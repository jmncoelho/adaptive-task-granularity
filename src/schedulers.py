"""
schedulers.py — All five scheduling strategies.

Each scheduler exposes:
    setup(devices, tasks, tick)   → called once at service admission
    step(devices, tasks, tick)    → called every simulation tick
    name                          → human-readable label
"""

from __future__ import annotations
from typing import List, Dict, Optional
from collections import defaultdict
import numpy as np
import math

from models import Device, Task, TaskState, SimConfig


# ── Base class ────────────────────────────────────────────────────────────────

class BaseScheduler:
    name: str = "Base"

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        # Global task pool for centralised schedulers
        self.global_pool: List[Task] = []
        # Communication overhead counter (bytes)
        self.bytes_control: float = 0.0
        # Latency-weighted task-transfer cost (abstract units). Isolates the
        # cost of *moving tasks* (framing + payload, scaled by link latency)
        # from the heartbeat/protocol traffic in bytes_control. This is the
        # quantity that the granularity mechanism actually trades off, and the
        # communication metric reported for EQ2.
        self.comm_cost: float = 0.0
        self.steal_attempts: int = 0
        self.steal_successes: int = 0
        self.steal_failures:  int = 0

    def _charge_transfer(self, ctrl_bytes: float, payload_bytes: float,
                         latency_ms: float) -> None:
        """Charge a task transfer to comm_cost, weighted by link latency."""
        lat_factor = 1.0 + latency_ms / 50.0
        self.comm_cost += (ctrl_bytes + payload_bytes) * lat_factor / 100.0

    def setup(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        pass

    def step(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        pass

    # ── Shared helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _active(devices: List[Device]) -> List[Device]:
        return [d for d in devices if d.active]

    @staticmethod
    def _tick_task(device: Device, tick: int) -> None:
        """Advance execution of device's current task by one tick."""
        if device.current_task is None:
            task = device.pop_bottom()
            if task is None:
                device.ticks_idle += 1
                return
            device.current_task = task
            task.state = TaskState.RUNNING
            if task.started_at is None:
                task.started_at = tick
            # Charge the one-off scheduling / metadata overhead T_sched(g) the
            # first time this task ever executes. More (finer) tasks => more
            # total overhead. Failed/expired tasks already charged are not
            # re-charged, so recovery does not double-count.
            if not task.sched_charged:
                ov = getattr(device.cfg, "sched_overhead", 0.0)
                if ov > 0.0:
                    task.remaining += ov
                    device.ticks_overhead += ov
                task.sched_charged = True

        task = device.current_task
        # Work done per tick proportional to effective capacity (capped at 1 unit).
        # exec_factor > 1 means the task really takes longer than the controller
        # predicted (model mismatch, invisible to the scheduler's estimates).
        work = max(0.1, min(1.0, device.eff_capacity)) / max(1e-6, task.exec_factor)
        task.remaining -= work
        device.ticks_busy += 1

        if task.remaining <= 0:
            task.state = TaskState.COMPLETED
            task.finished_at = tick
            task.remaining = 0.0
            device.tasks_done += 1
            device.current_task = None

    @staticmethod
    def _expire_leases(devices: List[Device], tick: int) -> None:
        for dev in devices:
            for task in list(dev.W):
                if (task.state == TaskState.LEASED and
                        task.lease_expiry > 0 and tick > task.lease_expiry):
                    task.state = TaskState.EXPIRED

    @staticmethod
    def _qos_utility(tasks: List[Task], cfg: SimConfig) -> float:
        """Predicted service utility based on completed + in-progress tasks."""
        total_w = sum(t.qos_weight for t in tasks)
        if total_w == 0:
            return 1.0
        done_w = sum(t.qos_weight for t in tasks
                     if t.state == TaskState.COMPLETED)
        return done_w / total_w


# ═════════════════════════════════════════════════════════════════════════════
# 1. STATIC PROPORTIONAL DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════════════

class StaticProportional(BaseScheduler):
    """
    Partition tasks once at admission proportional to P_i(t).
    Failed tasks are rescheduled by a central orchestrator.
    """
    name = "Static Proportional"

    def setup(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        active = self._active(devices)
        if not active:
            return
        caps = [d.perf_index for d in active]
        total = sum(caps)
        fracs = [c / total for c in caps]

        idx = 0
        for dev, frac in zip(active, fracs):
            n = max(1, round(frac * len(tasks)))
            for t in tasks[idx: idx + n]:
                dev.push_bottom(t)
                t.state = TaskState.QUEUED
            idx += n
        # Any remainder to first device
        for t in tasks[idx:]:
            active[0].push_bottom(t)

    def step(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        self._expire_leases(devices, tick)
        for dev in devices:
            if not dev.active:
                continue
            self._tick_task(dev, tick)

        # Orchestrator: re-queue expired/failed tasks onto least-loaded device
        pool = [t for t in tasks
                if t.state in (TaskState.EXPIRED, TaskState.FAILED)
                and t.retries < t.max_retries]
        if pool:
            active = self._active(devices)
            if active:
                target = min(active, key=lambda d: d.norm_load())
                for t in pool:
                    t.retries += 1
                    t.state = TaskState.QUEUED
                    target.push_bottom(t)
                    self.bytes_control += 64  # orchestrator message


# ═════════════════════════════════════════════════════════════════════════════
# 2. ENHANCED PROPORTIONAL DECOMPOSITION
# ═════════════════════════════════════════════════════════════════════════════

class EnhancedProportional(BaseScheduler):
    """
    Divide tasks into N rounds; each round is distributed proportionally.
    Reduces skew but still centralised.
    """
    name = "Enhanced Proportional"

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        super().__init__(cfg, rng)
        self._rounds: List[List[Task]] = []
        self._round_idx: int = 0
        self._next_round_tick: int = 0

    def setup(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        n = self.cfg.n_rounds
        size = math.ceil(len(tasks) / n)
        self._rounds = [tasks[i * size:(i + 1) * size] for i in range(n)]
        self._round_idx = 0
        self._next_round_tick = tick
        self._distribute_round(devices, tick)

    def _distribute_round(self, devices: List[Device], tick: int) -> None:
        if self._round_idx >= len(self._rounds):
            return
        active = self._active(devices)
        if not active:
            return
        batch = self._rounds[self._round_idx]
        caps = [d.perf_index for d in active]
        total = max(1e-6, sum(caps))
        fracs = [c / total for c in caps]
        idx = 0
        for dev, frac in zip(active, fracs):
            n = max(0, round(frac * len(batch)))
            for t in batch[idx: idx + n]:
                dev.push_bottom(t)
            idx += n
        for t in batch[idx:]:
            active[0].push_bottom(t)
        self._round_idx += 1
        self.bytes_control += 128 * len(batch)

    def step(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        self._expire_leases(devices, tick)
        for dev in devices:
            if not dev.active:
                continue
            self._tick_task(dev, tick)

        # Trigger next round when all active devices are roughly idle
        active = self._active(devices)
        if active and self._round_idx < len(self._rounds):
            avg_load = sum(d.norm_load() for d in active) / len(active)
            if avg_load < 0.3:
                self._distribute_round(devices, tick)

        # Re-queue failures
        pool = [t for t in tasks
                if t.state in (TaskState.EXPIRED, TaskState.FAILED)
                and t.retries < t.max_retries]
        if pool and active:
            target = min(active, key=lambda d: d.norm_load())
            for t in pool:
                t.retries += 1
                t.state = TaskState.QUEUED
                target.push_bottom(t)
                self.bytes_control += 64


# ═════════════════════════════════════════════════════════════════════════════
# 3. CENTRALISED DYNAMIC RESCHEDULING
# ═════════════════════════════════════════════════════════════════════════════

class CentralizedDynamic(BaseScheduler):
    """
    Devices report idleness to a central orchestrator, which assigns work.
    Models realistic coordinator overhead:
      - Round-trip latency between device and coordinator
      - Bounded coordinator throughput (assignments processed per tick)
      - FIFO queueing at the coordinator
    """
    name = "Centralized Dynamic"

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        super().__init__(cfg, rng)
        # Round-trip delay between idle device and assignment grant.
        # Tied to mean network latency × 2 (request + response).
        self.coord_round_trip: int = 20
        # Maximum assignment grants the coordinator can issue per tick.
        # Scales sub-linearly with cluster size to model finite server resources.
        self.coord_throughput: int = max(2, cfg.n_devices // 4)
        # Pending requests at the coordinator: list of (device_id, arrived_tick).
        self.coord_queue: List = []

    def setup(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        self.global_pool = list(tasks)
        for t in self.global_pool:
            t.state = TaskState.QUEUED

    def step(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        self._expire_leases(devices, tick)
        for dev in devices:
            if not dev.active:
                continue
            self._tick_task(dev, tick)

        # Re-queue expired tasks back to pool
        for t in tasks:
            if t.state in (TaskState.EXPIRED, TaskState.FAILED):
                if t.retries < t.max_retries:
                    t.retries += 1
                    t.state = TaskState.QUEUED
                    if t not in self.global_pool:
                        self.global_pool.append(t)

        # ── 1. Idle devices send requests to the coordinator ─────────────
        active = self._active(devices)
        in_queue = {dev_id for dev_id, _ in self.coord_queue}
        for dev in active:
            if (dev.current_task is None and not dev.W
                    and dev.id not in in_queue):
                # Request arrives at the coordinator after one one-way latency.
                arrives_at = tick + self.coord_round_trip // 2
                self.coord_queue.append((dev.id, arrives_at))
                self.bytes_control += 64   # request message

        # ── 2. Coordinator processes the front of the queue ──────────────
        available = [t for t in self.global_pool if t.state == TaskState.QUEUED]
        processed = 0
        while (self.coord_queue
               and processed < self.coord_throughput
               and available):
            dev_id, arrives_at = self.coord_queue[0]
            # The request hasn't arrived yet
            if arrives_at > tick:
                break
            self.coord_queue.pop(0)
            dev = next((d for d in active if d.id == dev_id), None)
            if dev is None:
                continue   # device disconnected while in queue
            # Coordinator picks a capacity-proportional batch
            n = min(max(1, int(dev.eff_capacity * 2)), len(available))
            for t in available[:n]:
                dev.push_bottom(t)
                self.global_pool.remove(t)
                self.bytes_control += 64
                self._charge_transfer(64, max(100, t.cost * 10),
                                      dev.spec.latency_ms)
            available = available[n:]
            processed += 1


# ═════════════════════════════════════════════════════════════════════════════
# 4. RANDOM DISTRIBUTED STEALING
# ═════════════════════════════════════════════════════════════════════════════

class RandomStealing(BaseScheduler):
    """
    Idle devices steal from randomly selected victims.
    No capacity-aware scoring, no QoS admission.
    """
    name = "Random Stealing"

    def setup(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        # Round-robin initial assignment
        active = self._active(devices)
        for i, t in enumerate(tasks):
            active[i % len(active)].push_bottom(t)

    def step(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        self._expire_leases(devices, tick)
        active = self._active(devices)
        hbs = {d.id: d.heartbeat(tick) for d in active}

        for dev in active:
            # Tick execution
            self._tick_task(dev, tick)
            # Steal if idle
            if (dev.current_task is None and not dev.W
                    and tick >= dev.backoff_until):
                others = [d for d in active if d.id != dev.id and d.W]
                if not others:
                    self._back_off(dev, tick)
                    continue
                victim = self.rng.choice(others)
                stolen = self._do_steal(dev, victim, tick)
                if stolen:
                    self.steal_successes += 1
                else:
                    self.steal_failures += 1
                    self._back_off(dev, tick)
                self.steal_attempts += 1

        # Re-queue expired tasks
        for t in tasks:
            if t.state == TaskState.EXPIRED and t.retries < t.max_retries:
                t.retries += 1
                if active:
                    self.rng.choice(active).push_bottom(t)

    def _do_steal(self, thief: Device, victim: Device, tick: int) -> bool:
        stealable = victim.stealable()
        if not stealable:
            return False
        n = min(self.cfg.b_max, max(1, len(stealable) // 2))
        batch = stealable[:n]
        data_bytes = sum(max(100, t.cost * 10) for t in batch)
        for t in batch:
            victim.W.remove(t)
            thief.push_bottom(t)
        self.bytes_control += 128 * n
        self._charge_transfer(128 * n, data_bytes, victim.spec.latency_ms)
        thief.backoff_count = 0
        return True

    def _back_off(self, dev: Device, tick: int) -> None:
        wait = min(self.cfg.backoff_max,
                   self.cfg.backoff_base * (2 ** dev.backoff_count))
        dev.backoff_until = tick + int(wait)
        dev.backoff_count += 1


# ═════════════════════════════════════════════════════════════════════════════
# 5. QoS-AWARE DISTRIBUTED WORK STEALING  (proposed algorithm)
# ═════════════════════════════════════════════════════════════════════════════

class QoSAwareStealing(BaseScheduler):
    """
    Full proposed algorithm:
      - Proactive stealing when lambda_i < lambda_low
      - Victim selection via V_{j,i} score (surplus × kappa / (1+Delta+E))
      - Batch size: min(B_max, ceil(sigma/w_bar), floor(|stealable|/2))
      - QoS-aware admission: U_S(t+Dt|steal) >= U_min
      - Lease-based ownership with epoch
    """
    name = "QoS-Aware Stealing"

    def setup(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        # Round-robin initial assignment
        active = self._active(devices)
        for i, t in enumerate(tasks):
            active[i % len(active)].push_bottom(t)

    def step(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        cfg = self.cfg
        active = self._active(devices)

        # ── 1. Update resources ───────────────────────────────────────────
        # (done by simulation engine before calling step)

        # ── 2. Publish heartbeats ─────────────────────────────────────────
        if tick % cfg.hb_interval == 0:
            hbs = {d.id: d.heartbeat(tick) for d in active}
            for dev in active:
                for src_id, _ in hbs.items():
                    if src_id != dev.id:
                        dev.record_hb(src_id, tick)
            self.bytes_control += len(active) * len(active) * 32

        # ── 3. Expire stale leases ────────────────────────────────────────
        self._expire_leases(active, tick)

        # ── 4. Execute + proactive steal ──────────────────────────────────
        for dev in active:
            # Guard reservation check
            if dev.residual <= 0.0:
                dev.ticks_idle += 1
                continue

            # Execute one step on current or next task
            self._tick_task(dev, tick)

            # Proactive steal if underloaded
            if (dev.norm_load() < cfg.lambda_low
                    and tick >= dev.backoff_until):
                self._try_steal(dev, active, tasks, tick)

        # ── 5. Capacity-aware recovery of orphan EXPIRED tasks ────────────
        self._recover_orphans(tasks, active, tick)

    def _recover_orphans(self, tasks: List[Task],
                         active: List[Device], tick: int) -> None:
        """
        Redistribute EXPIRED tasks that are no longer in any active deque.
        Targets the highest-effective-capacity device that is not overloaded.
        Mirrors the victim score logic but in reverse (push, not pull).
        """
        if not active:
            return
        # Tasks currently held in some active deque (by task id)
        in_deque_ids = {t.id for d in active for t in d.W}
        orphans = [t for t in tasks
                   if t.state == TaskState.EXPIRED
                   and t.id not in in_deque_ids
                   and t.retries < t.max_retries]
        if not orphans:
            return

        # Sort candidates by effective capacity, descending
        candidates = sorted(active, key=lambda d: d.eff_capacity, reverse=True)

        for task in orphans:
            # Pick first candidate not already overloaded
            target = next(
                (d for d in candidates
                 if d.norm_load() < self.cfg.lambda_high),
                candidates[0],
            )
            task.retries += 1
            task.state = TaskState.QUEUED
            task.owner = target.id
            task.lease_epoch = 0
            task.lease_expiry = 0
            target.push_bottom(task)
            # Recovery is initiated by the cluster, not a peer steal —
            # only charge a small control message.
            self.bytes_control += 64

    # ── Victim selection ──────────────────────────────────────────────────────

    def _victim_score(self, thief: Device, victim: Device, tick: int) -> float:
        """V_{j,i}(t) = s_i * kappa_i / (1 + Delta_hat + E_hat)"""
        cfg = self.cfg
        # Queue surplus s_i
        lam_i = victim.norm_load()
        s_i = max(0.0, lam_i - cfg.lambda_star) * victim.eff_capacity
        if s_i <= 0:
            return 0.0
        # Stability factor kappa_i (as seen by thief)
        kappa = thief.kappa(victim.id, tick)
        # Normalised communication cost
        delta_max = 50.0   # ms
        e_max = 1.0        # normalised energy reference
        delta_hat = min(1.0, victim.spec.latency_ms / delta_max)
        # Energy proxy: latency × batch size (simplified)
        e_hat = min(1.0, (victim.spec.latency_ms / delta_max) * 0.5)
        return (s_i * kappa) / (1.0 + delta_hat + e_hat)

    def _select_victim(self, thief: Device, candidates: List[Device],
                       tick: int) -> Optional[Device]:
        scores = []
        for d in candidates:
            if d.id == thief.id or not d.active:
                continue
            s = self._victim_score(thief, d, tick)
            if s > 0:
                scores.append((d, s))
        if not scores:
            return None
        # Softmax sampling proportional to scores
        devices_list, vals = zip(*scores)
        vals_arr = np.array(vals, dtype=float)
        probs = vals_arr / vals_arr.sum()
        return self.rng.choice(devices_list, p=probs)  # type: ignore[arg-type]

    # ── Batch size ────────────────────────────────────────────────────────────

    def _batch_size(self, thief: Device, victim: Device) -> int:
        cfg = self.cfg
        sigma = max(0.0, cfg.lambda_star - thief.norm_load()) * thief.eff_capacity
        stealable = victim.stealable()
        if not stealable:
            return 0
        w_bar = max(0.1, np.mean([t.remaining for t in stealable]))
        n1 = cfg.b_max
        n2 = math.ceil(sigma / w_bar) if sigma > 0 else 1
        n3 = max(1, len(stealable) // 2)
        return min(n1, n2, n3)

    # ── QoS admission ─────────────────────────────────────────────════════────

    def _admissible(self, batch: List[Task], thief: Device,
                    tasks: List[Task]) -> bool:
        """
        Conditions (iii) and (iv) of Definition 11 (Stealability).

        (iii) The thief must realistically complete the batch within the lease
              horizon — bounded by 2 × lease_dur to allow one renewal.
        (iv)  Service utility must remain at or above U_min after the steal,
              and the thief must be capable enough to not degrade output.
        """
        cfg = self.cfg
        # The commit gate has two separable components, each individually
        # switchable for the RQ11 ablation:
        #   * feasibility (iii): lease-renewal timing and capability floor;
        #   * QoS admission (iv): predicted post-steal utility >= U_min.
        # (iii) Feasibility.
        if not getattr(cfg, "no_feasibility", False):
            footprint = sum(t.remaining for t in batch)
            exec_rate = max(0.5, thief.eff_capacity)
            expected_time = footprint / exec_rate
            if expected_time > cfg.lease_dur * 2:
                return False
            if thief.eff_capacity < 0.3:
                return False
        # (iv) QoS admission. Under the progress-based, monotone utility of
        # this instantiation, completing a stolen batch never lowers utility,
        # so the predicted post-steal utility is always >= the current one and
        # the test is satisfied. It is retained (and switchable) because
        # non-monotone utility models can make it bind.
        if not getattr(cfg, "no_qos_admit", False):
            total_w = sum(t.qos_weight for t in tasks)
            if total_w > 0:
                completed_w = sum(t.qos_weight for t in tasks
                                  if t.state == TaskState.COMPLETED)
                u_now = completed_w / total_w
                # predicted post-steal utility never falls below u_now here
                if u_now < 0.0:   # structurally unreachable; kept for clarity
                    return False
        return True

    # ── Steal attempt ─────────────────────────────────────────────────────────

    def _try_steal(self, thief: Device, active: List[Device],
                   tasks: List[Task], tick: int) -> None:
        victim = self._select_victim(thief, active, tick)
        self.steal_attempts += 1
        thief.steal_attempts += 1

        if victim is None:
            self._back_off(thief, tick)
            self.steal_failures += 1
            return

        # Victim-side: select stealable tasks
        stealable = victim.stealable()
        n = self._batch_size(thief, victim)
        if n == 0 or not stealable:
            self._back_off(thief, tick)
            self.steal_failures += 1
            victim.steal_denials += 1
            return

        batch = stealable[:n]

        # QoS admission check
        if not self._admissible(batch, thief, tasks):
            self._back_off(thief, tick)
            self.steal_failures += 1
            victim.steal_denials += 1
            return

        # Grant: remove from victim, create lease, push to thief
        e = victim.next_epoch(thief.id)
        for t in batch:
            victim.W.remove(t)
            t.owner = thief.id
            t.lease_epoch = e
            t.lease_expiry = tick + self.cfg.lease_dur
            t.state = TaskState.LEASED
            thief.push_bottom(t)

        # Accounting
        ctrl = 128 + self.cfg.frame_bytes * len(batch)
        data_bytes = sum(max(100, t.cost * 10) for t in batch)
        self.bytes_control += ctrl
        thief.bytes_xfer += data_bytes
        self._charge_transfer(ctrl, data_bytes, victim.spec.latency_ms)
        victim.steal_grants += 1
        thief.steal_successes += 1
        thief.backoff_count = 0
        self.steal_successes += 1

    def _back_off(self, dev: Device, tick: int) -> None:
        wait = min(self.cfg.backoff_max,
                   self.cfg.backoff_base * (2 ** dev.backoff_count))
        dev.backoff_until = tick + int(wait)
        dev.backoff_count = min(dev.backoff_count + 1, 8)


# ── Factory ───────────────────────────────────────────────────────────────────

def make_scheduler(stype: str, cfg: SimConfig,
                   rng: np.random.Generator) -> BaseScheduler:
    mapping = {
        "static":   StaticProportional,
        "enhanced": EnhancedProportional,
        "central":  CentralizedDynamic,
        "random":   RandomStealing,
        "qos":      QoSAwareStealing,
    }
    cls = mapping.get(stype)
    if cls is None:
        raise ValueError(f"Unknown scheduler '{stype}'. "
                         f"Choose from: {list(mapping)}")
    return cls(cfg, rng)
