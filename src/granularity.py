"""
granularity.py — Adaptive task-granularity extension.

Implements, on top of the QoS-aware work-stealing simulator:

  * granularity-aware task generation at a fixed total service work,
  * the split / merge operators (Definitions in the follow-up paper),
  * the granularity-pressure function  Psi(t),
  * the AdaptiveQoSStealing scheduler (thief- and victim-side adaptation),
  * a six-policy registry used by the evaluation runner.

Total service work is conserved by split (children costs sum to the parent
cost) and by merge (the merged cost is the sum of inputs), and QoS weight is
likewise conserved, so the service utility accounting in simulation.py stays
exact across adaptation.  The live task list passed into the scheduler is
mutated in place by split/merge so the engine's termination and QoS checks
always see the current set of leaves.
"""

from __future__ import annotations
from typing import List, Optional, Tuple, Dict
import math
import numpy as np

from models import Device, Task, TaskState, SimConfig
from schedulers import QoSAwareStealing, StaticProportional, make_scheduler


# ══════════════════════════════════════════════════════════════════════════════
# GRANULARITY-AWARE TASK GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def total_service_work(cfg: SimConfig) -> float:
    """Total work of the service, held fixed across granularity levels."""
    if cfg.service_work and cfg.service_work > 0:
        return float(cfg.service_work)
    return float(cfg.n_tasks) * float(cfg.gran_levels[cfg.gran_init])


def generate_tasks_gran(cfg: SimConfig, rng: np.random.Generator,
                        level: int) -> List[Task]:
    """
    Generate tasks at granularity ``level`` whose total work equals
    ``total_service_work(cfg)``.  The cost distribution (uniform / lognormal /
    pareto) is the same machinery as the base simulator, but the *mean* task
    cost is the granularity level's nominal size.
    """
    level = int(np.clip(level, 0, len(cfg.gran_levels) - 1))
    mean = float(cfg.gran_levels[level])
    cv = cfg.task_cv
    W = total_service_work(cfg)
    n = max(1, int(round(W / mean)))

    if cfg.task_dist == "uniform":
        # Matches the accepted QoS-WS paper: U[1, 2*mean-1], i.e. U[1,39] at
        # the reference mean of 20 ticks (CV ~ 0.55), scaled per level.
        costs = rng.uniform(1.0, max(1.5, 2.0 * mean - 1.0), n)
    elif cfg.task_dist == "lognormal":
        # Matches the accepted paper at cv=1.0, mean 20.
        sigma2 = np.log(1 + cv ** 2)
        mu = np.log(mean) - sigma2 / 2
        costs = rng.lognormal(mu, np.sqrt(sigma2), n)
    elif cfg.task_dist == "pareto":
        # Matches the accepted paper: shape alpha=2 (infinite variance),
        # x_m = mean/2 so that E[X] = 2*x_m = mean.
        xm = mean / 2.0
        costs = (rng.pareto(2.0, n) + 1) * xm
    else:
        raise ValueError(f"Unknown distribution: {cfg.task_dist}")

    costs = np.clip(costs, 1.0, None)
    qos_w = rng.uniform(0.5, 1.5, n)
    ex_sig = getattr(cfg, "exec_mismatch", 0.0)
    ex_fac = (np.exp(rng.normal(0.0, ex_sig, n)) if ex_sig > 0.0
              else np.ones(n))
    tasks = []
    edges = np.linspace(0.0, 1.0, n + 1)
    for i in range(n):
        _t = Task(
            i, float(costs[i]), qos_weight=float(qos_w[i]),
            gran=level, root_id=i,
            data_lo=float(edges[i]), data_hi=float(edges[i + 1]),
        )
        _t.exec_factor = float(ex_fac[i])
        tasks.append(_t)
    return tasks


# ══════════════════════════════════════════════════════════════════════════════
# SPLIT / MERGE OPERATORS
# ══════════════════════════════════════════════════════════════════════════════

def split_task(task: Task, target_level: int, k: int,
               id_alloc, registry: List[Task]) -> List[Task]:
    """
    split(tau, g') -> {tau_1, ..., tau_k}  at finer granularity g' < Gamma(tau).

    Partitions the task's remaining work and data slice into ``k`` disjoint
    children. Children costs sum to the parent's remaining cost and children
    QoS weights sum to the parent weight, so total work and total QoS weight
    are conserved (idempotent refinement). The parent is removed from the live
    registry and replaced by its children.
    """
    if k < 2 or task.state not in (TaskState.QUEUED, TaskState.EXPIRED):
        return [task]
    base = task.remaining / k
    w = task.qos_weight / k
    _ef = getattr(task, "exec_factor", 1.0)
    span = (task.data_hi - task.data_lo) / k
    children: List[Task] = []
    for j in range(k):
        cid = id_alloc()
        c = Task(cid, base, qos_weight=w,
                 gran=target_level, root_id=task.root_id,
                 data_lo=task.data_lo + j * span,
                 data_hi=task.data_lo + (j + 1) * span)
        c.state = TaskState.QUEUED
        c.owner = task.owner
        c.retries = task.retries
        c.exec_factor = _ef
        children.append(c)
    # Swap in the registry: parent -> children
    try:
        registry.remove(task)
    except ValueError:
        pass
    registry.extend(children)
    return children


def merge_tasks(group: List[Task], target_level: int,
                id_alloc, registry: List[Task]) -> Optional[Task]:
    """
    merge(tau_1, ..., tau_k) -> tau'  at coarser granularity.

    Combines compatible queued tasks of the same service into one coarser task.
    Cost and QoS weight are summed (conserved). The inputs are removed from the
    live registry and replaced by the merged task. Returns None if fewer than
    two compatible tasks are supplied.
    """
    group = [t for t in group
             if t.state in (TaskState.QUEUED, TaskState.EXPIRED)
             and not t.sched_charged]
    if len(group) < 2:
        return None
    cost = sum(t.remaining for t in group)
    w = sum(t.qos_weight for t in group)
    lo = min(t.data_lo for t in group)
    hi = max(t.data_hi for t in group)
    mid = id_alloc()
    _tot = sum(t.remaining for t in group) or 1.0
    _mef = sum(getattr(t, "exec_factor", 1.0) * t.remaining for t in group) / _tot
    merged = Task(mid, cost, qos_weight=w,
                  gran=target_level, root_id=group[0].root_id,
                  data_lo=lo, data_hi=hi)
    merged.state = TaskState.QUEUED
    merged.owner = group[0].owner
    merged.retries = max(t.retries for t in group)
    merged.exec_factor = _mef
    for t in group:
        try:
            registry.remove(t)
        except ValueError:
            pass
    registry.append(merged)
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# GRANULARITY PRESSURE  Psi(t)
# ══════════════════════════════════════════════════════════════════════════════

def granularity_pressure(devices: List[Device], tasks: List[Task],
                         cfg: SimConfig, comm_cost: float,
                         work_done: float, recent_fail_rate: float,
                         tick: int = 0, load_view: Optional[Dict[int, float]] = None
                         ) -> Tuple[float, Dict[str, float]]:
    """
    Psi(t) = eta_I I(t) - eta_O O(t) + eta_Q Q_p(t) + eta_F F(t)

    Positive Psi favours finer granularity (split); negative favours coarser
    (merge). O(t) is the latency-weighted task-transfer cost per unit of useful
    work, so a congested / high-latency cluster pushes toward coarser units.
    """
    active = [d for d in devices if d.active]
    # ── I(t): normalised load imbalance (coefficient of variation of loads) ──
    # Loads may come from a possibly-stale heartbeat view; fall back to live
    # values when no view is supplied.
    if load_view is not None:
        loads = np.array([load_view.get(d.id, d.norm_load())
                          for d in active], dtype=float)
    else:
        loads = np.array([d.norm_load() for d in active], dtype=float)
    if loads.size and loads.mean() > 1e-9:
        I = float(loads.std() / (loads.mean() + 1e-9))
    else:
        I = 0.0
    if getattr(cfg, "raw_signals", False):
        I_n = float(I)  # unnormalised: raw coefficient of variation
    else:
        I_n = float(np.clip(I / cfg.i_ref, 0.0, cfg.i_cap))
    # ── O(t): transfer cost per unit of useful work, normalised to [0,1] ─────
    O = comm_cost / (work_done + 1e-9)
    O_n = float(O) if getattr(cfg, "raw_signals", False) \
        else float(np.clip(O / cfg.o_ref, 0.0, 1.0))
    # ── Q_p(t): normalised predicted QoS risk against the service deadline ──
    # Required-trajectory form: U_req(t) = U_min * min(1, t/ddl); the signal is
    # the normalised deficit relative to that trajectory, in [0,1]. At t=0 the
    # pressure is zero (full time remains); it rises only when progress falls
    # behind the pace needed to clear the floor by the deadline.
    total_w = sum(t.qos_weight for t in tasks)
    done_w = sum(t.qos_weight for t in tasks if t.state == TaskState.COMPLETED)
    U = done_w / total_w if total_w > 0 else 1.0
    ddl = getattr(cfg, "qos_deadline", None) or cfg.max_ticks
    u_req = cfg.u_min * min(1.0, tick / max(1, ddl))
    Q_n = float(np.clip((u_req - U) / max(1e-9, cfg.u_min), 0.0, 1.0))
    # ── F(t): failure pressure in [0,1] ──────────────────────────────────────
    F_n = float(np.clip(recent_fail_rate, 0.0, 1.0))

    psi = (cfg.eta_I * I_n - cfg.eta_O * O_n
           + cfg.eta_Q * Q_n + cfg.eta_F * F_n)
    return psi, {"I": I_n, "O": O_n, "Q": Q_n, "F": F_n, "U": U}


# ══════════════════════════════════════════════════════════════════════════════
# ADAPTIVE QoS-AWARE WORK STEALING
# ══════════════════════════════════════════════════════════════════════════════

class AdaptiveQoSStealing(QoSAwareStealing):
    """
    Granularity-aware extension of QoS-aware work stealing.

    On each steal the thief chooses not only a victim and batch size but also a
    target granularity g*, derived from the granularity pressure Psi(t) and its
    own residual capacity. The victim then exposes stealable work at g*:
      * Psi > deadband  -> split coarse tasks toward the finer g*,
      * Psi < -deadband -> merge small tasks toward the coarser g*.
    All adaptation is bounded (finite ladder), QoS-admission-preserving, and
    rate-limited by a cooldown to prevent oscillation.
    """
    name = "Adaptive QoS-WS"

    def __init__(self, cfg: SimConfig, rng: np.random.Generator):
        super().__init__(cfg, rng)
        self._next_id_val: int = 0
        self._psi: float = 0.0
        self._psi_parts: Dict[str, float] = {}
        self._work_done: float = 0.0
        self._fail_events: list = []      # ticks of recent failures (decaying)
        self._last_adapt: Dict[int, int] = {}   # task root_id -> last change tick
        self.n_splits: int = 0
        self.n_merges: int = 0
        self._registry: Optional[List[Task]] = None
        self.psi_trace: list = []
        # Current reference level of the (single) service lineage family,
        # initialised at the generation level and updated after each admitted
        # adaptation; enables gradual traversal of the full ladder.
        self._ref: int = int(self.cfg.gran_init)
        self._ref_changed: int = 0   # tick of the last reference movement
        # Heartbeat view of per-device load for the pressure signals, with
        # configurable staleness: entries are refreshed only every
        # hb_interval ticks, dropped with probability hb_loss, and the whole
        # view lags the true state. This is what RQ9 stresses.
        self._load_view: Dict[int, float] = {}
        self._view_age: Dict[int, int] = {}

    # ── id allocation for new (split/merge) tasks ──────────────────────────
    def _alloc_id(self) -> int:
        self._next_id_val += 1
        return self._next_id_val

    def setup(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        self._next_id_val = max((t.id for t in tasks), default=0) + 1
        self._registry = tasks
        super().setup(devices, tasks, tick)

    def _policy_psi(self, psi: float, parts: Dict[str, float],
                    devices: List[Device]) -> float:
        """Hook for alternative decision rules; the base controller keeps
        the weighted pressure unchanged."""
        return psi

    # ── desired granularity level from pressure ────────────────────────────
    def _desired_level(self, thief: Device) -> int:
        h = len(self.cfg.gran_levels)
        base = self._ref                # current reference, not the origin
        db = self.cfg.psi_deadband
        if self._psi > db:
            lvl = base - 1            # one level finer (split)
        elif self._psi < -db:
            lvl = base + 1            # one level coarser (merge)
        else:
            lvl = base
        # A weak thief should not aim for coarse work (lease feasibility);
        # nudge weak thieves toward the finer side.
        if thief.eff_capacity < 0.6:
            lvl = min(lvl, base)
        return int(np.clip(lvl, 0, h - 1))

    def _noisy(self, x: float) -> float:
        """Multiplicative estimation noise (decision estimates only)."""
        s = getattr(self.cfg, "est_noise", 0.0)
        if s <= 0.0:
            return x
        return float(x * (1.0 + self.rng.normal(0.0, s)))

    # ── victim-side adaptive exposure of stealable work ────────────────────
    def _expose(self, victim: Device, g_target: int, tick: int,
                need: int = 10**9) -> List[Task]:
        """Split or merge victim's stealable tasks toward g_target, adapting
        only the minimal subset required to expose up to `need` tasks at the
        target level. Mutates victim.W and the live registry. An admitted
        adaptation updates the current reference level."""
        reg = self._registry
        stealable = victim.stealable()
        if not stealable or reg is None:
            return stealable
        db = self.cfg.psi_deadband
        adapted = False

        if self._psi > db and not self.cfg.adapt_no_split:
            # Refine coarse tasks toward g_target: at most B_max source tasks
            # are adapted per grant --- exactly the per-steal adaptation
            # budget assumed by the overhead bound (N_s * B_max * K).
            budget = min(max(need, 1), self.cfg.b_max)
            adapted_src = 0
            for t in list(stealable):
                if adapted_src >= budget:
                    break
                if t.gran > g_target and tick - self._last_adapt.get(t.root_id, -10**9) >= self.cfg.gran_cooldown:
                    ratio = self.cfg.gran_levels[t.gran] / self.cfg.gran_levels[g_target]
                    k = int(np.clip(round(ratio), 2, 3))
                    children = split_task(t, g_target, k, self._alloc_id, reg)
                    if len(children) > 1:
                        if t in victim.W:
                            victim.W.remove(t)
                        for c in children:
                            victim.W.append(c)
                        victim.n_splits += 1
                        self.n_splits += 1
                        self._last_adapt[t.root_id] = tick
                        self.bytes_control += 32   # split metadata
                        adapted_src += 1
                        adapted = True
        elif self._psi < -db and not self.cfg.adapt_no_merge:
            # Coarsen: merge cooldown-eligible small tasks toward g_target,
            # producing at most `need` merged tasks.
            smalls = [t for t in stealable
                      if t.gran < g_target and not t.sched_charged
                      and tick - self._last_adapt.get(t.root_id, -10**9)
                          >= self.cfg.gran_cooldown]
            if len(smalls) >= 2:
                target_size = self.cfg.gran_levels[g_target]
                group: List[Task] = []
                acc, produced = 0.0, 0
                budget = min(max(need, 1), self.cfg.b_max)
                for t in smalls:
                    if produced >= budget:
                        break
                    group.append(t)
                    acc += t.remaining
                    if acc >= target_size and len(group) >= 2:
                        merged = merge_tasks(group, g_target, self._alloc_id, reg)
                        if merged is not None:
                            for gt in group:
                                if gt in victim.W:
                                    victim.W.remove(gt)
                                # merge is an adaptation of every member
                                # lineage: stamp all their cooldown timers.
                                self._last_adapt[gt.root_id] = tick
                            self._last_adapt[merged.root_id] = tick
                            victim.W.append(merged)
                            victim.n_merges += 1
                            self.n_merges += 1
                            self.bytes_control += 16
                            produced += 1
                            adapted = True
                        group, acc = [], 0.0
        if adapted and g_target != self._ref:
            # Admitted adaptation: advance the current reference toward the
            # target (one step), enabling full-ladder traversal over time.
            self._ref = int(np.clip(g_target, 0, len(self.cfg.gran_levels) - 1))
            self._ref_changed = tick
        return victim.stealable()

    # ── override the steal attempt to add granularity adaptation ───────────
    def _try_steal(self, thief: Device, active: List[Device],
                   tasks: List[Task], tick: int) -> None:
        victim = self._select_victim(thief, active, tick)
        self.steal_attempts += 1
        thief.steal_attempts += 1
        if victim is None:
            self._back_off(thief, tick)
            self.steal_failures += 1
            return

        g_target = self._desired_level(thief)
        # Requested batch, estimated at the target level BEFORE adaptation:
        # the victim adapts only the minimal subset needed for this request.
        lvl_size = self.cfg.gran_levels[g_target]
        sigma = max(0.0, (self.cfg.lambda_star - thief.norm_load())) * max(1e-9, thief.eff_capacity)
        b_req = max(1, min(self.cfg.b_max,
                           int(np.ceil(sigma / max(1e-9, lvl_size)))))
        stealable = self._expose(victim, g_target, tick, need=self.cfg.b_max)
        if not stealable:
            self._back_off(thief, tick)
            self.steal_failures += 1
            victim.steal_denials += 1
            return

        n = self._batch_size(thief, victim)
        if n == 0:
            self._back_off(thief, tick)
            self.steal_failures += 1
            victim.steal_denials += 1
            return
        batch = stealable[:n]

        # Final commit gate: feasibility of the assembled batch on the thief
        # (each task must fit within one lease at the thief's capacity) and
        # the QoS-safety re-check. Adapted descriptors stay in the victim's
        # queue if the transfer is denied; every individual split/merge has
        # already passed its own admissibility guard, so a denial leaves the
        # queue valid without descriptor rollback.
        # Feasibility under lease renewal: the task must be executable on
        # the thief within a bounded number of lease renewals (r_max), i.e.
        # it must make committed progress, not necessarily finish in one
        # lease horizon.
        cap = max(1e-9, thief.eff_capacity)
        r_max = 3
        batch = [t for t in batch
                 if t.remaining / cap <= r_max * self.cfg.lease_dur]
        # _admissible internally honours no_feasibility / no_qos_admit flags.
        if not batch or not self._admissible(batch, thief, tasks):
            self._back_off(thief, tick)
            self.steal_failures += 1
            victim.steal_denials += 1
            return

        e = victim.next_epoch(thief.id)
        for t in batch:
            if t in victim.W:
                victim.W.remove(t)
            t.owner = thief.id
            t.lease_epoch = e
            t.lease_expiry = tick + self.cfg.lease_dur
            t.state = TaskState.LEASED
            thief.push_bottom(t)

        # Accounting: control + per-task framing + latency-weighted transfer
        ctrl = 128 + self.cfg.frame_bytes * len(batch)
        data_bytes = sum(max(100, t.cost * 10) for t in batch)
        self.bytes_control += ctrl
        thief.bytes_xfer += data_bytes
        self._charge_transfer(ctrl, data_bytes, victim.spec.latency_ms)
        victim.steal_grants += 1
        thief.steal_successes += 1
        thief.backoff_count = 0
        self.steal_successes += 1

    # ── per-tick step: maintain Psi, then run base QoS logic ───────────────
    def step(self, devices: List[Device], tasks: List[Task], tick: int) -> None:
        self._registry = tasks
        # Detect devices that failed on this tick (set by the engine before step)
        new_fails = sum(1 for d in devices
                        if not d.active and d.failed_at == tick)
        for _ in range(new_fails):
            self._fail_events.append(tick)
        # decay failure events over a window
        self._fail_events = [ft for ft in self._fail_events if tick - ft < 500]
        n_dev = max(1, len(devices))
        recent_fail_rate = len(self._fail_events) / (n_dev * 5.0)
        # useful work done so far (busy ticks)
        self._work_done = sum(d.ticks_busy for d in devices)
        # ── Refresh the (possibly stale/lossy) heartbeat view of loads ──────
        hb_iv = max(1, self.cfg.hb_interval)
        hb_loss = getattr(self.cfg, "hb_loss", 0.0)
        est_noise = getattr(self.cfg, "est_noise", 0.0)
        if tick % hb_iv == 0:
            for d in devices:
                if not d.active:
                    continue
                if hb_loss > 0.0 and self.rng.random() < hb_loss:
                    continue   # this heartbeat is lost; view keeps ageing
                val = d.norm_load()
                if est_noise > 0.0:
                    val = max(0.0, val * (1.0 + self.rng.normal(0.0, est_noise)))
                self._load_view[d.id] = val
                self._view_age[d.id] = tick
        view = self._load_view if (self.cfg.hb_interval > 1
                                   or hb_loss > 0.0 or est_noise > 0.0) else None
        self._psi, self._psi_parts = granularity_pressure(
            devices, tasks, self.cfg, self.comm_cost,
            self._work_done, recent_fail_rate, tick, load_view=view)
        if self.cfg.adapt_random:
            # Negative control: replace the pressure by uniform noise so the
            # operators fire with random direction at the same bounded rate.
            self._psi = float(self.rng.uniform(-1.0, 1.0))
        # Subclass hook: lets alternative controllers replace the decision
        # signal BEFORE any stealing decision of this tick is taken.
        self._psi = self._policy_psi(self._psi, self._psi_parts, devices)
        # Reference relaxation: while the pressure rests inside the dead-band,
        # the current reference drifts one level per cooldown interval back
        # toward the generation anchor. Sustained pressure can therefore
        # traverse the whole ladder, while calm recentres the neighbourhood
        # (hysteresis with recentring); the reference movement itself is not a
        # task adaptation and performs no operation.
        if (abs(self._psi) <= self.cfg.psi_deadband
                and self._ref != self.cfg.gran_init
                and tick - self._ref_changed >= self.cfg.gran_cooldown):
            self._ref += 1 if self._ref < self.cfg.gran_init else -1
            self._ref_changed = tick
        if self.cfg.trace_psi:
            self.psi_trace.append({
                "tick": int(tick), "psi": float(self._psi),
                **{k: float(v) for k, v in self._psi_parts.items()},
                "n_splits": int(self.n_splits), "n_merges": int(self.n_merges),
            })
        super().step(devices, tasks, tick)


class ThresholdHeuristicStealing(AdaptiveQoSStealing):
    """Simple-rule baseline: split when idle devices exist and normalised
    overhead is low; merge when normalised overhead is high; otherwise hold.
    Reuses all operators, guards, cooldowns, and QoS admission of the adaptive
    scheduler --- only the pressure equation is replaced, isolating the value
    of the weighted multi-signal controller."""
    name = "Threshold heuristic"
    O_THRESH = 0.5

    def _policy_psi(self, psi, parts, devices):
        o_n = parts.get("O", 0.0)
        any_idle = any(d.active and len(d.W) == 0 for d in devices)
        if o_n > self.O_THRESH:
            return -1.0
        if any_idle:
            return 1.0
        return 0.0


# ══════════════════════════════════════════════════════════════════════════════
# POLICY REGISTRY  (six policies compared in the follow-up paper)
# ══════════════════════════════════════════════════════════════════════════════

def _level_indices(cfg: SimConfig) -> Dict[str, int]:
    h = len(cfg.gran_levels)
    return {"fine": 0, "g10": 1, "medium": h // 2, "g40": h - 2,
            "coarse": h - 1}


# policy_key -> (scheduler_factory_key, generation_level_name, is_adaptive)
POLICIES: Dict[str, Tuple[str, str, bool]] = {
    "static_coarse": ("static",  "coarse", False),
    "static_fine":   ("static",  "fine",   False),
    "qosws_coarse":  ("qos",     "coarse", False),
    "qosws_medium":  ("qos",     "medium", False),
    "qosws_fine":    ("qos",     "fine",   False),
    "adaptive":      ("adaptive", "medium", True),
}

# Core six compared throughout the paper; the rest serve RQ-specific roles.
CORE_POLICIES = list(POLICIES.keys())

# Extra fixed levels so the retrospective oracle spans the full ladder.
POLICIES.update({
    "qosws_g10": ("qos", "g10", False),
    "qosws_g40": ("qos", "g40", False),
})

# Ablations and controls (all generate at the medium level, like adaptive).
ABLATIONS: Dict[str, Dict] = {
    "adaptive_noQ":       {"eta_Q": 0.0},
    "adaptive_noF":       {"eta_F": 0.0},
    "adaptive_noO":       {"eta_O": 0.0},
    "adaptive_nosplit":   {"adapt_no_split": True},
    "adaptive_nomerge":   {"adapt_no_merge": True},
    "adaptive_batchonly": {"adapt_no_split": True, "adapt_no_merge": True},
    "adaptive_random":    {"adapt_random": True},
    "adaptive_noqos":     {"no_qos_admit": True},
    "adaptive_nofeas":    {"no_feasibility": True},
    "adaptive_nogate":    {"no_qos_admit": True, "no_feasibility": True},
    "adaptive_raw":       {"raw_signals": True},
}
for k in ABLATIONS:
    POLICIES[k] = ("adaptive", "medium", True)
POLICIES["threshold"] = ("threshold", "medium", True)
ABLATIONS["threshold_raw"] = {"raw_signals": True}
POLICIES["threshold_raw"] = ("threshold", "medium", True)

POLICY_LABELS: Dict[str, str] = {
    "static_coarse": "Static coarse",
    "static_fine":   "Static fine",
    "qosws_coarse":  "QoS-WS coarse",
    "qosws_medium":  "QoS-WS medium",
    "qosws_fine":    "QoS-WS fine",
    "adaptive":      "Adaptive QoS-WS",
    "qosws_g10":     "QoS-WS g=10",
    "qosws_g40":     "QoS-WS g=40",
    "adaptive_noQ":       "Adaptive ($\\eta_Q{=}0$)",
    "adaptive_noF":       "Adaptive ($\\eta_F{=}0$)",
    "adaptive_noO":       "Adaptive ($\\eta_O{=}0$)",
    "adaptive_nosplit":   "Adaptive (no split)",
    "adaptive_nomerge":   "Adaptive (no merge)",
    "adaptive_batchonly": "Adaptive (batch only)",
    "adaptive_random":    "Random adaptation",
    "adaptive_noqos":     "Adaptive (no QoS admit)",
    "adaptive_nofeas":    "Adaptive (no feasibility)",
    "adaptive_nogate":    "Adaptive (no commit gate)",
    "adaptive_raw":       "Adaptive (raw signals)",
    "threshold":          "Threshold heuristic",
    "threshold_raw":      "Threshold (raw signals)",
}


def make_policy_scheduler(policy_key: str, cfg: SimConfig,
                          rng: np.random.Generator):
    from dataclasses import replace
    sched_key, _, is_adaptive = POLICIES[policy_key]
    if policy_key in ABLATIONS:
        cfg = replace(cfg, **ABLATIONS[policy_key])
    if policy_key == "threshold_raw":
        return ThresholdHeuristicStealing(cfg, rng)
    if sched_key == "threshold":
        return ThresholdHeuristicStealing(cfg, rng)
    if is_adaptive:
        return AdaptiveQoSStealing(cfg, rng)
    return make_scheduler(sched_key, cfg, rng)


def generation_level_for(policy_key: str, cfg: SimConfig) -> int:
    _, level_name, _ = POLICIES[policy_key]
    return _level_indices(cfg)[level_name]
