"""
simulation.py — Simulation engine, device/task generation, metrics collection.
"""

from __future__ import annotations
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import copy

from models import Device, DeviceSpec, Task, TaskState, SimConfig
from schedulers import BaseScheduler


# ═════════════════════════════════════════════════════════════════════════════
# DEVICE GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_devices(cfg: SimConfig,
                     rng: np.random.Generator) -> List[Device]:
    """
    Create n_devices heterogeneous IoT devices.
    Capabilities span roughly Raspberry-Pi-class to laptop-class.
    """
    specs = []
    het = getattr(cfg, "het_scale", 1.0)
    lat_scale = getattr(cfg, "lat_scale", 1.0)
    for i in range(cfg.n_devices):
        # CPU: 1–4 cores at 0.8–3.5 GHz
        cores = rng.choice([1, 2, 4, 4, 8])
        speed = float(rng.uniform(0.8, 3.5))
        if het != 1.0:
            # widen/narrow the speed spread around its midpoint
            mid = 2.15
            speed = float(np.clip(mid + (speed - mid) * het, 0.4, 6.0))
        # Memory: 1–16 GB
        mem = float(rng.choice([1.0, 2.0, 4.0, 8.0, 16.0]))
        # Battery: 20% chance of being mains-powered
        is_mains = bool(rng.random() < 0.3)
        # Latency: 1–50 ms (scaled), bandwidth 10–150 Mbps
        lat = float(rng.uniform(1.0, 50.0)) * lat_scale
        bw  = float(rng.uniform(10.0, 150.0))
        # Guard: 10–40% CPU, 100MB–1GB memory
        guard_cpu = float(rng.uniform(0.10, 0.40))
        guard_mem = float(rng.uniform(0.1, min(1.0, mem * 0.3)))

        specs.append(DeviceSpec(
            cpu_speed=speed, n_cores=cores, memory_gb=mem,
            is_mains=is_mains, latency_ms=lat, bandwidth_mbps=bw,
            guard_cpu_frac=guard_cpu, guard_mem_gb=guard_mem,
        ))

    return [Device(i, specs[i], cfg) for i in range(cfg.n_devices)]


# ═════════════════════════════════════════════════════════════════════════════
# TASK GENERATION
# ═════════════════════════════════════════════════════════════════════════════

def generate_tasks(cfg: SimConfig, rng: np.random.Generator) -> List[Task]:
    """
    Generate n_tasks tasks whose costs follow the configured distribution.
    """
    mean = cfg.task_mean
    cv   = cfg.task_cv
    std  = mean * cv
    n    = cfg.n_tasks

    if cfg.task_dist == "uniform":
        lo = max(1.0, mean - std * 1.73)
        hi = mean + std * 1.73
        costs = rng.uniform(lo, hi, n)

    elif cfg.task_dist == "lognormal":
        sigma2 = np.log(1 + cv ** 2)
        mu = np.log(mean) - sigma2 / 2
        costs = rng.lognormal(mu, np.sqrt(sigma2), n)

    elif cfg.task_dist == "pareto":
        # Pareto with shape alpha such that mean = mean and CV = cv
        # For Pareto with scale xm: mean = alpha*xm/(alpha-1), need alpha > 1
        alpha_p = 1.0 / cv + 1.0 if cv > 0 else 3.0
        alpha_p = max(1.5, alpha_p)
        xm = mean * (alpha_p - 1) / alpha_p
        costs = (rng.pareto(alpha_p, n) + 1) * xm

    else:
        raise ValueError(f"Unknown distribution: {cfg.task_dist}")

    costs = np.clip(costs, 1.0, None)
    # QoS weights uniform in [0.5, 1.5] to simulate heterogeneous importance
    qos_w = rng.uniform(0.5, 1.5, n)
    return [Task(i, float(costs[i]), qos_weight=float(qos_w[i]))
            for i in range(n)]


# ═════════════════════════════════════════════════════════════════════════════
# METRICS
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class RunMetrics:
    scheduler_name: str
    n_devices: int
    n_tasks: int
    task_dist: str
    failure_model: str
    seed: int

    makespan: float              = 0.0
    tasks_completed: int         = 0
    tasks_failed: int            = 0
    completion_rate: float       = 0.0

    qos_final: float             = 0.0
    qos_violations: int          = 0   # ticks where U < U_min
    deadline_miss_ratio: float   = 0.0
    time_to_u_min: int           = -1   # tick when U first >= U_min  (-1 if never)
    time_to_u_desired: int       = -1   # tick when U first >= U_desired
    t95: int                     = -1   # tick when U first >= 0.95
    t99: int                     = -1   # tick when U first >= 0.99
    u_at_ddl: float              = -1.0 # utility at the service QoS deadline
    wdmr: float                  = -1.0 # weighted deadline miss ratio 1-U(ddl)

    avg_task_latency: float      = 0.0
    p95_task_latency: float      = 0.0

    avg_utilisation: float       = 0.0
    avg_idle_frac: float         = 0.0

    steal_attempts: int          = 0
    steal_successes: int         = 0
    steal_failures: int          = 0
    steal_success_rate: float    = 0.0

    bytes_control: float         = 0.0
    bytes_data: float            = 0.0
    comm_cost: float             = 0.0   # latency-weighted task-transfer cost (EQ2)

    energy_proxy: float          = 0.0
    fairness_jain: float         = 0.0  # Jain's fairness index on tasks done

    # ── Adaptive-granularity metrics ──────────────────────────────────────
    n_splits: int                = 0
    n_merges: int                = 0
    ticks_overhead: float        = 0.0   # total T_sched paid across cluster
    load_imbalance: float        = 0.0   # mean I(t) = std(lambda)/mean(lambda)
    n_tasks_final: int           = 0      # leaf-task count at end (after split/merge)
    avg_gran: float              = 0.0    # mean granularity level of executed work

    # Time-series for plotting (sampled every 50 ticks)
    ts_ticks: List[int]          = field(default_factory=list)
    ts_completed: List[int]      = field(default_factory=list)
    ts_qos: List[float]          = field(default_factory=list)
    ts_utilisation: List[float]  = field(default_factory=list)
    ts_steal_rate: List[float]   = field(default_factory=list)


# ═════════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

SAMPLE_INTERVAL = 50   # ticks between metric snapshots

def run_simulation(cfg: SimConfig,
                   scheduler: BaseScheduler,
                   devices: Optional[List[Device]] = None,
                   tasks: Optional[List[Task]] = None) -> RunMetrics:
    """
    Run one simulation instance and return a RunMetrics object.
    If devices/tasks are None they are generated from cfg.
    """
    rng = np.random.default_rng(cfg.seed)

    if devices is None:
        devices = generate_devices(cfg, rng)
    if tasks is None:
        tasks = generate_tasks(cfg, rng)

    # Initialise device dynamic state
    for dev in devices:
        dev.update_resources(rng)

    # Service admission at tick 0
    scheduler.setup(devices, tasks, tick=0)

    # ── Metrics accumulators ──────────────────────────────────────────────
    metrics = RunMetrics(
        scheduler_name=scheduler.name,
        n_devices=cfg.n_devices,
        n_tasks=cfg.n_tasks,
        task_dist=cfg.task_dist,
        failure_model=cfg.failure_model,
        seed=cfg.seed,
    )
    qos_violation_ticks = 0
    total_w = sum(t.qos_weight for t in tasks)
    recent_steals: List[int] = []   # per-tick steal counts for smoothing
    imbalance_samples: List[float] = []

    # ── Main loop ─────────────────────────────────────────────────────────
    makespan = 0
    for tick in range(1, cfg.max_ticks + 1):
        # ── Phased-workload support: apply scheduled config changes ─────
        ps = getattr(cfg, "phase_schedule", None)
        if ps:
            for p_tick, ov in ps:
                if tick == p_tick:
                    for k, v in ov.items():
                        if k == "lat_scale":
                            f = v / max(1e-9, getattr(cfg, "lat_scale", 1.0))
                            for dv in devices:
                                dv.spec.latency_ms *= f
                            cfg.lat_scale = v
                        else:
                            setattr(cfg, k, v)

        # 1. Update resource state
        for dev in devices:
            dev.update_resources(rng)

        # 2. Failure / recovery
        _apply_failures(devices, tasks, cfg, rng, tick)

        # 3. Scheduler step
        scheduler.step(devices, tasks, tick)

        # 4. Check termination
        n_done = sum(1 for t in tasks if t.state == TaskState.COMPLETED)
        n_fail = sum(1 for t in tasks
                     if t.state == TaskState.FAILED
                     or (t.state == TaskState.EXPIRED
                         and t.retries >= t.max_retries))
        if n_done + n_fail >= len(tasks):
            makespan = tick
            break

        # 5. QoS utility
        done_w = sum(t.qos_weight for t in tasks
                     if t.state == TaskState.COMPLETED)
        u_now = done_w / total_w if total_w > 0 else 1.0
        if u_now < cfg.u_min:
            qos_violation_ticks += 1
        # Record first-passage times for U_min and U_desired
        if metrics.time_to_u_min < 0 and u_now >= cfg.u_min:
            metrics.time_to_u_min = tick
        if metrics.time_to_u_desired < 0 and u_now >= cfg.u_desired:
            metrics.time_to_u_desired = tick
        if metrics.t95 < 0 and u_now >= 0.95:
            metrics.t95 = tick
        if metrics.t99 < 0 and u_now >= 0.99:
            metrics.t99 = tick
        ddl_q = getattr(cfg, "qos_deadline", None)
        if ddl_q and tick == ddl_q:
            metrics.u_at_ddl = u_now
            metrics.wdmr = max(0.0, 1.0 - u_now)

        # 6. Time-series sample
        recent_steals.append(scheduler.steal_successes)
        if tick % SAMPLE_INTERVAL == 0:
            active = [d for d in devices if d.active]
            util = (sum(d.ticks_busy for d in active) /
                    max(1, sum(d.ticks_busy + d.ticks_idle for d in active)))
            steal_rate = (recent_steals[-1] - recent_steals[max(0, len(recent_steals) - SAMPLE_INTERVAL - 1)]) / SAMPLE_INTERVAL
            metrics.ts_ticks.append(tick)
            metrics.ts_completed.append(n_done)
            metrics.ts_qos.append(u_now)
            metrics.ts_utilisation.append(util)
            metrics.ts_steal_rate.append(steal_rate)
            # Load imbalance I(t) over devices that still hold work
            loads = np.array([d.norm_load() for d in active
                              if d.norm_load() > 0], dtype=float)
            if loads.size >= 2 and loads.mean() > 1e-9:
                imbalance_samples.append(float(loads.std() / (loads.mean() + 1e-9)))
    else:
        makespan = cfg.max_ticks

    # ── Aggregate metrics ──────────────────────────────────────────────────
    metrics.makespan        = float(makespan)
    metrics.tasks_completed = sum(1 for t in tasks if t.state == TaskState.COMPLETED)
    metrics.tasks_failed    = len(tasks) - metrics.tasks_completed
    metrics.completion_rate = metrics.tasks_completed / len(tasks)

    done_w = sum(t.qos_weight for t in tasks if t.state == TaskState.COMPLETED)
    metrics.qos_final      = done_w / total_w if total_w > 0 else 1.0
    metrics.qos_violations = qos_violation_ticks

    latencies = [t.latency for t in tasks
                 if t.latency is not None and t.state == TaskState.COMPLETED]
    if latencies:
        metrics.avg_task_latency = float(np.mean(latencies))
        metrics.p95_task_latency = float(np.percentile(latencies, 95))

    total_ticks = sum(d.ticks_busy + d.ticks_idle for d in devices)
    total_busy  = sum(d.ticks_busy for d in devices)
    metrics.avg_utilisation = total_busy / max(1, total_ticks)
    metrics.avg_idle_frac   = 1.0 - metrics.avg_utilisation

    metrics.steal_attempts  = scheduler.steal_attempts
    metrics.steal_successes = scheduler.steal_successes
    metrics.steal_failures  = scheduler.steal_failures
    metrics.steal_success_rate = (scheduler.steal_successes /
                                  max(1, scheduler.steal_attempts))

    metrics.bytes_control = scheduler.bytes_control
    metrics.bytes_data    = sum(d.bytes_xfer for d in devices)
    metrics.comm_cost     = float(getattr(scheduler, "comm_cost", 0.0))

    # Energy proxy: CPU ticks × speed × cores
    metrics.energy_proxy = sum(
        d.ticks_busy * d.spec.cpu_speed * d.spec.n_cores
        for d in devices
    )

    # Jain's fairness on tasks completed per device
    done_per = np.array([d.tasks_done for d in devices], dtype=float)
    if done_per.sum() > 0:
        metrics.fairness_jain = (done_per.sum() ** 2 /
                                 (len(devices) * (done_per ** 2).sum() + 1e-9))

    # ── Adaptive-granularity metrics ───────────────────────────────────────
    metrics.n_splits = int(getattr(scheduler, "n_splits", 0)
                           or sum(d.n_splits for d in devices))
    metrics.n_merges = int(getattr(scheduler, "n_merges", 0)
                           or sum(d.n_merges for d in devices))
    metrics.ticks_overhead = float(sum(d.ticks_overhead for d in devices))
    metrics.load_imbalance = (float(np.mean(imbalance_samples))
                              if imbalance_samples else 0.0)
    metrics.n_tasks_final = len(tasks)
    gran_done = [t.gran for t in tasks if t.state == TaskState.COMPLETED]
    metrics.avg_gran = float(np.mean(gran_done)) if gran_done else 0.0

    return metrics


# ─── Failure helper ───────────────────────────────────────────────────────────

def _apply_failures(devices: List[Device], tasks: List[Task],
                    cfg: SimConfig, rng: np.random.Generator,
                    tick: int) -> None:
    if cfg.failure_model == "none":
        return

    for dev in devices:
        if dev.active:
            if rng.random() < cfg.fail_prob:
                lost = dev.fail(tick)
                # Lost tasks remain EXPIRED; they will be re-queued by scheduler
        else:
            # Recovery (only for transient model)
            if cfg.failure_model == "transient":
                if rng.random() < cfg.recover_prob:
                    dev.recover()
