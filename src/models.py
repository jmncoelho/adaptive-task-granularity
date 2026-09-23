"""
models.py — Task, Device, and SimConfig data models.

Notation follows the paper:
  rho_i(t)  = residual capacity
  P_i(t)    = device performance index
  W_i       = local work deque  (mathcal{W}_i in the paper)
  lambda_i  = normalised load
"""

from __future__ import annotations
from collections import deque, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional
import numpy as np


# ── Enumerations ──────────────────────────────────────────────────────────────

class TaskState(Enum):
    CREATED   = "created"
    QUEUED    = "queued"
    RUNNING   = "running"
    LEASED    = "leased"
    COMPLETED = "completed"
    FAILED    = "failed"
    EXPIRED   = "expired"


class SchedulerType(Enum):
    STATIC_PROP    = "Static Proportional"
    ENHANCED_PROP  = "Enhanced Proportional"
    CENTRAL_DYN    = "Centralized Dynamic"
    RANDOM_STEAL   = "Random Stealing"
    QOS_STEAL      = "QoS-Aware Stealing"


# ── Simulation configuration ──────────────────────────────────────────────────

@dataclass
class SimConfig:
    # ── Cluster
    n_devices: int   = 8
    # ── Tasks
    n_tasks: int     = 500
    task_dist: str   = "lognormal"   # "uniform" | "lognormal" | "pareto"
    task_mean: float = 20.0          # mean cost in ticks
    task_cv: float   = 1.0           # coefficient of variation
    # ── QoS
    u_min: float     = 0.4
    u_desired: float = 0.9
    # ── Work-stealing parameters
    lambda_low: float  = 0.5
    lambda_high: float = 2.5
    lambda_star: float = 1.2
    b_max: int         = 4
    lease_dur: int     = 100         # ticks
    hb_interval: int   = 10          # heartbeat interval in ticks
    backoff_base: int  = 5
    backoff_max: int   = 160
    # ── Failure model
    failure_model: str  = "none"     # "none" | "transient" | "permanent"
    fail_prob: float    = 0.001      # per device per tick
    recover_prob: float = 0.05       # per failed device per tick (transient)
    # ── Performance index weights (alpha, beta, gamma, delta)
    w_cpu: float  = 0.5
    w_mem: float  = 0.2
    w_bat: float  = 0.2
    w_lat: float  = 0.1
    # ── QoS utility dimension weights
    qos_weights: tuple = (0.6, 0.4)
    # ── Simulation control
    max_ticks: int = 30_000
    seed: int      = 42
    # ── Enhanced proportional: number of rounds
    n_rounds: int  = 4

    # ══════════════════════════════════════════════════════════════════════
    # ADAPTIVE TASK GRANULARITY EXTENSION
    # ══════════════════════════════════════════════════════════════════════
    # Ordered granularity ladder G = {g_1, ..., g_h}, finest first.
    # Each value is the *mean task cost in ticks* at that granularity level.
    # Index 0 = finest, last = coarsest. Default mirrors the suggestion
    # G = {1, 5, 15, 60}s in the follow-up paper, in simulator ticks.
    gran_levels: tuple = (5.0, 10.0, 20.0, 40.0, 60.0)
    # Initial generation level (index into gran_levels). For fixed-granularity
    # policies this fixes the task size; the adaptive policy starts here.
    gran_init: int = 2
    # Total service work is held fixed across granularities so that policies
    # at different granularities solve the *same* problem. If 0, falls back to
    # n_tasks * gran_levels[gran_init].
    service_work: float = 0.0

    # One-off scheduling / metadata overhead charged per task the first time it
    # executes (this is T_sched(g) in the cost model). Because finer granularity
    # produces more tasks, this is what makes fine decomposition genuinely
    # costlier in both makespan and bookkeeping.
    sched_overhead: float = 0.6
    # Per-task control framing bytes for a transfer (on top of the 128 B per
    # steal message). Finer granularity -> more tasks moved -> more framing.
    frame_bytes: float = 48.0

    # ── Granularity-pressure weights  Psi = eta_I I - eta_O O + eta_Q Q + eta_F F
    # Defaults match the worked example in the paper (all unit weights).
    eta_I: float = 1.0   # load imbalance      -> finer
    eta_O: float = 1.0   # communication cost  -> coarser
    eta_Q: float = 1.0   # QoS pressure        -> finer
    eta_F: float = 1.0   # failure pressure    -> finer (less work lost per loss)

    # ── Fase-B evaluation switches (ablations / controls / tracing) ─────────
    adapt_no_split: bool = False   # ablation: disable the split operator
    adapt_no_merge: bool = False   # ablation: disable the merge operator
    adapt_random: bool   = False   # negative control: Psi := U(-1,1) noise
    trace_psi: bool      = False   # record per-tick Psi components

    # ── Reviewer-round-2 controller revisions & new experiment knobs ────────
    qos_deadline: Optional[int] = 2000  # service deadline for the QoS-risk signal
    est_noise: float = 0.0     # multiplicative noise sigma on the decision signals
    hb_loss: float = 0.0       # per-heartbeat drop probability (staleness stress)
    phase_schedule: Optional[tuple] = None  # ((tick, {field: value}), ...)

    # ── Reviewer-round-3 ablation / robustness knobs ───────────────────────
    no_qos_admit: bool = False     # ablation: skip the QoS-safety commit gate
    no_feasibility: bool = False   # ablation: skip the feasibility commit checks
    raw_signals: bool = False      # ablation: use unnormalised pressure signals
    exec_mismatch: float = 0.0     # sigma of real vs predicted exec-time error
    bw_mismatch: float = 0.0       # sigma of real vs predicted bandwidth error
    # Anti-oscillation: minimum ticks a task must wait between successive
    # adaptive granularity changes, and a dead-band around Psi=0.
    gran_cooldown: int = 30
    psi_deadband: float = 0.15
    # Normalisation references so the components are comparable and the unit
    # eta weights remain interpretable. I is capped below 1 because some load
    # dispersion is intrinsic to a heterogeneous cluster and should not, on its
    # own, force maximal splitting; O can then dominate under expensive comm.
    i_ref: float = 2.5    # I_n = clip(I / i_ref, 0, i_cap)
    i_cap: float = 0.8
    o_ref: float = 0.22   # O_n = clip(O_raw / o_ref, 0, 1)
    # Latency profile multiplier (EQ5 sensitivity). Scales the generated base
    # communication latency of every device; >1 models an expensive network.
    lat_scale: float = 1.0
    # Heterogeneity multiplier (EQ5). >1 widens the CPU/memory spread.
    het_scale: float = 1.0


# ── Task ──────────────────────────────────────────────────────────────────────

class Task:
    __slots__ = (
        "id", "cost", "remaining", "state",
        "owner", "lease_epoch", "lease_expiry",
        "created_at", "started_at", "finished_at",
        "retries", "max_retries", "qos_weight",
        # ── Adaptive-granularity extensions ──────────────────────────────
        "gran",          # current granularity level Gamma(tau): index into G
        "root_id",       # id of the original (pre-split) task — idempotence key
        "data_lo",       # left edge of the data slice this task covers in [0,1]
        "data_hi",       # right edge of the data slice (data_hi-data_lo = share)
        "sched_charged", # whether the one-off T_sched(g) overhead was charged
        "exec_factor",   # real/predicted execution ratio (model mismatch, hidden from controller)
    )

    def __init__(self, task_id: int, cost: float,
                 created_at: int = 0,
                 qos_weight: float = 1.0,
                 max_retries: int = 3,
                 gran: int = 1,
                 root_id: Optional[int] = None,
                 data_lo: float = 0.0,
                 data_hi: float = 1.0):
        self.id          = task_id
        self.cost        = cost
        self.remaining   = cost
        self.state       = TaskState.CREATED
        self.owner: Optional[int] = None
        self.lease_epoch: int     = 0
        self.lease_expiry: int    = 0
        self.created_at           = created_at
        self.started_at: Optional[int]  = None
        self.finished_at: Optional[int] = None
        self.retries    = 0
        self.max_retries = max_retries
        self.qos_weight  = qos_weight
        # ── Granularity state ──────────────────────────────────────────────
        self.gran        = gran
        self.root_id     = root_id if root_id is not None else task_id
        self.data_lo     = data_lo
        self.data_hi     = data_hi
        self.sched_charged = False
        self.exec_factor   = 1.0

    @property
    def latency(self) -> Optional[float]:
        if self.finished_at is not None and self.started_at is not None:
            return self.finished_at - self.started_at
        return None

    def __repr__(self):
        return f"Task({self.id}, cost={self.cost:.1f}, state={self.state.value})"


# ── Device ────────────────────────────────────────────────────────────────────

@dataclass
class DeviceSpec:
    """Static hardware parameters for one device."""
    cpu_speed: float        # GHz
    n_cores: int
    memory_gb: float        # total RAM in GB
    is_mains: bool          # True → battery_level always 1.0
    latency_ms: float       # base communication latency in ms
    bandwidth_mbps: float
    guard_cpu_frac: float   # fraction of available CPU reserved for primary
    guard_mem_gb: float     # GB reserved for primary


class Device:
    """
    Models one IoT device in the cluster.

    Dynamic resource state is updated each tick by update_resources().
    The work deque (W_i) is a collections.deque:
        right end = bottom  (owner pops and pushes here)
        left  end = top     (thieves steal from here)
    """

    def __init__(self, dev_id: int, spec: DeviceSpec, cfg: SimConfig):
        self.id   = dev_id
        self.spec = spec
        self.cfg  = cfg

        # ── Dynamic resource state (reset each tick) ──────────────────────
        self.cpu_avail: float = 0.5
        self.mem_avail: float = spec.memory_gb * 0.5
        self.battery:   float = 1.0 if spec.is_mains else 0.8

        # ── Cluster membership ────────────────────────────────────────────
        self.active: bool = True
        self.failed_at: int = -1

        # ── Work deque  W_i ───────────────────────────────────────────────
        self.W: Deque[Task] = deque()
        self.current_task: Optional[Task] = None

        # ── Stealing control ──────────────────────────────────────────────
        self.backoff_until: int  = 0
        self.backoff_count: int  = 0

        # ── Heartbeat tracking {source_id: [tick, tick, …]} ──────────────
        self.hb_log: Dict[int, List[int]] = defaultdict(list)

        # ── Lease epoch counters {thief_id: epoch} ────────────────────────
        self.epoch_for: Dict[int, int] = defaultdict(int)

        # ── Per-device metrics ────────────────────────────────────────────
        self.ticks_busy: int   = 0
        self.ticks_idle: int   = 0
        self.steal_attempts:  int = 0
        self.steal_successes: int = 0
        self.steal_grants:    int = 0
        self.steal_denials:   int = 0
        self.tasks_done:      int = 0
        self.bytes_xfer:      float = 0.0
        # Adaptive-granularity bookkeeping
        self.ticks_overhead:  float = 0.0   # accumulated T_sched paid here
        self.n_splits:        int = 0
        self.n_merges:        int = 0

    # ── Resource helpers ──────────────────────────────────────────────────────

    def update_resources(self, rng: np.random.Generator) -> None:
        """Simulate fluctuating primary workload each tick."""
        if not self.active:
            return
        self.cpu_avail = float(np.clip(rng.normal(0.45, 0.2), 0.05, 0.95))
        self.mem_avail = float(np.clip(
            rng.uniform(0.2, 0.9) * self.spec.memory_gb,
            0.1, self.spec.memory_gb))
        if not self.spec.is_mains:
            self.battery = max(0.0, self.battery - 0.0001)

    @property
    def perf_index(self) -> float:
        """P_i(t) = alpha*(C^s * C^c * C^a) + beta*M + gamma*B - delta*L"""
        c = self.spec.cpu_speed * self.spec.n_cores * max(0.01, self.cpu_avail)
        m = min(1.0, self.mem_avail / max(0.01, self.spec.memory_gb))
        b = self.battery
        l = min(1.0, self.spec.latency_ms / 50.0)
        return (self.cfg.w_cpu * c + self.cfg.w_mem * m +
                self.cfg.w_bat * b - self.cfg.w_lat * l)

    @property
    def residual(self) -> float:
        """rho_i(t) = min over resources of (available - guard) / available"""
        cpu_r = max(0.0, (self.cpu_avail - self.spec.guard_cpu_frac) /
                    max(0.01, self.cpu_avail))
        mem_r = max(0.0, (self.mem_avail - self.spec.guard_mem_gb) /
                    max(0.01, self.mem_avail))
        return min(cpu_r, mem_r)

    @property
    def eff_capacity(self) -> float:
        """rho_i * P_i"""
        return self.residual * self.perf_index

    def norm_load(self, eps: float = 0.01) -> float:
        """lambda_i(t) = sum(w_hat) / max(eps, rho*P)"""
        pending = sum(t.remaining for t in self.W)
        if self.current_task:
            pending += self.current_task.remaining
        return pending / max(eps, self.eff_capacity)

    # ── Deque helpers ─────────────────────────────────────────────────────────

    def push_bottom(self, task: Task) -> None:
        task.owner = self.id
        task.state = TaskState.QUEUED
        self.W.append(task)          # append = right = bottom

    def pop_bottom(self) -> Optional[Task]:
        return self.W.pop() if self.W else None   # pop = right = bottom

    def stealable(self) -> List[Task]:
        return [t for t in self.W
                if t.state in (TaskState.QUEUED, TaskState.EXPIRED)]

    # ── Heartbeat & stability ─────────────────────────────────────────────────

    def record_hb(self, source_id: int, tick: int) -> None:
        self.hb_log[source_id].append(tick)

    def kappa(self, source_id: int, tick: int, W: int = 10) -> float:
        """Stability factor: fraction of expected heartbeats actually received."""
        window_start = tick - W * self.cfg.hb_interval
        count = sum(1 for t in self.hb_log[source_id] if t >= window_start)
        return min(1.0, count / max(1, W))

    def heartbeat(self, tick: int) -> dict:
        return {
            "id":       self.id,
            "q_len":    len(self.W),
            "residual": self.residual,
            "perf":     self.perf_index,
            "load":     self.norm_load(),
            "eff_cap":  self.eff_capacity,
            "active":   self.active,
            "tick":     tick,
        }

    def next_epoch(self, thief_id: int) -> int:
        self.epoch_for[thief_id] += 1
        return self.epoch_for[thief_id]

    # ── Failure ───────────────────────────────────────────────────────────────

    def fail(self, tick: int) -> List[Task]:
        """Mark device as failed; return tasks to re-queue."""
        self.active = False
        self.failed_at = tick
        lost = list(self.W)
        self.W.clear()
        if self.current_task and self.current_task.state == TaskState.RUNNING:
            lost.append(self.current_task)
            self.current_task = None
        for t in lost:
            t.state = TaskState.EXPIRED
            t.owner = None
        return lost

    def recover(self) -> None:
        self.active = True
        self.failed_at = -1
        self.cpu_avail = 0.3
        self.mem_avail = self.spec.memory_gb * 0.4
