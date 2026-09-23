"""
run_granularity.py — Evaluation runner for the adaptive task-granularity paper.

Runs one research question per invocation (or all of them) and appends
aggregated results (median + interquartile range over seeds) to a shared JSON
file, so figures can be regenerated without re-running the simulations.

    python run_granularity.py --rq all --seeds 5 --out results_gran.json
    python run_granularity.py --rq 2 --seeds 10 --jobs 8

Six policies are compared (see granularity.POLICIES):
    Static coarse/fine, QoS-WS coarse/medium/fine, Adaptive QoS-WS.

Determinism: every individual run is seeded independently (the RNGs are
constructed inside the run from cfg.seed), so results are identical whether
runs execute sequentially or in parallel via --jobs.
"""
from __future__ import annotations
import argparse, copy, json, os, sys, time
from typing import Dict, List
import numpy as np

from models import SimConfig
from simulation import run_simulation, generate_devices
from granularity import (POLICIES, POLICY_LABELS, make_policy_scheduler,
                         generation_level_for, generate_tasks_gran)

# number of parallel worker processes (set from --jobs in main)
N_JOBS = 1

METRICS = ["makespan", "comm_cost", "ticks_overhead", "completion_rate",
           "time_to_u_desired", "time_to_u_min", "qos_violations",
           "avg_utilisation", "n_splits", "n_merges", "load_imbalance",
           "fairness_jain", "energy_proxy", "bytes_data", "n_tasks_final",
           "avg_gran", "t95", "t99", "u_at_ddl", "wdmr"]


def run_policy(cfg: SimConfig, policy_key: str):
    rng = np.random.default_rng(cfg.seed)
    devices = generate_devices(cfg, rng)
    level = generation_level_for(policy_key, cfg)
    tasks = generate_tasks_gran(cfg, rng, level)
    sched = make_policy_scheduler(policy_key, cfg,
                                  np.random.default_rng(cfg.seed + 1))
    return run_simulation(cfg, sched, devices, tasks)


def _one_run(args):
    """Worker: run one (config-overrides, policy, seed) combination."""
    base_dict, ov, pk, s = args
    c = SimConfig(**base_dict)
    for k, v in ov.items():
        setattr(c, k, v)
    c.seed = s
    m = run_policy(c, pk)
    vals = {mt: float(getattr(m, mt)) for mt in METRICS}
    vals["_hit_horizon"] = bool(m.makespan >= c.max_ticks)
    return vals


def run_grid(base: SimConfig, points: List[tuple], seeds: List[int],
             label_fn, policies: List[str] = None) -> Dict:
    """points: list of (point_label, overrides_dict)."""
    from dataclasses import asdict
    from granularity import CORE_POLICIES
    pols = policies if policies is not None else CORE_POLICIES
    base_dict = asdict(base)
    # Build the full work list, then execute (optionally in parallel).
    work, keys = [], []
    for plabel, ov in points:
        for pk in pols:
            for s in seeds:
                work.append((base_dict, ov, pk, s))
                keys.append((plabel, pk))
    if N_JOBS > 1:
        import multiprocessing as mp
        with mp.Pool(N_JOBS) as pool:
            outs = pool.map(_one_run, work, chunksize=1)
    else:
        outs = [_one_run(wi) for wi in work]

    # Aggregate per (point, policy)
    grid: Dict = {}
    per: Dict = {}
    for (plabel, pk), vals in zip(keys, outs):
        per.setdefault((plabel, pk), []).append(vals)
    for plabel, _ in points:
        grid[plabel] = {}
        for pk in pols:
            runs = per[(plabel, pk)]
            agg = {}
            for mt in METRICS:
                v = np.array([r[mt] for r in runs], dtype=float)
                agg[mt] = {
                    "median": float(np.median(v)),
                    "q25": float(np.percentile(v, 25)),
                    "q75": float(np.percentile(v, 75)),
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                    "raw": [float(x) for x in v],
                }
            agg["_censored"] = bool(any(r["_hit_horizon"] for r in runs))
            grid[plabel][pk] = agg
            mk = agg["makespan"]["median"]
            cc = agg["comm_cost"]["median"]
            print(f"    {plabel:>14} {POLICY_LABELS[pk]:<16} "
                  f"mk={mk:7.0f} comm={cc:7.0f} "
                  f"sp={agg['n_splits']['median']:.0f} "
                  f"mg={agg['n_merges']['median']:.0f}"
                  f"{'  *censored' if agg['_censored'] else ''}")
    return grid


# ── Research questions ────────────────────────────────────────────────────────

def rq1(seeds, results):
    print("\n-- RQ1: Performance (makespan) --")
    # RQ1a: makespan by distribution at n=16
    base = SimConfig(n_devices=16, n_tasks=500, task_cv=1.0, max_ticks=40000)
    pts = [("uniform", {"task_dist": "uniform"}),
           ("lognormal", {"task_dist": "lognormal"}),
           ("pareto", {"task_dist": "pareto"})]
    from granularity import CORE_POLICIES
    pol_oracle = CORE_POLICIES + ["qosws_g10", "qosws_g40"]
    results["rq1a_dist"] = run_grid(base, pts, seeds, None,
                                    policies=pol_oracle)
    # RQ1b: makespan vs cluster size (lognormal)
    pts2 = [(str(n), {"n_devices": n, "n_tasks": 30 * n})
            for n in [4, 8, 16, 32]]
    base2 = SimConfig(task_dist="lognormal", task_cv=1.0, max_ticks=40000)
    results["rq1b_size"] = run_grid(base2, pts2, seeds, None,
                                    policies=pol_oracle)


def rq2(seeds, results):
    print("\n-- RQ2: Communication cost (latency sweep) --")
    base = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                     task_cv=1.0, max_ticks=40000)
    pts = [(str(l), {"lat_scale": float(l)}) for l in [1, 3, 5, 8, 12]]
    results["rq2_latency"] = run_grid(base, pts, seeds, None)


def rq3(seeds, results):
    print("\n-- RQ3: QoS preservation (strictness sweep) --")
    base = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                     task_cv=1.2, max_ticks=40000)
    pts = [("u0.80", {"u_desired": 0.80}),
           ("u0.90", {"u_desired": 0.90}),
           ("u0.95", {"u_desired": 0.95})]
    results["rq3_qos"] = run_grid(base, pts, seeds, None)


def rq4(seeds, results):
    print("\n-- RQ4: Robustness (failure models) --")
    base = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                     task_cv=1.0, max_ticks=40000)
    # Failure parameters aligned with the accepted QoS-WS paper:
    # transient: fail 1e-3 / recover 5e-2 per tick; permanent: fail 1.5e-4.
    pts = [("none", {"failure_model": "none"}),
           ("transient", {"failure_model": "transient",
                          "fail_prob": 1e-3, "recover_prob": 5e-2}),
           ("permanent", {"failure_model": "permanent", "fail_prob": 1.5e-4})]
    results["rq4_failures"] = run_grid(base, pts, seeds, None)


def rq5(seeds, results):
    print("\n-- RQ5: Sensitivity (heterogeneity) --")
    base = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                     task_cv=1.0, max_ticks=40000)
    pts = [("low", {"het_scale": 0.5}),
           ("med", {"het_scale": 1.0}),
           ("high", {"het_scale": 2.0})]
    results["rq5_het"] = run_grid(base, pts, seeds, None)


def rq6(seeds, results):
    print("\n-- RQ6: Ablations and adaptive controls --")
    pols = ["adaptive", "adaptive_noQ", "adaptive_noF", "adaptive_noO",
            "adaptive_nosplit", "adaptive_nomerge", "adaptive_batchonly",
            "adaptive_random", "threshold", "qosws_medium"]
    base = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                     task_cv=1.0, max_ticks=40000)
    pts = [("none", {"failure_model": "none"}),
           ("transient", {"failure_model": "transient",
                          "fail_prob": 1e-3, "recover_prob": 5e-2}),
           ("permanent", {"failure_model": "permanent", "fail_prob": 1.5e-4}),
           ("lat_x8", {"lat_scale": 8.0}),
           # conflicting-signal regime: expensive network AND churn, where
           # O argues for coarsening while F argues for refinement --- the
           # scenario in which weighted arbitration should differ from a
           # single-threshold rule.
           ("lat_x8_transient", {"lat_scale": 8.0,
                                 "failure_model": "transient",
                                 "fail_prob": 1e-3,
                                 "recover_prob": 5e-2})]
    results["rq6_ablation"] = run_grid(base, pts, seeds, None, policies=pols)


def rq7(seeds, results):
    print("\n-- RQ7: Strong scaling (fixed total work) --")
    # Fixed total work: 1200 medium tasks (~24000 work units) for every n.
    pols = ["qosws_coarse", "qosws_medium", "qosws_fine", "adaptive"]
    base = SimConfig(n_tasks=1200, task_dist="lognormal", task_cv=1.0,
                     max_ticks=60000)
    pts = [(str(n), {"n_devices": n, "qos_deadline": 96000 // n})
           for n in [4, 8, 16, 32]]
    results["rq7_strong"] = run_grid(base, pts, seeds, None, policies=pols)


def run_trace(out_path: str):
    """Single representative adaptive run with per-tick Psi logging, for the
    four-panel controller-response figure (transient churn regime)."""
    print("\n-- Psi trace run (n=16, lognormal, transient failures) --")
    from granularity import (make_policy_scheduler, generation_level_for,
                             generate_tasks_gran)
    from simulation import generate_devices
    from simulation import run_simulation
    cfg = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                    task_cv=1.0, max_ticks=40000, seed=42,
                    failure_model="transient", fail_prob=1e-3,
                    recover_prob=5e-2, trace_psi=True)
    rng = np.random.default_rng(cfg.seed)
    devices = generate_devices(cfg, rng)
    level = generation_level_for("adaptive", cfg)
    tasks = generate_tasks_gran(cfg, rng, level)
    sched = make_policy_scheduler("adaptive", cfg,
                                  np.random.default_rng(cfg.seed + 1))
    m = run_simulation(cfg, sched, devices, tasks)
    with open(out_path, "w") as f:
        json.dump({"config": "n16_lognormal_transient",
                   "makespan": float(m.makespan),
                   "trace": sched.psi_trace}, f)
    print(f"trace: {len(sched.psi_trace)} ticks -> {out_path}")


def rq8(seeds, results):
    print("\n-- RQ8: Phased workloads (regime transitions) --")
    pols = ["adaptive", "threshold", "adaptive_random", "qosws_medium",
            "qosws_fine"]
    base = SimConfig(n_devices=16, n_tasks=900, task_dist="lognormal",
                     task_cv=1.0, max_ticks=40000, qos_deadline=3600)
    pts = [("net_phases", {"phase_schedule": ((700, {"lat_scale": 8.0}),
                                              (1400, {"lat_scale": 1.0}))}),
           ("churn_phases", {"phase_schedule": ((700, {"failure_model": "transient",
                                                       "fail_prob": 1e-3,
                                                       "recover_prob": 5e-2}),
                                                (1400, {"failure_model": "none"}))}),
           ("net_then_churn", {"phase_schedule": ((600, {"lat_scale": 8.0}),
                                                  (1200, {"failure_model": "transient",
                                                          "fail_prob": 1e-3,
                                                          "recover_prob": 5e-2}),
                                                  (1800, {"lat_scale": 1.0,
                                                          "failure_model": "none"}))})]
    results["rq8_phases"] = run_grid(base, pts, seeds, None, policies=pols)


def rq9(seeds, results):
    print("\n-- RQ9: Observation staleness and estimation noise --")
    # Stress the controller's inputs: heartbeat interval (view lag), heartbeat
    # loss (view gaps), and multiplicative noise on the decision signals.
    pols = ["adaptive", "threshold", "qosws_medium"]
    # Anchor in the conflicting-signal regime (expensive network + churn),
    # where the pressure lives near the dead-band and observation quality
    # actually changes decisions; in benign regimes the imbalance signal
    # saturates its cap and the controller is insensitive to staleness.
    base = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                     task_cv=1.0, max_ticks=40000, lat_scale=8.0,
                     failure_model="transient", fail_prob=1e-3,
                     recover_prob=5e-2)
    pts = ([("clean", {})] +
           [(f"hb_x{k}", {"hb_interval": 10 * k}) for k in [2, 5, 10]] +
           [(f"hbloss_{int(p*100)}", {"hb_loss": p})
            for p in [0.05, 0.10, 0.20]] +
           [(f"noise_{int(s*100)}", {"est_noise": s})
            for s in [0.1, 0.2, 0.3]])
    results["rq9_staleness"] = run_grid(base, pts, seeds, None, policies=pols)


def rq10(seeds, results):
    print("\n-- RQ10: Controller parameter sensitivity (OFAT) --")
    pols = ["adaptive"]
    # Anchor in the conflicting-signal regime: in benign regimes the imbalance
    # signal saturates its cap and every controller parameter is inert, so a
    # meaningful sensitivity study must stress the regime where the pressure
    # lives near its decision boundaries.
    base = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                     task_cv=1.0, max_ticks=40000, lat_scale=8.0,
                     failure_model="transient", fail_prob=1e-3,
                     recover_prob=5e-2)
    pts = [("base", {})]
    for k, vals in [("eta_I", [0.5, 2.0]), ("eta_O", [0.5, 2.0]),
                    ("eta_Q", [0.5, 2.0]), ("eta_F", [0.5, 2.0]),
                    ("psi_deadband", [0.05, 0.30]),
                    ("gran_cooldown", [10, 90]),
                    ("i_ref", [1.25, 5.0]), ("o_ref", [0.11, 0.44])]:
        for v in vals:
            pts.append((f"{k}={v}", {k: v}))
    results["rq10_sensitivity"] = run_grid(base, pts, seeds, None,
                                           policies=pols)


def rq11(seeds, results):
    print("\n-- RQ11: QoS admission as a structural safeguard --")
    # In the progress-based utility instantiation, stealing never lowers
    # utility, so the admission gate is vacuously satisfied. We report this
    # directly: adaptive with and without the gate are compared under
    # deadline stress, and the (expected) equality is the finding --- the
    # gate is a structural safeguard for non-monotone utility models, not an
    # active constraint here.
    pols = ["adaptive", "adaptive_noqos", "adaptive_nofeas", "adaptive_nogate",
            "qosws_medium"]
    base = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                     task_cv=1.0, max_ticks=40000, u_desired=0.95,
                     qos_deadline=900)
    pts = [("tight", {"qos_deadline": 900}),
           ("mid", {"qos_deadline": 1200}),
           ("loose", {"qos_deadline": 2000})]
    results["rq11_qosadmit"] = run_grid(base, pts, seeds, None, policies=pols)


def rq12(seeds, results):
    print("\n-- RQ12: Normalisation ablation (2x2) --")
    # Normalised vs raw signals, for both the weighted controller and the
    # threshold rule, in the conflict regime.
    pols = ["adaptive", "adaptive_raw", "threshold", "threshold_raw",
            "qosws_medium"]
    base = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                     task_cv=1.0, max_ticks=40000, lat_scale=8.0,
                     failure_model="transient", fail_prob=1e-3,
                     recover_prob=5e-2)
    pts = [("conflict", {})]
    results["rq12_norm"] = run_grid(base, pts, seeds, None, policies=pols)


def rq13(seeds, results):
    print("\n-- RQ13: Cost-model mismatch (real != predicted) --")
    # The controller's estimates use nominal costs; the simulator executes
    # with a hidden per-task multiplier. This breaks the circularity RQ9
    # could not.
    pols = ["adaptive", "threshold", "qosws_medium"]
    base = SimConfig(n_devices=16, n_tasks=500, task_dist="lognormal",
                     task_cv=1.0, max_ticks=40000)
    pts = [("clean", {"exec_mismatch": 0.0})] + \
          [(f"mm_{int(s*100)}", {"exec_mismatch": s})
           for s in [0.2, 0.4, 0.6]]
    results["rq13_mismatch"] = run_grid(base, pts, seeds, None, policies=pols)


RQ_FNS = {1: rq1, 2: rq2, 3: rq3, 4: rq4, 5: rq5, 6: rq6, 7: rq7,
          8: rq8, 9: rq9, 10: rq10, 11: rq11, 12: rq12, 13: rq13}


def main():
    global N_JOBS
    # On Windows, piped/console output may default to cp1252; make output
    # encoding-safe regardless of platform and locale.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="Run the adaptive-granularity evaluation (RQ1-RQ5).")
    ap.add_argument("--rq", default="all",
                    help="which research question to run: 1..7 or 'all'")
    ap.add_argument("--seeds", type=int, default=5,
                    help="number of random seeds per configuration (default 5)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel worker processes (default 1; "
                         "use e.g. the number of physical cores)")
    ap.add_argument("--out", default="results_gran.json",
                    help="output JSON file (appended/updated per RQ)")
    ap.add_argument("--trace", action="store_true",
                    help="run the single Psi-trace scenario and exit")
    args = ap.parse_args()
    N_JOBS = max(1, args.jobs)
    if args.trace:
        run_trace("psi_trace.json")
        return
    seeds = list(range(42, 42 + args.seeds))

    results = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            results = json.load(f)

    t0 = time.time()
    if args.rq == "all":
        for n in sorted(RQ_FNS):
            RQ_FNS[n](seeds, results)
    else:
        RQ_FNS[int(args.rq)](seeds, results)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nSaved {args.out}  ({time.time()-t0:.0f}s, seeds={seeds})")


if __name__ == "__main__":
    main()
