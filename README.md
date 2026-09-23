# Adaptive Task Granularity — Simulator and Reproduction Package

This repository contains the discrete-event simulator, experiment driver,
statistical-analysis scripts, and canonical dataset behind the paper
*"Adaptive Task Granularity for QoS-Aware Distributed Work Stealing in
Heterogeneous IoT Clusters."* Everything reported in the paper — every table,
figure, and numerical claim — can be regenerated from the single canonical
dataset in `data/results_v3.json` using the scripts here.

The code is plain Python with a small scientific stack. There is no framework
to learn: one module defines the model, one runs the simulation, one drives the
experiments, and one does the statistics.

## 1. What is in the box

```
granularity-sim/
├── README.md            ← this file
├── requirements.txt     ← Python dependencies
├── src/
│   ├── models.py            core data types (Device, Task, SimConfig)
│   ├── schedulers.py        baseline schedulers (static, QoS-aware stealing)
│   ├── simulation.py        the discrete-event engine
│   ├── granularity.py       the adaptive-granularity framework + controller
│   ├── run_granularity.py   experiment driver (RQ1–RQ13)
│   ├── analyze_stats.py     paired statistics from a results file
│   ├── make_tikz.py         emit pgfplots figures + summary table
│   └── plot_psi_trace.py    emit the controller-trace figure
├── data/
│   └── results_v3.json      canonical dataset: 14 blocks, 30 seeds each
└── figures/                 (created when you generate figures)
```

## 2. Setup

Requires **Python 3.10+**. Create an environment and install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate          # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The dependencies are `numpy` and `scipy` (used by the statistics). The
simulator core itself needs only the standard library.

## 3. Quickest thing you can do (2 minutes)

Reproduce the statistical report from the shipped canonical dataset — no
simulation required, because `data/results_v3.json` already contains every
per-seed result reported in the paper:

```bash
cd src
python3 analyze_stats.py --in ../data/results_v3.json --out ../stats_v3.md
```

This writes `stats_v3.md` with the paired Wilcoxon tests, bootstrap confidence
intervals, and Cliff's delta effect sizes for every research question.

## 4. Running the simulation yourself

The driver runs one research question or all of them. Each RQ is a
self-contained experiment (see the paper's Section 8). Start small:

```bash
cd src
# one RQ, few seeds, quick smoke test (~1 minute)
python3 run_granularity.py --rq 1 --seeds 5 --out /tmp/smoke.json

# one RQ at full statistical power (30 seeds), parallelised
python3 run_granularity.py --rq 6 --seeds 30 --jobs 8 --out /tmp/rq6.json

# the whole suite at full power (this is how results_v3.json was produced)
python3 run_granularity.py --rq all --seeds 30 --jobs 8 --out results_v3.json
```

Arguments:

| flag        | meaning                                             | default            |
|-------------|-----------------------------------------------------|--------------------|
| `--rq`      | which research question: `1`–`13`, or `all`         | `all`              |
| `--seeds`   | number of random seeds (30 for paper-grade results) | `5`                |
| `--jobs`    | parallel worker processes                           | `1`                |
| `--out`     | output JSON path                                    | `results_gran.json`|
| `--trace`   | also emit a per-tick controller trace (for RQ-trace)| off                |

The full 13-RQ suite at 30 seeds takes on the order of an hour on 8 cores;
individual RQs take from under a minute to a few minutes.

> **Reproducibility note.** Each run is seeded, so a given `--seeds N` produces
> the same numbers on every machine (differences of one ULP aside). The shipped
> `data/results_v3.json` is the exact dataset the paper reports; re-running
> `--rq all --seeds 30` reproduces it.

## 5. Regenerating the figures

The paper's figures are native pgfplots (TikZ), emitted as `.tex` you include
directly in the manuscript:

```bash
cd src
python3 make_tikz.py                # reads ../data/results_v3.json
                                    # writes gran_figures.tex + gran_figs.json
python3 plot_psi_trace.py --in ../data/psi_trace.json   # controller-trace figure
```

`make_tikz.py` reads the canonical dataset and emits every evaluation figure
and the summary table. (The controller-trace figure needs a trace file produced
with `run_granularity.py --rq 8 --trace`, which writes `psi_trace.json`.)

## 6. A note on the dataset

`data/results_v3.json` is the **single source of truth**. It holds 14 blocks
(one per research question, RQ1 split into distribution and size sub-blocks),
each with 30 per-seed raw results per policy. If you re-run experiments, write
to a *different* `--out` file and compare, rather than overwriting the shipped
dataset — that way the canonical numbers stay intact and auditable.

## 7. Understanding the code

If you want to read rather than run, the natural order is:

1. `models.py` — the vocabulary: what a `Device`, a `Task`, and a `SimConfig`
   are.
2. `simulation.py` — the tick-based engine that advances the cluster.
3. `schedulers.py` — the baselines the adaptive policy is compared against.
4. `granularity.py` — the contribution: the split/merge operators, the
   normalised multi-signal pressure controller, the two-gate commit, and the
   registry of policy variants (including the ablations used in RQ6 and RQ11).
5. `run_granularity.py` — how each research question is set up and measured.

## 8. Citation

If you use this code or dataset, please cite the paper (see `CITATION.cff` /
the manuscript). This work builds on the QoS-aware distributed work-stealing
model of Coelho and Nogueira (*Wireless Networks*, 2026).
