#!/usr/bin/env python3
"""
Statistical analysis for the adaptive-granularity evaluation.

Reads a results JSON produced by run_granularity.py (with per-seed "raw"
arrays) and reports, for each research question's key comparisons:

  * median of each policy and of the paired difference (adaptive - other);
  * 95% bootstrap confidence interval for the median paired difference
    (10,000 resamples);
  * paired Wilcoxon signed-rank test (normal approximation with tie and
    zero corrections -- adequate for n >= 10, recommended n >= 30);
  * Cliff's delta effect size on the paired differences;
  * Holm-Bonferroni correction across each comparison family.

Also computes the retrospective fixed-granularity oracle for RQ1 (the best
fixed QoS-WS level per configuration) and the adaptive policy's distance
to it.

Usage:
    python3 analyze_stats.py --in results_gran.json [--out stats_report.md]

Pure numpy; no scipy required.
"""

import argparse
import json
import math
import sys

import numpy as np

FIXED_LEVELS = ["qosws_fine", "qosws_g10", "qosws_medium", "qosws_g40",
                "qosws_coarse"]

LABELS = {
    "qosws_fine": "fixed g=5", "qosws_g10": "fixed g=10",
    "qosws_medium": "fixed g=20", "qosws_g40": "fixed g=40",
    "qosws_coarse": "fixed g=60", "adaptive": "adaptive",
    "adaptive_noQ": "no QoS pressure", "adaptive_noF": "no failure pressure",
    "adaptive_noO": "no comm pressure", "adaptive_nosplit": "no split",
    "adaptive_nomerge": "no merge", "adaptive_batchonly": "batch only",
    "adaptive_random": "random", "threshold": "threshold heuristic",
    "static_coarse": "static coarse", "static_fine": "static fine",
}


# ── primitives ────────────────────────────────────────────────────────────────

def bootstrap_ci_median(diff, n_boot=10000, alpha=0.05, seed=7):
    """Percentile bootstrap CI for the median of paired differences."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, dtype=float)
    n = len(diff)
    idx = rng.integers(0, n, size=(n_boot, n))
    meds = np.median(diff[idx], axis=1)
    lo, hi = np.percentile(meds, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilcoxon_signed_rank(diff):
    """Paired Wilcoxon signed-rank test, normal approximation with tie and
    zero corrections (Pratt: zeros dropped). Returns (W, z, p_two_sided)."""
    d = np.asarray(diff, dtype=float)
    d = d[d != 0.0]
    n = len(d)
    if n < 5:
        return float("nan"), float("nan"), 1.0
    ranks = _rank(np.abs(d))
    w_pos = float(np.sum(ranks[d > 0]))
    w_neg = float(np.sum(ranks[d < 0]))
    W = min(w_pos, w_neg)
    mu = n * (n + 1) / 4.0
    # tie correction
    _, counts = np.unique(np.abs(d), return_counts=True)
    tie_term = float(np.sum(counts ** 3 - counts)) / 48.0
    sigma2 = n * (n + 1) * (2 * n + 1) / 24.0 - tie_term
    if sigma2 <= 0:
        return W, float("nan"), 1.0
    # continuity correction
    z = (W - mu + 0.5) / math.sqrt(sigma2)
    p = 2.0 * _norm_sf(abs(z))
    return W, float(z), float(min(1.0, p))


def _rank(x):
    """Average ranks (1-based) with ties."""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    sx = x[order]
    i = 0
    while i < len(sx):
        j = i
        while j + 1 < len(sx) and sx[j + 1] == sx[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def _norm_sf(z):
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def wilcoxon_one_sample(x, mu=1.0):
    """One-sample Wilcoxon signed-rank of x against mu (two-sided)."""
    return wilcoxon_signed_rank(np.asarray(x, dtype=float) - mu)


def cliffs_delta(a, b):
    """Cliff's delta between two samples (a vs b): P(a>b) - P(a<b)."""
    a = np.asarray(a, dtype=float)[:, None]
    b = np.asarray(b, dtype=float)[None, :]
    gt = np.sum(a > b)
    lt = np.sum(a < b)
    n = a.size * b.size // 1
    return float((gt - lt) / (a.shape[0] * b.shape[1]))


def delta_magnitude(d):
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def holm(pvals):
    """Holm-Bonferroni adjusted p-values (same order as input)."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m, dtype=float)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj.tolist()


# ── comparison machinery ──────────────────────────────────────────────────────

def raw(block, point, policy, metric="makespan"):
    try:
        return np.array(block[point][policy][metric]["raw"], dtype=float)
    except KeyError:
        return None


def compare_family(block, point, base_pol, other_pols, metric, title, lines):
    """Paired comparisons base vs each other policy, Holm-corrected."""
    rows, pvals = [], []
    a = raw(block, point, base_pol, metric)
    if a is None:
        lines.append(f"  [{title}] missing raw data for {base_pol}; "
                     "re-run with the updated runner.")
        return
    for pk in other_pols:
        b = raw(block, point, pk, metric)
        if b is None or len(b) != len(a):
            continue
        d = a - b                     # negative: adaptive lower/better
        lo, hi = bootstrap_ci_median(d)
        W, z, p = wilcoxon_signed_rank(d)
        cd = cliffs_delta(a, b)
        rows.append((pk, float(np.median(a)), float(np.median(b)),
                     float(np.median(d)), lo, hi, p, cd))
        pvals.append(p)
    if not rows:
        return
    padj = holm(pvals)
    lines.append(f"\n### {title}  (metric: {metric}; paired over "
                 f"{len(a)} seeds; Holm-corrected)")
    lines.append(f"| vs | med({LABELS.get(base_pol, base_pol)}) | med(other) "
                 "| med diff | 95% CI | p (Holm) | Cliff's d |")
    lines.append("|---|---|---|---|---|---|---|")
    for (pk, ma, mb, md, lo, hi, p, cd), pa in zip(rows, padj):
        sig = "**" if pa < 0.05 else ""
        lines.append(
            f"| {LABELS.get(pk, pk)} | {ma:.0f} | {mb:.0f} | {md:+.0f} "
            f"| [{lo:+.0f}, {hi:+.0f}] | {sig}{pa:.3g}{sig} "
            f"| {cd:+.2f} ({delta_magnitude(cd)}) |")


def oracle_distance(block, points, lines):
    lines.append("\n### Oracle distance (retrospective best fixed level "
                 "per configuration, by median makespan)")
    lines.append("| config | oracle level | oracle med | adaptive med "
                 "| ratio adaptive/oracle |")
    lines.append("|---|---|---|---|---|")
    for pt in points:
        meds = {}
        for pk in FIXED_LEVELS:
            r = raw(block, pt, pk)
            if r is None:
                m = block[pt].get(pk, {}).get("makespan", {}).get("median")
                if m is None:
                    continue
                meds[pk] = float(m)
            else:
                meds[pk] = float(np.median(r))
        if not meds:
            lines.append(f"| {pt} | (fixed-level data missing) | | | |")
            continue
        best = min(meds, key=meds.get)
        a = raw(block, pt, "adaptive")
        am = (float(np.median(a)) if a is not None
              else float(block[pt]["adaptive"]["makespan"]["median"]))
        lines.append(f"| {pt} | {LABELS[best]} | {meds[best]:.0f} "
                     f"| {am:.0f} | {am / meds[best]:.3f} |")


def strong_scaling(block, lines):
    lines.append("\n### Strong scaling (fixed total work; speedup and "
                 "efficiency relative to n=4)")
    lines.append("| policy | n | med makespan | speedup S(n) | "
                 "efficiency E(n) |")
    lines.append("|---|---|---|---|---|")
    for pk in ["qosws_coarse", "qosws_medium", "qosws_fine", "adaptive"]:
        base = None
        for n in ["4", "8", "16", "32"]:
            r = raw(block, n, pk)
            if r is None:
                continue
            med = float(np.median(r))
            if base is None:
                base = med
            s = base / med
            e = s / (int(n) / 4.0)
            lines.append(f"| {LABELS.get(pk, pk)} | {n} | {med:.0f} "
                         f"| {s:.2f} | {e:.2f} |")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="results_gran.json")
    ap.add_argument("--out", default="stats_report.md")
    args = ap.parse_args()

    R = json.load(open(args.inp))
    lines = ["# Statistical report — adaptive granularity evaluation", ""]
    lines.append(f"Source: `{args.inp}`. Tests: paired Wilcoxon signed-rank "
                 "(normal approx., tie/continuity corrected), 95% percentile "
                 "bootstrap CI of the median paired difference (10k "
                 "resamples), Cliff's delta; Holm correction per family. "
                 "Negative differences favour the adaptive policy "
                 "(lower makespan).")

    if "rq1a_dist" in R:
        lines.append("\n## RQ1a — distributions")
        for pt in R["rq1a_dist"]:
            compare_family(R["rq1a_dist"], pt, "adaptive",
                           FIXED_LEVELS + ["static_coarse", "static_fine"],
                           "makespan", f"{pt}: adaptive vs fixed", lines)
        oracle_distance(R["rq1a_dist"], list(R["rq1a_dist"].keys()), lines)

    if "rq1b_size" in R:
        lines.append("\n## RQ1b — weak scaling (work grows with n)")
        for pt in R["rq1b_size"]:
            compare_family(R["rq1b_size"], pt, "adaptive", FIXED_LEVELS,
                           "makespan", f"n={pt}: adaptive vs fixed", lines)
        oracle_distance(R["rq1b_size"], list(R["rq1b_size"].keys()), lines)

    if "rq6_ablation" in R:
        lines.append("\n## RQ6 — ablations and controls")
        abls = ["adaptive_noQ", "adaptive_noF", "adaptive_noO",
                "adaptive_nosplit", "adaptive_nomerge", "adaptive_batchonly",
                "adaptive_random", "threshold", "qosws_medium"]
        for pt in R["rq6_ablation"]:
            compare_family(R["rq6_ablation"], pt, "adaptive", abls,
                           "makespan", f"{pt}: adaptive vs ablations", lines)
            compare_family(R["rq6_ablation"], pt, "adaptive", abls,
                           "comm_cost", f"{pt}: communication", lines)

    if "rq7_strong" in R:
        lines.append("\n## RQ7 — strong scaling")
        strong_scaling(R["rq7_strong"], lines)
        for pt in R["rq7_strong"]:
            compare_family(R["rq7_strong"], pt, "adaptive",
                           ["qosws_coarse", "qosws_medium", "qosws_fine"],
                           "makespan", f"n={pt}: adaptive vs fixed", lines)

    if "rq4_failures" in R:
        lines.append("\n## RQ4 — failures (paired on makespan)")
        # One-sample tests of each stealing policy's slowdown against 1.0
        lines.append("\n### Slowdown vs 1.0 (one-sample Wilcoxon, per policy,"
                     " Holm within regime)")
        lines.append("| regime | policy | med slowdown | p (Holm) |")
        lines.append("|---|---|---|---|")
        for pt in ["transient", "permanent"]:
            rows, ps = [], []
            for pk in ["qosws_coarse", "qosws_medium", "qosws_fine",
                       "adaptive"]:
                a = raw(R["rq4_failures"], pt, pk)
                b = raw(R["rq4_failures"], "none", pk)
                if a is None or b is None:
                    continue
                slow = a / b
                _, _, p = wilcoxon_one_sample(slow, 1.0)
                rows.append((pk, float(np.median(slow)), p))
                ps.append(p)
            for (pk, ms, _), pa in zip(rows, holm(ps)):
                sig = "**" if pa < 0.05 else ""
                lines.append(f"| {pt} | {LABELS.get(pk, pk)} | {ms:.3f} "
                             f"| {sig}{pa:.3g}{sig} |")
        for pt in R["rq4_failures"]:
            compare_family(R["rq4_failures"], pt, "adaptive",
                           ["qosws_medium", "qosws_fine", "qosws_coarse"],
                           "makespan", f"{pt}: adaptive vs fixed stealing",
                           lines)

    for blk, title in [("rq8_phases", "RQ8 — phased workloads"),
                       ("rq9_staleness", "RQ9 — staleness and noise"),
                       ("rq10_sensitivity", "RQ10 — parameter sensitivity")]:
        if blk not in R:
            continue
        lines.append(f"\n## {title}")
        for pt in R[blk]:
            pols = list(R[blk][pt].keys())
            lines.append(f"\n### {pt}")
            lines.append("| policy | makespan | comm | splits | merges |")
            lines.append("|---|---|---|---|---|")
            for pk in pols:
                d = R[blk][pt][pk]
                lines.append(f"| {LABELS.get(pk, pk)} "
                             f"| {d['makespan']['median']:.0f} "
                             f"[{d['makespan']['q25']:.0f},{d['makespan']['q75']:.0f}] "
                             f"| {d['comm_cost']['median']:.0f} "
                             f"| {d['n_splits']['median']:.0f} "
                             f"| {d['n_merges']['median']:.0f} |")
            if blk == "rq8_phases":
                compare_family(R[blk], pt, "adaptive",
                               [p for p in pols if p != "adaptive"],
                               "makespan", f"{pt}: adaptive vs alternatives",
                               lines)

    text = "\n".join(lines) + "\n"
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(text[:2000])
    print(f"\n[full report -> {args.out}]")


if __name__ == "__main__":
    main()
