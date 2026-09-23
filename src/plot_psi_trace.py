#!/usr/bin/env python3
"""
Turn psi_trace.json (produced by `run_granularity.py --trace`) into:

  * a pgfplots four-panel figure block (fig_psi_trace.tex) matching the
    paper's style: (1) refinement signals I and F; (2) coarsening signal O
    and QoS pressure Q; (3) the pressure Psi with the dead-band; and
    (4) cumulative split and merge operations;
  * a quick-look PNG (psi_trace.png) with the same four panels.

The trace is downsampled to at most --points samples (default 300) to keep
the TikZ source light.

Usage:
    python3 plot_psi_trace.py [--in psi_trace.json] [--points 300]
"""

import argparse
import json

import numpy as np


def downsample(trace, max_pts):
    n = len(trace)
    if n <= max_pts:
        return trace
    idx = np.linspace(0, n - 1, max_pts).round().astype(int)
    return [trace[i] for i in idx]


def coords(tr, key):
    return " ".join(f"({x['tick']},{x[key]:.3f})" for x in tr)


def make_tex(tr, deadband):
    t_end = tr[-1]["tick"]
    parts = []
    parts.append(r"""\begin{figure}[t]
\centering
\begin{tikzpicture}
\begin{groupplot}[
  group style={group size=1 by 4, vertical sep=0.55cm,
               xlabels at=edge bottom, xticklabels at=edge bottom},
  width=0.92\columnwidth, height=0.30\columnwidth,
  xmin=0, xmax=""" + str(t_end) + r""",
  tick label style={font=\scriptsize}, label style={font=\scriptsize},
  legend style={font=\scriptsize, draw=none, fill=none},
  xlabel={time (ticks)},
]""")
    # panel 1: refinement signals
    parts.append(r"\nextgroupplot[ylabel={refine signals}, ymin=0, ymax=1.05,"
                 r" legend pos=north east, legend columns=2]")
    parts.append(r"\addplot[colad, thick] coordinates {" + coords(tr, "I")
                 + r"};\addlegendentry{$\widetilde{I}$}")
    parts.append(r"\addplot[colwf, thick] coordinates {" + coords(tr, "F")
                 + r"};\addlegendentry{$F$}")
    # panel 2: coarsening + QoS
    parts.append(r"\nextgroupplot[ylabel={coarsen / QoS}, ymin=0, ymax=1.05,"
                 r" legend pos=north east, legend columns=2]")
    parts.append(r"\addplot[colsf, thick] coordinates {" + coords(tr, "O")
                 + r"};\addlegendentry{$\widetilde{O}$}")
    parts.append(r"\addplot[colwm, thick] coordinates {" + coords(tr, "Q")
                 + r"};\addlegendentry{$Q_p$}")
    # panel 3: psi with dead-band
    lo, hi = -deadband, deadband
    parts.append(r"\nextgroupplot[ylabel={$\Psi(t)$},"
                 r" legend pos=north east]")
    parts.append(r"\addplot[colsc, very thick] coordinates {"
                 + coords(tr, "psi") + r"};\addlegendentry{$\Psi$}")
    parts.append(rf"\addplot[gray, dashed, domain=0:{t_end}] {{{hi}}};")
    parts.append(rf"\addplot[gray, dashed, domain=0:{t_end}] {{{lo}}};")
    # panel 4: cumulative operations
    parts.append(r"\nextgroupplot[ylabel={cumulative ops},"
                 r" legend pos=south east, legend columns=2]")
    parts.append(r"\addplot[colad, thick, const plot] coordinates {"
                 + coords(tr, "n_splits") + r"};\addlegendentry{splits}")
    parts.append(r"\addplot[colwc, thick, const plot] coordinates {"
                 + coords(tr, "n_merges") + r"};\addlegendentry{merges}")
    parts.append(r"""\end{groupplot}
\end{tikzpicture}
\caption{Controller response over one representative run ($n{=}16$,
lognormal workload, transient churn). Top to bottom: the refinement
signals (normalised imbalance $\widetilde{I}$ and failure pressure $F$);
the coarsening signal (normalised overhead $\widetilde{O}$) and QoS
pressure $Q_p$; the resulting pressure $\Psi(t)$ against the dead-band
(dashed); and the cumulative split and merge operations. Bursts of
failures raise $F$, push $\Psi$ above the dead-band, and are answered by
split activity; the adaptation rate remains bounded by the cooldown
throughout.}
\label{fig:psi-trace}
\end{figure}""")
    return "\n".join(parts)


def make_png(tr, deadband, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipped PNG")
        return
    t = [x["tick"] for x in tr]
    fig, axes = plt.subplots(4, 1, figsize=(7, 7), sharex=True)
    axes[0].plot(t, [x["I"] for x in tr], label=r"$\tilde{I}$")
    axes[0].plot(t, [x["F"] for x in tr], label=r"$F$")
    axes[0].set_ylabel("refine"); axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(t, [x["O"] for x in tr], label=r"$\tilde{O}$", color="tab:orange")
    axes[1].plot(t, [x["Q"] for x in tr], label=r"$Q_p$", color="tab:purple")
    axes[1].set_ylabel("coarsen/QoS"); axes[1].legend(ncol=2, fontsize=8)
    axes[2].plot(t, [x["psi"] for x in tr], color="tab:blue")
    axes[2].axhline(deadband, ls="--", c="gray"); axes[2].axhline(-deadband, ls="--", c="gray")
    axes[2].set_ylabel(r"$\Psi(t)$")
    axes[3].step(t, [x["n_splits"] for x in tr], where="post", label="splits")
    axes[3].step(t, [x["n_merges"] for x in tr], where="post", label="merges")
    axes[3].set_ylabel("cum. ops"); axes[3].set_xlabel("ticks")
    axes[3].legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="psi_trace.json")
    ap.add_argument("--points", type=int, default=300)
    ap.add_argument("--deadband", type=float, default=0.15)
    args = ap.parse_args()

    trace = json.load(open(args.inp))["trace"]
    tr = downsample(trace, args.points)
    tex = make_tex(tr, args.deadband)
    with open("fig_psi_trace.tex", "w") as f:
        f.write(tex)
    print(f"wrote fig_psi_trace.tex ({len(tr)} samples from "
          f"{len(trace)} ticks)")
    make_png(tr, args.deadband, "psi_trace.png")


if __name__ == "__main__":
    main()
