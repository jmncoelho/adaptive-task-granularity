"""make_tikz.py — emit native pgfplots figures + table from results_gran.json,
in the visual style of the main QoS-WS paper (evalbarstyle / evallinestyle,
median with IQR whiskers, six-policy palette)."""
from __future__ import annotations
import json

R = json.load(open("../data/results_v3.json"))

# policy order, colour, line mark, legend label
POL = [
    ("static_coarse", "colsc", "*",         "Static coarse"),
    ("static_fine",   "colsf", "square*",    "Static fine"),
    ("qosws_coarse",  "colwc", "triangle*",  "QoS-WS coarse"),
    ("qosws_medium",  "colwm", "diamond*",   "QoS-WS medium"),
    ("qosws_fine",    "colwf", "pentagon*",  "QoS-WS fine"),
    ("adaptive",      "colad", "star",       "Adaptive"),
]


def med(block, pt, pk, m):
    return R[block][pt][pk][m]["median"]


def iqr_half(block, pt, pk, m):
    a = R[block][pt][pk][m]
    return (a["q75"] - a["q25"]) / 2.0


def bar_fig(block, points, xcoords, metric, ylabel, caption, label,
            ymax=None, legend_cols=3, scale=1.0):
    lines = ["\\begin{figure}[t]", "\\centering", "\\begin{tikzpicture}",
             "\\begin{axis}[", "  evalbarstyle,",
             f"  ylabel={{{ylabel}}},",
             f"  symbolic x coords={{{', '.join(xcoords)}}},"]
    if ymax is not None:
        lines.append(f"  ymax={ymax},")
    lines.append(f"  legend columns={legend_cols},")
    lines.append("]")
    for pk, col, _mk, lab in POL:
        lines.append(f"\\addplot[{col}, fill={col}!85, draw={col}!60!black]")
        coords = []
        for pt, xc in zip(points, xcoords):
            v = med(block, pt, pk, metric) * scale
            e = iqr_half(block, pt, pk, metric) * scale
            coords.append(f"({xc}, {v:.1f}) +- (0, {e:.1f})")
        lines.append("  plot coordinates {" + " ".join(coords) + "};")
    lines.append("\\legend{" + ", ".join(l for *_x, l in POL) + "}")
    lines += ["\\end{axis}", "\\end{tikzpicture}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{figure}"]
    return "\n".join(lines)


def line_fig(block, points, xvals, metric, xlabel, ylabel, caption, label,
             ymin=None, ymax=None, legend_pos="north west"):
    lines = ["\\begin{figure}[t]", "\\centering", "\\begin{tikzpicture}",
             "\\begin{axis}[", "  evallinestyle,",
             f"  xlabel={{{xlabel}}}, ylabel={{{ylabel}}},",
             f"  legend pos={legend_pos}, legend columns=2,"]
    if ymin is not None:
        lines.append(f"  ymin={ymin},")
    if ymax is not None:
        lines.append(f"  ymax={ymax},")
    lines.append("]")
    for pk, col, mk, lab in POL:
        lines.append(f"\\addplot[{col}, mark={mk}, thick, mark size=2pt]")
        coords = " ".join(f"({x}, {med(block, pt, pk, metric):.1f})"
                          for pt, x in zip(points, xvals))
        lines.append(f"  coordinates {{{coords}}};")
        lines.append(f"\\addlegendentry{{{lab}}}")
    lines += ["\\end{axis}", "\\end{tikzpicture}",
              f"\\caption{{{caption}}}", f"\\label{{{label}}}", "\\end{figure}"]
    return "\n".join(lines)


# ── RQ1a: makespan by distribution (bar) ──────────────────────────────────────
fig_rq1a = bar_fig("rq1a_dist", ["uniform", "lognormal", "pareto"],
                   ["Uniform", "Lognormal", "Pareto"], "makespan",
                   "Makespan (ticks)",
                   "Median makespan across three task-cost distributions "
                   "($n{=}16$, fixed total work, $30$ seeds). Whiskers show the "
                   "interquartile range. Adaptive granularity matches or beats "
                   "the best fixed policy in every regime.",
                   "fig:rq1a", ymax=4300)

# ── RQ1b: makespan vs cluster size (line) ─────────────────────────────────────
fig_rq1b = line_fig("rq1b_size", ["4", "8", "16", "32"], [4, 8, 16, 32],
                    "makespan", "Number of devices", "Makespan (ticks)",
                    "Median makespan as a function of cluster size (lognormal "
                    "workload, total work scaled with $n$). The adaptive policy "
                    "tracks the lower envelope of the fixed policies and its "
                    "advantage grows with cluster size.",
                    "fig:rq1b", ymin=1000)

# ── RQ2: communication vs latency, and makespan vs latency (groupplot) ────────
def groupplot_rq2():
    lat_pts = ["1", "3", "5", "8", "12"]
    lat_x = [1, 3, 5, 8, 12]
    out = ["\\begin{figure}[t]", "\\centering", "\\begin{tikzpicture}",
           "\\begin{groupplot}[", "  group style={group size=2 by 1, "
           "horizontal sep=1.5cm}, evallinestyle, width=0.52\\columnwidth, "
           "height=0.5\\columnwidth, xlabel={Latency scale}, legend columns=2,",
           "]"]
    # panel a: comm cost
    out.append("\\nextgroupplot[ylabel={Comm.\\ cost (units)}, "
               "legend to name=rq2leg, legend style={font=\\scriptsize}]")
    for pk, col, mk, lab in POL:
        coords = " ".join(f"({x}, {med('rq2_latency', pt, pk, 'comm_cost'):.1f})"
                          for pt, x in zip(lat_pts, lat_x))
        out.append(f"\\addplot[{col}, mark={mk}, thick, mark size=1.8pt] "
                   f"coordinates {{{coords}}};")
        out.append(f"\\addlegendentry{{{lab}}}")
    # panel b: makespan
    out.append("\\nextgroupplot[ylabel={Makespan (ticks)}, ymin=1000]")
    for pk, col, mk, lab in POL:
        coords = " ".join(f"({x}, {med('rq2_latency', pt, pk, 'makespan'):.1f})"
                          for pt, x in zip(lat_pts, lat_x))
        out.append(f"\\addplot[{col}, mark={mk}, thick, mark size=1.8pt] "
                   f"coordinates {{{coords}}};")
    out += ["\\end{groupplot}",
            "\\node at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-1.1cm)$) "
            "{\\pgfplotslegendfromname{rq2leg}};",
            "\\end{tikzpicture}",
            "\\caption{Communication cost (left) and makespan (right) as the "
            "network latency profile is scaled ($n{=}16$, lognormal). "
            "Fixed-fine work stealing pays a communication cost that grows "
            "steeply with latency for no makespan benefit, whereas adaptive "
            "granularity keeps transfer cost close to the medium policy by "
            "coarsening under expensive communication.}",
            "\\label{fig:rq2}", "\\end{figure}"]
    return "\n".join(out)
fig_rq2 = groupplot_rq2()

# ── RQ3: time to U_desired by strictness (bar) ────────────────────────────────
fig_rq3 = bar_fig("rq3_qos", ["u0.80", "u0.90", "u0.95"],
                  ["$U_{des}{=}0.80$", "$U_{des}{=}0.90$", "$U_{des}{=}0.95$"],
                  "time_to_u_desired", "Time to $U_{des}$ (ticks)",
                  "Median time to reach the desired service utility "
                  "$U_{des}$ for three strictness levels ($n{=}16$, "
                  "lognormal). All stealing policies reach $U_{des}$ far "
                  "sooner than static decomposition; adaptive and medium "
                  "granularity are fastest.",
                  "fig:rq3", ymax=2600)

# ── RQ4: makespan slowdown under failures (bar, ratio to no-failure) ──────────
def fig_rq4_build():
    base = "rq4_failures"
    out = ["\\begin{figure}[t]", "\\centering", "\\begin{tikzpicture}",
           "\\begin{groupplot}[", "  group style={group size=2 by 1, "
           "horizontal sep=1.4cm}, evalbarstyle, width=0.47\\columnwidth, "
           "height=0.5\\columnwidth, /pgf/bar width=2.6pt, legend columns=3,", "]"]
    # panel A: slowdown ratios (both regimes)
    out.append("\\nextgroupplot[ylabel={Makespan slowdown $\\times$}, "
               "symbolic x coords={Transient, Permanent}, ymin=0.85, ymax=1.35, "
               "legend to name=rq4leg, legend style={font=\\scriptsize}]")
    for pk, col, _mk, lab in POL:
        none = med(base, "none", pk, "makespan")
        coords = []
        for pt, xc in [("transient", "Transient"), ("permanent", "Permanent")]:
            coords.append(f"({xc}, {med(base, pt, pk, 'makespan')/none:.3f})")
        out.append(f"\\addplot[{col}, fill={col}!85, draw={col}!60!black]")
        out.append("  plot coordinates {" + " ".join(coords) + "};")
        out.append(f"\\addlegendentry{{{lab}}}")
    # panel B: completion under transient churn, with IQR
    out.append("\\nextgroupplot[ylabel={Completion under transient (\\%)}, "
               "symbolic x coords={Completion}, ymin=84, ymax=101, "
               "enlarge x limits=0.5, xticklabels={}]")
    for pk, col, _mk, lab in POL:
        a = R[base]["transient"][pk]["completion_rate"]
        v = a["median"]*100
        lo = (a["median"]-a["q25"])*100
        hi = (a["q75"]-a["median"])*100
        out.append(f"\\addplot[{col}, fill={col}!85, draw={col}!60!black]")
        out.append(f"  plot coordinates {{(Completion, {v:.1f}) +- (0, 0)}};")
        out[-1] = f"  plot coordinates {{(Completion, {v:.1f})}};"
        out.insert(len(out), "")
        out.pop()
    # redo panel B with explicit error bars
    out = out[:next(i for i,l in enumerate(out) if "Completion under transient" in l)+0]
    out.append("\\nextgroupplot[ylabel={Completion under transient (\\%)}, "
               "symbolic x coords={Completion}, ymin=84, ymax=101, "
               "enlarge x limits=0.5, xticklabels={}]")
    for pk, col, _mk, lab in POL:
        a = R[base]["transient"][pk]["completion_rate"]
        v = a["median"]*100; lo=(a["median"]-a["q25"])*100; hi=(a["q75"]-a["median"])*100
        out.append(f"\\addplot[{col}, fill={col}!85, draw={col}!60!black, "
                   f"error bars/y dir=both, error bars/y explicit]")
        out.append(f"  plot coordinates {{(Completion, {v:.1f}) += (0, {hi:.1f}) -= (0, {lo:.1f})}};")
    out += ["\\end{groupplot}",
            "\\node at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-1.6cm)$) "
            "{\\pgfplotslegendfromname{rq4leg}};",
            "\\end{tikzpicture}",
            "\\caption{Robustness under the failure regimes of the companion "
            "paper ($n{=}16$, lognormal, $30$ seeds). Left: makespan slowdown "
            "relative to each policy's failure-free run ($1.0$ = unaffected). "
            "Right: median fraction of tasks completed under transient churn "
            "(whiskers show IQR). The static baselines leave a median "
            "$4$--$7\\%$ of tasks "
            "unfinished --- their near-unit slowdown bars therefore understate "
            "the damage --- while every stealing policy completes at least "
            "$99.7\\%$ (median) with slowdowns statistically "
            "indistinguishable from $1.0$; under "
            "permanent capacity loss all policies complete and the adaptive "
            "policy shows the lowest slowdown and completion time.}",
            "\\label{fig:rq4}", "\\end{figure}"]
    return "\n".join(out)
fig_rq4 = fig_rq4_build()

# ── RQ5: heterogeneity sweep (line) + trade-off scatter (groupplot) ───────────
def fig_rq5_build():
    het_pts = ["low", "med", "high"]
    het_x = [0.5, 1.0, 2.0]
    out = ["\\begin{figure}[t]", "\\centering", "\\begin{tikzpicture}",
           "\\begin{groupplot}[", "  group style={group size=2 by 1, "
           "horizontal sep=1.6cm}, evallinestyle, width=0.52\\columnwidth, "
           "height=0.5\\columnwidth, legend columns=2,", "]"]
    out.append("\\nextgroupplot[xlabel={Heterogeneity scale}, "
               "ylabel={Makespan (ticks)}, ymin=1000, "
               "legend to name=rq5leg, legend style={font=\\scriptsize}]")
    for pk, col, mk, lab in POL:
        coords = " ".join(f"({x}, {med('rq5_het', pt, pk, 'makespan'):.1f})"
                          for pt, x in zip(het_pts, het_x))
        out.append(f"\\addplot[{col}, mark={mk}, thick, mark size=1.8pt] "
                   f"coordinates {{{coords}}};")
        out.append(f"\\addlegendentry{{{lab}}}")
    # trade-off scatter: makespan vs comm at n=16 med-het (rq5_het['med'])
    out.append("\\nextgroupplot[xlabel={Comm.\\ cost (units)}, "
               "ylabel={Makespan (ticks)}, ymin=1000]")
    for pk, col, mk, lab in POL:
        x = med("rq5_het", "med", pk, "comm_cost")
        y = med("rq5_het", "med", pk, "makespan")
        out.append(f"\\addplot[{col}, mark={mk}, mark size=3pt, only marks] "
                   f"coordinates {{({x:.1f}, {y:.1f})}};")
    out += ["\\end{groupplot}",
            "\\node at ($(group c1r1.south)!0.5!(group c2r1.south)+(0,-1.1cm)$) "
            "{\\pgfplotslegendfromname{rq5leg}};",
            "\\end{tikzpicture}",
            "\\caption{Sensitivity to device heterogeneity. Left: median "
            "makespan as the CPU/memory spread is scaled; the adaptive "
            "advantage widens as heterogeneity grows. Right: the "
            "makespan--communication trade-off at $n{=}16$; the adaptive "
            "policy (\\textcolor{colad}{$\\star$}) occupies the favourable "
            "lower-left region, achieving low makespan without the "
            "communication cost of fixed-fine stealing "
            "(\\textcolor{colwf}{$\\bullet$}).}",
            "\\label{fig:rq5}", "\\end{figure}"]
    return "\n".join(out)
fig_rq5 = fig_rq5_build()

# ── Summary table (representative config: n=16, lognormal, no failures) ───────
def table_build():
    base, pt = "rq4_failures", "none"
    rows = []
    for pk, col, mk, lab in POL:
        rows.append((lab,
                     med(base, pt, pk, "makespan"),
                     med(base, pt, pk, "comm_cost"),
                     med(base, pt, pk, "ticks_overhead"),
                     med("rq3_qos", "u0.90", pk, "time_to_u_desired"),
                     med(base, pt, pk, "avg_utilisation") * 100,
                     med(base, pt, pk, "n_splits"),
                     med(base, pt, pk, "n_merges")))
    L = ["\\begin{table}[t]", "\\centering", "\\footnotesize",
         "\\caption{Aggregate median metrics at the representative operating "
         "point ($n{=}16$, $500$ tasks of equivalent total work, lognormal, "
         "no failures, $30$ seeds). Communication cost is the latency-weighted "
         "transfer cost; overhead is the cumulative per-task scheduling cost "
         "$T_{sched}$.}", "\\label{tab:summary}",
         "\\begin{tabular}{lrrrrrrr}", "\\toprule",
         "Policy & Makespan & Comm. & Ovhd. & $t_{U_{des}}$ & Util.\\% & "
         "Splits & Merges \\\\", "\\midrule"]
    for r in rows:
        L.append(f"{r[0]} & {r[1]:.0f} & {r[2]:.0f} & {r[3]:.0f} & {r[4]:.0f} "
                 f"& {r[5]:.1f} & {r[6]:.0f} & {r[7]:.0f} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(L)
table = table_build()

with open("gran_figures.tex", "w") as f:
    f.write("\n\n".join([fig_rq1a, fig_rq1b, fig_rq2, fig_rq3, fig_rq4,
                         fig_rq5, table]))
import json as _json
_named = {"rq1a": fig_rq1a, "rq1b": fig_rq1b, "rq2": fig_rq2, "rq3": fig_rq3,
          "rq4": fig_rq4, "rq5": fig_rq5, "table": table}
with open("gran_figs.json", "w") as f:
    _json.dump(_named, f)
print("wrote gran_figures.tex and gran_figs.json")
print("\n--- table preview ---\n")
print(table)
