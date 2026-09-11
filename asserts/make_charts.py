#!/usr/bin/env python3
"""Generate the figures used in README.md from the PAC 2026 optimization report.

All numbers are transcribed from PAC2026_indexer_mqa_logits_优化报告.md and
SVEconfig.md. Run from the repository root:

    python3 assets/make_charts.py

Writes PNG files into assets/.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# --- palette (validated with the dataviz six-checks validator, light mode) -----
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

S1 = "#2a78d6"   # blue   - Case 1
S2 = "#eb6834"   # orange - Case 2
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_2,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "grid.color": GRID,
    "axes.grid": True,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "legend.frameon": False,
})


def tidy(ax):
    """Recessive hairline grid on the value axis only."""
    ax.grid(axis="x", visible=False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_linewidth(0.8)


def save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)


# ---------------------------------------------------------------- 1. progress --
def chart_progression():
    versions = [
        "Baseline", "V2\nSVE BFDOT", "V3\nSME BFMOPA", "V6\nQ Packing",
        "V9\nZA K-pack", "V10\nPage sched.", "V10x\n2x2 BFMOPA",
        "V12\nK prefetch", "V12 final\n+ Q-ready",
    ]
    case1 = [0.038219666, 0.641099136, 7.79854, 8.02485, 8.463, 8.6, 9.2, 10.2, 10.5]
    case2 = [0.042419292, 0.729706728, 6.60030, 6.79449, 7.554, 9.5, 10.178, 14.7, 14.7]

    x = list(range(len(versions)))
    fig, ax = plt.subplots(figsize=(9.2, 5.0))

    ax.plot(x, case1, "-o", color=S1, linewidth=2, markersize=8,
            label="Case 1  (batch 32, next_n 2, avg_kv 1024)", zorder=3)
    ax.plot(x, case2, "-o", color=S2, linewidth=2, markersize=8,
            label="Case 2  (batch 128, next_n 1, avg_kv 4096)", zorder=3)

    ax.set_yscale("log")
    ax.set_ylim(0.02, 40)
    ax.set_xlim(-0.35, len(versions) - 0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(versions, fontsize=8.5)
    ax.set_ylabel("Throughput  (TFLOPS, log scale)")
    ax.set_title("Performance progression: 275x / 346x over the reference implementation",
                 fontsize=12, pad=14, loc="left")

    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(axis="y")

    # Selective direct labels: only the endpoints. Case 1 is the lower series at
    # both ends, so its label goes below and Case 2's above, keeping them apart.
    for xi, yi in ((0, case1[0]), (len(x) - 1, case1[-1])):
        ax.annotate(f"{yi:.3g}", (xi, yi), textcoords="offset points",
                    xytext=(0, -19), ha="center", fontsize=9, color=INK)
    for xi, yi in ((0, case2[0]), (len(x) - 1, case2[-1])):
        ax.annotate(f"{yi:.3g}", (xi, yi), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=9, color=INK)

    ax.annotate("page-level scheduling\nbreaks the 32-batch ceiling",
                xy=(5, 9.5), xytext=(3.6, 24),
                fontsize=8.5, color=INK_2, ha="center",
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.9))
    ax.annotate("K prefetch hidden\nunder SME compute",
                xy=(7, 14.7), xytext=(6.1, 30),
                fontsize=8.5, color=INK_2, ha="center",
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.9))

    ax.legend(loc="lower right", fontsize=9)
    tidy(ax)
    save(fig, "perf_progression.png")


# ---------------------------------------------------------------- 2. roofline --
def chart_roofline():
    i = [32, 320]
    bw_probe = 206.25      # GB/s, random page + K-packing order probe
    bw_lin = 356.4         # GB/s, random page + linear cache-line order
    sme_mix = 34.945       # TFLOPS, real 2x2 BFMOPA instruction mix
    sme_peak = 38.651      # TFLOPS, pure BFMOPA

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(30, 320)
    ax.set_ylim(4, 70)

    for b, style, label in (
        (bw_probe, "-", f"DRAM roof, probe order ({bw_probe:g} GB/s)"),
        (bw_lin, "--", f"DRAM roof, random page + linear order ({bw_lin:g} GB/s)"),
    ):
        ax.plot(i, [x * b / 1000 for x in i], style, color=MUTED,
                linewidth=1.4, zorder=2, label=label)

    for p, style, label in (
        (sme_mix, "-", f"SME compute roof, real instruction mix ({sme_mix:g} T)"),
        (sme_peak, ":", f"SME pure BFMOPA peak ({sme_peak:g} T)"),
    ):
        ax.plot(i, [p, p], style, color=INK_2, linewidth=1.4, zorder=2, label=label)

    ax.plot([128], [10.5], "o", color=S1, markersize=11, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=2)
    ax.plot([64], [14.7], "o", color=S2, markersize=11, zorder=4,
            markeredgecolor=SURFACE, markeredgewidth=2)

    ax.annotate("Case 1   128 FLOP/B   10.5 T", (128, 10.5), textcoords="offset points",
                xytext=(13, -4), ha="left", va="center", fontsize=9, color=INK)
    ax.annotate("Case 2   64 FLOP/B   14.7 T", (64, 14.7), textcoords="offset points",
                xytext=(13, 6), ha="left", va="bottom", fontsize=9, color=INK)

    ax.annotate("Case 2 sits above the probe roof:\nprefetch lifts effective\nbandwidth to ~219 GB/s",
                xy=(64.5, 14.2), xytext=(74, 5.6), fontsize=8.5, color=INK_2,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.9))

    ax.set_xlabel("Arithmetic intensity  (FLOP/Byte)")
    ax.set_ylabel("Throughput  (TFLOPS, log scale)")
    ax.set_title("Roofline position of the final kernel", fontsize=12, pad=14, loc="left")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=8.5)
    tidy(ax)
    save(fig, "roofline.png")


# ------------------------------------------------------------ 3. phase timing --
def chart_phase_timing():
    phases = ["Q Pack", "K Pack", "SME", "Postprocess", "Mask Init"]
    colors = [S1, S2, AQUA, YELLOW, MAGENTA]
    # percent of accounted phase time (sums to 100); overhead is reported separately.
    # (label, wall clock, [Q Pack, K Pack, SME, Postprocess, Mask Init], overhead)
    data = [
        ("Case 1 - V10x", "127.7 us", [13.96, 28.08, 43.08, 9.34, 5.54], 13.09),
        ("Case 1 - V12", "114.8 us", [15.79, 13.96, 53.41, 10.94, 5.91], 14.92),
        ("Case 2 - V10x", "916.7 us", [1.12, 51.22, 38.55, 8.36, 0.75], 5.08),
        ("Case 2 - V12", "680.3 us", [1.52, 25.80, 59.84, 11.73, 1.12], 6.69),
    ]

    x = list(range(len(data)))
    fig, ax = plt.subplots(figsize=(9.2, 5.4))

    bottoms = [0.0] * len(data)
    for pi, (phase, color) in enumerate(zip(phases, colors)):
        vals = [d[2][pi] for d in data]
        ax.bar(x, vals, 0.54, bottom=bottoms, color=color, label=phase,
               edgecolor=SURFACE, linewidth=2, zorder=3)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 8.0:  # only label a segment when the text fits inside it
                ax.text(xi, b + v / 2, f"{v:.1f}%", ha="center", va="center",
                        fontsize=8.5, color="white")
        bottoms = [b + v for b, v in zip(bottoms, vals)]

    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{d[0]}\n{d[1]}\noverhead {d[3]:.1f}%" for d in data], fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of phase time  (%)")
    ax.set_title("Prefetch moves the bottleneck out of K packing and into SME",
                 fontsize=12, pad=14, loc="left")
    ax.grid(axis="y")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=5, fontsize=9)
    tidy(ax)
    save(fig, "phase_timing.png")


# ------------------------------------------------------- 4. bandwidth matrix --
def chart_bandwidth():
    rows = [
        ("S-LIN   sequential page + linear", 398.8),
        ("S-KPK   sequential page + K-pack order", 357.5),
        ("R-LIN   random page + linear", 356.4),
        ("PACE-LIN  random + linear + 1-page prefetch", 337.8),
        ("PACE    random + K-pack + 1-page prefetch", 228.3),
        ("PACE x2 prefetch depth 2", 220.2),
        ("R-KPK   random page + K-pack order", 209.6),
        ("PACE x4 prefetch depth 4", 208.8),
    ]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    y = list(range(len(rows)))

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.barh(y, vals, 0.62, color=S1, zorder=3)
    for yi, v in zip(y, vals):
        ax.text(v + 4, yi, f"{v:.1f}", va="center", fontsize=9, color=INK)

    ax.axvline(206.25, color=INK_2, linewidth=1.4, linestyle="--", zorder=2)
    # caption sits left of the rule so the dashed line does not cut through it
    ax.text(201, len(rows) - 0.05, "pre-V12 probe roof 206 GB/s",
            ha="right", va="top", fontsize=8.5, color=INK_2)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.invert_yaxis()
    # headroom below the last bar for the reference-line caption
    ax.set_ylim(len(rows) + 0.75, -0.6)
    ax.set_xlim(0, 440)
    ax.set_xlabel("Effective read bandwidth  (GB/s)")
    ax.set_title("Random page mapping is nearly free; the page-interior access order is what costs",
                 fontsize=12, pad=14, loc="left")
    ax.grid(axis="x")
    tidy(ax)
    save(fig, "bandwidth_matrix.png")


# ---------------------------------------------------------- 5. measurement noise
def chart_stability():
    rounds = list(range(1, 7))
    case1 = [8.421, 8.536, 8.469, 8.457, 8.567, 5.301]
    case2 = [7.569, 7.546, 7.553, 7.558, 5.992, 7.556]

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.plot(rounds, case1, "-o", color=S1, linewidth=2, markersize=8,
            label="Case 1  (median 8.46 T)", zorder=3)
    ax.plot(rounds, case2, "-o", color=S2, linewidth=2, markersize=8,
            label="Case 2  (median 7.55 T)", zorder=3)

    # direct-label only the two extremes
    for xi, yi in ((6, case1[-1]), (5, case2[4])):
        ax.annotate(f"{yi:g}", (xi, yi), textcoords="offset points",
                    xytext=(0, -19), ha="center", fontsize=9, color=INK)

    ax.text(1.0, 5.9,
            "Two isolated dips, not reproducible:\n"
            "the other case stayed normal in both rounds.",
            fontsize=8.5, color=INK_2, ha="left", va="center")

    ax.set_xticks(rounds)
    ax.set_xlabel("Measurement round")
    ax.set_ylabel("Throughput  (TFLOPS)")
    ax.set_xlim(0.6, 6.5)
    ax.set_ylim(4.6, 9.9)
    ax.set_title("V9 six-round spread: the story lives in the median, not the best run",
                 fontsize=12, pad=14, loc="left")
    ax.legend(loc="upper left", fontsize=9)
    tidy(ax)
    save(fig, "v9_stability.png")


if __name__ == "__main__":
    chart_progression()
    chart_roofline()
    chart_phase_timing()
    chart_bandwidth()
    chart_stability()
