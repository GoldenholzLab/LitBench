#!/usr/bin/env python3
"""Figure 2 - the LitBench distractor grid as a visual (replaces old Table 1).

One row per test tier. Each row shows, schematically, what the system searches
(gold paper(s) among distractors of a given type) and the d-sweep (distractor
fill), so a reader sees at a glance how difficulty is graded from the T1 floor
to two-paper synthesis over the open corpus. Definitional figure - no data
dependency.

The last column carries the approved plain-language name of each condition,
word for word as the manuscript prose and Figure 1's footnote key give it, so a
reader meets one wording for a condition and not three. The two open-pool rows
have no approved name of their own; they are named after the bounded condition
they extend.

Color is used ONLY where it carries information (gold vs distractor TYPE), per
the clinical-ai-figure-design rule; everything else is neutral. NEJM AI / Lancet
Digital Health house style (scientific-figures skill): Helvetica, 600 DPI,
pdf.fonttype 42, no in-figure title.
Output: figures/figure2_grid.{pdf,png}
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
from pathlib import Path

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# --- color = information only ---
GOLD   = "#2E5A72"   # gold paper (the signal) - muted deep teal-blue
NEAR   = "#6E8598"   # near-neighbor epilepsy distractor - muted slate-blue
OOD    = "#CBD3DA"   # out-of-domain distractor - light blue-gray
INK    = "#2B2B2B"
MUTED  = "#6B7280"
OPEN_EDGE = "#2E5A72"  # open-corpus outline - matches gold signal

# tier rows: (label, n_papers, list of pictogram cells, d-sweep text, condition name, open?)
# pictogram cells are (kind, count) drawn left->right: 'gold','near','ood'
ROWS = [
    ("T1", "1", [("gold", 1)],                         "0",
     "the answering paper alone", False),
    ("T2", "1", [("gold", 1), ("ood", 8)],             "10 - 980",
     "hidden among unrelated papers", False),
    # T3 draws its distractors from the OTHER gold papers, and compose_haystack takes
    # min(n, len(pool)) of them: the pool is the 999 epilepsy papers that are not this
    # question's source, so the swept 1,026 realizes 999 distractors and a 1,000-paper
    # searched, not 1,027. The realized cap is printed; the legend states why.
    ("T3", "1", [("gold", 1), ("near", 8)],            "10 - 999",
     "hidden among similar epilepsy papers", False),
    ("T4", "1", [("gold", 1), ("near", 4), ("ood", 4)], "75 - 2000",
     "hidden among both kinds", False),
    ("T5", "2", [("gold", 2), ("near", 3), ("ood", 4)], "10 - 2000",
     "two papers, one fact from each", False),
    ("T4·∞", "1", [("gold", 1), ("open", 10)], "~127,000",
     "one paper, over the live PubMed pool", True),
    ("T5·∞", "2", [("gold", 2), ("open", 9)],  "~127,000",
     "two papers, one fact from each, over the live PubMed pool", True),
    ("T6", "1", [("ood", 9)], "1 - 980",
     "answer withheld, one paper; declining is the only credit", False),
    ("T7", "2", [("ood", 9)], "1 - 980",
     "answer withheld, two papers; the system must decline both", False),
]

CELL = 0.30       # square size (data units)
GAPX = 0.07       # gap between squares
ROW_H = 1.0
X_PICT = 2.35     # left x of pictogram band
X_DSWEEP = 7.2    # x of the d-sweep column (left of X_PROBE - "d (distractor fill)" is wider than the old "N sweep")
X_PROBE = 9.15    # x of the condition-name column
FIG_W = 11.6
# FIG_H scales with row count so the accepted round-5 rows (T1-T5,
# T4.inf/T5.inf) keep an identical on-page size when tiers are appended -
# the data-unit -> inch scale is held constant rather than letting the
# fixed-height canvas compress every row to make room for new ones.
_BASE_NROWS = 7        # round-5 row count this figure was accepted with
_BASE_FIG_H = 5.0      # round-5 figure height (in), tuned for _BASE_NROWS
_YLIM_PAD = 2.25        # ylim span beyond the row stack (bottom 1.15 + top 1.1)
FIG_H = _BASE_FIG_H * (len(ROWS) * ROW_H + _YLIM_PAD) / (_BASE_NROWS * ROW_H + _YLIM_PAD)

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
ax.set_xlim(0, 14.3)
ax.set_ylim(-1.15, len(ROWS) * ROW_H + 1.1)   # reserve a legend band below the rows
ax.axis("off")

nrows = len(ROWS)
top = nrows * ROW_H

# column headers
ax.text(0.15, top + 0.45, "Tier", fontsize=10.5, fontweight="bold", color=GOLD, va="center")
ax.text(X_PICT, top + 0.45, "Papers searched (schematic)", fontsize=10.5,
        fontweight="bold", color=GOLD, va="center")
ax.text(X_DSWEEP, top + 0.45, "d (distractor fill)", fontsize=10.5, fontweight="bold", color=GOLD, va="center")
ax.text(X_PROBE, top + 0.45, "What the condition is", fontsize=10.5, fontweight="bold", color=GOLD, va="center")
ax.plot([0.1, 14.1], [top + 0.15, top + 0.15], color="#D1D5DB", lw=0.9)


def draw_cell(x, y, kind):
    if kind == "gold":
        ax.add_patch(FancyBboxPatch((x, y), CELL, CELL, boxstyle="round,pad=0.002,rounding_size=0.04",
                                    facecolor=GOLD, edgecolor="white", lw=0.8, zorder=3))
        ax.plot([x + CELL*0.30, x + CELL*0.46], [y + CELL*0.5, y + CELL*0.30],
                color="white", lw=1.0, zorder=4)
        ax.plot([x + CELL*0.46, x + CELL*0.72], [y + CELL*0.30, y + CELL*0.70],
                color="white", lw=1.0, zorder=4)
    elif kind == "near":
        ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=NEAR, edgecolor="white", lw=0.8, zorder=2))
    elif kind == "ood":
        ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor=OOD, edgecolor="white", lw=0.8, zorder=2))
    elif kind == "open":
        ax.add_patch(Rectangle((x, y), CELL, CELL, facecolor="white",
                               edgecolor=OPEN_EDGE, lw=0.8, ls=(0, (1, 1)), zorder=2))


for i, (label, n_papers, cells, dsweep, condition, is_open) in enumerate(ROWS):
    y = top - (i + 1) * ROW_H + (ROW_H - CELL) / 2
    yc = y + CELL / 2
    # tier label + paper-count badge
    ax.text(0.15, yc, label, fontsize=12, fontweight="bold", color=INK, va="center")
    # The badge uses "paper(s)" rather than retrieval-literature jargon, which is
    # ambiguous here -- it can mean a step taken WITHIN one document, and would not
    # tell a clinician that two SEPARATE papers are required.
    badge_txt = "1 paper" if n_papers == "1" else "2 papers"
    ax.text(0.15, yc - 0.34, badge_txt, fontsize=7.2, color=MUTED, va="center")
    # pictogram
    x = X_PICT
    for kind, count in cells:
        for _ in range(count):
            draw_cell(x, y, kind)
            x += CELL + GAPX
        if kind in ("ood", "near", "open") and count >= 6:
            ax.text(x + 0.02, yc, "…", fontsize=12, color=MUTED, va="center")
            x += 0.32
    # "x N" growth annotation for distractor tiers
    if label != "T1":
        ax.annotate("", xy=(x + 0.05, yc), xytext=(X_PICT - 0.12, yc),
                    arrowprops=dict(arrowstyle="-|>", color="#B5BDC8", lw=0.8,
                                    shrinkA=0, shrinkB=0), zorder=1)
    # d sweep
    ax.text(X_DSWEEP, yc, dsweep, fontsize=9.2, color=INK, va="center")
    # the condition's approved plain-language name
    ax.text(X_PROBE, yc, condition, fontsize=8.4, color=INK, va="center")
    # faint row separator
    if i < nrows - 1:
        ax.plot([0.1, 14.1], [y - (ROW_H - CELL) / 2, y - (ROW_H - CELL) / 2],
                color="#F0F1F3", lw=0.6, zorder=0)

# legend (its own band below the rows; color = meaning)
ly = -0.72
ax.plot([0.1, 14.1], [-0.30, -0.30], color="#E5E7EB", lw=0.8, zorder=0)
lx = 0.15
def leg(x, kind, txt):
    draw_cell(x, ly, kind)
    ax.text(x + CELL + 0.14, ly + CELL/2, txt, fontsize=8.4, color=INK, va="center")
    return x + CELL + 0.20 + 0.088 * len(txt)
lx = leg(lx, "gold", "Gold paper (carries the answer)")
lx = leg(lx, "near", "Near-neighbor distractor")
lx = leg(lx, "ood", "Out-of-domain distractor")
lx = leg(lx, "open", "Open PubMed Central pool")

out = Path(__file__).resolve().parent
fig.savefig(out / "figure2_grid.pdf", dpi=600, bbox_inches="tight", pad_inches=0.12)
fig.savefig(out / "figure2_grid.png", dpi=600, bbox_inches="tight", pad_inches=0.12)
print("wrote figure2_grid.{pdf,png}")
