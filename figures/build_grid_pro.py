#!/usr/bin/env python3
"""Figure 3 — the FULL LitBench performance grid, rebuilt in the Hochberg/Willett
NEJM data-rich heatmap grammar (hochberg_kit.heatmap_values): viridis magnitude with a
bold printed value in EVERY cell, grouped tier column-brackets, Recall/Thinking row
brackets, hairline table gridlines. Shows every domain x every distractor fill x every
tier for all three systems — the same coverage as the big heatmap, made legible.

Three stacked panels (one per system: Gemma-4B, Gemma-12B, Sonnet-5) share one viridis
0-100 scale so cells are comparable across models. Data = consort.candidate_grid.assemble
(a5-ircot ladder). White cells = tiers structurally absent for a domain (methods/table/
figure Thinking carry no two-hop question).
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# In addition to LITBENCH_ROOT, this script imports `consort.candidate_grid`
# from the private LitBench working tree -- that module is not bundled in this
# repo. It will not run standalone; LITBENCH_ROOT (below) is inserted onto
# sys.path so `consort` is importable, provided your LITBENCH_ROOT tree
# actually contains the `consort` package.
LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
sys.path.insert(0, LITBENCH_ROOT)
# hochberg_kit is a personal Claude Code skill script (hochberg-figure-style) used for
# figure styling -- not part of the LitBench working tree, and not bundled in this repo.
HOCHBERG_STYLE_ROOT = os.environ.get("HOCHBERG_STYLE_ROOT")
if HOCHBERG_STYLE_ROOT is None:
    raise SystemExit(
        "Set HOCHBERG_STYLE_ROOT to the path of the hochberg-figure-style Claude Code "
        "skill's scripts directory to run this script (it uses hochberg_kit for figure "
        "styling, a personal skill not included in this repo)."
    )
sys.path.insert(0, HOCHBERG_STYLE_ROOT)
import hochberg_kit as hk  # noqa: E402
from consort.candidate_grid import CANDIDATES, assemble, ROW_LABELS, ALL_COLS, NCOL  # noqa: E402

OUT = Path(__file__).resolve().parent
hk.apply_style()

SYSTEMS = ["Gemma-4B", "Gemma-12B", "Sonnet-5"]

# column groups by tier (inclusive start, exclusive end) over the 27 tier·fill columns
tier_bounds = {}
for i, (tt, n) in enumerate(ALL_COLS):
    tier_bounds.setdefault(tt, [i, i])[1] = i + 1
COL_GROUPS = [(tt, b[0], b[1]) for tt, b in tier_bounds.items()]
COL_LABELS = [("∞" if n == "∞" else (f"{n}" if n < 1000 else f"{n//1000}k"))
              for tt, n in ALL_COLS]
# row groups: row 0 = ALL (aggregate); 1-7 Recall; 8-14 Thinking
ROW_GROUPS = [("Recall", 1, 8), ("Thinking", 8, 15)]
ROW_LABS = ["ALL"] + [r.replace(" Recall", "").replace(" Thinking", "")
                      for r in ROW_LABELS[1:]]


def system_matrix(matrix, bi):
    M = np.full((len(ROW_LABELS), NCOL), np.nan)
    for r in range(len(ROW_LABELS)):
        for c in range(NCOL):
            v = matrix[r][bi * NCOL + c]
            if v is not None:
                M[r, c] = v
    return M


def main():
    a5 = next(c for c in CANDIDATES if c.key == "a5-ircot")
    matrix, _, bands = assemble(a5)
    idx = {b: i for i, b in enumerate(bands)}

    fig, axes = plt.subplots(3, 1, figsize=(11.6, 11.2))
    plt.subplots_adjust(left=0.11, right=0.995, top=0.95, bottom=0.075, hspace=0.55)

    for k, (sysname, ax) in enumerate(zip(SYSTEMS, axes)):
        M = system_matrix(matrix, idx[sysname])
        last = (k == len(SYSTEMS) - 1)
        hk.heatmap_values(
            ax, M, row_labels=ROW_LABS,
            col_labels=(COL_LABELS if last else [""] * NCOL),
            cmap="viridis", vmin=0, vmax=100, fmt="{:.0f}", valfontsize=6.6,
            col_groups=(COL_GROUPS if k == 0 else None),
            row_groups=ROW_GROUPS, sep_rows=[1],
            col_group_title=("Test tier · distractor fill N" if k == 0 else None),
            cbar=("horizontal" if last else None),
            cbar_label=("Panel accuracy (%)  ·  three-LLM judge panel" if last else ""),
            cbar_ticks=[0, 25, 50, 75, 100],
            row_bracket_x=-0.075, row_label_x=-0.105)
        # bold system label at the left, panel letter
        ax.text(-0.105, 1.06, f"{'ABC'[k]}", transform=ax.transAxes, fontsize=13,
                fontweight="bold", va="bottom", ha="left", color=hk.INK)
        ax.text(0.5, 1.14 if k == 0 else 1.055, sysname, transform=ax.transAxes,
                fontsize=11, fontweight="bold", ha="center", va="bottom", color=hk.KEY)

    hk.save(fig, str(OUT / "figure3_grid_pro"))
    print("wrote figure3_grid_pro.{pdf,png}")


if __name__ == "__main__":
    main()
