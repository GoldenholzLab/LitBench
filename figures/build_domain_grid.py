#!/usr/bin/env python3
"""Render the IRCoT full within-corpus performance grid in two versions:

  figure3_domain_grid      -> MAIN Figure 3: three systems (Gemma-4B, Gemma-12B,
                              Sonnet-5), fully populated (Gemma-31B band dropped).
  figureS1_domain_heatmap  -> Figure S1: all four model bands, incl. the partial
                              Gemma-31B band (gray = its per-fill sweep was not run).

Same grid, same structure, same colorway (consort/combined_heatmap.render_combined,
continuous RdYlGn). Data + layout come straight from consort/candidate_grid.assemble;
this only slices out the Gemma-31B band for the main figure.
"""
import os
import sys
from pathlib import Path

# In addition to LITBENCH_ROOT, this script imports `consort.candidate_grid` and
# `consort.combined_heatmap` from the private LitBench working tree -- neither
# module is bundled in this repo. It will not run standalone; LITBENCH_ROOT
# (below) is inserted onto sys.path so `consort` is importable, provided your
# LITBENCH_ROOT tree actually contains the `consort` package.
LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
sys.path.insert(0, LITBENCH_ROOT)
from consort.candidate_grid import CANDIDATES, assemble, ROW_LABELS, COL_LABELS, NCOL  # noqa: E402
from consort.combined_heatmap import render_combined  # noqa: E402

OUT = Path(__file__).resolve().parent

# The grid's internal row vocabulary keys into archived run records and cannot be
# renamed upstream; "Recall" and "Thinking" are those internal keys and appear nowhere
# in the manuscript. Every reader-facing mention of the two fact types is "stated
# outright" / "requiring interpretation", so map at render time. The fact type goes on
# its own line under the section name: it is nearly the width of the longest one-line
# label the grid already carried, so the page does not have to grow to hold it.
FACT_TYPE = {"Recall": "stated outright", "Thinking": "requiring interpretation"}


def _paper_label(rl):
    section, _, fact = rl.rpartition(" ")
    return f"{section}\n{FACT_TYPE[fact]}" if fact in FACT_TYPE else rl


PAPER_ROW_LABELS = [_paper_label(rl) for rl in ROW_LABELS]
if sum("\n" in rl for rl in PAPER_ROW_LABELS) != 14:
    raise SystemExit("expected all 14 domain rows to carry a fact type; got "
                     f"{sum(chr(10) in rl for rl in PAPER_ROW_LABELS)}")

a5 = next(c for c in CANDIDATES if c.key == "a5-ircot")
matrix_all, status_all, band_labels_all = assemble(a5)   # 4 bands as scored: Gemma-4B, Gemma-12B, Gemma-31B, Sonnet-5
print("bands (as scored):", band_labels_all)

# Gemma-31B was a partial, exploratory run (never part of the main comparison) and has
# been dropped from the paper entirely. Every figure this script renders excludes it,
# so there is exactly one band-set from here on, not a per-figure special case.
drop = band_labels_all.index("Gemma-31B")
keep = [i for i in range(len(band_labels_all)) if i != drop]
cols = [b * NCOL + c for b in keep for c in range(NCOL)]
matrix = [[row[c] for c in cols] for row in matrix_all]
status = [[row[c] for c in cols] for row in status_all]
band_labels = [band_labels_all[b] for b in keep]
print("bands (rendered, Gemma-31B excluded):", band_labels)

# --- Figure S1: the three-system grid, = the approved version ---
render_combined([(a5.label, matrix, status)], PAPER_ROW_LABELS, band_labels, COL_LABELS,
                str(OUT / "figureS1_domain_heatmap"), title=None, caption=None)
print("wrote figureS1_domain_heatmap.{png,pdf}  (3 bands)")

# --- Figure 3: same three-band grid (kept for parity with figureS1_domain_heatmap;
# main-text Figure 3 is now built separately by build_degradation_curves.py) ---
render_combined([(a5.label, matrix, status)], PAPER_ROW_LABELS, band_labels, COL_LABELS,
                str(OUT / "figure3_domain_grid"), title=None, caption=None)
print("wrote figure3_domain_grid.{png,pdf}  (3 bands:", band_labels, ")")

# --- Figure S1, legible split -------------------------------------------------
# The full grid is natively wide (44 columns x 15 rows); placed at the supplement's
# text width its rows are too short to read in print. Splitting into two panels on a
# rotated page roughly triples the per-row height at the same page width, at the cost
# of one extra figure part. Same data, same colorway, no re-scoring. With Gemma-31B
# dropped, S1a keeps its original two bands and S1b now carries Sonnet-5 alone.
HALVES = [("S1a", ["Gemma-4B", "Gemma-12B"]), ("S1b", ["Sonnet-5"])]
for suffix, want in HALVES:
    idx = [band_labels.index(b) for b in want]
    cols = [b * NCOL + c for b in idx for c in range(NCOL)]
    m = [[row[c] for c in cols] for row in matrix]
    st = [[row[c] for c in cols] for row in status]
    render_combined([(a5.label, m, st)], PAPER_ROW_LABELS, want, COL_LABELS,
                    str(OUT / f"figure{suffix}_domain_heatmap"), title=None, caption=None)
    print(f"wrote figure{suffix}_domain_heatmap.{{png,pdf}}  ({want})")
