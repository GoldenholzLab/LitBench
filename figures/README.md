# Figure-generation code

Nine scripts build figures for the paper. This repository holds the
figure-development history, not a clean one-script-per-published-figure
mapping: several scripts below are successive iterations toward the same
published figure slot (three separate "Figure 2" candidates, five separate
"Figure 3" candidates), and not every published figure number is
guaranteed to have a builder here — extracting the actual output filenames
gives no Figure 4, S2, or S3 at all.

| Script | Output figure(s) | Data source |
|---|---|---|
| `build_paper_figures.py` | `figure3_tier_accuracy`, `figureS4_wall` | hardcoded in-file (no `LITBENCH_ROOT` needed) |
| `build_grid_figure.py` | `figure2_grid` | definitional — no data dependency |
| `build_report_card.py` | `figure1_report_card` | `LITBENCH_ROOT` |
| `build_heatmap.py` | `figure2_heatmap`, `figureA_heatmap_with31b` | `LITBENCH_ROOT` |
| `build_heatmap_full.py` | `figure2_heatmap_full` | `LITBENCH_ROOT` — **PREVIEW**: the script's own docstring labels this preview-only, pending a three-model judge-panel swap before it is final |
| `build_domain_grid.py` | `figure3_domain_grid`, `figureS1a_domain_heatmap`, `figureS1b_domain_heatmap` | `LITBENCH_ROOT` |
| `build_grid_pro.py` | `figure3_grid_pro` | `LITBENCH_ROOT` |
| `build_degradation_curves.py` | `figure3_degradation` | `LITBENCH_ROOT` |
| `build_tier_degradation.py` | `figure3_tier_degradation` | `LITBENCH_ROOT`, plus the closed question bank directly for `query_id`/`gold_paper_ids` only (no question or fact text) |

Each script that needs data outside this repo reads pre-computed aggregate
result files (e.g. `poll_verdicts.json`, `reconciled_query_map.json`, and
similar judge-verdict/summary JSON or CSV files) from the private LitBench
working tree via the `LITBENCH_ROOT` environment variable; point it at a
full LitBench working tree to regenerate figures from source. No figure
script reads `../analysis/data/` — that directory is read by nothing in
this repository (see `analysis/README.md`). Other than
`build_tier_degradation.py`'s two id columns noted above, none of the
scripts read the question bank, gold facts, or corpus text directly.
