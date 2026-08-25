# Analysis code

Statistics and reproducibility scripts behind the paper's reported numbers:

| Script | What it computes |
|---|---|
| `bootstrap_ci.py` | question-clustered bootstrap confidence intervals |
| `paper_clustered_ci.py` | single-paper accuracy CIs, clustered by source paper |
| `two_hop_graph_ci.py` | paper-level (graph-aware) bootstrap for the two-paper (T5) condition (**requires `LITBENCH_ROOT` — see file header**) |
| `two_hop_cluster_report.py` | the paper-pair connectivity check behind eTable S1.9 (**requires `LITBENCH_ROOT` — see file header**) |
| `compute_frontier_full.py` | the three-system head-to-head contrasts reported in Results (**requires `LITBENCH_ROOT` — see file header**) |
| `common_subset_report.py` | common-subset coverage/shortfall report (**requires `LITBENCH_ROOT` — see file header**) |
| `constituent_fact_analysis.py` | T5 constituent-fact (both-facts-credited) analysis |
| `realized_haystack_composition.py` | candidate-set composition check across N (**requires `LITBENCH_ROOT` — see file header**) |
| `merge_gap129_verdicts.py` | merges judge verdicts for the Sonnet-5 gap-129 regeneration (**requires `LITBENCH_ROOT` — see file header**) |
| `build_reconciled_map.py` | builds the reconciled query map (**requires private data — see file header**) |
| `build_sonnet_129_cells.py` | builds the Sonnet-5 gap-129 cell list (**requires private data — see file header**) |

`data/` holds published aggregate result numbers behind the paper's figures
and tables — accuracy values, confidence intervals, and opaque question
identifiers only, no question or fact text — kept here as static reference
data. No script in this directory reads `data/` back in: for example,
`common_subset_report.py` *writes* `common_subset_report.json`, but into the
private `LITBENCH_ROOT` tree, not into this bundled copy.

Only three scripts are pure library functions callable with no private data
at all: `bootstrap_ci.py`, `paper_clustered_ci.py`, and
`constituent_fact_analysis.py`. Every other script in this directory needs
`LITBENCH_ROOT` pointed at a full LitBench working tree to run;
`realized_haystack_composition.py` additionally imports
`litbench.core.haystack`, a private package not bundled in this repo (see
its file header).
