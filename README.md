# LitBench 
The list of published articles that make up the LitBench corpus: **1,980 open-access papers**,
of which **1,000 are epilepsy papers that carry the answers** and **980 are non-epilepsy decoys**
that carry none and exist only as competing papers to search through.

The benchmark itself — the question bank, the gold facts, and the per-cell scorecards — is **not
released**, so that LitBench stays an uncontaminated test set. Publishing the article list does not
compromise that: it says which papers were searched, not which facts were asked about or what the
correct answers are. The bank is available from the authors on reasonable request.

## Files

| file | rows | what it is |
|---|---|---|
| `corpus_papers.csv` | 1,980 | every paper in the corpus |
| `gold_papers.csv` | 1,000 | the epilepsy papers that carry the answers |
| `decoy_papers.csv` | 980 | the non-epilepsy papers that carry none |

## Columns

| column | meaning |
|---|---|
| `paper_id` | LitBench's internal identifier |
| `role` | `gold (carries answers)` or `decoy (no answer)` |
| `stratum` | sampling stratum — `epilepsy` for gold; the four decoy strata below |
| `identifier_type` | `PMID` for gold papers, `PMCID` for decoys |
| `identifier` | the PubMed or PubMed Central identifier |
| `doi` | DOI |
| `title` | article title |
| `journal` | journal name |
| `year` | publication year |
| `url` | resolvable link to the record |

Titles, journals, years and DOIs were resolved from NCBI E-utilities against the identifier in
`paper_id`, not taken from the parsed PDFs — the PDF-extracted titles in the build artifacts are
frequently running headers ("May 2017 | Volume 8 | Article 179") rather than titles. All 1,980
papers resolved.

## Decoy strata

The 980 decoys are drawn from four strata outside epilepsy:

| stratum | papers |
|---|---|
| `oncology_endo_cardio` | 250 |
| `internal_medicine` | 250 |
| `surgery_derm_id` | 249 |
| `neurology_non_epilepsy` | 231 |

Note the last one: 231 of the 980 decoys are **neurology papers that are not about epilepsy**.
They are the hardest decoys in the corpus — topically adjacent, so a system cannot exclude them on
subject matter alone — and they are why the paper describes the decoys as carrying no answer rather
than as being unrelated.

## Reproducing this file

```
python3 -c "import pandas as pd; print(pd.read_parquet('litbench/corpus.parquet').shape)"
```

The manifest is generated from `litbench/corpus.parquet` plus NCBI lookups. The corpus parquet is
not in this folder; it contains full article text.

## Analysis and figure-generation code

`analysis/` and `figures/` contain the code that produced the confidence
intervals, contrasts, and figures reported in the paper: bootstrap and
paper-clustered CI routines, the two-paper (T5) paper-level bootstrap and
hub-paper connectivity check behind eTable S1.9, the three-system contrast
computation, and the Figure 1-4 / S1-S4 builders.

This is the analysis layer, not the benchmark. It does not include the
question bank, the gold facts, judge verdicts, or the corpus text — those
stay closed for the reason given above. Three scripts
(`analysis/build_reconciled_map.py`, `analysis/build_sonnet_129_cells.py`,
`figures/build_tier_degradation.py`) read from the closed question bank and
are included for methodological transparency; they will not run without
that private data. Every other script in `analysis/` and `figures/` runs
against the small aggregate JSON files bundled in `analysis/data/`, which
contain only numbers, confidence intervals, and opaque question identifiers
— no question or fact text.

See `analysis/README.md` and `figures/README.md` for what each script does
and how to run it.
