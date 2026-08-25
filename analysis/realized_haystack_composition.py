"""Realized (not intended) T1-T5 haystack composition, tabulated from the
actual SHA-256-seeded generator in litbench/core/haystack.py.

Closes the "the T4 mix may be mathematically impossible" reviewer concern:
compose_haystack's docstring describes an *intended* ~1/3 near-neighbour +
~2/3 out-of-domain split for T4, but at large N the out-of-domain pool
(980 papers) is exhausted and the shortfall is backfilled with additional
near-neighbour papers. This module calls the real generator and counts what
it actually returned, rather than trusting the intended ratio.
"""
from __future__ import annotations

import os
from pathlib import Path

# In addition to LITBENCH_ROOT, this script imports `litbench.core.haystack`
# from the private LitBench working tree -- that module is not bundled in
# this repo. It will not run standalone; put your LITBENCH_ROOT tree on
# PYTHONPATH (or run it from within that tree) to make `litbench` importable.

LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
CORPUS_PATH = Path(LITBENCH_ROOT) / "litbench" / "corpus.parquet"

from litbench.core.haystack import _split_corpus, compose_haystack  # noqa: E402


def realized_composition(qid: str, test_type: str, n: int,
                          corpus_path: Path = CORPUS_PATH) -> dict:
    """Return the realized haystack composition for (qid, test_type, n).

    Uses fixed representative gold paper(s) (the first id(s) in the gold
    pool) as the query's gold paper(s): compose_haystack's sampling depends
    only on the seed and the pool sizes, not on which gold id(s) are
    excluded from the near-neighbour pool, so any fixed choice of gold ids
    is deterministic and sufficient to tabulate realized counts. T5 is the
    two-hop tier and is built from two gold papers; every other test type
    uses one.
    """
    gold_pool, distr_pool = _split_corpus(corpus_path)
    n_gold = 2 if test_type == "T5" else 1
    gold_ids = tuple(gold_pool[:n_gold])
    gold_set = set(gold_pool)
    distr_set = set(distr_pool)

    haystack = compose_haystack(corpus_path=corpus_path, query_id=qid,
                                test_type=test_type, n=n,
                                gold_paper_ids=gold_ids)

    gold_count = sum(1 for p in haystack if p in gold_ids)
    near_count = sum(1 for p in haystack if p in gold_set and p not in gold_ids)
    ood_count = sum(1 for p in haystack if p in distr_set)

    replaced = len(haystack) != len(set(haystack))
    backfilled_near = max(0, near_count - n // 3)

    return {
        "gold": gold_count,
        "near_neighbour": near_count,
        "out_of_domain": ood_count,
        "total": len(haystack),
        "replaced": replaced,
        "backfilled_near": backfilled_near,
    }
