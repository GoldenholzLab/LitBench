"""
NOTE: This script reads litbench's closed question bank
(queries_genuine.parquet), which is not included in this repository and is
not publicly available — the benchmark stays closed to keep it an
uncontaminated test set. This script is included for methodological
transparency only; it will not run without that private data and
LITBENCH_ROOT pointed at a full LitBench working tree.

Build the Sonnet-5 candidate cells for the 129 single-hop bank items the
July-17/18 native run never attempted.

The reconciled single-hop evaluation bank is 2,188 questions; the Sonnet-5 native
run covers 2,059 of them (its todo list was built before reconciliation). The 129
remainder is what `supplement.md` S1.9 discloses as "not attempted for Sonnet-5",
and what plan Task 0.2 (denominator reconciliation) needs closed. This script
enumerates that set from the data (never a hardcoded list) and writes the cell
files the existing workflow generators consume, at the matched N-ladder used for
the 2,059: T1 N=1, T2 N=100, T3 N=200, T4 N=500.

REPRODUCING THE ORIGINAL RUN CONDITIONS (verified, not assumed)
--------------------------------------------------------------
The 2,059 cells' haystacks are frozen in `consort/out/t2t4_haystack_lists.json`
(built 2026-07-17). Two things have changed in the repo since, so today's default
code does NOT reproduce them:

  * corpus: `litbench/corpus.parquet` was NFC-deduplicated 2026-07-23
    (2,008 -> 1,980 rows; gold pool 1,028 -> 1,000). T3/T4 sample the gold pool,
    so the seeded draw moved. `corpus.parquet.pre_dedup` reproduces T2 and T3
    byte-identically (verified against the stored lists).
  * T4 composition: `litbench/core/haystack.py` was fixed 2026-07-20 to fill to N
    distractors. The 2,059 ran under the OLD under-filled rule -- gold + floor(N/3)
    off-topic + floor(N/3) near-neighbour, drawn OFF-TOPIC FIRST, so the labelled
    N=500 haystack held 333 papers. Every stored T4 list is length 333 (2,059/2,059).

`--t4 old`   reproduces the 2,059's under-filled T4 (poolable with them; carries
             the known stale-T4 defect).
`--t4 fixed` uses the current filled-to-N builder (comparable with the Gemma
             `a5full_sh_t4fix` regeneration, NOT with the existing Sonnet T4 --
             which then has to be regenerated too).

Outputs
  consort/out/sonnet_129_t1_cells.json      T1 cells (gold paper read)
  consort/out/sonnet_129_t2t3_cells.json    T2/T3 cells WITH gold (collection/verification)
  consort/out/sonnet_129_t2t3_blind.json    T2/T3 cells WITHOUT gold (what the agent sees)
  consort/out/sonnet_129_t4_cells.json      T4 cells WITH gold        (only with --t4)
  consort/out/sonnet_129_t4_blind.json      T4 cells WITHOUT gold     (only with --t4)
  consort/out/t2t4_haystack_lists.json      appended in place (search_haystack.py reads this)

Usage:
  python3 analysis/build_sonnet_129_cells.py --dry-run
  python3 analysis/build_sonnet_129_cells.py                 # T1 + T2 + T3 only
  python3 analysis/build_sonnet_129_cells.py --t4 old        # add the T4 leg
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
ROOT = Path(LITBENCH_ROOT)
sys.path.insert(0, str(ROOT))

# In addition to LITBENCH_ROOT, this script imports `litbench.core.haystack` and
# `litbench.core.seeding` (below) from the private LitBench working tree -- neither
# module is bundled in this repo. It will not run standalone; the `sys.path.insert`
# above (pointing at LITBENCH_ROOT) is what makes `litbench` importable, provided
# your LITBENCH_ROOT tree actually contains the `litbench` package.
import litbench.core.haystack as _H  # noqa: E402
from litbench.core.haystack import compose_haystack  # noqa: E402
from litbench.core.seeding import make_rng  # noqa: E402

# Read the corpus split ONCE, not per cell.
_split_cache: dict[str, tuple] = {}
_orig_split = _H._split_corpus


def _cached_split(corpus_path):
    key = str(corpus_path)
    if key not in _split_cache:
        _split_cache[key] = _orig_split(corpus_path)
    return _split_cache[key]


_H._split_corpus = _cached_split

# The corpus AS IT WAS when the 2,059 ran (and when the Gemma O2 runs composed
# their haystacks: pools gold=1028 / off-topic=980).
CORPUS_PARQUET = ROOT / "litbench/corpus.parquet.pre_dedup"
QUERIES = ROOT / ".evie_cache/a5_fullgrid_t1t5/queries_genuine.parquet"
RECONCILED = ROOT / "consort/reconciled_query_map.json"
SONNET_VERDICTS = ROOT / ".evie_cache/sonnet_poll/full_t1t4/poll_verdicts.json"
LISTS_PATH = ROOT / "consort/out/t2t4_haystack_lists.json"
OUT = ROOT / "consort/out"

T2T3 = [("T2", 100), ("T3", 200)]
T4 = ("T4", 500)


def compose_t4_old(query_id: str, n: int, gold_paper_ids: tuple) -> tuple:
    """The pre-2026-07-20 T4 rule, reverse-engineered from the stored lists and
    verified to reproduce them exactly (off-topic drawn BEFORE near-neighbour)."""
    gold_pool, distr = _H._split_corpus(CORPUS_PARQUET)
    rng = make_rng(query_id=query_id, test_type="T4", n=n)
    pool3 = [p for p in gold_pool if p not in gold_paper_ids]
    s2 = list(rng.sample(distr, min(n // 3, len(distr))))
    s3 = list(rng.sample(pool3, min(n // 3, len(pool3))))
    return tuple(gold_paper_ids) + tuple(s2) + tuple(s3)


def verify_reproduction() -> dict:
    """Recompute stored haystacks for cells the 2,059 already ran. Any mismatch
    means the new cells would NOT be drawn under the original conditions."""
    lists = json.loads(LISTS_PATH.read_text())
    q = pd.read_parquet(QUERIES)
    sh = q[q.hop == 1].set_index("query_id")
    checks, sample = {}, ["Q00001", "Q00003", "Q00007", "Q01000", "Q02000"]
    for tt, n in T2T3 + [T4]:
        ok = checked = 0
        for qid in sample:
            cell = f"{qid}_{tt}_N{n}"
            if cell not in lists:      # not part of the 2,059 -> nothing to compare against
                continue
            checked += 1
            gold = tuple(sh.loc[qid, "gold_paper_ids"])
            got = (list(compose_t4_old(qid, n, gold)) if tt == "T4" else
                   list(compose_haystack(corpus_path=CORPUS_PARQUET, query_id=qid,
                                         test_type=tt, n=n, gold_paper_ids=gold)))
            ok += (got == lists[cell])
        checks[f"{tt}_N{n}"] = f"{ok}/{checked} stored haystacks reproduced exactly"
    return checks


def missing_query_ids() -> list[str]:
    bank = set(json.loads(RECONCILED.read_text())["single_hop"])
    done = {r["query_id"] for r in json.loads(SONNET_VERDICTS.read_text())}
    return sorted(bank - done)


def build(dry_run: bool = False, t4_mode: str | None = None) -> dict:
    repro = verify_reproduction()   # surfaced in the report; inspect before trusting the cells
    qids = missing_query_ids()
    q = pd.read_parquet(QUERIES)
    sh = q[q.hop == 1].set_index("query_id")

    t1_cells, t2t3_cells, t2t3_blind = [], [], []
    t4_cells, t4_blind, new_lists = [], [], {}

    conds = list(T2T3) + ([T4] if t4_mode else [])
    for qid in qids:
        row = sh.loc[qid]
        gold = tuple(row.gold_paper_ids)
        question = str(row.query_text)

        t1_cells.append({
            "cell_id": f"{qid}_T1_N1", "query_id": qid, "test_type": "T1", "n": 1,
            "question": question, "gold_paper_id": gold[0],
        })

        for test_type, n in conds:
            cell_id = f"{qid}_{test_type}_N{n}"
            if test_type == "T4" and t4_mode == "old":
                pids = compose_t4_old(qid, n, gold)
            else:
                pids = compose_haystack(corpus_path=CORPUS_PARQUET, query_id=qid,
                                        test_type=test_type, n=n, gold_paper_ids=gold)
            new_lists[cell_id] = list(pids)
            cell = {"cell_id": cell_id, "query_id": qid, "test_type": test_type, "n": n,
                    "question": question, "gold_paper_id": gold[0]}
            blind = {"cell_id": cell_id, "query_id": qid, "question": question}
            (t4_cells if test_type == "T4" else t2t3_cells).append(cell)
            (t4_blind if test_type == "T4" else t2t3_blind).append(blind)

    report = {
        "queries": len(qids),
        "reproduction_check": repro,
        "corpus": CORPUS_PARQUET.name,
        "t4_mode": t4_mode or "SKIPPED (decide old vs fixed first)",
        "t1_cells": len(t1_cells), "t2t3_cells": len(t2t3_cells), "t4_cells": len(t4_cells),
        "realized_sizes": {tt_n: sorted({len(new_lists[f"{qid}_{tt_n}"]) for qid in qids})
                           for tt_n in {f"{tt}_N{n}" for tt, n in conds}},
    }
    if dry_run:
        return report

    (OUT / "sonnet_129_t1_cells.json").write_text(json.dumps(t1_cells, indent=1))
    (OUT / "sonnet_129_t2t3_cells.json").write_text(json.dumps(t2t3_cells, indent=1))
    (OUT / "sonnet_129_t2t3_blind.json").write_text(json.dumps(t2t3_blind, indent=1))
    if t4_cells:
        (OUT / "sonnet_129_t4_cells.json").write_text(json.dumps(t4_cells, indent=1))
        (OUT / "sonnet_129_t4_blind.json").write_text(json.dumps(t4_blind, indent=1))

    lists = json.loads(LISTS_PATH.read_text())
    # Re-running this builder is fine (it is deterministic); CHANGING a haystack an
    # earlier run already answered under is not.
    changed = [c for c in set(lists) & set(new_lists) if lists[c] != new_lists[c]]
    if changed:
        raise SystemExit(f"refusing to change {len(changed)} existing lists, e.g. {sorted(changed)[:3]}")
    if not LISTS_PATH.with_suffix(".json.bak").exists():
        shutil.copy2(LISTS_PATH, LISTS_PATH.with_suffix(".json.bak"))
    lists.update(new_lists)
    LISTS_PATH.write_text(json.dumps(lists))
    report["haystack_lists_total_after"] = len(lists)
    return report


if __name__ == "__main__":
    mode = None
    if "--t4" in sys.argv:
        mode = sys.argv[sys.argv.index("--t4") + 1]
        if mode not in ("old", "fixed"):
            raise SystemExit("--t4 must be 'old' or 'fixed'")
    print(json.dumps(build(dry_run="--dry-run" in sys.argv, t4_mode=mode), indent=1))
