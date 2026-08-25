"""Matched single-hop comparison for the paper's headline reviewer fix: frontier
(Sonnet-5 native, NEW full-benchmark run) vs small (Gemma-4B) vs medium (Gemma-12B),
on the SAME reconciled queries at the SAME haystack N per tier, with question-
clustered bootstrap 95% CIs and paired differences.

Frontier single-hop now = .evie_cache/sonnet_poll/full_t1t4/poll_verdicts.json
(2,059 questions x T1-T4), replacing the old 132-cell sample.
"""
import json
import collections
import os
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bootstrap_ci import system_accuracy_ci, paired_diff_ci

LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
ROOT = Path(LITBENCH_ROOT)
LN = ROOT / ".evie_cache/a5_largeN_t1t5"
FRONT = ROOT / ".evie_cache/sonnet_poll/full_t1t4/poll_verdicts.json"
FN = {"T1": 1, "T2": 100, "T3": 200, "T4": 500}   # frontier's representative N per tier


def _load(p):
    return json.loads(Path(p).read_text())


def build_df():
    small = _load(LN / "single_judge_out/gemma-4b/poll_verdicts.json")
    medium = _load(LN / "single_judge_out/gemma-12b/poll_verdicts.json")
    front = _load(FRONT)
    kept = set(json.loads((ROOT / "consort/reconciled_query_map.json").read_text())["single_hop"])
    # matched query universe: present in all three systems AND kept by reconciliation
    qs = (set(r["query_id"] for r in small) & set(r["query_id"] for r in medium)
          & set(r["query_id"] for r in front) & kept)
    rows = []
    for sysname, v in (("small", small), ("medium", medium), ("frontier", front)):
        for r in v:
            tt = r["test_type"]
            if tt in FN and r["query_id"] in qs and r["n"] == FN[tt]:
                rows.append(dict(question_id=r["query_id"], system=sysname,
                                 test_type=tt, N=r["n"], credit=int(r["poll"])))
    return pd.DataFrame(rows), len(qs)


def pct(t):
    return tuple(round(100 * x, 1) for x in t)


if __name__ == "__main__":
    df, nq = build_df()
    print(f"matched single-hop queries: {nq} | rows: {len(df)}")
    print("\n== Per-system single-hop accuracy [95% CI], question-clustered bootstrap (n_boot=5000) ==")
    print(f"{'tier':<5} {'small (4B)':<22} {'medium (12B)':<22} {'frontier (Sonnet)':<22}")
    for tt in ("T1", "T2", "T3", "T4"):
        cells = []
        for s in ("small", "medium", "frontier"):
            p, lo, hi = pct(system_accuracy_ci(df, s, tt, n_boot=5000, seed=42))
            cells.append(f"{p:5.1f} [{lo:.1f}, {hi:.1f}]")
        print(f"{tt:<5} {cells[0]:<22} {cells[1]:<22} {cells[2]:<22}")
    print("\n== Paired differences [95% CI] (question-clustered, matched queries) ==")
    for tt in ("T1", "T4"):
        print(f" {tt}:")
        for a, b in (("frontier", "medium"), ("frontier", "small"), ("medium", "small")):
            d, lo, hi = pct(paired_diff_ci(df, a, b, tt, n_boot=5000, seed=42))
            print(f"   {a:8s} - {b:8s}: {d:+5.1f} pp  [{lo:+.1f}, {hi:+.1f}]")
