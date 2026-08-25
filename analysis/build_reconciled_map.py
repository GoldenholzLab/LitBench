"""
NOTE: This script reads litbench's closed question bank
(queries_genuine.parquet), which is not included in this repository and is
not publicly available — the benchmark stays closed to keep it an
uncontaminated test set. This script is included for methodological
transparency only; it will not run without that private data and
LITBENCH_ROOT pointed at a full LitBench working tree.

Build consort/reconciled_query_map.json — the three-signal (demote) reconciliation
of the single-hop question bank. Reduction + relabel only; T5 (M3xxxx) is untouched
(absent from the map => the grid passes it through frozen).
"""
import json, os, unicodedata
from pathlib import Path
import pandas as pd

# In addition to LITBENCH_ROOT, this script imports `gold_freeze.run` from the
# private LitBench working tree -- that module is not bundled in this repo. It
# will not run standalone; put your LITBENCH_ROOT tree on PYTHONPATH (or run it
# from within that tree) to make `gold_freeze` importable.

LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
REPO = Path(LITBENCH_ROOT)

from gold_freeze.run import build_frozen, DEFAULT_CONSENSUS, DEFAULT_FAKE_SECTION, DEFAULT_FAKE_WHOLE  # noqa: E402
SECOND_REVIEW_PATH = REPO / "reviews_bakcup_20260715.csv"
QGEN = REPO / ".evie_cache" / "a5_largeN_t1t5" / "queries_genuine.parquet"
OUT = REPO / "consort" / "reconciled_query_map.json"

def _nfc(s):
    return unicodedata.normalize("NFC", str(s)) if s is not None else s

def build():
    frozen, reviewed, _ = build_frozen(
        str(DEFAULT_CONSENSUS), str(SECOND_REVIEW_PATH), str(DEFAULT_FAKE_SECTION),
        str(DEFAULT_FAKE_WHOLE), fake_thinking_action="demote")
    # fact key -> (status, final_domain)
    fd = {(_nfc(r["paper_id"]), _nfc(r["fact_id"])): (r["frozen_status"], r["final_domain"])
          for r in frozen}
    df = pd.read_parquet(QGEN)
    single, dropped = {}, []
    for _, row in df[df["hop"] == 1].iterrows():
        qid = row["query_id"]
        key = (_nfc(row["gold_paper_ids"][0]), _nfc(row["gold_fact_ids"][0]))
        st = fd.get(key)
        if st is None or st[0] != "kept":
            dropped.append(qid)
        else:
            single[qid] = st[1]                      # reconciled (possibly demoted) domain
    OUT.write_text(json.dumps({
        "mode": "demote",
        "second_review_source": SECOND_REVIEW_PATH.name,
        "reviewed_papers": reviewed,
        "single_hop": single,
        "dropped": sorted(dropped),
    }, indent=0))
    print(f"reconciled_query_map.json: {len(single)} kept, {len(dropped)} dropped (second review)")

if __name__ == "__main__":
    build()
