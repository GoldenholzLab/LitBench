"""Common single-hop evaluation subset: bank size vs. matched-N vs. per-system shortfall.

The reconciled single-hop bank (2,188 items, `consort/reconciled_query_map.json`'s
"single_hop" key) is larger than the subset actually scored for the three-system
matched contrast (Gemma-4B + a5-ircot, Gemma-12B + a5-ircot, Sonnet-5 + native
retrieval): both Gemma systems ran a superset of the bank (2,252 raw single-hop
cells, of which 64 were later dropped by reconciliation), but the Sonnet-5
native-retrieval generation run was frozen at 2,059 cells before the two-reviewer
reconciliation was finalized, so 129 bank items were never submitted to it for an
answer. This module recomputes both numbers directly from the same three files
`consort/build_release_manifest.py` reads for its `runs` / `matched` block (never
hardcodes them), so the manuscript's denominator language stays honest, and
characterizes the shortfall (bank minus common) by which system(s) never ran it and
by its domain / fact-type distribution, to support the claim that the gap is a
run-coverage artifact rather than concentrated in one part of the benchmark.

Usage: python3 analysis/common_subset_report.py
"""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path

LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
ROOT = Path(LITBENCH_ROOT)
BANK_PATH = ROOT / "consort/reconciled_query_map.json"
OUT = ROOT / "consort/out/common_subset_report.json"

# Same three judged-verdict files consort/build_release_manifest.py reads for its
# per-system single-hop run sizes and matched-common universe.
SYSTEMS = {
    "gemma-4b": ROOT / "manifests/best3_rejudge/gemma-4b/poll_verdicts.json",
    "gemma-12b": ROOT / "manifests/best3_rejudge/gemma-12b/poll_verdicts.json",
    "sonnet5-native": ROOT / "manifests/best3_rejudge/frontier/poll_verdicts.json",
}


def _t1_qids(path: Path) -> set:
    rows = json.loads(path.read_text())
    return {r["query_id"] for r in rows if r.get("test_type") == "T1"}


def _split_tag(tag: str) -> tuple:
    """'abstract_thinking' -> ('abstract', 'thinking')."""
    domain, _, fact_type = tag.rpartition("_")
    return domain, fact_type


def compute_common_subset_report(bank_path: Path = BANK_PATH,
                                  systems: dict = None) -> dict:
    """Bank size, per-system single-hop (T1) run size, the common (matched)
    universe, the shortfall (bank minus common), which system(s) each shortfall
    query is missing from, and the shortfall's distribution by domain and by
    fact type (recall/thinking)."""
    systems = SYSTEMS if systems is None else systems
    bank = json.loads(bank_path.read_text())["single_hop"]
    bank_set = set(bank)

    run_qids = {name: _t1_qids(path) for name, path in systems.items()}
    common = set(bank_set)
    for qids in run_qids.values():
        common &= qids

    shortfall = sorted(bank_set - common)
    missing_from = {qid: sorted(name for name, qids in run_qids.items() if qid not in qids)
                    for qid in shortfall}

    domain_counts, fact_type_counts, tag_counts = Counter(), Counter(), Counter()
    for qid in shortfall:
        domain, fact_type = _split_tag(bank[qid])
        domain_counts[domain] += 1
        fact_type_counts[fact_type] += 1
        tag_counts[bank[qid]] += 1

    return {
        "bank_n": len(bank_set),
        "run_n": {name: len(qids) for name, qids in run_qids.items()},
        "common_n": len(common),
        "shortfall_n": len(shortfall),
        "shortfall_qids": shortfall,
        "shortfall_missing_from": missing_from,
        "shortfall_by_domain": dict(sorted(domain_counts.items(), key=lambda kv: -kv[1])),
        "shortfall_by_fact_type": dict(sorted(fact_type_counts.items(), key=lambda kv: -kv[1])),
        "shortfall_by_tag": dict(sorted(tag_counts.items(), key=lambda kv: -kv[1])),
    }


def main() -> None:
    report = compute_common_subset_report()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2))

    print(f"bank={report['bank_n']}  common={report['common_n']}  "
          f"shortfall={report['shortfall_n']}")
    print("per-system single-hop (T1) run sizes:", report["run_n"])

    only_missing = Counter(tuple(m) for m in report["shortfall_missing_from"].values())
    print("shortfall missing-from pattern (system(s) that never ran it):", dict(only_missing))

    print(f"shortfall by domain (of {report['shortfall_n']}):", report["shortfall_by_domain"])
    print(f"shortfall by fact type (of {report['shortfall_n']}):", report["shortfall_by_fact_type"])

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
