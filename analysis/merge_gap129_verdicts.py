"""Append the 129-question gap run's verdicts to the three Sonnet-5 artifacts.

The Sonnet-5 native run covered 2,059 of the reconciled 2,188-item single-hop bank.
The 129 missing questions were generated 2026-07-26 and judged on O2 by all three
scorers the paper uses. This script folds those verdicts into the artifacts the
analysis reads, taking each artifact to the full 2,188 x 4 = 8,752 cells.

  best3      manifests/best3_rejudge/frontier/poll_verdicts.json   (NUMBERS_v2 headline)
  poll       .evie_cache/sonnet_poll/full_t1t4/poll_verdicts.json  (figures, compute_frontier_full)
  composite  consort/composite_verdicts/composite_sonnet_sh.json   (span-gated secondary)

Each target gets a .bak first. Rows are emitted in the target's OWN schema (the shard
schemas differ: poll_credit/composite_credit/span_credit vs poll/composite/span; and the
best-3 shards carry per-judge `matched` rather than `votes`, so n_judges_voted is derived
from `matched`). Appending is refused if any (query_id, test_type, n) already exists.

Usage:
  python3 analysis/merge_gap129_verdicts.py --dry-run
  python3 analysis/merge_gap129_verdicts.py best3 poll composite
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import sys
from pathlib import Path

LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
ROOT = Path(LITBENCH_ROOT)
SHARDS = ROOT / ".evie_cache/sonnet_poll/gap129"

TARGETS = {
    "best3": ROOT / "manifests/best3_rejudge/frontier/poll_verdicts.json",
    "poll": ROOT / ".evie_cache/sonnet_poll/full_t1t4/poll_verdicts.json",
    "composite": ROOT / "consort/composite_verdicts/composite_sonnet_sh.json",
}
SHARD_DIR = {"best3": "best3_out", "poll": "poll_out", "composite": "composite_out"}
EXPECT_CELLS = 516  # 129 questions x T1/T2/T3/T4


def _judges(row) -> int:
    """How many panel members returned a verdict for this cell."""
    if row.get("votes"):
        return sum(1 for v in row["votes"].values() if v is not None)
    return len(row.get("matched") or {})


def _shard_rows(kind: str) -> list:
    rows = []
    for p in sorted(glob.glob(str(SHARDS / SHARD_DIR[kind] / "shard_*.json"))):
        if Path(p).name.startswith("._"):
            continue
        rows.extend(json.loads(Path(p).read_text()))
    # PoLL shard rows carry no cell_id, so key on the cell coordinates instead.
    dedup, seen = [], set()
    for r in rows:
        k = r.get("cell_id") or (r["query_id"], r["test_type"], r["n"])
        if k in seen:
            continue
        seen.add(k)
        dedup.append(r)
    return dedup


def _emit(kind: str, r: dict) -> dict:
    """Shard row -> the target artifact's schema."""
    if kind == "best3":
        c = r["poll_credit"]
        return {"query_id": r["query_id"], "candidate": r.get("candidate"),
                "test_type": r["test_type"], "n": r["n"],
                "poll": c, "both_hops": c, "any_hop": c, "n_judges_voted": _judges(r)}
    if kind == "poll":
        return {"query_id": r["query_id"], "candidate": r.get("candidate"),
                "test_type": r["test_type"], "n": r["n"],
                "both_hops": r["both_hops_credit"], "any_hop": r["any_hop_credit"],
                "poll": r["poll_credit"], "n_judges_voted": r.get("n_judges_voted", 0)}
    if kind == "composite":
        return {"query_id": r["query_id"], "candidate": r.get("candidate"),
                "model": r.get("model"), "test_type": r["test_type"], "n": r["n"],
                "cell_id": r["cell_id"], "poll": r["poll_credit"],
                "composite": r["composite_credit"], "both_hops": r["both_hops"],
                "both_hops_composite": r["both_hops_composite"], "any_hop": r["any_hop"],
                "any_hop_composite": r["any_hop_composite"], "span": r["span_credit"]}
    raise SystemExit(f"unknown kind {kind}")


def merge(kind: str, dry_run: bool = False) -> dict:
    target = TARGETS[kind]
    existing = json.loads(target.read_text())
    have = {(r["query_id"], r["test_type"], r["n"]) for r in existing}

    rows = _shard_rows(kind)
    new = [_emit(kind, r) for r in rows]
    clash = [k for k in ((r["query_id"], r["test_type"], r["n"]) for r in new) if k in have]
    report = {
        "target": str(target.relative_to(ROOT)), "existing": len(existing),
        "new_cells": len(new), "new_queries": len({r["query_id"] for r in new}),
        "collisions": len(clash), "total_after": len(existing) + len(new),
    }
    if clash:
        raise SystemExit(f"{kind}: {len(clash)} cells already present, e.g. {clash[:3]} — refusing")
    if len(new) != EXPECT_CELLS:
        raise SystemExit(f"{kind}: expected {EXPECT_CELLS} cells, got {len(new)} — refusing")
    if dry_run:
        return report

    bak = target.with_suffix(".json.pre_gap129.bak")
    if not bak.exists():
        shutil.copy2(target, bak)
    target.write_text(json.dumps(existing + new, indent=1))
    report["queries_after"] = len({r["query_id"] for r in existing + new})
    return report


if __name__ == "__main__":
    dry = "--dry-run" in sys.argv
    kinds = [a for a in sys.argv[1:] if a in TARGETS] or list(TARGETS)
    print(json.dumps({k: merge(k, dry) for k in kinds}, indent=1))
