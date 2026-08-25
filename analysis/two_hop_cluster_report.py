"""Two-hop pair-graph structure report: how independent are the 170 M3
two-hop questions, really?

Each M3 question is an edge between its two constituent papers (A, B). If the
paper-pair graph collapses into one giant connected component -- a few hub
review papers each co-occurring with many other papers -- then the 170
questions are not 170 independent draws from separate paper structures, even
though the question-level bootstrap used for the T5 headline (170 clusters)
is wider than a naive per-row bootstrap. This module quantifies that
structure (`pair_graph_stats`) and reruns the T5 both-hops headline with the
highest-degree hub papers excluded, so the paper can report whether the
headline depends on them (`hub_sensitivity`).

Reuses consort.compute_best3_t5.build_pair_components -- the same union-find
already used for that script's component-clustered CI attempt -- so the
"164 of 170 pairs in one component" figure can never diverge between scripts.

Usage: python3 analysis/two_hop_cluster_report.py
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from collections import Counter
from pathlib import Path

# In addition to LITBENCH_ROOT, this script imports `consort.compute_best3_t5`
# (below) and `consort.compute_composite_full` (below, inside `t5_both_hops_ci`)
# from the private LitBench working tree -- neither module is bundled in this
# repo. It will not run standalone; put your LITBENCH_ROOT tree on PYTHONPATH
# (or run it from within that tree) to make `consort` importable.

sys.path.insert(0, str(Path(__file__).resolve().parent))

LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
ROOT = Path(LITBENCH_ROOT)
M3_PATH = ROOT / "consort/out/m3_constituents.json"
OUT = ROOT / "consort/out/two_hop_cluster_report.json"

from consort.compute_best3_t5 import build_pair_components  # noqa: E402

# Natural break in the real degree distribution (25, 12, 10, then 8, 8, 8, ...):
# the three papers appearing in ten or more of the 170 pairs.
HUB_MIN_DEGREE = 10


def paper_degrees(m3: dict) -> Counter:
    """How many of the M3 pairs each constituent paper appears in."""
    degree: Counter = Counter()
    for rec in m3.values():
        degree[rec["A"]["paper_id"]] += 1
        degree[rec["B"]["paper_id"]] += 1
    return degree


def pair_graph_stats(m3: dict) -> dict:
    """Structure of the paper-pair graph underlying the two-hop (M3) question
    set: one node per constituent paper, one edge per M3 question linking its
    A and B papers. Reports the per-paper degree distribution, the share of
    pairs touching the single highest-degree ("hub") paper, and the size of
    the largest connected component, counted in pairs (not papers)."""
    degree = paper_degrees(m3)
    questions = len(m3)
    degrees = sorted(degree.values())

    hub_paper = max(degree, key=degree.get) if degree else None
    hub_touch = sum(1 for rec in m3.values()
                     if hub_paper in (rec["A"]["paper_id"], rec["B"]["paper_id"]))

    component_sizes = Counter(build_pair_components(m3).values())

    return {
        "n_papers": len(degree),
        "questions": questions,
        "max_degree": degrees[-1] if degrees else 0,
        "median_degree": statistics.median(degrees) if degrees else 0,
        "hub_paper": hub_paper,
        "hub_share": hub_touch / questions if questions else float("nan"),
        "n_components": len(component_sizes),
        "largest_component": max(component_sizes.values()) if component_sizes else 0,
    }


def hub_paper_ids(m3: dict, min_degree: int = HUB_MIN_DEGREE) -> set:
    """Papers appearing in at least `min_degree` of the M3 pairs."""
    return {p for p, d in paper_degrees(m3).items() if d >= min_degree}


def questions_touching(m3: dict, papers: set) -> set:
    return {qid for qid, rec in m3.items()
            if rec["A"]["paper_id"] in papers or rec["B"]["paper_id"] in papers}


def t5_both_hops_ci(question_ids: set = None) -> dict:
    """T5 both-hops (semantic, PRIMARY) accuracy per system, question-clustered
    bootstrap -- the same pipeline behind the paper's headline numbers
    (consort/compute_composite_full.t5_df + consort/bootstrap_ci), optionally
    restricted to a subset of M3 question ids (for the hub-exclusion check)."""
    from consort.compute_composite_full import t5_df
    from bootstrap_ci import system_accuracy_ci

    df = t5_df()
    df["test_type"] = "T5"
    df = df.drop(columns=["credit"]).rename(columns={"poll": "credit"})
    if question_ids is not None:
        df = df[df.question_id.isin(question_ids)]

    out = {}
    for system in ("small", "medium", "frontier"):
        p, lo, hi = system_accuracy_ci(df, system, "T5", n_boot=5000, seed=42)
        out[system] = {
            "acc": round(100 * p, 1) if p == p else float("nan"),
            "lo": round(100 * lo, 1) if lo == lo else float("nan"),
            "hi": round(100 * hi, 1) if hi == hi else float("nan"),
            "n_questions": int(df[df.system == system].question_id.nunique()),
        }
    return out


def hub_sensitivity(m3: dict, min_degree: int = HUB_MIN_DEGREE) -> dict:
    """Recompute the T5 both-hops headline excluding every M3 question that
    touches a hub paper (degree >= min_degree), so the full-set and
    hub-excluded headlines can be compared side by side."""
    hubs = hub_paper_ids(m3, min_degree)
    excluded = questions_touching(m3, hubs)
    retained = set(m3) - excluded
    return {
        "min_degree": min_degree,
        "hub_papers": sorted(hubs),
        "n_hub_papers": len(hubs),
        "n_excluded_questions": len(excluded),
        "n_retained_questions": len(retained),
        "full_set": t5_both_hops_ci(),
        "hub_excluded": t5_both_hops_ci(retained),
    }


def main() -> None:
    m3 = json.loads(M3_PATH.read_text())
    stats = pair_graph_stats(m3)
    sensitivity = hub_sensitivity(m3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"pair_graph_stats": stats, "hub_sensitivity": sensitivity}, indent=2))

    print(f"pair graph: {stats['n_papers']} papers, {stats['questions']} questions")
    print(f"  degree: max {stats['max_degree']}, median {stats['median_degree']}")
    print(f"  top hub: {stats['hub_paper']} ({100 * stats['hub_share']:.1f}% of pairs)")
    print(f"  components: {stats['n_components']}, largest = {stats['largest_component']} "
          f"of {stats['questions']} pairs")

    print(f"\nhub sensitivity (excluding {sensitivity['n_hub_papers']} papers with degree "
          f">= {sensitivity['min_degree']}, {sensitivity['n_excluded_questions']} questions dropped):")
    for system in ("small", "medium", "frontier"):
        full = sensitivity["full_set"][system]
        red = sensitivity["hub_excluded"][system]
        print(f"  {system:9s} full {full['acc']:5.1f}% [{full['lo']:.1f},{full['hi']:.1f}] "
              f"(n={full['n_questions']:3d})   hub-excluded {red['acc']:5.1f}% "
              f"[{red['lo']:.1f},{red['hi']:.1f}] (n={red['n_questions']:3d})")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
