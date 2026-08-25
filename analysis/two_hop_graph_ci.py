"""Graph-aware two-hop confidence intervals: resample the constituent PAPERS,
not the questions.

Each of the 170 two-hop (M3) questions is an edge between its two constituent
papers, and `consort/two_hop_cluster_report.py` already established that a few
hub papers place 164 of those 170 pairs in a single connected component. A
per-question bootstrap treats the 170 pairs as exchangeable draws, which they
are not: withholding one hub paper removes many pairs at once. This module
instead resamples the paper set with replacement and lets every pair incident
to a resampled paper carry its result forward (a pair whose two papers are
both drawn appears twice; a pair whose one paper is drawn k times appears k
times), so the graph's dependence structure is baked into the resample itself
rather than assumed away.

Usage: python3 analysis/two_hop_graph_ci.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

# In addition to LITBENCH_ROOT, this script imports `consort.compute_composite_full`
# (below, inside `t5_pair_scores_by_system`) from the private LitBench working tree --
# that module is not bundled in this repo. It will not run standalone; put your
# LITBENCH_ROOT tree on PYTHONPATH (or run it from within that tree) to make
# `consort` importable.

LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
ROOT = Path(LITBENCH_ROOT)
M3_PATH = ROOT / "consort/out/m3_constituents.json"
OUT = ROOT / "consort/out/two_hop_graph_ci.json"

N_BOOT = 5000
SEED = 42


def _pair_paper_indices(pairs, pair_papers):
    """Distinct paper count, plus each pair's two papers as indices into the
    sorted paper list (so a bootstrap draw of paper indices can be turned into
    per-pair multiplicities with a single bincount)."""
    papers = sorted({p for a, b in pair_papers.values() for p in (a, b)})
    idx = {p: i for i, p in enumerate(papers)}
    paper_a = np.array([idx[pair_papers[pid][0]] for pid in pairs])
    paper_b = np.array([idx[pair_papers[pid][1]] for pid in pairs])
    return len(papers), paper_a, paper_b


def _bootstrap_multiplicities(n_papers, paper_a, paper_b, rng):
    """One resample of the paper set (with replacement, same size as the
    original): how many times each pair appears when a pair is counted once
    per resampled occurrence of each of its two constituent papers."""
    draw = rng.integers(0, n_papers, n_papers)
    counts = np.bincount(draw, minlength=n_papers)
    return counts[paper_a] + counts[paper_b]


def paper_level_bootstrap(pair_scores: dict, pair_papers: dict,
                          n_boot: int = N_BOOT, seed: int = SEED) -> tuple:
    """Paper-level (graph-aware) bootstrap 95% CI for a two-hop headline.

    pair_scores: pair_id -> score (e.g. both-hops credit, mean over that
        pair's N-cells).
    pair_papers: pair_id -> (paperA_id, paperB_id).

    Returns (point, lo, hi). The point estimate is the plain mean over pairs
    (unchanged from a per-question bootstrap); only the interval reflects the
    paper-level resampling.
    """
    pairs = sorted(pair_scores)
    scores = np.array([pair_scores[pid] for pid in pairs], dtype=float)
    point = float(scores.mean())

    n_papers, paper_a, paper_b = _pair_paper_indices(pairs, pair_papers)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        mult = _bootstrap_multiplicities(n_papers, paper_a, paper_b, rng)
        total = mult.sum()
        boots[i] = (mult * scores).sum() / total if total else float("nan")
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paper_level_paired_diff_bootstrap(pair_scores_a: dict, pair_scores_b: dict,
                                       pair_papers: dict, n_boot: int = N_BOOT,
                                       seed: int = SEED) -> tuple:
    """Paired-difference counterpart of `paper_level_bootstrap`: both systems
    share one paper resample per iteration (not two independent ones), so the
    interval reflects the paired design the same way `bootstrap_ci.paired_diff_ci`
    does for the per-question version."""
    pairs = sorted(set(pair_scores_a) & set(pair_scores_b))
    scores_a = np.array([pair_scores_a[pid] for pid in pairs], dtype=float)
    scores_b = np.array([pair_scores_b[pid] for pid in pairs], dtype=float)
    point = float(scores_a.mean() - scores_b.mean())

    n_papers, paper_a, paper_b = _pair_paper_indices(pairs, pair_papers)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        mult = _bootstrap_multiplicities(n_papers, paper_a, paper_b, rng)
        total = mult.sum()
        if not total:
            boots[i] = float("nan")
            continue
        boots[i] = (mult * scores_a).sum() / total - (mult * scores_b).sum() / total
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def pair_papers_from_m3(m3: dict) -> dict:
    return {qid: (rec["A"]["paper_id"], rec["B"]["paper_id"]) for qid, rec in m3.items()}


def t5_pair_scores_by_system() -> dict:
    """Per-pair T5 both-hops (semantic, PRIMARY) score, mean over that pair's
    N-cells, one dict per system. Reuses `compute_composite_full.t5_df` and its
    `poll` (raw `both_hops`) field -- the same source and field as the paper's
    question-clustered headline (`two_hop_cluster_report.t5_both_hops_ci`) --
    so the point estimate here matches it exactly and only the interval
    changes."""
    from consort.compute_composite_full import t5_df

    df = t5_df().drop(columns=["credit"]).rename(columns={"poll": "credit"})
    return {system: df[df.system == system].groupby("question_id")["credit"].mean().to_dict()
            for system in ("small", "medium", "frontier")}


def main() -> None:
    m3 = json.loads(M3_PATH.read_text())
    pair_papers = pair_papers_from_m3(m3)
    by_system = t5_pair_scores_by_system()

    headline = {}
    print("== T5 both-hops, paper-level (graph-aware) bootstrap ==")
    for system in ("small", "medium", "frontier"):
        p, lo, hi = paper_level_bootstrap(by_system[system], pair_papers)
        headline[system] = {"acc": round(100 * p, 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}
        print(f"  {system:9s} {100*p:5.1f}%  [95% CI {100*lo:.1f}, {100*hi:.1f}]")

    diffs = {}
    print("\n== T5 paired differences, paper-level (graph-aware) bootstrap ==")
    for a, b in (("frontier", "medium"), ("frontier", "small"), ("medium", "small")):
        d, lo, hi = paper_level_paired_diff_bootstrap(by_system[a], by_system[b], pair_papers)
        diffs[f"{a}-{b}"] = {"diff": round(100 * d, 1), "lo": round(100 * lo, 1), "hi": round(100 * hi, 1)}
        print(f"  {a}-{b}: {100*d:+.1f} pp  [95% CI {100*lo:+.1f}, {100*hi:+.1f}]  "
              f"zero_included={lo <= 0 <= hi}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"headline": headline, "paired_diffs": diffs}, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
