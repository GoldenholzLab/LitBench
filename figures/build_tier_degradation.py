#!/usr/bin/env python3
"""
NOTE: This script reads litbench's closed question bank
(queries_genuine.parquet), which is not included in this repository and is
not publicly available — the benchmark stays closed to keep it an
uncontaminated test set. This script is included for methodological
transparency only; it will not run without that private data and
LITBENCH_ROOT pointed at a full LitBench working tree.

Figure 3 — accuracy by test tier for the three main systems.

Replaces the earlier per-N sweep version, which had three defects a review caught:
it plotted the SUPERSEDED judge panel (so every value contradicted the Results
paragraph printed above it), its Sonnet-5 series came from a 129-question sample
while the Gemma series came from the full bank (a population worth +2 to +8 points
to any system), and its bands were binomial Wilson intervals on data the Methods
argues are clustered within source papers.

This build fixes all three by construction:
  * ONE judge panel — the three-model panel that produced NUMBERS_v2, read from
    manifests/best3_rejudge/ (single-hop) and consort/composite_verdicts/ (T5).
  * ONE population — the complete reconciled bank, 2,188 questions per single-hop
    tier for every system; no subsample anywhere.
  * The paper's OWN interval — paper-clustered bootstrap for single-hop (resample
    the source paper) and component-clustered for two-hop (resample the connected
    component of the paper-pair graph, because hub papers put 164 of the 170 pairs
    in one component).

Layout follows the review: T1-T4 are a bracketed family that differ in distractor
TYPE (not one escalating ladder — T3 is empirically harder than T4), and T5 is
ruled off because it is a different metric (a conjunction over two papers) on a
different denominator (170 pairs). Panel letters A/B; no "100% ceiling" line;
system labels are the Table 1 configurations, not bare model names; the ordered
lightness ramp stays legible in greyscale and under protanopia.

Usage: LITBENCH_ROOT=/path/to/litbench python3 figures/build_tier_degradation.py
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import pandas as pd
from matplotlib import gridspec

# In addition to LITBENCH_ROOT, this script imports `consort.paper_clustered_ci`
# and (inside t5_stats()) `consort.two_hop_graph_ci` from the private LitBench
# working tree -- neither module is bundled in this repo. It will not run
# standalone; LITBENCH_ROOT (below) is inserted onto sys.path so `consort` is
# importable, provided your LITBENCH_ROOT tree actually contains the `consort`
# package.
LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
sys.path.insert(0, LITBENCH_ROOT)
# hochberg_kit is a personal Claude Code skill script (hochberg-figure-style) used for
# figure styling -- not part of the LitBench working tree, and not bundled in this repo.
HOCHBERG_STYLE_ROOT = os.environ.get("HOCHBERG_STYLE_ROOT")
if HOCHBERG_STYLE_ROOT is None:
    raise SystemExit(
        "Set HOCHBERG_STYLE_ROOT to the path of the hochberg-figure-style Claude Code "
        "skill's scripts directory to run this script (it uses hochberg_kit for figure "
        "styling, a personal skill not included in this repo)."
    )
sys.path.insert(0, HOCHBERG_STYLE_ROOT)
import hochberg_kit as hk  # noqa: E402
from consort.paper_clustered_ci import paper_clustered_ci  # noqa: E402

ROOT = Path(LITBENCH_ROOT)
OUT = Path(__file__).resolve().parent
hk.apply_style()

# Matched N per single-hop tier — the condition every system actually ran.
FN = {"T1": 1, "T2": 100, "T3": 200, "T4": 500}
TIERS = ["T1", "T2", "T3", "T4"]

# Ordered capability ladder -> sequential single-hue ramp with well separated
# lightness, so the ordering survives greyscale and colour-blind rendering.
# (label, colour, marker, linestyle)
SYS = [
    ("Gemma-4B + a5-ircot",      "#A8C4E0", "v", ":"),
    ("Gemma-12B + a5-ircot",     "#4E79A7", "s", "--"),
    ("Sonnet-5 + native retrieval", "#1F3A5F", "o", "-"),
]
SEM_PATH = {"Gemma-4B + a5-ircot": "gemma-4b",
            "Gemma-12B + a5-ircot": "gemma-12b",
            "Sonnet-5 + native retrieval": "frontier"}
T5_SYSTEM = {"Gemma-4B + a5-ircot": "small",
             "Gemma-12B + a5-ircot": "medium",
             "Sonnet-5 + native retrieval": "frontier"}

# What each tier manipulates. Deliberately NOT "floor"/"ladder" wording: T1 is the
# no-distractor control and T2-T4 vary distractor TYPE at a matched size.
TIER_LABEL = {"T1": "gold paper\nonly",
              "T2": "+ unrelated\npapers",
              "T3": "+ similar\npapers",
              "T4": "+ both\nkinds"}
TIER_SUB = {"T1": "no distractors", "T2": "non-epilepsy",
            "T3": "other epilepsy papers", "T4": "mixed"}


def load_single():
    """-> df(paper_id, system, test_type, credit) on the reconciled bank, one judge panel."""
    kept = set(json.loads((ROOT / "consort/reconciled_query_map.json").read_text())["single_hop"])
    q = pd.read_parquet(ROOT / ".evie_cache/a5_largeN_t1t5/queries_genuine.parquet")
    q2p = {qid: gp[0] for qid, gp in zip(q["query_id"], q["gold_paper_ids"]) if len(gp) > 0}
    rows = []
    for label, _, _, _ in SYS:
        verdicts = json.loads(
            (ROOT / f"manifests/best3_rejudge/{SEM_PATH[label]}/poll_verdicts.json").read_text())
        for r in verdicts:
            tt = r["test_type"]
            if tt in FN and r["n"] == FN[tt] and r["query_id"] in kept and r["query_id"] in q2p:
                rows.append({"paper_id": q2p[r["query_id"]], "system": label,
                             "test_type": tt, "credit": int(r["poll"])})
    return pd.DataFrame(rows)


def t5_stats():
    """-> {label: (point, lo, hi)} using the paper's OWN graph-aware bootstrap.

    Reuses consort.two_hop_graph_ci so the point estimates reproduce NUMBERS_v2
    exactly and the interval is the component-resampling one the Methods specifies,
    not a binomial approximation.
    """
    from consort.two_hop_graph_ci import (paper_level_bootstrap, pair_papers_from_m3,
                                          t5_pair_scores_by_system)
    m3 = json.loads((ROOT / "consort/out/m3_constituents.json").read_text())
    pair_papers = pair_papers_from_m3(m3)
    by_system = t5_pair_scores_by_system()
    return {label: paper_level_bootstrap(by_system[T5_SYSTEM[label]], pair_papers)
            for label, _, _, _ in SYS}


def main():
    sh = load_single()
    t5 = t5_stats()
    stats = {}
    for label, _, _, _ in SYS:
        for tt in TIERS:
            stats[(label, tt)] = paper_clustered_ci(sh, label, tt)
        stats[(label, "T5")] = t5[label]

    fig = plt.figure(figsize=(7.2, 3.3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1.35], wspace=0.30,
                           left=0.085, right=0.985, top=0.80, bottom=0.26)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    # ── Panel A: the four single-hop conditions ──────────────────────────────
    xs = range(len(TIERS))
    for label, color, mk, ls in SYS:
        ys = [stats[(label, t)][0] * 100 for t in TIERS]
        lo = [stats[(label, t)][1] * 100 for t in TIERS]
        hi = [stats[(label, t)][2] * 100 for t in TIERS]
        axA.fill_between(list(xs), lo, hi, color=color, alpha=0.18, lw=0)
        axA.plot(list(xs), ys, color=color, lw=2.2, ls=ls, marker=mk, ms=6.5,
                 mec="white", mew=0.6, label=label, zorder=3)
    axA.set_xticks(list(xs))
    axA.set_xticklabels([TIER_LABEL[t] for t in TIERS], fontsize=8)
    # The distractor definitions (TIER_SUB) live in the caption, not on the axis:
    # a second text row here collides with the two-line tick labels.
    axA.set_xlim(-0.35, len(TIERS) - 0.65)
    axA.set_ylabel("Accuracy (%)")
    hk.panel(axA, "A", dx=-0.10, dy=1.06, fontsize=10)
    axA.text(0.5, 1.10, "One gold paper  ·  2,188 questions per condition",
             transform=axA.transAxes, ha="center", va="bottom",
             fontsize=7.4, color=hk.MUTED, fontweight="bold")

    # ── Panel B: two-hop, ruled off as a different metric ────────────────────
    for label, color, mk, ls in SYS:
        if (label, "T5") not in stats:
            continue
        pt, lo, hi = [v * 100 for v in stats[(label, "T5")]]
        i = [l for l, _, _, _ in SYS].index(label)
        axB.errorbar(i, pt, yerr=[[pt - lo], [hi - pt]], color=color, marker=mk,
                     ms=7, mec="white", mew=0.6, lw=0, elinewidth=1.8, capsize=3, zorder=3)
        axB.annotate(f"{pt:.0f}", (i, pt), textcoords="offset points", xytext=(11, 0),
                     va="center", fontsize=7.5, fontweight="bold", color=color)
    axB.set_xticks(range(len(SYS)))
    axB.set_xticklabels(["4B", "12B", "Sonnet-5"], fontsize=7.4)
    axB.set_xlim(-0.6, len(SYS) - 0.4)
    axB.set_yticklabels([])
    hk.panel(axB, "B", dx=-0.16, dy=1.06, fontsize=10)
    axB.text(0.5, 1.10, "Two gold papers  ·  both facts required",
             transform=axB.transAxes, ha="center", va="bottom",
             fontsize=7.4, color=hk.MUTED, fontweight="bold")

    for ax in (axA, axB):
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 25, 50, 75, 100])
    axB.tick_params(axis="y", length=0)

    handles = [mlines.Line2D([], [], color=c, lw=2.2, ls=ls, marker=mk, ms=6.5,
                             mec="white", mew=0.6) for _, c, mk, ls in SYS]
    hk.legend_below(fig, handles, [l for l, _, _, _ in SYS], ncol=3, y=0.005, fontsize=7.8)
    hk.save(fig, str(OUT / "figure3_tier_degradation"))

    print("wrote figure3_tier_degradation.{pdf,png}")
    for label, _, _, _ in SYS:
        cells = " ".join(
            f"{t} {stats[(label, t)][0]*100:5.1f} [{stats[(label, t)][1]*100:.1f},{stats[(label, t)][2]*100:.1f}]"
            for t in TIERS + (["T5"] if (label, "T5") in stats else []))
        print(f"  {label:30} {cells}")


if __name__ == "__main__":
    main()
