#!/usr/bin/env python3
"""Figure 3 — LitBench performance as degradation curves over the FULL grid.

Every tier and every haystack size actually swept, for all three systems: the whole
per-N grid rendered as degradation curves rather than reduced to one point per tier.
Seven panels: T1 through the two-paper T5 (x = papers searched, log; y = accuracy),
plus T6/T7 (x = papers searched, log; y = correct-refusal rate) where the gold paper
is withheld entirely and the credited response is an explicit "I don't know".

Three trends read at once in A-E:
  * within a panel, curves slide down as more papers are added -> distractor pressure
  * panel to panel the whole family steps down                 -> task difficulty
  * the ordering 4B < 12B < Sonnet-5 holds everywhere          -> system strength
F-G read the same x-axis but the opposite of a degradation curve is the goal: refusal
falling with N means the system is getting WORSE at admitting it doesn't know.

Layout notes (from figure review): every line/marker panel (B-G; A is a categorical
bar chart, no log axis) shares ONE log x-range at EQUAL width with decade ticks and
decade gridlines, so a decade is the same physical distance in every panel and slopes
are comparable across them; an earlier version gave each panel its own range and
width, which is why the axis did not read as logarithmic. No 100% ceiling line (the
axis already ends at 100). Conditions are named by what they manipulate, not as a
"floor"-to-ceiling ladder. A blank spacer column separates E from F: A-E have a gold
answer in the haystack, F-G do not, and that regime change is the whole point of
including T6/T7 at all, so it gets a visible gap rather than reading as one long ladder.

Bands are 95% Wilson intervals computed from the TRUE number of questions/cells behind
each plotted point (system x tier x N), read from the rows that produce the point: the
two Gemma sweeps run the full ~2,252-question bank at every N whereas the Sonnet-5
per-N sweep is a 132-question sample (T1-T5) or a one-off partial collection (T6/T7),
so its bands are legitimately wider. CAVEAT, stated in the caption: this grid is
scored by the earlier PoLL panel, so its values are close to but not identical with
the three-model-panel numbers in NUMBERS_v2, which are matched-N and full-bank; the
per-domain, per-cell values are Figure S1.

Data sources: consort.candidate_grid.assemble (a5-ircot ladder, T1-T5, three main
bands) and consort/heatmaps_out/abstention_summary.json (T6/T7 refusal rate, Gemma
only; Sonnet-5's T6/T7 points are hardcoded from a separate partial collection run,
see load_absent()).
"""
import json
import os
import sys
from pathlib import Path
from math import sqrt
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import gridspec

# In addition to LITBENCH_ROOT, this script imports `consort.candidate_grid`
# from the private LitBench working tree -- that module is not bundled in this
# repo. It will not run standalone; LITBENCH_ROOT (below) is inserted onto
# sys.path so `consort` is importable, provided your LITBENCH_ROOT tree
# actually contains the `consort` package.
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
from consort.candidate_grid import (CANDIDATES, assemble, _accumulate,  # noqa: E402
                                    ALL_COLS, NCOL)

OUT = Path(__file__).resolve().parent
hk.apply_style()
# hochberg_kit sets 8pt body / 7.5pt ticks for a full-page canvas. This figure is
# downscaled into a 6.5in slot, so raise the base sizes to compensate.
plt.rcParams.update({'font.size': 12, 'axes.labelsize': 13, 'axes.labelweight': 'bold',
                     'xtick.labelsize': 11.5, 'ytick.labelsize': 11.5, 'legend.fontsize': 12})

TIERS = ["T1", "T2", "T3", "T4", "T5", "T6", "T7"]
# Name what each condition MANIPULATES. Not "floor" (T1 scores highest, so the word
# reads as floor performance and means the opposite), and not a difficulty ladder:
# T3 is empirically harder than T4, and T5 changes the task rather than the load.
# Kept SHORT: each panel is ~1.4in wide, so a label longer than about 14 characters
# collides with its neighbours once the type is large enough to survive downscaling.
TIER_LABEL = {"T1": "source only", "T2": "+ unrelated",
              "T3": "+ similar", "T4": "+ both",
              "T5": "two papers",
              "T6": "answer absent", "T7": "absent, both"}
TIER_SUB = {"T1": "no competitors", "T2": "non-epilepsy", "T3": "other epilepsy",
            "T4": "mixed", "T5": "both facts",
            "T6": "refusal correct", "T7": "refusal correct"}
# The T code gets its OWN line above the plain-language name rather than being
# prefixed to it: measured at 11pt bold in a 1.258in panel, "T6 answer absent"
# renders 1.284in wide and, centred, its left edge runs back under
# the panel letter sitting at dx=-0.06. Two lines keep both labels full-size and
# collision-free. A T code alone is not a label a clinician can read, so the
# plain-language name and its sub-label stay exactly as they were.
TITLE_Y_TIER = 1.24     # T code        (bold, largest)
TITLE_Y_LABEL = 1.15    # condition name (bold)  -- unchanged
TITLE_Y_SUB = 1.02      # sub-label      (muted) -- unchanged
PANEL_LETTER_Y = 1.31   # A-G, raised from 1.14 to sit on the T-code line.
# 1.31 is measured, not guessed: hk.panel draws va="top" at 14pt and the T code
# va="bottom" at 12.5pt, so equal y values do NOT align the two all-caps ink tops.
# At 120 dpi the T code's ink top lands at 490.5px; 1.31 puts the letter's at 489.5.
# T1-T5 have a gold answer somewhere in the haystack; T6/T7 withhold it, so the
# credited response flips from "recall the fact" to "say you can't find it".
ABSENT_TIERS = {"T6", "T7"}
# The candidate grid (ALL_COLS) only ever has T1-T5 columns -- T6/T7 don't exist in
# that accuracy matrix at all, they live in a separate refusal-rate source. load()
# must build its per-tier dict from this narrower list, not the full TIERS, or T6/T7
# end up as permanently-empty entries that crash the debug summary at the bottom.
ACCURACY_TIERS = [t for t in TIERS if t not in ABSENT_TIERS]
# NOTE: the per-point question count is now READ FROM THE DATA (see load()), never
# hardcoded — Gemma sweeps ~2,252 questions per N, the Sonnet-5 sweep 132, T5 170.

# bold, distinguishable — Sonnet-5 in the signature BrainGate blue (the headline result)
# Ordered capability ladder -> sequential ramp with separated lightness, so the
# ordering survives greyscale and colour-blind rendering; markers add redundancy.
# (band key in candidate_grid, display label = the Table 1 system, colour, marker)
SYS = [
    ("Gemma-4B",  "Gemma-4B + IRCoT"    ,         "#A8C4E0", "v"),
    ("Gemma-12B", "Gemma-12B + IRCoT"   ,        "#4E79A7", "s"),
    ("Sonnet-5",  "Sonnet-5 + native retrieval", "#1F3A5F", "o"),
]

# Systems evaluated on the REFUSAL tiers ONLY. They belong in panels F-G and in the
# legend, and they must stay out of panels A-E: DeepSeek never ran T1-T5, and a
# missing line at the bottom of an accuracy panel reads as a score near zero, which
# would invent a result the run never produced. Warm hue so it never reads as a
# fourth rung of the cool capability ladder above.
REFUSAL_ONLY_SYS = [
    ("DeepSeek-V4-Flash", "DeepSeek-V4-Flash + IRCoT (refusal tiers only)",
     "#E1943B", "D"),
]

# Panels F-G and the legend iterate ALL_SYS; panels A-E iterate SYS.
ALL_SYS = SYS + REFUSAL_ONLY_SYS

# Canvas size and panel widths, named so the layout tests can assert on them.
# The figure is PLACED in the manuscript at a fixed 6.5in width, so what governs
# legibility is the ratio between this canvas and that slot, not the canvas alone:
# a 14in canvas is squeezed to 42% and every label shrinks with it. Keep the canvas
# near its printed width and raise the font sizes instead.
FIGSIZE = (10.2, 4.2)   # printed at 6.5in wide -> ~69% scale, so 13pt renders as ~9pt
# Seven EQUAL panels. There is no spacer: an earlier version put a 0.28-wide empty
# column between E and F to mark the answer-present/answer-withheld regime change,
# and every reader took it for a layout mistake. Panels F and G name the change
# themselves ("answer absent", "absent, both"), so nothing above them is needed.
WIDTH_RATIOS = [1, 1, 1, 1, 1, 1, 1]


def _nfmt(n):
    """Axis label for a haystack size: 980 -> '980', 1000 -> '1k', 1026 -> '1,026'."""
    if n < 1000:
        return f"{n:g}"
    return f"{n // 1000}k" if n % 1000 == 0 else f"{n:,}"


def wilson(p, n, z=1.96):
    p = p / 100.0
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, (c - h) * 100), min(100.0, (c + h) * 100)


def load():
    """-> {system: {tier: [(N, accuracy, n_questions), ...]}}

    n_questions is len(agg[col]) for that band — the actual questions credited into
    the plotted aggregate — so each band reflects the evidence really behind it.
    """
    a5 = next(c for c in CANDIDATES if c.key == "a5-ircot")
    matrix, _, bands = assemble(a5)
    wanted = {s[0] for s in SYS}   # band keys
    counts = {}
    for band in a5.bands:
        if band.tier_label in wanted:
            agg, _, _ = _accumulate(band)
            counts[band.tier_label] = {col: len(v) for col, v in agg.items()}
    out = {}
    for bi, blabel in enumerate(bands):
        if blabel not in wanted:
            continue
        per = {t: [] for t in ACCURACY_TIERS}
        for c, (tt, n) in enumerate(ALL_COLS):
            if tt not in per or n == "∞":       # drop open-corpus (different regime -> Fig S1)
                continue
            v = matrix[0][bi * NCOL + c]         # ALL (aggregate) row
            if v is not None:
                nq = counts.get(blabel, {}).get((tt, n), 0)
                per[tt].append((n, v, nq))
        out[blabel] = per
    return out


ABSTENTION = Path(LITBENCH_ROOT) / "consort/heatmaps_out/abstention_summary.json"
# Band key in ALL_SYS -> model key in abstention_summary.json
_ABS_MODEL = {
    "Gemma-4B": "gemma-4b",
    "Gemma-12B": "gemma-12b",
    "DeepSeek-V4-Flash": "deepseek-0731",
}
# Sonnet-5 was sampled, not swept: T6 at N=50 only, T7 at N=1/10/50. Hardcoded from
# consort/collect_sonnet_abstention.py output (scratchpad/sonnet_abstention_partial
# .json) because that collector writes to the scratchpad, not to heatmaps_out.
# (refused, scorable) per (tier, N).
_SONNET_ABS = {
    ("T6", 50): (574, 605),
    ("T7", 1): (53, 53), ("T7", 10): (167, 167), ("T7", 50): (170, 170),
}


def load_absent():
    """-> {system: {tier: [(N, refusal_rate_pct, n_scored), ...]}}

    Points with no executed cells are DROPPED, never plotted as 0 - an unrun
    condition and a total-fabrication condition must not look identical.
    """
    out = {k: {t: [] for t in ABSENT_TIERS} for k, _, _, _ in ALL_SYS}
    summary = json.loads(ABSTENTION.read_text())
    inv = {v: k for k, v in _ABS_MODEL.items()}
    for row in summary.get("by_model_n") or []:
        band = inv.get(row["model"])
        if band is None or not row.get("scored") or row.get("refusal_rate") is None:
            continue
        out[band][row["test_type"]].append(
            (int(row["n"]), 100.0 * float(row["refusal_rate"]), int(row["scored"])))
    for (tt, n), (refused, scorable) in _SONNET_ABS.items():
        out["Sonnet-5"][tt].append((n, 100.0 * refused / scorable, scorable))
    return {k: {t: sorted(v) for t, v in d.items()} for k, d in out.items()}


def main():
    data = load()
    absent_data = load_absent()
    fig = plt.figure(figsize=FIGSIZE)
    # EQUAL panel widths and ONE shared log x-range across every tier, so a decade of
    # distractors is the same physical distance everywhere and slopes are comparable
    # across panels. (Previously each panel had its own range and width, which is why
    # the axis did not read as logarithmic.) No spacer column: the empty gap between E
    # and F read as a layout defect to every reader who saw it, and the shift from an
    # answer being present to withheld is already stated by the panel titles themselves.
    gs = gridspec.GridSpec(1, len(WIDTH_RATIOS), wspace=0.10, left=0.058, right=0.995,
                           top=0.80, bottom=0.235,
                           width_ratios=WIDTH_RATIOS)
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(WIDTH_RATIOS))]

    XLO, XHI = 0.62, 3400          # covers N=1 through the 2,000 sweep with margin
    DECADES = [1, 10, 100, 1000]

    for ti, (tier, ax) in enumerate(zip(TIERS, axes)):
        # T1-T5 read the accuracy grid; T6/T7 have no gold in the haystack at all, so
        # they read the refusal-rate grid instead (built by load_absent()).
        src = absent_data if tier in ABSENT_TIERS else data
        if tier == "T1":
            # One N, no distractors: a categorical bar reads better than three dots
            # on a log axis that spans four empty decades. Same colours, same
            # markers in the shared legend, same y-scale as B-E so heights compare.
            vals = []
            for key, disp, color, mk in SYS:
                pts = src.get(key, {}).get("T1") or []
                if pts:
                    n, v, nq = pts[0]
                    vals.append((disp, v, nq, color))
            xs_bar = range(len(vals))
            ax.bar(list(xs_bar), [v for _, v, _, _ in vals],
                   color=[c for _, _, _, c in vals], width=0.62, zorder=3)
            for i, (_, v, nq, _) in enumerate(vals):
                lo, hi = wilson(v, nq)
                ax.plot([i, i], [lo, hi], color=hk.INK, lw=1.0, zorder=4)
                ax.text(i, hi + 2.5, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=12.5, color=hk.INK)
            ax.set_xticks(list(xs_bar))
            ax.set_xticklabels([])          # identity comes from the shared legend
            ax.set_xlim(-0.7, len(vals) - 0.3)
            # Same limits as B-G, not (0, 100). Panel A owns the y tick labels for the
            # whole row, so if its scale differs from its neighbours' the printed labels
            # do not describe them: a reader lining Gemma-4B's flat T6/T7 series up
            # against panel A's "0" would misread 0% as roughly 3%. That series being
            # exactly zero is the finding, so the axis has to carry it honestly.
            ax.set_ylim(-3, 103)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.set_ylabel("Credited response (%)")
            hk.panel(ax, "A", dx=-0.30, dy=PANEL_LETTER_Y, fontsize=14.0)
            ax.text(0.5, TITLE_Y_TIER, tier, transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=12.5, fontweight="bold",
                    color=hk.INK)
            ax.text(0.5, TITLE_Y_LABEL, TIER_LABEL[tier], transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=11.0, fontweight="bold",
                    color=hk.INK)
            ax.text(0.5, TITLE_Y_SUB, TIER_SUB[tier], transform=ax.transAxes,
                    ha="center", va="bottom", fontsize=9.2, color=hk.MUTED)
            continue
        series = []
        # Refusal panels carry the refusal-only systems too; accuracy panels must not.
        for blabel, disp, color, mk in (ALL_SYS if tier in ABSENT_TIERS else SYS):
            pts = src.get(blabel, {}).get(tier) or []
            if not pts:
                continue          # unrun != zero; never draw a system with no cells here
            xs = [n for n, _, _ in pts]
            ys = [v for _, v, _ in pts]
            # Wilson band from the questions actually behind THIS point, not a global n.
            los = [wilson(v, nq)[0] for _, v, nq in pts]
            his = [wilson(v, nq)[1] for _, v, nq in pts]
            series.append({"y": ys, "lo": los, "hi": his, "color": color,
                           "marker": mk, "label": disp, "_x": xs})

        ax.set_xscale("log")
        ax.set_xlim(XLO, XHI)
        for dcd in DECADES:                                  # decade gridlines make the
            ax.axvline(dcd, color="#EEF1F4", lw=0.7, zorder=0)   # log scale self-evident
        if tier in ABSENT_TIERS:
            # Gemma was SWEPT (same 6-point N grid for every system); Sonnet-5 was
            # SAMPLED (1 point for T6, 3 for T7), so the systems in this panel do not
            # share one x grid. hk.perf_over_x plots one x against every series in the
            # list it's given, so a mismatched-length shared call would either crash
            # or misalign points -- call it once per system instead, unchanged
            # signature, each with its own "_x".
            for s in series:
                hk.perf_over_x(ax, s["_x"], [s], chance=None, annotate_last=False,
                               lw=2.2, ms=26, band_alpha=0.16)
        else:
            hk.perf_over_x(ax, series[-1]["_x"], series, chance=None,
                           annotate_last=False, lw=2.2, ms=26, band_alpha=0.16)

        # T7 sits at exactly 100% for every point (Appraise-5's ceiling), and Gemma-4B
        # sits at ~0% on T6/T7: a marker CENTERED on 0 or 100 gets clipped in half by
        # the axis frame with a flush ylim. A few points of headroom fixes it without
        # moving the 0/25/50/75/100 gridlines a reader would actually compare against.
        ax.set_ylim(-3, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(DECADES)
        ax.set_xticklabels([_nfmt(d) for d in DECADES], fontsize=12.5)
        ax.set_xticks([m * d for d in DECADES for m in range(2, 10) if XLO < m * d < XHI],
                      minor=True)
        ax.tick_params(axis="x", which="minor", length=1.5)
        ax.tick_params(axis="x", which="major", length=3)

        ax.set_ylabel("Accuracy (%)" if ti == 0 else "")
        if ti != 0:
            ax.set_yticklabels([])
        hk.panel(ax, "ABCDEFG"[ti], dx=(-0.30 if ti == 0 else -0.06),
                 dy=PANEL_LETTER_Y, fontsize=14.0)
        ax.text(0.5, TITLE_Y_TIER, tier, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=12.5, fontweight="bold", color=hk.INK)
        ax.text(0.5, TITLE_Y_LABEL, TIER_LABEL[tier], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=11.0, fontweight="bold", color=hk.INK)
        ax.text(0.5, TITLE_Y_SUB, TIER_SUB[tier], transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9.2, color=hk.MUTED)

    # x-centres from the axes actually drawn, so these never drift again when the
    # grid changes (they were stale twice already: 0.629 -> 0.591 -> now computed).
    boxes = [ax.get_position() for ax in axes]
    xlabel_c = (boxes[1].x0 + boxes[6].x1) / 2     # B-G share the log x-axis

    # No "answer present" / "answer withheld" banners across the top: the panel titles
    # already say it ("answer absent", "absent, both") and the sub-labels say "refusal
    # correct", so the banners repeated the panels underneath them and cost vertical room
    # the enlarged type needs.

    # Centred under panels B-G. Panel A is excluded: it's a bar chart with no log
    # axis, and a label spanning under it would be misleading. F/G belong in the
    # span because they carry the same log "papers searched" x-axis as B-E, just a
    # different credited quantity.
    fig.text(xlabel_c, 0.055, "Papers searched (source paper(s) + competitors; log scale)",
             ha="center", fontsize=14.0, fontweight="bold")
    # Four systems in two rows of two. The legend has to clear the x-axis label above
    # it: at the single-row y the second row rode up into "Papers searched ..." and the
    # two strings overprinted. "(refusal tiers only)" stays in the label rather than
    # moving to the caption -- it is the one thing a reader must not get wrong about
    # this figure, and a legend is read where a caption is skipped.
    handles = [hk.swatch(c, marker=mk) for _, _, c, mk in ALL_SYS]
    hk.legend_below(fig, handles, [d for _, d, _, _ in ALL_SYS], ncol=2, y=-0.155,
                    fontsize=13.0)
    hk.save(fig, str(OUT / "figure3_degradation"))
    print("wrote figure3_degradation.{pdf,png}")
    for blabel, _, _, _ in ALL_SYS:
        for tier in ACCURACY_TIERS:
            # Refusal-only systems have no accuracy grid at all; skip rather than
            # print a zero, for the same reason the panels don't draw one.
            pts = data.get(blabel, {}).get(tier) or []
            if not pts:
                continue
            rng = f"{pts[0][1]:.0f} -> {pts[-1][1]:.0f}" if len(pts) > 1 else f"{pts[0][1]:.0f}"
            print(f"  {blabel:11} {tier}: N {pts[0][0]}-{pts[-1][0]}  acc {rng}  (n={pts[0][2]})")
        for tier in ABSENT_TIERS:
            pts = absent_data[blabel][tier]
            if not pts:
                continue
            rng = f"{pts[0][1]:.1f} -> {pts[-1][1]:.1f}" if len(pts) > 1 else f"{pts[0][1]:.1f}"
            print(f"  {blabel:11} {tier}: N {pts[0][0]}-{pts[-1][0]}  refusal {rng}  (n={pts[0][2]})")


if __name__ == "__main__":
    main()
