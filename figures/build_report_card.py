#!/usr/bin/env python3
"""Figure 1 - the LitBench report card: three systems x two fact types x three
condition groups.

So: three stacked panels, one system each, top to bottom in capability order. Each
panel is a 2-row x 3-column heat map on ONE shared 0-100 colour scale with ONE colour
bar, so any cell can be compared with any other cell in the figure. Eighteen cells.

WHAT IS IN EACH CELL
--------------------
Every value is pooled at the level of individual scored responses - numerators and
denominators added, never an average of averages - exactly the way the previous card's
"All domains" row was built. A question that was asked at seven values of N contributes
seven responses to its cell, and the small figure under each value is the number of
distinct QUESTIONS behind it, so no reader mistakes eighteen cells for eighteen
equally-weighted measurements.

  * Columns T1-T4 and T5 come from consort.candidate_grid (the IRCoT ladder), the
    same matrix the supplement's per-domain grid, Figure S1, is drawn from. The
    bounded-corpus columns only: the two open-pool columns run a different retrieval
    engine over live PubMed, and folding them into a bounded-corpus cell would blend
    two regimes behind one number. They stay in Figure S1.
    ONE EXCEPTION, and it is the point of this build: the grid's Sonnet-5 single-paper
    source is still the 132-question pilot sweep (129 after reconciliation), while
    Table 2, Figure S3 and supplement S1.2 all report Sonnet-5 on the full reconciled
    bank. So Sonnet-5's T1-T4 columns are read straight from
    .evie_cache/sonnet_poll/full_t1t4/poll_verdicts.json - T1/T2/T3/T4 x 2,188
    questions - and all three systems' T1-T4 cells then rest on the same bank. That
    run carries one value of N per tier where the Gemma sweep carries 18; same
    questions, different sweep depth, and the footnote says so.
  * Column T6-T7 does not exist in that matrix at all. It is scored without a judge -
    the source paper is withheld, so declining is the only credited answer - and comes
    from consort/heatmaps_out/abstention_cells.csv, joined to
    consort/reconciled_query_map.json for the fact type. Credit is the strict rule
    build_abstention_results.py defines: on T7 BOTH facts must be declined.
    infra_error responses are out of the denominator; an unrun condition is never
    counted as a zero.

SONNET-5 ON T6-T7
-----------------
There is no per-question Sonnet-5 record for the withheld conditions - only the
(refused, scorable) aggregates that build_degradation_curves.py carries, collected as a
sample rather than a sweep and with NO fact-type breakdown. Printing one measured
number into two rows would invent a split that was never measured, so the two rows are
merged into a SINGLE cell carrying the pooled value, drawn tall enough that a reader
can see at a glance it is one measurement and not two. Its footnote says so.

The internal row vocabulary in consort/candidate_grid.py keys into archived run records
and cannot be renamed there; it is mapped to the reader-facing names at render time.

Output: figures/figure1_report_card.{png,pdf}
Build:  python3 figures/build_report_card.py
"""
import csv
import json
import os
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

# In addition to LITBENCH_ROOT, this script imports `consort.candidate_grid` and
# `consort.combined_heatmap` from the private LitBench working tree -- neither
# module is bundled in this repo. It will not run standalone; LITBENCH_ROOT
# (below) is inserted onto sys.path so `consort` is importable, provided your
# LITBENCH_ROOT tree actually contains the `consort` package.
LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
ROOT = Path(LITBENCH_ROOT)
sys.path.insert(0, str(ROOT))
from consort.candidate_grid import (  # noqa: E402
    ALL_COLS, CANDIDATES, DOMAIN_ROWS, _RECON_DROP, _T5_DOMAINS,
    domains_of, load_credit_by_query,
)
from consort.combined_heatmap import _cell_text_color, _perf_rgba  # noqa: E402

OUT = Path(__file__).resolve().parent
STEM = "figure1_report_card"

ABSTAIN_CELLS = ROOT / "consort" / "heatmaps_out" / "abstention_cells.csv"
QUERY_MAP = ROOT / "consort" / "reconciled_query_map.json"

# Sonnet-5's answer-present, single-paper run on the WHOLE reconciled bank. The band
# definition in consort/candidate_grid.py still points its Sonnet-5 single-paper source
# at sonnet_poll/{t1t2,t3,t4}, which is the 132-question PILOT sweep (129 after
# reconciliation); Table 2, Figure S3 and supplement S1.2 all report the full bank
# instead. This figure therefore reads the full-bank file directly rather than the band,
# so its T1-T4 cells rest on the same 2,188 questions the two Gemma systems do.
SONNET_FULL_T1T4 = (ROOT / ".evie_cache" / "sonnet_poll" / "full_t1t4"
                    / "poll_verdicts.json")
SONNET_FULL_BAND = "Sonnet-5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

INK, MUTED, RULE = "#2B2B2B", "#6B7280", "#D1D5DB"
ACCENT = "#2E5A72"

# ── reader-facing vocabulary ──────────────────────────────────────────────────
SYSTEMS = ["Gemma-4B", "Gemma-12B", "Sonnet-5"]
# the tints Figure S1 already gives these three bands, so the two figures rhyme
TINT = {"Gemma-4B": "#dbeafe", "Gemma-12B": "#ede9fe", "Sonnet-5": "#fef3c7"}

# Rows, top to bottom. The internal keys are derived from the upstream row order
# rather than written out, so this figure cannot drift from the matrix it reads.
FACT_DISPLAY = ["Stated outright", "Requiring interpretation"]
FACT_WRAP = ["Stated\noutright", "Requiring\ninterpretation"]

# Columns, left to right: the T codes, the plain-language description the paper already
# uses for each group, and the individual conditions inside the group, each in brackets
# after its cue, so a reader can tell which condition a code is without leaving the
# figure.
#
# The cues are abbreviated because the full approved names do not fit: "hidden among
# similar epilepsy papers (T3)" needs 1.81 in in a 1.60 in column and only clears it at
# 6.0 pt, below the 6.9 pt these subtitles are set in. The canonical names are therefore
# spelled out in full in the footnote key below, which is where FOOTNOTE_KEY comes from.
COLUMNS = [
    ("T1-T4", "answer present, one paper:\n"
              "alone (T1), + unrelated (T2),\n"
              "+ similar (T3), + both kinds (T4)"),
    ("T5", "answer present, two papers:\n"
           "one fact from each (T5)"),
    ("T6-T7", "answer withheld;\n"
              "declining is correct:\n"
              "one paper (T6), two papers (T7)"),
]
METRIC = "Credited response (%)"

# Model keys in abstention_cells.csv, for the two systems that ran the withheld tiers.
ABSTAIN_MODEL = {"Gemma-4B": "gemma-4b", "Gemma-12B": "gemma-12b"}
# Sonnet-5's withheld-condition collection: (refused, scorable) per (tier, N),
# the same partial collection build_degradation_curves.py:_SONNET_ABS carries, from
# consort/collect_sonnet_abstention.py. A SAMPLE, and with no fact-type breakdown -
# hence the single merged cell.
SONNET_ABSTAIN = {
    ("T6", 50): (574, 605),
    ("T7", 1): (53, 53), ("T7", 10): (167, 167), ("T7", 50): (170, 170),
}

# The seven conditions, spelled out with the names the manuscript prose and Figure 2
# use, so the artwork defines every code it draws. The column headers carry an
# abbreviated cue; this is the canonical key.
FOOTNOTE_KEY = (
    "Conditions: T1 = the answering paper alone; T2 = hidden among unrelated papers; "
    "T3 = hidden among similar epilepsy papers; T4 = hidden among both kinds; "
    "T5 = two papers, one fact from each; T6 = answer withheld, one paper; "
    "T7 = answer withheld, two papers."
)
FOOTNOTE_NOTES = (
    "Each cell pools every scored response in its group - numerators and denominators "
    "added, not an average of averages. All three systems answer the same "
    "2,188-question bank in T1-T4 but at different sweep depth: 18 values of N for each "
    "Gemma system, one per condition for Sonnet-5. T5 credits an answer only when both "
    "facts are right; T6-T7 credits an explicit refusal, and T7 requires both. "
    "n = distinct questions behind the cell. Sonnet-5's withheld-condition run is a "
    "995-response sample with no fact-type breakdown, so it is drawn as one cell across "
    "both rows. Per-condition, per-domain values are in Figure S1; the open-pool "
    "conditions run a different retrieval regime and appear there only."
)
FOOTNOTE = FOOTNOTE_KEY + " " + FOOTNOTE_NOTES
FOOTNOTE_WRAP = 145         # measured: 7 lines, widest 5.34 in inside the 5.67 in page
N_FOOTNOTE_LINES = 7

# ── geometry, in inches ───────────────────────────────────────────────────────
FIG_W = 5.83                # NEJM AI text width: rendered at final size
PAD_L, PAD_R, PAD_TOP, PAD_BOT = 0.06, 0.10, 0.10, 0.10
LAB_W = 0.86                # room for the wrapped row labels
GRID_L = PAD_L + LAB_W
GRID_W = FIG_W - GRID_L - PAD_R
COL_W = GRID_W / 3.0

H_HEAD = 0.80               # column codes + description + the per-condition cues
RH = 0.70                   # one fact-type row
H_SYS = 0.28                # system name band
PANEL_H = H_SYS + 2 * RH
GAP_PANEL = 0.20
GAP_LEGEND = 0.24
H_BAR = 0.50                # colour bar block
LINE_NOTE = 0.098           # baseline-to-baseline for a footnote line
H_NOTES = 0.10 + N_FOOTNOTE_LINES * LINE_NOTE

FIG_H = (PAD_TOP + H_HEAD + 3 * PANEL_H + 2 * GAP_PANEL
         + GAP_LEGEND + H_BAR + H_NOTES + PAD_BOT)

FS_VALUE, FS_N, FS_SYS = 15.0, 6.2, 9.5
FS_CODE, FS_SUB, FS_ROW = 9.0, 6.9, 7.6
FS_NOTE, FS_TICK = 5.7, 6.0


# ── data ──────────────────────────────────────────────────────────────────────
def fact_keys():
    """The two internal fact-type keys, in reader-facing row order, derived from the
    upstream row list and then CHECKED - never written out here.

    The 14 domain rows are seven article sections in one fact type followed by the same
    seven in the other. The three sections with no two-paper question by design must
    fall in the SECOND block; that is what identifies it as the block whose facts a
    reader has to work out.
    """
    keys = [d.rsplit("_", 1)[-1] for d in DOMAIN_ROWS]
    sections = [d.rsplit("_", 1)[0] for d in DOMAIN_ROWS]
    half = len(keys) // 2
    ok = (len(keys) == 14 and len(set(keys)) == 2
          and len(set(keys[:half])) == 1 and len(set(keys[half:])) == 1
          and set(sections[:half]) == set(sections[half:]))
    absent = [d for d in DOMAIN_ROWS if d not in _T5_DOMAINS]
    ok = ok and absent and all(d in DOMAIN_ROWS[half:] for d in absent)
    if not ok:
        raise SystemExit("upstream row order changed: the 14 rows no longer split into "
                         "two blocks of seven with the no-two-paper sections in the "
                         "second block")
    return [keys[0], keys[half]]


FACT_KEYS = fact_keys()

# The bounded-corpus columns of the candidate grid, split into the two answer-present
# bins. The open-pool columns are excluded on purpose (different retrieval regime).
BIN_COLS = {
    "T1-T4": [c for c in ALL_COLS if c[0] in ("T1", "T2", "T3", "T4") and c[1] != "∞"],
    "T5": [c for c in ALL_COLS if c[0] == "T5" and c[1] != "∞"],
}


def _blank():
    return {k: {"num": 0, "den": 0, "q": set()} for k in FACT_KEYS}


def _add(bucket, key, credit, qid):
    b = bucket[key]
    b["num"] += credit
    b["den"] += 1
    b["q"].add(qid)


def sonnet_full_t1t4_by_col():
    """Sonnet-5's full-bank single-paper run, in the same {col: [(qid, credit)]} shape
    load_credit_by_query returns, so it drops straight into the pooling loop below.

    Credit is the `poll` field, the same 3-judge panel verdict the Gemma bands use. Only
    the canonical grid columns are kept, so a column the grid does not carry cannot
    sneak into the pool.
    """
    rows = json.loads(SONNET_FULL_T1T4.read_text())
    out = {}
    for r in rows:
        if r.get("test_type") not in ("T1", "T2", "T3", "T4") or r.get("poll") is None:
            continue
        col = (r["test_type"], r["n"])
        if col in BIN_COLS["T1-T4"]:
            out.setdefault(col, []).append((r["query_id"], int(r["poll"])))
    return out


def load_answer_present():
    """-> {system: {bin: {fact key: {num, den, q}}}} for T1-T4 and T5."""
    a5 = next(c for c in CANDIDATES if c.key == "a5-ircot")
    sonnet_full = sonnet_full_t1t4_by_col()
    out = {}
    for band in a5.bands:
        if band.tier_label not in SYSTEMS:
            continue
        by_col = load_credit_by_query(band)
        if band.tier_label == SONNET_FULL_BAND:
            # replace the pilot sweep, keep this band's T5 columns untouched
            by_col = {c: v for c, v in by_col.items() if c not in BIN_COLS["T1-T4"]}
            by_col.update(sonnet_full)
        per_bin = {b: _blank() for b in BIN_COLS}
        for bname, cols in BIN_COLS.items():
            for col in cols:
                for qid, credit in by_col.get(col, []):
                    if qid in _RECON_DROP:      # out of the reconciled gold set
                        continue
                    # A two-paper question spans two article sections and so can belong
                    # to both fact types; it is counted once in each, never twice in one.
                    for key in {d.rsplit("_", 1)[-1] for d in domains_of(qid)}:
                        _add(per_bin[bname], key, credit, qid)
        out[band.tier_label] = per_bin
    check_sweep_depth(out, sonnet_full)
    return out


def check_sweep_depth(present, sonnet_full):
    """The footnote states two counts in words - 18 values of N per Gemma system and one
    per condition for Sonnet-5 - and a reader has no way to check them. Assert them
    against what was actually pooled, so the sentence cannot outlive the data it
    describes.
    """
    n_gemma, n_sonnet = len(BIN_COLS["T1-T4"]), len(sonnet_full)
    if (n_gemma, n_sonnet) != (18, 4):
        raise SystemExit(
            f"footnote says 18 values of N per Gemma system and one per condition (4) "
            f"for Sonnet-5; the data now gives {n_gemma} and {n_sonnet}")
    banks = {s: sum(len(present[s]["T1-T4"][k]["q"]) for k in FACT_KEYS)
             for s in present}
    if banks.get(SONNET_FULL_BAND) != 2188:
        raise SystemExit(
            f"the footnote says Sonnet-5 answers the same 2,188-question bank; its "
            f"T1-T4 cells now rest on {banks.get(SONNET_FULL_BAND)} questions")


def load_answer_withheld():
    """-> {system: {fact key: {num, den, q}}} for T6-T7, Gemma bands only.

    Credit is `both_abstained` (T7 needs both facts declined); infra_error responses
    are dropped from the denominator, per build_abstention_results.py.
    """
    single = json.loads(QUERY_MAP.read_text())["single_hop"]
    pair = json.loads((ROOT / ".evie_cache" / "t5_query_domains.json").read_text())
    inv = {v: k for k, v in ABSTAIN_MODEL.items()}
    out = {s: _blank() for s in ABSTAIN_MODEL}
    with ABSTAIN_CELLS.open() as fh:
        for r in csv.DictReader(fh):
            band = inv.get(r["model"])
            if band is None or r["outcome"] == "infra_error":
                continue
            qid = r["query_id"]
            if r["test_type"] == "T7":
                ab = pair.get(qid, {})
                doms = [d for d in (ab.get("A"), ab.get("B")) if d]
            else:
                doms = [single[qid]] if qid in single else []
            for key in {d.rsplit("_", 1)[-1] for d in doms}:
                _add(out[band], key, int(r["both_abstained"]), qid)
    return out


def sonnet_withheld():
    """(refused, scorable) pooled over Sonnet-5's whole partial collection."""
    num = sum(a for a, _ in SONNET_ABSTAIN.values())
    den = sum(b for _, b in SONNET_ABSTAIN.values())
    return num, den


def build_cells():
    """-> {(system, column code): cell}, where a cell is either
    {"split": [(num, den, nq), (num, den, nq)]}  - one value per row, or
    {"merged": (num, den)}                       - one value across both rows.
    """
    present = load_answer_present()
    withheld = load_answer_withheld()
    cells = {}
    for sysname in SYSTEMS:
        for bname in ("T1-T4", "T5"):
            b = present[sysname][bname]
            cells[(sysname, bname)] = {"split": [
                (b[k]["num"], b[k]["den"], len(b[k]["q"])) for k in FACT_KEYS]}
        if sysname in withheld:
            b = withheld[sysname]
            cells[(sysname, "T6-T7")] = {"split": [
                (b[k]["num"], b[k]["den"], len(b[k]["q"])) for k in FACT_KEYS]}
        else:
            cells[(sysname, "T6-T7")] = {"merged": sonnet_withheld()}
    return cells


def pct(num, den):
    return 100.0 * num / den


def fmt(v):
    """Integer percent, the convention Figure S1 uses. A value that is not zero but
    rounds to zero prints as "<1", so it is never confused with a measured zero."""
    if 0 < v < 0.5:
        return "<1"
    return f"{round(v):g}"


# ── fit guard ─────────────────────────────────────────────────────────────────
# No label may be wider than the space allotted to it, and nothing may run off the
# page. Measured from the rendered artists, not estimated from character counts.
CHECKS = []


def track(art, max_w, name):
    CHECKS.append((art, max_w, name))
    return art


def check_fit(fig):
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    fw, fh = fig.get_size_inches()
    bad = []
    for art, mw, name in CHECKS:
        bb = art.get_window_extent(renderer=rend)
        x0, y0 = bb.x0 / fig.dpi, bb.y0 / fig.dpi
        w, h = bb.width / fig.dpi, bb.height / fig.dpi
        if x0 < -1e-3 or y0 < -1e-3 or x0 + w > fw + 1e-3 or y0 + h > fh + 1e-3:
            bad.append(f"  {name}: runs off the page ({x0:.3f}, {y0:.3f}) to "
                       f"({x0 + w:.3f}, {y0 + h:.3f}) outside 0,0-{fw:.2f},{fh:.2f}")
        if mw is not None and w > mw + 1e-3:
            bad.append(f"  {name}: {w:.3f} in wide > {mw:.3f} in allowed")
    if bad:
        raise SystemExit("text overflows its box:\n" + "\n".join(bad))


# ── draw ──────────────────────────────────────────────────────────────────────
def _cell(ax, x, y, w, h, v, value_fs=FS_VALUE):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=_perf_rgba(v),
                           edgecolor="white", lw=1.2, zorder=2))
    return _cell_text_color(v, None)


def main():
    cells = build_cells()

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FIG_W)
    ax.set_ylim(0, FIG_H)
    ax.axis("off")

    # --- column heads ---------------------------------------------------------
    top = FIG_H - PAD_TOP
    for ci, (code, sub) in enumerate(COLUMNS):
        cx = GRID_L + (ci + 0.5) * COL_W
        track(ax.text(cx, top - 0.02, code, fontsize=FS_CODE, fontweight="bold",
                      color=INK, ha="center", va="top"), COL_W, f"code {code}")
        track(ax.text(cx, top - 0.21, sub, fontsize=FS_SUB, color=MUTED,
                      ha="center", va="top", linespacing=1.30),
              COL_W, f"subtitle {code}")
    head_rule = top - H_HEAD + 0.10
    ax.plot([GRID_L, GRID_L + GRID_W], [head_rule, head_rule], color=ACCENT, lw=0.9)

    # --- three panels ---------------------------------------------------------
    panel_top = top - H_HEAD
    for pi, sysname in enumerate(SYSTEMS):
        py = panel_top - pi * (PANEL_H + GAP_PANEL)

        ax.add_patch(Rectangle((GRID_L, py - H_SYS + 0.03), GRID_W, H_SYS - 0.06,
                               facecolor=TINT[sysname], edgecolor="none", zorder=1))
        ax.text(GRID_L + 0.10, py - H_SYS / 2.0, sysname, fontsize=FS_SYS,
                fontweight="bold", color=INK, ha="left", va="center", zorder=3)

        rows_top = py - H_SYS
        for ri, label in enumerate(FACT_WRAP):
            track(ax.text(GRID_L - 0.10, rows_top - (ri + 0.5) * RH, label,
                          fontsize=FS_ROW, color=INK, ha="right", va="center",
                          linespacing=1.25),
                  LAB_W - 0.04, f"row label {ri} {sysname}")

        for ci, (code, _) in enumerate(COLUMNS):
            x = GRID_L + ci * COL_W
            cell = cells[(sysname, code)]
            if "merged" in cell:
                num, den = cell["merged"]
                v = pct(num, den)
                tc = _cell(ax, x, rows_top - 2 * RH, COL_W, 2 * RH, v)
                cy = rows_top - RH
                # The merged cell is twice as tall, so its text block is centred on the
                # cell midpoint rather than sitting where a single row's would.
                ax.text(x + COL_W / 2.0, cy + 0.14, fmt(v), fontsize=FS_VALUE,
                        color=tc, ha="center", va="center", zorder=3)
                ax.text(x + COL_W / 2.0, cy - 0.06, f"{den:,} responses",
                        fontsize=FS_N, color=tc, ha="center", va="center", zorder=3)
                ax.text(x + COL_W / 2.0, cy - 0.17, "both fact types pooled",
                        fontsize=FS_N, color=tc, ha="center", va="center", zorder=3,
                        style="italic")
                continue
            for ri, (num, den, nq) in enumerate(cell["split"]):
                v = pct(num, den)
                y = rows_top - (ri + 1) * RH
                tc = _cell(ax, x, y, COL_W, RH, v)
                ax.text(x + COL_W / 2.0, y + RH / 2.0 + 0.07, fmt(v),
                        fontsize=FS_VALUE, color=tc, ha="center", va="center", zorder=3)
                ax.text(x + COL_W / 2.0, y + RH / 2.0 - 0.15, f"n = {nq:,}",
                        fontsize=FS_N, color=tc, ha="center", va="center", zorder=3)

    # --- one shared colour bar ------------------------------------------------
    grid_bot = panel_top - 3 * PANEL_H - 2 * GAP_PANEL
    by = grid_bot - GAP_LEGEND - 0.20
    bar_w, bar_h = 2.35, 0.17
    for i in range(96):
        ax.add_patch(Rectangle((GRID_L + i * bar_w / 96.0, by), bar_w / 96.0 + 0.004,
                               bar_h, facecolor=_perf_rgba(i * 100.0 / 95.0),
                               edgecolor="none"))
    ax.add_patch(Rectangle((GRID_L, by), bar_w, bar_h, facecolor="none",
                           edgecolor=RULE, lw=0.6))
    for lab, frac, ha in (("0", 0.0, "left"), ("50", 0.5, "center"), ("100", 1.0, "right")):
        ax.text(GRID_L + frac * bar_w, by - 0.03, lab, fontsize=FS_TICK, color=MUTED,
                ha=ha, va="top")
    ax.text(GRID_L, by + bar_h + 0.04, METRIC, fontsize=FS_SUB + 0.6,
            fontweight="bold", color=INK, ha="left", va="bottom")
    ax.text(GRID_L + bar_w + 0.16, by + bar_h / 2.0,
            "one scale, all three systems", fontsize=FS_NOTE + 0.4, color=MUTED,
            ha="left", va="center")

    # --- footnote -------------------------------------------------------------
    lines = textwrap.wrap(FOOTNOTE, width=FOOTNOTE_WRAP)
    if len(lines) != N_FOOTNOTE_LINES:
        raise SystemExit(f"footnote wraps to {len(lines)} lines, not {N_FOOTNOTE_LINES}; "
                         f"the figure height reserves room for exactly that many")
    ny = by - 0.20
    ax.plot([PAD_L, FIG_W - PAD_R], [ny, ny], color=RULE, lw=0.5)
    for i, line in enumerate(lines):
        track(ax.text(PAD_L, ny - 0.08 - i * LINE_NOTE, line, fontsize=FS_NOTE,
                      color=MUTED, ha="left", va="top"),
              FIG_W - PAD_L - PAD_R, f"footnote line {i + 1}")

    check_fit(fig)
    fig.savefig(OUT / f"{STEM}.png", dpi=600)
    fig.savefig(OUT / f"{STEM}.pdf")
    plt.close(fig)

    print(f"wrote {STEM}.{{png,pdf}}  {FIG_W:.2f} x {FIG_H:.2f} in")
    for sysname in SYSTEMS:
        for code, _ in COLUMNS:
            cell = cells[(sysname, code)]
            if "merged" in cell:
                num, den = cell["merged"]
                print(f"  {sysname:10} {code:6} MERGED         "
                      f"{num}/{den} = {pct(num, den):.2f}  -> {fmt(pct(num, den))}")
                continue
            for ri, (num, den, nq) in enumerate(cell["split"]):
                print(f"  {sysname:10} {code:6} {FACT_DISPLAY[ri]:24} "
                      f"{num}/{den} = {pct(num, den):.2f}  -> {fmt(pct(num, den))}"
                      f"  (q={nq})")


if __name__ == "__main__":
    main()
