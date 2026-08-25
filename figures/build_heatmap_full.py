#!/usr/bin/env python3
"""LitBench FULL per-N progressive-hardness heatmap for the paper (Figure 2).

Uses the paper's OWN colorway -- consort/combined_heatmap.py::render_combined (RdYlGn
ramp green-high/red-low + gray = not-run + colored band headers) -- over the FULL per-N
distractor sweep, so each row visibly fades green->red as N grows within and across tiers.
Three systems: small Gemma-4B / medium Gemma-12B / frontier Sonnet-5. Reconciled data
(post-second-review single-hop; T5 frozen). T1-T4 = single-hop same-fact %, T5 = two-hop both-hops %.

Sonnet-5's native single-hop run covers only the representative N per tier
(T1.1 / T2.100 / T3.200 / T4.500) -> its other single-hop N columns are gray (not run);
T5 is the full N-sweep for all three systems.

PREVIEW: current 3-LLM PoLL numbers; re-run after the three-model judge panel swap for the final.
Output: figures/figure2_heatmap_full.{pdf,png}
"""
import sys
import json
import os
from collections import defaultdict
from pathlib import Path
from statistics import mean

# In addition to LITBENCH_ROOT, this script imports `consort.combined_heatmap`
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
ROOT = Path(LITBENCH_ROOT)
sys.path.insert(0, str(ROOT))
from consort.combined_heatmap import render_combined  # noqa: E402

LN = ROOT / '.evie_cache/a5_largeN_t1t5'
SP = ROOT / '.evie_cache/sonnet_poll'
OUT = Path(__file__).resolve().parent / 'figure2_heatmap_full'

# columns: the full large-N per-N sweep (progressive hardness)
COLS = [
    ("T1", 1),
    ("T2", 10), ("T2", 50), ("T2", 100), ("T2", 200), ("T2", 500), ("T2", 980),
    ("T3", 10), ("T3", 50), ("T3", 100), ("T3", 200), ("T3", 500), ("T3", 1026),
    ("T4", 75), ("T4", 100), ("T4", 500), ("T4", 1000), ("T4", 2000),
    ("T5", 10), ("T5", 50), ("T5", 100), ("T5", 200), ("T5", 500), ("T5", 1000), ("T5", 2000),
]
COL_LABELS = [f"{tt}·{n}" for tt, n in COLS]
COL_IDX = {c: i for i, c in enumerate(COLS)}
NCOL = len(COLS)

SECTIONS = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'table', 'figure']
DOMAIN_ROWS = [f'{s}_thinking' for s in SECTIONS] + [f'{s}_recall' for s in SECTIONS]
ROW_LABELS = ['ALL (aggregate)'] + [d.replace('_', ' ').title() for d in DOMAIN_ROWS]
ROW_IDX = {d: i + 1 for i, d in enumerate(DOMAIN_ROWS)}  # +1 for ALL at row 0
NROW = len(ROW_LABELS)

# reconciliation (post-second-review): keep-set domains + dropped single-hop question ids
_recon = json.loads((ROOT / 'consort/reconciled_query_map.json').read_text())
QID2DOM = dict(_recon['single_hop'])
DROPPED = set(_recon['dropped'])
QDOM = json.loads((ROOT / '.evie_cache/t5_query_domains.json').read_text())


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def pct(vals):
    return round(mean(vals) * 100) if vals else None


def blank():
    return [[None] * NCOL for _ in range(NROW)]


def fill_single(mat, verdicts):
    """single-hop poll: ALL row + reconciled domain rows, per (tt,n); skip reconciliation-dropped."""
    acc = defaultdict(list)
    for r in verdicts or []:
        if r.get('query_id') in DROPPED:
            continue
        ci = COL_IDX.get((r.get('test_type'), r.get('n')))
        if ci is None:
            continue
        c = int(r['poll'])
        acc[(0, ci)].append(c)
        ri = ROW_IDX.get(QID2DOM.get(r['query_id']))
        if ri is not None:
            acc[(ri, ci)].append(c)
    for (ri, ci), vals in acc.items():
        mat[ri][ci] = pct(vals)


def fill_t5(mat, verdicts):
    """T5 both-hops (frozen): ALL row + both fact-domain rows, per (tt,n)."""
    acc = defaultdict(list)
    for r in verdicts or []:
        if r.get('test_type') != 'T5':
            continue
        ci = COL_IDX.get(('T5', r.get('n')))
        if ci is None:
            continue
        c = int(r['both_hops'])
        acc[(0, ci)].append(c)
        dd = QDOM.get(r.get('query_id')) or {}
        for d in (dd.get('A'), dd.get('B')):
            ri = ROW_IDX.get(d)
            if ri is not None:
                acc[(ri, ci)].append(c)
    for (ri, ci), vals in acc.items():
        mat[ri][ci] = pct(vals)


def band(kind):
    m = blank()
    if kind in ('gemma-4b', 'gemma-12b'):
        fill_single(m, _load(LN / 'single_judge_out' / kind / 'poll_verdicts.json'))
        fill_t5(m, _load(LN / 't5_judge_out' / kind / 'poll_verdicts.json'))
    elif kind == 'sonnet':
        fill_single(m, _load(SP / 'full_t1t4' / 'poll_verdicts.json'))
        fill_t5(m, _load(SP / 't5' / 'poll_verdicts.json'))
    return m


CAND_LABELS = ['small · Gemma-4B', 'medium · Gemma-12B', 'frontier · Sonnet-5']
b4, b12, bS = band('gemma-4b'), band('gemma-12b'), band('sonnet')

# sanity: ALL-row per band
print("=== ALL (aggregate) row, per band, per (tt,n) ===")
for lbl, m in [("4B", b4), ("12B", b12), ("Sonnet", bS)]:
    cells = " ".join(f"{COL_LABELS[c]}={'--' if m[0][c] is None else m[0][c]}" for c in range(NCOL))
    print(f"  {lbl:7s} {cells}")

# candidate-major big matrix: NROW x (3 * NCOL)
big = [b4[r] + b12[r] + bS[r] for r in range(NROW)]

caption = (
    "LitBench per-N accuracy across the full distractor sweep, 3-LLM PoLL (PREVIEW, current "
    "judge). T1-T4 = single-hop same-fact %; T5 = two-hop both-hops %. Each row fades "
    "green (high) -> red (low) as N grows within and across tiers -- the progressive-hardness "
    "signal. GRAY = not run (no data), NOT zero. Sonnet-5 single-hop was run at the "
    "representative N per tier only (T1.1 / T2.100 / T3.200 / T4.500), so its other single-hop "
    "N columns are gray; T5 is the full sweep for all three. Numbers update with the three-model judge panel."
)

render_combined([("", big)], ROW_LABELS, CAND_LABELS, COL_LABELS, str(OUT),
                title=None, caption=caption)
print(f"\nwrote {OUT}.png / .pdf")
