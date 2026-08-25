#!/usr/bin/env python3
"""LitBench canonical performance heatmap for the paper (Figure 2).

Domain (rows) x test-type T1-T5 (cols) x system band, PoLL accuracy, RdYlGn.
Main figure = Small (Gemma-4B) / Medium (Gemma-12B) / Frontier (Sonnet-5).
Appendix figure adds Strong (Gemma-31B), whose run is incomplete: single-hop is a
10-per-domain sample (all 14 domains populated); T5 covers 11 of 14 domains (methods,
table, and figure domains were not run and render gray), plus the ALL/aggregate cell.

N is aggregated within each test type (mean credit over the N sweep and queries), so
T1-T4 cells are single-hop PoLL same-fact accuracy and T5 is two-hop both-hops accuracy.
The ALL(aggregate) row is printed and checked against NUMBERS.md (4B T1 74.7 ...).

Data (per-query verdicts, reconciled 2026-07-15):
  4B/12B single-hop: .evie_cache/a5_largeN_t1t5/single_judge_out/<tier>/poll_verdicts.json
  4B/12B T5:         .evie_cache/a5_largeN_t1t5/t5_judge_out/<tier>/poll_verdicts.json
  Sonnet single-hop: .evie_cache/sonnet_t{1t2,3,4}_results_FULL.json (verdict.all_matched, domain)
  Sonnet T5:         .evie_cache/sonnet_t5_results_FULL.json (verdict.all_matched)
  31B single-hop:    .evie_cache/a5_t1t4/judge_out/gemma-31b/poll_verdicts.json (OLD sample)
  31B T5:            .evie_cache/a5_31b_largeN/t5_judge_out/gemma-31b/poll_verdicts.json
  T5 domain map:     .evie_cache/t5_query_domains.json (credit both-hops to both fact-domains)
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
from pathlib import Path

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size': 9, 'pdf.fonttype': 42, 'ps.fonttype': 42,
})

LITBENCH_ROOT = os.environ.get("LITBENCH_ROOT")
if LITBENCH_ROOT is None:
    raise SystemExit(
        "Set LITBENCH_ROOT to the path of your local LitBench working tree "
        "to run this script (it reads private benchmark data not included "
        "in this repo)."
    )
ROOT = Path(LITBENCH_ROOT)
LN = ROOT / '.evie_cache/a5_largeN_t1t5'
OUT = Path(__file__).resolve().parent

SECTIONS = ['abstract', 'introduction', 'methods', 'results', 'discussion', 'table', 'figure']
DOMAIN_ROWS = [f'{s}_thinking' for s in SECTIONS] + [f'{s}_recall' for s in SECTIONS]
ROW_KEYS = ['ALL'] + DOMAIN_ROWS
# Short row labels (the fact-type split is shown as a group band on the left,
# so the per-row label only needs the section name); ALL row keeps its full label.
ROW_LABELS = ['All domains'] + [s.title() for s in SECTIONS] * 2
TESTS = ['T1', 'T2', 'T3', 'T4', 'T5']
# Row-group structure for the clinical layout: (group label, [display row indices]).
# Row 0 = ALL; rows 1-7 = the seven Thinking domains; rows 8-14 = the seven Recall domains.
# The two internal fact-type keys ('thinking' / 'recall') key into the archived run
# records and cannot be renamed there; the band labels a reader sees are the two names
# the manuscript uses throughout.
ROW_GROUPS = [('', [0]), ('Requiring interpretation', list(range(1, 8))),
              ('Stated outright', list(range(8, 15)))]

# Reconciled single-hop domains (post-second-review: 2,188 kept, thinking->recall demotions
# applied); DROPPED = the 64 removed single-hop questions. T5 domains (QDOM) are frozen.
_recon = json.loads((ROOT / 'consort/reconciled_query_map.json').read_text())
QID2DOM = dict(_recon['single_hop'])
DROPPED = set(_recon['dropped'])
QDOM = json.loads((ROOT / '.evie_cache/t5_query_domains.json').read_text())


def _load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def _accumulate(rows, credit_key, dom_of, acc, exclude=None):
    """acc[(row_key, test)] -> list of 0/1 credits. dom_of(r) -> list of domain keys.
    exclude: query_ids to skip entirely (the reconciliation-dropped single-hop questions)."""
    for r in rows or []:
        tt = r.get('test_type')
        if tt not in TESTS:
            continue
        if exclude and r.get('query_id') in exclude:
            continue
        c = int(r[credit_key]) if not isinstance(r.get(credit_key), dict) else int(bool(r[credit_key]))
        acc[('ALL', tt)].append(c)
        for d in dom_of(r):
            if d in DOMAIN_ROWS:
                acc[(d, tt)].append(c)


def _sonnet_rows(paths):
    out = []
    for p in paths:
        for r in _load(p) or []:
            v = r.get('verdict') or {}
            out.append({'query_id': r.get('query_id'), 'test_type': r.get('test_type'),
                        'n': r.get('n'), 'domain': r.get('domain'),
                        'credit': 1 if v.get('all_matched') else 0})
    return out


def _t5_domains(r):
    dd = QDOM.get(r.get('query_id')) or {}
    return [d for d in (dd.get('A'), dd.get('B')) if d]


def band_matrix(kind):
    """kind in {gemma-4b, gemma-12b, gemma-31b, sonnet}. Returns NROW x 5 float/nan."""
    acc = defaultdict(list)
    if kind in ('gemma-4b', 'gemma-12b'):
        _accumulate(_load(LN / 'single_judge_out' / kind / 'poll_verdicts.json'),
                    'poll', lambda r: [QID2DOM.get(r['query_id'])], acc, exclude=DROPPED)
        _accumulate(_load(LN / 't5_judge_out' / kind / 'poll_verdicts.json'),
                    'both_hops', _t5_domains, acc)
    elif kind == 'gemma-31b':
        v = _load(ROOT / '.evie_cache/a5_largeN_t1t5/single_judge_out/gemma-31b/poll_verdicts.json') \
            or _load(ROOT / '.evie_cache/a5_t1t4/judge_out/gemma-31b/poll_verdicts.json')
        _accumulate(v, 'poll', lambda r: [QID2DOM.get(r['query_id'])], acc, exclude=DROPPED)
        # T5 lives in a separate 31b-only sweep directory, not under LN (that path never
        # existed — was silently returning no data, so the ALL/T5 cell rendered gray).
        _accumulate(_load(ROOT / '.evie_cache/a5_31b_largeN/t5_judge_out/gemma-31b/poll_verdicts.json'),
                    'both_hops', _t5_domains, acc)
    elif kind == 'sonnet':
        # single-hop = FULL-benchmark native run (2,059 q x T1-T4), reconciled 3-LLM PoLL,
        # judged 2026-07-18. Same schema as the gemma verdicts (poll). Supersedes the
        # earlier sonnet_poll/{t1t2,t3,t4} 132-question sample.
        SP = ROOT / '.evie_cache/sonnet_poll'
        _accumulate(_load(SP / 'full_t1t4' / 'poll_verdicts.json'),
                    'poll', lambda r: [QID2DOM.get(r['query_id'])], acc, exclude=DROPPED)
        # T5 two-hop FROZEN (separate run, unchanged).
        _accumulate(_load(SP / 't5' / 'poll_verdicts.json'), 'both_hops', _t5_domains, acc)
    m = np.full((len(ROW_KEYS), len(TESTS)), np.nan)
    for i, rk in enumerate(ROW_KEYS):
        for j, tt in enumerate(TESTS):
            vals = acc.get((rk, tt))
            if vals:
                m[i, j] = 100.0 * np.mean(vals)
    return m


# Colorblind-safe sequential colormap (viridis), gray for NaN. Dark = low, bright = high.
CMAP = plt.get_cmap('viridis').copy()
CMAP.set_bad('#D9DCDE')


GROUP_COLOR = '#1F2D5C'


def render(bands, band_labels, out_stem, figw=None):
    """Clinical domain x tier x system heatmap. Rows grouped ALL | requiring
    interpretation | stated outright
    with separators and left-margin band labels; per-tier cells carry the panel accuracy,
    gray = a tier that does not exist for that domain (structural not-applicable)."""
    nb = len(bands)
    big = np.hstack(bands)
    nrow, ncol = big.shape
    figw = figw or (2.6 + 0.60 * ncol)
    fig, ax = plt.subplots(figsize=(figw, 1.5 + 0.46 * nrow))
    im = ax.imshow(big, cmap=CMAP, vmin=0, vmax=100, aspect='auto')
    # thin white grid between every cell for legibility
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.0)
    ax.tick_params(which='minor', length=0)
    # cell values: white on dark (low) viridis cells, near-black on bright (high) cells
    for i in range(nrow):
        for j in range(ncol):
            v = big[i, j]
            if not np.isnan(v):
                tcol = 'white' if v < 55 else '#1A1A1A'
                wt = 'bold' if i == 0 else 'normal'   # emphasize the All-domains row
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                        fontsize=9.5, color=tcol, fontweight=wt)
    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.016, pad=0.010)
    cbar.set_label('Panel accuracy (%)', fontsize=10, color='#2B2B2B')
    cbar.ax.tick_params(labelsize=9)
    # row labels (short section names; group carried by the left band)
    ax.set_yticks(range(nrow))
    ax.set_yticklabels(ROW_LABELS, fontsize=9.5, color='#2B2B2B')
    ax.get_yticklabels()[0].set_fontweight('bold')  # All domains
    ax.tick_params(length=0)
    # column labels (T1..T5 per band) + bold panel headers + panel dividers
    ax.set_xticks(range(ncol))
    ax.set_xticklabels(TESTS * nb, fontsize=9, color='#2B2B2B')
    for b in range(nb):
        cx = b * len(TESTS) + (len(TESTS) - 1) / 2
        ax.text(cx, -1.5, band_labels[b], ha='center', va='center', fontsize=11.5,
                fontweight='bold', color=GROUP_COLOR)
        if b > 0:
            ax.axvline(b * len(TESTS) - 0.5, color='#2B2B2B', lw=2.0, zorder=5)
    # row-group separators (after All row 0, after Thinking block row 7) + left band labels
    for sep in (0.5, 7.5):
        ax.axhline(sep, color='#2B2B2B', lw=1.4, zorder=5)
    xband = -0.062 * ncol - 2.3   # left of the row labels, in data coords (tuned at render)
    for glabel, idxs in ROW_GROUPS:
        if not glabel:
            continue
        yc = sum(idxs) / len(idxs)
        ax.text(xband, yc, glabel, ha='center', va='center', rotation=90,
                fontsize=11, fontweight='bold', color=GROUP_COLOR, clip_on=False)
    # structural-N/A note
    ax.text(0, nrow + 0.35, 'Gray cell = tier not applicable to that domain',
            ha='left', va='top', fontsize=8.5, style='italic', color='#6B7280')
    ax.set_ylim(nrow - 0.5, -1.9)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(f'{out_stem}.pdf', dpi=600, bbox_inches='tight', pad_inches=0.14)
    fig.savefig(f'{out_stem}.png', dpi=600, bbox_inches='tight', pad_inches=0.14)
    plt.close(fig)


if __name__ == '__main__':
    m4 = band_matrix('gemma-4b')
    m12 = band_matrix('gemma-12b')
    mS = band_matrix('sonnet')
    m31 = band_matrix('gemma-31b')

    # reconcile ALL row against NUMBERS.md
    # 31B expect ~75/72/73/77/20 (DROPPED-excluded, matching supplement.md Appendix B);
    # NUMBERS.md's 31b row (76/72/73/78) predates that exclusion and is stale, not a target.
    print('ALL-row check (NUMBERS.md: 4B 74.7/65.2/59.5/61.2/10.3 | 12B 79.2/69.6/65.1/67.1/21.0 | Sonnet 90.9/92.1/92.1/92.1/46.4)')
    for name, m in [('4B', m4), ('12B', m12), ('Sonnet', mS), ('31B', m31)]:
        print(f'  {name:7s} ' + ' '.join(f'{"--" if np.isnan(x) else round(x)}' for x in m[0]))

    render([m4, m12, mS],
           ['Gemma-4B + IRCoT', 'Gemma-12B + IRCoT', 'Sonnet-5 + native retrieval'],
           str(OUT / 'figure2_heatmap'))
    print('wrote figure2_heatmap.{pdf,png}')

    render([m4, m12, mS, m31],
           ['Gemma-4B + IRCoT', 'Gemma-12B + IRCoT', 'Sonnet-5 + native retrieval',
            'Gemma-31B + IRCoT (partial)'],
           str(OUT / 'figureA_heatmap_with31b'))
    print('wrote figureA_heatmap_with31b.{pdf,png}')
