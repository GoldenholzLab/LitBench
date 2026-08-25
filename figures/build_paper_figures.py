#!/usr/bin/env python3
"""Build LitBench main-text data figures (Figure 3, Figure S4) in NEJM AI /
Lancet Digital Health house style (scientific-figures skill conventions).

Reframed to end-to-end SYSTEM configurations (not model tiers). No in-figure
titles (Lancet rule); titles live in the manuscript figure legends.

Data provenance: litbench/paper/NUMBERS_v2.md, three-model judge panel (DeepSeek-
V4-Flash + Qwen3.6-35B-A3B + Llama-4-Scout), reconciled 2026-07-22. Single-hop
values are matched N=2188 (semantic primary metric); T5 is all-N pooled,
both-hops. IRCoT harness for the Gemma systems; the Sonnet-5-based system
used Claude Code's own native retrieval.

Figure 3 (main text) is a tier-level accuracy line chart, T1 through T5: the
three-model judge panel's re-judge exists only at the representative N per tier (T1.1/T2.100/
T3.200/T4.500), not the full per-N sweep, so there is no per-N figure here.
Figure S4 (supplement) is the single-hop vs two-hop item-set accuracy
comparison — descriptive, since the two bars are drawn from different item
sets rather than a paired before/after measurement.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# --- Journal-standard configuration ---
plt.rcParams.update({
    'font.family':        'sans-serif',
    'font.sans-serif':    ['Helvetica', 'Helvetica Neue', 'Arial', 'DejaVu Sans'],
    'font.size':          10,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.spines.left':   True,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

# --- Lancet-inspired palette ---
LANCET_BLUE   = '#00468B'
LANCET_SALMON = '#FDAF91'
LABEL_COLOR   = '#2B2B2B'
MUTED_COLOR   = '#6B7280'
GRID_COLOR    = '#F3F4F6'
SPINE_COLOR   = '#9CA3AF'

OUT = Path(__file__).resolve().parent
OUT.mkdir(parents=True, exist_ok=True)

# --- Canonical data — three main systems: Gemma-4B / Gemma-12B / Sonnet-5 — three-model
# judge panel, matched single-hop N=2188 (T1=1/T2=100/T3=200/T4=500), semantic
# primary metric; T5 all-N pooled, both-hops. Provenance: NUMBERS_v2.md (single-hop updated 2026-07-26).
SYSTEMS = ['Gemma-4B\nIRCoT', 'Gemma-12B\nIRCoT', 'Sonnet-5\nnative retrieval']
T = {
    'T1': [71.9, 75.5, 90.0],
    'T2': [63.7, 68.2, 91.1],
    'T3': [55.9, 61.2, 91.3],
    'T4': [57.5, 63.9, 91.1],
    'T5': [5.7, 17.1, 37.8],
}
# 95% CI per tier, same [Gemma-4B, Gemma-12B, Sonnet-5] order as T, as (lo, hi).
# Single-hop = paper-clustered bootstrap (835 papers); T5 = the graph-aware paper-level
# bootstrap over connected components of the paper-pair graph (matches NUMBERS_v2).
CI = {
    'T1': [(69.8, 73.9), (73.5, 77.5), (88.4, 91.6)],
    'T2': [(61.4, 65.8), (66.0, 70.3), (89.5, 92.7)],
    'T3': [(53.7, 58.2), (58.9, 63.4), (89.7, 92.8)],
    'T4': [(55.2, 59.8), (61.6, 66.1), (89.5, 92.7)],
    'T5': [(3.6, 8.3), (12.3, 22.4), (32.5, 43.5)],
}
SINGLEHOP = ['T1', 'T2', 'T3', 'T4']
# Single-hop MEANS (T1-T4) for the Figure S4 comparison are the authoritative
# per-system means given in NUMBERS_v2.md, not recomputed from the rounded
# per-tier values above (recomputing gives 62.26/67.18/90.88 -- a rounding
# artifact of averaging already-rounded display figures).
SH_MEAN = [62.3, 67.2, 90.9]   # Gemma-4B, Gemma-12B, Sonnet-5
# 95% CI for the pooled single-hop mean (T1-T4 combined), paper-clustered
# bootstrap: 835 gold papers underlie the matched N=2188 single-hop set common
# to all three systems; each of 5,000 resamples draws papers with replacement
# and pools all of that paper's T1-T4 records before taking the mean (2.5th/
# 97.5th percentiles). Source: manifests/best3_rejudge/{gemma-4b,gemma-12b,
# frontier}/poll_verdicts.json joined to litbench/queries.parquet's
# gold_paper_ids (single-hop rows; one gold paper per question), 2026-07-22.
# Point estimates reproduce SH_MEAN to within rounding (62.26/67.18/90.88).
SH_MEAN_CI = [(60.3, 64.2), (65.3, 69.1), (89.3, 92.4)]
# Blue sequential ramp for the four single-hop tiers; salmon for the T5 two-hop tier.
TIER_COLORS = {
    'T1': '#C6D7E9', 'T2': '#9CB9D6', 'T3': '#6E96C0', 'T4': '#3E6DA6',
    'T5': LANCET_SALMON,
}
TIER_LABELS = {
    'T1': 'T1 single-hop', 'T2': 'T2 single-hop', 'T3': 'T3 single-hop',
    'T4': 'T4 single-hop', 'T5': 'T5 two-hop (both facts)',
}
# Per-system palette, held fixed across Figure 3 and Figure S4: Sonnet-5 is
# always blue, Gemma-12B always grey, Gemma-4B always salmon. Order matches
# SYSTEMS / T / CI ([Gemma-4B, Gemma-12B, Sonnet-5]).
SYS_COLORS = [LANCET_SALMON, '#ADB6B6', LANCET_BLUE]
SYS_NAMES = ['Gemma-4B + IRCoT', 'Gemma-12B + IRCoT', 'Sonnet-5 + native retrieval']


def _style_axes(ax):
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel('Accuracy (%)', fontsize=11, color=LABEL_COLOR)
    ax.yaxis.grid(True, color=GRID_COLOR, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=LABEL_COLOR, labelsize=10)


# ============================ FIGURE 3 ============================
# Tier-level accuracy per system: T1-T4 single-hop (matched N=2188) connected
# by a line, then a gap and T5 two-hop (both-hops, all-N pooled) as a lone,
# unconnected point -- the single-hop and two-hop items are different question
# sets, so the two are not drawn as one continuous trend.
def figure3_tier_accuracy():
    tiers_sh = SINGLEHOP
    x_sh = np.arange(len(tiers_sh))
    x_t5 = x_sh[-1] + 1.5
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for i, (name, color) in enumerate(zip(SYS_NAMES, SYS_COLORS)):
        y_sh = [T[t][i] for t in tiers_sh]
        err_sh = [[T[t][i] - CI[t][i][0] for t in tiers_sh],
                  [CI[t][i][1] - T[t][i] for t in tiers_sh]]
        ax.errorbar(x_sh, y_sh, yerr=err_sh, color=color, marker='o', markersize=6.5,
                    markeredgecolor='white', markeredgewidth=0.9, linewidth=1.8,
                    capsize=3, elinewidth=1.0, zorder=3, label=name)
        y5, (lo5, hi5) = T['T5'][i], CI['T5'][i]
        ax.errorbar([x_t5], [y5], yerr=[[y5 - lo5], [hi5 - y5]], color=color, marker='D',
                    markersize=6.5, markeredgecolor='white', markeredgewidth=0.9,
                    linewidth=0, capsize=3, elinewidth=1.0, zorder=3)
        ax.text(x_t5 + 0.22, y5, f'{y5:.1f}', ha='left', va='center', fontsize=8.3,
                color=color, fontweight='bold', zorder=4)
    # visual break between the single-hop tiers and the two-hop tier
    sep_x = (x_sh[-1] + x_t5) / 2
    ax.axvline(sep_x, color=SPINE_COLOR, linestyle=(0, (3, 2)), linewidth=0.9, zorder=1)
    _style_axes(ax)
    ax.set_xticks(list(x_sh) + [x_t5])
    ax.set_xticklabels(list(tiers_sh) + ['T5'], fontsize=10, color=LABEL_COLOR)
    ax.set_xlim(x_sh[0] - 0.5, x_t5 + 0.85)
    trans = ax.get_xaxis_transform()
    ax.text(np.mean(x_sh), -0.15, 'Single-hop', ha='center', va='top', fontsize=8.6,
            color=MUTED_COLOR, fontstyle='italic', transform=trans)
    ax.text(x_t5, -0.15, 'Two-hop (synthesis)', ha='center', va='top', fontsize=8.6,
            color=MUTED_COLOR, fontstyle='italic', transform=trans)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, -0.30), ncol=3,
              fontsize=9.5, frameon=False, handletextpad=0.5, columnspacing=1.6)
    fig.savefig(OUT / 'figure3_tier_accuracy.pdf', dpi=600, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(OUT / 'figure3_tier_accuracy.png', dpi=600, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print('wrote figure3_tier_accuracy.{pdf,png}')


# ============================ FIGURE S4 ============================
# Single-hop mean (T1-T4) versus T5 both-hops, per system, with the gap
# annotated. This is a descriptive comparison of two different item sets (a
# single gold paper's questions versus a two-paper synthesis question), not a
# paired before/after measurement on the same items -- see caption. Same
# per-system palette as Figure 3; a hatch (not a hue change) distinguishes the
# T5 bar within each system's pair. Both bars carry 95% CI whiskers: paper-
# clustered bootstrap for the single-hop mean (SH_MEAN_CI), question-clustered
# bootstrap for T5 (CI['T5']).
def figureS4_wall():
    t5 = T['T5']
    t5_err = [[t5[i] - CI['T5'][i][0] for i in range(len(SYSTEMS))],
              [CI['T5'][i][1] - t5[i] for i in range(len(SYSTEMS))]]
    sh_err = [[SH_MEAN[i] - SH_MEAN_CI[i][0] for i in range(len(SYSTEMS))],
              [SH_MEAN_CI[i][1] - SH_MEAN[i] for i in range(len(SYSTEMS))]]
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    x = np.arange(len(SYSTEMS))
    bar_w = 0.34
    ax.bar(x - bar_w / 2, SH_MEAN, bar_w, color=SYS_COLORS,
           edgecolor='white', linewidth=0.6, zorder=3,
           yerr=sh_err, capsize=3, error_kw=dict(ecolor='#555555', lw=1.0, zorder=5))
    ax.bar(x + bar_w / 2, t5, bar_w, color=SYS_COLORS, hatch='///',
           edgecolor=LABEL_COLOR, linewidth=0.7, zorder=3,
           yerr=t5_err, capsize=3, error_kw=dict(ecolor='#555555', lw=1.0, zorder=5))
    # value labels sit beside each bar, not above it, so they clear the error-bar whiskers
    for xi, v in zip(x - bar_w / 2, SH_MEAN):
        ax.text(xi - bar_w / 2 - 0.06, v, f'{v:.0f}', ha='right', va='center', fontsize=8.5, color=LABEL_COLOR, zorder=4)
    for xi, v in zip(x + bar_w / 2, t5):
        ax.text(xi + bar_w / 2 + 0.06, v, f'{v:.1f}', ha='left', va='center', fontsize=8.5, color=LABEL_COLOR, zorder=4)
    # gap annotation (Delta pp) above each pair, cleared against the taller CI whisker
    for i in range(len(SYSTEMS)):
        gap = SH_MEAN[i] - t5[i]
        top = max(SH_MEAN_CI[i][1], CI['T5'][i][1])
        ax.annotate(f'Δ {gap:.0f} pp', xy=(x[i], top + 4.5), ha='center', va='bottom',
                    fontsize=8.5, color=MUTED_COLOR, fontstyle='italic', zorder=4)
    _style_axes(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([s.replace(' + ', '\n+ ') for s in SYS_NAMES], fontsize=10, color=LABEL_COLOR)
    ax.set_xlim(-0.6, len(SYSTEMS) - 0.4)
    # in-figure subtitle: the two bars are different item sets, not a paired measurement
    ax.text(0.5, 1.06, 'Descriptive comparison of different item sets', transform=ax.transAxes,
            ha='center', va='bottom', fontsize=8.8, color=MUTED_COLOR, fontstyle='italic')
    # legend encodes the fact-set pattern only; system identity is already
    # carried by bar color and the x-axis labels.
    solid_patch = mpatches.Patch(facecolor='#D9DCDE', edgecolor=LABEL_COLOR, linewidth=0.8,
                                  label='Single-paper mean (T1–T4)')
    hatch_patch = mpatches.Patch(facecolor='#D9DCDE', edgecolor=LABEL_COLOR, linewidth=0.8,
                                  hatch='///', label='Two-paper, both facts (T5)')
    ax.legend(handles=[solid_patch, hatch_patch], loc='lower center',
              bbox_to_anchor=(0.5, -0.24), ncol=2, fontsize=9.5, frameon=False,
              handletextpad=0.5, columnspacing=2.2)
    fig.savefig(OUT / 'figureS4_wall.pdf', dpi=600, bbox_inches='tight', pad_inches=0.15)
    fig.savefig(OUT / 'figureS4_wall.png', dpi=600, bbox_inches='tight', pad_inches=0.15)
    plt.close(fig)
    print('wrote figureS4_wall.{pdf,png}')


if __name__ == '__main__':
    # Figure 1 (3-panel flowchart) is built by consort/make_consort.py; Figure 2 (distractor
    # grid) by figures/build_grid_figure.py. This script owns ONLY Figure 3 and Figure S4.
    # Do NOT re-add figure1()/figure2() here — they clobber the canonical figure1_overview.png.
    figure3_tier_accuracy()
    figureS4_wall()
