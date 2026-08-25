"""Synthesis-penalty analysis: isolate the cost of the second hop.

For each two-hop (M3xxxx) question we ask its two constituent facts INDEPENDENTLY
under matched distractor conditions, then compare how often BOTH are recovered
separately with how often the joint two-hop synthesis succeeds.

    synthesis penalty = P(both constituents recovered separately) - P(joint two-hop success)

A large positive penalty means the system could retrieve both facts on their own but
failed to combine them -> a genuine synthesis cost, not a retrieval cost.

df_single: rows (m3, system, constituent in {A,B}, credit in {0,1})   [matched-condition single-fact runs]
df_joint : rows (m3, system, both_hops in {0,1})                      [T5 both-hops]
"""
from __future__ import annotations

import pandas as pd


def synthesis_penalty(df_single, df_joint):
    """Returns {system: {both_separate, joint, penalty, n}} at the question level."""
    out = {}
    for sysname in sorted(set(df_single.system) | set(df_joint.system)):
        s = df_single[df_single.system == sysname]
        both = s.groupby("m3")["credit"].min()          # both constituents recovered -> min == 1
        both_separate = float(both.mean()) if len(both) else float("nan")
        j = df_joint[df_joint.system == sysname]
        joint = float(j.groupby("m3")["both_hops"].max().mean()) if len(j) else float("nan")
        out[sysname] = dict(both_separate=both_separate, joint=joint,
                            penalty=both_separate - joint, n=int(both.index.nunique()))
    return out


def decompose_two_hop(df_single, df_joint):
    """Extends synthesis_penalty with the two-hop credit decomposition.

    Every two-hop cell falls into exactly one of three buckets, derived from the
    joint T5 attempt's own any_hop and both_hops credit (df_joint must carry an
    `any_hop` column alongside `both_hops`, see composite_t5.json):

        neither = 1 - any_hop            neither gold fact credited
        one     = any_hop - both_hops    exactly one gold fact credited
        both    = both_hops              both gold facts credited (== `joint`)

    This does NOT require the constituent-isolation join used by both_separate;
    it only re-expresses the joint attempt's own any_hop/both_hops split. Do not
    assert this decomposition alone shows "combining evidence" is the cause of
    two-hop failure -- a large `one` share is also consistent with failing to
    retrieve the second gold paper (see the O2 retrieval-split check).

    Returns {system: {..., any_hop, neither, one, both}} at the question level.
    """
    out = synthesis_penalty(df_single, df_joint)
    for sysname, block in out.items():
        j = df_joint[df_joint.system == sysname]
        any_hop = float(j.groupby("m3")["any_hop"].max().mean()) if len(j) else float("nan")
        block.update(any_hop=any_hop, neither=1 - any_hop, one=any_hop - block["joint"], both=block["joint"])
    return out
