"""Paper-clustered bootstrap confidence intervals for LitBench accuracies.

Question-clustering alone (``consort/bootstrap_ci.py``) understates uncertainty
when many questions share a source paper, and two-hop questions share a paper
pair. This resamples the higher-level cluster instead: source paper for
single-hop, connected component of the paper-pair graph for two-hop.

    df columns: paper_id, system, test_type, credit  (credit in {0,1})
    two-hop callers pass cluster_col="pair_component"
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def paper_clustered_ci(df, system, test_type, n_boot=5000, seed=42, cluster_col="paper_id"):
    d = df[(df.system == system) & (df.test_type == test_type)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(d["credit"].mean())
    by_cluster = {c: g["credit"].to_numpy() for c, g in d.groupby(cluster_col)}
    clusters = np.array(sorted(by_cluster))
    n_clusters = len(clusters)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        draw = clusters[rng.integers(0, n_clusters, n_clusters)]
        pooled = np.concatenate([by_cluster[c] for c in draw])
        boots.append(pooled.mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)
