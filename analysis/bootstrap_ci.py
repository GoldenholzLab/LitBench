"""Question-clustered bootstrap confidence intervals for LitBench accuracies.

Resamples by ``question_id`` (cluster bootstrap) so that repeated haystack sizes
(N) on the same question are not treated as independent observations. Point
estimates are the mean over per-question mean credit, matching how LitBench
aggregates a (domain, test type, N) sweep.

    df columns: question_id, system, test_type, N, credit  (credit in {0,1})
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _question_means(df: pd.DataFrame, system: str, test_type: str) -> pd.Series:
    d = df[(df.system == system) & (df.test_type == test_type)]
    # collapse repeated N per question to the question's mean credit (cluster unit = question)
    return d.groupby("question_id")["credit"].mean()


def system_accuracy_ci(df, system, test_type, n_boot=5000, seed=0):
    qm = _question_means(df, system, test_type).to_numpy()
    if len(qm) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(qm.mean())
    rng = np.random.default_rng(seed)
    boots = [qm[rng.integers(0, len(qm), len(qm))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_diff_ci(df, sys_a, sys_b, test_type, n_boot=5000, seed=0):
    a = _question_means(df, sys_a, test_type)
    b = _question_means(df, sys_b, test_type)
    common = a.index.intersection(b.index)          # paired on shared questions
    da, db = a.loc[common].to_numpy(), b.loc[common].to_numpy()
    if len(common) == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(da.mean() - db.mean())
    rng = np.random.default_rng(seed)
    idx = np.arange(len(common))
    boots = [(da[s].mean() - db[s].mean())
             for s in (rng.integers(0, len(idx), len(idx)) for _ in range(n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, float(lo), float(hi)
