"""E6: univariate baselines on the E4 datasets (review 2).

Same scenarios and seeds as E4, so every method sees identical data. For each
candidate we compute a two-sided Mann-Whitney test of X_j between classes.
Rules:
  mwu_bh_q        keep if Benjamini-Hochberg adjusted p < q (q = 0.05, 0.1, 0.2)
  mwu_bonf        keep if p < 0.05 / 10
  mwu_shadow_max  keep if p is smaller than the smallest p of 10 shuffled
                  shadows: the max-shadow rule with a univariate importance
"""

from __future__ import annotations

import argparse

import numpy as np
from common import dataset, run_grid
from e4_mitigations import N_REAL, scenarios
from scipy import stats


def bh(p):
    p = np.asarray(p)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    adj[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(adj, 1)


def worker(s, r, seed):
    X, y = dataset(s, seed)
    rng = np.random.default_rng(seed)
    pos, neg = y == 1, y == 0
    pv = np.array([stats.mannwhitneyu(X[pos, j], X[neg, j]).pvalue for j in range(X.shape[1])])
    sh = np.column_stack([X[rng.permutation(len(y)), j] for j in range(X.shape[1])])
    psh = np.array([stats.mannwhitneyu(sh[pos, j], sh[neg, j]).pvalue for j in range(sh.shape[1])])
    rules = {
        "mwu_bh_0.05": bh(pv) < 0.05,
        "mwu_bh_0.1": bh(pv) < 0.1,
        "mwu_bh_0.2": bh(pv) < 0.2,
        "mwu_bonf": pv < 0.05 / len(pv),
        "mwu_shadow_max": pv < psh.min(),
    }
    return [
        dict(method=m, fnr=float(1 - k[:N_REAL].mean()), fpr=float(k[N_REAL:].mean()))
        for m, k in rules.items()
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e6_univariate", scenarios(ap.parse_args().quick), worker)
