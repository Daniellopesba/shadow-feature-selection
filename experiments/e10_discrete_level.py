"""E10: the level of the shadow rule with discrete features (paper, level of the shadow rule).

The level result assumes continuous columns (no ties). Here one candidate is an
irrelevant *binary* column among continuous candidates, or an irrelevant
continuous column among binary candidates, and we measure how often it beats
every shadow. Two shadow constructions: Boruta's (one permuted copy of each
candidate) and per-feature shadows (all K shadows are permutations of the
column under test), for which the level result holds without (A1) and (A3).
XGBoost (hist, 50 trees, depth 3, prevalence intercept), n = 1000, p = 0.1,
6 candidates (2 relevant at delta = 0.5), K = 6 shadows, column order shuffled
before every fit; mean gain, total gain and TreeSHAP importances.
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import methods as M
import numpy as np
from common import run_grid, xgb

D = 6


def data(case, rng, n=1000, p=0.1, delta=0.5):
    y = np.zeros(n, dtype=np.int8)
    y[: int(round(n * p))] = 1
    X = rng.standard_normal((n, D))
    X[y == 1, :2] += delta
    if case == "binary_null":  # test column 5 is binary, the rest continuous
        X[:, 5] = rng.random(n) < 0.3
    elif case == "continuous_null_among_binary":  # columns 0-4 binary, test column continuous
        for j in range(5):
            q = 0.3 + (0.15 if (j < 2) else 0.0) * (y == 1)
            X[:, j] = rng.random(n) < q
    perm = rng.permutation(n)
    return X[perm], y[perm]


def worker(s, r, seed):
    rng = np.random.default_rng(seed)
    X, y = data(s["case"], rng)
    j = D - 1  # the irrelevant column under test
    if s["shadows"] == "boruta":
        sh = M.permuted_shadows(X, rng)
    else:  # per-feature: K permutations of column j itself
        sh = np.column_stack([X[rng.permutation(len(y)), j] for _ in range(D)])
    XX = np.hstack([X, sh])
    order = rng.permutation(XX.shape[1])  # (A2): random column order
    inv = np.argsort(order)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = xgb(
            n_estimators=50, max_depth=3, random_state=seed, base_score=float(y.mean())
        ).fit(XX[:, order], y)
    rows = []
    for kind in ("gain", "total_gain", "shap"):
        if kind == "shap":
            imp = M.tree_shap_importance(m, XX[:, order], rng)[inv]
        else:
            imp = M.xgb_raw_importance(m, XX.shape[1], kind)[inv]
        rows.append(dict(importance=kind, hit=float(imp[j] > imp[D:].max())))
    return rows


def scenarios(quick):
    return [
        dict(case=c, shadows=sh, n_repeats=50 if quick else 2000)
        for c, sh in itertools.product(
            ("continuous", "binary_null", "continuous_null_among_binary"), ("boruta", "per_feature")
        )
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e10_discrete_level", scenarios(ap.parse_args().quick), worker)
