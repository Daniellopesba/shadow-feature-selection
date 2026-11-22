"""E1: first-split theory checks (paper: the first split).

(a) Null maximum of 2*Gain over thresholds for one uninformative feature, as a
    function of n_min at several n (Monte Carlo, XGBoost defaults lam=1,
    min_child_weight=1).
(b) Which feature does XGBoost's very first split use? One informative
    feature (shift delta) competes with ``n_noise`` uninformative ones. The
    theory predicts P(root = informative) depends on n_min * delta^2.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
import pandas as pd
from common import RESULTS, SUFFIX, run_grid, xgb

from shadowselect.simulation import make_dataset
from shadowselect.theory import max_null_gain


def null_max_table(n_rep: int) -> pd.DataFrame:
    rows = []
    for n in (1_000, 10_000, 100_000):
        for n_min in (5, 10, 25, 50, 100, 250, 500, 1000):
            if n_min >= n / 2:
                continue
            for lam, mcw, bs in ((0.0, 0.0, None), (1.0, 1.0, None), (1.0, 1.0, "auto")):
                # "auto": XGBoost 2.1's one-Newton-step intercept from 0.5
                base = None if bs is None else 1 / (1 + np.exp(-(4 * n_min / n - 2)))
                draws = max_null_gain(
                    n,
                    n_min,
                    n_rep=n_rep,
                    lam=lam,
                    min_child_weight=mcw,
                    base_score=base,
                    seed=n_min,
                )
                rows.append(
                    dict(
                        n=n,
                        n_min=n_min,
                        lam=lam,
                        min_child_weight=mcw,
                        intercept=bs or "prior",
                        mean=draws.mean(),
                        q50=np.quantile(draws, 0.5),
                        q90=np.quantile(draws, 0.9),
                        q95=np.quantile(draws, 0.95),
                    )
                )
    return pd.DataFrame(rows)


def root_worker(s, r, seed):
    X, y = make_dataset(s["n"], s["p"], s["delta"], 1, s["n_noise"], seed)
    # random column order: XGBoost breaks exact gain ties by column index
    perm = np.random.default_rng(seed).permutation(X.shape[1])
    real_col = int(np.flatnonzero(perm == 0)[0])
    kw = {"base_score": float(y.mean())} if s["intercept"] == "prior" else {}
    m = xgb(n_estimators=1, max_depth=1, tree_method=s["tree_method"], random_state=seed, **kw).fit(
        X[:, perm], y
    )
    trees = m.get_booster().trees_to_dataframe()
    root = trees[(trees.Tree == 0) & (trees.Node == 0)]
    feat = root.Feature.iloc[0]
    if feat == "Leaf":
        outcome = "no_split"
    else:
        outcome = "informative" if feat == f"f{real_col}" else "noise"
    return [
        dict(
            outcome=outcome,
            root_is_real=float(outcome == "informative"),
            root_gain=float(root.Gain.iloc[0]) if outcome != "no_split" else 0.0,
        )
    ]


def main(quick: bool, root_only: bool = False):
    RESULTS.mkdir(exist_ok=True)
    if not root_only:
        tab = null_max_table(100 if quick else 1000)
        tab.to_csv(RESULTS / f"e1_null_max{SUFFIX}.csv", index=False)
        print(tab.to_string())
    scen = []
    reps = 10 if quick else 100
    for n_min in (10, 25, 50, 100, 250, 500, 1000):
        for p in (0.001, 0.01, 0.1):
            n = int(round(n_min / p))
            if n > 200_000 or n < 200:
                continue
            for delta, icpt, tm in itertools.product(
                (0.1, 0.2, 0.3, 0.5, 1.0), ("auto", "prior"), ("hist", "exact")
            ):
                scen.append(
                    dict(
                        n=n,
                        p=p,
                        delta=delta,
                        n_noise=9,
                        n_min=n_min,
                        intercept=icpt,
                        tree_method=tm,
                        n_repeats=reps,
                    )
                )
    run_grid("e1_root_split", scen, root_worker)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    ap.add_argument("--root-only", action="store_true", help="skip the null-max table")
    a = ap.parse_args()
    main(a.quick, a.root_only)
