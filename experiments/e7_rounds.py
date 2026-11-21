"""E7: Boruta-style selection, number of rounds and FPR-matched curves (review 2).

E4 design (5 relevant + 5 irrelevant candidates, delta = 0.5) with XGBoost
training-set gain, prevalence intercept. Each repeat runs 60 rounds with fresh
shuffled shadows and stores the per-round hit vector, so decisions for any
n_iter <= 60 and any hit-count threshold can be computed afterwards.
"""

from __future__ import annotations

import argparse
import itertools

import numpy as np
from common import dataset, run_grid, xgb

ROUNDS = 60


def worker(s, r, seed):
    X, y = dataset(s, seed)
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    rows = []
    for k in range(ROUNDS):
        sh = np.column_stack([X[rng.permutation(len(y)), j] for j in range(d)])
        m = xgb(random_state=int(rng.integers(2**31 - 1)), base_score=float(y.mean()))
        m.fit(np.hstack([X, sh]), y)
        imp = m.feature_importances_
        hit_max = imp[:d] > imp[d:].max()
        hit_90 = imp[:d] > np.quantile(imp[d:], 0.9)
        rows.append(
            dict(
                round=k,
                hits_max="".join("1" if h else "0" for h in hit_max),
                hits_q90="".join("1" if h else "0" for h in hit_90),
            )
        )
    return rows


def scenarios(quick):
    out = []
    for n_min, p in itertools.product((10, 25, 50, 100, 250), (0.1, 0.01)):
        out.append(
            dict(
                n=int(round(n_min / p)),
                p=p,
                n_min=n_min,
                delta=0.5,
                n_real=5,
                n_noise=5,
                n_repeats=2 if quick else 10,
            )
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e7_rounds", scenarios(ap.parse_args().quick), worker)
