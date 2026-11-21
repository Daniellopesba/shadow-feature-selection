"""E8: realistic numbers of null features (review 2).

Single-shot max rule, XGBoost with the prevalence intercept, delta = 0.5,
5 informative features and 5, 50 or 200 null features (the nulls are the
shadows, as in E2).
"""

from __future__ import annotations

import argparse
import itertools

from common import dataset, run_grid, xgb

from shadowselect.simulation import selection_metrics


def worker(s, r, seed):
    X, y = dataset(s, seed)
    m = xgb(random_state=seed, base_score=float(y.mean())).fit(X, y)
    return [selection_metrics(m.feature_importances_, s["n_real"], s["n_noise"])]


def scenarios(quick):
    out = []
    for n_noise, n_min in itertools.product((5, 50, 200), (50, 100, 250, 500)):
        out.append(
            dict(
                n=n_min * 10,
                p=0.1,
                n_min=n_min,
                delta=0.5,
                n_real=5,
                n_noise=n_noise,
                n_repeats=2 if quick else 15,
            )
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e8_many_shadows", scenarios(ap.parse_args().quick), worker)
