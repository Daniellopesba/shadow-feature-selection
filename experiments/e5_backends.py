"""E5: is the effect specific to XGBoost? (paper Section 4.5)

Single-shot max-shadow rule (E2 design, delta = 0.5) with other learners'
native training-set importances:
  sklearn GradientBoostingClassifier (MSE-on-residuals splits, MDI),
  RandomForestClassifier (Gini MDI; Boruta's learner),
  LightGBM (split gain).
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np
from common import dataset, run_grid
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from shadowselect.simulation import selection_metrics


def model(name, seed):
    if name == "sklearn_gbm":
        return GradientBoostingClassifier(
            n_estimators=200, max_depth=3, learning_rate=0.1, random_state=seed
        )
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=300, n_jobs=1, random_state=seed)
    if name in ("lightgbm", "lightgbm_default"):
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            num_leaves=63 if name == "lightgbm" else 31,  # 31 is LightGBM's default
            learning_rate=0.1,
            importance_type="gain",
            n_jobs=1,
            verbose=-1,
            random_state=seed,
        )
    raise ValueError(name)


def worker(s, r, seed):
    X, y = dataset(s, seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = model(s["backend"], seed).fit(X, y)
    rows = [
        dict(
            importance="native",
            **selection_metrics(m.feature_importances_, s["n_real"], s["n_noise"]),
        )
    ]
    if s["backend"].startswith("lightgbm"):
        # mean gain per split, comparable to XGBoost's default "gain"
        booster = m.booster_
        tot = booster.feature_importance("gain")
        cnt = booster.feature_importance("split")
        mean = np.divide(tot, cnt, out=np.zeros_like(tot, dtype=float), where=cnt > 0)
        rows.append(
            dict(importance="mean_gain", **selection_metrics(mean, s["n_real"], s["n_noise"]))
        )
    return rows


BACKENDS = ("sklearn_gbm", "random_forest", "lightgbm", "lightgbm_default")


def scenarios(quick):
    out = []
    for backend, n_min, p in itertools.product(BACKENDS, (10, 25, 50, 100, 250, 500), (0.1, 0.01)):
        n = int(round(n_min / p))
        out.append(
            dict(
                backend=backend,
                n=n,
                p=p,
                n_min=n_min,
                delta=0.5,
                n_real=5,
                n_noise=5,
                n_repeats=2 if quick else 15,
            )
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e5_backends", scenarios(ap.parse_args().quick), worker)
