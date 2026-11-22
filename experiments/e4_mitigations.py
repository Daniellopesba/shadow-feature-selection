"""E4: mitigation strategies (paper Section 5).

Unlike E2, the data hold 5 informative AND 5 irrelevant *candidate*
features, and every method builds its own shadows (permuted copies of the
10 candidates). That lets us report both error types:

  FNR = informative features rejected,  FPR = irrelevant features kept.

Methods (all XGBoost, same hyper-parameters as E2):
  naive_auto     one fit, max-shadow rule, XGBoost 2.1 auto intercept
  naive          one fit, max-shadow rule, intercept = prevalence
  p90            one fit, 90th-percentile shadow rule
  class_weight   one fit, scale_pos_weight = n_neg / n_pos
  downsample     one fit on a 1:1 majority-downsampled sample
  early_stop     one fit with early stopping on a stratified 20% split
  boruta_gain    ShadowFeatureSelector, training-set gain, 20 rounds
  boruta_perm    ShadowFeatureSelector, held-out permutation importance, 20 rounds

For the two selector methods, ``<name>`` keeps accepted + tentative features
and ``<name>_strict`` keeps accepted features only.
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import numpy as np
from common import dataset, run_grid, xgb
from sklearn.model_selection import train_test_split

from shadowselect import ShadowFeatureSelector

N_REAL, N_IRR = 5, 5


def _one_shot(X, y, rng, model, q=1.0, fit_kw=None):
    d = X.shape[1]
    shadows = np.column_stack([X[rng.permutation(len(y)), j] for j in range(d)])
    model.fit(np.hstack([X, shadows]), y, **(fit_kw or {}))
    imp = model.feature_importances_
    return imp[:d] > np.quantile(imp[d:], q)


def run_method(name, X, y, seed):
    rng = np.random.default_rng(seed)
    prior = float(y.mean())
    if name == "naive_auto":
        return _one_shot(X, y, rng, xgb(random_state=seed))
    if name == "naive":
        return _one_shot(X, y, rng, xgb(random_state=seed, base_score=prior))
    if name == "p90":
        return _one_shot(X, y, rng, xgb(random_state=seed, base_score=prior), q=0.9)
    if name == "class_weight":
        w = (len(y) - y.sum()) / y.sum()
        return _one_shot(X, y, rng, xgb(random_state=seed, base_score=0.5, scale_pos_weight=w))
    if name == "downsample":
        pos = np.flatnonzero(y == 1)
        neg = rng.choice(np.flatnonzero(y == 0), size=len(pos), replace=False)
        idx = np.concatenate([pos, neg])
        return _one_shot(X[idx], y[idx], rng, xgb(random_state=seed, base_score=0.5))
    if name == "early_stop":
        d = X.shape[1]
        shadows = np.column_stack([X[rng.permutation(len(y)), j] for j in range(d)])
        XX = np.hstack([X, shadows])
        Xtr, Xva, ytr, yva = train_test_split(XX, y, test_size=0.2, stratify=y, random_state=seed)
        m = xgb(random_state=seed, base_score=prior, early_stopping_rounds=20)
        m.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
        imp = m.feature_importances_
        if not np.isfinite(imp).all() or imp.sum() == 0:  # no split at all
            return np.zeros(d, dtype=bool)
        return imp[:d] > imp[d:].max()
    if name in ("boruta_gain", "boruta_perm"):
        sel = ShadowFeatureSelector(
            xgb(base_score=prior),
            importance="native" if name == "boruta_gain" else "permutation",
            n_iter=20,
            n_repeats=3,
            min_minority=0,
            random_state=seed,
        ).fit(X, y)
        return sel.decision_  # array of accepted / tentative / rejected
    raise ValueError(name)


METHODS = (
    "naive_auto",
    "naive",
    "p90",
    "class_weight",
    "downsample",
    "early_stop",
    "boruta_gain",
    "boruta_perm",
)


def worker(s, r, seed):
    X, y = dataset(s, seed)
    rows = []
    for m in METHODS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out = run_method(m, X, y, seed)
        if out.dtype == object:  # selector decisions: report both keep policies
            variants = {m: out != "rejected", f"{m}_strict": out == "accepted"}
        else:
            variants = {m: out}
        for name, keep in variants.items():
            rows.append(
                dict(
                    method=name,
                    fnr=float(1 - keep[:N_REAL].mean()),
                    fpr=float(keep[N_REAL:].mean()),
                )
            )
    return rows


def scenarios(quick):
    out = []
    for n_min, p in itertools.product((10, 25, 50, 100, 250, 500), (0.1, 0.01)):
        n = int(round(n_min / p))
        out.append(
            dict(
                n=n,
                p=p,
                n_min=n_min,
                delta=0.5,
                n_real=N_REAL,
                n_noise=N_IRR,
                n_repeats=2 if quick else 20,
            )
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e4_mitigations", scenarios(ap.parse_args().quick), worker)
