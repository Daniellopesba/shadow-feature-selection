"""The failure mode in one screen.

A fraud-like target: 40 000 rows, 0.1% positives (n_min = 40). Five features
carry a 0.5 SD signal, five are irrelevant. We compare
  (a) the common one-shot recipe: XGBoost training-set gain vs. the max shadow,
  (b) ShadowFeatureSelector with held-out permutation importance.

Run:  python examples/02_small_minority_pitfall.py
"""

import warnings

import numpy as np
from xgboost import XGBClassifier

from shadowselect import ShadowFeatureSelector
from shadowselect.simulation import make_dataset

X, y = make_dataset(n=40_000, p=0.001, delta=0.5, n_real=5, n_noise=5, seed=3)
print(f"n={len(y)}, positives={y.sum()}")
truth = np.r_[np.ones(5, bool), np.zeros(5, bool)]


def report(name, keep):
    fnr = 1 - keep[truth].mean()
    fpr = keep[~truth].mean()
    print(f"{name:<38s} kept={keep.astype(int)}  FNR={fnr:.1f}  FPR={fpr:.1f}")


# (a) one-shot gain vs. max shadow
rng = np.random.default_rng(0)
shadows = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])])
model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, verbosity=0)
model.fit(np.hstack([X, shadows]), y)
imp = model.feature_importances_
report("one-shot gain vs max shadow", imp[:10] > imp[10:].max())

# (b) repeated rounds, held-out permutation importance, intercept at the prevalence
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # we know the minority class is small
    sel = ShadowFeatureSelector(
        XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            verbosity=0,
            base_score=float(y.mean()),
        ),
        importance="permutation",
        n_iter=20,
        keep_tentative=True,
        random_state=0,
    ).fit(X, y)
report("selector (permutation, keep tentative)", sel.support_)
print(sel.summary()[["feature", "decision", "hits"]].to_string(index=False))
