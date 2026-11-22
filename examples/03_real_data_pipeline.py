"""Real data, selection inside cross-validation, and a shrinking minority class.

Uses the Wisconsin breast-cancer data (569 rows), keeping only its 15 weaker
features (single-feature ROC AUC below 0.8) and adding 15 pure-noise columns,
so that selection has something to get wrong. We subsample the malignant
class to 212, 60 and 25 rows.

The selector sits *inside* the pipeline that is cross-validated, so each fold
selects on its own training rows. Selecting on all rows and then
cross-validating the model on the kept columns leaks the test folds into the
selection and overstates performance.

Run:  python examples/03_real_data_pipeline.py
"""

import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shadowselect import ImbalanceWarning, ShadowFeatureSelector

data = load_breast_cancer()
X_all, y_all = data.data, (data.target == 0).astype(int)  # 1 = malignant (212 rows)
auc = np.array([roc_auc_score(y_all, x) for x in X_all.T])
weak = np.flatnonzero(np.maximum(auc, 1 - auc) < 0.8)  # the 15 weaker features
rng = np.random.default_rng(0)
X_all = np.hstack([X_all[:, weak], rng.standard_normal((len(y_all), 15))])
n_real = len(weak)
print(f"{n_real} real features (single-feature AUC 0.51-0.79) + 15 noise columns\n")


def model():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for n_pos in (212, 60, 25):
    pos = rng.choice(np.flatnonzero(y_all == 1), n_pos, replace=False)
    idx = np.concatenate([pos, np.flatnonzero(y_all == 0)])
    X, y = X_all[idx], y_all[idx]

    selected = make_pipeline(ShadowFeatureSelector(n_jobs=-1, random_state=0), model())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ImbalanceWarning)
        res = cross_validate(
            selected, X, y, cv=cv, scoring="average_precision", return_estimator=True
        )
    full = cross_validate(model(), X, y, cv=cv, scoring="average_precision")
    kept = np.array([p[0].get_support() for p in res["estimator"]])
    flag = "  (ImbalanceWarning)" if caught else ""
    print(
        f"positives={n_pos:3d} (~{int(0.8 * n_pos)} per training fold){flag}\n"
        f"  kept per fold: {kept[:, :n_real].sum(1).mean():4.1f}/{n_real} real, "
        f"{kept[:, n_real:].sum(1).mean():4.1f}/15 noise\n"
        f"  PR-AUC with selection {res['test_score'].mean():.3f}, "
        f"all features {full['test_score'].mean():.3f}"
    )
