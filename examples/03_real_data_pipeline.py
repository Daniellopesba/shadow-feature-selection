"""Real data in a scikit-learn pipeline, with a shrinking minority class.

Uses the Wisconsin breast-cancer data (569 rows, 30 features) plus 10 added
pure-noise columns. We subsample the malignant class to show how the number
of retained real features and the number of noise columns let through change
with the minority count.

Run:  python examples/03_real_data_pipeline.py
"""

import warnings

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from shadowselect import ShadowFeatureSelector

data = load_breast_cancer()
X0, y0 = data.data, (data.target == 0).astype(int)  # 1 = malignant (212 rows)
rng = np.random.default_rng(0)
X0 = np.hstack([X0, rng.standard_normal((len(y0), 10))])  # 10 noise columns
n_real = data.data.shape[1]

for n_pos in (212, 80, 30, 12):
    pos = rng.choice(np.flatnonzero(y0 == 1), n_pos, replace=False)
    idx = np.concatenate([pos, np.flatnonzero(y0 == 0)])
    X, y = X0[idx], y0[idx]
    sel = ShadowFeatureSelector(
        RandomForestClassifier(n_estimators=100, n_jobs=1),
        importance="native",
        n_iter=20,
        keep_tentative=True,
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sel.fit(X, y)
    kept = sel.get_support()
    flag = " (ImbalanceWarning)" if any("Minority" in str(x.message) for x in w) else ""
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    auc = (
        cross_val_score(pipe, X[:, kept], y, scoring="roc_auc", cv=3).mean()
        if kept.any()
        else float("nan")
    )
    print(
        f"positives={n_pos:4d}: kept {kept[:n_real].sum():2d}/30 real, "
        f"{kept[n_real:].sum()}/10 noise; CV AUC on kept={auc:.3f}{flag}"
    )
