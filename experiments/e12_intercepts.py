"""E12: XGBoost's default intercept by version (paper Section 3.1).

Fits a one-tree model at three prevalences and records the learned
base_score. Run once per installed XGBoost version; appends to
results/e12_intercepts.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost
from common import RESULTS

from shadowselect.simulation import make_dataset

rows = []
for p in (0.001, 0.01, 0.1):
    X, y = make_dataset(100_000, p, 0.5, 2, 2, 0)
    for tm in ("hist", "exact"):
        m = xgboost.XGBClassifier(n_estimators=1, max_depth=1, tree_method=tm, n_jobs=1).fit(X, y)
        cfg = m.get_booster().save_config()
        import json

        bs = json.loads(cfg)["learner"]["learner_model_param"]["base_score"]
        bs = float(bs.strip("[]").split(",")[0])
        rows.append(dict(xgboost=xgboost.__version__, p=p, tree_method=tm, base_score=bs))
path = RESULTS / "e12_intercepts.csv"
old = pd.read_csv(path) if path.exists() else pd.DataFrame()
if len(old):
    old = old[old.xgboost != xgboost.__version__]
pd.concat([old, pd.DataFrame(rows)]).to_csv(path, index=False)
print(pd.DataFrame(rows))
