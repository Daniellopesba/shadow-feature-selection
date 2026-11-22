"""A fraud-style feature table: missing values, categoricals, booleans, weights, time.

60,000 card transactions ordered in time, about 0.5% fraud (about 300 rows). Eight columns
drive the label (one with 30% missing values, two categoricals, one boolean),
and 20 feature-store columns are noise. The table goes into the selector as a
pandas DataFrame: no imputation, no encoding. XGBoost and LightGBM read the
missing values and categoricals themselves.

Shows:
  * the screen and the selector on a raw DataFrame;
  * XGBoost and LightGBM settings that do not memorise a few hundred positives;
  * cost weights passed through the pipeline (``sample_weight``);
  * selection inside time-ordered cross-validation.

Run:  python examples/04_fraud_table.py     (needs xgboost and lightgbm)
"""

import warnings

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline

from shadowselect import ShadowFeatureSelector, UnivariateScreen, assess_imbalance

rng = np.random.default_rng(0)
n = 60_000
df = pd.DataFrame(
    {
        "amount": np.exp(rng.normal(3.5, 1.2, n)),
        "merchant_cat": pd.Categorical(
            rng.choice(["grocery", "travel", "electronics", "gaming", "fuel", "other"], n)
        ),
        "country": pd.Categorical(rng.choice(["US", "GB", "BR", "NG", "DE", "IN"], n)),
        "device_age_days": rng.exponential(200, n),
        "n_txn_24h": rng.poisson(2, n),
        "card_present": rng.random(n) < 0.5,
        "credit_score": rng.normal(680, 60, n),
    }
)
df["amount_rounded"] = df["amount"].round(-1) + rng.normal(0, 1, n)  # near-duplicate
for j in range(20):
    df[f"fs_{j:02d}"] = rng.normal(size=n)
logit = (
    0.6 * np.log(df.amount)
    + 0.9 * df.merchant_cat.isin(["electronics", "gaming"])
    - 0.4 * np.log1p(df.device_age_days)
    + 0.25 * df.n_txn_24h
    - 0.6 * df.card_present
    + 0.7 * (df.country == "NG")
    - 0.006 * (df.credit_score - 680)
).to_numpy(dtype=float)
y = (rng.random(n) < 1 / (1 + np.exp(-(logit - 6.8)))).astype(int)
df.loc[rng.random(n) < 0.3, "credit_score"] = np.nan  # no bureau record
df.loc[rng.random(n) < 0.05, "device_age_days"] = np.nan
real = ["amount", "merchant_cat", "country", "device_age_days", "n_txn_24h"]
real += ["card_present", "credit_score", "amount_rounded"]
weight = np.where(y == 1, np.clip(df.amount, 5, 500) / 50, 1.0)  # chargeback cost
print(f"{n} rows, {y.sum()} frauds, {df.shape[1]} columns\n")

print(assess_imbalance(y, n_features=df.shape[1]), "\n")

# 1. Rank-test screen (seconds). Categoricals get a chi-square test.
screen = UnivariateScreen().fit(df, y)
print("screen keeps:", list(screen.get_feature_names_out()), "\n")

# 2. Selector with two learners that cannot memorise ~300 positives. Both read
#    the categoricals and missing values natively.
from xgboost import XGBClassifier  # noqa: E402

lgbm = LGBMClassifier(
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=15,
    min_child_samples=50,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    verbose=-1,
)
xgb = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1, enable_categorical=True, verbosity=0
)
for name, learner in [("XGBoost", xgb), ("LightGBM", lgbm)]:
    selector = ShadowFeatureSelector(learner, random_state=0)
    selector.fit(df, y, sample_weight=weight)
    s = selector.summary()
    kept = set(s.feature[selector.support_])
    print(f"{name}: keeps {len(kept & set(real))}/8 real, {len(kept - set(real))}/20 noise")
    print(s[["feature", "decision", "hits", "delta_hat", "s_hat"]].head(8).round(2).to_string())
    print("adjustments:", selector.estimator_adjustments_, "\n")

# 3. Selection inside time-ordered CV; weights routed to both steps.
model = XGBClassifier(
    n_estimators=200, max_depth=3, learning_rate=0.05, enable_categorical=True, verbosity=0
)
pipe = Pipeline([("select", ShadowFeatureSelector(xgb, random_state=0)), ("model", model)])
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # early folds have few positives
    res = cross_validate(
        pipe,
        df,
        y,
        cv=TimeSeriesSplit(n_splits=3),
        scoring="average_precision",
        params={"select__sample_weight": weight, "model__sample_weight": weight},
    )
print(
    "time-ordered CV PR-AUC per fold:",
    np.round(res["test_score"], 3),
    f"(a random ranking scores about {y.mean():.3f})",
)
