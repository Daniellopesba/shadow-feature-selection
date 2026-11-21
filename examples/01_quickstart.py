"""Quickstart: check the minority count, then select features.

Run:  python examples/01_quickstart.py
"""

from sklearn.ensemble import RandomForestClassifier

from shadowselect import ShadowFeatureSelector, assess_imbalance
from shadowselect.simulation import make_dataset

# 4 informative features (shift 0.7 SD in the positive class) + 4 irrelevant ones,
# 5% positives -> 150 minority samples.
X, y = make_dataset(n=3000, p=0.05, delta=0.7, n_real=4, n_noise=4, seed=0)

# 1. Before selecting anything: how small is the minority class?
print(assess_imbalance(y))

# 2. Select. Defaults: permuted shadows (Boruta), held-out permutation
#    importance, 20 rounds, binomial test with Bonferroni correction.
selector = ShadowFeatureSelector(
    RandomForestClassifier(n_estimators=200, min_samples_leaf=5, n_jobs=-1),
    n_iter=20,
    random_state=0,
)
selector.fit(X, y)
print(selector.summary().to_string(index=False))
print("kept columns:", selector.get_support(indices=True))
