"""Quickstart: check the minority count, select, and read the summary.

Run:  python examples/01_quickstart.py
"""

from shadowselect import ShadowFeatureSelector, UnivariateScreen, assess_imbalance
from shadowselect.simulation import make_dataset

# 4 informative features (shift 0.7 SD in the positive class) + 4 irrelevant ones,
# 5% positives -> 250 minority rows, so s = n_min * delta^2 = 122 for the real ones.
X, y = make_dataset(n=5000, p=0.05, delta=0.7, n_real=4, n_noise=4, seed=0)

# 1. Before selecting anything: what can this many positives support?
print(assess_imbalance(y, n_features=X.shape[1]), "\n")

# 2. A rank-test screen: seconds, uses every row, controls false discoveries.
print(UnivariateScreen().fit(X, y).summary().round(4).to_string(index=False), "\n")

# 3. The shadow-feature selector with its defaults: a shallow, class-balanced
#    random forest, 10 shadows per round, 20 rounds, native (impurity)
#    importance, binomial tests with Benjamini-Hochberg correction, tentative
#    features kept.
selector = ShadowFeatureSelector(random_state=0, n_jobs=-1).fit(X, y)
cols = ["feature", "decision", "hits", "delta_hat", "s_hat", "underpowered"]
print(selector.summary()[cols].round(2).to_string(index=False))
print("kept columns:", selector.get_support(indices=True))
print(f"expected share of irrelevant features kept: {selector.expected_null_keep_rate_:.1%}")
