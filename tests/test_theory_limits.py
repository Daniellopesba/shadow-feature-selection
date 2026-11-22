"""Numerical checks of the paper's limit results.

Covers the scaling variable s, the level of the max-shadow rule, the rank-test
power and the gain accounting.
"""

import numpy as np
import pytest
from scipy import stats
from sklearn.tree import DecisionTreeClassifier

from shadowselect.theory import (
    detection_probability,
    limit_fixed_threshold_noncentrality,
    rank_test_power,
)


def test_fixed_threshold_noncentrality_peaks_at_median_with_2_over_pi():
    u = np.linspace(0.01, 0.99, 981)
    for delta in (0.05, 0.02):
        nc = limit_fixed_threshold_noncentrality(1000, delta, u)
        assert u[np.argmax(nc)] == pytest.approx(0.5, abs=0.05)  # peak drifts by O(delta)
        assert nc.max() / (1000 * delta**2) == pytest.approx(2 / np.pi, rel=0.05)


def test_mann_whitney_is_mean_rank_fraction_of_positives():
    # Prop 8: U / (n_pos n_neg) = mean over positives of F_neg(Z_i) exactly
    rng = np.random.default_rng(0)
    n, m, delta = 20000, 40, 0.5
    x = rng.standard_normal(n)
    pos = np.zeros(n, bool)
    pos[:m] = True
    x[pos] += delta
    u = stats.mannwhitneyu(x[pos], x[~pos]).statistic / (m * (n - m))
    neg = np.sort(x[~pos])
    direct = np.mean(np.searchsorted(neg, x[pos]) / (n - m))
    assert u == pytest.approx(direct, abs=1e-12)
    # and it is close to the limit (1/m) sum Phi(Z_i)
    assert u == pytest.approx(np.mean(stats.norm.cdf(x[pos])), abs=0.01)


@pytest.mark.parametrize("s", [2.0, 8.0])
def test_rank_test_power_matches_simulation(s):
    rng = np.random.default_rng(1)
    m, alpha = 400, 0.1
    delta = np.sqrt(s / m)
    w = stats.norm.cdf(rng.standard_normal((4000, m)) + delta).mean(axis=1)
    z = np.sqrt(12 * m) * (w - 0.5)
    emp = np.mean(np.abs(z) > stats.norm.ppf(1 - alpha / 2))
    assert emp == pytest.approx(rank_test_power(s, alpha), abs=0.03)


def test_null_candidate_beats_max_shadow_one_in_s_plus_1():
    # Prop 6: rank-based, column-symmetric importance => P(hit) = 1/(S+1)
    rng = np.random.default_rng(2)
    S, reps, hits = 4, 400, 0
    for _ in range(reps):
        n = 300
        y = (rng.random(n) < 0.2).astype(int)
        real = rng.standard_normal(n) + 0.8 * y
        null = rng.standard_normal(n)
        cols = [real, null] + [rng.permutation(c) for c in (real, null, real, null)][:S]
        X = np.column_stack(cols)
        order = rng.permutation(X.shape[1])  # random column order breaks ties fairly
        imp = np.empty(X.shape[1])
        tree = DecisionTreeClassifier(max_depth=3, random_state=int(rng.integers(1 << 30)))
        imp[order] = tree.fit(X[:, order], y).feature_importances_
        hits += imp[1] > imp[2:].max()
    lo, hi = stats.binomtest(hits, reps).proportion_ci(0.99)
    assert lo <= 1 / (S + 1) <= hi + 0.01


def test_detection_probability_decreases_with_number_of_shadows():
    p = [detection_probability(50, 0.5, n_null=k, n_rep=1500, seed=0) for k in (1, 10, 100)]
    assert p[0] > p[1] > p[2]


def test_tree_gains_telescope_to_leaves_minus_root():
    # Prop 9: sum of a tree's split gains = sum_leaves G^2/(H+lam) - G_root^2/(H_root+lam)
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(3)
    n, lam, eta = 4000, 1.0, 0.3
    X = rng.standard_normal((n, 3))
    y = (rng.random(n) < 0.05 + 0.1 * (X[:, 0] > 0)).astype(float)
    b = y.mean()
    m = xgb.XGBClassifier(
        n_estimators=1,
        max_depth=3,
        tree_method="exact",
        reg_lambda=lam,
        learning_rate=eta,
        base_score=b,
        min_child_weight=1.0,
    ).fit(X, y)
    df = m.get_booster().trees_to_dataframe()
    leaves = df[df.Feature == "Leaf"]
    # leaf value = -eta G / (H + lam), Cover = H
    G = -leaves.Gain.to_numpy() * (leaves.Cover.to_numpy() + lam) / eta
    H = leaves.Cover.to_numpy()
    G_root = (b - y).sum()
    H_root = n * b * (1 - b)
    lhs = df[df.Feature != "Leaf"].Gain.sum()
    rhs = np.sum(G**2 / (H + lam)) - G_root**2 / (H_root + lam)
    assert lhs == pytest.approx(rhs, rel=1e-3)
