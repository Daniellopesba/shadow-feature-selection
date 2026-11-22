import numpy as np
import pytest

from shadowselect.theory import (
    best_signal_noncentrality,
    max_null_gain,
    min_detectable_delta,
    pearson_chi2_2x2,
    signal_noncentrality,
    split_gains,
)


@pytest.mark.parametrize("n,k", [(200, 5), (1000, 50), (500, 250)])
def test_two_gain_equals_pearson_chi2(n, k):
    rng = np.random.default_rng(n + k)
    x = rng.standard_normal(n)
    y = np.zeros(n)
    y[rng.choice(n, k, replace=False)] = 1
    thr, g2 = split_gains(x, y, lam=0.0)
    for j in rng.choice(len(thr), 10, replace=False):
        assert g2[j] == pytest.approx(pearson_chi2_2x2(x, y, thr[j]), rel=1e-9, abs=1e-9)


def test_null_gain_mean_is_n_over_n_minus_1():
    # E[2 Gain(t)] = n/(n-1) at every fixed split position, any n_min.
    n, k, pos = 60, 4, 17
    rng = np.random.default_rng(0)
    vals = []
    for _ in range(40000):
        y = np.zeros(n)
        y[rng.choice(n, k, replace=False)] = 1
        p = k / n
        GL = np.sum(p - y[:pos])
        HL, HR = p * (1 - p) * pos, p * (1 - p) * (n - pos)
        vals.append(GL**2 / HL + GL**2 / HR)
    assert np.mean(vals) == pytest.approx(n / (n - 1), rel=0.02)


def test_small_delta_noncentrality_matches_2_over_pi():
    n, p, delta = 100000, 0.01, 0.05
    nc, t = best_signal_noncentrality(n, p, delta)
    approx = 2 / np.pi * n * p * (1 - p) * delta**2
    assert nc == pytest.approx(approx, rel=0.03)
    assert abs(t) < 0.2


def test_noncentrality_zero_without_signal():
    assert np.all(signal_noncentrality(1000, 0.1, 0.0, np.linspace(-2, 2, 9)) == 0)
    assert best_signal_noncentrality(1000, 0.1, 0.0) == (0.0, 0.0)


def test_noncentrality_scales_with_minority_count():
    a, _ = best_signal_noncentrality(10000, 0.01, 0.5)
    b, _ = best_signal_noncentrality(20000, 0.01, 0.5)
    assert b / a == pytest.approx(2.0, rel=1e-6)


def test_max_null_gain_is_reproducible_and_positive():
    a = max_null_gain(500, 20, n_rep=50, seed=3)
    b = max_null_gain(500, 20, n_rep=50, seed=3)
    np.testing.assert_array_equal(a, b)
    assert np.all(a > 0)


def test_min_detectable_delta_monotone_in_n_min():
    assert min_detectable_delta(10, 8.0) > min_detectable_delta(100, 8.0)
    assert min_detectable_delta(0, 8.0) == float("inf")


def test_wrong_intercept_scales_chi2_without_constraints():
    # With constant Hessian and lam = 0, a wrong base_score rescales every
    # split's gain by p(1-p) / (b(1-b)): the ranking of splits is unchanged.
    rng = np.random.default_rng(1)
    n, k, b = 2000, 20, 0.12
    x = rng.standard_normal(n)
    y = np.zeros(n)
    y[rng.choice(n, k, replace=False)] = 1
    _, g_prior = split_gains(x, y)
    _, g_auto = split_gains(x, y, base_score=b)
    p = k / n
    np.testing.assert_allclose(g_auto, g_prior * p * (1 - p) / (b * (1 - b)), rtol=1e-6, atol=1e-9)


def test_null_max_depends_on_n_min_not_n():
    a = max_null_gain(5_000, 50, n_rep=400, lam=1.0, min_child_weight=1.0, seed=0)
    b = max_null_gain(100_000, 50, n_rep=400, lam=1.0, min_child_weight=1.0, seed=1)
    assert abs(np.median(a) - np.median(b)) < 0.15 * np.median(a)


def test_limit_process_null_median_grows_slowly():
    from shadowselect.theory import limit_max_gain

    m10 = np.median(limit_max_gain(10, 0.0, n_rep=300))
    m800 = np.median(limit_max_gain(800, 0.0, n_rep=300))
    assert 1.0 < m10 < m800 < 3 * m10


def test_limit_process_matches_finite_n():
    from shadowselect.theory import limit_max_gain

    lim = limit_max_gain(50, 0.0, n_rep=600, seed=0)
    fin = max_null_gain(100_000, 50, n_rep=600, lam=1.0, min_child_weight=1.0, seed=1)
    assert abs(np.median(lim) - np.median(fin)) < 0.1 * np.median(lim)


def test_detection_probability_monotone():
    from shadowselect.theory import detection_probability

    a = detection_probability(50, 0.2, n_null=5, n_rep=300)
    b = detection_probability(50, 0.8, n_null=5, n_rep=300)
    assert a < b and b > 0.9


def test_intercept_toll_identity():
    # Proposition 5, eq. (7): exact decomposition of the gain under a wrong intercept.
    rng = np.random.default_rng(4)
    n, k, b, lam = 3000, 30, 0.12, 1.0
    x = rng.standard_normal(n)
    y = np.zeros(n)
    y[rng.choice(n, k, replace=False)] = 1
    _, gains = split_gains(x, y, lam=lam, base_score=b)
    order = np.argsort(x)
    ys = y[order]
    p, h = k / n, b * (1 - b)
    H = h * n
    r = (b - p) / h
    nL = np.arange(1, n)
    kL = np.cumsum(ys)[:-1]
    DL = -(kL - p * nL)
    DR = -DL
    HL, HR = h * nL, h * (n - nL)
    pred = (DL - r * lam) ** 2 / (HL + lam) + (DR - r * lam) ** 2 / (HR + lam)
    pred -= lam * r**2 * (H + 2 * lam) / (H + lam)
    np.testing.assert_allclose(gains, pred, rtol=1e-7, atol=1e-8)


def test_xgboost_reported_root_gain_is_pearson_chi2():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(5)
    n, k = 3000, 60
    x = rng.standard_normal((n, 1))
    y = np.zeros(n)
    y[rng.choice(n, k, replace=False)] = 1
    m = xgb.XGBClassifier(
        n_estimators=1,
        max_depth=1,
        tree_method="exact",
        reg_lambda=0.0,
        min_child_weight=0.0,
        base_score=k / n,
        learning_rate=1.0,
    ).fit(x, y)
    root = m.get_booster().trees_to_dataframe().iloc[0]
    thr, gains = split_gains(x[:, 0], y)
    assert root.Gain == pytest.approx(gains.max(), rel=1e-4)
    assert root.Gain == pytest.approx(pearson_chi2_2x2(x[:, 0], y, root.Split), rel=1e-3)
