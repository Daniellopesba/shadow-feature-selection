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
