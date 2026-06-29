import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shadow_lib import (
    aggregate_repeats,
    compute_metrics,
    derive_seed,
    make_dataset,
)


def test_derive_seed_is_deterministic():
    a = derive_seed("abc123", 0)
    b = derive_seed("abc123", 0)
    assert a == b
    assert 0 <= a < 2**32


def test_derive_seed_differs_by_repeat():
    seeds = {derive_seed("abc123", i) for i in range(20)}
    assert len(seeds) == 20


def test_derive_seed_differs_by_scenario_id():
    a = derive_seed("abc123", 0)
    b = derive_seed("def456", 0)
    assert a != b


def test_make_dataset_shape_and_counts():
    X, y = make_dataset(n=1000, p=0.1, delta=0.5, n_real=5, n_noise=5, seed=42)
    assert X.shape == (1000, 10)
    assert y.shape == (1000,)
    assert y.sum() == 100


def test_make_dataset_exact_minority_count_rounding():
    X, y = make_dataset(n=1000, p=0.001, delta=0.5, n_real=5, n_noise=5, seed=0)
    assert y.sum() == 1


def test_make_dataset_real_features_shifted_in_minority_class():
    X, y = make_dataset(n=20000, p=0.5, delta=1.0, n_real=5, n_noise=5, seed=7)
    pos_means = X[y == 1, :5].mean(axis=0)
    neg_means = X[y == 0, :5].mean(axis=0)
    assert np.allclose(pos_means - neg_means, 1.0, atol=0.05)


def test_make_dataset_noise_features_independent_of_y():
    X, y = make_dataset(n=20000, p=0.5, delta=1.0, n_real=5, n_noise=5, seed=7)
    pos_noise = X[y == 1, 5:].mean(axis=0)
    neg_noise = X[y == 0, 5:].mean(axis=0)
    assert np.allclose(pos_noise - neg_noise, 0.0, atol=0.05)


def test_make_dataset_control_no_signal():
    X, y = make_dataset(n=20000, p=0.5, delta=0.0, n_real=5, n_noise=5, seed=7)
    pos_means = X[y == 1, :5].mean(axis=0)
    neg_means = X[y == 0, :5].mean(axis=0)
    assert np.allclose(pos_means - neg_means, 0.0, atol=0.05)


def test_make_dataset_determinism():
    X1, y1 = make_dataset(n=500, p=0.1, delta=0.5, n_real=5, n_noise=5, seed=42)
    X2, y2 = make_dataset(n=500, p=0.1, delta=0.5, n_real=5, n_noise=5, seed=42)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)


def test_compute_metrics_naive_floor_is_max_of_noise():
    importances = np.array(
        [0.30, 0.25, 0.10, 0.05, 0.04, 0.10, 0.08, 0.04, 0.02, 0.02]
    )
    m = compute_metrics(importances, n_real=5, n_noise=5)
    assert m["naive_floor"] == 0.10


def test_compute_metrics_p90_floor_is_p90_of_noise():
    rng = np.random.default_rng(0)
    real = rng.uniform(0.1, 0.2, size=5)
    noise = np.linspace(0.01, 0.10, 10)
    importances = np.concatenate([real, noise])
    importances = importances / importances.sum()
    m = compute_metrics(importances, n_real=5, n_noise=10)
    noise_norm = importances[5:]
    assert np.isclose(m["p90_floor"], np.percentile(noise_norm, 90))


def test_compute_metrics_fnr_max_counts_real_below_floor():
    importances = np.array(
        [0.30, 0.25, 0.03, 0.02, 0.01, 0.05, 0.04, 0.03, 0.02, 0.02]
    )
    m = compute_metrics(importances, n_real=5, n_noise=5)
    assert np.isclose(m["fnr_max"], 0.6)


def test_compute_metrics_fnr_p90_uses_p90_floor():
    importances = np.array(
        [0.30, 0.25, 0.10, 0.05, 0.04, 0.10, 0.08, 0.04, 0.02, 0.02]
    )
    importances = importances / importances.sum()
    m = compute_metrics(importances, n_real=5, n_noise=5)
    real = importances[:5]
    noise = importances[5:]
    p90 = np.percentile(noise, 90)
    expected_fnr = float((real < p90).mean())
    assert np.isclose(m["fnr_p90"], expected_fnr)


def test_compute_metrics_mean_real_importance():
    importances = np.array(
        [0.30, 0.25, 0.10, 0.05, 0.04, 0.10, 0.08, 0.04, 0.02, 0.02]
    )
    importances = importances / importances.sum()
    m = compute_metrics(importances, n_real=5, n_noise=5)
    assert np.isclose(m["mean_real_importance"], importances[:5].mean())


def test_aggregate_repeats_basic():
    per_repeat = {
        "fnr_max": [0.0, 0.2, 0.4, 0.6, 0.8],
        "naive_floor": [0.05, 0.05, 0.05, 0.05, 0.05],
    }
    agg = aggregate_repeats(per_repeat)
    assert np.isclose(agg["fnr_max"]["mean"], 0.4)
    assert np.isclose(agg["fnr_max"]["median"], 0.4)
    assert agg["naive_floor"]["std"] == 0.0


def test_pipeline_smoke_xgboost():
    from xgboost import XGBClassifier

    X, y = make_dataset(n=2000, p=0.5, delta=1.0, n_real=5, n_noise=5, seed=42)
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        tree_method="hist",
        importance_type="gain",
        n_jobs=1,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X, y)
    m = compute_metrics(clf.feature_importances_, n_real=5, n_noise=5)
    assert m["mean_real_importance"] > m["naive_floor"]
    assert m["fnr_max"] == 0.0
