import numpy as np
import pytest

from shadowselect.simulation import derive_seed, make_dataset, selection_metrics


def test_exact_minority_count_and_shape():
    X, y = make_dataset(n=1000, p=0.013, delta=0.5, n_real=3, n_noise=4, seed=1)
    assert X.shape == (1000, 7)
    assert y.sum() == 13


def test_shift_only_on_real_features():
    X, y = make_dataset(n=200000, p=0.5, delta=1.0, n_real=2, n_noise=2, seed=0)
    diff = X[y == 1].mean(0) - X[y == 0].mean(0)
    np.testing.assert_allclose(diff, [1, 1, 0, 0], atol=0.03)


def test_deterministic():
    a = make_dataset(100, 0.1, 0.5, 2, 2, seed=7)
    b = make_dataset(100, 0.1, 0.5, 2, 2, seed=7)
    np.testing.assert_array_equal(a[0], b[0])
    np.testing.assert_array_equal(a[1], b[1])


def test_invalid_p():
    with pytest.raises(ValueError):
        make_dataset(100, 1.5, 0.5, 2, 2, seed=0)


def test_derive_seed():
    assert derive_seed("a", 0) == derive_seed("a", 0)
    assert len({derive_seed("a", i) for i in range(50)}) == 50


def test_selection_metrics():
    imp = np.array([0.3, 0.05, 0.2, 0.1, 0.15, 0.2])  # 3 real, 3 noise
    m = selection_metrics(imp, 3, 3)
    assert m["floor_q100"] == pytest.approx(0.2)
    assert m["fnr_q100"] == pytest.approx(1 / 3)  # 0.05 < 0.2; 0.2 is not < 0.2
    assert m["fnr_q90"] == pytest.approx(1 / 3)


def test_selection_metrics_shape_check():
    with pytest.raises(ValueError):
        selection_metrics(np.ones(5), 3, 3)
