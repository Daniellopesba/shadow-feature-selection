"""Synthetic data and evaluation metrics used by the paper's experiments.

The data model (Section 2 of the paper):

* ``n`` samples, of which exactly ``n_min = round(n * p)`` are positives.
* ``n_real`` informative features: ``X_j ~ N(0, 1)`` for negatives and
  ``X_j ~ N(delta, 1)`` for positives (a location shift of ``delta``
  standard deviations).
* ``n_noise`` uninformative features: ``X_j ~ N(0, 1)`` independent of ``y``.

Columns are ordered ``[real..., noise...]``.
"""

from __future__ import annotations

import hashlib

import numpy as np

__all__ = [
    "derive_seed",
    "make_dataset",
    "selection_metrics",
]


def derive_seed(key: str, repeat_idx: int) -> int:
    """Deterministic 32-bit seed from a string key and a repeat index."""
    h = hashlib.md5(f"{key}:{repeat_idx}".encode()).hexdigest()
    return int(h[:8], 16)


def make_dataset(
    n: int,
    p: float,
    delta: float,
    n_real: int,
    n_noise: int,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gaussian location-shift dataset with an exact minority count.

    Parameters
    ----------
    n : int
        Number of samples.
    p : float
        Minority proportion. The number of positives is ``round(n * p)``.
    delta : float
        Mean shift of every informative feature in the positive class, in
        units of the (unit) within-class standard deviation.
    n_real, n_noise : int
        Number of informative and uninformative features.
    seed : int, Generator or None
        Random state.

    Returns
    -------
    X : ndarray of shape (n, n_real + n_noise)
    y : ndarray of shape (n,), dtype int8
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    if n <= 0 or n_real < 0 or n_noise < 0:
        raise ValueError("n must be positive and feature counts non-negative")
    rng = np.random.default_rng(seed)
    n_min = int(round(n * p))

    X = rng.standard_normal((n, n_real + n_noise))
    y = np.zeros(n, dtype=np.int8)
    y[:n_min] = 1
    if delta != 0.0 and n_min > 0:
        X[:n_min, :n_real] += delta
    perm = rng.permutation(n)
    return X[perm], y[perm]


def selection_metrics(
    importances: np.ndarray,
    n_real: int,
    n_noise: int,
    quantiles: tuple[float, ...] = (1.0, 0.9),
) -> dict[str, float]:
    """Single-shot shadow-threshold metrics for one fitted model.

    ``importances`` lists the real features first, then the noise
    features. They are normalised to sum to one (the convention of
    ``feature_importances_`` in scikit-learn and XGBoost).

    For each quantile ``q`` the threshold is the ``q``-quantile of the noise
    importances (``q = 1`` is the classical max-shadow rule) and the false
    negative rate is the fraction of real features strictly below it.

    Returns keys ``floor_q{100q}``, ``fnr_q{100q}`` for each quantile, plus
    ``mean_real_importance`` and ``mean_noise_importance``.
    """
    imp = np.asarray(importances, dtype=float)
    if imp.shape != (n_real + n_noise,):
        raise ValueError(f"expected {n_real + n_noise} importances, got shape {imp.shape}")
    if n_real == 0 or n_noise == 0:
        raise ValueError("need at least one real and one noise feature")
    total = imp.sum()
    imp = imp / total if total > 0 else imp
    real, noise = imp[:n_real], imp[n_real:]
    out: dict[str, float] = {
        "mean_real_importance": float(real.mean()),
        "mean_noise_importance": float(noise.mean()),
    }
    for q in quantiles:
        tag = f"q{int(round(100 * q))}"
        floor = float(np.quantile(noise, q))
        out[f"floor_{tag}"] = floor
        out[f"fnr_{tag}"] = float((real < floor).mean())
    return out
