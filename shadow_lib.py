"""Numeric core for the shadow-feature experiment.

Pure functions, importable from both the parameterized notebook and the
unit test suite. No I/O lives here.
"""
from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(scenario_id: str, repeat_idx: int) -> int:
    """Deterministic 32-bit seed from scenario id and repeat index."""
    h = hashlib.md5(f"{scenario_id}:{repeat_idx}".encode()).hexdigest()
    return int(h[:8], 16)


def make_dataset(
    n: int,
    p: float,
    delta: float,
    n_real: int,
    n_noise: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic Gaussian dataset with exact minority count.

    Real features (first n_real columns) are shifted by +delta in the minority
    class. Noise features (next n_noise columns) are independent of y.
    """
    rng = np.random.default_rng(seed)
    n_min = int(round(n * p))
    if n_min < 0 or n_min > n:
        raise ValueError(f"n_min={n_min} out of range for n={n}, p={p}")

    n_features = n_real + n_noise
    X = rng.standard_normal((n, n_features))
    y = np.zeros(n, dtype=np.int8)
    y[:n_min] = 1
    if delta != 0.0 and n_min > 0:
        X[:n_min, :n_real] += delta
    perm = rng.permutation(n)
    return X[perm], y[perm]


def compute_metrics(
    importances: np.ndarray, n_real: int, n_noise: int
) -> dict[str, float]:
    """Per-repeat metrics from normalized gain importances.

    Convention: the first n_real entries are real features, the next n_noise
    are noise. Importances are normalized here defensively.
    """
    importances = np.asarray(importances, dtype=float)
    assert importances.shape == (n_real + n_noise,)
    total = importances.sum()
    normed = importances / total if total > 0 else importances
    real = normed[:n_real]
    noise = normed[n_real : n_real + n_noise]
    naive_floor = float(noise.max())
    p90_floor = float(np.percentile(noise, 90))
    return {
        "naive_floor": naive_floor,
        "p90_floor": p90_floor,
        "mean_real_importance": float(real.mean()),
        "fnr_max": float((real < naive_floor).mean()),
        "fnr_p90": float((real < p90_floor).mean()),
    }


def aggregate_repeats(
    per_repeat: dict[str, list[float]],
) -> dict[str, dict[str, float]]:
    """Mean/median/p90/std per metric across repeats."""
    out: dict[str, dict[str, float]] = {}
    for name, values in per_repeat.items():
        arr = np.asarray(values, dtype=float)
        out[name] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }
    return out
