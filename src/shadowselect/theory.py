"""Closed-form and Monte Carlo tools for the paper's first-split analysis.

Notation (paper, Section 3). At the first boosting round with log-loss and
the intercept set to the empirical log-odds, every sample has gradient
``g_i = p_hat - y_i`` with ``p_hat = n_min / n`` and Hessian
``h = p_hat (1 - p_hat)``. XGBoost scores a candidate split ``(L, R)`` by

    Gain = 1/2 [ G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam) ] - gamma.

**Proposition (gain = Pearson chi-square).** With ``lam = gamma = 0``,
``2 * Gain`` equals Pearson's chi-square statistic for the 2x2 table
``{L, R} x {y=0, y=1}``. Hence, for an uninformative feature and a fixed
threshold, ``2 * Gain`` has exact mean ``n / (n - 1)`` and is asymptotically
chi-square(1) when every expected cell count is large.

**Proposition (signal non-centrality).** For a feature whose positives are
shifted by ``delta``, the split at threshold ``t`` has
``2 * Gain ~ noncentral chi-square(1, nc)`` asymptotically with

    nc(t) = n p (1-p) (Phi(t) - Phi(t - delta))^2 / (q (1 - q)),
    q = p Phi(t - delta) + (1 - p) Phi(t).

For small ``delta`` and small ``p`` the best threshold is near the majority
median and ``nc* ~= (2/pi) n_min delta^2``. The informative feature's
advantage over noise therefore grows with ``n_min delta^2`` while the null
maximum grows only logarithmically with the number of distinct thresholds.
"""

from __future__ import annotations

import numpy as np
from scipy import optimize, stats

__all__ = [
    "split_gains",
    "pearson_chi2_2x2",
    "signal_noncentrality",
    "best_signal_noncentrality",
    "max_null_gain",
    "min_detectable_delta",
]


def split_gains(
    x: np.ndarray,
    y: np.ndarray,
    lam: float = 0.0,
    min_child_weight: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """XGBoost first-round split gains (without the 1/2 factor) for every threshold.

    Returns ``(thresholds, two_gain)`` where ``two_gain[k]`` is
    ``G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam)`` for the split
    ``x <= thresholds[k]``. Ties in ``x`` are respected. Splits violating
    ``min_child_weight`` (on Hessian sums) are dropped.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = len(xs)
    p = ys.mean()
    g = p - ys
    h = p * (1 - p)
    G, H = g.sum(), h * n
    # candidate split after position k only where the next value differs
    valid = np.nonzero(xs[1:] > xs[:-1])[0]
    GL = np.cumsum(g)[valid]
    HL = h * (valid + 1)
    GR, HR = G - GL, H - HL
    ok = (HL >= min_child_weight) & (HR >= min_child_weight)
    two_gain = GL**2 / (HL + lam) + GR**2 / (HR + lam) - G**2 / (H + lam)
    thr = 0.5 * (xs[valid] + xs[valid + 1])
    return thr[ok], two_gain[ok]


def pearson_chi2_2x2(x: np.ndarray, y: np.ndarray, threshold: float) -> float:
    """Pearson chi-square (no continuity correction) for ``x <= threshold`` vs ``y``."""
    x = np.asarray(x)
    y = np.asarray(y).astype(bool)
    left = x <= threshold
    a, b = np.sum(left & y), np.sum(left & ~y)
    c, d = np.sum(~left & y), np.sum(~left & ~y)
    n = a + b + c + d
    denom = (a + b) * (c + d) * (a + c) * (b + d)
    return float(n * (a * d - b * c) ** 2 / denom) if denom > 0 else 0.0


def signal_noncentrality(n: int, p: float, delta: float, t: float | np.ndarray) -> np.ndarray:
    """Non-centrality ``n * phi^2`` of the split ``x <= t`` for a shifted feature."""
    t = np.asarray(t, dtype=float)
    a = stats.norm.cdf(t - delta)  # P(left | y = 1)
    b = stats.norm.cdf(t)  # P(left | y = 0)
    q = p * a + (1 - p) * b
    return n * p * (1 - p) * (a - b) ** 2 / (q * (1 - q))


def best_signal_noncentrality(n: int, p: float, delta: float) -> tuple[float, float]:
    """Maximise :func:`signal_noncentrality` over ``t``. Returns ``(nc*, t*)``."""
    if delta == 0:
        return 0.0, 0.0
    res = optimize.minimize_scalar(
        lambda t: -signal_noncentrality(n, p, delta, t),
        bounds=(-6.0, 6.0 + abs(delta)),
        method="bounded",
    )
    return float(-res.fun), float(res.x)


def max_null_gain(
    n: int,
    n_min: int,
    n_rep: int = 1000,
    lam: float = 0.0,
    min_child_weight: float = 0.0,
    seed: int | None = 0,
) -> np.ndarray:
    """Monte Carlo draws of ``max_t 2*Gain(t)`` for one uninformative feature.

    Under the null only the rank pattern of the positives matters, so each
    draw places ``n_min`` positives uniformly at random among ``n`` ordered
    positions.
    """
    rng = np.random.default_rng(seed)
    p = n_min / n
    h = p * (1 - p)
    idx = np.arange(1, n)
    HL = h * idx
    HR = h * n - HL
    ok = (HL >= min_child_weight) & (HR >= min_child_weight)
    out = np.empty(n_rep)
    for r in range(n_rep):
        y = np.zeros(n)
        y[rng.choice(n, size=n_min, replace=False)] = 1.0
        GL = np.cumsum(p - y)[:-1]
        two_gain = GL**2 / (HL + lam) + GL**2 / (HR + lam)  # G = 0
        out[r] = two_gain[ok].max() if ok.any() else 0.0
    return out


def min_detectable_delta(n_min: int, null_max: float, p: float = 0.0) -> float:
    """Smallest shift whose expected best first split beats ``null_max``.

    Solves ``1 + nc* = null_max`` with ``nc* = (2/pi) n_min (1-p) delta^2``
    (small-``delta`` approximation). ``null_max`` is a quantile of the null
    distribution of ``max_t 2*Gain(t)``, e.g. from :func:`max_null_gain`.
    """
    if n_min <= 0:
        return float("inf")
    excess = max(null_max - 1.0, 0.0)
    return float(np.sqrt(excess / ((2 / np.pi) * n_min * (1 - p))))
