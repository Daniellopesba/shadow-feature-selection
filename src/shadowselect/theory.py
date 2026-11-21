"""Closed-form and Monte Carlo tools for the paper's first-round analysis.

Conventions (paper, Section 3). "Gain" is what XGBoost computes and reports,

    Gain = G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam),

(the XGBoost paper writes half of this; the software does not). In the first
tree every sample has gradient ``g_i = b - y_i`` and Hessian ``b (1 - b)``,
where ``b`` is the base score. Functions below return this quantity; some
variable names still say ``two_gain`` for historical reasons.

* Proposition 1: with ``lam = 0``, Gain equals Pearson's chi-square of the
  2x2 split-by-class table times ``p (1-p) / (b (1-b))``.
* Proposition 2: for a null feature at a fixed split, ``E[Gain] = n / (n-1)``
  (prevalence intercept, ``lam = 0``).
* Proposition 4: non-centrality ``n p (1-p) (Phi(t) - Phi(t-delta))^2 / (q (1-q))``;
  for small ``delta`` (about 0.3 or less) and small ``p`` its maximum is about
  ``(2/pi) n_min delta^2``.
* Theorem 1: as ``n -> inf`` with ``n_min`` fixed (prevalence intercept), the
  gain process converges to ``Gamma_delta(u)``; see :func:`limit_max_gain`.
* Proposition 5: under a wrong intercept every split pays a toll of about
  ``lam * r^2`` with ``r = (b - p) / (b (1 - b))``.

The null maximum grows slowly with ``n_min`` (roughly like ``2 log log n_min``),
so detectability is governed to first order by ``n_min * delta^2``.
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
    "limit_max_gain",
    "detection_probability",
    "detectable_delta",
]


def split_gains(
    x: np.ndarray,
    y: np.ndarray,
    lam: float = 0.0,
    min_child_weight: float = 0.0,
    base_score: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """XGBoost first-round split gains (as reported by XGBoost) for every threshold.

    Returns ``(thresholds, two_gain)`` where ``two_gain[k]`` is
    ``G_L^2/(H_L+lam) + G_R^2/(H_R+lam) - G^2/(H+lam)`` for the split
    ``x <= thresholds[k]``. Ties in ``x`` are respected. Splits violating
    ``min_child_weight`` (on Hessian sums) are dropped. ``base_score`` is the
    initial predicted probability; ``None`` uses the prevalence of ``y``.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    order = np.argsort(x, kind="stable")
    xs, ys = x[order], y[order]
    n = len(xs)
    b = ys.mean() if base_score is None else float(base_score)
    g = b - ys
    h = b * (1 - b)
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
    base_score: float | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """Monte Carlo draws of ``max_t 2*Gain(t)`` for one uninformative feature.

    Under the null only the rank pattern of the positives matters, so each
    draw places ``n_min`` positives uniformly at random among ``n`` ordered
    positions. ``base_score`` is the initial predicted probability (``None``
    = prevalence ``n_min / n``, which makes ``G = 0``).
    """
    rng = np.random.default_rng(seed)
    b = n_min / n if base_score is None else float(base_score)
    h = b * (1 - b)
    idx = np.arange(1, n)
    HL = h * idx
    HR = h * n - HL
    H = h * n
    ok = (HL >= min_child_weight) & (HR >= min_child_weight)
    out = np.empty(n_rep)
    for r in range(n_rep):
        y = np.zeros(n)
        y[rng.choice(n, size=n_min, replace=False)] = 1.0
        g = b - y
        G = g.sum()
        GL = np.cumsum(g)[:-1]
        two_gain = GL**2 / (HL + lam) + (G - GL) ** 2 / (HR + lam) - G**2 / (H + lam)
        out[r] = two_gain[ok].max() if ok.any() else 0.0
    return out


def limit_max_gain(
    n_min: int,
    delta: float = 0.0,
    n_rep: int = 2000,
    lam: float = 1.0,
    min_child_weight: float = 1.0,
    n_bins: int | None = None,
    seed: int | None = 0,
) -> np.ndarray:
    """Draws of ``sup_u Gamma_delta(u)``, the small-prevalence limit of the best
    first-round split (``2*Gain`` units, intercept at the prevalence).

    ``Gamma_delta(u) = (N(u) - n_min u)^2 [1/(n_min u + lam) + 1/(n_min (1-u) + lam)]``
    on ``u in [w/n_min, 1 - w/n_min]``, where ``N(u)`` counts the positives with
    ``Phi(Z_i) <= u`` and ``Z_i ~ N(delta, 1)``. ``n_bins=None`` takes the
    supremum over all thresholds (exact greedy); an integer restricts ``u`` to
    the ``n_bins - 1`` interior quantile cuts, mimicking ``tree_method="hist"``.
    """
    if n_min <= 2 * min_child_weight:
        raise ValueError("need n_min > 2 * min_child_weight")
    rng = np.random.default_rng(seed)
    m = n_min
    a = min_child_weight / m
    if n_bins is not None:
        grid = np.arange(1, n_bins) / n_bins
        grid = grid[(grid >= a) & (grid <= 1 - a)]
    out = np.empty(n_rep)
    for r in range(n_rep):
        u_pos = np.sort(stats.norm.cdf(rng.standard_normal(m) + delta))
        if n_bins is None:
            # sup is attained at a boundary or as a one-sided limit at a jump
            eps = 1e-12
            cand = np.concatenate([u_pos - eps, u_pos, [a, 1 - a]])
            cand = cand[(cand >= a) & (cand <= 1 - a)]
        else:
            cand = grid
        k = np.searchsorted(u_pos, cand, side="right")
        val = (k - m * cand) ** 2 * (1 / (m * cand + lam) + 1 / (m * (1 - cand) + lam))
        out[r] = val.max() if len(val) else 0.0
    return out


def detection_probability(
    n_min: int,
    delta: float,
    n_null: int = 10,
    n_rep: int = 2000,
    lam: float = 1.0,
    min_child_weight: float = 1.0,
    n_bins: int | None = None,
    seed: int | None = 0,
) -> float:
    """P(an informative feature's best first split beats the best of ``n_null``
    null features), in the small-prevalence limit."""
    kw = dict(lam=lam, min_child_weight=min_child_weight, n_bins=n_bins)
    sig = limit_max_gain(n_min, delta, n_rep=n_rep, seed=seed, **kw)
    null = limit_max_gain(
        n_min, 0.0, n_rep=n_rep * n_null, seed=None if seed is None else seed + 1, **kw
    )
    null = null.reshape(n_rep, n_null).max(axis=1)
    return float(np.mean(sig > null))


def detectable_delta(
    n_min: int,
    n_null: int = 10,
    power: float = 0.9,
    n_rep: int = 1000,
    seed: int | None = 0,
    **kw,
) -> float:
    """Smallest ``delta`` (grid search) whose first-round detection probability
    against ``n_null`` null features reaches ``power``."""
    if n_min <= 2 * kw.get("min_child_weight", 1.0):
        return float("inf")
    grid = np.round(np.arange(0.05, 3.0001, 0.05), 2)
    lo, hi = 0, len(grid) - 1
    if detection_probability(n_min, grid[hi], n_null, n_rep, seed=seed, **kw) < power:
        return float("inf")
    while lo < hi:  # detection probability is increasing in delta
        mid = (lo + hi) // 2
        if detection_probability(n_min, grid[mid], n_null, n_rep, seed=seed, **kw) >= power:
            hi = mid
        else:
            lo = mid + 1
    return float(grid[lo])


def min_detectable_delta(n_min: int, null_max: float, p: float = 0.0) -> float:
    """Closed-form rough guide: shift whose non-centrality reaches ``null_max``.

    Optimistic-to-conservative rule of thumb only; it compares a mean with a
    quantile and ignores that the informative feature is also maximised over
    thresholds. Prefer :func:`detectable_delta`.

    Solves ``1 + nc* = null_max`` with ``nc* = (2/pi) n_min (1-p) delta^2``
    (small-``delta`` approximation). ``null_max`` is a quantile of the null
    distribution of ``max_t 2*Gain(t)``, e.g. from :func:`max_null_gain`.
    """
    if n_min <= 0:
        return float("inf")
    excess = max(null_max - 1.0, 0.0)
    return float(np.sqrt(excess / ((2 / np.pi) * n_min * (1 - p))))
