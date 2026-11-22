"""Data-generating processes for the robustness study (E9, paper Section 6).

Every generator returns ``X, y, roles`` with exactly ``n_min = round(n p)``
positives. ``roles`` labels each column:

  "rel"    relevant through its marginal distribution (location or scale)
  "inter"  relevant only jointly with another column (interaction)
  "proxy"  a noisy copy of a relevant column: marginally associated with y,
           conditionally independent of y given the relevant columns
  "null"   independent of y and of every other column

The baseline is the Gaussian location-shift model of Section 2.1 with 5
relevant and 5 null columns.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

DGPS = (
    "gauss",
    "corr_rel",
    "proxy",
    "t3",
    "lognormal",
    "mixed_binary",
    "scale",
    "interaction",
)


def _labels(n, p, rng):
    n_min = int(round(n * p))
    y = np.zeros(n, dtype=np.int8)
    y[:n_min] = 1
    return y, n_min


def _std_t3(rng, size):
    return rng.standard_t(3, size=size) / np.sqrt(3.0)


def _std_lognormal(rng, size):
    s = 1.0
    m, v = np.exp(s**2 / 2), (np.exp(s**2) - 1) * np.exp(s**2)
    return (rng.lognormal(0.0, s, size=size) - m) / np.sqrt(v)


def make(dgp, n, p, delta, seed, n_rel=5, n_null=5):
    rng = np.random.default_rng(seed)
    y, n_min = _labels(n, p, rng)
    pos = y == 1
    if dgp in ("gauss", "t3", "lognormal"):
        draw = {"gauss": rng.standard_normal, "t3": None, "lognormal": None}[dgp]
        size = (n, n_rel + n_null)
        if dgp == "gauss":
            X = draw(size)
        elif dgp == "t3":
            X = _std_t3(rng, size)
        else:
            X = _std_lognormal(rng, size)
        X[pos, :n_rel] += delta
        roles = ["rel"] * n_rel + ["null"] * n_null
    elif dgp == "corr_rel":
        rho = 0.6  # within-class correlation among the relevant columns
        F = rng.standard_normal((n, 1))
        E = rng.standard_normal((n, n_rel))
        R = np.sqrt(rho) * F + np.sqrt(1 - rho) * E
        R[pos] += delta
        X = np.hstack([R, rng.standard_normal((n, n_null))])
        roles = ["rel"] * n_rel + ["null"] * n_null
    elif dgp == "proxy":
        rho = 0.9  # correlation of each proxy with its relevant column
        R = rng.standard_normal((n, n_rel))
        R[pos] += delta
        P = rho * R + np.sqrt(1 - rho**2) * rng.standard_normal((n, n_rel))
        X = np.hstack([R, P, rng.standard_normal((n, n_null))])
        roles = ["rel"] * n_rel + ["proxy"] * n_rel + ["null"] * n_null
    elif dgp == "mixed_binary":
        # columns 0,2,4 relevant continuous; 1,3 relevant binary; nulls alternate
        # binary / continuous. Binary columns: P(x = 1 | y = 0) = 0.3, and
        # P(x = 1 | y = 1) chosen to give the same AUC as the continuous columns.
        auc = stats.norm.cdf(delta / np.sqrt(2))
        q0 = 0.3
        q1 = min(0.99, q0 + 2 * (auc - 0.5) / (1.0))  # AUC of a binary = 0.5 + (q1 - q0)/2
        X = rng.standard_normal((n, n_rel + n_null))
        X[pos, :n_rel] += delta
        for j in (1, 3):
            X[:, j] = np.where(pos, rng.random(n) < q1, rng.random(n) < q0)
        for j in (n_rel + 0, n_rel + 2, n_rel + 4):
            if j < n_rel + n_null:
                X[:, j] = rng.random(n) < q0
        roles = ["rel"] * n_rel + ["null"] * n_null
    elif dgp == "scale":
        X = rng.standard_normal((n, n_rel + n_null))
        X[pos, :n_rel] *= 1.6  # same mean, larger spread among positives
        roles = ["rel"] * n_rel + ["null"] * n_null
    elif dgp == "interaction":
        # two pairs that are marginally N(0,1) in both classes but correlated
        # (rho = 0.6) among positives only, plus one location-shift column
        X = rng.standard_normal((n, n_rel + n_null))
        rho = 0.6
        for a, b in ((0, 1), (2, 3)):
            z = rng.standard_normal(n)
            X[pos, b] = rho * X[pos, a] + np.sqrt(1 - rho**2) * z[pos]
        X[pos, 4] += delta
        roles = ["inter"] * 4 + ["rel"] + ["null"] * n_null
    else:
        raise ValueError(dgp)
    perm = rng.permutation(n)
    return X[perm].astype(float), y[perm], np.array(roles)


def population_auc(dgp, delta, seed=0, n=400_000):
    """Single-feature ROC AUC of each column in the large-sample limit (Monte Carlo)."""
    X, y, roles = make(dgp, n, 0.5, delta, seed)
    pos, neg = X[y == 1], X[y == 0]
    U = stats.mannwhitneyu(pos, neg, axis=0).statistic
    return U / (len(pos) * len(neg)), roles
