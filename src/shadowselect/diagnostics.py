"""Pre-flight diagnostics: is shadow-feature selection trustworthy on this target?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["ImbalanceWarning", "minority_count", "ImbalanceReport", "assess_imbalance"]


class ImbalanceWarning(UserWarning):
    """Shadow-feature selection is unlikely to be reliable at this minority count."""


def minority_count(y) -> int:
    """Number of samples in the smallest class of ``y``."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    if len(counts) < 2:
        raise ValueError("y must contain at least two classes")
    return int(counts.min())


@dataclass(frozen=True)
class ImbalanceReport:
    n: int
    n_minority: int
    minority_fraction: float
    min_detectable_delta: float
    """Smallest standardized mean shift whose best first split beats the
    95th percentile of the null max (first-round theory, Section 3)."""
    recommendation: str

    def __str__(self) -> str:
        return (
            f"n={self.n}, n_minority={self.n_minority} "
            f"(p={self.minority_fraction:.4f}); first-split minimum detectable "
            f"shift ~ {self.min_detectable_delta:.2f} SD. {self.recommendation}"
        )


def _recommendation(n_min: int) -> str:
    if n_min < 50:
        return (
            "Unreliable: do not prune on training-set gain vs. shadows. Use permutation "
            "importance on held-out data with many rounds, keep tentative features, or "
            "collect more minority samples."
        )
    if n_min < 200:
        return (
            "Marginal: use ShadowFeatureSelector(importance='permutation') with n_iter >= 20 "
            "and treat 'tentative' features as kept."
        )
    if n_min < 1000:
        return "Usually fine for moderate effects; prefer held-out permutation importance."
    return "Shadow-feature selection is expected to work for all but very weak features."


def assess_imbalance(y, null_quantile: float = 0.95, n_rep: int = 500, seed: int = 0):
    """Summarise how much a small minority count limits shadow-feature selection.

    The minimum detectable shift uses :func:`shadowselect.theory.max_null_gain`
    (Monte Carlo of the null maximum of the first-split chi-square) and the
    small-effect non-centrality ``(2/pi) n_min (1-p) delta^2``. It is a
    first-round, single-split heuristic: full boosting ensembles need larger
    effects (see the paper, Section 4).
    """
    from .theory import max_null_gain, min_detectable_delta

    y = np.asarray(y)
    n = len(y)
    n_min = minority_count(y)
    p = n_min / n
    # the null maximum depends on n mostly through n_min; cap n for speed
    n_sim = min(n, 20000)
    k_sim = max(1, int(round(n_sim * p)))
    null = max_null_gain(n_sim, k_sim, n_rep=n_rep, min_child_weight=1.0, seed=seed)
    q = float(np.quantile(null, null_quantile))
    mdd = min_detectable_delta(n_min, q, p)
    return ImbalanceReport(n, n_min, p, mdd, _recommendation(n_min))
