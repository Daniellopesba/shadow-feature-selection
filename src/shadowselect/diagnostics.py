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
    """Smallest standardised mean shift that wins the first split against the
    shadows with probability 0.9 (first-round theory; a lower bound)."""
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


def assess_imbalance(y, n_shadows: int = 10, power: float = 0.9, n_rep: int = 500, seed: int = 0):
    """Summarise how much a small minority count limits shadow-feature selection.

    ``min_detectable_delta`` is the smallest standardised mean shift for which
    an informative feature wins the *first* split of a gradient-boosted tree
    against ``n_shadows`` null features with probability ``power``. It uses
    the small-prevalence limit of the paper (Theorem 1) with XGBoost's default
    regularisation (lambda = 1, min_child_weight = 1) and an intercept at the
    prevalence, via :func:`shadowselect.theory.detectable_delta`. Full
    300-round ensembles need larger effects than this (paper, Section 4), so
    read it as a lower bound on what is detectable.
    """
    from .theory import detectable_delta

    y = np.asarray(y)
    n = len(y)
    n_min = minority_count(y)
    mdd = detectable_delta(n_min, n_null=n_shadows, power=power, n_rep=n_rep, seed=seed)
    return ImbalanceReport(n, n_min, n_min / n, mdd, _recommendation(n_min))
