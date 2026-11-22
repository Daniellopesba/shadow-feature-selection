"""Pre-flight diagnostics: is shadow-feature selection trustworthy on this target?"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "ImbalanceWarning",
    "LearnerWarning",
    "minority_count",
    "null_keep_rate",
    "ImbalanceReport",
    "assess_imbalance",
]


class ImbalanceWarning(UserWarning):
    """Shadow-feature selection is unlikely to be reliable at this minority count."""


class LearnerWarning(UserWarning):
    """The learner gives shadow features as much importance as strong real ones."""


def null_keep_rate(
    n_features: int,
    n_iter: int,
    hit_rate: float,
    alpha: float = 0.05,
    p0: float = 0.5,
    keep_tentative: bool = True,
) -> float:
    """Probability that an irrelevant feature is kept by the selector.

    Assumes the feature beats the shadow threshold independently in each
    round with probability ``hit_rate`` (``1 / (K + 1)`` for ``K`` shadows and
    the max-shadow rule; the paper's level of the max-shadow rule) and Bonferroni-corrected
    binomial tests against ``p0`` at level ``alpha``. With
    ``keep_tentative=True`` the feature is kept unless rejected; otherwise
    only if accepted. Benjamini-Hochberg rejects at least as many features,
    so with ``keep_tentative=True`` this is an upper bound for it.

    The rate depends on the number of features: more features mean a
    stricter per-feature level and fewer rejections.

    >>> round(null_keep_rate(10, 20, 1 / 11), 3)
    0.102
    >>> round(null_keep_rate(200, 20, 1 / 11), 3)
    0.272
    """
    from scipy import stats

    level = alpha / n_features
    h = np.arange(n_iter + 1)
    if keep_tentative:
        rejected = stats.binom.cdf(h, n_iter, p0) < level
        r = h[rejected].max() if rejected.any() else -1
        return float(stats.binom.sf(r, n_iter, hit_rate))
    accepted = stats.binom.sf(h - 1, n_iter, p0) < level
    if not accepted.any():
        return 0.0
    a = h[accepted].min()
    return float(stats.binom.sf(a - 1, n_iter, hit_rate))


def minority_count(y) -> int:
    """Number of samples in the smallest class of ``y``."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    if len(counts) < 2:
        raise ValueError("y must contain at least two classes")
    return int(counts.min())


@dataclass(frozen=True)
class ImbalanceReport:
    """What a target with this many positives can support.

    Attributes
    ----------
    n, n_minority, minority_fraction
        Rows, rows in the smallest class, and their ratio.
    min_detectable_delta
        Smallest standardised mean shift that wins the *first* split against
        ``n_shadows`` shadows with probability ``power`` (first-round theory;
        a lower bound for full ensembles).
    reliable_delta
        Shift at which ``n_min * delta^2 = 25``, the paper's rule of thumb for
        reliable detection by repeated rounds with 5-10 shadows (the paper's guidance).
    n_shadows
        Shadow count the numbers refer to.
    null_keep_rate
        Expected fraction of irrelevant features the selector keeps (with
        ``keep_tentative=True``), when ``n_features`` is given; else ``None``.
    recommendation
        One-paragraph advice for this band of ``n_minority``.
    """

    n: int
    n_minority: int
    minority_fraction: float
    min_detectable_delta: float
    recommendation: str
    reliable_delta: float = float("nan")
    n_shadows: int = 10
    null_keep_rate: float | None = None

    def __str__(self) -> str:
        out = (
            f"n={self.n}, n_minority={self.n_minority} (p={self.minority_fraction:.4f}). "
            f"Reliably detectable: shifts of about {self.reliable_delta:.2f} SD or more "
            f"(n_min * delta^2 >= 25); first-split lower bound {self.min_detectable_delta:.2f} SD "
            f"with {self.n_shadows} shadows."
        )
        if self.null_keep_rate is not None:
            out += (
                f" Expect about {100 * self.null_keep_rate:.0f}% of irrelevant features to be "
                "kept as tentative."
            )
        return f"{out} {self.recommendation}"


def _recommendation(n_min: int) -> str:
    # Bands follow the paper's practical guidance.
    if n_min < 25:
        return (
            "Unreliable: no importance-threshold method recovered moderate effects at this "
            "count. Use UnivariateScreen, keep features on domain grounds, or collect more "
            "positives; do not prune on importance."
        )
    if n_min < 100:
        return (
            "Marginal: screen with UnivariateScreen, and use the selector with "
            "keep_tentative=True; features with n_min * delta^2 below about 25 are usually "
            "missed, so read rejections with summary()['s_hat']."
        )
    if n_min < 500:
        return (
            "Usually fine for moderate effects (n_min * delta^2 >= 25) with a learner that does "
            "not memorise the minority class (shallow trees, prevalence intercept)."
        )
    return (
        "Shadow-feature selection is expected to work for all but very weak features, "
        "provided the learner does not memorise the minority class."
    )


def assess_imbalance(
    y,
    n_shadows: int = 10,
    n_features: int | None = None,
    n_iter: int = 20,
    alpha: float = 0.05,
    power: float = 0.9,
    n_rep: int = 500,
    seed: int = 0,
) -> ImbalanceReport:
    """Summarise how much a small minority count limits shadow-feature selection.

    ``min_detectable_delta`` is the smallest standardised mean shift for which
    an informative feature wins the *first* split of a gradient-boosted tree
    against ``n_shadows`` null features with probability ``power``. It uses
    the paper's small-prevalence limit theorem with XGBoost's default
    regularisation (lambda = 1, min_child_weight = 1) and an intercept at the
    prevalence, via :func:`shadowselect.theory.detectable_delta`. Full
    300-round ensembles need larger effects than this (see the paper), so
    read it as a lower bound; ``reliable_delta`` is the practical guide.

    Pass ``n_features`` (and the selector's ``n_iter`` and ``alpha``) to get
    the expected share of irrelevant features kept, which grows with the
    number of features.

    The report depends only on the labels. It cannot see whether the learner
    overfits the minority class; :class:`ShadowFeatureSelector` checks that
    after fitting (:class:`LearnerWarning`).
    """
    from .theory import detectable_delta

    y = np.asarray(y)
    n = len(y)
    n_min = minority_count(y)
    mdd = detectable_delta(n_min, n_null=n_shadows, power=power, n_rep=n_rep, seed=seed)
    keep = None
    if n_features is not None:
        keep = null_keep_rate(n_features, n_iter, 1.0 / (n_shadows + 1), alpha=alpha)
    return ImbalanceReport(
        n=n,
        n_minority=n_min,
        minority_fraction=n_min / n,
        min_detectable_delta=mdd,
        recommendation=_recommendation(n_min),
        reliable_delta=float(np.sqrt(25.0 / n_min)),
        n_shadows=n_shadows,
        null_keep_rate=keep,
    )
