"""Univariate screening: a per-feature rank test with false-discovery control.

At small minority counts the shadow rule asks a tree model to rank features
from a few dozen positive rows, and it throws away most of the information
in doing so. A two-sample rank test of each feature between the classes uses
every row, has a known null distribution, and controls false discoveries
directly. In the paper's experiments (E6) it found more real features than any
tree-shadow method at the same false-positive rate when ``n_min <= 50``.

It only sees marginal shifts. A feature that matters only through an
interaction, or only in the tails, is invisible to it, so treat it as a
complement to model-based selection, not a replacement.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted, check_X_y

__all__ = ["UnivariateScreen", "benjamini_hochberg"]


def benjamini_hochberg(pvalues) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (step-up, monotone, capped at 1)."""
    p = np.asarray(pvalues, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    adj = np.empty(m)
    adj[order] = np.minimum.accumulate(scaled[::-1])[::-1]
    return np.minimum(adj, 1.0)


class UnivariateScreen(SelectorMixin, BaseEstimator):
    """Keep features whose distribution differs between the two classes.

    Each feature is tested with a two-sided Mann-Whitney U test (minority
    vs majority; with more than two classes, the smallest class against the
    rest). Features are kept when the adjusted p-value is below
    ``alpha``.

    Parameters
    ----------
    alpha : float, default=0.1
        Level for the adjusted p-values.
    correction : {"bh", "bonferroni", "none"}, default="bh"
        ``"bh"`` controls the false discovery rate (Benjamini-Hochberg);
        ``"bonferroni"`` controls the family-wise error rate.

    Attributes
    ----------
    pvalues_ : ndarray, shape (n_features,)
        Raw two-sided p-values.
    adjusted_pvalues_ : ndarray, shape (n_features,)
    effect_ : ndarray, shape (n_features,)
        Common-language effect size P(X_minority > X_majority) - 0.5,
        in [-0.5, 0.5].
    support_ : ndarray of bool
    n_minority_ : int

    Examples
    --------
    >>> from shadowselect import UnivariateScreen
    >>> from shadowselect.simulation import make_dataset
    >>> X, y = make_dataset(n=20000, p=0.0025, delta=0.7, n_real=3, n_noise=3, seed=2)
    >>> UnivariateScreen().fit(X, y).get_support().tolist()
    [True, True, True, False, False, False]
    """

    def __init__(self, alpha: float = 0.1, correction: str = "bh"):
        self.alpha = alpha
        self.correction = correction

    def fit(self, X, y):
        if self.correction not in ("bh", "bonferroni", "none"):
            raise ValueError(
                f"correction must be 'bh', 'bonferroni' or 'none'; got {self.correction!r}"
            )
        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha}")
        if hasattr(X, "columns"):
            names = np.asarray(X.columns, dtype=object)
            if all(isinstance(c, str) for c in names):
                self.feature_names_in_ = names
        X, y = check_X_y(X, y, dtype=float)
        self.n_features_in_ = X.shape[1]
        classes, counts = np.unique(y, return_counts=True)
        if len(classes) < 2:
            raise ValueError("y must contain at least two classes; got 1 class")
        self.classes_ = classes
        minority = classes[np.argmin(counts)]
        pos = y == minority
        self.n_minority_ = int(pos.sum())
        res = stats.mannwhitneyu(X[pos], X[~pos], axis=0, alternative="two-sided")
        p = np.nan_to_num(np.asarray(res.pvalue, dtype=float), nan=1.0)
        self.pvalues_ = p
        self.effect_ = np.asarray(res.statistic, dtype=float) / (pos.sum() * (~pos).sum()) - 0.5
        if self.correction == "bh":
            self.adjusted_pvalues_ = benjamini_hochberg(p)
        elif self.correction == "bonferroni":
            self.adjusted_pvalues_ = np.minimum(p * p.size, 1.0)
        else:
            self.adjusted_pvalues_ = p.copy()
        self.support_ = self.adjusted_pvalues_ < self.alpha
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def summary(self):
        """Per-feature table: name, effect, p-value, adjusted p-value, kept."""
        import pandas as pd

        check_is_fitted(self, "support_")
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        return pd.DataFrame(
            dict(
                feature=list(names),
                effect=self.effect_,
                pvalue=self.pvalues_,
                adjusted_pvalue=self.adjusted_pvalues_,
                kept=self.support_,
            )
        )
