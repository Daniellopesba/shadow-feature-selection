"""Univariate screening: a per-feature rank test with false-discovery control.

At small minority counts the shadow rule asks a tree model to rank features
from a few dozen positive rows, and it throws away most of the information
in doing so. A two-sample rank test of each feature between the classes uses
every row, has a known null distribution, and controls false discoveries
directly. In the paper's experiments (E6) it found more real features than any
tree-shadow method at the same false-positive rate when ``n_min <= 50``.

Missing values are dropped per feature (their share is reported), pandas
categorical, object and string columns get a chi-square test of independence
(with missing values as their own level), and columns are processed in blocks
so memory stays close to the size of one block.

It only sees marginal shifts. A feature that matters only through an
interaction, or only in the tails, is invisible to it, so treat it as a
complement to model-based selection, not a replacement.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectorMixin
from sklearn.utils.validation import check_is_fitted

from ._validation import FrameTransformMixin, check_target, prepare_X, set_feature_names

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


class UnivariateScreen(FrameTransformMixin, SelectorMixin, BaseEstimator):
    """Keep features whose distribution differs between the two classes.

    Each numeric feature is tested with a two-sided Mann-Whitney U test
    (minority vs majority; with more than two classes, the smallest class
    against the rest), ignoring missing values. Categorical columns of a
    DataFrame are tested with a chi-square test of independence between
    category (missing values as a level) and class. Features are kept when
    the adjusted p-value is below ``alpha``.

    Parameters
    ----------
    alpha : float, default=0.1
        Level for the adjusted p-values.
    correction : {"bh", "bonferroni", "none"}, default="bh"
        ``"bh"`` controls the false discovery rate (Benjamini-Hochberg);
        ``"bonferroni"`` controls the family-wise error rate.
    block_size : int, default=32
        Number of numeric columns tested at once. Peak memory is about
        ``4 * n_samples * block_size`` floats.

    Attributes
    ----------
    pvalues_ : ndarray, shape (n_features,)
        Raw two-sided p-values (``1.0`` for constant or empty columns).
    adjusted_pvalues_ : ndarray, shape (n_features,)
    effect_ : ndarray, shape (n_features,)
        Common-language effect size P(X_minority > X_majority) - 0.5, in
        [-0.5, 0.5], for numeric features; ``NaN`` for categorical ones.
    test_ : ndarray of str, shape (n_features,)
        ``"mann-whitney"`` or ``"chi2"``.
    missing_fraction_ : ndarray, shape (n_features,)
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

    def __init__(self, alpha: float = 0.1, correction: str = "bh", block_size: int = 32):
        self.alpha = alpha
        self.correction = correction
        self.block_size = block_size

    def fit(self, X, y):
        """Test every feature against the class label."""
        if self.correction not in ("bh", "bonferroni", "none"):
            raise ValueError(
                f"correction must be 'bh', 'bonferroni' or 'none'; got {self.correction!r}"
            )
        if not isinstance(self.alpha, (int, float)) or not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1); got {self.alpha!r}")
        if not isinstance(self.block_size, (int, np.integer)) or self.block_size < 1:
            raise ValueError(f"block_size must be a positive integer; got {self.block_size!r}")
        set_feature_names(self, X)
        X, kind = prepare_X(X)
        n, d = X.shape
        y, classes, counts = check_target(y, n)
        self.n_features_in_ = d
        self.classes_ = classes
        minority = classes[np.argmin(counts)]
        pos = y == minority
        self.n_minority_ = int(pos.sum())

        p = np.ones(d)
        effect = np.full(d, np.nan)
        missing = np.zeros(d)
        test = np.empty(d, dtype=object)
        numeric, categorical = self._split_columns(X, kind)
        for j in categorical:
            test[j] = "chi2"
            p[j], missing[j] = self._chi2(X.iloc[:, j], pos)
        test[numeric] = "mann-whitney"
        bs = int(self.block_size)
        for start in range(0, len(numeric), bs):
            cols = numeric[start : start + bs]
            block = self._dense_block(X, kind, cols)
            nan = np.isnan(block)
            missing[cols] = nan.mean(axis=0)
            if not nan.any():
                p[cols], effect[cols] = self._mwu(block[pos], block[~pos])
                continue
            for c, j in enumerate(cols):
                ok = ~nan[:, c]
                a, b = block[ok & pos, c], block[ok & ~pos, c]
                if len(a) and len(b):
                    pj, ej = self._mwu(a[:, None], b[:, None])
                    p[j], effect[j] = pj[0], ej[0]
        self.pvalues_ = p
        self.effect_ = effect
        self.test_ = test
        self.missing_fraction_ = missing
        if self.correction == "bh":
            self.adjusted_pvalues_ = benjamini_hochberg(p)
        elif self.correction == "bonferroni":
            self.adjusted_pvalues_ = np.minimum(p * p.size, 1.0)
        else:
            self.adjusted_pvalues_ = p.copy()
        self.support_ = self.adjusted_pvalues_ < self.alpha
        return self

    @staticmethod
    def _split_columns(X, kind):
        if kind != "frame":
            return np.arange(X.shape[1]), []
        import pandas as pd

        numeric, categorical = [], []
        for j, t in enumerate(X.dtypes):
            if pd.api.types.is_bool_dtype(t) or (
                pd.api.types.is_numeric_dtype(t) and not isinstance(t, pd.CategoricalDtype)
            ):
                numeric.append(j)
            else:
                categorical.append(j)
        return np.asarray(numeric, dtype=int), categorical

    @staticmethod
    def _dense_block(X, kind, cols):
        if kind == "dense":
            return np.asarray(X[:, cols], dtype=float)
        if kind == "sparse":
            return X[:, cols].toarray().astype(float, copy=False)
        import pandas as pd

        return np.column_stack(
            [pd.to_numeric(X.iloc[:, j]).to_numpy(dtype=float, na_value=np.nan) for j in cols]
        )

    @staticmethod
    def _mwu(a, b):
        res = stats.mannwhitneyu(a, b, axis=0, alternative="two-sided")
        pv = np.nan_to_num(np.asarray(res.pvalue, dtype=float), nan=1.0)
        eff = np.asarray(res.statistic, dtype=float) / (len(a) * len(b)) - 0.5
        return pv, eff

    @staticmethod
    def _chi2(col, pos):
        import pandas as pd

        codes = pd.Series(col).astype("category").cat.codes.to_numpy()  # -1 = missing
        missing = float(np.mean(codes < 0))
        levels, inv = np.unique(codes, return_inverse=True)
        if len(levels) < 2:
            return 1.0, missing
        table = np.zeros((len(levels), 2))
        np.add.at(table, (inv, pos.astype(int)), 1)
        return float(stats.chi2_contingency(table, correction=False).pvalue), missing

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = True
        return tags

    def summary(self):
        """Per-feature table: name, test, effect, missing share, p-values, kept."""
        import pandas as pd

        check_is_fitted(self, "support_")
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        return pd.DataFrame(
            dict(
                feature=list(names),
                test=self.test_.astype(str),
                effect=self.effect_,
                missing_fraction=self.missing_fraction_,
                pvalue=self.pvalues_,
                adjusted_pvalue=self.adjusted_pvalues_,
                kept=self.support_,
            )
        )
