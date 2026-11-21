"""Scikit-learn compatible shadow-feature selector."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from numbers import Integral, Real

import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import SelectorMixin
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted, check_random_state, check_X_y

from .diagnostics import ImbalanceWarning, minority_count

__all__ = ["ShadowFeatureSelector"]

ACCEPTED, TENTATIVE, REJECTED = "accepted", "tentative", "rejected"


def _default_estimator(random_state):
    return HistGradientBoostingClassifier(
        max_iter=200, early_stopping=True, random_state=random_state
    )


class ShadowFeatureSelector(SelectorMixin, BaseEstimator):
    """All-relevant feature selection against shadow (null) features.

    Each round appends shadow columns to ``X``, fits ``estimator``, and records
    a *hit* for every real feature whose importance exceeds the
    ``threshold_quantile`` of the shadow importances. After ``n_iter`` rounds
    a two-sided binomial test against ``p0`` (Bonferroni-corrected across
    features) labels each feature accepted, rejected or tentative, as in
    Boruta (Kursa and Rudnicki, 2010).

    Parameters
    ----------
    estimator : classifier, default=None
        Any scikit-learn compatible classifier. With
        ``importance="native"`` it must expose ``feature_importances_`` after
        fitting. ``None`` uses ``HistGradientBoostingClassifier`` with early
        stopping, which forces ``importance="permutation"``.
    importance : {"native", "permutation"} or callable, default="permutation"
        ``"native"`` reads ``estimator.feature_importances_`` (training-set
        gain/impurity importance). ``"permutation"`` fits on a stratified
        ``1 - test_size`` split and computes permutation importance on the
        held-out part. A callable ``f(fitted_estimator, X_val, y_val) ->
        ndarray`` is also accepted. The paper shows training-set gain is the
        importance most damaged by a small minority class.
    shadow : {"permute", "gaussian"}, default="permute"
        ``"permute"`` shuffles each real column independently (Boruta;
        preserves marginal distributions and cardinality). ``"gaussian"``
        draws i.i.d. standard normal columns.
    n_shadows : int or None, default=None
        Number of shadow columns. ``None`` means one per real feature.
        For ``"permute"`` columns are cycled when ``n_shadows`` exceeds
        the number of features.
    threshold_quantile : float in (0, 1], default=1.0
        Quantile of shadow importances a feature must exceed for a hit.
        ``1.0`` is the classical max-shadow rule.
    n_iter : int, default=20
        Number of rounds.
    alpha : float, default=0.05
        Family-wise significance level of the binomial test.
    p0 : float, default=0.5
        Hit probability under the null in the binomial test (Boruta's choice).
    test_size : float, default=0.3
        Held-out fraction for ``importance="permutation"``.
    scoring : str or callable, default="neg_log_loss"
        Scorer for permutation importance.
    n_repeats : int, default=5
        Permutation repeats per round.
    keep_tentative : bool, default=False
        Whether tentative features are kept by :meth:`transform`.
    min_minority : int, default=50
        Emit :class:`ImbalanceWarning` if the minority class has fewer
        samples than this (in the full data for ``"native"``; in the
        training split for ``"permutation"``).
    random_state : int, RandomState or None, default=None
    n_jobs : int or None, default=None
        Passed to :func:`sklearn.inspection.permutation_importance`.

    Attributes
    ----------
    support_ : ndarray of bool, shape (n_features,)
    decision_ : ndarray of str, shape (n_features,)
        ``"accepted"``, ``"tentative"`` or ``"rejected"``.
    hits_ : ndarray of int, shape (n_features,)
    importance_history_ : ndarray, shape (n_iter, n_features)
    shadow_threshold_history_ : ndarray, shape (n_iter,)
    pvalues_accept_, pvalues_reject_ : ndarray, shape (n_features,)
        One-sided binomial p-values (uncorrected).
    n_minority_ : int
    n_features_in_ : int
    feature_names_in_ : ndarray of str
        Only when ``X`` has string column names.

    Examples
    --------
    >>> from shadowselect import ShadowFeatureSelector
    >>> from shadowselect.simulation import make_dataset
    >>> X, y = make_dataset(n=2000, p=0.1, delta=1.0, n_real=3, n_noise=3, seed=0)
    >>> sel = ShadowFeatureSelector(n_iter=10, random_state=0).fit(X, y)
    >>> sel.get_support().tolist()
    [True, True, True, False, False, False]
    """

    def __init__(
        self,
        estimator=None,
        *,
        importance: str | Callable = "permutation",
        shadow: str = "permute",
        n_shadows: int | None = None,
        threshold_quantile: float = 1.0,
        n_iter: int = 20,
        alpha: float = 0.05,
        p0: float = 0.5,
        test_size: float = 0.3,
        scoring="neg_log_loss",
        n_repeats: int = 5,
        keep_tentative: bool = False,
        min_minority: int = 50,
        random_state=None,
        n_jobs: int | None = None,
    ):
        self.estimator = estimator
        self.importance = importance
        self.shadow = shadow
        self.n_shadows = n_shadows
        self.threshold_quantile = threshold_quantile
        self.n_iter = n_iter
        self.alpha = alpha
        self.p0 = p0
        self.test_size = test_size
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.keep_tentative = keep_tentative
        self.min_minority = min_minority
        self.random_state = random_state
        self.n_jobs = n_jobs

    # ------------------------------------------------------------------ utils
    def _validate_params_(self):
        if self.shadow not in ("permute", "gaussian"):
            raise ValueError(f"shadow must be 'permute' or 'gaussian', got {self.shadow!r}")
        if not (callable(self.importance) or self.importance in ("native", "permutation")):
            raise ValueError("importance must be 'native', 'permutation' or a callable")
        if not isinstance(self.n_iter, Integral) or self.n_iter < 1:
            raise ValueError("n_iter must be a positive integer")
        if not isinstance(self.threshold_quantile, Real) or not 0 < self.threshold_quantile <= 1:
            raise ValueError("threshold_quantile must be in (0, 1]")
        if not 0 < self.alpha < 1 or not 0 < self.p0 < 1:
            raise ValueError("alpha and p0 must be in (0, 1)")
        if self.n_shadows is not None and (
            not isinstance(self.n_shadows, Integral) or self.n_shadows < 1
        ):
            raise ValueError("n_shadows must be a positive integer or None")
        if self.estimator is None and self.importance == "native":
            raise ValueError("the default estimator has no native importance; pass an estimator")

    def _make_shadows(self, X, rng):
        n, d = X.shape
        k = d if self.n_shadows is None else int(self.n_shadows)
        if self.shadow == "gaussian":
            return rng.standard_normal((n, k))
        cols = [X[rng.permutation(n), j % d] for j in range(k)]
        return np.column_stack(cols)

    def _importances(self, est, X, y, rng):
        if self.importance == "native":
            est.fit(X, y)
            if not hasattr(est, "feature_importances_"):
                raise TypeError(
                    f"{type(est).__name__} has no feature_importances_; "
                    "use importance='permutation'"
                )
            return np.asarray(est.feature_importances_, dtype=float)
        seed = int(rng.integers(2**31 - 1))
        X_tr, X_va, y_tr, y_va = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=seed
        )
        est.fit(X_tr, y_tr)
        if callable(self.importance):
            return np.asarray(self.importance(est, X_va, y_va), dtype=float)
        r = permutation_importance(
            est,
            X_va,
            y_va,
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            random_state=seed,
            n_jobs=self.n_jobs,
        )
        return r.importances_mean

    # -------------------------------------------------------------------- API
    def fit(self, X, y):
        """Run the shadow-feature rounds and label every feature."""
        self._validate_params_()
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
        self.n_minority_ = minority_count(y)

        effective_min = self.n_minority_
        if self.importance != "native":
            effective_min = int(np.floor(self.n_minority_ * (1 - self.test_size)))
        if effective_min < self.min_minority:
            warnings.warn(
                f"Minority class has {effective_min} training samples (< {self.min_minority}). "
                "Shadow-feature selection loses power quickly at small minority counts; "
                "weak features are likely to be rejected. See shadowselect.diagnostics.",
                ImbalanceWarning,
                stacklevel=2,
            )

        level = self.alpha / X.shape[1]
        if max(self.p0, 1 - self.p0) ** self.n_iter >= level:
            warnings.warn(
                f"n_iter={self.n_iter} is too small for any feature to reach significance at "
                f"alpha={self.alpha} with {X.shape[1]} features; every feature will be "
                "'tentative'. Increase n_iter.",
                UserWarning,
                stacklevel=2,
            )

        rng = np.random.default_rng(check_random_state(self.random_state).randint(2**31 - 1))
        base = self.estimator if self.estimator is not None else _default_estimator(None)
        d = X.shape[1]
        hits = np.zeros(d, dtype=int)
        imp_hist, thr_hist = [], []
        for _ in range(self.n_iter):
            shadows = self._make_shadows(X, rng)
            XX = np.hstack([X, shadows])
            est = clone(base)
            if "random_state" in est.get_params():
                est.set_params(random_state=int(rng.integers(2**31 - 1)))
            imp = self._importances(est, XX, y, rng)
            real, shadow_imp = imp[:d], imp[d:]
            thr = float(np.quantile(shadow_imp, self.threshold_quantile))
            hits += real > thr
            imp_hist.append(real)
            thr_hist.append(thr)

        self.hits_ = hits
        self.importance_history_ = np.vstack(imp_hist)
        self.shadow_threshold_history_ = np.asarray(thr_hist)
        self.pvalues_accept_ = stats.binom.sf(hits - 1, self.n_iter, self.p0)
        self.pvalues_reject_ = stats.binom.cdf(hits, self.n_iter, self.p0)
        decision = np.full(d, TENTATIVE, dtype=object)
        decision[self.pvalues_accept_ < level] = ACCEPTED
        decision[self.pvalues_reject_ < level] = REJECTED
        self.decision_ = decision
        keep = decision == ACCEPTED
        if self.keep_tentative:
            keep |= decision == TENTATIVE
        self.support_ = keep
        return self

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def summary(self):
        """Per-feature table as a list of dicts (or a DataFrame if pandas is installed)."""
        check_is_fitted(self, "support_")
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            names = np.array([f"x{i}" for i in range(self.n_features_in_)])
        rows = [
            {
                "feature": str(names[i]),
                "decision": self.decision_[i],
                "hits": int(self.hits_[i]),
                "n_iter": self.n_iter,
                "median_importance": float(np.median(self.importance_history_[:, i])),
                "p_accept": float(self.pvalues_accept_[i]),
                "p_reject": float(self.pvalues_reject_[i]),
            }
            for i in range(self.n_features_in_)
        ]
        try:
            import pandas as pd
        except ImportError:  # pragma: no cover
            return rows
        return pd.DataFrame(rows)
