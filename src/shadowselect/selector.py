"""Scikit-learn compatible shadow-feature selector."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from numbers import Integral, Real

import numpy as np
from scipy import sparse, stats
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectorMixin
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.utils import _safe_indexing
from sklearn.utils.validation import check_is_fitted, check_random_state

from ._validation import (
    FrameTransformMixin,
    check_target,
    prepare_X,
    set_feature_names,
    standardized_mean_difference,
)
from .diagnostics import ImbalanceWarning, LearnerWarning, null_keep_rate
from .screening import benjamini_hochberg

__all__ = ["ShadowFeatureSelector"]

ACCEPTED, TENTATIVE, REJECTED = "accepted", "tentative", "rejected"

#: ``s = n_min * delta^2`` below which a feature is reported as underpowered
#: (the paper's guidance: features are found reliably only above about 25 with 5-10 shadows).
S_UNDERPOWERED = 25.0

_SEED_PARAMS = ("random_state", "random_seed", "seed")


def _default_estimator(n_jobs=None):
    # A shallow, class-balanced forest. Depth-limited trees do not memorise a
    # small minority class, which keeps shadow importances low, and cost a
    # fraction of fully grown trees; balanced bootstrap samples give the
    # minority class enough weight to be split on. In our benchmarks it had
    # the lowest false-negative rate of the learners tried at every minority
    # count. It handles missing values.
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=7,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=n_jobs,
    )


def _fit_param_rows(fit_params, idx, n):
    """Index every row-aligned fit parameter (e.g. ``sample_weight``) by ``idx``."""
    out = {}
    for k, v in fit_params.items():
        if v is not None and hasattr(v, "__len__") and not isinstance(v, str) and len(v) == n:
            out[k] = _safe_indexing(v, idx)
        else:
            out[k] = v
    return out


class ShadowFeatureSelector(FrameTransformMixin, SelectorMixin, BaseEstimator):
    """All-relevant feature selection against shadow (null) features.

    Each round appends ``n_shadows`` shadow columns (shuffled copies of
    randomly chosen real columns) to ``X``, shuffles the column order, fits
    ``estimator`` and records a *hit* for every real feature whose importance
    exceeds the ``threshold_quantile`` of the shadow importances. After
    ``n_iter`` rounds two one-sided binomial tests against ``p0`` label each
    feature accepted (hit rate above ``p0``), rejected (below) or tentative,
    as in Boruta (Kursa and Rudnicki, 2010), with Bonferroni or
    Benjamini-Hochberg correction across features.

    Read the paper's practical guidance before choosing settings. In short: the number
    of minority-class rows ``n_min`` and the effect size ``delta`` decide what
    can be found, through ``s = n_min * delta^2`` (reported per feature by
    :meth:`summary`), and every extra shadow raises the bar a real feature must
    clear.

    Parameters
    ----------
    estimator : classifier, default=None
        Any scikit-learn compatible classifier. ``None`` uses a 200-tree
        random forest with ``max_depth=7`` and
        ``class_weight="balanced_subsample"``. XGBoost and LightGBM are faster on
        large tables and read missing values and pandas categoricals natively.
        Two adjustments are made to the per-round copies (never to the object
        you pass): XGBoost < 3.0 gets ``base_score`` set to the prevalence
        (the paper's intercept toll; XGBoost >= 3.0 already does this), and LightGBM
        in native mode is scored by mean gain per split (what XGBoost calls
        ``"gain"``), because LightGBM's split counts and total gain favour
        continuous columns, shadows included. Both are listed in
        ``estimator_adjustments_``.
    importance : {"auto", "native", "permutation"} or callable, default="auto"
        ``"native"`` reads ``estimator.feature_importances_`` after fitting on
        all rows. ``"permutation"`` fits on a stratified ``1 - test_size``
        split and computes permutation importance on the held-out part; it is
        model-agnostic but spends ``test_size`` of the scarce positives on
        validation and costs ``n_repeats`` extra passes per feature. ``"auto"``
        uses native importance when the estimator provides it and permutation
        otherwise. A callable ``f(fitted_estimator, X_val, y_val) -> ndarray``
        is also accepted (it is given the held-out split).
    shadow : {"permute", "gaussian"}, default="permute"
        ``"permute"`` shuffles the rows of a real column (Boruta; preserves
        marginal distributions, missing values and categories).
        ``"gaussian"`` draws i.i.d. standard normal columns.
    n_shadows : int or None, default=10
        Number of shadow columns per round. Each round draws a fresh random
        set of source columns. The paper's detection thresholds assume 5 to
        10 shadows; more shadows make the maximum shadow importance larger
        and drop more real features (the paper's corollary on the number of shadows). ``None``
        means one shadow per feature, Boruta's choice, which is only
        advisable when there are few features.
    threshold_quantile : float in (0, 1], default=1.0
        Quantile of shadow importances a feature must exceed for a hit.
        ``1.0`` is the classical max-shadow rule.
    n_iter : int, default=20
        Number of rounds. It must be large enough for a feature to reach
        significance at ``alpha`` after correction across all features
        (``p0**n_iter < alpha / n_features`` for the default ``p0``); a
        ``ValueError`` is raised otherwise. More rounds lower the false
        positive rate of ``keep_tentative=True``.
    alpha : float, default=0.05
        Significance level of the binomial tests after correction.
    correction : {"bh", "bonferroni"}, default="bh"
        Multiple-testing correction across features. ``"bh"``
        (Benjamini-Hochberg) controls the false discovery rate; with many
        features it rejects irrelevant ones much sooner than Bonferroni, so
        fewer are kept as tentative. ``"bonferroni"`` controls the
        family-wise error rate (Boruta's choice).
    p0 : float, default=0.5
        Hit probability under the null in the binomial tests (Boruta's choice).
    test_size : float, default=0.3
        Held-out fraction for ``importance="permutation"``.
    scoring : str or callable, default="neg_log_loss"
        Scorer for permutation importance.
    n_repeats : int, default=5
        Permutation repeats per round.
    keep_tentative : bool, default=True
        Whether tentative features (neither significantly better nor worse
        than the shadows) are kept by :meth:`transform`. ``True`` is the
        paper's recommendation when positives are few: keeping only accepted
        features dropped most moderately predictive features below about
        100 positives in the paper's experiments. The price is a higher false positive
        rate; ``expected_null_keep_rate_`` reports it for your settings.
    min_minority : int, default=100
        Emit :class:`ImbalanceWarning` if the minority class has fewer rows
        than this (in the full data for ``"native"``; in the training split
        for ``"permutation"``). The paper's bands: below 25 positives no
        importance threshold is reliable; 25 to 100 is marginal.
    n_jobs : int or None, default=None
        Number of rounds run in parallel (threads). ``None`` runs them one
        after another. Results do not depend on ``n_jobs``. With a learner
        that is itself multi-threaded, leave this at ``None`` and set the
        learner's own ``n_jobs``.
    verbose : int, default=0
        ``1`` prints one line per round (elapsed time and decisions so far).
    random_state : int, RandomState or None, default=None
        Seeds the shadows, the column order, the held-out splits and every
        ``random_state`` / ``random_seed`` / ``seed`` parameter of the
        estimator (including inside a ``Pipeline``). Set it for reproducible
        selections.

    Attributes
    ----------
    support_ : ndarray of bool, shape (n_features,)
    decision_ : ndarray of str, shape (n_features,)
        ``"accepted"``, ``"tentative"`` or ``"rejected"``.
    hits_ : ndarray of int, shape (n_features,)
    importance_mode_ : str
        ``"native"``, ``"permutation"`` or ``"callable"`` (resolves ``"auto"``).
    importance_history_ : ndarray, shape (n_iter, n_features)
    shadow_importance_history_ : ndarray, shape (n_iter, n_shadows_)
    shadow_threshold_history_ : ndarray, shape (n_iter,)
    pvalues_accept_, pvalues_reject_ : ndarray, shape (n_features,)
        One-sided binomial p-values (uncorrected).
    effect_size_ : ndarray, shape (n_features,)
        Standardised mean difference (minority minus rest, pooled SD),
        ``NaN`` for categorical columns.
    n_minority_ : int
    n_shadows_ : int
    expected_null_keep_rate_ : float
        Probability that an irrelevant feature is kept by :meth:`transform`,
        if it beats the shadows with probability ``1/(n_shadows_ + 1)`` in
        every round (the paper's level of the max-shadow rule). With ``correction="bh"`` and
        ``keep_tentative=True`` this is an upper bound.
    estimator_adjustments_ : list of str
    n_features_in_ : int
    feature_names_in_ : ndarray of str
        Only when ``X`` has string column names.

    Examples
    --------
    >>> from shadowselect import ShadowFeatureSelector
    >>> from shadowselect.simulation import make_dataset
    >>> X, y = make_dataset(n=2000, p=0.1, delta=1.0, n_real=3, n_noise=3, seed=0)
    >>> sel = ShadowFeatureSelector(random_state=0).fit(X, y)
    >>> sel.get_support().tolist()
    [True, True, True, False, False, False]
    """

    def __init__(
        self,
        estimator=None,
        *,
        importance: str | Callable = "auto",
        shadow: str = "permute",
        n_shadows: int | None = 10,
        threshold_quantile: float = 1.0,
        n_iter: int = 20,
        alpha: float = 0.05,
        correction: str = "bh",
        p0: float = 0.5,
        test_size: float = 0.3,
        scoring="neg_log_loss",
        n_repeats: int = 5,
        keep_tentative: bool = True,
        min_minority: int = 100,
        n_jobs: int | None = None,
        verbose: int = 0,
        random_state=None,
    ):
        self.estimator = estimator
        self.importance = importance
        self.shadow = shadow
        self.n_shadows = n_shadows
        self.threshold_quantile = threshold_quantile
        self.n_iter = n_iter
        self.alpha = alpha
        self.correction = correction
        self.p0 = p0
        self.test_size = test_size
        self.scoring = scoring
        self.n_repeats = n_repeats
        self.keep_tentative = keep_tentative
        self.min_minority = min_minority
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.random_state = random_state

    # ------------------------------------------------------------- validation
    def _check_params(self):
        if self.shadow not in ("permute", "gaussian"):
            raise ValueError(f"shadow must be 'permute' or 'gaussian'; got {self.shadow!r}")
        if not (callable(self.importance) or self.importance in ("auto", "native", "permutation")):
            raise ValueError("importance must be 'auto', 'native', 'permutation' or a callable")
        if self.correction not in ("bonferroni", "bh"):
            raise ValueError(f"correction must be 'bonferroni' or 'bh'; got {self.correction!r}")
        if (
            not isinstance(self.n_iter, Integral)
            or isinstance(self.n_iter, bool)
            or self.n_iter < 1
        ):
            raise ValueError(f"n_iter must be a positive integer; got {self.n_iter!r}")
        if not isinstance(self.threshold_quantile, Real) or not 0 < self.threshold_quantile <= 1:
            raise ValueError(
                f"threshold_quantile must be in (0, 1]; got {self.threshold_quantile!r}"
            )
        for name in ("alpha", "p0", "test_size"):
            v = getattr(self, name)
            if not isinstance(v, Real) or isinstance(v, bool) or not 0 < v < 1:
                raise ValueError(f"{name} must be a number in (0, 1); got {v!r}")
        if self.n_shadows is not None and (
            not isinstance(self.n_shadows, Integral)
            or isinstance(self.n_shadows, bool)
            or self.n_shadows < 1
        ):
            raise ValueError(
                f"n_shadows must be a positive integer or None; got {self.n_shadows!r}"
            )
        if not isinstance(self.n_repeats, Integral) or self.n_repeats < 1:
            raise ValueError(f"n_repeats must be a positive integer; got {self.n_repeats!r}")
        if not isinstance(self.min_minority, Integral) or self.min_minority < 0:
            raise ValueError(
                f"min_minority must be a non-negative integer; got {self.min_minority!r}"
            )

    # --------------------------------------------------------------- learner
    def _base_estimator(self, y, classes):
        if self.estimator is None:
            # use the cores inside each fit unless rounds run in parallel
            parallel_rounds = self.n_jobs not in (None, 1)
            return _default_estimator(n_jobs=1 if parallel_rounds else -1), []
        est = clone(self.estimator)
        notes = []
        mod = type(est).__module__.split(".")[0]
        params = est.get_params(deep=False)
        if mod == "xgboost" and len(classes) == 2 and params.get("base_score") is None:
            import xgboost

            if int(xgboost.__version__.split(".")[0]) < 3:
                prev = float(np.mean(y == classes[1]))
                est.set_params(base_score=prev)
                notes.append(
                    f"base_score set to the prevalence {prev:.4g} "
                    f"(XGBoost {xgboost.__version__} defaults to a fixed intercept)"
                )
        self._mean_gain = False
        if mod == "lightgbm" and self._importance_mode == "native":
            self._mean_gain = True
            notes.append(
                "importance is LightGBM's mean gain per split (XGBoost's 'gain'), not "
                f"importance_type={params.get('importance_type')!r}: split counts and total "
                "gain favour continuous columns, shadows included"
            )
        return est, notes

    @staticmethod
    def _seed(est, seed):
        params = est.get_params(deep=True)
        upd = {k: seed for k in params if k.rsplit("__", 1)[-1] in _SEED_PARAMS}
        if upd:
            est.set_params(**upd)

    # --------------------------------------------------------------- rounds
    def _round_matrix(self, X, kind, src, pos, rng, buf):
        """Real columns plus shadows of ``src`` placed at column positions ``pos``.

        ``pos[i]`` is the position of stacked column ``i`` (reals first, then
        shadows) in the matrix the learner sees.
        """
        n, d = X.shape
        k = len(src)
        m = d + k
        if kind == "dense":
            if buf is None or buf.shape != (n, m):
                buf = np.empty((n, m), dtype=X.dtype, order="F")
            for i in range(d):
                buf[:, pos[i]] = X[:, i]
            for j, s in enumerate(src):
                if self.shadow == "gaussian":
                    buf[:, pos[d + j]] = rng.standard_normal(n)
                else:
                    buf[:, pos[d + j]] = X[rng.permutation(n), s]
            return buf, buf
        if kind == "sparse":
            if self.shadow == "gaussian":
                sh = sparse.csc_matrix(rng.standard_normal((n, k)).astype(X.dtype))
            else:
                sh = sparse.hstack(
                    [X[:, [s]][rng.permutation(n)] for s in src], format="csc", dtype=X.dtype
                )
            stacked = sparse.hstack([X, sh], format="csc")
            order = np.argsort(pos)  # stacked column shown at each position
            return stacked[:, order], None
        # DataFrame: keep each column's dtype (categoricals, NaN) intact
        import pandas as pd

        cols = [None] * m
        for i in range(d):
            cols[pos[i]] = X.iloc[:, i]
        for j, s in enumerate(src):
            if self.shadow == "gaussian":
                cols[pos[d + j]] = pd.Series(rng.standard_normal(n))
            else:
                cols[pos[d + j]] = X.iloc[:, s].take(rng.permutation(n)).reset_index(drop=True)
        frame = pd.concat(cols, axis=1, ignore_index=True)
        frame.columns = [f"f{c}" for c in range(m)]
        return frame, None

    def _importances(self, est, XX, y, fit_params, rng):
        if self._importance_mode == "native":
            est.fit(XX, y, **fit_params)
            if not hasattr(est, "feature_importances_"):
                raise TypeError(
                    f"{type(est).__name__} has no feature_importances_; "
                    "use importance='permutation'"
                )
            if getattr(self, "_mean_gain", False):
                booster = est.booster_
                gain = booster.feature_importance("gain")
                count = booster.feature_importance("split")
                return np.where(count > 0, gain / np.maximum(count, 1), 0.0)
            return np.asarray(est.feature_importances_, dtype=float)
        seed = int(rng.integers(2**31 - 1))
        n = len(y)
        tr, va = train_test_split(
            np.arange(n), test_size=self.test_size, stratify=y, random_state=seed
        )
        X_tr, X_va = _safe_indexing(XX, tr), _safe_indexing(XX, va)
        est.fit(X_tr, y[tr], **_fit_param_rows(fit_params, tr, n))
        if callable(self.importance):
            return np.asarray(self.importance(est, X_va, y[va]), dtype=float)
        sw = fit_params.get("sample_weight")
        r = permutation_importance(
            est,
            X_va,
            y[va],
            scoring=self.scoring,
            n_repeats=self.n_repeats,
            random_state=seed,
            n_jobs=1,
            sample_weight=None if sw is None else _safe_indexing(sw, va),
        )
        return r.importances_mean

    def _one_round(self, base, X, kind, y, fit_params, seed, buf=None):
        rng = np.random.default_rng(seed)
        n, d = X.shape
        k = self.n_shadows_
        if k <= d:
            src = rng.choice(d, size=k, replace=False) if k < d else np.arange(d)
        else:
            src = np.concatenate([rng.permutation(d) for _ in range(-(-k // d))])[:k]
        # shuffle the column order so that ties are not broken in favour of
        # real features (an assumption of the paper's max-shadow level result)
        pos = rng.permutation(d + k)
        XX, buf = self._round_matrix(X, kind, src, pos, rng, buf)
        est = clone(base)
        self._seed(est, int(rng.integers(2**31 - 1)))
        imp_pos = self._importances(est, XX, y, fit_params, rng)
        imp = imp_pos[pos]  # back to stacked order
        if not np.all(np.isfinite(imp)):
            imp = np.nan_to_num(imp, nan=0.0)
        return imp[:d], imp[d:], buf

    # -------------------------------------------------------------------- API
    def fit(self, X, y, **fit_params):
        """Run the shadow-feature rounds and label every feature.

        Parameters
        ----------
        X : array-like, sparse matrix or DataFrame of shape (n_samples, n_features)
            Missing values are allowed if the estimator handles them.
            DataFrames with categorical columns are passed to the estimator
            with their dtypes intact.
        y : array-like of shape (n_samples,)
            Class labels.
        **fit_params : dict
            Passed to ``estimator.fit`` every round, e.g. ``sample_weight``.
            Row-aligned parameters are split with the rows in permutation
            mode; ``sample_weight`` is also used to score permutation
            importance. Parameters that refer to columns (``eval_set``,
            LightGBM's ``categorical_feature``) are not supported, because
            each round reorders the columns; use pandas categoricals instead.
        """
        self._check_params()
        set_feature_names(self, X)
        X, kind = prepare_X(X)
        n, d = X.shape
        y, classes, counts = check_target(y, n)
        self.n_features_in_ = d
        self.classes_ = classes
        self.n_minority_ = int(counts.min())
        self.n_shadows_ = d if self.n_shadows is None else int(self.n_shadows)

        probe = self.estimator if self.estimator is not None else _default_estimator()
        if self.importance == "auto":
            native = hasattr(probe, "feature_importances_") or hasattr(
                type(probe), "feature_importances_"
            )
            mode = "native" if native else "permutation"
        elif callable(self.importance):
            mode = "permutation"
        else:
            mode = self.importance
        self._importance_mode = mode
        self.importance_mode_ = "callable" if callable(self.importance) else mode
        base, self.estimator_adjustments_ = self._base_estimator(y, classes)

        level = self.alpha / d
        if max(self.p0, 1 - self.p0) ** self.n_iter >= level:
            need = int(np.floor(np.log(level) / np.log(max(self.p0, 1 - self.p0)))) + 1
            raise ValueError(
                f"n_iter={self.n_iter} is too small: with {d} features no feature can reach "
                f"significance at alpha={self.alpha}, so every feature would be 'tentative'. "
                f"Use n_iter >= {need}."
            )

        effective_min = self.n_minority_
        if mode != "native":
            effective_min = int(np.floor(self.n_minority_ * (1 - self.test_size)))
        if effective_min < self.min_minority:
            band = (
                "Below about 25 no importance threshold is reliable: prefer UnivariateScreen "
                "or do not prune on importance."
                if effective_min < 25
                else "This is the marginal band (25-100): only features with "
                "n_min * delta^2 above about 25 are found reliably; check summary()['s_hat'] "
                "and consider UnivariateScreen."
            )
            warnings.warn(
                f"The minority class has {effective_min} training rows. {band}",
                ImbalanceWarning,
                stacklevel=2,
            )

        minority = classes[np.argmin(counts)]
        self.effect_size_ = standardized_mean_difference(X, kind, y == minority)

        if kind == "dense" and not X.flags.f_contiguous:
            X = np.asfortranarray(X)  # one copy; columns are then contiguous

        seeds = np.random.default_rng(
            check_random_state(self.random_state).randint(2**31 - 1)
        ).integers(2**31 - 1, size=self.n_iter)

        real_hist = np.empty((self.n_iter, d))
        shadow_hist = np.empty((self.n_iter, self.n_shadows_))
        t0 = time.perf_counter()

        def record(r, real, shadow_imp):
            real_hist[r] = real
            shadow_hist[r] = shadow_imp
            if self.verbose:
                thr = np.quantile(shadow_hist[: r + 1], self.threshold_quantile, axis=1)
                h = (real_hist[: r + 1] > thr[:, None]).sum(axis=0)
                print(
                    f"[ShadowFeatureSelector] round {r + 1}/{self.n_iter} "
                    f"{time.perf_counter() - t0:.1f}s; features with a hit this round: "
                    f"{int((real > thr[-1]).sum())}/{d}; median hits so far: "
                    f"{np.median(h):.0f}",
                    flush=True,
                )

        if self.n_jobs is None or self.n_jobs == 1:
            buf = None
            for r, s in enumerate(seeds):
                real, sh, buf = self._one_round(base, X, kind, y, fit_params, int(s), buf)
                record(r, real, sh)
        else:
            from joblib import Parallel, delayed

            results = Parallel(n_jobs=self.n_jobs, prefer="threads")(
                delayed(self._one_round)(base, X, kind, y, fit_params, int(s)) for s in seeds
            )
            for r, (real, sh, _) in enumerate(results):
                record(r, real, sh)

        thr = np.quantile(shadow_hist, self.threshold_quantile, axis=1)
        hits = (real_hist > thr[:, None]).sum(axis=0)
        self.hits_ = hits
        self.importance_history_ = real_hist
        self.shadow_importance_history_ = shadow_hist
        self.shadow_threshold_history_ = thr
        self.pvalues_accept_ = stats.binom.sf(hits - 1, self.n_iter, self.p0)
        self.pvalues_reject_ = stats.binom.cdf(hits, self.n_iter, self.p0)
        if self.correction == "bonferroni":
            acc = np.minimum(self.pvalues_accept_ * d, 1.0) < self.alpha
            rej = np.minimum(self.pvalues_reject_ * d, 1.0) < self.alpha
        else:
            acc = benjamini_hochberg(self.pvalues_accept_) < self.alpha
            rej = benjamini_hochberg(self.pvalues_reject_) < self.alpha
        decision = np.full(d, TENTATIVE, dtype=object)
        decision[acc] = ACCEPTED
        decision[rej & ~acc] = REJECTED
        self.decision_ = decision
        keep = decision == ACCEPTED
        if self.keep_tentative:
            keep |= decision == TENTATIVE
        self.support_ = keep

        pi0 = (
            1.0 / (self.n_shadows_ + 1)
            if self.threshold_quantile == 1
            else 1.0 - self.threshold_quantile
        )
        self.expected_null_keep_rate_ = null_keep_rate(
            d, self.n_iter, pi0, alpha=self.alpha, p0=self.p0, keep_tentative=self.keep_tentative
        )
        self._check_learner(hits)
        return self

    def _check_learner(self, hits):
        """Warn when the learner gives shadows as much importance as real features.

        A learner that memorises the minority class (deep or leaf-wise trees,
        split-count importance) spreads importance over every column it can
        split on, shadows included. Then no feature beats the shadows often,
        whatever its effect. The check: no feature reaches a hit rate of 1/2
        although some feature has a clear marginal effect (``s_hat >= 25``).
        """
        s_hat = self.n_minority_ * np.nan_to_num(self.effect_size_, nan=0.0) ** 2
        strong = s_hat >= 4 * S_UNDERPOWERED
        if strong.any() and (hits[strong] < self.n_iter / 2).all():
            warnings.warn(
                f"{int(strong.sum())} feature(s) have a large marginal effect "
                f"(n_min * delta_hat^2 >= {4 * S_UNDERPOWERED:.0f}) but none beat the shadows in "
                "half of the rounds. The learner is probably memorising the minority class, "
                "which inflates shadow importances. Use shallower trees or more regularisation "
                "(for LightGBM: num_leaves <= 15, min_child_samples >= 50, subsampling).",
                LearnerWarning,
                stacklevel=3,
            )

    def _get_support_mask(self):
        check_is_fitted(self, "support_")
        return self.support_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.input_tags.sparse = True
        return tags

    def summary(self):
        """Per-feature table (pandas DataFrame), sorted as the input columns.

        Columns: ``feature``, ``decision``, ``hits``, ``hit_rate``,
        ``median_importance``, ``median_shadow_threshold`` (what the feature
        had to beat, in the learner's units), ``p_accept``, ``p_reject``
        (uncorrected), ``delta_hat`` (standardised mean difference, minority
        minus rest), ``s_hat = n_min * delta_hat**2`` and ``underpowered``
        (``s_hat < 25``: with this many positives a feature of this marginal
        effect is usually not detectable, so a rejection says little).
        ``delta_hat`` only sees marginal mean shifts; it is ``NaN`` for
        categorical columns.
        """
        import pandas as pd

        check_is_fitted(self, "support_")
        names = getattr(self, "feature_names_in_", None)
        if names is None:
            names = [f"x{i}" for i in range(self.n_features_in_)]
        s_hat = self.n_minority_ * self.effect_size_**2
        return pd.DataFrame(
            {
                "feature": [str(c) for c in names],
                "decision": self.decision_.astype(str),
                "hits": self.hits_.astype(int),
                "hit_rate": self.hits_ / self.n_iter,
                "median_importance": np.median(self.importance_history_, axis=0),
                "median_shadow_threshold": np.full(
                    self.n_features_in_, float(np.median(self.shadow_threshold_history_))
                ),
                "p_accept": self.pvalues_accept_,
                "p_reject": self.pvalues_reject_,
                "delta_hat": self.effect_size_,
                "s_hat": s_hat,
                "underpowered": np.where(np.isnan(s_hat), False, s_hat < S_UNDERPOWERED),
            }
        )
