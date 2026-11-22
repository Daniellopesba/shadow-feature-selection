"""Selection methods compared in the paper (Sections 5 and 6).

Every function takes a candidate matrix ``X`` (n x d), labels ``y`` and a
seed, and returns ``(keep, score)``: a boolean keep-mask over the d candidates
and a per-candidate score in which larger means "more relevant" (used for the
FPR-matched comparison). The functions are self-contained, so the experiments
do not depend on the defaults of the ``shadowselect`` package; the package's
own defaults are evaluated separately (``e4_package_default.py``).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
from common import xgb
from scipy import stats
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedKFold, train_test_split


# --------------------------------------------------------------------------- helpers
def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    adj[order] = np.minimum.accumulate((p[order] * m / np.arange(1, m + 1))[::-1])[::-1]
    return np.minimum(adj, 1.0)


def permuted_shadows(X, rng, k=None):
    """``k`` shadow columns, each a uniformly random row-permutation of a candidate
    column (cycling through the candidates; ``k=None`` means one per candidate)."""
    d = X.shape[1]
    k = d if k is None else k
    return np.column_stack([X[rng.permutation(len(X)), j % d] for j in range(k)])


def xgb_raw_importance(model, d_total, kind):
    s = model.get_booster().get_score(importance_type=kind)
    return np.array([s.get(f"f{j}", 0.0) for j in range(d_total)], dtype=float)


def tree_shap_importance(model, X, rng, max_rows=5000):
    """Mean |TreeSHAP| per column on a uniform subsample of at most ``max_rows`` rows."""
    from xgboost import DMatrix

    idx = rng.choice(len(X), size=min(max_rows, len(X)), replace=False)
    phi = model.get_booster().predict(DMatrix(X[idx]), pred_contribs=True)[:, :-1]
    return np.abs(phi).mean(axis=0)


def beats(cand, shadow):
    """Number of shadows each candidate strictly exceeds (0..K)."""
    return (cand[:, None] > shadow[None, :]).sum(axis=1)


# --------------------------------------------------------------------------- one-shot shadow rules
def one_shot_xgb(X, y, seed, n_shadows=None, base_score="prior", shap=True, **xgb_kw):
    """One XGBoost fit on [X, shadows]; returns per-importance (keep, score).

    Importances: ``gain`` (mean gain per split, XGBoost's default), ``total_gain``
    and, if ``shap``, mean |TreeSHAP|. The rule keeps a candidate that beats
    every shadow; the score is the number of shadows it beats.
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    sh = permuted_shadows(X, rng, n_shadows)
    kw = dict(random_state=seed, **xgb_kw)
    if base_score == "prior":
        kw["base_score"] = float(y.mean())
    m = xgb(**kw).fit(np.hstack([X, sh]), y)
    D = d + sh.shape[1]
    out = {}
    for kind in ("gain", "total_gain"):
        imp = xgb_raw_importance(m, D, kind)
        sc = beats(imp[:d], imp[d:])
        out[kind] = (sc == sh.shape[1], sc.astype(float))
    if shap:
        imp = tree_shap_importance(m, np.hstack([X, sh]), rng)
        sc = beats(imp[:d], imp[d:])
        out["shap"] = (sc == sh.shape[1], sc.astype(float))
    return out


# --------------------------------------------------------------------------- Boruta-style rounds
def boruta_rounds(
    X,
    y,
    seed,
    n_iter=20,
    n_shadows=None,
    importance="gain",
    alpha=0.05,
    correction="bonferroni",
    test_size=0.3,
    n_repeats=3,
):
    """Boruta-style selection with a fixed number of rounds (no removal).

    Each round draws fresh permuted shadows, fits XGBoost (prevalence intercept)
    and records a hit when a candidate beats the largest shadow. Hits are tested
    two-sided against Binomial(n_iter, 1/2) with a Bonferroni or BH correction
    over the d candidates. Returns (keep_with_tentative, keep_accepted_only, hits).
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    hits = np.zeros(d)
    for _ in range(n_iter):
        sh = permuted_shadows(X, rng, n_shadows)
        XX = np.hstack([X, sh])
        m = xgb(random_state=int(rng.integers(2**31 - 1)), base_score=float(y.mean()))
        if importance == "permutation":
            Xtr, Xva, ytr, yva = train_test_split(
                XX, y, test_size=test_size, stratify=y, random_state=int(rng.integers(2**31 - 1))
            )
            m.fit(Xtr, ytr)
            imp = permutation_importance(
                m,
                Xva,
                yva,
                scoring="neg_log_loss",
                n_repeats=n_repeats,
                random_state=int(rng.integers(2**31 - 1)),
            ).importances_mean
        else:
            m.fit(XX, y)
            imp = xgb_raw_importance(m, XX.shape[1], importance)
        hits += imp[:d] > imp[d:].max()
    p_acc = stats.binom.sf(hits - 1, n_iter, 0.5)
    p_rej = stats.binom.cdf(hits, n_iter, 0.5)
    if correction == "bonferroni":
        acc, rej = p_acc < alpha / d, p_rej < alpha / d
    else:
        acc, rej = bh_adjust(p_acc) < alpha, bh_adjust(p_rej) < alpha
    return ~rej, acc, hits


def borutapy(X, y, seed, class_weight=None, max_iter=100):
    """BorutaPy 0.4 (iterative removal, two-step correction) with a shallow forest."""
    import sklearn.ensemble  # noqa: F401  (import before patching numpy aliases)

    for name, val in (("int", int), ("float", float), ("bool", np.bool_)):
        if not hasattr(np, name):
            setattr(np, name, val)
    from boruta import BorutaPy
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=100, max_depth=5, n_jobs=1, class_weight=class_weight, random_state=seed
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = BorutaPy(rf, n_estimators=100, max_iter=max_iter, random_state=seed).fit(X, y)
    acc = b.support_.copy()
    tent = b.support_weak_.copy()
    score = -b.ranking_.astype(float)
    return acc | tent, acc, score


# --------------------------------------------------------------------------- no-shadow baselines
def oof_permutation(X, y, seed, n_splits=5, n_repeats=5):
    """Out-of-fold permutation importance: every positive is used once for validation.

    Per fold, fit XGBoost (prevalence intercept) on the training folds and compute
    the log-loss increase when a column of the held-out fold is shuffled. Rules:
    keep if the mean over folds is > 0, or if mean - 2 SE > 0 (SE over folds).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_fold = []
    for k, (tr, te) in enumerate(skf.split(X, y)):
        m = xgb(random_state=seed + k, base_score=float(y[tr].mean())).fit(X[tr], y[tr])
        r = permutation_importance(
            m, X[te], y[te], scoring="neg_log_loss", n_repeats=n_repeats, random_state=seed + k
        )
        per_fold.append(r.importances_mean)
    a = np.array(per_fold)
    mean = a.mean(axis=0)
    se = a.std(axis=0, ddof=1) / np.sqrt(n_splits)
    z = mean / np.maximum(se, 1e-12)
    return mean > 0, mean - 2 * se > 0, z


def mann_whitney(X, y, seed, n_shadows=None):
    """Two-sided Mann-Whitney test of each candidate. Returns p-values and the
    keep-masks of BH (q = 0.1, 0.2) and of the max-shadow rule on p-values."""
    rng = np.random.default_rng(seed)
    pos, neg = y == 1, y == 0
    pv = stats.mannwhitneyu(X[pos], X[neg], axis=0).pvalue
    sh = permuted_shadows(X, rng, n_shadows)
    psh = stats.mannwhitneyu(sh[pos], sh[neg], axis=0).pvalue
    adj = bh_adjust(pv)
    return {
        "mwu_bh_0.1": (adj < 0.1, -np.log10(pv)),
        "mwu_bh_0.2": (adj < 0.2, -np.log10(pv)),
        "mwu_shadow_max": (pv < psh.min(), -np.log10(pv)),
    }


def gaussian_knockoffs(X, rng):
    """Second-order (Gaussian, equicorrelated) model-X knockoffs."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    Z = (X - mu) / sd
    S = np.corrcoef(Z, rowvar=False)
    lam_min = np.linalg.eigvalsh(S).min()
    s = min(1.0, 2 * lam_min) * 0.999
    Sinv = np.linalg.inv(S)
    d = X.shape[1]
    Ds = s * np.eye(d)
    mean = Z - Z @ Sinv @ Ds
    V = 2 * Ds - Ds @ Sinv @ Ds
    C = np.linalg.cholesky(V + 1e-10 * np.eye(d))
    Zk = mean + rng.standard_normal(Z.shape) @ C.T
    return Zk * sd + mu


def knockoff_select(W, q, offset=1):
    """Knockoff (offset 0) or knockoff+ (offset 1) selection at target FDR q."""
    ts = np.sort(np.abs(W[W != 0]))
    for t in ts:
        if (offset + (W <= -t).sum()) / max(1, (W >= t).sum()) <= q:
            return W >= t
    return np.zeros_like(W, dtype=bool)


def knockoffs_xgb(X, y, seed):
    """Model-X knockoffs with W_j = gain_j - gain of its knockoff (XGBoost, prevalence
    intercept). Returns knockoff+ at q = 0.2, knockoff (offset 0) at q = 0.2, and W."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    Xk = gaussian_knockoffs(X, rng)
    m = xgb(random_state=seed, base_score=float(y.mean())).fit(np.hstack([X, Xk]), y)
    imp = xgb_raw_importance(m, 2 * d, "total_gain")
    W = imp[:d] - imp[d:]
    return knockoff_select(W, 0.2, 1), knockoff_select(W, 0.2, 0), W


def l1_logistic_cv(X, y, seed):
    """L1-penalised logistic regression, penalty by 5-fold CV on log-loss."""
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.preprocessing import StandardScaler

    Z = StandardScaler().fit_transform(X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegressionCV(
            Cs=12,
            cv=StratifiedKFold(5, shuffle=True, random_state=seed),
            l1_ratios=(1.0,),
            solver="saga",
            scoring="neg_log_loss",
            max_iter=2000,
            random_state=seed,
        ).fit(Z, y)
    coef = np.abs(m.coef_.ravel())
    return coef > 0, coef


def stability_selection(X, y, seed, n_sub=50, pi_thr=0.6, ev=1.0):
    """Stability selection (Meinshausen & Buehlmann 2010) with L1-logistic.

    The penalty grid is cut where the mean number selected per half-sample
    reaches q = sqrt((2 pi_thr - 1) d EV), which bounds E[false selections] by EV.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    Z = StandardScaler().fit_transform(X)
    n, d = Z.shape
    q = np.sqrt((2 * pi_thr - 1) * d * ev)
    Cs = np.logspace(-4, 1, 16)
    freq = np.zeros((len(Cs), d))
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    for _ in range(n_sub):
        idx = np.concatenate(
            [rng.choice(pos, len(pos) // 2, replace=False), rng.choice(neg, len(neg) // 2, replace=False)]
        )
        for i, C in enumerate(Cs):
            m = LogisticRegression(penalty="l1", C=C, solver="liblinear").fit(Z[idx], y[idx])
            freq[i] += m.coef_.ravel() != 0
    freq /= n_sub
    ok = freq.sum(axis=1) <= q  # admissible penalties
    prob = freq[ok].max(axis=0) if ok.any() else np.zeros(d)
    return prob >= pi_thr, prob


# --------------------------------------------------------------------------- driver
def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t
