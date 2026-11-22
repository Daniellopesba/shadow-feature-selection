"""Evaluate the pre-stated acceptance criteria (``criteria.py``) against the results.

Every theory-versus-simulation comparison in the paper is computed here, from the
files in ``results/`` and, for P1, P2, P6 and P15, from small direct computations.
No verdict is written by hand: each row applies the rule recorded in
``criteria.CRITERIA``. Run after the experiments:

    python paper_numbers.py            # all rows
    python paper_numbers.py P4 P13     # selected rows (the table keeps the others)

Outputs
-------
* ``paper/generated/table_agreement.tex``: the agreement table of Appendix D.
* ``paper/generated/numbers.tex``: LaTeX macros for the numbers quoted in the text.
* ``results/paper_numbers.json``: every row, machine readable.

Conventions (see ``criteria.py``): standard errors are clustered by data set; a
prediction that is itself simulated contributes its Monte Carlo variance; cells
whose prediction and measurement are both below 0.02 or both above 0.98 are
excluded as uninformative; each row reports a p-value for exact agreement,
Bonferroni-combined over its cells, and the table gives Holm-adjusted values.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from common import RESULTS
from criteria import CRITERIA
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "paper" / "generated"
JSON = RESULTS / "paper_numbers.json"
LIMIT_REPS = 20_000

ROWS: dict[str, dict] = {}
MACROS: dict[str, str] = {}


# ------------------------------------------------------------------ helpers
def load(name, **kw):
    p = RESULTS / name
    if not p.exists():
        return None
    return pd.read_csv(p, low_memory=False, **kw)


def informative(pred, meas):
    pred, meas = np.asarray(pred, float), np.asarray(meas, float)
    return ~(((pred < 0.02) & (meas < 0.02)) | ((pred > 0.98) & (meas > 0.98)))


def bonferroni_min(p):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    return float(min(1.0, p.min() * len(p))) if len(p) else np.nan


def cell_stats(d, by, col):
    """Mean of ``col`` per cell and its SE, the data set being the unit."""
    g = d.groupby(by)[col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out["se"] = out["std"].fillna(0) / np.sqrt(out["count"])
    return out


def row(pid, measured, verdict, pvalue=np.nan, **extra):
    c = CRITERIA[pid]
    ROWS[pid] = dict(
        id=pid,
        statement=c["statement"],
        test=c["test"],
        rule=c["rule"],
        measured=measured,
        verdict=verdict,
        p=None if pvalue is None or not np.isfinite(pvalue) else float(pvalue),
        **extra,
    )
    print(f"{pid}: {verdict} | p={pvalue} | {measured}")


def macro(name, value, fmt="{:.2f}"):
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value


# ------------------------------------------------------- the limit object (4)
def limit_sup(m, delta, reps, rng, lam=1.0, w=1.0, n_bins=None, chunk=2000):
    """Draws of sup_u Gamma(u) of eq. (4) for a Gaussian feature with shift delta.

    Exact search: the supremum of a piecewise-continuous process is attained at a
    window end or as a one-sided limit at a jump, so it suffices to evaluate at
    each placement from the left and from the right and at the window ends.
    ``n_bins`` restricts u to the grid j / n_bins (histogram search under (H1)).
    """
    a = w / m
    out = np.empty(reps)
    for s in range(0, reps, chunk):
        r = min(chunk, reps - s)
        v = np.sort(stats.norm.cdf(rng.standard_normal((r, m)) + delta), axis=1)
        if n_bins is None:
            i = np.arange(m)
            # left limit at v_i: count i; value at v_i: count i + 1; window ends
            u = np.concatenate([v, v, np.full((r, 1), a), np.full((r, 1), 1 - a)], axis=1)
            k = np.concatenate(
                [
                    np.broadcast_to(i, (r, m)),
                    np.broadcast_to(i + 1, (r, m)),
                    (v <= a).sum(1, keepdims=True),
                    (v <= 1 - a).sum(1, keepdims=True),
                ],
                axis=1,
            )
        else:
            grid = np.arange(1, n_bins) / n_bins
            grid = grid[(grid >= a) & (grid <= 1 - a)]
            off = 2.0 * np.arange(r)[:, None]
            k = np.searchsorted((v + off).ravel(), (grid[None, :] + off).ravel(), side="right")
            k = k.reshape(r, -1) - m * np.arange(r)[:, None]
            u = np.broadcast_to(grid, k.shape)
        val = (k - m * u) ** 2 * (1 / (m * u + lam) + 1 / (m * (1 - u) + lam))
        val = np.where((u >= a) & (u <= 1 - a), val, -np.inf)
        out[s : s + r] = val.max(axis=1)
    return out


_NULL_CACHE: dict = {}


def null_sup(m, n_bins=None, reps=LIMIT_REPS):
    key = (m, n_bins, reps)
    if key not in _NULL_CACHE:
        rng = np.random.default_rng([m, n_bins or 0, 2])
        _NULL_CACHE[key] = np.sort(limit_sup(m, 0.0, reps, rng, n_bins=n_bins))
    return _NULL_CACHE[key]


def root_win(m, delta, k, n_bins=None, reps=LIMIT_REPS):
    """E[Q(M)^K] (ties counted as losses) and its Monte Carlo SE."""
    null = null_sup(m, n_bins)
    rng = np.random.default_rng([m, int(round(1000 * delta)), n_bins or 0, 1])
    sig = limit_sup(m, delta, reps, rng, n_bins=n_bins)
    q = np.searchsorted(null, sig, side="left") / len(null)
    x = q**k
    return float(x.mean()), float(x.std() / np.sqrt(len(x)))


# ------------------------------------------------------------------ P1, P2
def p1():
    import xgboost as xgb

    from shadowselect.theory import pearson_chi2_2x2

    errs = []
    for seed in range(20):
        rng = np.random.default_rng(seed)
        n, k = 3000, 10 + 5 * seed
        x = rng.standard_normal((n, 1))
        y = np.zeros(n)
        y[rng.choice(n, k, replace=False)] = 1
        m = xgb.XGBClassifier(
            n_estimators=1,
            max_depth=1,
            tree_method="exact",
            reg_lambda=0.0,
            min_child_weight=0.0,
            base_score=k / n,
        ).fit(x, y)
        root = m.get_booster().trees_to_dataframe().iloc[0]
        errs.append(abs(root.Gain / pearson_chi2_2x2(x[:, 0], y, root.Split) - 1))
    e = max(errs)
    macro("PoneErr", f"{e:.0e}".replace("e-0", "e-"), "{}")
    row("P1", f"max relative error {e:.1e} over 20 fits", "agrees" if e <= 1e-6 else "disagrees")


def p2():
    rng = np.random.default_rng(0)
    parts, z = [], []
    for n, k, pos in ((60, 3, 17), (200, 10, 150), (1000, 5, 500), (5000, 5, 50)):
        p = k / n
        reps = 40_000
        # positives' ranks are a uniformly random subset: k_L is hypergeometric
        kl = rng.hypergeometric(k, n - k, pos, size=reps)
        gl = pos * p - kl
        vals = gl**2 / (p * (1 - p)) * (1 / pos + 1 / (n - pos))
        target = n / (n - 1)
        se = vals.std() / np.sqrt(reps)
        z.append((vals.mean() - target) / se)
        parts.append(f"{vals.mean():.3f} (SE {se:.3f}) vs {target:.3f}")
    ok = all(abs(v) <= 3 for v in z)
    p = float(stats.chi2.sf(np.sum(np.square(z)), len(z)))
    row("P2", "; ".join(parts), "agrees" if ok else "disagrees", p)


# ------------------------------------------------------------- P3-P6 (E1)
def p3():
    t = load("e1_null_max.csv")
    if t is None:
        return row("P3", "results missing", "pending")
    t = t[(t.intercept == "prior") & (t.lam == 1.0) & (t.min_child_weight == 1.0)]
    spread = t.groupby("n_min").q50.agg(lambda v: (v.max() - v.min()) / v.mean())
    macro("PthreeSpread", 100 * spread.max(), "{:.1f}")
    row(
        "P3",
        f"largest relative range of the median across n = 1e3..1e5: {spread.max():.3f} "
        f"(n_min {int(spread.idxmax())}); medians {t.q50.min():.2f} to {t.q50.max():.2f}",
        "agrees" if spread.max() <= 0.10 else "disagrees",
    )


def p4():
    r = load("e1_root_split.csv.gz")
    if r is None:
        return row("P4", "results missing", "pending")
    out = {}
    for tm, nb in (("exact", None), ("hist", 256)):
        s = r[(r.intercept == "prior") & (r.tree_method == tm)]
        g = cell_stats(s, ["n_min", "p", "delta"], "root_is_real")
        preds = {}
        for m, dlt in g[["n_min", "delta"]].drop_duplicates().itertuples(index=False):
            preds[(m, dlt)] = root_win(int(m), float(dlt), 9, n_bins=nb)
        g["pred"] = [preds[(m, d)][0] for m, d in zip(g.n_min, g.delta)]
        g["pred_se"] = [preds[(m, d)][1] for m, d in zip(g.n_min, g.delta)]
        g = g[informative(g.pred, g["mean"])]
        pv = g.pred.clip(0.005, 0.995)
        var = pv * (1 - pv) / g["count"] + g.pred_se**2
        chi2 = float(np.sum((g["mean"] - g.pred) ** 2 / var))
        p = float(stats.chi2.sf(chi2, len(g)))
        mae = float(np.mean(np.abs(g["mean"] - g.pred)))
        out[tm] = (len(g), chi2, p, mae)
    n, chi2, p, mae = out["exact"]
    nh, chi2h, ph, maeh = out["hist"]
    macro("PfourMAE", mae)
    macro("PfourP", p)
    macro("PfourCells", n, "{}")
    macro("PfourMAEhist", maeh)
    ok = p >= 0.01 and mae <= 0.05
    row(
        "P4",
        f"exact: {n} cells, chi-square {chi2:.1f}, p = {p:.2f}, mean |error| {mae:.3f}; "
        f"hist (conditional on H1): chi-square {chi2h:.1f} on {nh} cells, p = {ph:.2g}, "
        f"mean |error| {maeh:.3f}",
        "agrees" if ok else "disagrees",
        p,
        hist_p=ph,
    )


def logit_cluster(X, y, groups, iters=50):
    """Logistic regression by Newton's method with a cluster-robust covariance."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        W = mu * (1 - mu)
        H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = 1 / (1 + np.exp(-X @ beta))
    Hinv = np.linalg.pinv(X.T @ (X * (mu * (1 - mu))[:, None]))
    sc = X * (y - mu)[:, None]
    S = pd.DataFrame(sc).groupby(np.asarray(groups)).sum().to_numpy()
    G = len(S)
    cov = Hinv @ (S.T @ S) @ Hinv * G / (G - 1)
    return beta, cov, mu


def p5():
    from sklearn.preprocessing import SplineTransformer

    r = load("e1_root_split.csv.gz")
    if r is None:
        return row("P5", "results missing", "pending")
    s = r[(r.intercept == "prior") & (r.tree_method == "exact")].copy()
    cm = s.groupby(["n_min", "p", "delta"]).root_is_real.transform("mean")
    s = s[(cm >= 0.02) & (cm <= 0.98)].copy()  # saturated cells carry no information
    s["logs"] = np.log(s.n_min * s.delta**2)
    B = SplineTransformer(n_knots=6, degree=3, include_bias=True).fit_transform(s[["logs"]])
    X = np.column_stack([B, np.log10(s.p)])
    cells = s.n_min.astype(str) + "_" + s.p.astype(str) + "_" + s.delta.astype(str)
    beta, cov, mu = logit_cluster(X, s.root_is_real.to_numpy(float), cells)
    b, se = beta[-1], np.sqrt(cov[-1, -1])
    scale = float(np.mean(mu * (1 - mu)))
    lo, hi = (b - 1.96 * se) * scale, (b + 1.96 * se) * scale
    p = float(2 * stats.norm.sf(abs(b / se)))
    macro("PfiveAME", b * scale, "{:+.3f}")
    macro("PfiveLo", lo, "{:+.3f}")
    macro("PfiveHi", hi, "{:+.3f}")
    ok = lo >= -0.05 and hi <= 0.05
    row(
        "P5",
        f"change in root win probability per decade of p at fixed s: {b * scale:+.3f} "
        f"(95% CI {lo:+.3f} to {hi:+.3f}; {cells.nunique()} cells, {len(s)} runs)",
        "agrees" if ok else "disagrees",
        p,
    )


def p6():
    r = load("e1_root_split.csv.gz")
    if r is None:
        return row("P6", "results missing", "pending")
    from shadowselect.simulation import make_dataset
    from shadowselect.theory import split_gains

    a = r[(r.intercept == "auto") & (r.n_min == 50) & (r.delta == 0.5)]
    ex = a[a.tree_method == "exact"]
    agree, n_runs, pred, meas = 0, 0, {}, {}
    for p in sorted(ex.p.unique()):
        rows = ex[ex.p == p]
        cnt = 0
        for _, rw in rows.iterrows():
            X, y = make_dataset(int(rw.n), p, 0.5, 1, 9, int(rw.seed))
            b = 1 / (1 + np.exp(-(4 * y.mean() - 2)))
            best = max(
                split_gains(X[:, j], y, lam=1.0, min_child_weight=1.0, base_score=b)[1].max()
                for j in range(X.shape[1])
            )
            cnt += best <= 0
            agree += (best <= 0) == (rw.outcome == "no_split")
            n_runs += 1
        pred[p] = cnt / len(rows)
        meas[p] = float((rows.outcome == "no_split").mean())
    rate = agree / n_runs
    macro("PsixAgree", 100 * rate, "{:.1f}")
    for p in pred:
        macro(f"PsixNoSplit{str(p).replace('0.', '').replace('.', '')}", meas[p])
    row(
        "P6",
        f"per-run agreement {agree}/{n_runs} ({rate:.1%}); P(no split) "
        + ", ".join(f"p = {p}: {meas[p]:.2f}" for p in pred),
        "agrees" if rate >= 0.99 else "disagrees",
    )


# ------------------------------------------------------------- P7, P8 (E2)
def e2_prior():
    d = load("e2_sweep.csv.gz")
    if d is None:
        return None
    return d[(d.intercept == "prior") & (d.importance == "gain") & (d.delta > 0)]


def p7():
    d = e2_prior()
    if d is None:
        return row("P7", "results missing", "pending")
    k = int(d.n_noise.iloc[0])
    g = cell_stats(d, ["n_min", "p", "delta"], "fnr_q100")
    preds = {}
    for m, dlt in g[["n_min", "delta"]].drop_duplicates().itertuples(index=False):
        preds[(m, dlt)] = root_win(int(m), float(dlt), k, n_bins=256)
    g["pred"] = [1 - preds[(m, d)][0] for m, d in zip(g.n_min, g.delta)]
    g["pred_se"] = [preds[(m, d)][1] for m, d in zip(g.n_min, g.delta)]
    g = g[informative(g.pred, g["mean"])]
    se = np.sqrt(g.se**2 + g.pred_se**2)
    ok = g["mean"] >= g.pred - 2 * se
    # one-sided p-value for "measured below prediction"; SE floor avoids 0/0
    z = (g["mean"] - g.pred) / np.maximum(se, 1e-3)
    p = bonferroni_min(stats.norm.cdf(z))
    macro("PsevenPass", int(ok.sum()), "{}")
    macro("PsevenCells", len(g), "{}")
    macro("PsevenExcess", float((g["mean"] - g.pred).median()))
    row(
        "P7",
        f"{int(ok.sum())} of {len(g)} informative cells at or above the root-split "
        f"prediction (K = {k}, hist grid); median excess {float((g['mean'] - g.pred).median()):+.2f}",
        "agrees" if ok.all() else "disagrees",
        p,
    )


def p8():
    d = e2_prior()
    if d is None:
        return row("P8", "results missing", "pending")
    g = cell_stats(d, ["n_min", "delta", "p"], "fnr_q100").sort_values(["n_min", "delta", "p"])
    diffs = []
    for (m, dlt), v in g.groupby(["n_min", "delta"]):
        v = v.sort_values("p")
        for i in range(len(v) - 1):
            lo, hi = v.iloc[i], v.iloc[i + 1]  # lo has the smaller p
            if not informative([lo["mean"]], [hi["mean"]])[0]:
                continue
            diffs.append(
                dict(
                    n_min=m,
                    delta=dlt,
                    p_small=lo.p,
                    p_large=hi.p,
                    diff=lo["mean"] - hi["mean"],
                    se=np.sqrt(lo.se**2 + hi.se**2),
                )
            )
    if not diffs:
        return row("P8", "no informative pairs", "pending")
    t = pd.DataFrame(diffs)
    ok = t["diff"] >= -2 * t.se
    z = t["diff"] / np.maximum(t.se, 1e-3)
    p = bonferroni_min(stats.norm.cdf(z))
    w = 1 / np.maximum(t.se, 1e-3) ** 2
    pooled = float(np.sum(w * t["diff"]) / np.sum(w))
    pooled_se = float(1 / np.sqrt(np.sum(w)))
    macro("PeightPass", int(ok.sum()), "{}")
    macro("PeightPairs", len(t), "{}")
    macro("PeightPooled", pooled, "{:+.3f}")
    macro("PeightPooledSE", pooled_se, "{:.3f}")
    row(
        "P8",
        f"{int(ok.sum())} of {len(t)} adjacent-p pairs with FNR(smaller p) >= FNR(larger p) - 2 SE; "
        f"precision-weighted mean difference {pooled:+.3f} (SE {pooled_se:.3f})",
        "agrees" if ok.all() else "disagrees",
        p,
    )


# ------------------------------------------------------------ P9-P12 (level)
def p9():
    parts, ok_all, ps = [], True, []
    bound = 1 / 11
    e4, e4b = load("e4_mitigations.csv.gz"), load("e4b_baselines.csv.gz")
    sets = []
    if e4 is not None:
        for m in ("naive_auto", "class_weight", "downsample", "early_stop"):
            sets.append((m, e4[e4.method == m]))
    if e4b is not None:
        for m in ("oneshot_gain", "oneshot_total_gain", "oneshot_shap", "mwu_shadow_max"):
            sets.append((m, e4b[e4b.method == m]))
    if not sets:
        return row("P9", "results missing", "pending")
    for m, v in sets:
        mean, se = v.fpr.mean(), v.fpr.std() / np.sqrt(len(v))
        ok = mean <= bound + 2 * se
        ok_all &= ok
        ps.append(stats.norm.sf((mean - bound) / se))
        parts.append(f"{m.replace('_', ' ')} {mean:.3f} ({se:.3f})")
        macro(f"Pnine{m.replace('_', '')}", mean, "{:.3f}")
    row(
        "P9",
        "FPR (SE) against 1/11 = 0.091: " + "; ".join(parts),
        "agrees" if ok_all else "disagrees",
        bonferroni_min(ps),
    )


def p10():
    d = load("e10_discrete_level.csv.gz")
    tag = ""
    if d is None:
        d = load("e10_discrete_level_quick.csv.gz")
        tag = " (quick run)"
    if d is None:
        return row("P10", "results missing", "pending")
    k = 6
    bound = 1 / (k + 1)
    g = cell_stats(d, ["shadows", "case", "importance"], "hit")
    pf = g[g.shadows == "per_feature"]
    ok = pf["mean"] <= bound + 2 * pf.se
    ps = stats.norm.sf((pf["mean"] - bound) / np.maximum(pf.se, 1e-3))
    bo = g[g.shadows == "boruta"]
    for _, rw in g.iterrows():
        macro(f"Pten{rw.shadows.replace('_', '')}{rw.case.replace('_', '')}{rw.importance.replace('_', '')}", rw["mean"], "{:.3f}")
    hi = pf.loc[pf["mean"].idxmax()]
    bhi = bo.loc[bo["mean"].idxmax()]
    row(
        "P10",
        f"per-feature shadows: {int(ok.sum())} of {len(pf)} case/importance pairs within the bound 1/7 = 0.143 + 2 SE; "
        f"largest {hi['mean']:.3f} ({hi.case.replace('_', ' ')}, {hi.importance.replace('_', ' ')}). "
        f"Boruta shadows, for comparison: largest {bhi['mean']:.3f} ({bhi.case.replace('_', ' ')}, {bhi.importance.replace('_', ' ')})"
        + tag,
        ("agrees" if ok.all() else "disagrees") + tag,
        bonferroni_min(ps),
    )


def p11():
    d = load("e8_many_shadows.csv.gz")
    if d is None:
        return row("P11", "results missing", "pending")
    g = cell_stats(d, ["n_min", "n_noise"], "fnr_q100")
    res, ps = [], []
    for m, v in g.groupby("n_min"):
        v = v.set_index("n_noise")
        ks = sorted(v.index)
        for i, k1 in enumerate(ks):
            for k2 in ks[i + 1 :]:
                a, b = v.loc[k1], v.loc[k2]
                if not informative([a["mean"]], [b["mean"]])[0]:
                    continue
                se = np.sqrt(a.se**2 + b.se**2)
                res.append(b["mean"] >= a["mean"] - 2 * se)
                ps.append(stats.norm.cdf((b["mean"] - a["mean"]) / max(se, 1e-3)))
    txt = "; ".join(
        f"n_min {m}: " + ", ".join(f"K={int(k)} {x:.2f}" for k, x in zip(v.n_noise, v["mean"]))
        for m, v in g.groupby("n_min")
    )
    row(
        "P11",
        f"{sum(res)} of {len(res)} informative comparisons monotone within 2 SE; {txt}",
        "agrees" if all(res) else "disagrees",
        bonferroni_min(ps),
    )


def p12():
    d = load("e7_rounds.csv.gz", dtype={"hits_max": str, "hits_q90": str})
    if d is None:
        return row("P12", "results missing", "pending")
    d = d.sort_values(["n_min", "p", "repeat", "round"])
    n_cand, k, n_real = 10, 10, 5
    level = 0.05 / n_cand
    parts, ok_all, ps = [], True, []
    rmax = int(d["round"].max()) + 1
    theta2 = []
    for R in (10, 20, 40, rmax):
        cdf = stats.binom.cdf(np.arange(R + 1), R, 0.5)
        r_R = int(np.max(np.nonzero(cdf < level)[0])) if (cdf < level).any() else -1
        keep = []
        for _, v in d[d["round"] < R].groupby(["n_min", "p", "repeat"]):
            hits = np.array([[c == "1" for c in h] for h in v.hits_max]).sum(axis=0)
            keep.append(float((hits[n_real:] > r_R).mean()))
            if R == rmax:
                h = hits[n_real:]
                theta2.append(float(np.mean(h * (h - 1) / (R * (R - 1)))))
        keep = np.array(keep)
        mean, se = keep.mean(), keep.std() / np.sqrt(len(keep))
        if r_R >= 0:
            bound = min(1.0, R / ((k + 1) * (r_R + 1)))
            ok = mean <= bound + 2 * se
            ok_all &= ok
            ps.append(stats.norm.sf((mean - bound) / max(se, 1e-3)))
            parts.append(f"R={R}: keep {mean:.3f} (SE {se:.3f}), bound {bound:.3f}")
        else:
            parts.append(f"R={R}: no rejection possible (r_R = -1), keep {mean:.3f}")
        macro(f"PtwelveKeep{R}", mean, "{:.3f}")
    th = np.array(theta2)
    lim, lim_se = 4 * th.mean(), 4 * th.std() / np.sqrt(len(th))
    ok = mean <= lim + 2 * np.sqrt(se**2 + lim_se**2)
    ok_all &= ok
    ps.append(stats.norm.sf((mean - lim) / max(np.sqrt(se**2 + lim_se**2), 1e-3)))
    macro("PtwelveLimit", lim, "{:.3f}")
    row(
        "P12",
        "; ".join(parts) + f"; at R={rmax}: 4 E[theta^2] = {lim:.3f} (SE {lim_se:.3f})",
        "agrees" if ok_all else "disagrees",
        bonferroni_min(ps),
    )


# ------------------------------------------------------- P13, P14 (rank test)
def mw_predict(mu, n_rel, n_null, n_shadows, reps=200_000, seed=0):
    """FNR of the max-shadow rule and of BH (q = 0.1, 0.2) when each relevant
    candidate's standardised statistic is N(mu, 1) and null ones N(0, 1)."""
    rng = np.random.default_rng(seed)
    mu = np.broadcast_to(np.asarray(mu, float), (n_rel,))
    zr = rng.standard_normal((reps, n_rel)) + mu
    zn = rng.standard_normal((reps, n_null))
    zs = rng.standard_normal((reps, n_shadows))
    pr, pn, psh = (2 * stats.norm.sf(np.abs(z)) for z in (zr, zn, zs))
    out = {"mwu_shadow_max": 1 - (pr < psh.min(axis=1, keepdims=True)).mean(axis=0)}
    pv = np.concatenate([pr, pn], axis=1)
    m = pv.shape[1]
    o = np.argsort(pv, axis=1)
    ps = np.take_along_axis(pv, o, 1) * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ps[:, ::-1], axis=1)[:, ::-1]
    bh = np.empty_like(adj)
    np.put_along_axis(bh, o, adj, 1)
    for q in (0.1, 0.2):
        out[f"mwu_bh_{q}"] = 1 - (bh[:, :n_rel] < q).mean(axis=0)
    return {k: (float(np.mean(v)), v) for k, v in out.items()}


def sauc_mu(auc, n_min):
    sauc = 4 * np.pi * n_min * (np.asarray(auc) - 0.5) ** 2
    return np.sign(np.asarray(auc) - 0.5) * np.sqrt(3 * sauc / np.pi)


def p13():
    e6, e4b = load("e6_univariate.csv.gz"), load("e4b_baselines.csv.gz")
    if e6 is None:
        return row("P13", "results missing", "pending")
    auc = stats.norm.cdf(0.5 / np.sqrt(2))
    res, ps, parts = [], [], []
    for meth in ("mwu_shadow_max", "mwu_bh_0.2"):
        g = cell_stats(e6[e6.method == meth], ["n_min", "p"], "fnr")
        for _, rw in g.iterrows():
            pred = mw_predict(sauc_mu(auc, rw.n_min), 5, 5, 10, reps=50_000, seed=int(rw.n_min))[meth][0]
            if not informative([pred], [rw["mean"]])[0]:
                continue
            ok = abs(rw["mean"] - pred) <= 0.05 + 2 * rw.se
            res.append(ok)
            ps.append(2 * stats.norm.sf(abs(rw["mean"] - pred) / max(rw.se, 1e-3)))
            parts.append((abs(rw["mean"] - pred), f"{meth.replace('mwu_', '').replace('_', ' ')}, n_min {int(rw.n_min)}, p {rw.p}: {rw['mean']:.2f} vs {pred:.2f}"))
    worst = max(parts)[1]
    row(
        "P13",
        f"{sum(res)} of {len(res)} informative cells within 0.05 + 2 SE; largest deviation {worst}",
        "agrees" if all(res) else "disagrees",
        bonferroni_min(ps),
    )


def p14():
    d = load("e9_robustness.csv.gz")
    tag = ""
    if d is None:
        d = load("e9_robustness_quick.csv.gz")
        tag = " (quick run)"
    if d is None:
        return row("P14", "results missing", "pending")
    from dgp import population_auc

    res, ps, parts = [], [], []
    for dgp in ("t3", "lognormal", "corr_rel"):
        auc, roles = population_auc(dgp, 0.5)
        a = float(np.mean(np.asarray(auc)[roles == "rel"]))
        for meth in ("mwu_shadow_max", "mwu_bh_0.1"):
            g = cell_stats(d[(d.dgp == dgp) & (d.method == meth)], ["n_min"], "keep_rel")
            for _, rw in g.iterrows():
                pred = 1 - mw_predict(sauc_mu(a, rw.n_min), 5, 5, 10, reps=50_000, seed=int(rw.n_min))[meth][0]
                if not informative([pred], [rw["mean"]])[0]:
                    continue
                ok = abs(rw["mean"] - pred) <= 0.05 + 2 * rw.se
                res.append(ok)
                ps.append(2 * stats.norm.sf(abs(rw["mean"] - pred) / max(rw.se, 1e-3)))
                parts.append((abs(rw["mean"] - pred), f"{dgp.replace('_', ' ')}, {meth.replace('mwu_', '').replace('_', ' ')}, n_min {int(rw.n_min)}: {rw['mean']:.2f} vs {pred:.2f}"))
    worst = max(parts)[1] if parts else "none"
    row(
        "P14",
        f"{sum(res)} of {len(res)} informative cells within 0.05 + 2 SE{tag}; largest deviation {worst}",
        ("agrees" if all(res) else "disagrees") + tag,
        bonferroni_min(ps),
    )


# ------------------------------------------------------------------ P15
def p15():
    import xgboost as xgb

    worst_t, worst_n = 0.0, 0.0
    for seed in range(10):
        rng = np.random.default_rng(seed)
        n = 4000
        X = rng.standard_normal((n, 5))
        y = (rng.random(n) < 1 / (1 + np.exp(-(X[:, 0] - 3)))).astype(float)
        b, eta, lam = y.mean(), 0.1, 1.0
        m = xgb.XGBClassifier(
            n_estimators=1, max_depth=3, learning_rate=eta, reg_lambda=lam, gamma=0.0,
            min_child_weight=1.0, tree_method="exact", base_score=b,
        ).fit(X, y)
        t = m.get_booster().trees_to_dataframe()
        leaves = t[t.Feature == "Leaf"]
        H = leaves.Cover.to_numpy()
        G = -leaves.Gain.to_numpy() * (H + lam) / eta  # leaf value is -eta G / (H + lam)
        Gt, Ht = float(np.sum(b - y)), float(n * b * (1 - b))
        split_sum = t[t.Feature != "Leaf"].Gain.sum()
        identity = np.sum(G**2 / (H + lam)) - Gt**2 / (Ht + lam)
        worst_t = max(worst_t, abs(split_sum / identity - 1))
        # Newton decrease computed directly from the leaf values, against eq. (14)
        v = -eta * G / (H + lam)
        direct = -np.sum(G * v + 0.5 * H * v**2)
        formula = eta * (1 - eta / 2) * np.sum(G**2 / (H + lam)) + eta**2 * lam / 2 * np.sum(
            G**2 / (H + lam) ** 2
        )
        worst_n = max(worst_n, abs(direct / formula - 1))
    worst = max(worst_t, worst_n)
    row(
        "P15",
        f"max relative error: telescoping {worst_t:.1e}, Newton decrease {worst_n:.1e} (10 trees)",
        "agrees" if worst <= 1e-5 else "disagrees",
    )


# ------------------------------------------------------------- fit counts
def fit_counts():
    """Number of data sets (one model fit or one selection run each) per experiment."""
    files = dict(
        Etwo=["e2_sweep.csv.gz", "e2_sweep_xgb3.csv.gz"],
        Ethree=["e3_boosting_rounds.csv.gz"],
        Efour=["e4b_baselines.csv.gz"],
        Efive=["e5_backends.csv.gz"],
        Enine=["e9_robustness.csv.gz"],
        Eten=["e10_discrete_level.csv.gz"],
        Eeleven=["e11_real_data.csv.gz"],
    )
    for name, fs in files.items():
        total = 0
        for f in fs:
            d = load(f)
            if d is None:
                continue
            keys = [c for c in ("regime", "dgp", "dataset", "case", "shadows", "backend", "n", "p",
                                "n_min", "delta", "intercept", "repeat", "seed") if c in d.columns]
            total += len(d[keys].drop_duplicates())
        macro(f"{name}Fits", f"{total:,}".replace(",", "{,}") if total else "--", "{}")


# ------------------------------------------------------------------ output
def holm(ps):
    ids = [k for k, v in ps.items() if v is not None]
    order = sorted(ids, key=lambda k: ps[k])
    m, run, out = len(order), 0.0, {}
    for i, k in enumerate(order):
        run = max(run, min(1.0, (m - i) * ps[k]))
        out[k] = run
    return out


def tex_escape(s):
    return (
        s.replace("\\", r"\textbackslash{}")
        .replace("_", r"\_")
        .replace("%", r"\%")
        .replace("&", r"\&")
        .replace("#", r"\#")
        .replace("^", r"\^{}")
        .replace(">=", r"$\ge$")
        .replace("<=", r"$\le$")
    )


def write_outputs():
    GEN.mkdir(parents=True, exist_ok=True)
    adj = holm({k: v["p"] for k, v in ROWS.items()})
    for k, v in ROWS.items():
        v["p_holm"] = adj.get(k)
    n_tests = len(adj)
    n_rej = sum(1 for v in adj.values() if v < 0.05)
    n_agree = sum(1 for v in ROWS.values() if v["verdict"].startswith("agrees"))
    macro("AgreeRows", n_agree, "{}")
    macro("TotalRows", len(ROWS), "{}")
    macro("HolmTests", n_tests, "{}")
    macro("HolmRejected", n_rej, "{}")

    def key(pid):
        return int(pid[1:])

    lines = [
        r"\begin{longtable}{@{}l p{0.17\textwidth} p{0.47\textwidth} l l@{}}",
        r"\caption{Theory against simulation. Each rule was fixed in advance (\code{experiments/criteria.py}); "
        r"the verdict applies it to the results, and $p$ is the Holm-adjusted p-value for exact agreement "
        r"(-- where the comparison is deterministic).}\label{tab:agreement}\\",
        r"\toprule",
        r"Row & Statement and rule & Result & Verdict & $p$ \\",
        r"\midrule\endfirsthead",
        r"\toprule Row & Statement and rule & Result & Verdict & $p$ \\ \midrule\endhead",
    ]
    for pid in sorted(ROWS, key=key):
        v = ROWS[pid]
        lab = v["statement"].split(" ")[0]
        rest = " ".join(v["statement"].split(" ")[1:])
        stmt = rf"\cref{{{lab}}}{(' ' + tex_escape(rest)) if rest else ''}: {tex_escape(v['rule'])}"
        p = "--" if v.get("p_holm") is None else f"{v['p_holm']:.2g}"
        lines.append(
            rf"{pid} & \raggedright {stmt} & \raggedright {tex_escape(v['measured'])} & {tex_escape(v['verdict'])} & {p} \\"
        )
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{longtable}"]
    (GEN / "table_agreement.tex").write_text("\n".join(lines) + "\n")

    words = "Zero One Two Three Four Five Six Seven Eight Nine".split()

    def cs(name):  # LaTeX command names take letters only
        return "".join(words[int(c)] if c.isdigit() else c for c in name if c.isalnum())

    (GEN / "numbers.tex").write_text(
        "% generated by experiments/paper_numbers.py; do not edit\n"
        + "\n".join(rf"\newcommand{{\num{cs(k)}}}{{{v}}}" for k, v in sorted(MACROS.items()))
        + "\n"
    )
    JSON.write_text(json.dumps(dict(rows=ROWS, macros=MACROS), indent=1, default=float))
    print(f"wrote {GEN / 'table_agreement.tex'}, {GEN / 'numbers.tex'}, {JSON}")


ALL = dict(
    P1=p1, P2=p2, P3=p3, P4=p4, P5=p5, P6=p6, P7=p7, P8=p8, P9=p9, P10=p10,
    P11=p11, P12=p12, P13=p13, P14=p14, P15=p15,
)

if __name__ == "__main__":
    wanted = sys.argv[1:] or list(ALL)
    if sys.argv[1:] and JSON.exists():
        saved = json.loads(JSON.read_text())
        ROWS.update(saved["rows"])
        MACROS.update(saved["macros"])
    for pid in wanted:
        ALL[pid]()
    fit_counts()
    write_outputs()
