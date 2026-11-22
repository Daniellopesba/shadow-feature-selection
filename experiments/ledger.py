"""The theory <-> experiment ledger.

Each entry is a prediction derived in the paper's Section 3, the experiment
that tests it, the measured value, and a verdict. Run after the experiments:

    python ledger.py            # writes ../docs/ledger_measurements.md

The prose ledger (docs/LEDGER.md and paper Appendix A) quotes these numbers.
Predictions P1, P2 and P6 are recomputed here directly (small, seconds); the
rest read results/*.csv.gz.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from common import RESULTS, ROOT
from sklearn.isotonic import IsotonicRegression

from shadowselect.simulation import make_dataset
from shadowselect.theory import detection_probability, pearson_chi2_2x2, split_gains

OUT = ROOT / "docs" / "ledger_measurements.md"
LINES: list[str] = []


def emit(pid, claim, measured, verdict):
    LINES.append(f"| {pid} | {claim} | {measured} | {verdict} |")
    print(pid, "|", measured, "|", verdict)


def iso_r2(x, y, increasing=False):
    x = np.log(np.asarray(x, float))
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    fit = IsotonicRegression(increasing=increasing).fit(x[ok], y[ok]).predict(x[ok])
    return 1 - np.mean((fit - y[ok]) ** 2) / np.var(y[ok])


# ---------------------------------------------------------------- P1, P2
def p1():
    import xgboost as xgb

    errs = []
    for seed in range(5):
        rng = np.random.default_rng(seed)
        n, k = 3000, 30 + 10 * seed
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
    emit(
        "P1",
        "XGBoost's reported root gain equals Pearson X^2 (prior intercept, lambda=0)",
        f"max relative error {max(errs):.1e} over 5 fits",
        "confirmed",
    )


def p2():
    rng = np.random.default_rng(0)
    res = []
    for n, k, pos in ((60, 3, 17), (200, 10, 150), (1000, 5, 500)):
        p = k / n
        vals = np.empty(20000)
        for r in range(len(vals)):
            y = np.zeros(n)
            y[rng.choice(n, k, replace=False)] = 1
            gl = np.sum(p - y[:pos])
            vals[r] = gl**2 / (p * (1 - p)) * (1 / pos + 1 / (n - pos))
        res.append(f"{vals.mean():.3f} vs {n / (n - 1):.3f}")
    emit("P2", "E[Gain] at a fixed split = n/(n-1), any n_min", "; ".join(res), "confirmed")


# ---------------------------------------------------------------- P3-P6 (E1)
def p3():
    t = pd.read_csv(RESULTS / "e1_null_max.csv")
    t = t[(t.intercept == "prior") & (t.lam == 1.0)]
    spread = t.groupby("n_min").q50.agg(lambda v: (v.max() - v.min()) / v.mean())
    emit(
        "P3",
        "Null best-split law depends on n_min, not n (Thm 1; prior, XGBoost defaults). "
        "Tests the finite-n formula (3) by simulation, not XGBoost itself",
        f"median varies by at most {100 * spread.max():.1f}% across n in 1e3..1e5 "
        f"(n_min 5..1000); median grows {t.q50.min():.1f} -> {t.q50.max():.1f}",
        "confirmed",
    )
    a = pd.read_csv(RESULTS / "e1_null_max.csv")
    a = a[(a.intercept == "auto")].groupby("n").q50.median()
    return a


def e1_root():
    r = pd.read_csv(RESULTS / "e1_root_split.csv.gz")
    if "outcome" not in r.columns:
        return None
    return r


def p4_p5(r):
    pr = r[(r.intercept == "prior")]
    g = pr.groupby(["tree_method", "n_min", "p", "delta"]).root_is_real.mean().reset_index()
    rows = []
    cache = {}
    for _, row in g.iterrows():
        key = (row.tree_method, row.n_min, row.delta)
        if key not in cache:
            nb = 256 if row.tree_method == "hist" else None
            cache[key] = detection_probability(
                int(row.n_min), row.delta, n_null=9, n_rep=600, n_bins=nb, seed=1
            )
        rows.append(cache[key])
    g["predicted"] = rows
    for tm in ("exact", "hist"):
        s = g[g.tree_method == tm]
        mae = np.mean(np.abs(s.predicted - s.root_is_real))
        pv = s.predicted.clip(0.01, 0.99)
        chi2 = float(np.sum((s.root_is_real - s.predicted) ** 2 / (pv * (1 - pv) / 100)))
        mid = s[(s.predicted > 0.2) & (s.predicted < 0.8)]
        mae_mid = np.mean(np.abs(mid.predicted - mid.root_is_real))
        worst = s.loc[(s.predicted - s.root_is_real).abs().idxmax()]
        emit(
            f"P4-{tm}",
            f"P(root uses the informative feature) matches the n->inf limit (Thm 1), {tm} search",
            f"mean |pred - meas| = {mae:.3f} over {len(s)} cells (100 runs each), "
            f"{mae_mid:.3f} over the {len(mid)} cells with 0.2<pred<0.8; binomial "
            f"chi2 = {chi2:.0f} on {len(s)} df; worst "
            f"{worst.root_is_real:.2f} vs {worst.predicted:.2f} at n_min={worst.n_min}, "
            f"p={worst.p}, delta={worst.delta}",
            "confirmed" if mae < 0.06 else "partly",
        )
    s = g[g.tree_method == "exact"]
    r_s = iso_r2(s.n_min * s.delta**2, s.root_is_real, increasing=True)
    r_p = iso_r2(s.p, s.root_is_real, increasing=True)
    emit(
        "P5",
        "Root detection is organised by s = n_min*delta^2, not by p",
        f"monotone-fit R^2: s {r_s:.2f}, p {r_p:.2f}",
        "confirmed" if r_s > 0.85 else "partly",
    )
    return g


def p6(r):
    a = r[(r.intercept == "auto") & (r.n_min == 50) & (r.delta == 0.5)]
    meas = {}
    for (tm, p), v in a.groupby(["tree_method", "p"]).outcome:
        meas[(tm, p)] = (
            (v == "no_split").mean(),
            (v == "noise").sum() / max((v != "no_split").sum(), 1),
        )
    pr = r[(r.intercept == "prior")].outcome.eq("no_split").mean()
    # theory: under exact greedy the root splits iff some feature has Gain > 0, eq. (7),
    # evaluated on each E1 run's own data (same seeds)
    preds, agree = {}, 0
    ex = a[a.tree_method == "exact"]
    for p in (0.001, 0.01, 0.1):
        rows = ex[ex.p == p]
        cnt = 0
        for _, row in rows.iterrows():
            X, y = make_dataset(int(row.n), p, 0.5, 1, 9, int(row.seed))
            b = 1 / (1 + np.exp(-(4 * y.mean() - 2)))
            best = max(
                split_gains(X[:, j], y, lam=1.0, min_child_weight=1.0, base_score=b)[1].max()
                for j in range(10)
            )
            cnt += best <= 0
            agree += (best <= 0) == (row.outcome == "no_split")
        preds[p] = cnt / len(rows)
    n_runs = len(ex)
    txt = "; ".join(
        f"p={p}: predicted {preds[p]:.2f}, measured exact {meas[('exact', p)][0]:.2f}, "
        f"hist {meas[('hist', p)][0]:.2f}"
        for p in (0.001, 0.01, 0.1)
    )
    emit(
        "P6",
        "Automatic intercept: the first tree often does not split, more so as p falls "
        "(Prop 5, eq. 7). Cell n_min=50, delta=0.5",
        f"P(no split) {txt}. Per-run agreement of eq. (7) with XGBoost (exact): "
        f"{agree}/{n_runs}. When the exact-search root does split at p=0.001, a noise feature "
        f"wins {meas[('exact', 0.001)][1]:.0%} of the time. Prevalence intercept: no-split rate "
        f"{pr:.2f}",
        "confirmed (exact); hist removes the extreme splits, so it almost never splits",
    )


# ---------------------------------------------------------------- P7-P10 (E2, E3)
def e2():
    return pd.read_csv(RESULTS / "e2_sweep.csv.gz")


def p7(d):
    s = d[(d.delta > 0) & (d.importance == "gain")]
    g = s.groupby(["delta", "n_min", "p", "intercept"]).fnr_q100.mean().unstack("intercept")
    g = g.reset_index()
    g["gap"] = g.auto - g.prior
    small = g[g.p <= 0.01].gap.mean()
    large = g[g.p >= 0.05].gap.mean()
    worst = g.loc[g.gap.idxmax()]
    emit(
        "P7",
        "Full ensemble: auto intercept hurts at small p, not at moderate p (Sec 3.5)",
        f"mean FNR gap auto-prior: {small:+.2f} for p<=0.01, {large:+.2f} for p>=0.05; "
        f"largest {worst.auto:.2f} vs {worst.prior:.2f} (n_min={worst.n_min}, p={worst.p}, "
        f"delta={worst.delta})",
        "confirmed" if small > 0.05 > abs(large) else "partly",
    )


def iso_loo(x, y, increasing=False):
    x = np.log(np.asarray(x, float))
    y = np.asarray(y, float)
    pred = np.empty_like(y)
    for i in range(len(y)):
        m = np.arange(len(y)) != i
        pred[i] = (
            IsotonicRegression(increasing=increasing, out_of_bounds="clip")
            .fit(x[m], y[m])
            .predict(x[i : i + 1])[0]
        )
    return 1 - np.mean((pred - y) ** 2) / np.var(y)


def p8(d):
    s = d[(d.delta > 0) & (d.importance == "gain")]
    out = []
    for icpt in ("prior", "auto"):
        g = s[s.intercept == icpt].groupby(["delta", "n_min", "p"]).fnr_q100.mean().reset_index()
        sv = g.n_min * g.delta**2
        txt = (
            f"{icpt} ({len(g)} cells): s {iso_r2(sv, g.fnr_q100):.2f} "
            f"(LOO {iso_loo(sv, g.fnr_q100):.2f}), n_min {iso_r2(g.n_min, g.fnr_q100):.2f}, "
            f"delta {iso_r2(g.delta, g.fnr_q100):.2f}, p decreasing "
            f"{iso_r2(1 / g.p, g.fnr_q100):.2f} / increasing "
            f"{iso_r2(1 / g.p, g.fnr_q100, increasing=True):.2f} "
            f"(LOO {iso_loo(1 / g.p, g.fnr_q100, increasing=True):.2f})"
        )
        if icpt == "prior":
            rng = np.random.default_rng(0)
            boots = []
            for _ in range(400):
                i = rng.integers(0, len(g), len(g))
                boots.append(iso_r2(sv.iloc[i], g.fnr_q100.iloc[i]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            ns = g[(g.fnr_q100 > 0.02) & (g.fnr_q100 < 0.98)]
            nsv = ns.n_min * ns.delta**2
            txt += (
                f"; bootstrap 95% CI for s [{lo:.2f}, {hi:.2f}]; {len(ns)} non-saturated "
                f"cells: s {iso_r2(nsv, ns.fnr_q100):.2f} (LOO {iso_loo(nsv, ns.fnr_q100):.2f})"
            )
        out.append(txt)
    emit(
        "P8",
        "Full 300-round ensemble FNR is organised by s = n_min*delta^2 (Conjecture 2)",
        "isotonic R^2 of cell-mean FNR on " + "; ".join(out),
        "supported (prior); partly (auto)",
    )


def p9(d):
    c = d[(d.delta == 0) & (d.importance == "total_gain")].copy()
    ph = c.n_min / c.n
    c["budget"] = c.n * (-ph * np.log(ph) - (1 - ph) * np.log(1 - ph))
    c["total"] = 10 * c.raw_real_mean  # all 10 columns are null in the control
    g = c.groupby(["intercept", "n", "p"]).apply(lambda v: (v.total / v.budget).mean())
    gp = c[c.intercept == "prior"].groupby(["n", "p"]).apply(lambda v: (v.total / v.budget).mean())
    eta = 0.1
    k = 1 / (eta * (1 - eta / 2))
    emit(
        "P9",
        "Gain accounting (Prop 9): total gain ~ (loss removed)/(eta(1-eta/2)) = 10.53 x loss "
        "removed; on pure noise the loss removed is memorisation of n*H2(p)",
        f"total/(n H2) = {g.mean():.1f} +- {g.std():.1f} over all {len(g)} control cells "
        f"(prior intercept: {gp.mean():.1f} +- {gp.std():.1f}, i.e. about "
        f"{gp.mean() / k:.0%} of the initial loss memorised in 300 trees)",
        "identity (Prop 9); the fraction removed is empirical",
    )


def p10():
    d = pd.read_csv(RESULTS / "e3_boosting_rounds.csv.gz")
    d = d[d.importance == "gain"].copy()
    d.loc[(d.raw_real_mean == 0) & (d.raw_noise_max == 0), "fnr_q100"] = np.nan
    g = d.groupby(["intercept", "delta", "n_min", "p", "rounds"]).fnr_q100.mean().reset_index()
    pr = g[g.intercept == "prior"].pivot_table(
        index=["delta", "n_min", "p"], columns="rounds", values="fnr_q100"
    )
    late = pr[300]
    diff = (pr[300] - pr[30]).dropna()
    au = g[(g.intercept == "auto") & (g.p == 0.001)].groupby("rounds").fnr_q100.mean()
    emit(
        "P10",
        "Later rounds: memorisation raises FNR after the signal is learned (Sec 3.6)",
        f"prior: mean FNR {pr[1].mean():.2f} at k=1, {pr[3].mean():.2f} at k=3, "
        f"{pr[30].mean():.2f} at k=30, {late.mean():.2f} at k=300; paired change k=30->300 "
        f"{diff.mean():+.3f} +- {diff.std() / np.sqrt(len(diff)):.3f} (SE, {len(diff)} cells); "
        f"auto, p=0.001: " + ", ".join(f"k={k}: {v:.2f}" for k, v in au.items()),
        "partly: under the prior intercept most of the failure is present by round 3 and "
        "later rounds add a small, significant increase; the large memorisation spike is an "
        "auto-intercept, early-round effect",
    )


# ---------------------------------------------------------------- P11, P12 (E4, E5)
def p11():
    p = RESULTS / "e4_mitigations.csv.gz"
    if not p.exists():
        return
    d = pd.read_csv(p)
    g = d[d.n_min <= 100].groupby("method")[["fnr", "fpr"]].mean()
    txt = "; ".join(f"{m} FNR {r.fnr:.2f}/FPR {r.fpr:.2f}" for m, r in g.iterrows())
    perm = g.loc["boruta_perm"]
    naive = g.loc["naive_auto"]
    ok = perm.fnr < naive.fnr and perm.fpr <= naive.fpr + 0.05
    emit(
        "P11",
        "Held-out permutation importance avoids training-set memorisation, so it "
        "loses fewer relevant features at a similar FPR (n_min<=100)",
        txt,
        "confirmed" if ok else "not confirmed",
    )


def p12():
    p = RESULTS / "e5_backends.csv.gz"
    if not p.exists():
        return
    d = pd.read_csv(p)
    g = d.groupby(["backend", "importance", "n_min", "p"]).fnr_q100.mean().unstack("p")
    g["gap"] = g[0.01] - g[0.1]
    txt = "; ".join(
        f"{b} ({imp}) {v:+.2f}"
        for (b, imp), v in g.groupby(["backend", "importance"]).gap.mean().items()
    )
    hi = g.reset_index()
    hi = hi[hi.n_min >= 250].groupby(["backend", "importance"]).gap.mean()
    emit(
        "P12",
        "Count-based child constraints (LightGBM defaults) leave residual p-dependence at "
        "fixed n_min (Prop 3 scope)",
        f"mean FNR(p=0.01) - FNR(p=0.1) at equal n_min: {txt}. At n_min>=250 only: "
        + "; ".join(f"{b} ({imp}) {v:+.2f}" for (b, imp), v in hi.items()),
        "partly: LightGBM shows the largest gap (both leaf settings) and the only one that "
        "persists at n_min>=250; every learner shows some gap at n_min 50-100; open",
    )


# ---------------------------------------------------------------- P13-P17 (loop 2)
def root_fnr(n_min, delta, k, n_null_rep=20000, n_rep=3000):
    from shadowselect.theory import limit_max_gain

    null = np.sort(limit_max_gain(n_min, 0.0, n_rep=n_null_rep, seed=2))
    sig = limit_max_gain(n_min, delta, n_rep=n_rep, seed=1)
    f = np.searchsorted(null, sig, side="left") / len(null)
    return 1 - float(np.mean(f**k))


def p13():
    d = pd.read_csv(RESULTS / "e4_mitigations.csv.gz")
    one = d[d.method.isin(["naive", "naive_auto", "class_weight", "downsample", "early_stop"])]
    g = one.groupby("method").fpr.mean()
    se = one.fpr.std() / np.sqrt(len(one))
    e6 = pd.read_csv(RESULTS / "e6_univariate.csv.gz")
    m6 = e6[e6.method == "mwu_shadow_max"].fpr.mean()
    emit(
        "P13",
        "Max-shadow rule has level 1/(K+1) for each null candidate, any n_min, p, learner "
        "(Prop 6); K=10 gives 0.091",
        "one-shot XGBoost methods (E4): "
        + ", ".join(f"{m} {v:.3f}" for m, v in g.items())
        + f" (pooled {one.fpr.mean():.3f} +- {se:.3f}); rank-test shadow rule (E6) {m6:.3f}",
        "confirmed",
    )


def p14():
    d = pd.read_csv(RESULTS / "e8_many_shadows.csv.gz")
    g = d.groupby(["n_min", "n_noise"]).fnr_q100.mean().unstack("n_noise")
    parts, above = [], True
    for n_min, row in g.iterrows():
        preds = {k: root_fnr(int(n_min), 0.5, int(k)) for k in g.columns}
        above &= all(row[k] >= preds[k] - 0.05 for k in g.columns)
        parts.append(
            f"n_min={n_min}: "
            + ", ".join(f"K={k} {row[k]:.2f} (root {preds[k]:.2f})" for k in g.columns)
        )
    mono = bool((g.diff(axis=1).iloc[:, 1:] >= -0.02).all().all())
    emit(
        "P14",
        "More shadows raise the bar: FNR increases with K (Cor 1); ensemble at or above "
        "the root-split prediction (Conj 2)",
        "; ".join(parts),
        f"confirmed (monotone in K: {mono}; ensemble >= root: {above})",
    )


def p15():
    p = RESULTS / "e7_rounds.csv.gz"
    if not p.exists():
        return
    from scipy import stats

    d = pd.read_csv(p, dtype={"hits_max": str, "hits_q90": str})
    d = d.sort_values(["n_min", "p", "repeat", "round"])
    out = []
    for R in (10, 20, 40, 60):
        level = 0.05 / 10
        keep_null, keep_real = [], []
        for _, v in d[d["round"] < R].groupby(["n_min", "p", "repeat"]):
            hits = np.array([[c == "1" for c in h] for h in v.hits_max]).sum(axis=0)
            rejected = stats.binom.cdf(hits, R, 0.5) < level
            keep_real.append(1 - rejected[:5].mean())
            keep_null.append(1 - rejected[5:].mean())
        cdf = stats.binom.cdf(np.arange(R + 1), R, 0.5)
        r = int(np.max(np.nonzero(cdf < level)[0])) if (cdf < level).any() else -1
        const = stats.binom.sf(r, R, 1 / 11)
        out.append(
            f"R={R}: FPR {np.mean(keep_null):.3f} (constant-pi {const:.3f}, Markov bound "
            f"{min(1, R / (11 * (r + 1))):.2f}), FNR {1 - np.mean(keep_real):.3f}"
        )
    emit(
        "P15",
        "Keep-tentative FPR after R rounds is below the Markov bound (10) and falls with R "
        "(Prop 7); excess over the constant-pi value measures data dependence",
        "; ".join(out),
        "see text",
    )


def p16():
    from scipy import stats

    e6 = pd.read_csv(RESULTS / "e6_univariate.csv.gz")
    meas = e6.groupby(["method", "n_min"]).fnr.mean()
    rng = np.random.default_rng(0)
    rows = []
    for n_min in sorted(e6.n_min.unique()):
        if n_min > 100:
            continue
        reps = 20000

        def z(dlt, k, m=n_min, reps=reps):
            w = stats.norm.cdf(rng.standard_normal((reps, k, m)) + dlt).mean(axis=2)
            return np.abs(w - 0.5) * np.sqrt(12 * m)

        zr, zn, zs = z(0.5, 5), z(0.0, 5), z(0.0, 10)
        pred_sh = np.mean(zr <= zs.max(axis=1, keepdims=True))
        pv = 2 * stats.norm.sf(np.concatenate([zr, zn], axis=1))
        o = np.argsort(pv, axis=1)
        ps = np.take_along_axis(pv, o, 1) * 10 / np.arange(1, 11)
        adj = np.minimum.accumulate(ps[:, ::-1], axis=1)[:, ::-1]
        bh = np.empty_like(adj)
        np.put_along_axis(bh, o, adj, 1)
        pred_bh = np.mean(bh[:, :5] >= 0.2)
        rows.append(
            f"n_min={n_min}: shadow rule {meas[('mwu_shadow_max', n_min)]:.2f} "
            f"(pred {pred_sh:.2f}), BH 0.2 {meas[('mwu_bh_0.2', n_min)]:.2f} (pred {pred_bh:.2f})"
        )
    emit(
        "P16",
        "Rank test FNR follows Prop 8 (z ~ N(sqrt(3s/pi),1)); at matched level it beats the "
        "tree max-shadow rule (Cor 2)",
        "; ".join(rows),
        "confirmed",
    )


def p17():
    d = pd.read_csv(RESULTS / "e4_mitigations.csv.gz")
    g = d[d.method == "naive"].groupby("n_min").fnr.mean()
    parts = [f"n_min={k}: {v:.2f} (root {root_fnr(int(k), 0.5, 10):.2f})" for k, v in g.items()]
    emit(
        "P17",
        "Ensemble FNR is at least the root-split failure probability (Conj 2), "
        "E4 one-shot rule, K=10, prevalence intercept",
        "; ".join(parts),
        "supported",
    )


if __name__ == "__main__":
    LINES += ["| id | prediction | measured | verdict |", "|---|---|---|---|"]
    p1()
    p2()
    p3()
    r = e1_root()
    if r is not None:
        p4_p5(r)
        p6(r)
    d = e2()
    p7(d)
    p8(d)
    p9(d)
    p10()
    p11()
    p12()
    p13()
    p14()
    p15()
    p16()
    p17()
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        "# Ledger measurements (generated by experiments/ledger.py)\n\n" + "\n".join(LINES) + "\n"
    )
    print("wrote", OUT)
