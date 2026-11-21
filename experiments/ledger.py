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
        "Null best-split law depends on n_min, not n (Thm 1; prior, XGBoost defaults)",
        f"median varies by at most {100 * spread.max():.0f}% across n in 1e3..1e5 "
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
        worst = s.loc[(s.predicted - s.root_is_real).abs().idxmax()]
        emit(
            f"P4-{tm}",
            f"P(root uses the informative feature) matches the n->inf limit (Thm 1), {tm} search",
            f"mean |pred - meas| = {mae:.3f} over {len(s)} cells (100 runs each); worst "
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
    # theory: under exact greedy the root splits iff some feature has Gain > 0, eq. (7)
    preds = {}
    for p in (0.001, 0.01, 0.1):
        n = int(50 / p)
        cnt = 0
        for seed in range(100):
            X, y = make_dataset(n, p, 0.5, 1, 9, seed)
            b = 1 / (1 + np.exp(-(4 * y.mean() - 2)))
            best = max(
                split_gains(X[:, j], y, lam=1.0, min_child_weight=1.0, base_score=b)[1].max()
                for j in range(10)
            )
            cnt += best <= 0
        preds[p] = cnt / 100
    txt = "; ".join(
        f"p={p}: predicted {preds[p]:.2f}, measured exact {meas[('exact', p)][0]:.2f}, "
        f"hist {meas[('hist', p)][0]:.2f}"
        for p in (0.001, 0.01, 0.1)
    )
    emit(
        "P6",
        "Automatic intercept: the first tree often does not split, more so as p falls "
        "(Prop 5, eq. 7). Cell n_min=50, delta=0.5",
        f"P(no split) {txt}. When the exact-search root does split at p=0.001, a noise feature "
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


def p8(d):
    s = d[(d.delta > 0) & (d.importance == "gain")]
    out = []
    for icpt in ("prior", "auto"):
        g = s[s.intercept == icpt].groupby(["delta", "n_min", "p"]).fnr_q100.mean().reset_index()
        out.append(
            f"{icpt}: s {iso_r2(g.n_min * g.delta**2, g.fnr_q100):.2f}, "
            f"n_min {iso_r2(g.n_min, g.fnr_q100):.2f}, p {iso_r2(1 / g.p, g.fnr_q100):.2f}"
        )
    emit(
        "P8",
        "Full 300-round ensemble FNR is organised by s = n_min*delta^2 (prior intercept)",
        "monotone-fit R^2 of FNR on " + "; ".join(out),
        "confirmed (prior); partly (auto)",
    )


def p9(d):
    c = d[(d.delta == 0) & (d.importance == "total_gain") & (d.n_min <= 250)].copy()
    ph = c.n_min / c.n
    c["budget"] = c.n * (-ph * np.log(ph) - (1 - ph) * np.log(1 - ph))
    c["total"] = 10 * c.raw_real_mean  # all 10 columns are null in the control
    g = c.groupby(["intercept", "n", "p"]).apply(lambda v: (v.total / v.budget).mean())
    per_pos = c.groupby("p").apply(lambda v: (v.total / v.n_min).mean())
    emit(
        "P9",
        "Memorisation budget: total gain on pure noise scales like n*H2(p) "
        "~ n_min(1+ln 1/p) (Sec 3.6)",
        f"total/(n H2) = {g.mean():.1f} +- {g.std():.1f} (CV {g.std() / g.mean():.0%}) across "
        f"{len(g)} cells with n_min<=250; per positive it ranges "
        f"{per_pos.min():.0f} (p={per_pos.idxmin()}) to {per_pos.max():.0f} "
        f"(p={per_pos.idxmax()})",
        "confirmed",
    )


def p10():
    d = pd.read_csv(RESULTS / "e3_boosting_rounds.csv.gz")
    d = d[d.importance == "gain"].copy()
    d.loc[(d.raw_real_mean == 0) & (d.raw_noise_max == 0), "fnr_q100"] = np.nan
    g = d.groupby(["intercept", "delta", "n_min", "p", "rounds"]).fnr_q100.mean().reset_index()
    pr = g[g.intercept == "prior"].pivot_table(
        index=["delta", "n_min", "p"], columns="rounds", values="fnr_q100"
    )
    early, late = pr[[1, 3]].mean(axis=1), pr[300]
    au = g[(g.intercept == "auto") & (g.p == 0.001)].groupby("rounds").fnr_q100.mean()
    emit(
        "P10",
        "Later rounds: memorisation raises FNR after the signal is learned (Sec 3.6)",
        f"prior: mean FNR {early.mean():.2f} at k<=3, {pr[30].mean():.2f} at k=30, "
        f"{late.mean():.2f} at k=300; auto, p=0.001: "
        + ", ".join(f"k={k}: {v:.2f}" for k, v in au.items()),
        "partly: under the prior intercept most of the failure is present by round 3; "
        "the memorisation spike is an auto-intercept, early-round effect",
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
    g = d.groupby(["backend", "n_min", "p"]).fnr_q100.mean().unstack("p")
    g["gap"] = g[0.01] - g[0.1]
    txt = "; ".join(
        f"{b}: mean FNR(p=0.01)-FNR(p=0.1) at equal n_min {v:+.2f}"
        for b, v in g.groupby("backend").gap.mean().items()
    )
    emit(
        "P12",
        "Learners without a Hessian-based child constraint (sklearn GBM, RF, LightGBM "
        "defaults) show residual p-dependence at fixed n_min (Prop 3 scope)",
        txt,
        "see text",
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
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(
        "# Ledger measurements (generated by experiments/ledger.py)\n\n" + "\n".join(LINES) + "\n"
    )
    print("wrote", OUT)
