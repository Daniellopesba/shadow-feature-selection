"""Build every figure and table in the paper from results/*.csv.gz.

Run after the experiments:  python make_figures.py
Writes PNG + PDF figures to figures/ and markdown tables to figures/tables.md.
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from common import FIGURES, RESULTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito (colour-blind safe)
C = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#F0E442"]
P_COLOR = {0.2: C[0], 0.1: C[5], 0.05: C[2], 0.01: C[4], 0.005: C[1], 0.001: C[3]}
plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "font.size": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
    }
)
TABLES: list[str] = []


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def table(title, df, floatfmt=".2f"):
    df = df.copy()
    for col in df.columns:
        if col in ("p",):
            df[col] = df[col].map(lambda v: f"{v:g}")
        elif col in ("n", "n_min") and df[col].dtype.kind in "if":
            df[col] = df[col].map(lambda v: f"{int(v):,}")
    TABLES.append(f"### {title}\n\n{df.to_markdown(index=False, floatfmt=floatfmt)}\n")


def se(x):
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


# --------------------------------------------------------------------- E1
def fig_e1():
    t = pd.read_csv(RESULTS / "e1_null_max.csv")
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.4))
    ax = axes[0]
    styles = {
        ("prior", 1.0): ("XGBoost defaults (λ=1, w=1), prior intercept", "-"),
        ("auto", 1.0): ("XGBoost defaults, auto intercept", "--"),
        ("prior", 0.0): ("no regularisation (λ=0, w=0)", ":"),
    }
    for (icpt, lam), (label, ls) in styles.items():
        for k, n in enumerate((1_000, 10_000, 100_000)):
            s = t[(t.intercept == icpt) & (t.lam == lam) & (t.n == n)].sort_values("n_min")
            ax.plot(
                s.n_min,
                s.q50,
                ls,
                color=C[k],
                marker="o",
                ms=2.5,
                label=f"{label}" if k == 2 else None,
            )
    for k, n in enumerate((1_000, 10_000, 100_000)):
        ax.plot([], [], color=C[k], label=f"n = {n:,}")
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="minority count $n_{min}$",
        ylabel="median of max$_t$ Gain (null feature)",
        title="(a) best split of a pure-noise feature, round 1",
    )
    ax.text(
        0.02,
        0.02,
        "auto intercept: best split < 0 (no split) at small $n_{min}$",
        transform=ax.transAxes,
        fontsize=6.5,
        color="grey",
    )
    ax.legend(fontsize=6, loc="lower right")

    r = pd.read_csv(RESULTS / "e1_root_split.csv.gz")
    ex = r[r.tree_method == "exact"]
    g = ex.groupby(["intercept", "p", "n_min", "delta"]).root_is_real.mean().reset_index()
    g["s"] = g.n_min * g.delta**2
    ax = axes[1]
    for icpt, mk in (("prior", "o"), ("auto", "x")):
        for p in sorted(g.p.unique()):
            s = g[(g.intercept == icpt) & (g.p == p)]
            ax.scatter(
                s.s,
                s.root_is_real,
                marker=mk,
                s=14,
                color=P_COLOR[p],
                alpha=0.85,
                label=f"p={p}, {icpt}",
            )
    # Theorem 1 prediction (limit process), n_min = 50 and 500 across delta
    from shadowselect.theory import detection_probability

    for n_min, ls in ((50, "-"), (500, "--")):
        deltas = np.linspace(0.05, 1.0, 14)
        pred = [detection_probability(n_min, dl, n_null=9, n_rep=400, seed=2) for dl in deltas]
        ax.plot(
            n_min * deltas**2,
            pred,
            ls,
            color="black",
            lw=1,
            label=f"Theorem 1 limit, $n_{{min}}$={n_min}",
        )
    ax.set(
        xscale="log",
        xlabel=r"$s = n_{min}\,\delta^2$",
        ylabel="P(root split uses the informative feature)",
        title="(b) 1 informative vs 9 noise features, exact search",
    )
    ax.axhline(0.1, color="grey", lw=0.8, ls=":")
    ax.legend(fontsize=5.5, ncol=2, loc="upper left")
    ax = axes[2]
    h = r[(r.tree_method == "hist")]
    ns = (
        h.groupby(["intercept", "p", "n_min"])
        .outcome.apply(lambda v: (v == "no_split").mean())
        .reset_index()
    )
    for icpt, mk, ls in (("auto", "x", "-"), ("prior", "o", ":")):
        for p in sorted(ns.p.unique()):
            s = ns[(ns.intercept == icpt) & (ns.p == p)].sort_values("n_min")
            ax.plot(
                s.n_min, s.outcome, ls, marker=mk, ms=3, color=P_COLOR[p], label=f"p={p}, {icpt}"
            )
    ax.set(
        xscale="log",
        xlabel="$n_{min}$",
        ylabel="P(first tree does not split)",
        title="(c) the intercept toll (Prop. 5)",
    )
    ax.legend(fontsize=5.5)
    save(fig, "fig1_first_split")

    piv = t[(t.lam == 1.0)].pivot_table(index="n_min", columns=["intercept", "n"], values="q50")
    piv.columns = [f"{a} n={b:,}" for a, b in piv.columns]
    table("E1: median null max of 2·Gain, XGBoost defaults", piv.reset_index())


# --------------------------------------------------------------------- E2
def load_e2(name="e2_sweep"):
    d = pd.read_csv(RESULTS / f"{name}.csv.gz")
    return d


def fig_e2():
    d = load_e2()
    sig = d[(d.delta > 0) & (d.importance == "gain")]
    agg = (
        sig.groupby(["regime", "n", "p", "n_min", "delta", "intercept"])
        .agg(
            fnr=("fnr_q100", "mean"),
            fnr_se=("fnr_q100", se),
            fnr90=("fnr_q90", "mean"),
            floor=("floor_q100", "mean"),
            real=("mean_real_importance", "mean"),
        )
        .reset_index()
    )

    # Fig 2: FNR vs n_min, per delta, auto vs prior
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2), sharex=True, sharey=True)
    for i, icpt in enumerate(("auto", "prior")):
        for j, delta in enumerate((0.3, 0.5, 1.0)):
            ax = axes[i, j]
            for p in sorted(agg.p.unique(), reverse=True):
                s = agg[(agg.intercept == icpt) & (agg.delta == delta) & (agg.p == p)]
                s = s.sort_values("n_min")
                ax.errorbar(
                    s.n_min,
                    s.fnr,
                    yerr=s.fnr_se,
                    marker="o" if p >= 0.05 else "s",
                    ms=3,
                    lw=1,
                    capsize=1.5,
                    color=P_COLOR[p],
                    label=f"p={p} ({'A' if p >= 0.05 else 'B'})",
                )
            ax.set(xscale="log", title=f"δ = {delta}, {icpt} intercept")
            if j == 0:
                ax.set_ylabel("FNR (max-shadow rule)")
            if i == 1:
                ax.set_xlabel("minority count $n_{min}$")
    axes[0, 0].legend(fontsize=6.5)
    save(fig, "fig2_fnr_vs_nmin")

    # Fig 3: collapse on s = n_min delta^2 (prior) and the p-axis view
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    pr = agg[agg.intercept == "prior"].copy()
    pr["s"] = pr.n_min * pr.delta**2
    for p in sorted(pr.p.unique(), reverse=True):
        s = pr[pr.p == p]
        axes[0].scatter(s.p, s.fnr, color=P_COLOR[p], s=12)
        axes[1].scatter(s.s, s.fnr, color=P_COLOR[p], s=14, label=f"p={p}")
    axes[0].set(
        xscale="log",
        xlabel="imbalance ratio p",
        ylabel="FNR",
        title="(a) against p: no single curve",
    )
    axes[1].set(
        xscale="log",
        xlabel=r"$s = n_{min}\,\delta^2$",
        ylabel="FNR",
        title="(b) against $n_{min}\\delta^2$ (prior intercept)",
    )
    axes[1].legend(fontsize=6.5)
    save(fig, "fig3_collapse")

    # Fig 4: mechanism - raw total gain of real vs best noise, and normalised floor
    tg = d[(d.importance == "total_gain") & (d.intercept == "prior")]
    a2 = (
        tg.groupby(["p", "n_min", "delta"])
        .agg(real=("raw_real_mean", "mean"), noise=("raw_noise_max", "mean"))
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.2))
    for k, delta in enumerate((0.0, 0.3, 0.5, 1.0)):
        s = a2[a2.delta == delta].sort_values("n_min")
        if delta > 0:
            axes[0].scatter(s.n_min, s.real / s.noise, color=C[k], s=12, label=f"δ={delta}")
    axes[0].axhline(1, color="grey", lw=0.8)
    axes[0].set(
        xscale="log",
        yscale="log",
        xlabel="$n_{min}$",
        ylabel="mean real / max noise total gain",
        title="(a) informative advantage (raw total gain)",
    )
    axes[0].legend(fontsize=7)
    ctrl = a2[a2.delta == 0].sort_values("n_min")
    axes[1].scatter(ctrl.n_min, ctrl.noise / ctrl.n_min, color=C[6], s=12)
    axes[1].set(
        xscale="log",
        yscale="log",
        xlabel="$n_{min}$",
        ylabel="max noise total gain / $n_{min}$",
        title="(b) pure-noise control (δ = 0)",
    )
    save(fig, "fig4_mechanism")

    # Table: by n_min bin
    pr = agg[agg.intercept == "prior"]
    au = agg[agg.intercept == "auto"]
    bins = [0, 25, 75, 200, 500, 1500, 10**9]
    labels = ["<25", "25-74", "75-199", "200-499", "500-1499", "1500+"]
    rows = []
    for name, df in (("auto", au), ("prior", pr)):
        df = df.assign(bin=pd.cut(df.n_min, bins, right=False, labels=labels))
        t = df.pivot_table(
            index="bin", columns="delta", values="fnr", aggfunc="mean", observed=False
        )
        t.columns = [f"{name} δ={c}" for c in t.columns]
        rows.append(t)
    tab = pd.concat(rows, axis=1).reset_index().rename(columns={"bin": "n_min"})
    table("E2: mean FNR of the max-shadow rule by n_min bin", tab)
    full = agg.pivot_table(
        index=["regime", "n", "p", "n_min"], columns=["intercept", "delta"], values="fnr"
    ).reset_index()
    full.columns = [
        "_".join(str(x) for x in c if x != "") if isinstance(c, tuple) else c for c in full.columns
    ]
    table("E2: FNR (max rule, gain) per scenario", full)
    full.to_csv(FIGURES / "table_e2_fnr.csv", index=False)


def fig_e2_noise20():
    p = RESULTS / "e2_sweep_noise20.csv.gz"
    if not p.exists():
        return
    d20 = pd.read_csv(p)
    d5 = load_e2()
    keys = ["n", "p", "n_min"]
    a = d5[(d5.delta == 0.5) & (d5.intercept == "prior") & (d5.importance == "gain")]
    b = d20[d20.importance == "gain"]
    a = a.groupby(keys).fnr_q100.mean().rename("fnr_5_shadows")
    b = b.groupby(keys).fnr_q100.mean().rename("fnr_20_shadows")
    m = pd.concat([a, b], axis=1).reset_index()
    fig, ax = plt.subplots(figsize=(4.2, 3.2))
    ax.scatter(m.n_min, m.fnr_5_shadows, color=C[0], s=14, label="5 noise features")
    ax.scatter(m.n_min, m.fnr_20_shadows, color=C[1], s=14, marker="s", label="20 noise features")
    ax.set(xscale="log", xlabel="$n_{min}$", ylabel="FNR", title="δ = 0.5, prior intercept")
    ax.legend(fontsize=7)
    save(fig, "fig8_more_shadows")
    table("E2 robustness: 5 vs 20 noise features (δ=0.5, prior)", m)


# --------------------------------------------------------------------- E3
def fig_e3():
    p = RESULTS / "e3_boosting_rounds.csv.gz"
    if not p.exists():
        return
    d = pd.read_csv(p)
    d = d[d.importance == "gain"].copy()
    d.loc[(d.raw_real_mean == 0) & (d.raw_noise_max == 0), "fnr_q100"] = np.nan  # no split
    agg = d.groupby(["n_min", "p", "delta", "intercept", "rounds"]).fnr_q100.mean().reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(9, 5.2), sharex=True, sharey=True)
    for i, delta in enumerate((0.3, 0.5)):
        for j, n_min in enumerate((25, 100, 500)):
            ax = axes[i, j]
            for p in sorted(agg.p.unique(), reverse=True):
                for icpt, ls in (("prior", "-"), ("auto", "--")):
                    s = agg[
                        (agg.delta == delta)
                        & (agg.n_min == n_min)
                        & (agg.p == p)
                        & (agg.intercept == icpt)
                    ]
                    if len(s):
                        ax.plot(
                            s.rounds,
                            s.fnr_q100,
                            ls,
                            marker="o",
                            ms=2.5,
                            color=P_COLOR[p],
                            label=f"p={p}, {icpt}" if (i, j) == (0, 0) else None,
                        )
            ax.set(xscale="log", title=f"$n_{{min}}$ = {n_min}, δ = {delta}")
            if i == 1:
                ax.set_xlabel("boosting rounds used for importance")
            if j == 0:
                ax.set_ylabel("FNR (max rule)")
    axes[0, 0].legend(fontsize=6)
    save(fig, "fig5_rounds")
    piv = agg[(agg.rounds.isin([1, 10, 300]))].pivot_table(
        index=["delta", "n_min", "p"], columns=["intercept", "rounds"], values="fnr_q100"
    )
    piv.columns = [f"{a} k={b}" for a, b in piv.columns]
    table("E3: FNR after k rounds", piv.reset_index())


# --------------------------------------------------------------------- E4
LABELS = {
    "naive_auto": "max rule, auto intercept",
    "naive": "max rule, prior intercept",
    "p90": "P90 rule",
    "class_weight": "class weights",
    "downsample": "1:1 downsampling",
    "early_stop": "early stopping",
    "boruta_gain": "Boruta rounds, gain (keep tentative)",
    "boruta_gain_strict": "Boruta rounds, gain (accepted only)",
    "boruta_perm": "Boruta rounds, held-out permutation (keep tentative)",
    "boruta_perm_strict": "Boruta rounds, held-out permutation (accepted only)",
}


def fig_e4():
    p = RESULTS / "e4_mitigations.csv.gz"
    if not p.exists():
        return
    d = pd.read_csv(p)
    agg = d.groupby(["method", "p", "n_min"]).agg(fnr=("fnr", "mean"), fpr=("fpr", "mean"))
    agg = agg.reset_index()
    methods = [m for m in LABELS if m in set(agg.method)]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for k, m in enumerate(methods):
        s = agg[agg.method == m].groupby("n_min")[["fnr", "fpr"]].mean().reset_index()
        ls = "--" if m.endswith("strict") else "-"
        col = C[k % len(C)]
        axes[0].plot(s.n_min, s.fnr, ls, marker="o", ms=3, color=col, label=LABELS[m])
        axes[1].plot(s.n_min, s.fpr, ls, marker="o", ms=3, color=col)
    axes[0].set(
        xscale="log",
        xlabel="$n_{min}$",
        ylabel="FNR (relevant dropped)",
        title="(a) false negative rate, δ = 0.5",
    )
    axes[1].set(
        xscale="log",
        xlabel="$n_{min}$",
        ylabel="FPR (irrelevant kept)",
        title="(b) false positive rate",
    )
    axes[0].legend(fontsize=6, loc="upper right")
    save(fig, "fig6_mitigations")
    tab = agg.groupby(["method", "n_min"])[["fnr", "fpr"]].mean().reset_index()
    tab["method"] = tab.method.map(LABELS)
    piv = tab.pivot_table(index="method", columns="n_min", values=["fnr", "fpr"])
    piv.columns = [f"{a} n_min={b}" for a, b in piv.columns]
    table("E4: mitigations, mean over p in {0.1, 0.01}", piv.reset_index())


# --------------------------------------------------------------------- E5
def fig_e5():
    p = RESULTS / "e5_backends.csv.gz"
    if not p.exists():
        return
    d = pd.read_csv(p)
    agg = d.groupby(["backend", "p", "n_min"]).fnr_q100.mean().reset_index()
    fig, ax = plt.subplots(figsize=(4.6, 3.3))
    names = {"sklearn_gbm": "sklearn GBM", "random_forest": "random forest", "lightgbm": "LightGBM"}
    for k, (b, label) in enumerate(names.items()):
        for p, ls in ((0.1, "-"), (0.01, "--")):
            s = agg[(agg.backend == b) & (agg.p == p)].sort_values("n_min")
            ax.plot(s.n_min, s.fnr_q100, ls, marker="o", ms=3, color=C[k], label=f"{label}, p={p}")
    ax.set(
        xscale="log", xlabel="$n_{min}$", ylabel="FNR (max rule)", title="other learners, δ = 0.5"
    )
    ax.legend(fontsize=6.5)
    save(fig, "fig7_backends")
    piv = agg.pivot_table(index="n_min", columns=["backend", "p"], values="fnr_q100")
    piv.columns = [f"{a} p={b}" for a, b in piv.columns]
    table("E5: FNR for other learners (δ=0.5)", piv.reset_index())


if __name__ == "__main__":
    for f in (fig_e1, fig_e2, fig_e2_noise20, fig_e3, fig_e4, fig_e5):
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
    (FIGURES / "tables.md").write_text("\n".join(TABLES))
    print("wrote tables.md")
