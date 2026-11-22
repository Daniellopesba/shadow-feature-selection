"""Build every figure and LaTeX table in the paper from results/.

Run after the experiments:  python make_figures.py
Figures go to figures/ (PDF and PNG), named after their number in the paper
(fig1_..., figA1_... for appendix figures). LaTeX tables go to
paper/generated/. Error bars are two standard errors over data sets (the
repeat is the independent unit).
"""

from __future__ import annotations

import matplotlib
import numpy as np
import pandas as pd
from common import FIGURES, RESULTS, ROOT

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

GENERATED = ROOT / "paper" / "generated"

# Okabe-Ito, colour-blind safe; every series also has its own marker
OI = {
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "black": "#000000",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
}
P_STYLE = {  # prevalence -> (colour, marker)
    0.2: (OI["blue"], "o"),
    0.1: (OI["sky"], "o"),
    0.05: (OI["green"], "o"),
    0.01: (OI["orange"], "s"),
    0.005: (OI["vermillion"], "s"),
    0.001: (OI["purple"], "s"),
}
DELTA_STYLE = {0.3: (OI["vermillion"], "o"), 0.5: (OI["blue"], "s"), 1.0: (OI["green"], "^")}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "font.size": 8.5,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8.5,
        "legend.fontsize": 7,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "legend.frameon": False,
        "errorbar.capsize": 1.5,
    }
)


def save(fig, name):
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def write_tex(name, body):
    GENERATED.mkdir(parents=True, exist_ok=True)
    (GENERATED / f"{name}.tex").write_text(body)
    print("wrote generated/", name)


def se(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def f2(v, d=2):
    return "--" if not np.isfinite(v) else f"{v:.{d}f}"


def mean_se(df, keys, col):
    """Mean and SE of ``col`` over repeats (the independent unit) per ``keys``."""
    return df.groupby(keys)[col].agg(["mean", se]).reset_index()


def legend_below(fig, axes, ncol=4, y=-0.02):
    h, lab = [], []
    for ax in np.atleast_1d(axes).ravel():
        for hh, ll in zip(*ax.get_legend_handles_labels(), strict=True):
            if ll not in lab:
                h.append(hh)
                lab.append(ll)
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, y), ncol=ncol)


# --------------------------------------------------------------------- E1: first split
def root_prediction_curve(n_min, deltas, k=9):
    from shadowselect.theory import detection_probability

    return [detection_probability(n_min, d, n_null=k, n_rep=600, seed=2) for d in deltas]


def fig1_first_split():
    t = pd.read_csv(RESULTS / "e1_null_max.csv")
    r = pd.read_csv(RESULTS / "e1_root_split.csv.gz")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))

    ax = axes[0]
    for (lam, ls, lab) in ((1.0, "-", "XGBoost defaults ($\\lambda = w = 1$)"),
                           (0.0, ":", "no regularisation ($\\lambda = w = 0$)")):
        for k, (n, col) in enumerate(((1_000, OI["sky"]), (10_000, OI["blue"]), (100_000, OI["black"]))):
            s = t[(t.intercept == "prior") & (t.lam == lam) & (t.n == n)].sort_values("n_min")
            ax.plot(s.n_min, s.q50, ls, color=col, marker="o", ms=2.5,
                    label=f"$n$ = {n:,}" if lam == 1.0 else None)
        ax.plot([], [], ls, color=OI["grey"], label=lab)
    ax.set(xscale="log", yscale="log", xlabel="minority count $n_{\\min}$",
           ylabel="median best split gain (eq. 1)",
           title="(a) null feature: depends on $n_{\\min}$, not $n$")
    ax.set_yticks([2, 3, 4, 6])
    ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(loc="lower right", fontsize=6.5)

    ax = axes[1]
    ex = r[(r.tree_method == "exact") & (r.intercept == "prior")]
    g = ex.groupby(["p", "n_min", "delta"]).root_is_real.agg(["mean", "count"]).reset_index()
    g["s"] = g.n_min * g.delta**2
    for p in sorted(g.p.unique(), reverse=True):
        s = g[g.p == p]
        col, mk = P_STYLE[p]
        err = 2 * np.sqrt(s["mean"] * (1 - s["mean"]) / s["count"])
        ax.errorbar(s.s, s["mean"], yerr=err, fmt=mk, ms=3, color=col, alpha=0.85,
                    elinewidth=0.6, label=f"XGBoost, $p$ = {p}")
    for n_min, ls in ((50, "-"), (500, "--")):
        deltas = np.linspace(0.05, 1.0, 14)
        ax.plot(n_min * deltas**2, root_prediction_curve(n_min, deltas), ls, color="black",
                lw=1, label=f"limit (Thm 1), $n_{{\\min}}$ = {n_min}")
    ax.axhline(0.1, color=OI["grey"], lw=0.8, ls=":")
    ax.set(xscale="log", xlabel="$s = n_{\\min}\\delta^2$",
           ylabel="P(root uses the informative feature)",
           title="(b) 1 informative vs 9 null features")
    ax.legend(loc="upper left", fontsize=6, ncol=1)
    fig.tight_layout()
    save(fig, "fig1_first_split")


def figA1_intercept_toll():
    r = pd.read_csv(RESULTS / "e1_root_split.csv.gz")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax = axes[0]
    ex = r[(r.tree_method == "exact")]
    g = ex.groupby(["intercept", "p", "n_min", "delta"]).root_is_real.mean().reset_index()
    g["s"] = g.n_min * g.delta**2
    for icpt, fill in (("prior", True), ("auto", False)):
        for p in sorted(g.p.unique(), reverse=True):
            s = g[(g.intercept == icpt) & (g.p == p)]
            col, mk = P_STYLE[p]
            ax.scatter(s.s, s.root_is_real, marker=mk, s=13, edgecolor=col,
                       facecolor=col if fill else "none", lw=0.8,
                       label=f"$p$ = {p}, {'prevalence' if fill else 'XGBoost 2.x default'}")
    ax.set(xscale="log", xlabel="$s = n_{\\min}\\delta^2$",
           ylabel="P(root uses the informative feature)",
           title="(a) exact search, both intercepts")
    ax = axes[1]
    h = r[r.tree_method == "hist"]
    ns = h.groupby(["intercept", "p", "n_min"]).outcome.apply(
        lambda v: (v == "no_split").mean()).reset_index()
    for icpt, ls, fill in (("auto", "-", False), ("prior", ":", True)):
        for p in sorted(ns.p.unique(), reverse=True):
            s = ns[(ns.intercept == icpt) & (ns.p == p)].sort_values("n_min")
            col, mk = P_STYLE[p]
            ax.plot(s.n_min, s.outcome, ls, marker=mk, ms=3, color=col,
                    mfc=col if fill else "none")
    ax.set(xscale="log", xlabel="$n_{\\min}$", ylabel="P(first tree does not split)",
           title="(b) the intercept toll, hist")
    fig.tight_layout()
    legend_below(fig, axes, ncol=3)
    save(fig, "figA1_intercept_toll")


# --------------------------------------------------------------------- E2: full ensemble
def load_e2(name="e2_sweep"):
    d = pd.read_csv(RESULTS / f"{name}.csv.gz")
    return d[d.importance == "gain"]


def e2_cells(d):
    agg = d.groupby(["regime", "n", "p", "n_min", "delta", "intercept"]).fnr_q100.agg(
        ["mean", se, "count"]).reset_index()
    agg["s"] = agg.n_min * agg.delta**2
    return agg


def root_failure(n_min, delta, k):
    """1 - E[Q(M)^K]: root-split failure in the small-prevalence limit."""
    from shadowselect.theory import limit_max_gain

    null = np.sort(limit_max_gain(n_min, 0.0, n_rep=20000, seed=11, n_bins=256))
    sig = limit_max_gain(n_min, delta, n_rep=3000, seed=12, n_bins=256)
    f = np.searchsorted(null, sig, side="left") / len(null)
    return float(1 - np.mean(f**k))


def fig2_scaling():
    d = load_e2()
    agg = e2_cells(d[d.delta > 0])
    pr = agg[agg.intercept == "prior"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    ax = axes[0]
    for delta, (col, mk) in DELTA_STYLE.items():
        s = pr[pr.delta == delta]
        ax.errorbar(s.s, s["mean"], yerr=2 * s["se"], fmt=mk, ms=3, color=col, elinewidth=0.6,
                    label=f"$\\delta$ = {delta}")
    ss = np.unique(np.round(np.geomspace(1, 1500, 18)))
    pred = [root_failure(200, np.sqrt(x / 200), 5) for x in ss]
    ax.plot(ss, pred, "-", color="black", lw=1, label="root-split failure, $n_{\\min}$ = 200, $K$ = 5")
    ax.axhline(5 / 6, color=OI["grey"], lw=0.8, ls=":")
    ax.text(300, 5 / 6 + 0.02, "chance: 5/6", fontsize=6.5, color=OI["grey"])
    ax.set(xscale="log", xlabel="$s = n_{\\min}\\delta^2$", ylabel="FNR (max-shadow rule)",
           title="(a) the full ensemble against $s$", ylim=(-0.03, 1.0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.5, ncol=2, frameon=False)

    ax = axes[1]
    for delta, ls in ((0.3, "-"), (0.5, "--")):
        for n_min in (50, 100, 250, 500):
            s = pr[(pr.delta == delta) & (pr.n_min == n_min)].sort_values("p")
            if s.p.nunique() < 2:
                continue
            col = {50: OI["vermillion"], 100: OI["orange"], 250: OI["green"], 500: OI["blue"]}[n_min]
            ax.errorbar(s.p, s["mean"], yerr=2 * s["se"], fmt=ls, marker="o", ms=3, color=col,
                        elinewidth=0.6, label=f"$n_{{\\min}}$ = {n_min}" if delta == 0.3 else None)
    ax.plot([], [], "-", color=OI["grey"], label="$\\delta$ = 0.3")
    ax.plot([], [], "--", color=OI["grey"], label="$\\delta$ = 0.5")
    ax.set(xscale="log", xlabel="prevalence $p$ (same $n_{\\min}$)", ylabel="FNR",
           title="(b) residual effect of $p$ at fixed $n_{\\min}$")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=6.5, ncol=3, frameon=False)
    fig.tight_layout()
    save(fig, "fig2_scaling")


def figA2_fnr_vs_nmin():
    agg = e2_cells(load_e2().query("delta > 0"))
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for i, icpt in enumerate(("prior", "auto")):
        for j, delta in enumerate((0.3, 0.5, 1.0)):
            ax = axes[i, j]
            for p in sorted(agg.p.unique(), reverse=True):
                s = agg[(agg.intercept == icpt) & (agg.delta == delta) & (agg.p == p)]
                s = s.sort_values("n_min")
                col, mk = P_STYLE[p]
                ax.errorbar(s.n_min, s["mean"], yerr=2 * s["se"], marker=mk, ms=3, lw=0.9,
                            color=col, elinewidth=0.6, label=f"$p$ = {p}")
            name = "prevalence" if icpt == "prior" else "XGBoost 2.x default"
            ax.set(xscale="log", title=f"$\\delta$ = {delta}, {name} intercept")
            if j == 0:
                ax.set_ylabel("FNR")
            if i == 1:
                ax.set_xlabel("$n_{\\min}$")
    fig.tight_layout()
    legend_below(fig, axes, ncol=6)
    save(fig, "figA2_fnr_vs_nmin")


def table_e2():
    agg = e2_cells(load_e2().query("delta > 0"))
    bins = [0, 25, 75, 200, 500, 1500, 10**9]
    labels = ["$<25$", "25--74", "75--199", "200--499", "500--1499", "$\\ge 1500$"]
    agg = agg.assign(bin=pd.cut(agg.n_min, bins, right=False, labels=labels))
    lines = []
    for b in labels:
        row = [b]
        for icpt in ("prior", "auto"):
            for delta in (0.3, 0.5, 1.0):
                s = agg[(agg.bin == b) & (agg.intercept == icpt) & (agg.delta == delta)]
                row.append(f2(s["mean"].mean()))
        n_cells = agg[(agg.bin == b) & (agg.intercept == "prior") & (agg.delta == 0.5)].shape[0]
        row.append(str(n_cells))
        lines.append(" & ".join(row) + r" \\")
    write_tex("table_e2", "\n".join(lines) + "\n")


def table_sbands():
    """FNR of the one-shot rule by band of s (E2, prevalence intercept): guidance table."""
    agg = e2_cells(load_e2().query("delta > 0"))
    c = agg[agg.intercept == "prior"].groupby(["n_min", "p", "delta"])["mean"].mean().reset_index()
    c["s"] = c.n_min * c.delta**2
    bands = [(0, 5, "$< 5$", "50, 0.3"), (5, 10, "5--10", "100, 0.3"), (10, 25, "10--25", "50, 0.5"),
             (25, 60, "25--60", "150, 0.5"), (60, np.inf, "$\\ge 60$", "250, 0.5")]
    lines = []
    for lo, hi, lab, ex in bands:
        x = c[(c.s >= lo) & (c.s < hi)]["mean"]
        n, d = ex.split(", ")
        lines.append(f"{lab} & {x.min():.2f}--{x.max():.2f} & {x.median():.2f} & {len(x)} & "
                     f"$\\nmin = {n}$, $\\delta = {d}$ \\\\")
    write_tex("table_sbands", "\n".join(lines) + "\n")


# --------------------------------------------------------------------- E3, E5, E8: appendix
def figA3_rounds():
    p = RESULTS / "e3_boosting_rounds.csv.gz"
    d = pd.read_csv(p)
    d = d[d.importance == "gain"].copy()
    nosplit = (d.raw_real_mean == 0) & (d.raw_noise_max == 0)
    d["fnr_missing"] = d.fnr_q100.where(~nosplit)  # no split: undefined
    d["fnr_drop_all"] = d.fnr_q100.where(~nosplit, 1.0)  # no split: every feature dropped
    agg = d.groupby(["n_min", "p", "delta", "intercept", "rounds"]).agg(
        fnr=("fnr_missing", "mean"), fnr_all=("fnr_drop_all", "mean"),
        n_ok=("fnr_missing", "count")).reset_index()
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True, sharey=True)
    for i, delta in enumerate((0.3, 0.5)):
        for j, n_min in enumerate((25, 100, 500)):
            ax = axes[i, j]
            for p in sorted(agg.p.unique(), reverse=True):
                for icpt, ls in (("prior", "-"), ("auto", "--")):
                    s = agg[(agg.delta == delta) & (agg.n_min == n_min) & (agg.p == p)
                            & (agg.intercept == icpt)]
                    if len(s):
                        col, mk = P_STYLE[p]
                        ax.plot(s.rounds, s.fnr_all, ls, marker=mk, ms=2.5, color=col,
                                label=f"$p$ = {p}")
            ax.set(xscale="log", title=f"$n_{{\\min}}$ = {n_min}, $\\delta$ = {delta}")
            if i == 1:
                ax.set_xlabel("boosting rounds used")
            if j == 0:
                ax.set_ylabel("FNR")
    axes[0, 0].plot([], [], "-", color=OI["grey"], label="prevalence intercept")
    axes[0, 0].plot([], [], "--", color=OI["grey"], label="XGBoost 2.x default")
    fig.tight_layout()
    legend_below(fig, axes, ncol=5)
    save(fig, "figA3_rounds")
    agg.to_csv(RESULTS / "e3_summary.csv", index=False)


def figA4_mechanism():
    d = pd.read_csv(RESULTS / "e2_sweep.csv.gz")
    tg = d[(d.importance == "total_gain") & (d.intercept == "prior")]
    a2 = tg.groupby(["p", "n_min", "delta"]).agg(
        real=("raw_real_mean", "mean"), noise=("raw_noise_max", "mean")).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))
    for delta, (col, mk) in DELTA_STYLE.items():
        s = a2[a2.delta == delta].sort_values("n_min")
        axes[0].scatter(s.n_min, s.real / s.noise, color=col, marker=mk, s=12,
                        label=f"$\\delta$ = {delta}")
    axes[0].axhline(1, color=OI["grey"], lw=0.8)
    axes[0].set(xscale="log", yscale="log", xlabel="$n_{\\min}$",
                ylabel="mean relevant / max null total gain",
                title="(a) advantage of relevant features")
    axes[0].legend(fontsize=6.5)
    ctrl = a2[a2.delta == 0].sort_values("n_min")
    for p in sorted(ctrl.p.unique(), reverse=True):
        s = ctrl[ctrl.p == p]
        col, mk = P_STYLE[p]
        axes[1].scatter(s.n_min, s.noise / s.n_min, color=col, marker=mk, s=12, label=f"$p$ = {p}")
    axes[1].set(xscale="log", yscale="log", xlabel="$n_{\\min}$",
                ylabel="max null total gain / $n_{\\min}$", title="(b) pure-noise control ($\\delta$ = 0)")
    axes[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    save(fig, "figA4_mechanism")


BACKEND_STYLE = {
    "sklearn_gbm": ("scikit-learn GBM", OI["blue"], "o"),
    "random_forest": ("random forest, full depth", OI["vermillion"], "s"),
    "rf_shallow": ("random forest, depth 5", OI["orange"], "D"),
    "rf_shallow_balanced": ("random forest, depth 5, balanced", OI["purple"], "v"),
    "lightgbm": ("LightGBM, 63 leaves", OI["green"], "^"),
    "lightgbm_default": ("LightGBM, 31 leaves", OI["sky"], "<"),
}


def figA5_backends():
    d = pd.read_csv(RESULTS / "e5_backends.csv.gz")
    d = d[d.importance == "native"]
    agg = d.groupby(["backend", "p", "n_min"]).fnr_q100.agg(["mean", se]).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), sharey=True)
    for ax, p in zip(axes, (0.1, 0.01), strict=True):
        for b, (lab, col, mk) in BACKEND_STYLE.items():
            s = agg[(agg.backend == b) & (agg.p == p)].sort_values("n_min")
            if len(s):
                ax.errorbar(s.n_min, s["mean"], yerr=2 * s["se"], marker=mk, ms=3, color=col,
                            lw=0.9, elinewidth=0.6, label=lab)
        ax.set(xscale="log", xlabel="$n_{\\min}$", title=f"$p$ = {p}, $\\delta$ = 0.5")
    axes[0].set_ylabel("FNR (max-shadow rule)")
    fig.tight_layout()
    legend_below(fig, axes, ncol=3)
    save(fig, "figA5_backends")


def figA6_shadows():
    e8 = pd.read_csv(RESULTS / "e8_many_shadows.csv.gz")
    col = "fnr_q100" if "fnr_q100" in e8 else "fnr"
    k = "n_noise" if "n_noise" in e8 else "n_shadows"
    agg = e8.groupby([k, "n_min"])[col].agg(["mean", se]).reset_index()
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    for kk, (c, mk) in zip(sorted(agg[k].unique()), ((OI["blue"], "o"), (OI["orange"], "s"),
                                                      (OI["vermillion"], "^")), strict=False):
        s = agg[agg[k] == kk].sort_values("n_min")
        ax.errorbar(s.n_min, s["mean"], yerr=2 * s["se"], marker=mk, ms=3, color=c, lw=0.9,
                    elinewidth=0.6, label=f"$K$ = {kk}")
    ax.set(xscale="log", xlabel="$n_{\\min}$", ylabel="FNR (max-shadow rule)",
           title="number of shadows, $\\delta$ = 0.5, $p$ = 0.1")
    ax.legend()
    fig.tight_layout()
    save(fig, "figA6_shadows")


# --------------------------------------------------------------------- methods (E4, E4b, E4c, E6)
METHOD_LABELS = {
    # one-shot rules at level 1/(K+1)
    "oneshot_gain": "max rule, gain",
    "oneshot_total_gain": "max rule, total gain",
    "oneshot_shap": "max rule, TreeSHAP",
    "mwu_shadow_max": "max rule, Mann–Whitney",
    # other one-shot variants (E4)
    "naive_auto": "max rule, gain, XGBoost 2.x intercept",
    "p90": "P90 rule, gain",
    "class_weight": "max rule, class weights",
    "downsample": "max rule, 1:1 downsampling",
    "early_stop": "max rule, early stopping",
    # multi-fit and shadow-free selectors
    "boruta_gain": "Boruta rounds, gain, keep tentative",
    "boruta_gain_strict": "Boruta rounds, gain, accepted only",
    "boruta_perm": "Boruta rounds, held-out permutation, keep tentative",
    "boruta_perm_strict": "Boruta rounds, held-out permutation, accepted only",
    "borutapy": "BorutaPy, keep tentative",
    "borutapy_strict": "BorutaPy, accepted only",
    "borutapy_balanced": "BorutaPy, balanced forest, keep tentative",
    "borutapy_balanced_strict": "BorutaPy, balanced forest, accepted only",
    "oof_perm_pos": "out-of-fold permutation, mean $>0$",
    "oof_perm_2se": "out-of-fold permutation, mean $-$ 2 SE $>0$",
    "knockoff_plus": "model-X knockoffs+, $q$ = 0.2",
    "knockoff": "model-X knockoffs, $q$ = 0.2",
    "l1_cv": "$\\ell_1$-logistic, CV penalty",
    "stability": "stability selection ($\\ell_1$), $E[V] \\le 1$",
    "mwu_bh_0.1": "Mann–Whitney, BH $q$ = 0.1",
    "mwu_bh_0.2": "Mann–Whitney, BH $q$ = 0.2",
}


def load_methods():
    frames = []
    for name in ("e4_mitigations", "e4b_baselines", "e4c_borutapy", "e6_univariate"):
        p = RESULTS / f"{name}.csv.gz"
        if p.exists():
            f = pd.read_csv(p)
            f["source"] = name
            frames.append(f)
    d = pd.concat(frames, ignore_index=True)
    # E6 and E4b both contain the Mann-Whitney rules on identical data; keep E4b's
    if "e4b_baselines" in set(d.source):
        d = d[~((d.source == "e6_univariate") & d.method.isin(set(d[d.source == "e4b_baselines"].method)))]
    d = d[d.method != "naive"]  # identical to E4b's oneshot_gain (same fit)
    return d


def method_summary(d):
    """Per method and n_min: mean FNR and FPR (over p and repeats) and their SEs."""
    g = d.groupby(["method", "n_min"]).agg(
        fnr=("fnr", "mean"), fnr_se=("fnr", se), fpr=("fpr", "mean"), fpr_se=("fpr", se),
        n=("fnr", "size")).reset_index()
    return g


FIG4_PANELS = {
    "one-shot shadow rules (level 1/11)": [
        ("oneshot_gain", OI["blue"], "o", "-"),
        ("oneshot_total_gain", OI["sky"], "s", "-"),
        ("oneshot_shap", OI["green"], "^", "-"),
        ("mwu_shadow_max", OI["black"], "D", "-"),
    ],
    "other selectors": [
        ("boruta_gain", OI["vermillion"], "o", "-"),
        ("boruta_gain_strict", OI["vermillion"], "o", ":"),
        ("borutapy_balanced", OI["purple"], "s", "-"),
        ("oof_perm_2se", OI["orange"], "^", "-"),
        ("knockoff_plus", OI["green"], "v", "-"),
        ("mwu_bh_0.1", OI["black"], "D", "-"),
    ],
}


def fig3_methods():
    g = method_summary(load_methods())
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
    for j, (title, series) in enumerate(FIG4_PANELS.items()):
        for m, col, mk, ls in series:
            s = g[g.method == m].sort_values("n_min")
            if not len(s):
                continue
            for i, (y, e) in enumerate((("fnr", "fnr_se"), ("fpr", "fpr_se"))):
                axes[i, j].errorbar(s.n_min, s[y], yerr=2 * s[e], ls=ls, marker=mk, ms=3,
                                    color=col, lw=1, elinewidth=0.6,
                                    mfc=col if ls == "-" else "white",
                                    label=METHOD_LABELS[m] if i == 0 else None)
        axes[0, j].set(title=f"({'ab'[j]}) {title}", ylim=(-0.03, 1.0))
        axes[1, j].set(xscale="log", xlabel="$n_{\\min}$ ($\\delta$ = 0.5, so $s = n_{\\min}/4$)",
                       ylim=(-0.01, 0.35))
        axes[1, j].axhline(1 / 11, color=OI["grey"], ls="--", lw=0.8)
        axes[1, j].text(11, 1 / 11 + 0.008, "1/(K+1) = 1/11", fontsize=6.5, color=OI["grey"])
        h, lab = axes[0, j].get_legend_handles_labels()
        axes[1, j].legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.3), fontsize=6.5,
                          ncol=1, frameon=False)
    axes[0, 0].set_ylabel("FNR (relevant dropped)")
    axes[1, 0].set_ylabel("FPR (irrelevant kept)")
    for ax in axes[0]:
        ax.set_xscale("log")
    fig.tight_layout()
    save(fig, "fig3_methods")


TABLE2_ROWS = [
    ("One fit, max-shadow rule ($K = 10$)", [
        "oneshot_gain", "oneshot_total_gain", "oneshot_shap", "mwu_shadow_max",
        "class_weight", "downsample", "early_stop", "p90"]),
    ("Repeated rounds (20 fits) and Boruta", [
        "boruta_gain", "boruta_gain_strict", "boruta_perm", "boruta_perm_strict",
        "borutapy", "borutapy_strict", "borutapy_balanced", "borutapy_balanced_strict"]),
    ("Without shadows", [
        "oof_perm_pos", "oof_perm_2se", "knockoff_plus", "knockoff", "l1_cv", "stability",
        "mwu_bh_0.1", "mwu_bh_0.2"]),
]


def table_methods():
    d = load_methods()
    g = method_summary(d)
    sec = d.groupby("method").seconds.median() if "seconds" in d else pd.Series(dtype=float)
    nmins = (10, 25, 50, 100, 250)
    lines = []
    for head, methods in TABLE2_ROWS:
        lines.append(rf"\multicolumn{{{len(nmins) + 2}}}{{l}}{{\emph{{{head}}}}} \\")
        for m in methods:
            s = g[g.method == m].set_index("n_min")
            if not len(s):
                continue
            cells = []
            for k in nmins:
                if k in s.index:
                    cells.append(f"{s.loc[k, 'fnr']:.2f} / {s.loc[k, 'fpr']:.2f}")
                else:
                    cells.append("--")
            t = sec.get(m, np.nan)
            ts = "--" if not np.isfinite(t) else (f"{t:.2f}" if t < 10 else f"{t:.0f}")
            lines.append(f"\\quad {METHOD_LABELS[m]} & " + " & ".join(cells) + f" & {ts} \\\\")
    write_tex("table_methods", "\n".join(lines) + "\n")
    # standard errors, for the text
    g.to_csv(RESULTS / "methods_summary.csv", index=False)


WIDE_ROWS = [
    ("oneshot_gain_K10", "max rule, $K = 10$"),
    ("oneshot_gain_K100", "max rule, $K = 100$"),
    ("boruta_K10_bonferroni", "rounds, $K = 10$, Bonferroni"),
    ("boruta_K10_bh", "rounds, $K = 10$, BH"),
    ("boruta_K10_bh_strict", "rounds, $K = 10$, BH, accepted only"),
    ("boruta_K100_bonferroni", "rounds, $K = 100$, Bonferroni (Boruta's $K$)"),
    ("boruta_K100_bh", "rounds, $K = 100$, BH"),
    ("mwu_bh_0.1", "Mann--Whitney, BH $q = 0.1$"),
]


def table_wide():
    d = pd.read_csv(RESULTS / "e4w_wide.csv.gz")
    g = d.groupby(["method", "n_min"])[["fnr", "fpr"]].mean()
    nmins = sorted(d.n_min.unique())
    lines = []
    for m, lab in WIDE_ROWS:
        cells = [f"{g.loc[(m, k), 'fnr']:.2f} / {g.loc[(m, k), 'fpr']:.3f}" for k in nmins]
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    write_tex("table_wide", "\n".join(lines) + "\n")


# --------------------------------------------------------------------- E9 robustness, E11 real data
DGP_LABELS = {
    "gauss": "Gaussian (baseline)",
    "corr_rel": "correlated relevant ($\\rho$ = 0.6)",
    "proxy": "proxies of relevant ($\\rho$ = 0.9)",
    "t3": "heavy tails ($t_3$)",
    "lognormal": "skewed (log-normal)",
    "mixed_binary": "mixed binary / continuous",
    "scale": "scale only (sd 1.6)",
    "interaction": "interaction only",
}
ROBUST_METHODS = ["oneshot_gain", "oneshot_shap", "boruta_gain", "oof_perm_2se", "mwu_bh_0.1"]


def table_robustness(n_min=100):
    d = pd.read_csv(RESULTS / "e9_robustness.csv.gz")
    d = d[d.n_min == n_min]
    lines = []
    for dgp, lab in DGP_LABELS.items():
        s = d[d.dgp == dgp]
        cells = []
        for m in ROBUST_METHODS:
            t = s[s.method == m]
            rel = t.keep_inter.mean() if dgp == "interaction" else t.keep_rel.mean()
            cells.append(f"{1 - rel:.2f} / {t.keep_null.mean():.2f}")
        lines.append(f"{lab} & " + " & ".join(cells) + r" \\")
    write_tex("table_robustness", "\n".join(lines) + "\n")


REAL_METHODS = ["oneshot_gain", "oneshot_shap", "boruta_gain", "boruta_gain_strict",
                "borutapy_balanced", "oof_perm_2se", "mwu_bh_0.1", "all_features"]


def table_realdata():
    d = pd.read_csv(RESULTS / "e11_real_data.csv.gz")
    base = d[d.method == "all_features"].set_index(["dataset", "n_min", "p", "repeat"]).pr_auc
    d = d.join(base.rename("pr_all"), on=["dataset", "n_min", "p", "repeat"])
    d["rel_pr"] = d.pr_auc / d.pr_all
    lines = []
    for n_min in (25, 50, 100, 250):
        s = d[d.n_min == n_min]
        cells = []
        for m in REAL_METHODS:
            t = s[s.method == m]
            if not len(t):
                cells.append("--")
                continue
            if m == "all_features":
                cells.append(f"{t.pr_auc.mean():.3f}")
            else:
                cells.append(f"{t.fnr.mean():.2f} / {t.fpr.mean():.2f} / {t.rel_pr.mean():.2f}")
        lines.append(f"{n_min} & " + " & ".join(cells) + r" \\")
    write_tex("table_realdata", "\n".join(lines) + "\n")


if __name__ == "__main__":
    import sys

    jobs = [fig1_first_split, figA1_intercept_toll, fig2_scaling, figA2_fnr_vs_nmin, table_e2, table_sbands,
            figA3_rounds, figA4_mechanism, figA5_backends, figA6_shadows,
            fig3_methods, table_methods, table_wide, table_robustness, table_realdata]
    only = set(sys.argv[1:])
    for f in jobs:
        if only and f.__name__ not in only:
            continue
        try:
            f()
        except FileNotFoundError as e:
            print("skip", f.__name__, e)
