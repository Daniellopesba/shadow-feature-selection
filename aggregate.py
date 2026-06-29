"""Cross-scenario aggregation: figures + summary CSV for the paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REGIME_COLOR = {"A": "#1f77b4", "B": "#d62728"}
P_MARKER = {
    0.001: "v",
    0.005: "^",
    0.01: "s",
    0.05: "o",
    0.10: "D",
    0.20: "P",
}

FIG_DIR = Path("figures")


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        params = r["params"]
        agg = r["aggregate"]
        rows.append(
            {
                "scenario_id": r["scenario_id"],
                "regime": params["regime"],
                "n": params["n"],
                "p": params["p"],
                "delta": params["delta"],
                "n_real": params["n_real"],
                "n_noise": params["n_noise"],
                "control": params["control"],
                "backend": params["backend"],
                "n_repeats": params["n_repeats"],
                "n_min": r["n_min"],
                "naive_floor_mean": agg["naive_floor"]["mean"],
                "naive_floor_std": agg["naive_floor"]["std"],
                "p90_floor_mean": agg["p90_floor"]["mean"],
                "mean_real_importance_mean": agg["mean_real_importance"]["mean"],
                "fnr_max_mean": agg["fnr_max"]["mean"],
                "fnr_max_std": agg["fnr_max"]["std"],
                "fnr_p90_mean": agg["fnr_p90"]["mean"],
            }
        )
    return pd.DataFrame(rows)


def save_fig(fig: plt.Figure, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.png", dpi=160, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_metric_vs_nmin(
    df: pd.DataFrame, metric: str, ylabel: str, title: str
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (regime, p), sub in df.groupby(["regime", "p"]):
        sub = sub.sort_values("n_min")
        ax.plot(
            sub["n_min"],
            sub[metric],
            marker=P_MARKER.get(p, "o"),
            color=REGIME_COLOR.get(regime, "gray"),
            linestyle="-",
            linewidth=1.2,
            markersize=6,
            label=f"regime {regime}, p={p}",
            alpha=0.85,
        )
    ax.set_xscale("log")
    ax.set_xlabel("n_min  =  round(n · p)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    return fig


def plot_collapse(df: pd.DataFrame) -> None:
    primary = df[
        (df["control"] == False)
        & (df["backend"] == "xgboost")
        & (df["n_noise"] == 5)
    ]
    if primary.empty:
        return
    for d in sorted(primary["delta"].unique()):
        sub = primary[primary["delta"] == d]
        fig = _plot_metric_vs_nmin(
            sub,
            "fnr_max_mean",
            "FNR (naive shadow floor)",
            f"Collapse: FNR_max vs n_min   (δ={d})",
        )
        save_fig(fig, f"01_collapse_fnr_max_delta_{d}")


def plot_noise_floor_inflation(df: pd.DataFrame) -> None:
    primary = df[
        (df["control"] == False)
        & (df["backend"] == "xgboost")
        & (df["n_noise"] == 5)
        & (df["delta"] == 0.5)
    ]
    if primary.empty:
        return
    fig = _plot_metric_vs_nmin(
        primary,
        "naive_floor_mean",
        "normalized noise floor (max over noise features)",
        "Noise floor inflation vs n_min  (δ=0.5)",
    )
    save_fig(fig, "02_noise_floor_inflation")


def plot_mechanism(df: pd.DataFrame) -> None:
    base = df[
        (df["backend"] == "xgboost")
        & (df["n_noise"] == 5)
        & (df["delta"] == 0.5)
    ]
    signal = base[base["control"] == False].sort_values("n_min")
    control = base[base["control"] == True].sort_values("n_min")
    if signal.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(
        signal["n_min"],
        signal["mean_real_importance_mean"],
        "o-",
        color="green",
        label="mean real-feature importance (signal)",
    )
    ax.plot(
        signal["n_min"],
        signal["naive_floor_mean"],
        "s-",
        color="red",
        label="noise floor (signal)",
    )
    if not control.empty:
        ax.plot(
            control["n_min"],
            control["naive_floor_mean"],
            "x--",
            color="gray",
            label="noise floor (pure-noise control)",
        )
    ax.set_xscale("log")
    ax.set_xlabel("n_min")
    ax.set_ylabel("normalized importance")
    ax.set_title(
        "Mechanism: real-feature collapse vs noise-share inflation (δ=0.5)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "03_mechanism_decomposition")


def plot_mitigation(df: pd.DataFrame) -> None:
    base = df[
        (df["control"] == False)
        & (df["backend"] == "xgboost")
        & (df["n_noise"] == 5)
        & (df["delta"] == 0.5)
    ].sort_values("n_min")
    if base.empty:
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for regime, sub in base.groupby("regime"):
        ax.plot(
            sub["n_min"],
            sub["fnr_max_mean"],
            "o-",
            color=REGIME_COLOR.get(regime, "gray"),
            label=f"FNR_max  regime {regime}",
        )
        ax.plot(
            sub["n_min"],
            sub["fnr_p90_mean"],
            "s--",
            color=REGIME_COLOR.get(regime, "gray"),
            alpha=0.6,
            label=f"FNR_p90  regime {regime}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("n_min")
    ax.set_ylabel("FNR")
    ax.set_title(
        "Mitigation: naive (max) vs P90 noise-floor thresholds (δ=0.5)"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "04_mitigation_p90")


def plot_heatmaps(df: pd.DataFrame) -> None:
    base = df[
        (df["control"] == False)
        & (df["backend"] == "xgboost")
        & (df["n_noise"] == 5)
        & (df["delta"] == 0.5)
    ]
    for regime, sub in base.groupby("regime"):
        pivot = sub.pivot_table(
            index="n", columns="p", values="fnr_max_mean", aggfunc="mean"
        )
        if pivot.empty:
            continue
        fig, ax = plt.subplots(figsize=(5.5, 4))
        im = ax.imshow(
            pivot.values, aspect="auto", cmap="magma", vmin=0.0, vmax=1.0
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{c:g}" for c in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{r:g}" for r in pivot.index])
        ax.set_xlabel("p")
        ax.set_ylabel("n")
        ax.set_title(f"FNR_max heatmap — regime {regime} (δ=0.5)")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = pivot.values[i, j]
                if not np.isnan(v):
                    ax.text(
                        j,
                        i,
                        f"{v:.2f}",
                        ha="center",
                        va="center",
                        color="white" if v < 0.6 else "black",
                        fontsize=9,
                    )
        fig.colorbar(im, ax=ax, label="FNR_max")
        fig.tight_layout()
        save_fig(fig, f"05_heatmap_regime_{regime}")


def plot_signal_interaction(df: pd.DataFrame) -> None:
    base = df[
        (df["control"] == False)
        & (df["backend"] == "xgboost")
        & (df["n_noise"] == 5)
    ]
    deltas = sorted(base["delta"].unique())
    if not deltas:
        return
    fig, axes = plt.subplots(
        1, len(deltas), figsize=(5 * len(deltas), 4), sharey=True
    )
    if len(deltas) == 1:
        axes = [axes]
    for ax, d in zip(axes, deltas):
        sub = base[base["delta"] == d]
        for (regime, p), grp in sub.groupby(["regime", "p"]):
            grp = grp.sort_values("n_min")
            ax.plot(
                grp["n_min"],
                grp["fnr_max_mean"],
                marker=P_MARKER.get(p, "o"),
                color=REGIME_COLOR.get(regime, "gray"),
                linewidth=1.2,
                label=f"{regime}, p={p}",
                alpha=0.85,
            )
        ax.set_xscale("log")
        ax.set_xlabel("n_min")
        ax.set_title(f"δ = {d}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("FNR_max")
    axes[-1].legend(fontsize=7, loc="best")
    fig.suptitle("Signal × imbalance interaction")
    fig.tight_layout()
    save_fig(fig, "06_signal_interaction")


def plot_robustness(df: pd.DataFrame) -> None:
    n20 = df[
        (df["control"] == False)
        & (df["n_noise"] == 20)
        & (df["backend"] == "xgboost")
        & (df["delta"] == 0.5)
    ]
    if not n20.empty:
        fig = _plot_metric_vs_nmin(
            n20,
            "fnr_max_mean",
            "FNR_max (n_noise=20)",
            "Robustness: same collapse with n_noise=20  (δ=0.5)",
        )
        save_fig(fig, "07a_robustness_n_noise_20")

    sk = df[
        (df["control"] == False)
        & (df["backend"] == "sklearn_gbm")
        & (df["delta"] == 0.5)
    ]
    if not sk.empty:
        fig = _plot_metric_vs_nmin(
            sk,
            "fnr_max_mean",
            "FNR_max (sklearn GBM)",
            "Robustness: sklearn GBM cross-check  (δ=0.5)",
        )
        save_fig(fig, "07b_robustness_sklearn_gbm")


def write_summary_table(df: pd.DataFrame) -> Path:
    base = df[
        (df["control"] == False)
        & (df["backend"] == "xgboost")
        & (df["n_noise"] == 5)
        & (df["delta"] == 0.5)
    ].copy()
    bins = [0, 25, 75, 200, 500, 1500, np.inf]
    labels = ["<25", "25-74", "75-199", "200-499", "500-1499", "1500+"]
    base["n_min_bin"] = pd.cut(
        base["n_min"], bins=bins, labels=labels, right=False
    )
    grouped = (
        base.groupby("n_min_bin", observed=True)
        .agg(
            n_scenarios=("scenario_id", "count"),
            n_min_min=("n_min", "min"),
            n_min_max=("n_min", "max"),
            mean_noise_floor=("naive_floor_mean", "mean"),
            mean_fnr_max=("fnr_max_mean", "mean"),
            mean_fnr_p90=("fnr_p90_mean", "mean"),
        )
        .reset_index()
    )

    def recommend(row):
        if pd.isna(row["mean_fnr_max"]):
            return "(no data)"
        if row["mean_fnr_max"] < 0.05:
            return "shadow heuristic OK"
        if row["mean_fnr_p90"] < 0.10:
            return "use P90 floor (multi-shadow)"
        return (
            "shadow approach unreliable; prefer stratified CV w/ direct "
            "gain ranking"
        )

    grouped["recommendation"] = grouped.apply(recommend, axis=1)
    out = FIG_DIR / "summary_table.csv"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(out, index=False)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="smoke", help="(metadata only)")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    df = load_results(Path(args.results_dir))
    if df.empty:
        print("[aggregate] no result files found")
        return 1
    print(f"[aggregate] loaded {len(df)} scenarios from {args.results_dir}/")

    plot_collapse(df)
    plot_noise_floor_inflation(df)
    plot_mechanism(df)
    plot_mitigation(df)
    plot_heatmaps(df)
    plot_signal_interaction(df)
    plot_robustness(df)
    out = write_summary_table(df)
    print(f"[aggregate] wrote summary table: {out}")
    print(f"[aggregate] figures in: {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
