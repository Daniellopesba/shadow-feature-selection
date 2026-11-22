"""E4w: a wide table, and the number of shadows Boruta uses (paper: selection methods).

100 candidates (5 relevant at delta = 0.5, 95 irrelevant), p = 0.05,
n_min in {25, 50, 100, 250}. Compares K = 10 shadows with Boruta's default of
one shadow per candidate (K = 100), and Bonferroni with Benjamini-Hochberg
for the per-feature binomial decisions (20 rounds, XGBoost gain, prevalence
intercept), next to the one-shot rule and the Mann-Whitney screen.
"""

from __future__ import annotations

import argparse
import warnings

import methods as M
from common import run_grid
from scipy import stats

from shadowselect.simulation import make_dataset

N_REAL, D = 5, 100


def worker(s, r, seed):
    X, y = make_dataset(s["n"], s["p"], s["delta"], N_REAL, D - N_REAL, seed)
    rows = []

    def add(name, keep, sec):
        rows.append(
            dict(
                method=name,
                fnr=float(1 - keep[:N_REAL].mean()),
                fpr=float(keep[N_REAL:].mean()),
                n_kept_null=int(keep[N_REAL:].sum()),
                seconds=sec,
            )
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for K in (10, D):
            out, t = M.timed(M.one_shot_xgb, X, y, seed, n_shadows=K, shap=False)
            add(f"oneshot_gain_K{K}", out["gain"][0], t)
            (_, _, hits), t = M.timed(M.boruta_rounds, X, y, seed, n_shadows=K)
            p_acc = stats.binom.sf(hits - 1, 20, 0.5)
            p_rej = stats.binom.cdf(hits, 20, 0.5)
            for corr in ("bonferroni", "bh"):
                if corr == "bonferroni":
                    acc, rej = p_acc < 0.05 / D, p_rej < 0.05 / D
                else:
                    acc, rej = M.bh_adjust(p_acc) < 0.05, M.bh_adjust(p_rej) < 0.05
                add(f"boruta_K{K}_{corr}", ~rej, t)
                add(f"boruta_K{K}_{corr}_strict", acc, t)
        out, t = M.timed(M.mann_whitney, X, y, seed)
        add("mwu_bh_0.1", out["mwu_bh_0.1"][0], t)
    return rows


def scenarios(quick):
    return [
        dict(n=int(round(m / 0.05)), p=0.05, n_min=m, delta=0.5, n_repeats=1 if quick else 10)
        for m in (25, 50, 100, 250)
    ]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e4w_wide", scenarios(ap.parse_args().quick), worker)
