"""E2: the main sweep (paper Section 4.1-4.3).

Single-shot shadow rule on XGBoost training-set importance across a
credit-risk regime (A) and a fraud regime (B) that overlap in n_min, with
three effect sizes and a pure-noise control (delta = 0).

Factor ``intercept``:
  * ``auto``  -- XGBoost 2.1's default base_score estimate. In 2.1 this is a
    single Newton step from 0.5 and lands near 0.12 for any p << 0.1.
  * ``prior`` -- base_score set to the empirical prevalence n_min / n,
    which is what the first-split theory assumes.

Both intercepts see the same datasets (common random numbers). Run under
XGBoost >= 3 (see requirements-xgb3.txt) to write ``e2_sweep_xgb3``: the same
datasets with XGBoost 3's own default intercept.

The noise features in the data play the role of the shadows here.
"""

from __future__ import annotations

import argparse
import itertools

from common import dataset, run_grid, xgb, xgb_importances

from shadowselect.simulation import selection_metrics

REGIMES = {
    "A": dict(ns=[500, 1000, 2500, 5000], ps=[0.05, 0.10, 0.20]),
    "B": dict(ns=[10_000, 50_000, 200_000], ps=[0.001, 0.005, 0.01]),
}


def scenarios(quick: bool, n_noise: int = 5, deltas=(0.0, 0.3, 0.5, 1.0)):
    out = []
    for regime, g in REGIMES.items():
        for n, p, delta, icpt in itertools.product(g["ns"], g["ps"], deltas, ("auto", "prior")):
            n_min = int(round(n * p))
            if n_min < 5:
                continue
            reps = 3 if quick else (25 if n >= 50_000 else 40)
            out.append(
                dict(
                    regime=regime,
                    n=n,
                    p=p,
                    n_min=n_min,
                    delta=delta,
                    n_real=5,
                    n_noise=n_noise,
                    intercept=icpt,
                    n_repeats=reps,
                )
            )
    return out


def worker(s, r, seed):
    X, y = dataset(s, seed)
    kw = {"base_score": float(y.mean())} if s["intercept"] == "prior" else {}
    m = xgb(random_state=seed, **kw).fit(X, y)
    d = s["n_real"] + s["n_noise"]
    rows = []
    for kind, raw in xgb_importances(m, d).items():
        met = selection_metrics(raw, s["n_real"], s["n_noise"])
        real, noise = raw[: s["n_real"]], raw[s["n_real"] :]
        rows.append(
            dict(
                importance=kind,
                raw_real_mean=float(real.mean()),
                raw_noise_max=float(noise.max()),
                **met,
            )
        )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    ap.add_argument("--n-noise", type=int, default=5)
    a = ap.parse_args()
    import xgboost

    if int(xgboost.__version__.split(".")[0]) >= 3:
        # XGBoost 3.x: its own default intercept only, on the same datasets
        sc = [s for s in scenarios(a.quick, a.n_noise) if s["intercept"] == "auto"]
        run_grid("e2_sweep_xgb3", sc, worker, seed_exclude=("intercept",))
    elif a.n_noise == 5:
        run_grid("e2_sweep", scenarios(a.quick), worker, seed_exclude=("intercept",))
    else:  # robustness: more shadows, delta = 0.5, prior intercept only
        sc = [s for s in scenarios(a.quick, a.n_noise, deltas=(0.5,)) if s["intercept"] == "prior"]
        run_grid(f"e2_sweep_noise{a.n_noise}", sc, worker)
