"""E3: where does the failure come from? (paper Section 4.4)

Fixes n_min and varies p (hence n), and reads importances after the first
k boosting rounds of a single 300-round fit. Separates three effects:
  * first-round detectability (k = 1, governed by n_min * delta^2),
  * accumulation of noise-feature gain in later rounds (memorisation of the
    few positives), and
  * the intercept effect (XGBoost 2.1 auto base_score vs the prevalence).
"""

from __future__ import annotations

import argparse
import itertools

from common import dataset, run_grid, xgb

from shadowselect.simulation import selection_metrics

ROUNDS = (1, 3, 10, 30, 100, 300)


def scenarios(quick: bool):
    out = []
    for n_min, p, delta, icpt in itertools.product(
        (25, 100, 500), (0.1, 0.01, 0.001), (0.3, 0.5), ("auto", "prior")
    ):
        n = int(round(n_min / p))
        if n > 200_000:
            continue
        out.append(
            dict(
                n=n,
                p=p,
                n_min=n_min,
                delta=delta,
                n_real=5,
                n_noise=5,
                intercept=icpt,
                n_repeats=3 if quick else 15,
            )
        )
    return out


def worker(s, r, seed):
    X, y = dataset(s, seed)
    kw = {"base_score": float(y.mean())} if s["intercept"] == "prior" else {}
    m = xgb(random_state=seed, **kw).fit(X, y)
    b = m.get_booster()
    d = s["n_real"] + s["n_noise"]
    rows = []
    for k in ROUNDS:
        bk = b[:k]
        for kind in ("gain", "total_gain"):
            sc = bk.get_score(importance_type=kind)
            raw = [sc.get(f"f{j}", 0.0) for j in range(d)]
            met = selection_metrics(raw, s["n_real"], s["n_noise"])
            rows.append(
                dict(
                    rounds=k,
                    importance=kind,
                    raw_real_mean=sum(raw[:5]) / 5,
                    raw_noise_max=max(raw[5:]),
                    **met,
                )
            )
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e3_boosting_rounds", scenarios(ap.parse_args().quick), worker)
