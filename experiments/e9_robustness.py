"""E9: departures from the Gaussian location-shift model (paper Section 6).

Eight data-generating processes (``dgp.py``): the Gaussian baseline,
correlated relevant columns, proxies of relevant columns, heavy-tailed (t3)
and skewed (log-normal) marginals, mixed binary and continuous columns, a
scale-only signal, and an interaction-only signal. delta = 0.5 throughout,
p = 0.05, n_min in {25, 50, 100, 250}. For each method we record the keep
rate of every column role (relevant, interaction, proxy, null).
"""

from __future__ import annotations

import argparse
import itertools
import warnings

import methods as M
import numpy as np
from common import run_grid
from dgp import DGPS, make


def worker(s, r, seed):
    X, y, roles = make(s["dgp"], s["n"], s["p"], s["delta"], seed)
    res = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, t = M.timed(M.one_shot_xgb, X, y, seed)
        for kind, (keep, _) in out.items():
            res[f"oneshot_{kind}"] = keep
        (keep, acc, _), t = M.timed(M.boruta_rounds, X, y, seed)
        res["boruta_gain"] = keep
        res["boruta_gain_strict"] = acc
        (_, two, _), t = M.timed(M.oof_permutation, X, y, seed)
        res["oof_perm_2se"] = two
        for name, (keep, _) in M.mann_whitney(X, y, seed).items():
            res[name] = keep
    rows = []
    for name, keep in res.items():
        row = dict(method=name)
        for role in ("rel", "inter", "proxy", "null"):
            m = roles == role
            row[f"keep_{role}"] = float(keep[m].mean()) if m.any() else np.nan
        rows.append(row)
    return rows


def scenarios(quick):
    out = []
    for dgp, n_min in itertools.product(DGPS, (25, 50, 100, 250)):
        p = 0.05
        out.append(
            dict(
                dgp=dgp,
                n=int(round(n_min / p)),
                p=p,
                n_min=n_min,
                delta=0.5,
                n_repeats=2 if quick else 15,
            )
        )
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e9_robustness", scenarios(ap.parse_args().quick), worker)
