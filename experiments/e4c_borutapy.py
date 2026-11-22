"""E4c: BorutaPy, the Boruta implementation most Python users run (paper Section 5).

Same datasets and seeds as E4 (first 10 repeats; n_min <= 250). BorutaPy 0.4
with iterative removal of rejected features and its two-step correction, on a
shallow random forest (100 trees, depth 5), with and without
``class_weight="balanced_subsample"``. Reports accepted-only and
accepted + tentative selections.
"""

from __future__ import annotations

import argparse

import methods as M
from common import dataset, run_grid
from e4_mitigations import N_REAL
from e4_mitigations import scenarios as e4_scenarios


def worker(s, r, seed):
    X, y = dataset(s, seed)
    rows = []
    for cw in (None, "balanced_subsample"):
        (keep, acc, score), t = M.timed(M.borutapy, X, y, seed, class_weight=cw)
        tag = "borutapy" if cw is None else "borutapy_balanced"
        for name, k in ((tag, keep), (f"{tag}_strict", acc)):
            rows.append(
                dict(
                    method=name,
                    fnr=float(1 - k[:N_REAL].mean()),
                    fpr=float(k[N_REAL:].mean()),
                    keep="".join("1" if v else "0" for v in k),
                    seconds=t,
                )
            )
    return rows


def scenarios(quick):
    out = []
    for s in e4_scenarios(quick):
        if s["n_min"] <= 250:
            out.append({**s, "n_repeats": 1 if quick else 10})
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e4c_borutapy", scenarios(ap.parse_args().quick), worker)
