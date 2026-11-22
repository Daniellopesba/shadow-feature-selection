"""E4b: further baselines on the E4 datasets (paper Section 5).

Same scenarios and seeds as E4 and E6, so every method sees identical data
(5 relevant candidates at delta = 0.5, 5 irrelevant candidates, 10 permuted
shadows where a method uses shadows). Methods:

  oneshot_gain / _total_gain / _shap   one XGBoost fit, max-shadow rule, three
                                       importances from the same fit (the
                                       "gain" arm reproduces E4's "naive")
  oof_perm_pos / oof_perm_2se          5-fold out-of-fold permutation importance
  knockoff_plus / knockoff             Gaussian model-X knockoffs, q = 0.2
  l1_cv                                L1-logistic regression, CV-chosen penalty
  stability                            stability selection with L1-logistic
  mwu_*                                Mann-Whitney (as E6; recomputed for scores)

Every row stores the keep-mask, a per-candidate score (for FPR-matched
comparisons) and the wall time of the method.
"""

from __future__ import annotations

import argparse
import warnings

import methods as M
from common import dataset, run_grid
from e4_mitigations import N_REAL, scenarios


def fmt(a):
    return " ".join(f"{float(v):.6g}" for v in a)


def worker(s, r, seed):
    X, y = dataset(s, seed)
    rows = []

    def add(name, keep, score, sec):
        rows.append(
            dict(
                method=name,
                fnr=float(1 - keep[:N_REAL].mean()),
                fpr=float(keep[N_REAL:].mean()),
                keep="".join("1" if k else "0" for k in keep),
                score=fmt(score),
                seconds=sec,
            )
        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, t = M.timed(M.one_shot_xgb, X, y, seed)
        for kind, (keep, sc) in out.items():
            add(f"oneshot_{kind}", keep, sc, t)
        (pos, two, z), t = M.timed(M.oof_permutation, X, y, seed)
        add("oof_perm_pos", pos, z, t)
        add("oof_perm_2se", two, z, t)
        (kp, k0, W), t = M.timed(M.knockoffs_xgb, X, y, seed)
        add("knockoff_plus", kp, W, t)
        add("knockoff", k0, W, t)
        (keep, coef), t = M.timed(M.l1_logistic_cv, X, y, seed)
        add("l1_cv", keep, coef, t)
        (keep, prob), t = M.timed(M.stability_selection, X, y, seed)
        add("stability", keep, prob, t)
        out, t = M.timed(M.mann_whitney, X, y, seed)
        for name, (keep, sc) in out.items():
            add(name, keep, sc, t)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    run_grid("e4b_baselines", scenarios(ap.parse_args().quick), worker)
