"""E11: semi-synthetic benchmark on real tables (paper: real data).

Five public tables from the Penn Machine Learning Benchmarks (PMLB): adult
(income), churn (telecom churn), coil2000 (insurance purchase), magic
(gamma-ray telescope) and spambase. The minority class is labelled 1.

Protocol
  1. Append 10 known-null columns: each a row-permutation of a randomly chosen
     real column (so marginals and cardinality are realistic).
  2. Pseudo-truth on the full table. A real column is "relevant" if a
     Mann-Whitney test with Benjamini-Hochberg control (q = 0.001) or a
     Boruta-style selection (XGBoost gain, 10 shadows, 40 rounds, accepted)
     finds it; the two sets are recorded separately. Real columns neither
     method finds are excluded from both error rates.
  3. Split 70/30 (stratified). From the 70% pool draw n_min positives and
     n_min (1 - p) / p negatives, for n_min in {25, 50, 100, 250} and
     p in {0.1, 0.02} where the pool allows it; 10 draws per cell.
  4. Run each method on the subsample. FNR is measured on the relevant real
     columns, FPR on the 10 appended nulls.
  5. Downstream: refit XGBoost (prevalence intercept) on the subsample using
     the kept columns and report PR-AUC (average precision) on the fixed 30%
     test split.

Data are downloaded once into experiments/data_cache/ (not version
controlled) and checked against SHA-256 digests.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import urllib.request
import warnings
from pathlib import Path

import methods as M
import numpy as np
import pandas as pd
from common import run_grid, xgb
from scipy import stats
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split

CACHE = Path(__file__).resolve().parent / "data_cache"
URL = "https://media.githubusercontent.com/media/EpistasisLab/pmlb/master/datasets/{0}/{0}.tsv.gz"
SHA256 = {
    "adult": "fb8bc76eafa39558b6044907f23416faf99099cefb8afd06a2cf5afbdae1b506",
    "churn": "7a8bf22bd4c0845187d4d6eb39b215181ba149834b851430a68052a7a3a86bcc",
    "coil2000": "849976657dc371578819f225ce81eb9a76bd084ce743b5ff6753e415567c68dd",
    "magic": "c823ea26d676538f9194d41f290798ff2f3fe59255c340418fd9439d8fc9c07f",
    "spambase": "735e8d43f7ee78ac54ca17404510e95f4c8b364a721e4d9532a1b3221ed70961",
}
N_NULL = 10


def load(name):
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"{name}.tsv.gz"
    if not path.exists():
        urllib.request.urlretrieve(URL.format(name), path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SHA256[name]:
        raise RuntimeError(f"{name}: SHA-256 mismatch ({digest})")
    df = pd.read_csv(path, sep="\t")
    y = df.pop("target").to_numpy()
    minority = pd.Series(y).value_counts().idxmin()
    y = (y == minority).astype(np.int8)
    X = df.to_numpy(dtype=float)
    rng = np.random.default_rng(int(digest[:8], 16))
    src = rng.choice(X.shape[1], size=N_NULL, replace=X.shape[1] < N_NULL)
    nulls = np.column_stack([X[rng.permutation(len(X)), j] for j in src])
    return np.hstack([X, nulls]), y, list(df.columns)


def pseudo_truth(name):
    """Relevant real columns on the full table (cached)."""
    path = CACHE / f"{name}_truth.json"
    if path.exists():
        return json.loads(path.read_text())
    X, y, cols = load(name)
    d = len(cols)
    Xr = X[:, :d]
    pv = stats.mannwhitneyu(Xr[y == 1], Xr[y == 0], axis=0).pvalue
    mw = M.bh_adjust(np.nan_to_num(pv, nan=1.0)) < 0.001
    _, acc, _ = M.boruta_rounds(Xr, y, seed=12345, n_iter=40, n_shadows=10)
    truth = {"mw": np.flatnonzero(mw).tolist(), "boruta": np.flatnonzero(acc).tolist(), "d": d}
    path.write_text(json.dumps(truth))
    return truth


def pr_auc(Xtr, ytr, Xte, yte, keep, seed):
    if not keep.any():
        return float(yte.mean())  # no features: the base rate
    m = xgb(random_state=seed, base_score=float(ytr.mean())).fit(Xtr[:, keep], ytr)
    return float(average_precision_score(yte, m.predict_proba(Xte[:, keep])[:, 1]))


def worker(s, r, seed):
    X, y, cols = load(s["dataset"])
    truth = pseudo_truth(s["dataset"])
    d = truth["d"]
    rel = np.zeros(X.shape[1], bool)
    rel[truth["mw"]] = True
    rel[truth["boruta"]] = True
    mw_only = np.zeros_like(rel)
    mw_only[truth["mw"]] = True
    bo = np.zeros_like(rel)
    bo[truth["boruta"]] = True
    null = np.zeros_like(rel)
    null[d:] = True
    Xpool, Xte, ypool, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    rng = np.random.default_rng(seed)
    n_neg = int(round(s["n_min"] * (1 - s["p"]) / s["p"]))
    idx = np.concatenate(
        [
            rng.choice(np.flatnonzero(ypool == 1), s["n_min"], replace=False),
            rng.choice(np.flatnonzero(ypool == 0), n_neg, replace=False),
        ]
    )
    Xs, ys = Xpool[idx], ypool[idx]
    res = {"all_features": (np.ones(X.shape[1], bool), 0.0)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out, t = M.timed(M.one_shot_xgb, Xs, ys, seed, n_shadows=10)
        res["oneshot_gain"] = (out["gain"][0], t)
        res["oneshot_shap"] = (out["shap"][0], t)
        (keep, acc, _), t = M.timed(M.boruta_rounds, Xs, ys, seed, n_shadows=10)
        res["boruta_gain"] = (keep, t)
        res["boruta_gain_strict"] = (acc, t)
        (_, two, _), t = M.timed(M.oof_permutation, Xs, ys, seed)
        res["oof_perm_2se"] = (two, t)
        out, t = M.timed(M.mann_whitney, np.nan_to_num(Xs), ys, seed)
        res["mwu_bh_0.1"] = (out["mwu_bh_0.1"][0], t)
        if len(ys) <= 2500:
            (keep, acc, _), t = M.timed(M.borutapy, Xs, ys, seed, class_weight="balanced_subsample")
            res["borutapy_balanced"] = (keep, t)
    rows = []
    for name, (keep, t) in res.items():
        rows.append(
            dict(
                method=name,
                fnr=float(1 - keep[rel].mean()) if rel.any() else np.nan,
                fnr_mw=float(1 - keep[mw_only].mean()) if mw_only.any() else np.nan,
                fnr_boruta=float(1 - keep[bo].mean()) if bo.any() else np.nan,
                fpr=float(keep[null].mean()),
                n_kept=int(keep.sum()),
                keep="".join("1" if k else "0" for k in keep),
                pr_auc=pr_auc(Xs, ys, Xte, yte, keep, seed),
                test_prevalence=float(yte.mean()),
                seconds=t,
            )
        )
    return rows


def scenarios(quick):
    out = []
    for name in SHA256:
        X, y, _ = load(name)
        n_pos_pool = int(0.7 * y.sum())
        n_neg_pool = int(0.7 * (len(y) - y.sum()))
        for n_min, p in itertools.product((25, 50, 100, 250), (0.1, 0.02)):
            n_neg = int(round(n_min * (1 - p) / p))
            if n_min <= n_pos_pool and n_neg <= n_neg_pool:
                out.append(dict(dataset=name, n_min=n_min, p=p, n=n_min + n_neg,
                                n_repeats=1 if quick else 10))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="tiny profile; writes *_quick files")
    quick = ap.parse_args().quick
    for name in SHA256:
        print(name, pseudo_truth(name), flush=True)
    run_grid("e11_real_data", scenarios(quick), worker)
