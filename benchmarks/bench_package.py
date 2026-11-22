"""Runtime and memory benchmarks for shadowselect.

Measures, on synthetic data from ``shadowselect.simulation.make_dataset``:

1. one selector round with each learner (default forest, XGBoost, LightGBM);
2. building one round's design matrix (real columns + shadows, shuffled);
3. ``UnivariateScreen.fit`` time and peak memory.

Peak memory is the peak of Python-tracked allocations (``tracemalloc``;
NumPy reports its buffers there), above what was allocated before the call.

Run:  python benchmarks/bench_package.py            (about 5 minutes on 4 cores)
      python benchmarks/bench_package.py --quick    (seconds; used in CI)
"""

from __future__ import annotations

import argparse
import platform
import time
import tracemalloc
import warnings

import numpy as np

import shadowselect
from shadowselect import ShadowFeatureSelector, UnivariateScreen
from shadowselect.selector import _default_estimator
from shadowselect.simulation import make_dataset


def measure(fn):
    tracemalloc.start()
    base = tracemalloc.get_traced_memory()[0]
    t = time.perf_counter()
    out = fn()
    sec = time.perf_counter() - t
    peak = tracemalloc.get_traced_memory()[1] - base
    tracemalloc.stop()
    return out, sec, peak / 1e9


def learners():
    out = {"default forest (200 trees, depth 7, balanced)": _default_estimator(n_jobs=-1)}
    try:
        from xgboost import XGBClassifier

        out["XGBoost (300 trees, depth 6)"] = XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, verbosity=0
        )
    except ImportError:
        pass
    try:
        from lightgbm import LGBMClassifier

        out["LightGBM (300 trees, 15 leaves)"] = LGBMClassifier(
            n_estimators=300, num_leaves=15, min_child_samples=50, verbose=-1
        )
    except ImportError:
        pass
    return out


def bench_rounds(n, d):
    X, y = make_dataset(n=n, p=0.01, delta=0.5, n_real=10, n_noise=d - 10, seed=0)
    print(f"\n## One selector round, {n:,} x {d} (+10 shadows), 1% positives")
    print("| learner | seconds per round | 20 rounds |")
    print("|---|---:|---:|")
    for name, est in learners().items():
        sel = ShadowFeatureSelector(est, n_iter=20, random_state=0)
        sel.n_shadows_ = 10
        sel._importance_mode = "native"
        _, sec, _ = measure(lambda sel=sel, est=est: sel._one_round(est, X, "dense", y, {}, 0))
        print(f"| {name} | {sec:.1f} | {20 * sec / 60:.1f} min |")


def bench_round_matrix(n, d):
    X, _ = make_dataset(n=n, p=0.01, delta=0.5, n_real=10, n_noise=d - 10, seed=0)
    X = np.asfortranarray(X)
    sel = ShadowFeatureSelector()
    rng = np.random.default_rng(0)
    src = rng.choice(d, 10, replace=False)
    pos = rng.permutation(d + 10)
    _, sec, peak = measure(lambda: sel._round_matrix(X, "dense", src, pos, rng, None))
    print(f"\n## Building one round's matrix, {n:,} x {d} float64 ({X.nbytes / 1e9:.2f} GB)")
    print(f"{sec:.2f} s, {peak:.2f} GB allocated (one buffer, reused across rounds)")


def bench_screen(n, d):
    X, y = make_dataset(n=n, p=0.01, delta=0.3, n_real=10, n_noise=d - 10, seed=0)
    _, sec, peak = measure(lambda: UnivariateScreen().fit(X, y))
    print(f"\n## UnivariateScreen.fit, {n:,} x {d} ({X.nbytes / 1e9:.2f} GB)")
    print(f"{sec:.1f} s, peak {peak:.2f} GB above the input")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true", help="small sizes, for CI")
    args = ap.parse_args()
    warnings.simplefilter("ignore")
    print(
        f"shadowselect {shadowselect.__version__}, Python {platform.python_version()}, "
        f"{platform.machine()}"
    )
    if args.quick:
        bench_rounds(5_000, 20)
        bench_round_matrix(20_000, 50)
        bench_screen(20_000, 50)
    else:
        bench_rounds(100_000, 100)
        bench_round_matrix(500_000, 200)
        bench_screen(200_000, 200)


if __name__ == "__main__":
    main()
