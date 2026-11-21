"""Shared helpers for the paper's experiments.

Every experiment writes a tidy per-repeat CSV to ``results/<name>.csv.gz``;
``make_figures.py`` turns those into the figures in ``figures/``. Seeds are
derived from the scenario parameters, so reruns are bit-for-bit identical
for the pinned library versions in ``requirements-experiments.txt``.
"""

from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
sys.path.insert(0, str(ROOT / "src"))
# --quick runs write *_quick files so they never overwrite the paper's results
SUFFIX = "_quick" if "--quick" in sys.argv else ""

from shadowselect.simulation import derive_seed, make_dataset  # noqa: E402

# The model studied in the paper. Deliberately "reasonable defaults" with no
# early stopping, as commonly used inside shadow-feature selection loops.
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    eval_metric="logloss",
    verbosity=0,
    n_jobs=1,
)


def xgb(**overrides):
    from xgboost import XGBClassifier

    return XGBClassifier(**{**XGB_PARAMS, **overrides})


def xgb_importances(model, d: int) -> dict[str, np.ndarray]:
    """Raw (unnormalised) importances of every type from a fitted XGBClassifier."""
    booster = model.get_booster()
    out = {}
    for kind in ("gain", "total_gain", "weight"):
        s = booster.get_score(importance_type=kind)
        out[kind] = np.array([s.get(f"f{j}", 0.0) for j in range(d)], dtype=float)
    return out


def scenario_key(params: dict) -> str:
    return json.dumps(params, sort_keys=True)


def run_grid(
    name: str,
    scenarios: Iterable[dict],
    worker: Callable[[dict, int, int], list[dict]],
    n_jobs: int = -1,
) -> pd.DataFrame:
    """Run ``worker(scenario, repeat_idx, seed)`` for every scenario x repeat.

    Each scenario needs an ``n_repeats`` key. Workers return a list of row
    dicts; scenario parameters are merged into every row.
    """
    name += SUFFIX
    scenarios = list(scenarios)
    tasks = [(s, r) for s in scenarios for r in range(s["n_repeats"])]
    # big jobs first for better load balancing
    tasks.sort(key=lambda t: -t[0].get("n", 0))
    t0 = time.time()
    print(f"[{name}] {len(scenarios)} scenarios, {len(tasks)} tasks", flush=True)

    def _one(s, r):
        seed = derive_seed(scenario_key({k: v for k, v in s.items() if k != "n_repeats"}), r)
        rows = worker(s, r, seed)
        return [{**s, "repeat": r, "seed": seed, **row} for row in rows]

    out = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_one)(s, r) for s, r in tasks)
    df = pd.DataFrame([row for rows in out for row in rows])
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.csv.gz"
    df.to_csv(path, index=False)
    meta = {
        "experiment": name,
        "n_scenarios": len(scenarios),
        "n_tasks": len(tasks),
        "wall_seconds": round(time.time() - t0, 1),
        "versions": versions(),
    }
    (RESULTS / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{name}] wrote {path} in {meta['wall_seconds']}s", flush=True)
    return df


def versions() -> dict[str, str]:
    import scipy
    import sklearn

    v = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "cpu_count": str(os.cpu_count()),
    }
    try:
        import xgboost

        v["xgboost"] = xgboost.__version__
    except ImportError:
        pass
    return v


def dataset(s: dict, seed: int):
    return make_dataset(s["n"], s["p"], s["delta"], s["n_real"], s["n_noise"], seed)
