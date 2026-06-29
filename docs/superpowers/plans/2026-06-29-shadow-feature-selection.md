# Shadow Feature Selection Experiment Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible experiment harness that demonstrates the shadow-feature heuristic's failure mode under class imbalance and produces all figures + tables for the methods paper.

**Architecture:** A papermill-driven parameterized notebook (`experiment.ipynb`) runs one scenario per execution and writes a JSON result file. An idempotent orchestration loop (`run_sweep.py`) enumerates scenarios, skips completed ones, executes the notebook via papermill with retries, and loops until coverage is achieved. A separate `aggregate.py` reads all result files and produces cross-scenario figures and the summary table. All numeric computation is factored into `shadow_lib.py` so it is unit-testable independently of the notebook.

**Tech Stack:** Python 3.12 (uv-managed venv), xgboost, scikit-learn, numpy, pandas, matplotlib, papermill, jupyter, pyyaml. JSON for result files. pytest for unit tests.

**Spec:** `docs/superpowers/specs/2026-06-29-shadow-feature-selection-design.md`

**Note on file layout vs spec:** The spec lists `experiment.ipynb` as the place where computation happens. For testability, this plan adds `shadow_lib.py` (data gen, metrics, seed derivation) and `scenarios.py` (grid building). The notebook becomes a thin orchestration layer that imports both. This is a clarification of the spec, not a deviation.

---

## Task 1: Scaffold project (venv, requirements, dirs, README)

**Files:**
- Create: `requirements.txt`
- Create: `README.md`
- Create: `results/.gitkeep`, `figures/.gitkeep`
- Modify: `.gitignore` (already exists; verify only)

- [ ] **Step 1: Verify uv and git are present**

Run: `which uv && uv --version && which git && git --version`
Expected: paths for both, uv 0.x+, git 2.x+.

- [ ] **Step 2: Write `requirements.txt` with pinned versions**

Use these specific versions (known-good on Python 3.12 / macOS arm64):

```
numpy==2.1.3
scipy==1.14.1
pandas==2.2.3
scikit-learn==1.5.2
xgboost==2.1.3
matplotlib==3.9.2
pyyaml==6.0.2
papermill==2.6.0
jupyter==1.1.1
ipykernel==6.29.5
pytest==8.3.3
```

- [ ] **Step 3: Create the venv on Python 3.12 and install**

Run:
```
cd /Users/dan/projects/shadow-feature-selection
uv venv --python 3.12 .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name shadow-feature-selection --display-name "shadow-feature-selection"
```
Expected: clean installs, kernel registered.

- [ ] **Step 4: Verify all imports work**

Run:
```
.venv/bin/python -c "import xgboost, sklearn, numpy, pandas, matplotlib, papermill, yaml; \
print('xgb', xgboost.__version__); print('sk', sklearn.__version__); print('np', numpy.__version__); \
print('pd', pandas.__version__); print('mpl', matplotlib.__version__); print('pm', papermill.__version__)"
```
Expected: prints versions; no errors.

- [ ] **Step 5: Create directories with .gitkeep**

Run: `mkdir -p results figures executed && touch results/.gitkeep figures/.gitkeep`

- [ ] **Step 6: Verify `.gitignore` excludes `.venv/`, `executed/`, `__pycache__/`, `.ipynb_checkpoints/`, `.DS_Store`, `*.pyc`**

If any line is missing, append it.

- [ ] **Step 7: Write `README.md`**

```markdown
# Shadow Feature Selection Under Class Imbalance

Experiment harness for the methods paper showing that the shadow-feature
heuristic (Boruta-style) fails systematically on normalized gain importance
under class imbalance, with the failure controlled by the absolute minority
count `n_min = round(n·p)` rather than the imbalance ratio `p`.

See `docs/superpowers/specs/2026-06-29-shadow-feature-selection-design.md`
for the design and `FINDINGS.md` for the results.

## Reproduce

```
uv venv --python 3.12 .venv
.venv/bin/python -m pip install -r requirements.txt

# Smoke profile (validates pipeline; single-digit minutes)
.venv/bin/python run_sweep.py --profile smoke
.venv/bin/python aggregate.py --profile smoke

# Full profile (the paper's evidence; ~hour-plus)
.venv/bin/python run_sweep.py --profile full
.venv/bin/python aggregate.py --profile full
```

Outputs:
- `results/` — one JSON per scenario (per-repeat values + aggregates).
- `executed/` — papermill-executed notebooks per scenario (self-contained records).
- `figures/` — PNG + PDF cross-scenario plots and `summary_table.csv`.
- `FINDINGS.md` — write-up.
```

- [ ] **Step 8: Commit**

```
git add requirements.txt README.md results/.gitkeep figures/.gitkeep .gitignore
git -c user.name="dan" -c user.email="daniellopes.ba@gmail.com" commit -m "Scaffold project: venv requirements, README, result/figure dirs"
```

---

## Task 2: shadow_lib.py — data generation (TDD)

**Files:**
- Create: `shadow_lib.py`
- Test: `tests/test_shadow_lib.py`

- [ ] **Step 1: Write failing test for `derive_seed`**

Create `tests/test_shadow_lib.py`:

```python
from shadow_lib import derive_seed


def test_derive_seed_is_deterministic():
    a = derive_seed("abc123", 0)
    b = derive_seed("abc123", 0)
    assert a == b
    assert 0 <= a < 2**32


def test_derive_seed_differs_by_repeat():
    seeds = {derive_seed("abc123", i) for i in range(20)}
    assert len(seeds) == 20


def test_derive_seed_differs_by_scenario_id():
    a = derive_seed("abc123", 0)
    b = derive_seed("def456", 0)
    assert a != b
```

- [ ] **Step 2: Run test, see it fail**

Run: `.venv/bin/python -m pytest tests/test_shadow_lib.py -v`
Expected: ImportError / cannot import `derive_seed`.

- [ ] **Step 3: Implement `derive_seed` in `shadow_lib.py`**

```python
"""Numeric core for the shadow-feature experiment.

Pure functions, importable from both the parameterized notebook and the
unit test suite. No I/O lives here.
"""
from __future__ import annotations

import hashlib

import numpy as np


def derive_seed(scenario_id: str, repeat_idx: int) -> int:
    """Deterministic 32-bit seed from scenario id and repeat index."""
    h = hashlib.md5(f"{scenario_id}:{repeat_idx}".encode()).hexdigest()
    return int(h[:8], 16)
```

- [ ] **Step 4: Run test, see it pass**

Run: `.venv/bin/python -m pytest tests/test_shadow_lib.py -v`
Expected: 3 passed.

- [ ] **Step 5: Write failing tests for `make_dataset`**

Append to `tests/test_shadow_lib.py`:

```python
import numpy as np

from shadow_lib import make_dataset


def test_make_dataset_shape_and_counts():
    X, y = make_dataset(n=1000, p=0.1, delta=0.5, n_real=5, n_noise=5, seed=42)
    assert X.shape == (1000, 10)
    assert y.shape == (1000,)
    assert y.sum() == 100  # round(1000 * 0.1)


def test_make_dataset_exact_minority_count_rounding():
    X, y = make_dataset(n=1000, p=0.001, delta=0.5, n_real=5, n_noise=5, seed=0)
    assert y.sum() == 1  # round(1000 * 0.001) = 1


def test_make_dataset_real_features_shifted_in_minority_class():
    X, y = make_dataset(n=20000, p=0.5, delta=1.0, n_real=5, n_noise=5, seed=7)
    pos_means = X[y == 1, :5].mean(axis=0)
    neg_means = X[y == 0, :5].mean(axis=0)
    # Each real feature should be ~1.0 higher in the positive class.
    assert np.allclose(pos_means - neg_means, 1.0, atol=0.05)


def test_make_dataset_noise_features_independent_of_y():
    X, y = make_dataset(n=20000, p=0.5, delta=1.0, n_real=5, n_noise=5, seed=7)
    pos_noise = X[y == 1, 5:].mean(axis=0)
    neg_noise = X[y == 0, 5:].mean(axis=0)
    assert np.allclose(pos_noise - neg_noise, 0.0, atol=0.05)


def test_make_dataset_control_no_signal():
    X, y = make_dataset(n=20000, p=0.5, delta=0.0, n_real=5, n_noise=5, seed=7)
    pos_means = X[y == 1, :5].mean(axis=0)
    neg_means = X[y == 0, :5].mean(axis=0)
    assert np.allclose(pos_means - neg_means, 0.0, atol=0.05)


def test_make_dataset_determinism():
    X1, y1 = make_dataset(n=500, p=0.1, delta=0.5, n_real=5, n_noise=5, seed=42)
    X2, y2 = make_dataset(n=500, p=0.1, delta=0.5, n_real=5, n_noise=5, seed=42)
    assert np.array_equal(X1, X2)
    assert np.array_equal(y1, y2)
```

- [ ] **Step 6: Run tests, see failures**

Run: `.venv/bin/python -m pytest tests/test_shadow_lib.py -v`
Expected: import errors for `make_dataset`.

- [ ] **Step 7: Implement `make_dataset` in `shadow_lib.py`**

Append:

```python
def make_dataset(
    n: int,
    p: float,
    delta: float,
    n_real: int,
    n_noise: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic Gaussian dataset with exact minority count.

    Real features (first n_real columns) are shifted by +delta in the minority
    class. Noise features (next n_noise columns) are independent of y.
    """
    rng = np.random.default_rng(seed)
    n_min = int(round(n * p))
    if n_min < 0 or n_min > n:
        raise ValueError(f"n_min={n_min} out of range for n={n}, p={p}")

    n_features = n_real + n_noise
    X = rng.standard_normal((n, n_features))
    y = np.zeros(n, dtype=np.int8)
    # First n_min rows are the minority class.
    y[:n_min] = 1
    if delta != 0.0 and n_min > 0:
        X[:n_min, :n_real] += delta
    # Shuffle rows so the minority class isn't a contiguous prefix.
    perm = rng.permutation(n)
    return X[perm], y[perm]
```

- [ ] **Step 8: Run tests, see them pass**

Run: `.venv/bin/python -m pytest tests/test_shadow_lib.py -v`
Expected: 8 passed.

- [ ] **Step 9: Commit**

```
git add shadow_lib.py tests/test_shadow_lib.py
git -c user.name="dan" -c user.email="daniellopes.ba@gmail.com" commit -m "Add shadow_lib: derive_seed and make_dataset with unit tests"
```

---

## Task 3: shadow_lib.py — metrics (TDD)

**Files:**
- Modify: `shadow_lib.py`
- Modify: `tests/test_shadow_lib.py`

- [ ] **Step 1: Write failing tests for `compute_metrics`**

Append to `tests/test_shadow_lib.py`:

```python
from shadow_lib import compute_metrics


def test_compute_metrics_naive_floor_is_max_of_noise():
    importances = np.array([0.30, 0.25, 0.10, 0.05, 0.04,  # real (5)
                            0.10, 0.08, 0.04, 0.02, 0.02])  # noise (5)
    m = compute_metrics(importances, n_real=5, n_noise=5)
    assert m["naive_floor"] == 0.10


def test_compute_metrics_p90_floor_is_p90_of_noise():
    rng = np.random.default_rng(0)
    real = rng.uniform(0.1, 0.2, size=5)
    noise = np.linspace(0.01, 0.10, 10)
    importances = np.concatenate([real, noise])
    m = compute_metrics(importances, n_real=5, n_noise=10)
    assert np.isclose(m["p90_floor"], np.percentile(noise, 90))


def test_compute_metrics_fnr_max_counts_real_below_floor():
    importances = np.array([0.30, 0.25, 0.03, 0.02, 0.01,  # real
                            0.05, 0.04, 0.03, 0.02, 0.02])  # noise
    m = compute_metrics(importances, n_real=5, n_noise=5)
    # naive floor = max(noise) = 0.05; reals below: 0.03, 0.02, 0.01 -> 3/5
    assert np.isclose(m["fnr_max"], 0.6)


def test_compute_metrics_fnr_p90_uses_p90_floor():
    importances = np.array([0.30, 0.25, 0.10, 0.05, 0.04,  # real
                            0.10, 0.08, 0.04, 0.02, 0.02])  # noise
    m = compute_metrics(importances, n_real=5, n_noise=5)
    p90 = np.percentile([0.10, 0.08, 0.04, 0.02, 0.02], 90)
    expected_fnr = sum(r < p90 for r in [0.30, 0.25, 0.10, 0.05, 0.04]) / 5
    assert np.isclose(m["fnr_p90"], expected_fnr)


def test_compute_metrics_mean_real_importance():
    importances = np.array([0.30, 0.25, 0.10, 0.05, 0.04,  # real
                            0.10, 0.08, 0.04, 0.02, 0.02])  # noise
    m = compute_metrics(importances, n_real=5, n_noise=5)
    assert np.isclose(m["mean_real_importance"], np.mean([0.30, 0.25, 0.10, 0.05, 0.04]))
```

- [ ] **Step 2: Run tests, see failures**

Run: `.venv/bin/python -m pytest tests/test_shadow_lib.py -v`
Expected: ImportError for `compute_metrics`.

- [ ] **Step 3: Implement `compute_metrics`**

Append to `shadow_lib.py`:

```python
def compute_metrics(
    importances: np.ndarray, n_real: int, n_noise: int
) -> dict[str, float]:
    """Per-repeat metrics from normalized gain importances.

    Convention: the first n_real entries are real features, the next n_noise
    are noise. Importances do not need to be pre-normalized; we normalize here
    defensively for the dict-of-floats output.
    """
    assert importances.shape == (n_real + n_noise,)
    total = importances.sum()
    if total <= 0:
        # All-zero importance (no splits found); treat noise floor as 0 → no prune.
        normed = importances
    else:
        normed = importances / total
    real = normed[:n_real]
    noise = normed[n_real : n_real + n_noise]
    naive_floor = float(noise.max())
    p90_floor = float(np.percentile(noise, 90))
    return {
        "naive_floor": naive_floor,
        "p90_floor": p90_floor,
        "mean_real_importance": float(real.mean()),
        "fnr_max": float((real < naive_floor).mean()),
        "fnr_p90": float((real < p90_floor).mean()),
    }
```

- [ ] **Step 4: Run tests, see them pass**

Run: `.venv/bin/python -m pytest tests/test_shadow_lib.py -v`
Expected: 13 passed.

- [ ] **Step 5: Add aggregator helper `aggregate_repeats`**

Append to `shadow_lib.py`:

```python
def aggregate_repeats(per_repeat: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """Mean/median/p90/std per metric across repeats."""
    out: dict[str, dict[str, float]] = {}
    for name, values in per_repeat.items():
        arr = np.asarray(values, dtype=float)
        out[name] = {
            "mean": float(arr.mean()),
            "median": float(np.median(arr)),
            "p90": float(np.percentile(arr, 90)),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        }
    return out
```

- [ ] **Step 6: Add an end-to-end smoke test with an actual XGBoost fit**

Append to `tests/test_shadow_lib.py`:

```python
def test_pipeline_smoke_xgboost():
    """Sanity check: with strong signal and balanced data, real-feature
    importance should clearly dominate the noise floor."""
    from xgboost import XGBClassifier
    X, y = make_dataset(n=2000, p=0.5, delta=1.0, n_real=5, n_noise=5, seed=42)
    clf = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.1,
        tree_method="hist", importance_type="gain", n_jobs=1,
        eval_metric="logloss", verbosity=0,
    )
    clf.fit(X, y)
    m = compute_metrics(clf.feature_importances_, n_real=5, n_noise=5)
    assert m["mean_real_importance"] > m["naive_floor"]
    assert m["fnr_max"] == 0.0
```

- [ ] **Step 7: Run tests, see them pass**

Run: `.venv/bin/python -m pytest tests/test_shadow_lib.py -v`
Expected: 14 passed.

- [ ] **Step 8: Commit**

```
git add shadow_lib.py tests/test_shadow_lib.py
git -c user.name="dan" -c user.email="daniellopes.ba@gmail.com" commit -m "Add compute_metrics, aggregate_repeats, and pipeline smoke test"
```

---

## Task 4: scenarios.py — grid building (TDD)

**Files:**
- Create: `scenarios.py`
- Test: `tests/test_scenarios.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_scenarios.py`:

```python
import pytest

from scenarios import (
    build_scenarios,
    scenario_id,
    SMOKE_PROFILE,
    FULL_PROFILE,
)


def test_scenario_id_deterministic():
    p1 = {"regime": "A", "n": 1000, "p": 0.1, "delta": 0.5,
          "n_real": 5, "n_noise": 5, "control": False,
          "backend": "xgboost", "n_repeats": 25}
    p2 = dict(p1)
    assert scenario_id(p1) == scenario_id(p2)


def test_scenario_id_differs_on_any_param_change():
    base = {"regime": "A", "n": 1000, "p": 0.1, "delta": 0.5,
            "n_real": 5, "n_noise": 5, "control": False,
            "backend": "xgboost", "n_repeats": 25}
    sid = scenario_id(base)
    for key in base:
        modified = dict(base)
        if isinstance(base[key], (int, float)):
            modified[key] = base[key] + (1 if isinstance(base[key], int) else 0.01)
        elif isinstance(base[key], bool):
            modified[key] = not base[key]
        else:
            modified[key] = base[key] + "_x"
        assert scenario_id(modified) != sid, f"id collision on {key}"


def test_smoke_profile_is_small():
    scenarios = build_scenarios(SMOKE_PROFILE)
    assert 10 <= len(scenarios) <= 40


def test_full_profile_covers_both_regimes():
    scenarios = build_scenarios(FULL_PROFILE)
    regimes = {s["regime"] for s in scenarios}
    assert regimes == {"A", "B"}


def test_full_profile_has_pure_noise_control():
    scenarios = build_scenarios(FULL_PROFILE)
    assert any(s["control"] for s in scenarios)
    assert any(not s["control"] for s in scenarios)


def test_full_profile_includes_n_noise_20_robustness():
    scenarios = build_scenarios(FULL_PROFILE)
    assert any(s["n_noise"] == 20 for s in scenarios)


def test_full_profile_includes_sklearn_cross_check():
    scenarios = build_scenarios(FULL_PROFILE)
    backends = {s["backend"] for s in scenarios}
    assert "sklearn_gbm" in backends


def test_degenerate_scenarios_excluded():
    scenarios = build_scenarios(FULL_PROFILE)
    for s in scenarios:
        n_min = round(s["n"] * s["p"])
        assert n_min >= 5, f"degenerate scenario kept: {s}"


def test_n_min_overlap_between_regimes():
    """The whole point: Regime A and Regime B should share an n_min range."""
    scenarios = build_scenarios(FULL_PROFILE)
    a_n_min = {round(s["n"] * s["p"]) for s in scenarios if s["regime"] == "A"}
    b_n_min = {round(s["n"] * s["p"]) for s in scenarios if s["regime"] == "B"}
    # They don't need to coincide on identical values, just to overlap in range.
    assert min(a_n_min) <= max(b_n_min)
    assert min(b_n_min) <= max(a_n_min)


def test_repeats_reduced_for_large_n():
    scenarios = build_scenarios(FULL_PROFILE)
    for s in scenarios:
        if s["n"] >= 50000:
            assert s["n_repeats"] <= 15
```

- [ ] **Step 2: Run tests, see ImportError**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py -v`
Expected: cannot import `scenarios`.

- [ ] **Step 3: Implement `scenarios.py`**

```python
"""Scenario grid construction for the shadow-feature experiment.

Each scenario is a dict with these keys (all hash inputs for scenario_id):
    regime, n, p, delta, n_real, n_noise, control, backend, n_repeats
"""
from __future__ import annotations

import hashlib
import itertools
import json
from typing import Any, Iterable


def scenario_id(params: dict[str, Any]) -> str:
    """Stable 12-char id from a scenario param dict."""
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode()).hexdigest()[:12]


def _repeats_for(n: int, base: int) -> int:
    return min(base, 15) if n >= 50000 else base


def _cross(
    regime: str,
    ns: Iterable[int],
    ps: Iterable[float],
    deltas: Iterable[float],
    *,
    n_real: int = 5,
    n_noise: int = 5,
    control_too: bool = True,
    backend: str = "xgboost",
    base_repeats: int = 25,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n, p, delta in itertools.product(ns, ps, deltas):
        if round(n * p) < 5:
            continue
        for control in ([False, True] if control_too else [False]):
            out.append(
                {
                    "regime": regime,
                    "n": int(n),
                    "p": float(p),
                    "delta": float(delta),
                    "n_real": n_real,
                    "n_noise": n_noise,
                    "control": bool(control),
                    "backend": backend,
                    "n_repeats": _repeats_for(n, base_repeats),
                }
            )
    return out


# ----- Profiles -----------------------------------------------------------------

SMOKE_PROFILE = "smoke"
FULL_PROFILE = "full"


def _smoke() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    # Regime A
    scenarios.extend(_cross(
        "A", ns=[500, 1000], ps=[0.10, 0.20], deltas=[0.5],
        base_repeats=2,
    ))
    # Regime B
    scenarios.extend(_cross(
        "B", ns=[5000], ps=[0.01], deltas=[0.5],
        base_repeats=2,
    ))
    # 1 n_noise=20 robustness scenario
    scenarios.extend(_cross(
        "A", ns=[1000], ps=[0.10], deltas=[0.5],
        n_noise=20, control_too=False, base_repeats=2,
    ))
    # 1 sklearn cross-check
    scenarios.extend(_cross(
        "A", ns=[1000], ps=[0.10], deltas=[0.5],
        control_too=False, backend="sklearn_gbm", base_repeats=2,
    ))
    return _dedupe(scenarios)


def _full() -> list[dict[str, Any]]:
    scenarios: list[dict[str, Any]] = []
    # Regime A: full cross
    scenarios.extend(_cross(
        "A", ns=[500, 1000, 2500, 5000], ps=[0.05, 0.10, 0.20],
        deltas=[0.3, 0.5, 1.0],
    ))
    # Regime B: full cross
    scenarios.extend(_cross(
        "B", ns=[10000, 50000, 200000], ps=[0.001, 0.005, 0.01],
        deltas=[0.3, 0.5, 1.0],
    ))
    # n_noise=20 robustness: δ=0.5 only, no control, both regimes
    scenarios.extend(_cross(
        "A", ns=[500, 1000, 2500, 5000], ps=[0.05, 0.10, 0.20],
        deltas=[0.5], n_noise=20, control_too=False,
    ))
    scenarios.extend(_cross(
        "B", ns=[10000, 50000, 200000], ps=[0.001, 0.005, 0.01],
        deltas=[0.5], n_noise=20, control_too=False,
    ))
    # sklearn cross-check: 6 scenarios from Regime A, moderate n, δ=0.5
    scenarios.extend(_cross(
        "A", ns=[1000, 2500, 5000], ps=[0.10, 0.20],
        deltas=[0.5], control_too=False, backend="sklearn_gbm",
    ))
    return _dedupe(scenarios)


def _dedupe(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for s in scenarios:
        seen[scenario_id(s)] = s
    return list(seen.values())


def build_scenarios(profile: str) -> list[dict[str, Any]]:
    if profile == SMOKE_PROFILE:
        return _smoke()
    if profile == FULL_PROFILE:
        return _full()
    raise ValueError(f"unknown profile: {profile!r}")


if __name__ == "__main__":
    import sys
    prof = sys.argv[1] if len(sys.argv) > 1 else SMOKE_PROFILE
    scenarios = build_scenarios(prof)
    print(f"profile={prof} n_scenarios={len(scenarios)}")
    for s in scenarios[:5]:
        print(scenario_id(s), s)
```

- [ ] **Step 4: Run tests, see them pass**

Run: `.venv/bin/python -m pytest tests/test_scenarios.py -v`
Expected: 10 passed.

- [ ] **Step 5: Print profile sizes to sanity-check the grid**

Run: `.venv/bin/python scenarios.py smoke` and `.venv/bin/python scenarios.py full`
Expected: smoke ~10–20 scenarios; full ~200–300 scenarios.

- [ ] **Step 6: Commit**

```
git add scenarios.py tests/test_scenarios.py
git -c user.name="dan" -c user.email="daniellopes.ba@gmail.com" commit -m "Add scenarios.py: smoke and full grids with deterministic ids"
```

---

## Task 5: experiment.ipynb — parameterized notebook

**Files:**
- Create: `experiment.ipynb`

The notebook runs ONE scenario. papermill injects parameters into the `parameters`-tagged cell. The notebook writes `results/<scenario_id>.json` and renders inline plots for the scenario.

- [ ] **Step 1: Create the notebook via a Python script**

Run this snippet (one-time, to produce the notebook structure):

```
.venv/bin/python - <<'PY'
import nbformat as nbf

nb = nbf.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "name": "shadow-feature-selection",
    "display_name": "shadow-feature-selection",
    "language": "python",
}

cells = []

cells.append(nbf.v4.new_markdown_cell(
    "# Shadow-feature scenario\n\n"
    "Parameterized notebook. One run = one scenario. Writes "
    "`results/<scenario_id>.json` and shows the per-feature importance "
    "distribution for this scenario."
))

# Parameters cell (papermill injects values here).
params_cell = nbf.v4.new_code_cell(
    "# Default parameters; papermill overrides via the `parameters` tag.\n"
    "scenario_id = 'smoke-default'\n"
    "regime = 'A'\n"
    "n = 1000\n"
    "p = 0.10\n"
    "delta = 0.5\n"
    "n_real = 5\n"
    "n_noise = 5\n"
    "control = False\n"
    "backend = 'xgboost'\n"
    "n_repeats = 5\n"
    "results_dir = 'results'\n"
)
params_cell.metadata["tags"] = ["parameters"]
cells.append(params_cell)

cells.append(nbf.v4.new_code_cell(
    "import json, os, sys, time, platform\n"
    "from pathlib import Path\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "import matplotlib.pyplot as plt\n"
    "import sklearn, xgboost\n"
    "\n"
    "sys.path.insert(0, os.getcwd())\n"
    "from shadow_lib import derive_seed, make_dataset, compute_metrics, aggregate_repeats\n"
    "\n"
    "effective_delta = 0.0 if control else float(delta)\n"
    "n_min = int(round(n * p))\n"
    "print(f'scenario_id={scenario_id} regime={regime} n={n} p={p} delta={delta} '\n"
    "      f'control={control} backend={backend} n_repeats={n_repeats} n_min={n_min}')\n"
))

cells.append(nbf.v4.new_code_cell(
    "def make_clf(backend, seed):\n"
    "    if backend == 'xgboost':\n"
    "        from xgboost import XGBClassifier\n"
    "        return XGBClassifier(\n"
    "            n_estimators=300, max_depth=6, learning_rate=0.1,\n"
    "            tree_method='hist', importance_type='gain', n_jobs=-1,\n"
    "            eval_metric='logloss', verbosity=0, random_state=seed,\n"
    "        )\n"
    "    if backend == 'sklearn_gbm':\n"
    "        from sklearn.ensemble import GradientBoostingClassifier\n"
    "        return GradientBoostingClassifier(\n"
    "            n_estimators=200, max_depth=3, learning_rate=0.1,\n"
    "            random_state=seed,\n"
    "        )\n"
    "    raise ValueError(f'unknown backend: {backend!r}')\n"
))

cells.append(nbf.v4.new_code_cell(
    "per_repeat = {k: [] for k in\n"
    "    ['naive_floor', 'p90_floor', 'mean_real_importance', 'fnr_max', 'fnr_p90']}\n"
    "importances_all = []\n"
    "wall_start = time.time()\n"
    "for r in range(n_repeats):\n"
    "    seed = derive_seed(scenario_id, r)\n"
    "    X, y = make_dataset(n=n, p=p, delta=effective_delta,\n"
    "                       n_real=n_real, n_noise=n_noise, seed=seed)\n"
    "    clf = make_clf(backend, seed)\n"
    "    clf.fit(X, y)\n"
    "    imp = np.asarray(clf.feature_importances_, dtype=float)\n"
    "    importances_all.append(imp)\n"
    "    m = compute_metrics(imp, n_real=n_real, n_noise=n_noise)\n"
    "    for k, v in m.items():\n"
    "        per_repeat[k].append(v)\n"
    "wall_time = time.time() - wall_start\n"
    "print(f'wall_time={wall_time:.2f}s')\n"
))

cells.append(nbf.v4.new_code_cell(
    "aggregate = aggregate_repeats(per_repeat)\n"
    "lib_versions = {\n"
    "    'python': platform.python_version(),\n"
    "    'numpy': np.__version__,\n"
    "    'pandas': pd.__version__,\n"
    "    'sklearn': sklearn.__version__,\n"
    "    'xgboost': xgboost.__version__,\n"
    "}\n"
    "result = {\n"
    "    'scenario_id': scenario_id,\n"
    "    'params': {'regime': regime, 'n': n, 'p': p, 'delta': delta,\n"
    "               'n_real': n_real, 'n_noise': n_noise, 'control': control,\n"
    "               'backend': backend, 'n_repeats': n_repeats},\n"
    "    'effective_delta': effective_delta,\n"
    "    'n_min': n_min,\n"
    "    'n_repeats_used': n_repeats,\n"
    "    'wall_time_seconds': wall_time,\n"
    "    'library_versions': lib_versions,\n"
    "    'per_repeat': per_repeat,\n"
    "    'aggregate': aggregate,\n"
    "}\n"
    "Path(results_dir).mkdir(parents=True, exist_ok=True)\n"
    "out_path = Path(results_dir) / f'{scenario_id}.json'\n"
    "with open(out_path, 'w') as f:\n"
    "    json.dump(result, f, indent=2)\n"
    "print('wrote', out_path)\n"
))

cells.append(nbf.v4.new_code_cell(
    "# Inline plot: per-feature importance distribution across repeats.\n"
    "imp_matrix = np.array(importances_all)  # (n_repeats, n_features)\n"
    "imp_norm = imp_matrix / imp_matrix.sum(axis=1, keepdims=True).clip(min=1e-12)\n"
    "labels = [f'real_{i}' for i in range(n_real)] + [f'noise_{j}' for j in range(n_noise)]\n"
    "fig, ax = plt.subplots(figsize=(8, 4))\n"
    "positions = np.arange(n_real + n_noise)\n"
    "ax.boxplot(imp_norm, positions=positions, widths=0.6, showfliers=False)\n"
    "ax.axhline(aggregate['naive_floor']['mean'], color='red', linestyle='--',\n"
    "           label=f\"naive floor (mean={aggregate['naive_floor']['mean']:.3f})\")\n"
    "ax.axhline(aggregate['p90_floor']['mean'], color='orange', linestyle=':',\n"
    "           label=f\"p90 floor (mean={aggregate['p90_floor']['mean']:.3f})\")\n"
    "ax.set_xticks(positions)\n"
    "ax.set_xticklabels(labels, rotation=45)\n"
    "ax.set_ylabel('normalized gain importance')\n"
    "ax.set_title(f'{scenario_id} | n={n} p={p} delta={delta} control={control} n_min={n_min}')\n"
    "ax.legend()\n"
    "fig.tight_layout()\n"
    "plt.show()\n"
))

cells.append(nbf.v4.new_code_cell(
    "# Print headline numbers for quick eyeballing.\n"
    "print('FNR_max (mean):', aggregate['fnr_max']['mean'])\n"
    "print('FNR_p90 (mean):', aggregate['fnr_p90']['mean'])\n"
    "print('Noise floor (mean):', aggregate['naive_floor']['mean'])\n"
    "print('Mean real importance (mean):', aggregate['mean_real_importance']['mean'])\n"
))

nb.cells = cells
nbf.write(nb, 'experiment.ipynb')
print('wrote experiment.ipynb with', len(cells), 'cells')
PY
```
Expected: prints "wrote experiment.ipynb with 8 cells".

- [ ] **Step 2: Verify the notebook is valid JSON and has a `parameters` tag**

Run:
```
.venv/bin/python -c "import nbformat; nb=nbformat.read('experiment.ipynb', 4); \
print([c['metadata'].get('tags') for c in nb.cells if c['cell_type']=='code'])"
```
Expected: at least one cell with `['parameters']`.

- [ ] **Step 3: Smoke-execute the notebook by hand via papermill (default params)**

Run:
```
.venv/bin/python -m papermill experiment.ipynb /tmp/exp_default.ipynb \
  -p scenario_id smoke_manual -p n 500 -p p 0.20 -p n_repeats 2 -k shadow-feature-selection
```
Expected: completes; `results/smoke_manual.json` exists.

- [ ] **Step 4: Inspect the JSON**

Run: `.venv/bin/python -m json.tool results/smoke_manual.json | head -40`
Expected: keys `scenario_id`, `params`, `n_min`, `aggregate`, etc.

- [ ] **Step 5: Clean up the smoke run**

Run: `rm results/smoke_manual.json /tmp/exp_default.ipynb`

- [ ] **Step 6: Commit**

```
git add experiment.ipynb
git -c user.name="dan" -c user.email="daniellopes.ba@gmail.com" commit -m "Add parameterized experiment notebook (papermill-ready)"
```

---

## Task 6: config.yaml + run_sweep.py — orchestration

**Files:**
- Create: `config.yaml`
- Create: `run_sweep.py`

- [ ] **Step 1: Write `config.yaml`**

```yaml
# Profile selection for the experiment sweep. Defaults to `smoke`.
default_profile: smoke

# Orchestration controls (shared by both profiles).
orchestration:
  max_passes: 3
  max_attempts_per_pass: 2
  results_dir: results
  executed_dir: executed
  failed_path: failed.json
  notebook: experiment.ipynb
  kernel: shadow-feature-selection
```

- [ ] **Step 2: Write `run_sweep.py`**

```python
"""Idempotent orchestration loop for the shadow-feature sweep.

Enumerates the scenario set for the chosen profile, skips scenarios whose
results already exist, executes the parameterized notebook via papermill for
the rest, retries transient failures, and loops until coverage is reached or
all remaining are marked failed.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import papermill as pm
import yaml

from scenarios import build_scenarios, scenario_id


def load_config(path: Path = Path("config.yaml")) -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_failed(failed_path: Path) -> dict[str, dict[str, Any]]:
    if not failed_path.exists():
        return {}
    with open(failed_path) as f:
        return json.load(f)


def write_failed(failed_path: Path, failed: dict[str, dict[str, Any]]) -> None:
    with open(failed_path, "w") as f:
        json.dump(failed, f, indent=2)


def is_done(results_dir: Path, sid: str) -> bool:
    p = results_dir / f"{sid}.json"
    if not p.exists():
        return False
    try:
        with open(p) as f:
            json.load(f)
        return True
    except Exception:
        return False


def run_one(scenario: dict[str, Any], sid: str, orch: dict[str, Any]) -> None:
    out_nb = Path(orch["executed_dir"]) / f"{sid}.ipynb"
    Path(orch["executed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(orch["results_dir"]).mkdir(parents=True, exist_ok=True)
    parameters = {
        "scenario_id": sid,
        "regime": scenario["regime"],
        "n": scenario["n"],
        "p": scenario["p"],
        "delta": scenario["delta"],
        "n_real": scenario["n_real"],
        "n_noise": scenario["n_noise"],
        "control": scenario["control"],
        "backend": scenario["backend"],
        "n_repeats": scenario["n_repeats"],
        "results_dir": orch["results_dir"],
    }
    pm.execute_notebook(
        orch["notebook"],
        str(out_nb),
        parameters=parameters,
        kernel_name=orch["kernel"],
        progress_bar=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None,
                        help="smoke | full (default from config.yaml)")
    args = parser.parse_args()

    cfg = load_config()
    profile = args.profile or cfg["default_profile"]
    orch = cfg["orchestration"]

    results_dir = Path(orch["results_dir"])
    failed_path = Path(orch["failed_path"])
    max_passes = int(orch["max_passes"])
    max_attempts = int(orch["max_attempts_per_pass"])

    scenarios = build_scenarios(profile)
    targets = [(scenario_id(s), s) for s in scenarios]
    print(f"[sweep] profile={profile} target_count={len(targets)}")

    failed = load_failed(failed_path)

    for pass_idx in range(1, max_passes + 1):
        remaining = [(sid, s) for sid, s in targets
                     if not is_done(results_dir, sid) and sid not in failed]
        if not remaining:
            print(f"[sweep] pass {pass_idx}: nothing to do (all covered or failed)")
            break
        print(f"[sweep] pass {pass_idx}: remaining={len(remaining)} "
              f"covered={sum(1 for sid, _ in targets if is_done(results_dir, sid))} "
              f"failed={len(failed)}")
        for i, (sid, s) in enumerate(remaining, 1):
            t0 = time.time()
            last_exc: BaseException | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    run_one(s, sid, orch)
                    elapsed = time.time() - t0
                    print(f"[sweep]   [{i}/{len(remaining)}] {sid} OK "
                          f"({elapsed:.1f}s, attempt {attempt}) {_compact(s)}")
                    last_exc = None
                    break
                except Exception as exc:
                    last_exc = exc
                    print(f"[sweep]   [{i}/{len(remaining)}] {sid} FAIL "
                          f"attempt {attempt}: {exc!r}")
            if last_exc is not None:
                failed[sid] = {
                    "scenario": s,
                    "pass": pass_idx,
                    "exception": repr(last_exc),
                    "traceback": traceback.format_exception(
                        type(last_exc), last_exc, last_exc.__traceback__),
                }
                write_failed(failed_path, failed)
                print(f"[sweep]   [{i}/{len(remaining)}] {sid} marked FAILED")

    covered = sum(1 for sid, _ in targets if is_done(results_dir, sid))
    still_missing = [sid for sid, _ in targets
                     if not is_done(results_dir, sid) and sid not in failed]
    print(f"[sweep] DONE: covered={covered}/{len(targets)} "
          f"failed={len(failed)} still_missing={len(still_missing)}")
    return 0 if not still_missing else 2


def _compact(s: dict[str, Any]) -> str:
    return (f"regime={s['regime']} n={s['n']} p={s['p']} delta={s['delta']} "
            f"control={s['control']} backend={s['backend']} "
            f"n_noise={s['n_noise']} n_repeats={s['n_repeats']}")


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 3: Dry-run the orchestrator on smoke profile (no execution check yet)**

Run: `.venv/bin/python -c "from scenarios import build_scenarios, scenario_id; \
ss = build_scenarios('smoke'); print(len(ss)); [print(scenario_id(s), s) for s in ss[:3]]"`
Expected: prints count and first 3 scenarios.

- [ ] **Step 4: Commit**

```
git add config.yaml run_sweep.py
git -c user.name="dan" -c user.email="daniellopes.ba@gmail.com" commit -m "Add run_sweep.py orchestration loop and config.yaml"
```

---

## Task 7: Run smoke profile end-to-end

- [ ] **Step 1: Execute the smoke sweep**

Run: `.venv/bin/python run_sweep.py --profile smoke`
Expected: prints per-scenario OK lines; final `DONE: covered=N/N failed=0 still_missing=0`. Should take single-digit minutes.

- [ ] **Step 2: Verify result files exist and parse**

Run:
```
ls results/*.json | wc -l
.venv/bin/python -c "import json, glob; \
n = sum(1 for p in glob.glob('results/*.json') if isinstance(json.load(open(p)), dict)); \
print('valid_json_count', n)"
```
Expected: equals the smoke target count.

- [ ] **Step 3: Verify executed notebooks exist**

Run: `ls executed/*.ipynb | wc -l`
Expected: equals the number of newly executed scenarios.

- [ ] **Step 4: Re-run to confirm idempotence**

Run: `.venv/bin/python run_sweep.py --profile smoke`
Expected: first pass reports `remaining=0`; exits 0.

- [ ] **Step 5: Commit results**

```
git add results/ failed.json 2>/dev/null || true
git -c user.name="dan" -c user.email="daniellopes.ba@gmail.com" commit -m "Smoke profile results (sanity-check of pipeline)" || echo "no changes"
```

---

## Task 8: aggregate.py — figures + summary table

**Files:**
- Create: `aggregate.py`

Aggregator reads `results/*.json`, builds a tidy DataFrame, writes
publication figures (PNG + PDF) and a CSV summary table. Visual quality is
checked by opening the figures; pipeline correctness is verified by the
smoke run.

- [ ] **Step 1: Write `aggregate.py`**

```python
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
    0.001: "v", 0.005: "^", 0.01: "s",
    0.05: "o", 0.10: "D", 0.20: "P",
}

FIG_DIR = Path("figures")


def load_results(results_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(results_dir.glob("*.json")):
        with open(p) as f:
            r = json.load(f)
        params = r["params"]
        agg = r["aggregate"]
        rows.append({
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
        })
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
            sub["n_min"], sub[metric],
            marker=P_MARKER.get(p, "o"),
            color=REGIME_COLOR[regime],
            linestyle="-", linewidth=1.2, markersize=6,
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
    primary = df[(df["control"] == False) & (df["backend"] == "xgboost") &
                 (df["n_noise"] == 5)]
    for d in sorted(primary["delta"].unique()):
        sub = primary[primary["delta"] == d]
        fig = _plot_metric_vs_nmin(
            sub, "fnr_max_mean",
            "FNR (naive shadow floor)",
            f"Collapse: FNR_max vs n_min   (δ={d})",
        )
        save_fig(fig, f"01_collapse_fnr_max_delta_{d}")


def plot_noise_floor_inflation(df: pd.DataFrame) -> None:
    primary = df[(df["control"] == False) & (df["backend"] == "xgboost") &
                 (df["n_noise"] == 5) & (df["delta"] == 0.5)]
    fig = _plot_metric_vs_nmin(
        primary, "naive_floor_mean",
        "normalized noise floor (max over noise features)",
        "Noise floor inflation vs n_min  (δ=0.5)",
    )
    save_fig(fig, "02_noise_floor_inflation")


def plot_mechanism(df: pd.DataFrame) -> None:
    base = df[(df["backend"] == "xgboost") & (df["n_noise"] == 5) &
              (df["delta"] == 0.5)]
    signal = base[base["control"] == False].sort_values("n_min")
    control = base[base["control"] == True].sort_values("n_min")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(signal["n_min"], signal["mean_real_importance_mean"],
            "o-", color="green", label="mean real-feature importance (signal)")
    ax.plot(signal["n_min"], signal["naive_floor_mean"],
            "s-", color="red", label="noise floor (signal)")
    if not control.empty:
        ax.plot(control["n_min"], control["naive_floor_mean"],
                "x--", color="gray", label="noise floor (pure-noise control)")
    ax.set_xscale("log")
    ax.set_xlabel("n_min")
    ax.set_ylabel("normalized importance")
    ax.set_title("Mechanism: real-feature collapse vs noise-share inflation (δ=0.5)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    save_fig(fig, "03_mechanism_decomposition")


def plot_mitigation(df: pd.DataFrame) -> None:
    base = df[(df["control"] == False) & (df["backend"] == "xgboost") &
              (df["n_noise"] == 5) & (df["delta"] == 0.5)].sort_values("n_min")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for regime, sub in base.groupby("regime"):
        ax.plot(sub["n_min"], sub["fnr_max_mean"], "o-",
                color=REGIME_COLOR[regime], label=f"FNR_max  regime {regime}")
        ax.plot(sub["n_min"], sub["fnr_p90_mean"], "s--",
                color=REGIME_COLOR[regime], alpha=0.6,
                label=f"FNR_p90  regime {regime}")
    ax.set_xscale("log")
    ax.set_xlabel("n_min")
    ax.set_ylabel("FNR")
    ax.set_title("Mitigation: naive (max) vs P90 noise-floor thresholds (δ=0.5)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, "04_mitigation_p90")


def plot_heatmaps(df: pd.DataFrame) -> None:
    base = df[(df["control"] == False) & (df["backend"] == "xgboost") &
              (df["n_noise"] == 5) & (df["delta"] == 0.5)]
    for regime, sub in base.groupby("regime"):
        pivot = sub.pivot_table(index="n", columns="p",
                                values="fnr_max_mean", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="magma",
                       vmin=0.0, vmax=1.0)
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
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="white" if v < 0.6 else "black", fontsize=9)
        fig.colorbar(im, ax=ax, label="FNR_max")
        fig.tight_layout()
        save_fig(fig, f"05_heatmap_regime_{regime}")


def plot_signal_interaction(df: pd.DataFrame) -> None:
    base = df[(df["control"] == False) & (df["backend"] == "xgboost") &
              (df["n_noise"] == 5)]
    deltas = sorted(base["delta"].unique())
    fig, axes = plt.subplots(1, len(deltas), figsize=(5 * len(deltas), 4),
                             sharey=True)
    if len(deltas) == 1:
        axes = [axes]
    for ax, d in zip(axes, deltas):
        sub = base[base["delta"] == d]
        for (regime, p), grp in sub.groupby(["regime", "p"]):
            grp = grp.sort_values("n_min")
            ax.plot(grp["n_min"], grp["fnr_max_mean"],
                    marker=P_MARKER.get(p, "o"),
                    color=REGIME_COLOR[regime], linewidth=1.2,
                    label=f"{regime}, p={p}", alpha=0.85)
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
    n20 = df[(df["control"] == False) & (df["n_noise"] == 20) &
             (df["backend"] == "xgboost") & (df["delta"] == 0.5)]
    if not n20.empty:
        fig = _plot_metric_vs_nmin(
            n20, "fnr_max_mean", "FNR_max (n_noise=20)",
            "Robustness: same collapse with n_noise=20  (δ=0.5)",
        )
        save_fig(fig, "07a_robustness_n_noise_20")

    sk = df[(df["control"] == False) & (df["backend"] == "sklearn_gbm") &
            (df["delta"] == 0.5)]
    if not sk.empty:
        fig = _plot_metric_vs_nmin(
            sk, "fnr_max_mean", "FNR_max (sklearn GBM)",
            "Robustness: sklearn GBM cross-check  (δ=0.5)",
        )
        save_fig(fig, "07b_robustness_sklearn_gbm")


def write_summary_table(df: pd.DataFrame) -> Path:
    base = df[(df["control"] == False) & (df["backend"] == "xgboost") &
              (df["n_noise"] == 5) & (df["delta"] == 0.5)].copy()
    bins = [0, 25, 75, 200, 500, 1500, np.inf]
    labels = ["<25", "25-74", "75-199", "200-499", "500-1499", "1500+"]
    base["n_min_bin"] = pd.cut(base["n_min"], bins=bins, labels=labels,
                               right=False)
    grouped = base.groupby("n_min_bin", observed=True).agg(
        n_scenarios=("scenario_id", "count"),
        n_min_min=("n_min", "min"),
        n_min_max=("n_min", "max"),
        mean_noise_floor=("naive_floor_mean", "mean"),
        mean_fnr_max=("fnr_max_mean", "mean"),
        mean_fnr_p90=("fnr_p90_mean", "mean"),
    ).reset_index()

    def recommend(row):
        if pd.isna(row["mean_fnr_max"]):
            return "(no data)"
        if row["mean_fnr_max"] < 0.05:
            return "shadow heuristic OK"
        if row["mean_fnr_p90"] < 0.10:
            return "use P90 floor (multi-shadow)"
        return "shadow approach unreliable; prefer stratified CV w/ direct gain ranking"

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
```

- [ ] **Step 2: Run aggregate on the smoke results**

Run: `.venv/bin/python aggregate.py --profile smoke`
Expected: prints `[aggregate] loaded N scenarios`, writes PNG + PDF files in `figures/`, writes `figures/summary_table.csv`. Some figures may have sparse points (smoke is intentionally small); no exceptions.

- [ ] **Step 3: Inspect the figures**

Run: `ls figures/`
Expected: at least `01_collapse_fnr_max_delta_0.5.png/.pdf`, `02_noise_floor_inflation.*`, `03_mechanism_decomposition.*`, `04_mitigation_p90.*`, `05_heatmap_regime_A.*`, `05_heatmap_regime_B.*`, `06_signal_interaction.*`, `summary_table.csv`. (`07a_*` and `07b_*` if smoke included those scenarios.)

- [ ] **Step 4: Commit aggregator + smoke figures**

```
git add aggregate.py figures/
git -c user.name="dan" -c user.email="daniellopes.ba@gmail.com" commit -m "Add aggregate.py: cross-scenario figures and summary CSV (smoke run)"
```

---

## Task 9: Run full profile

- [ ] **Step 1: Estimate runtime upfront**

Run: `.venv/bin/python -c "from scenarios import build_scenarios; \
ss = build_scenarios('full'); \
big = [s for s in ss if s['n'] >= 50000]; \
print('total', len(ss), 'large_n', len(big))"`
Expected: a few hundred total; ~tens with n ≥ 50000 (these dominate runtime).

- [ ] **Step 2: Execute the full sweep**

Run: `.venv/bin/python run_sweep.py --profile full`
Expected: per-scenario OK lines streaming; the 200k cells take noticeably longer (minutes each, but still finite with `tree_method='hist'`). Final: `DONE: covered=X/X failed=0`.

If the run is interrupted, just re-invoke — it is idempotent and will resume.

- [ ] **Step 3: Verify coverage**

Run:
```
.venv/bin/python - <<'PY'
import json, glob
from scenarios import build_scenarios, scenario_id
target = {scenario_id(s) for s in build_scenarios('full')}
got = {p.split('/')[-1][:-5] for p in glob.glob('results/*.json')}
missing = target - got
print('target', len(target), 'got', len(got), 'missing', len(missing))
PY
```
Expected: `missing 0` (or only IDs listed in `failed.json`).

- [ ] **Step 4: Regenerate figures from the full results**

Run: `.venv/bin/python aggregate.py --profile full`
Expected: prints number of scenarios > smoke. Figures should now visibly show the collapse.

- [ ] **Step 5: Eyeball the headline figure**

Open `figures/01_collapse_fnr_max_delta_0.5.png`. Expected: points from regime A (blue) and regime B (red) lie on approximately the same curve when plotted against `n_min`.

- [ ] **Step 6: Commit full-profile results + figures**

```
git add results/ figures/ failed.json 2>/dev/null || true
git -c user.name="dan" -c user.email="daniellopes.ba@gmail.com" commit -m "Full profile results, figures, and summary table"
```

---

## Task 10: FINDINGS.md write-up

**Files:**
- Create: `FINDINGS.md`

- [ ] **Step 1: Pull headline numbers from the aggregated data**

Run:
```
.venv/bin/python - <<'PY'
import json, glob
import pandas as pd
rows = []
for p in glob.glob('results/*.json'):
    r = json.load(open(p))
    rows.append({
        "regime": r["params"]["regime"], "n": r["params"]["n"],
        "p": r["params"]["p"], "delta": r["params"]["delta"],
        "control": r["params"]["control"], "backend": r["params"]["backend"],
        "n_noise": r["params"]["n_noise"],
        "n_min": r["n_min"],
        "fnr_max": r["aggregate"]["fnr_max"]["mean"],
        "fnr_p90": r["aggregate"]["fnr_p90"]["mean"],
        "noise_floor": r["aggregate"]["naive_floor"]["mean"],
        "mean_real": r["aggregate"]["mean_real_importance"]["mean"],
    })
df = pd.DataFrame(rows)
base = df[(~df.control) & (df.backend=='xgboost') & (df.n_noise==5) & (df.delta==0.5)]
for nm in [10, 50, 200, 1000]:
    closest = base.iloc[(base.n_min - nm).abs().argsort()[:3]]
    print(f"--- n_min ~ {nm} ---")
    print(closest[["regime","n","p","n_min","fnr_max","fnr_p90","noise_floor"]].to_string(index=False))
PY
```
Use the output to fill in the table below.

- [ ] **Step 2: Write `FINDINGS.md`**

```markdown
# Findings — Shadow Feature Selection Under Class Imbalance

## Headline

When the minority class is small, the shadow-feature heuristic on normalized
gain importance prunes genuinely predictive features at a rate that depends
almost entirely on the absolute minority count `n_min = round(n·p)`, **not**
on the imbalance ratio `p` itself. A credit-risk regime (p ∈ [0.05, 0.20])
and a fraud regime (p ∈ [0.001, 0.01]) collapse onto the same FNR-vs-`n_min`
curve once they share a common `n_min` range.

See **figures/01_collapse_fnr_max_delta_0.5.png** for the headline result.

## Representative numbers (δ = 0.5, n_noise = 5, XGBoost)

| n_min ≈ | mean noise floor | FNR_max | FNR_p90 |
|---:|---:|---:|---:|
| 10 | <FILL FROM STEP 1> | <…> | <…> |
| 50 | <…> | <…> | <…> |
| 200 | <…> | <…> | <…> |
| 1000 | <…> | <…> | <…> |

## Mechanism

The normalized gain importance constraint (importances sum to 1 across
features) makes the noise-feature *share* a function of the *real* features'
absolute gain. As `n_min` shrinks, the trees' real-feature gain collapses
toward the noise-feature gain — not because the noise features find any
signal, but because the real ones can no longer find theirs. The result is
that the normalized noise floor inflates mechanically. This is confirmed by
the **pure-noise control** (figures/03_mechanism_decomposition.*): with no
real signal, the noise floor stays roughly flat across `n_min`, ruling out
absolute noise inflation as the driver.

## Mitigation

Using the **90th-percentile** noise importance (i.e. a multi-shadow vote)
instead of the **max** noise importance roughly halves the FNR in the
small-`n_min` regime (figures/04_mitigation_p90.*), but it does not save the
heuristic at very small `n_min` — the floor simply outruns the real
features' gain.

## Robustness

The collapse and inflation are reproduced with `n_noise = 20`
(figures/07a_robustness_n_noise_20.*) and with sklearn's
`GradientBoostingClassifier` as a cross-implementation check
(figures/07b_robustness_sklearn_gbm.*).

## Practical guideline

See `figures/summary_table.csv` for an `n_min` → recommended-approach table
suitable for the paper's "Practical Guidelines" section.
```

Replace the `<FILL FROM STEP 1>` cells with the actual numbers from Step 1.

- [ ] **Step 3: Commit**

```
git add FINDINGS.md
git -c user.name="dan" -c user.email="daniellopes.ba@gmail.com" commit -m "FINDINGS.md: headline numbers, mechanism, mitigation, robustness"
```

---

## Self-Review Notes

- Spec coverage: every spec section maps to a task (scaffold → 1; data gen → 2; metrics → 3; scenarios → 4; experiment notebook → 5; orchestration → 6; smoke → 7; aggregation/figures/table → 8; full run → 9; FINDINGS → 10).
- Determinism: `derive_seed` + `scenario_id` ensure reproducibility (tasks 2 & 4).
- Idempotence + retries: covered in task 6.
- Pure-noise control + n_noise=20 robustness + sklearn cross-check: present in `scenarios.py` and verified in tests (task 4).
- All figure types in the spec map to functions in `aggregate.py` (task 8).
- Summary table written to CSV (task 8 / `write_summary_table`).
- No "TBD"s, no "similar to" references; every code step shows the actual code.
