# Shadow Feature Selection Under Class Imbalance — Experiment Harness Design

**Date:** 2026-06-29
**Status:** Approved (for implementation)

## 1. Purpose

Produce the reproducible empirical evidence for a methods paper showing that the
**shadow-feature heuristic for feature selection** (inject random noise
features; prune any real feature whose importance falls below the noise floor)
has a systematic failure mode under class imbalance when applied to
**normalized gain importance**. As the minority class shrinks, real features
lose absolute gain, so noise features' *normalized share* rises mechanically
and genuinely predictive features get pruned.

The headline empirical claim the harness must support: when failure rate
(FNR) is plotted against the **absolute minority count `n_min = round(n·p)`**
(log x), points from a moderate-imbalance "credit risk" regime and a
severe-imbalance "fraud" regime **collapse onto a single curve** — i.e. the
controlling variable is `n_min`, not the imbalance ratio `p`.

Boruta is the inspiration only. We do **not** implement or test Boruta. The
object of study is the generic shadow-feature heuristic on normalized gain
importance.

## 2. Fixed Decisions (locked by user; do not relitigate)

- **Importance metric:** normalized gain importance (sums to 1 across features).
- **Primary model:** `xgboost.XGBClassifier`, `importance_type="gain"`,
  `tree_method="hist"`. Reasonable defaults; not the subject of the paper.
- **Cross-check model:** sklearn `GradientBoostingClassifier` on a small subset
  of scenarios (its `feature_importances_` is gain-based MDI). **Not**
  `HistGradientBoostingClassifier` (no gain importance).
- **x-axis of primary plots:** `n_min = round(n·p)`.
- **Two regimes are intentionally chosen to overlap in `n_min`** so the
  collapse is visible: Credit-risk (moderate imbalance / modest n) and Fraud
  (severe imbalance / large n).
- **Pure-noise control** runs alongside every signal scenario, to prove the
  noise-floor inflation is a normalization effect (the floor should stay
  roughly flat in the control).
- **Exact positives:** use `round(n·p)` positives, not a Bernoulli draw.
- **Deterministic seeding:** every repeat's seed is derived from its
  scenario id, so the full sweep is reproducible.

## 3. Project Layout

```
shadow-feature-selection/
├── .venv/                                  # uv-managed, Python 3.12
├── .gitignore
├── README.md                               # reproduce: smoke then full
├── requirements.txt                        # pinned versions
├── config.yaml                             # `smoke` and `full` profiles
├── scenarios.py                            # builds scenario list w/ deterministic ids
├── experiment.ipynb                        # parameterized notebook (papermill)
├── run_sweep.py                            # idempotent orchestration loop
├── aggregate.py                            # results → figures + summary CSV
├── results/                                # one JSON per scenario_id
├── executed/                               # papermill-executed notebooks per scenario
├── figures/                                # PNG + PDF cross-scenario plots
├── failed.json                             # scenarios that exhausted retries
└── FINDINGS.md                             # ~1 page write-up
```

## 4. Environment & Reproducibility

- `uv venv --python 3.12 .venv` (Python 3.14 wheels for xgboost / pyarrow may
  be incomplete; 3.12 is safe).
- `uv pip install -r requirements.txt`.
- `requirements.txt` pins concrete versions for numpy, scipy, pandas,
  scikit-learn, xgboost, matplotlib, jupyter, papermill, pyyaml, pyarrow,
  ipykernel. Versions are recorded at install time and re-emitted by every
  executed notebook (env block).
- A `scenario_id = md5(json.dumps(params, sort_keys=True))[:12]` keys every
  scenario file system entry.
- A `per_repeat_seed = int(md5(f"{scenario_id}:{repeat_idx}").hexdigest()[:8], 16)`
  derives every per-repeat seed.

## 5. Data Generation (per repeat)

- `n_real = 5` genuinely predictive features, `n_noise = 5` pure-noise features
  (plus a separate robustness pass with `n_noise = 20`).
- All features i.i.d. `N(0, 1)`.
- For the minority class, shift the `n_real` real features by `+δ`
  (location-shift signal). Noise features are independent of `y` always.
- Use **exactly `round(n·p)` positives**, the rest are negatives.
- **Pure-noise control:** δ effectively 0 (real features unshifted). The
  noise floor should stay roughly flat across `n_min` in the control.
- Degenerate cases (`n_min < 5`) are skipped and flagged.

## 6. Scenario Grid

### Regime A — Credit risk
- `p ∈ {0.05, 0.10, 0.20}`
- `n ∈ {500, 1000, 2500, 5000}`
- `n_min` spans ~25 → ~1000

### Regime B — Fraud
- `p ∈ {0.001, 0.005, 0.01}`
- `n ∈ {10000, 50000, 200000}`
- `n_min` spans ~10 → ~2000

### Crossed with
- `δ ∈ {0.3, 0.5, 1.0}`
- `control ∈ {False, True}` (control = δ unused, real features unshifted)
- All scenarios use `n_noise = 5` and the XGBoost backend by default.

### Robustness additions
- `n_noise = 20` pass: δ=0.5, full grid (no control), XGBoost only.
- sklearn-GBM cross-check: ~6 scenarios from Regime A at moderate n with δ=0.5
  (e.g. n∈{1000, 2500, 5000} × p∈{0.10, 0.20}). XGBoost run already exists for
  these for direct comparison.

### Smoke profile (validates pipeline cheaply)
- Regime A: n∈{500, 1000}, p∈{0.10, 0.20}, δ=0.5
- Regime B: n=5000, p=0.01, δ=0.5
- Plus pure-noise control variants
- Plus 1 `n_noise=20` and 1 sklearn-GBM scenario
- `n_repeats=2`
- ~20 scenarios, single-digit minutes total

## 7. Modeling

- `XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
  tree_method="hist", importance_type="gain", n_jobs=-1, eval_metric="logloss",
  verbosity=0)`.
- `GradientBoostingClassifier(n_estimators=200, max_depth=3,
  learning_rate=0.1)` for cross-check.
- Default `n_repeats = 25`; auto-reduce to `15` for `n ≥ 50000`. Smoke uses 2.

## 8. Metrics (per repeat → aggregated)

For each fitted model, normalize gain importances to sum to 1, then compute:

- **Naive shadow floor:** `max(noise_importances)`.
- **Multi-shadow P90 floor:** `np.percentile(noise_importances, 90)`.
- **Mean real-feature importance.**
- **FNR_max:** fraction of real features with importance < naive floor.
- **FNR_p90:** fraction of real features with importance < P90 floor.
- **n_min** (carried through for plotting).

Per-scenario aggregates: mean, median, p90, std. Per-repeat values retained
for distributional plots.

## 9. Per-Scenario Output

`results/<scenario_id>.json`:

```json
{
  "scenario_id": "...",
  "params": { ...complete scenario params... },
  "library_versions": { "xgboost": "...", "sklearn": "...", "numpy": "...", "python": "..." },
  "n_min": 50,
  "n_repeats_attempted": 25,
  "n_repeats_used": 25,
  "wall_time_seconds": 12.34,
  "per_repeat": {
    "naive_floor": [...],
    "p90_floor": [...],
    "mean_real_importance": [...],
    "fnr_max": [...],
    "fnr_p90": [...]
  },
  "aggregate": {
    "naive_floor": {"mean": ..., "median": ..., "p90": ..., "std": ...},
    "p90_floor": {...},
    "mean_real_importance": {...},
    "fnr_max": {...},
    "fnr_p90": {...}
  }
}
```

## 10. Orchestration Loop (`run_sweep.py`)

1. Build target scenario set from `config.yaml` profile via `scenarios.py`.
2. Read `results/` and `failed.json`; compute remaining = target − done − failed.
3. Pass loop, up to `max_passes` (default 3):
   - For each remaining scenario, run `experiment.ipynb` via papermill (up to
     `max_attempts_per_pass` = 2 retries on transient failure).
   - On success: result file is written by the notebook; executed copy lands
     in `executed/<scenario_id>.ipynb`.
   - On exhausted retries this pass: record in `failed.json` with last
     exception + traceback, drop from must-cover set, log loudly.
   - After each scenario, print `covered N / remaining M / failed F`.
4. Terminate when remaining is empty or max passes reached.
5. Return non-zero exit if any scenario is still uncovered and unfailed.

## 11. Aggregation (`aggregate.py`)

Reads all `results/*.json`, builds a tidy DataFrame, and emits:

**Figures (PNG @ 150+ dpi + PDF/SVG):**

1. **Collapse plot (headline):** FNR_max vs `n_min` (log x); regimes overlaid;
   color=regime, marker=p. Main panel δ=0.5; δ∈{0.3,1.0} as side panels.
2. **Noise floor inflation:** normalized noise floor (max) vs `n_min`, both regimes.
3. **Mechanism / decomposition:** mean real-feature importance and noise floor
   on shared axes vs `n_min`, with pure-noise control overlay (flat floor).
4. **Mitigation:** FNR_max vs FNR_p90 vs `n_min`.
5. **FNR heatmaps** over the (n, p) grid, one per regime.
6. **Signal interaction:** FNR vs `n_min` faceted by δ.
7. **Robustness:** collapse plot repeated with `n_noise=20` and sklearn-GBM.

**Tables:**

- `figures/summary_table.csv` — `n_min` bins → mean noise floor → FNR at δ=0.5
  → recommended approach. Spans both regimes. This is the paper's
  "Practical Guidelines" table.

## 12. Acceptance Criteria

1. Smoke profile completes end-to-end; results, executed notebooks, and at least
   one example of each figure type are produced without error.
2. Full profile completes (all target scenarios covered or explicitly failed
   with reasons in `failed.json`).
3. The collapse plot visibly demonstrates regime-independent dependence on `n_min`.
4. The mechanism plot + pure-noise control show the normalization mechanism.
5. `figures/summary_table.csv` is written.
6. `FINDINGS.md` exists with headline numbers (FNR at representative `n_min`
   values), confirms the collapse, and states the mechanism in one paragraph
   with figure references.

## 13. Out of Scope

- Boruta itself.
- Hyperparameter tuning of XGBoost / sklearn-GBM (the paper isn't about tuning).
- Permutation importance, SHAP, or other importance metrics.
- Real-world datasets — synthetic Gaussian only.
- Statistical hypothesis testing on the collapse curve.
