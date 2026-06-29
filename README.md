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
.venv/bin/python -m ipykernel install --user --name shadow-feature-selection --display-name "shadow-feature-selection"

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
