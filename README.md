# shadowselect

**Shadow-feature selection that is honest about class imbalance.** This
repository has three parts:

1. a paper, *Shadow-Feature Selection Under Class Imbalance: The Minority
   Count Decides* ([`paper/`](paper/));
2. a small, tested Python package, `shadowselect`, for running Boruta-style
   selection and checking whether your data can support it;
3. scripts that reproduce every figure and number in the paper
   ([`experiments/`](experiments/)).

## The finding in one paragraph

The shadow-feature trick adds known-useless features ("shadows") and drops
real features whose importance falls below theirs. When the positive class
is rare, the trick drops genuinely predictive features, often most of them.
How often is governed by the **number of positive examples** $n_{\min}$, not the
imbalance ratio. More precisely, it is governed by $s = n_{\min}\cdot\delta^2$,
where $\delta$ is the feature's standardised effect size. At the first boosting
round this is a theorem: XGBoost's split gain is exactly Pearson's chi-square
statistic, and its distribution depends on $n_{\min}$ and $\delta$ only. Across 63
simulated scenarios with 300-tree models, $s$ explains 94% of the variance in
the failure rate, and $p$ explains none of it. Part of what looked like an
imbalance effect is XGBoost 2.x's default intercept, which is about 0.12
whenever $p$ is small. Set `base_score` to the prevalence.

| positives × effect² ($s = n_{\min}\delta^2$) | features wrongly dropped by the max-shadow rule |
|:--|:--|
| below 5 (e.g. 50 positives, δ = 0.3) | more than half |
| 10–25 (e.g. 50 positives, δ = 0.5) | 10–30% |
| above 60 (e.g. 250 positives, δ = 0.5) | about none |

## Install

```bash
pip install git+https://github.com/Daniellopesba/shadow-feature-selection
# optional learners
pip install xgboost lightgbm
```

Requires Python 3.10 or later, NumPy, SciPy and scikit-learn.

## Use

```python
from xgboost import XGBClassifier
from shadowselect import ShadowFeatureSelector, assess_imbalance

# 1. Before selecting: what can this target support?
print(assess_imbalance(y))
# n=40000, n_minority=40 (p=0.0010); first-split minimum detectable shift ~ 0.75 SD. Unreliable: ...

# 2. Select. Boruta-style rounds with fresh shuffled shadows and a binomial test.
selector = ShadowFeatureSelector(
    XGBClassifier(n_estimators=300, max_depth=6, base_score=y.mean()),
    importance="permutation",   # held-out permutation importance; "native" = training gain
    n_iter=20,
    keep_tentative=True,        # recommended below ~100 positives
    random_state=0,
)
X_kept = selector.fit_transform(X, y)
print(selector.summary())       # decision, hits, p-values per feature
```

`ShadowFeatureSelector` is a scikit-learn transformer, so it works in
`Pipeline` and `GridSearchCV`. It accepts pandas DataFrames, and
`get_feature_names_out()` returns the kept columns. It passes scikit-learn's
estimator checks. When the training data has fewer than `min_minority`
positives (default 50) it raises an `ImbalanceWarning`.

| parameter | default | meaning |
|---|---|---|
| `estimator` | `HistGradientBoostingClassifier` | any classifier |
| `importance` | `"permutation"` | `"native"` (`feature_importances_`), `"permutation"` (held-out), or a callable |
| `shadow` | `"permute"` | shuffled copies of the features (Boruta) or `"gaussian"` noise |
| `n_iter` | 20 | rounds |
| `threshold_quantile` | 1.0 | quantile of the shadow importances to beat (1.0 = max) |
| `alpha` | 0.05 | family-wise level of the two-sided binomial test (Bonferroni) |
| `keep_tentative` | `False` | keep features that are neither accepted nor rejected |

Practical rules (paper, Section 6):

1. Count positives in the data you select on, not the ratio.
2. Expect to detect only features with $n_{\min}\delta^2 \gtrsim 25$.
3. With XGBoost 2.x, set `base_score=y.mean()`.
4. Use repeated rounds, and keep tentative features when positives are few.
5. Class weights, downsampling and early stopping do not fix selection.
6. Below about 25 positives, don't prune on importance at all.

## Examples

| script | shows |
|---|---|
| [`examples/01_quickstart.py`](examples/01_quickstart.py) | `assess_imbalance`, then selection with a random forest |
| [`examples/02_small_minority_pitfall.py`](examples/02_small_minority_pitfall.py) | the one-shot recipe vs. the selector at 40 positives |
| [`examples/03_real_data_pipeline.py`](examples/03_real_data_pipeline.py) | real data (breast cancer + noise columns) as the minority class shrinks |

## Reproduce the paper

```bash
pip install -e . -r requirements-experiments.txt
cd experiments
for e in e1_first_split e2_sweep e3_boosting_rounds e4_mitigations e5_backends; do python $e.py; done
python e2_sweep.py --n-noise 20
python make_figures.py && python ledger.py
cd ../paper && ./build.sh            # needs pandoc >= 3
```

The whole set takes about 1.5 hours on 4 cores. Add `--quick` to any
experiment for a smoke run. Results are committed in `results/`, figures in
`figures/`.

## How the paper was checked

Each theoretical claim is a numbered prediction, tested by an experiment and
recorded with the measured value and a verdict in
[`docs/LEDGER.md`](docs/LEDGER.md). When an experiment disagreed, the theory
was revised and independently re-reviewed. The ledger keeps that history.

## Repository layout

```
src/shadowselect/   selector.py, diagnostics.py, theory.py, simulation.py
tests/              unit tests (pytest), incl. numerical checks of the propositions
experiments/        e1..e5 experiments, make_figures.py, ledger.py
results/            per-repeat results (csv.gz) + run metadata
figures/            paper figures (png, pdf) and tables.md
paper/              paper sources (Markdown + BibTeX) and build script
docs/               LEDGER.md (theory vs experiment), history/ (original design notes)
```

## License

MIT
