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
where $\delta$ is the feature's standardised effect size. The paper proves this
for the first boosting round: XGBoost's split gain is exactly Pearson's
chi-square statistic, its small-prevalence limit depends only on $n_{\min}$ and
$\delta$, and as $n_{\min}$ grows only on $s$. It also proves that the max-shadow
rule keeps an irrelevant feature with probability at most $1/(K+1)$ for $K$
shadows whatever the imbalance, so imbalance only costs power, and that a
simple rank test on the same data has more power ($3s/\pi$ against $2s/\pi$).
Across 63 simulated scenarios with 300-tree models, $s$ explains 94% of the
variance in the failure rate (90% out of sample), and $p$ explains almost
none. Part of what looked like an imbalance effect is XGBoost 2.x's default
intercept, which is about 0.12 whenever $p$ is small. Set `base_score` to the
prevalence.

| positives × effect² ($s = n_{\min}\delta^2$) | features wrongly dropped by the one-shot max-shadow rule (5 shadows) |
|:--|:--|
| below 5 (e.g. 50 positives, δ = 0.3) | 64–84% |
| 10–25 (e.g. 50 positives, δ = 0.5) | 22–67%, typically 30–45% |
| above 60 (e.g. 250 positives, δ = 0.5) | at most 4% |

More shadows make it worse: with 200 shadows, 100 positives and δ = 0.5,
99% are dropped.

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
from shadowselect import ShadowFeatureSelector, UnivariateScreen, assess_imbalance

# 1. Before selecting: what can this target support?
print(assess_imbalance(y))
# n=20000, n_minority=40 (p=0.0020); first-split minimum detectable shift ~ 0.65 SD. Marginal: ...

# 2. With few positives, screen with a rank test (Mann-Whitney + Benjamini-Hochberg).
screen = UnivariateScreen(alpha=0.1).fit(X, y)
print(screen.summary())  # effect, p-value, adjusted p-value, kept

# 3. Select. Boruta-style rounds with fresh shuffled shadows and a binomial test.
selector = ShadowFeatureSelector(
    XGBClassifier(n_estimators=300, max_depth=6, base_score=y.mean()),
    n_iter=20,
    random_state=0,
)  # importance="auto", keep_tentative=True by default
X_kept = selector.fit_transform(X, y)
print(selector.summary())  # decision, hits, p-values per feature
```

`ShadowFeatureSelector` and `UnivariateScreen` are scikit-learn transformers,
so they work in `Pipeline` and `GridSearchCV`. They accept pandas
DataFrames, and `get_feature_names_out()` returns the kept columns. Both pass
scikit-learn's estimator checks. When the training data has fewer than
`min_minority` positives (default 50) the selector raises an
`ImbalanceWarning`.

| parameter | default | meaning |
|---|---|---|
| `estimator` | `RandomForestClassifier(200 trees)` | any classifier |
| `importance` | `"auto"` | native `feature_importances_` when available, else held-out permutation; or `"native"`, `"permutation"`, a callable |
| `shadow` | `"permute"` | shuffled copies of the features (Boruta) or `"gaussian"` noise |
| `n_iter` | 20 | rounds; more rounds lower the false positive rate (paper, Prop. 7) |
| `threshold_quantile` | 1.0 | quantile of the shadow importances to beat (1.0 = max) |
| `alpha` | 0.05 | family-wise level of each one-sided binomial test (Bonferroni) |
| `keep_tentative` | `True` | keep features that are neither accepted nor rejected |

Practical rules (paper, Section 6):

1. Count positives in the data you select on, not the ratio.
2. Expect to detect only features with $n_{\min}\delta^2 \gtrsim 25$, fewer with many shadows.
3. With XGBoost 2.x, set `base_score=y.mean()`.
4. With few positives, screen with a rank test (`UnivariateScreen`); it cannot see interactions.
5. Use repeated rounds, and keep tentative features when positives are few.
6. Class weights, downsampling and early stopping do not fix selection.
7. Below about 25 positives, don't prune on importance at all.

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
for e in e1_first_split e2_sweep e3_boosting_rounds e4_mitigations e5_backends \
         e6_univariate e7_rounds e8_many_shadows; do python $e.py; done
python e2_sweep.py --n-noise 20
python make_figures.py && python ledger.py
cd ../paper && ./build.sh            # needs pandoc >= 3; PDF via xelatex or Node+Playwright
```

The whole set takes about 2.5 hours on 4 cores. Add `--quick` to any
experiment for a smoke run. Results are committed in `results/`, figures in
`figures/`.

## How the paper was checked

Each main claim is either proved (paper Section 3 and Appendix C) or
labelled a conjecture. Each proved result was turned into a numbered
prediction, tested by an experiment and recorded with the measured value and
a verdict in [`docs/LEDGER.md`](docs/LEDGER.md). When an experiment disagreed, the theory
was revised and independently re-reviewed. The ledger keeps that history.

## Repository layout

```
src/shadowselect/   selector.py, screening.py, diagnostics.py, theory.py, simulation.py
tests/              unit tests (pytest), incl. numerical checks of the propositions
experiments/        e1..e8 experiments, make_figures.py, ledger.py
results/            per-repeat results (csv.gz) + run metadata
figures/            paper figures (png, pdf) and tables.md
paper/              paper sources (Markdown + BibTeX) and build script
docs/               LEDGER.md (theory vs experiment), history/ (original design notes)
```

## License

MIT
