# shadowselect

**Shadow-feature selection that is honest about class imbalance.** This
repository has three parts:

1. a paper, *Shadow-Feature Selection Under Class Imbalance: The Minority
   Count Decides* ([PDF](paper/paper.pdf), sources in [`paper/`](paper/));
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

Requires Python 3.10 or later, NumPy, SciPy, pandas and scikit-learn 1.6 or later.

## Use

```python
from shadowselect import ShadowFeatureSelector, UnivariateScreen, assess_imbalance
from shadowselect.simulation import make_dataset

# 20,000 rows, 0.5% positives (n_min = 100): 5 features shifted by 0.5 SD, 45 irrelevant
X, y = make_dataset(n=20_000, p=0.005, delta=0.5, n_real=5, n_noise=45, seed=0)

# 1. Before selecting: what can this many positives support?
print(assess_imbalance(y, n_features=X.shape[1]))

# 2. Screen with a rank test (Mann-Whitney + Benjamini-Hochberg). Seconds, any size.
screen = UnivariateScreen(alpha=0.1).fit(X, y)
print(screen.summary())  # test, effect, missing share, p-value, adjusted p-value, kept

# 3. Select: Boruta-style rounds against 10 fresh shadows, then binomial tests.
selector = ShadowFeatureSelector(random_state=0, n_jobs=-1)
X_kept = selector.fit_transform(X, y)
print(selector.summary())  # decision, hits, delta_hat, s_hat = n_min * delta_hat^2, ...
```

Both selectors are scikit-learn transformers: they work in `Pipeline`,
`cross_validate` and `GridSearchCV`, pass scikit-learn's estimator checks, and
`get_feature_names_out()` returns the kept columns. Put the selector inside the
pipeline you cross-validate, so that each fold selects on its own rows.

**Inputs.** NumPy arrays, SciPy sparse matrices, and pandas DataFrames with
missing values, `bool`, nullable (`Int64`, `boolean`) and `category` columns.
DataFrames with categoricals reach the learner unchanged, so XGBoost
(`enable_categorical=True`) and LightGBM use them natively; shadows are row
permutations of real columns, so they keep each column's dtype and missing
values. `UnivariateScreen` drops missing values per feature and tests
categoricals with a chi-square test. Pass fit parameters such as
`sample_weight` to `fit` (or `select__sample_weight` through a `Pipeline`).

**Reading the summary.** `s_hat = n_min * delta_hat**2` puts the paper's main
quantity next to each decision. A rejected feature with `underpowered=True`
(`s_hat < 25`) was not shown to be useless: with this many positives a
marginal effect of that size is usually not detectable.

### Parameters

| parameter | default | meaning |
|---|---|---|
| `estimator` | random forest, 200 trees, `max_depth=7`, `class_weight="balanced_subsample"` | any classifier |
| `importance` | `"auto"` | native `feature_importances_` when available, else held-out permutation; or `"native"`, `"permutation"`, a callable |
| `n_shadows` | 10 | shadows per round, drawn from random columns each round; `None` = one per feature (Boruta), which costs power on wide tables |
| `shadow` | `"permute"` | row-shuffled copies of real columns, or `"gaussian"` noise |
| `n_iter` | 20 | rounds; too few for any feature to reach significance is an error |
| `threshold_quantile` | 1.0 | quantile of the shadow importances to beat (1.0 = max) |
| `alpha`, `correction` | 0.05, `"bh"` | level of the one-sided binomial tests, Benjamini-Hochberg across features; `"bonferroni"` (Boruta) controls the family-wise error rate instead |
| `keep_tentative` | `True` | keep features that are neither accepted nor rejected |
| `min_minority` | 100 | `ImbalanceWarning` below this many positives (training split) |
| `n_jobs`, `verbose` | `None`, 0 | parallel rounds (threads); one progress line per round |

After fitting, `expected_null_keep_rate_` is the share of irrelevant features
that Bonferroni decisions would keep as tentative; Benjamini-Hochberg keeps
fewer, so for the default it is an upper bound. It grows with the number of
features and falls with `n_iter`: with 10 shadows and 20 rounds it is 10% for
10 features and 27% for 200 (`shadowselect.null_keep_rate`). In our
benchmark with 50 features, Benjamini-Hochberg kept 12-18% of the irrelevant
ones where Bonferroni kept 18-27%, at almost the same false-negative rate.

### Choosing the learner

The selector only works if the learner does not memorise the minority class.
A learner that does spreads importance over every column it can split on,
shadows included, and then no real feature beats the shadows. The selector
warns (`LearnerWarning`) when features with a large marginal effect never beat
the shadows.

| learner | settings that behaved well with a few hundred positives |
|---|---|
| default random forest | as shipped (depth 7, class-balanced bootstrap); in our benchmark it missed the fewest real features of the learners tried, at every minority count. Fully grown forests inflate shadow importances and are slow |
| XGBoost | `n_estimators=300, max_depth=6, learning_rate=0.1`; 3.x sets the intercept to the prevalence itself, and for 2.x the selector sets `base_score` to the prevalence |
| LightGBM | `num_leaves=15, min_child_samples=50`, row and column subsampling; the selector replaces split-count importance by gain |

With a multi-threaded learner (XGBoost, LightGBM) leave `n_jobs=None` and let
the learner use the cores; with the default forest `n_jobs=-1` runs rounds in
parallel. On 100,000 rows x 100 features (4 cores) one round took 30 s with
the default forest, 5 s with XGBoost and 3 s with LightGBM, so prefer a
booster on large tables. `benchmarks/bench_package.py` measures this on your
machine.

### Practical rules (from the paper)

1. Count positives in the data you select on, not the ratio.
2. Expect to detect only features with $n_{\min}\delta^2 \gtrsim 25$, fewer with many shadows.
3. With XGBoost 2.x, set `base_score=y.mean()` (the selector does this for you; 3.x does it by default).
4. With few positives, screen with a rank test (`UnivariateScreen`); it cannot see interactions.
5. Use repeated rounds, and keep tentative features when positives are few.
6. Class weights, downsampling and early stopping do not fix selection.
7. Below about 25 positives, don't prune on importance at all.

## Examples

| script | shows |
|---|---|
| [`examples/01_quickstart.py`](examples/01_quickstart.py) | `assess_imbalance`, the screen, the selector and its summary |
| [`examples/02_small_minority_pitfall.py`](examples/02_small_minority_pitfall.py) | the one-shot recipe vs. the selector at 40 positives |
| [`examples/03_real_data_pipeline.py`](examples/03_real_data_pipeline.py) | real data, selection inside cross-validation, as the minority class shrinks |
| [`examples/04_fraud_table.py`](examples/04_fraud_table.py) | a raw fraud-style DataFrame (missing values, categoricals, weights) with LightGBM and time-ordered CV |

## Reproduce the paper

```bash
pip install -e . -r requirements-experiments.txt
cd experiments
for e in e1_first_split e2_sweep e3_boosting_rounds e4_mitigations e5_backends \
         e6_univariate e7_rounds e8_many_shadows; do python $e.py; done
python e2_sweep.py --n-noise 20
python make_figures.py && python paper_numbers.py
cd ../paper && ./build.sh            # needs pandoc >= 3; PDF via xelatex or Node+Playwright
```

The whole set takes about 2.5 hours on 4 cores. Add `--quick` to any
experiment for a smoke run. Results are committed in `results/`, figures in
`figures/`.

## The paper

[`paper/paper.pdf`](paper/paper.pdf) is the current version. Every main claim
is either proved (Section 3 and Appendix C) or stated as a conjecture, and
Appendix A compares each result with the simulations. To cite it:

```bibtex
@misc{lopes2026shadow,
  author = {Daniel Lopes},
  title  = {Shadow-Feature Selection Under Class Imbalance: The Minority Count Decides},
  year   = {2026},
  howpublished = {\url{https://github.com/Daniellopesba/shadow-feature-selection}}
}
```

## Repository layout

```
src/shadowselect/   selector.py, screening.py, diagnostics.py, theory.py, simulation.py
benchmarks/         runtime and memory benchmarks of the package
tests/              unit tests (pytest), incl. numerical checks of the propositions
experiments/        e1..e8 experiments, make_figures.py, paper_numbers.py
results/            per-repeat results (csv.gz), run metadata, paper_numbers.md
figures/            paper figures (png, pdf) and tables.md
paper/              paper.pdf, sources (Markdown + BibTeX) and build script
```

## License

MIT
