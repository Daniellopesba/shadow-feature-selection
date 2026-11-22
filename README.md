# shadowselect

**Shadow-feature selection that is honest about class imbalance.** This
repository has three parts:

1. a paper, *Shadow-Feature Selection Under Class Imbalance: How the Minority
   Count Limits What Can Be Found* ([PDF](paper/paper.pdf), sources in [`paper/`](paper/));
2. a small, tested Python package, `shadowselect`, for running Boruta-style
   selection and checking whether your data can support it;
3. scripts that reproduce every figure and number in the paper
   ([`experiments/`](experiments/)).

## The finding in one paragraph

The shadow-feature trick adds known-useless features ("shadows") and drops
real features whose importance falls below theirs. When the positive class
is rare, the trick drops genuinely predictive features, often most of them.
How often is governed by the **number of positive examples** $n_{\min}$ and
the effect size, not by the imbalance ratio: through $s = n_{\min}\delta^2$ for
a standardised mean shift $\delta$, or its rank analogue
$s_{\mathrm{AUC}} = 4\pi n_{\min}(\mathrm{AUC} - 1/2)^2$ for any feature.
The paper proves this for the first tree of a boosted ensemble (its split gain
is Pearson's chi-square, and as the prevalence goes to zero the root split
depends only on $n_{\min}$ and where the positives sit among the negatives).
It also proves that the max-shadow rule keeps an irrelevant feature with
probability at most $1/(K+1)$ for $K$ shadows whatever the imbalance, so
imbalance costs power, not false positives, and that a Mann-Whitney test on
the same data has power that depends only on $s_{\mathrm{AUC}}$. Every proved
statement was checked against simulation criteria fixed in advance, and full
300-tree ensembles fail at least as often as the first-tree theory predicts.
In XGBoost 1.x and 2.x much of what looks like an imbalance effect is the
default intercept: set `base_score` to the prevalence (XGBoost 3 does this by
default).

| positives × effect² ($s = n_{\min}\delta^2$) | example | features wrongly dropped by the one-shot max-shadow rule (5 shadows), median over scenarios |
|:--|:--|:--|
| below 5 | 50 positives, δ = 0.3 | 67% (58–85%) |
| 5–10 | 100 positives, δ = 0.3 | 48% (38–53%) |
| 10–25 | 50 positives, δ = 0.5 | 32% (26–58%) |
| 25–60 | 150 positives, δ = 0.5 | 7% (0–25%) |
| 60 or more | 250 positives, δ = 0.5 | 0% (0–3%) |

More shadows make it worse: with 200 shadows, 100 positives and δ = 0.5,
99% are dropped. Below about 25 positives no method we tested separates weak
signal from noise reliably.

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
for e in e1_first_split e2_sweep e3_boosting_rounds e4_mitigations e4b_baselines \
         e4c_borutapy e4w_wide e5_backends e6_univariate e7_rounds e8_many_shadows \
         e9_robustness e10_discrete_level e11_real_data e12_intercepts; do python $e.py; done
python make_figures.py      # figures/ and paper/generated/ tables
python paper_numbers.py     # acceptance criteria and every number quoted in the paper
cd ../paper && ./build.sh   # pdflatex + bibtex (TeX Live 2023 or later)
```

Two runs need other XGBoost versions in separate environments (only `xgboost`
changes): `e2_sweep.py` again under XGBoost 3.1.3, and `e12_intercepts.py`
under 1.7.6, 2.1.3 and 3.1.3; see `requirements-experiments.txt`.
`e11_real_data.py` downloads five tables from PMLB and checks their SHA-256
digests. Add `--quick` to any experiment for a smoke run. Results are
committed in `results/`, figures in `figures/`, and `results/*.meta.json`
records each run's library versions, commit and wall time.

## The paper

[`paper/paper.pdf`](paper/paper.pdf) is the current version;
`paper/build.sh` also writes an anonymised build for double-blind review to
`paper/build/paper_anonymous.pdf`. Every main claim is either proved (Section
3 and Appendix A) or stated as a conjecture, and Appendix D compares each
result with the simulations under acceptance criteria fixed in advance
(`experiments/criteria.py`). To cite it:

```bibtex
@misc{lopes2026shadow,
  author = {Daniel Lopes},
  title  = {Shadow-Feature Selection Under Class Imbalance: How the Minority Count Limits What Can Be Found},
  year   = {2026},
  howpublished = {\url{https://github.com/Daniellopesba/shadow-feature-selection}}
}
```

## Repository layout

```
src/shadowselect/   selector.py, screening.py, diagnostics.py, theory.py, simulation.py
tests/              unit tests (pytest), incl. numerical checks of the propositions
experiments/        e1..e12 experiments, criteria.py, make_figures.py, paper_numbers.py
results/            per-repeat results (csv.gz), run metadata, paper_numbers.json
figures/            paper figures (png, pdf)
paper/              paper.pdf, LaTeX sources (TMLR style), generated tables, build script
```

## License

MIT
