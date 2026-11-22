# 6. Practical guidance

## 6.1 Decision rules

**Rule 1: compute $n_{\min}$ before selecting anything.** The number of
minority-class samples, not the imbalance ratio, sets what can be detected.
A 10% default rate on 300 loans ($n_{\min} = 30$) is harder than a 0.5% fraud
rate on 100,000 transactions ($n_{\min} = 500$). If you use cross-validation or a
held-out split, count the positives in the training part.

**Rule 2: translate $n_{\min}$ into a detectable effect.** By Theorem 2, a
feature's detectability depends on $s = n_{\min}\delta^2$, where $\delta$ is its
standardised mean difference between classes. The ranges below are the
cell-mean FNRs of the one-shot max-shadow rule in E2 (300-tree XGBoost,
prevalence intercept, 5 shadows), grouped by $s$:

| $s = n_{\min}\delta^2$ | max-shadow FNR (range over cells; median) | example |
|:--|:--|:--|
| < 5 | 64–84%; 66% | $n_{\min} = 50$, δ = 0.3 |
| 5–10 | 38–60%; 49% | $n_{\min} = 100$, δ = 0.3 |
| 10–25 | 22–67%; 35% | $n_{\min} = 50$, δ = 0.5 |
| 25–60 | 0–16%; 10% | $n_{\min} = 150$, δ = 0.5 |
| > 60 | at most 4% | $n_{\min} = 250$, δ = 0.5 |

These numbers assume about five shadows. With more shadows the bar rises
(Corollary 1): with 200 shadows, FNR at $n_{\min} = 100$, δ = 0.5 ($s = 25$) is
99% instead of 11% (E8). `shadowselect.assess_imbalance(y, n_shadows=...)`
prints the smallest $\delta$ that wins the first split against the given number
of shadows. It is a lower bound: in every cell we measured, full ensembles
need more (Conjecture 2).

**Rule 3: if you use XGBoost 2.x, set `base_score` to the prevalence.**
`XGBClassifier(base_score=y.mean())` removes most of the extra failure at
small $p$ (Proposition 5, Section 4.3) and makes the result depend on $n_{\min}$
rather than on $n$. It is not free in every cell: at $n_{\min} = 10$, δ = 1 it
raised FNR from 0.20 to 0.36. Outside that corner it helped or made no
difference.

**Rule 4: screen with a rank test when positives are few.** For each
feature, a Mann–Whitney test between the classes with Benjamini–Hochberg
control (`shadowselect.UnivariateScreen`) uses every row and has a known
null distribution. Under a location shift it needs about $s \approx 9$ for 90%
power where a tree split needs about 13.5 (Corollary 2), and at matched
false positive rate it lost far fewer relevant features than any tree-shadow
method at $n_{\min} \le 50$ (Table 2). It cannot see interactions, so use it to
decide what is clearly in, not to drop what a model finds useful.

**Rule 5: use repeated rounds, and decide what to do with tentative features.**
Prefer a Boruta-style test over a single comparison with the max shadow. A
one-shot threshold also depends on how many shadows you add (Corollary 1).
With fewer than about 100 positives, keeping only "accepted" features drops
most moderately predictive ones. Keep tentative features as well
(`keep_tentative=True`, the package default). The price is an
irrelevant-feature rate that depends on the number of rounds and shadows
(Proposition 7): about 15% with 20 rounds and 10 shadows in our
simulations, lower with more rounds (Section 5.1). Alternatively, treat
tentative features as a list for a human to review.

**Rule 6: do not rely on class weights, downsampling or early stopping to fix
selection.** None of them changes $n_{\min}$, and none of them helped in E4.

**Rule 7: below about 25 positives, do not prune on importance.** No method
we tested worked well there, the rank test included: at $s < 6$ its power is
below 0.7 (Proposition 8). Keep features on domain grounds, use a penalised model
whose regularisation path you can inspect, or collect more positive examples.
Doubling $n_{\min}$ helps as much as a $\sqrt 2$ stronger feature does.

## 6.2 The `shadowselect` package

The package implements these rules as a scikit-learn transformer.

```python
from xgboost import XGBClassifier
from shadowselect import ShadowFeatureSelector, UnivariateScreen, assess_imbalance

print(assess_imbalance(y))  # n_min, detectable effect, recommendation

screen = UnivariateScreen(alpha=0.1).fit(X, y)  # Rule 4: rank test + BH
screen.summary()

selector = ShadowFeatureSelector(
    XGBClassifier(n_estimators=300, max_depth=6, base_score=y.mean()),  # Rule 3
    n_iter=20,  # rounds, each with fresh shuffled shadows
    random_state=0,
)  # importance="auto", keep_tentative=True (Rule 5)
X_selected = selector.fit_transform(X, y)
selector.summary()  # decision, hits, p-values per feature
```

`importance="auto"` uses the estimator's native importances when it has
them and held-out permutation importance otherwise; the experiments found
no advantage for permutation importance (Section 5, item 7). The default
estimator is a random forest. The selector works with any classifier. It
emits an `ImbalanceWarning` when the minority count in the training data is
below a configurable threshold (default 50). It warns when `n_iter` is too
small for any feature to reach significance, and it fits inside `Pipeline`
and `GridSearchCV`. The test suite includes scikit-learn's estimator checks
and numerical checks of Propositions 1, 2, 4, 5, 6, 8 and 9.
