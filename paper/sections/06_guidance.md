# 6. Practical guidance

## 6.1 Decision rules

**Rule 1: compute $n_{\min}$ before selecting anything.** The number of
minority-class samples, not the imbalance ratio, sets what can be detected.
A 10% default rate on 300 loans ($n_{\min} = 30$) is harder than a 0.5% fraud
rate on 100,000 transactions ($n_{\min} = 500$). If you use cross-validation or a
held-out split, count the positives in the training part.

**Rule 2: translate $n_{\min}$ into a detectable effect.** By Section 3.4, a
feature is detectable when $s = n_{\min}\delta^2$ is large enough. Here $\delta$ is
its standardised mean difference between classes. From Table 1 and Figure 3,
with the max-shadow rule:

| $s = n_{\min}\delta^2$ | max-shadow FNR | example |
|:--|:--|:--|
| < 5 | above 50% | $n_{\min} = 50$, δ = 0.3 |
| 10–25 | 10–30% | $n_{\min} = 50$, δ = 0.5 |
| > 60 | about 0 | $n_{\min} = 250$, δ = 0.5 |

`shadowselect.assess_imbalance(y)` prints the smallest $\delta$ that is detectable
at the first split for your $n_{\min}$. It is a lower bound: full ensembles
need somewhat more.

**Rule 3: if you use XGBoost 2.x, set `base_score` to the prevalence.**
`XGBClassifier(base_score=y.mean())` costs nothing. It removes most of the
extra failure at small $p$ (Section 4.3) and makes the result depend on
$n_{\min}$ rather than on $n$. This advice also stands on its own, outside
feature selection.

**Rule 4: use repeated rounds, and decide what to do with tentative features.**
Prefer a Boruta-style test over a single comparison with the max shadow. A
one-shot threshold also depends on how many shadows you add (Section 4.3).
With fewer than about 100 positives, keeping only "accepted" features drops
most moderately predictive ones. Keep tentative features as well
(`keep_tentative=True`) and accept an irrelevant-feature rate of roughly 15%.
Alternatively, treat tentative features as a list for a human to review.

**Rule 5: do not rely on class weights, downsampling or early stopping to fix
selection.** None of them changes $n_{\min}$, and none of them helped in E4.

**Rule 6: below about 25 positives, do not prune on importance.** No method
we tested worked there. Keep features on domain grounds, use a penalised model
whose regularisation path you can inspect, or collect more positive examples.
Doubling $n_{\min}$ helps as much as a $\sqrt 2$ stronger feature does.

## 6.2 The `shadowselect` package

The package implements these rules as a scikit-learn transformer.

```python
from xgboost import XGBClassifier
from shadowselect import ShadowFeatureSelector, assess_imbalance

print(assess_imbalance(y))          # n_min, detectable effect, recommendation

selector = ShadowFeatureSelector(
    XGBClassifier(n_estimators=300, max_depth=6, base_score=y.mean()),
    importance="permutation",        # or "native" for training-set gain
    n_iter=20,                       # rounds, each with fresh shuffled shadows
    keep_tentative=True,             # Rule 4
    random_state=0,
)
X_selected = selector.fit_transform(X, y)
selector.summary()                   # decision, hits, p-values per feature
```

The selector works with any classifier. It emits an `ImbalanceWarning` when
the minority count in the training data is below a configurable threshold
(default 50). It warns when `n_iter` is too small for any feature to reach
significance, and it fits inside `Pipeline` and `GridSearchCV`. The test
suite includes scikit-learn's estimator checks.
