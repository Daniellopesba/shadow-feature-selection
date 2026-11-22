# 2. Setup

## 2.1 Data and notation

We observe $n$ samples $(x_i, y_i)$ with $x_i \in \mathbb R^d$ and $y_i \in \{0,1\}$.
The **minority count** is $n_{\min} = \sum_i y_i$ (we label the minority class
1), and $\hat p = n_{\min}/n$ is the prevalence. Throughout, "imbalance ratio"
means $p$ and "minority count" means $n_{\min}$.

A feature $X_j$ is **relevant** if it is not independent of $Y$, and
**irrelevant** (null) otherwise. The experiments use a Gaussian location-shift
model with exactly $n_{\min} = \operatorname{round}(np)$ positives:

* $d_{\text{real}}$ relevant features, $X_j \mid Y{=}0 \sim N(0,1)$ and
  $X_j \mid Y{=}1 \sim N(\delta, 1)$;
* $d_{\text{noise}}$ irrelevant features, $X_j \sim N(0,1)$ independent of $Y$;
* all features mutually independent given $Y$.

The effect size $\delta$ is in units of the within-class standard deviation.
$\delta = 0.3$, 0.5 and 1.0 correspond to single-feature ROC AUCs of 0.58, 0.64
and 0.76 [$\Phi(\delta/\sqrt 2)$]. These are typical of individual features in
credit and fraud models.

## 2.2 The shadow rule and its errors

Let $C$ be the candidate features and $S$ a set of shadow features, generated
independently of $Y$ (by shuffling candidate columns or drawing noise). Fit a
model on $C \cup S$ and compute an importance $I_j$ for every column. The
**one-shot shadow rule** with quantile $q$ keeps candidate $j$ if

$$
I_j > \tau_q, \qquad \tau_q = \operatorname{quantile}_q\{I_s : s \in S\},
$$

with $q = 1$ (the maximum) as the classical choice. **Boruta-style selection**
repeats this over $R$ rounds with fresh shadows. It counts each candidate's
wins $W_j$ and tests $W_j \sim \mathrm{Binomial}(R, 1/2)$ in both directions with a
Bonferroni correction. A significantly high count accepts the feature, a
significantly low count rejects it, and anything in between is *tentative*.

We report both error rates:

* **false negative rate (FNR):** fraction of relevant candidates dropped;
* **false positive rate (FPR):** fraction of irrelevant candidates kept.

In the main sweep (Section 4.3) the irrelevant features themselves serve as
the shadows. There FPR is not defined
and we report FNR only. The mitigation study (Section 5) separates
irrelevant candidates from method-generated shadows and reports both rates.

## 2.3 Importances

For gradient-boosted trees we use the learner's native, training-set
importance. In XGBoost the default `importance_type="gain"` is the **average**
gain (equation (1)) over all splits that use the feature. `total_gain` is the
sum, and `weight` is the number of splits. `feature_importances_` normalises
the chosen type to sum to one. For scikit-learn's `GradientBoostingClassifier`
and `RandomForestClassifier` it is the normalised impurity decrease (MDI). For
LightGBM we use `importance_type="gain"`, which is the total gain. **Held-out
permutation importance** is the drop in a validation score when one column of
the validation set is shuffled [@breiman2001random].

## 2.4 Model

Unless stated otherwise the learner is `xgboost.XGBClassifier` 2.1.3 with
300 trees, depth 6, learning rate 0.1, `tree_method="hist"`, and the
library defaults $\lambda = 1$, $\gamma = 0$, `min_child_weight` $= 1$. There is no
early stopping. These are "reasonable defaults" of the kind used inside
selection loops, not tuned settings. The intercept is either XGBoost's
automatic estimate (*auto*) or the prevalence (*prior*,
`base_score` $= \hat p$). Section 3.1 explains why this matters.
