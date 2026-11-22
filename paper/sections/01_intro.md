# 1. Introduction

## 1.1 The shadow-feature heuristic

A common feature-selection recipe in applied machine learning is to add
columns that are known to be useless and see which real features beat them.
The useless columns, called *shadow features*, are either random noise or
copies of the real features with their rows shuffled, which breaks any link to
the target. After fitting a model (usually tree-based), any real feature whose
importance falls below the largest shadow importance is dropped. Boruta
[@kursa2010boruta] turns this into a procedure: it repeats the comparison over
many rounds with fresh shuffles and applies a binomial test to each feature's
win count. BorutaSHAP [@keany2020borutashap] uses SHAP values instead of
impurity importance. Many teams use the one-shot version directly, on the
gain importance of a gradient-boosted model.

The appeal is that the shadows calibrate the threshold automatically: whatever
importance pure noise can reach is, by construction, the bar to clear.

## 1.2 What goes wrong with a small minority class

Credit default, fraud, medical diagnosis and churn all have rare positive
classes. In this setting the recipe fails in a specific, predictable way.
Features that really do predict the target are dropped at high rates. The
failure is controlled by the **absolute number of minority-class samples**
$n_{\min}$, not by the imbalance ratio $p$. A credit model with 5% defaults
among 1,000 loans ($n_{\min} = 50$) and a fraud model with 0.1% fraud among
50,000 transactions (also $n_{\min} = 50$) are in the same danger.

The mechanism is not that noise features get more important as $p$ shrinks.
Under the null, a split on an uninformative feature has the same expected gain
at any imbalance (Proposition 2), and the shadow rule keeps an irrelevant
feature with the same probability at any imbalance (Proposition 6). What
changes is the informative feature's advantage: it scales with
$s = n_{\min}\delta^2$ (Theorem 2), while the best split pure noise can find grows
only slowly with $n_{\min}$. Two further factors make it worse:

1. In XGBoost 2.0 and 2.1, the default starting prediction is about 0.12 for
   any small $p$. That raises the evidence needed for the first splits
   (Proposition 5) and disables the protection `min_child_weight` normally
   gives against splits isolating a handful of samples (Proposition 3).
2. Later boosting rounds memorise the few positives, and those splits land on
   informative and noise features alike. This effect is second-order under
   a correct intercept (Section 4.4).

## 1.3 Contributions

1. **Theory (Section 3), with proofs.** XGBoost's first-round split gain is
   exactly Pearson's chi-square statistic for the split-by-class table
   (Proposition 1). In the small-prevalence limit, the root split depends on
   the data only through $n_{\min}$ and the effect size (Theorem 1), and, as
   $n_{\min}$ grows, only through $s = n_{\min}\delta^2$ (Theorem 2). For any
   rank-based learner the max-shadow rule has level $1/(K+1)$ (Proposition 6),
   more shadows raise the bar for real features (Corollary 1), and repeated
   Boruta rounds have a false positive rate we bound (Proposition 7). A
   Mann–Whitney test is driven by the same counting process and has
   non-centrality $(3/\pi)s$, against $(2/\pi)s$ for the best single split
   (Proposition 8, Corollary 2). A misestimated intercept raises the
   evidence a split needs by a factor of order $1/\hat p$ (Proposition 5). What
   we cannot prove, about full ensembles, is stated as conjectures with the
   evidence for them.
2. **Experiments that test each statement (Section 4).** About 57,000 fitted
   models, two imbalance regimes chosen to overlap in $n_{\min}$, four learners.
   Every proved statement yields a numerical prediction; Appendix A lists
   each prediction next to its measurement.
3. **Mitigations with both error rates (Section 5).** We compare eight
   tree-based strategies and a rank-test baseline on false negative rate *and*
   false positive rate.
4. **A tested, scikit-learn compatible tool (Section 6).** The `shadowselect`
   package provides `ShadowFeatureSelector` (Boruta-style rounds),
   `UnivariateScreen` (rank tests with false-discovery control) and
   `assess_imbalance`, a pre-flight check computed from the theory. Every
   number and figure in this paper is produced by scripts in the same
   repository with fixed seeds.

## 1.4 Related work

**Shadow, probe and knockoff methods.** The idea of judging features against
artificial ones that are irrelevant by construction is older than Boruta:
random probe features [@stoppiglia2003ranking], pseudovariables that
calibrate a selection threshold [@wu2007controlling], and artificial
contrasts for tree ensembles [@tuv2009feature]. Boruta [@kursa2010boruta] introduced
permuted shadow copies with a random-forest learner and a repeated binomial
test. BorutaSHAP [@keany2020borutashap] swaps in SHAP importances. Model-X
knockoffs [@candes2018panning] are the principled relative of shadows. They
construct synthetic copies that mimic the joint distribution of the real
features and give finite-sample false discovery rate control. Their power
still depends on the importance statistic, so the loss of power we describe
applies to them too, though we do not test knockoffs here. We are not aware of
work that analyses shadow-based selection as a function of the minority count.

**Bias and noise in tree importances.** Impurity-based importances are biased
toward features with many split points [@strobl2007bias]. Permutation
importance computed on held-out data avoids that bias [@breiman2001random;
@strobl2007bias], and can be turned into p-values against a permutation null
[@altmann2010permutation]. Split-selection bias is also why conditional
inference trees and related methods select splits by p-values instead of raw
gain [@loh1997split; @hothorn2006unbiased; @strobl2007unbiased;
@nembrini2018revival]. Uninformative features receive non-zero training-set gain,
and cross-validated gain has been proposed to correct this
[@adler2022feature; @zhang2023unbiased]. None of these works studies how the minority count
changes the comparison.

**Maximally selected statistics.** The best split of a null feature is a
maximally selected chi-square statistic, studied for cut-point selection in
biostatistics [@miller1982maximally; @lausen1992maximally; @hothorn2003exact]. Our Theorem 1
connects that literature to gradient-boosting importance under imbalance. The
loss from splitting a continuous variable at one threshold is the classical
efficiency of grouping [@cox1957note] and of the median test relative to the
Wilcoxon test [@mood1954asymptotic; @mann1947test].

**Imbalanced learning.** Resampling (e.g. SMOTE, @chawla2002smote), cost-sensitive
learning and ensembles are surveyed by @fernandez2018learning. That literature
targets predictive performance, not the reliability of feature selection.
