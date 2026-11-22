# 4. Experiments: testing the theory

Section 3 makes specific predictions. This section tests each one against
simulation. Where the data disagreed with an earlier version of the theory,
we changed the theory and say so. Appendix A lists every prediction with the
measured value and a verdict.

## 4.1 Protocol and reproducibility

All experiments use the data model of Section 2.1 and the learner of
Section 2.4, unless stated otherwise. Seeds are derived from a hash of each
scenario's parameters and the repeat index, so every run is bit-for-bit
reproducible with the pinned library versions (`requirements-experiments.txt`:
Python 3.13, NumPy 2.5, scikit-learn 1.9, XGBoost 2.1.3, LightGBM 4.7). Each
experiment is one script in `experiments/` that writes a tidy per-repeat
table to `results/`. `make_figures.py` and `ledger.py` produce every figure,
table and number quoted here. The full set runs in about 1.5 hours on four
CPU cores. Each script also has a `--quick` profile that CI runs.

| id | question | design | fits |
|---|---|---|---|
| E1 | Does the first-split theory hold? | depth-1, one-tree XGBoost; 1 informative + 9 null features; $n_{\min}\in\{10,\dots,1000\}$, $p \in \{0.001, 0.01, 0.1\}$, $\delta \in \{0.1,\dots,1\}$; both intercepts; exact and hist; 100 repeats | 34,000 |
| E2 | Full-ensemble failure rate | 300-tree XGBoost; credit regime A ($n \le 5000$, $p \ge 0.05$) and fraud regime B ($n \ge 10^4$, $p \le 0.01$) overlapping in $n_{\min}$; $\delta \in \{0, 0.3, 0.5, 1\}$; both intercepts; 25 repeats (15 for $n \ge 5\cdot 10^4$); plus a 20-noise-feature variant | 4,185 |
| E3 | Where in boosting does it fail? | as E2, importances read after $k = 1,\dots,300$ rounds; $n_{\min} \in \{25, 100, 500\}$ with $p$ over two to three decades | 480 |
| E4 | Mitigations, both error rates | 5 relevant + 5 irrelevant candidates, method-built shadows; 8 strategies; 15 repeats | 11,040 |
| E5 | Other learners | sklearn GBM, random forest, LightGBM (63 and 31 leaves); single-shot max rule; 15 repeats | 720 |
| E6 | Rank-test baseline | Mann–Whitney on the E4 datasets (same seeds), BH and shadow rules | 180 |
| E7 | Number of Boruta rounds | E4 design, XGBoost gain, prevalence intercept, 60 rounds per dataset stored round by round; 10 repeats | 6,000 |
| E8 | Number of shadows | one-shot max rule with 5, 50 or 200 null features; $n_{\min} \in \{50, \dots, 500\}$, $p = 0.1$; 15 repeats | 180 |

## 4.2 E1: the first split

**Gain is chi-square (P1).** With $\lambda = 0$, exact search and the prevalence
intercept, the root gain XGBoost reports matches Pearson's $X^2$ to within
$2\times 10^{-7}$ relative error, which is float32 precision. This confirms
Proposition 1 and the software convention in equation (1).

**The null best split depends on $n_{\min}$, not $n$ (P3).** This check
simulates formula (3) directly at finite $n$, so it tests the limit in
Theorem 1, not XGBoost (the next paragraph tests XGBoost). Under XGBoost's
default regularisation and the prevalence intercept, the median of the best
null split varies by at most 4.5% across $n = 10^3$ to $10^5$ at every
$n_{\min}$ (Figure 1a, solid lines). It grows only from 1.8 to 5.3 as $n_{\min}$
goes from 5 to 1000. Without regularisation ($\lambda = w = 0$, dotted) the
finite-$n$ medians are 1.2 to 2.5 times larger and grow with $n$, as
remark (iii) of Theorem 1 says they must: the unregularised limit is
infinite. Under the automatic intercept (dashed) the toll of Proposition 5
makes the best split *negative* for small $\hat p$: no split is worth making.

**Theorem 1 predicts XGBoost's root decision quantitatively (P4, P5).**
Figure 1b shows how often a depth-1 tree picks the single informative
feature over nine null ones, across 85 combinations of $n_{\min}$, $p$ and
$\delta$, with 100 runs each. Columns are shuffled so that ties are broken at
random. The black curves are the predictions of the $n \to \infty$ limit (4),
computed by Monte Carlo without fitting anything. With the prevalence
intercept, predicted and measured probabilities differ by 0.025 on average
for exact search (0.026 for `hist`, using the binned limit), at $p = 0.1$ as
well as $p = 0.001$ and $n$ from 250 to 100,000. Much of that is binomial noise
from 100 runs per cell: the goodness-of-fit statistic is $\chi^2 = 86$ on 85
degrees of freedom (85 for `hist`), so the data are consistent with the
limit being exact. In the 28 cells where the prediction is between 0.2 and
0.8, the mean error is 0.042. A monotone fit on $s = n_{\min}\delta^2$ explains 99%
of the variance in the measured probability, and $p$ explains 6%. The
informative feature wins half the time at $s \approx 5$ and almost always above
$s \approx 25$.

**The intercept toll is real, and it is large (P6).** Under the automatic
intercept at $p = 0.001$ (crosses in Figure 1b), the informative feature
rarely wins, even at large $s$. Figure 1c shows why: with `hist` search the
first tree does not split at all in 94% of runs. With exact search, at
$n_{\min} = 50$ and $\delta = 0.5$, equation (7) predicts no split in 56% of runs,
and XGBoost made no split in 56%. Evaluating (7) on each run's own data
predicts split or no split correctly in all 300 exact-search runs of this cell (100 at each $p$). When it did split, a null feature won 68%
of the time. Those are the extreme, positive-isolating splits Proposition 5
describes, and the informative feature has no advantage there. At $p = 0.01$
the toll is smaller and splits happen. At $p = 0.1$ the automatic intercept
is close to the prevalence and nothing changes.

![**Figure 1.** First-round theory against XGBoost. (a) Median best split of a null feature: with XGBoost's defaults and the prevalence intercept it depends on $n_{\min}$, not $n$. (b) Probability that the root uses the informative feature, against $s = n_{\min}\delta^2$. Dots: prevalence intercept, crosses: automatic intercept, black: Theorem 1 limit. (c) How often the first tree does not split at all.](figures/fig1_first_split.png)

## 4.3 E2: the full ensemble

Figure 2 shows the false negative rate of the one-shot max-shadow rule against
$n_{\min}$, for each effect size and both intercepts. Table 1 aggregates by
$n_{\min}$.

**Table 1.** Mean FNR of the max-shadow rule (300-tree XGBoost, `gain`
importance, 5 null features), by minority count, effect size and intercept.

| $n_{\min}$ | auto, δ=0.3 | auto, δ=0.5 | auto, δ=1 | prior, δ=0.3 | prior, δ=0.5 | prior, δ=1 |
|:--|--:|--:|--:|--:|--:|--:|
| < 25 | 0.83 | 0.68 | 0.20 | 0.80 | 0.66 | 0.36 |
| 25–74 | 0.71 | 0.39 | 0.01 | 0.70 | 0.32 | 0.02 |
| 75–199 | 0.58 | 0.11 | 0.00 | 0.51 | 0.13 | 0.00 |
| 200–499 | 0.58 | 0.15 | 0.00 | 0.41 | 0.02 | 0.00 |
| 500–1499 | 0.29 | 0.00 | 0.00 | 0.08 | 0.00 | 0.00 |
| ≥ 1500 | 0.07 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

The first and last rows each contain a single $(n, p)$ scenario; the others
average two to five. Standard errors over repeats are 0.02–0.06 per scenario
(Figure 2).

![**Figure 2.** False negative rate of the max-shadow rule in 300-tree XGBoost, against $n_{\min}$. Top: automatic intercept. Bottom: prevalence intercept. Circles: credit regime A, squares: fraud regime B. Error bars: one standard error over repeats.](figures/fig2_fnr_vs_nmin.png)

![**Figure 3.** The same FNRs (prevalence intercept) against (a) the imbalance ratio $p$ and (b) $s = n_{\min}\delta^2$.](figures/fig3_collapse.png)

**$s = n_{\min}\delta^2$ organises the full ensemble too (P8, Conjecture 2).** With
the prevalence intercept, a monotone (isotonic) fit of cell-mean FNR on $s$
explains 94% of the variance across the 63 signal scenarios (Figure 3b).
This is an in-sample fit of a flexible curve, so we also report
out-of-sample versions: leave-one-out 0.90, and a bootstrap 95% interval of
0.91–0.98. Many cells sit at FNR 0 or 1, which flatters any monotone fit; on
the 34 cells with FNR strictly between 0.02 and 0.98 the fit is 0.89
(leave-one-out 0.81). The same fit on $n_{\min}$ alone explains 36%, on $\delta$
alone 43%. For $p$ the answer depends on the direction of the fit: FNR
decreasing in $1/p$ explains nothing, and FNR increasing in $1/p$ explains
0.15 in sample and 0.03 out of sample (Figure 3a). As a rule of thumb from
these cells, FNR is 64–84% for $s < 5$, 22–67% (median 35%) for $s$ between 10
and 25, and at most 4% for $s > 60$ (Section 6, Rule 2).

**The intercept accounts for the fraud-regime gap (P7).** The earlier draft
of this paper, and the first version of these experiments, used XGBoost's
automatic intercept. They found the fraud regime failing more often than the
credit regime at the same $n_{\min}$, and attributed that to "a small residual
penalty" of extreme majority counts. Proposition 5 predicted instead that the
gap comes from the misestimated intercept and should disappear with
`base_score` $= \hat p$. It does, for the most part. Averaged over cells with
$p \le 0.01$, the automatic intercept raises FNR by 0.10. For $p \ge 0.05$ the
difference is 0.01. The largest single effect is at $n = 200{,}000$, $p = 0.005$,
$\delta = 0.3$: FNR 0.61 under the automatic intercept and 0.04 with the prevalence
intercept. The effect is not uniform. At $p = 0.001$, $\delta = 0.5$ it falls
from 0.51 to 0.04 at $n = 200{,}000$, but only from 0.49 to 0.37 and from 0.68 to
0.66 in the other two $p = 0.001$ cells. And in one corner the prevalence
intercept is worse: at $n_{\min} = 10$, $\delta = 1$, FNR is 0.20 under the automatic
intercept and 0.36 with the prevalence.

**A smaller residual $p$-dependence remains at weak signal, and we cannot
fully explain it.** With the prevalence intercept and $\delta = 0.3$, cells with
the same $n_{\min}$ still differ: at $n_{\min} = 200$, FNR is 0.67 at $p = 0.001$ and
0.28 at $p = 0.2$. At $n_{\min} = 250$, it is 0.49 at $p = 0.005$ and 0.25 at
$p = 0.05$. Two partial explanations are proved. First, finite-$n$ effects
favour large $p$: at small $n$ the null best split is smaller (Theorem 1,
remark (ii)), and a two-sample test's information is $n_{\min}(1-\hat p)$ rather
than $n_{\min}$ (Section 3.7). Second, the gain accounting of Proposition 9
ties the total gain to the training loss removed, and the initial loss
$nH_2(\hat p) \approx n_{\min}(1 + \ln(1/\hat p))$ grows as $\hat p$ falls. In the pure-noise
control ($\delta = 0$, P9) the total gain recorded on ten null features is
$8.8 \pm 1.6$ times $nH_2(\hat p)$ across all 42 control cells. By Proposition 9
that means 300 trees memorise about 85% of the initial loss, and more loss is
available to memorise at small $\hat p$. We have not turned this into a
prediction of FNR, and dividing $s$ by $1 + \ln(1/\hat p)$ or by $\ln(en)$ does not
improve the monotone fit (R² 0.92 and 0.95, against 0.94 for $s$ alone). We
therefore record the residual as unexplained (Conjecture 2).

**More shadows raise the bar (P14, Corollary 1).** With 20 null features
instead of 5 (E2, δ = 0.5, prevalence intercept), FNR rises only where it was
neither near 0 nor near 1, that is, for $n_{\min} \le 50$: at $n_{\min} = 25$ from
0.44 to 0.72, at $n_{\min} = 50$ from 0.22–0.30 to 0.42–0.52. E8 varies the
number of shadows directly. At $n_{\min} = 100$ ($s = 25$), FNR is 0.11 with 5
shadows, 0.36 with 50 and 0.99 with 200. FNR increases with $K$ in every
cell, as Corollary 1 requires, and in every cell it is at or above the
root-split prediction ($1 - \mathbb E[F_0(M)^K]$: 0.01, 0.07 and 0.15 at
$n_{\min} = 100$). The gap between ensemble and root grows with $K$: with many
shadows, memorised gain spread over many null columns makes the largest of
them large.

## 4.4 E3: when in boosting does it fail?

An earlier version of Section 3 predicted that memorisation in later rounds
would push FNR up after the signal was learned (P10). The data only partly
agree, and the disagreement changed our account. (This experiment was rerun
after a fix so that a model with no splits counts as missing, not as a
failure.)

* **With the prevalence intercept, most of the failure is there from the
  start.** Averaged over cells, FNR is 0.46 using only the first tree, 0.40
  using three, 0.34 at 30 and 0.40 at 300 (Figure 4). Late rounds add a
  small but significant increase: the paired change from 30 to 300 rounds is
  $+0.059 \pm 0.021$ (standard error over 16 cells). It is largest at weak
  signal, for example 0.08 to 0.27 between rounds 10 and 300 at
  $n_{\min} = 500$, $p = 0.01$, $\delta = 0.3$. So the first-round detectability of
  Section 3.4 is the main mechanism, and memorisation a second-order one.
* **With the automatic intercept at small $p$, failure peaks early.** At
  $p = 0.001$ the first tree has no splits at all. FNR is 0.40 after three
  rounds, 0.66 after 10 and 0.84 after 30, and eases back to 0.71 by round
  300. This is the intercept-correction window of Proposition 5: while the
  split threshold (9) is high, the only splits that pass isolate a few
  positives, and null features get as many of them as informative ones.

![**Figure 4.** FNR of the max-shadow rule using importances from the first $k$ boosting rounds only. Solid: prevalence intercept, dashed: automatic intercept.](figures/fig5_rounds.png)

We therefore present memorisation (Section 3.8) as a second-order effect
under the prevalence intercept, visible at weak signal and small $p$, and as
the main early-round effect under the automatic intercept.

![**Figure 5.** Mechanism checks. (a) Mean raw total gain of informative features over the best null feature. (b) Pure-noise control: best null total gain per minority sample. It is larger at small $p$, as the gain accounting of Proposition 9 implies, since the initial loss $n H_2(\hat p)$ is larger.](figures/fig4_mechanism.png)

## 4.5 E5: other learners

The single-shot max rule fails at small $n_{\min}$ with every learner we tried
(Figure 7). At $n_{\min} = 25$, FNR is 0.48–0.56 for LightGBM, random forest
(Boruta's learner) and scikit-learn GBM alike. This is what Proposition 6
and Corollary 1 lead one to expect: the level of the rule and the shape of
the comparison do not depend on the learner.

P12 asked whether learners that constrain children by sample count rather
than Hessian (Proposition 3) show a residual dependence on $p$ at fixed
$n_{\min}$. Averaged over $n_{\min}$, FNR at $p = 0.01$ exceeds FNR at $p = 0.1$ by
0.13–0.16 for LightGBM and by 0.04 for random forest and scikit-learn GBM.
At $n_{\min} \ge 250$ the gap is 0.24 for LightGBM with 63 leaves, 0.07–0.10 with
its default 31 leaves, and zero for the other two. Using LightGBM's mean
gain per split, comparable to XGBoost's `gain`, instead of its total gain
does not change this. At $n_{\min} = 50$ every learner shows a gap (random forest
0.51 against 0.20). So the count constraint is not the whole story. Depth
limits, bootstrap and feature subsampling, and leaf-wise growth all differ
between these learners, and we leave the question open.

![**Figure 7.** Single-shot max rule with other learners, δ = 0.5. Solid: p = 0.1, dashed: p = 0.01.](figures/fig7_backends.png)

## 4.6 Tests of Sections 3.6 and 3.7

**The shadow rule's level is $1/(K+1)$ (P13, Proposition 6).** In E4 every
one-shot method compares each irrelevant candidate with the maximum of
$K = 10$ shadows, so Proposition 6 gives an FPR of $1/11 = 0.091$, whatever
$n_{\min}$ and $p$. Pooled over all cells, the five one-shot XGBoost methods
have FPR $0.083 \pm 0.004$ (between 0.075 and 0.089 by method), slightly below
0.091 because of ties at zero importance, which count as losses. The rank
test with the same shadow rule has 0.078.

**The rank test follows Proposition 8 (P16).** Treating each Mann–Whitney
statistic as $N(\sqrt{3s/\pi}, 1)$, with $s = n_{\min}/4$, predicts the rank test's
FNR in E6 without any fitting: 0.62, 0.31, 0.08 and 0.00 for the shadow rule
at $n_{\min} = 10$, 25, 50 and 100, against 0.66, 0.38, 0.06 and 0.01 measured;
and 0.66, 0.23, 0.03 and 0.00 for Benjamini–Hochberg at $q = 0.2$, against 0.66,
0.24, 0.04 and 0.00.

**The full ensemble fails at least as often as the root split (P17,
Conjecture 2).** For the one-shot rule in E4 the root-split prediction of
Corollary 1 ($K = 10$) is 0.67, 0.40, 0.16 and 0.02 at $n_{\min} = 10$, 25, 50
and 100; the 300-tree ensemble measures 0.83, 0.66, 0.33 and 0.12. Together
with E8 (above), the ensemble is at or above the root prediction in every
cell we have. We do not have a proof of this inequality.
