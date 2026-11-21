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
| E4 | Mitigations, both error rates | 5 relevant + 5 irrelevant candidates, method-built shadows; 8 strategies; 20 repeats | 11,040 |
| E5 | Other learners | sklearn GBM, random forest, LightGBM; single-shot max rule | 540 |

## 4.2 E1: the first split

**Gain is chi-square (P1).** With $\lambda = 0$, exact search and the prevalence
intercept, the root gain XGBoost reports matches Pearson's $X^2$ to within
$2\times 10^{-7}$ relative error, which is float32 precision. This confirms
Proposition 1 and the software convention in equation (1).

**The null best split depends on $n_{\min}$, not $n$ (P3).** Under XGBoost's
default regularisation and the prevalence intercept, the median of the best
null split varies by at most 4% across $n = 10^3$ to $10^5$ at every
$n_{\min}$ (Figure 1a, solid lines). It grows only from 1.8 to 5.3 as $n_{\min}$
goes from 5 to 1000. Without regularisation ($\lambda = w = 0$, dotted) it is an
order of magnitude larger and heavy-tailed, as Section 3.3 predicts. Under
the automatic intercept (dashed) the toll of Proposition 5 makes the best
split *negative* for small $\hat p$: no split is worth making.

**Theorem 1 predicts XGBoost's root decision quantitatively (P4, P5).**
Figure 1b shows how often a depth-1 tree picks the single informative
feature over nine null ones, across 85 combinations of $n_{\min}$, $p$ and
$\delta$, with 100 runs each. Columns are shuffled so that ties are broken at
random. The black curves are the predictions of the $n \to \infty$ limit (4),
computed by Monte Carlo without fitting anything. With the prevalence
intercept, predicted and measured probabilities differ by 0.025 on average,
for exact search (0.026 for `hist`, using the binned limit). This holds for
$p = 0.1$ as well as $p = 0.001$, at $n$ from 250 to 200,000. A monotone fit on
$s = n_{\min}\delta^2$ explains 99% of the variance in the measured
probability, and $p$ explains 6%. The informative feature wins half the time
at $s \approx 5$ and almost always above $s \approx 25$.

**The intercept toll is real, and it is large (P6).** Under the automatic
intercept at $p = 0.001$ (crosses in Figure 1b), the informative feature
rarely wins, even at large $s$. Figure 1c shows why: with `hist` search the
first tree does not split at all in 94% of runs. With exact search, at
$n_{\min} = 50$ and $\delta = 0.5$, equation (7) predicts no split in 57% of runs,
and XGBoost made no split in 56%. When it did split, a null feature won 68%
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

![**Figure 2.** False negative rate of the max-shadow rule in 300-tree XGBoost, against $n_{\min}$. Top: automatic intercept. Bottom: prevalence intercept. Circles: credit regime A, squares: fraud regime B. Error bars: one standard error over repeats.](figures/fig2_fnr_vs_nmin.png)

![**Figure 3.** The same FNRs (prevalence intercept) against (a) the imbalance ratio $p$ and (b) $s = n_{\min}\delta^2$.](figures/fig3_collapse.png)

**$s = n_{\min}\delta^2$ organises the full ensemble too (P8).** With the
prevalence intercept, a monotone (isotonic) fit of FNR on $s$ explains 94% of
the variance across the 63 signal scenarios (Figure 3b). The same fit on
$n_{\min}$ alone explains 36%, because $\delta$ varies, and on $p$ it explains
0% (Figure 3a). The first-round summary of Section 3.4 thus carries over to
300-round ensembles. As a rule of thumb from Table 1 and Figure 3b, FNR is
above 50% for $s \lesssim 5$, around 10–30% for $s \approx 10$–$25$, and near zero
for $s \gtrsim 60$.

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
intercept. At $p = 0.001$, $\delta = 0.5$ it falls from 0.51 to 0.04.

**A smaller residual $p$-dependence remains at weak signal.** With the
prevalence intercept and $\delta = 0.3$, cells with the same $n_{\min}$ still
differ: at $n_{\min} = 200$, FNR is 0.67 at $p = 0.001$ and 0.28 at $p = 0.2$. At
$n_{\min} = 250$, it is 0.49 at $p = 0.005$ and 0.25 at $p = 0.05$. Section 3.6
predicts a residual $\ln(1/\hat p)$ effect through the memorisation budget. The
pure-noise control ($\delta = 0$) confirms that budget (P9). The total gain
recorded on ten null features is $9.6 \pm 0.4$ times $n H_2(\hat p)$ across all
30 control cells with $n_{\min} \le 250$, a coefficient of variation of 5%. Per
minority sample, that is 25 at $p = 0.2$ and 74 at $p = 0.001$. Dividing $s$ by
$1 + \ln(1/\hat p)$ or by $\ln(en)$ does not improve the monotone fit (R² 0.92 and
0.95, against 0.94 for $s$ alone). So the budget effect is real, but it only
matters at weak signal, and $s$ remains the best single summary.

**More shadows lower the bar for false negatives.** With 20 null features
instead of 5 (δ = 0.5, prevalence intercept), FNR rises wherever it was not
already near 0 or 1. At $n_{\min} = 25$ it goes from 0.44 to 0.72, and at
$n_{\min} = 50$ from 0.22–0.30 to 0.42–0.52. The maximum of more draws is
larger. A max-shadow rule's threshold depends on how many shadows are used,
which is one more reason to prefer a repeated test (Section 5).

## 4.4 E3: when in boosting does it fail?

Section 3.6 predicted that memorisation in later rounds would push FNR up
after the signal was learned (P10). The data only partly agree, and the
disagreement changed our account.

* **With the prevalence intercept, most of the failure is there from the
  start.** Averaged over cells, FNR is 0.43 using only the first three trees,
  0.34 at 30 and 0.40 at 300 (Figure 5). Importances computed from the first
  tree alone already show most of the effect at small $n_{\min}$. Late rounds
  add a moderate increase at weak signal, for example 0.08 to 0.27 between
  rounds 10 and 300 at $n_{\min} = 500$, $p = 0.01$, $\delta = 0.3$. So the
  first-round detectability of Section 3.4 is the main mechanism, not
  memorisation.
* **With the automatic intercept at small $p$, failure peaks early.** At
  $p = 0.001$ the first tree has no splits at all, then FNR rises to 0.66 by
  round 10 and 0.84 by round 30. It eases back to 0.71 by round 300. This is
  the intercept-correction window of Proposition 5: while the toll is high,
  the only splits that pass isolate a few positives, and null features get
  as many of them as informative ones.

![**Figure 5.** FNR of the max-shadow rule using importances from the first $k$ boosting rounds only. Solid: prevalence intercept, dashed: automatic intercept.](figures/fig5_rounds.png)

We therefore present memorisation (Section 3.6) as a second-order effect
under the prevalence intercept, visible at weak signal and small $p$, and as
the main early-round effect under the automatic intercept.

![**Figure 4.** Mechanism checks. (a) Mean raw total gain of informative features over the best null feature. (b) Pure-noise control: best null total gain per minority sample. It is larger at small $p$, as the memorisation budget $n H_2(\hat p)$ predicts.](figures/fig4_mechanism.png)

## 4.5 E5: other learners

The single-shot max rule fails at small $n_{\min}$ with every learner we tried
(Figure 7). At $n_{\min} = 25$, FNR is 0.52–0.56 for LightGBM, random forest
(Boruta's learner) and scikit-learn GBM alike, and it falls to near zero by
$n_{\min} = 250$ for all but one setting.

The exception tests the scope of Proposition 3 (P12). LightGBM's defaults
constrain children by sample count (20) rather than Hessian. With a
$p$-free count constraint we expected a residual dependence on $p$ at fixed
$n_{\min}$. We found it. At $n_{\min} = 250$ and 500, LightGBM's FNR is 0.23–0.28 at
$p = 0.01$ and 0.01 at $p = 0.1$. Random forest and scikit-learn GBM also have
count-based constraints (`min_samples_leaf=1`), but they show no consistent
$p$-dependence (for example 0.16 vs 0.09 and 0.09 vs 0.13 at $n_{\min} = 100$). Our
theory does not explain this difference. Plausible factors are depth limits
(scikit-learn GBM uses depth 3), bootstrap and feature subsampling in random
forests, and LightGBM's leaf-wise growth, which spends many splits on small
leaves. We leave it open.

![**Figure 7.** Single-shot max rule with other learners, δ = 0.5. Solid: p = 0.1, dashed: p = 0.01.](figures/fig7_backends.png)
