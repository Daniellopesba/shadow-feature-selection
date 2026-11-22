# 5. Mitigations: both error rates

A mitigation that lowers the threshold will lose fewer relevant features, and
it will also keep more irrelevant ones. To see the trade-off, E4 puts 5
relevant ($\delta = 0.5$) and 5 irrelevant candidates in each dataset. Every
method builds its own shadows (shuffled copies of the 10 candidates), and we
report FNR on the relevant candidates and FPR on the irrelevant ones. Each
cell is 15 repeats at each of $p \in \{0.01, 0.1\}$, for $n_{\min}$ from 10 to 500.
Table 2 averages over the two values of $p$. They mostly agree; the largest
difference is at $n_{\min} = 50$, where the max rule's FNR is 0.22 at $p = 0.01$
and 0.44 at $p = 0.1$. With 75 relevant features per cell, a single entry has
a standard error of up to 0.06. E6 applies univariate rank tests to the
same datasets, with the same seeds.

**Table 2.** FNR / FPR by method and minority count ($\delta = 0.5$, so
$s = n_{\min}/4$; 300-tree XGBoost unless stated; 10 shadows).

| method | $n_{\min}$=10 | 25 | 50 | 100 | 250 | 500 |
|:--|:--:|:--:|:--:|:--:|:--:|:--:|
| max rule, automatic intercept | .81 / .06 | .72 / .08 | .40 / .07 | .21 / .09 | .00 / .09 | .00 / .11 |
| max rule, prevalence intercept | .83 / .04 | .66 / .07 | .33 / .07 | .12 / .09 | .01 / .12 | .00 / .12 |
| P90 rule | .73 / .10 | .50 / .16 | .15 / .16 | .06 / .16 | .00 / .19 | .00 / .16 |
| class weights | .82 / .10 | .74 / .05 | .42 / .13 | .23 / .09 | .01 / .08 | .00 / .10 |
| 1:1 downsampling | .85 / .07 | .86 / .04 | .70 / .07 | .38 / .11 | .06 / .08 | .00 / .09 |
| early stopping | .81 / .07 | .69 / .10 | .43 / .05 | .19 / .11 | .01 / .09 | .00 / .11 |
| Boruta rounds, gain, accepted only | .96 / .00 | .90 / .01 | .53 / .01 | .24 / .01 | .00 / .00 | .00 / .00 |
| Boruta rounds, gain, keep tentative | .63 / .14 | .38 / .17 | .07 / .15 | .03 / .18 | .00 / .15 | .00 / .15 |
| Boruta rounds, permutation, accepted only | .99 / .00 | .95 / .00 | .83 / .00 | .56 / .00 | .04 / .00 | .00 / .00 |
| Boruta rounds, permutation, keep tentative | .55 / .18 | .27 / .19 | .10 / .13 | .02 / .17 | .00 / .15 | .00 / .17 |
| *Mann–Whitney, shadow rule (level 1/11)* | .66 / .08 | .38 / .09 | .06 / .07 | .01 / .08 | .00 / .07 | .00 / .08 |
| *Mann–Whitney, BH q = 0.1* | .78 / .02 | .45 / .04 | .05 / .05 | .01 / .04 | .00 / .04 | .00 / .04 |
| *Mann–Whitney, BH q = 0.2* | .66 / .06 | .24 / .15 | .04 / .09 | .00 / .16 | .00 / .10 | .00 / .11 |

![**Figure 6.** Mitigations: (a) relevant features lost and (b) irrelevant features kept, by method. Dashed: Boruta-style selection keeping accepted features only.](figures/fig6_mitigations.png)

What the table shows:

1. **Below $s \approx 6$ no tree-shadow method we tested works.** At $n_{\min} \le 25$
   ($s \le 6.25$), every XGBoost-based method either loses most relevant
   features or keeps a substantial share of irrelevant ones. The rank test
   does better but not well: at $n_{\min} = 25$, Mann–Whitney with
   Benjamini–Hochberg at $q = 0.2$ [@benjamini1995controlling] loses 24% at an
   FPR of 0.15. At $n_{\min} = 10$ nothing works. This is the regime where
   Proposition 8 gives power below 0.7 even for the best test.
2. **At matched false positive level, the rank test beats the tree.** The
   max-shadow rule with $K = 10$ has level $1/11 = 0.091$ for every method
   (Proposition 6; the measured FPRs are 0.04–0.12, pooled 0.08). The rank
   test with the same shadow rule has the same level and loses far fewer
   relevant features: 0.38 against 0.66 at $n_{\min} = 25$, and 0.06 against
   0.33 at $n_{\min} = 50$. Corollary 2 predicts this: dichotomising costs a
   factor 2/3 in non-centrality and the threshold search costs more. The
   rank test's FNR also matches Proposition 8 quantitatively (predicted
   0.62, 0.31, 0.08 and 0.00 at $n_{\min} = 10$, 25, 50 and 100; Appendix A, P16).
   The caveat of Section 3.7 applies: in this simulation every relevant
   feature is a location shift. A rank test cannot find a feature that acts
   only through an interaction.

![**Figure 8.** (a) Power at level $1/(K+1) = 0.1$ against $s$, computed from the limits: Mann–Whitney (Proposition 8), a fixed median split (Proposition 4), and the max-shadow rule on the best root split ($K = 9$, $n_{\min} = 50$, Theorem 1). (b) Measured FNR at δ = 0.5 for rank tests (E6) and XGBoost shadow methods (E4) on the same datasets; the legend gives each method's FPR.](figures/fig9_rank_vs_split.png)
3. **Class weights, downsampling and early stopping do not help.** Class
   weights and early stopping leave FNR where the max rule had it.
   Downsampling to 1:1 is the worst option. It discards majority samples
   without adding minority ones, and $n_{\min}$, the quantity that matters, is
   unchanged.
4. **The intercept matters at small $p$.** Setting `base_score` to the
   prevalence lowers FNR at $n_{\min} = 25$–100 (for example 0.21 to 0.12 at
   $n_{\min} = 100$) without raising FPR here. The gain is concentrated at
   $p = 0.01$: 0.28 to 0.15 at $n_{\min} = 100$. It is not free everywhere: in E2,
   at $n_{\min} = 10$ and $\delta = 1$, FNR is 0.20 with the automatic intercept and
   0.36 with the prevalence (Section 4.3).
5. **Lower thresholds trade one error for the other.** The P90 rule halves FNR
   at $n_{\min} = 50$ (0.33 to 0.15) and roughly doubles FPR (0.07 to 0.16).
6. **Repeated rounds help, if tentative features are kept, and the FPR
   depends on the number of rounds.** Boruta-style selection with 20 rounds
   that keeps accepted *and* tentative features reaches FNR 0.07–0.10 at
   $n_{\min} = 50$ with FPR about 0.15. That FPR is not a property of the method.
   By Proposition 7 it depends on $R$ and $K$: with 20 rounds and 10 shadows
   a data-independent hit rate would give 0.10, and the excess shows how
   much the hit rate varies between datasets. Section 5.1 measures how FPR
   and FNR change with $R$. Keeping only accepted features gives essentially
   zero FPR but loses most relevant features below $n_{\min} = 100$: with 20
   rounds, too few wins are significant after the Bonferroni correction.
7. **Held-out permutation importance is not better here.** Within Boruta
   rounds, permutation importance on a 30% held-out split performed about as
   well as training-set gain when tentative features were kept. It was worse
   in the strict version, because it holds out 30% of the already scarce
   positives. Its advantages are elsewhere: it is model-agnostic and immune
   to the cardinality bias of impurity importance [@strobl2007bias]. Neither
   matters in this Gaussian simulation. We expected (P11) that it would avoid
   the memorisation part of the failure and win outright. It did not,
   consistent with Section 4.4's finding that memorisation is a secondary
   effect under the prevalence intercept.

## 5.1 Number of rounds (E7)

E7 repeats the Boruta-style selector of Table 2 (XGBoost gain, prevalence
intercept, 10 shadows) for 60 rounds on each dataset and stores every round,
so the decisions for any number of rounds $R \le 60$ can be read off the same
fits. Keeping accepted and tentative features, averaged over $n_{\min}$ from 10
to 250:

| rounds $R$ | 10 | 20 | 40 | 60 |
|:--|--:|--:|--:|--:|
| FPR (irrelevant kept) | 0.39 | 0.15 | 0.08 | 0.07 |
| Markov bound (10) | 0.91 | 0.45 | 0.30 | 0.27 |
| FNR (relevant dropped) | 0.13 | 0.21 | 0.26 | 0.28 |

Every measured FPR is below the bound of Proposition 7, and FPR falls with
$R$ as it predicts. The per-round hit rate of irrelevant candidates averages
0.085, consistent with $\mathbb E\,\pi_j \le 1/11$ (Proposition 6), but it varies a lot
between datasets: its standard deviation is 0.14, and 2.8% of irrelevant
candidates have $\pi_j \ge 1/2$. By Proposition 7 that 2.8% is roughly where FPR
settles as $R$ grows; at 60 rounds it is 0.07. The variation is also why 20
rounds give 0.15 instead of the 0.10 a constant hit rate would give.

More rounds buy a lower FPR with a higher FNR. At roughly matched FPR the
methods rank as the theory says. With $R = 60$ (FPR 0.07) the selector loses
0.44 of relevant features at $n_{\min} = 25$ and 0.15 at $n_{\min} = 50$; the
one-shot max rule (FPR 0.08) loses 0.66 and 0.33; the rank test with the
shadow rule (FPR 0.08) loses 0.38 and 0.06.
