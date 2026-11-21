# 5. Mitigations: both error rates

A mitigation that lowers the threshold will lose fewer relevant features, and
it will also keep more irrelevant ones. To see the trade-off, E4 puts 5
relevant ($\delta = 0.5$) and 5 irrelevant candidates in each dataset. Every
method builds its own shadows (shuffled copies of the 10 candidates), and we
report FNR on the relevant candidates and FPR on the irrelevant ones. Each
cell is 20 repeats at $p \in \{0.01, 0.1\}$ and $n_{\min}$ from 10 to 500.
Table 2 averages over the two values of $p$, which behave alike.

**Table 2.** FNR / FPR by method and minority count ($\delta = 0.5$, 300-tree
XGBoost, 10 shadows).

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

![**Figure 6.** Mitigations: (a) relevant features lost and (b) irrelevant features kept, by method. Dashed: Boruta-style selection keeping accepted features only.](figures/fig6_mitigations.png)

What the table shows:

1. **Nothing rescues $n_{\min} \le 25$.** At $\delta = 0.5$ that is $s \le 6$.
   Every method either loses most relevant features or keeps a substantial
   share of irrelevant ones. A feature with a 0.5 SD effect and 25 positive
   examples is close to undetectable by any importance-threshold method.
   Section 3.4 explains why.
2. **Class weights, downsampling and early stopping do not help.** Class
   weights and early stopping leave FNR where the max rule had it.
   Downsampling to 1:1 is the worst option. It discards majority samples
   without adding minority ones, and $n_{\min}$, the quantity that matters, is
   unchanged.
3. **The intercept matters most at small $p$.** Setting `base_score` to the
   prevalence lowers FNR at $n_{\min} = 25$–100 (e.g. 0.21 to 0.12 at
   $n_{\min} = 100$), at no cost in FPR. The gain is concentrated at $p = 0.01$:
   0.28 to 0.15 at $n_{\min} = 100$.
4. **Lower thresholds trade one error for the other.** The P90 rule halves FNR
   at $n_{\min} = 50$ (0.33 to 0.15) and roughly doubles FPR (0.07 to 0.16).
5. **Repeated rounds give the best trade-off, if tentative features are
   kept.** Boruta-style selection with 20 rounds that keeps accepted *and*
   tentative features reaches FNR 0.07–0.10 at $n_{\min} = 50$ and 0.02–0.03 at
   $n_{\min} = 100$, with FPR about 0.15. Keeping only accepted features gives
   essentially zero FPR but loses most relevant features below
   $n_{\min} = 100$. With 20 rounds, too few wins are significant after the
   Bonferroni correction. The choice between the two is a choice of error
   rate, not of method.
6. **Held-out permutation importance is not better here.** Within Boruta
   rounds, permutation importance on a 30% held-out split performed about as
   well as training-set gain when tentative features were kept. It was worse
   in the strict version, because it holds out 30% of the already scarce
   positives. Its advantages are elsewhere: it is model-agnostic and immune
   to the cardinality bias of impurity importance [@strobl2007bias]. Neither
   matters in this Gaussian simulation. We expected (P11) that it would avoid
   the memorisation part of the failure and win outright. It did not,
   consistent with Section 4.4's finding that memorisation is a secondary
   effect under the prevalence intercept.
