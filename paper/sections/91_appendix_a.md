# Appendix A. Theory–experiment ledger

Every claim in Section 3 was written as a prediction before it was tested.
The table lists each prediction, the test, the measured value and a verdict.
`experiments/ledger.py` recomputes the measured column from `results/`.
`docs/LEDGER.md` in the repository also records how each disagreement
changed the theory.

| id | prediction (source) | test | measured | verdict |
|:--|:--|:--|:--|:--|
| P1 | XGBoost's reported root gain equals Pearson $X^2$ (Prop 1) | 5 exact fits, $\lambda=0$ | max relative error $1.5\times10^{-7}$ | confirmed |
| P2 | $\mathbb E[\text{Gain}] = n/(n-1)$ at a fixed split, any $n_{\min}$ (Prop 2) | 20,000-draw Monte Carlo, 3 settings | 1.014 vs 1.017; 0.985 vs 1.005; 1.007 vs 1.001 | confirmed |
| P3 | Null best split depends on $n_{\min}$, not $n$ (Thm 1) | E1 Monte Carlo, $n = 10^3$–$10^5$ | median varies ≤ 4% across $n$; grows 1.8 → 5.3 over $n_{\min}$ 5 → 1000 | confirmed |
| P4 | $P$(root uses informative feature) equals the Thm 1 limit | E1, 85 cells × 100 runs | mean abs. error 0.025 (exact), 0.026 (hist) | confirmed |
| P5 | Root detection organised by $s = n_{\min}\delta^2$, not $p$ (Prop 4) | E1, isotonic R² | $s$: 0.99, $p$: 0.06 | confirmed |
| P6 | Automatic intercept: first tree often does not split; worse as $p$ falls (Prop 5) | E1, $n_{\min}=50$, δ=0.5 | no split at $p=0.001$: predicted 0.57, measured 0.56 (exact), 0.99 (hist); 0 at $p \ge 0.01$ | confirmed |
| P7 | Full ensemble: automatic intercept hurts at small $p$ only (Sec 3.5) | E2 | FNR gap +0.10 ($p \le 0.01$), +0.01 ($p \ge 0.05$); worst 0.61 vs 0.04 | confirmed |
| P8 | Full-ensemble FNR organised by $s$ (Sec 3.4) | E2, isotonic R² | prevalence intercept: $s$ 0.94, $n_{\min}$ 0.36, $p$ 0.00; automatic: $s$ 0.75 | confirmed (prevalence); partly (automatic) |
| P9 | Memorisation budget ∝ $nH_2(\hat p) \approx n_{\min}(1+\ln 1/\hat p)$ (Sec 3.6) | E2 pure-noise control | total gain / $nH_2$ = 9.6 ± 0.4 (CV 5%) over 30 cells | confirmed |
| P10 | Later rounds raise FNR after the signal is learned (Sec 3.6, first version) | E3 | prevalence: 0.43 (k ≤ 3), 0.34 (k=30), 0.40 (k=300); automatic, $p=0.001$: peak 0.84 at k=30 | partly; theory revised |
| P11 | Held-out permutation importance beats training gain (Sec 3.6, first version) | E4 | equal with tentative kept (FNR 0.24 vs 0.28); worse when strict (0.83 vs 0.66) | not confirmed; claim dropped |
| P12 | Count-based child constraints leave residual $p$-dependence (Prop 3 scope) | E5 | LightGBM +0.13; random forest +0.04; sklearn GBM +0.04 | LightGBM only; open |

**Iteration history.**

1. *Draft paper.* It claimed that $p$ inflates the noise floor. The
   repository's sweep contradicted this. The failure tracked $n_{\min}$, and
   the draft's own corollary predicted a falling mean noise floor.
2. *Theory v1.* We introduced the chi-square identity, the $n_{\min}$ limit,
   and a claim that the automatic intercept makes the null maximum grow with
   $n$.
3. *Independent review 1.* The reviewer found that the automatic intercept
   does something stronger: it imposes a toll that often prevents any split.
   The reviewer also caught a factor of 2 in the gain convention, the
   exact-versus-hist distinction, gaps in the Theorem 1 proof (ranks,
   Skorokhod, ties, atoms), and the limited range of the $2/\pi$
   approximation. All were fixed. Proposition 5 is new, and detectability is
   now computed from the limit process.
4. *Experiments round 1.* P1–P9 were confirmed, P4 and P6 quantitatively.
   P10 and P11 were not. Memorisation turned out to be secondary under the
   prevalence intercept and an early-round effect under the automatic one.
   Sections 3.6, 4.4 and 5 were rewritten accordingly. P12 held for LightGBM
   only, and the scope statement in Section 3 was narrowed.
5. *Independent review 2* of the full paper: see `docs/LEDGER.md`.
