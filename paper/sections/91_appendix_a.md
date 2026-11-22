# Appendix A. Theory–experiment ledger

Every result in Section 3 was turned into a numerical prediction before it was tested.
The table lists each prediction, the test, the measured value and a verdict.
`experiments/ledger.py` recomputes the measured column from `results/`.
`docs/LEDGER.md` in the repository also records how each disagreement
changed the theory.

| id | prediction (source) | test | measured | verdict |
|:--|:--|:--|:--|:--|
| P1 | XGBoost's reported root gain equals Pearson $X^2$ (Prop 1) | 5 exact fits, $\lambda=0$ | max relative error $1.5\times10^{-7}$ | confirmed |
| P2 | $\mathbb E[\text{Gain}] = n/(n-1)$ at a fixed split, any $n_{\min}$ (Prop 2) | 20,000-draw Monte Carlo, 3 settings | 1.014 vs 1.017; 0.985 vs 1.005; 1.007 vs 1.001 | confirmed |
| P3 | Null best split depends on $n_{\min}$, not $n$ (Thm 1) | simulation of formula (3), $n = 10^3$–$10^5$ | median varies ≤ 4.5% across $n$; grows 1.8 → 5.3 over $n_{\min}$ 5 → 1000 | confirmed |
| P4 | $P$(root uses informative feature) equals the Thm 1 limit | E1, 85 cells × 100 XGBoost fits | mean abs. error 0.025 (exact), 0.026 (hist); $\chi^2$ = 86 and 85 on 85 df; 0.042 on the 28 mid-range cells | confirmed |
| P5 | Root detection organised by $s = n_{\min}\delta^2$, not $p$ (Prop 4, Thm 2) | E1, isotonic R² | $s$: 0.99, $p$: 0.06 | confirmed |
| P6 | Automatic intercept: first tree often does not split; worse as $p$ falls (Prop 5) | E1, $n_{\min}=50$, δ=0.5 | no split at $p=0.001$: predicted 0.56, measured 0.56 (exact); per-run agreement 300/300; 0.99 (hist) | confirmed |
| P7 | Full ensemble: automatic intercept hurts at small $p$ only (Sec 3.5) | E2 | FNR gap +0.10 ($p \le 0.01$), +0.01 ($p \ge 0.05$); worst 0.61 vs 0.04; one cell reversed (0.20 vs 0.36) | confirmed, not uniform |
| P8 | Full-ensemble FNR organised by $s$ (Conj 2) | E2, isotonic R² | prevalence intercept: $s$ 0.94 (LOO 0.90, bootstrap 0.91–0.98; non-saturated 0.89, LOO 0.81), $\delta$ 0.43, $n_{\min}$ 0.36, $p$ ≤ 0.15 (LOO 0.03); automatic: $s$ 0.75 | supported (prevalence); partly (automatic) |
| P9 | Total gain = Newton loss decrease / $\eta(1-\eta/2)$ (Prop 9) | identity tested exactly in the unit tests; E2 pure-noise control | total gain / $nH_2$ = 8.8 ± 1.6 over 42 cells, i.e. about 85% of the initial loss memorised | identity; fraction empirical |
| P10 | Later rounds raise FNR after the signal is learned (first version of Sec 3.8) | E3 (rerun) | prevalence: 0.46 (k=1), 0.40 (k=3), 0.34 (k=30), 0.40 (k=300); paired 30→300 +0.059 ± 0.021; automatic, $p=0.001$: peak 0.84 at k=30 | partly; theory revised |
| P11 | Held-out permutation importance beats training gain (first version) | E4 | equal with tentative kept (FNR 0.24 vs 0.28); worse when strict (0.83 vs 0.66) | not confirmed; claim dropped |
| P12 | Count-based child constraints leave residual $p$-dependence (Prop 3 scope) | E5 | gap 0.13–0.16 for LightGBM (63 or 31 leaves, total or mean gain), 0.04 for RF and sklearn GBM; only LightGBM keeps a gap at $n_{\min} \ge 250$ | partly; open |
| P13 | Max-shadow rule has level $1/(K+1)$ for any learner and imbalance (Prop 6) | E4 one-shot methods, E6 | FPR 0.083 ± 0.004 (5 XGBoost methods), 0.078 (rank test), bound 0.091 | confirmed |
| P14 | FNR increases with the number of shadows (Cor 1) | E8, $K$ = 5, 50, 200 | monotone in every cell; e.g. $n_{\min}=100$: 0.11, 0.36, 0.99 | confirmed |
| P15 | Keep-tentative FPR below the bound (10), limit ≤ $P(\pi_j \ge 1/2)$ (Prop 7) | E7, $R$ = 10–60 | FPR 0.39, 0.15, 0.08, 0.07 vs bound 0.91, 0.45, 0.30, 0.27; $\mathbb E\pi_j$ = 0.085; $P(\pi_j \ge 1/2)$ = 0.028 | confirmed |
| P16 | Rank-test FNR from $z \sim N(\sqrt{3s/\pi}, 1)$ (Prop 8) | E6 | shadow rule 0.66/0.38/0.06/0.01 vs predicted 0.62/0.31/0.08/0.00; BH 0.2: 0.66/0.24/0.04/0.00 vs 0.66/0.23/0.03/0.00 | confirmed |
| P17 | Ensemble FNR ≥ root-split failure (Conj 2) | E4, E8 | holds in all 18 cells; e.g. E4 $n_{\min}=25$: 0.66 vs 0.40 | supported |

The Mann–Whitney power formula (12), the fixed-threshold non-centrality (5),
the shadow level (Prop 6), the shadow-count monotonicity (Cor 1) and the gain
identity (13) are also checked numerically in the unit tests
(`tests/test_theory_limits.py`).

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
   P12 held for LightGBM only, and the scope statement in Section 3 was
   narrowed.
5. *Independent review 2 (paper against data).* The reviewer re-derived
   every number from `results/`. Main findings: the memorisation "budget"
   (old P9) is an accounting identity, not a prediction, and the
   $n_{\min} \le 250$ filter was post hoc; the 94% was an in-sample fit (now
   reported with leave-one-out, bootstrap and non-saturated versions); a
   univariate rank test beat every tree method at small $n_{\min}$; the 15% FPR of
   Boruta rounds depends on the number of rounds and shadows; several
   examples were selective. The library defaults contradicted the data
   (P11), and were changed.
6. *Proofs.* At the author's request every main claim was turned into a
   proved statement or labelled a conjecture. New: Theorem 2 (the $s$ scaling),
   Propositions 6–9, Corollaries 1–2, Conjectures 1–2, Appendix C. Each new
   result was converted into a numerical prediction (P13–P17) and tested by
   three new experiments (E6–E8) and a rerun of E3.
7. *Independent review 3 (proofs).* A referee checked Section 3 and
   Appendix C. No central result was wrong. The fixes: Proposition 6 now
   states its assumptions exactly (it does not cover held-out permutation
   importance with exact search, and needs a random column order, which the
   selector now uses); the histogram clause of Theorem 1 became a corollary
   with explicit hypotheses; two citations (Cox's grouping efficiency and
   Anderson's inequality in infinite dimensions) were replaced by short
   proofs; the error of the toll approximation is now bounded (7'); and
   statements with unproved clauses were split into proved parts and
   numerical evaluations. E4 and E7 were run before the column-order fix;
   it only affects exact ties.
