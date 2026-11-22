# Appendix A. Theory against simulation

Each row states a quantitative consequence of Section 3, the experiment that
measures it, the measured value, and how well the two agree.
`experiments/paper_numbers.py` recomputes the measured column from `results/`.

| id | statement (source) | test | measured | agreement |
|:--|:--|:--|:--|:--|
| P1 | XGBoost's reported root gain equals Pearson $X^2$ (Prop 1) | 5 exact fits, $\lambda=0$ | max relative error $1.5\times10^{-7}$ | yes |
| P2 | $\mathbb E[\text{Gain}] = n/(n-1)$ at a fixed split, any $n_{\min}$ (Prop 2) | 20,000-draw Monte Carlo, 3 settings | 1.014 vs 1.017; 0.985 vs 1.005; 1.007 vs 1.001 | yes |
| P3 | Null best split depends on $n_{\min}$, not $n$ (Thm 1) | simulation of formula (3), $n = 10^3$–$10^5$ | median varies ≤ 4.5% across $n$; grows 1.8 → 5.3 over $n_{\min}$ 5 → 1000 | yes |
| P4 | $P$(root uses informative feature) equals the Thm 1 limit | E1, 85 cells × 100 XGBoost fits | mean abs. error 0.025 (exact), 0.026 (hist); $\chi^2$ = 86 and 85 on 85 df; 0.042 on the 28 mid-range cells | yes |
| P5 | Root detection organised by $s = n_{\min}\delta^2$, not $p$ (Prop 4, Thm 2) | E1, isotonic R² | $s$: 0.99, $p$: 0.06 | yes |
| P6 | Automatic intercept: first tree often does not split; worse as $p$ falls (Prop 5) | E1, $n_{\min}=50$, δ=0.5 | no split at $p=0.001$: predicted 0.56, measured 0.56 (exact); per-run agreement 300/300; 0.99 (hist) | yes |
| P7 | Full ensemble: automatic intercept hurts at small $p$ only (Sec 3.5) | E2 | FNR gap +0.10 ($p \le 0.01$), +0.01 ($p \ge 0.05$); worst 0.61 vs 0.04; one cell reversed (0.20 vs 0.36) | yes, not uniform |
| P8 | Full-ensemble FNR organised by $s$ (Conj 2) | E2, isotonic R² | prevalence intercept: $s$ 0.94 (LOO 0.90, bootstrap 0.91–0.98; non-saturated 0.89, LOO 0.81), $\delta$ 0.43, $n_{\min}$ 0.36, $p$ ≤ 0.15 (LOO 0.03); automatic: $s$ 0.75 | strong (prevalence); partial (automatic) |
| P9 | Total gain = Newton loss decrease / $\eta(1-\eta/2)$ (Prop 9) | identity tested exactly in the unit tests; E2 pure-noise control | total gain / $nH_2$ = 8.8 ± 1.6 over 42 cells, i.e. about 85% of the initial loss memorised | identity; fraction empirical |
| P10 | Late-round memorisation and FNR (Sec 3.8) | E3 | prevalence: 0.46 (k=1), 0.40 (k=3), 0.34 (k=30), 0.40 (k=300); paired 30→300 +0.059 ± 0.021; automatic, $p=0.001$: peak 0.84 at k=30 | second-order (prevalence); early-round (automatic) |
| P11 | Held-out permutation importance against training gain (Sec 5) | E4 | equal with tentative kept (FNR 0.24 vs 0.28); worse when strict (0.83 vs 0.66) | no advantage |
| P12 | Count-based child constraints leave residual $p$-dependence (Prop 3 scope) | E5 | gap 0.13–0.16 for LightGBM (63 or 31 leaves, total or mean gain), 0.04 for RF and sklearn GBM; only LightGBM keeps a gap at $n_{\min} \ge 250$ | partial; open |
| P13 | Max-shadow rule has level $1/(K+1)$ for any learner and imbalance (Prop 6) | E4 one-shot methods, E6 | FPR 0.083 ± 0.004 (5 XGBoost methods), 0.078 (rank test), bound 0.091 | yes |
| P14 | FNR increases with the number of shadows (Cor 1) | E8, $K$ = 5, 50, 200 | monotone in every cell; e.g. $n_{\min}=100$: 0.11, 0.36, 0.99 | yes |
| P15 | Keep-tentative FPR below the bound (10), limit ≤ $P(\pi_j \ge 1/2)$ (Prop 7) | E7, $R$ = 10–60 | FPR 0.39, 0.15, 0.08, 0.07 vs bound 0.91, 0.45, 0.30, 0.27; $\mathbb E\pi_j$ = 0.085; $P(\pi_j \ge 1/2)$ = 0.028 | yes |
| P16 | Rank-test FNR from $z \sim N(\sqrt{3s/\pi}, 1)$ (Prop 8) | E6 | shadow rule 0.66/0.38/0.06/0.01 vs predicted 0.62/0.31/0.08/0.00; BH 0.2: 0.66/0.24/0.04/0.00 vs 0.66/0.23/0.03/0.00 | yes |
| P17 | Ensemble FNR ≥ root-split failure (Conj 2) | E4, E8 | holds in all 18 cells; e.g. E4 $n_{\min}=25$: 0.66 vs 0.40 | yes, no proof |

The Mann–Whitney power formula (12), the fixed-threshold non-centrality (5),
the shadow level (Prop 6), the shadow-count monotonicity (Cor 1) and the gain
identity (13) are also checked numerically in the unit tests
(`tests/test_theory_limits.py`).
