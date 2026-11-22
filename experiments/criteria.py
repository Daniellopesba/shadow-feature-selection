"""Acceptance criteria for every theory-versus-simulation comparison (paper Appendix D).

Each entry fixes, before the comparison is computed, the statistic, the
prediction it is compared with, and the rule that turns the two into a
verdict. ``paper_numbers.py`` evaluates the rule; no verdict is written by hand.

Conventions
-----------
* SE is the Monte Carlo standard error of the measured quantity, clustered by
  data set (repeats are the independent unit; features within a data set are
  not), combined in quadrature with the Monte Carlo error of the prediction
  when the prediction is itself simulated.
* "within tau + 2 SE" means |measured - predicted| <= tau + 2 * SE.
* For bounds (one-sided statements), "below bound + 2 SE" means
  measured <= bound + 2 * SE.
* Cells whose prediction and measurement are both below 0.02 or both above
  0.98 carry no information and are excluded from counts such as "k of K
  cells".
* Multiplicity: besides the verdicts, each row reports a p-value for exact
  agreement; Appendix D gives the Holm-adjusted p-values over all rows and the
  number of disagreements expected by chance.
"""

CRITERIA = {
    "P1": dict(
        statement="prop:chi2",
        test="root gain reported by XGBoost (lambda=0, exact, prevalence intercept) vs Pearson X^2",
        rule="max relative error <= 1e-6 (float32 precision)",
    ),
    "P2": dict(
        statement="prop:nullmean",
        test="Monte Carlo mean of the root gain of a null split at fixed n_L vs n/(n-1)",
        rule="within 0 + 3 SE in every setting",
    ),
    "P3": dict(
        statement="thm:limit",
        test="median best null split, n = 1e3..1e5, at each n_min (XGBoost defaults, prevalence intercept)",
        rule="relative range across n <= 0.10 at every n_min",
    ),
    "P4": dict(
        statement="thm:limit",
        test="P(root uses the informative feature): XGBoost depth-1 fits vs the limit (4) by Monte Carlo",
        rule=(
            "chi-square goodness of fit over cells, with the variance of each cell the sum of the "
            "binomial variance of the measurement and the Monte Carlo variance of the prediction: "
            "p >= 0.01; and mean absolute error <= 0.05. Exact search; hist reported conditional on (H1)."
        ),
    ),
    "P5": dict(
        statement="thm:scaling",
        test=(
            "incremental effect of log10 p on the root decision given s: logistic regression of the "
            "per-run root choice on a monotone spline of s plus beta * log10 p, clustered by cell"
        ),
        rule="95% CI of the change in win probability per decade of p lies within [-0.05, 0.05]",
    ),
    "P6": dict(
        statement="prop:toll",
        test="no split in the first tree under the XGBoost 2.1 default intercept: identity (7) on each run's data vs XGBoost",
        rule="per-run agreement >= 99%",
    ),
    "P7": dict(
        statement="conj:ensemble (a)",
        test="ensemble FNR (one-shot rule, prevalence intercept) vs root failure 1 - E[Q(M)^K], per informative cell",
        rule="measured >= predicted - 2 SE in every informative cell",
    ),
    "P8": dict(
        statement="conj:ensemble (b)",
        test="paired difference FNR(smaller p) - FNR(larger p) at equal n_min, delta (E2, prevalence intercept)",
        rule="difference >= -2 SE in every informative pair",
    ),
    "P9": dict(
        statement="prop:level",
        test="pooled FPR of one-shot max-shadow rules with K = 10 (E4, E4b)",
        rule="measured <= 1/(K+1) + 2 SE (data-set-clustered SE), for each method",
    ),
    "P10": dict(
        statement="prop:level (per-feature shadows)",
        test="hit rate of a null column with per-feature shadows, continuous and discrete cases (E10)",
        rule="measured <= 1/(K+1) + 2 SE in every case and importance type",
    ),
    "P11": dict(
        statement="cor:shadows",
        test="ensemble FNR as K grows (E8): K = 5, 50, 200",
        rule="FNR(K') >= FNR(K) - 2 SE for K' > K in every informative cell",
    ),
    "P12": dict(
        statement="prop:rounds",
        test="keep rate of irrelevant candidates after R rounds (E7) vs bound (10) and the limit bound 4 E[theta^2]",
        rule="measured <= bound + 2 SE at every R; FPR at R = 60 <= 4 E[theta^2] + 2 SE",
    ),
    "P13": dict(
        statement="prop:mw",
        test="Mann-Whitney FNR (shadow rule and BH q = 0.2) vs the prediction from N(sqrt(3 s_AUC / pi), 1) (E6)",
        rule="within 0.05 + 2 SE in every informative cell",
    ),
    "P14": dict(
        statement="prop:mw (non-Gaussian features)",
        test="Mann-Whitney keep rate of relevant columns under t3, log-normal and correlated-Gaussian features (E9) vs the s_AUC prediction",
        rule="within 0.05 + 2 SE in every informative cell",
    ),
    "P15": dict(
        statement="prop:accounting",
        test="sum of split gains vs leaf-sum identity (13) and Newton decrease (14), in the unit tests",
        rule="relative error <= 1e-5",
    ),
}
