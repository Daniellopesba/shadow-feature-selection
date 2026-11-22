# 7. Limitations

* **What is proved and what is not.** Theorems 1–2 and Propositions 1–5
  are about the first tree; Propositions 6–7 hold for any rank-based learner
  and any number of trees but concern only the level of the rule, not its
  power; Proposition 9 is an accounting identity. The power of the shadow rule
  on a full ensemble is Conjecture 2, supported by experiments but not
  proved. Theorem 2 restricts thresholds to a fixed window; the full-window
  behaviour is Conjecture 1. Theorem 1 gives no rate of convergence in $n$.
* **Simulation design.** Features are Gaussian, conditionally independent,
  and differ only by a location shift. Correlated features, heavy tails,
  categorical features and interaction-only signals are not studied.
  Correlation in particular splits importance between correlated features,
  which should make selection harder, not easier.
* **One importance family in depth.** We analyse split-gain importance. SHAP
  importances, as in BorutaSHAP, aggregate the same tree structure and are
  likely affected, but we have not tested them. Model-X knockoffs
  [@candes2018panning; @barber2015controlling] control false discoveries, but
  their power also depends on the importance statistic. We did not test them.
* **Learner-specific behaviour.** The automatic-intercept effect is specific
  to XGBoost 2.0–2.1, and XGBoost 3.x estimates the intercept differently. We
  did not explain why LightGBM shows a larger residual $p$-dependence than
  the other learners (Section 4.5), nor the residual $p$-dependence of
  XGBoost at weak signal (Section 4.3).
* **Binary targets only.** Multi-class problems with a rare class reduce to
  the same quantities one-vs-rest, but we did not test them.
* **Real data.** The repository includes a real-data example (breast cancer
  data with injected noise and a subsampled minority class) that shows the
  same pattern. It is an illustration, not a benchmark.

# 8. Conclusion

The shadow-feature heuristic compares a feature's importance with the best
importance that pure noise achieves. With a rare positive class, informative
features lose that comparison more and more often. The quantity that
controls this is the number of minority samples $n_{\min}$, combined with the
effect size as $s = n_{\min}\delta^2$, not the imbalance ratio. For the first
boosting round this is proved: split gain is Pearson's chi-square statistic,
in the small-prevalence limit its law depends only on $n_{\min}$ and $\delta$, and
as $n_{\min}$ grows only on $s$. The rule's false positive level, $1/(K+1)$, does
not depend on imbalance at all; imbalance only takes away power. For full
300-round ensembles, $s$ explains 94% of the variance in measured failure
rates (90% out of sample), and a proof remains open.

Much of what looks like an imbalance effect in practice is the default intercept of
XGBoost 2.x, which is about 0.12 whenever $p$ is small. Setting `base_score`
to the prevalence removes most of it.

The same theory explains why a simple rank test does better: it averages the
evidence over thresholds where a split takes the maximum, and it does not
pay for searching. At equal false positive rate it lost far fewer relevant
features than any tree-based shadow method below about 50 positives. Below
about 25 positives nothing we tested works well. The `shadowselect` package
implements the diagnostic, the rank-test screen and the selector, and every
number in this paper can be regenerated from the repository.
