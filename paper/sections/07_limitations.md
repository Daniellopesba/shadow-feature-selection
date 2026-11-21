# 7. Limitations

* **The theory covers the first round.** Theorem 1 and Propositions 1–5 are
  about the first tree. Later rounds are described heuristically
  (Section 3.6), and their effect is measured, not derived. Under the
  prevalence intercept the first-round account explains most of what we
  measure (Section 4.4), but it is not a proof for full ensembles.
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
  did not explain why LightGBM shows a residual $p$-dependence and random
  forests do not (Section 4.5).
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
effect size as $s = n_{\min}\delta^2$, not the imbalance ratio. At the first
boosting round this is a theorem: split gain is Pearson's chi-square
statistic, and in the small-prevalence limit its distribution depends on the
data only through $n_{\min}$ and $\delta$. For full 300-round ensembles, $s$
explains 94% of the variance in measured failure rates.

Part of what looked like an imbalance effect in earlier experiments,
including the first draft of this paper, was the default intercept of
XGBoost 2.x, which is about 0.12 whenever $p$ is small. Setting `base_score`
to the prevalence removes it.

No importance-threshold method recovers moderate effects from 25 or fewer
positives. Above that, repeated rounds that keep tentative features give the
best balance of missed and spurious features. The `shadowselect` package
implements the diagnostics and the selector, and every number in this paper
can be regenerated from the repository.
