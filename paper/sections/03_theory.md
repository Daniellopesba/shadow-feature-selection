# 3. First-round theory: split gain is a chi-square statistic

This section analyses the first tree of a gradient-boosted ensemble with
log-loss. Statements labelled *proposition* are exact. *Theorem 1* is
asymptotic. Section 3.6 is heuristic. Section 4 checks each claim by
simulation and measures what happens in later rounds, where we have no
closed form.

**Scope.** We analyse XGBoost's second-order split criterion with *exact
greedy* threshold search (every midpoint between distinct values is a
candidate). `tree_method="hist"`, the default for large data, restricts
thresholds to at most `max_bin` $-1$ quantile cuts (default 255). We note
where this matters. LightGBM [@ke2017lightgbm] uses the same criterion. Its defaults
(prevalence intercept, $\lambda = 0$, `min_sum_hessian_in_leaf` $=10^{-3}$,
`min_data_in_leaf` $=20$) constrain children by sample count rather than by
Hessian, so Proposition 3 does not carry over to it.

## 3.1 Gradients at initialisation

For log-loss with raw score $F$ and $\sigma(F) = 1/(1+e^{-F})$,

$$
g_i = \sigma(F_i) - y_i, \qquad h_i = \sigma(F_i)\bigl(1-\sigma(F_i)\bigr).
$$

Before the first tree every sample has the same score $F_0 = \operatorname{logit}(b)$,
where $b$ is the *base score*, the initial predicted probability. So in the
whole first tree $h_i = h = b(1-b)$ is constant. Write $\hat p = n_{\min}/n$. Two
choices of $b$ matter:

* **Prevalence intercept**, $b = \hat p$: the maximum-likelihood intercept. It
  makes the root gradient sum $G = \sum_i g_i$ exactly zero.
* **Automatic intercept in XGBoost 2.0–2.1.** The intercept is estimated by one
  Newton step on the log-loss from $F = 0$:
  $F_0 = -\bar g/\bar h = (\hat p - \tfrac12)/\tfrac14 = 4\hat p - 2$, so
  $b = \sigma(4\hat p - 2)$. As $\hat p \to 0$, $b \to \sigma(-2) \approx 0.119$. At
  $\hat p = 0.001$ the initial prediction is about 120 times the true rate. We
  confirmed these values in XGBoost 2.1.3 (Section 4.1). XGBoost 3.x estimates
  the intercept differently. We did not study it.

## 3.2 The split gain

For a set of samples $S$, write $G_S = \sum_{i \in S} g_i$, $H_S = \sum_{i\in S} h_i$,
$n_S = |S|$, $k_S$ for the number of positives in $S$, and
$\hat p_S = k_S/n_S$. Splitting a node $S$ into $L$ and $R$ has gain

$$
\operatorname{Gain} = \frac{G_L^2}{H_L+\lambda} + \frac{G_R^2}{H_R+\lambda}
 - \frac{G_S^2}{H_S+\lambda}, \qquad (1)
$$

and the split is made only if $\operatorname{Gain} > \gamma$ and
$H_L, H_R \ge w$. Here $\lambda$ is the L2 penalty on leaf weights (default 1),
$\gamma$ the minimum split loss (default 0) and $w$ is `min_child_weight` (default 1).
@chen2016xgboost write the objective reduction as half of (1) minus
$\gamma$. Equation (1) is what the software computes, compares with $\gamma$, and
reports in `trees_to_dataframe()` and in the `gain` and `total_gain`
importances. We use the software's convention throughout. Importance type
`gain` is the *mean* of (1) over the splits that use a feature. `total_gain`
is the *sum*.

**Proposition 1 (gain is Pearson's chi-square).** In the first tree, with
$\lambda = 0$, the gain of splitting any node $S$ is

$$
\operatorname{Gain} = \frac{\hat p_S(1-\hat p_S)}{b(1-b)}\; X_S^2, \qquad
X_S^2 = \frac{n_S\,(k_L n_R - k_R n_L)^2}{n_L\, n_R\, k_S\,(n_S-k_S)}, \qquad (2)
$$

where $X_S^2$ is Pearson's chi-square statistic, without continuity correction,
for the $2\times 2$ table of child by class within $S$. At the root, with the
prevalence intercept, the factor is 1 and $\operatorname{Gain} = X^2$.

*Proof.* With constant $h$, $H_T = h\,n_T$. For reals $a, c$ and $x, y > 0$,
$a^2/x + c^2/y - (a+c)^2/(x+y) = (ay - cx)^2/\bigl(xy(x+y)\bigr)$. Take $a = G_L$,
$c = G_R$, $x = h n_L$, $y = h n_R$, which gives
$\operatorname{Gain} = (G_L n_R - G_R n_L)^2/(h\, n_L n_R n_S)$. Since $G_T = b\,n_T - k_T$,
the $b$ terms cancel, leaving $G_L n_R - G_R n_L = k_R n_L - k_L n_R$. Multiply and
divide by $\hat p_S(1-\hat p_S) = k_S(n_S - k_S)/n_S^2$. $\square$

So the first-tree gain is a test statistic for "this split separates the
classes". With $\lambda = 0$ the base score rescales it by the same factor for
every feature and threshold in a node. With $\lambda > 0$ the base score also
adds a penalty (Proposition 5).

## 3.3 The null distribution at a fixed threshold

Call feature $X_j$ *null* if it is continuous and independent of $Y$. Under the
null, given the labels, the ranks of the $n_{\min}$ positives among the $n$
values of $X_j$ are a uniformly random subset. So a split that sends $n_L$
samples left has $k_L \sim \mathrm{Hypergeometric}(n, n_{\min}, n_L)$.

**Proposition 2 (exact null mean).** At the root, for a null feature, a fixed
$n_L$, the prevalence intercept and $\lambda = 0$,
$\mathbb E[\operatorname{Gain}] = \mathbb E[X^2] = n/(n-1)$, for every $n_L$ and
every $n_{\min}$.

*Proof.* With $b = \hat p$, $G = 0$, so $G_R = -G_L$ with $G_L = n_L\hat p - k_L$, and
$\operatorname{Gain} = G_L^2\,(1/H_L + 1/H_R) = G_L^2\, n/\bigl(\hat p(1-\hat p)\,n_L n_R\bigr)$.
Here $\mathbb E[G_L] = 0$ and
$\operatorname{Var}(k_L) = n_L\hat p(1-\hat p)(n-n_L)/(n-1)$. Substituting gives $n/(n-1)$. $\square$

Imbalance therefore does not change the expected gain of an individual
uninformative split. What it changes is the *shape* of the distribution.
$X^2$ is approximately $\chi^2_1$ only when the expected positive counts in both
children, $n_L\hat p$ and $n_R\hat p$, are large. When a child is expected to
hold about one positive or fewer, the right tail is much heavier. For a child
of size $m \ll n$ holding $j$ positives,
$X^2 \approx (j - m\hat p)^2/(m\hat p)$. With $m\hat p = 0.1$ and $j = 2$ that is 36,
against a $\chi^2_1$ 99.9th percentile of 10.8.

**Proposition 3 (`min_child_weight` counts expected positives).** In the first
tree, $H_L \ge w$ means $n_L \ge w/(b(1-b))$. Under the null, a child of that
size holds on average $n_L\hat p \ge w\,\hat p/(b(1-b))$ positives. With the
prevalence intercept this is $w/(1-\hat p) \approx w$. XGBoost's default $w = 1$
thus rules out the splits whose children are expected to hold fewer than about
one positive, which are where the heavy tail lives. With the automatic
intercept at $\hat p = 0.001$ ($b \approx 0.12$) the bound is about $0.0095\,w$ expected
positives, a child of about 10 samples. The constraint then protects almost
nothing. Under `hist`, children are also limited to unions of bins, which
bounds them from below at roughly $n/$`max_bin` samples.

## 3.4 The best split, and why $n_{\min}$ governs it

The root split takes the best threshold of every feature and then the best
feature. Index thresholds by $u = n_L/n$. With the prevalence intercept,
$G_L = n_{\min}u - k_L$, $H_L = n_{\min}u(1-\hat p)$ and $H_R = n_{\min}(1-u)(1-\hat p)$,
so

$$
\operatorname{Gain}(u) = \bigl(k_L(u) - n_{\min}u\bigr)^2
\left[\frac{1}{n_{\min}u(1-\hat p) + \lambda} + \frac{1}{n_{\min}(1-u)(1-\hat p) + \lambda}\right],
\qquad u \in [u_w,\, 1-u_w], \qquad (3)
$$

with $u_w = w/\bigl(n_{\min}(1-\hat p)\bigr)$. Apart from $k_L(u)$, everything here is a
function of $n_{\min}$, $\hat p$, $\lambda$ and $w$.

**Theorem 1 (small-prevalence limit).** Use the Gaussian data model of
Section 2 with $d_{\text{real}}$ informative features (shift $\delta$) and
$d_{\text{noise}}$ null features, the prevalence intercept, exact greedy search,
and $w > 0$ with $n_{\min} > 2w$. Fix $n_{\min}$, $\delta$, $\lambda$, $w$, $d_{\text{real}}$
and $d_{\text{noise}}$, and let $n \to \infty$, so $\hat p \to 0$. Then each feature's
gain process (3) converges in distribution, as a step process in $u$ (Skorokhod
$J_1$ topology), to

$$
\Gamma_\delta(u) = \bigl(N_\delta(u) - n_{\min}u\bigr)^2
\left[\frac{1}{n_{\min}u+\lambda} + \frac{1}{n_{\min}(1-u)+\lambda}\right],
\quad u \in [w/n_{\min},\, 1 - w/n_{\min}], \qquad (4)
$$

independently across features. Here $N_\delta(u) = \#\{i \le n_{\min} : \Phi(Z_i) \le u\}$
and $Z_1, \dots, Z_{n_{\min}}$ are i.i.d. $N(\delta, 1)$ ($\delta = 0$ for a null
feature). The suprema $\sup_u \Gamma_\delta(u)$ converge jointly, and their
law depends only on $(n_{\min}, \delta, \lambda, w)$. With `hist` and a fixed
`max_bin` $= B$, the same holds with $u$ restricted to $\{1/B, \dots, (B-1)/B\}$.

*Proof sketch.* Let $F_n$ be the empirical CDF of the feature over all $n$
samples. A positive with value $Z_i$ has rank fraction $F_n(Z_i)$. The
negatives are a fraction $1 - \hat p \to 1$ of the sample and are i.i.d. $N(0,1)$,
so $\sup_z |F_n(z) - \Phi(z)| \to 0$ almost surely (Glivenko–Cantelli, with the
$n_{\min}$ positives changing $F_n$ by at most $\hat p$). Hence $k_L(u)$, the
number of positives with rank fraction at most $u$, converges as a step
process to $N_\delta(u)$: the jump locations $F_n(Z_i)$ converge to $\Phi(Z_i)$,
which are a.s. distinct. The bracket in (3) converges uniformly on
$[u_w, 1-u_w]$, and $u_w \to w/n_{\min}$. Convergence in $J_1$ follows.
Between jumps $\Gamma_\delta$ is continuous, so its supremum on the compact
interval is attained at an endpoint or as a one-sided limit at a jump. The
supremum is $J_1$-continuous at paths whose jumps do not sit on the interval
endpoints, which holds a.s. Independence across features holds because the
features are conditionally independent given $Y$. $\square$

*Remarks.* (i) The limit law of $\sup \Gamma_\delta$ has atoms. With positive
probability (1–7% in our simulations) the null supremum sits at the boundary
$u = w/n_{\min}$ and takes one of finitely many values. Two features can then
tie exactly. XGBoost breaks exact ties by column order, so in experiments we
randomise column order. The *probability that an informative feature wins*
is therefore a limit of probabilities of strict events plus a tie term.
(ii) The theorem gives no rate. By simulation, at $n_{\min} = 50$ the 95th
percentile of the null supremum is 7.3, 8.0, 8.5 and 8.7 for
$n = 100, 200, 500, 10^5$, against 8.9 in the limit. So "$n_{\min}$ governs" is
accurate once $\hat p \lesssim 0.1$. Exact finite-$n$ terms enter through
$n\hat p(1-\hat p)$. (iii) For a null feature the limit is the maximally selected
chi-square statistic of $n_{\min}$ uniform points
[@miller1982maximally; @lausen1992maximally]. With $\lambda = w = 0$ it is the
supremum of a squared standardised uniform empirical process, whose maximum
grows without bound [@jaeschke1979asymptotic; @donoho2004higher]. The truncation
$u \ge w/n_{\min}$ is what keeps it small.

**Proposition 4 (signal strength).** For an informative feature, the Pearson
statistic at threshold $t$ (on the feature's scale) has local-alternative
non-centrality

$$
\nu(t) = n\,\hat p(1-\hat p)\,\frac{\bigl(\Phi(t) - \Phi(t-\delta)\bigr)^2}{q(1-q)},
\qquad q = \hat p\,\Phi(t-\delta) + (1-\hat p)\,\Phi(t). \qquad (5)
$$

For $\hat p \to 0$ and small $\delta$ the maximum over $t$ is near $t = 0$:

$$
\nu^\star \approx \tfrac{2}{\pi}\, n_{\min}\,\delta^2 \qquad (\delta \lesssim 0.3). \qquad (6)
$$

*Proof.* $X^2/n$ converges to the squared phi coefficient
$\phi^2 = \bigl(P(L, Y{=}1) - P(L)\,\hat p\bigr)^2/\bigl(P(L)P(R)\,\hat p(1-\hat p)\bigr)$, and
$\nu = n\phi^2$ [standard for Pearson's statistic under local alternatives].
Here $P(L, Y{=}1) - P(L)\,\hat p = \hat p(1-\hat p)\bigl(\Phi(t-\delta) - \Phi(t)\bigr)$, which
gives (5). For small $\delta$, $\Phi(t) - \Phi(t-\delta) \approx \delta\varphi(t)$, and
$\varphi(t)^2/\bigl(\Phi(t)(1-\Phi(t))\bigr)$ peaks at $t = 0$ with value $2/\pi$. $\square$

For larger $\delta$ the optimal threshold moves into the positives' tail and
(6) underestimates. At $\hat p = 0.001$, $\nu^\star$ is 1.09 times (6) at
$\delta = 0.5$ and 1.47 times at $\delta = 1$.

**Consequence: detectability scales with $n_{\min}\delta^2$, to first order.**
The informative feature's best split competes with the best split of each
shadow. By Theorem 1 both distributions depend on $n_{\min}$ and $\delta$ only,
and by (6) the informative one is shifted by an amount proportional to
$s = n_{\min}\delta^2$. The null side grows only slowly with $n_{\min}$: in the
limit (4) with $\lambda = w = 1$, the median null supremum is 2.3, 3.7, 4.4 and
4.9 at $n_{\min} = 10, 50, 200, 800$. So $s$ is the first-order summary. At fixed
$s = 12.5$, the probability that one informative feature wins the root against
nine null features is 0.84, 0.87, 0.82 and 0.79 at those four values of
$n_{\min}$. That is close, but not exactly constant, because the null side
drifts upward. Halving $\delta$ costs roughly as much as dividing $n_{\min}$ by
four.

The library computes these probabilities by Monte Carlo on (4)
(`theory.detection_probability`). `assess_imbalance` reports the smallest
$\delta$ that wins against ten shadows with probability 0.9.

## 3.5 What the automatic intercept changes

**Proposition 5 (the intercept toll).** Let $b \neq \hat p$, so the root has
$G \neq 0$. Write $r = G/H = (b - \hat p)/(b(1-b))$ and, for a child $c$,
$D_c = G_c - rH_c = -(k_c - \hat p\, n_c)$, the child's excess of positives over
the prevalence. Then, exactly,

$$
\operatorname{Gain} = \sum_{c \in \{L,R\}} \frac{(D_c - r\lambda)^2}{H_c+\lambda}
\;-\; \lambda r^2\,\frac{H+2\lambda}{H+\lambda}. \qquad (7)
$$

When both children satisfy $H_c \gg \lambda$,

$$
\operatorname{Gain} \approx \frac{\hat p(1-\hat p)}{b(1-b)}\,X^2 \;-\; \lambda r^2 . \qquad (8)
$$

*Proof.* Write $rH_c + D_c = r(H_c+\lambda) + (D_c - r\lambda)$ and expand:
$G_c^2/(H_c+\lambda) = r^2(H_c+\lambda) + 2r(D_c - r\lambda) + (D_c-r\lambda)^2/(H_c+\lambda)$.
Summing over the two children and using $D_L + D_R = 0$ and $H_L + H_R = H$
gives $r^2H - 2\lambda r^2 + \sum_c (D_c - r\lambda)^2/(H_c+\lambda)$. The parent term is
$(rH)^2/(H+\lambda) = r^2H - \lambda r^2 H/(H+\lambda)$. Subtracting gives (7). For
$H_c \gg \lambda$, $(D_c - r\lambda)^2/(H_c+\lambda) = D_c^2/H_c + O(\lambda/H_c)$ relative
terms, and $\sum_c D_c^2/H_c$ is the right-hand side of Proposition 1 at the root.
$\square$

So every balanced split pays a fixed toll $\lambda r^2$. It is positive only if

$$
X^2 > \lambda\,\frac{(b-\hat p)^2}{b(1-b)\,\hat p(1-\hat p)} \approx \frac{0.136\,\lambda}{\hat p}
\quad\text{(automatic intercept, small } \hat p). \qquad (9)
$$

At $\hat p = 0.001$ that is $X^2 > 136$, far above anything a feature with a
moderate effect produces. The first tree is then often a single leaf, which
only lowers the intercept. Section 4.2 measures how often the root does not split at all. The toll shrinks as later rounds pull the score toward
$\operatorname{logit}\hat p$, which at learning rate 0.1 takes tens of rounds.
Until then, the splits that pass are the exceptions to (8): small children
that are strongly enriched in positives. By Proposition 3 the automatic
intercept allows children of about ten samples. Any continuous feature,
informative or not, can isolate a few positives that way. These are
memorisation splits, and they are made before the signal is learned.

The prediction is that, under the automatic intercept, failure at a fixed
$n_{\min}$ gets worse as $\hat p$ falls. The toll (9) grows like $1/\hat p$, and the
intercept-correction phase grows like $\log(b/\hat p)$. With the prevalence
intercept this effect should disappear. Sections 4.2–4.4 test both
predictions.

## 3.6 Later rounds (heuristic)

After the first tree the scores differ across samples and Proposition 1 no
longer holds exactly. We describe the two effects that the experiments in
Section 4.4 measure.

1. **The signal is used up.** Once leaf values have absorbed the location
   shift, informative features have little left to explain. What remains is
   idiosyncratic: positives the model has not fit, each with $g_i \approx -1$.
2. **Memorisation is open to every feature.** A depth-6 tree can carve out
   regions around a few unfit positives, and any continuous feature serves
   about equally well for that. For a leaf $S$ the gain contribution is
   $G_S^2/(H_S+\lambda) \le G_S^2/(w+\lambda)$, since $H_S \ge w$. An unfit positive
   with predicted probability $\pi_i$ contributes $g_i^2/h_i = (1-\pi_i)/\pi_i$, which
   is large when $\pi_i$ is small. With the prevalence intercept,
   $\pi_i \approx \hat p$ early on. A leaf then needs about $w/\hat p$ samples
   (Proposition 3), so isolating single positives is blocked until the
   scores spread out. Under the automatic intercept it is possible from the
   first rounds.

How much memorisation gain is available? The training log-loss at the
prevalence intercept is $n\,H_2(\hat p) \approx n_{\min}\,(1 + \ln(1/\hat p))$ nats, where
$H_2$ is the binary entropy. Since gain is a second-order estimate of loss
reduction, this caps the total the ensemble can record. Only the part due to
the location shift is specific to informative features, and that part grows
with $n_{\min}\delta^2$, not with $\ln(1/\hat p)$. At fixed $n_{\min}$, a smaller $\hat p$
therefore leaves a larger budget for memorisation, a residual $\log(1/\hat p)$
dependence that the first-round theory does not capture.

The two importance types dilute differently. Under `total_gain`,
memorisation gain adds to every feature, so normalised shares drift toward
$1/d$. Under `gain` (the mean per split), each feature's many small
memorisation splits pull its average down toward the typical memorisation
gain. That erases the informative feature's advantage from its few large,
early splits. Either way, the max-shadow rule starts rejecting informative
features. Section 4.4 reads importances after $k = 1, 3, 10, 30, 100, 300$ rounds
to measure this.
