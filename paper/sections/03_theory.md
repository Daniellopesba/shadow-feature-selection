# 3. Theory

This section derives the behaviour of the shadow rule from the split
criterion. Every statement carries one of four labels, and we use them
strictly:

* **Proposition** or **Theorem**: proved. Short proofs are given in the text;
  longer ones are in Appendix C.
* **Corollary**: follows from a proved statement in a few lines.
* **Numerical evaluation**: a quantity that is defined exactly by a proved
  limit, but has no closed form. We compute it by Monte Carlo on the limit
  object itself, not on XGBoost.
* **Conjecture**: not proved. We state the evidence for it and the experiment
  that tests it.

The proved results describe the *first tree* of a gradient-boosted ensemble
with log-loss (Sections 3.1–3.5), the shadow comparison itself
(Section 3.6, any learner whose importances depend on each column only
through its ranks), and a rank test applied to the same data (Section 3.7).
For the full 300-tree ensemble we prove one accounting identity and state
the rest as a conjecture (Section 3.8). Section 4 tests every statement.

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
$\lambda = 0$, the gain of splitting any node $S$ with $0 < k_S < n_S$ into
non-empty children is

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
nothing.

*Remark.* Under `hist`, children are also unions of bins, which in practice
bounds them from below at roughly $n/$`max_bin` samples. This is a heuristic
description of the binning, not part of the proposition.

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
gain process (3), a càdlàg function of $u$, converges in distribution to

$$
\Gamma_\delta(u) = \bigl(N_\delta(u) - n_{\min}u\bigr)^2
\left[\frac{1}{n_{\min}u+\lambda} + \frac{1}{n_{\min}(1-u)+\lambda}\right],
\quad u \in [w/n_{\min},\, 1 - w/n_{\min}], \qquad (4)
$$

in the sense that the counting processes $k_L$ converge in the Skorokhod $J_1$
topology and the gains converge uniformly after a time change that tends to
the identity, jointly and independently across features. Here $N_\delta(u) = \#\{i \le n_{\min} : \Phi(Z_i) \le u\}$
and $Z_1, \dots, Z_{n_{\min}}$ are i.i.d. $N(\delta, 1)$ ($\delta = 0$ for a null
feature). The suprema $M_j = \sup_u \Gamma_{\delta_j}(u)$ converge jointly, and their
law depends only on $(n_{\min}, \delta, \lambda, w)$.

**Corollary (histogram search).** With `tree_method="hist"` and a fixed
`max_bin` $= B$, suppose (H1) the cut points have empirical-CDF positions
converging to $j/B$ almost surely, and (H2) no $j/B$ equals $w/n_{\min}$ or
$1 - w/n_{\min}$. Then the same conclusion holds with $u$ restricted to
$\{1/B, \dots, (B-1)/B\} \cap [w/n_{\min}, 1 - w/n_{\min}]$. (H1) is an assumption about
XGBoost's quantile sketch, which we have not verified analytically.

*Proof.* Appendix C.1. The idea: a positive's rank fraction among all $n$
values converges to $\Phi(Z_i)$ because the negatives, which are almost all of
the sample, have empirical CDF converging uniformly to $\Phi$. So the counting
process $k_L$ converges to $N_\delta$ and the bracket converges uniformly. $\square$

*Remarks.* (i) The limit law of $M$ has atoms. With positive probability
(1–7% in our simulations) the null supremum sits at the boundary
$u = w/n_{\min}$ and takes one of finitely many values, so two features can
tie exactly. XGBoost breaks exact ties by column order, so the experiments
randomise column order. (ii) The theorem gives no rate. By simulation, at
$n_{\min} = 50$ the 95th percentile of the null supremum is 7.3, 8.0, 8.8 and 8.9
for $n = 100, 200, 500, 10^5$, against 9.1 in the limit. So "$n_{\min}$ governs"
is accurate once $\hat p \lesssim 0.1$. (iii) For a null feature the limit is
the maximally selected chi-square statistic of $n_{\min}$ uniform points
[@miller1982maximally; @lausen1992maximally]. With $\lambda = w = 0$ it is the
supremum of a squared standardised uniform empirical process. That supremum
is finite almost surely but has infinite mean (it is at least of order
$1/(n_{\min}U_{(1)})$ for the smallest uniform $U_{(1)}$), and it tends to infinity in
probability as $n_{\min} \to \infty$ [@jaeschke1979asymptotic; @donoho2004higher]. The
truncation $u \ge w/n_{\min}$ and the penalty $\lambda$ are what keep it small.

**Proposition 4 (signal at a fixed threshold).** In the limit of Theorem 1,
$N_\delta(u) \sim \mathrm{Binomial}(n_{\min}, q_\delta(u))$ with
$q_\delta(u) = \Phi\bigl(\Phi^{-1}(u) - \delta\bigr)$. The standardised deviation
$(N_\delta(u) - n_{\min}u)^2/(n_{\min}u(1-u))$, which is $\Gamma_\delta(u)$ for
$\lambda = 0$, has mean $q(1-q)/\bigl(u(1-u)\bigr) + \nu(u)$ with non-centrality

$$
\nu(u) = n_{\min}\,\frac{\bigl(q_\delta(u) - u\bigr)^2}{u(1-u)}, \qquad
\lim_{\delta \to 0}\ \frac{\max_u \nu(u)}{n_{\min}\delta^2} = \frac{2}{\pi}, \qquad (5)
$$

with the maximum attained at $u \to 1/2$, the median split. At finite $n$ the
same non-centrality is
$\nu(t) = n\hat p(1-\hat p)\bigl(\Phi(t) - \Phi(t-\delta)\bigr)^2/\bigl(q(1-q)\bigr)$ with
$q = \hat p\,\Phi(t-\delta) + (1-\hat p)\Phi(t)$, which tends to (5) as $\hat p \to 0$.

*Proof.* A positive satisfies $\Phi(Z_i) \le u$ iff $Z_i - \delta \le \Phi^{-1}(u) - \delta$,
which has probability $q_\delta(u)$; the $Z_i$ are independent, so $N_\delta(u)$ is
binomial and the mean follows. For the limit, write $x = \Phi^{-1}(u)$. By the
mean value theorem $u - q_\delta(u) = \delta\int_0^1 \varphi(x - \theta\delta)\,d\theta$, so

$$
\frac{\nu(u)}{n_{\min}\delta^2} = \frac{\bigl(\int_0^1\varphi(x-\theta\delta)\,d\theta\bigr)^2}{\Phi(x)\bigl(1-\Phi(x)\bigr)}
\;\longrightarrow\; \frac{\varphi(x)^2}{\Phi(x)\bigl(1-\Phi(x)\bigr)}
$$

uniformly in $x$ (Appendix C.2). The right side, $f_0(x)$, is the efficiency of
dichotomising a normal sample at $x$ [@cox1957note]. It is at most $2/\pi$, with
equality only at $x = 0$. For independent standard normals $Z_1, Z_2$ and
$x \neq 0$, $(2\Phi(x) - 1)^2 = P(|Z_1| \le |x|, |Z_2| \le |x|) < P(Z_1^2 + Z_2^2 \le 2x^2) = 1 - e^{-x^2}$,
because the square lies strictly inside the disc of radius $|x|\sqrt 2$. Hence
$\Phi(x)(1-\Phi(x)) = \tfrac14\bigl[1 - (2\Phi(x)-1)^2\bigr] > \tfrac14 e^{-x^2}$, and
$f_0(x) < (e^{-x^2}/2\pi)/(e^{-x^2}/4) = 2/\pi = f_0(0)$. With the uniform convergence of
Appendix C.2, the maximiser of $\nu$ tends to $u = 1/2$. $\square$

So $\max_u \nu \approx (2/\pi)\, n_{\min}\delta^2$ for small $\delta$. For larger $\delta$ the
optimal threshold moves into the positives' tail and the approximation
underestimates: at $\delta = 0.5$ the exact maximum of (5) is 1.09 times
$(2/\pi)n_{\min}\delta^2$, and at $\delta = 1$ it is 1.47 times.

**Theorem 2 (the scaling variable $s = n_{\min}\delta^2$).** Let $n_{\min} \to \infty$ and
$\delta \to 0$ with $n_{\min}\delta^2 \to s \in [0, \infty)$ (take $\delta \ge 0$; for $\delta < 0$ replace
$\mathbb B$ by $-\mathbb B$), after the limit $n \to \infty$ of Theorem 1, and fix $\lambda \ge 0$, $w \ge 0$ and
a window $[\varepsilon, 1-\varepsilon]$ with $0 < \varepsilon < 1/2$. On that window, the
limit gain processes (4) of independent features converge jointly and
uniformly in distribution to independent copies of

$$
\Gamma^{(s)}(u) = \frac{\bigl(\mathbb B(u) - \sqrt s\,\psi(u)\bigr)^2}{u(1-u)}, \qquad
\psi(u) = \varphi\bigl(\Phi^{-1}(u)\bigr), \qquad (6)
$$

where $\mathbb B$ is a Brownian bridge, $s$ is the feature's own scaling
variable, and $s = 0$ for a null feature. Consequently, if candidate thresholds
are restricted to the window, the probability that an informative feature's
best split beats the best splits of $K$ null features converges to

$$
\pi_\varepsilon(s, K) = P\left(\sup_{[\varepsilon,1-\varepsilon]} \Gamma^{(s)} > \max_{k \le K}\ \sup_{[\varepsilon,1-\varepsilon]} \Gamma^{(0)}_k\right),
$$

which depends on $s$, $K$ and the window, and not on $\lambda$, $w$ or how $n_{\min}$
and $\delta$ combine to give $s$. It is non-decreasing in $s$, and
$\pi_\varepsilon(0, K) = 1/(K+1)$.

*Proof.* Appendix C.3. The counting process satisfies
$N_\delta(u) - n_{\min}u = \bigl[N_\delta(u) - n_{\min}q_\delta(u)\bigr] + n_{\min}\bigl(q_\delta(u) - u\bigr)$.
After dividing by $\sqrt{n_{\min}}$, the first term is a uniform empirical process
evaluated at $q_\delta(u) \to u$, which converges to $\mathbb B(u)$ (Donsker), and the
second term converges to $-\sqrt s\,\psi(u)$ uniformly (Proposition 4's
expansion). The bracket in (4) times $n_{\min}$ converges to $1/\bigl(u(1-u)\bigr)$
uniformly on the window (this needs no condition on $w$, which the window
makes irrelevant). Monotonicity in $s$ is Anderson's inequality
[@anderson1955integral; @borell1974convex] applied to the symmetric convex set
$\{f : \sup |f|/\sqrt{u(1-u)} \le c\}$. At $s = 0$ the $K+1$ suprema are i.i.d. with
a continuous law. $\square$

Theorem 2 is the precise sense in which $s = n_{\min}\delta^2$ governs detection:
in the iterated limit ($n \to \infty$, then $n_{\min} \to \infty$), $n_{\min}$ and $\delta$ enter only
through $s$. We prove no rate that is joint in $n$ and $n_{\min}$. Halving $\delta$
costs exactly as much as dividing $n_{\min}$ by four.

Theorem 2 restricts thresholds to a fixed window. XGBoost searches all
thresholds with $H_L, H_R \ge w$, which is the window $[w/n_{\min}, 1 - w/n_{\min}]$
of Theorem 1. That window grows with $n_{\min}$, and near its edges the null
supremum keeps growing.

**Conjecture 1 (slow drift on the full window).** On the full window, the
median null supremum grows like $2\log\log n_{\min}$, and at fixed $s$ and $K$ the
win probability decreases slowly toward $1/(K+1)$ as $n_{\min} \to \infty$.

*Evidence.* A classical result covers the unregularised analogue: the
squared supremum of the standardised uniform empirical process over
$[a_m, 1-a_m]$, with $a_m \to 0$ and $m a_m \to \infty$, is $2\log\log(1/a_m)\,(1 + o_P(1))$
[@jaeschke1979asymptotic; @csorgo1993weighted]. With $a_m$ of order
$\log m/m$ or $m^{-c}$, this is $2\log\log m\,(1 + o_P(1))$. Near the edges the drift
$\sqrt s\,\psi(u)/\sqrt{u(1-u)}$ in (6) tends to zero, so all $K+1$ processes
look alike where the supremum is attained. What we have not proved is the
behaviour of the regularised boundary region $u \approx w/n_{\min}$. Numerical
evaluation of (4) with $\lambda = w = 1$ and $K = 9$ null features:

| $n_{\min}$ | 10 | 50 | 200 | 1,000 | 5,000 | 20,000 |
|:--|--:|--:|--:|--:|--:|--:|
| median null supremum | 2.40 | 3.63 | 4.54 | 5.24 | 5.74 | 6.14 |
| $P$(informative wins), $s = 12.5$ | 0.87 | 0.85 | 0.82 | 0.79 | 0.80 | 0.76 |

So over the range that matters in practice, $s$ fixes the win probability to
within about $\pm 0.05$, and the residual dependence on $n_{\min}$ is a slow
decrease, not an increase. (1,500 signal and 6,000 null draws per column;
Monte Carlo standard error about 0.01.)

The library computes these probabilities by Monte Carlo on (4)
(`theory.detection_probability`). `assess_imbalance` reports the smallest
$\delta$ that wins against a given number of shadows with probability 0.9.

## 3.5 What the automatic intercept changes

**Proposition 5 (the intercept toll).** Let $b \neq \hat p$, so the root has
$G \neq 0$. Write $r = G/H = (b - \hat p)/(b(1-b))$ and, for a child $c$,
$D_c = G_c - rH_c = -(k_c - \hat p\, n_c)$, the child's excess of positives over
the prevalence. Then, exactly,

$$
\operatorname{Gain} = \sum_{c \in \{L,R\}} \frac{(D_c - r\lambda)^2}{H_c+\lambda}
\;-\; \lambda r^2\,\frac{H+2\lambda}{H+\lambda}. \qquad (7)
$$

With $\kappa = \hat p(1-\hat p)/\bigl(b(1-b)\bigr)$ and $H_{\min} = \min(H_L, H_R)$, the remainder is
bounded by

$$
\left|\operatorname{Gain} - (\kappa X^2 - \lambda r^2)\right| \le \frac{\lambda}{H_{\min}}\bigl(\kappa X^2 + 2\lambda r^2\bigr)
+ 2\lambda r^2 \sum_c \frac{|\hat p_c - \hat p|}{|b - \hat p|}, \qquad (7')
$$

so when both children satisfy $H_c \gg \lambda$ and their prevalences are close to
the root's, $|\hat p_c - \hat p| \ll |b - \hat p|$,

$$
\operatorname{Gain} \approx \frac{\hat p(1-\hat p)}{b(1-b)}\,X^2 \;-\; \lambda r^2 . \qquad (8)
$$

*Proof.* Write $rH_c + D_c = r(H_c+\lambda) + (D_c - r\lambda)$ and expand:
$G_c^2/(H_c+\lambda) = r^2(H_c+\lambda) + 2r(D_c - r\lambda) + (D_c-r\lambda)^2/(H_c+\lambda)$.
Summing over the two children and using $D_L + D_R = 0$ and $H_L + H_R = H$
gives $r^2H - 2\lambda r^2 + \sum_c (D_c - r\lambda)^2/(H_c+\lambda)$. The parent term is
$(rH)^2/(H+\lambda) = r^2H - \lambda r^2 H/(H+\lambda)$. Subtracting gives (7). For
$H_c \gg \lambda$, $\sum_c D_c^2/(H_c+\lambda) = \sum_c D_c^2/H_c\,(1 + O(\lambda/H_c))$, and
$\sum_c D_c^2/H_c$ is the right-hand side of Proposition 1 at the root. The
remaining terms are $-2r\lambda\sum_c D_c/(H_c+\lambda) + r^2\lambda^2\sum_c 1/(H_c+\lambda)$
plus the last term of (7), which tends to $\lambda r^2$ when $H \gg \lambda$. Since
$D_c/H_c = (\hat p - \hat p_c)/\bigl(b(1-b)\bigr)$ and $r = (b-\hat p)/\bigl(b(1-b)\bigr)$, the cross
term is $\lambda r^2 \cdot O\bigl(|\hat p_c - \hat p|/|b - \hat p|\bigr)$, which is small under the
second condition. Collecting the four remainder terms,
$-\sum_c D_c^2\lambda/\bigl(H_c(H_c+\lambda)\bigr) - 2r\lambda\sum_c D_c/(H_c+\lambda) + r^2\lambda^2\sum_c 1/(H_c+\lambda) - \lambda^2r^2/(H+\lambda)$,
and bounding each gives (7'). $\square$

So every balanced split pays a toll $\lambda r^2$. The toll itself is roughly
constant: with the automatic intercept, $r \to 1/(1-\sigma(-2)) \approx 1.135$ as
$\hat p \to 0$, and $\lambda r^2 = 1.10\lambda$ at $\hat p = 0.01$ and $1.27\lambda$ at $\hat p = 0.001$. What
changes is the factor in front of $X^2$: $\hat p(1-\hat p)/(b(1-b))$ is about $0.09$ at
$\hat p = 0.01$ and $0.0095$ at $\hat p = 0.001$. Under (8), that is, up to the remainder (7'), the gain is therefore positive only if

$$
X^2 > \lambda\,\frac{(b-\hat p)^2}{b(1-b)\,\hat p(1-\hat p)} \;\approx\; \frac{e^{-2}\,\lambda}{\hat p}
\quad\text{(automatic intercept, } \hat p \to 0\text{)}, \qquad (9)
$$

since $b \to \sigma(-2)$ and $b/(1-b) \to e^{-2} \approx 0.135$. The exact threshold is
$12.0\lambda$ at $\hat p = 0.01$ and $133.7\lambda$ at $\hat p = 0.001$, far above anything a
feature with a moderate effect produces at the root. The first tree is then
often a single leaf, which only lowers the intercept. Section 4.2 measures
how often the root does not split at all. The barrier falls as later rounds
pull the score toward $\operatorname{logit}\hat p$, which at learning rate 0.1
takes tens of rounds. Until then, the splits that pass are the exceptions to
(8): small children that are strongly enriched in positives, where
$|\hat p_c - \hat p|$ is not small. By Proposition 3 the automatic intercept allows
children of about ten samples. Any continuous feature, informative or not,
can isolate a few positives that way. These are memorisation splits, and
they are made before the signal is learned.

The prediction is that, under the automatic intercept, failure at a fixed
$n_{\min}$ gets worse as $\hat p$ falls. The threshold (9) grows like $1/\hat p$, and the
intercept-correction phase grows like $\log(b/\hat p)$. With the prevalence
intercept this effect should disappear. Sections 4.2–4.4 test both
predictions.

## 3.6 The shadow comparison: exact level, and the number of shadows

The results so far concern one feature's gain. The shadow rule compares a
candidate with $K$ shadows. Two facts about that comparison hold for any
sample size and any number of boosting rounds, under three conditions on the
learner and its importance measure:

* **(A0) exchangeable rows.** The rows are i.i.d., or at least the rows of a
  null column are exchangeable given $Y$ and the other columns.
* **(A1) rank dependence.** The importances are a function of $Y$, the rank
  vectors of the columns, and randomness independent of the data. Tree
  learners with native training-set importance satisfy this with exact search;
  with `hist` we verified it numerically (importances unchanged under
  $x \mapsto x^3$ and $x \mapsto e^{3x}$) but have not proved it. Held-out permutation
  importance with exact search does *not* satisfy it, because thresholds are
  midpoints of training values. Replacing every column by its ranks before
  fitting makes (A1) hold for any learner and any importance.
* **(A2) column symmetry.** For every data set and every column permutation
  $P$, the importances of the permuted data have the law of the permuted
  importances. Ties broken by column order violate this; shuffling the column
  order before each fit restores it, and `ShadowFeatureSelector` does so.
* **(A3) continuity.** Feature values have no ties within a column.

**Proposition 6 (level of the shadow rule).** Assume (A1)–(A3). Let candidate
$j$ be null in the strong sense that $X_j$ is independent of $(Y, X_{C\setminus j})$ (a
null correlated with other candidates is not covered), and let each of the
$K$ shadows be an independent, uniformly random permutation of some candidate
column, or a column of i.i.d. continuous noise. Then, for the importances of the fitted model,

$$
P\left(I_j > \max_{k \le K} I_{s_k}\right) \le \frac{1}{K+1},
$$

with equality when importances tie with probability zero. This holds for
every $n$, $n_{\min}$, $\hat p$ and number of trees.

*Proof.* By (A1) and (A3) the importances are a function of $Y$, the rank
vectors of all columns, and the learner's own randomness. The rank vector of
$X_j$ is uniform over permutations and independent of $Y$ and the other
candidates, because $X_j$ is null with a continuous law. A shadow's rank
vector is the rank vector of its source column composed with an independent
uniform permutation, so it is again uniform and independent of everything
else; so is the rank vector of a column of i.i.d. continuous noise. Hence the rank vectors of $X_j$ and of the $K$ shadows are i.i.d. and
independent of the remaining columns and $Y$. By (A2) the $K+1$ importances
are exchangeable, so each is the strict maximum with probability at most
$1/(K+1)$, with equality when there are no ties. $\square$

Proposition 6 says that the one-shot max-shadow rule is a test of level
$1/(K+1)$ for each null candidate, whatever the imbalance. Imbalance can only
change the rule's *power*. If all $K$ shadows are permutations of column $j$
itself, the same conclusion holds without (A1) and (A3): given the multiset
of its values, $X_j$ and its permutations are i.i.d. uniform arrangements.
For an informative candidate, Theorem 1 describes the power at the root.

**Corollary 1 (number of shadows).** In the limit of Theorem 1, let $M_j$ be an
informative candidate's best root gain and $M_0$ a null feature's, and write
$F_0(x) = P(M_0 < x)$. Then, counting exact ties as losses,
$P(\text{root win}) = \mathbb E\bigl[F_0(M_j)^K\bigr]$, which is strictly decreasing in $K$
whenever $P\bigl(0 < F_0(M_j) < 1\bigr) > 0$.

*Proof.* The root gain depends on a column only through the ranks of the
positives in it. A permuted column's rank vector is uniform and independent
of its source, so the $K$ shadow suprema are i.i.d. copies of $M_0$ and
independent of $M_j$ (Theorem 1). Given $M_j$, the candidate wins iff every shadow is below
it, which has probability $F_0(M_j)^K$. Take expectations; $x \mapsto x^K$ is
strictly decreasing in $K$ on $(0,1)$. $\square$

Because the limit law has atoms (remark (i)), Theorem 1 does not by itself
give convergence of the finite-$n$ win probability to this value. It gives
the sandwich
$P(M_j > \max_k M_{s_k}) \le \liminf_n P(\text{strict win at } n)$ and
$\limsup_n P(\text{win or tie at } n) \le P(M_j \ge \max_k M_{s_k})$; in our simulations
the two ends differ by at most the atom mass, 1–6%.

So adding shadows lowers the false positive level $1/(K+1)$ and raises the
bar for real features at the same time. The detection thresholds in this
paper assume 5–10 shadows. With $K = 200$, at $n_{\min} = 50$ and $\delta = 0.5$
($s = 12.5$), the root win probability falls from 0.90 ($K = 5$) to 0.47.
Boruta's default uses one shadow per candidate, so $K$ grows with the number
of candidates.

**Proposition 7 (repeated rounds).** Run $R$ rounds on fixed data $D$, each with
fresh shadows and fresh learner randomness, and let
$\pi_j(D) = P(\text{round hit} \mid D)$. A null candidate's hit count satisfies
$H_j \mid D \sim \mathrm{Binomial}(R, \pi_j(D))$ with $\mathbb E\,\pi_j(D) \le 1/(K+1)$. Suppose
the candidate is rejected when $H_j \le r_R$, where $r_R$ is the largest $r$ with
$P\bigl(\mathrm{Binomial}(R, \tfrac12) \le r\bigr) < \alpha/d$ (one-sided, Bonferroni over $d$
candidates; $r_R = -1$ if there is none, and then nothing is rejected). Assume
the hypotheses of Proposition 6 hold in every round. Then the probability that a null candidate is kept (accepted
or tentative) is

$$
P(\text{keep}) = \mathbb E\left[P\bigl(\mathrm{Binomial}(R, \pi_j(D)) > r_R\bigr)\right]
\;\le\; \frac{R}{(K+1)(r_R+1)}, \qquad (10)
$$

and, as $R \to \infty$ with $\alpha$, $d$ and $K$ fixed,
$\limsup_R P(\text{keep}) \le P\bigl(\pi_j(D) \ge \tfrac12\bigr) \le 2/(K+1)$.

*Proof.* Given $D$ the rounds are i.i.d., which gives the binomial law;
$\mathbb E\,\pi_j(D) = P(\text{hit}) \le 1/(K+1)$ by Proposition 6. The bound in (10) is
Markov's inequality for $H_j$. As $R \to \infty$, $r_R/R \to 1/2$ and
$H_j/R \to \pi_j(D)$ almost surely given $D$, so the keep event converges to
$\{\pi_j(D) > 1/2\}$ except on $\{\pi_j(D) = 1/2\}$: for $\pi < 1/2$,
$P(\mathrm{Binomial}(R, \pi) > r_R) \to 0$, and dominated convergence gives the limit.
Markov's inequality bounds
$P(\pi_j \ge 1/2)$ by $2\,\mathbb E\,\pi_j$. $\square$

For $R = 20$, $K = d = 10$ and $\alpha = 0.05$, $r_R = 3$. If $\pi_j$ did not depend on
the data, (10) would be $P(\mathrm{Binomial}(20, 1/11) \ge 4) = 0.10$. Any larger
value measures how much $\pi_j(D)$ varies between datasets. The bound (10)
decreases with $R$, and the limit is at most $P(\pi_j \ge 1/2)$. Proposition 7
does not say that the false positive rate itself falls monotonically in $R$,
nor what the limit is, which depends on the law of $\pi_j(D)$. Section 5.1
measures both.

## 3.7 A rank test on the same data

The gain process (4) is built from the counting process $N_\delta$. A two-sample
rank test is built from the same process.

**Proposition 8 (the Mann–Whitney test in the same limit).** In the setting of
Theorem 1, the normalised Mann–Whitney statistic of feature $j$,
$W_n = U/\bigl(n_{\min}(n - n_{\min})\bigr)$, converges almost surely on the coupling of
Appendix C.1 (hence in distribution) to

$$
W = \frac{1}{n_{\min}}\sum_{i=1}^{n_{\min}} \Phi(Z_i) \;=\; 1 - \frac{1}{n_{\min}}\int_0^1 N_\delta(u)\,du. \qquad (11)
$$

If moreover $n_{\min} \to \infty$ and $n_{\min}\delta^2 \to s$ as in Theorem 2, then
$\sqrt{12\,n_{\min}}\,(W - \tfrac12) \Rightarrow N\bigl(\sqrt{3s/\pi},\, 1\bigr)$ (for $\delta \ge 0$), and the test that
rejects when $|\sqrt{12 n_{\min}}(W - \tfrac12)| > z_{\alpha/2}$ has asymptotic power

$$
\Phi\left(\sqrt{3s/\pi} - z_{\alpha/2}\right) + \Phi\left(-\sqrt{3s/\pi} - z_{\alpha/2}\right). \qquad (12)
$$

*Proof.* Without ties, $U$ counts pairs (positive, negative) with the
positive larger, so $W_n = n_{\min}^{-1}\sum_i F^-_{n}(Z_i)$, where $F^-_n$ is the
empirical CDF of the negatives. By Glivenko–Cantelli $F^-_n \to \Phi$ uniformly,
which gives the first form of (11). The second follows from
$\int_0^1 \mathbf 1\{\Phi(Z_i) \le u\}\,du = 1 - \Phi(Z_i)$. For the normal limit, the
$V_i = \Phi(Z_i)$ are i.i.d., bounded, with
$\mathbb E V_i = P(\xi' \le \delta + \xi) = \Phi(\delta/\sqrt2)$ for independent standard normals
$\xi, \xi'$, and $\operatorname{Var} V_i \to 1/12$ as $\delta \to 0$. The Lindeberg–Feller theorem gives
asymptotic normality, and
$\sqrt{12 n_{\min}}\,\bigl(\Phi(\delta/\sqrt2) - \tfrac12\bigr) = \sqrt{12 n_{\min}}\,\delta\,\varphi(0)/\sqrt2\,(1 + o(1)) \to \sqrt{3s/\pi}$. $\square$

At finite $n$ the exact null variance of $U$, $n_{\min}(n - n_{\min})(n+1)/12$, gives the
same normal limit with $s$ replaced by $s\,(n - n_{\min})/(n+1) \approx s(1 - \hat p)$. For a
two-sample test the information is $n_{\min}(n - n_{\min})/n$, which is $n_{\min}$ when
positives are rare: the rank test, too, is governed by the minority count.

Equation (11) shows how the two procedures use the same information. The
split criterion takes the *largest* standardised deviation of $N_\delta(u)$
from $n_{\min}u$ over thresholds. The rank test takes the *average* deviation,
and its null distribution is known exactly.

**Corollary 2 (where the shadow rule loses power).** Compare the two at the
same per-candidate false positive level $1/(K+1)$. By Proposition 6 this is
the level of the max-shadow rule, so the rank test is run at
$\alpha = 1/(K+1)$. The best single threshold has non-centrality at most
$(2/\pi)s\,(1 + o(1))$ (Proposition 4), the rank test $(3/\pi)s$ (Proposition 8). Hence
any test based on one split needs 1.5 times as many positives as the rank
test for the same power. The ratio $2/3$ is the Pitman efficiency of the
median test relative to the Wilcoxon test under a normal shift
[@mood1954asymptotic].

*Numerical evaluation (the scanning loss).* The shadow rule does not use a
fixed threshold. It compares the best of many thresholds for the candidate
with the best of many thresholds for each of $K$ shadows. We have no closed
form for the effect of this search, so we evaluate the limits numerically, with
$K = 9$ (level 0.1) and $n_{\min} = 50$:

| $s = n_{\min}\delta^2$ | 2 | 5 | 10 | 15 | 25 |
|:--|--:|--:|--:|--:|--:|
| max-shadow rule, best root split (limit (4)) | 0.27 | 0.51 | 0.77 | 0.90 | 0.98 |
| fixed median split, level 0.1 (non-centrality $2s/\pi$) | 0.31 | 0.56 | 0.81 | 0.93 | 0.99 |
| Mann–Whitney, level 0.1 (equation (12)) | 0.40 | 0.71 | 0.93 | 0.98 | 1.00 |

To reach power 0.9, the rank test needs $s \approx 9.0$, the median split
$s \approx 13.5$ and the max-shadow rule on the root split $s \approx 15$. The
ordering is the same at $n_{\min} = 25$ and $250$. Formula (12) agrees with an
exact finite-$n_{\min}$ simulation to within 0.003 at $n_{\min} = 25$.

*Scope.* Proposition 8 is about location shifts, which is the model of
Section 2. A rank test of each feature on its own cannot see a feature that
matters only through an interaction, or only through its variance, and a
tree can. The rank test is therefore a first screen when positives are few,
not a replacement for model-based selection. Section 5 compares the two
empirically.

## 3.8 The full ensemble

After the first tree the scores differ across samples, and Proposition 1 no
longer holds exactly. We have one exact statement about the whole ensemble
and one conjecture.

**Proposition 9 (gain accounting).** Consider one tree with $\gamma = 0$ and leaves
$\ell$ with gradient and Hessian sums $(G_\ell, H_\ell)$, and let $(G, H)$ be the sums
over all samples. Then the reported gains of the tree's splits add up to

$$
\sum_{\text{splits}} \operatorname{Gain} \;=\; \sum_\ell \frac{G_\ell^2}{H_\ell+\lambda} - \frac{G^2}{H+\lambda}. \qquad (13)
$$

With leaf values $-\eta\,G_\ell/(H_\ell+\lambda)$ (learning rate $\eta$), the decrease of the
second-order (Newton) approximation of the training loss is

$$
\Delta\tilde L = \eta\left(1 - \frac{\eta}{2}\right)\sum_\ell \frac{G_\ell^2}{H_\ell+\lambda}
+ \frac{\eta^2\lambda}{2}\sum_\ell \frac{G_\ell^2}{(H_\ell+\lambda)^2}. \qquad (14)
$$

*Proof.* Each split's gain (1) is "children minus parent". Summing over the
internal nodes, every internal node's term appears once with a plus sign (as
a child) and once with a minus sign (as a parent), except the root, so the
sum telescopes to (13). For (14), the Newton model of the loss change is
$\sum_\ell \bigl(G_\ell v_\ell + \tfrac12 H_\ell v_\ell^2\bigr)$ with $v_\ell = -\eta G_\ell/(H_\ell+\lambda)$; write
$H_\ell = (H_\ell + \lambda) - \lambda$ and collect terms. $\square$

*Heuristic consequence.* When $\lambda \ll H_\ell$ and $G \approx 0$ (the intercept matches the current mean),
a tree's total gain is its Newton loss decrease divided by $\eta(1-\eta/2)$,
which is 10.5 at $\eta = 0.1$. If the Newton model approximates the true loss
change, then over the ensemble the total gain recorded by all
features is approximately $10.5 \times$ the training log-loss removed. That loss
starts at $nH_2(\hat p) \approx n_{\min}\bigl(1 + \ln(1/\hat p)\bigr)$ nats under the prevalence
intercept, where $H_2$ is the binary entropy. In pure-noise data every bit of
loss removed is memorisation. The identity does not say how much is
removed; that is an empirical quantity (Appendix A, P9).

How the memorised gain is divided between features determines the
importances, and we have no theorem for it. We state the claim that the
experiments support.

**Conjecture 2 (the full ensemble).** With the prevalence intercept, the false
negative rate of the one-shot max-shadow rule on a boosted ensemble is, to
first order, a function of $s$ and $K$, and it is at least the root-split
failure probability $1 - \mathbb E[F_0(M_j)^K]$ of Corollary 1.

*Evidence.* In the E2 sweep, $s$ explains 94% of the variance of the
ensemble's FNR (leave-one-out 0.90), $\hat p$ explains none of it, and the
failure is mostly present after three trees (Sections 4.3–4.4). In every
E4 and E8 cell the ensemble's FNR is at or above the root-split prediction.
A residual dependence on $\hat p$ at fixed $n_{\min}$ remains, and we cannot
explain it. One candidate is the $\ln(1/\hat p)$ term in the memorisation budget.
