# Appendix C. Proofs

Notation: $m = n_{\min}$, $\varphi$ and $\Phi$ are the standard normal density and CDF,
$\psi(u) = \varphi(\Phi^{-1}(u))$, and $\|\cdot\|$ is the supremum norm.

## C.1 Proof of Theorem 1

*Coupling.* The claim is about laws, so we may build all sample sizes on one
probability space. Fix the $m$ positives $Z_1, \dots, Z_m$ i.i.d. $N(\delta, 1)$ and an
infinite i.i.d. $N(0,1)$ sequence $\xi_1, \xi_2, \dots$, and let the sample of size $n$
consist of the positives and $\xi_1, \dots, \xi_{n-m}$. Different features use
independent copies, which gives independence across features at every $n$
(features are independent given $Y$ in the model of Section 2). We show
almost-sure convergence, which implies convergence in distribution.

*Rank fractions.* Let $F^-_n$ be the empirical CDF of $\xi_1, \dots, \xi_{n-m}$. The rank
fraction of positive $i$ among all $n$ values is
$\rho_i^n = n^{-1}\bigl[(n-m)F^-_n(Z_i) + \#\{l \le m : Z_l \le Z_i\}\bigr]$, so

$$
\bigl|\rho_i^n - \Phi(Z_i)\bigr| \le \|F^-_n - \Phi\| + \frac{2m}{n} \longrightarrow 0
\quad\text{a.s.}
$$

by the Glivenko–Cantelli theorem. Let $V_i = \Phi(Z_i)$. Almost surely the $V_i$ are
distinct, lie in $(0,1)$, and differ from $a = w/m$ and $1 - a$.

*The counting process.* Exact greedy search evaluates, for $j = 1, \dots, n-1$,
the split that sends the $j$ smallest values left, so $u = j/n$ and
$k_L(j/n) = K_n(j/n)$ with $K_n(u) = \#\{i : \rho_i^n \le u\}$. Fix $\omega$ outside a
null set and $n$ large enough that the $\rho_i^n$ are in the same order as the
$V_i$. Let $\lambda_n$ be the increasing piecewise-linear bijection of $[0,1]$ with
$\lambda_n(0) = 0$, $\lambda_n(1) = 1$ and $\lambda_n(\rho_i^n) = V_i$. Then $K_n = N \circ \lambda_n$ exactly,
where $N(u) = \#\{i : V_i \le u\}$, and $\|\lambda_n - \mathrm{id}\| \le \max_i |\rho_i^n - V_i| \to 0$. The
slopes of $\lambda_n$ tend to 1.

*The bracket.* With the prevalence intercept, $H_L = n_L\,\hat p(1-\hat p) = mu(1-\hat p)$,
so the admissible thresholds are $u \in [a_n, 1 - a_n]$ with
$a_n = w/\bigl(m(1-\hat p)\bigr) \to a$, and the bracket of (3) is
$\beta_n(u) = [mu(1-\hat p)+\lambda]^{-1} + [m(1-u)(1-\hat p)+\lambda]^{-1}$. Since $w > 0$,
$\beta_n \to \beta$ uniformly on $[a/2, 1 - a/2]$, where $\beta$ is the bracket of (4), and
$\beta$ is uniformly continuous there (this also covers $\lambda = 0$).

*Process convergence.* Extend $\Gamma$ by formula (4) to $[a/2, 1-a/2]$. Since
$K_n = N\circ\lambda_n$ with $\|\lambda_n - \mathrm{id}\| \to 0$, the counting processes converge in
$D[0,1]$ with the $J_1$ topology [@billingsley1999convergence]. For the gains, on
$[a_n, 1-a_n]$,
$\operatorname{Gain}_n(u) - \Gamma(\lambda_n(u)) = (N(\lambda_n u) - mu)^2\beta_n(u) - (N(\lambda_n u) - m\lambda_n u)^2\beta(\lambda_n u)$.
Since $0 \le N \le m$, $|mu - m\lambda_n u| \le m\|\lambda_n - \mathrm{id}\|$, $\beta_n \to \beta$ uniformly and $\beta$ is
uniformly continuous, $\sup_{u \in [a_n, 1-a_n]} |\operatorname{Gain}_n(u) - \Gamma(\lambda_n(u))| \to 0$. This is
the sense of convergence in Theorem 1.

*Suprema.* The search maximises over the grid $T_n = \{j/n\} \cap [a_n, 1-a_n]$. Its
image $\lambda_n(T_n)$ has mesh at most $(1+o(1))/n$ and end points tending to $a$ and
$1-a$. $\Gamma$ is continuous except for finitely many jumps at the $V_i$, none
at $a$ or $1-a$.

* Lower bound. The supremum of $\Gamma$ over $[a, 1-a]$ is a value at a continuity
  point, or a one-sided limit at a jump. Every open interval inside
  $[a, 1-a]$ eventually contains points of $\lambda_n(T_n)$, so
  $\liminf_n \max_{T_n}\operatorname{Gain}_n \ge \sup_{[a,1-a]}\Gamma$.
* Upper bound. For any $\eta > 0$, eventually $\lambda_n(T_n) \subset [a-\eta, 1-a+\eta]$, so
  $\limsup_n \max_{T_n}\operatorname{Gain}_n \le \sup_{[a-\eta,\,1-a+\eta]}\Gamma$. Let $\eta \to 0$, using the
  continuity of $\Gamma$ at $a$ and $1-a$.

Hence $\max_{T_n}\operatorname{Gain}_n \to \sup_{[a,1-a]}\Gamma$ almost surely, jointly over the
independently generated features. The law of $\Gamma$ is a function of
$(m, \delta, \lambda, w)$ only. $\square$

*Corollary (histogram search).* Under (H1) the cut $c^n_j$ has
$F_n(c^n_j) \to j/B$, where $F_n$ is the empirical CDF of all $n$ values, so
$k_L$ at that cut is $K_n(F_n(c^n_j)) = N(\lambda_n(F_n(c^n_j))) \to N(j/B)$, because $N$ is
almost surely continuous at $j/B$. The bracket converges as before. Under
(H2) each cut is eventually admissible in the finite problem if and only if
$j/B \in (a, 1-a)$. So the finite maximum over cuts converges to the maximum of
$\Gamma$ over $\{j/B\} \cap [a, 1-a]$. $\square$

## C.2 The limit in Proposition 4

Write $f_\delta(x) = \bigl(\int_0^1 \varphi(x - \theta\delta)\,d\theta\bigr)^2/\bigl(\Phi(x)(1-\Phi(x))\bigr)$, so that
$\nu(u)/(m\delta^2) = f_\delta(\Phi^{-1}(u))$. For $|\delta| \le 1$,
$\varphi(x - \theta\delta) = \varphi(x)\,e^{\theta\delta x - \theta^2\delta^2/2} \le \varphi(x)\,e^{|x|}$, so

$$
f_\delta(x) \le g(x) = \frac{\varphi(x)^2 e^{2|x|}}{\Phi(x)\bigl(1-\Phi(x)\bigr)}.
$$

By symmetry and the Mills-ratio bound $1 - \Phi(x) \ge \varphi(x)\,x/(1+x^2)$ for $x > 0$,
$g(x) \le \varphi(x)e^{2x}(1+x^2)/\bigl(x\,\Phi(x)\bigr) \to 0$ as $|x| \to \infty$. Given $\epsilon > 0$, pick
$X$ with $g < \epsilon$ outside $[-X, X]$. On $[-X, X]$, $f_\delta \to f_0$ uniformly as $\delta \to 0$ by
uniform continuity of $\varphi$. Hence $\sup_x f_\delta(x) \to \sup_x f_0(x)$. The function
$f_0(x) = \varphi(x)^2/\bigl(\Phi(x)(1-\Phi(x))\bigr)$ is uniquely maximised at $x = 0$, with
$f_0(0) = 2/\pi$, by the disc argument in the proof of Proposition 4. Since
$f_0$ is continuous with a unique maximiser and $f_\delta \to f_0$ uniformly on
compacts with the envelope $g$, every maximiser of $f_\delta$ tends to 0. $\square$

## C.3 Proof of Theorem 2

Fix one feature with shift $\delta = \delta_m$ and write $s_m = m\delta_m^2 \to s$. Let
$\xi_i = Z_i - \delta$ and $U_i = \Phi(\xi_i)$, i.i.d. uniform. Then $\Phi(Z_i) \le u$ iff
$U_i \le q_\delta(u) = \Phi(\Phi^{-1}(u) - \delta)$, so $N_\delta(u) = mG_m(q_\delta(u))$ with $G_m$ the empirical
CDF of the $U_i$. With the uniform empirical process
$\alpha_m(t) = \sqrt m\,(G_m(t) - t)$,

$$
\frac{N_\delta(u) - mu}{\sqrt m} = \alpha_m\bigl(q_\delta(u)\bigr) + \sqrt m\,\bigl(q_\delta(u) - u\bigr).
$$

*(a) Drift.* With $x = \Phi^{-1}(u)$, $u - q_\delta(u) = \delta\int_0^1 \varphi(x - \theta\delta)\,d\theta$ and
$|\varphi'| \le \varphi(1)$, so $\bigl|\sqrt m\,(u - q_\delta(u)) - \sqrt m\,\delta\,\psi(u)\bigr| \le \varphi(1)\,s_m/\sqrt m \to 0$
uniformly in $u$. Since $\sqrt m\,\delta_m \to \sqrt s$, the drift converges uniformly to
$-\sqrt s\,\psi(u)$.

*(b) Noise.* $\|q_\delta - \mathrm{id}\| \le \varphi(0)\,\delta \to 0$. The uniform empirical process
converges to a Brownian bridge $\mathbb B$ in $D[0,1]$ and is asymptotically
equicontinuous [@shorack1986empirical;
@billingsley1999convergence], so
$\|\alpha_m \circ q_\delta - \alpha_m\| \to 0$ in probability and $\alpha_m \circ q_\delta \Rightarrow \mathbb B$. Because
the limit is continuous, the convergence holds in the supremum norm (by
Skorokhod representation).

*(c) Bracket.* On $[\varepsilon, 1-\varepsilon]$,
$m\bigl[(mu+\lambda)^{-1} + (m(1-u)+\lambda)^{-1}\bigr] \to 1/\bigl(u(1-u)\bigr)$ uniformly.

Combining (a)–(c) with the continuous mapping theorem,
$\Gamma_{\delta_m} \Rightarrow \Gamma^{(s)}$ uniformly on the window. Independent features give
independent limits.

*(d) Win probability.* The map $f \mapsto \sup_{[\varepsilon,1-\varepsilon]} f$ is continuous for the
supremum norm, so the suprema converge jointly. For $s = 0$,
$\sup\Gamma^{(0)} = \bigl(\sup_u |\mathbb B(u)|/\sqrt{u(1-u)}\bigr)^2$. Apply Tsirelson's theorem
[@tsirelson1975density; @lifshits1995gaussian] to the symmetrised centred
Gaussian family $\{\pm\mathbb B(u)/\sqrt{u(1-u)}\}$: the law of its supremum has no atoms
except possibly at the left end of its support. That end is 0, because small
balls have positive Gaussian measure, and $P(\sup = 0) = 0$. So the null suprema have continuous laws.
As they are independent of the signal supremum, the event of a tie has
probability zero, and the win probability converges by the portmanteau
theorem.

*(e) Monotonicity in $s$.* For $c > 0$ let
$A_c = \{f \in C[\varepsilon, 1-\varepsilon] : |f(u)| \le \sqrt{c\,u(1-u)}\ \text{for all } u\}$, a closed,
convex, symmetric set. Then $P(\sup\Gamma^{(s)} \le c) = P(\mathbb B - \sqrt s\,\psi \in A_c)$. The law
$\gamma$ of $\mathbb B$ on $C[\varepsilon, 1-\varepsilon]$ is a centred Gaussian Radon measure, hence
log-concave [@borell1974convex; @bogachev1998gaussian]. For convex symmetric
$A_c$, $A_c + (\theta t_1 + (1-\theta)t_2)\psi \supseteq \theta(A_c + t_1\psi) + (1-\theta)(A_c + t_2\psi)$, so
$t \mapsto \gamma(A_c + t\psi)$ is log-concave; it is even by symmetry of $\gamma$ and $A_c$,
hence non-increasing for $t \ge 0$ (this is Anderson's inequality
[@anderson1955integral] in infinite dimensions, and needs no condition on $\psi$). So $\sup\Gamma^{(s)}$ is stochastically non-decreasing in
$s$, and since the null suprema are independent of it, $\pi_\varepsilon(s, K)$ is
non-decreasing in $s$. We do not claim strict monotonicity.

*(f) $s = 0$.* The $K+1$ suprema are i.i.d. with a continuous law, so each is
the largest with probability $1/(K+1)$. $\square$
