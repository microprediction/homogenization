# Portfolio theory under fast regime switching: research memo

Drafted 2026-09-23. Scope: prior art on portfolio choice when returns, volatilities or correlations are switched by a
regime or driven by a fast factor; what the site's first-order (Green-Kubo) rule gives for these problems; and
checkable examples. A second part covers switching and stochastic correlation.

Notation follows tools/pages/any-equation.html. The chain has generator Q, stationary law pi and group inverse
Q#. For regime functions f and h,
K(f, h) = -pi . (f Q# h) = int_0^inf Cov_pi(f(y_0), h(y_t)) dt,
with the small parameter (switching time) absorbed into K. gamma is relative risk aversion throughout,
U(x) = x^(1-gamma)/(1-gamma). lambda is the Sharpe ratio, and for several assets lambda^2 = (mu-r)' Sigma^-1 (mu-r).

How to read the status labels:
- "Known" means stated in a paper whose abstract or text we read.
- "Derived here" means a calculation made for this memo. Some of these were checked numerically in a scratch
  script, which is not in the repo; the rest are unchecked.
- "Not read" means the reference is verified (title, first author, year and DOI checked on Crossref), but we
  describe its content only from its title or from how other papers cite it.

---

## 1. Summary

1. **Nobody seems to have computed the Green-Kubo correction for portfolio choice.** The literature treats two
   cases, but not the first correction under a fast chain:
   - An observed regime at ordinary speed, solved exactly: Zhou and Yin (2003), Bäuerle and Rieder (2004),
     Sotomayor and Cadenillas (2009).
   - A fast chain treated by aggregation (Yin, Zhang and coauthors). This proves convergence to a limit problem
     and that controls built from the limit are nearly optimal. From the abstracts we read, it does not give an
     explicit first-order correction.

   The closest asymptotic work is Fouque, Sircar and Zariphopoulou (2017), for a fast mean-reverting diffusion
   factor:
   - Leading order is Merton with the square-averaged Sharpe ratio <lambda^2>.
   - The first correction is of order sqrt(eps) and proportional to the correlation rho_1 between the factor and
     the stock.
   - They do not compute the order-eps term, which is the Green-Kubo term. Their accuracy proof for power utility
     bounds the error by C eps.

   A Markov chain is independent of the price Brownian motion, so the sqrt(eps) term vanishes. The first
   correction is then exactly the order-eps Green-Kubo term.
2. **For CRRA with an observed regime, the problem is linear and the rule applies unchanged.** The HJB factors as
   V_i = x^(1-gamma)/(1-gamma) Psi_i(t), and Psi solves the site's engine system. So:
   - The certainty-equivalent rate is psi_bar + (1-gamma) K(psi, psi), with psi_i = r_i + lambda_i^2/(2 gamma).
   - The optimal weights are exactly myopic in every regime, so they get no correction at any order.
   - All the "operators" are scalars, so they commute and no antisymmetric (cycle-direction) term appears.
3. **The mean-variance frontier of Zhou and Yin (2003) is also driven by a linear ODE system.** Its slope depends
   on beta = E exp(-int lambda^2 dt), and the rule gives beta ≈ exp(-T <lambda^2> + T K(lambda^2, lambda^2)).
   This is derived here and checked numerically.
4. **For a regime-blind investor, the correction is a Green-Kubo penalty that moves the weights.** This investor
   holds a constant fraction and cannot see the regime. The correction involves K_mu,mu, K_mu,s and K_s,s, and
   it shifts the weights at first order (derived here, checked numerically).

   Caveat: an investor who filters the regime from prices gains a learning term of the same order eps. That term
   is not in the rule. Also, in continuous time a switching volatility is revealed by quadratic variation.
5. **Nonlinear HJB.** The expansion still works when the Hamiltonian is nonlinear:
   - Leading order is the average of the optimized Hamiltonians, not the Hamiltonian of averaged parameters.
     This matches FSZ's <lambda^2>.
   - At first order, A_j A_k is replaced by DN_j(v0)[N_k(v0)], where N_k is the (possibly nonlinear) operator that
     the switched parameter multiplies and DN_j is its derivative. The antisymmetric part becomes a Lie bracket of
     the N's.
   - For Merton with switched r and lambda^2, the bracket vanishes, for any utility. So the cycle direction cannot
     matter at first order in the plain Merton problem. It can matter once a slow factor with non-commuting
     dynamics is switched, for example the reversion speed and factor volatility in Kim-Omberg.

   All of point 5 is derived here and unchecked.
6. **Correlation (second part).**
   - Only rho switches, all operators constant-coefficient: they commute. The first-order price is a Hull-White
     style mixing over the integrated covariance matrix, with correction t sum K_ab d^2C/dSigma_a dSigma_b. For
     rho alone this is (K_rho,rho / t) d^2C/drho^2, a "correlation volga" that makes an implied-correlation
     smile.
   - Drift and correlation co-switch: K_mu,c12 produces a third cross-cumulant (co-skewness). This is the
     first-order fingerprint of "correlation rises in down markets" (derived and checked).
   - A fast diffusive correlation factor correlated with price shocks gives an order-sqrt(eps) FPS-type skew
     instead.

   We found no published fast-switching or fast-factor asymptotics for correlation in pricing. The nearest is
   Fouque, Pun and Wong (2016): an ambiguous (worst-case) correlation with fast stochastic volatilities.

---

## 2. Prior art: regime-switching allocation (ordinary speed)

**Zariphopoulou (1992).** Infinite-horizon investment and consumption with transaction fees and Markov-chain
parameters. The value function is shown to be the unique viscosity solution of a system of variational
inequalities. The paper is about existence and characterization; it has no asymptotics. (Known; abstract read.)

**Zhou and Yin (2003).** Continuous-time Markowitz mean-variance problem with rate, drifts and volatilities
switched by a chain independent of the Brownian motion, solved by stochastic LQ control.
- The efficient frontier is given in closed form "based on solutions of two systems of linear ordinary
  differential equations".
- With a deterministic interest rate, the results look like the no-switching case.

(Known; abstract read.) This linearity is what lets the Green-Kubo rule apply directly (Section 4.3).

**Yin and Zhou (2004).** Discrete-time version of the same problem and its continuous-time limit. It also sets up
a two-time-scale formulation: a large chain is aggregated into a smaller one, the limit is shown to be a switching
diffusion by weak convergence, and strategies built from the limit are asymptotically optimal. (Known; abstract
read.)

**Bäuerle and Rieder (2004).** Utility of terminal wealth with rate, drift and volatility modulated by an
observable chain. Explicit solutions allow a comparison with a market using constant (average) data. (Known;
abstract read.) This is the ordinary-speed version of our item 2. Our first-order formula quantifies their
comparison when switching is fast.

**Rieder and Bäuerle (2005).** Unobservable Markov-modulated drift. Filtering reduces the problem to full
observation, and it is solved explicitly for log and power utility. (Known; abstract read.)

**Sass and Haussmann (2004).** Terminal wealth under partial information, with the drift a continuous-time
Markov chain. (Not read.)

**Honda (2003).** Unobservable, regime-switching mean returns. (Not read.)

**Elliott, Siu and Badescu (2010).** Mean-variance selection under a hidden Markov regime model. (Not read.)

**Sotomayor and Cadenillas (2009).** Consumption and investment with an observable regime, including
regime-dependent utility. Explicit solutions for specific HARA utilities.
- The risky proportion is larger in a bull regime than a bear regime, whatever the risk preference.
- The consumption-wealth ratio depends on regime and risk tolerance.

(Known; abstract read.) With consumption, the CRRA reduction is no longer linear (Section 4.2).

**Bäuerle and Rieder (2011).** Book: Markov decision processes with finance applications, including partially
observable problems. (Not read in detail.)

**Empirical allocation.**
- Ang and Bekaert (2002) solve a CRRA investor's international allocation with a regime-switching model. In the
  working-paper version they find a high-volatility regime with higher correlations and lower means. Ignoring
  regimes costs little for moderate risk aversion, and hedging demands from time-varying correlations are
  negligible. (Abstract of NBER WP 7056 read. That title, "International asset allocation with time-varying
  correlations", differs from the published one; we take it to be the earlier version.)
- Ang and Bekaert (2004, FAJ): regime strategies add most value when choosing among cash, bonds and equity.
  (Abstract of NBER WP 10080 read.)
- Guidolin and Timmermann (2007, 2008): multivariate regime switching in allocation. (Not read.)
- Tu (2010) reports certainty-equivalent losses above 2% a year from ignoring regimes. (Known; abstract read.)
- Ang and Timmermann (2012) survey. (Not read.)

None of these works is asymptotic in the switching rate.

---

## 3. Prior art: fast factors and asymptotic expansions

**Fouque, Sircar and Zariphopoulou (2017, Math. Finance 27(3) 704-745; online 2015).** Read in full from the
author-posted PDF on Fouque's UCSB page. Merton problem with a fast mean-reverting factor Y (scale eps) and a slow
factor Z (scale delta).
- **Leading order.** v0 = M(t, x; lambda_bar), the constant-parameter Merton value, with the square-averaged
  Sharpe ratio lambda_bar^2 = <mu^2/sigma^2>.
- **First correction** (their Proposition 2.7): sqrt(eps) v1 with v1 = -(T-t) rho_1 B D_1^2 v0, where
  B = <lambda a theta'> and D_k = R^k d^k/dx^k. It is proportional to the correlation rho_1 between the fast
  factor and the stock.
- **Structure.** "Remarkable similarities with the linear option pricing problem", through the risk-tolerance
  operators D_k.
- **Zeroth-order strategy.** pi0 = (lambda(y)/sigma(y)) R(t, x; lambda_bar) already achieves the value to order
  sqrt(eps).
- **"Practical" regime-blind strategy** (Section 5 and Appendix C). The best Y-independent strategy uses
  mu_bar = <mu> and sigma_bar^2 = <sigma^2>. Its leading-order loss is set by the Cauchy-Schwarz gap
  <mu^2/sigma^2> >= mu_bar^2/sigma_bar^2.
- **Power utility.** In their notation (RRA gamma) the value has distortion form V = x^(1-gamma)/(1-gamma) Psi^q,
  with q = gamma/(gamma + (1-gamma) rho^2), and Psi solves a linear PDE (their eq. 77-79, attributed to
  Zariphopoulou 2001). With rho = 0, q = 1 and the value is linear in Psi.
- **Proof of accuracy** (their Theorem 6.2): error at most C eps. The order-eps average term (the Green-Kubo term)
  is not computed.

Weights (power utility): pi = lambda/(gamma sigma) + sqrt(eps) rho_1 (...) + ..., so the correction to the
weights is also proportional to rho_1.

**Other fast-factor portfolio papers.**
- Fouque and Hu (2017, SICON): slowly varying environment, general utility. A zeroth-order strategy is
  asymptotically optimal within a class. (Abstract read.)
- Fouque and Hu (2020, MMS): fast and slow factors together, first-order value approximation, optimality of the
  zeroth-order strategy within a class. It notes that a pure PDE approach fails in the multiscale case. (Abstract
  read.)
- Fouque and Hu, fractional and rough environments: SIFIN 2018, Appl. Math. Finance 2018, Math. Finance 2019.
  (Verified; not read.)
- Jonsson and Sircar (2002): "Optimal investment problems and volatility homogenization approximations", a book
  chapter and an early use of homogenization for investment. (Not read.)
- The monograph by Fouque, Papanicolaou, Sircar and Sølna (2011) covers the pricing side and has a
  portfolio-optimization chapter. (Blurb read.)

**Exact affine and quadratic solutions**, which are what the rule should reproduce in its limits:
- Kim and Omberg (1996): OU risk premium, HARA utility, nonmyopic demand. (Abstract read.)
- Wachter (2002): mean-reverting returns, complete market. (Not read.)
- Chacko and Viceira (2005): stochastic volatility (precision) with recursive utility. Hedging demand is negative
  when risk aversion exceeds one and the volatility-return correlation is negative, and small in US data.
  (Abstract read.)
- Liu (2007): quadratic returns and CRRA, solved up to an ODE, including Heston. (Abstract read.)
- Merton (1969, 1971): the classical problem.

---

## 4. The first-order rule applied to Merton and mean-variance

### 4.1 Nonlinear Hamiltonians: what changes

Write the per-regime HJB in backward form as v_t + H_y(v) = 0, with
H_y(v) = H_bar(v) + sum_k phi_k(y) N_k(v).
Here N_k is the (possibly nonlinear) operator that the switched parameter multiplies. For Merton with the regime
observed:
- N_lambda(v) = -½ v_x^2 / v_xx is the optimized Sharpe term;
- N_r(v) = x v_x is the rate term.

Expanding with Q/eps as on the site (derived here, formal):

- **Order eps^-1.** v0 does not depend on the regime.
- **Order 1.** Solvability gives v0_t + H_bar(v0) = 0, where H_bar = sum_y pi_y H_y is the average of the
  optimized Hamiltonians. For Merton this is the square-averaged Sharpe ratio of FSZ, not Merton with averaged
  mu and sigma. The regime memory is v1_y = -sum_k (Q# phi_k)_y N_k(v0).
- **Order eps.** The average correction solves
  (v1_bar)_t + DH_bar(v0)[v1_bar] + sum_jk K_jk DN_j(v0)[N_k(v0)] = 0,  v1_bar(T) = 0.
  So A_j A_k becomes DN_j(v0)[N_k(v0)], and the correction is propagated by the linearized averaged HJB. That
  linearization is FSZ's operator L_t,x(lambda_bar) = d_t + ½ lambda_bar^2 D_2 + lambda_bar^2 D_1.
- **Antisymmetric part.** It enters through DN_j[N_k] - DN_k[N_j], the Lie bracket of the two operators viewed as
  vector fields on functions. For linear operators this is the commutator.
- **Merton bracket.** For N_r and N_lambda above, a direct calculation shows the bracket is zero, because
  N_lambda is invariant under scaling of x. So in Merton's problem with switched (r, mu, sigma), for any
  utility, only K_sym enters at first order. The cycle direction cannot matter.
- **General utility.** The source is K_lambda2,lambda2 (D_1 + ½ D_2)(½ D_1 v0), in FSZ's operators with
  lambda_bar. The first-order weights also move, through the memory term v1_y. The explicit solution for
  v1_bar is open: FSZ's Lemma 2.5 gives [L, D_1] = 0, but we did not check whether D_2 D_1 commutes with L.
  For power utility the formula reduces to item 4.2.

Caveat: this is a formal expansion. A proof would follow FSZ Section 6.3 (power utility) or the aggregation
arguments of Yin and Zhang.

### 4.2 CRRA, observed regime (linear, exact reduction)

HJB: V_t + max_pi[...] + sum_j Q_ij V_j = 0. Try V_i = x^(1-gamma)/(1-gamma) Psi_i(t). Then
Psi_t + (Q + diag((1-gamma) psi)) Psi = 0,  Psi(T) = 1,  psi_i = r_i + lambda_i^2/(2 gamma).
This is the engine system of the yield-curve page with forcing (1-gamma) psi. It is standard: Bäuerle and Rieder
(2004) solve this case explicitly. The rule gives (derived here):
- **Certainty-equivalent rate** (regime drawn from pi): CE ≈ psi_bar + (1-gamma) K(psi, psi).
  - gamma > 1: fluctuating opportunities lower the CE.
  - gamma < 1: they raise it.
  - Log utility: no effect.
  - Checked: the three-state chain Q_A (x10, x40), T = 5 and gamma = 3 match to 1e-6.
- **Known starting regime i.** Add the memory term: log Psi_i ≈ log(pi.Psi) - (1-gamma)(Q# psi~)_i. For a single
  mode with time constant tau this is +(1-gamma) tau psi~_i.
- **Weights.** pi_i* = Sigma_i^-1 (mu_i - r_i)/gamma exactly, so there is no correction. The chain is independent
  of W and Psi does not depend on x, so there is no hedging demand.
- **Symmetry.** Only K_sym enters, because all operators are scalars.

**Consumption breaks the linearity.** Here Psi_i^(1/gamma) enters the consumption term, so the reduced system is
nonlinear (Sotomayor and Cadenillas solve it explicitly for specific HARA cases). Item 4.1 applies with N_c the
consumption term.

**Correlated factor.** Zariphopoulou's distortion Psi^q with q ≠ 1 linearizes a single diffusion factor but not
the chain coupling: the coupling becomes sum_j Q_ij Psi_j^q. So a chain combined with a correlated slow factor is
genuinely nonlinear, and item 4.1 is the tool.

### 4.3 Mean-variance frontier (linear)

For wealth dx = [r x + b'u] dt + u' sigma dW with regime-switched (b, sigma) and constant r, the LQ value is
P_i(t)(x - h(t))^2 with h(t) = d e^{-r(T-t)}, where d is the target that fixes the point on the frontier. P solves
P' + (2r - lambda_i^2) P + (Q P)_i = 0,  P(T) = 1.
We derived this; it is consistent with the "two systems of linear ODEs" of Zhou and Yin (2003). The frontier then
has the no-switching shape
Var x_T = [beta/(1 - beta)] (E x_T - x0 e^{rT})^2,  beta = E_i0 exp(-int_0^T lambda^2(y_s) ds),
which matches Zhou and Yin's remark about deterministic rates. The exact frontier formula should be compared with
their theorem before the site uses it.

The rule gives (derived here):
beta ≈ exp(-T <lambda^2> + T K(lambda^2, lambda^2)),
plus a memory factor for a known start. So to first order the frontier sees only <lambda^2> and K(lambda^2,
lambda^2), the same kind of identification as on the yield-curve page. Checked with Q_A x10: beta 0.57861 exact
against 0.57863 from the rule (0.57690 averaged).

### 4.4 Regime-blind investor, constant proportion

The investor does not see the regime and holds a constant fraction pi. Write s = sigma^2. The CE rate in regime y
is h_y(pi) = r + pi(mu_y - r) - ½ gamma pi^2 s_y, and E X_T^(1-gamma) = E exp((1-gamma) int h). The rule gives
(derived here):
CE(pi) ≈ h_bar(pi) + (1-gamma)[pi^2 K_mu,mu - gamma pi^3 K_sym_mu,s + ¼ gamma^2 pi^4 K_s,s].
- Checked: Q_A x10 and x40 with gamma = 3 match to about 3e-7 or better.
- **Leading weight.** pi0 = (mu_bar - r)/(gamma s_bar), the FSZ practical strategy.
- **First correction.**
  delta_pi = (1-gamma)[2 pi0 K_mu,mu - 3 gamma pi0^2 K_sym_mu,s + gamma^2 pi0^3 K_s,s] / (gamma s_bar).
- **Effect for gamma > 1.** Persistent drift regimes act like extra return variance of about 2 K_mu,mu, which
  shrinks positions. A negative mu-s covariance (bear regimes more volatile) shrinks them further.

Caveats:
- **Learning.** An investor who filters the regime from returns gains an amount of the same order. A Kalman-type
  heuristic gives Var(mu_hat) ≈ eps v^2/(2 sigma^2) for a drift fluctuation of variance v. This term is outside
  the rule and would need a separate derivation, e.g. from Rieder and Bäuerle (2005) or the Wonham filter.
- **Visible volatility.** A switching volatility is visible in continuous time through quadratic variation. The
  regime-blind case with s switching is realistic only with discrete rebalancing or as FSZ's "practical"
  restriction.

### 4.5 Where the antisymmetric term can appear

It needs non-commuting operators acting on a slow state. Example: Kim-Omberg with an OU Sharpe factor z,
dz = kappa(theta - z) dt + beta dB independent of W (so q = 1 and the problem is linear), with kappa and beta^2
switched by an observed fast chain.
- The linear equation for Psi contains A_kappa = (theta - z) d_z and A_beta2 = ½ d_zz. Their commutator is
  [A_kappa, A_beta2] = d_zz, as on the Vasicek page.
- So 2 K_anti_kappa,beta2 acts as extra factor variance and changes the value, with the weights still myopic
  because rho = 0.
- With rho ≠ 0 the weights carry a hedging term and the problem is nonlinear (item 4.1).

A regime-dependent payoff or running source opens the other route, as noted on the site. Examples: a
regime-dependent utility (Sotomayor and Cadenillas) or a regime-dependent consumption weight.

---

## 5. Correlation: switching, stochastic and asymmetric

### 5.1 Prior art

**Correlations rise in bear markets.**
- Longin and Solnik (2001): with extreme-value methods, correlation is linked to the market trend, not to
  volatility as such. It rises in bear markets but not bull markets. (Abstract read.)
- Ang and Chen (2002): asymmetric correlations of US equity portfolios. (Not read. It is commonly cited as finding
  that regime-switching models fit the asymmetry better than GARCH-type models, but we have not confirmed this.)
- Hong, Tu and Zhou (2007): a model-free test for asymmetric correlation, beta and covariance. They find strong
  asymmetry for size and momentum portfolios and value to a disappointment-averse investor. (Abstract read.)
- Chesnay and Jondeau (2001): a multivariate Markov-switching model in which correlations rose significantly in
  turbulent regimes (S&P, DAX, FTSE, weekly 1988-99). (Abstract read.)
- Ang and Bekaert (2002, 2004): a high-volatility regime with higher correlations and lower means, which is the
  "bear regime" of co-switching. (Abstracts of the NBER versions read.)
- Okimoto (2008), regime-switching copulas, and Cappiello, Engle and Sheppard (2006), asymmetric DCC. (Not read.)
- Pelletier (2006), regime switching in correlations. (Not read.)
- Engle (2002), DCC. (Verified; not read.)

**Correlation risk in portfolios and prices.**
- Buraschi, Porchia and Trojani (2010): intertemporal portfolio choice with stochastic correlation (a Wishart
  covariance). Optimal portfolios hedge volatility and correlation separately, and the covariance hedge grows
  with persistence, leverage effects, dimension and constraints. (Abstract read.)
- Driessen, Maenhout and Vilkov (2009): correlation risk is priced in S&P 100 index against component options.
  (Abstract read.)
- Bäuerle and Li (2013): power and log utility with Wishart volatility, explicit when the drift is linear in the
  covariance. (Abstract read.)
- Branger, Muck, Seifried and Weisheit (2017): jumps in variances and covariances. (Not read.)

**Wishart models.**
- Bru (1991), the Wishart process. (Not read.)
- Gourieroux, Jasiak and Sufana (2009), the Wishart autoregressive process. (Not read.)
- Gourieroux and Sufana (2010), derivative pricing. (Not read.)
- Da Fonseca, Grasselli and Tebaldi (2007, RDR), options with stochastic correlation. (Not read.)
- Da Fonseca, Grasselli and Tebaldi (2008, QF): a Wishart multifactor Heston model, priced by FFT, with a factor
  for stochastic leverage. (Abstract read.)

**Asymptotics with correlation.**
- Fouque, Pun and Wong (2016): ambiguous (worst-case) correlation between two assets with fast mean-reverting
  stochastic volatilities, with an asymptotic closed form for general utilities. (Abstract read.) This is the only
  fast-scale correlation asymptotics we found, and it treats correlation as ambiguous rather than switching.
- Deelstra and Simon (2017): multivariate options in Markov-modulated Lévy models. (Not read.)
- Carmona and Durrleman (2003): spread options. (Not read.)
- We searched for fast-switching or fast mean-reverting stochastic correlation expansions (FPS-type) for spread
  or basket options and found none. This is not proof of absence.

### 5.2 Rule: only rho switches (co-switching variances allowed)

Take multi-asset Black-Scholes in log prices under the pricing measure, where drift is r in every regime. Switch
the covariance entries c_ab(y) (c_12 = rho sigma_1 sigma_2).
- **Operators.** A_c12 = d_12 and A_caa = ½(d_aa - d_a). They have constant coefficients, so they commute and
  only K_sym enters.
- **Mixing formula.** Each A equals the derivative of the price with respect to the integrated covariance entry.
  So the rule gives
  C ≈ C(Sigma_bar) + t sum_ab K_ab d^2C/dSigma_a dSigma_b.
  This is Hull-White mixing over the integrated covariance matrix, whose covariance is 2tK. (Derived here.)
- **Only rho switching:**
  C ≈ C(rho_bar) + (K_rho,rho / t) d^2C/drho^2.
  This is a "correlation volga". It changes sign across strikes, so it produces an implied-correlation smile that
  no single rho absorbs.
- **Exchange (Margrabe) option.** It depends only on the spread variance s = c11 + c22 - 2 c12, so the rule
  reduces exactly to the site's single-asset smile in the ratio S1/S2, with
  K_s,s = K(c11~ + c22~ - 2 c12~, same). When only rho switches, K_s,s = 4 sigma_1^2 sigma_2^2 K_rho,rho.
- **Weights, observed regime (CRRA).** Replace lambda^2 by the multi-asset lambda_y^2 = theta' Sigma_y^-1 theta
  in item 4.2. For two symmetric assets, lambda^2 = 2 l^2/(1 + rho). CE = psi_bar + (1-gamma) K(psi, psi), and
  the weights are myopic per regime.
- **Weights, regime-blind.** h_y = r + pi' theta - gamma pi_1 pi_2 sigma_1 sigma_2 rho_y + ... gives the extra
  term (1-gamma) gamma^2 pi_1^2 pi_2^2 sigma_1^2 sigma_2^2 K_rho,rho. For gamma > 1 this penalizes any two-asset
  position quartically. Log utility is unaffected.

### 5.3 Co-switching drift and correlation (route 1)

Under the physical measure, switch mu and c12 together. Then K_mu_k,c12 couples d_k with d_12 and adds the
term 2 K_sym_mu_k,c12 d_k d_12. Its first-order fingerprint in the joint return law is a third cross-cumulant
(derived here):
kappa_112(t) = 2t [2 K_sym(mu_1, c12) + K_sym(mu_2, c11)],
and symmetrically for kappa_122.
- **Sign.** When the bear regime has low mean and high correlation, K(mu, c12) < 0, which gives negative
  co-skewness.
- **Check.** Three-state chain, Q_A x20: exact -9.46e-5, rule -9.59e-5. At x80: -2.392e-5 against -2.397e-5.
  The exact values come from finite differences of the exact Markov-modulated cumulant generating function.
- **Pricing.** Under the risk-neutral measure the drift is r, so this term matters for allocation, density
  forecasting and the likelihood, not pricing. The exception is a regime risk premium that changes Q.
  K_c11,c12 and K_c22,c12 enter if volatilities co-switch.
- **Allocation, regime-blind.** h~ = pi' mu~ - gamma pi_1 pi_2 sigma_1 sigma_2 rho~. The cross term
  -2 gamma pi_1 pi_2 sigma_1 sigma_2 sum_k pi_k K_sym(mu_k, rho), times (1-gamma), penalizes long-long positions
  for gamma > 1 beyond the effect of the average correlation. This is a first-order, closed-form version of the
  Ang-Bekaert question.
- **Literature.** Co-switching (bear regime: low mean, high correlation, high volatility) is well supported
  empirically (Ang and Bekaert; Chesnay and Jondeau; Longin and Solnik on the direction). We found no first-order
  fast-switching treatment of its effect.

### 5.4 Leverage-type stochastic correlation (route 2)

Let rho = rho(Y), with Y a fast OU factor whose Brownian motion is correlated with W_1 and W_2 (coefficients
rho_1Y and rho_2Y). The FPS expansion then gives an order-sqrt(eps) correction:
- With L_0 phi = rho(y) - rho_bar, the source is <L_1 L_0^-1 (L_2 - <L_2>)>, where
  L_1 = sum_k rho_kY a(y) sigma_k d_y d_k.
- This gives P1 = (T-t) sum_k V_k sigma_1 sigma_2 d_k d_12 P0 with V_k ∝ rho_kY sigma_k <a phi'>.
  (Structure derived here; signs and constants unchecked.)
- These are odd, third-order terms, so they skew the implied correlation across strikes.
- A Markov chain independent of W cannot produce this term. It needs a diffusive factor, or jumps in prices at
  regime switches.
- The empirical relevance (index against component option skews) is suggested by Driessen, Maenhout and Vilkov.
  Beyond Fouque, Pun and Wong, we found no FPS-type fast correlation expansion.

---

## 6. Checkable examples for the site

1. **Merton with an observed fast regime (CRRA).**
   - Model: three regimes switching (r, mu, sigma), chain Q_A.
   - Formula: CE = psi_bar + (1-gamma) K(psi, psi), with a memory term for a known start; weights exactly
     myopic.
   - Check: exact via expm of Q + diag((1-gamma) psi). Show the error falling fourfold faster when the chain is
     sped up fourfold.
   - Note that the effect vanishes for log utility.
2. **Regime-blind constant-proportion investor.**
   - Formula: CE(pi) and delta_pi from item 4.4.
   - Check: exact CE via expm for a grid of pi, then compare the numerical argmax with pi0 + delta_pi.
   - Pair with FSZ's Cauchy-Schwarz gap at leading order.
3. **Mean-variance frontier.**
   - Formula: beta ≈ exp(-T <lambda^2> + T K(lambda^2, lambda^2)).
   - Check: exact beta from the linear system; Monte Carlo of the Zhou-Yin optimal policy to confirm Var and E.
   - Identification: two chains with the same <lambda^2> and K give the same frontier to first order.
4. **Kim-Omberg with switched kappa and beta^2 (the antisymmetric term).**
   - Model: rho = 0, an observed chain cycling one way against the other way, with the same averages and K_sym.
   - Formula: the effective factor variance shifts by 2 K_anti.
   - Check: finite differences in z for the coupled linear system, as on the Vasicek page.
5. **Correlation smile.**
   - Margrabe exchange option with switched rho: this reuses the smile code in the ratio S1/S2.
   - Spread option with K ≠ 0: C(rho_bar) + (K_rho,rho/T) d^2C/drho^2. Since the operators commute, the exact
     price is E[C(rho_hat_T)] with rho_hat the time-average of rho. Compute it from the exact occupation-time
     law (two states) or by Monte Carlo over chain paths only, with C by one-dimensional Gauss-Hermite.
6. **Downside correlation (route 1).**
   - Formula: kappa_112 = 2t[2 K(mu_1, c12) + K(mu_2, c11)].
   - Check: already done in a scratch script, by finite differences of the exact cumulant generating function.
     A likelihood-page analogue would add a He_{1,1,2}-type term to the bivariate transition density.
7. **Leverage-type correlation (route 2).**
   - Derive the sqrt(eps) coefficient carefully.
   - Check against Monte Carlo of the fast OU correlation factor with correlated shocks, at two values of eps.
     This is the least developed item.

---

## 7. References (verified on Crossref: title, first author, year, DOI)

"Read" means the abstract (or the paper) was read. "Not read" means we verified the metadata only.

Regime switching and allocation
- Zariphopoulou, T. (1992). Investment-consumption models with transaction fees and Markov-chain parameters.
  SIAM J. Control Optim. 30, 613-636. doi:10.1137/0330035. Read.
- Zariphopoulou, T. (2001). A solution approach to valuation with unhedgeable risks. Finance Stoch. 5, 61-82.
  doi:10.1007/pl00000040. Not read; distortion transform as stated in FSZ 2017.
- Zhou, X.Y., Yin, G. (2003). Markowitz's mean-variance portfolio selection with regime switching: a
  continuous-time model. SIAM J. Control Optim. 42, 1466-1482. doi:10.1137/s0363012902405583. Read.
- Yin, G., Zhou, X.Y. (2004). Markowitz's mean-variance portfolio selection with regime switching: from
  discrete-time models to their continuous-time limits. IEEE TAC 49, 349-360. doi:10.1109/tac.2004.824479. Read.
- Bäuerle, N., Rieder, U. (2004). Portfolio optimization with Markov-modulated stock prices and interest rates.
  IEEE TAC 49, 442-447. doi:10.1109/tac.2004.824471. Read.
- Rieder, U., Bäuerle, N. (2005). Portfolio optimization with unobservable Markov-modulated drift process.
  J. Appl. Probab. 42, 362-378. doi:10.1239/jap/1118777176. Read.
- Sass, J., Haussmann, U.G. (2004). Optimizing the terminal wealth under partial information: the drift process
  as a continuous time Markov chain. Finance Stoch. 8. doi:10.1007/s00780-004-0132-9. Not read.
- Honda, T. (2003). Optimal portfolio choice for unobservable and regime-switching mean returns. JEDC 28, 45-78.
  doi:10.1016/s0165-1889(02)00106-9. Not read.
- Elliott, R.J., Siu, T.K., Badescu, A. (2010). On mean-variance portfolio selection under a hidden Markovian
  regime-switching model. Economic Modelling 27, 678-686. doi:10.1016/j.econmod.2010.01.007. Not read.
- Sotomayor, L.R., Cadenillas, A. (2009). Explicit solutions of consumption-investment problems in financial
  markets with regime switching. Math. Finance 19, 251-279. doi:10.1111/j.1467-9965.2009.00366.x. Read.
- Bäuerle, N., Rieder, U. (2011). Markov Decision Processes with Applications to Finance. Springer Universitext.
  doi:10.1007/978-3-642-18324-9. Not read.
- Ang, A., Bekaert, G. (2002). International asset allocation with regime shifts. RFS 15, 1137-1187.
  doi:10.1093/rfs/15.4.1137. Abstract of NBER WP 7056 (doi:10.3386/w7056) read.
- Ang, A., Bekaert, G. (2004). How regimes affect asset allocation. Financial Analysts Journal 60(2), 86-99.
  doi:10.2469/faj.v60.n2.2612. Abstract of NBER WP 10080 read.
- Guidolin, M., Timmermann, A. (2007). Asset allocation under multivariate regime switching. JEDC 31, 3503-3544.
  doi:10.1016/j.jedc.2006.12.004. Not read.
- Guidolin, M., Timmermann, A. (2008). International asset allocation under regime switching, skew, and kurtosis
  preferences. RFS 21, 889-935. doi:10.1093/rfs/hhn006. Not read.
- Tu, J. (2010). Is regime switching in stock returns important in portfolio decisions? Management Sci. 56,
  1198-1215. doi:10.1287/mnsc.1100.1181. Read.
- Ang, A., Timmermann, A. (2012). Regime changes and financial markets. Annu. Rev. Financ. Econ. 4, 313-337.
  doi:10.1146/annurev-financial-110311-101808. Not read.

Two-time-scale (fast chain) control
- Yin, G., Zhang, Q. (1998; 2nd ed. 2013). Continuous-Time Markov Chains and Applications: A Two-Time-Scale
  Approach. Springer. doi:10.1007/978-1-4612-0627-9 and doi:10.1007/978-1-4614-4346-9. Not read.
- Zhang, Q., Yin, G. (2004). Nearly-optimal asset allocation in hybrid stock investment models. JOTA 121,
  419-444. doi:10.1023/b:jota.0000037412.23243.6c. Not read; the abstract of the companion CDC paper (Zhang and
  Yin 2002, "Hybrid stock-investment models and asset allocation", doi:10.1109/cdc.2002.1184525) was read. It
  aggregates each weakly irreducible class of a large chain, obtains a limit switching diffusion, and derives
  nearly optimal allocations.
- Liu, Y.J., Yin, G., Zhou, X.Y. (2005). Near-optimal controls of random-switching LQ problems with indefinite
  control weight costs. Automatica 41, 1063-1070. doi:10.1016/j.automatica.2005.01.002. Not read.
- Yin, G., Talafha, Y. (2012). Mean-variance portfolio selection under regime-switching diffusion asset models:
  a two-time-scale limit. In Advances in Statistics, Probability and Actuarial Science, World Scientific,
  375-390. doi:10.1142/9789814383318_0016. Not read.
- Yang, Z., Yin, G., Wang, L.Y., Zhang, H. (2013). Near-optimal mean-variance controls under two-time-scale
  formulations and applications. Stochastics 85, 723-741. doi:10.1080/17442508.2013.795567. Read: a limit
  problem as eps -> 0, and controls built from it are nearly optimal. No explicit correction is mentioned.
- Badowski, G., Yin, G., Zhang, Q. (2003). Near-optimal controls of discrete-time dynamic systems driven by
  singularly-perturbed Markov chains. JOTA 116, 131-166. doi:10.1023/a:1022166304069. Not read.

Fast factors and asymptotics
- Fouque, J.-P., Sircar, R., Zariphopoulou, T. (2017). Portfolio optimization and stochastic volatility
  asymptotics. Math. Finance 27(3), 704-745 (online 2015). doi:10.1111/mafi.12109. Read (author PDF).
- Fouque, J.-P., Hu, R. (2017). Asymptotic optimal strategy for portfolio optimization in a slowly varying
  stochastic environment. SIAM J. Control Optim. 55, 1990-2023. doi:10.1137/16m1066762. Read.
- Fouque, J.-P., Hu, R. (2020). Multiscale asymptotic analysis for portfolio optimization under stochastic
  environment. Multiscale Model. Simul. 18, 1318-1342. doi:10.1137/19m1245967. Read.
- Fouque, J.-P., Hu, R. (2018). Optimal portfolio under fast mean-reverting fractional stochastic environment.
  SIAM J. Financial Math. 9, 564-601. doi:10.1137/17m1134068. Not read.
- Fouque, J.-P., Hu, R. (2018). Portfolio optimization under fast mean-reverting and rough fractional stochastic
  environment. Appl. Math. Finance 25, 361-388. doi:10.1080/1350486x.2019.1584532. Not read.
- Fouque, J.-P., Hu, R. (2019). Optimal portfolio under fractional stochastic environment. Math. Finance 29,
  697-734. doi:10.1111/mafi.12195. Not read.
- Jonsson, M., Sircar, R. (2002). Optimal investment problems and volatility homogenization approximations. In
  Modern Methods in Scientific Computing and Applications, Springer, 255-281. doi:10.1007/978-94-010-0510-4_7.
  Not read.
- Fouque, J.-P., Papanicolaou, G., Sircar, R., Sølna, K. (2011). Multiscale Stochastic Volatility for Equity,
  Interest Rate, and Credit Derivatives. CUP. doi:10.1017/cbo9781139020534. Blurb read.
- Kim, T.S., Omberg, E. (1996). Dynamic nonmyopic portfolio behavior. RFS 9, 141-161. doi:10.1093/rfs/9.1.141.
  Read.
- Wachter, J.A. (2002). Portfolio and consumption decisions under mean-reverting returns: an exact solution for
  complete markets. JFQA 37, 63-91. doi:10.2307/3594995. Not read.
- Chacko, G., Viceira, L.M. (2005). Dynamic consumption and portfolio choice with stochastic volatility in
  incomplete markets. RFS 18, 1369-1402. doi:10.1093/rfs/hhi035. Read.
- Liu, J. (2007). Portfolio selection in stochastic environments. RFS 20, 1-39. doi:10.1093/rfs/hhl001. Read.
- Merton, R.C. (1969). Lifetime portfolio selection under uncertainty: the continuous-time case. Rev. Econ.
  Stat. 51, 247-257. doi:10.2307/1926560. Not read.
- Merton, R.C. (1971). Optimum consumption and portfolio rules in a continuous-time model. JET 3, 373-413.
  doi:10.1016/0022-0531(71)90038-x. Not read.

Correlation
- Longin, F., Solnik, B. (2001). Extreme correlation of international equity markets. J. Finance 56, 649-676.
  doi:10.1111/0022-1082.00340. Read.
- Ang, A., Chen, J. (2002). Asymmetric correlations of equity portfolios. JFE 63, 443-494.
  doi:10.1016/s0304-405x(02)00068-5. Not read.
- Hong, Y., Tu, J., Zhou, G. (2007). Asymmetries in stock returns: statistical tests and economic evaluation.
  RFS 20, 1547-1581 (online 2006). doi:10.1093/rfs/hhl037. Read.
- Chesnay, F., Jondeau, E. (2001). Does correlation between stock returns really increase during turbulent
  periods? Economic Notes 30, 53-80. doi:10.1111/1468-0300.00047. Read.
- Okimoto, T. (2008). New evidence of asymmetric dependence structures in international equity markets. JFQA
  43, 787-815. doi:10.1017/s0022109000004294. Not read.
- Cappiello, L., Engle, R.F., Sheppard, K. (2006). Asymmetric dynamics in the correlations of global equity and
  bond returns. J. Financial Econometrics 4, 537-572. doi:10.1093/jjfinec/nbl005. Not read.
- Pelletier, D. (2006). Regime switching for dynamic correlations. J. Econometrics 131, 445-473.
  doi:10.1016/j.jeconom.2005.01.013. Not read.
- Engle, R. (2002). Dynamic conditional correlation. JBES 20, 339-350. doi:10.1198/073500102288618487.
  Not read.
- Buraschi, A., Porchia, P., Trojani, F. (2010). Correlation risk and optimal portfolio choice. J. Finance 65,
  393-420. doi:10.1111/j.1540-6261.2009.01533.x. Read.
- Driessen, J., Maenhout, P.J., Vilkov, G. (2009). The price of correlation risk: evidence from equity options.
  J. Finance 64, 1377-1406. doi:10.1111/j.1540-6261.2009.01467.x. Read.
- Bäuerle, N., Li, Z. (2013). Optimal portfolios for financial markets with Wishart volatility. J. Appl. Probab.
  50, 1025-1043. doi:10.1239/jap/1389370097. Read.
- Branger, N., Muck, M., Seifried, F.T., Weisheit, S. (2017). Optimal portfolios when variances and covariances
  can jump. JEDC 85, 59-89. doi:10.1016/j.jedc.2017.09.008. Not read.
- Bru, M.-F. (1991). Wishart processes. J. Theoret. Probab. 4, 725-751. doi:10.1007/bf01259552. Not read.
- Gourieroux, C., Jasiak, J., Sufana, R. (2009). The Wishart autoregressive process of multivariate stochastic
  volatility. J. Econometrics 150, 167-181. doi:10.1016/j.jeconom.2008.12.016. Not read.
- Gourieroux, C., Sufana, R. (2010). Derivative pricing with Wishart multivariate stochastic volatility. JBES 28,
  438-451. doi:10.1198/jbes.2009.08105. Not read.
- Da Fonseca, J., Grasselli, M., Tebaldi, C. (2007). Option pricing when correlations are stochastic: an
  analytical framework. Rev. Deriv. Res. 10, 151-180. doi:10.1007/s11147-008-9018-x. Not read.
- Da Fonseca, J., Grasselli, M., Tebaldi, C. (2008). A multifactor volatility Heston model. Quant. Finance 8,
  591-604. doi:10.1080/14697680701668418. Read.
- Fouque, J.-P., Pun, C.S., Wong, H.Y. (2016). Portfolio optimization with ambiguous correlation and stochastic
  volatilities. SIAM J. Control Optim. 54, 2309-2338. doi:10.1137/15m1032533. Read.
- Deelstra, G., Simon, M. (2017). Multivariate European option pricing in a Markov-modulated Lévy framework.
  J. Comput. Appl. Math. 317, 171-187. doi:10.1016/j.cam.2016.11.040. Not read.
- Carmona, R., Durrleman, V. (2003). Pricing and hedging spread options. SIAM Review 45, 627-685.
  doi:10.1137/s0036144503424798. Not read.
