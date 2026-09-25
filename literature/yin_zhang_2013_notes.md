# Yin & Zhang (2013), Continuous-Time Markov Chains and Applications: A Two-Time-Scale Approach, 2nd ed.

Springer, Stochastic Modelling and Applied Probability 37. 437 pages. First edition 1998 (subtitle "A Singular
Perturbation Approach"). Second edition (preface dated August 2012) adds two chapters, 6 (backward equations)
and 10 (hybrid LQG), and merges first-edition chapters 4+6 into chapter 4 and 5+7 into chapter 5.

Notes are section by section, with page numbers of the 2013 edition. Theorems are paraphrased; short quotes only.
The question these notes answer first: does the book contain the fast-switching expansion of a *forced* backward
equation a' = (Q/ε + diag g) a, i.e. the Feynman–Kac / bond-price construction of Cotton (2001)?

## Verdict (from chapters 4 and 6; confirmed by the full read, see the end)

**No.** Every expansion in the book is for the chain alone, or for the chain with a *slow generator* perturbation:

- Chapter 4: forward equation dp/dt = p Q(t)/ε (irreducible case, §4.2) and dp/dt = p (Q̃(t)/ε + Q̂(t)) with
  Q̂ another generator (weak and strong interactions, §4.3–4.5). The unknown is a probability vector.
- Chapter 6: backward equation du/dt = −(Q̃(t)/ε + Q̂(t)) u, u(T) = u₀, again with Q̂ a generator (eq. 6.4,
  p. 237). No potential, no discounting, no terminal exponent; u is E[u₀(α(T)) | α(t)], the expectation of a
  function of the chain alone. Based on Yin & Nguyen, Acta Math. Appl. Sinica 25 (2009) 457–476.
- The word "bond", "Vasicek", "interest rate" and "Cotton" do not occur in the book. "Option" occurs only in the
  bibliography entry [223]: G. Yin, "Asymptotic expansions of option price under regime-switching diffusions with
  a fast-varying switching process", Asymptotic Analysis 65 (2009) 203–222, DOI 10.3233/asy-2009-0953 (cited on
  p. 236 for the orthogonality-uniqueness remark of Lemma 6.1). Its Crossref abstract: asymptotic expansions of
  the coupled Black–Scholes equations under fast switching, leading term a Black–Scholes equation with mean
  return and volatility averaged with respect to the stationary measure, uniform error bounds. **That paper, not
  the book, is the post-2001 work closest to the thesis**: same programme (Feynman–Kac backward equation with the
  chain fast) applied to equity options instead of bonds and survival probabilities. Not yet read (paywalled);
  its treatment of higher orders, the initial layer and the Green–Kubo structure is unknown.

What the book does supply, and the paper cites it for: the singular-perturbation machinery itself. The
recursion Q ϕ_{k+1} = −ϕ̇_k − Q̂ ϕ_k with the Fredholm alternative, the terminal/initial layer by Taylor-expanding
Q at the boundary, exponential decay of layer terms through orthogonality to the stationary projector, and the
error lemma |L^ε e| = O(ε^{k+1}) ⇒ |e| = O(ε^k) with the "back up one step" trick. Our Theorem 1 is this argument
with diag g added; the potential changes the outer equations (ϕ₀ acquires the exponential of the averaged
forcing) but not the structure of the proof.

## Front matter

- p. xiii, Preface to the 2nd edition (August 2012): two new chapters (6, backward equations; 10, hybrid LQG);
  first-edition chapters 4+6 → 4 and 5+7 → 5; notation simplified.
- p. xv, Preface to the first edition (March 1998): singularly perturbed nonstationary chains whose states form
  groups with fast switching within and slow switching between; asymptotic expansions of distributions, error
  estimates, asymptotic normality, exponential bounds, weak and strong interactions; applications to hierarchical
  production planning, MDPs, control. Thanks Fleming, Kushner, Khasminskii ("expertise in probability and singular
  perturbations"), Bensoussan, Kokotovic, Kurtz, Karatzas.
- p. xvii Convention: equations (chapter.number); theorems etc. numbered together within a chapter; (A6.1) =
  assumption 1 of chapter 6; ϕ_n, ψ_n sequences; ϕ^i partitioned block.
- p. xix Notation: K generic constant (K + K = K); ν(t) quasi-stationary distribution; 1 column of ones; Qf(·)(i) =
  Σ_{j≠i} q_ij (f(j) − f(i)); O₁(y) with sup ≤ 1; |y|_T sup norm on [0,T].

## Chapter 1: Introduction and overview (pp. 3–16)

### 1.1 Introduction
Two motivating problems. (i) Production planning of a failure-prone machine: capacity α^ε with generator
Q(u(t))/ε, surplus ẋ = u − z, discounted cost; for small ε the limit problem replaces the chain by its average
distribution ν(t) (Sethi & Zhang [192]); asymptotically optimal controls from the limit. (ii) Large-scale systems
with weak and strong interactions: Q^ε = Q̃/ε + Q̂ with Q̃ = diag(Q̃¹,…,Q̃^l) irreducible blocks; the aggregated
process ᾱ^ε(t) = Σ i 1{α^ε ∈ M_i} converges to a chain with generator Q̄ = diag(ν¹,…,ν^l) Q̂ diag(1_{m₁},…,1_{m_l}),
an l × l matrix: decomposition and aggregation (Simon, Simon–Ando). Hybrid systems with Markovian switching;
stability (Kac–Krasovskii); the Wang–Khargonekar–Beydoun example (1.1): two stable linear systems switched at
period 0.01 give an unstable system; the book's limit system explains it (details §5.6). The illustrative model
(1.2)–(1.3): dp^ε/dt = p^ε Q/ε, p^ε = p⁰ exp(Qt/ε) → ν at rate O(exp(−κ₀t/ε)). The six questions the book answers
for time-dependent Q(t): existence and identification of the limit, rate, the CLT for the centred scaled
occupation measure n^ε(t) = ε^{−1/2}∫₀^t (χ^ε − ν) ds, weak and strong interactions, measurable generators. "ε = 0.1
might be considered as a small quantity from a practical point of view" (p. 10).

### 1.2 A brief survey (p. 10)
Markov chains: Markov 1907, Kolmogorov, Doeblin, Doob, Lévy, Kendall, Reuter, Chung; surveys listed; nonstationary
cases (Bernstein, Doeblin, Linnik, Dobrushin; Iosifescu); Davis's piecewise-deterministic processes as the
framework. Singular perturbations: Prandtl, WKB (Liouville–Green 1837), Krylov–Bogoliubov averaging, Friedrichs,
Levinson, Mitropolskii; texts Kevorkian–Cole, O'Malley, Smith, Vasil'eva–Butuzov, Wasow; physics (Gardiner,
Risken, van Kampen, Schuss, Hänggi–Talkner–Borkovec); control (Bensoussan, Kokotovic et al., Pervozvanskii–
Gaitsgori); stochastic averaging: Khasminskii [112], Freidlin–Wentzell, Kushner. No finance references here.

### 1.3 Outline (p. 12)
Part I (1–3) background; Part II (4–6) asymptotics: 4 forward equations (irreducible, then Q̃/ε + Q̂ with recurrent,
absorbing, transient states; countable state spaces; remarks on diffusions), 5 occupation measures (exponential
bounds, CLT with initial layers in the covariance, weak and strong interactions where the limit is a Gaussian
mixture / switching diffusion, measurable generators), 6 backward equations ("adjoint" of 4; "one of the crucial
results is Lemma 6.1"). Part III (7–10): MDPs with Q^ε(u) = Q̃(u)/ε + Q̂(u), near-optimal controls of dynamic
systems, numerical methods (Markov chain approximation, threshold policies by stochastic approximation), hybrid
LQG. Figure 1.1: dependence chart, chapters 5 and 6 both depend on 4.

## Chapter 2: Mathematical preliminaries (pp. 17–29)

- 2.2 Martingales: filtrations, progressive measurability (Davis), stopping times, martingales, local martingales.
- 2.3 Markov chains (p. 18): jump processes on finite or countable M; transition matrix P(t,s), Chapman–Kolmogorov.
  Definition 2.1 (q-Property): q_ij Borel, uniformly bounded, q_ij ≥ 0 off-diagonal, rows sum to zero.
  Definition 2.2 (generator): Q(t) generates α if f(α(t)) − ∫₀ᵗ Q(ς)f(·)(α(ς)) dς is a martingale for bounded f.
  Lemma 2.4: equivalently the indicator vector minus ∫ (indicator vector) Q is a martingale.
- 2.4 Piecewise-deterministic construction (p. 21): jump times with density exp(∫₀ᵗ q_ii)(−q_ii(t)), post-jump
  law q_ij(τ)/(−q_ii(τ)); Theorem 2.5: the construction is a Markov chain with generator Q(t); forward equation
  dP(t,s)/dt = P(t,s)Q(t) (2.5); if Q continuous, backward equation dP(t,s)/ds = −Q(s)P(t,s) (2.6).
- 2.5 Irreducibility (p. 23): Definition 2.7: weakly irreducible = ν(t)Q(t) = 0, Σν_i = 1 has a unique solution
  with ν ≥ 0; (strongly) irreducible if ν > 0. Two-state example: weak needs λ+μ > 0, strong needs both > 0.
  Definition 2.8 quasi-stationary distribution. Rank of a weakly irreducible Q is m−1. Definition 2.10: the
  algebraic form (fQ = 0, f·1 = 0 only trivially). Nonhomogeneous fQ = g solvable iff g·1 = 0 (Fredholm, Cor. A.38);
  unique solution with prescribed f·1 via Q_c = (1 ⋮ Q), f (Q_c Q_cᵀ) = g_c Q_cᵀ (2.11) — "used repeatedly".
- 2.6 Gaussian processes and diffusions (p. 25): Brownian motion, stochastic integral, diffusion generator L (2.12).
- 2.7 Switching diffusions (p. 27): dx = b(x,α)dt + σ(x,α)dw with state-dependent switching q_ij(x) (2.13–2.14);
  the pair is Markov; generator (2.15) with Q(x)f(x,·)(ι); Poisson-random-measure representation (2.16–2.17);
  generalized Itô lemma (2.18). No pricing.
- 2.8 Notes: Chung, Davis, Ethier–Kurtz, Elliott [55], Yin–Zhu [244] for switching diffusions.

## Chapter 3: Markovian models (pp. 31–56)

- 3.2 Birth and death processes; M/M/1 stationary distribution ν_i = (1−λ/μ)(λ/μ)^i.
- 3.3.1 Queues with finite capacity; singularly perturbed queues Q(t)/ε; Massey–Whitt uniform acceleration
  (Q(εt), τ = εt gives dp/dτ = pQ(τ)/ε, eq. 3.4: "studying such a singularly perturbed model is the objective of
  this book"); a two-customer-type queue with Q̃/ε + Q̂ (bank example: check deposits vs account opening); "it is
  the relative rates that count".
- 3.3.2 Reliability: parallel components; standby system (3.5) with ν_i ∝ (μ/λ)^{i−1}; burn-in phenomenon with a
  mixed exponential up-time, five-state generator.
- 3.3.3 Competing risks: Q̃/ε + Q̂ with Q̃ = diag(Q₀, 0): risky states are absorbing under the fast part.
- 3.3.4 Two-time-scale Cox processes: intensity modulated by a fast chain; compensator G^ε(t) = G₀ + Σ a_i ∫ 1{α^ε=i}
  is a weighted occupation measure, converges in mean square (chapter 5), even for measurable Q.
- **3.3.5 Random evolutions (p. 41)**: dM/ds = −V(α(s))M; u(t,α) = E_α M(0,t) satisfies du/dt = V(α)u + Qu (3.6),
  "known as a generalized Feynman–Kac formula in potential theory"; the sped-up version du/dt = V(α)u + Qu/ε
  (Hersh [85], Rocky Mountain J. Math. 1974). **This is exactly the reduced system a' = (Q/ε + diag g)a with a
  constant forcing.** The book stops at the remark that chapter 4 handles the distribution of α^ε; it does not
  expand (3.6) or its sped-up form anywhere.
- 3.3.6 Seasonal temperature model: nine states in three seasonal groups, ε = 1/20, aggregated three-state chain
  with rates λ̄_i(t), μ̄_i(t); group averages 30, 60, 90 degrees.
- 3.4 Stochastic optimization: simulated annealing (Chiang–Chow, Wentzell eigenvalues), continuous-time stochastic
  approximation with Markovian noise, Polyak–Ruppert averaging as a singular perturbation (Yin–Gupta [224]),
  controlled systems with Markovian disturbances (chapter 8).
- 3.5 Jump linear systems: LQ and LQG with Markov jumps (Ji–Chizeck), wide-band noise ξ(t/ε²)/ε, large-scale
  systems by decomposition/aggregation (Phillips–Kokotovic, Pan–Başar), the transient-state model.
- 3.6 Time-scale separation (p. 53): "typically works well in practice when ε is sufficiently small (i.e., less
  than a 'small' constant, e.g., 0.5)"; worked 4×4 decomposition of a generator into Q̃/ε + Q̂ with ε = 0.1 by
  separating orders of magnitude, making each part a generator, and permuting to block-diagonal form; the
  decomposition is not unique; Avramovic's eigenvalue-based reduction as the systematic alternative.
- 3.7 Notes: manufacturing origins (Sethi–Zhang); Yin, H. Zhang & Q. Zhang, "Applications of two-time-scale
  Markovian systems", preprint 2012 [232], for queueing, financial market models and insurance risk; discrete-time
  counterpart Yin–Zhang 2005 [238].

## Chapter 4: Asymptotic expansions of solutions for forward equations (pp. 59–139)

### 4.1 Introduction (p. 59)
Aim: approximate p^ε(t), the distribution of a chain with generator Q(t)/ε, by outer expansion plus initial-layer
correction in the stretched time τ = t/ε; construct, prove regularity and exponential decay, bound the error
uniformly. Remark: the usual singular-perturbation stability condition fails (the generator has eigenvalue 0);
Markov structure replaces it. Lemma 4.1: the solution of the forward equation stays a probability vector.

### 4.2 Irreducible case (pp. 62–84)
Setting: dp^ε/dt = p^ε Q(t)/ε, p^ε(0) = p⁰ (eq. 4.3). Assumptions (A4.1) Q(t) weakly irreducible on [0,T]
(unique nonnegative solution of f Q = 0, f·1 = 1: the quasi-stationary distribution ν(t)); (A4.2) Q ∈ C^{n+1}
with Lipschitz (n+1)st derivative. Lemma 4.4 (Appendix A.2): |exp(As) − P| ≤ K e^{−κs}, P = 1ν.

**Theorem 4.5** (p. 65): under (A4.1)–(A4.2) there are ϕ_i ∈ C^{n+1−i}[0,T] and ψ_i with |ψ_i(t/ε)| ≤ K e^{−κ₀ t/ε}
such that sup_[0,T] |p^ε − Σ_{i≤n} ε^i ϕ_i − Σ_{i≤n} ε^i ψ_i(t/ε)| ≤ K ε^{n+1}. ϕ₀ = ν(t). Corollary 4.7: p^ε → ν(t).
Remark 4.8: if p⁰ = ϕ₀(0) the zeroth layer vanishes but higher layers remain.

Construction (4.2.2, p. 66): L^ε f = ε f' − f Q; equating powers, ϕ₀ Q = 0, ϕ_k Q = ϕ̇_{k−1} (eq. 4.13). Remark 4.9:
solvability needs ϕ̇_{k−1} ⊥ N(Q) = span 1, which holds since ϕ₀·1 = 1 and ϕ_k·1 = 0. Uniqueness by replacing one
equation with the normalisation; Cramer's rule with determinant Δ(t) bounded away from 0 (eq. 4.15). Remark 4.10:
alternatively ϕ₀ = (1,0,…,0) Q_cᵀ (Q_c Q_cᵀ)^{−1} with Q_c = (1 ⋮ Q).

Initial layer (4.2.3, p. 69): τ = t/ε; Taylor-expand Q(ετ) about 0; dψ₀/dτ = ψ₀ Q(0), dψ_k/dτ = ψ_k Q(0) + r_k(τ),
r_k = Σ_{i=1}^k (τ^i/i!) ψ_{k−i} Q^{(i)}(0) (eq. 4.18–4.19); matching ϕ₀(0)+ψ₀(0) = p⁰, ϕ_k(0)+ψ_k(0) = 0; solutions
ψ₀ = (p⁰ − ϕ₀(0)) e^{Q(0)τ}, ψ_k = −ϕ_k(0) e^{Q(0)τ} + ∫₀^τ r_k(s) e^{Q(0)(τ−s)} ds (eq. 4.20–4.21).

Decay (4.2.4, p. 72): Proposition 4.11: |ψ_k(τ)| ≤ c_{2k}(τ) e^{−κ₀₀ τ}, polynomial prefactor of degree 2k, because
ψ₀(0) ⊥ 1 so ψ₀(0)P = 0, and Q^{(k)}(0) 1 = 0 so Q^{(k)}(0) P = 0. Corollary 4.12: pure exponential with a smaller
rate.

Validation (4.2.5, p. 74): Lemma 4.13: sup|L^ε v| = O(ε^{k+1}), v(0) = 0 ⇒ sup|v| = O(ε^k), via the bounded principal
matrix solution and v = ε^{−1}∫ η X. Proposition 4.14: e_k = O(ε^{k+1}) for k ≤ n, proved by bounding L^ε e_{n+1}
(the ϕ terms telescope to −ε^{n+2} ϕ̇_{n+1}; the ψ terms leave Taylor remainders t^{n+2−i} e^{−κ₀t/ε} = O(ε^{n+2−i}))
then "backing up one step" (eq. 4.28–4.29). Remark 4.15: L^ε e_k = O(ε^{k+1} + (εt + … + ε^k t) e^{−κ₀t/ε}), used
later on [0, ∞). Piecewise-smooth Q gives inner boundary layers at the breakpoints.

Examples (4.2.6, p. 78): 4.16 constant irreducible Q: series terminates, ϕ₀ = ν, ψ₀ = (p⁰−ν) e^{Qt/ε}; two-state
formulas with λ, μ. 4.17 two-state time-varying: closed form; ϕ₁ = ((λ̇μ − μ̇λ)/(λ+μ)³, −same); weak irreducibility
allows λ(t) or μ(t) to vanish as long as λ+μ > 0.

4.2.7 Two-time-scale expansion (p. 81): the Kevorkian–Cole form y^ε(t,τ) = Σ ε^i y_i(t,τ) with d/dt = ∂_t +
ε^{−1}∂_τ; ∂_τ y₀ = y₀ Q(t), y₀ = p⁰ e^{Q(t)τ}, y_i = −∫₀^τ ∂_t y_{i−1} e^{Q(t)(τ−s)} ds (4.37), t "frozen".
Theorem 4.18: same O(ε^{n+1}) bound. Relationship (p. 83): y₀ = ν(t) + [p⁰ e^{Q(t)τ} − P(t)] (4.38), so each
y_i = ϕ_i + an exponentially decaying part; the two-state case worked out. Comparison list: same conditions,
no significant difference, both use a stationary chain (Q(0) vs frozen Q(t)); the separable form is more
transparent, better for weak/strong interactions and for asymptotic normality; the book uses the separable form.

### 4.3 Multiple weakly irreducible classes (pp. 84–107)
Generator Q^ε(t) = Q̃(t)/ε + Q̂(t) (4.39), both generators; Q̃ = diag(Q̃¹,…,Q̃^l) after relabelling (4.41), each
weakly irreducible, groups M_k. Example 4.20: two-machine flowshop, machine 1 breaks down faster.
(A4.3) each Q̃^k(t) weakly irreducible; (A4.4) Q̃ ∈ C^{n+1}, Q̂ ∈ C^n with Lipschitz top derivatives.
L^ε f = ε f' − f(Q̃ + εQ̂) (4.42). Outer equations ϕ₀Q̃ = 0, ϕ_i Q̃ = ϕ̇_{i−1} − ϕ_{i−1}Q̂ (4.44); layer equations by
Taylor expansion of both Q̃ and Q̂ at 0 (4.45); matching ϕ₀(0)+ψ₀(0) = p⁰, ϕ_i(0)+ψ_i(0) = 0 (4.46); ϕ₀·1 = 1,
ϕ_i·1 = 0 (4.47). Key difference from §4.2: ϕ_i and ψ_i must be found jointly; the initial conditions are the
delicate point.
Leading term: ϕ₀^k = ν^k(t) ϑ₀^k(t) (Lemma 4.21), ϑ₀^k the probability of the aggregated state M_k; the solvability
(Fredholm) condition gives the aggregated equation dϑ₀/dt = ϑ₀ Q̄(t), Q̄ = diag(ν¹,…,ν^l) Q̂ 1̃ (4.50–4.51),
Remark 4.23; initial-value consistency ϕ₀(0)1̃ = lim_δ lim_ε p^ε(δ)1̃ = p⁰1̃ (4.53); Proposition 4.24 summarises.
ψ₀ = (p⁰ − ϕ₀(0)) e^{Q̃(0)τ} (4.56), Proposition 4.25 exponential decay via orthogonality to π = 1̃ diag(ν^k(0)).
Higher terms: ϕ₁^k = b̄₀^k + ϑ₁^k ν^k (4.59) with b̄₀^k the 1-orthogonal particular solution; dϑ₁/dt = ϑ₁Q̄ +
b̄₀ Q̂ 1̃ (4.61); ϑ₁(0) fixed by demanding ψ₁(τ) → 0: ψ₁(0)π + ψ̄₀π = 0 with ψ̄₀ = (∫₀^∞ ψ₀(0)e^{Q̃(0)s} ds) Q̂(0)
(4.64–4.67), giving ϑ₁^k(0) = ψ̄₀^k 1_{m_k}; then ψ₁(0) = −ϕ₁(0). Induction for all i (4.68). Proposition 4.26
collects: (a) the algebraic-differential system (4.69), (b) the initial-condition recipe, (c) exponential decay,
(d) ϕ_i ∈ C^{n+1−i}.
4.3.2 Remainder: Lemma 4.27 (as 4.13), Proposition 4.28 e_i = O(ε^{i+1}). 4.3.3 User's guide: two-stage
procedure (initialisation, iteration with the unspecified ϑ_i^k(0) fixed from the decay condition).
**Theorem 4.29** (p. 103): the expansion y_n^ε with (a) smoothness, (b) exponential decay, (c) O(ε^{n+1}).
Remark 4.30: the decay rate κ₀ shrinks with n (polynomial prefactor). Corollary 4.31: time-invariant case,
|P(α^ε(t) = s_kj) − ν_j^k ϑ^k(t)| ≤ K(ε(t+1) + e^{−κ₀t/ε}) with constants independent of ε and t (used in
chapters 5, 7). Remark 4.32: alternative decomposition ϕ_i = v_i diag(ν) + U_i. 4.3.5 the flowshop example
solved in closed form (4.73), ϕ₀ = (ν¹ϑ₀¹, ν²ϑ₀²) with ϑ₀ from the 2×2 aggregated generator.

### 4.4 Inclusion of absorbing states (pp. 107–114)
Q̃ = diag(Q̃¹,…,Q̃^l, 0_{m_a×m_a}) (4.74); motivation competing risks (Remark 4.33). ϕ₀ = (ϑ₀ν¹,…,ϑ₀ν^l, ϕ₀^a) with
the absorbing block ϕ₀^a not determined by Q̃ but by the i = 1 equation: d(ϑ₀, ϕ₀^a)/dt = (ϑ₀, ϕ₀^a) Q̄, Q̄ =
diag(ν¹,…,ν^l, I_{m_a}) Q̂ 1̃_a (4.77–4.78), initial (ϑ₀(0), ϕ₀^a(0)) = p⁰1̃_a. Remark 4.34: absorbing states cannot
be aggregated; ϕ₀^a stabilises if the Q̂₂₂ block is stable. ψ₀^a ≡ 0; Lemma 4.35 decay with π_a. Higher terms as
in §4.3 with ϕ₁^a from Q̂ and initial conditions from ψ₁(0)π_a = −ψ̄₀π_a (4.81). **Theorem 4.36**: same three
conclusions. Example 4.37: 3-state chain with one absorbing state solved exactly; ϕ₀ carries an e^{−t} term
from the absorbing state that decays with the slow generator, not with ε.

### 4.5 Inclusion of transient states (pp. 115–126)
Q̃ with recurrent blocks plus a transient block row (Q̃_*^1,…,Q̃_*^l, Q̃_*) (4.82), Q̃_* Hurwitz (A4.6); inspired by
Phillips–Kokotovic, Delebecque–Quadrat, Pan–Başar. ϕ₀^* = 0 (transient states have vanishing leading
probability, Remark 4.39); 1̃_*(t) with a_{m_k}(t) = −Q̃_*^{−1}Q̃_*^k 1 (4.85); Q̄_* = diag(ν)(Q̂₁₁1̃ + Q̂₁₂(a_{m₁},…,
a_{m_l})) (p. 118); Remark 4.40: Q̂₁₂ (recurrent → transient jump rates) decides whether transient states
matter. Initial condition needs care: Theorem 4.41 (proof in Appendix A.4) gives ϑ₀^k(0) = p^{0,k}1 −
p^{0,*}Q̃_*^{−1}(0)Q̃_*^k(0)1 (4.88): the initial transient mass is redistributed to the recurrent groups through
−Q̃_*^{−1} = ∫₀^∞ e^{Q̃_*t}dt, and Σ_kϑ₀^k(0) = 1 (4.89). Lemma 4.43 decay with π_*. ϕ₁^* = b₀^*Q̃_*^{−1}; (4.93)
for ϑ₁ has an extra term with db₀^*/dt; Remark 4.44: the ((d/dt)Q̃(0))π_* term no longer vanishes. **Theorem
4.45**: same conclusions. Example 4.46: four states, two transient, solved exactly, ϕ₀ = (½, ½, 0, 0).
Remark 4.47 summarises: the Hurwitz condition is what gives the layer decay.

### 4.6 Countable state spaces (pp. 126–133)
4.6.1 Q̃ block-diagonal with infinitely many finite blocks (4.95); (A4.7) each block weakly irreducible; (A4.8)
the blocks' decay rates κ_k bounded below by κ₀ > 0; semigroup exp(Q̃(0)τ) on l¹; Theorem 4.48: Theorem 4.29
holds. 4.6.2 Q̃ itself infinite-dimensional: Definitions 4.49–4.51 (weak irreducibility via Q_c = (1 ⋮ Q),
quasi-stationary distribution, F-Property = wQ_c = b uniquely solvable); (A4.9): smoothness, F-Property, bounded
|·|₁ norms, eigenvalue 0 simple and isolated, empty residual spectrum. Theorem 4.53: Theorem 4.29 holds;
smoothness of ϕ₀ from difference quotients since Cramer's rule is unavailable. 4.6.3 Finite-dimensional
truncation fails in general: the limits ε → 0 and N → ∞ do not commute; example with entries 2^{−|i−j|}-type
decay where every truncation has strictly negative eigenvalues, so the truncated expansion has no regular part.

### 4.7 Remarks on singularly perturbed diffusions (pp. 133–137)
Example 4.54: controlled slow–fast diffusion (fast component with drift/ε and diffusion/√ε); Khasminskii's
averaging with the quasi-stationary density μ(t, x₁, x₂). The forward (Fokker–Planck) equation
∂_t p = L₂^* p/ε + L₁^* p (4.101); on the periodic (compact) setting with (4.102) smoothness, the matched
expansion holds with error O(ε^{n+1}) in the weak sense ⟨p^ε − s_n^ε, h⟩_H (4.103); leading term ≈ density of X₁
times the quasi-stationary conditional density of X₂; regular terms μ v_i + U_i, "note the resemblance" to the
chain case. Proofs in Khasminskii–Yin [116]; uniform-topology versions via stochastic representation or energy
methods (Il'in–Khasminskii–Yin [94]). Again no potential term.

### 4.8 Notes (pp. 137–140)
Singular-perturbation literature; the distinct feature that Q(t) is singular so the usual stability condition
fails, replaced by the q-Property. The expansion of (4.3) was initiated in Khasminskii, Yin & Zhang [119]; §4.3
refines [120]. Weak/strong-interaction predecessors: Delebecque–Quadrat–Kokotovic, Gaitsgori–Pervozvanskii,
Phillips–Kokotovic (time-invariant, single ergodic class of Q̃/ε + Q̂ for small ε); here nonstationary and only
weak irreducibility of blocks. Countable-state extensions Yin–Zhang [230, 231] (quasi-birth-death queues);
diffusions Khasminskii–Yin [115, 116]; "averaging principles and related backward equations" Khasminskii–Yin
[117, 118]; applications to queueing, financial engineering, insurance risk in Yin, Zhang & Zhang [232]
(preprint 2012).

## Chapter 5: Occupation measures (pp. 141–234)

### 5.1–5.2.2 (p. 141)
Probabilistic counterpart of chapter 4: law of large numbers for unscaled occupation measures, exponential
bounds, CLT for the √ε-scaled centred occupation time n^ε(t,i) = ε^{−1/2}∫₀ᵗ(1{α^ε=i} − ν_i)β_i ds, with the
limit covariance depending on the initial-layer terms. Centred occupation measure Z_i^ε(t) (5.1) with a weight
β_i(·) (a control in chapter 8). (A5.1) weak irreducibility, (A5.2) Q ∈ C¹ with Lipschitz derivative.
Lemma 5.1: the transition matrix expands as P^ε(t,t₀) = P₀(t) + O(ε + e^{−κ₀(t−t₀)/ε}) (5.2), and to second
order P₀ + εP₁ + Q₀((t−t₀)/ε, t₀) + εQ₁(…) + O(ε²) (5.3) with identical rows ν(t), ϕ₁(t) and layer matrices
Q₀, Q₁ decaying uniformly in t₀ (proof by Gronwall and a compactness argument, (5.5)–(5.7)). Unscaled measure:
sup_t E|Z^ε(t)|² → 0, i.e. ∫₀ᵗ 1{α^ε=i} ds → ∫₀ᵗ ν_i ds in probability (LLN); Example 5.3 the Cox-process
compensator G^ε → G₀ + Σ a_i∫ν_i.

### 5.2.3 Exponential bounds (p. 148)
**Theorem 5.4**: for ε ≤ ε₀, all T ≥ 0 and bounded measurable β, E exp(θ_T (T+1)^{−3/2} sup_{t≤T}|n^ε(t)|) ≤ K
with 0 ≤ θ_T ≤ min(1, κ₀)/(K_T|β|_T) (5.13–5.14); constants independent of T (Remark 5.5). Proof via the
martingale w^ε(t) = χ^ε(t) − χ^ε(0) − ε^{−1}∫χ^εQ (Elliott), the representation χ^ε(t) = χ^ε(0)P^ε(t,0) +
∫(dw^ε)P^ε(t,s), the expansion (5.11) of P^ε − P₀, and an exponential-martingale (Doléans) estimate with a
tail-sum bound Ee^ξ ≤ e + (e−1)Σ e^j P(ξ ≥ j) (5.18). Steps 2–5: n^ε is "nearly a martingale"
(E[n^ε(t)|F_s] − n^ε(s) = O(√ε) deterministically, from the expansion), a convexity argument, a stopping-time
maximal inequality P(sup x^ε ≥ x) ≤ K/x (5.23), and integration of the tail. The jump count is dominated by a
Poisson process of rate a/ε (the piecewise-deterministic construction), giving the Stirling-type tail
P(N(T) ≥ a_j) ≤ 2γ₀^{a_j−1}.
Corollary 5.6: constant Q gives the sharper E exp(θ(1+T)^{−1/2} sup|n^ε|) ≤ K with θ independent of T (ϕ₁ ≡ 0).
Corollary 5.7: moment bounds E sup|n^ε|^{2j} ≤ K_j(1+T)^{3j} (or (1+T)^j when Q is constant). Corollary 5.8:
P(sup_t |∫₀ᵗ(1{α^ε=i} − ν_i)β_i| ≥ ε^{1/2−δ}) ≤ K exp(−θ_T ε^{−δ}(T+1)^{−3/2}) (5.27–5.28): large-deviation-type
decay of the occupation-measure deviation. Used for hierarchical controls in manufacturing (Sethi–Zhang).

### 5.2.4 Asymptotic normality (p. 159)
**Theorem 5.9**: under (A5.1) and Q ∈ C² with Lipschitz second derivative, n^ε(·) ⇒ a Gaussian process n(·)
with independent increments, mean 0, covariance ∫₀ᵗ A(s) ds with
A_ij(t) = β_i β_j [ν_i ∫₀^∞ q₀,ij(r,t) dr + ν_j ∫₀^∞ q₀,ji(r,t) dr] (5.30), Q₀(r,t) the zeroth layer matrix of the
transition-matrix expansion (5.3). Remark 5.10: E[nᵀ(t₁)n(t₂)] = ∫₀^{min} A. Remark 5.11: the second derivative
is needed only to characterise A. Proof in five steps: Lemma 5.12 mean → 0 (O(√ε)); Lemma 5.13 covariance by
splitting the double integral into D₁, D₂ and using P^ε(ς,r) expansion, change of variable ς − r = εs, and the
exponential decay of q₀ (5.33–5.34); Lemma 5.14 exponential φ-mixing of α^ε with rate κ/ε, proved by iterating
Chapman–Kolmogorov over blocks of length Nε (5.37–5.38); Lemma 5.16 tightness in D via a fourth-moment bound
E|ñ^ε(t+ς) − ñ^ε(t)|⁴ ≤ Kς² using the mixing inequality (5.39–5.43), and a.s. continuity of limits; Step 5
characteristic functions factorise across disjoint intervals by mixing, so the limit has independent
increments and continuous paths, hence Gaussian (Skorohod). Example 5.18: two-state chain, Q₀(s,t₀) =
−e^{−(μ₁+μ₂)s}Q(t₀)/(μ₁+μ₂), A(t) = 2μ₁μ₂/(μ₁+μ₂)³ · [[β₁², −β₁β₂],[−β₁β₂, β₂²]].
[Note for our purposes: ∫₀^∞ Q₀(r,t) dr is exactly the group-inverse object (−Q^# restricted to the centred
subspace), i.e. their asymptotic covariance A is the Green–Kubo matrix K of the indicator functions weighted by
β; this is the probabilistic face of the first-order term in the reduced system.]

### 5.2.5 Extensions (p. 169)
Q^ε = Q(t)/ε + Q̂(t) with Q weakly irreducible: (A5.3)–(A5.4); outer/layer equations (5.44)–(5.45) now with Q̂
entering at order ε; Theorem 5.19 (expansion), Theorem 5.20 (exponential bound), Corollary 5.21 (constant
generators: T-independent constants), Theorem 5.22 (CLT with the same A, Q̂ irrelevant to the covariance);
Remark 5.23: irreducibility of the fast part is what matters; Sethi–Zhang Lemma J.10 for weak irreducibility of
Q/ε + Q̂ at small ε; Corollary 5.6 fails when Q̂ ≠ 0 since ϕ₁ ≠ 0.

### 5.3 Weak and strong interactions (p. 173)
Convention: "weak and strong interaction" = all states recurrent, Q̃ = diag(Q̃¹,…,Q̃^l). Plan: aggregate M_k to a
super state k; the aggregated process ᾱ^ε converges weakly to a Markov chain with generator Q̄ = diag(ν)Q̂ 1̃
(5.48); error bounds via linear time-varying SDEs driven by a square-integrable martingale; the CLT limit is a
switching diffusion (no independent increments) by martingale problem and perturbed test functions; measurable
generators in §5.4.
5.3.1 Aggregation: (A5.5) blocks weakly irreducible, (A5.6) Q̃ ∈ C¹ Lipschitz derivative, Q̂ Lipschitz.
Lemma 5.24: P^ε(t,t₀) = P₀(t,t₀) + O(ε + e^{−κ₀(t−t₀)/ε}) with P₀(t,t₀) = 1̃Θ(t,t₀)diag(ν^k(t)) (5.49), Θ the
principal solution of dΘ/dt = ΘQ̄(t) (5.50). Aggregated process ᾱ^ε(t) = k if α^ε(t) ∈ M_k (5.51). (Continues.)

Theorem 5.25: E[∫₀ᵀ(1{α^ε = s_ij} − ν_j^i(t)1{ᾱ^ε = i})β_ij dt]² = O(ε): the chain is approximated by its
aggregate times the block quasi-stationary distribution (Liapunov-function proof via the expansion, (5.52)–(5.57)).
Example 5.26: α^ε itself is not tight (two-state Q/ε; a limit would have to be the constant ν₁ + 2ν₂). Theorem
5.27: the aggregate ᾱ^ε ⇒ ᾱ, a Markov chain with generator Q̄(t) (tightness by Kurtz's criterion, f.d.d. by
Lemma 5.24). Example 5.28: when Q̂ = (q̄_ij I_{m₀}) with identical blocks the aggregate is exactly Markov.

### 5.3.2 Exponential bounds with interactions (p. 182)
Centred weights W_ij(t,α) = (1{α = s_ij} − ν_j^i(t)1{α ∈ M_i})β_ij(t) (5.61). **Theorem 5.29**: E exp(θ_T(T+1)^{−3}
sup|n^ε|) ≤ K with θ_T ≤ min(1,κ₀)/(K_T|β|_T(1+|Q̂|_T)) (5.65–5.66). Proof: the SDE dχ^ε = χ^εQ^ε dt + dw^ε, the
aggregate indicator χ̄^ε, the identity (5.75) comparing χ^ε with χ̂^ε = χ̄^ε diag(ν), and a change of variables
through the principal solutions P^ε, Φ̌^ε, Ψ̌^ε with Ψ̌^ε(t,s) solving dΨ̌/dt = Ψ̌ Q̃/ε (5.80), hence entries in [0,1].
Remark 5.31: the comparison is with the random aggregate, not a deterministic function. Corollary 5.32: identical
blocks give the (T+1)^{3/2} exponent; Corollary 5.33: constant generators give (T+1)^{5/2}, with Q̌₀ = Q̂1̃diag(ν)
and Q̂P₀(t,s) = Q̌₀e^{Q̌₀(t−s)}; Remark 5.34: K_T = K(T+1) then.

### 5.3.3 Asymptotic distributions with interactions (p. 191)
n^ε(·) ⇒ a switching diffusion modulated by ᾱ(·), a Gaussian mixture, not Gaussian. Plan: tightness of
(n^ε, ᾱ^ε); the limit solves a martingale problem with a unique solution; characterise; construct the switching
diffusion. Lemma 5.35: E[n^ε(t) − n^ε(s)|F_s] = O(√ε), E[|n^ε(t) − n^ε(s)|²|F_s] = O(t−s). Lemma 5.36: (n^ε, ᾱ^ε)
tight in D([0,T]; R^m × M). Lemma 5.37: ∫ W_ij(s,α^ε)ξ(s, n^ε(s)) ds → 0 in mean square for Lipschitz ξ, by a
piecewise-constant approximation of ξ on a partition of mesh ε^{1−δ} (Remark 5.38). (Continues.)

Martingale problem (p. 198): generator G^ε f = ∂_t f + ε^{−1/2}⟨W(t,α), ∇_x f⟩ + Q^ε f; perturbed test function
f(x,α) + √ε h(t,x,α) with Q̃ h = −⟨W, ∇_x f⟩ (5.93), solvable by Fredholm since W is centred within each block
(orthogonal to 1_{m_k}); g(s,x,α) = ⟨W, ∇_x h⟩ (5.95), block-averaged ḡ(s,x,i) = Σ_j ν_j^i g(s,x,s_ij) which is a
quadratic form ½Σ a_{j₁j₂}(s,i)∂²f⁰ (5.99). Remark 5.39: h is unique up to a block constant, g is well defined.
Lemma 5.40: any weak limit solves the martingale problem with operator L = ½Σ a ∂² + Q̄(s); Lemma 5.41
uniqueness (characteristic functions satisfy a linear ODE (5.102)); Lemma 5.42: A(s,j) = (a_{j₁j₂}) symmetric
nonnegative definite; explicit A(s,j) = 2A⁰(s,j) with A⁰ = β_diag[ν_diag ∫₀^∞ Q₀(r,t,j) dr + (∫Q₀ dr)ᵀ ν_diag]β_diag,
Q₀(r,t,j) = (I − 1ν^j)e^{Q̃^j r}, σ⁰σ⁰ᵀ = A⁰ (5.104), σ(s,j) block-diagonal with the jth block only (5.105).
**Theorem 5.43** (p. 210): n^ε ⇒ n(t) = ∫₀ᵗ σ(s, ᾱ(s)) dw(s), a switching diffusion modulated by the limit
aggregate chain. Remark 5.45: no independent increments (E[n^εᵀ(s)(n^ε(t) − n^ε(s))] ↛ 0 because it involves
P₁(t,s)); the direct mixing approach of §5.2 fails, the perturbed test function approach treats (x, α) jointly.
Corollary 5.46: one block recovers the Gaussian limit with A of (5.30). Example 5.47: the four-state flowshop
chain, σ⁰(s,j) = 2^{1/2}(λ₁μ₁/(λ₁+μ₁)³)^{1/2}·[[β_{j1}, 0],[−β_{j2}, 0]].

### 5.4 Measurable generators (p. 213)
Only measurability and boundedness (the q-Property) plus weak irreducibility (A5.7); no expansion possible, so
convergence is in the weak topology of L²([0,T]). Case I: Lemma 5.49: P(α^ε(t) = i) and the transition
probabilities converge weakly in L² to ν_i(t) (weak-L² compactness, then p(t)Q̃(t) = 0 a.e.); Theorem 5.50:
E|∫₀ᵗ(1{α^ε=i} − ν_i)β_i|² → 0. Case II: block-diagonal Q̃ (A5.8): Lemma 5.51: P(α^ε(t) = s_ij) → ν_j^i(t)ϑ^i(t)
weakly in L², ϑ solving the aggregated integral equation; Theorem 5.52: the analogue of Theorem 5.25 with → 0
instead of O(ε); Theorem 5.53: ᾱ^ε ⇒ ᾱ. Used in §8.6 (relaxed controls). (Continues.)

Theorem 5.53's proof uses the martingale identity (5.115) for the aggregate indicator χ̄^ε and Theorem 5.52 to
identify the limit's generator as Q̄. Remark 5.54: same result as Theorem 5.27 under weaker conditions, different
proof. Remark 5.55: a weak-derivative (Sobolev H^n) formulation might carry the expansion over to measurable
generators; not pursued.

### 5.5 Transient and absorbing states (p. 222)
Transient: (A5.9) blocks weakly irreducible, Q̃_* Hurwitz; Q̄_* = diag(ν)(Q̂₁₁1̃ + Q̂₁₂(a_{m₁},…,a_{m_l})) (5.120);
a_{m_i}(t) = −Q̃_*^{−1}Q̃_*^i 1 form a probability row vector across i for each transient state (5.122), since
−Q̃_*^{−1} = ∫₀^∞ e^{Q̃_* s} ds ≥ 0. Theorem 5.56: centred occupation measures are O(ε) for recurrent states and O(ε²)
for transient ones (5.124). The aggregate is redefined with a randomisation ξ_j(t) that assigns a transient state
to a recurrent group with probabilities a_{m_i, j}(t) (5.125); Theorem 5.57: ᾱ^ε ⇒ ᾱ with generator Q̄_*. Theorem
5.58: switching-diffusion limit with σ padded by a zero transient block (5.127–5.128). Theorem 5.59: measurable
generators, same conclusions in the mean-square/weak sense.
Absorbing: (A5.10); Q̄ = diag(ν¹,…,ν^l, I_{m_a}) Q̂ 1̃_a (5.131); expansion p^ε = (ϑ, ϑ^a)diag(ν, I) + O(ε + e^{−κ₀t/ε});
transition-matrix expansion (5.132); aggregate keeps absorbing states as themselves (5.133); centred measures
(5.134)–(5.135) with the absorbing part centred at ϑ^a_j(t) and not scaled by ε^{−1/2}; Theorem 5.60: recurrent
measures O(ε); ᾱ^ε ⇒ ᾱ; the pair (n^ε, ᾱ^ε) ⇒ solution of a martingale problem with a first-order drift term
Σ b_j ∂_{a,j} for the absorbing part; measurable case; covariance blocks W^rr → diffusion part, cross terms → 0,
W^aa → ∫(δ_jk ϑ^a_j − ϑ^a_jϑ^a_k) ds (5.137–5.138). Proofs in Yin, Zhang & Badowski [241].

### 5.6 Remarks on a stability problem (p. 229)
The Wang–Khargonekar–Beydoun example redone with Markovian switching Q/ε, Q = [[−1,1],[1,−1]]: G(1), G(2) both
Hurwitz, averaged Ḡ = (G(1)+G(2))/2 has eigenvalues −210 and 10 (saddle); x^ε ⇒ x of the averaged system, with a
large-deviations bound P(ρ_{0,T}(x^ε, x) ≥ δ) ≤ exp(−c₁/ε) (He–Yin–Zhang [84]); perturbed Liapunov functions
V + V₂^ε with V₂^ε = O(ε)V show E(y₂^ε)² ≥ E(y₂^ε(0))² e^{9t} and E(y₁^ε)² ≤ … e^{−209t}: unstable in probability.
Second example (5.145): two unstable G(i) whose average is stable. Moral: stability of the averaged system
"implies" that of the original for small ε, in both directions.

### 5.7 Notes (p. 233)
§5.2 from Zhang & Yin [252], §5.3 from Zhang & Yin [253]; T-dependence of the exponential bounds matters for
discounted infinite-horizon control; martingale problem (Stroock–Varadhan), perturbed test functions
(Blankenship–Papanicolaou, Papanicolaou–Stroock–Varadhan, Kushner); Kurtz suggested treating (n^ε, ᾱ^ε) jointly;
transient/measurable/absorbing results in Yin, Zhang & Badowski [239–241]. Random-evolution CLT: Pinsky [176].

## Chapter 6: Asymptotic expansions of solutions for backward equations (pp. 235–257)

### 6.1 Introduction (p. 235)
Dual of chapter 4: backward equations; matched asymptotic expansions; uniform error bounds; recurrent states
only (§6.3–6.4) and with transient states (§6.5).

### 6.2 Problem formulation (p. 236)
Lemma 6.1 (Q constant, weakly irreducible): Qζ = b solvable iff νb = 0; solutions differ by c₀1; unique solution
with νξ = 0, computable as ξ = (Q̌ᵀQ̌)^{−1} Q̌ᵀ B with Q̌ = (Q; ν) (eq. 6.3). Cites Yin [223] (the 2009 option paper) for
the orthogonality remark.
Formulation: Q^ε(t) = Q̃(t)/ε + Q̂(t), both generators (eq. 6.4); Q̃ block-diagonal of l weakly irreducible blocks
(6.5) or with a transient block and Hurwitz Q̃_* (6.6). The equation: du^ε/dt = −Q^ε(t) u^ε, u^ε(T) = u₀ (eq. 6.10),
"the backward Kolmogorov equation"; u^ε is a probability vector only if u₀ is one.

### 6.3 Construction (p. 238)
(A6.1) each Q̃^i(t) weakly irreducible with quasi-stationary ν^i(t); (A6.2) Q̃, Q̂ ∈ C^{n+2}. Ansatz Φ_n(t) +
Ψ_n((T−t)/ε), τ = (T−t)/ε a *terminal* layer (eq. 6.11). Outer equations Q̃ ϕ₀ = 0, Q̃ ϕ_{i+1} = −ϕ̇_i − Q̂ ϕ_i
(6.12). Layer equations by Taylor expansion of Q̃, Q̂ at T (6.13–6.14): dψ₀/dτ = Q̃(T)ψ₀, dψ_i/dτ = Q̃(T)ψ_i + r_i.
Matching ϕ₀(T) + ψ₀(0) = u₀, ϕ_i(T) + ψ_i(0) = 0 (6.15); ψ_i explicit by variation of constants (6.16).
Lemma 6.2: |e^{Q̃(T)τ} − P| ≤ C e^{−κτ}, P = diag(1 ν^i(T)).
6.3.1 Leading term: ϕ₀ = 1̃ β₀(t), β₀ ∈ R^l (6.18); β̇₀ = −Q̄(t) β₀ with Q̄ = ν Q̂ 1̃ the aggregated generator (6.22–6.23);
ψ₀ → 0 forces P ψ₀(0) = 0 hence β₀(T) = ν(T) u₀ (6.28).
6.3.2 Higher terms: ϕ_i = 1̃ β_i + ϕ̃_i with ϕ̃_i = (Q_vᵀ Q_v)^{−1} Q_vᵀ b̄_{i−1} the ν-orthogonal particular solution
(6.31; Lemma 6.3 on rank); β̇_i = −Q̄ β_i − ν ϕ̃̇_i (6.32); the terminal condition from decay of ψ_i:
β_i(T) = ∫₀^∞ ν(T) r_i(s) ds (6.37). Proposition 6.4: ϕ_i ∈ C^{n+2−i}. Proposition 6.5: |ψ_i| ≤ C e^{−κ_i τ}.

### 6.4 Error estimates (p. 246)
L^ε f = ε f' + (Q̃ + εQ̂) f (6.39). Lemma 6.6: |L^ε ξ| ≤ Cε^{k+1}, ξ(T) = 0 ⇒ sup|ξ| = O(ε^k) (bounded transition
matrix Σ^ε). Proposition 6.7 and **Theorem 6.8** (p. 250): sup_[0,T] |u^ε − Φ_n − Ψ_n((T−t)/ε)| = O(ε^{n+1}) under
(A6.1)–(A6.2), with the explicit construction listed.

### 6.5 Transient states (p. 250)
(A6.3) Q̃_*(t) Hurwitz. ϕ₀^* = Σ β₀^i a_i with a_i = −Q̃_*^{−1} Q̃_*^i 1 (6.45–6.46); 1̃_*(t), ν_*(t), P_* (6.48–6.49);
Q̄_* = ν_* Q̂ 1̃_* (6.50); Lemma 6.9 decay for the partitioned layer system; β₀(T) = ν_*(T) u₀ (6.54).
**Theorem 6.10** (p. 254): the same construction and O(ε^{n+1}) bound under (A6.1)–(A6.3).

### 6.6 Remarks (p. 255)
Example 6.11: Q₀(t)/ε + Q̂(t) with Q₀ irreducible gives u^ε(t) = ν₀(t) + O(ε + e^{−κ₀(T−t)/ε}) [as printed; the
leading term is the ν₀-average of u₀ when u₀ is a probability vector]. 6.6.1 Related problems: the fully
degenerate switching ODE Ẋ = b(X, α^ε), backward equation −∂_t u = b ∂_x u + Q^ε u (6.56–6.57) on a compact set,
expansion sketched, details omitted, refers to Khasminskii & Yin [117, 118]. No potential term here either.

### 6.7 Notes (p. 257)
Motivation from nearly completely decomposable systems (Simon–Ando, Courtois). "The result of this chapter is
based on Yin and Nguyen [226]." Future direction: generator depending on the continuous state, Q(x, t).

## Part III: Controls, Numerical Methods, and LQG with Switching

## Chapter 7: Markov decision problems (pp. 261–283)

### 7.1 Introduction (p. 261)
Finite-state continuous-time MDPs with weak and strong interactions. Hierarchical control: replace the fast
process by its average under the quasi-stationary distributions, solve the limit problem, lift the decision back.
Literature: Derman [46], Ross [184], White [218], Guo and Hernández-Lerma [78]; Gershwin [71], Sethi and Zhang
[192]; Simon and Ando [196]. Claimed advantages over Chapter 8: no Lipschitz condition on the limit control,
better convergence rates, long-run average cost handled. Nothing on pricing.

### 7.2 Problem formulation (p. 263)
Controlled generator Q^ε(u) = Q̃(u)/ε + Q̂(u) (7.1), Q̃ = diag(Q̃¹, …, Q̃ˡ), feedback u(i) ∈ Γ compact; admissible
class A_f. Discounted cost J^ε(i, u) = E ∫₀^∞ e^{−ρt} G(x^ε(t), u(x^ε(t))) dt; problem P^ε (7.2); DP equation
ρv^ε(i) = min_u {G(i, u) + Q^ε(u) v^ε(·)(i)} (7.3), unique solution by Theorem A.30, optimal minimiser by A.31.
Example 7.1 (p. 264): two failure-prone machines in a flowshop, four states, first machine fast (rates λ₁/ε, μ₁/ε),
second slow (λ₂, μ₂), control u = preventive-maintenance rate.

### 7.3 Limit problem (p. 265)
Limit control set Γ̄ = Γ₁ × … × Γ_l, each U^k assigning a control to every substate of group k. Q^ε(U) obtained by
replacing u in row i by u_i (7.4). Remark 7.2: the chain generated by Q^ε(u(x)) and the one generated by
Q̄^ε(U) with U = (u(s_ij)) have the same law. Assumptions (A7.1) continuity in u, Q̃₀^k(U^k) weakly irreducible
for every U^k and irreducible for some U₀; (A7.2) G continuous. Aggregated generator
Q̄(U) = ν(U) Q̂₀(U) 1̃ (7.5), row k depends only on U^k. Averaged cost Ḡ(k, U^k) = Σ_j ν_j^k(U^k) G(s_kj, u^kj).
Limit problem P⁰ (7.6) with DP equation (7.7) in l unknowns instead of m (Remark 7.3). Example 7.4 computes
ν^1, ν^2 and the 2 × 2 aggregated generator with rates η₁(U), η₂(U) explicitly.

### 7.4 Asymptotic optimality (p. 269)
Lemma 7.5: any subsequential limit of v^ε(i) is constant on each group (multiply DP by ε, Lemma A.39 on
Q̃₀^k v ≥ 0). Theorem 7.6: v^ε(i) → v(k) for i ∈ M_k (ν^k-average the DP inequality; reverse direction by
compactness of the minimisers). Constructed control u_c(x) = Σ I_{x = s_kj} u^kj_* (7.10). Lemma 7.7 (from
Corollary 4.31): |P(x^ε(t) = s_kj) − ν_j^k(U) f_k(t)| ≤ K(ε(t + 1) + e^{−κ₀ t/ε}) (7.11) with f solving
ḟ = f Q̄(U). Theorem 7.8: |J^ε(i, u_c) − v^ε(i)| → 0, bound O(ε) in (7.14). Remark 7.9: the t-dependent bound
does not serve long-run average costs.

### 7.5 Convergence rate and error bound (p. 272)
Theorem 7.10 (Γ finite): v^ε(i) − v(k) = O(ε) and J^ε(i, u_c) − v^ε(i) = O(ε). Proof partitions (0, 1) into sets
E_j on which the ε-optimal control is the constant γ_j. Remark 7.11: for Sethi–Zhang's production models the best
rate is √ε; here O(ε) because the dynamics are the chain itself, not an ODE driven by it.

### 7.6 Long-run average cost (p. 274)
Assumptions (A7.3) irreducibility of every Q̃₀^k(U^k) and of Q̄^ε(U) for small ε, (A7.4) Γ finite. Problem
P^ε_av, DP equation λ^ε = min_u {G(i, u) + Q^ε(u) h^ε(·)(i)} (7.15); Theorem 7.13 existence, uniqueness up to a
constant, optimality (Ross). Limit problem P⁰_av. Lemma 7.14: Q̄(U) is irreducible (Gaussian-elimination
argument on the aggregated matrix, rank l − 1, then positivity of ν̄). Theorem 7.15 verification for the limit.
Lemma 7.16: quasi-stationary distribution of Q̄^ε(U) equals ν₀(U) = (ν¹ ν̄¹, …, νˡ ν̄ˡ) + O(ε), the O(ε) from the
rationality in ε of the solution via the Cramer-type formula of Remark 4.10. Theorem 7.17: J^ε(u_c) − λ^ε = O(ε),
and λ^ε = λ⁰ + O(ε) (7.23). Remark 7.18: if Q̂(u) is irreducible the constructed control is exactly optimal.

### 7.7 Computational procedures (p. 280)
Discounted: rewrite (7.7) as v(k) = min {Ḡ/(ρ + |q̄_kk|) + Σ q̄_kk'/(ρ + |q̄_kk|) v(k')} (7.25), i.e. a discrete-time
DP with discount ϱ = max |q̄_kk|/(ρ + |q̄_kk|) < 1 and transition weights p̃_kk'; solve by successive
approximation, policy improvement or LP (Ross). Long-run average: analogous discrete version with
λ̃⁰ = λ⁰ max 1/|q̄_kk|; Kushner and Dupuis [141, Ch. 6].

### 7.8 Notes (p. 282)
Based on Zhang [249]. Related: Delebecque and Quadrat [44], Phillips and Kokotovic [175]; Guo and Hernández-Lerma
[78] for advanced criteria; Sethi and Zhang [193] for non-finite spaces; Costa and Dufour [33, 34] on singularly
perturbed piecewise-deterministic Markov processes and general-state MDPs; risk-sensitive costs in Zhang [248],
Fleming and Zhang [66].

## Chapter 8: Stochastic control of dynamical systems (pp. 285–318)

### 8.1 Introduction (p. 285)
Piecewise-deterministic control system dx^ε/dt = f(x^ε, α^ε, u) (8.1), α^ε generated by Q̃/ε + Q̂ (weak and strong
interactions among recurrent groups), discounted (8.2) or finite-horizon (8.3) costs. Sections 8.2–8.5 use dynamic
programming and viscosity solutions with time-independent generators (8.4); §8.6 the weak-convergence and
relaxed-control approach allowing Q^ε(t).

### 8.2 Problem formulation (p. 287)
(A8.1) f(x, α, u) = f₁(x, α) + f₂(x, α) u, bounded and Lipschitz, Γ convex compact; (A8.2) G jointly convex,
locally Lipschitz with polynomial growth; (A8.3) each Q̃^k irreducible. Definitions 8.1–8.2 admissible and feedback
controls; problem P^ε (8.5). Example 8.3: one machine, one part type, capacity c^ε ∈ {0, 1} and demand
z^ε ∈ {z₁, z₂}, surplus x^ε, cost c⁺x⁺ + c⁻x⁻; the two orderings of the 4 × 4 generator make either capacity or
demand the fast component.

### 8.3 Properties of the value functions (p. 290)
HJB (8.6): ρv^ε = min_u {f ∂_x v^ε + G + Q^ε v^ε(x, ·)(α)}; Hamiltonian (8.7); Remark 8.4 on inner-product
notation. Lemma 8.5: v^ε locally Lipschitz, unique viscosity solution (Theorem A.30), convex when f is
x-independent. Lemma 8.6: subsequential limits of v^ε depend only on the group index k (first-jump argument
using P(α^ε(τ^ε) = s_i₀j | τ^ε) → −q̃_j₀j/q̃_j₀j₀ and Lemma A.39). Remark 8.7 the smooth shortcut. Limit problem
P⁰ (8.14) with averaged f̄(x, k, U) = Σ_j ν_j^k f(x, s_kj, u^kj), Ḡ likewise, chain generated by
Q̄ = diag(ν¹, …, νˡ) Q̂ 1̃; l HJB equations instead of m. Theorem 8.8: v^ε(x, α) → v⁰(x, k) by showing the
limit is a viscosity sub- and supersolution of (8.15) through ν^k-weighted Hamiltonians (8.16), then uniqueness.
Remark 8.9: linearity in u is only needed for the feedback construction.

### 8.4 Asymptotic optimal controls (p. 296)
(A8.4): f = f(α, u), G twice differentiable in u with ∂²G/∂u² ≥ c₀ I, second-order bound in x. Lemma 8.10: v⁰
convex and C¹, the limit optimal feedback U*(x) locally Lipschitz (Lemma A.32), and optimal. Construction (8.19):
u(x, α) = Σ I_{α = s_kj} u*_j(x, k). Proof of near-optimality via an intermediate process driven by the aggregated
chain ᾱ^ε: E|x^ε − x̄^ε| → 0 by integration by parts and Theorem 5.25 (8.20); x̄^ε → x* w.p.1 by weak convergence
of ᾱ^ε to ᾱ (Theorem 5.27) and Skorohod representation (8.21–8.22). Theorem 8.11:
|J^ε(x, α, u^ε) − v^ε(x, α)| → 0.

### 8.5 Convergence rate (p. 300)
Restricted to Q̃ irreducible (single block) (A8.5), f = B₁(α) u + B₂(α), G = G₁(x) + G₂(α, u) convex (A8.6).
Theorem 8.12: |v^ε(x, α) − v⁰(x)| ≤ K(1 + |x|^κ) √ε. Upper bound from Corollary 5.21 (E|x^ε − x|² = O(εt²)) and the
Theorem 4.5 expansion for P(α^ε(t) = j) = ν_j + O(ε + e^{−κ₀t/ε}); lower bound with the conditional control
u_j(t) = E[u^ε(t) | α^ε(t) = j] and Jensen, giving −K(1 + |x|^κ) ε. Remark 8.13: √ε is sharp (Sethi–Zhang
one-dimensional example). Open-loop constructed controls inherit the √ε bound; feedback error bounds only under
extra conditions such as large ρ. Example 8.14 continues Example 8.3 with fast demand: limit demand
z̄ = ν₁¹ z₁ + ν₂¹ z₂, explicit threshold x* from the negative eigenvalue a₋ of a 2 × 2 matrix A₁, bang-bang
optimal control; the limit control is not Lipschitz so Theorem 8.11 does not apply directly, but near-optimality
still holds as in Sethi and Zhang [192, Ch. 5].

### 8.6 Weak convergence approach (p. 306)
8.6.1–8.6.2: finite-horizon problem (8.26) rewritten with a relaxed control m̃ (8.27); integral form of the
dynamics. 8.6.3: (A8.7) Q^ε(t) = Q̃(t)/ε + Q̂(t) with each Q̃^k(t) weakly irreducible (measurability suffices,
Remark 8.15); (A8.8) Γ compact, f continuous, Lipschitz, linear growth, f(x, α, Γ) convex; (A8.9) G bounded
continuous. Limit problem P⁰ in relaxed form with ν_j^i(t) I_{ᾱ(t) = i}. Lemma 8.16 a priori bounds and
continuity of the control-to-state map; Lemma 8.17 existence of optimal relaxed controls, δ-optimal ordinary and
piecewise-constant Lipschitz feedback controls (chattering lemma). Theorem 8.18: tightness of (x^ε, m̃^ε), limit
satisfies P⁰ (decomposition (8.29), the occupation-measure term killed by Theorem 5.52, weak convergence of ᾱ^ε
by Theorem 5.53, martingale-problem characterisation (8.31)), and J^ε(m̃^ε) → J(m̃). Remark 8.19: direct
averaging in the sense of Kushner [139, Ch. 5]; nonstationary ν^i(t) handled through Chapter 5. Theorem 8.20:
v^ε → v⁰, and a δ-optimal Lipschitz feedback for P⁰ is (δ + o(1))-optimal for P^ε (8.32–8.36).

### 8.7 Notes (p. 316)
Based on Zhang, Yin and Boukas [256] and Yin and Zhang [233, 234]. Finite-horizon extension (Zhang and Yin
[251], Zhang [247] for the irreducible case). Multilevel hierarchies P^{ε₁,…,ε_n₀} → P^{0,ε₂,…} → … as a
multiresolution scheme (Sethi and Zhang [193]). Manufacturing literature; Saksena, O'Reilly and Kokotovic [186]
survey. Infinite-horizon relaxed formulations. Weak-convergence lineage: Kushner and Runggaldier [142], Bensoussan
[8], Kushner [140], Kokotovic [126, 127]. Phillips and Kokotovic [175] expand the cost function itself; "it will
be interesting to see if an asymptotic expansion of the cost function can be derived under the formulation of
the current chapter" (p. 318). Costa and Dufour [33, 34]. Hybrid filtering and the IMM algorithm (Blom and
Bar-Shalom [17], Li [146]) as an open application.

## Chapter 9: Numerical methods for control and optimization (pp. 319–339)

### 9.1 Introduction (p. 319)
Two angles: Kushner's finite-difference (Markov-chain approximation) method for the HJB of the limit problem, and
stochastic optimisation over threshold (hedging) policies for long-run average costs (Kimemia and Gershwin,
Caramanis and Liberopoulos [24]).

### 9.2 Numerical methods for optimal control (p. 320)
Problem P with chain generated by Q (no ε); HJB (9.1). Upwind one-sided differences in each x_i according to the
sign of f_i, giving the discrete DP (9.2) with denominator ρ + |q_αα| + Σ|f_i|/Δx_i, i.e. an auxiliary discounted
MDP solvable by value or policy iteration. Theorem 9.1: v^Δ → v as Δ → 0 (contraction for fixed Δ; upper and
lower semicontinuous envelopes v*, v_* are viscosity sub/supersolutions; uniqueness Theorem A.24). Remark 9.2:
Yan and Zhang [221] show controls built from an approximate value function are nearly optimal. Remark 9.3: apply
to the limit problem, not to P^ε.

### 9.3 Optimization under threshold policy (p. 324)
9.3.1: long-run average cost (9.6) for dx/dt = f(x, α, u); Definition 9.4 threshold policies u = Σ c_i I_{x ∈ A_i}.
Example 9.5 (Bielecki and Kumar [12]): one machine, demand z, cost c⁺x⁺ + c⁻x⁻; optimal policy is a hedging point
θ* with explicit formula and explicit optimal cost. Algorithm (9.7): θ_{k+1} = θ_k − (η/T) ∫_{kT}^{(k+1)T}
g(θ_k, ξ(t)) dt with an IPA gradient estimate (Ho and Cao [87], Glasserman [74]); for Example 9.5 the integrand is
c⁺ I_{x > 0} − c⁻ I_{x < 0}. Remark 9.6: T need not grow with η (Kushner and Vázquez-Abad [143]); a little bias is
harmless. 9.3.2 Convergence: (A9.1) ergodicity of conditional averages of g toward ∇J, (A9.2) continuity in
expectation, (A9.3) uniform integrability. Theorem 9.8: interpolated iterates θ^η(·) are tight in D and converge
weakly to the ODE θ̇ = −∇J(θ) (9.8); proof by N-truncation, martingale characterisation (9.11–9.14), Kushner
[139, Thm 2.2]. Theorem 9.10: with a unique asymptotically stable point θ* and bounded-in-probability iterates,
θ^η(t_η + ·) ⇒ θ*. 9.3.3 Examples: 9.11 recovers Bielecki–Kumar's θ* = 66.96, cost 142.89, with θ̃* = 67.23
(95% CI [66.64, 67.80]) and J = 139.43 from 100 replications, η = 0.5, θ₀ = 20; 9.12 two-machine cascade
(Yan, Yin and Lou [220]), surplus variables s₁ = x₁ + x₂, s₂ = x₂, two-threshold policy, IPA gradient integrands
g₁, g₂ written with stopping times τ_k^i, γ_k^i (Kushner and Vázquez-Abad); Figure 9.2 shows iterates from two
initial points converging to the same region. 9.3.4 Error bounds: Theorem 9.13, perturbed-Liapunov argument
gives lim sup_k V(θ_k) = O(η) (9.20) under mixing-type bounds (9.19); Remark 9.15 on the √η-scaled local
diffusion limit.

### 9.4 Notes (p. 339)
Kushner [138], Kushner and Dupuis [141] for the numerics (value/policy iteration, Jacobi, Gauss–Seidel,
multigrid as open topic); Yin, Yan and Lou [228], Yan, Yin and Lou [220] for the threshold optimisation;
Kimemia and Gershwin [121]; Kushner and Yin [145, Ch. 9]. Computations by Houmin Yan.

## Chapter 10: Hybrid LQG problems (pp. 341–372)

### 10.1 Introduction (p. 341)
Linear system with Markov-modulated coefficient matrices and additive Brownian noise; value functions solve a
system of Riccati equations coupled through the generator; large chains make them costly, so use two-time-scale
averaging; three cases (recurrent, transient, absorbing); one-dimensional numerical example. Based on Zhang and
Yin [254].

### 10.2 Problem formulation (p. 343)
dx = [A(α) x + B(α) u] dt + σ dw (10.1), finite horizon, quadratic cost (10.2) with M(i) ≥ 0, N(i) > 0, D > 0;
Q^ε = Q̃/ε + Q̂ (10.3); α^ε independent of w.

### 10.3 Optimal controls (p. 344)
HJB system (10.4) with ½ tr(σσ' ∂²v) and Q^ε v(s, x, ·)(i). Quadratic ansatz v^ε = x' K^ε(s, i) x + q^ε(s, i)
(10.6) gives the coupled Riccati equations K̇^ε = −K^ε A − A' K^ε − M + K^ε B N^{−1} B' K^ε − Q^ε K^ε(s, ·)(i)
(10.7), K^ε(T, i) = D, and q̇^ε = −tr(σσ' K^ε) − Q^ε q^ε (10.8); optimal control u^{ε,*} = −N^{−1} B' K^ε x (10.9).
[Note for the paper: this is the only place in the book where an ODE system of the form "Riccati + Q^ε acting on
the regime index" is expanded; the potential-free structure is the same as chapter 6, the unknown is a matrix,
and the expansion is only to leading order.]

### 10.4 Recurrent states (p. 345)
Q̃ = diag(Q̃¹, …, Q̃ˡ) weakly irreducible blocks (10.10); aggregated chain ᾱ^ε ⇒ ᾱ generated by
Q̄ = diag(ν¹, …, νˡ) Q̂ diag(1_{m₁}, …, 1_{m_l}); occupation bound (10.11) E(∫[I_{α^ε = s_kj} − ν_j^k I_{ᾱ^ε = k}] β)²
= O(ε). 10.4.1 Theorem 10.1: K^ε(s, s_kj) → K(s, k), q^ε → q(s, k) uniformly, where K solves the l-dimensional
Riccati system (10.12) with ν^k-averaged coefficients (Ā, M̄, and B N^{−1} B' averaged as a product) and Q̄, q
solves (10.13). Lemma 10.2 K^ε positive definite (representation through the principal matrix Ψ(t, s)); Lemma
10.3 bounds |K^ε|, |q^ε| ≤ C₀ e^{k₀T}(T + 1) via v^ε(s, aξ, i)/a²; Lemma 10.4 uniform Lipschitz continuity in s
(shift arguments with Dynkin's formula, (10.18)–(10.22)). Proof of Theorem 10.1: Arzelà–Ascoli, multiply the
integral Riccati equation by ε to get Q̃^k K⁰(s, ·) = 0 (10.24) hence K⁰ constant on groups, then ν^k-average the
equation and use uniqueness. 10.4.2 Nearly optimal controls: limit HJB (10.25) with f̄(s, x, k, U) =
Ā(k) x + Σ_j ν_j^k B(s_kj) u^kj and N̄(k, U); limit optimal u^{kj*} = −N^{−1}(s_kj) B'(s_kj) K(s, k) x;
constructed u^ε (10.26) = −N^{−1}(α) B'(α) K(s, k) x for α ∈ M_k, and (10.27–10.28) using only the group index
when B, N are constant within groups. Theorem 10.5: both are nearly optimal (trajectory convergence (10.29) via
weak convergence of ᾱ^ε, Skorohod and (10.11)). Remark 10.6: 40 states in 5 groups, n₁ = 3, reduces 240 Riccati
unknowns to 30. Example 10.7: Q̃ irreducible gives a single classical Riccati equation (10.31) with averaged
coefficients. Remark 10.8: σ = σ(α^ε) handled the same way, with σσ̄'(k) averaged.

### 10.5 Transient states (p. 358)
Q̃ = [[Q̃_r, 0], [Q̃₀, Q̃_*]] (10.35), Q̃_* Hurwitz. Aggregated generator Q_* = diag(ν)(Q̂₁₁ 1̃ + Q̂₁₂ (a_{m₁}, …, a_{m_l}))
(10.36) with a_{m_j} = −Q̃_*^{−1} Q̃_*^j 1 (10.37) the absorption probabilities from transient states; ᾱ^ε assigns a
transient state to group ξ_j with law a_{m_j}. Bounds (10.38): recurrent occupation error O(ε), transient
occupation O(ε²). Theorem 10.9: K^ε(s, s_kj) → K(s, k) and K^ε(s, s_*j) → K*(s, j) = Σ_k a_{m_k}(j) K(s, k);
Riccati system (10.39)–(10.40) with Q_*. Constructed controls (10.41)–(10.42); Theorem 10.10 near-optimality.

### 10.6 Absorbing states (p. 362)
Q̃ = diag(Q̃¹, …, Q̃ˡ, 0_{m_a × m_a}) (10.43); Q_a = diag(ν¹, …, νˡ, I_{m_a}) Q̂ 1̃_a (10.44); aggregated chain
(10.45) keeps absorbing states as their own super-states l + j. Theorem 10.11: limits K(s, k) for k ≤ l and
K(s, l + j) for absorbing states, Riccati system (10.47)–(10.48) with Q_a. Controls (10.49)–(10.50); Theorem
10.12 near-optimality.

### 10.7 A numerical example (p. 366)
Two-state chain Q^ε = (1/ε) [[−0.5, 0.5], [0.5, −0.5]], one-dimensional system with A(1) = 0.5, A(2) = −0.1,
B(1) = 1, B(2) = 2, σ = 1, M = N = D = 1, T = 5, h = 0.01, 100 sample paths. Table 10.1: |K^ε − K| ≈ 2.2–2.5 ε,
|x^ε − x̄^ε| ≈ 0.1–0.2 ε, |v^ε − v| ≈ 4.2–5.1 ε, |J^ε − v^ε| ≈ 0.4–4.2 √ε for ε = 0.1, 0.01, 0.001, 0.0001. Figures
10.1–10.4 sample paths of K^ε(·, i), K(·), α^ε, x^ε, x. "Very good approximation … with only half of the
computational effort."

### 10.8 Remarks on indefinite control weights (p. 368)
Switching diffusion with control in the diffusion coefficient dx = [A x + B u] dt + C(α^ε) u dw (10.52), cost
(10.53) with D(α^ε(T)); indefinite N allowed if "not too negative" (Chen, Li and Zhou [26], Yong and Zhou [246]);
backward SDE techniques (Pardoux and Peng [171]). Limit Riccati (10.54) with (N̄ + C' K C)^{−1} and terminal
D̄(k) = Σ_j ν_j^k D(s_kj). Details in Liu, Yin and Zhou [148]. Near-optimality of the constructed control stated.

### 10.9 Notes (p. 371)
Based on Zhang and Yin [254]. Hybrid LQG literature: Blair and Sworder [15], Caines and Chen [21], Mariton
[154], Rishel [181], Ji and Chizeck [97, 98], Zhang [250]; Chow, Menaldi and Robin [29]. Two-time-scale roots:
Delebecque [43], Delebecque and Quadrat [44], Phillips and Kokotovic [175], Pan and Başar [164, 167]. Finance
applications named: Markowitz mean–variance selection with regime switching, Zhou and Yin [259] (SIAM J. Control
Optim. 42, 2003) and Yin and Zhou [243] (IEEE TAC 49, 2004).

## Appendix A: Background materials (pp. 373–406)

### A.1 Properties of generators (p. 373)
Lemma A.1 Gerschgorin. Lemma A.2: weakly irreducible Q has a simple zero eigenvalue, all others with negative
real part, and |exp(Qs) − 1ν| ≤ K e^{−κ̃s} with κ̃ = −½ max Re λ_i (proof through Chung's limit theorems and the
Jordan form). Remark A.3: literature (Cox and Miller, Iosifescu, Doob, Karlin and McGregor, Keilson's Green's
function approach, Campbell and Rose on convergence of exp((A + B/ε)t) iff B semistable). Lemma A.4: a unique
solution of νQ = 0, ν1 = 1 is nonnegative. Lemma A.5: rank Q = m − 1 implies weak irreducibility. Lemma A.6:
continuous weakly irreducible Q(t) on [0, T] has a uniform spectral gap Re λ_t ≤ −κ.

### A.2 Weak convergence (p. 376)
Definitions A.7–A.9 (weak convergence, Skorohod topology on D([0, ∞); Rʳ), tightness); Theorem A.10 Prohorov;
Theorem A.11 Skorohod representation; Definition A.12 and Theorems A.13–A.14 martingale problem and its
uniqueness (Ethier and Kurtz); Theorem A.15 tightness for ODE solutions with uniformly integrable right-hand
sides; p-lim and the operator A^ε (Ethier–Kurtz/Kushner); Lemma A.16 mixing inequalities; Lemma A.17 Kurtz'
tightness criterion (A.4)–(A.5), Remark A.18; Lemma A.19 perturbed-test-function tightness and the N-truncation
device (A.8); Lemma A.20 weak convergence via perturbed test functions (A.9)–(A.10); Theorem A.21 a Lipschitz
continuous-time martingale is constant.

### A.3 Relaxed control (p. 382)
Space IM of measures on Γ × [0, ∞) with m̃(Γ × [0, t]) = t; admissibility as progressive measurability; derivative
m̃_t (A.11); metric (A.12); relaxed feedback controls; Theorem A.22 chattering theorem (Kushner [140]): relaxed
controls approximated by piecewise-constant finite-valued ordinary controls to accuracy γ in state and cost.

### A.4 Viscosity solutions of HJB equations (p. 383)
Definition A.23 (polynomial growth, sub/supersolution tests with C¹ test functions, per regime α₀). Theorem A.24
uniqueness for ρv = min_u {b ∂_x v + G + Qv(x, ·)(α)} (A.14) under (A8.1)-type conditions; proof by doubling
variables with penalty |x₁ − x₂|²/δ and weight γη(x), η = exp(a(1 + |x|²)^{1/2}), the coupling term (A.22) handled
by the maximum over α. Lemma A.25 (Crandall, Evans and Lions) touching test functions.

### A.5 Value functions and optimal controls (p. 388)
One-dimensional model with chain-adapted admissible controls (Definitions A.26–A.27). Lemma A.28: v locally
Lipschitz if G is; convex if G jointly convex and b x-independent. Lemma A.29 dynamic programming principle at
stopping times, full proof (countably-valued τ first, then dyadic approximation). Theorem A.30: v is the unique
viscosity solution of (A.29) (Dynkin's formula up to the first jump time, ψ(x, α) test function). Theorem A.31
verification theorem (Fleming and Rishel). Lemma A.32: minimiser of uV(x) + c(u) with c'' > 0 is locally Lipschitz.

### A.6 Miscellany (p. 399)
Definition A.33 convexity; Lemma A.34 (Clarke) convex ⇒ locally Lipschitz and a.e. differentiable; Theorem A.35
Arzelà–Ascoli; Lemma A.36 weak sequential compactness in Hilbert space; Lemma A.37 Fredholm alternative and
Corollary A.38 for yB = b (solvable iff b ⊥ null(B)); Lemma A.39: irreducible Q and f(i) ≤ Σ_{j ≠ i} γ_ij f(j)
force f constant (elimination on the normalised matrix Q₁); Lemmas A.40–A.41 continuity of integrals under the
Skorohod topology; Lemma A.42 conditional expectation on a discrete variable. Proof of Theorem 4.41 (transient
states, p. 403): Lemma A.43 compares the fundamental matrix of ẏ = y A^ε(t) with exp(A(t − s)) when
|A^ε(t) − A| ≤ Kε(t + 1) (bound (A.44) with (t + 1)² and ε(t + 1)⁴ e^{Kε(t+1)²} factors); the transient
initial-layer integral (A.46) evaluated by the substitution s → s/ε.

## Bibliography (pp. 407–423, 260 entries) and Index (pp. 425–427)

Finance-related entries only: [6] Barone-Adesi and Whaley 1987 (American options, J. Finance, cited in chapter 3
for the Markov-modulated diffusion example); [223] Yin 2009 Asymptotic Analysis 65, 203–222 (option price
expansions under fast switching, cited on p. 236); [243] Yin and Zhou 2004 and [259] Zhou and Yin 2003
(Markowitz with regime switching); [255] Zhang and Yin, nearly optimal asset allocation in hybrid stock
investment models; [257] Zhang, Yin and Liu, near-optimal selling rule for a two-time-scale market model;
[229] Yin and Yang, two-time-scale jump diffusions with Markov switching; [244] Yin and Zhu, Hybrid Switching
Diffusions (Springer 2010). Machinery lineage: Khasminskii [111–114], Khasminskii and Yin [115–118], Khasminskii,
Yin and Zhang [119, 120] (asymptotic expansions of singularly perturbed chains, 1996–1997); Il'in and Khasminskii
[93]; Il'in, Khasminskii and Yin [94]; Griego and Hersh [76, 77], Hersh [85], Pinsky [176, 177] (random
evolutions); Kurtz [135]; Papanicolaou [168–170]; Delebecque, Quadrat, Kokotovic [43–45, 175]; Pervozvanskii and
Gaitsgori [69, 174]; Courtois [35]; Simon and Ando [196]; Di Masi and Kabanov [47, 48]; Kabanov and
Pergamenshchikov [100]. [232] Yin, H. Zhang and Q. Zhang, "Applications of two-time-scale Markovian systems",
preprint 2012, is the only entry newer than 2010 besides the authors' own discrete-time book [238]. No entry by
Cotton, Naik, Elliott–Mamon, Buffington–Elliott, Hamilton or Sircar–Papanicolaou. The index (p. 425–427) has
"Random evolution, 41" and "Two-time scale, 5, 81, 341, 362"; no entry for option, bond, Feynman–Kac, potential
or discount.

## Verdict after the full read

Unchanged from the chapter 4/6 verdict above, and now unconditional: nowhere in the book is the backward equation
carrying a potential or discount term expanded. The closest objects are (i) the potential-free backward equation
of chapter 6 (Yin and Nguyen 2009), (ii) the random-evolution remark of §3.3.5, (iii) the coupled Riccati system of
chapter 10, expanded only to leading order, and (iv) the unexpanded cost-function remark on p. 318 asking whether
"an asymptotic expansion of the cost function can be derived". The candidate for overlap with the thesis remains
Yin (2009) [223], which the book cites but does not reproduce.
