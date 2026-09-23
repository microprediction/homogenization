# Epidemic models in fast-switching or random environments: literature memo

Prepared 2026-09-23 for the homogenization site. Covers compartmental models (SIR, SIS, SEIR, structured) whose
rates are switched by a periodic or random environment, what is known, and what the engine and the first-order rule
could add. Every reference in the list at the end was checked against Crossref (title, first author, year, DOI) or the
arXiv API. The notes on what each paper says come from its abstract, or from the full text where one was open; the
list says which. Numerical checks were run for this memo (scripts in the session scratchpad, not in the repo).

## Summary

- The distinction the site draws, expected growth versus almost-sure (typical) growth, is standard in this
  literature and is the main theme of the random-environment R0 papers. Bacaër and Khaladi (2013) say so directly: their
  R0 decides growth or decay of simulated populations, while an earlier parameter (Hernandez-Suarez et al. 2012) decides
  growth of the expectation. Gray et al. (2012) give both for SIS with telegraph noise: a next-generation-matrix R0 and a
  different almost-sure threshold, the π-average of β N over the π-average of μ + γ.
- Extinction is governed by the almost-sure exponent, not by the mean. That holds for branching processes in random
  environments (Smith and Wilkinson 1969: extinction iff E log(mean offspring) ≤ 0), for SIS/SIRS with Markov switching
  (Gray et al. 2012; Li, Liu and Cui 2017), and for the multitype SIS with switching (Prodhomme and Strickler 2024: the sign
  of the top Lyapunov exponent of the linearized switched system decides).
- Fast-switching asymptotics exist and are recent. Monmarché, Schreiber and Strickler (2025, Bull. Math. Biol.), using
  Monmarché and Strickler (2025, J. Appl. Probab.), give the first-order fast-switching correction to the almost-sure
  Lyapunov exponent of x' = A(y_t) x for Metzler matrices, for both Markov and periodic switching. Their Markov formula
  is a Green-Kubo expression projected off the Perron direction (shown below). For periodic switching the first-order term
  is a sum of commutators and vanishes for two environments. **This is the closest prior art; the site should cite it.**
- As far as this search found, nobody has written the matching first-order formula for the **expected** growth rate
  (first moment) in the Green-Kubo form, or pointed out that the gap between expected and almost-sure growth is the
  Green-Kubo integral of the projected rate ȳᵀA(y)x̄. Both follow in a few lines from the site's rule and were checked
  numerically here. Treat them as new until a more targeted search says otherwise.
- The ideas note needs one correction. "Fluctuation raises the mean growth" holds for SIR, where one infected
  compartment makes the problem scalar. It fails in SEIR when only β switches. β multiplies a nilpotent operator
  (E is fed by I), so the first-order correction to the **mean** growth is zero. The **almost-sure** growth still drops,
  by K_ββ (ȳ_E x̄_I)². The order of regimes (the antisymmetric part of K) enters SEIR when β and γ both switch, because
  their operators do not commute.

## 1. R0 and growth in periodic and random environments

**Periodic.** Heesterbeek and Roberts (1995) gave threshold quantities for periodic compartmental systems. Bacaër and
Guernaoui (2006) defined R0 for periodic environments as the spectral radius of a next-generation operator on periodic
functions. Wang and Zhao (2008) made this general for compartmental ODEs and proved that it is the invasion threshold. The
standard reading of their paper, visible in the abstract snippet but not checked in the full text (paywalled), is that
the time-averaged autonomous R0 is right when the new-infection matrix and the transition matrix are both diagonal, and
can be too high or too low otherwise. Thieme (2009) placed spectral bound and R0 in an abstract setting that includes time
heterogeneity (evolution semigroups). Inaba (2012, 2019) develops the generation-evolution view of R0 in time-heterogeneous
settings (titles and DOIs checked, content not read).

Bacaër and Ouifki (2007) find, for a sinusoidal coefficient, that growth rate and R0 are the largest roots of continued-
fraction equations. In their SEIS example with a fixed latent period, the threshold depends on the amplitude of the
fluctuations as well as the mean contact rate. Bacaër (2007) expands R0 for a vector population p0(1 + ε cos(ωt − φ))
with small ε. The first term uses the average vector population. The second is O(ε²), at most (ε²/8)%, and always
lowers R0. That is a small-amplitude expansion, not a fast one. Mitchell and Kribs (2017) compare the ways of computing
periodic R0. Their summary: seasonality alone cannot change persistence, and it does so only together with
"non-hierarchical heterogeneity" in the infected classes. That fits the scalar case below, where the Floquet exponent is
exactly the averaged rate.

For SIR/SEIR with one class of infectives and seasonal β, the linearized equation is scalar. Its exponent over a period is
exactly mean(β)S0/N − γ, and the periodic R0 is the mean contact rate over γ. Periodic forcing matters for thresholds only
when there are several infected compartments. Dietz (1976), Grassly and Fraser (2006), Altizer et al. (2006) and Keeling and
Rohani (2008) cover seasonal forcing above threshold (resonance, multi-annual cycles). Keeling, Rohani and Grenfell (2001)
explain forced dynamics as switching between attractors. That is nonlinear and outside the linear engine.

**Random.** Gray, Greenhalgh, Mao and Pan (2012), read in full from the author copy, analyse SIS with a two-state (then
finite) Markov chain switching β, μ and γ. They derive a next-generation-matrix "R0S" by counting infections between switches.
That is the threshold of the mean linear system. They then prove that almost-sure extinction and persistence are decided by
a **different** quantity, T0S = (π1β1N + π2β2N)/(π1(μ1+γ1) + π2(μ2+γ2)), that is, by the sign of π·(βN − μ − γ). Greenhalgh,
Liang and Mao (2016) extend this to SIRS, and Li, Liu and Cui (2017) to SIRS with nonlinear incidence (R0 = π-average of
the per-regime R0 values when only β switches). Economou and Lopez-Herrero (2015) study the deterministic SIS in a Markovian
environment (summary only read). Artalejo, Economou and Lopez-Herrero (2013) study quasi-stationarity, extinction and final
size of stochastic epidemics in a random environment (title and DOI only). Ed-Darraz and Khaladi (2015) prove a final-size
result for SIR driven by a periodic-inhomogeneous Markov environment (summary only).

Bacaër and Khaladi (2013), HAL abstract read, handle mostly discrete-time Markov environments. They define R0 as the spectral
radius of a next-generation operator, show that its position relative to 1 decides growth or decay in simulations, and
contrast it with the Hernandez-Suarez et al. (2012) parameter, which decides growth of the expected population. Bacaër (2015)
treats stochastic SIS with periodic β: log of the mean extinction time over N converges to a constant given by a periodic
Hamilton-Jacobi equation, with approximations for small amplitude, high frequency and low frequency following Assaf,
Kamenev and Meerson (2008). Bacaër (2016) treats a two-state Markov environment. The mean extinction time grows
exponentially in N when both states are favourable and as a power of N when one state is unfavourable. Prodhomme and
Strickler (2024) prove the multitype version, with a power-law exponent p* characterized explicitly.

**The white-noise case.** Gray et al. (2011) perturb β by white noise, βdt → βdt + σdB (Itô). The almost-sure threshold is
R0S = R0D − σ²N²/(2(μ+γ)), so noise makes extinction easier. That is the Lewontin-Cohen gap in Itô form: the mean still
grows at βN − μ − γ, while the typical path grows at βN − μ − γ − σ²N²/2. A fast telegraph β with variance scaled so that
2K_ββ = σ² converges to Stratonovich white noise (Wong and Zakai 1965). In the switching model the almost-sure rate is
π·β N − μ − γ with no correction, and the mean gains +K_ββN². Both models have mean minus almost-sure = σ²N²/2 = K_ββN². They
differ only in which of the two carries the parameter β. The site could state this dictionary, because papers comparing the
two noise types tend to present them as different effects.

**Emergence probability.** For a birth-death process with periodic rates (Kendall 1948), Carmona and Gandon (2020) find that
fast fluctuations give an emergence probability that depends only on average transmission, 1 − μ/mean(λ). Slow fluctuations
make it depend strongly on the timing of introduction (their "winter is coming" effect). Their fast limit is leading order
only.

## 2. Fast-switching asymptotics and comparison with Green-Kubo

**What exists.** Monmarché, Schreiber and Strickler (2025), read in full on arXiv, consider x' = A(σ(ωt))x with Metzler
A_1..A_N, switched either periodically or by a Markov chain. They give first-order expansions of the Lyapunov exponent Λ(ω)
in the slow (ω → 0) and fast (ω → ∞) limits. Let Ā = Σ α_i A_i with right and left Perron vectors x̄, ȳ (1ᵀx̄ = 1, ȳᵀx̄ = 1),
and let P = I − x̄ȳᵀ. Then:

- Fast periodic: Λ = λ(Ā) + ω⁻¹ ȳᵀ(½ Σ_{i<j} α_iα_j [A_j, A_i]) x̄ + o(ω⁻¹). This is only a commutator (the Magnus second
  term), and it vanishes for two environments.
- Fast Markov (from Monmarché and Strickler 2025, Prop. 2): Λ = λ(Ā) + ω⁻¹ c_fM, with
  c_fM = ȳᵀ(Σ_ij α_i (Q − I)^#_ij A_j (x̄ȳᵀ − I) A_i) x̄, where Q is the jump chain and # the group inverse.

Write A_y = Ā + Σ_k φ_k(y) B_k and let K_lk = ∫_0^∞ Cov_π(φ_l(y_0), φ_k(y_t)) dt be the site's Green-Kubo matrix. Their
Markov formula becomes

  Λ_as ≈ λ(Ā) + Σ_{k,l} K_lk ȳᵀ B_k P B_l x̄ .

This matches the site's first-order rule except for the projection P. The later parameter's operator sits on the left
because the population equation runs forward in time. I checked this rewriting against their stage-structured example:
it reproduces their −(a − b)²/(8(a + b)). With two environments and diagonal B the correction is non-negative by Jensen:
at finite speed Λ sits above λ(Ā), and faster switching lowers it. That is the inflationary effect. An off-diagonal
fluctuation (a fecundity or transmission entry) gives the opposite sign. They also prove that for dispersal between patches the fast random correction
exceeds the fast periodic one, which is zero: mode matters, not just tempo. Monmarché and Strickler (2025) also build two
matrices whose convex combinations are all stable while the switched system has a positive top exponent at a suitable
rate. Benaïm, Le Borgne, Malrieu and Zitt (2014) found the same phenomenon for planar systems. Benaïm, Lobry, Sari and
Strickler (2025) give fast and slow asymptotic formulas for the top exponent of cooperative systems driven by a general
ergodic Feller process.

The ecology literature on the "inflationary effect" of autocorrelated fluctuations in coupled sinks (Gonzalez and Holt
2002; Roy, Holt and Barfield 2005), Katriel's dispersal-induced growth (2022), and the epidemic version in Kortessis et al.
(2020) are the same phenomenon at finite frequency. Kortessis et al. show that asynchronous variable control plus movement
between two populations raises the long-run regional Rt. Tuljapurkar (1982, 1990) and Tuljapurkar and Haridas (2006) give
small-noise approximations of the stochastic growth rate for discrete-time structured populations in Markovian environments.
Arnold (1984) links the two growth rates: the p-th moment exponent g(p) is convex with g(0) = 0 and g′(0) = λ_as.

**What the engine adds (new as far as found).** The mean m_i(t) = E[x(t); y_t = i] solves the linear block system
m′ = (Qᵀ ⊗ I + blockdiag A_i) m. This is exact for the linearized ODE with random coefficients, and also for the mean of
the multitype branching process in the same environment. Degenerate perturbation theory on the zero eigenspace of Qᵀ ⊗ I
gives

  Λ_mean ≈ λ(Ā) + Σ_{k,l} K_lk ȳᵀ B_k B_l x̄ ,

the site's rule with ȳ and x̄ as the payoff and initial state. Subtracting gives

  Λ_mean − Λ_as ≈ Σ_{k,l} K_lk (ȳᵀB_k x̄)(ȳᵀB_l x̄) = ∫_0^∞ Cov_π(r(y_0), r(y_t)) dt ≥ 0,   r(y) = ȳᵀ A_y x̄ .

The Lewontin-Cohen gap for a structured population is the Green-Kubo integral of the instantaneous growth rate
projected on the averaged Perron pair. Only the symmetric part of K enters the gap. The antisymmetric part enters both
exponents in the same way. I did not find this identity stated in the papers above.

Two consequences for epidemics:

- One infected compartment (SIR/SIS linearization): P = 0, so Λ_as = π·β S0/N − γ exactly. That matches Gray et al.'s
  T0S and the scalar identity used in Li et al. The mean gains exactly K_ββ at first order.
- SEIR, x = (E, I), A = [[−σ, β], [σ, −γ]], β switched: B_β = e_E e_Iᵀ with B_β² = 0. So Λ_mean has **no** first-order
  correction, and Λ_as = λ(Ā) − K_ββ (ȳ_E x̄_I)². The latent stage stops a transmission burst from compounding within one
  correlation time. This contradicts the ideas-note sentence for multi-compartment models and should be corrected there.

## 3. Order of fluctuations (antisymmetric K) in multi-compartment models

Nothing in the epidemic literature found here isolates an antisymmetric covariance or cycle-direction term. The nearest
results are:

- The periodic fast-switching term in Monmarché, Schreiber and Strickler (2025) is a pure commutator Σ α_iα_j [A_j, A_i].
  It depends on the order of the environments in the cycle, and it vanishes for two environments or for commuting matrices.
- Their Markov formula contains the full non-symmetric (Q − I)^#. For an irreversible chain it therefore contains the
  antisymmetric part of K acting through [B_k, B_l], but they do not separate or interpret it.
- Wang and Zhao (2008), Mitchell and Kribs (2017) and the phase-lag vector-borne models of Bacaër (2006, 2007) show that
  periodic R0 depends on relative phases once there are several infected compartments. That is the deterministic,
  finite-frequency version of the same effect.

So the explicit statement is new as far as found. With switched parameters (β, γ) in SEIR, B_β B_γ = −e_E e_Iᵀ,
B_γ B_β = 0 and B_γ² = e_I e_Iᵀ, so

  Λ_mean ≈ λ(Ā) + K_γγ ȳ_I x̄_I − K_γβ ȳ_E x̄_I ,

with K_γβ = ∫ Cov(γ(y_0), β(y_t)) dt: recovery rate now, transmission later. Reversing the cycle turns K_γβ into K_βγ.
The difference between the two directions is 2 K^anti_βγ ȳ_E x̄_I. Only one ordering of the product is non-zero, which
is exactly the commutator [B_β, B_γ] = −e_E e_Iᵀ. Age-structured next-generation matrices with a switched contact
matrix C(y) = C̄ + Σ φ_k C_k fit the same form. The cycle term is Σ K^anti_kl ȳᵀ[C_k, C_l]x̄. For example, it would pick up
whether school-age contact peaks come before or after adult-contact peaks in a policy cycle. I found no reference for
this.

## 4. Case counts under a hidden regime

The count models the site already has apply directly to **imported or spillover** cases arriving at a hidden-regime
rate. That is a Markov-modulated Poisson process (Fischer and Meier-Hellstern 1993), and the counts page gives its
first-order overdispersion t(ℓ1 − ℓ2)²/(4λ) (two states), or in general 2t·K_ℓℓ. Held, Höhle and Hofmann (2005) model
surveillance counts as a Poisson (or negative-binomial) branching process with immigration, with seasonal and
overdispersion extensions. Le Strat and Carrat (1999) fit hidden Markov models to surveillance series (Poisson mixtures
for polio counts). Martínez-Beneito et al. (2008) and Conesa et al. (2015) use Markov switching between epidemic and
non-epidemic phases for influenza. Cori et al. (2013) estimate Rt by renewal-equation Poisson likelihood. Parag (2021)
improves Rt estimation at low incidence (abstract not read). Dureau, Kalogeropoulos and Baguelin (2013) instead give β a
diffusion and fit by particle MCMC.

Overdispersion has two separate sources that the site should keep apart:

- Individual heterogeneity: the negative-binomial offspring distribution of Lloyd-Smith et al. (2005).
- Temporal environment: the Green-Kubo excess. For transmitted (not imported) cases, counts given the regime path are
  not Poisson because of branching, so the MMPP formula does not apply directly. The first and second moments of a
  branching process in a switching environment do satisfy closed linear systems. The second moment is an augmented
  engine system with the first moment as a source. The first-order overdispersion of transmitted counts could be derived
  that way. I found no published version.

Standard overdispersion score tests are Cameron and Trivedi (1990) and Dean (1992). The latter derives tests against
arbitrary mixtures specified by two moments, which covers the first-order MMPP alternative. Dean's statistic should be
the score test at K = 0 in the site's parameterization, one-sided as on the likelihood page. That is plausible but not
checked.

## 5. Concrete examples the site could add, with checks run

All three were checked with a numerical eigenvalue of the block matrix Qᵀ ⊗ I + blockdiag A_i (mean) and, where
relevant, Monte Carlo of the switched ODE (almost-sure exponent, 2×10⁴ time units, renormalized products of matrix
exponentials).

**A. Linearized SIR with a two-state switched β.** Expected prevalence given the starting regime is
E[I_t | y_0 = i] = I_0 a_i(t) with a′ = (Q + diag(β_i − γ))a, a(0) = 1. That is the engine with constant g_i and exact
Feynman-Kac. First order: a_i(t) ≈ exp(t(β̄ − γ + K_ββ))(1 − [Q^#(β − β̄)]_i), with K_ββ = (Δβ/2)²/(2λ) for symmetric
switching at rate λ. The almost-sure rate is exactly β̄ − γ. With β = (0.5, 0.1) and γ = 0.25, the exact mean growth at
λ = 2, 4, 8, 16 is 0.059975, 0.054997, 0.052500, 0.051250, against the prediction 0.05 + 0.02/λ = 0.06, 0.055, 0.0525,
0.05125.

**B. Linearized SEIR, β switched (the counterexample to "fluctuation raises the mean").** σ = 0.5, γ = 0.25,
β = (0.6, 0.2), λ(Ā) = 0.089354, ȳ_E x̄_I = 0.538. The exact mean growth minus λ(Ā), times λ², tends to about 0.0026.
So the correction is second order, as predicted.

| λ | a.s. Monte Carlo | a.s. predicted λ(Ā) − K_ββ(ȳ_E x̄_I)² |
|---|---|---|
| 2 | 0.08643 | 0.08646 |
| 4 | 0.08809 | 0.08791 |
| 8 | 0.08850 | 0.08863 |
| 16 | 0.08905 | 0.08899 |

**C. SEIR with β and γ switched on the irreversible three-state chain Q_A of the any-equation page.** σ = 0.5,
β = (0.9, 0.1, 0.5), γ = (0.1, 0.3, 0.5), chain c·Q_A, compared with its time reversal (same π, same K^sym). The formula
λ(Ā) + K_γγ ȳ_I x̄_I − K_γβ ȳ_E x̄_I gives 0.049141, 0.044680 and 0.043565 at c = 1, 4, 16, against exact values 0.050461,
0.044766 and 0.043570. The difference between the forward and reversed cycles is:

| c | exact | predicted 2K^anti ȳ_E x̄_I |
|---|---|---|
| 1 | 2.87e-4 | 4.57e-4 |
| 2 | 1.79e-4 | 2.29e-4 |
| 4 | 1.01e-4 | 1.14e-4 |
| 8 | 5.36e-5 | 5.71e-5 |
| 16 | 2.77e-5 | 2.86e-5 |

The ratio tends to 1. The effect is small in absolute size, since it is second order in the rate differences and first
order in 1/c, but it is cleanly isolated.

Suggested page checks: exact block eigenvalue and exact matrix-exponential mean (as the certificates do now); Monte Carlo
of the switched ODE for the almost-sure exponent; and, for the branching version, Gillespie simulation of the SEIR birth-death
process in the switching environment to confirm that the mean follows the same block system while extinction follows the
almost-sure exponent (as in Smith and Wilkinson 1969 and Prodhomme and Strickler 2024).

Beyond the linear phase: the invasion results above cover only the linearization. For the nonlinear SIR the site's rule
gives a corrected generator for the joint law. Monmarché and Strickler (2025) expand the invariant measure of
Markov-modulated ODEs near a globally attracting point, which would be the starting point for an endemic-equilibrium
version.

## References (all verified; DOI given)

Read in full (open copy): Gray et al. 2011, 2012 (Strathprints); Monmarché, Schreiber and Strickler 2025 (arXiv);
Li, Liu and Cui 2017 (arXiv, threshold only); Carmona and Gandon 2020 (PLOS). Abstract or HAL summary read: the Bacaër
papers, Thieme, Heesterbeek and Roberts, Greenhalgh et al., Prodhomme and Strickler, Monmarché and Strickler, Benaïm et al.
(note), Assaf et al., Kortessis et al., Roy et al., Tuljapurkar and Haridas, Arnold, Smith and Wilkinson, Held et al.,
Le Strat and Carrat, Martínez-Beneito et al., Conesa et al., Dureau et al., Allen, Britton, Lloyd-Smith et al., Dean, Katriel.
Title and bibliographic data only: the rest.

- Allen LJS (2017). A primer on stochastic epidemic models: formulation, numerical simulation, and analysis. Infect. Dis. Model. 2:128-142. doi:10.1016/j.idm.2017.03.001
- Altizer S et al. (2006). Seasonality and the dynamics of infectious diseases. Ecol. Lett. 9:467-484. doi:10.1111/j.1461-0248.2005.00879.x
- Arnold L (1984). A formula connecting sample and moment stability of linear stochastic systems. SIAM J. Appl. Math. 44:793-802. doi:10.1137/0144057
- Artalejo JR, Economou A, Lopez-Herrero MJ (2013; online 2012). Stochastic epidemic models with random environment: quasi-stationarity, extinction and final size. J. Math. Biol. 67:799-831. doi:10.1007/s00285-012-0570-5
- Assaf M, Kamenev A, Meerson B (2008). Population extinction in a time-modulated environment. Phys. Rev. E 78:041123. doi:10.1103/PhysRevE.78.041123
- Athreya KB, Karlin S (1971). On branching processes with random environments: I: extinction probabilities. Ann. Math. Stat. 42:1499-1520. doi:10.1214/aoms/1177693150
- Bacaër N (2007). Approximation of the basic reproduction number R0 for vector-borne diseases with a periodic vector population. Bull. Math. Biol. 69:1067-1091. doi:10.1007/s11538-006-9166-9
- Bacaër N (2009). Periodic matrix population models: growth rate, basic reproduction number, and entropy. Bull. Math. Biol. 71:1781-1792. doi:10.1007/s11538-009-9426-6
- Bacaër N (2015; online 2014). On the stochastic SIS epidemic model in a periodic environment. J. Math. Biol. 71:491-511. doi:10.1007/s00285-014-0828-1
- Bacaër N (2016). Le modèle stochastique SIS pour une épidémie dans un environnement aléatoire. J. Math. Biol. 73:847-866. doi:10.1007/s00285-016-0974-8
- Bacaër N, Ait Dads EH (2011; online 2010). Genealogy with seasonality, the basic reproduction number, and the influenza pandemic. J. Math. Biol. 62:741-762. doi:10.1007/s00285-010-0354-8
- Bacaër N, Guernaoui S (2006). The epidemic threshold of vector-borne diseases with seasonality. J. Math. Biol. 53:421-436. doi:10.1007/s00285-006-0015-0
- Bacaër N, Khaladi M (2013; online 2012). On the basic reproduction number in a random environment. J. Math. Biol. 67:1729-1739. doi:10.1007/s00285-012-0611-0
- Bacaër N, Ouifki R (2007). Growth rate and basic reproduction number for population models with a simple periodic factor. Math. Biosci. 210:647-658. doi:10.1016/j.mbs.2007.07.005
- Benaïm M, Le Borgne S, Malrieu F, Zitt P-A (2014). On the stability of planar randomly switched systems. Ann. Appl. Probab. 24:292-311. doi:10.1214/13-AAP924
- Benaïm M, Lobry C, Sari T, Strickler É (2025). A note on the top Lyapunov exponent of linear cooperative systems. Ann. Fac. Sci. Toulouse Math. 34:225-241. doi:10.5802/afst.1811 (arXiv:2302.05874)
- Benaïm M, Strickler É (2019). Random switching between vector fields having a common zero. Ann. Appl. Probab. 29:326-375. doi:10.1214/18-AAP1418
- Black AJ, McKane AJ (2010). Stochastic amplification in an epidemic model with seasonal forcing. J. Theor. Biol. 267:85-94. doi:10.1016/j.jtbi.2010.08.014
- Britton T (2010). Stochastic epidemic models: a survey. Math. Biosci. 225:24-35. doi:10.1016/j.mbs.2010.01.006
- Cameron AC, Trivedi PK (1990). Regression-based tests for overdispersion in the Poisson model. J. Econometrics 46:347-364. doi:10.1016/0304-4076(90)90014-K
- Carmona P, Gandon S (2020). Winter is coming: pathogen emergence in seasonal environments. PLoS Comput. Biol. 16:e1007954. doi:10.1371/journal.pcbi.1007954
- Conesa D, Martínez-Beneito MA, Amorós R, López-Quílez A (2015; online 2011). Bayesian hierarchical Poisson models with a hidden Markov structure for the detection of influenza epidemic outbreaks. Stat. Methods Med. Res. 24:206-223. doi:10.1177/0962280211414853
- Cori A, Ferguson NM, Fraser C, Cauchemez S (2013). A new framework and software to estimate time-varying reproduction numbers during epidemics. Am. J. Epidemiol. 178:1505-1512. doi:10.1093/aje/kwt133
- Dean CB (1992). Testing for overdispersion in Poisson and binomial regression models. J. Am. Stat. Assoc. 87:451-457. doi:10.1080/01621459.1992.10475225
- Dietz K (1976). The incidence of infectious diseases under the influence of seasonal fluctuations. Lecture Notes in Biomathematics 11:1-15. doi:10.1007/978-3-642-93048-5_1
- Dureau J, Kalogeropoulos K, Baguelin M (2013). Capturing the time-varying drivers of an epidemic using stochastic dynamical systems. Biostatistics 14:541-555. doi:10.1093/biostatistics/kxs052
- Economou A, Lopez-Herrero MJ (2016; online 2015). The deterministic SIS epidemic model in a Markovian random environment. J. Math. Biol. 73:91-121. doi:10.1007/s00285-015-0943-7
- Ed-Darraz A, Khaladi M (2015). On the final size of epidemics in random environment. Math. Biosci. 266:10-14. doi:10.1016/j.mbs.2015.05.004
- Fischer W, Meier-Hellstern K (1993). The Markov-modulated Poisson process (MMPP) cookbook. Perform. Eval. 18:149-171. doi:10.1016/0166-5316(93)90035-S
- Gonzalez A, Holt RD (2002). The inflationary effects of environmental fluctuations in source-sink systems. PNAS 99:14872-14877. doi:10.1073/pnas.232589299
- Grassly NC, Fraser C (2006). Seasonal infectious disease epidemiology. Proc. R. Soc. B 273:2541-2550. doi:10.1098/rspb.2006.3604
- Gray A, Greenhalgh D, Hu L, Mao X, Pan J (2011). A stochastic differential equation SIS epidemic model. SIAM J. Appl. Math. 71:876-902. doi:10.1137/10081856X
- Gray A, Greenhalgh D, Mao X, Pan J (2012). The SIS epidemic model with Markovian switching. J. Math. Anal. Appl. 394:496-516. doi:10.1016/j.jmaa.2012.05.029
- Greenhalgh D, Liang Y, Mao X (2016). Modelling the effect of telegraph noise in the SIRS epidemic model using Markovian switching. Physica A 462:684-704. doi:10.1016/j.physa.2016.06.125
- Heesterbeek JAP, Roberts MG (1995). Threshold quantities for infectious diseases in periodic environments. J. Biol. Syst. 3:779-787. doi:10.1142/S021833909500071X
- Held L, Höhle M, Hofmann M (2005). A statistical framework for the analysis of multivariate infectious disease surveillance counts. Stat. Model. 5:187-199. doi:10.1191/1471082X05st098oa
- Hernandez-Suarez C, Rabinovich J, Hernandez K (2012). The long-run distribution of births across environments under environmental stochasticity and its use in the calculation of unconditional life-history parameters. Theor. Popul. Biol. 82:264-274. doi:10.1016/j.tpb.2012.05.004
- Inaba H (2012; online 2011). On a new perspective of the basic reproduction number in heterogeneous environments. J. Math. Biol. 65:309-348. doi:10.1007/s00285-011-0463-z
- Inaba H (2019). The basic reproduction number R0 in time-heterogeneous environments. J. Math. Biol. 79:731-764. doi:10.1007/s00285-019-01375-y
- Katriel G (2022). Dispersal-induced growth in a time-periodic environment. J. Math. Biol. 85:24. doi:10.1007/s00285-022-01791-7
- Keeling MJ, Rohani P (2008). Modeling Infectious Diseases in Humans and Animals. Princeton University Press. doi:10.1515/9781400841035
- Keeling MJ, Rohani P, Grenfell BT (2001). Seasonally forced disease dynamics explored as switching between attractors. Physica D 148:317-335. doi:10.1016/S0167-2789(00)00187-1
- Kendall DG (1948). On the generalized "birth-and-death" process. Ann. Math. Stat. 19:1-15. doi:10.1214/aoms/1177730285
- Kortessis N, Simon MW, Barfield M, Glass GE, Singer BH, Holt RD (2020). The interplay of movement and spatiotemporal variation in transmission degrades pandemic control. PNAS 117:30104-30106. doi:10.1073/pnas.2018286117
- Le Strat Y, Carrat F (1999). Monitoring epidemiologic surveillance data using hidden Markov models. Stat. Med. 18:3463-3478. doi:10.1002/(SICI)1097-0258(19991230)18:24<3463::AID-SIM409>3.0.CO;2-I
- Lewontin RC, Cohen D (1969). On population growth in a randomly varying environment. PNAS 62:1056-1060. doi:10.1073/pnas.62.4.1056
- Li D, Liu S, Cui J (2017). Threshold dynamics and ergodicity of an SIRS epidemic model with Markovian switching. J. Differential Equations 263:8873-8915. doi:10.1016/j.jde.2017.08.066
- Lloyd-Smith JO, Schreiber SJ, Kopp PE, Getz WM (2005). Superspreading and the effect of individual variation on disease emergence. Nature 438:355-359. doi:10.1038/nature04153
- Martínez-Beneito MA, Conesa D, López-Quílez A, López-Maside A (2008). Bayesian Markov switching models for the early detection of influenza epidemics. Stat. Med. 27:4455-4468. doi:10.1002/sim.3320
- Mitchell C, Kribs C (2017). A comparison of methods for calculating the basic reproductive number for periodic epidemic systems. Bull. Math. Biol. 79:1846-1869. doi:10.1007/s11538-017-0309-y
- Monmarché P, Schreiber SJ, Strickler É (2025). Impacts of tempo and mode of environmental fluctuations on population growth: slow- and fast-limit approximations of Lyapunov exponents for periodic and random environments. Bull. Math. Biol. 87. doi:10.1007/s11538-025-01443-z (arXiv:2408.11179)
- Monmarché P, Strickler É (2025). Asymptotic expansion of the invariant measure for Markov-modulated ODEs at high frequency. J. Appl. Probab. 62:898-924. doi:10.1017/jpr.2024.107 (arXiv:2309.16464)
- Parag KV (2021). Improved estimation of time-varying reproduction numbers at low case incidence and between epidemic waves. PLoS Comput. Biol. 17:e1009347. doi:10.1371/journal.pcbi.1009347
- Prodhomme A, Strickler É (2024). Large population asymptotics for a multitype stochastic SIS epidemic model in randomly switching environment. Ann. Appl. Probab. 34. doi:10.1214/23-AAP2035
- Roy M, Holt RD, Barfield M (2005). Temporal autocorrelation can enhance the persistence and abundance of metapopulations comprised of coupled sinks. Am. Nat. 166:246-261. doi:10.1086/431286
- Smith WL, Wilkinson WE (1969). On branching processes in random environments. Ann. Math. Stat. 40:814-827. doi:10.1214/aoms/1177697589
- Thieme HR (2009). Spectral bound and reproduction number for infinite-dimensional population structure and time heterogeneity. SIAM J. Appl. Math. 70:188-211. doi:10.1137/080732870
- Tuljapurkar SD (1982). Population dynamics in variable environments. II. Correlated environments, sensitivity analysis and dynamics. Theor. Popul. Biol. 21:114-140. doi:10.1016/0040-5809(82)90009-0
- Tuljapurkar S, Haridas CV (2006). Temporal autocorrelation and stochastic population growth. Ecol. Lett. 9:327-337. doi:10.1111/j.1461-0248.2006.00881.x
- Wang W, Zhao X-Q (2008). Threshold dynamics for compartmental epidemic models in periodic environments. J. Dyn. Differ. Equ. 20:699-717. doi:10.1007/s10884-008-9111-8
- Wong E, Zakai M (1965). On the convergence of ordinary integrals to stochastic integrals. Ann. Math. Stat. 36:1560-1564. doi:10.1214/aoms/1177699916

Not verified or not used: Heesterbeek and Roberts (1995) "Threshold quantities for helminth infections" (J. Math. Biol. 33,
doi:10.1007/BF00176380) exists but was not read. The Tuljapurkar (1990) monograph is cited only through the survey page.
Inaba's content was not read.
