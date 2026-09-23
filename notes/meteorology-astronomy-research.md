# Fast-switching homogenization in meteorology and astronomy (research memo, 2026-09-23)

Scope: where the site's problem, E[exp(-int g(y_s) ds)] with y a fast finite Markov chain, and its expansion
(averaged answer, Green-Kubo first correction K = int_0^inf Cov_pi(g(y_0), g(y_t)) dt, higher orders with an
initial layer) already appears in meteorology and astronomy, what is known, and what is not.

Verification: every reference below was checked against Crossref (title, first author, year) unless marked
UNVERIFIED. "Abstract read" means the abstract was read via Crossref, OpenAlex or the publisher; "read" means an
open-access or author-posted PDF was read. Nothing was taken from shadow libraries.

Numerical check: `notes/binary_mixture_check.py` (numpy, scipy).

---

## 1. Stochastic radiative transfer in binary Markovian mixtures (main lead)

### The problem is the engine's problem

A beam crosses a medium made of two materials, A and B, with absorption coefficients Sigma_A and Sigma_B. Along
the ray the material alternates, and the chord lengths in A and B are exponential with means lambda_A and
lambda_B. The material seen along the ray is then a two-state Markov chain indexed by path length s, with
switching rates 1/lambda_A (A to B) and 1/lambda_B (B to A). For a purely absorbing medium the ensemble-averaged
transmission of a slab of thickness L is

    T(L) = E[ exp(-int_0^L Sigma(y_s) ds) ],

with y_0 drawn from the volume fractions p_A = lambda_A/(lambda_A+lambda_B), p_B = 1 - p_A. This is the
regime-switching survival problem with path length in place of time, g = Sigma, and no slow variable x.

The mapping holds in any dimension when the mixture is a Poisson tessellation, because a straight line through a
Poisson tessellation meets it in a Poisson sequence of interfaces. It fails once scattering is present: a
scattered particle revisits regions it has already crossed, so the material along its path is no longer Markov in
path length.

### What is known

**The exact answer.** The two-exponential law

    T(L) = [ (S~ - r_-) e^{-r_- L} + (r_+ - S~) e^{-r_+ L} ] / (r_+ - r_-),
    2 r_+- = <S> + S~ +- sqrt( (<S> - S~)^2 + 4 beta ),
    <S> = p_A S_A + p_B S_B,  S~ = p_B S_A + p_A S_B + 1/lambda_A + 1/lambda_B,  beta = p_A p_B (S_A - S_B)^2

is quoted by d'Eon (arXiv 2019, read), who credits it to Avaste and Vainikko (1974), Levermore et al. (1986)
and Vanderhaegen (1986). Levermore, Pomraning, Sanzo and Wong (1986, abstract read) derive a formally exact
equation for the purely absorbing case and solve it exactly for two-fluid Markov statistics, and they note that it
agrees with averaging exponential attenuation directly. The script checks that the law equals the 2x2
matrix-exponential answer to 1e-12 and matches a Monte Carlo of the chain (0.1293 +- 0.0002 against 0.1294).

**The Levermore-Pomraning (LP) closure.** The LP model is two coupled transport equations for the
material-conditioned intensities. It is exact for purely absorbing Markovian mixtures and approximate once there
is scattering or non-Markovian mixing (stated in the LLNL/OSTI benchmark reports found in the search, e.g. the Brantley
Levermore-Pomraning verification reports). The book is Pomraning (1991). Chord length sampling
(Zimmerman and Adams 1991, UNVERIFIED) is its Monte Carlo counterpart, and is also exact only for pure absorbers.

**Atomic mix.** Replace the cross sections by their volume averages, T = exp(-<S> L). This is the averaged
model. It is exact as lambda_A, lambda_B -> 0.

**Small-correlation-length asymptotics.** These exist, but the papers were not read in full:
- Malvagi, Levermore and Pomraning (1989, abstract read) study three limits of the LP model. Limit (1), "fluid
  packets small compared to the particle mean free path in the packet", reduces the two coupled equations to a
  single transport equation of the usual form. This is the small-correlation-length limit. Whether the paper
  writes out the first correction to the atomic-mix cross section was not verified.
- Levermore, Wong and Pomraning (1988, abstract read) use renewal theory for the pure absorber with arbitrary
  chord-length distributions. They obtain "effective cross sections and an effective source" for a standard
  transport equation, and they report that the mean and variance of the chord lengths suffice for good accuracy.
- Su and Pomraning (1994, summary read in the DOE progress report OSTI 10158144) construct small and large
  correlation length limiting solutions in rod geometry, with albedo and mean free path both fluctuating, and use
  them to build new closures.
- Pomraning (1996) has "small correlation length solutions" for beam transport (title only).
- Kiedrowski and Vu (arXiv 2024, read in part) map transport in a binary Markovian mixture to the telegraph
  process. They derive the distribution of the distance travelled in each material and give an asymptotic
  Gaussian form for highly mixed media, whose variance scales with the correlation length. They report that the
  asymptotic form is more accurate than atomic mix. A Gaussian transit length is the central-limit form of the
  Green-Kubo correction.

**Benchmarks.** Adams, Larsen and Pomraning (1989) give the standard planar benchmark suite. Brantley (2011)
compares Monte Carlo algorithms on it. Larmier, Hugot, Malvagi, Mazzolo and Zoia (2017, abstract read) extend
the benchmarks to d = 1, 2, 3 on Poisson tessellations. Larmier, Lam, Brantley, Malvagi et al. (2018) study
chord length sampling in d dimensions. Vasques and Larsen (2014) and Larsen and Vasques (2011) develop
non-classical transport with non-exponential path lengths, applied to pebble beds.

### The first correction is the Green-Kubo term (derivation)

For the chain above, Cov_pi(Sigma(y_0), Sigma(y_s)) = p_A p_B (S_A - S_B)^2 e^{-s/lambda_c}, with the
correlation length lambda_c = lambda_A lambda_B/(lambda_A + lambda_B) = 1/(1/lambda_A + 1/lambda_B). So

    K = int_0^inf Cov ds = p_A p_B (S_A - S_B)^2 lambda_c = beta lambda_c,

and the engine's first-order answer is

    T(L) ~ exp( -(<S> - beta lambda_c) L ).

From the exact law: r_- solves (r - <S>)(r - S~) = beta, so r_- = <S> - beta/(S~ - r_-) exactly. Since
S~ - <S> = 1/lambda_c - (p_A - p_B)(S_A - S_B),

    r_- = <S> - beta lambda_c - beta (p_A - p_B)(S_A - S_B) lambda_c^2 + O(lambda_c^3).

The long-path decay rate therefore agrees with the Green-Kubo correction at first order. The O(lambda_c^2) term
is the engine's second-order term for unequal switching rates. The weight on e^{-r_+ L} is (<S> - r_-)/(r_+ - r_-)
= beta lambda_c^2 + O(lambda_c^3). It decays on the scale lambda_c and is the initial layer: the ray enters in
whichever material it enters. Its permanent trace is a constant -beta lambda_c^2 in log T. The first iterate of
the fixed point, <S> - beta/S~, has the same first-order term but a different second-order term. If the classical
"effective cross section" has that form (recalled, UNVERIFIED), it is correct to first order only.

Numerical check (S_A = 2, S_B = 0.2, lambda_A = lambda_B/2, so p_A = 1/3, L = 3). Errors are in log T:

| lambda_c | exact T | atomic mix | + Green-Kubo | + second order and layer |
|---|---|---|---|---|
| 0.4 | 0.16329 | 5.9e-1 | 2.8e-1 | 4.6e-2 |
| 0.1 | 0.11038 | 2.0e-1 | 2.0e-2 | 3.4e-4 |
| 0.025 | 0.09563 | 5.3e-2 | 1.3e-3 | 2.4e-6 |
| 0.0125 | 0.09317 | 2.7e-2 | 3.1e-4 | 2.4e-7 |

The three columns fall like lambda_c, lambda_c^2 and roughly lambda_c^3. Atomic mix underestimates transmission,
because clumping lets light through the gaps. The Green-Kubo term is the size of that effect.

### Known versus new

- Known: the exact purely absorbing answer, the LP closure, atomic mix as the limit, and small-correlation-length
  analyses. The expansion of r_- in lambda_c is one line from the exact law and is very likely written down
  somewhere in Pomraning's book or in Malvagi-Levermore-Pomraning (1989). Neither was read here, so priority for
  the first-order term should not be claimed.
- Probably new as a framing: that the atomic-mix correction is a Green-Kubo integral of the opacity
  autocovariance along the ray. This carries over directly to non-binary mixtures (N-phase chains, as in Hobson
  and Scheuer 1993) and to any stationary opacity process with integrable covariance. The same holds for the
  all-orders series with the initial layer. The site's engine (constant g_i, no x) produces it at once.
- Open: with scattering the ray-indexed chain picture breaks. Whether the first correction to atomic mix in the
  scattering case is still given by a Green-Kubo term (for example through an effective cross section in a
  single transport equation, as in limit (1) above) is the natural next question. The rod-geometry results of
  Su and Pomraning (1994) are where to check.

### Clouds and interstellar dust (the same mathematics)

- Broken clouds. Titov (1990, abstract read) models cloud fields with Poisson point fluxes whose inputs are cloud
  fraction and mean horizontal cloud size. These are p and lambda in the notation above. Malvagi, Byrne,
  Pomraning and Somerville (1993, abstract read) show that Titov's integral Markovian model is equivalent to a
  special case of a simple low-order differential model (the LP type) and recommend the differential form for
  GCMs. Kassianov (2003) extends the Markovian approach to multilayer broken clouds. Prigarin, Kargin and Oppel
  (1998) simulate random broken-cloud fields and the direct solar radiation. For direct-beam transmission along a
  slant path through a Markovian cloud/clear field, the Green-Kubo form gives
  T ~ exp(-(c S_cloud - c(1-c) S_cloud^2 lambda_c) L) for cloud fraction c, cloud extinction S_cloud and clear air taken as non-extinguishing. It is exact to first
  order in the cloud size divided by the photon mean free path in cloud. When cloud chords are optically thick
  the expansion is poor; the exact two-exponential law should be used.
- Non-exponential transmission. Kostinski (2001) (summary read) shows that a statistically homogeneous but
  spatially correlated medium attenuates more slowly than exponentially. Davis and Marshak (2004) show enhanced
  mean free paths and wider-than-exponential free-path distributions. Cahalan, Ridgway, Wiscombe and Bell (1994)
  study the plane-parallel albedo bias, and Barker (1996, abstract read) the gamma-distributed independent-pixel
  parameterization. All are the same Jensen-type effect: E exp(-tau) > exp(-E tau). For a Markov field the
  Green-Kubo term is the leading correction in the fast-switching limit.
- Clumpy interstellar dust. Natta and Panagia (1984, abstract read) derive analytic extinction for clumpy dust
  layers. Boisse (1990/1991) treats UV transfer in fragmented clouds; the proceedings version is Crossref-verified
  and the 1990 A&A paper is UNVERIFIED. Hobson and Scheuer (1993, abstract read) solve an N-phase Markov-mixture
  slab analytically, with absorption and isotropic scattering, and give "effective homogeneous" solutions. Witt
  and Gordon (1996) use Monte Carlo for clumpy scattering media. Stalevski et al. (2012) model AGN tori as a
  clumpy two-phase medium. Checkable example: apparent extinction of a star behind an N-phase Markov line of
  sight. The averaged model is A_V proportional to <kappa> L. The first correction is -K L, with K from the N-state
  covariance. The engine gives it to all orders.

---

## 2. Meteorology

**Stochastic climate models and mode reduction.** Hasselmann (1976) treats fast weather as noise forcing slow
climate variables. Majda, Timofeyev and Vanden-Eijnden (1999; 2001, abstract read) derive the reduced
stochastic equations systematically. The drift and diffusion of the reduced model are time integrals of
correlations of the fast modes, which are Green-Kubo integrals. Franzke, Majda and Vanden-Eijnden (2005) apply it
to a realistic barotropic model, and Gottwald, Crommelin and Franzke (2016) review it. Scaling difference: this
is the diffusive (homogenization) scaling, where the fast forcing has zero mean and its effect appears on the long
time scale 1/epsilon. The site's expansion is the averaging scaling, where the averaged model leads and the
Green-Kubo term is the O(epsilon) correction. The integral is the same. An expansion in the fast time scale does
exist here, and it is the core of the method.
Checkable example: a Hasselmann-type slow variable dT = (-a T + F(y)) dt with F switching between two values. The
averaged model is relaxation to the mean forcing. The first correction is Var T ~ K/a with K = pq dF^2/gamma, a
red spectrum. The exact two-state answer is available for comparison.

**Weather regimes as hidden Markov chains.** Majda, Franzke, Fischer and Crommelin (2006) and Franzke, Crommelin,
Fischer and Majda (2008, abstract read) fit hidden Markov models to planetary-flow indices and identify
metastable regimes. Kondrashov, Ide and Ghil (2004) study preferred regime transitions. The averaged model for a
seasonal-mean quantity is its regime-weighted mean. The first correction is the Green-Kubo variance of the
seasonal mean, about 2K/T, the "climate noise" floor. No expansion in the regime switching time was found in this
literature. The HMM papers estimate the chain; they do not expand in it.
Checkable example: from a fitted two-regime HMM for a blocking index, predict the interannual variance of seasonal
blocking-day counts as 2K/T plus the layer term, and compare with the observed series.

**Stochastic parameterization.** Palmer (2001), Buizza, Miller and Palmer (1999, stochastically perturbed
tendencies) and Palmer (2019, review). Crommelin and Vanden-Eijnden (2008, abstract read) represent subgrid
processes by a Markov chain conditioned on the resolved state and fit it on Lorenz '96. Khouider, Majda and
Katsoulakis (2003) build coarse-grained stochastic lattice models for convection. The conditional Markov chain is
exactly the engine's setting with a state-dependent generator. The first-order rule then gives the
noise-induced drift and diffusion that the chain adds to the resolved dynamics. No paper found writes this first
correction explicitly for the conditional-chain scheme; this is not established, only not found.

**Precipitation occurrence.** Gabriel and Neumann (1962) introduce the two-state Markov chain for daily rain
occurrence. Katz (1974) computes probabilities under it. Richardson (1981) and Wilks (1998) build weather
generators on it. Katz and Parlange (1998, abstract read) document "overdispersion": chain-dependent models
underestimate the variance of monthly totals. Part of the gap is removed by higher-order occurrence chains; the
rest is attributed to low-frequency variation. Katz, Parlange and Tebaldi (2003) condition on a circulation index.
The discrete-time Green-Kubo formula gives the wet-day count variance: Var N ~ n pi(1-pi)(1+d)/(1-d) with
d = p11 - p01, plus a boundary constant. The long-window version is standard. For p01 = 0.25, p11 = 0.66 and
n = 30 it gives 17.5, against 16.9 exact and 7.3 binomial. The remaining overdispersion is what a hidden slow
regime switching (p01, p11) would add. That is a two-level (nested) Green-Kubo computation and a concrete open
exercise.

---

## 3. Astronomy

**State-switching sources and photon counts.** Given the source state, the counts are a Markov-modulated Poisson
process (Fischer and Meier-Hellstern 1993, the MMPP "cookbook"). The averaged model is Poisson at the mean rate.
The first correction is the Fano factor for windows much longer than the switching time, F = 1 + 2K/lambda_bar
(the site's counts page). Astronomy uses Bayesian blocks (Scargle 1998) and continuous autoregressive models
(Kelly, Bechtold and Siemiginowska 2009, the damped random walk for quasars). No astronomy paper found uses an
MMPP with a fast-switching expansion; this was a light search.

**Pulsar mode switching and timing noise.** Kramer et al. (2006, abstract read): PSR B1931+24 is on for 5-10
days and off for 25-35 days, and spins down 50% faster when on. Lyne, Hobbs, Kramer, Stairs et al. (2010, abstract
read): timing noise in many pulsars comes from abrupt switching between two spin-down rates, correlated with
pulse-shape changes. This is the engine's problem with g = spin-down rate. The averaged model is a constant mean
spin-down. The first correction makes the accumulated frequency a random walk, Var(int nudot) ~ 2KT, where
K = pq (nudot_1 - nudot_2)^2/gamma for Markov switching. The exact two-state variance is
2K(T - (1 - e^{-gamma T})/gamma), so the Green-Kubo form is off only by the layer constant (for switching times of
30 and 100 days: 2001 exact against 2048 at T = 1000 days). The characteristic function of the
timing residual is E exp(i k int nudot), the engine with imaginary g. Timing-noise analysis uses Gaussian-process
red noise (van Haasteren and Levin 2012, abstract read). The switching model predicts a specific spectrum:
Lorentzian in nudot, flattening below the switching frequency. It also predicts non-Gaussian residuals at
second order.
Checkable example: B1931+24, with p ~ 0.2 on and a correlation time of about 6 days if Markov is assumed. The real
switching is quasi-periodic, so K must be computed from the observed on/off covariance, not the Markov formula.
Compare the predicted random-walk strength with the published timing residuals.

**Wave propagation in randomly layered media.** Fouque, Garnier, Papanicolaou and Solna (2007, book). Solna and
Papanicolaou (2000, author PDF read): a pulse through a randomly layered medium, viewed in its random arrival-time
frame, is the effective-medium pulse convolved with a Gaussian of variance z gamma_0^2 l/2. Here
l = int_0^inf C(s) ds is the integrated covariance of the fluctuations. This is the O'Doherty-Anstey (1971)
result. The averaged model is the effective medium, and the correction is a Green-Kubo integral. The expansion is
the diffusion approximation in the layer scale, so it is known. For the interstellar medium, scintillation and pulse
broadening (Rickett 1990) come from 3D Kolmogorov turbulence, not layering. The connection there is through
Markov-approximation parabolic equations rather than a finite chain, and it is weaker.

**Clumpy ISM.** See section 1. Hobson and Scheuer (1993) is the closest astronomical analogue of the
binary-mixture theory and already uses N-state Markov statistics.

---

## Candidate site pages, in order of fit

1. "Light through clumpy media": the binary Markovian mixture as the regime-switching survival problem. Exact
   law, atomic mix, Green-Kubo first correction, second order and layer, with the table above. Links to
   Levermore-Pomraning.
2. Pulsar spin-down switching: timing residuals as an integrated switched rate. Real parameters from Kramer et
   al. (2006).
3. Precipitation overdispersion: the discrete-time Green-Kubo and a nested hidden regime.

---

## References (Crossref-verified unless marked)

- Adams, M.L., Larsen, E.W., Pomraning, G.C. (1989) Benchmark results for particle transport in a binary Markov statistical medium. JQSRT. doi:10.1016/0022-4073(89)90072-1
- Avaste, O.A., Vainikko, G.M. (1974) solar radiative transfer in broken clouds, Izv. Atmos. Ocean. Phys. UNVERIFIED (cited by d'Eon 2019; no Crossref record).
- Barker, H.W. (1996) A parameterization for computing grid-averaged solar fluxes for inhomogeneous marine boundary layer clouds. Part I. J. Atmos. Sci. doi:10.1175/1520-0469(1996)053<2289:APFCGA>2.0.CO;2
- Boisse, P. (1991) Transfer of continuum UV radiation inside fragmented clouds. IAU Symp. doi:10.1017/S0074180900199127
- Brantley, P.S. (2011) A benchmark comparison of Monte Carlo particle transport algorithms for binary stochastic mixtures. JQSRT. doi:10.1016/j.jqsrt.2010.06.007
- Buizza, R., Miller, M., Palmer, T.N. (1999) Stochastic representation of model uncertainties in the ECMWF ensemble prediction system. QJRMS. doi:10.1002/qj.49712556006
- Cahalan, R.F., Ridgway, W., Wiscombe, W.J., Bell, T.L. (1994) The albedo of fractal stratocumulus clouds. J. Atmos. Sci. doi:10.1175/1520-0469(1994)051<2434:TAOFSC>2.0.CO;2
- Crommelin, D., Vanden-Eijnden, E. (2008) Subgrid-scale parameterization with conditional Markov chains. J. Atmos. Sci. doi:10.1175/2008JAS2566.1
- Davis, A.B., Marshak, A. (2004) Photon propagation in heterogeneous optical media with spatial correlations. JQSRT. doi:10.1016/S0022-4073(03)00114-6
- d'Eon, E. (2019) A reciprocal formulation of nonexponential radiative transfer. 3: Binary mixtures. arXiv:1903.08783 (no Crossref record).
- Fischer, W., Meier-Hellstern, K. (1993) The Markov-modulated Poisson process (MMPP) cookbook. Performance Evaluation. doi:10.1016/0166-5316(93)90035-S
- Fouque, J.-P., Garnier, J., Papanicolaou, G., Solna, K. (2007) Wave Propagation and Time Reversal in Randomly Layered Media. Springer. doi:10.1007/978-0-387-49808-9
- Franzke, C., Majda, A.J., Vanden-Eijnden, E. (2005) Low-order stochastic mode reduction for a realistic barotropic model climate. J. Atmos. Sci. doi:10.1175/JAS3438.1
- Franzke, C., Crommelin, D., Fischer, A., Majda, A.J. (2008) A hidden Markov model perspective on regimes and metastability in atmospheric flows. J. Climate. doi:10.1175/2007JCLI1751.1
- Gabriel, K.R., Neumann, J. (1962) A Markov chain model for daily rainfall occurrence at Tel Aviv. QJRMS. doi:10.1002/qj.49708837511
- Gottwald, G.A., Crommelin, D.T., Franzke, C.L.E. (2016) Stochastic climate theory. In Nonlinear and Stochastic Climate Dynamics. doi:10.1017/9781316339251.009
- Hasselmann, K. (1976) Stochastic climate models Part I. Theory. Tellus. doi:10.1111/j.2153-3490.1976.tb00696.x
- Hobson, M.P., Scheuer, P.A.G. (1993) Radiative transfer in a clumpy medium - I. Analytical Markov-process solution for an N-phase slab. MNRAS. doi:10.1093/mnras/264.1.145
- Kassianov, E. (2003) Stochastic radiative transfer in multilayer broken clouds. Part I: Markovian approach. JQSRT. doi:10.1016/S0022-4073(02)00170-X
- Katz, R.W. (1974) Computing probabilities associated with the Markov chain model for precipitation. J. Appl. Meteor. doi:10.1175/1520-0450(1974)013<0953:CPAWTM>2.0.CO;2
- Katz, R.W., Parlange, M.B. (1998) Overdispersion phenomenon in stochastic modeling of precipitation. J. Climate. doi:10.1175/1520-0442(1998)011<0591:OPISMO>2.0.CO;2
- Katz, R.W., Parlange, M.B., Tebaldi, C. (2003) Stochastic modeling of the effects of large-scale circulation on daily weather in the southeastern U.S. Climatic Change. doi:10.1023/A:1026054330406
- Kelly, B.C., Bechtold, J., Siemiginowska, A. (2009) Are the variations in quasar optical flux driven by thermal fluctuations? ApJ. doi:10.1088/0004-637X/698/1/895
- Khouider, B., Majda, A.J., Katsoulakis, M.A. (2003) Coarse-grained stochastic models for tropical convection and climate. PNAS. doi:10.1073/pnas.1634951100
- Kiedrowski, B.C., Vu, E.H. (2024) Transit-length distribution for particle transport in binary Markovian mixed media. arXiv:2412.19359; SSRN doi:10.2139/ssrn.5200776
- Kondrashov, D., Ide, K., Ghil, M. (2004) Weather regimes and preferred transition paths in a three-level quasigeostrophic model. J. Atmos. Sci. doi:10.1175/1520-0469(2004)061<0568:WRAPTP>2.0.CO;2
- Kostinski, A.B. (2001) On the extinction of radiation by a homogeneous but spatially correlated random medium. JOSA A. doi:10.1364/JOSAA.18.001929
- Kramer, M., Lyne, A.G., O'Brien, J.T., Jordan, C.A., et al. (2006) A periodically active pulsar giving insight into magnetospheric physics. Science. doi:10.1126/science.1124060
- Larmier, C., Hugot, F.-X., Malvagi, F., Mazzolo, A., Zoia, A. (2017) Benchmark solutions for transport in d-dimensional Markov binary mixtures. JQSRT. doi:10.1016/j.jqsrt.2016.11.015
- Larmier, C., Lam, A., Brantley, P., Malvagi, F., et al. (2018) Monte Carlo chord length sampling for d-dimensional Markov binary mixtures. JQSRT. doi:10.1016/j.jqsrt.2017.09.014
- Larsen, E.W., Vasques, R. (2011) A generalized linear Boltzmann equation for non-classical particle transport. JQSRT. doi:10.1016/j.jqsrt.2010.07.003
- Levermore, C.D., Pomraning, G.C., Sanzo, D.L., Wong, J. (1986) Linear transport theory in a random medium. J. Math. Phys. doi:10.1063/1.527320
- Levermore, C.D., Wong, J., Pomraning, G.C. (1988) Renewal theory for transport processes in binary statistical mixtures. J. Math. Phys. doi:10.1063/1.527997
- Lyne, A., Hobbs, G., Kramer, M., Stairs, I., et al. (2010) Switched magnetospheric regulation of pulsar spin-down. Science. doi:10.1126/science.1186683
- Majda, A.J., Timofeyev, I., Vanden-Eijnden, E. (1999) Models for stochastic climate prediction. PNAS. doi:10.1073/pnas.96.26.14687
- Majda, A.J., Timofeyev, I., Vanden-Eijnden, E. (2001) A mathematical framework for stochastic climate models. CPAM. doi:10.1002/cpa.1014
- Majda, A.J., Franzke, C.L., Fischer, A., Crommelin, D.T. (2006) Distinct metastable atmospheric regimes despite nearly Gaussian statistics: a paradigm model. PNAS. doi:10.1073/pnas.0602641103
- Malvagi, F., Levermore, C.D., Pomraning, G.C. (1989) Asymptotic limits of a statistical transport description. Transp. Theory Stat. Phys. doi:10.1080/00411458908204690
- Malvagi, F., Pomraning, G.C. (1990) Renormalized equations for linear transport in stochastic media. J. Math. Phys. doi:10.1063/1.528824
- Malvagi, F., Byrne, R.N., Pomraning, G.C., Somerville, R.C.J. (1993) Stochastic radiative transfer in a partially cloudy atmosphere. J. Atmos. Sci. doi:10.1175/1520-0469(1993)050<2146:SRTIPC>2.0.CO;2
- Natta, A., Panagia, N. (1984) Extinction in inhomogeneous clouds. ApJ. doi:10.1086/162681
- O'Doherty, R.F., Anstey, N.A. (1971) Reflections on amplitudes. Geophys. Prospecting. doi:10.1111/j.1365-2478.1971.tb00610.x
- Palmer, T.N. (2001) A nonlinear dynamical perspective on model error. QJRMS. doi:10.1002/qj.49712757202
- Palmer, T.N. (2019) Stochastic weather and climate models. Nature Reviews Physics. doi:10.1038/s42254-019-0062-2
- Pomraning, G.C. (1991) Linear Kinetic Theory and Particle Transport in Stochastic Mixtures. World Scientific. doi:10.1142/1549
- Pomraning, G.C. (1996) Small correlation length solutions for planar symmetry beam transport in a stochastic medium. Ann. Nucl. Energy. doi:10.1016/0306-4549(95)00080-1
- Pomraning, G.C. (1998) Radiative transfer and transport phenomena in stochastic media. Int. J. Eng. Sci. doi:10.1016/S0020-7225(98)00050-0
- Pomraning, G.C. (1994) Linear kinetic theory and particle transport in stochastic mixtures, progress report (OSTI). doi:10.2172/10158144
- Prigarin, S.M., Kargin, B.A., Oppel, U.G. (1998) Random fields of broken clouds and their associated direct solar radiation, scattered transmission and albedo. Pure Appl. Opt. doi:10.1088/0963-9659/7/6/017
- Richardson, C.W. (1981) Stochastic simulation of daily precipitation, temperature, and solar radiation. Water Resour. Res. doi:10.1029/WR017i001p00182
- Rickett, B.J. (1990) Radio propagation through the turbulent interstellar plasma. ARA&A. doi:10.1146/annurev.aa.28.090190.003021
- Scargle, J.D. (1998) Studies in astronomical time series analysis. V. Bayesian blocks. ApJ. doi:10.1086/306064
- Solna, K., Papanicolaou, G. (2000) Ray theory for a locally layered random medium. Waves in Random Media. doi:10.1088/0959-7174/10/1/311
- Stalevski, M., Fritz, J., Baes, M., Nakos, T., et al. (2012) 3D radiative transfer modelling of the dusty tori around AGN as a clumpy two-phase medium. MNRAS. doi:10.1111/j.1365-2966.2011.19775.x
- Su, B., Pomraning, G.C. (1994) Limiting correlation length solutions in stochastic radiative transfer. JQSRT. doi:10.1016/0022-4073(94)90019-1
- Titov, G.A. (1990) Statistical description of radiation transfer in clouds. J. Atmos. Sci. doi:10.1175/1520-0469(1990)047<0024:SDORTI>2.0.CO;2
- van Haasteren, R., Levin, Y. (2012) Understanding and analysing time-correlated stochastic signals in pulsar timing. MNRAS. doi:10.1093/mnras/sts097
- Vanderhaegen, D. (1986) cited by d'Eon for the exact two-exponential law. UNVERIFIED.
- Vasques, R., Larsen, E.W. (2014) Non-classical particle transport with angular-dependent path-length distributions. I: Theory. Ann. Nucl. Energy. doi:10.1016/j.anucene.2013.12.021
- Wilks, D.S. (1998) Multisite generalization of a daily stochastic precipitation generation model. J. Hydrology. doi:10.1016/S0022-1694(98)00186-3
- Witt, A.N., Gordon, K.D. (1996) Multiple scattering in clumpy media. I. ApJ. doi:10.1086/177282
- Zimmerman, G.B., Adams, M.L. (1991) Algorithms for Monte Carlo particle transport in binary statistical mixtures. UNVERIFIED (no Crossref match).
