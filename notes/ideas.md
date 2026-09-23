# Ideas to consider

## Portfolio theory with fast regimes (noted 2026-09-23)
- Merton's problem with returns or volatilities switched by a fast chain: the value function solves an HJB
  equation, so the averaged problem plus a Green-Kubo correction should apply, including the antisymmetric
  (cycle-direction) term when switched operators do not commute.
- Prior art: Fouque, Sircar and Zariphopoulou (2017), portfolio optimization under fast and slow stochastic volatility.
- Questions: first correction to the optimal weights; whether a mean-variance allocator sees only the Green-Kubo matrix
  of expected returns and variances; connection to allocation work on schur.microprediction.org.

## Compartmental models of disease spread (noted 2026-09-23)
- Early epidemic phase: the linearized SIR/SEIR system is linear in the infected compartments, so a transmission rate
  switched by a fast environment (seasons, behaviour, policy regimes) gives exactly the engine's system
  a' = (Q + diag(beta_i - gamma)) a for expected prevalence. Expected growth rate = averaged rate plus the
  Green-Kubo term K(beta, beta): fluctuation raises the mean growth, while the typical (almost sure) growth is lower,
  the Lewontin-Cohen distinction already in the survey.
- Multi-compartment (SEIR, age structure): the next-generation matrices do not commute, so the antisymmetric
  (cycle-direction) term should appear, e.g. whether high contact precedes or follows high susceptibility.
- Case counts given a hidden regime are a Markov-modulated Poisson or branching process: the counts page and the
  likelihood page's Jarque-Bera-type test carry over to overdispersion in case data.
- Beyond the linear phase, the nonlinear SIR needs the first-order rule applied to a nonlinear generator (closure
  questions, as in the survey's turbulence section).
- Prior art to find and verify: basic reproduction number in random or periodic environments, epidemics in
  Markov-switching environments, stochastic SIS/SIR with telegraph noise.
