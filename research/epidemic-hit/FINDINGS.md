# Findings log

## 2026-09-23: the mechanism works in simulation (analysis/01_communities.py)
4000 independent SIR communities, gamma = 0.2/day, beta = beta_bar +/- delta switching at rate lam each way, equal seeds.
- National early growth equals the exact mean rate, the top eigenvalue of Q + diag(beta_i - gamma):
  r_mean = beta_bar - gamma - lam + sqrt(lam^2 + delta^2) (e.g. 0.224 predicted, 0.221 simulated). One community
  grows at the typical rate beta_bar - gamma. First order: r_mean - r_typ = K = delta^2 / (2 lam).
- Fitting homogeneous SIR to the national early growth overstates R. With R_bar = 1.5:

| delta, lam | R from growth | 1 - 1/R | attack at national peak | final attack | z(R growth) |
|---|---|---|---|---|---|
| 0, - | 1.50 | 0.33 | 0.34 | 0.58 | 0.58 |
| 0.1, 0.1 | 1.69 | 0.41 | 0.27 | 0.63 | 0.69 |
| 0.2, 0.1 | 2.11 | 0.53 | 0.20 | 0.72 | 0.82 |
| 0.2, 0.03 | 2.34 | 0.57 | 0.13 | 0.78 | 0.87 |
| 0.25, 0.05 | 2.49 | 0.60 | 0.10 | 0.80 | 0.89 |

  The national curve turns over at 10-27% attack while the growth-implied threshold is 41-60%: the hot communities
  peak first. Infection continues after the national peak (final 63-80%), below the growth-implied final size.
- The effect needs many independent environments. In one community the observed growth is the typical rate.

## 2026-09-23: US counties, 2020 (analysis/02, 03)
Weekly county incidence, counties with >= 100 cases in both weeks.
- Cross-county sd of weekly log growth 0.25-0.47. Deviations do not persist week to week (lag-1 correlation -0.10,
  lags 2-8 within +/-0.07): the heterogeneity switches rather than persists.
- They do compound: the cross-county variance of log(new_{w+k}/new_w) is 0.103 + 0.051 k, so K = 0.026 per week
  on top of level noise (variance 0.05 per week-count).
- National growth minus median county growth: -0.0006 +/- 0.009 per week at k = 1, rising to about +0.04 at k = 12.
  Independent weights would give K k, about 0.3 at k = 12. Case-heavy counties grow more slowly next, which offsets
  most of the aggregate tilt (local depletion and behaviour).
- Size: K = 0.026 per week moves R by about 0.02 at a 5-day generation interval. At county scale the switching
  mechanism is real but far too small to explain R near 3 against a turnover near 20% immunity.

Reading: if switching explains the gap it has to act at a much finer scale (households, venues, individuals'
activity), where K is large. That is the setting of Tkachenko et al. (2021). What the homogenization view can add is
the analytic size: the gap in growth rates is the Green-Kubo integral K of the transmission rate, and the exact
linear-phase mean rate is an eigenvalue. Next: individual-level switching activity (SIR with activity states,
infection proportional to activity of both parties), derive K and the transient HIT analytically, and compare with
Tkachenko et al.'s fitted activity variance and correlation time and with serology (NYC, Chicago).

## 2026-09-23: individual activity switching, solved exactly (analysis/04, 05)
Tkachenko et al. (2021) have the non-local version: R_e ~ R0 S^lam with lam_eff = lam_inf + an incidence-weighted
integral of the activity autocorrelation (their Eqs. 17-19); they fit lam_eff ~ 4 for NYC and Chicago, lam_inf ~ 2,
and NYC seroprevalence 23% after the first wave.

Solvable version: activity a in {a1, a2} (mean 1, variance s2, delta^2 = c delta + s2), relaxing at rate kappa;
infection force on activity a is beta a Phi, Phi = sum_k a_k I_k. The resolvent of the activity chain splits on
{p, delta p}, which gives in closed form:
- growth rate: 1 = beta (1/(r+gamma) + s2/(r+gamma+kappa));   R0 = beta (1/gamma + s2/(gamma+kappa));
- immunity factor in the growth phase, rho = r/(r+kappa):
  lam = ((1 + s2 rho)/gamma + s2 (1 + (1+c) rho)/(gamma+kappa)) / (1/gamma + s2/(gamma+kappa)).
  kappa -> 0 gives <a^3>/<a^2> (Tkachenko's persistent value); kappa -> inf gives 1.
- Fast switching, first order: lam - 1 ~ r K_a, K_a = s2/kappa the Green-Kubo integral of individual activity.
  The excess immunity factor is the growth rate times the Green-Kubo number.
Checked against the ODE: r to 4 digits; lam to 0.1-0.3% (fit window attack < 0.3%).

NYC calibration (05): r0 = ln 2 / 3 per day, gamma = 0.2, gamma-like skew c = 2 s2, no mitigation.
Homogeneous reading R = 2.16, threshold 54%. Attack at the first peak of prevalence:

| s2 | 1/kappa (days) | lam | attack at first peak | long-run threshold 1 - 1/R0 |
|---|---|---|---|---|
| 0.5 | 10 | 1.6 | 0.40 | 0.51 |
| 1 | 10 | 2.3 | 0.31 | 0.50 |
| 1 | 100 | 2.9 | 0.25 | 0.53 |
| 2 | 10 | 3.6 | 0.22 | 0.48 |
| 2 | 33 | 4.5 | 0.17 | 0.51 |
| 4 | 10 | 6.3 | 0.13 | 0.46 |

s2 = 2 with a 10-day correlation time turns the first wave at 22% with lam = 3.6: NYC's 23% serology and
Tkachenko's lam ~ 4, with depletion alone. The long-run threshold stays near 50%.
Caveat: NYC locked down on 22 March 2020, which also cut transmission; these numbers show depletion is sufficient,
not that it was the cause. Separating the two needs places or periods without strong mitigation, or the
post-wave rebound timing, which the model ties to 1/kappa.
