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
