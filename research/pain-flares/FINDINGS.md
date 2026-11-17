# Findings log

## 2026-11-17: published curves (analysis/01_published_curves.py, out/01_published_curves.json)
J = (m - d)' C^-1 (m - d) on the published numbers, C their sampling covariance (empirical-process correlations
between shares of one curve; Costa's share correlated 0.60 with Henschke's one-year share, since the people overlap).

| curve (numbers) | coin, exact | first order, inside validity (abs(A) <= 0.1) | hidden two-state, exact | regime memory, 95% |
|---|---|---|---|---|
| episode length (4) | 56.5 | 56.5 | 4.8 | >= 16.4 days |
| time to next episode (4) | 34.4 | 34.4 | 0.5 | >= 73.3 days |
| care-seekers' recovery (5) | 586.7 | 557.9 | 18.4 | >= 68.7 days |

- Fast switching averages the hidden regime to one exponential, the coin. To first order, from a stationary start, a
  curve identifies only hbar - q1 q2 (h1 - h2)^2 / lam: the average hazard and its spread are confounded.
- The data want slow switching: the profile is flat down to a frozen regime (episodes or people of two kinds) and
  rises above the 95% line once the memory is shorter than the bounds above. On the first two curves the first-order
  amplitude A cancels (the numbers are conditional on a day, or an entry lag, already survived), so there the equality
  with the coin is an identity.
- Two kinds are still not enough for the care-seekers (J = 18.4 on 5 numbers, about 2 degrees of freedom).
- Migraine diaries (Barra 2020, five daily chances in the csv): a hidden regime lowers J by 1.5 only (not in a
  script here; the coin's J is 10.9).

## 2026-11-17: daily series (analysis/02_daily_fit.py, out/02_daily_fit.json)
density_first_order fitted by maximum likelihood (two-state chain, so the Green-Kubo matrix is rank one); estimate,
standard error, truth:

| series | kappa | level | s_eff | K_uw | K_ww | LM+ |
|---|---|---|---|---|---|---|
| 1 hour | 0.706 (0.029), 0.700 | 3.017 (0.035), 3.000 | 2.084 (0.070), 2.082 | 0.078 (0.030), 0.083 | 0.131 (0.057), 0.083 | 14.1 |
| 2 hours | 0.685 (0.028), 0.700 | 2.961 (0.036), 3.000 | 2.156 (0.071), 2.163 | 0.129 (0.029), 0.167 | 0.144 (0.055), 0.167 | 42.1 |
| 10 days | 0.300 (0.015), 0.700 | 3.142 (0.076), 3.000 | 1.880 (0.059), 21.6 | 0.729 (0.066), 20.0 | 0.501 (0.066), 20.0 | 472.3 |

- At hour-scale memory the first-order likelihood recovers every identified quantity within 1.3 standard errors;
  K_uw and K_ww carry 23 to 43% standard errors on ten years of daily data.
- At 10 days it breaks down: kappa is estimated at 0.30 against 0.70, and the first-order Green-Kubo values (20) are
  far outside the expansion's range.
- LM+ (95% point 5.14) rejects "no switching" on all three series. On real diaries, with whole-number ratings and
  many zeros, its Gaussian null would need recalibrating first.
