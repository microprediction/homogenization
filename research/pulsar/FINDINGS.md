# Findings log

## 2026-09-23: data loaded
17 pulsars, spans about 23.6 years (B1828-11: MJD 49202 to 57818), nudot sampled every 1.5 days after GP smoothing.

## 2026-09-23: first look (analysis/01_explore.py)
Units: nudot in 1e-15 Hz/s, time in days, K in (1e-15 Hz/s)^2 days.
Series shapes (out/01_series.png): B1828-11, B1540-06 and B1929+20 are quasi-periodic. B2035+36 and J2043+2740 are
step-like. B1822-09 is spiky.
The Gaussian-mixture BIC is not a two-level test: a sinusoid is bimodal too, so every series "prefers" two levels.

## 2026-09-23: K has no plateau for most pulsars (analysis/02_regularity.py, out/02_batch_means.png)
The batch-means estimate tau Var(block mean)/2 against block length tau falls in two groups.
- Falling at long blocks: B1540-06, B1642-03, B1714-34, B1826-17, B1828-11, B2148+63, B1929+20. Successive swings
  cancel, so the long-run integral of the autocovariance is much smaller than variance times correlation time.
  K_meas / (Var x tau_c) ranges from 0.075 (B1828-11) to 0.4.
- Still rising at 23 years: B2035+36, J2043+2740, B1822-09, B0919+06, B0740-28. The switching time is not short
  compared with the span, so K is at best bounded below.
The plateau median used in 01 is therefore not a usable K. From here on K_data is the six-block estimate
(block length = span/6, chi-square 68% interval, width roughly a factor of 4).

## 2026-09-23: renewal beats Markov (analysis/03_dwells.py, out/03_dwells.json)
Two-state fit by median threshold with hysteresis (0.25 sd). The square wave explains 50-73% of the variance
(B1822-09: 20%). From completed dwells: Delta, mean dwells m, coefficient of variation of dwells cv.
Predictions compared with K_data (log10 of prediction / data):

| predictor | median abs log10 error | mean log10 error | inside 68% interval |
|---|---|---|---|
| Markov, Delta^2 pi1 pi2 / gamma | 0.42 | +0.29 | 6/17 |
| renewal, Delta^2 (m2^2 s1^2 + m1^2 s2^2) / (2 (m1+m2)^3) | 0.29 | -0.14 | 8/17 |

The difference is carried by the regular switchers (dwell cv < 0.6: B0950+08, B1540-06, B1714-34, B1828-11,
B1929+20, B2148+63). Markov overpredicts all six, by factors 1.7 to 38. Renewal errors are
-0.39, -1.37, +0.09, +0.88, -0.72, +0.59 in log10: right order on average, no bias sign.
- B1714-34: Markov 1.07, renewal 0.064, data 0.052 [0.033, 0.13].
- B1828-11: Markov 25, renewal 5.0, data 0.67 [0.42, 1.6]. Still high: up and down dwells alternate
  (a quasi-period), which the independent-dwell renewal formula ignores.
- B1540-06 (cv 0.07-0.10): renewal 0.00017, data 0.0040. Nearly clockwork dwells send the renewal K to zero; the
  residual K in the data comes from amplitude and period wander not in the two-state model.
Bursty step-like pulsars have cv > 1 and K above Markov, as renewal says:
- B2035+36 (cv 2.25, 1.73): Markov 29.5, renewal 119, data 79 [50, 192].
- J2043+2740: Markov 1310, renewal 1460, data 1940 [1220, 4700]; only 8 switches.

Reading: the Green-Kubo number that sets the long-run frequency and phase wander is controlled by dwell-time
regularity, not by the size or rate of the switch. Clock-like switchers accumulate far less timing noise than a
Markov model with the same levels and rates predicts. This is the renewal K formula's content, and the data follow it.

Caveats to settle next:
- K_data is from six blocks. A spectral estimate at low frequency and a simulation check of the estimator on
  fitted renewal models (bias and interval coverage at this span) are needed before any claim.
- Threshold and hysteresis choices; detrending (linear) interacts with the longest dwells.
- Alternating renewal with correlated consecutive dwells (up then down) for the quasi-periodic group.

## 2026-09-23: the estimator is biased, and the fair test is mixed (analysis/04, 05)
04: simulated renewal square waves on each pulsar's grid, same detrend and six-block estimator.
- Typical estimate/true ratio 0.6-0.75, 68% interval coverage about 0.6: mildly low, usable.
- Clockwork dwells (B1540-06): estimate is 13x the true K. Blocks cut cycles, and the edge pieces dominate a
  tiny true K. So "renewal predicts too little" in the table above is mostly the estimator.
- Bursty long dwells (B2035+36, J2043+2740): estimate is 0.2-0.4 of true K.
So the comparison in the previous entry was not like for like. It is replaced by the test below.

05: percentile of the observed K among 400 simulated six-block K under each fitted model (strained: <5 or >95).

| group | pulsars | Markov | renewal |
|---|---|---|---|
| excess low-frequency power, both fail high | B0740-28, B0919+06, B1822-09, B2035+36, J2043+2740 | 99-100 | 100 |
| renewal fits, Markov too high | B1714-34, B2148+63 | 0, 2 | 28, 10 |
| both fit | B0950+08, B1642-03, B1818-04, B1826-17, B1839+09, B1907+00 | 22-73 | 38-88 |
| Markov fits, renewal too low | B1540-06, B1929+20, B1903+07 | 7-77 | 96-100 |
| both too high | B1828-11 | 0 | 2 |

Strained: Markov 8 of 17, renewal 9 of 17. The earlier "renewal beats Markov" does not survive a like-for-like test.

Reading now:
- Five pulsars have more long-run K than any two-state fit gives. There is a slower process under the switching,
  the usual red timing noise. A two-state model with a slow Gaussian component (K adds) is the next model.
- Dwell regularity matters for B1714-34 and B2148+63, where Markov is rejected and renewal is not.
- B1828-11 has less K than either model: consecutive dwells are anticorrelated (a genuine quasi-period).
  Needs an alternating renewal with correlated dwells, or a phase-diffusion oscillator, whose K is set by the
  phase diffusion rate, not by the dwell spread.
- B1540-06 and B1929+20: independent-dwell renewal gives too little. Amplitude or period wander adds K.

## 2026-09-23: stage (Erlang ring) models fitted to the autocovariance (analysis/06_stage_fit.py)
Fitting to the autocovariance, not to thresholded dwells. Thresholding a noisy series chops dwells and
overstates their spread. k stages per regime give dwell cv 1/sqrt(k); K comes from the group inverse of the stage
chain, checked against the renewal formula to 14 digits (scratch check, k = 1 to 30). Fits need multi-start
(periodogram and dwell hints); the first runs landed in poor local minima.

Percentile of observed six-block K among 400 simulated paths of the fitted model:
- Ring alone is consistent (5-95) for 12 of 17.
- Ring fails high for B0740-28, B0919+06, B1822-09, B1839+09, B1929+20. Adding a slow OU component brings
  B0740-28 (94), B1839+09 (64), B1929+20 (63) in; B0919+06 is marginal (96); B1822-09 fails (99): it is spiky and
  glitch-like, not a switcher.
- B1828-11 now fits: ring k = 64, dwells 359 and 130 days, percentile 24. The threshold fit had too much dwell spread.
- Compare the threshold-based Markov test: 8 of 17 strained.

Selected k is never 1 (Markov). Lowest is 2 (B1822-09); most are 8 to 64, dwell cv 0.12 to 0.35.

Caveat that may explain the large k: the series are Gaussian-process smoothed. Smoothing flattens the
autocovariance near lag 0, and a large-k ring also has a flat top there. Need the GP kernel length used by
Keith & Nitu, then fit the ring convolved with that kernel, or fit only lags beyond a few kernel lengths.
OU time scales hitting the 5e4-day bound (B1903+07, B1907+00, J2043+2740) are not identified; those K_model
values are meaningless, although the percentile test still simulates the fitted model.
