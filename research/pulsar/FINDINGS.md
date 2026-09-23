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
