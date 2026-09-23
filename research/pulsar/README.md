# Pulsar spin-down switching: the Green-Kubo number in real data

## Question
Many pulsars switch between two spin-down rates. To first order the long-run timing noise of a switching pulsar is set
by one number, the Green-Kubo integral of the spin-down rate,

    K = int_0^inf Cov(nudot(0), nudot(t)) dt,

so that the frequency wanders with variance 2 K T and the phase with variance (2/3) K T^3
(see https://homogenization.microprediction.org/pulsars.html). Timing data identify the averaged spin-down and K, not
the individual rates and switching times. Questions:
1. Can K be measured directly from real spin-down series, and how large is it for known switchers?
2. Does a two-level switching model, fitted to the levels and dwell times, reproduce the measured K, and does the
   Markov or the quasi-periodic (renewal) form fit better?
3. Is K a useful, physically parameterized alternative to the power-law red-noise model in timing analysis?

## Data
Keith & Nitu (2023), spin-down time series for 17 Jodrell Bank pulsars, Zenodo doi:10.5281/zenodo.7664166 (CC-BY 4.0).
Columns: MJD, nudot (Hz/s), nudot error. The series are derived by Fourier-basis Gaussian-process regression of timing
residuals, so they are smoothed: short-time structure is suppressed, but the low-frequency content that K measures
is constrained by the timing data. Run `python3 fetch_data.py` to download into data/ (gitignored).

## Layout
- fetch_data.py: download the data.
- analysis/: numbered scripts; each writes its outputs to analysis/out/.
- FINDINGS.md: dated log of results, with the numbers and what they do and do not show.

## Plan
1. Explore: detrend each series, plot, look for two-level structure and quasi-periodicity.
2. Measure K from the integrated autocovariance (with the window and detrending choices stated).
3. Fit two-level models (levels, occupation fractions, dwell times) where switching is visible; compute the Markov and
   renewal predictions of K; compare with step 2.
4. Timing-noise check: predict the frequency and phase wander from K and compare with the integrated series.
5. Only then consider fitting timing residuals directly (TOAs are not in this data set).
