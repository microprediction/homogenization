# Pain flares: what a hidden two-state regime can and cannot be seen in

## Context
MIRP, the Mathematical Institute for Research of Pain (https://painmath.org), studies pain as a stochastic process.
Under MIRP's LinkedIn post "Who Gets There First?", Peter Cotton suggested homogenization for our simplest rival model
and invited us to "dump raw time series as an issue or PR". This folder is that dump, with the analysis.

The pain question is plain: from today's state, does a flare or a recovery come first, and how long until either?
MIRP's main model is a double-well diffusion whose committor answers it. Its rivals are coins with no memory: a
two-state chain with constant flare and recovery rates, and the same chain with a hidden regime switching those rates.

The data are sparse. Each study prints 4 to 10 summary numbers (shares of episodes over by a date, medians), not
patient records: de Campos 2023, da Silva 2019, Henschke 2008, Costa 2009 and Barra 2020 (citations below). On such
numbers, homogenization says a fast hidden regime is invisible; the data instead want a slow one. Day-by-day ratings
are where the first-order likelihood earns its place.

**No real patient or personal data are included.** `published_flare_numbers.csv` holds printed summary statistics;
`data/` holds generated series only.

## Data
- `published_flare_numbers.csv`: 18 published numbers with standard errors (columns: curve, time in days, what, value,
  standard error, rounding allowance, unit, source). An episode is pain above 2/10 for at least 24 h; it is over at
  average pain 0 or 1/10 for 7 days. Study entry came within a month of recovery (lag median 14 days, IQR 7 to 27.5).
  Care-seekers recover at the start of a pain-free month; Costa's 406 are Henschke's people still in pain at 3 months.
- `make_data.py`: writes three synthetic daily series into `data/` (gitignored) from fixed seeds. The model is
  dX = kappa (theta_y - X) dt + sqrt(s_y) dW with kappa = 0.7/day, theta = (2, 5), s = (1, 4)/day, bad a third of the
  time, regime memory 1/lam of 1 hour, 2 hours and 10 days (lam = a + b; sha256 412463f2..., 2656c88d..., 7a9a8a82...).
  Columns: day, level, rating (level rounded and clipped to 0-10), and the ground truth regime and bad_share.

Sources: de Campos TF et al. Eur J Pain 2023;27:602-610, doi:10.1002/ejp.2083 · da Silva T et al. J Physiother
2019;65:159-165, doi:10.1016/j.jphys.2019.04.010 · Henschke N et al. BMJ 2008;337:a171, doi:10.1136/bmj.a171 ·
Costa LCM et al. BMJ 2009;339:b3829, doi:10.1136/bmj.b3829 · Barra M et al. Sci Rep 2020;10:3631,
doi:10.1038/s41598-020-60505-5.

## Layout
- `analysis/01_published_curves.py`: coin against a hidden two-state hazard on the three low back pain curves; profile
  likelihood over the regime memory (about 30 s).
- `analysis/02_daily_fit.py`: `papers/likelihood/density.py`, unchanged, fitted to the synthetic series; ends with PASS.
- `FINDINGS.md`: dated results.

Run: `python3 research/pain-flares/make_data.py`, then the two analysis scripts.
