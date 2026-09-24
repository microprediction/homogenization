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

## 2026-09-23: placing the work against Cotton (2020)
arXiv:2006.07341 already has the multi-community mechanism of 01_communities.py: mean early growth G, harmonic mean
J at the peak, peak dispersion F. The simulation in 01 is a regime-switching instance of that paper's growth
convexity, and it agrees with it: national growth runs at the mean rate, the national peak comes early.
What the homogenization work adds to the 2020 paper:
- Markov-switching transmission in place of a static mixture or an OU rate: the mean growth rate is exactly the top
  eigenvalue of Q + diag(beta_i - gamma), and to first order G = 1 + K_bb / beta_bar with K the Green-Kubo number.
- SEIR and cycle direction (epidemics page): non-commuting compartments, where the antisymmetric part of K matters.
- Individual activity switching (04): closed-form growth rate, R0 and immunity factor, lam - 1 ~ r K_a.
County data (02, 03) are too noisy to test either version, as expected.

## 2026-09-23: what fits the county waves (analysis/06_waves.py, 07_model_fits.py)
Deaths, not cases. NYT county deaths 2020 to June 2021 (NYC as one unit), Census 2019 populations, counties over
50,000 people, 3-week centred weekly deaths. A wave is a peak with prominence >= 30% of the county maximum and
>= 10 deaths a week. Early growth r: log-linear slope over the contiguous rise between about 10% and 60% of the
peak. R = (1 + r theta)^k (generation interval mean 5.5 d, sd 2.1 d). Attack = cumulative deaths / (IFR * pop),
at the start of the rise and at the death peak (which marks the infection peak about 3 weeks earlier).
471 waves in 372 counties at IFR 0.7%.

- 95-100% of waves turn over below the homogeneous threshold 1 - (1 - a_start)/R.
- The attack at the turnover is flat across growth quartiles: spring 2020 about 4%, summer 5-7%, winter 15-17%
  cumulative, while the homogeneous threshold rises from about 21% to 45% across the same quartiles.
- Model fits for the log depletion during the wave, Delta = -ln((1 - a_turn)/(1 - a_start)) (winter, IFR 0.7%):

| model | resid sd | BIC | parameters |
|---|---|---|---|
| homogeneous, Delta = ln R | 1.49 | 271 | - |
| persistent, ln R / lam | 0.79 | -151 | lam 3.5 |
| switching, ln R / (lam0 + K r) | 0.70 | -237 | lam0 1 (bound), K 54 d |
| fixed increment, Delta = c | 0.68 | -254 | c 0.071 |
| free power, c (ln R)^b | 0.68 | -249 | b -0.03 |

  Same ranking at IFR 0.5% and 1% and in summer 2020. Spring 2020: switching, fixed increment and free power tie
  (b = 0.34), all far ahead of homogeneous and persistent.
- Reading: a wave depletes about the same share of susceptibles whatever its early growth. Homogeneous SIR and
  constant-lam heterogeneity both need depletion to scale with ln R, and fail. Switching activity reproduces the
  flat relation because lam grows with r (lam - 1 ~ r K), with K = 54 days at IFR 0.7% (e.g. activity variance 2
  and a 27-day correlation time), but it has to sit at lam0 = 1 and mimic a fixed increment; it does not beat it.
  A behavioural response triggered by deaths, or mitigation, gives the fixed increment directly.
- Residual sd 0.7 in log depletion: county IFR varies with age structure, and reporting varies. Age-adjusted IFR is
  the next refinement. The discriminating test between switching and behaviour is the rebound: switching predicts
  a new wave after about the activity correlation time, independent of deaths; a behavioural response predicts
  relaxation tied to falling deaths.

## 2026-09-23: correction to the model ranking above
The "fixed increment" row is not a mechanism: it is the observed flat relation summarized by one constant fitted
after the fact, and it says nothing about the level. It is the target a mechanism has to hit, not a competitor.
The switching model does predict a level: with lam0 = 1, Delta = ln R / (1 + K r) and ln R ~ r T_g, so fast waves
plateau at Delta ~ T_g / K (0.10 at K = 54 d; winter median observed 0.075).
Out-of-sample by season (IFR 0.7%), median log(observed / predicted):

| K fitted on | spring | summer | winter |
|---|---|---|---|
| spring (119 d) | -0.01 | -0.16 | +0.69 |
| summer (130 d) | +0.07 | -0.09 | +0.77 |
| winter (54 d) | -0.69 | -0.80 | +0.10 |
| homogeneous | -2.28 | -2.17 | -1.26 |

One K carries across spring and summer; winter waves depleted about twice as much as that K predicts. K is not
stable across seasons, which it should be if it were a property of activity alone. Mitigation (strong in spring
and summer, weaker in winter) lowers transmission outside the switching model and contaminates K; a changing IFR
over 2020 may also contribute. Next: an independent K (activity variance and correlation time from contact or
mobility data) to predict the level without fitting it, and an age-adjusted IFR.

## 2026-09-23: the flat relation is largely a noise artefact (analysis/08_errors_in_variables.py)
Growth from a few weeks of smoothed deaths is noisy (reporting delays, small counts), and noise in the regressor
flattens a fitted slope. Instrument: case growth over the same rise window shifted 3 weeks earlier. Elasticity b in
log Delta = a + b log(ln R): b = 1 for homogeneous or constant-lam heterogeneity, b = 0 for a flat relation.

| season | n | corr(r_deaths, r_cases) | OLS b | IV b | IV 90% bootstrap |
|---|---|---|---|---|---|
| spring 2020 | 56 | 0.57 | 0.39 | 0.91 | [0.46, 1.50] |
| summer 2020 | 66 | 0.18 | -0.23 | 3.61 | [-14.7, 16.8] |
| winter 2020-21 | 326 | 0.23 | 0.01 | 2.93 | [0.70, 12.5] |

Death-based and case-based growth agree poorly (correlation 0.2-0.6), so the death-based growth is mostly noise
and the OLS slope is attenuated toward zero. Corrected, spring gives b = 0.91 and winter excludes b = 0; summer is
uninformative. The data are consistent with depletion proportional to ln R, i.e. a constant immunity factor
(winter about 3.5, spring about 9.5 at IFR 0.7%), not with the growth-dependent factor that is the switching
signature. The earlier fits (07) and the out-of-sample K table were fitting the attenuated slope.
What survives: waves turn over far below the homogeneous threshold (a large lam or mitigation), and the level is
not explained by homogeneous SIR. Spatial or persistent heterogeneity fits the shape at least as well as switching
once the noise is accounted for. Caveats: the instrument is weak in summer and winter (IV then has wide intervals
and small-sample bias toward OLS), and case growth carries testing trends.

## 2026-09-23: the slope is not identified by these growth measures (analysis/09_rule_table.py)
Winter waves binned by case-based R (quartiles). Observed share of susceptibles infected before the turnover,
IFR 0.7%: 6.5%, 7.0%, 6.9%, 8.8% at R_cases 1.08, 1.16, 1.23, 1.34 (textbook 1 - 1/R: 7.8%, 13.8%, 18.9%, 25.5%).
Death-based R across the same bins: 1.28, 1.25, 1.28, 1.35. The two growth measures barely agree, so binned on case
growth the relation is flat again, and the large IV slope in 08 mostly reflects the weak instrument (dividing by a
small covariance). The earlier entry's "the flat relation is largely a noise artefact" overstates it: with these
data the slope is not identified; the level is. Constant-lam fit: 3.4 (death growth) or 2.1 (case growth) at
IFR 0.7%. A cleaner growth measure (hospital admissions, or serology-anchored national data) is needed for the slope.

## 2026-09-23: Spain, first wave, 52 provinces (analysis/10_spain.py, data/spain via build_spain.py)
ENE-COVID round 3 serology (rapid IgG test, 8-22 June 2020), ISCIII daily admissions and deaths by province.
- Turnovers are set by the lockdown: admissions peak between 23 and 31 March in almost every province, 9-17 days
  after the 14 March state of alarm, regardless of growth or immunity.
- Attack at the turnover 0.3-4.6% (median 2%) against textbook thresholds 46-76%: an immunity factor near 60,
  which is the lockdown, not heterogeneity.
- Growth from admissions and from deaths are uncorrelated across provinces (0.04); the slope is not identified.
- Spain's first wave does not test the paradox. The autumn 2020 wave, with ENE-COVID round 4 (November 2020)
  serology by province, would: change in seroprevalence June to November against the growth of the autumn wave,
  without a national lockdown. Round-4 provincial tables still to fetch.

## 2026-09-23: Spain, autumn 2020, by province (analysis/11_spain_autumn.py)
ENE-COVID round 4 seroconversion (infections June to early November among first-wave susceptibles), scaled to the
autumn death peak with cumulative deaths from 1 July. Weekly growth from admissions and deaths. 24 provinces with
growth and a death peak before the round-4 sampling.
- Depletion at the turnover: median 2.6% of susceptibles, textbook median 17%; lam about 7; every province below.
- Across growth quartiles the depletion is 2.4-4.2% while the textbook rises from 12% to 26%.
- Admissions-based and death-based growth are again uncorrelated (0.04): the slope is not identified.
- Confound: national curfew from 25 October 2020 (RD 926/2020) and regional restrictions; many provinces peaked
  one to two weeks later.
Same picture as the US counties: a robust level (waves turn over far below the textbook threshold), no usable slope.

## 2026-09-23: age, Sweden, England, Geneva, Manaus (analysis/12-14)
Age structure alone (12_age.py; Prem et al. 2021 synthetic contact matrices, World Bank 2020 populations, no
fitting): lam_age = [sum u v^2 / sum u v] / sum n v = 1.3-1.7 with equal susceptibility, 1.1-1.2 with under-20s at
half. First-wave peak attack at R = 2.5 is 51-56% against the textbook 60%. Age cannot supply lam of 2-7; the
heterogeneity that matters is within age groups (activity).

Turnovers anchored by serology (13_sweden.py, 14_points.py):

| place | control | turnover attack | lam |
|---|---|---|---|
| England, 9 ONS regions (REACT-2) | lockdown 23 Mar; death peaks 6-18 Apr | 1-5% | 17-94 |
| Geneva (SEROCoV-POP) | partial lockdown 16 Mar | 5% | 23 |
| Stockholm (FHM outpatient; Castro Dopico) | voluntary; ICU peak 4 Apr | 2-3% | 33-44 |
| Skane, Vastra Gotaland | voluntary | 1-3% | 20-45 |
| Spain spring (ENE-COVID) | national lockdown | 0.3-4.6% | ~60 |
| Spain autumn (ENE-COVID round 4) | curfew 25 Oct | 2.6% of susceptibles | ~7 |
| US counties, winter 2020-21 (IFR 0.7%) | mixed | ~7% of susceptibles | 2-3.5 |
| Manaus (Buss et al., excess deaths) | little effective control | ~17% at the peak; 66-76% final | 2.5 |

Where contacts were cut, by law or voluntarily (Sweden), the first wave turned at 1-5% and lam is 20-90: that is
the contact cut, not immunity. Where control was weak (Manaus, US winter) lam is 2-3.5, which needs activity
heterogeneity with CV^2 about 0.5-1.25 within age groups, close to the CV^2 ~ 1 of measured social contacts.
Manaus' final attack (66-76%) is close to the homogeneous final size at R = 1.6 (64%): the early turnover there
did not stop the epidemic from reaching high attack, as time-varying (transient) heterogeneity predicts and
persistent heterogeneity does not. Caveats: Manaus serology is monthly (the peak attack is interpolated between
5% on 11 April and 46% on 10 May) and blood donors are not the population; Stockholm growth from ICU admissions.
Download scripts copied to fetch/ (data/ stays gitignored).
