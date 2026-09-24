"""Sweden, first wave (no lockdown), by region: early growth against the attack at the turnover.

Growth from daily ICU admissions (Svenska Intensivvårdsregistret episodes, by region), 7-day mean, contiguous rise.
Serology: FHM outpatient residual-sera survey by region (weeks 17-24, 2020) and, for Stockholm, Castro Dopico et al.
(2021) weekly Bayesian estimates. Antibodies at date t reflect infections to about t - 14 d; ICU admission follows
infection by about 12 d, so antibodies at t correspond to ICU admissions to about t - 2 d. Attack at the infection
peak = seroprevalence(t) * ICU_cum(ICU peak) / ICU_cum(t - 2 d), averaged over first-wave samples (to 14 June)."""
import numpy as np, pandas as pd, os
import waves_lib as wl
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "data", "sweden")
d = pd.read_csv(os.path.join(D, "sweden_daily.csv"), dtype={"region_code": str}, parse_dates=["date"])
s = pd.read_csv(os.path.join(D, "sero_estimates.csv"), dtype={"region_code": str}, parse_dates=["sample_start", "sample_end"])
s["mid"] = s.sample_start + (s.sample_end - s.sample_start) / 2
rows = []
for study in ("FHM_outpatient", "FHM_blooddonors", "CastroDopico2021_Bayes"):
    ss = s[(s.study_id == study) & (s.mid <= "2020-06-14") & (s.region_code != "SE")]
    for reg, gs in ss.groupby("region_name"):
        g = d[d.region_name == reg].set_index("date").sort_index()
        icu = g.icu_admissions_sir_episodes.astype(float).loc[:"2020-07-31"]
        if icu.sum() < 40:
            continue
        r, _, _, pk = wl.growth(icu, end="2020-07-31")
        if not np.isfinite(r):
            r, _, _, pkw = wl.growth_weekly(icu, end="2020-07-31")
            pk = wl.smooth(icu).idxmax() if pk is None else pk
        C = icu.cumsum()
        att = [row.prevalence_pct / 100 * C.loc[:pk].iloc[-1] / C.loc[:row.mid - pd.Timedelta(days=2)].iloc[-1]
               for row in gs.itertuples() if C.loc[:row.mid - pd.Timedelta(days=2)].iloc[-1] > 0]
        a = float(np.mean(att)); R = wl.R_from_r(r)
        rows.append(dict(region=reg, study=study, n_sero=len(att), icu=int(icu.sum()), r_icu=r, R=R, icu_peak=str(pk.date()),
                         sero_mean=float(gs.prevalence_pct.mean() / 100), attack_at_peak=a,
                         textbook=1 - 1 / R if R > 1 else np.nan, lam=np.log(R) / -np.log(1 - a) if R > 1 and 0 < a < 1 else np.nan))
S = pd.DataFrame(rows); S.to_csv(os.path.join(HERE, "out", "13_sweden.csv"), index=False)
print(S.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
