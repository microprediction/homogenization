"""Point checks: England ONS regions (REACT-2), Geneva (SEROCoV-POP), Manaus (Buss et al.).

England: deaths by date of death (ONS certificates) per ONS region; REACT-2 round 1 (20 Jun-13 Jul, midpoint 1 Jul),
  test-adjusted and weighted; attack at the turnover by cumulative-death scaling (waves_lib.summarize).
Geneva: FOPH deaths and admissions; SEROCoV-POP week 5 (ref 6 May), Table 2 adjusted.
Manaus: excess deaths (all-cause 2020 minus the 2015-2019 same-day mean, SIM); the seroreversion-corrected cumulative
  attack series (Buss et al. Table S2), interpolated at the death peak minus 7 days (infection peak = death peak - 21 d,
  antibodies at t cover infections to t - 14 d)."""
import numpy as np, pandas as pd, os
import waves_lib as wl
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "data", "points")
rows = []
e = pd.read_csv(os.path.join(D, "england_daily.csv"), parse_dates=["date"])
es = pd.read_csv(os.path.join(D, "england_serology.csv"))
es = es[es.method.str.contains("re-weighted")]
for reg, g in e[e.geography == "ONS region"].groupby("place"):
    g = g.set_index("date").sort_index()
    sero = es[es.place == reg].prevalence_pct.iloc[0] / 100
    o = wl.summarize(f"England: {reg}", g.deaths.fillna(0), None, sero, "2020-07-01"); rows.append(o)
gv = pd.read_csv(os.path.join(D, "geneva_daily.csv"), parse_dates=["date"]).set_index("date").sort_index()
gs = pd.read_csv(os.path.join(D, "geneva_serology.csv"))
sero_g = gs[(gs.adjusted == "adjusted") & (gs.week == 5)].prevalence_pct.iloc[0] / 100
rows.append(wl.summarize("Geneva", gv.deaths.fillna(0), gv.hosp_admissions.fillna(0), sero_g, "2020-05-06"))
m = pd.read_csv(os.path.join(D, "manaus_daily.csv"), parse_dates=["date"]).set_index("date").sort_index()
base = m[[f"deaths_all_cause_{y}_same_day" for y in range(2015, 2020)]].mean(axis=1)
exc = (m.deaths_all_cause - base).clip(lower=0).loc[:"2020-10-31"]
r, _, _, pk = wl.growth(exc, end="2020-08-31")
ms = pd.read_csv(os.path.join(D, "manaus_serology.csv"), parse_dates=["date_start", "date_end"])
ms = ms[(ms.place == "Manaus") & ms.method.str.contains("seroreversion")].copy()
ms["mid"] = ms.date_start + (ms.date_end - ms.date_start) / 2
t = (pk - pd.Timedelta(days=7))
a = float(np.interp(t.value, ms.mid.astype("int64"), ms.prevalence_pct / 100))
R = wl.R_from_r(r)
rows.append(dict(place="Manaus (excess deaths)", r_deaths=r, R_deaths=R, peak_deaths=str(pk.date()), sero=float(ms.prevalence_pct.max() / 100),
                 attack_at_peak=a, textbook_deaths=1 - 1 / R, lam_deaths=np.log(R) / -np.log(1 - a)))
S = pd.DataFrame(rows); S.to_csv(os.path.join(HERE, "out", "14_points.csv"), index=False)
cols = ["place", "r_deaths", "R_deaths", "peak_deaths", "sero", "attack_at_peak", "textbook_deaths", "lam_deaths", "r_adm", "R_adm"]
print(S[[c for c in cols if c in S]].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
