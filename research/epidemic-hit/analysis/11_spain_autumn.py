"""Spain, autumn 2020 wave, by province, anchored by ENE-COVID round-4 seroconversion.

Seroconversion (Tabla 12) = share of those seronegative through rounds 1-3 who were positive in round 4
(16-29 Nov 2020): infections among first-wave susceptibles from about June to early November. Antibodies at the
round-4 midpoint (22 Nov) reflect infections to about 8 Nov, i.e. deaths to about 29 Nov. The depletion at the
autumn turnover is seroconversion scaled by cumulative deaths from 1 July to the autumn death peak over cumulative
deaths from 1 July to 29 Nov. Growth from the contiguous rise of admissions and of deaths after 15 July.
R here is the effective R at the start of the autumn rise, so the textbook turnover is 1 - 1/R of the susceptibles."""
import numpy as np, pandas as pd, os
import waves_lib as wl
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "data", "spain")
P = pd.read_csv(os.path.join(D, "spain_provinces.csv"), keep_default_na=False)
d = pd.read_csv(os.path.join(D, "spain_daily.csv"), keep_default_na=False, parse_dates=["date"])
rows = []
for _, p in P.iterrows():
    g = d[d.province_code == p.province_code].set_index("date").sort_index().loc["2020-07-01":]
    if g.deaths.sum() < 30 or p.seroconv_r4_pct in ("", None):
        continue
    rA, _, _, pkA = wl.growth_weekly(g.hosp_admissions, start="2020-07-15")
    rD, _, _, pkD = wl.growth_weekly(g.deaths, start="2020-07-15")
    if pkD is None:
        continue
    Dc = g.deaths.cumsum()
    ratio = Dc.loc[:pkD].iloc[-1] / Dc.loc[:"2020-11-29"].iloc[-1]
    sc = float(p.seroconv_r4_pct) / 100
    dep = sc * ratio
    RA, RD = wl.R_from_r(rA), wl.R_from_r(rD)
    rows.append(dict(place=p["name"], r_adm=rA, R_adm=RA, r_deaths=rD, R_deaths=RD, adm_peak=None if pkA is None else str(pkA.date()),
                     death_peak=str(pkD.date()), seroconv=sc, peak_after_sero=pkD > pd.Timestamp("2020-11-29"),
                     depletion_at_peak=dep, textbook=1 - 1 / RA if RA > 1 else np.nan,
                     lam=np.log(RA) / -np.log(1 - dep) if RA > 1 and 0 < dep < 1 else np.nan))
S = pd.DataFrame(rows); S.to_csv(os.path.join(HERE, "out", "11_spain_autumn.csv"), index=False)
print(S.sort_values("R_adm").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
ok = S.dropna(subset=["R_adm", "R_deaths", "lam"]); ok = ok[(ok.R_deaths > 1) & ~ok.peak_after_sero]
y = np.log(-np.log(1 - ok.depletion_at_peak)); xa = np.log(np.log(ok.R_adm)); xd = np.log(np.log(ok.R_deaths))
print(f"\n{len(ok)} provinces with the death peak before 29 Nov. corr(r_adm, r_deaths) = {np.corrcoef(ok.r_adm, ok.r_deaths)[0,1]:.2f}")
print(f"elasticity: OLS adm {np.cov(y,xa)[0,1]/xa.var():.2f}, OLS deaths {np.cov(y,xd)[0,1]/xd.var():.2f}, "
      f"IV adm|deaths {np.cov(y,xd)[0,1]/np.cov(xa,xd)[0,1]:.2f}, IV deaths|adm {np.cov(y,xa)[0,1]/np.cov(xd,xa)[0,1]:.2f}")
print(f"median depletion {ok.depletion_at_peak.median():.3f}, median textbook {ok.textbook.median():.3f}, median lam {ok.lam.median():.2f}, "
      f"share below textbook {np.mean(ok.depletion_at_peak < ok.textbook):.2f}")
lamfit = float(np.exp(np.mean(xa) - np.mean(y)))
t = ok.groupby(pd.qcut(ok.R_adm, 4), observed=True).agg(n=("R_adm", "size"), R=("R_adm", "median"), observed=("depletion_at_peak", "median"))
t["textbook"] = 1 - 1 / t.R; t[f"rule lam={lamfit:.1f}"] = 1 - t.R ** (-1 / lamfit)
print(t.to_string(float_format=lambda v: f"{v:.3f}"))
