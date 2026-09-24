"""Spain, first wave, 52 provinces: early growth against the attack rate at the turnover, anchored by serology.

ENE-COVID round 3 (8-22 June 2020, midpoint 15 June) gives the attack; daily hospital admissions and deaths (ISCIII,
by province of residence) give the growth (waves_lib). The national lockdown (14 March) is common to all provinces.
Growth is measured on the contiguous rise; the attack at the turnover is the round-3 serology scaled back to the
death peak with cumulative deaths. Also: log depletion against log ln R with admissions and deaths as mutual
instruments."""
import numpy as np, pandas as pd, os, json
import waves_lib as wl
HERE = os.path.dirname(os.path.abspath(__file__)); D = os.path.join(HERE, "..", "data", "spain")
P = pd.read_csv(os.path.join(D, "spain_provinces.csv"), keep_default_na=False)
d = pd.read_csv(os.path.join(D, "spain_daily.csv"), keep_default_na=False, parse_dates=["date"])
print(P.columns.tolist(), d.columns.tolist())
rows = []
for _, p in P.iterrows():
    g = d[d.province_code == p.province_code].set_index("date").sort_index()
    if g.deaths.sum() < 30:
        continue
    o = wl.summarize(p["name"], g.deaths, g.hosp_admissions, p.sero_prevalence_pct / 100, "2020-06-15")
    o.update(code=p.province_code, pop=int(p.population_2020), deaths=int(g.deaths.sum()),
             adm_peak=str(wl.smooth(g.hosp_admissions.astype(float)).idxmax().date()))
    rows.append(o)
S = pd.DataFrame(rows)
S.to_csv(os.path.join(HERE, "out", "10_spain.csv"), index=False)
cols = ["place", "r_adm", "R_adm", "r_deaths", "R_deaths", "adm_peak", "peak_deaths", "sero", "attack_at_peak", "textbook_adm", "lam_adm"]
print(S[cols].sort_values("R_adm").to_string(index=False, float_format=lambda v: f"{v:.3f}"))
ok = S.dropna(subset=["R_adm", "R_deaths", "attack_at_peak"])
ok = ok[(ok.R_adm > 1) & (ok.R_deaths > 1)]
y = np.log(-np.log(1 - ok.attack_at_peak)); xa = np.log(np.log(ok.R_adm)); xd = np.log(np.log(ok.R_deaths))
print(f"\n{len(ok)} provinces. corr(r_adm, r_deaths) = {np.corrcoef(ok.r_adm, ok.r_deaths)[0,1]:.2f}")
print(f"elasticity of depletion in ln R: OLS on admissions {np.cov(y, xa)[0,1]/xa.var():.2f}, OLS on deaths {np.cov(y, xd)[0,1]/xd.var():.2f}, "
      f"IV (admissions, deaths instrument) {np.cov(y, xd)[0,1]/np.cov(xa, xd)[0,1]:.2f}, IV (deaths, admissions instrument) {np.cov(y, xa)[0,1]/np.cov(xd, xa)[0,1]:.2f}")
print(f"median attack at turnover {ok.attack_at_peak.median():.3f}, median textbook (admissions) {ok.textbook_adm.median():.3f}, "
      f"median lam (admissions) {ok.lam_adm.median():.2f}; share below textbook {np.mean(ok.attack_at_peak < ok.textbook_adm):.2f}")
lam_fit = float(np.exp(np.mean(np.log(np.log(ok.R_adm))) - np.mean(y)))
print(f"constant-lam fit (admissions): {lam_fit:.2f}")
q = pd.qcut(ok.R_adm, 4)
t = ok.groupby(q, observed=True).agg(n=("R_adm", "size"), R=("R_adm", "median"), observed=("attack_at_peak", "median"), sero=("sero", "median"))
t["textbook"] = 1 - 1 / t.R; t[f"rule lam={lam_fit:.1f}"] = 1 - t.R ** (-1 / lam_fit)
print(t.to_string(float_format=lambda v: f"{v:.3f}"))
