"""County epidemic waves from deaths: early growth against the attack rate at the turnover.

Weekly deaths per county (NYT, 2020 to June 2021; New York City as one unit), 3-week centred mean. Each peak with
prominence >= 30% of the county maximum and >= 10 deaths a week is a wave. For each wave:
- r: log-linear slope of weekly deaths on the rising side between 10% and 60% of the peak, per day;
- R: from r with a gamma generation interval (mean 5.5 d, sd 2.1 d): R = (1 + r theta)^k;
- attack at turnover a*: cumulative deaths up to the peak week / (IFR * population). Deaths lag infections by about
  3 weeks, so the death peak marks the infection peak about 3 weeks earlier; cumulative deaths to the death peak
  count infections up to the infection peak.
- homogeneous threshold 1 - 1/R; immunity factor lam = ln R / (-ln(1 - a*)), from R S^lam = 1 at the turnover.
Homogeneous: lam = 1. Persistent heterogeneity: lam > 1, the same for fast and slow waves. Switching: lam grows with r."""
import numpy as np, pandas as pd, json, os, sys
from scipy.signal import find_peaks

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
IFR = float(os.environ.get("IFR", 0.007))
MEAN_GI, SD_GI = 5.5, 2.1
k_gi, th_gi = (MEAN_GI / SD_GI) ** 2, SD_GI ** 2 / MEAN_GI

d = pd.concat([pd.read_csv(os.path.join(DATA, f), dtype={"fips": str}, parse_dates=["date"])
               for f in ("us-counties-2020.csv", "us-counties-2021.csv")])
d.loc[(d.county == "New York City") & (d.state == "New York"), "fips"] = "NYC"
d = d[d.fips.notna() & (d.date <= "2021-06-30")]
cum = d.pivot_table(index="date", columns="fips", values="deaths").sort_index().ffill().fillna(0)
wk = cum.resample("W-SUN").last().diff().clip(lower=0).iloc[1:]
pop = pd.read_csv(os.path.join(DATA, "co-est2019-alldata.csv"), encoding="latin-1", dtype={"STATE": str, "COUNTY": str})
pop = pop[pop.SUMLEV == 50]
P = dict(zip(pop.STATE.str.zfill(2) + pop.COUNTY.str.zfill(3), pop.POPESTIMATE2019))
P["NYC"] = sum(P[f] for f in ("36005", "36047", "36061", "36081", "36085"))

waves = []
for f in wk.columns:
    if P.get(f, 0) < 50000:
        continue
    y = wk[f].values.astype(float); sm = pd.Series(y).rolling(3, center=True, min_periods=1).mean().values
    if sm.max() < 10:
        continue
    pk, props = find_peaks(sm, prominence=0.3 * sm.max(), height=10)
    cs = np.cumsum(y)
    for j, p in enumerate(pk):
        i1 = p
        while i1 > 0 and sm[i1] > 0.6 * sm[p]:
            i1 -= 1
        i0 = i1
        while i0 > 0 and sm[i0 - 1] < sm[i0] and sm[i0 - 1] >= 0.1 * sm[p]:
            i0 -= 1                                   # contiguous rise, monotone in the smoothed series
        seg = np.arange(i0, i1 + 1)
        if len(seg) < 3 or sm[i0] > 0.35 * sm[p]:
            continue
        r = np.polyfit(seg, np.log(sm[seg]), 1)[0] / 7
        if r <= 0:
            continue
        R = (1 + r * th_gi) ** k_gi                   # R at the start of the wave, R0 * S_start^lam
        a0 = cs[seg[0]] / (IFR * P[f])                # attack when the wave took off
        a = cs[p] / (IFR * P[f])
        hit = 1 - (1 - a0) / R                        # homogeneous turnover given the immunity at the start
        lam = np.log(R) / -np.log((1 - a) / (1 - a0)) if 0 <= a0 < a < 1 else np.nan
        waves.append(dict(fips=f, pop=int(P[f]), peak=str(wk.index[p].date()), r=r, R=R, attack_start=a0, attack=a,
                          hit_hom=hit, lam=lam, peak_deaths=float(sm[p]), npts=len(seg),
                          rise_start=str(wk.index[seg[0]].date()), rise_end=str(wk.index[seg[-1]].date())))
W = pd.DataFrame(waves)
W["season"] = pd.cut(pd.to_datetime(W.peak), pd.to_datetime(["2020-01-01", "2020-06-30", "2020-09-30", "2021-06-30"]),
                     labels=["spring 2020", "summer 2020", "winter 2020-21"])
W.to_csv(os.path.join(HERE, "out", f"06_waves_ifr{IFR}.csv"), index=False)
print(f"IFR {IFR}: {len(W)} waves in {W.fips.nunique()} counties")
for sname, g in W.groupby("season", observed=True):
    ok = g.lam.notna()
    print(f"\n{sname}: {len(g)} waves; median r {g.r.median():.3f}/day, median R {g.R.median():.2f}, "
          f"median homogeneous threshold {g.hit_hom.median():.2f}, median attack at start {g.attack_start.median():.2f}, at turnover {g.attack.median():.2f}; "
          f"share turning below threshold {np.mean(g.attack < g.hit_hom):.2f}; median lam {g.lam[ok].median():.2f}")
    q = pd.qcut(g.r, 4, duplicates="drop")
    t = g.groupby(q, observed=True).agg(n=("r", "size"), R=("R", "median"), hit=("hit_hom", "median"),
                                        attack=("attack", "median"), lam=("lam", "median"))
    print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    gg = g[ok & (g.lam > 0)]
    b = np.polyfit(gg.r, np.log(gg.lam), 1)
    print(f"  log lam = {b[1]:.2f} + {b[0]:.1f} r   (n = {len(gg)})")
