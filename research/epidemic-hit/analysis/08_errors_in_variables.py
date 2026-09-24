"""Is the flat relation an artefact of noisy growth estimates? An instrumental-variable check.

Each wave's growth r_D comes from a few weeks of smoothed deaths, with reporting delays and noise. Noise in the
regressor attenuates a fitted slope toward zero. Cases give a second estimate r_C over the same rise window shifted
3 weeks earlier (infection to death), with errors from testing rather than death reporting.
Elasticity b in log Delta = a + b log(ln R): OLS on the death-based R, and IV with the case-based R as instrument.
b = 1 for homogeneous or constant-lam heterogeneity; b = 0 for a flat relation."""
import numpy as np, pandas as pd, os
HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "..", "data")
MEAN_GI, SD_GI = 5.5, 2.1; k_gi, th_gi = (MEAN_GI / SD_GI) ** 2, SD_GI ** 2 / MEAN_GI
d = pd.concat([pd.read_csv(os.path.join(DATA, f), dtype={"fips": str}, parse_dates=["date"])
               for f in ("us-counties-2020.csv", "us-counties-2021.csv")])
d.loc[(d.county == "New York City") & (d.state == "New York"), "fips"] = "NYC"
d = d[d.fips.notna() & (d.date <= "2021-06-30")]
cum = d.pivot_table(index="date", columns="fips", values="cases").sort_index().ffill().fillna(0)
wkc = cum.resample("W-SUN").last().diff().clip(lower=0).iloc[1:]
W = pd.read_csv(os.path.join(HERE, "out", "06_waves_ifr0.007.csv"), dtype={"fips": str}, parse_dates=["rise_start", "rise_end"])
W = W[(W.attack > W.attack_start) & (W.attack < 1)].copy()
rc = []
for _, w in W.iterrows():
    c = wkc[w.fips].rolling(3, center=True, min_periods=1).mean()
    seg = c.loc[w.rise_start - pd.Timedelta(weeks=3): w.rise_end - pd.Timedelta(weeks=3)]
    seg = seg[seg > 0]
    rc.append(np.polyfit(np.arange(len(seg)), np.log(seg.values), 1)[0] / 7 if len(seg) >= 3 else np.nan)
W["r_C"] = rc
W = W[W.r_C > 0].copy()
W["D"] = -np.log((1 - W.attack) / (1 - W.attack_start))
W["x"] = np.log(np.log(W.R)); W["z"] = np.log(np.log((1 + W.r_C * th_gi) ** k_gi)); W["y"] = np.log(W.D)
print(f"{'season':15s} {'n':>4s} {'corr(rD,rC)':>11s} {'OLS b':>7s} {'IV b':>7s} {'IV b 90% (bootstrap)':>22s}")
rng = np.random.default_rng(0)
for s, g in W.groupby("season"):
    ols = np.cov(g.y, g.x)[0, 1] / g.x.var()
    iv = np.cov(g.y, g.z)[0, 1] / np.cov(g.x, g.z)[0, 1]
    bs = []
    for _ in range(2000):
        h = g.sample(len(g), replace=True, random_state=rng.integers(1 << 31))
        bs.append(np.cov(h.y, h.z)[0, 1] / np.cov(h.x, h.z)[0, 1])
    lo, hi = np.percentile(bs, [5, 95])
    print(f"{s:15s} {len(g):4d} {np.corrcoef(g.r, g.r_C)[0,1]:11.2f} {ols:7.2f} {iv:7.2f}   [{lo:.2f}, {hi:.2f}]")
