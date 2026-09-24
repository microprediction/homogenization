"""Do county growth deviations compound? Cross-county variance of log(new_{w+k} / new_w) against k.

A random walk in log incidence (fluctuating growth rates that integrate) gives variance 2 K k plus a constant
from level noise; pure level noise (reporting timing, batches) gives a constant. The aggregate-minus-median gap
over k weeks is also reported: compounding fluctuations make it grow like K k."""
import numpy as np, pandas as pd, json, os
HERE = os.path.dirname(os.path.abspath(__file__))
MIN = 100
df = pd.read_csv(os.path.join(HERE, "..", "data", "us-counties-2020.csv"), dtype={"fips": str}, parse_dates=["date"])
df = df[df.fips.notna()]
cum = df.pivot_table(index="date", columns="fips", values="cases").sort_index().ffill().fillna(0)
wk = cum.resample("W-SUN").last().diff().clip(lower=0).iloc[1:]
wk = wk.loc["2020-04-05":"2020-12-27"]
out = {}
print(" k   var(log ratio)   median over start weeks   aggregate - median gap")
for k in range(1, 13):
    vs, gs = [], []
    for i in range(len(wk) - k):
        a, b = wk.iloc[i], wk.iloc[i + k]
        ok = (a >= MIN) & (b >= MIN)
        if ok.sum() < 30: continue
        g = np.log(b[ok] / a[ok])
        vs.append(g.var()); gs.append(np.log(b[ok].sum() / a[ok].sum()) - g.median())
    out[k] = dict(var=float(np.median(vs)), gap=float(np.mean(gs)), n=len(vs))
    print(f"{k:2d}   {np.median(vs):.3f}   {len(vs):3d}   {np.mean(gs):+.3f}")
ks = np.array(list(out)); v = np.array([out[k]["var"] for k in ks])
slope, icpt = np.polyfit(ks[ks >= 2], v[ks >= 2], 1)
print(f"\nvariance = {icpt:.3f} + {slope:.4f} k  (k >= 2)  ->  K = {slope/2:.4f} per week, level-noise variance {icpt/2:.3f}")
json.dump(dict(by_k=out, slope=slope, intercept=icpt), open(os.path.join(HERE, "out", "03_compounding.json"), "w"), indent=1)
