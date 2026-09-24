"""US counties, 2020: is county transmission heterogeneous, and does the heterogeneity switch or persist?

Weekly new cases per county (differences of NYT cumulative counts, 7-day blocks ending Sundays). Weekly log growth
g = log(new_w / new_{w-1}) for counties with at least MIN cases in both weeks.
- National growth G_w = log(sum new_w / sum new_{w-1}) over the same counties: the aggregate, which weights counties
  by their cases.
- Typical growth: the median of g over counties.
- Deviations d = g - (weekly median). Persistence: correlation across counties of d_w and d_{w+k}, k = 1..8.
  A fixed county effect keeps this correlation up at every lag; switching makes it decay.
Poisson noise in weekly counts biases lag-1 correlation down; MIN keeps it small (sd of log count about 1/sqrt(MIN))."""
import numpy as np, pandas as pd, json, os

HERE = os.path.dirname(os.path.abspath(__file__))
MIN = 100
df = pd.read_csv(os.path.join(HERE, "..", "data", "us-counties-2020.csv"), dtype={"fips": str}, parse_dates=["date"])
df = df[df.fips.notna()]
cum = df.pivot_table(index="date", columns="fips", values="cases").sort_index().ffill().fillna(0)
wk = cum.resample("W-SUN").last().diff().clip(lower=0).iloc[1:]
weeks = wk.index
rows, D = [], {}
for i in range(1, len(weeks)):
    a, b = wk.iloc[i - 1], wk.iloc[i]
    ok = (a >= MIN) & (b >= MIN)
    if ok.sum() < 30:
        continue
    g = np.log(b[ok] / a[ok])
    G = np.log(b[ok].sum() / a[ok].sum())
    med = g.median()
    D[weeks[i]] = g - med
    rows.append(dict(week=str(weeks[i].date()), n=int(ok.sum()), national=G, median=med, gap=G - med,
                     sd=g.std(), iqr=g.quantile(.75) - g.quantile(.25)))
T = pd.DataFrame(rows)
print(T.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
dev = pd.DataFrame(D).T  # weeks x counties
cors = {}
for k in range(1, 9):
    cs = []
    for i in range(len(dev) - k):
        x, y = dev.iloc[i], dev.iloc[i + k]
        m = x.notna() & y.notna()
        if m.sum() >= 30:
            cs.append(np.corrcoef(x[m], y[m])[0, 1])
    cors[k] = (float(np.median(cs)), len(cs))
print("\nlag (weeks): median cross-county correlation of growth deviations")
for k, (c, n) in cors.items():
    print(f"  {k}: {c:+.3f}  ({n} week pairs)")
json.dump(dict(weeks=rows, lag_corr=cors, MIN=MIN), open(os.path.join(HERE, "out", "02_county_growth.json"), "w"), indent=1)
