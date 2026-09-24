"""Which model fits the county waves? Fit the log attack gained in each wave against its early growth.

Delta = -ln((1 - a_turn)/(1 - a_start)), the log susceptible depletion during the wave; ln R from early growth.
Models for Delta (fitted by least squares on log Delta, per season, IFR 0.7%):
  homogeneous      Delta = ln R                     (0 parameters)
  persistent       Delta = ln R / lam               (lam)
  switching        Delta = ln R / (lam0 + K r)      (lam0 >= 1, K >= 0)
  fixed increment  Delta = c                        (c)
  free power       Delta = c (ln R)^b               (c, b): b = 1 persistent, b = 0 fixed increment
Reported: residual sd of log Delta, BIC, and parameters."""
import numpy as np, pandas as pd, os
from scipy.optimize import least_squares
HERE = os.path.dirname(os.path.abspath(__file__))
IFR = os.environ.get("IFR", "0.007")
W = pd.read_csv(os.path.join(HERE, "out", f"06_waves_ifr{IFR}.csv"))
W = W[(W.attack > W.attack_start) & (W.attack < 1)].copy()
W["D"] = -np.log((1 - W.attack) / (1 - W.attack_start)); W["lnR"] = np.log(W.R)
y_all = np.log(W.D.values)

def fit(name, f, p0, lb, ub, g):
    y = np.log(g.D.values)
    if p0 is None:
        res = y - np.log(f(None, g)); k = 0
    else:
        s = least_squares(lambda p: y - np.log(f(p, g)), p0, bounds=(lb, ub)); res = s.fun; k = len(p0); p0 = s.x
    n = len(y); sd = np.sqrt(np.mean(res ** 2))
    return name, sd, n * np.log(np.mean(res ** 2)) + k * np.log(n), p0

M = [("homogeneous", lambda p, g: g.lnR.values, None, None, None),
     ("persistent", lambda p, g: g.lnR.values / p[0], [2.0], [0.01], [100]),
     ("switching", lambda p, g: g.lnR.values / (p[0] + p[1] * g.r.values), [1.5, 20.0], [1.0, 0.0], [100, 1e4]),
     ("fixed increment", lambda p, g: np.full(len(g), p[0]), [0.05], [1e-4], [5]),
     ("free power", lambda p, g: p[0] * g.lnR.values ** p[1], [0.1, 0.5], [1e-4, -3], [10, 3])]
for season, g in W.groupby("season"):
    print(f"\n{season} (n = {len(g)}), IFR {IFR}")
    for name, f, p0, lb, ub in M:
        nm, sd, bic, p = fit(name, f, p0, lb, ub, g)
        ps = "" if p is None else " ".join(f"{v:.3g}" for v in p)
        print(f"  {nm:16s} resid sd {sd:.3f}  BIC {bic:8.1f}  params {ps}")
