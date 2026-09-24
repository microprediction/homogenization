"""The turnover rule against the county waves.

Winter 2020-21 waves (06, IFR as set), binned by the case-based growth rate (independent of death reporting noise).
Per bin: median R from case growth and from death growth, the textbook share 1 - 1/R, the rule 1 - R^(-1/lam),
and the observed share of susceptibles infected during the wave, 1 - (1 - a_turn)/(1 - a_start)."""
import numpy as np, pandas as pd, os, importlib.util, sys
HERE = os.path.dirname(os.path.abspath(__file__))
IFR = os.environ.get("IFR", "0.007")
os.environ["IFR"] = IFR
spec = importlib.util.spec_from_file_location("e", os.path.join(HERE, "08_errors_in_variables.py"))
# reuse the case-growth computation from 08 without printing its table
src = open(os.path.join(HERE, "08_errors_in_variables.py")).read().split('print(f"{\'season\'')[0]
src = src.replace('"06_waves_ifr0.007.csv"', f'"06_waves_ifr{IFR}.csv"')
ns = {"__file__": os.path.join(HERE, "08_errors_in_variables.py")}; exec(src, ns); W = ns["W"]; k_gi, th_gi = ns["k_gi"], ns["th_gi"]
g = W[W.season == "winter 2020-21"].copy()
g["R_C"] = (1 + g.r_C * th_gi) ** k_gi
g["share"] = 1 - (1 - g.attack) / (1 - g.attack_start)
lam = float(np.exp(np.mean(np.log(np.log(g.R))) - np.mean(np.log(g.D))))        # constant-lam fit, death growth
lamC = float(np.exp(np.mean(np.log(np.log(g.R_C))) - np.mean(np.log(g.D))))      # constant-lam fit, case growth
g["bin"] = pd.qcut(g.R_C, 4)
t = g.groupby("bin", observed=True).agg(n=("R", "size"), R_cases=("R_C", "median"), R_deaths=("R", "median"),
                                        observed=("share", "median"))
t["textbook"] = 1 - 1 / t.R_cases
for L in (2, 3.5, 5):
    t[f"rule lam={L}"] = 1 - t.R_cases ** (-1 / L)
print(f"IFR {IFR}, winter 2020-21, {len(g)} waves; constant-lam fit: lam = {lam:.2f} (death growth), {lamC:.2f} (case growth)")
print(t.to_string(float_format=lambda v: f"{v:.3f}"))
