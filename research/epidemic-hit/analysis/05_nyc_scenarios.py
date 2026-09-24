"""Calibrate the activity-switching SIR to an early growth rate and read off the first-wave turnover.

For each activity variance s2 (gamma-like skew, c = 2 s2) and relaxation rate kappa, beta is set so that the
early growth rate equals r0 (NYC, March 2020: deaths doubling in about 3 days, r0 = ln 2 / 3 per day).
No mitigation is applied, so the attack at the first peak is the depletion effect alone.
Reported: homogeneous SIR reading of the growth, R_hom = 1 + r0/gamma and its threshold 1 - 1/R_hom; the model's
R0 and long-run threshold 1 - 1/R0; lam in the growth phase; attack rate at the first peak of prevalence;
the trough after it and the size of a second peak, if any, within 2 years."""
import numpy as np, json, os, importlib.util
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__))
s = importlib.util.spec_from_file_location("m", os.path.join(HERE, "04_activity_switching.py"))
m = importlib.util.module_from_spec(s); s.loader.exec_module(m)

gamma, r0 = 0.2, np.log(2) / 3
rows = []
print(f"{'s2':>4} {'kappa':>6} {'R_hom':>6} {'HIT_hom':>7} {'R0':>6} {'HIT_long':>8} {'lam':>5} {'attack@peak':>11} {'final/2y':>8} {'2nd peak':>8}")
for s2 in (0.5, 1.0, 2.0, 4.0):
    for kappa in (0.01, 0.03, 0.1, 0.3, 1.0):
        c = 2 * s2
        beta = brentq(lambda b: m.growth(b, gamma, kappa, s2) - r0, 1e-3, 5)
        R0 = m.R0(beta, gamma, kappa, s2)
        sol, a, p, G = m.simulate(beta, gamma, kappa, s2, c, T=730)
        ts = np.linspace(0, 730, 73001); y = sol.sol(ts)
        I = y[2:].sum(0); att = 1 - y[:2].sum(0) - I
        from scipy.signal import find_peaks
        pk, _ = find_peaks(I, prominence=1e-6)
        k1 = pk[0] if len(pk) else int(np.argmax(I))
        second = float(I[pk[1]] / I[k1]) if len(pk) > 1 else 0.0
        Rh = 1 + r0 / gamma
        row = dict(s2=s2, kappa=kappa, R_hom=Rh, HIT_hom=1 - 1 / Rh, R0=R0, HIT_long=1 - 1 / R0,
                   lam=m.lam(r0, gamma, kappa, s2, c), attack_at_peak=float(att[k1] + I[k1]), final=float(1 - y[:2, -1].sum()),
                   second_peak_rel=second)
        rows.append(row)
        print(f"{s2:4.1f} {kappa:6.2f} {Rh:6.2f} {row['HIT_hom']:7.2f} {R0:6.2f} {row['HIT_long']:8.2f} {row['lam']:5.2f} "
              f"{row['attack_at_peak']:11.2f} {row['final']:8.2f} {second:8.2f}", flush=True)
json.dump(rows, open(os.path.join(HERE, "out", "05_nyc_scenarios.json"), "w"), indent=1)
