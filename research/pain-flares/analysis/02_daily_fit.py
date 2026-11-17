"""Fit the first-order likelihood of papers/likelihood/density.py (unchanged) to the synthetic daily series.

Free parameters: kappa, the average level, the average noise s and the Green-Kubo terms (K_uu, K_uw, K_ww) of a
two-state chain, which make that matrix rank one (K_uw^2 = K_uu K_ww). They are passed to density_first_order as an
equivalent chain (shares 1/2, rate 1/2 each way), for which K_ff = (f_1 - f_0)^2 / 4. Reported: what the daily data
identify, kappa, the level, s_eff = s + 2 kappa^2 K_uu, K_uw and K_ww, against the truth, with standard errors from
the observed information. Ends with PASS when the two fast series recover every one within 3 standard errors."""
import csv, importlib.util, json, math, os
import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("density", os.path.join(HERE, "..", "..", "..", "papers", "likelihood", "density.py"))
dn = importlib.util.module_from_spec(spec); spec.loader.exec_module(dn)
KAPPA, THETA, S, P_BAD = 0.7, (2.0, 5.0), (1.0, 4.0), 1 / 3
SERIES = [("fast_1h", 24.0), ("fast_2h", 12.0), ("slow_10d", 0.1)]   # lam = a + b per day
Q_EQ = np.array([[-0.5, 0.5], [0.5, -0.5]])


def loglik(z, x):
    kappa, th, s, gu, gw = math.exp(z[0]), z[1], math.exp(z[2]), z[3], z[4]
    if s <= abs(gw):
        return -1e12
    p, _ = dn.density_first_order(x[1:], Q_EQ, kappa, [th - gu, th + gu], [s - gw, s + gw], x[:-1], 1.0)
    return float(np.sum(np.log(p))) if np.all(p > 0) else -1e12


def report(z):  # kappa, level, s_eff, K_uw, K_ww
    kappa, s, gu, gw = math.exp(z[0]), math.exp(z[2]), z[3], z[4]
    return np.array([kappa, z[1], s + 2 * kappa ** 2 * gu * gu, gu * gw, gw * gw])


def truth(lam):
    a, b = lam * P_BAD, lam * (1 - P_BAD)
    K = lambda f, h: float(dn.gk(np.array([[-a, a], [b, -b]]), np.array(f), np.array(h)))
    sbar = (1 - P_BAD) * S[0] + P_BAD * S[1]
    return np.array([KAPPA, (1 - P_BAD) * THETA[0] + P_BAD * THETA[1], sbar + 2 * KAPPA ** 2 * K(THETA, THETA), K(THETA, S), K(S, S)])


def lm_plus(x):  # Cotton's one-sided score statistic on Gaussian AR(1) residuals; 95% point 5.1384
    A = np.vstack([np.ones(len(x) - 1), x[:-1]]).T
    r = x[1:] - A @ np.linalg.lstsq(A, x[1:], rcond=None)[0]; z = (r - r.mean()) / r.std()
    return len(z) * (np.mean(z ** 3) ** 2 / 6 + max(np.mean(z ** 4) - 3, 0) ** 2 / 24)


def main():
    out, ok = {}, True
    for name, lam in SERIES:
        x = np.array([float(r["level"]) for r in csv.DictReader(open(os.path.join(HERE, "..", "data", name + ".csv")))])
        f = lambda z: -loglik(z, x)
        z0 = np.array([math.log(0.5), x.mean(), math.log(x[1:].var() - 0.0), 0.3, 0.3])
        z = min((minimize(f, z0 + d, method="Nelder-Mead", options=dict(xatol=1e-7, fatol=1e-7, maxfev=20000))
                 for d in (0, np.array([0, 0, 0, -0.6, 0.6]))), key=lambda o: o.fun).x
        h = 1e-4; I = np.zeros((5, 5)); E = np.eye(5) * h   # observed information, central differences
        for i in range(5):
            for j in range(5):
                I[i, j] = (f(z + E[i] + E[j]) - f(z + E[i] - E[j]) - f(z - E[i] + E[j]) + f(z - E[i] - E[j])) / (4 * h * h)
        G = np.array([(report(z + E[i]) - report(z - E[i])) / (2 * h) for i in range(5)])   # d report / d z
        se = np.sqrt(np.clip(np.diag(G.T @ np.linalg.pinv(I) @ G), 0, None))
        est, tru = report(z), truth(lam)
        zs = (est - tru) / se
        out[name] = dict(memory_hours=24 / lam, estimate=est.tolist(), se=se.tolist(), truth=tru.tolist(), lm_plus=lm_plus(x))
        print(f"{name}: memory {24 / lam:g} h, LM+ {out[name]['lm_plus']:.1f}")
        for lab, e, s, t, q in zip(("kappa", "level", "s_eff", "K_uw", "K_ww"), est, se, tru, zs):
            print(f"   {lab:6s} estimate {e:8.4f} +- {s:.4f}   truth {t:8.4f}   ({q:+.1f} se)")
        if name.startswith("fast"):
            ok &= bool(np.all(np.abs(zs) < 3))
    json.dump(out, open(os.path.join(HERE, "out", "02_daily_fit.json"), "w"))
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
