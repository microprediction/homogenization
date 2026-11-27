"""Coin with no memory against a hidden two-state hazard, on published low back pain sojourn curves.

A sojourn ends at hazard h_y while a hidden regime switches (1 -> 2 at rate a, 2 -> 1 at b, lam = a + b), so
S(t) = nu exp((Q - H) t) 1. Fast switching averages this to one exponential (the coin); the first-order term is
S = (1 + A) exp(-r t) after a layer of length 1/lam. J = (m - d)' C^-1 (m - d), C the sampling covariance of the
published numbers. Profiles J over the regime memory 1/lam and bounds it at 95% by bisection."""
import csv, json, math, os
import numpy as np
from scipy.optimize import brentq, minimize, minimize_scalar

HERE = os.path.dirname(os.path.abspath(__file__))
P = {(r["curve"], r["time_days"]): r for r in csv.DictReader(open(os.path.join(HERE, "..", "published_flare_numbers.csv")))}
v = lambda c, t: tuple(float(P[(c, t)][f]) for f in ("value", "standard_error", "allowance"))
CURVES = {  # points (days, share ended, se); median (days, se); entry lags; minimum length; conditional (t0, t, share, se)
    "episode": dict(c="lbp_episode_length", pts=[7, 21, 42, 84], med=None, lags=None, min=1.0, cond=None),
    "gap": dict(c="lbp_time_to_next_episode", pts=[91, 183, 365], med="", lags=[7, 10.5, 20.75, 29], min=None, cond=None),
    "care": dict(c="lbp_care_time_to_recovery", pts=[42, 84, 365], med="", lags=None, min=None, cond="456"),
}


def targets(k):
    c = CURVES[k]; d, se, al, F = [], [], [], []
    for t in c["pts"] + ([""] if c["med"] is not None else []):
        x, s, a = v(c["c"], str(t)); d.append(x); se.append(s); al.append(a); F.append(x if t != "" else 0.5)
    n = len(d); C = np.diag(np.square(se) + np.square(al))  # allowance (whole-day rounding) on the diagonal only
    for i in range(n):
        for j in range(i):
            lo, hi = min(F[i], F[j]), max(F[i], F[j])
            r = math.sqrt(lo * (1 - hi) / (hi * (1 - lo))) * (-1 if (c["med"] is not None and n - 1 in (i, j)) else 1)
            C[i, j] = C[j, i] = r * se[i] * se[j]   # empirical-process correlation; a median moves against shares
    if c["cond"]:  # Costa's people are Henschke's still in pain at 91 days: correlate with the 365-day share
        x, s, _ = v(c["c"], c["cond"]); F0 = d[1]; g = (d[2] - F0) / (1 - F0); h = (x - g) / (1 - g)
        C = np.pad(C, (0, 1)); C[n, n] = s * s
        C[2, n] = C[n, 2] = (1 - F0) * (1 - h) * g * (1 - g) / (x * (1 - x) / s ** 2)
        d.append(x)
    return np.array(d), C


def S_hidden(t, h1, h2, a, b, nu1):
    tr, det = -(a + b + h1 + h2), a * h2 + b * h1 + h1 * h2
    r2 = (tr - math.sqrt((a - b + h1 - h2) ** 2 + 4 * a * b)) / 2; r1 = det / r2
    if abs(r1 - r2) < 1e-12 * abs(r2):
        return np.exp(r1 * t) * (1 - (nu1 * h1 + (1 - nu1) * h2 + r1) * t)
    c1 = (-(nu1 * h1 + (1 - nu1) * h2) - r2) / (r1 - r2)   # nu (M - r2 I) 1 / (r1 - r2), M 1 = -h
    return c1 * np.exp(r1 * t) + (1 - c1) * np.exp(r2 * t)


def predict(k, S):
    c = CURVES[k]
    if c["min"]:
        s1 = float(S(np.array([c["min"]]))[0]); S0 = S
        S = lambda t: S0(np.maximum(t, c["min"])) / s1
    if c["lags"]:  # time from entry, a lag after the previous sojourn ended
        L = np.array(c["lags"]); SL = S(L)
        Se = lambda t: np.mean(S(L[None, :] + np.atleast_1d(t)[:, None]) / SL[None, :], axis=1)
    else:
        Se = lambda t: S(np.atleast_1d(np.asarray(t, float)))
    m = list(1 - Se(np.array(c["pts"], float)))
    if c["med"] is not None:
        tg = np.concatenate([[0], np.geomspace(0.01, 1e5, 300)]); i = int(np.argmax(Se(tg) <= 0.5))
        m.append(brentq(lambda t: Se(t)[0] - 0.5, tg[i - 1], tg[i]) if i > 0 else 1e5)
    if c["cond"]:
        m.append(1 - S(np.array([456.0]))[0] / S(np.array([91.0]))[0])
    return np.array(m)


def J(k, S):
    d, C = targets(k)
    try:
        e = predict(k, S) - d
    except (ValueError, ZeroDivisionError, OverflowError):
        return 1e9
    return float(e @ np.linalg.solve(C, e)) if np.all(np.isfinite(e)) else 1e9


sig = lambda z: 0.5 * (1 + math.tanh(z / 2))


def fit_hidden(k, lam, starts):
    f = lambda z: J(k, lambda t: S_hidden(t, math.exp(z[0]), math.exp(z[1]), lam * (1 - sig(z[2])), lam * sig(z[2]), sig(z[3])))
    best = min((minimize(f, z0, method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8, maxfev=3000)) for z0 in starts),
               key=lambda o: o.fun)
    return best.fun, best.x


def main():
    out, base = {}, [np.array([math.log(h1), math.log(0.005), 0.0, z]) for h1 in (0.3, 0.05) for z in (-1.5, 1.5)]
    for k in CURVES:
        coin = minimize_scalar(lambda lr: J(k, lambda t: np.exp(-math.exp(lr) * t)), bounds=(-12, 3), method="bounded").fun
        homog = min(minimize(lambda z: J(k, lambda t: (1 + 0.1 * math.tanh(z[1])) * np.exp(-math.exp(z[0]) * t)),
                             np.array([lr, a0]), method="Nelder-Mead", options=dict(xatol=1e-8, fatol=1e-10, maxiter=4000)).fun
                    for lr in (-6, -4, -2) for a0 in (-2, 0, 2))
        prof, warm = [], None
        for lam in np.logspace(-4, 2, 25):
            Jl, z = fit_hidden(k, lam, base + ([warm] if warm is not None else []))
            warm = z; prof.append((float(lam), Jl, z))
        Jmin = min(p[1] for p in prof); cut = Jmin + 3.841458820694124
        i = next(j for j in range(int(np.argmin([p[1] for p in prof])), 24) if prof[j][1] <= cut < prof[j + 1][1])
        lo, hi, st = math.log(prof[i][0]), math.log(prof[i + 1][0]), [prof[i][2], prof[i + 1][2]] + base
        for _ in range(14):
            mid = (lo + hi) / 2; Jm, z = fit_hidden(k, math.exp(mid), st); st = [z] + st[:6]
            lo, hi = (mid, hi) if Jm <= cut else (lo, mid)
        out[k] = dict(n=len(targets(k)[0]), J_coin=coin, J_first_order_valid=homog, J_hidden=Jmin,
                      memory95_days=math.floor(10 * math.exp(-lo)) / 10, profile=[(p[0], p[1]) for p in prof])
        print(f"{k:8s} n {out[k]['n']}  coin J {coin:7.2f}  first order (|A| <= 0.1) J {homog:7.2f}  "
              f"hidden exact J {Jmin:6.2f}  regime memory >= {out[k]['memory95_days']} days (95%)")
    json.dump(out, open(os.path.join(HERE, "out", "01_published_curves.json"), "w"))


if __name__ == "__main__":
    main()
