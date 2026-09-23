"""Fit stage models to the autocovariance and test the observed K against them.

Model A (ring): two levels 0 and Delta; dwells Erlang(k) with means m1, m2 (k stages per regime, constant
hazard per stage). Model B: model A plus an independent slow Ornstein-Uhlenbeck component (variance v, time
scale theta). Both add white noise (a spike at lag 0, not fitted). Parameters are fitted by least squares to the
sample autocovariance at lags 1.5 days to span/4, k by grid search.
K of the ring comes from the group inverse of the stage chain; the OU adds v * theta.
The observed six-block K is then placed among 400 simulated six-block K from the fitted model, same grid,
detrend and estimator as the data."""
import json, os
import numpy as np
from scipy.linalg import expm
from scipy.optimize import least_squares
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
def mod(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ex, dw = mod("01_explore"), mod("03_dwells")
rng = np.random.default_rng(7)
N = 400


def ring(m1, m2, k):
    n = 2 * k; Q = np.zeros((n, n))
    for i in range(n):
        r = k / m1 if i < k else k / m2
        Q[i, i] = -r; Q[i, (i + 1) % n] = r
    pi = np.r_[np.full(k, m1 / k), np.full(k, m2 / k)] / (m1 + m2)
    f = np.r_[np.ones(k), np.zeros(k)]
    return Q, pi, f - pi @ f


def ring_acov(m1, m2, k, lags):
    Q, pi, g = ring(m1, m2, k)
    w, V = np.linalg.eig(Q); a = np.linalg.solve(V, g); b = (pi * g) @ V
    return np.real(np.exp(np.outer(lags, w)) @ (b * a))


def ring_K(m1, m2, k):
    Q, pi, g = ring(m1, m2, k)
    P = np.outer(np.ones(len(pi)), pi)
    return pi @ (g * ((np.linalg.inv(P - Q) - P) @ g))


def sample_acov(x, nlag):
    y = x - x.mean(); n = len(y)
    f = np.fft.rfft(y, 2 * n)
    return (np.fft.irfft(f * np.conj(f))[:nlag] / np.arange(n, n - nlag, -1))


def fit(t, x, ou, start=None, hints=()):
    dt = np.median(np.diff(t)); nlag = int((t[-1] - t[0]) / 4 / dt)
    lags = dt * np.arange(1, nlag); c = sample_acov(x, nlag)[1:]
    wts = 1 / np.sqrt(1 + lags / 100)  # early lags are better measured
    best = None
    ks = (start["k"],) if start else (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
    for k in ks:
        def res(p):
            m1, m2, D2 = np.exp(p[:3]); r = D2 * ring_acov(m1, m2, k, lags)
            if ou: r = r + np.exp(p[3]) * np.exp(-lags / np.exp(p[4]))
            return (r - c) * wts
        lo = [np.log(5 * dt)] * 2 + [-50] + ([-50, np.log(200.0)] if ou else [])
        hi = [np.log(2e4)] * 2 + [50] + ([50, np.log(5e4)] if ou else [])
        if start:
            base = [np.log(start["m1"]), np.log(start["m2"]), 2 * np.log(start["Delta"])]
            starts = [base + [np.log(f * x.var()), np.log(th)] for f in (0.05, 0.3) for th in (500.0, 2000.0, 8000.0)]
        else:
            starts = [[np.log(a), np.log(b), np.log(4 * x.var())] for a, b in ((100.0, 150.0), (200.0, 200.0), (400.0, 300.0), (1500.0, 1000.0)) + tuple(hints)]
        for p0 in starts:
            p0 = list(np.clip(p0, np.array(lo) + 1e-9, np.array(hi) - 1e-9))
            try:
                s = least_squares(res, p0, bounds=(lo, hi))
            except Exception:
                continue
            if best is None or s.cost < best[1].cost:
                best = (k, s)
    k, s = best
    m1, m2, D2 = np.exp(s.x[:3])
    p = dict(k=k, m1=m1, m2=m2, Delta=np.sqrt(D2), v=0.0, theta=1.0)
    if ou:
        p["v"], p["theta"] = np.exp(s.x[3]), np.exp(s.x[4])
    p["K_model"] = D2 * ring_K(m1, m2, k) + p["v"] * p["theta"]
    p["rel_rmse"] = float(np.sqrt(np.mean((s.fun / wts) ** 2)) / c[0])
    return p


def simulate(t, p, noise):
    m, k = (p["m1"], p["m2"]), p["k"]
    edges, st = [t[0] - 10 * sum(m)], []
    s = int(rng.random() < m[1] / sum(m))
    while edges[-1] < t[-1]:
        edges.append(edges[-1] + rng.gamma(k, m[s] / k)); st.append(s); s = 1 - s
    y = p["Delta"] * (1 - np.array(st)[np.searchsorted(edges, t, side="right") - 1])
    if p["v"] > 0:
        a = np.exp(-np.diff(t) / p["theta"]); z = np.empty(len(t)); z[0] = rng.standard_normal()
        e = rng.standard_normal(len(t))
        for i in range(1, len(t)):
            z[i] = a[i - 1] * z[i - 1] + np.sqrt(1 - a[i - 1] ** 2) * e[i]
        y = y + np.sqrt(p["v"]) * z
    return y + noise * rng.standard_normal(len(t))


def percentile(t, x, p, K_data):
    nlag = 2
    noise = np.sqrt(max(sample_acov(x, nlag)[0] - sample_acov(x, nlag)[1], 0))
    ks = []
    for _ in range(N):
        y = simulate(t, p, noise)
        c = np.polyfit(t - t.mean(), y, 1); y = y - np.polyval(c, t - t.mean())
        ks.append(dw.K_long(t, y)[0])
    return 100 * float(np.mean(np.array(ks) < K_data))


def main():
    rows = json.load(open(os.path.join(HERE, "out", "01_summary.json")))
    dwf = {d["psr"]: d for d in json.load(open(os.path.join(HERE, "out", "03_dwells.json")))}
    only = os.environ.get("PSR")
    out = []
    print(f"{'pulsar':11s} {'model':5s} {'k':>3s} {'m1':>6s} {'m2':>6s} {'cv':>5s} {'OU share':>8s} {'theta':>6s} "
          f"{'fit err':>7s} {'K model':>9s} {'K data':>9s} {'pct':>4s}")
    for r in rows:
        if only and r["psr"] not in only.split(","):
            continue
        t, _, x, _ = ex.load(r["psr"])
        K_data = dw.K_long(t, x)[0]
        prev = None
        for name, ou in (("ring", False), ("ring+OU", True)):
            d = dwf[r["psr"]]
            hints = [(r["period_days"] / 2, r["period_days"] / 2)] + ([(d["m_up"], d["m_dn"]), (d["m_dn"], d["m_up"])] if "m_up" in d else [])
            p = fit(t, x, ou, start=prev, hints=hints); prev = p
            p["pct"] = percentile(t, x, p, K_data)
            p.update(psr=r["psr"], model=name, K_data=K_data)
            out.append(p)
            share = p["v"] / (p["v"] + p["Delta"] ** 2 * p["m1"] * p["m2"] / (p["m1"] + p["m2"]) ** 2)
            flag = "*" if p["pct"] < 5 or p["pct"] > 95 else " "
            print(f"{r['psr']:11s} {name:7s} {p['k']:3d} {p['m1']:6.0f} {p['m2']:6.0f} {1/np.sqrt(p['k']):5.2f} "
                  f"{share:8.2f} {p['theta']:6.0f} {p['rel_rmse']:7.3f} {p['K_model']:9.3g} {K_data:9.3g} {p['pct']:4.0f}{flag}",
                  flush=True)
    json.dump(out, open(os.path.join(HERE, "out", "06_stage_fit.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
