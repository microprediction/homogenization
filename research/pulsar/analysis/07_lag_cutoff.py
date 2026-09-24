"""Is the large stage count an artefact of the Gaussian-process smoothing?

Smoothing changes the autocovariance only at lags shorter than the kernel length. Refit the ring model
(06_stage_fit.ring_acov) using only lags >= L, for L = 0, 60, 150, 300 days, with the lag-0 region free, and report
the selected k (dwell cv = 1/sqrt k) and the weighted cost for k = 1 against the best k."""
import json, os
import numpy as np
from scipy.optimize import least_squares
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
def mod(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ex, sf = mod("01_explore"), mod("06_stage_fit")
KS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)


def fit_cut(t, x, L, hints):
    dt = np.median(np.diff(t)); nlag = int((t[-1] - t[0]) / 4 / dt)
    lags = dt * np.arange(nlag); c = sf.sample_acov(x, nlag)
    keep = lags >= max(L, dt); lags, c = lags[keep], c[keep]
    wts = 1 / np.sqrt(1 + lags / 100)
    costs = {}
    for k in KS:
        def res(p):
            m1, m2, D2 = np.exp(p); return (D2 * sf.ring_acov(m1, m2, k, lags) - c) * wts
        best = np.inf
        for a, b in ((100.0, 150.0), (200.0, 200.0), (400.0, 300.0), (1500.0, 1000.0)) + tuple(hints):
            try:
                s = least_squares(res, [np.log(a), np.log(b), np.log(4 * x.var())],
                                  bounds=([np.log(5 * dt)] * 2 + [-50], [np.log(2e4)] * 2 + [50]))
                best = min(best, s.cost)
            except Exception:
                pass
        costs[k] = best
    kb = min(costs, key=costs.get)
    return kb, costs[1] / costs[kb]


def main():
    rows = json.load(open(os.path.join(HERE, "out", "01_summary.json")))
    dwf = {d["psr"]: d for d in json.load(open(os.path.join(HERE, "out", "03_dwells.json")))}
    cuts = (0, 60, 150, 300)
    print(f"{'pulsar':11s} " + " ".join(f"{'L=' + str(L):>14s}" for L in cuts) + "   (best k, cost(k=1)/cost(best))")
    out = []
    for r in rows:
        t, _, x, _ = ex.load(r["psr"]); d = dwf[r["psr"]]
        hints = [(r["period_days"] / 2,) * 2] + ([(d["m_up"], d["m_dn"]), (d["m_dn"], d["m_up"])] if "m_up" in d else [])
        res = [fit_cut(t, x, L, hints) for L in cuts]
        out.append(dict(psr=r["psr"], cuts=cuts, best_k=[k for k, _ in res], markov_cost_ratio=[q for _, q in res]))
        print(f"{r['psr']:11s} " + " ".join(f"{k:4d} {q:9.1f}" for k, q in res), flush=True)
    json.dump(out, open(os.path.join(HERE, "out", "07_lag_cutoff.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
