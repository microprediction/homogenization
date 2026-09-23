"""Place the observed six-block K among simulated six-block K under two fitted models.

Both models use the fitted Delta, mean dwells and residual noise level, on the pulsar's own grid, with the same
detrend and estimator. Markov: exponential dwells (cv = 1). Renewal: gamma dwells with the fitted cv.
Reported: the percentile of K_data in each simulated distribution (400 paths). A model is strained when the
percentile is below 5 or above 95."""
import json, os
import numpy as np
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
def mod(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ex, dw, chk = mod("01_explore"), mod("03_dwells"), mod("04_estimator_check")
N = 400


def sim_K(t, r, cv, noise):
    ks = []
    for _ in range(N):
        y = chk.path(t, r["Delta"], (r["m_up"], r["m_dn"]), cv) + noise * chk.rng.standard_normal(len(t))
        c = np.polyfit(t - t.mean(), y, 1); y = y - np.polyval(c, t - t.mean())
        ks.append(dw.K_long(t, y)[0])
    return np.array(ks)


def main():
    fits = [r for r in json.load(open(os.path.join(HERE, "out", "03_dwells.json"))) if "Delta" in r]
    out = []
    print(f"{'pulsar':11s} {'cv up/dn':>10s} {'K data':>9s} {'Markov pct':>10s} {'renewal pct':>11s}")
    for r in fits:
        t, _, x, _ = ex.load(r["psr"])
        noise = np.sqrt(1 - r["var_explained"]) * x.std()
        km = sim_K(t, r, (1.0, 1.0), noise)
        kr = sim_K(t, r, (r["cv_up"], r["cv_dn"]), noise)
        pm, pr = 100 * np.mean(km < r["K_data"]), 100 * np.mean(kr < r["K_data"])
        out.append(dict(psr=r["psr"], K_data=r["K_data"], markov_pct=pm, renewal_pct=pr,
                        markov_median=float(np.median(km)), renewal_median=float(np.median(kr))))
        flag = lambda p: "*" if p < 5 or p > 95 else " "
        print(f"{r['psr']:11s} {r['cv_up']:4.2f}/{r['cv_dn']:4.2f} {r['K_data']:9.3g} {pm:9.0f}{flag(pm)} {pr:10.0f}{flag(pr)}")
    json.dump(out, open(os.path.join(HERE, "out", "05_model_test.json"), "w"), indent=1)
    for k in ("markov_pct", "renewal_pct"):
        print(k, "strained:", sum(o[k] < 5 or o[k] > 95 for o in out), "of", len(out))


if __name__ == "__main__":
    main()
