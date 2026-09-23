"""Does the six-block K estimate recover the true K at this span?

For each pulsar, simulate an alternating renewal square wave with the fitted Delta and gamma-distributed dwells
(the fitted means and cv), on the pulsar's own time grid, plus white noise at the fitted residual level. The true K
is the renewal formula. Apply the same linear detrend and six-block estimator, and report the median ratio
K_est / K_true and the coverage of the 68% interval over 400 paths."""
import json, os
import numpy as np
from scipy import stats
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
def mod(name):
    s = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
ex, dw = mod("01_explore"), mod("03_dwells")
rng = np.random.default_rng(1)


def gamma_draw(m, cv):
    k = 1 / cv ** 2
    return rng.gamma(k, m / k)


def path(t, D, m, cv):
    t0 = t[0] - 5 * (m[0] + m[1])  # burn in so the start is stationary
    edges, st, s = [t0], [], int(rng.random() < m[1] / sum(m))
    while edges[-1] < t[-1]:
        edges.append(edges[-1] + gamma_draw(m[s], cv[s])); st.append(s); s = 1 - s
    idx = np.searchsorted(edges, t, side="right") - 1
    return D * np.array(st)[idx]


def main():
    fits = [r for r in json.load(open(os.path.join(HERE, "out", "03_dwells.json"))) if "Delta" in r]
    out = []
    print(f"{'pulsar':11s} {'median est/true':>15s} {'coverage':>9s}")
    for r in fits:
        t, _, x, _ = ex.load(r["psr"])
        m = (r["m_up"], r["m_dn"]); cv = (r["cv_up"], r["cv_dn"])
        s1, s2 = cv[0] * m[0], cv[1] * m[1]
        Ktrue = r["Delta"] ** 2 * (m[1] ** 2 * s1 ** 2 + m[0] ** 2 * s2 ** 2) / (2 * (m[0] + m[1]) ** 3)
        noise = np.sqrt(1 - r["var_explained"]) * x.std()
        ratios, cover = [], 0
        for _ in range(400):
            y = path(t, r["Delta"], m, cv)
            y = y + noise * rng.standard_normal(len(t))  # white noise on the 1.5-day grid adds little to K
            c = np.polyfit(t - t.mean(), y, 1); y = y - np.polyval(c, t - t.mean())
            K, lo, hi = dw.K_long(t, y)
            ratios.append(K / Ktrue); cover += lo <= Ktrue <= hi
        med = float(np.median(ratios))
        out.append(dict(psr=r["psr"], K_true=Ktrue, median_ratio=med, q10=float(np.quantile(ratios, .1)),
                        q90=float(np.quantile(ratios, .9)), coverage=cover / 400))
        print(f"{r['psr']:11s} {med:15.2f} {cover / 400:9.2f}   10-90%: [{out[-1]['q10']:.2f}, {out[-1]['q90']:.2f}]")
    json.dump(out, open(os.path.join(HERE, "out", "04_estimator_check.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
