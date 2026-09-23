"""Two-level switching fits and the renewal prediction of K.

Each detrended series is split into high and low states by a threshold at the median with hysteresis
(a switch is declared only when the series crosses median +/- h, h = 0.25 sd). From the state sequence:
- Delta: difference of the state means;
- m1, m2, s1, s2: mean and sd of the completed dwell times in each state (first and last dwells dropped);
- K_markov = Delta^2 pi1 pi2 / gamma with gamma = 1/m1 + 1/m2 (exponential dwells with the same means);
- K_renewal = Delta^2 (m2^2 s1^2 + m1^2 s2^2) / (2 (m1 + m2)^3) (the same means and the observed dwell spread).
K_long is the batch-means estimate at the longest block with at least six blocks, with a chi-square 68% interval,
computed on the square-wave reconstruction and on the data itself."""
import json, os
import numpy as np
from scipy import stats
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ex", os.path.join(HERE, "01_explore.py"))
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)


def states(x, h):
    med = np.median(x)
    s = np.empty(len(x), int); cur = int(x[0] > med)
    for i, v in enumerate(x):
        if cur == 0 and v > med + h: cur = 1
        elif cur == 1 and v < med - h: cur = 0
        s[i] = cur
    return s


def dwells(t, s):
    idx = np.flatnonzero(np.diff(s)) + 1
    edges = np.r_[0, idx, len(s)]
    d = {0: [], 1: []}
    for a, b in zip(edges[1:-2], edges[2:-1]):
        d[s[a]].append(t[b] - t[a])
    return np.array(d[1]), np.array(d[0])


def K_long(t, x, nmin=6):
    dt = np.median(np.diff(t)); span = t[-1] - t[0]
    nb = nmin; tau = span / nb; k = int(round(tau / dt))
    means = np.array([x[i * k:(i + 1) * k].mean() for i in range(nb)])
    v = means.var(ddof=1); K = tau * v / 2
    lo, hi = K * (nb - 1) / stats.chi2.ppf([0.84, 0.16], nb - 1)
    return K, lo, hi


def main():
    rows = json.load(open(os.path.join(HERE, "out", "01_summary.json")))
    out = []
    for r in rows:
        t, nd, x, _ = ex.load(r["psr"])
        s = states(x, 0.25 * x.std())
        up, dn = dwells(t, s)
        if len(up) < 2 or len(dn) < 2:
            out.append(dict(psr=r["psr"], n_switches=int(np.sum(np.diff(s) != 0)))); continue
        D = x[s == 1].mean() - x[s == 0].mean()
        m1, m2, s1, s2 = up.mean(), dn.mean(), up.std(ddof=1), dn.std(ddof=1)
        g = 1 / m1 + 1 / m2; p1 = m1 / (m1 + m2)
        Km = D ** 2 * p1 * (1 - p1) / g
        Kr = D ** 2 * (m2 ** 2 * s1 ** 2 + m1 ** 2 * s2 ** 2) / (2 * (m1 + m2) ** 3)
        sq = np.where(s == 1, x[s == 1].mean(), x[s == 0].mean())
        Kd, lo, hi = K_long(t, x); Ks, _, _ = K_long(t, sq)
        frac = 1 - np.var(x - sq) / np.var(x)
        out.append(dict(psr=r["psr"], n_switches=int(np.sum(np.diff(s) != 0)), Delta=D, m_up=m1, m_dn=m2,
                        cv_up=s1 / m1, cv_dn=s2 / m2, var_explained=frac, K_markov=Km, K_renewal=Kr,
                        K_square=Ks, K_data=Kd, K_data_lo=lo, K_data_hi=hi))
    json.dump(out, open(os.path.join(HERE, "out", "03_dwells.json"), "w"), indent=1)
    hdr = f"{'pulsar':11s} {'sw':>3s} {'Delta':>7s} {'m_up':>5s} {'m_dn':>5s} {'cv_up':>5s} {'cv_dn':>5s} {'R2':>4s} " \
          f"{'K markov':>9s} {'K renew':>9s} {'K square':>9s} {'K data':>9s} {'68% interval':>20s}"
    print(hdr)
    for o in out:
        if "Delta" not in o:
            print(f"{o['psr']:11s} {o['n_switches']:3d}  too few complete dwells"); continue
        print(f"{o['psr']:11s} {o['n_switches']:3d} {o['Delta']:7.3g} {o['m_up']:5.0f} {o['m_dn']:5.0f} "
              f"{o['cv_up']:5.2f} {o['cv_dn']:5.2f} {o['var_explained']:4.2f} {o['K_markov']:9.3g} {o['K_renewal']:9.3g} "
              f"{o['K_square']:9.3g} {o['K_data']:9.3g}   [{o['K_data_lo']:.3g}, {o['K_data_hi']:.3g}]")


if __name__ == "__main__":
    main()
