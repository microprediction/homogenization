"""Certificate for the fast-switching expansion of survival under a regime-switching OU hazard.

Model: dx = kappa (theta_y - x) dt + sigma_y dW, y a two-state chain switching at rate lam each way,
u_y(t, x) = E[exp(-int_0^t x_r dr) | x_0 = x, y_0 = y].

Two checks, run with `python3 verify_expansion.py`:

1. The exact reduction (exact.py) agrees with a Monte Carlo referee that has no time grid: the regime path is
   sampled by exponential holding times and the Gaussian integral of the hazard is integrated out given the path.
2. The expansion in eps = 1/lam (expansion.py) has error of order eps, eps^2, eps^3 after
   0, 1 and 2 correction orders: the log2 error ratio under lam -> 2 lam tends to 1, 2, 3.
"""
import math
import numpy as np
from exact import exact_u
from expansion import Corrected

BASE = dict(kappa=2.0, thetas=[0.15, 0.02], sigmas=[math.sqrt(0.0555), math.sqrt(0.0055)])
X0 = 0.12


def monte_carlo(t, x, s, kappa, thetas, sigmas, lmbd, n_paths=200_000, seed=7):
    """No time grid. The regime path is sampled exactly, by exponential holding times, and given the path
    int_0^t x is Gaussian: its mean is x B(t) + kappa sum over holding periods [a, b] of theta_y int_a^b B(t - r) dr
    and its variance sum of sigma_y^2 int_a^b B(t - r)^2 dr, B(u) = (1 - exp(-kappa u)) / kappa. Each path then
    contributes exp(-mean + variance / 2) in closed form, so the only error is sampling error."""
    rng = np.random.default_rng(seed)
    th, sg = np.array(thetas), np.array(sigmas)
    B = lambda u: (1 - np.exp(-kappa * u)) / kappa
    G1 = lambda u: (u - B(u)) / kappa                                                    # int_0^u B
    G2 = lambda u: (u - 2 * B(u) + (1 - np.exp(-2 * kappa * u)) / (2 * kappa)) / kappa ** 2   # int_0^u B^2
    ys = np.full(n_paths, s, dtype=int)
    now = np.zeros(n_paths)
    alive = np.ones(n_paths, bool)
    mean, var = np.full(n_paths, x * float(B(t))), np.zeros(n_paths)
    while alive.any():
        idx = np.where(alive)[0]
        nxt = np.minimum(now[idx] + rng.exponential(1 / lmbd, idx.size), t)
        ta, tb = t - now[idx], t - nxt
        mean[idx] += kappa * th[ys[idx]] * (G1(ta) - G1(tb))
        var[idx] += sg[ys[idx]] ** 2 * (G2(ta) - G2(tb))
        now[idx] = nxt
        alive[idx] = nxt < t
        ys[idx] = 1 - ys[idx]
    v = np.exp(-mean + var / 2)
    return v.mean(), v.std(ddof=1) / math.sqrt(n_paths)


def main():
    ok = True
    print("1. Exact reduction vs Monte Carlo (lam = 1.7)")
    for t, s in [(1.0, 0), (4.0, 1)]:
        e = exact_u(t, X0, s, lmbd=1.7, **BASE)
        m, se = monte_carlo(t, X0, s, lmbd=1.7, **BASE)
        z = (m - e) / se
        print(f"   t={t} s={s}: exact {e:.6f}  MC {m:.6f} +/- {se:.6f}  z={z:+.2f}")
        ok &= abs(z) < 4  # no time discretization: the gap is sampling error only

    print("2. Convergence orders of the expansion (log2 of error ratio as lam doubles)")
    lams = [10, 20, 40, 80, 160, 320]
    for t in [1.0, 4.0]:
        for s in [0, 1]:
            sgn = 1 if s == 0 else -1
            errs = []
            for lm in lams:
                c = Corrected(lmbd=lm, **BASE)
                e = exact_u(t, X0, s, lmbd=lm, **BASE)
                ser = c.series(t, X0, sgn)
                errs.append([ser[0] - e, sum(ser[:3]) - e, sum(ser) - e])
            last = [math.log2(abs(a / b)) for a, b in zip(errs[-2], errs[-1])]
            print(f"   t={t} s={s}: orders {last[0]:.3f} {last[1]:.3f} {last[2]:.3f}")
            ok &= abs(last[0] - 1) < 0.05 and abs(last[1] - 2) < 0.05 and abs(last[2] - 3) < 0.05

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
