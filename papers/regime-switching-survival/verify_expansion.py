"""Certificate for the fast-switching expansion of survival under a regime-switching OU hazard.

Model: dx = kappa (theta_y - x) dt + sigma_y dW, y a two-state chain switching at rate lam each way,
u_y(t, x) = E[exp(-int_0^t x_r dr) | x_0 = x, y_0 = y].

Two checks, run with `python3 verify_expansion.py`:

1. The exact reduction (exact.py) agrees with a Monte Carlo referee.
2. The expansion in eps = 1/lam (expansion.py) has error of order eps, eps^2, eps^3 after
   0, 1 and 2 correction orders: the log2 error ratio under lam -> 2 lam tends to 1, 2, 3.
"""
import math
import numpy as np
from exact import exact_u
from expansion import Corrected

BASE = dict(kappa=2.0, thetas=[0.15, 0.02], sigmas=[math.sqrt(0.0555), math.sqrt(0.0055)])
X0 = 0.12


def monte_carlo(t, x, s, kappa, thetas, sigmas, lmbd, n_paths=200_000, n_steps=400, seed=7):
    """Exact-in-law OU steps between switch checks; trapezoid for the integral."""
    rng = np.random.default_rng(seed)
    dt = t / n_steps
    th, sg = np.array(thetas), np.array(sigmas)
    xs = np.full(n_paths, float(x))
    ys = np.full(n_paths, s, dtype=int)
    integral = np.zeros(n_paths)
    decay = math.exp(-kappa * dt)
    sd_factor = math.sqrt((1 - decay ** 2) / (2 * kappa))
    for _ in range(n_steps):
        prev = xs.copy()
        xs = th[ys] + (xs - th[ys]) * decay + sg[ys] * sd_factor * rng.standard_normal(n_paths)
        integral += 0.5 * dt * (prev + xs)
        flip = rng.random(n_paths) < 1 - math.exp(-lmbd * dt)
        ys = np.where(flip, 1 - ys, ys)
    v = np.exp(-integral)
    return v.mean(), v.std() / math.sqrt(n_paths)


def main():
    ok = True
    print("1. Exact reduction vs Monte Carlo (lam = 1.7)")
    for t, s in [(1.0, 0), (4.0, 1)]:
        e = exact_u(t, X0, s, lmbd=1.7, **BASE)
        m, se = monte_carlo(t, X0, s, lmbd=1.7, **BASE)
        z = (m - e) / se
        print(f"   t={t} s={s}: exact {e:.6f}  MC {m:.6f} +/- {se:.6f}  z={z:+.2f}")
        ok &= abs(z) < 4 and abs(m - e) < 2e-3  # time-step bias is O(dt^2) for the trapezoid

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
