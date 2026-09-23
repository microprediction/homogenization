"""Monte Carlo check of the two-name credit correlation with no time discretization.

Given the regime path, theta(s) is piecewise constant and each CIR survival is exact:
    E[exp(-int_0^T x) | path] = exp(-B(T) x0 - kappa int_0^T theta(s) B(T - s) ds),
and int over a holding period [a, b] of B(T - s) is I1(T - a) - I1(T - b), with I1 the closed-form CIR integral.
The two names are independent given the path, so the joint survival is the product. Only the regime path is
simulated, exactly, by exponential holding times."""
import math
import numpy as np
from explicit import cir_B, cir_int_B

kap, sig, T = 2.0, 0.2, 3.0
th = np.array([[0.8, 0.005], [0.6, 0.005]])
x0 = np.array([0.05, 0.05])


def mc(lam, N=400000, seed=1):
    rng = np.random.default_rng(seed)
    I1 = np.vectorize(lambda r: cir_int_B(r, kap, sig))
    BT = cir_B(T, kap, sig)
    y = rng.integers(0, 2, N)                                  # stationary start
    t = np.zeros(N)
    acc = np.zeros((N, 2))                                     # int theta_j(s) B(T - s) ds for each name
    alive = np.ones(N, bool)
    while alive.any():
        idx = np.where(alive)[0]
        tn = np.minimum(t[idx] + rng.exponential(1 / lam, idx.size), T)
        w = I1(T - t[idx]) - I1(T - tn)
        acc[idx, 0] += th[0][y[idx]] * w
        acc[idx, 1] += th[1][y[idx]] * w
        t[idx] = tn
        y[idx] = 1 - y[idx]
        alive[idx] = tn < T
    e1 = np.exp(-BT * x0[0] - kap * acc[:, 0])
    e2 = np.exp(-BT * x0[1] - kap * acc[:, 1])
    e12 = e1 * e2
    corr = lambda a, b, ab: (ab - a * b) / math.sqrt((1 - a) * a * (1 - b) * b)
    c = corr(e1.mean(), e2.mean(), e12.mean())
    batches = [corr(e1[s].mean(), e2[s].mean(), e12[s].mean()) for s in np.array_split(np.arange(N), 20)]
    return c, np.std(batches) / math.sqrt(20)


if __name__ == '__main__':
    for lam in (1.0, 2.0, 4.0, 8.0):
        c, se = mc(lam)
        print(f"lam {lam}: Monte Carlo corr {c:.5f} +/- {se:.5f}")
