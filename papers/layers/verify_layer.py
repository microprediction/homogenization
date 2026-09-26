"""Certificate for the two-state initial layer to second order, with a forcing that does not vanish at the start.

    omega' = q(t) (1 - omega^2) - 2 omega / eps,   omega(0) = 0,      (log m)' = b(t) + q(t) omega.
Composite expansion, uniform on [0, T], with E(t) = e^{-2t/eps} and I(t) = int_0^t q(s) E(s) ds:
    omega = (eps/2) {q - q(0) E} - (eps^2/4) {q' - q'(0) E} + O(eps^3),
    log m = int b + (eps/2) int q^2 - (eps q(0)/2) I - (eps^2/8) {q^2 - q(0)^2} + (eps^2 q'(0)/4) I + O(eps^3).
The outer series alone is O(eps) wrong in omega near t = 0 when q(0) != 0, which costs O(eps^2) in log m; the first
composite is O(eps^2) uniformly in both; the second is O(eps^3). Measured on q(t) = 0.4 + 0.3 sin(2t) - 0.2 t e^{-t}, b = 0, over [0, 3].
"""
import math
import numpy as np
from scipy.integrate import solve_ivp, quad

q = lambda t: 0.4 + 0.3 * math.sin(2 * t) - 0.2 * t * math.exp(-t)
dq = lambda t: 0.6 * math.cos(2 * t) - 0.2 * math.exp(-t) + 0.2 * t * math.exp(-t)
T = 3.0


def exact(eps, ts):
    rhs = lambda t, y: [q(t) * (1 - y[0] ** 2) - 2 * y[0] / eps, q(t) * y[0]]
    sol = solve_ivp(rhs, (0, T), [0.0, 0.0], method='Radau', rtol=1e-12, atol=1e-15, dense_output=True)
    return sol.sol(ts)


def composites(eps, ts):
    q0, dq0 = q(0), dq(0)
    E = lambda t: math.exp(-2 * t / eps)
    I = lambda t: quad(lambda s: q(s) * E(s), 0, t, epsabs=1e-15, epsrel=1e-13, limit=200)[0]
    Q2 = lambda t: quad(lambda s: q(s) ** 2, 0, t, epsabs=1e-15, epsrel=1e-13)[0]
    om_out = np.array([eps / 2 * q(t) - eps ** 2 / 4 * dq(t) for t in ts])
    om1 = np.array([eps / 2 * (q(t) - q0 * E(t)) for t in ts])
    om2 = np.array([eps / 2 * (q(t) - q0 * E(t)) - eps ** 2 / 4 * (dq(t) - dq0 * E(t)) for t in ts])
    lm_out = np.array([eps / 2 * Q2(t) - eps ** 2 / 8 * (q(t) ** 2 - q0 ** 2) for t in ts])
    lm1 = np.array([eps / 2 * Q2(t) - eps * q0 / 2 * I(t) for t in ts])
    lm2 = np.array([eps / 2 * Q2(t) - eps * q0 / 2 * I(t) - eps ** 2 / 8 * (q(t) ** 2 - q0 ** 2) + eps ** 2 * dq0 / 4 * I(t) for t in ts])
    return om_out, om1, om2, lm_out, lm1, lm2


def main():
    ok = True
    epss = [0.2, 0.1, 0.05, 0.025]
    names = ['omega: outer to second order', 'omega: first composite', 'omega: second composite',
             'log m: outer to second order', 'log m: first composite', 'log m: second composite']
    want = [1, 2, 3, 2, 2, 3]      # the outer series misses omega by O(eps) only over a window of width eps: O(eps^2) in log m
    errs = np.zeros((len(epss), 6))
    for i, eps in enumerate(epss):
        ts = np.concatenate([np.linspace(0, 4 * eps, 200), np.linspace(4 * eps, T, 300)])
        om, lm = exact(eps, ts)
        for j, approx in enumerate(composites(eps, ts)):
            errs[i, j] = np.abs((om if j < 3 else lm) - approx).max()
    print(f"   q(0) = {q(0):.3f}, q'(0) = {dq(0):.3f}; largest error on [0, 3] for eps = {epss}")
    for j, name in enumerate(names):
        o = math.log2(errs[-2, j] / errs[-1, j])
        ok &= abs(o - want[j]) < 0.15
        print(f"   {name:30s} " + "  ".join(f"{e:.2e}" for e in errs[:, j]) + f"   order {o:.2f} (expected {want[j]})")
    print("PASS" if ok else "FAIL")


if __name__ == '__main__':
    main()
