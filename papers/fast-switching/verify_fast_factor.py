"""Certificate for the fast mean-reverting factor: fastswitch_op.py and fastswitch_gen.py.

1. The operator engine reproduces the chain engine when the coupling is diagonal.
2. The multi-scale engine with q = 2 and no correlation reproduces order n in eps at order 2n in sqrt(eps).
3. With correlation, the error after n orders in delta = sqrt(eps) falls like delta^(n+1) against the exact
   Gaussian solution.
4. Monte Carlo of the two-factor model agrees with the exact solution.   `python3 verify_fast_factor.py`
"""
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastswitch import FastSwitch, ExpSum
from fastswitch_op import FastSwitchOp, Op, hermite_eval
from fastswitch_gen import FastSwitchGen
from models import fast_factor, fast_factor_exact

OK = True


def check(name, cond, detail):
    global OK
    OK &= bool(cond)
    print(f"{'ok ' if cond else 'FAIL'} {name}: {detail}")


def main():
    k = 2.0

    def G(theta, v):
        return ExpSum({0: -theta + v / (2 * k * k), k: theta - 2 * v / (2 * k * k), 2 * k: v / (2 * k * k)})
    g = [G(th, v) for th, v in zip([0.20, 0.08, 0.02], [0.04, 0.01, 0.0025])]
    Q = 8 * np.array([[-3, 2, 1], [1, -2, 1], [0.5, 1.5, -2]], float)
    fs = FastSwitch(Q, g, order=4)
    op = Op([(g[i], np.diag([1.0 if j == i else 0.0 for j in range(3)])) for i in range(3)])
    fo = FastSwitchOp(fs.Q0, fs.eps, op, fs.pi, np.ones(3), order=4)
    d = max(abs(fo.a(t, o) - fs.a(t, o)).max() for t in (0.2, 1.0) for o in range(5))
    check("operator engine = chain engine", d < 1e-13, f"max difference {d:.1e}")

    par = dict(kappa=1.0, theta0=0.05, theta1=0.03, sig0=0.25, sig1=0.2)
    L, pi, one, Gs = fast_factor(rho=0.0, order=6, **par)
    eps = 0.01
    f1 = FastSwitchOp(L, eps, Gs[0], pi, one, order=3)
    f2 = FastSwitchGen(L, math.sqrt(eps), 2, Gs, pi, one, order=6)
    d = max(abs(f1.a(t, o) - f2.a(t, 2 * o)).max() for t in (0.3, 2.0) for o in range(4))
    check("sqrt(eps) scaling without correlation", d < 1e-13, f"max difference {d:.1e}")

    rho, t, y = -0.7, 1.0, 0.3
    L, pi, one, Gs = fast_factor(rho=rho, order=6, **par)
    errs = {}
    for e in (0.0025, 0.000625):
        fg = FastSwitchGen(L, math.sqrt(e), 2, Gs, pi, one, order=6)
        ex = fast_factor_exact(t, y, e, rho=rho, **par)
        errs[e] = [abs(hermite_eval(fg.a(t, o), y) - ex) for o in range(6)]
    rates = [math.log2(a / b) for a, b in zip(errs[0.0025], errs[0.000625])]
    check("correlated orders in sqrt(eps)", all(abs(r - (n + 1)) < 0.3 for n, r in enumerate(rates)),
          " ".join(f"{r:.2f}" for r in rates) + " (expected 1..6)")

    rng = np.random.default_rng(4)
    P, n, e, x0 = 100000, 4000, 0.04, 0.04
    dt = t / n
    x, yv, I = np.full(P, x0), np.full(P, y), np.zeros(P)
    for _ in range(n):
        z1 = rng.standard_normal(P)
        z2 = rho * z1 + math.sqrt(1 - rho * rho) * rng.standard_normal(P)
        prev = x
        x = x + par['kappa'] * (par['theta0'] + par['theta1'] * yv - x) * dt + (par['sig0'] + par['sig1'] * yv) * math.sqrt(dt) * z1
        yv = yv - yv / e * dt + math.sqrt(2 / e * dt) * z2
        I += 0.5 * dt * (prev + x)
    v = np.exp(-I)
    se = v.std() / math.sqrt(P)
    B = 1 - math.exp(-par['kappa'] * t)
    ex = fast_factor_exact(t, y, e, rho=rho, **par) * math.exp(-B / par['kappa'] * x0)
    check("reduction by Monte Carlo", abs(v.mean() - ex) < 4 * se, f"exact {ex:.6f}, Monte Carlo {v.mean():.6f} +/- {se:.6f}")
    print("PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
