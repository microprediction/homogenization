"""Certificate for fastswitch.py, the all-orders fast-switching engine on any finite chain.

1. On two states it reproduces general_orders.py term for term (sums of exponentials and Chebyshev series).
2. On a non-reversible three-state chain the error after n orders falls like eps^(n+1) against a
   30-digit numerical solution.  `python3 verify_engine.py`  (about a minute)
"""
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'regime-switching-survival'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastswitch import FastSwitch, ExpSum, Cheb, numerical_a
from general_orders import GeneralSeries

k = 2.0


def G(theta, s2):
    """-k theta B + s2 B^2 / 2 with B = (1 - exp(-k t)) / k, as a sum of exponentials"""
    return ExpSum({0: -theta + s2 / (2 * k * k), k: theta - 2 * s2 / (2 * k * k), 2 * k: s2 / (2 * k * k)})


def Gf(theta, s2):
    return lambda t: -theta * (1 - math.exp(-k * t)) + 0.5 * s2 * ((1 - math.exp(-k * t)) / k) ** 2


def main():
    ok = True
    th, sg = [0.15, 0.02], [math.sqrt(0.0555), math.sqrt(0.0055)]
    gs = GeneralSeries(kappa=k, thetas=th, sigmas=sg, lam12=14, lam21=6, order=6)
    Q = [[-14, 14], [6, -6]]
    engines = {'exponential sums': FastSwitch(Q, [G(th[i], sg[i] ** 2) for i in range(2)], order=6),
               'Chebyshev': FastSwitch(Q, [Cheb.fit(Gf(th[i], sg[i] ** 2), 10.0, 60) for i in range(2)], order=6)}
    for name, fs in engines.items():
        w = 0.0
        for t in (0.3, 2.0, 6.0):
            B = (1 - math.exp(-k * t)) / k
            for o in range(7):
                a = fs.a(t, o)
                for i in range(2):
                    w = max(w, abs(a[i] * math.exp(-B * 0.12) - gs.u(t, 0.12, i, o)))
        print(f"1. two states, {name}: max difference {w:.1e}")
        ok &= w < 1e-12
    th3, s3 = [0.20, 0.08, 0.02], [0.04, 0.01, 0.0025]
    g = [G(th3[i], s3[i]) for i in range(3)]
    base = np.array([[-3, 2, 1], [1, -2, 1], [0.5, 1.5, -2]], float)
    errs = {}
    for sc in (8, 16):
        Qs = sc * base
        ex = numerical_a(0.3, Qs, g, dps=25)
        fs = FastSwitch(Qs, g, order=6)
        errs[sc] = [float(fs.a(0.3, o)[0] - ex[0]) for o in range(1, 7)]
    rates = [math.log2(abs(a / b)) for a, b in zip(errs[8], errs[16])]
    print("2. three states: rates " + " ".join(f"{r:.2f}" for r in rates) + " (expected 2..7)")
    ok &= all(abs(r - (n + 2)) < 0.5 for n, r in enumerate(rates))
    # a defective generator: eigenvalues 0, -3, -3 with a Jordan block, so no eigenvector basis exists
    J = np.array([[-1, 1, 0], [0, -1, 1], [4, 0, -4]], float)
    gJ = [ExpSum({0: c}) for c in (1.0, 2.0, 3.0)]
    exJ = np.array(numerical_a(0.3, 100 * J, gJ, dps=30), float)
    fsJ = FastSwitch(100 * J, gJ, order=4)
    errJ = [float(np.abs(fsJ.a(0.3, o) - exJ).max()) for o in range(5)]
    print("3. Jordan-block chain: errors by order " + " ".join(f"{e:.1e}" for e in errJ))
    ok &= errJ[4] < 1e-10 and all(b < a / 20 for a, b in zip(errJ, errJ[1:]))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
