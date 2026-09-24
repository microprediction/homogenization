"""Certificate for fastswitch.py, the all-orders fast-switching engine on any finite chain.

1. On two states it reproduces general_orders.py term for term (sums of exponentials and Chebyshev series).
2. On a non-reversible three-state chain the error after n orders falls like eps^(n+1) against a numerical
   solution computed at 30 digits; the errors are formed at that precision.
3. A chain with a Jordan block.
4. A complex forcing that is real to high order at t = 0 keeps its imaginary part.
5. A terminal vector with pi . a0 = 0: the error after n orders still falls like eps^(n+1).
6. Nearly equal but distinct eigenvalues of Q: with g = 0 the engine reproduces exp(Q t) a0.
7. The numerical solver keeps complex arithmetic for a forcing that is real at some times.
8. A Chebyshev product above the degree cap raises instead of losing terms.
9. An irreducible defective three-state generator (issue 7: Q* = J - I + 0.1 u v^T with v.u = 0, so the eigenvalue
   -1 has a Jordan block): the error after n orders falls like eps^(n+1), in both engines.
`python3 verify_engine.py`  (about a minute)
"""
import math
import cmath
import os
import sys
import numpy as np
import mpmath as mp
from numpy.polynomial import Chebyshev
from scipy.linalg import expm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'regime-switching-survival'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastswitch import FastSwitch, ExpSum, Cheb, numerical_a, numerical_a_callable
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
        ex = numerical_a(0.3, Qs, g, dps=30, mp_values=True)
        fs = FastSwitch(Qs, g, order=6)
        errs[sc] = [float(mp.mpf(float(fs.a(0.3, o)[0])) - ex[0]) for o in range(1, 7)]
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

    # g = i (1 - e^{-t})^5 in both regimes: real through t^4 at 0; the answer is exp(int g)
    gc = ExpSum({j: 1j * (-1) ** j * math.comb(5, j) for j in range(6)})
    ac = FastSwitch([[-10, 10], [10, -10]], [gc, gc], order=1).a(2.0)
    errc = float(np.abs(ac - cmath.exp(gc.integral(2.0))).max())
    print(f"4. complex forcing flat at t = 0: error {errc:.1e}")
    ok &= errc < 1e-13

    # terminal vector with stationary mean zero
    pi3 = FastSwitch(base, g, order=0).pi
    a0 = np.array([1.0, -1.0, 0.0])
    a0[2] = -(pi3[0] - pi3[1]) / pi3[2]
    errz = {}
    for sc in (8, 16):
        exz = np.array(numerical_a(0.3, sc * base, g, dps=20, a0=a0), float)
        fz = FastSwitch(sc * base, g, order=4, a0=a0)
        errz[sc] = [float(np.abs(fz.a(0.3, o) - exz).max()) for o in range(5)]
    rz = [math.log2(a / b) for a, b in zip(errz[8], errz[16])]
    print("5. pi . a0 = 0: rates " + " ".join(f"{r:.2f}" for r in rz) + " (orders 2..4 expected 3..5)")
    ok &= all(abs(rz[n] - (n + 1)) < 0.5 for n in (2, 3, 4))

    # nearly equal, distinct eigenvalues (-1.4999994 and -1.5000006 for Q / 1000)
    u = np.array([1., -1., 0.]) / np.sqrt(2)
    v = np.array([1., 1., -2.]) / np.sqrt(6)
    Qd = 1000 * (np.ones((3, 3)) / 3 - np.eye(3) + 4e-7 * (np.outer(u, u) - np.outer(v, v)))
    e1 = np.array([1., 0., 0.])
    fd = FastSwitch(Qd, [ExpSum.const(0)] * 3, order=4, a0=e1)
    errd = max(float(np.abs(fd.a(t, 4) - expm(Qd * t) @ e1).max()) for t in (fd.eps / 4, fd.eps, 4 * fd.eps))
    print(f"6. nearly equal eigenvalues, g = 0: max difference from exp(Qt) a0 {errd:.1e}")
    ok &= errd < 1e-14

    # g = i (t - 0.3): real at t = 0.3 only
    gi = lambda t: 1j * (t - 0.3)
    an = numerical_a_callable(1.0, [[-5, 5], [5, -5]], [gi, gi])
    erri = float(np.abs(an - cmath.exp(0.2j)).max())
    print(f"7. numerical solution, complex forcing real at t = 0.3: error {erri:.1e}")
    ok &= erri < 1e-10

    # T_61 squared has a T_122 component above the cap of 120; T_40 squared fits and is kept exactly
    f61 = Cheb(Chebyshev([0.] * 61 + [1.], domain=[0, 1]))
    try:
        FastSwitch([[-20, 20], [20, -20]], [f61, f61.scale(-1)], order=1)
        raised = False
    except ValueError:
        raised = True
    f40 = Cheb(Chebyshev([0.] * 40 + [1.], domain=[0, 1]))
    lt = FastSwitch([[-20, 20], [20, -20]], [f40, f40.scale(-1)], order=1).log_terms[1]
    errc2 = abs(lt.integral(0.4) - (f40.s * f40.s * 0.5).integ(lbnd=0)(0.4))
    print(f"8. Chebyshev degree cap: degree 122 raises {raised}; degree 80 first-order term error {errc2:.1e}")
    ok &= raised and errc2 < 1e-13

    # irreducible but defective: every off-diagonal entry of Q* is positive and N = 0.1 u v^T is nilpotent
    from fastswitch_gen import FastSwitchGen
    from fastswitch_op import Op
    Qstar = np.ones((3, 3)) / 3 - np.eye(3) + 0.1 * np.outer([1., -1., 0.], [1., 1., -2.])
    g9 = [ExpSum.const(c) for c in (-0.2, 0.3, 0.6)]
    err9, errg = {}, 0.0
    for sc in (10, 20):
        Q9 = sc * Qstar
        ex9 = numerical_a(1.0, Q9, g9, dps=30, mp_values=True)
        fs9 = FastSwitch(Q9, g9, order=5)
        err9[sc] = [max(abs(float(mp.mpf(float(fs9.a(1.0, o)[i])) - ex9[i])) for i in range(3)) for o in range(6)]
        op9 = Op([(g9[i], np.diag([1.0 if j == i else 0.0 for j in range(3)])) for i in range(3)])
        fg9 = FastSwitchGen(fs9.Q0, fs9.eps, 1, {0: op9}, fs9.pi, np.ones(3), order=5)
        errg = max(errg, max(float(np.abs(fg9.a(t, o) - fs9.a(t, o)).max()) for t in (0.3, 1.0) for o in range(6)))
    r9 = [math.log2(a / b) for a, b in zip(err9[10], err9[20])]
    print("9. defective irreducible chain: errors at lam = 10 " + " ".join(f"{e:.1e}" for e in err9[10]) +
          "; rates " + " ".join(f"{r:.2f}" for r in r9) + " (expected 1..6); general engine differs by "
          f"{errg:.1e}")
    ok &= err9[10][5] < 1e-7 and all(abs(r - (n + 1)) < 0.5 for n, r in enumerate(r9)) and errg < 1e-12
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
