"""Certificate for general_orders.py: arbitrary switching rates lam12, lam21.

1. With lam12 = lam21 = lam the series equals all_orders.py term for term.
2. With unequal occupancy (p = 0.3) the error after n orders falls like eps^(n+1), eps = 1/(lam12 + lam21),
   against a 30-digit numerical solution of the two-state linear ODE.  `python3 verify_general_orders.py`
"""
import math
from general_orders import GeneralSeries, numerical_u
from all_orders import FullSeries

BASE = dict(kappa=2.0, thetas=[0.15, 0.02], sigmas=[math.sqrt(0.0555), math.sqrt(0.0055)])


def main():
    ok = True
    w = 0.0
    for lam in (5, 20):
        g = GeneralSeries(lam12=lam, lam21=lam, order=6, **BASE)
        f = FullSeries(lmbd=lam, order=6, **BASE)
        for t in (0.3, 2.0):
            for s in (0, 1):
                for o in range(7):
                    w = max(w, abs(g.u(t, 0.12, s, o) - f.u(t, 0.12, s, o)))
    print(f"1. symmetric case: max difference from all_orders.py {w:.1e}")
    ok &= w < 1e-14
    t = 0.3
    for s in (0, 1):
        errs = {}
        for gam in (20, 40):
            ex = numerical_u(t, 0.12, s, lam12=0.7 * gam, lam21=0.3 * gam, dps=30, **BASE)
            g = GeneralSeries(lam12=0.7 * gam, lam21=0.3 * gam, order=6, **BASE)
            errs[gam] = [float(g.u(t, 0.12, s, o) - ex) for o in range(1, 7)]
        rates = [math.log2(abs(a / b)) for a, b in zip(errs[20], errs[40])]
        print(f"2. p = 0.3, regime {s + 1}: rates " + " ".join(f"{r:.2f}" for r in rates) + " (expected 2..7)")
        ok &= all(abs(r - (n + 2)) < 0.5 for n, r in enumerate(rates))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
