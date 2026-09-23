"""Certificate for the all-orders expansion (outer series plus initial layer), all_orders.py.

Against a 30-digit exact solution of the Riccati system, the error after n correction orders
must fall like eps^(n+1): the log2 ratio as lam doubles approaches n + 1 until double-precision
round-off (about 1e-16) is reached. Runs in a few minutes (mpmath). `python3 verify_all_orders.py`
"""
import math
from all_orders import FullSeries, exact_mp

BASE = dict(kappa=2.0, thetas=[0.15, 0.02], sigmas=[math.sqrt(0.0555), math.sqrt(0.0055)])


def main():
    t, x, s, N = 0.3, 0.12, 0, 6
    errs = {}
    for lm in (20, 40):
        ex = exact_mp(t, x, s, lmbd=lm, **BASE)
        fs = FullSeries(lmbd=lm, order=N, **BASE)
        errs[lm] = [float(fs.u(t, x, s, order=n) - ex) for n in range(1, N + 1)]
    ok = True
    for n in range(1, N + 1):
        r = math.log2(abs(errs[20][n - 1] / errs[40][n - 1]))
        print(f"order {n}: error {errs[40][n - 1]:+.2e} at lam=40, observed rate {r:.2f} (expected {n + 1})")
        ok &= abs(r - (n + 1)) < 0.5
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
