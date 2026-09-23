"""Certificate for the QuantLib models added to the engine: Merton76, variance gamma and Bates with switched parameters.
For each, the expansion's error against the numerical solution of the reduced system falls one order per order.
Orders whose error has reached rounding (below 1e-13) are not rated."""
import math
import numpy as np
from fastswitch import FastSwitch, numerical_a, numerical_a_callable
from quantlib_models import merton76, variance_gamma, bates

Q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])


def rates(make, t, u_list, lams=(20.0, 40.0), orders=range(0, 5), floor=1e-13):
    out = []
    for u in u_list:
        errs = {}
        for lam in lams:
            g, gf = make(u)[:2]
            ref = numerical_a_callable(t, lam * Q0, gf, rtol=1e-13)[0]
            fs = FastSwitch(lam * Q0, g, order=max(orders))
            errs[lam] = [abs(fs.a(t, o)[0] - ref) for o in orders]
        out.append([math.log2(a / b) for a, b in zip(errs[lams[0]], errs[lams[1]]) if b > floor])
    return out


def main():
    ok = True
    for name, make in [
        ("Merton76: volatility and jump intensity", lambda u: merton76(u, [0.1, 0.3], [0.5, 3.0], -0.1, 0.15)),
        ("variance gamma: sigma, nu, theta", lambda u: variance_gamma(u, [0.12, 0.3], [0.2, 0.5], [-0.05, -0.2])),
        ("Bates: variance level and jump intensity", lambda u: bates(u, 2.0, [0.09, 0.02], 0.4, -0.6, [0.5, 3.0], -0.1, 0.15, 1.0)),
    ]:
        rr = rates(make, 1.0, [0.5, 2.0], lams=(4.0, 8.0))
        for u, r in zip([0.5, 2.0], rr):
            good = all(abs(x - (k + 1)) < 0.35 for k, x in enumerate(r))
            ok &= good
            print(f"{'ok ' if good else 'BAD'} {name}, u={u}: convergence orders " + " ".join(f"{x:.2f}" for x in r) + " (expected 1..5)")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
