"""Certificate for the G2++ remark on the instruments page: Jamshidian's decomposition needs one factor.

Under G2++ the bond maturing tau after expiry is A exp(-B_a(tau) x - B_b(tau) z), B_c(tau) = (1 - exp(-c tau)) / c.
1. The ratio B_a / B_b is strictly monotone in tau when a != b, so two payment dates never share a loading direction.
   For a = 0.5, b = 0.1 and residual maturities 1 and 4 the loading determinant is 0.94870454.
2. The exercise boundary of a two-payment coupon bond is therefore curved: its slope dz/dx moves between the two
   ratios -B_a / B_b, and a zero-coupon bond's boundary is a straight line.
`python3 verify_g2_coupon.py`
"""
import math
import numpy as np
from scipy.optimize import brentq

A_, B_ = 0.5, 0.1


def B(c, tau):
    return (1 - math.exp(-c * tau)) / c


def boundary(x, taus, cash, strike):
    """z at which the coupon bond equals the strike, and the slope dz/dx there (unit A for each cash flow)."""
    value = lambda z: sum(c * math.exp(-B(A_, t) * x - B(B_, t) * z) for c, t in zip(cash, taus)) - strike
    z = brentq(value, -50, 50, xtol=1e-14)
    w = [c * math.exp(-B(A_, t) * x - B(B_, t) * z) for c, t in zip(cash, taus)]
    return z, -sum(wi * B(A_, t) for wi, t in zip(w, taus)) / sum(wi * B(B_, t) for wi, t in zip(w, taus))


def main():
    ok = True
    print("1. loadings")
    det = B(A_, 1) * B(B_, 4) - B(A_, 4) * B(B_, 1)
    print(f"   (B_a, B_b) at 1: ({B(A_, 1):.8f}, {B(B_, 1):.8f}); at 4: ({B(A_, 4):.8f}, {B(B_, 4):.8f}); determinant {det:.8f}")
    ok &= abs(det - 0.94870454) < 1e-8
    taus = np.linspace(0.01, 40, 4000)
    ratio = np.array([B(A_, t) / B(B_, t) for t in taus])
    print(f"   B_a / B_b falls from {ratio[0]:.4f} to {ratio[-1]:.4f}, monotone: {bool(np.all(np.diff(ratio) < 0))}")
    ok &= bool(np.all(np.diff(ratio) < 0))

    print("2. exercise boundary of cash flows 0.5 at 1 and 0.6 at 4, strike 1")
    xs = [-0.3, -0.1, 0.0, 0.1, 0.3]
    slopes = [boundary(x, [1, 4], [0.5, 0.6], 1.0)[1] for x in xs]
    r1, r4 = -B(A_, 1) / B(B_, 1), -B(A_, 4) / B(B_, 4)
    print("   slope dz/dx at x = " + ", ".join(f"{x:+.1f}" for x in xs) + ": " + ", ".join(f"{s:.6f}" for s in slopes))
    print(f"   single-payment slopes: {r1:.6f} (maturity 1), {r4:.6f} (maturity 4)")
    ok &= max(slopes) - min(slopes) > 1e-3 and all(r1 < s < r4 for s in slopes)
    zc = [boundary(x, [4], [0.6], 0.5)[1] for x in xs]
    ok &= max(abs(s - r4) for s in zc) < 1e-12
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
