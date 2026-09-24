"""Certificate for the closed-form helpers in explicit.py and bond_option_explicit.py.

1. cir_B, cir_int_B and cir_int_B2 against 25-digit quadrature, including volatilities near zero, short and long horizons.
2. two_state_constant_exact against the matrix exponential, at switching rates where cosh and sinh overflow, at a
   complex Black-Scholes frequency, and where s = 0.
3. The Vasicek moments int_0^T e^{-k kappa t} B^m dt against 30-digit quadrature, for kappa T from 1e-8 to 50.
4. The first-order bond call at slow mean reversion (kappa = 1e-4) against the numerical Gil-Pelaez price.
`python3 verify_explicit.py`
"""
import numpy as np
import mpmath as mp
from scipy.linalg import expm
from explicit import cir_B, cir_int_B, cir_int_B2, two_state_constant_exact, vasicek_moment, I_k
import bond_option_explicit as bo


def main():
    ok = True
    mp.mp.dps = 25
    worst = 0.0
    for kappa in (0.5, 2.0):
        for sigma in (0.0, 1e-4, 1e-3, 5e-3, 1e-2, 0.1, 1.0):
            h = mp.sqrt(kappa ** 2 + 2 * mp.mpf(sigma) ** 2)

            def B(z):
                p = -mp.expm1(-h * z)
                return 2 * p / ((h + kappa) * p + 2 * h * (1 - p))
            for t in (1e-3, 0.1, 0.49, 0.51, 1.0, 10.0, 400.0):
                pts = mp.linspace(0, t, 21)
                errs = [abs(cir_int_B2(t, kappa, sigma) / mp.quad(lambda z: B(z) ** 2, pts) - 1),
                        abs(cir_B(t, kappa, sigma) / B(t) - 1)]
                if sigma:
                    errs.append(abs(cir_int_B(t, kappa, sigma) / mp.quad(B, pts) - 1))
                worst = max(worst, float(max(errs)))
    good = worst < 1e-13
    ok &= good
    print(f"{'ok ' if good else 'BAD'} 1. CIR B, int B, int B^2: largest relative error {worst:.1e}")

    def ref2(gbar, gt, lam, t):
        return (expm(np.array([[gbar + gt - lam, lam], [lam, gbar - gt - lam]], complex) * t) @ np.ones(2))
    cases = [(0.0, 0.0, lam, 1.0) for lam in (700.0, 800.0, 1000.0)]            # PGF at z = 1: exactly 1
    u, r, sig = 1 - 0.5j, 0.03, (0.4, 0.1)                                         # Black-Scholes, Lewis frequency
    gs = [1j * u * (r - s * s / 2) - u * u * s * s / 2 for s in sig]
    cases += [((gs[0] + gs[1]) / 2, (gs[0] - gs[1]) / 2, lam, 1.0) for lam in (5.0, 800.0)]
    cases += [(-0.3, 2j, 2.0, 1.5), (0.1, 0.5, 3.0, 0.2)]                         # s = 0, and an ordinary case
    worst = 0.0
    for gbar, gt, lam, t in cases:
        ex = ref2(gbar, gt, lam, t)
        for i, sign in enumerate((+1, -1)):
            worst = max(worst, abs(two_state_constant_exact(gbar, gt, lam, t, sign) - ex[i]) / abs(ex[i]))
    good = worst < 1e-12
    ok &= good
    print(f"{'ok ' if good else 'BAD'} 2. two-state constant closed form: largest relative error {worst:.1e}")

    mp.mp.dps = 30
    worst = 0.0
    for x in (1e-8, 1e-4, 1e-2, 0.1, 0.5, 1.0, 3.0, 10.0, 50.0):
        T = 3.0
        kappa = x / T
        for k in range(5):
            for m in range(8 if k == 0 else 5):
                f = lambda t: mp.e ** (-k * kappa * t) * (-mp.expm1(-kappa * t) / kappa) ** m
                ref = mp.quad(f, mp.linspace(0, T, 9))
                worst = max(worst, float(abs(vasicek_moment(k, m, T, kappa) / ref - 1)))
                if k == 0:
                    worst = max(worst, float(abs(I_k(m, T, kappa) / ref - 1)))
    good = worst < 1e-13
    ok &= good
    print(f"{'ok ' if good else 'BAD'} 3. Vasicek moments, kappa T from 1e-8 to 50: largest relative error {worst:.1e}")

    from options import zcb_call
    args = (1e-4, [.05, .03], [.3, .1], .04, 1., 4., .9, 20.)
    c1 = bo.call(*args)
    num = zcb_call(1., 4., .9, .04, 0, 1e-4, [.05, .03], [.3, .1], 20 * np.array([[-1., 1.], [1., -1.]]), n=80)
    good = abs(c1 - num) < 3e-4
    ok &= good
    print(f"{'ok ' if good else 'BAD'} 4. bond call, kappa = 1e-4, lam = 20: first order {c1:.8f}, numerical {num:.8f}, "
          f"difference {abs(c1 - num):.1e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
