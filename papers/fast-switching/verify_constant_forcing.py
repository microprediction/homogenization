"""Certificate for the exact constant-forcing decomposition and its Fourier-frequency scope.

The two-state characteristic function has an exact slow outer mode plus a fast initial layer.  The slow mode is
analytic in z = g_tilde/lambda for |z| < 1, while the fast mode is exponentially small uniformly on |z| <= rho < 1.
The final check records the large-frequency growth of the variance-gamma forcing used by model_pages.py.
"""
import cmath
import math
import numpy as np


def exact(gbar, gtilde, eps, maturity, sign):
    lam = 1.0 / eps
    s = cmath.sqrt(lam * lam + gtilde * gtilde)
    return cmath.exp((gbar - lam) * maturity) * (
        cmath.cosh(s * maturity) + (lam + sign * gtilde) * cmath.sinh(s * maturity) / s
    )


def split(gbar, gtilde, eps, maturity, sign):
    z = eps * gtilde
    d = cmath.sqrt(1.0 + z * z)
    a = 0.5 * (1.0 + (1.0 + sign * z) / d)
    b = 0.5 * (1.0 - (1.0 + sign * z) / d)
    outer = cmath.exp(gbar * maturity) * a * cmath.exp((d - 1.0) * maturity / eps)
    layer = cmath.exp(gbar * maturity) * b * cmath.exp(-(d + 1.0) * maturity / eps)
    return outer, layer, d, b


def outer_quadratic(gbar, gtilde, eps, maturity, sign):
    z = eps * gtilde
    tg = maturity * gtilde
    p2 = 1.0 + 0.5 * (tg + sign) * z
    p2 += (tg * tg / 8.0 + sign * tg / 4.0 - 0.25) * z * z
    return cmath.exp(gbar * maturity) * p2


def variance_gamma_forcing(u):
    sigma = (0.25, 0.12)
    nu = (0.5, 0.2)
    theta = (-0.25, -0.10)

    def psi(z, s, n, t):
        return -cmath.log(1.0 - 1j * t * n * z + 0.5 * s * s * n * z * z) / n

    omega = tuple(math.log(1.0 - t * n - 0.5 * s * s * n) / n
                  for s, n, t in zip(sigma, nu, theta))
    omega_tilde = 0.5 * (omega[0] - omega[1])
    z = u - 0.5j
    psi_tilde = 0.5 * (psi(z, sigma[0], nu[0], theta[0]) - psi(z, sigma[1], nu[1], theta[1]))
    return 1j * z * omega_tilde + psi_tilde, omega_tilde


def main():
    rng = np.random.default_rng(31004)
    rho = 0.72
    eps = 0.08
    maturity = 0.65
    layer_constant = (math.sqrt(1.0 + rho * rho) + 1.0 + rho) / (2.0 * math.sqrt(1.0 - rho * rho))
    max_identity_error = 0.0
    max_layer_ratio = 0.0

    for sign in (-1, 1):
        for _ in range(1000):
            radius = rho * math.sqrt(rng.random())
            z = radius * cmath.exp(2j * math.pi * rng.random())
            gtilde = z / eps
            gbar = -0.4 + 0.3j * rng.normal()
            outer, layer, d, _ = split(gbar, gtilde, eps, maturity, sign)
            ref = exact(gbar, gtilde, eps, maturity, sign)
            max_identity_error = max(max_identity_error, abs(ref - outer - layer))

            assert d.real >= math.sqrt(1.0 - rho * rho) - 2e-15
            bound = (math.exp(gbar.real * maturity) * layer_constant
                     * math.exp(-(1.0 + math.sqrt(1.0 - rho * rho)) * maturity / eps))
            max_layer_ratio = max(max_layer_ratio, abs(layer) / bound)
            assert abs(layer) <= bound * (1.0 + 2e-14)

    assert max_identity_error < 2e-13

    # With g_tilde fixed, the slow-mode Taylor remainder is cubic in eps.  The fast mode is omitted here because its
    # separate bound is beyond all algebraic orders.
    gbar, gtilde, maturity = -0.37 + 0.11j, 0.43 - 0.27j, 0.9
    errors = []
    for eps in (0.10, 0.05, 0.025, 0.0125):
        outer = split(gbar, gtilde, eps, maturity, +1)[0]
        errors.append(abs(outer - outer_quadratic(gbar, gtilde, eps, maturity, +1)))
    orders = [math.log(errors[i] / errors[i + 1], 2.0) for i in range(len(errors) - 1)]
    assert min(orders) > 2.9

    # psi_1 - psi_2 is logarithmic, but the switched martingale correction is i*u*omega_tilde.  Consequently
    # g_tilde/(i*u) -> omega_tilde and the admissible large-frequency window is O(lambda), not exponential in lambda.
    rows = []
    for u in (200.0, 2_000.0, 20_000.0, 200_000.0):
        forcing, omega_tilde = variance_gamma_forcing(u)
        rows.append((u, abs(forcing), forcing / (1j * (u - 0.5j))))
    assert abs(rows[-1][2] - omega_tilde) < 2e-4
    assert abs(rows[-1][1] / rows[-1][0] - abs(omega_tilde)) < 2e-4

    print('exact split max error:', f'{max_identity_error:.3e}')
    print('largest layer / rigorous bound:', f'{max_layer_ratio:.6f}')
    print('outer Taylor errors:', [f'{x:.3e}' for x in errors])
    print('observed orders:', [f'{x:.4f}' for x in orders])
    print('VG omega_tilde:', f'{omega_tilde:.11f}')
    for u, size, ratio in rows:
        print(f'  u={u:8.0f}  |g_tilde|={size:12.6f}  g_tilde/(iu)={ratio.real:.9f}{ratio.imag:+.9f}i')
    print('PASS')


if __name__ == '__main__':
    main()
