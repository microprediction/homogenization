"""Certificate for the exact constant-forcing decomposition and its Fourier-frequency scope.

The two-state characteristic function has an exact slow outer mode plus a fast initial layer.  The slow mode is
analytic in z = g_tilde/lambda for |z| < 1, while the fast mode is exponentially small uniformly on |z| <= rho < 1.
The same statements are checked for unequal transition rates after replacing
z by the forcing contrast divided by the total switching rate.  The final
check records the large-frequency growth of the variance-gamma forcing used
by model_pages.py.
"""
import cmath
import math
import numpy as np
from scipy.linalg import expm


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


def unequal_split(g1, g2, rate12, rate21, speed, maturity, sign):
    """Exact slow/fast split for Q=speed*[[-rate12,rate12],[rate21,-rate21]]."""
    total = rate12 + rate21
    theta = (rate12 - rate21) / total
    eps = 1.0 / (speed * total)
    delta = g1 - g2
    z = eps * delta
    d = cmath.sqrt(1.0 - 2.0 * theta * z + z * z)
    gbar = (rate21 * g1 + rate12 * g2) / total
    a = 0.5 * (1.0 + (1.0 + sign * z) / d)
    b = 0.5 * (1.0 - (1.0 + sign * z) / d)
    outer = (cmath.exp(gbar * maturity) * a
             * cmath.exp((d - 1.0 + theta * z) * maturity / (2.0 * eps)))
    layer = (cmath.exp(gbar * maturity) * b
             * cmath.exp(-(d + 1.0 - theta * z) * maturity / (2.0 * eps)))
    return outer, layer, d, z


def unequal_direct(g1, g2, rate12, rate21, speed, maturity):
    matrix = np.array([
        [g1 - speed * rate12, speed * rate12],
        [speed * rate21, g2 - speed * rate21],
    ], dtype=complex)
    return expm(matrix * maturity) @ np.ones(2)


def unequal_outer_quadratic(g1, g2, rate12, rate21, speed,
                            maturity, sign):
    """Quadratic Taylor polynomial of the unequal-rate slow mode."""
    total = rate12 + rate21
    theta = (rate12 - rate21) / total
    eps = 1.0 / (speed * total)
    delta = g1 - g2
    z = eps * delta
    gbar = (rate21 * g1 + rate12 * g2) / total
    c1 = maturity * delta * (1.0 - theta * theta) / 4.0
    c2 = maturity * delta * theta * (1.0 - theta * theta) / 4.0
    a1 = (theta + sign) / 2.0
    a2 = (3.0 * theta * theta - 1.0) / 4.0 + sign * theta / 2.0
    p1 = a1 + c1
    p2 = a2 + c2 + 0.5 * c1 * c1 + a1 * c1
    return cmath.exp(gbar * maturity) * (1.0 + p1 * z + p2 * z * z)


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

    # Unequal transition rates.  The discriminant is
    # d(z)^2=1-2*theta*z+z^2, whose two roots have modulus one for positive
    # rates.  Hence the radius-one analytic domain survives asymmetry.
    rate12, rate21 = 1.7, 0.4
    maturity = 0.8
    unequal_identity_error = 0.0
    rng = np.random.default_rng(41004)
    for _ in range(500):
        speed = rng.uniform(3.0, 40.0)
        g1 = rng.normal(-0.2, 0.5) + 1j * rng.normal(0.0, 0.4)
        g2 = rng.normal(-0.2, 0.5) + 1j * rng.normal(0.0, 0.4)
        direct = unequal_direct(
            g1, g2, rate12, rate21, speed, maturity
        )
        split_values = []
        for sign in (1, -1):
            outer, layer, _, _ = unequal_split(
                g1, g2, rate12, rate21, speed, maturity, sign
            )
            split_values.append(outer + layer)
        unequal_identity_error = max(
            unequal_identity_error,
            np.max(np.abs(direct - np.asarray(split_values))),
        )
    assert unequal_identity_error < 3e-13

    theta = (rate12 - rate21) / (rate12 + rate21)
    branch_points = np.array([
        theta + 1j * math.sqrt(1.0 - theta * theta),
        theta - 1j * math.sqrt(1.0 - theta * theta),
    ])
    assert np.max(np.abs(np.abs(branch_points) - 1.0)) < 3e-16

    # The slow-mode polynomial is third-order accurate in inverse total
    # switching speed.  The fast mode is omitted because its separate compact-
    # disk bound is exponentially small.
    unequal_g1, unequal_g2 = 0.31 - 0.22j, -0.47 + 0.19j
    unequal_errors = []
    for speed in (8.0, 16.0, 32.0, 64.0):
        outer = unequal_split(
            unequal_g1, unequal_g2, rate12, rate21, speed,
            maturity, +1,
        )[0]
        approximation = unequal_outer_quadratic(
            unequal_g1, unequal_g2, rate12, rate21, speed,
            maturity, +1,
        )
        unequal_errors.append(abs(outer - approximation))
    unequal_orders = [
        math.log(unequal_errors[i] / unequal_errors[i + 1], 2.0)
        for i in range(len(unequal_errors) - 1)
    ]
    assert min(unequal_orders) > 2.9

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
    print('unequal-rate exact split max error:',
          f'{unequal_identity_error:.3e}')
    print('unequal-rate branch-point moduli:',
          [f'{abs(x):.12f}' for x in branch_points])
    print('unequal-rate outer Taylor errors:',
          [f'{x:.3e}' for x in unequal_errors])
    print('unequal-rate observed orders:',
          [f'{x:.4f}' for x in unequal_orders])
    print('VG omega_tilde:', f'{omega_tilde:.11f}')
    for u, size, ratio in rows:
        print(f'  u={u:8.0f}  |g_tilde|={size:12.6f}  g_tilde/(iu)={ratio.real:.9f}{ratio.imag:+.9f}i')
    print('PASS')


if __name__ == '__main__':
    main()
