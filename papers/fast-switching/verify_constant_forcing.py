"""Certificate for the exact constant-forcing decomposition and its Fourier-frequency scope.

The two-state characteristic function has an exact slow outer mode plus a fast initial layer.  The slow mode is
analytic in z = g_tilde/lambda for |z| < 1, while the fast mode is exponentially small uniformly on |z| <= rho < 1.
The same statements are checked for unequal transition rates after replacing
z by the forcing contrast divided by the total switching rate.  The final
checks record the large-frequency growth of the variance-gamma forcing and
the separate compact-frequency and tail terms in Lewis inversion, and a
uniform cubic Lewis-price theorem for the variance-gamma benchmark on a
parabolic frequency window.
"""
import cmath
import math
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from scipy.optimize import brentq


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


def variance_gamma_parts(u, rate=0.03):
    sigma = (0.25, 0.12)
    nu = (0.5, 0.2)
    theta = (-0.25, -0.10)

    def psi(z, s, n, t):
        return -cmath.log(1.0 - 1j * t * n * z + 0.5 * s * s * n * z * z) / n

    omega = tuple(math.log(1.0 - t * n - 0.5 * s * s * n) / n
                  for s, n, t in zip(sigma, nu, theta))
    omega_tilde = 0.5 * (omega[0] - omega[1])
    g = tuple(1j * u * (rate + o) + psi(u, s, n, t)
              for s, n, t, o in zip(sigma, nu, theta, omega))
    return 0.5 * (g[0] + g[1]), 0.5 * (g[0] - g[1]), omega_tilde


def variance_gamma_forcing(u):
    _, forcing, omega_tilde = variance_gamma_parts(u - 0.5j)
    return forcing, omega_tilde


def variance_gamma_shifted_decay(rate=0.03):
    """Constants in a switching-rate-uniform bound on phi(u-i/2).

    For regime i, put a_i=sigma_i^2*nu_i/2, b_i=theta_i*nu_i,
    and c_i=1-b_i/2-a_i/4.  On the Lewis line z=u-i/2,

        Re(1-i*b_i*z+a_i*z^2) = c_i+a_i*u^2.

    If c_i>0, conditioning on the regime path therefore gives

        |phi(u-i/2)| <= exp(H*T) (1+q*u^2)^(-alpha*T),

    where q=min_i a_i/c_i, alpha=min_i 1/nu_i, and H is below.
    The estimate is independent of the transition generator.
    """
    sigma = (0.25, 0.12)
    nu = (0.5, 0.2)
    theta = (-0.25, -0.10)
    omega = tuple(math.log(1.0 - t * n - 0.5 * s * s * n) / n
                  for s, n, t in zip(sigma, nu, theta))
    a = tuple(0.5 * s * s * n for s, n in zip(sigma, nu))
    b = tuple(t * n for t, n in zip(theta, nu))
    c = tuple(1.0 - bi / 2.0 - ai / 4.0 for ai, bi in zip(a, b))
    assert min(c) > 0.0
    q = min(ai / ci for ai, ci in zip(a, c))
    alpha = min(1.0 / n for n in nu)
    h = max(0.5 * (rate + oi) - math.log(ci) / n
            for oi, ci, n in zip(omega, c, nu))
    return q, alpha, h, c


def symmetric_bad_occupation_bound(lam, maturity):
    """Chernoff bound for spending less than one third of the time in state 2.

    The chain starts in state 1 and jumps in either direction at rate ``lam``.
    With L_2 its state-2 occupation time, Markov's inequality at s=lam and
    the exact two-state Feynman--Kac transform give

      P(L_2 <= T/3) <= C_occ exp(-kappa_occ*lam*T).
    """
    c_occ = 0.5 * (1.0 + 3.0 / math.sqrt(5.0))
    kappa_occ = 7.0 / 6.0 - math.sqrt(5.0) / 2.0
    return c_occ * math.exp(-kappa_occ * lam * maturity), c_occ, kappa_occ


def symmetric_occupation_laplace(lam, maturity, penalty):
    """E_1 exp(-penalty*L_2) for the symmetric two-state chain."""
    delta = math.sqrt(lam * lam + penalty * penalty / 4.0)
    center = -lam - penalty / 2.0
    ratio = (lam + penalty / 2.0) / delta
    # Combine the outer exponential with the two hyperbolic modes first, so
    # the certificate remains evaluable when lam*T is much larger than 700.
    return (0.5 * (1.0 + ratio)
            * math.exp((center + delta) * maturity)
            + 0.5 * (1.0 - ratio)
            * math.exp((center - delta) * maturity))


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

    # A pointwise transform estimate becomes a Lewis-price estimate only after
    # accounting for the omitted frequency tail.  If X is the log return,
    # |phi(u-i/2)| <= E exp(X/2) gives the model-independent bound
    #
    #   int_R^infty |phi(u-i/2)|/(u^2+1/4) du
    #     <= M_{1/2} {pi - 2 atan(2R)}.
    #
    # Variance gamma supplies a sharper certificate.  On the Lewis line the
    # real part of each regime's quadratic denominator is c_i+a_i*u^2.
    # Conditioning on the regime path then yields, uniformly in its generator,
    #
    # |phi(u-i/2)| <= exp(H*T)*(1+q*u^2)^(-alpha*T).
    #
    # Thus its Lewis tail is at most
    #
    # exp(H*T)*q^(-alpha*T)*R^(-(2*alpha*T+1))/(2*alpha*T+1).
    #
    # We independently integrate the exact full transform and the quadratic
    # slow mode stopped at the largest R for which |g_tilde|/lambda <= r.
    # The calculation illustrates both parts of the rigorous triangle bound.
    rate, maturity, spot, strike = 0.03, 1.0, 100.0, 100.0
    envelope_ratio = 0.35
    decay_q, decay_alpha, decay_h, decay_c = variance_gamma_shifted_decay(rate)
    decay_power = 2.0 * decay_alpha * maturity + 1.0

    def vg_cf(u, lam, approximate=False):
        gbar, gtilde, _ = variance_gamma_parts(u - 0.5j, rate)
        if approximate:
            return outer_quadratic(gbar, gtilde, 1.0 / lam,
                                   maturity, +1)
        return exact(gbar, gtilde, 1.0 / lam, maturity, +1)

    def lewis_integral(fun, upper):
        return quad(lambda u: (fun(u) / (u * u + 0.25)).real,
                    0.0, upper, epsabs=2e-11, epsrel=2e-11,
                    limit=1000)[0]

    price_prefactor = math.sqrt(spot * strike) * math.exp(-rate * maturity) / math.pi
    inversion_rows = []
    for lam in (25.0, 50.0, 100.0, 200.0):
        window = brentq(
            lambda u: abs(variance_gamma_parts(u - 0.5j, rate)[1]) / lam
            - envelope_ratio,
            1.0, 10_000.0,
        )
        grid_envelope = max(
            abs(variance_gamma_parts(window * j / 2000 - 0.5j, rate)[1]) / lam
            for j in range(2001)
        )
        assert grid_envelope <= envelope_ratio * (1.0 + 2e-12)

        exact_full = lewis_integral(lambda u: vg_cf(u, lam), np.inf)
        approx_window = lewis_integral(
            lambda u: vg_cf(u, lam, approximate=True), window
        )
        interior_l1 = quad(
            lambda u: abs(vg_cf(u, lam)
                          - vg_cf(u, lam, approximate=True))
            / (u * u + 0.25),
            0.0, window, epsabs=2e-10, epsrel=2e-10,
            limit=1000,
        )[0]
        half_moment = vg_cf(0.0, lam).real
        assert half_moment > 0.0
        universal_tail = half_moment * (
            math.pi - 2.0 * math.atan(2.0 * window)
        )
        vg_tail = (math.exp(decay_h * maturity)
                   * decay_q ** (-decay_alpha * maturity)
                   * window ** (-decay_power) / decay_power)
        assert vg_tail < universal_tail

        # Direct pointwise checks are not part of the proof, but guard the
        # implementation of its constants over five frequency decades.
        for u in np.geomspace(1.0e-2, 1.0e4, 200):
            decay_bound = (math.exp(decay_h * maturity)
                           * (1.0 + decay_q * u * u)
                           ** (-decay_alpha * maturity))
            assert abs(vg_cf(u, lam)) <= decay_bound * (1.0 + 2e-11)

        price_error = price_prefactor * abs(exact_full - approx_window)
        price_bound = price_prefactor * (interior_l1 + vg_tail)
        assert price_error <= price_bound * (1.0 + 2e-10)
        inversion_rows.append((lam, window, price_error, price_bound,
                               price_prefactor * interior_l1,
                               price_prefactor * vg_tail,
                               price_prefactor * universal_tail))

    inversion_orders = [
        math.log(inversion_rows[j][2] / inversion_rows[j + 1][2], 2.0)
        for j in range(len(inversion_rows) - 1)
    ]
    interior_orders = [
        math.log(inversion_rows[j][4] / inversion_rows[j + 1][4], 2.0)
        for j in range(len(inversion_rows) - 1)
    ]
    assert min(inversion_orders) > 2.8
    assert min(interior_orders) > 2.8
    # The universal tail certificate is only first order when R is linear in
    # lambda.  The VG-specific certificate is fifth order here because
    # alpha=2 and T=1.  The computed triangle bound is consequently dominated
    # by its numerically integrated third-order interior term at large lambda.
    vg_tail_orders = [
        math.log(inversion_rows[j][5] / inversion_rows[j + 1][5], 2.0)
        for j in range(len(inversion_rows) - 1)
    ]
    universal_tail_orders = [
        math.log(inversion_rows[j][6] / inversion_rows[j + 1][6], 2.0)
        for j in range(len(inversion_rows) - 1)
    ]
    assert min(vg_tail_orders) > 4.8
    assert max(vg_tail_orders) < 5.3
    assert min(universal_tail_orders) > 0.8
    assert max(universal_tail_orders) < 1.2

    # A genuinely uniform cubic price theorem uses the smaller parabolic
    # window R_lambda=c*sqrt(lambda).  On this window z=g_tilde/lambda tends
    # to zero and the third Taylor remainder is bounded by
    #
    #   C lambda^-3 exp(T Re(g_bar)) (1+|g_tilde|)^6.
    #
    # For this benchmark exp(T Re(g_bar)) has power beta=7/2, so the displayed
    # envelope is integrable after division by the Lewis denominator.  The
    # exact tail needs a sharper argument than the worst-regime alpha=2 bound.
    # If L_2 is the occupation time of regime 2, its conditional decay exponent
    # is 2*T+3*L_2.  On L_2>=T/3 it is at least 3*T, while the complement has
    # the exponentially small Chernoff probability returned above.  Hence
    #
    # tail(R) <= exp(H*T) [q^-3 R^-7/7
    #              + P(L_2<T/3) q^-2 R^-5/5]                 (T=1).
    #
    # At R=c*sqrt(lambda) this is O(lambda^-7/2) plus an
    # exponentially small term, leaving the integrated Taylor remainder as
    # the cubic leading error.
    average_beta = 0.5 * sum(1.0 / n for n in (0.5, 0.2)) * maturity
    assert average_beta == 3.5
    assert average_beta > 2.5
    parabolic_scale = 10.0
    parabolic_rows = []
    weighted_remainder_ratios = []
    for lam in (50.0, 100.0, 200.0, 400.0):
        window = parabolic_scale * math.sqrt(lam)
        exact_full = lewis_integral(lambda u: vg_cf(u, lam), np.inf)
        approx_window = lewis_integral(
            lambda u: vg_cf(u, lam, approximate=True), window
        )
        interior_l1 = quad(
            lambda u: abs(vg_cf(u, lam)
                          - vg_cf(u, lam, approximate=True))
            / (u * u + 0.25),
            0.0, window, epsabs=2e-10, epsrel=2e-10,
            limit=1000,
        )[0]
        bad_probability, occupation_constant, occupation_rate = (
            symmetric_bad_occupation_bound(lam, maturity)
        )
        direct_chernoff = (math.exp(lam * maturity / 3.0)
                           * symmetric_occupation_laplace(
                               lam, maturity, lam
                           ))
        assert direct_chernoff <= bad_probability * (1.0 + 2e-14)
        good_tail = (math.exp(decay_h * maturity) * decay_q ** -3.0
                     * window ** -7.0 / 7.0)
        bad_tail = (math.exp(decay_h * maturity) * bad_probability
                    * decay_q ** -2.0 * window ** -5.0 / 5.0)
        exact_tail_bound = good_tail + bad_tail
        price_error = price_prefactor * abs(exact_full - approx_window)
        price_bound = price_prefactor * (interior_l1 + exact_tail_bound)
        assert price_error <= price_bound * (1.0 + 2e-10)

        # This grid is a diagnostic for the analytic differentiated-remainder
        # envelope, not a replacement for it.  The ratio stays bounded as the
        # parabolic window expands.
        eps_lam = 1.0 / lam
        for u in np.linspace(0.0, window, 401):
            gbar_u, gtilde_u, _ = variance_gamma_parts(u - 0.5j, rate)
            remainder = abs(vg_cf(u, lam)
                            - vg_cf(u, lam, approximate=True))
            scale = (eps_lam ** 3 * math.exp(gbar_u.real * maturity)
                     * (1.0 + abs(gtilde_u)) ** 6)
            if scale > 0.0:
                weighted_remainder_ratios.append(remainder / scale)
        for u in np.geomspace(1.0e-2, 1.0e4, 200):
            occupation_envelope = math.exp(decay_h * maturity) * (
                (1.0 + decay_q * u * u) ** -3.0
                + bad_probability * (1.0 + decay_q * u * u) ** -2.0
            )
            assert abs(vg_cf(u, lam)) <= occupation_envelope * (
                1.0 + 2e-11
            )
        parabolic_rows.append((lam, window, price_error, price_bound,
                               price_prefactor * interior_l1,
                               price_prefactor * good_tail,
                               price_prefactor * bad_tail))

    parabolic_orders = [
        math.log(parabolic_rows[j][2] / parabolic_rows[j + 1][2], 2.0)
        for j in range(len(parabolic_rows) - 1)
    ]
    parabolic_interior_orders = [
        math.log(parabolic_rows[j][4] / parabolic_rows[j + 1][4], 2.0)
        for j in range(len(parabolic_rows) - 1)
    ]
    assert min(parabolic_orders) > 2.8
    assert min(parabolic_interior_orders) > 2.8
    assert max(weighted_remainder_ratios) < 1.0

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
    print('VG shifted-decay constants (c, q, alpha, H, tail power):',
          [f'{x:.9f}' for x in decay_c], f'{decay_q:.12f}',
          f'{decay_alpha:.6f}', f'{decay_h:.12f}',
          f'{decay_power:.6f}')
    print('Lewis inversion certificate '
          '(lambda, R, error, bound, interior, VG tail, universal tail):')
    for row in inversion_rows:
        print(' ', f'{row[0]:5.0f}', f'{row[1]:10.4f}',
              *(f'{x:.6e}' for x in row[2:]))
    print('windowed-price observed orders:',
          [f'{x:.4f}' for x in inversion_orders])
    print('interior-L1 observed orders:',
          [f'{x:.4f}' for x in interior_orders])
    print('VG-tail-bound observed orders:',
          [f'{x:.4f}' for x in vg_tail_orders])
    print('universal-tail-bound observed orders:',
          [f'{x:.4f}' for x in universal_tail_orders])
    print('occupation Chernoff constants (C, kappa):',
          f'{occupation_constant:.12f}', f'{occupation_rate:.12f}')
    print('parabolic-window cubic certificate '
          '(lambda, R, error, bound, interior, good tail, bad tail):')
    for row in parabolic_rows:
        print(' ', f'{row[0]:5.0f}', f'{row[1]:10.4f}',
              *(f'{x:.6e}' for x in row[2:]))
    print('parabolic-price observed orders:',
          [f'{x:.4f}' for x in parabolic_orders])
    print('parabolic-interior observed orders:',
          [f'{x:.4f}' for x in parabolic_interior_orders])
    print('largest weighted cubic-remainder ratio:',
          f'{max(weighted_remainder_ratios):.9f}')
    print('PASS')


if __name__ == '__main__':
    main()
