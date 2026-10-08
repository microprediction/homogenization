"""Numerical certificate for correlation-tail homogenization scales.

The proof is on tools/pages/mixing-scale.html.  This certificate independently
integrates the exact finite-horizon covariance and checks the Green--Kubo,
long-memory, critical, and zero-Green--Kubo regimes and the scalar and vector
periodic counterexamples, including the Gaussian fractional-Brownian functional
limit, the critical Gaussian finite-dimensional limit and failure of
path-space tightness, an ergodic second-chaos counterexample with nonvanishing
limiting skewness, a covariance-matched hierarchy of arbitrary Hermite ranks,
and the sharp Fourier-decay and Cesaro-mean conditions at the
saturated epsilon-squared boundary.
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import sici, zeta


def integrated_variance(covariance, epsilon, maturity=1.0):
    """Variance of integral_0^T phi(t/epsilon) dt under stationarity."""
    value, error = quad(
        lambda u: 2.0 * (maturity - u) * covariance(u / epsilon),
        0.0,
        maturity,
        epsabs=2e-13,
        epsrel=2e-11,
        points=[epsilon] if epsilon < maturity else None,
        limit=800,
    )
    assert error < 2e-9 * max(value, 1e-12)
    return value


def measured_order(values):
    return math.log(values[-2] / values[-1], 2.0)


def verify_integrable_case():
    # A symmetric two-state chain accelerated by 1/epsilon has covariance
    # exp(-2t/epsilon).  Its Green--Kubo integral before acceleration is 1/2.
    epsilons = 2.0 ** -np.arange(4, 11)
    exact = np.array([
        integrated_variance(lambda t: math.exp(-2.0 * t), eps)
        for eps in epsilons
    ])
    closed = epsilons - 0.5 * epsilons ** 2 * (1.0 - np.exp(-2.0 / epsilons))
    assert np.max(np.abs(exact - closed)) < 2e-12
    assert abs(measured_order(exact) - 1.0) < 0.003
    assert abs(exact[-1] / epsilons[-1] - 1.0) < 6e-4
    return measured_order(exact)


def verify_zero_frequency_atom():
    """Check the invariant spectral atom omitted by the punctured integral.

    Let X_t=Z+Y_t, where Var(Z)=atom_mass and Y has covariance exp(-|t|).
    The random constant is the shift-invariant projection and contributes
    atom_mass*T^2 at every acceleration rate.
    """
    atom_mass = 0.37
    maturity = 1.3
    quadrature_epsilons = 2.0 ** -np.arange(3, 11)

    covariance_values = np.array([
        integrated_variance(
            lambda t: atom_mass + math.exp(-t), eps, maturity
        )
        for eps in quadrature_epsilons
    ])
    quadrature_exact = (
        atom_mass * maturity ** 2
        + 2.0 * maturity * quadrature_epsilons
        - 2.0 * quadrature_epsilons ** 2
        * (1.0 - np.exp(-maturity / quadrature_epsilons))
    )
    identity_error = float(np.max(np.abs(
        covariance_values - quadrature_exact
    )))
    assert identity_error < 2e-12

    epsilons = 2.0 ** -np.arange(3, 15)
    exact = (
        atom_mass * maturity ** 2
        + 2.0 * maturity * epsilons
        - 2.0 * epsilons ** 2
        * (1.0 - np.exp(-maturity / epsilons))
    )
    invariant_variance = atom_mass * maturity ** 2
    omitted_atom_error = float(np.max(np.abs(
        exact
        - (2.0 * maturity * epsilons
           - 2.0 * epsilons ** 2
           * (1.0 - np.exp(-maturity / epsilons)))
        - invariant_variance
    )))
    assert omitted_atom_error < 2e-16

    normalized_limit = exact[-1] / maturity ** 2
    assert abs(normalized_limit - atom_mass) < 3e-4
    remainder = exact - invariant_variance
    green_kubo_ratio = remainder[-1] / (2.0 * maturity * epsilons[-1])
    assert abs(green_kubo_ratio - 1.0) < 1e-4
    return (identity_error, omitted_atom_error, normalized_limit,
            green_kubo_ratio)


def normalized_ou_variance(scale):
    """Normalized integrated variance for covariance exp(-|t|).

    The direct formula 2*(R - 1 + exp(-R))/R^2 loses digits at small R,
    so use its Taylor expansion there.
    """
    scale = np.asarray(scale, dtype=float)
    values = np.empty_like(scale)
    small = scale < 1e-3
    r = scale[small]
    values[small] = (
        1.0 - r / 3.0 + r ** 2 / 12.0
        - r ** 3 / 60.0 + r ** 4 / 360.0
    )
    r = scale[~small]
    values[~small] = 2.0 * (r + np.expm1(-r)) / r ** 2
    return values


def verify_all_scale_crossover():
    """Check the complete T/epsilon crossover, including an invariant atom.

    For C(t)=atom_mass+exp(-|t|), the normalized variance is exactly

        atom_mass + 2*(R - 1 + exp(-R))/R^2,  R=T/epsilon.

    It tends to C(0) as R->0, has a nontrivial finite-R crossover, and
    tends to the invariant atom as R->infinity.  Independent covariance
    quadrature also checks that maturity and epsilon enter only through R.
    """
    atom_mass = 0.37
    scales = 2.0 ** np.arange(-8, 9)
    maturities = (0.2, 1.3, 7.0)
    exact = atom_mass + normalized_ou_variance(scales)

    normalized_quadrature = []
    for maturity in maturities:
        normalized_quadrature.append(np.array([
            integrated_variance(
                lambda t: atom_mass + math.exp(-t),
                maturity / scale,
                maturity,
            ) / maturity ** 2
            for scale in scales
        ]))
    normalized_quadrature = np.array(normalized_quadrature)
    quadrature_error = float(np.max(np.abs(
        normalized_quadrature - exact[None, :]
    )))
    collapse_error = float(np.max(np.ptp(normalized_quadrature, axis=0)))

    frozen_value = float(
        atom_mass + normalized_ou_variance(np.array([2.0 ** -24]))[0]
    )
    finite_value = float(
        atom_mass + normalized_ou_variance(np.array([1.0]))[0]
    )
    invariant_value = float(
        atom_mass + normalized_ou_variance(np.array([2.0 ** 24]))[0]
    )
    frozen_error = abs(frozen_value - (atom_mass + 1.0))
    finite_error = abs(finite_value - (atom_mass + 2.0 / math.e))
    invariant_error = abs(invariant_value - atom_mass)

    assert quadrature_error < 3e-12
    assert collapse_error < 3e-12
    assert frozen_error < 3e-8
    assert finite_error < 2e-16
    assert invariant_error < 2e-7
    return (quadrature_error, collapse_error, frozen_value, finite_value,
            invariant_value, frozen_error, invariant_error)


def fejer_kernel(scale, frequency):
    """Normalized variance kernel K_R(omega)=sinc(R*omega/2)^2."""
    return float(np.sinc(scale * frequency / (2.0 * math.pi)) ** 2)


def verify_vector_rank_crossover():
    """Check the matrix theorem and show that finite-R rank is nonmonotone.

    With independent uniform phases U,V, the stationary ergodic torus flow

        X_t=(sqrt(2) cos(U+t), sqrt(2) cos(V+sqrt(2)t))

    has covariance diag(cos(t), cos(sqrt(2)t)).  The normalized covariance of
    its time integral is diag(K_R(1), K_R(sqrt(2))).  Its rank is 2, 1, 2 at
    R=pi, 2*pi, 3*pi: one frequency lands on a Fejer zero only in the middle.
    """
    scales = math.pi * np.arange(1.0, 4.0)
    frequencies = (1.0, math.sqrt(2.0))
    exact = np.array([
        [fejer_kernel(scale, frequency) for frequency in frequencies]
        for scale in scales
    ])

    maturity = 1.7

    def normalized_covariance(scale, frequency):
        value, error = quad(
            lambda u: 2.0 * (maturity - u)
            * math.cos(frequency * u * scale / maturity),
            0.0,
            maturity,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=400,
        )
        assert error < 5e-13
        return value / maturity ** 2

    quadrature = np.array([
        [normalized_covariance(scale, frequency)
         for frequency in frequencies]
        for scale in scales
    ])
    identity_error = float(np.max(np.abs(quadrature - exact)))
    ranks = tuple(
        int(np.linalg.matrix_rank(np.diag(row), tol=1e-12))
        for row in exact
    )
    resonant_eigenvalues = tuple(float(value) for value in exact[1])

    assert identity_error < 2e-12
    assert ranks == (2, 1, 2)
    assert resonant_eigenvalues[0] < 1e-30
    assert resonant_eigenvalues[1] > 0.01
    return identity_error, ranks, resonant_eigenvalues


def sign_gaussian_covariance(t, alpha, delta=1.0):
    correlation = (1.0 + t * t) ** (-alpha / 2.0)
    return 2.0 * delta * delta * math.asin(correlation) / math.pi


def verify_long_memory_case(alpha):
    epsilons = 2.0 ** -np.arange(8, 19)
    exact = np.array([
        integrated_variance(lambda t: sign_gaussian_covariance(t, alpha), eps)
        for eps in epsilons
    ])
    constant = 4.0 / (math.pi * (1.0 - alpha) * (2.0 - alpha))
    asymptotic = constant * epsilons ** alpha
    ratio = exact[-1] / asymptotic[-1]
    assert abs(ratio - 1.0) < (0.025 if alpha <= 0.4 else 0.08)
    order = measured_order(exact)
    assert abs(order - alpha) < 0.025
    return order, ratio


def verify_critical_case():
    # C(t)=1/(1+t) is an exponential mixture and hence positive definite.
    epsilons = 2.0 ** -np.arange(8, 19)
    exact = np.array([
        integrated_variance(lambda t: 1.0 / (1.0 + t), eps)
        for eps in epsilons
    ])
    leading = 2.0 * epsilons * np.log(1.0 / epsilons)
    ratio = exact[-1] / leading[-1]
    assert abs(ratio - 1.0) < 0.09
    return ratio


def antipersistent_covariance(t, alpha):
    """Valid covariance with zero Green--Kubo integral for 1 < alpha < 2.

    This is the cosine transform of the Gamma(alpha, 1) density:

        C(t) = Re (1-it)^(-alpha).

    Its tail coefficient cos(pi alpha/2) is negative.
    """
    return (1.0 + t * t) ** (-alpha / 2.0) * math.cos(alpha * math.atan(t))


def gamma_spectral_density(omega, alpha):
    """One-sided spectral density of antipersistent_covariance."""
    return (math.exp(-omega) * omega ** (alpha - 1.0)
            / math.gamma(alpha))


def antipersistent_variance(epsilon, alpha, maturity=1.0):
    """Exact integrated variance for antipersistent_covariance."""
    x = maturity / epsilon
    if alpha == 2.0:
        return epsilon ** 2 * math.log1p(x * x)
    numerator = ((1.0 - 1j * x) ** (2.0 - alpha)).real - 1.0
    return (2.0 * epsilon ** 2 * numerator
            / ((alpha - 1.0) * (2.0 - alpha)))


def spectral_abelian_constant(alpha, density_coefficient=1.0):
    """Constant for f(omega) ~ a omega^(alpha-1), 0 < alpha < 2."""
    return (density_coefficient * math.pi
            / (math.gamma(3.0 - alpha)
               * math.sin(math.pi * alpha / 2.0)))


def atomic_spectral_variance(epsilon, alpha, maturity=1.0,
                             cutoff_multiple=6.0, tail_terms=10):
    """Integrated variance for a purely atomic low-frequency spectrum.

    Put mass n^(-alpha-1) at frequency 1/n.  The direct sum is cut off
    beyond ``cutoff_multiple * maturity / epsilon``.  There the cosine
    series is absolutely convergent, and each remaining power sum is a
    Hurwitz zeta value.
    """
    scale = maturity / epsilon
    cutoff = int(math.ceil(cutoff_multiple * scale))
    indices = np.arange(1, cutoff + 1, dtype=float)
    direct = np.sum(
        indices ** (1.0 - alpha) * (1.0 - np.cos(scale / indices))
    )
    tail = 0.0
    for order in range(1, tail_terms + 1):
        tail += ((-1.0) ** (order + 1)
                 * scale ** (2 * order) / math.factorial(2 * order)
                 * zeta(alpha + 2 * order - 1.0, cutoff + 1.0))
    return 2.0 * epsilon ** 2 * (direct + tail)


def verify_measure_level_spectral_theorem():
    """Check the density-free theorem on a pure-point spectrum.

    For masses w_n=n^(-alpha-1) at omega_n=1/n,

        F(x)=sum_{n >= ceil(1/x)} w_n ~ x^alpha/alpha.

    The measure has no density.  The general cumulative-mass theorem
    nevertheless predicts the same constant as density coefficient one.
    """
    alpha = 1.3
    small_x = 2.0 ** -20
    first_index = int(math.ceil(1.0 / small_x))
    cumulative_mass = zeta(alpha + 1.0, first_index)
    mass_ratio = cumulative_mass / (small_x ** alpha / alpha)

    epsilons = 2.0 ** -np.arange(7, 16)
    exact = np.array([
        atomic_spectral_variance(eps, alpha) for eps in epsilons
    ])
    independently_refined = np.array([
        atomic_spectral_variance(
            eps, alpha, cutoff_multiple=8.0, tail_terms=12
        )
        for eps in epsilons
    ])
    summation_error = float(np.max(np.abs(exact - independently_refined)))

    coefficient = spectral_abelian_constant(alpha)
    leading = coefficient * epsilons ** alpha
    order = measured_order(exact)
    ratio = exact[-1] / leading[-1]

    assert abs(mass_ratio - 1.0) < 1e-6
    assert summation_error < 2e-13
    assert abs(order - alpha) < 0.004
    assert abs(ratio - 1.0) < 0.002
    return mass_ratio, summation_error, order, ratio


def verify_measure_level_critical_theorem():
    """Check the density-free alpha=2 endpoint on a pure-point spectrum.

    Put mass n^(-3) at frequency 1/n.  Then

        F(x) ~ x^2 / 2,

    so b=1/2, L=1 and H(x)=log(x).  The cumulative-mass endpoint
    predicts

        Var A_epsilon(1) ~ 2 epsilon^2 log(1/epsilon).

    The convergence is only logarithmic.  Accordingly, the certificate
    records both the terminal ratio and the slope of Var/epsilon^2 against
    log(1/epsilon), whose predicted value is two.
    """
    small_x = 2.0 ** -20
    first_index = int(math.ceil(1.0 / small_x))
    cumulative_mass = zeta(3.0, first_index)
    mass_ratio = cumulative_mass / (small_x ** 2 / 2.0)

    epsilons = 2.0 ** -np.arange(8, 21)
    exact = np.array([
        atomic_spectral_variance(
            eps, 2.0, cutoff_multiple=6.0, tail_terms=10
        )
        for eps in epsilons
    ])
    independently_refined = np.array([
        atomic_spectral_variance(
            eps, 2.0, cutoff_multiple=8.0, tail_terms=12
        )
        for eps in epsilons
    ])
    summation_error = float(np.max(np.abs(exact - independently_refined)))

    logarithm = np.log(1.0 / epsilons)
    normalized = exact / epsilons ** 2
    log_slope = float(np.polyfit(logarithm, normalized, 1)[0])
    ratio = float(exact[-1] / (2.0 * epsilons[-1] ** 2 * logarithm[-1]))

    assert abs(mass_ratio - 1.0) < 2e-6
    assert summation_error < 2e-13
    assert abs(log_slope - 2.0) < 0.05
    assert abs(ratio - 1.0) < 0.03
    return mass_ratio, summation_error, log_slope, ratio


def verify_joint_maturity_spectral_theorem():
    """Check that the spectral asymptotic only needs T/epsilon -> infinity.

    The exact normalized variance depends on epsilon and maturity only through
    R=T/epsilon.  We evaluate the same R grid along fixed-, vanishing-, and
    growing-maturity paths.  The paths therefore have to collapse before the
    asymptotic approximation is even invoked.
    """
    scales = 2.0 ** np.arange(8, 25)
    maturities = (
        np.ones_like(scales),
        scales ** -0.5,
        scales ** 0.5,
    )

    alpha = 1.7
    coefficient = spectral_abelian_constant(
        alpha, density_coefficient=1.0 / math.gamma(alpha)
    )
    power_ratios = []
    normalized_power = []
    critical_ratios = []
    normalized_critical = []
    for maturity in maturities:
        epsilon = maturity / scales
        power_variance = np.array([
            antipersistent_variance(eps, alpha, term)
            for eps, term in zip(epsilon, maturity)
        ])
        power_leading = (
            coefficient * maturity ** (2.0 - alpha)
            * epsilon ** alpha
        )
        power_ratios.append(power_variance / power_leading)
        normalized_power.append(power_variance / maturity ** 2)

        critical_variance = np.array([
            antipersistent_variance(eps, 2.0, term)
            for eps, term in zip(epsilon, maturity)
        ])
        critical_leading = 2.0 * epsilon ** 2 * np.log(scales)
        critical_ratios.append(critical_variance / critical_leading)
        normalized_critical.append(critical_variance / maturity ** 2)

    power_ratios = np.array(power_ratios)
    normalized_power = np.array(normalized_power)
    critical_ratios = np.array(critical_ratios)
    normalized_critical = np.array(normalized_critical)
    power_collapse_error = float(np.max(np.ptp(normalized_power, axis=0)))
    critical_collapse_error = float(np.max(
        np.ptp(normalized_critical, axis=0)
    ))
    power_terminal_spread = float(np.ptp(power_ratios[:, -1]))
    critical_terminal_spread = float(np.ptp(critical_ratios[:, -1]))
    power_terminal_ratio = float(power_ratios[0, -1])
    critical_terminal_ratio = float(critical_ratios[0, -1])

    assert power_collapse_error < 3e-15
    assert critical_collapse_error < 3e-15
    assert power_terminal_spread < 3e-15
    assert critical_terminal_spread < 3e-15
    assert abs(power_terminal_ratio - 1.0) < 0.009
    assert abs(critical_terminal_ratio - 1.0) < 1e-12
    return (power_collapse_error, critical_collapse_error,
            power_terminal_spread, critical_terminal_spread,
            power_terminal_ratio, critical_terminal_ratio)


def band_variance(epsilon, weight, lower, upper, maturity=1.0):
    """Exact contribution from a flat spectral band.

    The one-sided spectral density is ``weight`` on [lower, upper].
    """
    scale = maturity / epsilon

    def primitive(omega):
        sine_integral = sici(scale * omega)[0]
        return (-(1.0 - math.cos(scale * omega)) / omega
                + scale * sine_integral)

    return (2.0 * epsilon ** 2 * weight
            * (primitive(upper) - primitive(lower)))


def verify_spectral_abelian_theorem():
    """Check the low-frequency theorem beyond one-signed covariance tails.

    Add a flat band away from zero to the Gamma spectral density.  Its
    covariance contribution oscillates like 1/t and dominates the signed
    pointwise tail when alpha > 1, but its integrated-variance contribution
    is only O(epsilon^2).  The low-frequency omega^(alpha-1) term still gives
    the sharp epsilon^alpha law.
    """
    # First reconcile the frequency-domain constant with both time-domain
    # Karamata constants for the Gamma spectral benchmark.
    constant_errors = []
    for alpha in (0.4, 1.4, 1.7):
        spectral = spectral_abelian_constant(
            alpha, density_coefficient=1.0 / math.gamma(alpha)
        )
        covariance_tail = math.cos(math.pi * alpha / 2.0)
        if alpha < 1.0:
            temporal = (2.0 * covariance_tail
                        / ((1.0 - alpha) * (2.0 - alpha)))
        else:
            temporal = (-2.0 * covariance_tail
                        / ((alpha - 1.0) * (2.0 - alpha)))
        constant_errors.append(abs(spectral - temporal))
    constant_error = max(constant_errors)
    assert constant_error < 2e-14

    alpha = 1.7
    weight, lower, upper = 0.8, 4.0, 5.0

    def mixed_covariance(t):
        gamma_part = antipersistent_covariance(t, alpha)
        if t == 0.0:
            band_part = weight * (upper - lower)
        else:
            band_part = (weight
                         * (math.sin(upper * t) - math.sin(lower * t)) / t)
        return gamma_part + band_part

    # Independent time-domain quadrature agrees with the exact spectral
    # decomposition at moderate switching rates.
    quadrature_epsilons = 2.0 ** -np.arange(3, 8)
    covariance_values = np.array([
        integrated_variance(mixed_covariance, eps)
        for eps in quadrature_epsilons
    ])
    spectral_values = np.array([
        antipersistent_variance(eps, alpha)
        + band_variance(eps, weight, lower, upper)
        for eps in quadrature_epsilons
    ])
    identity_error = float(np.max(np.abs(
        covariance_values - spectral_values
    )))
    assert identity_error < 2e-15

    # The exact spectral expression can be evaluated deep in the asymptotic
    # regime without oscillatory quadrature.
    epsilons = 2.0 ** -np.arange(8, 25)
    exact = np.array([
        antipersistent_variance(eps, alpha)
        + band_variance(eps, weight, lower, upper)
        for eps in epsilons
    ])
    coefficient = spectral_abelian_constant(
        alpha, density_coefficient=1.0 / math.gamma(alpha)
    )
    leading = coefficient * epsilons ** alpha
    order = measured_order(exact)
    ratio = exact[-1] / leading[-1]
    band_fraction = band_variance(
        epsilons[-1], weight, lower, upper
    ) / exact[-1]
    assert abs(order - alpha) < 0.004
    assert abs(ratio - 1.0) < 0.009
    assert band_fraction < 1e-4

    # The remote band makes the covariance oscillate forever.  Count sign
    # changes on a long deterministic mesh to certify that eventual sign is
    # absent in this concrete example.
    times = np.linspace(20.0, 2000.0, 100001)
    covariance = np.array([mixed_covariance(t) for t in times])
    signs = np.sign(covariance)
    sign_changes = int(np.count_nonzero(signs[1:] * signs[:-1] < 0.0))
    assert sign_changes > 3000
    return (constant_error, identity_error, order, ratio,
            band_fraction, sign_changes)


def verify_antipersistent_case(alpha):
    """Check the zero-Green--Kubo, regularly varying negative-tail law."""
    # First check the closed form against independent covariance quadrature.
    quadrature_epsilons = 2.0 ** -np.arange(4, 13)
    quadrature = np.array([
        integrated_variance(lambda t: antipersistent_covariance(t, alpha), eps)
        for eps in quadrature_epsilons
    ])
    closed = np.array([
        antipersistent_variance(eps, alpha) for eps in quadrature_epsilons
    ])
    identity_error = float(np.max(np.abs(quadrature - closed)))
    assert identity_error < 2e-15

    # The exact formula remains stable much farther into the asymptotic
    # regime than cancellation-prone numerical integration.
    epsilons = 2.0 ** -np.arange(8, 25)
    exact = np.array([
        antipersistent_variance(eps, alpha) for eps in epsilons
    ])
    tail_coefficient = math.cos(math.pi * alpha / 2.0)
    constant = (-2.0 * tail_coefficient
                / ((alpha - 1.0) * (2.0 - alpha)))
    leading = constant * epsilons ** alpha
    ratio = exact[-1] / leading[-1]
    order = measured_order(exact)
    assert tail_coefficient < 0.0
    assert abs(order - alpha) < 0.004
    assert abs(ratio - 1.0) < 0.009
    return order, ratio, identity_error


def verify_gaussian_fractional_limit():
    """Check covariance convergence to fractional Brownian motion.

    A centered Gaussian process is determined by its covariance.  For the
    Gamma spectral benchmark, the exact variance of every integrated increment
    therefore gives an exact finite-dimensional certificate for the functional
    limit.  We check both persistent and antipersistent exponents.
    """
    times = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    results = []
    for alpha, powers in ((0.7, (8, 12, 16, 20)),
                          (1.7, (10, 20, 30))):
        hurst = 1.0 - alpha / 2.0
        target = np.array([
            [0.5 * (s ** (2.0 * hurst) + t ** (2.0 * hurst)
                    - abs(t - s) ** (2.0 * hurst))
             for t in times]
            for s in times
        ])
        errors = []
        for power in powers:
            epsilon = 2.0 ** -power
            variances = np.array([
                antipersistent_variance(epsilon, alpha, t)
                if t > 0.0 else 0.0
                for t in times
            ])
            terminal_variance = variances[-1]
            covariance = np.empty_like(target)
            for i, s in enumerate(times):
                for j, t in enumerate(times):
                    lag_variance = (
                        antipersistent_variance(
                            epsilon, alpha, abs(t - s)
                        ) if t != s else 0.0
                    )
                    covariance[i, j] = (
                        variances[i] + variances[j] - lag_variance
                    ) / (2.0 * terminal_variance)
            errors.append(float(np.max(np.abs(covariance - target))))

        assert all(x > y for x, y in zip(errors, errors[1:]))
        if alpha < 1.0:
            assert errors[-1] < 1.3e-6
        else:
            assert errors[-1] < 1.2e-3
        results.append((alpha, hurst, tuple(errors), powers[-1]))
    return tuple(results)


def verify_critical_gaussian_boundary():
    """Check the alpha=2 Gaussian finite-dimensional boundary.

    The one-sided spectral density f(omega)=omega exp(-omega) has

        F(x) ~ x^2/2,        V(R)=log(1+R^2).

    Consequently the normalized integrated process has limiting covariance
    one on the positive-time diagonal and one half at every pair of distinct
    positive times.  Its increments over every fixed positive time interval
    retain asymptotic variance one, which is the obstruction to C[0,1]
    tightness.
    """
    scales = np.array([0.25, 1.0, 4.0, 16.0])
    quadrature = np.array([
        integrated_variance(
            lambda t: antipersistent_covariance(t, 2.0), 1.0, scale
        )
        for scale in scales
    ])
    closed = np.log1p(scales ** 2)
    identity_error = float(np.max(np.abs(quadrature - closed)))

    times = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    target = np.zeros((len(times), len(times)))
    for i, s in enumerate(times):
        for j, t in enumerate(times):
            if s > 0.0 and t > 0.0:
                target[i, j] = 1.0 if i == j else 0.5

    powers = (10, 20, 40, 80, 160)
    errors = []
    increment_variances = []
    delta = 0.01
    for power in powers:
        horizon = 2.0 ** power

        def variance(t):
            return math.log1p((horizon * t) ** 2)

        terminal_variance = variance(1.0)
        covariance = np.empty_like(target)
        for i, s in enumerate(times):
            for j, t in enumerate(times):
                covariance[i, j] = (
                    variance(s) + variance(t) - variance(abs(t - s))
                ) / (2.0 * terminal_variance)
        errors.append(float(np.max(np.abs(covariance - target))))
        increment_variances.append(variance(delta) / terminal_variance)

    assert identity_error < 3e-15
    assert all(x > y for x, y in zip(errors, errors[1:]))
    assert errors[-1] < 0.021
    assert increment_variances[-1] > 0.958
    return (identity_error, tuple(errors), tuple(increment_variances),
            powers, delta)


def verify_ergodic_second_chaos_counterexample():
    """Certify that matching covariance scaling need not give an fBm limit.

    Let G be stationary standard Gaussian with covariance
    r(t)=(1+|t|)^(-beta), and put X=G^2-1.  The variance and third cumulant of
    I_R=int_0^R X_t dt reduce to one- and two-dimensional deterministic
    integrals.  Their normalized ratio tends to a strictly positive constant,
    whereas every centered Gaussian has zero third cumulant.
    """
    beta = 0.3
    horizons = (16.0, 256.0, 4096.0, 65536.0)

    variance_constant = (
        4.0 / ((1.0 - 2.0 * beta) * (2.0 - 2.0 * beta))
    )
    simplex_constant = (
        math.gamma(1.0 - beta) ** 2
        / math.gamma(2.0 - 2.0 * beta)
        / ((2.0 - 3.0 * beta) * (3.0 - 3.0 * beta))
    )
    limiting_skewness = (
        48.0 * simplex_constant / variance_constant ** 1.5
    )

    skewnesses = []
    variance_ratios = []
    cumulant_ratios = []
    for horizon in horizons:
        variance = 4.0 * quad(
            lambda lag: (
                (horizon - lag) * (1.0 + lag) ** (-2.0 * beta)
            ),
            0.0,
            horizon,
            epsabs=1e-7,
            epsrel=2e-10,
            limit=500,
        )[0]

        # On the ordered simplex, set x=t-s, y=u-t, w=x+y and
        # q=x/(x+y).  The Jacobian is w and the six orderings are equal.
        def cumulant_integrand(total_lag):
            split_integral = quad(
                lambda split: (
                    (1.0 + total_lag * split) ** (-beta)
                    * (1.0 + total_lag * (1.0 - split)) ** (-beta)
                ),
                0.0,
                1.0,
                epsabs=1e-10,
                epsrel=2e-10,
            )[0]
            return (
                (horizon - total_lag)
                * (1.0 + total_lag) ** (-beta)
                * total_lag
                * split_integral
            )

        third_cumulant = 48.0 * quad(
            cumulant_integrand,
            0.0,
            horizon,
            epsabs=1e-4,
            epsrel=2e-8,
            limit=500,
        )[0]
        skewnesses.append(third_cumulant / variance ** 1.5)
        variance_ratios.append(
            variance
            / (variance_constant * horizon ** (2.0 - 2.0 * beta))
        )
        cumulant_ratios.append(
            third_cumulant
            / (48.0 * simplex_constant * horizon ** (3.0 - 3.0 * beta))
        )

    assert all(x > y for x, y in zip(skewnesses, skewnesses[1:]))
    assert abs(skewnesses[-1] / limiting_skewness - 1.0) < 0.024
    assert variance_ratios[-1] > 0.96
    assert cumulant_ratios[-1] > 0.98
    assert limiting_skewness > 2.0
    return (
        beta,
        limiting_skewness,
        tuple(skewnesses),
        variance_ratios[-1],
        cumulant_ratios[-1],
    )


def verify_covariance_matched_hermite_hierarchy():
    """Check that every Hermite rank can have exactly the same covariance.

    Fix alpha in (0,1).  For each q, let G_q be standard stationary Gaussian
    with correlation (1+|t|)^(-alpha/q), and set

        X_q(t) = He_q(G_q(t)) / sqrt(q!).

    The Gaussian-Hermite identity gives Cov(X_q(0),X_q(t))=
    (1+|t|)^(-alpha), independently of q.  Taqqu's theorem then distinguishes
    the rank-q functional limits even though every second-order statistic is
    identical.  Tensor Gauss-Hermite quadrature checks the covariance identity;
    independent time quadrature checks the common integrated variance.
    """
    alpha = 0.6
    orders = tuple(range(1, 6))
    lag = 1.3
    target_covariance = (1.0 + lag) ** (-alpha)

    nodes, weights = np.polynomial.hermite_e.hermegauss(24)
    normal_weights = weights / math.sqrt(2.0 * math.pi)
    covariance_errors = []
    for order in orders:
        beta = alpha / order
        correlation = (1.0 + lag) ** (-beta)
        coefficients = np.zeros(order + 1)
        coefficients[-1] = 1.0
        first = np.polynomial.hermite_e.hermeval(nodes, coefficients)
        second = np.polynomial.hermite_e.hermeval(
            correlation * nodes[:, None]
            + math.sqrt(1.0 - correlation ** 2) * nodes[None, :],
            coefficients,
        )
        expectation = np.sum(
            normal_weights[:, None] * normal_weights[None, :]
            * first[:, None] * second
        ) / math.factorial(order)
        covariance_errors.append(abs(expectation - target_covariance))

    horizon = 257.0
    quadrature_variance = 2.0 * quad(
        lambda lag_value: (
            (horizon - lag_value) * (1.0 + lag_value) ** (-alpha)
        ),
        0.0,
        horizon,
        epsabs=2e-10,
        epsrel=2e-12,
        limit=500,
    )[0]
    exact_integral = (
        (horizon + 1.0)
        * math.expm1((1.0 - alpha) * math.log1p(horizon))
        / (1.0 - alpha)
        - math.expm1((2.0 - alpha) * math.log1p(horizon))
        / (2.0 - alpha)
    )
    exact_variance = 2.0 * exact_integral
    variance_identity_error = abs(quadrature_variance - exact_variance)

    large_horizon = 2.0 ** 24
    large_integral = (
        (large_horizon + 1.0)
        * math.expm1((1.0 - alpha) * math.log1p(large_horizon))
        / (1.0 - alpha)
        - math.expm1((2.0 - alpha) * math.log1p(large_horizon))
        / (2.0 - alpha)
    )
    leading_constant = 2.0 / ((1.0 - alpha) * (2.0 - alpha))
    asymptotic_ratio = (
        2.0 * large_integral
        / (leading_constant * large_horizon ** (2.0 - alpha))
    )
    hurst = 1.0 - alpha / 2.0

    assert max(covariance_errors) < 2e-12
    assert variance_identity_error < 2e-9
    assert abs(asymptotic_ratio - 1.0) < 0.002
    return (
        alpha,
        hurst,
        orders,
        max(covariance_errors),
        variance_identity_error,
        asymptotic_ratio,
    )


def verify_zero_gk_boundary():
    """Check the second critical index and finite-first-moment limit."""
    # At alpha=2 the exact formula is epsilon^2 log(1+epsilon^-2),
    # whose leading term is 2 epsilon^2 log(1/epsilon).
    quadrature_epsilons = 2.0 ** -np.arange(4, 13)
    critical_quadrature = np.array([
        integrated_variance(lambda t: antipersistent_covariance(t, 2.0), eps)
        for eps in quadrature_epsilons
    ])
    critical_closed = np.array([
        antipersistent_variance(eps, 2.0) for eps in quadrature_epsilons
    ])
    critical_identity_error = float(np.max(np.abs(
        critical_quadrature - critical_closed
    )))
    assert critical_identity_error < 2e-15

    critical_epsilons = 2.0 ** -np.arange(8, 25)
    critical_exact = np.array([
        antipersistent_variance(eps, 2.0) for eps in critical_epsilons
    ])
    critical_leading = (
        2.0 * critical_epsilons ** 2 * np.log(1.0 / critical_epsilons)
    )
    critical_ratio = critical_exact[-1] / critical_leading[-1]
    assert abs(critical_ratio - 1.0) < 1e-12

    # At alpha=3, the first covariance moment is finite and equals -1/2.
    # The exact variance is epsilon^2 T^2/(T^2+epsilon^2), so its leading
    # coefficient -2M is one.
    finite_alpha = 3.0
    finite_quadrature_epsilons = 2.0 ** -np.arange(4, 12)
    finite_quadrature = np.array([
        integrated_variance(
            lambda t: antipersistent_covariance(t, finite_alpha), eps
        )
        for eps in finite_quadrature_epsilons
    ])
    finite_closed = np.array([
        antipersistent_variance(eps, finite_alpha)
        for eps in finite_quadrature_epsilons
    ])
    finite_identity_error = float(np.max(np.abs(
        finite_quadrature - finite_closed
    )))
    assert finite_identity_error < 2e-15

    first_moment, moment_error = quad(
        lambda t: t * antipersistent_covariance(t, finite_alpha),
        0.0,
        np.inf,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=800,
    )
    assert moment_error < 2e-12
    assert abs(first_moment + 0.5) < 2e-13

    inverse_spectral_moment, spectral_error = quad(
        lambda omega: gamma_spectral_density(omega, finite_alpha)
        / omega ** 2,
        0.0,
        np.inf,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=800,
    )
    assert spectral_error < 2e-12
    assert abs(inverse_spectral_moment - 0.5) < 2e-13
    spectral_identity_error = abs(first_moment + inverse_spectral_moment)
    assert spectral_identity_error < 2e-13

    finite_epsilons = 2.0 ** -np.arange(8, 20)
    finite_exact = np.array([
        antipersistent_variance(eps, finite_alpha)
        for eps in finite_epsilons
    ])
    finite_ratio = finite_exact[-1] / finite_epsilons[-1] ** 2
    finite_order = measured_order(finite_exact)
    assert abs(finite_ratio - 1.0) < 2e-10
    assert abs(finite_order - 2.0) < 2e-10
    return (critical_ratio, critical_identity_error, first_moment,
            inverse_spectral_moment, spectral_identity_error,
            finite_order, finite_ratio, finite_identity_error)


def verify_saturated_spectral_criterion():
    """Check pointwise and Cesaro laws at the saturated spectral boundary.

    Write nu(d omega)=omega^-2 mu(d omega).  If nu is finite, the exact
    normalized variance is M_-2-Re(nu_hat(R)).  An exponential density makes
    the transform vanish, whereas one nonzero atom makes the coefficient
    oscillate forever despite having the same finite inverse moment.
    """
    scales = 2.0 ** np.arange(2, 21)

    # Let nu have density exp(-omega), so mu has density omega^2 exp(-omega).
    # Then M_-2=1 and Re nu_hat(R)=1/(1+R^2).
    continuous = scales ** 2 / (1.0 + scales ** 2)
    continuous_mass = quad(
        lambda omega: math.exp(-omega), 0.0, np.inf,
        epsabs=2e-13, epsrel=2e-13,
    )[0]
    continuous_exact = np.array([
        continuous_mass - quad(
            lambda omega: math.exp(-omega),
            0.0,
            np.inf,
            weight="cos",
            wvar=scale,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=800,
        )[0]
        for scale in scales[:8]
    ])
    continuous_identity_error = float(np.max(np.abs(
        continuous_exact - continuous[:8]
    )))
    assert continuous_identity_error < 2e-11
    assert abs(continuous[-1] - 1.0) < 1e-12

    # The stationary Gaussian process X_t=U cos(t)+V sin(t) has one-sided
    # spectral measure delta_1.  Its inverse moment is also one, but the
    # normalized variance 1-cos(R) has distinct subsequential limits.
    indices = np.arange(1, 9, dtype=float)
    resonant_scales = 2.0 * math.pi * indices
    antiresonant_scales = (2.0 * indices + 1.0) * math.pi
    resonant = 1.0 - np.cos(resonant_scales)
    antiresonant = 1.0 - np.cos(antiresonant_scales)
    assert np.max(np.abs(resonant)) < 2e-14
    assert np.max(np.abs(antiresonant - 2.0)) < 2e-14

    # Pointwise convergence can fail while a sharp averaged law survives.
    # For q(R)=M_-2-Re nu_hat(R), direct cosine averaging gives
    #
    #   average q -> M_-2,
    #   average (q-M_-2)^2 -> (1/2) sum_x nu({x})^2.
    #
    # Check both constants for two incommensurate atoms.  The finite-window
    # values below use the exact antiderivatives rather than a sampled grid.
    frequencies = np.array([1.0, math.sqrt(2.0)])
    masses = np.array([0.4, 0.25])
    inverse_moment = float(np.sum(masses))

    def exact_cesaro_values(window):
        transform_mean = float(np.sum(
            masses * np.sin(window * frequencies) / (window * frequencies)
        ))
        coefficient_mean = inverse_moment - transform_mean
        transform_square_mean = float(np.sum(
            masses ** 2 * (
                0.5
                + np.sin(2.0 * window * frequencies)
                / (4.0 * window * frequencies)
            )
        ))
        for first in range(len(frequencies)):
            for second in range(first + 1, len(frequencies)):
                omega_minus = frequencies[first] - frequencies[second]
                omega_plus = frequencies[first] + frequencies[second]
                transform_square_mean += (
                    masses[first] * masses[second] / window
                    * (
                        math.sin(window * omega_minus) / omega_minus
                        + math.sin(window * omega_plus) / omega_plus
                    )
                )
        return coefficient_mean, transform_square_mean

    # Independently integrate one finite window before using the closed forms
    # at a much larger window to certify the limits.
    check_window = 64.0
    check_exact = exact_cesaro_values(check_window)
    check_quadrature = (
        quad(
            lambda scale: inverse_moment - float(np.sum(
                masses * np.cos(scale * frequencies)
            )),
            0.0,
            check_window,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=800,
        )[0] / check_window,
        quad(
            lambda scale: float(np.sum(
                masses * np.cos(scale * frequencies)
            )) ** 2,
            0.0,
            check_window,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=800,
        )[0] / check_window,
    )
    cesaro_identity_error = max(
        abs(check_exact[0] - check_quadrature[0]),
        abs(check_exact[1] - check_quadrature[1]),
    )
    assert cesaro_identity_error < 2e-13

    window = 2.0 ** 20
    coefficient_mean, transform_square_mean = exact_cesaro_values(window)
    cesaro_mean_error = abs(coefficient_mean - inverse_moment)
    wiener_limit = 0.5 * float(np.sum(masses ** 2))
    cesaro_square_error = abs(transform_square_mean - wiener_limit)
    assert cesaro_mean_error < 1e-6
    assert cesaro_square_error < 1e-6

    return (continuous_identity_error, continuous[-1],
            float(np.max(np.abs(resonant))),
            float(np.max(np.abs(antiresonant - 2.0))),
            cesaro_mean_error, transform_square_mean,
            wiener_limit, cesaro_square_error, cesaro_identity_error)


def exponential_mixture_variance(epsilon, alpha, maturity=1.0):
    """Exact variance for a valid logarithmically modified covariance.

    The covariance is the positive OU mixture

        C(t) = integral_0^(e^-1) exp(-lambda t)
               lambda^(alpha-1) log(1/lambda) d lambda.

    Hence C(t) ~ Gamma(alpha) t^-alpha log(t).  The integration below
    first evaluates the finite-horizon variance of each exponential
    component exactly and then integrates over the mixing measure.
    """
    scale = maturity / epsilon
    upper = scale / math.e
    log_scale = math.log(scale)

    def exponential_kernel(x):
        # 1/x - (1-exp(-x))/x^2, evaluated without cancellation.
        if x < 1e-4:
            return 0.5 - x / 6.0 + x * x / 24.0 - x ** 3 / 120.0
        return 1.0 / x + math.expm1(-x) / (x * x)

    def integrand(x):
        return (x ** (alpha - 1.0) * (log_scale - math.log(x))
                * exponential_kernel(x))

    split = min(1.0, upper)
    value = quad(integrand, 0.0, split, epsabs=2e-12,
                 epsrel=2e-11, limit=800)[0]
    if upper > 1.0:
        value += quad(integrand, 1.0, upper, epsabs=2e-12,
                      epsrel=2e-11, limit=800)[0]
    return 2.0 * maturity ** 2 * (epsilon / maturity) ** alpha * value


def verify_regular_variation():
    """Check nonconstant slowly varying and critical logarithmic factors."""
    epsilons = 2.0 ** -np.arange(8, 19)

    alpha = 0.4
    long_values = np.array([
        exponential_mixture_variance(eps, alpha) for eps in epsilons
    ])
    long_leading = (
        2.0 * math.gamma(alpha) / ((1.0 - alpha) * (2.0 - alpha))
        * epsilons ** alpha * np.log(1.0 / epsilons)
    )
    long_ratio = long_values[-1] / long_leading[-1]
    assert abs(long_ratio - 1.0) < 0.03

    critical_values = np.array([
        exponential_mixture_variance(eps, 1.0) for eps in epsilons
    ])
    critical_leading = epsilons * np.log(1.0 / epsilons) ** 2
    critical_ratio = critical_values[-1] / critical_leading[-1]
    assert abs(critical_ratio - 1.0) < 0.07
    return long_ratio, critical_ratio


def verify_periodic_case():
    # With U uniform, phi(U+t)=cos(U+t), so the exact variance is
    # epsilon^2 (1-cos(T/epsilon)).  Choose a nonresonant sequence so the
    # oscillatory coefficient stays away from zero.
    epsilons = 1.0 / (2.0 * math.pi * np.arange(20, 80) + 1.1)
    exact = np.array([
        quad(lambda u: 1.0 - u, 0.0, 1.0, weight="cos", wvar=1.0 / eps,
             epsabs=2e-14, epsrel=2e-14, limit=400)[0]
        for eps in epsilons
    ])
    closed = epsilons ** 2 * (1.0 - np.cos(1.0 / epsilons))
    assert np.max(np.abs(exact - closed)) < 3e-12
    assert np.max(np.abs(exact / epsilons ** 2 - (1.0 - math.cos(1.1)))) < 2e-8
    return exact[-1] / epsilons[-1] ** 2


def main():
    short_order = verify_integrable_case()
    zero_atom = verify_zero_frequency_atom()
    all_scale = verify_all_scale_crossover()
    vector_rank = verify_vector_rank_crossover()
    long_04 = verify_long_memory_case(0.4)
    long_07 = verify_long_memory_case(0.7)
    critical_ratio = verify_critical_case()
    antipersistent_14 = verify_antipersistent_case(1.4)
    antipersistent_17 = verify_antipersistent_case(1.7)
    gaussian_fclt = verify_gaussian_fractional_limit()
    critical_gaussian = verify_critical_gaussian_boundary()
    second_chaos = verify_ergodic_second_chaos_counterexample()
    hermite_hierarchy = verify_covariance_matched_hermite_hierarchy()
    zero_gk_boundary = verify_zero_gk_boundary()
    saturated_spectral = verify_saturated_spectral_criterion()
    spectral_abelian = verify_spectral_abelian_theorem()
    measure_level = verify_measure_level_spectral_theorem()
    measure_level_critical = verify_measure_level_critical_theorem()
    joint_maturity = verify_joint_maturity_spectral_theorem()
    regular_variation = verify_regular_variation()
    periodic_coefficient = verify_periodic_case()

    print("Correlation-tail scaling certificate")
    print(f"integrable-correlation order: {short_order:.6f} (target 1)")
    print("zero-frequency atom identity error, omitted-atom error, "
          "normalized variance limit, residual Green--Kubo ratio: "
          f"{zero_atom[0]:.3e}, {zero_atom[1]:.3e}, "
          f"{zero_atom[2]:.9f}, {zero_atom[3]:.9f}")
    print("all-scale crossover quadrature/collapse errors: "
          f"{all_scale[0]:.3e}, {all_scale[1]:.3e}")
    print("all-scale frozen/finite/invariant normalized variances: "
          f"{all_scale[2]:.12f}, {all_scale[3]:.12f}, "
          f"{all_scale[4]:.12f}")
    print("all-scale frozen/invariant limit errors: "
          f"{all_scale[5]:.3e}, {all_scale[6]:.3e}")
    print("vector spectral identity error and ranks at pi, 2pi, 3pi: "
          f"{vector_rank[0]:.3e}, {vector_rank[1]}")
    print("vector eigenvalues at the rank-drop scale 2pi: "
          f"{vector_rank[2][0]:.3e}, {vector_rank[2][1]:.12f}")
    print(f"alpha=0.4 order, asymptotic ratio: {long_04[0]:.6f}, {long_04[1]:.6f}")
    print(f"alpha=0.7 order, asymptotic ratio: {long_07[0]:.6f}, {long_07[1]:.6f}")
    print(f"critical epsilon-log ratio: {critical_ratio:.6f}")
    print("zero-Green--Kubo antipersistent alpha=1.4 order, ratio, identity error: "
          f"{antipersistent_14[0]:.6f}, {antipersistent_14[1]:.6f}, "
          f"{antipersistent_14[2]:.3e}")
    print("zero-Green--Kubo antipersistent alpha=1.7 order, ratio, identity error: "
          f"{antipersistent_17[0]:.6f}, {antipersistent_17[1]:.6f}, "
          f"{antipersistent_17[2]:.3e}")
    for alpha, hurst, errors, terminal_power in gaussian_fclt:
        print("Gaussian fractional limit alpha/H, covariance errors, "
              f"terminal R=2^{terminal_power}: {alpha:.1f}, {hurst:.2f}, "
              f"{tuple(f'{error:.3e}' for error in errors)}")
    print("critical Gaussian spectral identity/covariance errors: "
          f"{critical_gaussian[0]:.3e}, "
          f"{tuple(f'{error:.6f}' for error in critical_gaussian[1])}")
    print("critical Gaussian delta and normalized increment variances at "
          f"R=2^{critical_gaussian[3]}: {critical_gaussian[4]:.2f}, "
          f"{tuple(f'{value:.6f}' for value in critical_gaussian[2])}")
    print("ergodic second-chaos beta, limiting/finite normalized third "
          "cumulants: "
          f"{second_chaos[0]:.1f}, {second_chaos[1]:.9f}, "
          f"{tuple(f'{value:.9f}' for value in second_chaos[2])}")
    print("ergodic second-chaos terminal variance/cumulant leading ratios: "
          f"{second_chaos[3]:.9f}, {second_chaos[4]:.9f}")
    print("covariance-matched Hermite hierarchy alpha/H and ranks: "
          f"{hermite_hierarchy[0]:.1f}, {hermite_hierarchy[1]:.1f}, "
          f"{hermite_hierarchy[2]}")
    print("Hermite hierarchy covariance/variance identity errors and "
          "leading ratio: "
          f"{hermite_hierarchy[3]:.3e}, {hermite_hierarchy[4]:.3e}, "
          f"{hermite_hierarchy[5]:.9f}")
    print("zero-Green--Kubo alpha=2 log-corrected ratio, identity error: "
          f"{zero_gk_boundary[0]:.12f}, {zero_gk_boundary[1]:.3e}")
    print("zero-Green--Kubo alpha=3 first moment, order, ratio, identity error: "
          f"{zero_gk_boundary[2]:.12f}, {zero_gk_boundary[5]:.9f}, "
          f"{zero_gk_boundary[6]:.12f}, {zero_gk_boundary[7]:.3e}")
    print("alpha=3 inverse spectral moment, spectral identity error: "
          f"{zero_gk_boundary[3]:.12f}, {zero_gk_boundary[4]:.3e}")
    print("saturated continuous-spectrum identity error and terminal ratio: "
          f"{saturated_spectral[0]:.3e}, {saturated_spectral[1]:.12f}")
    print("saturated pure-tone resonant and antiresonant errors: "
          f"{saturated_spectral[2]:.3e}, {saturated_spectral[3]:.3e}")
    print("saturated two-atom Cesaro mean error and mean-square limit: "
          f"{saturated_spectral[4]:.3e}, {saturated_spectral[5]:.12f} "
          f"(target {saturated_spectral[6]:.12f}, "
          f"error {saturated_spectral[7]:.3e}, "
          f"identity {saturated_spectral[8]:.3e})")
    print("spectral Abelian constant error, covariance/spectral identity error: "
          f"{spectral_abelian[0]:.3e}, {spectral_abelian[1]:.3e}")
    print("oscillatory-band alpha=1.7 order, ratio, remote-band fraction, "
          "sign changes: "
          f"{spectral_abelian[2]:.6f}, {spectral_abelian[3]:.6f}, "
          f"{spectral_abelian[4]:.3e}, {spectral_abelian[5]}")
    print("pure-point spectral mass ratio, summation error, order, ratio: "
          f"{measure_level[0]:.9f}, {measure_level[1]:.3e}, "
          f"{measure_level[2]:.6f}, {measure_level[3]:.6f}")
    print("pure-point alpha=2 mass ratio, summation error, log slope, ratio: "
          f"{measure_level_critical[0]:.9f}, "
          f"{measure_level_critical[1]:.3e}, "
          f"{measure_level_critical[2]:.6f}, "
          f"{measure_level_critical[3]:.6f}")
    print("joint maturity power/critical collapse errors and terminal ratios: "
          f"{joint_maturity[0]:.3e}, {joint_maturity[1]:.3e}, "
          f"{joint_maturity[4]:.6f}, {joint_maturity[5]:.12f}")
    print("joint maturity terminal-ratio spreads (power, critical): "
          f"{joint_maturity[2]:.3e}, {joint_maturity[3]:.3e}")
    print("OU-mixture regular-variation ratios: "
          f"power-log {regular_variation[0]:.6f}, "
          f"critical-log^2 {regular_variation[1]:.6f}")
    print(f"periodic epsilon^2 coefficient: {periodic_coefficient:.6f}")
    print("PASS")


if __name__ == "__main__":
    main()
