"""Numerical certificate for correlation-tail homogenization scales.

The proof is on tools/pages/mixing-scale.html.  This certificate independently
integrates the exact finite-horizon covariance and checks the Green--Kubo,
long-memory, critical, and zero-Green--Kubo regimes and the periodic
counterexample.
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
    long_04 = verify_long_memory_case(0.4)
    long_07 = verify_long_memory_case(0.7)
    critical_ratio = verify_critical_case()
    antipersistent_14 = verify_antipersistent_case(1.4)
    antipersistent_17 = verify_antipersistent_case(1.7)
    zero_gk_boundary = verify_zero_gk_boundary()
    spectral_abelian = verify_spectral_abelian_theorem()
    measure_level = verify_measure_level_spectral_theorem()
    measure_level_critical = verify_measure_level_critical_theorem()
    regular_variation = verify_regular_variation()
    periodic_coefficient = verify_periodic_case()

    print("Correlation-tail scaling certificate")
    print(f"integrable-correlation order: {short_order:.6f} (target 1)")
    print("zero-frequency atom identity error, omitted-atom error, "
          "normalized variance limit, residual Green--Kubo ratio: "
          f"{zero_atom[0]:.3e}, {zero_atom[1]:.3e}, "
          f"{zero_atom[2]:.9f}, {zero_atom[3]:.9f}")
    print(f"alpha=0.4 order, asymptotic ratio: {long_04[0]:.6f}, {long_04[1]:.6f}")
    print(f"alpha=0.7 order, asymptotic ratio: {long_07[0]:.6f}, {long_07[1]:.6f}")
    print(f"critical epsilon-log ratio: {critical_ratio:.6f}")
    print("zero-Green--Kubo antipersistent alpha=1.4 order, ratio, identity error: "
          f"{antipersistent_14[0]:.6f}, {antipersistent_14[1]:.6f}, "
          f"{antipersistent_14[2]:.3e}")
    print("zero-Green--Kubo antipersistent alpha=1.7 order, ratio, identity error: "
          f"{antipersistent_17[0]:.6f}, {antipersistent_17[1]:.6f}, "
          f"{antipersistent_17[2]:.3e}")
    print("zero-Green--Kubo alpha=2 log-corrected ratio, identity error: "
          f"{zero_gk_boundary[0]:.12f}, {zero_gk_boundary[1]:.3e}")
    print("zero-Green--Kubo alpha=3 first moment, order, ratio, identity error: "
          f"{zero_gk_boundary[2]:.12f}, {zero_gk_boundary[5]:.9f}, "
          f"{zero_gk_boundary[6]:.12f}, {zero_gk_boundary[7]:.3e}")
    print("alpha=3 inverse spectral moment, spectral identity error: "
          f"{zero_gk_boundary[3]:.12f}, {zero_gk_boundary[4]:.3e}")
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
    print("OU-mixture regular-variation ratios: "
          f"power-log {regular_variation[0]:.6f}, "
          f"critical-log^2 {regular_variation[1]:.6f}")
    print(f"periodic epsilon^2 coefficient: {periodic_coefficient:.6f}")
    print("PASS")


if __name__ == "__main__":
    main()
