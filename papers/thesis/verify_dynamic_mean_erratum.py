"""Certificate for the Chapter 3 dynamic-mean Vasicek erratum.

The 2001 thesis uses two incompatible conventions for the symmetric chain's
switching parameter and reverses the Vasicek remaining-maturity loading in its
dynamic-mean approximation.  This certificate checks the corrected kernel,
its closed form, its first-order agreement with the exact switched bond ODE,
and the two numerical examples printed in the thesis.  It also checks a
fixed-contrast, fast-switching refinement: the corrected dynamic mean captures
the regime log-price contrast through order lambda^-2, but omits the common
order-lambda^-1 Green--Kubo convexity term.  The endpoint coefficients of both
next remainders are also checked.  Finally, it checks a nonasymptotic error
bound that is uniform over every maturity and every finite contrast-to-
switching ratio, and permits the contrast to vary with the switching rate.
An unequal-rate extension makes the cancellation mechanism precise: the
linear log-ratio error is generally quadratic in contrast/switching speed,
and improves to cubic exactly when the two transition rates coincide.
For constant forcing, a signed fourth-order equilibrium expansion identifies
the sharp quadratic and cubic coefficients behind those rates.
"""

from __future__ import annotations

import math

import mpmath as mp
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm


def vasicek_loading(time: float, kappa: float) -> float:
    """Return B(time)=(1-exp(-kappa*time))/kappa stably."""
    return -math.expm1(-kappa * time) / kappa


def exponential_difference_ratio(
    first_rate: float, second_rate: float, horizon: float
) -> float:
    """Return (exp(-second*T)-exp(-first*T))/(first-second)."""
    difference = first_rate - second_rate
    if difference == 0.0:
        return horizon * math.exp(-second_rate * horizon)
    return (
        math.exp(-second_rate * horizon)
        * -math.expm1(-difference * horizon)
        / difference
    )


def memory_loading(
    decay: float, kappa: float, horizon: float, power: int
) -> float:
    """Closed form for int exp(-decay*s) B(T-s)^power ds."""
    if decay <= 0.0 or kappa <= 0.0 or horizon < 0.0 or power < 0:
        raise ValueError("rates must be positive; horizon and power nonnegative")
    return sum(
        (-1.0) ** index
        * math.comb(power, index)
        * exponential_difference_ratio(
            decay, index * kappa, horizon
        )
        for index in range(power + 1)
    ) / kappa**power


def memory_loading_quadrature(
    decay: float, kappa: float, horizon: float, power: int
) -> float:
    """Independent adaptive-quadrature check of ``memory_loading``."""
    return quad(
        lambda elapsed: math.exp(-decay * elapsed)
        * vasicek_loading(horizon - elapsed, kappa) ** power,
        0.0,
        horizon,
        epsabs=2e-14,
        epsrel=2e-14,
    )[0]


def response_coefficient(
    switching_rate: float,
    kappa: float,
    horizon: float,
    theta_contrast: float,
    variance_contrast: float,
) -> float:
    """First antisymmetric log-bond coefficient for per-state jump rate lambda."""
    decay = 2.0 * switching_rate
    return (
        -kappa
        * theta_contrast
        * memory_loading(decay, kappa, horizon, 1)
        + 0.5
        * variance_contrast
        * memory_loading(decay, kappa, horizon, 2)
    )


def uniform_log_ratio_bound(
    switching_rate: float, forcing_bound: float
) -> tuple[float, float]:
    """Return eta and an all-ratio, maturity-uniform log-ratio bound.

    If |q(t)| <= forcing_bound and eta=forcing_bound/(2*switching_rate),
    the exact Riccati contrast omega stays in [-rho,rho], where

        rho = tanh(asinh(2*eta)/2).

    Its linearization D then obeys

        sup_T |2*atanh(omega(T))-2*D(T)| <= returned bound.

    No smallness assumption on eta is required.  The bound is asymptotic to
    (8/3)*eta^3 as eta tends to zero.
    """
    if switching_rate <= 0.0 or forcing_bound < 0.0:
        raise ValueError("switching rate must be positive and bound nonnegative")
    eta = forcing_bound / (2.0 * switching_rate)
    rho = math.tanh(0.5 * math.asinh(2.0 * eta))
    if rho < 1e-3:
        rho_squared = rho * rho
        atanh_remainder = rho**3 * (
            1.0 / 3.0
            + rho_squared * (
                1.0 / 5.0
                + rho_squared * (1.0 / 7.0 + rho_squared / 9.0)
            )
        )
    else:
        atanh_remainder = math.atanh(rho) - rho
    bound = (
        2.0 * eta * rho**2
        + 2.0 * atanh_remainder
    )
    return eta, bound


def riccati_log_ratio_error(
    switching_rate: float,
    horizon: float,
    forcing,
) -> float:
    """Solve the exact and linear contrast equations and return their error."""

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        value = forcing(time)
        omega, linear = state
        return np.array(
            [
                value * (1.0 - omega**2) - 2.0 * switching_rate * omega,
                value - 2.0 * switching_rate * linear,
            ]
        )

    solution = solve_ivp(
        rhs,
        (0.0, horizon),
        np.zeros(2),
        method="DOP853",
        rtol=2e-13,
        atol=2e-15,
    )
    assert solution.success
    omega, linear = solution.y[:, -1]
    assert abs(omega) < 1.0
    return abs(2.0 * math.atanh(omega) - 2.0 * linear)


def positive_ratio_equilibrium(
    rate_12: float, rate_21: float, forcing: float
) -> float:
    """Positive equilibrium y=exp(ell) for constant log-ratio forcing.

    The exact log ratio solves

        ell' = q + rate_12(exp(-ell)-1) - rate_21(exp(ell)-1).

    Hence y is the positive root of

        rate_21*y^2 - (q-rate_12+rate_21)*y - rate_12 = 0.

    The alternate quadratic formula avoids cancellation when the linear
    coefficient is negative.
    """
    if rate_12 <= 0.0 or rate_21 <= 0.0:
        raise ValueError("transition rates must be positive")
    linear = forcing - rate_12 + rate_21
    radical = math.hypot(linear, 2.0 * math.sqrt(rate_12 * rate_21))
    if linear >= 0.0:
        return (linear + radical) / (2.0 * rate_21)
    return 2.0 * rate_12 / (radical - linear)


def constant_forcing_expansion_coefficients(
    rate_12: float, rate_21: float
) -> tuple[float, float, float]:
    """Coefficients of the constant-forcing equilibrium error.

    Put ``s=rate_12+rate_21``, ``x=q/s`` and
    ``alpha=(rate_12-rate_21)/s``.  The analytic equilibrium branch satisfies

        ell_*(q)-x = c2*x**2 + c3*x**3 + c4*x**4 + O(x**5).

    This function returns ``(c2,c3,c4)``.  In particular, ``c2=alpha/2``;
    symmetry removes it and leaves ``c3=-1/6``.
    """
    if rate_12 <= 0.0 or rate_21 <= 0.0:
        raise ValueError("transition rates must be positive")
    asymmetry = (rate_12 - rate_21) / (rate_12 + rate_21)
    return (
        asymmetry / 2.0,
        asymmetry**2 / 2.0 - 1.0 / 6.0,
        asymmetry * (5.0 * asymmetry**2 - 3.0) / 8.0,
    )


def unequal_rate_log_ratio_bound(
    rate_12: float, rate_21: float, forcing_bound: float
) -> tuple[float, float, float]:
    """Return invariant radius, rate asymmetry, and a uniform error bound.

    If ``|q(t)| <= forcing_bound``, the exact log ratio ``ell`` is compared
    with ``D' = q-(rate_12+rate_21)D``.  The interval obtained from the two
    extremal constant forcings is invariant.  On that interval,

        |ell-D| <= delta*(cosh(L)-1) + sinh(L)-L,

    where ``delta=|rate_12-rate_21|/(rate_12+rate_21)``.  The first term is
    quadratic for unequal rates; it vanishes exactly for symmetric rates.
    """
    if rate_12 <= 0.0 or rate_21 <= 0.0 or forcing_bound < 0.0:
        raise ValueError("transition rates must be positive and bound nonnegative")
    upper = math.log(
        positive_ratio_equilibrium(rate_12, rate_21, forcing_bound)
    )
    lower = math.log(
        positive_ratio_equilibrium(rate_12, rate_21, -forcing_bound)
    )
    assert lower <= 1e-14 and upper >= -1e-14
    radius = max(upper, -lower)
    asymmetry = abs(rate_12 - rate_21) / (rate_12 + rate_21)
    cosh_remainder = 2.0 * math.sinh(0.5 * radius) ** 2
    if radius < 1e-3:
        radius_squared = radius * radius
        sinh_remainder = radius**3 * (
            1.0 / 6.0
            + radius_squared * (
                1.0 / 120.0
                + radius_squared * (1.0 / 5040.0 + radius_squared / 362880.0)
            )
        )
    else:
        sinh_remainder = math.sinh(radius) - radius
    bound = (
        asymmetry * cosh_remainder
        + sinh_remainder
    )
    return radius, asymmetry, bound


def unequal_rate_log_ratio_error(
    rate_12: float,
    rate_21: float,
    horizon: float,
    forcing,
) -> float:
    """Solve the exact unequal-rate log ratio and its linearization."""
    total_rate = rate_12 + rate_21

    def rhs(time: float, state: np.ndarray) -> np.ndarray:
        value = forcing(time)
        log_ratio, linear = state
        return np.array(
            [
                value
                + rate_12 * math.expm1(-log_ratio)
                - rate_21 * math.expm1(log_ratio),
                value - total_rate * linear,
            ]
        )

    solution = solve_ivp(
        rhs,
        (0.0, horizon),
        np.zeros(2),
        method="DOP853",
        rtol=2e-13,
        atol=2e-15,
    )
    assert solution.success
    return abs(solution.y[0, -1] - solution.y[1, -1])


def switched_bond_factors(
    switching_rate: float,
    kappa: float,
    horizon: float,
    theta_mean: float,
    theta_contrast: float,
    variance_mean: float,
    variance_contrast: float,
    contrast_scale: float = 1.0,
) -> np.ndarray:
    """Exact regime factors a_i(T) in P_i=exp(-B(T)x)a_i(T)."""
    theta = theta_mean + contrast_scale * np.array(
        [theta_contrast, -theta_contrast]
    )
    variance = variance_mean + contrast_scale * np.array(
        [variance_contrast, -variance_contrast]
    )
    if np.min(variance) < 0.0:
        raise ValueError("both regime variances must be nonnegative")

    def rhs(time: float, factors: np.ndarray) -> np.ndarray:
        loading = vasicek_loading(time, kappa)
        forcing = (
            -kappa * theta * loading
            + 0.5 * variance * loading**2
        )
        switching = switching_rate * np.array(
            [factors[1] - factors[0], factors[0] - factors[1]]
        )
        return forcing * factors + switching

    solution = solve_ivp(
        rhs,
        (0.0, horizon),
        np.ones(2),
        method="DOP853",
        rtol=2e-13,
        atol=2e-15,
    )
    assert solution.success
    return solution.y[:, -1]


def averaged_bond_factor(
    kappa: float,
    horizon: float,
    theta_mean: float,
    variance_mean: float,
) -> float:
    """Vasicek factor for the regime-averaged parameters."""
    log_factor = quad(
        lambda time: -kappa
        * theta_mean
        * vasicek_loading(time, kappa)
        + 0.5
        * variance_mean
        * vasicek_loading(time, kappa) ** 2,
        0.0,
        horizon,
        epsabs=2e-14,
        epsrel=2e-14,
    )[0]
    return math.exp(log_factor)


def printed_dynamic_mean_loading(
    switching_rate: float, kappa: float, horizon: float
) -> float:
    """The thesis's printed int exp(-lambda*s) B(s) ds."""
    return quad(
        lambda elapsed: math.exp(-switching_rate * elapsed)
        * vasicek_loading(elapsed, kappa),
        0.0,
        horizon,
        epsabs=2e-14,
        epsrel=2e-14,
    )[0]


def main() -> None:
    # Under the site's convention lambda is the jump rate out of each state,
    # so the centered state decays at 2 lambda, not lambda.  Check directly
    # against the two-state matrix exponential.
    for switching_rate in (0.3, 1.0, 5.0):
        generator = switching_rate * np.array([[-1.0, 1.0], [1.0, -1.0]])
        contrast = np.array([1.0, -1.0])
        for elapsed in (0.0, 0.1, 0.7, 2.0):
            propagated = expm(elapsed * generator) @ contrast
            expected = math.exp(-2.0 * switching_rate * elapsed) * contrast
            assert np.max(np.abs(propagated - expected)) < 2e-14

    # Check the closed form, including removable resonances decay=j*kappa.
    rng = np.random.default_rng(20260927)
    largest_loading_error = 0.0
    cases = [(1.0, 1.0, 2.3), (2.0, 1.0, 2.3)]
    cases += [
        (
            float(10 ** rng.uniform(-1.0, 1.0)),
            float(10 ** rng.uniform(-1.0, 1.0)),
            float(10 ** rng.uniform(-1.0, 1.0)),
        )
        for _ in range(200)
    ]
    for decay, kappa, horizon in cases:
        for power in (1, 2):
            closed = memory_loading(decay, kappa, horizon, power)
            numerical = memory_loading_quadrature(
                decay, kappa, horizon, power
            )
            largest_loading_error = max(
                largest_loading_error, abs(closed - numerical)
            )
    assert largest_loading_error < 3e-12

    # Scale both regime contrasts by delta.  Symmetry makes half the log-price
    # ratio odd in delta; the corrected dynamic-mean kernel is its exact linear
    # coefficient, so the remainder must be cubic.
    switching_rate, kappa, horizon = 1.3, 0.7, 4.0
    theta_mean, theta_contrast = 0.18, 0.09
    variance_mean, variance_contrast = 0.025, 0.007
    coefficient = response_coefficient(
        switching_rate,
        kappa,
        horizon,
        theta_contrast,
        variance_contrast,
    )
    scales = np.array([1.0, 0.5, 0.25, 0.125, 0.0625])
    residuals = []
    for scale in scales:
        factors = switched_bond_factors(
            switching_rate,
            kappa,
            horizon,
            theta_mean,
            theta_contrast,
            variance_mean,
            variance_contrast,
            scale,
        )
        log_contrast = 0.5 * math.log(factors[0] / factors[1])
        residuals.append(abs(log_contrast - scale * coefficient))
    residuals = np.asarray(residuals)
    orders = np.log2(residuals[:-1] / residuals[1:])
    assert np.min(orders[-2:]) > 2.999

    # Keep the regime contrast fixed and increase the switching rate.  If
    # D_lambda is the corrected dynamic-mean contrast, the exact amplitude
    # ratio satisfies log(a_1/a_2)=2 D_lambda+O(lambda^-3).  The geometric
    # mean has a separate common O(lambda^-1) Green--Kubo correction that the
    # deterministic dynamic-mean price omits.
    kappa, horizon = 1.3, 4.0
    theta = np.array([0.18, 0.04])
    volatility = np.array([0.16, 0.05])
    theta_mean = float(np.mean(theta))
    theta_contrast = float((theta[0] - theta[1]) / 2.0)
    variance = volatility**2
    variance_mean = float(np.mean(variance))
    variance_contrast = float((variance[0] - variance[1]) / 2.0)

    def contrast_growth(time: float) -> float:
        loading = vasicek_loading(time, kappa)
        return (
            -kappa * theta_contrast * loading
            + 0.5 * variance_contrast * loading**2
        )

    endpoint_growth = contrast_growth(horizon)
    ratio_cubic_coefficient = -(endpoint_growth**3) / 6.0
    common_quadratic_coefficient = -(endpoint_growth**2) / 4.0

    log_averaged = math.log(
        averaged_bond_factor(kappa, horizon, theta_mean, variance_mean)
    )
    green_kubo_integral = quad(
        lambda time: contrast_growth(time) ** 2,
        0.0,
        horizon,
        epsabs=2e-14,
        epsrel=2e-14,
    )[0]
    fast_rates = np.array([4.0, 8.0, 16.0, 32.0, 64.0, 128.0])
    fast_rows = []
    ratio_residuals = []
    common_shifts = []
    corrected_common_residuals = []
    corrected_common_errors = []
    refined_ratio_errors = []
    refined_common_errors = []
    for rate in fast_rates:
        factors = switched_bond_factors(
            rate,
            kappa,
            horizon,
            theta_mean,
            theta_contrast,
            variance_mean,
            variance_contrast,
        )
        log_factors = np.log(factors)
        dynamic_contrast = response_coefficient(
            rate,
            kappa,
            horizon,
            theta_contrast,
            variance_contrast,
        )
        ratio_residual = (
            log_factors[0] - log_factors[1] - 2.0 * dynamic_contrast
        )
        common_shift = 0.5 * float(np.sum(log_factors)) - log_averaged
        corrected_common_residual = (
            common_shift - green_kubo_integral / (2.0 * rate)
        )
        refined_ratio_error = abs(
            ratio_residual - ratio_cubic_coefficient / rate**3
        )
        refined_common_error = abs(
            corrected_common_residual
            - common_quadratic_coefficient / rate**2
        )
        ratio_residuals.append(ratio_residual)
        common_shifts.append(abs(common_shift))
        corrected_common_residuals.append(corrected_common_residual)
        corrected_common_errors.append(abs(corrected_common_residual))
        refined_ratio_errors.append(refined_ratio_error)
        refined_common_errors.append(refined_common_error)
        fast_rows.append(
            (
                rate,
                ratio_residual,
                common_shift,
                corrected_common_residual,
            )
        )
    ratio_orders = np.log2(
        np.abs(np.asarray(ratio_residuals[:-1]) / ratio_residuals[1:])
    )
    common_orders = np.log2(
        np.asarray(common_shifts[:-1]) / common_shifts[1:]
    )
    corrected_common_orders = np.log2(
        np.asarray(corrected_common_errors[:-1])
        / corrected_common_errors[1:]
    )
    refined_ratio_orders = np.log2(
        np.asarray(refined_ratio_errors[:-1]) / refined_ratio_errors[1:]
    )
    refined_common_orders = np.log2(
        np.asarray(refined_common_errors[:-1]) / refined_common_errors[1:]
    )
    assert ratio_orders[-1] > 2.98
    assert common_orders[-1] > 0.995
    assert corrected_common_orders[-1] > 2.0
    # The final ratio datum is at the ODE solver's double-precision floor;
    # use the preceding 32-to-64 doubling for its fourth-order check.
    assert refined_ratio_orders[-2] > 3.9
    assert refined_common_orders[-1] > 2.98
    assert abs(
        fast_rates[-1] ** 3 * ratio_residuals[-1]
        - ratio_cubic_coefficient
    ) < 4e-8
    assert abs(
        fast_rates[-1] ** 2
        * corrected_common_residuals[-1]
        - common_quadratic_coefficient
    ) < 4e-6

    # The Riccati invariant-interval proof gives a genuinely nonasymptotic
    # bound, uniform in T and valid at every finite contrast-to-switching
    # ratio.  Exercise it first on constant forcing all the way to the
    # equilibrium limit, then on bounded time-dependent forcings.  The latter
    # are normalized by an analytic envelope, not by a sampled maximum.
    utilization = []
    small_eta = 2.0 ** -16
    small_eta_ratio = (
        uniform_log_ratio_bound(1.0, 2.0 * small_eta)[1]
        / small_eta**3
    )
    assert abs(small_eta_ratio - 8.0 / 3.0) < 2e-9
    eta_grid = np.geomspace(0.01, 100.0, 100)
    for eta in eta_grid:
        equilibrium = math.tanh(0.5 * math.asinh(2.0 * eta))
        equilibrium_error = abs(2.0 * math.atanh(equilibrium) - 2.0 * eta)
        _, bound = uniform_log_ratio_bound(1.0, 2.0 * eta)
        assert equilibrium_error <= bound
        utilization.append(equilibrium_error / bound)

    rng = np.random.default_rng(20261001)
    random_cases = 160
    for _ in range(random_cases):
        rate = float(10 ** rng.uniform(-0.5, 1.0))
        eta = float(10 ** rng.uniform(-2.0, 2.0))
        forcing_bound = 2.0 * rate * eta
        coefficients = rng.normal(size=3)
        envelope = float(np.sum(np.abs(coefficients)))
        frequencies = rate * 10 ** rng.uniform(-1.0, 1.0, size=2)

        def bounded_forcing(time: float) -> float:
            raw = (
                coefficients[0]
                + coefficients[1] * math.sin(frequencies[0] * time)
                + coefficients[2] * math.cos(frequencies[1] * time)
            )
            return forcing_bound * raw / envelope

        horizon = float(10 ** rng.uniform(-2.0, 1.0) / rate)
        error = riccati_log_ratio_error(rate, horizon, bounded_forcing)
        _, bound = uniform_log_ratio_bound(rate, forcing_bound)
        assert error <= bound * (1.0 + 2e-11)
        utilization.append(error / bound)

    # A joint contrast/switching sequence: ||q_lambda|| grows like lambda^.6,
    # while the theorem predicts a uniform O(lambda^-1.2) error.  A fixed
    # smooth forcing exposes that rate without relying on the Vasicek endpoint.
    joint_rates = 2.0 ** np.arange(4, 11)
    joint_errors = []
    joint_bounds = []
    for rate in joint_rates:
        scale = rate**0.6
        forcing_bound = 0.9 * scale

        def joint_forcing(time: float, scale: float = scale) -> float:
            return scale * (0.6 + 0.3 * math.sin(time))

        joint_errors.append(
            riccati_log_ratio_error(rate, 1.7, joint_forcing)
        )
        joint_bounds.append(
            uniform_log_ratio_bound(rate, forcing_bound)[1]
        )
    joint_errors = np.asarray(joint_errors)
    joint_bounds = np.asarray(joint_bounds)
    assert np.all(joint_errors <= joint_bounds)
    joint_orders = np.log2(joint_errors[:-1] / joint_errors[1:])
    assert joint_orders[-1] > 1.18

    # Unequal transition rates break the odd symmetry responsible for the
    # cubic error.  Work directly with ell=log(a_1/a_2).  The extremal
    # constant-forcing equilibria provide an invariant interval, and the
    # nonlinear remainder splits exactly into an asymmetric quadratic term
    # plus the symmetric cubic term.
    unequal_utilization = []
    rng = np.random.default_rng(20261004)
    unequal_random_cases = 200
    for _ in range(unequal_random_cases):
        rate_12 = float(10 ** rng.uniform(-1.0, 1.5))
        rate_21 = float(10 ** rng.uniform(-1.0, 1.5))
        total_rate = rate_12 + rate_21
        eta = float(10 ** rng.uniform(-2.0, 1.0))
        forcing_bound = eta * total_rate
        coefficients = rng.normal(size=3)
        envelope = float(np.sum(np.abs(coefficients)))
        frequencies = total_rate * 10 ** rng.uniform(-1.0, 1.0, size=2)

        def unequal_forcing(time: float) -> float:
            raw = (
                coefficients[0]
                + coefficients[1] * math.sin(frequencies[0] * time)
                + coefficients[2] * math.cos(frequencies[1] * time)
            )
            return forcing_bound * raw / envelope

        horizon = float(10 ** rng.uniform(-2.0, 1.0) / total_rate)
        error = unequal_rate_log_ratio_error(
            rate_12, rate_21, horizon, unequal_forcing
        )
        radius, _, bound = unequal_rate_log_ratio_bound(
            rate_12, rate_21, forcing_bound
        )
        assert radius >= 0.0 and error <= bound * (1.0 + 3e-11)
        if bound > 0.0:
            unequal_utilization.append(error / bound)

    # Constant forcing reaches the exact equilibrium and exposes the sharp
    # change of order.  Hold the rate proportions fixed while multiplying
    # both rates by lambda.
    forcing_level = 0.8
    asymmetry_scales = 2.0 ** np.arange(2, 11)
    asymmetric_errors = []
    symmetric_errors = []
    asymmetric_bounds = []
    symmetric_bounds = []
    for scale in asymmetry_scales:
        asymmetric_rates = (0.35 * scale, 1.65 * scale)
        symmetric_rates = (scale, scale)
        for rates, errors_out, bounds_out in (
            (asymmetric_rates, asymmetric_errors, asymmetric_bounds),
            (symmetric_rates, symmetric_errors, symmetric_bounds),
        ):
            if rates[0] == rates[1]:
                eta_here = forcing_level / sum(rates)
                # Stable value of eta-asinh(eta) on this small-eta grid.
                error = eta_here**3 * (
                    1.0 / 6.0
                    - eta_here**2 * (
                        3.0 / 40.0
                        - eta_here**2 * (
                            5.0 / 112.0
                            - eta_here**2 * (
                                35.0 / 1152.0 - eta_here**2 * 63.0 / 2816.0
                            )
                        )
                    )
                )
            else:
                equilibrium = math.log(
                    positive_ratio_equilibrium(*rates, forcing_level)
                )
                linear_equilibrium = forcing_level / sum(rates)
                error = abs(equilibrium - linear_equilibrium)
            errors_out.append(error)
            bounds_out.append(
                unequal_rate_log_ratio_bound(*rates, forcing_level)[2]
            )
    asymmetric_errors = np.asarray(asymmetric_errors)
    symmetric_errors = np.asarray(symmetric_errors)
    asymmetric_bounds = np.asarray(asymmetric_bounds)
    symmetric_bounds = np.asarray(symmetric_bounds)
    assert np.all(asymmetric_errors <= asymmetric_bounds * (1.0 + 2e-13))
    assert np.all(symmetric_errors <= symmetric_bounds * (1.0 + 1e-10))
    asymmetric_orders = np.log2(
        asymmetric_errors[:-1] / asymmetric_errors[1:]
    )
    symmetric_orders = np.log2(symmetric_errors[:-1] / symmetric_errors[1:])
    assert asymmetric_orders[-1] > 1.998
    assert symmetric_orders[-1] > 2.998

    final_scale = asymmetry_scales[-1]
    final_eta = forcing_level / (2.0 * final_scale)
    final_asymmetry = abs(0.35 - 1.65) / (0.35 + 1.65)
    scaled_asymmetric_error = asymmetric_errors[-1] / final_eta**2
    assert abs(scaled_asymmetric_error - final_asymmetry / 2.0) < 4e-4

    # The constant-forcing equilibrium admits a signed analytic expansion.
    # Verify all displayed coefficients independently against the exact
    # quadratic root at 80-digit precision, including both rate orderings.
    expansion_asymmetries = (-0.9, -0.65, -0.2, 0.0, 0.2, 0.65, 0.9)
    expansion_orders = []
    leading_coefficient_errors = []
    with mp.workdps(80):
        expansion_grid = [mp.mpf("0.08") / 2**index for index in range(5)]
        for asymmetry in expansion_asymmetries:
            asymmetry_mp = mp.mpf(str(asymmetry))
            rate_12_mp = (1 + asymmetry_mp) / 2
            rate_21_mp = (1 - asymmetry_mp) / 2
            c2, c3, c4 = constant_forcing_expansion_coefficients(
                float(rate_12_mp), float(rate_21_mp)
            )
            residuals = []
            for scaled_forcing in expansion_grid:
                linear = scaled_forcing - rate_12_mp + rate_21_mp
                equilibrium_ratio = (
                    linear
                    + mp.sqrt(
                        linear**2 + 4 * rate_12_mp * rate_21_mp
                    )
                ) / (2 * rate_21_mp)
                equilibrium = mp.log(equilibrium_ratio)
                expansion = (
                    scaled_forcing
                    + c2 * scaled_forcing**2
                    + c3 * scaled_forcing**3
                    + c4 * scaled_forcing**4
                )
                residuals.append(abs(equilibrium - expansion))

            orders = [
                mp.log(residuals[index] / residuals[index + 1], 2)
                for index in range(len(residuals) - 1)
            ]
            expansion_orders.append(float(orders[-1]))

            smallest = expansion_grid[-1]
            linear = smallest - rate_12_mp + rate_21_mp
            exact = mp.log(
                (
                    linear
                    + mp.sqrt(linear**2 + 4 * rate_12_mp * rate_21_mp)
                )
                / (2 * rate_21_mp)
            )
            if asymmetry != 0.0:
                observed = (exact - smallest) / smallest**2
                target = asymmetry_mp / 2
            else:
                observed = (exact - smallest) / smallest**3
                target = -mp.mpf(1) / 6
            leading_coefficient_errors.append(float(abs(observed - target)))

    assert min(expansion_orders) > 4.98
    assert max(leading_coefficient_errors) < 2.1e-3

    # At equal rates the general log-ratio theorem has no quadratic term.
    # Its radius and bound reduce to asinh(M/S) and sinh(L)-L exactly.
    symmetric_radius, symmetric_asymmetry, symmetric_general_bound = (
        unequal_rate_log_ratio_bound(3.0, 3.0, 1.7)
    )
    expected_radius = math.asinh(1.7 / 6.0)
    assert symmetric_asymmetry == 0.0
    assert abs(symmetric_radius - expected_radius) < 2e-15
    assert abs(
        symmetric_general_bound
        - (math.sinh(expected_radius) - expected_radius)
    ) < 2e-15

    # Reproduce the two thesis examples.  The reported dynamic-mean values use
    # exp(-lambda*s)B(s); the correction uses exp(-2lambda*s)B(T-s).
    examples = (
        (5.0, 2.0, 15.0, 0.010020),
        (1.0, 25.0, 5.0, 0.0516),
    )
    rows = []
    for rate, speed, maturity, thesis_simulation in examples:
        theta_mean, theta_contrast = 0.25, 0.10
        variance_mean = 1e-8
        printed_exponent = (
            speed
            * theta_contrast
            * printed_dynamic_mean_loading(rate, speed, maturity)
        )
        corrected_exponent = abs(
            response_coefficient(
                rate, speed, maturity, theta_contrast, 0.0
            )
        )
        printed = math.sinh(printed_exponent)
        corrected = math.sinh(corrected_exponent)
        factors = switched_bond_factors(
            rate,
            speed,
            maturity,
            theta_mean,
            theta_contrast,
            variance_mean,
            0.0,
        )
        averaged = averaged_bond_factor(
            speed, maturity, theta_mean, variance_mean
        )
        exact = abs(factors[1] - factors[0]) / (2.0 * averaged)
        assert abs(exact - corrected) < abs(exact - printed)
        assert abs(exact - thesis_simulation) < 7e-4
        rows.append((rate, speed, maturity, printed, corrected, exact))

    print("Chapter 3 dynamic-mean Vasicek erratum")
    print(f"maximum closed-form/quadrature error: {largest_loading_error:.3e}")
    print(f"linear response coefficient: {coefficient:.10f}")
    print(f"last two contrast-remainder orders: {orders[-2]:.6f}, {orders[-1]:.6f}")
    print("fixed contrast, fast switching")
    print("lambda  contrast residual   common log shift   corrected common residual")
    for rate, ratio_residual, common_shift, corrected_common_residual in fast_rows:
        print(
            f" {rate:5.0f}   {ratio_residual:16.7e}"
            f"   {common_shift:16.7e}   {corrected_common_residual:25.7e}"
        )
    print(
        "last fast-switching orders: contrast "
        f"{ratio_orders[-1]:.6f}, omitted common term "
        f"{common_orders[-1]:.6f}, corrected common residual "
        f"{corrected_common_orders[-1]:.6f}"
    )
    print(
        "sharp endpoint limits: lambda^3 contrast residual "
        f"{fast_rates[-1] ** 3 * ratio_residuals[-1]:.9e} -> "
        f"{ratio_cubic_coefficient:.9e}; lambda^2 corrected common residual "
        f"{fast_rates[-1] ** 2 * corrected_common_residuals[-1]:.9e} -> "
        f"{common_quadratic_coefficient:.9e}"
    )
    print(
        "orders after endpoint corrections: contrast "
        f"{refined_ratio_orders[-2]:.6f} (32-to-64), common "
        f"{refined_common_orders[-1]:.6f} (64-to-128)"
    )
    print(
        "maturity-uniform bound: "
        f"{len(eta_grid) + random_cases} constant/random cases, maximum utilization "
        f"{max(utilization):.6f}; joint contrast/switching last order "
        f"{joint_orders[-1]:.6f}"
    )
    print(
        "uniform-bound small-eta coefficient: "
        f"{small_eta_ratio:.9f} versus {8.0 / 3.0:.9f}"
    )
    print(
        "unequal-rate bound: "
        f"{unequal_random_cases} random cases, maximum utilization "
        f"{max(unequal_utilization):.6f}; last constant-forcing orders "
        f"{asymmetric_orders[-1]:.6f} unequal and "
        f"{symmetric_orders[-1]:.6f} symmetric"
    )
    print(
        "unequal-rate quadratic coefficient: "
        f"{scaled_asymmetric_error:.9f} versus "
        f"{final_asymmetry / 2.0:.9f}"
    )
    print(
        "constant-forcing signed expansion: minimum fourth-order-residual "
        f"order {min(expansion_orders):.6f}; maximum leading-coefficient "
        f"error {max(leading_coefficient_errors):.3e}"
    )
    print("lambda  kappa   T       printed       corrected      exact switched")
    for rate, speed, maturity, printed, corrected, exact in rows:
        print(
            f" {rate:4.1f}   {speed:5.1f}  {maturity:4.0f}"
            f"    {printed:11.8f}   {corrected:11.8f}   {exact:13.8f}"
        )
    print(
        "PASS: per-state rate convention, remaining-maturity kernel, closed "
        "forms, cubic contrast remainder, fixed-contrast fast-switching "
        "orders and endpoint coefficients, maturity-uniform finite-rate "
        "all-ratio bound, unequal-rate quadratic/cubic transition and signed "
        "equilibrium coefficients, joint "
        "contrast/switching rate, and both thesis examples"
    )


if __name__ == "__main__":
    main()
