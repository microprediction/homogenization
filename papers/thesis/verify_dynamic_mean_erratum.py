"""Certificate for the Chapter 3 dynamic-mean Vasicek erratum.

The 2001 thesis uses two incompatible conventions for the symmetric chain's
switching parameter and reverses the Vasicek remaining-maturity loading in its
dynamic-mean approximation.  This certificate checks the corrected kernel,
its closed form, its first-order agreement with the exact switched bond ODE,
and the two numerical examples printed in the thesis.  It also checks a
fixed-contrast, fast-switching refinement: the corrected dynamic mean captures
the regime log-price contrast through order lambda^-2, but omits the common
order-lambda^-1 Green--Kubo convexity term.
"""

from __future__ import annotations

import math

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
    ratio_errors = []
    common_shifts = []
    corrected_common_errors = []
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
        ratio_error = abs(
            log_factors[0] - log_factors[1] - 2.0 * dynamic_contrast
        )
        common_shift = 0.5 * float(np.sum(log_factors)) - log_averaged
        corrected_common_error = abs(
            common_shift - green_kubo_integral / (2.0 * rate)
        )
        ratio_errors.append(ratio_error)
        common_shifts.append(abs(common_shift))
        corrected_common_errors.append(corrected_common_error)
        fast_rows.append(
            (rate, ratio_error, common_shift, corrected_common_error)
        )
    ratio_orders = np.log2(
        np.asarray(ratio_errors[:-1]) / ratio_errors[1:]
    )
    common_orders = np.log2(
        np.asarray(common_shifts[:-1]) / common_shifts[1:]
    )
    corrected_common_orders = np.log2(
        np.asarray(corrected_common_errors[:-1])
        / corrected_common_errors[1:]
    )
    assert ratio_orders[-1] > 2.98
    assert common_orders[-1] > 0.995
    assert corrected_common_orders[-1] > 2.0

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
    print("lambda  contrast error   common log shift   corrected common error")
    for rate, ratio_error, common_shift, corrected_common_error in fast_rows:
        print(
            f" {rate:5.0f}   {ratio_error:13.7e}"
            f"   {common_shift:16.7e}   {corrected_common_error:20.7e}"
        )
    print(
        "last fast-switching orders: contrast "
        f"{ratio_orders[-1]:.6f}, omitted common term "
        f"{common_orders[-1]:.6f}, corrected common residual "
        f"{corrected_common_orders[-1]:.6f}"
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
        "orders, and both thesis examples"
    )


if __name__ == "__main__":
    main()
