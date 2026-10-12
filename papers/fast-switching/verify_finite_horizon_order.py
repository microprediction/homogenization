"""Certificate for finite-horizon switching-speed order in two states.

For a stationary two-state chain and real nonconstant forcing, the complete
Feynman--Kac transform at every positive horizon decreases strictly with the
common speed multiplier.  The certificate compares the closed transform and
its analytic speed derivative with independent matrix-exponential and Frechet
derivative calculations.  It also records why stationarity is essential.
"""

import numpy as np
from scipy.linalg import expm, expm_frechet


EPS = np.finfo(float).eps
ROUND_TOL = 2048.0 * EPS


def closed_value_and_derivative(a, b, g1, g2, speed, maturity):
    """Stationary transform and its derivative in the speed multiplier."""
    total = a + b
    theta = (a - b) / total
    delta = g1 - g2
    mean = (b * g1 + a * g2) / total
    r = speed * total - theta * delta
    beta2 = (1.0 - theta * theta) * delta * delta
    discriminant = np.sqrt(r * r + beta2)
    half_time = maturity / 2.0
    x = half_time * discriminant
    multiplier = np.exp(mean * maturity - half_time * r)
    value = multiplier * (
        np.cosh(x) + r * np.sinh(x) / discriminant
    )
    derivative = (
        total * multiplier * beta2 / discriminant**3
        * (np.sinh(x) - x * np.cosh(x))
    )
    return value, derivative


def matrix_value_and_derivative(a, b, g1, g2, speed, maturity, prior):
    """Independent matrix exponential and Frechet speed derivative."""
    generator = np.array([[-a, a], [b, -b]], dtype=float)
    tilted = speed * generator + np.diag([g1, g2])
    semigroup, derivative = expm_frechet(
        maturity * tilted, maturity * generator
    )
    one = np.ones(2)
    return float(prior @ semigroup @ one), float(prior @ derivative @ one)


def scaled_error(lhs, rhs):
    return abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))


def main():
    rng = np.random.default_rng(20261012)
    speeds = np.geomspace(0.02, 20.0, 13)
    max_value_error = 0.0
    max_derivative_error = 0.0
    max_envelope_violation = 0.0
    least_negative_derivative = np.inf

    for _ in range(64):
        a, b = rng.uniform(0.2, 3.0, size=2)
        g1, g2 = rng.uniform(-1.0, 1.0, size=2)
        if abs(g1 - g2) < 0.05:
            g2 += 0.2
        maturity = rng.uniform(0.05, 3.0)
        prior = np.array([b, a]) / (a + b)
        mean = float(prior @ np.array([g1, g2]))
        slow_limit = float(prior @ np.exp(maturity * np.array([g1, g2])))
        fast_limit = float(np.exp(maturity * mean))
        previous = np.inf

        for speed in speeds:
            value, derivative = closed_value_and_derivative(
                a, b, g1, g2, speed, maturity
            )
            matrix_value, matrix_derivative = matrix_value_and_derivative(
                a, b, g1, g2, speed, maturity, prior
            )
            max_value_error = max(
                max_value_error, scaled_error(value, matrix_value)
            )
            max_derivative_error = max(
                max_derivative_error,
                scaled_error(derivative, matrix_derivative),
            )
            max_envelope_violation = max(
                max_envelope_violation,
                max(0.0, fast_limit - value, value - slow_limit)
                / max(1.0, abs(fast_limit), abs(slow_limit)),
            )
            assert value < previous
            assert derivative < 0.0
            previous = value
            least_negative_derivative = min(
                least_negative_derivative, -derivative
            )

    assert max_value_error <= ROUND_TOL
    assert max_derivative_error <= ROUND_TOL
    assert max_envelope_violation <= ROUND_TOL
    assert least_negative_derivative > 0.0

    # A deterministic unequal-rate benchmark quoted on the page.
    a, b = 1.7, 0.4
    g1, g2 = 0.35, -0.25
    maturity = 1.3
    speed = 0.8
    prior = np.array([b, a]) / (a + b)
    benchmark_value, benchmark_derivative = closed_value_and_derivative(
        a, b, g1, g2, speed, maturity
    )
    benchmark_matrix = matrix_value_and_derivative(
        a, b, g1, g2, speed, maturity, prior
    )
    assert scaled_error(benchmark_value, benchmark_matrix[0]) <= ROUND_TOL
    assert scaled_error(benchmark_derivative, benchmark_matrix[1]) <= ROUND_TOL

    # Without a stationary start the speed derivative can have either sign.
    # Starting in the lower-forcing state makes faster switching beneficial.
    low_prior = np.array([1.0, 0.0])
    low_value, low_derivative = matrix_value_and_derivative(
        1.0, 1.0, 0.0, 1.0, 0.2, 1.0, low_prior
    )
    high_value, high_derivative = matrix_value_and_derivative(
        1.0, 1.0, 0.0, 1.0, 0.2, 1.0, 1.0 - low_prior
    )
    assert low_derivative > 0.0
    assert high_derivative < 0.0

    print("finite-horizon switching-speed order certificate")
    print("  random unequal-rate parameter sets", 64)
    print("  speeds per parameter set", len(speeds))
    print("  maximum normalized transform error", f"{max_value_error:.3e}")
    print("  maximum normalized derivative error", f"{max_derivative_error:.3e}")
    print("  maximum stationary-envelope violation", f"{max_envelope_violation:.3e}")
    print("  minimum magnitude of negative derivative", f"{least_negative_derivative:.3e}")
    print("  unequal-rate benchmark value", f"{benchmark_value:.10f}")
    print("  unequal-rate benchmark speed derivative", f"{benchmark_derivative:.10f}")
    print("  low-state nonstationary value and derivative",
          f"{low_value:.10f}", f"{low_derivative:.10f}")
    print("  high-state nonstationary value and derivative",
          f"{high_value:.10f}", f"{high_derivative:.10f}")
    print("PASS")


if __name__ == "__main__":
    main()
