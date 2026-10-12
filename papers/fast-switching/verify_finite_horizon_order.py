"""Certificate for finite-horizon switching-speed order in two states.

For a stationary two-state chain and real nonconstant forcing, the complete
Feynman--Kac transform at every positive horizon decreases strictly and is
strictly convex in the common speed multiplier.  The certificate compares the
closed transform and its first two analytic speed derivatives with independent
block-matrix exponential calculations.  It also records why stationarity is
essential and gives a stationary nonreversible three-state counterexample to
extending strict convexity beyond the two-state theorem.
"""

import numpy as np
from scipy.linalg import expm, expm_frechet


EPS = np.finfo(float).eps
ROUND_TOL = 2048.0 * EPS
CAUCHY_TOL = 1.0e-8


def closed_value_and_derivatives(a, b, g1, g2, speed, maturity):
    """Stationary transform and its first two speed derivatives."""
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
    f = x * np.cosh(x) - np.sinh(x)
    u = r / discriminant
    curvature = (
        total**2 * multiplier * beta2 / discriminant**4
        * (x * f + u * (3.0 * f - x * x * np.sinh(x)))
    )
    return value, derivative, curvature


def general_matrix_value_and_derivatives(
    generator, forcing, speed, maturity, prior
):
    """Matrix exponential and first two speed derivatives in any dimension."""
    generator = np.asarray(generator, dtype=float)
    forcing = np.asarray(forcing, dtype=float)
    prior = np.asarray(prior, dtype=float)
    dimension = len(forcing)
    tilted = speed * generator + np.diag(forcing)
    semigroup, derivative = expm_frechet(
        maturity * tilted, maturity * generator
    )
    one = np.ones(dimension)
    block = np.zeros((3 * dimension, 3 * dimension))
    block_generator = maturity * generator
    block_tilted = maturity * tilted
    for index in range(3):
        start = dimension * index
        block[start:start + dimension, start:start + dimension] = block_tilted
    block[0:dimension, dimension:2 * dimension] = block_generator
    block[dimension:2 * dimension, 2 * dimension:3 * dimension] = block_generator
    second_derivative = 2.0 * expm(block)[
        0:dimension, 2 * dimension:3 * dimension
    ]
    return (
        float(prior @ semigroup @ one),
        float(prior @ derivative @ one),
        float(prior @ second_derivative @ one),
    )


def matrix_value_and_derivatives(a, b, g1, g2, speed, maturity, prior):
    """Independent two-state specialization of the matrix calculation."""
    generator = np.array([[-a, a], [b, -b]], dtype=float)
    return general_matrix_value_and_derivatives(
        generator, [g1, g2], speed, maturity, prior
    )


def cauchy_speed_derivatives(
    generator, forcing, speed, maturity, prior, radius=0.003, nodes=64
):
    """Independent Cauchy-contour derivatives of the entire speed transform."""
    angles = 2.0 * np.pi * np.arange(nodes) / nodes
    circle = speed + radius * np.exp(1j * angles)
    one = np.ones(len(forcing))
    values = np.array([
        prior @ expm(
            maturity * (point * generator + np.diag(forcing))
        ) @ one
        for point in circle
    ])
    first = np.mean(values * np.exp(-1j * angles)) / radius
    second = (
        2.0 * np.mean(values * np.exp(-2j * angles)) / radius**2
    )
    return first, second


def scaled_error(lhs, rhs):
    return abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))


def main():
    rng = np.random.default_rng(20261012)
    speeds = np.geomspace(0.02, 20.0, 13)
    max_value_error = 0.0
    max_derivative_error = 0.0
    max_curvature_error = 0.0
    max_envelope_violation = 0.0
    least_negative_derivative = np.inf
    least_positive_curvature = np.inf

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
            value, derivative, curvature = closed_value_and_derivatives(
                a, b, g1, g2, speed, maturity
            )
            (
                matrix_value,
                matrix_derivative,
                matrix_curvature,
            ) = matrix_value_and_derivatives(
                a, b, g1, g2, speed, maturity, prior
            )
            max_value_error = max(
                max_value_error, scaled_error(value, matrix_value)
            )
            max_derivative_error = max(
                max_derivative_error,
                scaled_error(derivative, matrix_derivative),
            )
            max_curvature_error = max(
                max_curvature_error,
                scaled_error(curvature, matrix_curvature),
            )
            max_envelope_violation = max(
                max_envelope_violation,
                max(0.0, fast_limit - value, value - slow_limit)
                / max(1.0, abs(fast_limit), abs(slow_limit)),
            )
            assert value < previous
            assert derivative < 0.0
            assert curvature > 0.0
            previous = value
            least_negative_derivative = min(
                least_negative_derivative, -derivative
            )
            least_positive_curvature = min(
                least_positive_curvature, curvature
            )

    assert max_value_error <= ROUND_TOL
    assert max_derivative_error <= ROUND_TOL
    assert max_curvature_error <= ROUND_TOL
    assert max_envelope_violation <= ROUND_TOL
    assert least_negative_derivative > 0.0
    assert least_positive_curvature > 0.0

    # A deterministic unequal-rate benchmark quoted on the page.
    a, b = 1.7, 0.4
    g1, g2 = 0.35, -0.25
    maturity = 1.3
    speed = 0.8
    prior = np.array([b, a]) / (a + b)
    (
        benchmark_value,
        benchmark_derivative,
        benchmark_curvature,
    ) = closed_value_and_derivatives(
        a, b, g1, g2, speed, maturity
    )
    benchmark_matrix = matrix_value_and_derivatives(
        a, b, g1, g2, speed, maturity, prior
    )
    assert scaled_error(benchmark_value, benchmark_matrix[0]) <= ROUND_TOL
    assert scaled_error(benchmark_derivative, benchmark_matrix[1]) <= ROUND_TOL
    assert scaled_error(benchmark_curvature, benchmark_matrix[2]) <= ROUND_TOL

    # Without a stationary start the speed derivative can have either sign.
    # Starting in the lower-forcing state makes faster switching beneficial.
    low_prior = np.array([1.0, 0.0])
    low_value, low_derivative, _ = matrix_value_and_derivatives(
        1.0, 1.0, 0.0, 1.0, 0.2, 1.0, low_prior
    )
    high_value, high_derivative, _ = matrix_value_and_derivatives(
        1.0, 1.0, 0.0, 1.0, 0.2, 1.0, 1.0 - low_prior
    )
    assert low_derivative > 0.0
    assert high_derivative < 0.0

    # The two-state convexity conclusion is not dimension-free.  This
    # irreducible three-state chain is stationary under (47, 3, 49) / 99 but
    # is nonreversible: pi_0 Q_01 != pi_1 Q_10.  Its stationary transform has
    # a strictly negative second speed derivative at the displayed point.
    three_generator = np.array([
        [-5.0, 4.0, 1.0],
        [13.0, -79.0, 66.0],
        [4.0, 1.0, -5.0],
    ])
    three_prior = np.array([47.0, 3.0, 49.0]) / 99.0
    three_forcing = np.array([-4.0, 3.0, 5.0])
    three_speed = 0.01
    three_maturity = 0.5
    assert np.max(abs(three_prior @ three_generator)) <= ROUND_TOL
    assert abs(three_prior[0] * three_generator[0, 1]
               - three_prior[1] * three_generator[1, 0]) > 1.0
    (
        three_value,
        three_derivative,
        three_curvature,
    ) = general_matrix_value_and_derivatives(
        three_generator,
        three_forcing,
        three_speed,
        three_maturity,
        three_prior,
    )
    cauchy_derivative, cauchy_curvature = cauchy_speed_derivatives(
        three_generator,
        three_forcing,
        three_speed,
        three_maturity,
        three_prior,
    )
    cauchy_derivative_error = abs(cauchy_derivative - three_derivative)
    cauchy_curvature_error = abs(cauchy_curvature - three_curvature)
    assert cauchy_derivative_error <= CAUCHY_TOL
    assert cauchy_curvature_error <= CAUCHY_TOL
    assert three_derivative < 0.0
    assert three_curvature < 0.0

    print("finite-horizon switching-speed order certificate")
    print("  random unequal-rate parameter sets", 64)
    print("  speeds per parameter set", len(speeds))
    print("  maximum normalized transform error", f"{max_value_error:.3e}")
    print("  maximum normalized derivative error", f"{max_derivative_error:.3e}")
    print("  maximum normalized curvature error", f"{max_curvature_error:.3e}")
    print("  maximum stationary-envelope violation", f"{max_envelope_violation:.3e}")
    print("  minimum magnitude of negative derivative", f"{least_negative_derivative:.3e}")
    print("  minimum positive curvature", f"{least_positive_curvature:.3e}")
    print("  unequal-rate benchmark value", f"{benchmark_value:.10f}")
    print("  unequal-rate benchmark speed derivative", f"{benchmark_derivative:.10f}")
    print("  unequal-rate benchmark speed curvature", f"{benchmark_curvature:.10f}")
    print("  low-state nonstationary value and derivative",
          f"{low_value:.10f}", f"{low_derivative:.10f}")
    print("  high-state nonstationary value and derivative",
          f"{high_value:.10f}", f"{high_derivative:.10f}")
    print("  three-state stationary value", f"{three_value:.10f}")
    print("  three-state speed derivative", f"{three_derivative:.10f}")
    print("  three-state speed curvature", f"{three_curvature:.10f}")
    print("  Cauchy derivative discrepancy", f"{cauchy_derivative_error:.3e}")
    print("  Cauchy curvature discrepancy", f"{cauchy_curvature_error:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
