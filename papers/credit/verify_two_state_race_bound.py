"""Certificate for the exact two-state first-default finite-rate bound."""

import numpy as np


ROUNDOFF_TOL = 128 * np.finfo(float).eps


def stationary(a, b):
    """Stationary row distribution of [[-a,a],[b,-b]]."""
    return np.array([b, a], dtype=float) / (a + b)


def resolvent_probability(a, b, cause_one, cause_two, speed):
    """Exact stationary probability that cause one arrives first."""
    generator = speed * np.array([[-a, a], [b, -b]], dtype=float)
    total = cause_one + cause_two
    return float(
        stationary(a, b)
        @ np.linalg.solve(np.diag(total) - generator, cause_one)
    )


def formula(a, b, cause_one, cause_two, speed):
    """Linear-fractional formula, first correction, and remainder bound."""
    d0, d1 = cause_one + cause_two
    slope_d = a * d1 + b * d0
    intercept_d = d0 * d1
    slope_n = a * cause_one[1] + b * cause_one[0]
    intercept_n = (
        b * d1 * cause_one[0] + a * d0 * cause_one[1]
    ) / (a + b)
    leading = slope_n / slope_d
    boundary = intercept_n - leading * intercept_d
    coefficient = boundary / slope_d
    exact = leading + boundary / (speed * slope_d + intercept_d)
    remainder = exact - leading - coefficient / speed
    signed_remainder = (
        -boundary * intercept_d
        / (speed * slope_d * (speed * slope_d + intercept_d))
    )
    bound = (
        abs(boundary) * intercept_d
        / (speed * speed * slope_d * slope_d)
    )
    return exact, leading, coefficient, remainder, signed_remainder, bound


def green_kubo_coefficient(a, b, cause_one, cause_two):
    """Generic group-inverse first-order coefficient for comparison."""
    generator = np.array([[-a, a], [b, -b]], dtype=float)
    pi = stationary(a, b)
    projection = np.outer(np.ones(2), pi)
    group_inverse = np.linalg.inv(generator + projection) - projection
    hazards = np.array([cause_one, cause_two])
    means = hazards @ pi
    centered = hazards - means[:, None]
    green_kubo = np.array([
        [
            -pi @ (centered[j] * (group_inverse @ centered[k]))
            for k in range(2)
        ]
        for j in range(2)
    ])
    mean_total = means.sum()
    return (
        (means[0] / mean_total) * green_kubo.sum()
        - green_kubo[:, 0].sum()
    ) / mean_total


def main():
    cases = (
        (1.7, 0.4, np.array([0.07, 0.01]), np.array([0.01, 0.05])),
        (0.3, 2.2, np.array([0.02, 0.11]), np.array([0.08, 0.03])),
        (3.0, 1.0, np.array([0.15, 0.04]), np.array([0.02, 0.09])),
    )
    maximum_formula_error = 0.0
    maximum_remainder_error = 0.0
    maximum_ratio = 0.0
    headline = None
    for a, b, cause_one, cause_two in cases:
        coefficient_gk = green_kubo_coefficient(
            a, b, cause_one, cause_two
        )
        for speed in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
            exact, leading, coefficient, remainder, signed, bound = formula(
                a, b, cause_one, cause_two, speed
            )
            direct = resolvent_probability(
                a, b, cause_one, cause_two, speed
            )
            formula_error = abs(exact - direct)
            remainder_error = abs(remainder - signed)
            maximum_formula_error = max(maximum_formula_error, formula_error)
            maximum_remainder_error = max(
                maximum_remainder_error, remainder_error
            )
            maximum_ratio = max(maximum_ratio, abs(remainder) / bound)
            assert formula_error <= ROUNDOFF_TOL
            assert remainder_error <= ROUNDOFF_TOL
            assert abs(coefficient - coefficient_gk) <= ROUNDOFF_TOL
            assert abs(remainder) <= bound * (1 + ROUNDOFF_TOL)
            assert np.sign(remainder) == -np.sign(coefficient)
            if a == 1.7 and b == 0.4 and speed == 1.0:
                headline = exact, leading, coefficient, remainder, bound

    exact, leading, coefficient, remainder, bound = headline
    print("two-state first-default finite-rate certificate")
    print(f"  speed 1 exact probability {exact:.10f}")
    print(f"  homogenized probability {leading:.10f}")
    print(f"  first-order coefficient {coefficient:.10e}")
    print(f"  signed remainder {remainder:.10e}")
    print(f"  global bound {bound:.10e}")
    print(f"  maximum formula error {maximum_formula_error:.3e}")
    print(f"  maximum signed-remainder error {maximum_remainder_error:.3e}")
    print(f"  maximum remainder/bound ratio {maximum_ratio:.10f}")
    print("PASS")


if __name__ == "__main__":
    main()
