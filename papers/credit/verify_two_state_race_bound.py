"""Certificate for exact two-state first-default finite-rate bounds."""

import numpy as np


ROUNDOFF_TOL = 128 * np.finfo(float).eps


def stationary(a, b):
    """Stationary row distribution of [[-a,a],[b,-b]]."""
    return np.array([b, a], dtype=float) / (a + b)


def resolvent_probabilities(a, b, hazards, speed):
    """Exact stationary first-arrival probabilities for all causes."""
    generator = speed * np.array([[-a, a], [b, -b]], dtype=float)
    total = hazards.sum(axis=0)
    return (
        stationary(a, b)
        @ np.linalg.solve(np.diag(total) - generator, hazards.T)
    )


def formula(a, b, hazards, speed):
    """Vector linear-fractional law, correction, and remainder bounds."""
    d0, d1 = hazards.sum(axis=0)
    slope_d = a * d1 + b * d0
    intercept_d = d0 * d1
    slope_n = a * hazards[:, 1] + b * hazards[:, 0]
    intercept_n = (
        b * d1 * hazards[:, 0] + a * d0 * hazards[:, 1]
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
        np.abs(boundary) * intercept_d
        / (speed * speed * slope_d * slope_d)
    )
    return exact, leading, coefficient, remainder, signed_remainder, bound


def green_kubo_coefficients(a, b, hazards):
    """Generic group-inverse first-order coefficient vector."""
    generator = np.array([[-a, a], [b, -b]], dtype=float)
    pi = stationary(a, b)
    projection = np.outer(np.ones(2), pi)
    group_inverse = np.linalg.inv(generator + projection) - projection
    means = hazards @ pi
    centered = hazards - means[:, None]
    green_kubo = np.array([
        [
            -pi @ (centered[j] * (group_inverse @ centered[k]))
            for k in range(len(hazards))
        ]
        for j in range(len(hazards))
    ])
    mean_total = means.sum()
    return (
        (means / mean_total) * green_kubo.sum()
        - green_kubo.sum(axis=0)
    ) / mean_total


def main():
    cases = (
        (1.7, 0.4, np.array([[0.07, 0.01], [0.01, 0.05]])),
        (0.3, 2.2, np.array([[0.02, 0.11], [0.08, 0.03]])),
        (3.0, 1.0, np.array([
            [0.15, 0.04], [0.02, 0.09], [0.06, 0.03],
        ])),
    )
    maximum_formula_error = 0.0
    maximum_remainder_error = 0.0
    maximum_ratio = 0.0
    headline = None
    maximum_simplex_error = 0.0
    for a, b, hazards in cases:
        coefficient_gk = green_kubo_coefficients(a, b, hazards)
        for speed in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
            exact, leading, coefficient, remainder, signed, bound = formula(
                a, b, hazards, speed
            )
            direct = resolvent_probabilities(a, b, hazards, speed)
            formula_error = np.max(np.abs(exact - direct))
            remainder_error = np.max(np.abs(remainder - signed))
            simplex_error = max(
                abs(exact.sum() - 1), abs(leading.sum() - 1),
                abs(coefficient.sum()), abs(remainder.sum()),
            )
            maximum_formula_error = max(maximum_formula_error, formula_error)
            maximum_remainder_error = max(
                maximum_remainder_error, remainder_error
            )
            maximum_simplex_error = max(maximum_simplex_error, simplex_error)
            active = bound > 0
            maximum_ratio = max(
                maximum_ratio, np.max(np.abs(remainder[active]) / bound[active])
            )
            assert formula_error <= ROUNDOFF_TOL
            assert remainder_error <= ROUNDOFF_TOL
            assert simplex_error <= ROUNDOFF_TOL
            assert np.max(np.abs(coefficient - coefficient_gk)) <= ROUNDOFF_TOL
            assert np.all(np.abs(remainder) <= bound * (1 + ROUNDOFF_TOL))
            assert np.all(np.sign(remainder[active]) == -np.sign(coefficient[active]))
            if a == 1.7 and b == 0.4 and speed == 1.0:
                headline = exact, leading, coefficient, remainder, bound

    exact, leading, coefficient, remainder, bound = headline
    print("two-state multi-cause first-default finite-rate certificate")
    print("  speed 1 exact probabilities "
          + " ".join(f"{value:.10f}" for value in exact))
    print("  homogenized probabilities "
          + " ".join(f"{value:.10f}" for value in leading))
    print("  first-order coefficients "
          + " ".join(f"{value:.10e}" for value in coefficient))
    print("  signed remainders "
          + " ".join(f"{value:.10e}" for value in remainder))
    print("  global bounds " + " ".join(f"{value:.10e}" for value in bound))
    print(f"  maximum formula error {maximum_formula_error:.3e}")
    print(f"  maximum signed-remainder error {maximum_remainder_error:.3e}")
    print(f"  maximum probability-simplex error {maximum_simplex_error:.3e}")
    print(f"  maximum remainder/bound ratio {maximum_ratio:.10f}")
    print("PASS")


if __name__ == "__main__":
    main()
