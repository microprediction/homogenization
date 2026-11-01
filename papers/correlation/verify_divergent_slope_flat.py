#!/usr/bin/env python3
"""Certificate for the general divergent-slope covariance expansion."""

import math

from scipy.optimize import minimize_scalar

from verify_flat_map import log_profile_moment


LOG_SCALES = (4.0, 8.0, 16.0, 32.0, 64.0)


def profile(value):
    """Index-one profile h(t)=t log(1+t)."""
    return value * math.log1p(value)


def derivative(value):
    """Derivative of the index-one profile."""
    return math.log1p(value) + value / (1.0 + value)


def curvature(value):
    """Second derivative of the index-one profile."""
    return 1.0 / (1.0 + value) + 1.0 / (1.0 + value) ** 2


def quadratic_saddle(speed, power):
    """Exact Gaussian-rate contraction for one moment."""
    logarithmic_scale = 0.5 * math.log(speed)

    def objective(log_rescaled_radius):
        return (
            power * profile(logarithmic_scale - log_rescaled_radius)
            + 0.5 * math.exp(2.0 * log_rescaled_radius)
        )

    result = minimize_scalar(
        objective,
        bounds=(-5.0, logarithmic_scale),
        method="bounded",
        options={"xatol": 1e-12},
    )
    radius_log = 1.0 + logarithmic_scale - float(result.x)
    return radius_log, float(result.fun)


def exact_log_moment(speed, power):
    """Log moment under the exact atom-plus-Bessel telegraph law."""
    return log_profile_moment(
        speed,
        power,
        lambda radius_log: profile(radius_log - 1.0),
        lambda radius_log: derivative(radius_log - 1.0),
        quadratic_saddle,
    )


def expansion(logarithmic_scale, power):
    """The theorem's two-term moment cost."""
    local_derivative = derivative(logarithmic_scale)
    return (
        power * profile(logarithmic_scale)
        - 0.5 * power * local_derivative
        * (math.log(power * local_derivative) - 1.0)
    )


def row(logarithmic_scale):
    """Return the profile, exact-law and asymptotic comparisons."""
    speed = math.exp(2.0 * logarithmic_scale)
    _, first_profile_cost = quadratic_saddle(speed, 1.0)
    _, second_profile_cost = quadratic_saddle(speed, 2.0)
    first_cost = -exact_log_moment(speed, 1.0)
    second_cost = -exact_log_moment(speed, 2.0)
    mean_separation = 2.0 * first_cost - second_cost
    variance_cost = second_cost - math.log1p(-math.exp(-mean_separation))
    predicted_second = expansion(logarithmic_scale, 2.0)
    local_derivative = derivative(logarithmic_scale)
    flatness = (
        curvature(logarithmic_scale) / local_derivative
        * math.log(local_derivative) ** 2
    )
    return (
        logarithmic_scale,
        local_derivative,
        flatness,
        second_profile_cost / predicted_second,
        second_cost / second_profile_cost,
        variance_cost / predicted_second,
        mean_separation / (local_derivative * math.log(2.0)),
        (2.0 * first_profile_cost - second_profile_cost)
        / (local_derivative * math.log(2.0)),
    )


def main():
    rows = [row(scale) for scale in LOG_SCALES]

    print("General divergent-slope covariance certificate")
    print("profile h(t) = t log(1+t), telegraph Green-Kubo variance 1")
    for values in rows:
        (
            scale, local_derivative, flatness, profile_ratio,
            exact_ratio, variance_ratio, separation, profile_separation,
        ) = values
        print(
            f"log scale {scale:4.0f}: derivative {local_derivative:.9f}, "
            f"flatness {flatness:.9f}, profile/expansion {profile_ratio:.9f}, "
            f"exact/profile {exact_ratio:.9f}, variance/expansion "
            f"{variance_ratio:.9f}, separation/leading {separation:.9f}, "
            f"profile separation {profile_separation:.9f}"
        )

    ok = True
    ok &= all(
        rows[index][2] > rows[index + 1][2] > 0.0
        for index in range(len(rows) - 1)
    )
    ok &= all(
        1.0 < rows[index + 1][3] < rows[index][3] < 1.02
        for index in range(len(rows) - 1)
    )
    ok &= all(
        0.97 < rows[index][4] < rows[index + 1][4] < 1.0
        for index in range(len(rows) - 1)
    )
    ok &= all(abs(values[5] - 1.0) < 0.02 for values in rows)
    ok &= all(
        0.75 < rows[index][6] < rows[index + 1][6] < 1.0
        for index in range(len(rows) - 1)
    )
    ok &= rows[-1][3] < 1.00005
    ok &= rows[-1][4] > 0.9993
    ok &= rows[-1][5] > 0.9994
    ok &= rows[-1][6] > 0.90
    ok &= rows[-1][7] > 0.996
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
