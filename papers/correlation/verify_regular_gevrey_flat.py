#!/usr/bin/env python3
"""Certificate for regularly varying Gevrey-gauge covariance."""

import math

from scipy.optimize import brentq, minimize_scalar

from verify_gauge_flat import chain_data, contracted_gauge_rate


GAUGE_CONSTANT = 4.60437900568596
PROFILE_INDEX = 2.0


def profile(value):
    """A non-power regularly varying penalty of index two."""
    return value ** 2 * math.log1p(value)


def inverse_scale(speed):
    """Solve B^2 F(B) = speed."""
    return float(brentq(
        lambda value: value ** 2 * profile(value) - speed,
        1e-8,
        speed ** 0.25,
        xtol=1e-12,
    ))


def limiting_constant(moment_power):
    """Regular-variation contraction constant for one moment."""
    q = PROFILE_INDEX
    return (
        (q + 2.0) / 2.0
        * moment_power ** (2.0 / (q + 2.0))
        * (q * GAUGE_CONSTANT) ** (-q / (q + 2.0))
    )


def profile_contraction(scale, moment_power):
    """Retain the finite-scale slowly varying profile, but use the quadratic rate."""
    denominator = profile(scale)

    def objective(log_multiplier):
        multiplier = math.exp(log_multiplier)
        return (
            moment_power * profile(scale * multiplier) / denominator
            + multiplier ** -2 / (2.0 * GAUGE_CONSTANT)
        )

    result = minimize_scalar(
        objective,
        bounds=(-2.0, 2.0),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.fun), math.exp(float(result.x))


def exact_saddle(speed, moment_power, scores, green_kubo):
    """Use the exact finite-chain rate at the profile-corrected saddle."""
    scale = inverse_scale(speed)
    _, multiplier = profile_contraction(scale, moment_power)
    predicted_radius = 1.0 / (scale * multiplier)

    def objective(log_radius):
        radius = math.exp(log_radius)
        rate, _ = contracted_gauge_rate(scores, green_kubo, radius)
        return moment_power * profile(1.0 / radius) + speed * rate

    result = minimize_scalar(
        objective,
        bounds=(math.log(predicted_radius) - 1.5,
                math.log(predicted_radius) + 1.5),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return math.exp(float(result.x)), float(result.fun)


def main():
    stationary, scores, green_kubo = chain_data()
    first_constant = limiting_constant(1.0)
    second_constant = limiting_constant(2.0)
    separation_constant = 2.0 * first_constant - second_constant

    rows = []
    for speed in (1e6, 1e10, 1e14, 1e18):
        scale = inverse_scale(speed)
        rate_scale = profile(scale)
        first_profile, _ = profile_contraction(scale, 1.0)
        second_profile, _ = profile_contraction(scale, 2.0)
        second_radius, second_cost = exact_saddle(
            speed, 2.0, scores, green_kubo
        )
        _, first_cost = exact_saddle(speed, 1.0, scores, green_kubo)
        profile_separation = 2.0 * first_profile - second_profile
        rows.append((
            speed,
            scale,
            second_radius,
            second_profile / second_constant,
            second_cost / (second_profile * rate_scale),
            profile_separation / separation_constant,
            (2.0 * first_cost - second_cost)
            / (profile_separation * rate_scale),
        ))

    print("Regularly varying Gevrey-gauge covariance certificate")
    print(f"stationary law {stationary}")
    print("profile F(x) = x^2 log(1+x)")
    print(f"regular-variation index {PROFILE_INDEX:.6f}")
    print(f"gauge constant {GAUGE_CONSTANT:.9f}")
    print(f"first-moment limiting constant {first_constant:.9f}")
    print(f"variance limiting constant {second_constant:.9f}")
    print(f"mean-separation limiting constant {separation_constant:.9f}")
    for row in rows:
        speed, scale, radius, profile_rate, exact_rate, profile_sep, exact_sep = row
        print(
            f"speed {speed:.0e}: inverse {scale:.9f}, radius {radius:.9f}, "
            f"profile variance {profile_rate:.9f}, exact variance {exact_rate:.9f}, "
            f"profile separation {profile_sep:.9f}, exact separation {exact_sep:.9f}"
        )

    ok = True
    ok &= separation_constant > 0.0
    ok &= all(0.88 < rows[i][3] < rows[i + 1][3] < 1.0
              for i in range(len(rows) - 1))
    ok &= all(0.97 < rows[i][5] < rows[i + 1][5] < 1.0
              for i in range(len(rows) - 1))
    ok &= all(abs(row[4] - 1.0) < 1.5e-3 for row in rows)
    ok &= all(abs(row[6] - 1.0) < 8e-4 for row in rows)
    ok &= abs(rows[-1][4] - 1.0) < 2e-5
    ok &= abs(rows[-1][6] - 1.0) < 1e-5
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
