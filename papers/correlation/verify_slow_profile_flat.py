#!/usr/bin/env python3
"""Certificate for the slowly varying flat-profile covariance expansion."""

import itertools
import math

import numpy as np

from scipy.optimize import minimize_scalar

from verify_gauge_flat import (
    NEGATIVE_WEIGHTS,
    POSITIVE_WEIGHTS,
    chain_data,
    contracted_gauge_rate,
)


GAUGE_CONSTANT = 4.60437900568596


def derived_gauge_constant(green_kubo):
    """Maximize the gauge dual quadratic form over its box vertices."""
    vertices = np.array(list(itertools.product(
        (-NEGATIVE_WEIGHTS[0], POSITIVE_WEIGHTS[0]),
        (-NEGATIVE_WEIGHTS[1], POSITIVE_WEIGHTS[1]),
    )))
    values = np.einsum("ni,ij,nj->n", vertices, green_kubo, vertices)
    return float(np.max(values))


def profile_saddle(speed, moment_power):
    """Contract h(t)=t^2 against the quadratic gauge rate exactly."""
    logarithmic_scale = 0.5 * math.log(speed)

    def objective(log_rescaled_radius):
        radius = math.exp(log_rescaled_radius)
        return (
            moment_power * (logarithmic_scale - log_rescaled_radius) ** 2
            + radius ** 2 / (2.0 * GAUGE_CONSTANT)
        )

    result = minimize_scalar(
        objective,
        bounds=(-4.0, math.log(max(2.0, logarithmic_scale)) + 3.0),
        method="bounded",
        options={"xatol": 1e-11},
    )
    return float(result.fun), math.exp(float(result.x))


def exact_saddle(speed, moment_power, scores, green_kubo):
    """Replace the quadratic rate by the exact finite-chain contraction."""
    profile_cost, rescaled_radius = profile_saddle(speed, moment_power)
    predicted_radius = rescaled_radius / math.sqrt(speed)

    def objective(log_radius):
        radius = math.exp(log_radius)
        rate, _ = contracted_gauge_rate(scores, green_kubo, radius)
        return moment_power * math.log(1.0 / radius) ** 2 + speed * rate

    result = minimize_scalar(
        objective,
        bounds=(math.log(predicted_radius) - 1.0,
                math.log(predicted_radius) + 1.0),
        method="bounded",
        options={"xatol": 1e-9},
    )
    return (
        float(result.fun),
        math.exp(float(result.x)),
        profile_cost,
        rescaled_radius,
    )


def second_order_cost(speed, moment_power):
    """The theorem's two-term moment expansion for h(t)=t^2."""
    logarithmic_scale = 0.5 * math.log(speed)
    derivative = 2.0 * logarithmic_scale
    return (
        moment_power * logarithmic_scale ** 2
        - 0.5 * moment_power * derivative
        * (math.log(moment_power * GAUGE_CONSTANT * derivative) - 1.0)
    )


def main():
    stationary, scores, green_kubo = chain_data()
    computed_gauge_constant = derived_gauge_constant(green_kubo)
    rows = []
    for speed in (1e4, 1e6, 1e8, 1e10):
        logarithmic_scale = 0.5 * math.log(speed)
        derivative = 2.0 * logarithmic_scale
        exact_second, radius, profile_second, rescaled = exact_saddle(
            speed, 2.0, scores, green_kubo
        )
        exact_first, _, profile_first, _ = exact_saddle(
            speed, 1.0, scores, green_kubo
        )
        expansion = second_order_cost(speed, 2.0)
        profile_separation = 2.0 * profile_first - profile_second
        exact_separation = 2.0 * exact_first - exact_second
        rows.append((
            speed,
            logarithmic_scale,
            rescaled,
            radius * math.sqrt(speed) / rescaled,
            profile_second / expansion,
            exact_second / profile_second,
            profile_separation / (derivative * math.log(2.0)),
            exact_separation / profile_separation,
        ))

    print("Slowly varying flat-profile covariance certificate")
    print(f"stationary law {stationary}")
    print("logarithmic profile h(t) = t^2")
    print(f"gauge constant {computed_gauge_constant:.9f}")
    for row in rows:
        speed, scale, saddle, radius_ratio, expansion, exact, separation, exact_sep = row
        print(
            f"speed {speed:.0e}: log scale {scale:.9f}, saddle {saddle:.9f}, "
            f"radius ratio {radius_ratio:.9f}, profile/expansion {expansion:.9f}, "
            f"exact/profile {exact:.9f}, separation/leading {separation:.9f}, "
            f"exact separation {exact_sep:.9f}"
        )

    ok = True
    ok &= abs(computed_gauge_constant - GAUGE_CONSTANT) < 1e-12
    ok &= all(rows[i][4] > rows[i + 1][4] > 1.0
              for i in range(len(rows) - 1))
    ok &= all(0.60 < rows[i][6] < rows[i + 1][6] < 1.0
              for i in range(len(rows) - 1))
    ok &= all(abs(row[5] - 1.0) < 6e-4 for row in rows)
    ok &= all(abs(row[7] - 1.0) < 6e-4 for row in rows)
    ok &= abs(rows[-1][3] - 1.0) < 5e-5
    ok &= abs(rows[-1][5] - 1.0) < 1e-6
    ok &= abs(rows[-1][7] - 1.0) < 1e-5
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
