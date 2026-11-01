#!/usr/bin/env python3
"""Certificate for the tangent-gauge Green--Kubo flat constant."""

import math

import numpy as np
from scipy.optimize import minimize_scalar

from verify_gauge_flat import (
    FLAT_SCALE,
    SIGMA,
    chain_data,
    gauge,
    rate,
)


PERTURBATION = 0.8
GAUGE_CONSTANT = 4.60437900568596


def local_scale(value):
    """Nonhomogeneous scale tangent to the asymmetric gauge at zero."""
    return gauge(value) * (1.0 + PERTURBATION * np.linalg.norm(value))


def boundary_radius(direction, level):
    """Solve local_scale(t * direction) = level for unit direction."""
    gauge_direction = gauge(direction)
    discriminant = math.sqrt(
        1.0 + 4.0 * PERTURBATION * level / gauge_direction
    )
    return 2.0 * level / (gauge_direction * (1.0 + discriminant))


def contracted_local_rate(scores, green_kubo, level):
    """Contract the exact rate over a nonhomogeneous level set."""
    def angular_rate(angle):
        direction = np.array([math.cos(angle), math.sin(angle)])
        radius = boundary_radius(direction, level)
        return rate(scores, green_kubo, radius * direction)

    grid = np.linspace(0.0, 2.0 * math.pi, 65)[:-1]
    values = [angular_rate(angle) for angle in grid]
    index = int(np.argmin(values))
    half_width = 2.0 * math.pi / 64.0
    result = minimize_scalar(
        angular_rate,
        bounds=(grid[index] - half_width, grid[index] + half_width),
        method="bounded",
        options={"xatol": 1e-10},
    )
    return float(result.fun), float(result.x % (2.0 * math.pi))


def saddle(log_speed, power, scores, green_kubo):
    beta = SIGMA - 1.0

    def objective(profile):
        level = FLAT_SCALE ** beta * profile ** (-beta)
        value, _ = contracted_local_rate(scores, green_kubo, level)
        return power * math.exp(profile - log_speed) + value

    result = minimize_scalar(
        objective,
        bounds=(10.0, log_speed),
        method="bounded",
        options={"xatol": 1e-7},
    )
    return float(result.x), float(result.fun)


def main():
    stationary, scores, green_kubo = chain_data()
    beta = SIGMA - 1.0
    leading = FLAT_SCALE ** (2.0 * beta) / (2.0 * GAUGE_CONSTANT)

    maximizing_polar = np.array([-2.0, -1.4])
    tangent_direction = green_kubo @ maximizing_polar
    tangent_angle = float(
        math.atan2(tangent_direction[1], tangent_direction[0])
        % (2.0 * math.pi)
    )

    small_level = 0.005
    exact_rate, angle = contracted_local_rate(
        scores, green_kubo, small_level
    )
    quadratic = small_level ** 2 / (2.0 * GAUGE_CONSTANT)
    rate_ratio = exact_rate / quadratic
    angle_shift = abs(angle - tangent_angle)

    coordinate_map = np.array([[1.2, -0.4], [0.3, 1.1]])
    probe = np.array([-0.02, 0.01])
    transported_error = abs(
        local_scale(probe)
        - local_scale(np.linalg.solve(coordinate_map, coordinate_map @ probe))
    )

    rows = []
    for log_speed in (128.0, 512.0, 2048.0, 8192.0):
        _, second = saddle(log_speed, 2.0, scores, green_kubo)
        _, first = saddle(log_speed, 1.0, scores, green_kubo)
        scale = log_speed ** (2.0 * beta)
        variance_rate = second * scale / leading
        mean_separation = (2.0 * first - second) * scale / leading
        rows.append((log_speed, variance_rate, mean_separation))

    print("Tangent-gauge Green--Kubo flat-map certificate")
    print(f"stationary law {stationary}")
    print(f"perturbation coefficient {PERTURBATION:.6f}")
    print(f"tangent gauge constant {GAUGE_CONSTANT:.9f}")
    print(f"predicted logarithmic constant {leading:.9f}")
    print(f"transported scale discrepancy {transported_error:.3e}")
    print(f"tangent minimizing angle {tangent_angle:.9f}")
    print(f"finite-level minimizing angle {angle:.9f}")
    print(f"finite-level angle shift {angle_shift:.9f}")
    print(f"small-level normalized rate {rate_ratio:.9f}")
    for log_speed, variance_rate, mean_separation in rows:
        print(
            f"log speed {log_speed:7.0f}: variance {variance_rate:.9f}, "
            f"mean separation {mean_separation:.9f}"
        )

    ok = True
    ok &= abs(local_scale(np.array([0.0, 0.0]))) < 1e-14
    ok &= transported_error < 2e-14
    ok &= abs(rate_ratio - 1.0) < 4e-3
    ok &= angle_shift < 2e-4
    ok &= all(rows[i][1] > rows[i + 1][1] > 1.0
              for i in range(len(rows) - 1))
    ok &= all(rows[i][2] > rows[i + 1][2] > 1.0
              for i in range(len(rows) - 1))
    ok &= rows[-1][1] < 1.01
    ok &= rows[-1][2] < 1.01
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
