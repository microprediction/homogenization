#!/usr/bin/env python3
"""Certificate for the asymmetric-gauge Green--Kubo flat constant."""

import itertools
import math

import numpy as np
from scipy.optimize import minimize_scalar

from verify_norm_flat import (
    FLAT_SCALE,
    SIGMA,
    chain_data,
    rate,
    scgf,
)


# rho(x) = sum_i a_i^+ x_i^+ + a_i^- x_i^-.
POSITIVE_WEIGHTS = np.array([1.0, 0.7])
NEGATIVE_WEIGHTS = np.array([2.0, 1.4])


def gauge(value):
    """Asymmetric weighted l1 gauge."""
    return float(
        POSITIVE_WEIGHTS @ np.maximum(value, 0.0)
        + NEGATIVE_WEIGHTS @ np.maximum(-value, 0.0)
    )


def contracted_gauge_rate(scores, green_kubo, radius):
    """Contract the exact rate over the asymmetric gauge sphere."""
    def angular_rate(angle):
        direction = np.array([math.cos(angle), math.sin(angle)])
        boundary = direction / gauge(direction)
        return rate(scores, green_kubo, radius * boundary)

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
        radius = FLAT_SCALE ** beta * profile ** (-beta)
        value, _ = contracted_gauge_rate(scores, green_kubo, radius)
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

    # The polar unit ball is the asymmetric box
    # product_i[-a_i^-, a_i^+]. A convex quadratic reaches its maximum
    # at a vertex.
    polar_vertices = np.array(list(itertools.product(
        (-NEGATIVE_WEIGHTS[0], POSITIVE_WEIGHTS[0]),
        (-NEGATIVE_WEIGHTS[1], POSITIVE_WEIGHTS[1]),
    )))
    vertex_values = np.einsum(
        "ni,ij,nj->n", polar_vertices, green_kubo, polar_vertices
    )
    index = int(np.argmax(vertex_values))
    gauge_constant = float(vertex_values[index])
    maximizing_vertex = polar_vertices[index]
    leading = FLAT_SCALE ** (2.0 * beta) / (2.0 * gauge_constant)

    coordinate_map = np.array([[1.2, -0.4], [0.3, 1.1]])
    transformed_green = coordinate_map @ green_kubo @ coordinate_map.T
    transformed_vertices = np.array([
        np.linalg.solve(coordinate_map.T, vertex)
        for vertex in polar_vertices
    ])
    transformed_values = np.einsum(
        "ni,ij,nj->n",
        transformed_vertices,
        transformed_green,
        transformed_vertices,
    )
    coordinate_error = abs(float(np.max(transformed_values)) - gauge_constant)

    step = 2e-4
    hessian = np.empty((2, 2))
    for i in range(2):
        for j in range(2):
            ei = np.eye(2)[i]
            ej = np.eye(2)[j]
            hessian[i, j] = (
                scgf(scores, step * (ei + ej))[0]
                - scgf(scores, step * (ei - ej))[0]
                - scgf(scores, step * (-ei + ej))[0]
                + scgf(scores, -step * (ei + ej))[0]
            ) / (4.0 * step ** 2)
    hessian_error = float(np.max(np.abs(hessian - green_kubo)))

    small_radius = 0.005
    exact_rate, angle = contracted_gauge_rate(
        scores, green_kubo, small_radius
    )
    quadratic = small_radius ** 2 / (2.0 * gauge_constant)
    rate_ratio = exact_rate / quadratic

    rows = []
    for log_speed in (128.0, 512.0, 2048.0, 8192.0):
        _, second = saddle(log_speed, 2.0, scores, green_kubo)
        _, first = saddle(log_speed, 1.0, scores, green_kubo)
        scale = log_speed ** (2.0 * beta)
        variance_rate = second * scale / leading
        mean_separation = (2.0 * first - second) * scale / leading
        rows.append((log_speed, variance_rate, mean_separation))

    print("Asymmetric-gauge Green--Kubo flat-map certificate")
    print(f"stationary law {stationary}")
    print(f"Green--Kubo matrix\n{green_kubo}")
    print(f"gauge constant {gauge_constant:.9f}")
    print(f"maximizing polar vertex {maximizing_vertex}")
    print(f"predicted logarithmic constant {leading:.9f}")
    print(f"coordinate-invariance discrepancy {coordinate_error:.3e}")
    print(f"tilted Hessian maximum discrepancy {hessian_error:.3e}")
    print(f"small-radius direction {angle:.9f}")
    print(f"small-radius normalized rate {rate_ratio:.9f}")
    for log_speed, variance_rate, mean_separation in rows:
        print(
            f"log speed {log_speed:7.0f}: variance {variance_rate:.9f}, "
            f"mean separation {mean_separation:.9f}"
        )

    ok = True
    ok &= abs(gauge_constant - 4.60437900568596) < 1e-12
    ok &= np.max(np.abs(maximizing_vertex - np.array([-2.0, -1.4]))) < 1e-12
    ok &= coordinate_error < 2e-14
    ok &= hessian_error < 2e-8
    ok &= abs(rate_ratio - 1.0) < 8e-4
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
