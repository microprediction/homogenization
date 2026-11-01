#!/usr/bin/env python3
"""Certificate for the arbitrary-norm Green--Kubo flat-map constant."""

import itertools
import math

import numpy as np
from scipy.linalg import eig
from scipy.optimize import minimize_scalar, root


Q = np.array([
    [-4.0, 3.0, 1.0],
    [1.0, -5.0, 4.0],
    [2.0, 1.0, -3.0],
])
RAW_SCORES = np.array([
    [-1.0, 0.3],
    [0.5, -1.2],
    [2.0, 1.5],
])
SIGMA = 2.0
FLAT_SCALE = 1.5


def chain_data():
    """Stationary law, centered scores and Poisson Green--Kubo form."""
    values, vectors = np.linalg.eig(Q.T)
    stationary = np.real(vectors[:, np.argmin(np.abs(values))])
    stationary /= stationary.sum()
    scores = RAW_SCORES - np.ones((3, 1)) * (stationary @ RAW_SCORES)
    projection = np.ones((3, 1)) * stationary[None, :]
    poisson = np.linalg.solve(-Q + projection, scores)
    one_sided = scores.T @ (stationary[:, None] * poisson)
    return stationary, scores, one_sided + one_sided.T


def scgf(scores, tilt):
    """Principal tilted eigenvalue and gradient."""
    values, left, right = eig(
        Q + np.diag(scores @ tilt), left=True, right=True
    )
    index = int(np.argmax(np.real(values)))
    left_vector = left[:, index]
    right_vector = right[:, index]
    denominator = np.vdot(left_vector, right_vector)
    gradient = np.array([
        np.vdot(left_vector, scores[:, j] * right_vector) / denominator
        for j in range(2)
    ])
    return float(np.real(values[index])), np.real(gradient)


def rate(scores, green_kubo, value):
    """Exact Legendre transform of the principal eigenvalue."""
    solution = root(
        lambda tilt: scgf(scores, tilt)[1] - value,
        np.linalg.solve(green_kubo, value),
        tol=1e-11,
    )
    residual = np.linalg.norm(solution.fun)
    if residual > 1e-9:
        raise RuntimeError(f"tilt residual {residual:.3e}")
    eigenvalue, _ = scgf(scores, solution.x)
    return float(solution.x @ value - eigenvalue)


def contracted_l1_rate(scores, green_kubo, radius):
    """Contract the exact rate over the l1 sphere."""
    def angular_rate(angle):
        direction = np.array([math.cos(angle), math.sin(angle)])
        boundary = direction / np.linalg.norm(direction, ord=1)
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
        value, _ = contracted_l1_rate(scores, green_kubo, radius)
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
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=2)))
    sign_values = np.einsum(
        "ni,ij,nj->n", signs, green_kubo, signs
    )
    dual_constant = float(np.max(sign_values))
    maximizing_sign = signs[int(np.argmax(sign_values))]
    leading = FLAT_SCALE ** (2.0 * beta) / (2.0 * dual_constant)

    coordinate_map = np.array([[1.2, -0.4], [0.3, 1.1]])
    transformed_green = coordinate_map @ green_kubo @ coordinate_map.T
    transformed_dual_vertices = np.array([
        np.linalg.solve(coordinate_map.T, sign) for sign in signs
    ])
    transformed_values = np.einsum(
        "ni,ij,nj->n",
        transformed_dual_vertices,
        transformed_green,
        transformed_dual_vertices,
    )
    coordinate_error = abs(float(np.max(transformed_values)) - dual_constant)

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
    exact_rate, angle = contracted_l1_rate(
        scores, green_kubo, small_radius
    )
    quadratic = small_radius ** 2 / (2.0 * dual_constant)
    rate_ratio = exact_rate / quadratic

    rows = []
    for log_speed in (128.0, 512.0, 2048.0, 8192.0):
        _, second = saddle(log_speed, 2.0, scores, green_kubo)
        _, first = saddle(log_speed, 1.0, scores, green_kubo)
        scale = log_speed ** (2.0 * beta)
        variance_rate = second * scale / leading
        mean_separation = (2.0 * first - second) * scale / leading
        rows.append((log_speed, variance_rate, mean_separation))

    print("Arbitrary-norm Green--Kubo flat-map certificate")
    print(f"stationary law {stationary}")
    print(f"Green--Kubo matrix\n{green_kubo}")
    print(f"l1 dual constant {dual_constant:.9f}")
    print(f"maximizing dual sign {maximizing_sign}")
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
