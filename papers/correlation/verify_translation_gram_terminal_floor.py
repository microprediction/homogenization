#!/usr/bin/env python3
"""Certificate for the finite-dimensional end of the rare-tail staircase."""

import mpmath as mp

from verify_translation_gram_conditioning import (
    covariance_gram,
    derivative_gram,
    descending_eigenvalues,
    gaussian_absolute_moment,
)


mp.mp.dps = 240
CENTER = mp.mpf(1)
NODES = tuple(mp.mpf(value) for value in (-3, -2, -1, 0, 1, 2))
LEVEL = 1
TAIL_CENTER = mp.mpf(1)
TAIL_WEIGHTS = (mp.mpf(1), mp.mpf(2))
CRITICAL_RATIO = mp.mpf("1.5")
TERMINAL_LEVEL = len(NODES) - LEVEL - 2


def orthogonal_projector(rows):
    """Project onto the orthogonal complement of the supplied rows."""
    matrix = mp.matrix(rows)
    return mp.eye(len(NODES)) - matrix.T * (matrix * matrix.T) ** -1 * matrix


def contaminated_gram(size, delta):
    """Return the exact Gaussian-plus-two-atom covariance Gram."""
    size = mp.mpf(size)
    epsilon = 1 / size
    total_weight = sum(TAIL_WEIGHTS)
    offsets = (
        -TAIL_WEIGHTS[1] / total_weight,
        TAIL_WEIGHTS[0] / total_weight,
    )
    slopes = tuple(TAIL_CENTER + offset * delta for offset in offsets)
    exponents = tuple(CENTER + epsilon * node for node in NODES)
    base_gram = covariance_gram(exponents)
    base_means = mp.matrix([
        gaussian_absolute_moment(exponent) for exponent in exponents
    ])
    probabilities = tuple(
        weight * epsilon ** (2 * LEVEL)
        * mp.exp(-2 * CENTER * slope / epsilon)
        for weight, slope in zip(TAIL_WEIGHTS, slopes)
    )
    atoms = tuple(
        mp.matrix([
            mp.exp(slope * exponent / epsilon) for exponent in exponents
        ])
        for slope in slopes
    )
    base_weight = 1 - sum(probabilities)
    gram = base_weight * base_gram
    for probability, atom in zip(probabilities, atoms):
        difference = atom - base_means
        gram += base_weight * probability * difference * difference.T
    difference = atoms[0] - atoms[1]
    gram += probabilities[0] * probabilities[1] * difference * difference.T
    return epsilon, gram


def squared_norm(vector):
    """Euclidean squared norm of a column vector."""
    return (vector.T * vector)[0]


def main():
    dimension = len(NODES)
    derivative_cholesky = mp.cholesky(derivative_gram(CENTER, dimension))
    polynomial_rows = [
        [node ** order / mp.factorial(order) for node in NODES]
        for order in range(dimension)
    ]
    tail_row = [mp.exp(TAIL_CENTER * node) for node in NODES]
    derivative_row = [node * mp.exp(TAIL_CENTER * node) for node in NODES]

    intermediate_coefficients = []
    for step in range(1, TERMINAL_LEVEL):
        projector = orthogonal_projector(
            polynomial_rows[:LEVEL + step] + [tail_row]
        )
        direction = projector * mp.matrix(polynomial_rows[LEVEL + step])
        intermediate_coefficients.append(
            derivative_cholesky[LEVEL + step, LEVEL + step] ** 2
            * squared_norm(direction)
        )

    terminal_projector = orthogonal_projector(
        polynomial_rows[:LEVEL + TERMINAL_LEVEL] + [tail_row]
    )
    gaussian_direction = terminal_projector * mp.matrix(
        polynomial_rows[LEVEL + TERMINAL_LEVEL]
    )
    tail_direction = terminal_projector * mp.matrix(derivative_row)
    gaussian_floor = (
        derivative_cholesky[
            LEVEL + TERMINAL_LEVEL, LEVEL + TERMINAL_LEVEL
        ] ** 2
        * squared_norm(gaussian_direction)
    )
    tail_increment = (
        TAIL_WEIGHTS[0] * TAIL_WEIGHTS[1] / sum(TAIL_WEIGHTS)
        * CRITICAL_RATIO ** 2 * squared_norm(tail_direction)
    )
    terminal_coefficient = gaussian_floor + tail_increment

    normalized_gaussian = gaussian_direction / mp.sqrt(
        squared_norm(gaussian_direction)
    )
    normalized_tail = tail_direction / mp.sqrt(squared_norm(tail_direction))
    sign = mp.sign((normalized_gaussian.T * normalized_tail)[0])
    collinearity_residual = mp.sqrt(
        squared_norm(normalized_gaussian - sign * normalized_tail)
    )

    gaussian_rank_rows = (
        polynomial_rows[:LEVEL + TERMINAL_LEVEL]
        + [tail_row, polynomial_rows[LEVEL + TERMINAL_LEVEL]]
    )
    tail_rank_rows = (
        polynomial_rows[:LEVEL + TERMINAL_LEVEL]
        + [tail_row, derivative_row]
    )
    gaussian_rank_eigenvalues = descending_eigenvalues(
        mp.matrix(gaussian_rank_rows) * mp.matrix(gaussian_rank_rows).T
    )
    tail_rank_eigenvalues = descending_eigenvalues(
        mp.matrix(tail_rank_rows) * mp.matrix(tail_rank_rows).T
    )

    rows = []
    for size in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        epsilon = mp.mpf(1) / size
        critical_delta = CRITICAL_RATIO * epsilon ** TERMINAL_LEVEL
        _, critical_gram = contaminated_gram(size, critical_delta)
        critical_eigenvalues = descending_eigenvalues(critical_gram)
        intermediate_ratios = tuple(
            critical_eigenvalues[LEVEL + 1 + step]
            / epsilon ** (2 * (LEVEL + step))
            / intermediate_coefficients[step - 1]
            for step in range(1, TERMINAL_LEVEL)
        )
        critical_scaled = (
            critical_eigenvalues[-1]
            / epsilon ** (2 * (LEVEL + TERMINAL_LEVEL))
        )

        subfloor_delta = epsilon ** (TERMINAL_LEVEL + mp.mpf("0.5"))
        _, subfloor_gram = contaminated_gram(size, subfloor_delta)
        subfloor_eigenvalues = descending_eigenvalues(subfloor_gram)
        subfloor_scaled = (
            subfloor_eigenvalues[-1]
            / epsilon ** (2 * (LEVEL + TERMINAL_LEVEL))
        )
        rows.append((
            size,
            intermediate_ratios,
            critical_scaled,
            critical_scaled / terminal_coefficient,
            subfloor_scaled,
            subfloor_scaled / gaussian_floor,
        ))

    print("Translation-Gram terminal-floor certificate")
    print(f"terminal level {TERMINAL_LEVEL}")
    print(
        "intermediate coefficients "
        + " ".join(
            f"{float(value):.15f}" for value in intermediate_coefficients
        )
    )
    print(f"Gaussian floor {float(gaussian_floor):.15f}")
    print(f"critical tail increment {float(tail_increment):.15f}")
    print(f"critical total {float(terminal_coefficient):.15f}")
    print(f"collinearity residual {float(collinearity_residual):.3e}")
    print(
        "Gaussian evaluation minimum eigenvalue "
        f"{float(gaussian_rank_eigenvalues[-1]):.15f}"
    )
    print(
        "tail evaluation minimum eigenvalue "
        f"{float(tail_rank_eigenvalues[-1]):.15f}"
    )
    print("n Gaussian-1-ratio Gaussian-2-ratio critical critical-ratio "
          "subfloor subfloor-ratio")
    for row in rows:
        values = row[1] + row[2:]
        print(
            f"{row[0]:4d} "
            + " ".join(f"{float(value):.12f}" for value in values)
        )

    ok = True
    ok &= gaussian_rank_eigenvalues[-1] > mp.mpf("0.00000001")
    ok &= tail_rank_eigenvalues[-1] > mp.mpf("0.00000001")
    ok &= collinearity_residual < mp.mpf("0.0000000001")
    ok &= gaussian_floor > 0
    ok &= tail_increment > 0
    ok &= all(
        abs(rows[index + 1][mode] - 1) < abs(rows[index][mode] - 1)
        for index in range(3, len(rows) - 1)
        for mode in (3, 5)
    )
    ok &= all(
        abs(rows[-1][1][index] - 1) < mp.mpf("0.01")
        for index in range(TERMINAL_LEVEL - 1)
    )
    ok &= abs(rows[-1][3] - 1) < mp.mpf("0.01")
    ok &= abs(rows[-1][5] - 1) < mp.mpf("0.01")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
