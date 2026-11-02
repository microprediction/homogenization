#!/usr/bin/env python3
"""Certificate for the second tail/Gaussian Gram collision."""

import mpmath as mp

from verify_translation_gram_conditioning import (
    covariance_gram,
    derivative_gram,
    descending_eigenvalues,
    gaussian_absolute_moment,
    vandermonde_matrix,
)


mp.mp.dps = 240
CENTER = mp.mpf(1)
NODES = tuple(mp.mpf(value) / 2 for value in (-6, -4, -2, 0, 2, 4, 6))
LEVEL = 2
TAIL_CENTER = mp.mpf(1)
TAIL_WEIGHTS = (mp.mpf(1), mp.mpf(2))
CRITICAL_RATIO = mp.mpf("1.5")


def orthogonal_projector(rows):
    """Project onto the orthogonal complement of the supplied rows."""
    matrix = mp.matrix(rows)
    return mp.eye(len(NODES)) - matrix.T * (matrix * matrix.T) ** -1 * matrix


def contaminated_gram(size):
    """Exact Gram with tail separation rho times epsilon squared."""
    size = mp.mpf(size)
    epsilon = 1 / size
    delta = CRITICAL_RATIO * epsilon ** 2
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


def main():
    dimension = len(NODES)
    derivative_cholesky = mp.cholesky(derivative_gram(CENTER, dimension))
    node_matrix = vandermonde_matrix(NODES)
    node_cholesky = mp.cholesky(node_matrix * node_matrix.T)
    gaussian_coefficients = tuple(
        (derivative_cholesky[index, index]
         * node_cholesky[index, index]) ** 2
        for index in range(dimension)
    )

    polynomial_rows = [
        [node ** order / mp.factorial(order) for node in NODES]
        for order in range(LEVEL + 3)
    ]
    tail_row = [mp.exp(TAIL_CENTER * node) for node in NODES]
    derivative_row = [node * mp.exp(TAIL_CENTER * node) for node in NODES]

    first_projector = orthogonal_projector(
        polynomial_rows[:LEVEL + 1] + [tail_row]
    )
    first_direction = first_projector * mp.matrix(
        polynomial_rows[LEVEL + 1]
    )
    first_coefficient = (
        derivative_cholesky[LEVEL + 1, LEVEL + 1] ** 2
        * (first_direction.T * first_direction)[0]
    )

    critical_projector = orthogonal_projector(
        polynomial_rows[:LEVEL + 2] + [tail_row]
    )
    gaussian_direction = critical_projector * mp.matrix(
        polynomial_rows[LEVEL + 2]
    )
    tail_direction = critical_projector * mp.matrix(derivative_row)
    gaussian_variance = derivative_cholesky[LEVEL + 2, LEVEL + 2] ** 2
    tail_variance = (
        TAIL_WEIGHTS[0] * TAIL_WEIGHTS[1] / sum(TAIL_WEIGHTS)
        * CRITICAL_RATIO ** 2
    )
    critical_block = (
        gaussian_variance * gaussian_direction * gaussian_direction.T
        + tail_variance * tail_direction * tail_direction.T
    )
    critical_coefficients = tuple(descending_eigenvalues(critical_block)[:2])

    level_projector = orthogonal_projector(polynomial_rows[:LEVEL])
    level_direction = level_projector * mp.matrix(polynomial_rows[LEVEL])
    level_direction /= mp.sqrt((level_direction.T * level_direction)[0])
    projected_tail = level_projector * mp.matrix(tail_row)
    primary_block = (
        gaussian_coefficients[LEVEL]
        * level_direction * level_direction.T
        + sum(TAIL_WEIGHTS) * projected_tail * projected_tail.T
    )
    primary_coefficients = tuple(descending_eigenvalues(primary_block)[:2])

    evaluation_rows = polynomial_rows + [tail_row, derivative_row]
    evaluation_gram = mp.matrix(evaluation_rows) * mp.matrix(evaluation_rows).T
    evaluation_eigenvalues = descending_eigenvalues(evaluation_gram)

    rows = []
    for size in (32, 64, 128, 256, 512, 1024, 2048):
        epsilon, gram = contaminated_gram(size)
        eigenvalues = descending_eigenvalues(gram)
        primary = tuple(
            eigenvalues[LEVEL + index] / epsilon ** (2 * LEVEL)
            for index in range(2)
        )
        first = eigenvalues[LEVEL + 2] / epsilon ** (2 * (LEVEL + 1))
        critical = tuple(
            eigenvalues[LEVEL + 3 + index]
            / epsilon ** (2 * (LEVEL + 2))
            for index in range(2)
        )
        ratios = tuple(
            critical[index] / critical_coefficients[index]
            for index in range(2)
        )
        rows.append((size, primary, first, critical, ratios))

    print("Translation-Gram second critical-collision certificate")
    print(
        "primary coefficients "
        + " ".join(f"{float(value):.15f}" for value in primary_coefficients)
    )
    print(f"first Gaussian coefficient {float(first_coefficient):.15f}")
    print(
        "critical coefficients "
        + " ".join(f"{float(value):.15f}" for value in critical_coefficients)
    )
    print(
        "confluent evaluation minimum eigenvalue "
        f"{float(evaluation_eigenvalues[-1]):.15f}"
    )
    print("n primary-1 primary-2 first critical-1 critical-2 ratio-1 ratio-2")
    for size, primary, first, critical, ratios in rows:
        values = primary + (first,) + critical + ratios
        print(
            f"{size:4d} "
            + " ".join(f"{float(value):.12f}" for value in values)
        )

    ok = True
    ok &= evaluation_eigenvalues[-1] > mp.mpf("0.00000001")
    ok &= all(value > 0 for value in critical_coefficients)
    ok &= all(
        abs(rows[index + 1][4][mode] - 1)
        < abs(rows[index][4][mode] - 1)
        for index in range(len(rows) - 1)
        for mode in range(2)
    )
    ok &= all(
        abs(rows[-1][1][index] / primary_coefficients[index] - 1)
        < mp.mpf("0.005")
        for index in range(2)
    )
    ok &= abs(rows[-1][2] / first_coefficient - 1) < mp.mpf("0.02")
    ok &= all(
        abs(rows[-1][4][index] - 1) < mp.mpf("0.02")
        for index in range(2)
    )
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
