#!/usr/bin/env python3
"""Certificate for every open interval in the rare-tail Gram staircase."""

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
NODES = tuple(mp.mpf(value) for value in (-3, -2, -1, 0, 1, 2, 3))
LEVEL = 1
INTERVAL_LEVEL = 2
TAIL_CENTER = mp.mpf(1)
TAIL_WEIGHTS = (mp.mpf(1), mp.mpf(2))
SEPARATION_POWER = mp.mpf("2.5")


def orthogonal_projector(rows):
    """Project onto the orthogonal complement of the supplied rows."""
    matrix = mp.matrix(rows)
    return mp.eye(len(NODES)) - matrix.T * (matrix * matrix.T) ** -1 * matrix


def contaminated_gram(size):
    """Exact Gram with epsilon^(m+1) << delta << epsilon^m."""
    size = mp.mpf(size)
    epsilon = 1 / size
    delta = epsilon ** SEPARATION_POWER
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
    return epsilon, delta, gram


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
        for order in range(dimension)
    ]
    tail_row = [mp.exp(TAIL_CENTER * node) for node in NODES]
    derivative_row = [node * mp.exp(TAIL_CENTER * node) for node in NODES]

    intermediate_coefficients = []
    for step in range(1, INTERVAL_LEVEL + 1):
        projector = orthogonal_projector(
            polynomial_rows[:LEVEL + step] + [tail_row]
        )
        direction = projector * mp.matrix(polynomial_rows[LEVEL + step])
        coefficient = (
            derivative_cholesky[LEVEL + step, LEVEL + step] ** 2
            * (direction.T * direction)[0]
        )
        intermediate_coefficients.append(coefficient)

    tail_projector = orthogonal_projector(
        polynomial_rows[:LEVEL + INTERVAL_LEVEL + 1] + [tail_row]
    )
    tail_direction = tail_projector * mp.matrix(derivative_row)
    tail_coefficient = (
        TAIL_WEIGHTS[0] * TAIL_WEIGHTS[1] / sum(TAIL_WEIGHTS)
        * (tail_direction.T * tail_direction)[0]
    )

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

    evaluation_rows = (
        polynomial_rows[:LEVEL + INTERVAL_LEVEL + 1]
        + [tail_row, derivative_row]
    )
    evaluation_gram = mp.matrix(evaluation_rows) * mp.matrix(evaluation_rows).T
    evaluation_eigenvalues = descending_eigenvalues(evaluation_gram)

    rows = []
    for size in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        epsilon, delta, gram = contaminated_gram(size)
        eigenvalues = descending_eigenvalues(gram)
        primary = tuple(
            eigenvalues[LEVEL + index] / epsilon ** (2 * LEVEL)
            for index in range(2)
        )
        intermediate = tuple(
            eigenvalues[LEVEL + 1 + step]
            / epsilon ** (2 * (LEVEL + step))
            for step in range(1, INTERVAL_LEVEL + 1)
        )
        tail_index = LEVEL + INTERVAL_LEVEL + 2
        tail = (
            eigenvalues[tail_index]
            / (epsilon ** (2 * LEVEL) * delta ** 2)
        )
        remainder_ratio = (
            eigenvalues[tail_index + 1]
            / (epsilon ** (2 * LEVEL) * delta ** 2)
        )
        ratios = tuple(
            intermediate[index] / intermediate_coefficients[index]
            for index in range(INTERVAL_LEVEL)
        ) + (tail / tail_coefficient,)
        rows.append((size, primary, intermediate, tail, remainder_ratio, ratios))

    print("Translation-Gram interlacing-staircase certificate")
    print(f"open interval m={INTERVAL_LEVEL}")
    print(f"separation power {float(SEPARATION_POWER):.1f}")
    print(
        "primary coefficients "
        + " ".join(f"{float(value):.15f}" for value in primary_coefficients)
    )
    print(
        "intermediate coefficients "
        + " ".join(
            f"{float(value):.15f}" for value in intermediate_coefficients
        )
    )
    print(f"tail coefficient {float(tail_coefficient):.15f}")
    print(
        "confluent evaluation minimum eigenvalue "
        f"{float(evaluation_eigenvalues[-1]):.15f}"
    )
    print("n primary-1 primary-2 gaussian-1 gaussian-2 tail remainder "
          "ratio-g1 ratio-g2 ratio-tail")
    for size, primary, intermediate, tail, remainder, ratios in rows:
        values = primary + intermediate + (tail, remainder) + ratios
        print(
            f"{size:4d} "
            + " ".join(f"{float(value):.12f}" for value in values)
        )

    ok = True
    ok &= evaluation_eigenvalues[-1] > mp.mpf("0.00000001")
    ok &= all(value > 0 for value in intermediate_coefficients)
    ok &= tail_coefficient > 0
    ok &= all(
        abs(rows[-1][1][index] / primary_coefficients[index] - 1)
        < mp.mpf("0.005")
        for index in range(2)
    )
    ok &= all(
        abs(rows[-1][5][index] - 1) < mp.mpf("0.03")
        for index in range(INTERVAL_LEVEL + 1)
    )
    ok &= all(
        abs(rows[index + 1][5][mode] - 1)
        < abs(rows[index][5][mode] - 1)
        for index in range(3, len(rows) - 1)
        for mode in range(INTERVAL_LEVEL + 1)
    )
    ok &= rows[-1][4] < mp.mpf("0.03")
    ok &= all(
        rows[index + 1][4] < rows[index][4]
        for index in range(3, len(rows) - 1)
    )
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
