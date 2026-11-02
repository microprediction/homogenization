#!/usr/bin/env python3
"""Certificate for the quadratic mode split of two coalescing rare tails."""

import mpmath as mp

from verify_translation_gram_conditioning import (
    covariance_gram,
    derivative_gram,
    descending_eigenvalues,
    gaussian_absolute_moment,
    vandermonde_matrix,
)


mp.mp.dps = 220
CENTER = mp.mpf(1)
NODES = tuple(mp.mpf(value) / 2 for value in (-5, -3, -1, 1, 3, 5))
LEVEL = 2
TAIL_CENTER = mp.mpf(1)
TAIL_WEIGHTS = (mp.mpf(1), mp.mpf(2))


def orthogonal_projector(rows):
    """Project onto the orthogonal complement of the supplied rows."""
    matrix = mp.matrix(rows)
    return mp.eye(len(NODES)) - matrix.T * (matrix * matrix.T) ** -1 * matrix


def contaminated_gram(size):
    """Exact Gram for a Gaussian plus two weighted-centered coalescing atoms."""
    size = mp.mpf(size)
    epsilon = 1 / size
    delta = mp.sqrt(epsilon)
    total_weight = sum(TAIL_WEIGHTS)
    offsets = (
        -TAIL_WEIGHTS[1] / total_weight,
        TAIL_WEIGHTS[0] / total_weight,
    )
    slopes = tuple(
        TAIL_CENTER + offset * delta for offset in offsets
    )
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
    rare_scales = tuple(
        probability
        * mp.exp(2 * CENTER * slope / epsilon)
        / epsilon ** (2 * LEVEL)
        for probability, slope in zip(probabilities, slopes)
    )
    return epsilon, delta, gram, rare_scales


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
        for order in range(LEVEL + 1)
    ]
    tail_row = [mp.exp(TAIL_CENTER * node) for node in NODES]
    derivative_row = [
        node * mp.exp(TAIL_CENTER * node) for node in NODES
    ]
    split_projector = orthogonal_projector(polynomial_rows + [tail_row])
    split_direction = split_projector * mp.matrix(derivative_row)
    split_coefficient = (
        TAIL_WEIGHTS[0] * TAIL_WEIGHTS[1] / sum(TAIL_WEIGHTS)
        * (split_direction.T * split_direction)[0]
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
    primary_coefficients = tuple(
        descending_eigenvalues(primary_block)[:2]
    )

    evaluation_rows = polynomial_rows + [tail_row, derivative_row]
    evaluation_gram = mp.matrix(evaluation_rows) * mp.matrix(evaluation_rows).T
    evaluation_eigenvalues = descending_eigenvalues(evaluation_gram)

    rows = []
    for size in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        epsilon, delta, gram, rare_scales = contaminated_gram(size)
        eigenvalues = descending_eigenvalues(gram)
        normalized_split = (
            eigenvalues[LEVEL + 2]
            / epsilon ** (2 * LEVEL)
            / delta ** 2
        )
        normalized_primary = tuple(
            eigenvalues[LEVEL + index] / epsilon ** (2 * LEVEL)
            for index in range(2)
        )
        rows.append((
            size,
            rare_scales,
            epsilon / delta,
            normalized_primary,
            normalized_split,
            normalized_split / split_coefficient,
        ))

    print("Translation-Gram coalescing-tail split certificate")
    print(f"split coefficient {float(split_coefficient):.15f}")
    print(
        "primary coefficients "
        + " ".join(f"{float(value):.15f}" for value in primary_coefficients)
    )
    print(
        "confluent evaluation minimum eigenvalue "
        f"{float(evaluation_eigenvalues[-1]):.15f}"
    )
    print("n epsilon/delta primary-1 primary-2 normalized-split ratio")
    for size, _, separation, primary, normalized_split, ratio in rows:
        print(
            f"{size:4d} {float(separation):.12f} "
            + " ".join(f"{float(value):.12f}" for value in primary)
            + " "
            f"{float(normalized_split):.12f} {float(ratio):.12f}"
        )

    ok = True
    ok &= evaluation_eigenvalues[-1] > mp.mpf("0.000001")
    ok &= split_coefficient > 0
    ok &= all(
        all(
            abs(scale - weight) < mp.mpf("1e-14")
            for scale, weight in zip(rare_scales, TAIL_WEIGHTS)
        )
        for _, rare_scales, _, _, _, _ in rows
    )
    ok &= all(
        abs(rows[index + 1][5] - 1) < abs(rows[index][5] - 1)
        for index in range(len(rows) - 1)
    )
    ok &= all(
        abs(rows[-1][3][index] / primary_coefficients[index] - 1)
        < mp.mpf("0.02")
        for index in range(2)
    )
    ok &= abs(rows[-1][5] - 1) < mp.mpf("0.05")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
