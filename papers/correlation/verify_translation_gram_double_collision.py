#!/usr/bin/env python3
"""Certificate for two successive three-tail/Gaussian collision blocks."""

import mpmath as mp

from verify_translation_gram_conditioning import (
    covariance_gram, derivative_gram, descending_eigenvalues,
    gaussian_absolute_moment, vandermonde_matrix,
)
from verify_translation_gram_multitail_confluence import (
    CENTER, LEVEL, RAW_OFFSETS, TAIL_CENTER, TAIL_WEIGHTS,
    weighted_residual,
)


mp.mp.dps = 260
CRITICAL_RATIO = mp.mpf("1.5")
NODES = tuple(mp.mpf(value) for value in (-3, -2, -1, 0, 1, 2, 3))


def orthogonal_projector(rows):
    """Project onto the orthogonal complement of the supplied rows."""
    matrix = mp.matrix(rows)
    return mp.eye(len(NODES)) - matrix.T * (matrix * matrix.T) ** -1 * matrix


def contaminated_gram(size):
    """Exact Gram with delta equal to rho times epsilon."""
    size = mp.mpf(size)
    epsilon = 1 / size
    delta = CRITICAL_RATIO * epsilon
    mean_offset = sum(
        weight * offset for weight, offset in zip(TAIL_WEIGHTS, RAW_OFFSETS)
    ) / sum(TAIL_WEIGHTS)
    offsets = tuple(offset - mean_offset for offset in RAW_OFFSETS)
    slopes = tuple(TAIL_CENTER + offset * delta for offset in offsets)
    exponents = tuple(CENTER + epsilon * node for node in NODES)
    base_gram = covariance_gram(exponents)
    base_means = mp.matrix(
        [gaussian_absolute_moment(exponent) for exponent in exponents]
    )
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
    for first in range(len(atoms)):
        for second in range(first + 1, len(atoms)):
            difference = atoms[first] - atoms[second]
            gram += probabilities[first] * probabilities[second] * difference * difference.T
    rare_scales = tuple(
        probability * mp.exp(2 * CENTER * slope / epsilon)
        / epsilon ** (2 * LEVEL)
        for probability, slope in zip(probabilities, slopes)
    )
    return epsilon, gram, rare_scales, offsets


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
    tail_rows = [
        [node ** order / mp.factorial(order) * mp.exp(TAIL_CENTER * node)
         for node in NODES]
        for order in range(3)
    ]

    _, _, _, offsets = contaminated_gram(32)
    label_rows = [[offset ** order for offset in offsets] for order in range(3)]
    label_residuals = [weighted_residual(label_rows[order], label_rows[:order])
                       for order in range(3)]
    label_coefficients = tuple(
        (residual.T * mp.diag(TAIL_WEIGHTS) * residual)[0]
        for residual in label_residuals
    )

    first_projector = orthogonal_projector(
        polynomial_rows[:LEVEL + 1] + tail_rows[:1]
    )
    first_gaussian = first_projector * mp.matrix(polynomial_rows[LEVEL + 1])
    first_tail = first_projector * mp.matrix(tail_rows[1])
    first_block = (
        derivative_cholesky[LEVEL + 1, LEVEL + 1] ** 2
        * first_gaussian * first_gaussian.T
        + label_coefficients[1] * CRITICAL_RATIO ** 2
        * first_tail * first_tail.T
    )
    first_coefficients = tuple(descending_eigenvalues(first_block)[:2])

    second_projector = orthogonal_projector(
        polynomial_rows[:LEVEL + 2] + tail_rows[:2]
    )
    second_gaussian = second_projector * mp.matrix(polynomial_rows[LEVEL + 2])
    second_tail = second_projector * mp.matrix(tail_rows[2])
    second_block = (
        derivative_cholesky[LEVEL + 2, LEVEL + 2] ** 2
        * second_gaussian * second_gaussian.T
        + label_coefficients[2] * CRITICAL_RATIO ** 4
        * second_tail * second_tail.T
    )
    second_coefficients = tuple(descending_eigenvalues(second_block)[:2])

    level_projector = orthogonal_projector(polynomial_rows[:LEVEL])
    level_direction = level_projector * mp.matrix(polynomial_rows[LEVEL])
    level_direction /= mp.sqrt((level_direction.T * level_direction)[0])
    projected_tail = level_projector * mp.matrix(tail_rows[0])
    primary_block = (
        gaussian_coefficients[LEVEL] * level_direction * level_direction.T
        + sum(TAIL_WEIGHTS) * projected_tail * projected_tail.T
    )
    primary_coefficients = tuple(descending_eigenvalues(primary_block)[:2])

    evaluation_rows = polynomial_rows + tail_rows
    evaluation_gram = mp.matrix(evaluation_rows) * mp.matrix(evaluation_rows).T
    evaluation_eigenvalues = descending_eigenvalues(evaluation_gram)

    rows = []
    for size in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        epsilon, gram, rare_scales, _ = contaminated_gram(size)
        eigenvalues = descending_eigenvalues(gram)
        primary = tuple(
            eigenvalues[LEVEL + index] / epsilon ** (2 * LEVEL)
            for index in range(2)
        )
        first = tuple(
            eigenvalues[LEVEL + 2 + index] / epsilon ** (2 * (LEVEL + 1))
            for index in range(2)
        )
        second = tuple(
            eigenvalues[LEVEL + 4 + index] / epsilon ** (2 * (LEVEL + 2))
            for index in range(2)
        )
        ratios = tuple(
            value / coefficient
            for value, coefficient in zip(
                first + second, first_coefficients + second_coefficients
            )
        )
        rows.append((size, rare_scales, primary, first, second, ratios))

    print("Translation-Gram double-collision certificate")
    print("primary coefficients "
          + " ".join(f"{float(value):.15f}" for value in primary_coefficients))
    print("first collision coefficients "
          + " ".join(f"{float(value):.15f}" for value in first_coefficients))
    print("second collision coefficients "
          + " ".join(f"{float(value):.15f}" for value in second_coefficients))
    print("confluent evaluation minimum eigenvalue "
          f"{float(evaluation_eigenvalues[-1]):.15f}")
    print("n primary-1 primary-2 first-1 first-2 second-1 second-2 ratios")
    for size, _, primary, first, second, ratios in rows:
        values = primary + first + second + ratios
        print(f"{size:5d} " + " ".join(f"{float(value):.12f}" for value in values))

    ok = True
    ok &= abs(sum(
        weight * offset for weight, offset in zip(TAIL_WEIGHTS, offsets)
    )) < mp.mpf("1e-14")
    ok &= evaluation_eigenvalues[-1] > mp.mpf("0.0000001")
    ok &= all(value > 0 for value in first_coefficients + second_coefficients)
    ok &= all(
        all(abs(scale - weight) < mp.mpf("1e-14")
            for scale, weight in zip(rare_scales, TAIL_WEIGHTS))
        for _, rare_scales, _, _, _, _ in rows
    )
    ok &= all(
        abs(rows[-1][5][mode] - 1) < abs(rows[1][5][mode] - 1)
        for mode in range(4)
    )
    ok &= all(
        abs(rows[-1][2][index] / primary_coefficients[index] - 1)
        < mp.mpf("0.01") for index in range(2)
    )
    ok &= all(abs(ratio - 1) < mp.mpf("0.03") for ratio in rows[-1][5])
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
