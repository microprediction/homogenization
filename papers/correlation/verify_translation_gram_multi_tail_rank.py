#!/usr/bin/env python3
"""Certificate for multi-tail promotion of several coalescing Gram modes."""

import mpmath as mp

from verify_translation_gram_conditioning import (
    covariance_gram,
    derivative_gram,
    descending_eigenvalues,
    gaussian_absolute_moment,
    vandermonde_matrix,
)


mp.mp.dps = 200
CENTER = mp.mpf(1)
NODES = tuple(mp.mpf(value) / 2 for value in (-5, -3, -1, 1, 3, 5))
LEVEL = 2
TAIL_SLOPES = (mp.mpf("0.5"), mp.mpf(1))


def orthogonal_projector(rows):
    """Project onto the orthogonal complement of the supplied row vectors."""
    matrix = mp.matrix(rows)
    return mp.eye(len(NODES)) - matrix.T * (matrix * matrix.T) ** -1 * matrix


def contaminated_gram(size):
    """Exact six-output covariance Gram for a Gaussian plus two rare atoms."""
    size = mp.mpf(size)
    epsilon = 1 / size
    exponents = tuple(CENTER + epsilon * node for node in NODES)
    base_gram = covariance_gram(exponents)
    base_means = mp.matrix([
        gaussian_absolute_moment(exponent) for exponent in exponents
    ])
    probabilities = tuple(
        epsilon ** (2 * LEVEL)
        * mp.exp(-2 * CENTER * slope / epsilon)
        for slope in TAIL_SLOPES
    )
    atoms = tuple(
        mp.matrix([
            mp.exp(slope * exponent / epsilon) for exponent in exponents
        ])
        for slope in TAIL_SLOPES
    )
    base_weight = 1 - sum(probabilities)
    gram = base_weight * base_gram
    for probability, atom in zip(probabilities, atoms):
        difference = atom - base_means
        gram += (
            base_weight * probability
            * difference * difference.T
        )
    for first in range(len(atoms)):
        for second in range(first + 1, len(atoms)):
            difference = atoms[first] - atoms[second]
            gram += (
                probabilities[first] * probabilities[second]
                * difference * difference.T
            )
    rare_scales = tuple(
        probability
        * mp.exp(2 * CENTER * slope / epsilon)
        / epsilon ** (2 * LEVEL)
        for probability, slope in zip(probabilities, TAIL_SLOPES)
    )
    return epsilon, gram, rare_scales


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
    projector = orthogonal_projector(polynomial_rows[:LEVEL])
    level_direction = projector * mp.matrix(polynomial_rows[LEVEL])
    level_direction /= mp.sqrt((level_direction.T * level_direction)[0])
    projected_tails = tuple(
        projector * mp.matrix([
            mp.exp(slope * node) for node in NODES
        ])
        for slope in TAIL_SLOPES
    )

    limiting_block = (
        gaussian_coefficients[LEVEL]
        * level_direction * level_direction.T
    )
    for projected_tail in projected_tails:
        limiting_block += projected_tail * projected_tail.T
    limiting_eigenvalues = descending_eigenvalues(limiting_block)
    predicted = tuple(limiting_eigenvalues[:3])

    evaluation_rows = [
        [node ** order for node in NODES]
        for order in range(LEVEL + 1)
    ] + [
        [mp.exp(slope * node) for node in NODES]
        for slope in TAIL_SLOPES
    ]
    evaluation_gram = mp.matrix(evaluation_rows) * mp.matrix(evaluation_rows).T
    evaluation_eigenvalues = descending_eigenvalues(evaluation_gram)

    rows = []
    for size in (32, 64, 128, 256, 512, 1024):
        epsilon, gram, rare_scales = contaminated_gram(size)
        eigenvalues = descending_eigenvalues(gram)
        leading_ratios = (
            eigenvalues[0] / gaussian_coefficients[0],
            eigenvalues[1] / epsilon ** 2 / gaussian_coefficients[1],
        )
        promoted = tuple(
            eigenvalues[LEVEL + index] / epsilon ** (2 * LEVEL)
            for index in range(3)
        )
        rows.append((
            size,
            rare_scales,
            leading_ratios,
            promoted,
            tuple(
                promoted[index] / predicted[index] for index in range(3)
            ),
        ))

    print("Translation-Gram multi-tail rank-promotion certificate")
    print(
        "Gaussian level-two coefficient "
        f"{float(gaussian_coefficients[LEVEL]):.15f}"
    )
    print(
        "promoted coefficients "
        + " ".join(f"{float(value):.15f}" for value in predicted)
    )
    print(
        "Chebyshev evaluation minimum eigenvalue "
        f"{float(evaluation_eigenvalues[-1]):.15f}"
    )
    print("n first-ratio second-ratio mode-1 mode-2 mode-3 ratio-1 ratio-2 ratio-3")
    for size, _, leading_ratios, promoted, ratios in rows:
        values = leading_ratios + promoted + ratios
        print(
            f"{size:4d} "
            + " ".join(f"{float(value):.12f}" for value in values)
        )

    expected_direction = (
        mp.matrix([5, -1, -4, -4, -1, 5]) / mp.sqrt(84)
    )
    ok = True
    ok &= mp.norm(level_direction - expected_direction) < mp.mpf("1e-14")
    ok &= max(
        abs((mp.matrix(polynomial_rows[order]).T * level_direction)[0])
        for order in range(LEVEL)
    ) < mp.mpf("1e-14")
    ok &= evaluation_eigenvalues[-1] > mp.mpf("0.0001")
    ok &= all(value > 0 for value in predicted)
    ok &= max(
        abs(value) for value in limiting_eigenvalues[3:]
    ) < mp.mpf("1e-14")
    ok &= all(
        all(abs(scale - 1) < mp.mpf("1e-14") for scale in rare_scales)
        for _, rare_scales, _, _, _ in rows
    )
    ok &= all(
        all(
            abs(rows[index + 1][4][mode] - 1)
            < abs(rows[index][4][mode] - 1)
            for mode in range(3)
        )
        for index in range(len(rows) - 1)
    )
    ok &= abs(rows[-1][2][0] - 1) < mp.mpf("0.00001")
    ok &= abs(rows[-1][2][1] - 1) < mp.mpf("0.0001")
    ok &= abs(rows[-1][4][0] - 1) < mp.mpf("0.003")
    ok &= abs(rows[-1][4][1] - 1) < mp.mpf("0.002")
    ok &= abs(rows[-1][4][2] - 1) < mp.mpf("0.004")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
