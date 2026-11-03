#!/usr/bin/env python3
"""Certificate for a rational-power multi-tail spectral merge."""

import mpmath as mp

from verify_translation_gram_conditioning import (
    covariance_gram, derivative_gram, descending_eigenvalues,
    gaussian_absolute_moment, vandermonde_matrix,
)


mp.mp.dps = 260
CENTER = mp.mpf(1)
LEVEL = 1
TAIL_CENTER = mp.mpf(1)
TAIL_COUNT = 3
POWER_NUMERATOR = 3
POWER_DENOMINATOR = 2
CRITICAL_RATIO = mp.mpf("1.25")
NODES = tuple(mp.mpf(value) for value in (-4, -3, -2, -1, 0, 1, 2, 3))
TAIL_WEIGHTS = tuple(mp.mpf(value) for value in (1, 2, 3))
TAIL_OFFSETS = tuple(mp.mpf(value) for value in (
    mp.mpf(-4) / 3, mp.mpf(-1) / 3, mp.mpf(2) / 3,
))


def orthogonal_projector(rows):
    """Project onto the orthogonal complement of the supplied rows."""
    matrix = mp.matrix(rows)
    return mp.eye(len(NODES)) - matrix.T * (matrix * matrix.T) ** -1 * matrix


def weighted_residual(values, previous):
    """Return the weighted residual from earlier label vectors."""
    weights = mp.diag(TAIL_WEIGHTS)
    vector = mp.matrix(values)
    if not previous:
        return vector
    matrix = mp.matrix(previous)
    return vector - matrix.T * (matrix * weights * matrix.T) ** -1 * matrix * weights * vector


def contaminated_gram(size):
    """Exact three-tail Gram with delta proportional to epsilon^(3/2)."""
    size = mp.mpf(size)
    epsilon = 1 / size
    alpha = mp.mpf(POWER_NUMERATOR) / POWER_DENOMINATOR
    delta = CRITICAL_RATIO * epsilon ** alpha
    slopes = tuple(TAIL_CENTER + offset * delta for offset in TAIL_OFFSETS)
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
    return epsilon, gram, rare_scales


def main():
    dimension = len(NODES)
    alpha = mp.mpf(POWER_NUMERATOR) / POWER_DENOMINATOR
    final_gaussian_order = POWER_NUMERATOR
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
        for order in range(LEVEL + final_gaussian_order + 1)
    ]
    tail_rows = [
        [node ** order / mp.factorial(order) * mp.exp(TAIL_CENTER * node)
         for node in NODES]
        for order in range(TAIL_COUNT)
    ]
    label_rows = [[offset ** order for offset in TAIL_OFFSETS]
                  for order in range(TAIL_COUNT)]
    label_residuals = [weighted_residual(label_rows[order], label_rows[:order])
                       for order in range(TAIL_COUNT)]
    label_coefficients = tuple(
        (residual.T * mp.diag(TAIL_WEIGHTS) * residual)[0]
        for residual in label_residuals
    )

    gaussian_one_projector = orthogonal_projector(
        polynomial_rows[:LEVEL + 1] + tail_rows[:1]
    )
    gaussian_one_direction = gaussian_one_projector * mp.matrix(polynomial_rows[LEVEL + 1])
    gaussian_one_coefficient = (
        derivative_cholesky[LEVEL + 1, LEVEL + 1] ** 2
        * (gaussian_one_direction.T * gaussian_one_direction)[0]
    )

    tail_one_projector = orthogonal_projector(
        polynomial_rows[:LEVEL + 2] + tail_rows[:1]
    )
    tail_one_direction = tail_one_projector * mp.matrix(tail_rows[1])
    tail_one_coefficient = (
        label_coefficients[1] * CRITICAL_RATIO ** 2
        * (tail_one_direction.T * tail_one_direction)[0]
    )

    gaussian_two_projector = orthogonal_projector(
        polynomial_rows[:LEVEL + 2] + tail_rows[:2]
    )
    gaussian_two_direction = gaussian_two_projector * mp.matrix(polynomial_rows[LEVEL + 2])
    gaussian_two_coefficient = (
        derivative_cholesky[LEVEL + 2, LEVEL + 2] ** 2
        * (gaussian_two_direction.T * gaussian_two_direction)[0]
    )

    collision_projector = orthogonal_projector(
        polynomial_rows[:LEVEL + 3] + tail_rows[:2]
    )
    collision_gaussian = collision_projector * mp.matrix(polynomial_rows[LEVEL + 3])
    collision_tail = collision_projector * mp.matrix(tail_rows[2])
    collision_block = (
        derivative_cholesky[LEVEL + 3, LEVEL + 3] ** 2
        * collision_gaussian * collision_gaussian.T
        + label_coefficients[2] * CRITICAL_RATIO ** 4
        * collision_tail * collision_tail.T
    )
    collision_coefficients = tuple(descending_eigenvalues(collision_block)[:2])
    predicted = (
        gaussian_one_coefficient,
        tail_one_coefficient,
        gaussian_two_coefficient,
    ) + collision_coefficients

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

    scale_offsets = (mp.mpf(1), alpha, mp.mpf(2), mp.mpf(3), mp.mpf(3))
    rows = []
    for size in (32, 128, 512, 2048, 8192):
        epsilon, gram, rare_scales = contaminated_gram(size)
        eigenvalues = descending_eigenvalues(gram)
        primary = tuple(
            eigenvalues[LEVEL + index] / epsilon ** (2 * LEVEL)
            for index in range(2)
        )
        values = tuple(
            eigenvalues[LEVEL + 2 + index]
            / epsilon ** (2 * (LEVEL + scale_offsets[index]))
            for index in range(len(predicted))
        )
        ratios = tuple(value / coefficient
                       for value, coefficient in zip(values, predicted))
        rows.append((size, rare_scales, primary, values, ratios))

    print("Translation-Gram rational-power spectral-merge certificate")
    print("label residual coefficients "
          + " ".join(f"{float(value):.15f}" for value in label_coefficients))
    print("primary coefficients "
          + " ".join(f"{float(value):.15f}" for value in primary_coefficients))
    print(f"gaussian-1 coefficient {float(gaussian_one_coefficient):.15f}")
    print(f"tail-1 coefficient {float(tail_one_coefficient):.15f}")
    print(f"gaussian-2 coefficient {float(gaussian_two_coefficient):.15f}")
    print("collision coefficients "
          + " ".join(f"{float(value):.15f}" for value in collision_coefficients))
    print("confluent evaluation minimum eigenvalue "
          f"{float(evaluation_eigenvalues[-1]):.15f}")
    print("n primary-1 primary-2 merged coefficients and ratios")
    for size, _, primary, values, ratios in rows:
        printed = primary + values + ratios
        print(f"{size:5d} " + " ".join(f"{float(value):.12f}" for value in printed))

    ok = True
    ok &= abs(sum(weight * offset for weight, offset
                  in zip(TAIL_WEIGHTS, TAIL_OFFSETS))) < mp.mpf("1e-14")
    ok &= evaluation_eigenvalues[-1] > mp.mpf("0.00000000001")
    ok &= all(value > 0 for value in predicted)
    ok &= all(
        all(abs(scale - weight) < mp.mpf("1e-14")
            for scale, weight in zip(rare_scales, TAIL_WEIGHTS))
        for _, rare_scales, _, _, _ in rows
    )
    ok &= all(abs(rows[-1][4][mode] - 1) < abs(rows[1][4][mode] - 1)
              for mode in range(len(predicted)))
    ok &= all(abs(rows[-1][2][index] / primary_coefficients[index] - 1)
              < mp.mpf("0.02") for index in range(2))
    ok &= all(abs(ratio - 1) < mp.mpf("0.04") for ratio in rows[-1][4])
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
