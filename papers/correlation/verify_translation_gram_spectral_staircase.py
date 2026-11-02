#!/usr/bin/env python3
"""Certificate for the intermediate-scale rare-tail spectral staircase."""

import mpmath as mp

from verify_translation_gram_conditioning import (
    covariance_gram,
    derivative_gram,
    descending_eigenvalues,
    gaussian_absolute_moment,
    vandermonde_matrix,
)


mp.mp.dps = 180
CENTER = mp.mpf(1)
NODES = tuple(map(mp.mpf, (-2, -1, 0, 1, 2)))
LEVEL = 2


def orthogonal_projector(rows):
    """Project onto the orthogonal complement of the supplied row vectors."""
    matrix = mp.matrix(rows)
    return mp.eye(len(NODES)) - matrix.T * (matrix * matrix.T) ** -1 * matrix


def contaminated_gram(size):
    """Exact five-output covariance Gram at the level-two tail scale."""
    size = mp.mpf(size)
    epsilon = 1 / size
    atom = mp.exp(size)
    probability = mp.exp(-2 * CENTER * size) / size ** (2 * LEVEL)
    exponents = tuple(CENTER + epsilon * node for node in NODES)
    means = mp.matrix([
        gaussian_absolute_moment(exponent) for exponent in exponents
    ])
    atom_difference = mp.matrix([
        atom ** exponents[index] - means[index]
        for index in range(len(NODES))
    ])
    gram = (
        (1 - probability) * covariance_gram(exponents)
        + probability * (1 - probability)
        * atom_difference * atom_difference.T
    )
    rare_scale = (
        probability * atom ** (2 * CENTER)
        / epsilon ** (2 * LEVEL)
    )
    return epsilon, gram, rare_scale


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

    exponential_vector = mp.matrix([mp.exp(node) for node in NODES])
    projected_exponential = projector * exponential_vector
    alpha = (level_direction.T * projected_exponential)[0]
    projected_norm_squared = (
        projected_exponential.T * projected_exponential
    )[0]
    beta_squared = projected_norm_squared - alpha ** 2
    gaussian_level = gaussian_coefficients[LEVEL]
    trace = gaussian_level + projected_norm_squared
    determinant = gaussian_level * beta_squared
    discriminant = mp.sqrt(trace ** 2 - 4 * determinant)
    predicted = ((trace + discriminant) / 2, (trace - discriminant) / 2)
    limiting_block = (
        gaussian_level * level_direction * level_direction.T
        + projected_exponential * projected_exponential.T
    )
    limiting_eigenvalues = descending_eigenvalues(limiting_block)

    divided_difference_direction = (
        mp.matrix([-1, 3, -3, 1, 0]) / mp.sqrt(20)
    )
    divided_difference = (
        divided_difference_direction.T * exponential_vector
    )[0]

    rows = []
    for size in (32, 64, 128, 256, 512, 1024):
        epsilon, gram, rare_scale = contaminated_gram(size)
        eigenvalues = descending_eigenvalues(gram)
        leading_ratios = (
            eigenvalues[0] / gaussian_coefficients[0],
            eigenvalues[1] / epsilon ** 2 / gaussian_coefficients[1],
        )
        lifted = (
            eigenvalues[2] / epsilon ** 4,
            eigenvalues[3] / epsilon ** 4,
        )
        rows.append((
            size,
            rare_scale,
            leading_ratios[0],
            leading_ratios[1],
            lifted[0],
            lifted[1],
            lifted[0] / predicted[0],
            lifted[1] / predicted[1],
        ))

    print("Translation-Gram rare-tail spectral-staircase certificate")
    print(f"Gaussian level-two coefficient {float(gaussian_level):.15f}")
    print(f"projected exponential norm squared {float(projected_norm_squared):.15f}")
    print(f"orthogonal remainder squared {float(beta_squared):.15f}")
    print(f"divided-difference witness {float(divided_difference):.15f}")
    print(f"lifted coefficients {float(predicted[0]):.15f} {float(predicted[1]):.15f}")
    print("n rare-scale first-ratio second-ratio upper lower upper-ratio lower-ratio")
    for row in rows:
        print(
            f"{row[0]:4d} {float(row[1]):.12f} "
            f"{float(row[2]):.12f} {float(row[3]):.12f} "
            f"{float(row[4]):.12f} {float(row[5]):.12f} "
            f"{float(row[6]):.12f} {float(row[7]):.12f}"
        )

    expected_direction = mp.matrix([2, -1, -2, -1, 2]) / mp.sqrt(14)
    ok = True
    ok &= mp.norm(level_direction - expected_direction) < mp.mpf("1e-14")
    ok &= max(
        abs((mp.matrix(polynomial_rows[order]).T * level_direction)[0])
        for order in range(LEVEL)
    ) < mp.mpf("1e-14")
    ok &= max(
        abs((mp.matrix(polynomial_rows[order]).T
             * divided_difference_direction)[0])
        for order in range(LEVEL + 1)
    ) < mp.mpf("1e-14")
    ok &= abs(
        divided_difference
        - (mp.e - 1) ** 3 / (mp.e ** 2 * mp.sqrt(20))
    ) < mp.mpf("1e-14")
    ok &= beta_squared > divided_difference ** 2 > 0
    ok &= abs(predicted[0] * predicted[1] - determinant) < mp.mpf("1e-14")
    ok &= max(
        abs(limiting_eigenvalues[index] - predicted[index])
        for index in range(2)
    ) < mp.mpf("1e-14")
    ok &= max(abs(value) for value in limiting_eigenvalues[2:]) < mp.mpf("1e-14")
    ok &= all(abs(row[1] - 1) < mp.mpf("1e-14") for row in rows)
    ok &= all(
        abs(rows[index + 1][6] - 1) < abs(rows[index][6] - 1)
        and abs(rows[index + 1][7] - 1) < abs(rows[index][7] - 1)
        for index in range(len(rows) - 1)
    )
    ok &= abs(rows[-1][2] - 1) < mp.mpf("0.00001")
    ok &= abs(rows[-1][3] - 1) < mp.mpf("0.0001")
    ok &= abs(rows[-1][6] - 1) < mp.mpf("0.003")
    ok &= abs(rows[-1][7] - 1) < mp.mpf("0.0002")
    ok &= predicted[1] > 0
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
