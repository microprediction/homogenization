#!/usr/bin/env python3
"""Certificate for rare-contamination lifting of a Gram eigenvalue order."""

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
NODES = tuple(map(mp.mpf, (0, 1, 2, 3)))


def contaminated_gram(size):
    """Exact four-output covariance Gram at the penultimate tail scale."""
    size = mp.mpf(size)
    epsilon = 1 / size
    atom = mp.exp(size)
    probability = mp.exp(-2 * CENTER * size) / size ** 4
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
    rare_scale = probability * atom ** (2 * CENTER) / epsilon ** 4
    return epsilon, gram, rare_scale


def main():
    derivative_cholesky = mp.cholesky(derivative_gram(CENTER, 4))
    node_matrix = vandermonde_matrix(NODES)
    node_cholesky = mp.cholesky(node_matrix * node_matrix.T)
    gaussian_penultimate = (
        derivative_cholesky[2, 2] * node_cholesky[2, 2]
    ) ** 2

    penultimate_direction = mp.matrix([1, -1, -1, 1]) / 2
    last_direction = mp.matrix([-1, 3, -3, 1]) / mp.sqrt(20)
    exponential_vector = mp.matrix([mp.exp(node) for node in NODES])
    alpha = (penultimate_direction.T * exponential_vector)[0]
    beta = (last_direction.T * exponential_vector)[0]
    trace = gaussian_penultimate + alpha ** 2 + beta ** 2
    determinant = gaussian_penultimate * beta ** 2
    discriminant = mp.sqrt(trace ** 2 - 4 * determinant)
    predicted = ((trace + discriminant) / 2, (trace - discriminant) / 2)

    rows = []
    for size in (32, 64, 128, 256, 512, 1024):
        epsilon, gram, rare_scale = contaminated_gram(size)
        eigenvalues = descending_eigenvalues(gram)
        coefficients = (
            eigenvalues[2] / epsilon ** 4,
            eigenvalues[3] / epsilon ** 4,
        )
        rows.append((
            size,
            rare_scale,
            coefficients[0],
            coefficients[1],
            coefficients[0] / predicted[0],
            coefficients[1] / predicted[1],
        ))

    print("Translation-Gram eigenvalue-order lifting certificate")
    print(f"Gaussian penultimate coefficient {float(gaussian_penultimate):.15f}")
    print(f"projected exponential alpha {float(alpha):.15f}")
    print(f"barycentric exponential beta {float(beta):.15f}")
    print(f"lifted coefficients {float(predicted[0]):.15f} {float(predicted[1]):.15f}")
    print("n rare-scale upper lower upper-ratio lower-ratio")
    for row in rows:
        print(
            f"{row[0]:4d} {float(row[1]):.12f} "
            f"{float(row[2]):.12f} {float(row[3]):.12f} "
            f"{float(row[4]):.12f} {float(row[5]):.12f}"
        )

    polynomial_rows = [mp.matrix([
        node ** order / mp.factorial(order) for node in NODES
    ]) for order in range(3)]
    ok = True
    ok &= abs(alpha - (mp.e - 1) ** 2 * (mp.e + 1) / 2) < mp.mpf("1e-14")
    ok &= abs(beta - (mp.e - 1) ** 3 / mp.sqrt(20)) < mp.mpf("1e-14")
    ok &= abs((penultimate_direction.T * last_direction)[0]) < mp.mpf("1e-14")
    ok &= max(abs((polynomial_rows[order].T * penultimate_direction)[0])
              for order in range(2)) < mp.mpf("1e-14")
    ok &= max(abs((polynomial_rows[order].T * last_direction)[0])
              for order in range(3)) < mp.mpf("1e-14")
    ok &= abs(predicted[0] * predicted[1] - determinant) < mp.mpf("1e-14")
    ok &= all(abs(row[1] - 1) < mp.mpf("1e-14") for row in rows)
    ok &= all(
        abs(rows[index + 1][4] - 1) < abs(rows[index][4] - 1)
        and abs(rows[index + 1][5] - 1) < abs(rows[index][5] - 1)
        for index in range(len(rows) - 1)
    )
    ok &= abs(rows[-1][4] - 1) < mp.mpf("0.004")
    ok &= abs(rows[-1][5] - 1) < mp.mpf("0.0002")
    ok &= predicted[1] > 0
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
