#!/usr/bin/env python3
"""Certificate for arbitrary-rank rare-contamination Gram limits."""

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


def barycentric_weights(nodes):
    """Return polynomial interpolation weights at distinct nodes."""
    return mp.matrix([
        1 / mp.fprod(
            nodes[index] - nodes[other]
            for other in range(len(nodes))
            if other != index
        )
        for index in range(len(nodes))
    ])


def contaminated_gram(size):
    """Exact four-output covariance Gram for the concrete mixture."""
    size = mp.mpf(size)
    epsilon = 1 / size
    atom = mp.exp(size)
    probability = mp.exp(-2 * CENTER * size) / size ** 6
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
    rare_scale = probability * atom ** (2 * CENTER) / epsilon ** 6
    return epsilon, gram, rare_scale


def main():
    derivative_cholesky = mp.cholesky(derivative_gram(CENTER, 4))
    node_matrix = vandermonde_matrix(NODES)
    node_cholesky = mp.cholesky(node_matrix * node_matrix.T)
    gaussian_coefficient = (
        derivative_cholesky[3, 3] * node_cholesky[3, 3]
    ) ** 2

    weights = barycentric_weights(NODES)
    unit_weights = weights / mp.norm(weights)
    exponential_vector = mp.matrix([mp.exp(node) for node in NODES])
    contamination_increment = (unit_weights.T * exponential_vector)[0] ** 2
    closed_form_increment = (mp.e - 1) ** 6 / 20
    contaminated_coefficient = gaussian_coefficient + contamination_increment

    rows = []
    for size in (32, 64, 128, 256, 512, 1024):
        epsilon, gram, rare_scale = contaminated_gram(size)
        smallest = descending_eigenvalues(gram)[-1]
        rows.append((
            size,
            rare_scale,
            smallest / epsilon ** 6,
            smallest / (contaminated_coefficient * epsilon ** 6),
        ))

    print("Arbitrary-rank rare-contamination Gram certificate")
    print(f"Gaussian coefficient {float(gaussian_coefficient):.15f}")
    print(f"barycentric increment {float(contamination_increment):.15f}")
    print(f"contaminated coefficient {float(contaminated_coefficient):.15f}")
    print(f"contaminated/Gaussian {float(contaminated_coefficient / gaussian_coefficient):.12f}")
    print("n rare-scale coefficient limit-ratio")
    for size, rare_scale, coefficient, limit_ratio in rows:
        print(
            f"{size:4d} {float(rare_scale):.12f} "
            f"{float(coefficient):.12f} {float(limit_ratio):.12f}"
        )

    ok = True
    ok &= abs(contamination_increment - closed_form_increment) < mp.mpf("1e-14")
    ok &= max(abs((weights.T * mp.matrix([
        node ** order for node in NODES
    ]))[0]) for order in range(3)) < mp.mpf("1e-14")
    ok &= abs(gaussian_coefficient - mp.mpf(
        "0.00260311732198459579545510842315"
    )) < mp.mpf("1e-14")
    ok &= all(abs(row[1] - 1) < mp.mpf("1e-14") for row in rows)
    ok &= all(
        abs(rows[index + 1][2] - contaminated_coefficient)
        < abs(rows[index][2] - contaminated_coefficient)
        for index in range(len(rows) - 1)
    )
    ok &= abs(rows[-1][3] - 1) < mp.mpf("0.004")
    ok &= contaminated_coefficient / gaussian_coefficient > 495
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
