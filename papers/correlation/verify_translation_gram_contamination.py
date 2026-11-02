#!/usr/bin/env python3
"""Certificate for the rare-contamination translation-Gram limit."""

import mpmath as mp

from verify_translation_gram_conditioning import (
    covariance_gram,
    derivative_gram,
    descending_eigenvalues,
    gaussian_absolute_moment,
)


mp.mp.dps = 100
CENTER = mp.mpf(1)


def contaminated_gram(size):
    """Exact covariance Gram for the concrete Gaussian mixture."""
    size = mp.mpf(size)
    epsilon = 1 / size
    atom = mp.exp(size)
    probability = mp.exp(-2 * CENTER * size) / size ** 2
    exponents = (CENTER - epsilon / 2, CENTER + epsilon / 2)
    means = mp.matrix([
        gaussian_absolute_moment(exponent) for exponent in exponents
    ])
    atom_difference = mp.matrix([
        atom ** exponents[index] - means[index] for index in range(2)
    ])
    gram = (
        (1 - probability) * covariance_gram(exponents)
        + probability * (1 - probability)
        * atom_difference * atom_difference.T
    )
    second_moment_excess = probability * (
        atom ** (2 * CENTER)
        - gaussian_absolute_moment(2 * CENTER)
    )
    return epsilon, gram, second_moment_excess


def main():
    derivatives = derivative_gram(CENTER, 2)
    gaussian_coefficient = mp.det(derivatives) / (2 * derivatives[0, 0])
    contamination_increment = mp.cosh(1) - 1
    contaminated_coefficient = gaussian_coefficient + contamination_increment

    rows = []
    for size in (16, 32, 64, 128, 256, 512, 1024, 2048):
        epsilon, gram, moment_excess = contaminated_gram(size)
        smallest = descending_eigenvalues(gram)[1]
        rows.append((
            size,
            moment_excess / epsilon ** 2,
            smallest / epsilon ** 2,
            smallest / (contaminated_coefficient * epsilon ** 2),
            smallest / (gaussian_coefficient * epsilon ** 2),
        ))

    print("Rare-contamination translation-Gram certificate")
    print(f"Gaussian coefficient {float(gaussian_coefficient):.12f}")
    print(f"contamination increment {float(contamination_increment):.12f}")
    print(f"contaminated coefficient {float(contaminated_coefficient):.12f}")
    print("n moment-scale coefficient limit-ratio Gaussian-ratio")
    for size, moment_scale, coefficient, limit_ratio, gaussian_ratio in rows:
        print(
            f"{size:4d} {float(moment_scale):.12f} "
            f"{float(coefficient):.12f} {float(limit_ratio):.12f} "
            f"{float(gaussian_ratio):.12f}"
        )

    ok = True
    ok &= abs(
        contaminated_coefficient - mp.mpf("0.5774508304064466588")
    ) < mp.mpf("1e-14")
    ok &= all(
        abs(rows[index + 1][2] - contaminated_coefficient)
        < abs(rows[index][2] - contaminated_coefficient)
        for index in range(len(rows) - 1)
    )
    ok &= abs(rows[1][1] - 1) < mp.mpf("1e-14")
    ok &= abs(rows[-1][3] - 1) < mp.mpf("0.001")
    ok &= rows[-1][4] > mp.mpf("16.7")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
