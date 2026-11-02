#!/usr/bin/env python3
"""Certificate for the moment-buffer translation-Gram criterion."""

import mpmath as mp

from verify_translation_gram_conditioning import (
    covariance_gram,
    derivative_gram,
    descending_eigenvalues,
    gaussian_absolute_moment,
)


mp.mp.dps = 100
CENTER = mp.mpf(1)
BUFFER = mp.mpf("0.25")


def buffered_mixture(size):
    """Return the exact Gram and tail diagnostics for a buffered mixture."""
    size = mp.mpf(size)
    epsilon = 1 / size
    atom = mp.exp(size)
    probability = mp.exp(-2 * (CENTER + BUFFER) * size)
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
    top_moment_atom = probability * atom ** (2 * (CENTER + BUFFER))
    derivative_tail = probability * atom ** (2 * CENTER) * size ** 2
    return epsilon, gram, top_moment_atom, derivative_tail


def main():
    derivatives = derivative_gram(CENTER, 2)
    gaussian_coefficient = mp.det(derivatives) / (2 * derivatives[0, 0])
    rows = []
    for size in (16, 20, 24, 32, 40, 48, 64):
        epsilon, gram, top_moment_atom, derivative_tail = buffered_mixture(
            size
        )
        smallest = descending_eigenvalues(gram)[1]
        rows.append((
            size,
            top_moment_atom,
            derivative_tail,
            smallest / epsilon ** 2,
            smallest / (gaussian_coefficient * epsilon ** 2),
        ))

    print("Moment-buffer translation-Gram certificate")
    print(f"Gaussian coefficient {float(gaussian_coefficient):.12f}")
    print("n top-moment-atom derivative-tail coefficient Gaussian-ratio")
    for size, top_moment_atom, derivative_tail, coefficient, ratio in rows:
        print(
            f"{size:3d} {float(top_moment_atom):.12f} "
            f"{float(derivative_tail):.12e} "
            f"{float(coefficient):.12f} {float(ratio):.12f}"
        )

    ok = True
    ok &= all(
        abs(row[1] - 1) < mp.mpf("1e-14")
        for row in rows
    )
    ok &= all(
        rows[index + 1][2] < rows[index][2]
        for index in range(len(rows) - 1)
    )
    ok &= rows[0][4] > 2
    ok &= abs(rows[-1][4] - 1) < mp.mpf("0.0002")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
