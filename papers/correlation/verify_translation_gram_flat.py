#!/usr/bin/env python3
"""Certificate for the translation-hull covariance Gram theorem."""

import math

import numpy as np

from verify_flat_map import continuous_integral


EXPONENTS = (1.0, 2.0, 3.0, 4.0)


def gaussian_absolute_moment(power):
    """Return E|G|^power for a standard Gaussian G."""
    return (
        2.0 ** (0.5 * power)
        * math.gamma(0.5 * (power + 1.0))
        / math.sqrt(math.pi)
    )


def telegraph_absolute_moment(speed, power):
    """Exact absolute moment of the symmetric telegraph average."""
    continuous = continuous_integral(
        speed,
        lambda position: position ** power,
        absolute_tolerance=0.0,
    )
    return continuous + math.exp(-speed)


def covariance_gram(exponents, moment):
    """Covariance Gram of powers under the supplied moment function."""
    return np.array([
        [
            moment(left + right) - moment(left) * moment(right)
            for right in exponents
        ]
        for left in exponents
    ])


def normalized_telegraph_gram(speed, exponents):
    """Componentwise-normalized exact telegraph covariance matrix."""
    moments = {
        power: telegraph_absolute_moment(speed, power)
        for power in set(exponents) | {
            left + right for left in exponents for right in exponents
        }
    }
    return np.array([
        [
            speed ** (0.5 * (left + right))
            * (moments[left + right] - moments[left] * moments[right])
            for right in exponents
        ]
        for left in exponents
    ])


def main():
    gaussian = covariance_gram(EXPONENTS, gaussian_absolute_moment)
    eigenvalues = np.linalg.eigvalsh(gaussian)
    determinant = np.linalg.det(gaussian)

    pair = gaussian[:2, :2]
    pair_determinant = np.linalg.det(pair)
    exact_pair_determinant = 2.0 - 6.0 / math.pi

    rows = []
    for speed in (64.0, 256.0, 1024.0, 4096.0):
        exact = normalized_telegraph_gram(speed, EXPONENTS)
        exact_eigenvalues = np.linalg.eigvalsh(exact)
        rows.append((
            speed,
            np.max(np.abs(exact / gaussian - 1.0)),
            exact_eigenvalues[0] / eigenvalues[0],
            np.linalg.det(exact) / determinant,
        ))

    print("Translation-hull covariance Gram certificate")
    print("Gaussian rank-one feature, powers 1, 2, 3, 4")
    print(f"two-output determinant {pair_determinant:.12f}")
    print(f"two-output exact formula {exact_pair_determinant:.12f}")
    print(f"four-output determinant {determinant:.12f}")
    print(f"four-output smallest eigenvalue {eigenvalues[0]:.12f}")
    for speed, entry_error, eigen_ratio, determinant_ratio in rows:
        print(
            f"speed {speed:4.0f}: largest relative entry error "
            f"{entry_error:.9f}, smallest eigenvalue ratio "
            f"{eigen_ratio:.9f}, determinant ratio {determinant_ratio:.9f}"
        )

    ok = True
    ok &= abs(pair[0, 0] - (1.0 - 2.0 / math.pi)) < 5e-14
    ok &= abs(pair[0, 1] - math.sqrt(2.0 / math.pi)) < 5e-14
    ok &= abs(pair[1, 1] - 2.0) < 5e-14
    ok &= abs(pair_determinant - exact_pair_determinant) < 5e-14
    ok &= determinant > 0.0075
    ok &= eigenvalues[0] > 0.0012
    ok &= all(
        rows[index + 1][1] < rows[index][1]
        and rows[index + 1][2] > rows[index][2]
        and rows[index + 1][3] > rows[index][3]
        for index in range(len(rows) - 1)
    )
    ok &= rows[-1][1] < 0.0021
    ok &= rows[-1][2] > 0.998
    ok &= rows[-1][3] > 0.992
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
