#!/usr/bin/env python3
"""Certificate for coalescing translation-hull Gram conditioning."""

import math

import mpmath as mp


mp.mp.dps = 120
CENTER = mp.mpf(1)


def gaussian_absolute_moment(power):
    """Return E|G|^power for a standard Gaussian G."""
    return (
        mp.power(2, power / 2)
        * mp.gamma((power + 1) / 2)
        / mp.sqrt(mp.pi)
    )


def moment_derivative(power, order):
    """Differentiate the Gaussian absolute-moment function."""
    return mp.diff(gaussian_absolute_moment, power, order)


def covariance_gram(exponents):
    """Covariance Gram of Gaussian absolute powers."""
    return mp.matrix([
        [
            gaussian_absolute_moment(left + right)
            - gaussian_absolute_moment(left)
            * gaussian_absolute_moment(right)
            for right in exponents
        ]
        for left in exponents
    ])


def derivative_gram(center, size):
    """Gram of centered exponent derivatives through order size - 1."""
    return mp.matrix([
        [
            moment_derivative(2 * center, left + right)
            - moment_derivative(center, left)
            * moment_derivative(center, right)
            for right in range(size)
        ]
        for left in range(size)
    ])


def vandermonde(nodes):
    """Return the ordinary Vandermonde product."""
    product = mp.mpf(1)
    for left in range(len(nodes)):
        for right in range(left + 1, len(nodes)):
            product *= nodes[right] - nodes[left]
    return product


def descending_eigenvalues(matrix):
    """Return symmetric eigenvalues in descending order."""
    return list(reversed(list(mp.eigsy(matrix, eigvals_only=True))))


def main():
    pair_derivatives = derivative_gram(CENTER, 2)
    variance = pair_derivatives[0, 0]
    pair_coefficient = mp.det(pair_derivatives)
    smallest_coefficient = pair_coefficient / (2 * variance)
    condition_coefficient = 4 * variance ** 2 / pair_coefficient

    pair_rows = []
    for delta in map(mp.mpf, ("0.2", "0.1", "0.05", "0.025")):
        gram = covariance_gram((CENTER - delta / 2, CENTER + delta / 2))
        eigenvalues = descending_eigenvalues(gram)
        pair_rows.append((
            delta,
            mp.det(gram) / (pair_coefficient * delta ** 2),
            eigenvalues[1] / (smallest_coefficient * delta ** 2),
            (eigenvalues[0] / eigenvalues[1])
            * delta ** 2 / condition_coefficient,
        ))

    nodes = tuple(map(mp.mpf, (0, 1, 2, 3)))
    four_derivatives = derivative_gram(CENTER, 4)
    factorial_product = mp.mpf(math.prod(math.factorial(i) for i in range(4)))
    four_coefficient = (
        mp.det(four_derivatives)
        * (vandermonde(nodes) / factorial_product) ** 2
    )

    determinant_rows = []
    for epsilon in map(mp.mpf, ("0.05", "0.025", "0.0125", "0.00625", "0.003125")):
        exponents = tuple(CENTER + epsilon * node for node in nodes)
        gram = covariance_gram(exponents)
        determinant_rows.append((
            epsilon,
            mp.det(gram) / (four_coefficient * epsilon ** 12),
        ))

    coarse = mp.mpf("0.00625")
    fine = coarse / 2
    coarse_eigenvalues = descending_eigenvalues(covariance_gram(
        tuple(CENTER + coarse * node for node in nodes)
    ))
    fine_eigenvalues = descending_eigenvalues(covariance_gram(
        tuple(CENTER + fine * node for node in nodes)
    ))
    eigenvalue_exponents = [
        mp.log(coarse_eigenvalues[index] / fine_eigenvalues[index], 2)
        for index in range(4)
    ]

    print("Coalescing translation-hull Gram conditioning certificate")
    print(f"c=1 variance {float(variance):.12f}")
    print(f"two-output determinant coefficient {float(pair_coefficient):.12f}")
    print(f"smallest-eigenvalue coefficient {float(smallest_coefficient):.12f}")
    print(f"condition-number coefficient {float(condition_coefficient):.12f}")
    for delta, determinant_ratio, eigenvalue_ratio, condition_ratio in pair_rows:
        print(
            f"delta {float(delta):.3f}: determinant ratio "
            f"{float(determinant_ratio):.9f}, smallest eigenvalue ratio "
            f"{float(eigenvalue_ratio):.9f}, condition ratio "
            f"{float(condition_ratio):.9f}"
        )
    print(f"four-output leading coefficient {float(four_coefficient):.15f}")
    for epsilon, determinant_ratio in determinant_rows:
        print(
            f"epsilon {float(epsilon):.6f}: determinant ratio "
            f"{float(determinant_ratio):.9f}"
        )
    print("four-output eigenvalue exponents " + " ".join(
        f"{float(exponent):.6f}" for exponent in eigenvalue_exponents
    ))

    ok = True
    ok &= variance > mp.mpf("0.36")
    ok &= pair_coefficient > mp.mpf("0.024")
    ok &= all(
        abs(row[1] - 1) < abs(pair_rows[index - 1][1] - 1)
        and abs(row[2] - 1) < abs(pair_rows[index - 1][2] - 1)
        and abs(row[3] - 1) < abs(pair_rows[index - 1][3] - 1)
        for index, row in enumerate(pair_rows[1:], start=1)
    )
    ok &= abs(pair_rows[-1][1] - 1) < mp.mpf("0.0002")
    ok &= abs(pair_rows[-1][2] - 1) < mp.mpf("0.0004")
    ok &= abs(pair_rows[-1][3] - 1) < mp.mpf("0.0006")
    ok &= four_coefficient > mp.mpf("0.00005")
    ok &= all(
        abs(determinant_rows[index][1] - 1)
        < abs(determinant_rows[index - 1][1] - 1)
        for index in range(1, len(determinant_rows))
    )
    ok &= abs(determinant_rows[-1][1] - 1) < mp.mpf("0.006")
    ok &= abs(eigenvalue_exponents[0]) < mp.mpf("0.02")
    ok &= max(
        abs(eigenvalue_exponents[index] - 2 * index)
        for index in range(1, 4)
    ) < mp.mpf("0.01")
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
