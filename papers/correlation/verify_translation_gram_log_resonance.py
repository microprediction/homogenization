#!/usr/bin/env python3
"""Certificate for the logarithmic layer around a rational power."""

import mpmath as mp

import verify_translation_gram_rational_lattice as base


mp.mp.dps = 260
RATIONAL_POWER = mp.mpf(3) / 2
NOMINAL_RATIO = mp.mpf(3) / 2
THETA = mp.log(mp.mpf(3) / 2)
EFFECTIVE_RATIO = NOMINAL_RATIO * mp.exp(-THETA)


def predictions():
    """Return the rational-layer coefficients at the effective ratio."""
    dimension = len(base.NODES)
    derivative_cholesky = mp.cholesky(base.derivative_gram(base.CENTER, dimension))
    polynomial_rows = [
        [node ** order / mp.factorial(order) for node in base.NODES]
        for order in range(base.LEVEL + base.POWER_NUMERATOR + 1)
    ]
    tail_rows = [
        [node ** order / mp.factorial(order) * mp.exp(base.TAIL_CENTER * node)
         for node in base.NODES]
        for order in range(base.TAIL_COUNT)
    ]
    label_rows = [[offset ** order for offset in base.TAIL_OFFSETS]
                  for order in range(base.TAIL_COUNT)]
    label_residuals = [
        base.weighted_residual(label_rows[order], label_rows[:order])
        for order in range(base.TAIL_COUNT)
    ]
    label_coefficients = tuple(
        (residual.T * mp.diag(base.TAIL_WEIGHTS) * residual)[0]
        for residual in label_residuals
    )

    gaussian_one_projector = base.orthogonal_projector(
        polynomial_rows[:base.LEVEL + 1] + tail_rows[:1]
    )
    gaussian_one = gaussian_one_projector * mp.matrix(polynomial_rows[base.LEVEL + 1])
    gaussian_one_coefficient = (
        derivative_cholesky[base.LEVEL + 1, base.LEVEL + 1] ** 2
        * (gaussian_one.T * gaussian_one)[0]
    )

    tail_one_projector = base.orthogonal_projector(
        polynomial_rows[:base.LEVEL + 2] + tail_rows[:1]
    )
    tail_one = tail_one_projector * mp.matrix(tail_rows[1])
    tail_one_coefficient = (
        label_coefficients[1] * EFFECTIVE_RATIO ** 2
        * (tail_one.T * tail_one)[0]
    )

    gaussian_two_projector = base.orthogonal_projector(
        polynomial_rows[:base.LEVEL + 2] + tail_rows[:2]
    )
    gaussian_two = gaussian_two_projector * mp.matrix(polynomial_rows[base.LEVEL + 2])
    gaussian_two_coefficient = (
        derivative_cholesky[base.LEVEL + 2, base.LEVEL + 2] ** 2
        * (gaussian_two.T * gaussian_two)[0]
    )

    collision_projector = base.orthogonal_projector(
        polynomial_rows[:base.LEVEL + 3] + tail_rows[:2]
    )
    collision_gaussian = collision_projector * mp.matrix(
        polynomial_rows[base.LEVEL + 3]
    )
    collision_tail = collision_projector * mp.matrix(tail_rows[2])
    collision_block = (
        derivative_cholesky[base.LEVEL + 3, base.LEVEL + 3] ** 2
        * collision_gaussian * collision_gaussian.T
        + label_coefficients[2] * EFFECTIVE_RATIO ** 4
        * collision_tail * collision_tail.T
    )
    collision_coefficients = tuple(base.descending_eigenvalues(collision_block)[:2])
    return (
        gaussian_one_coefficient,
        tail_one_coefficient,
        gaussian_two_coefficient,
    ) + collision_coefficients


def moving_gram(size):
    """Evaluate the exact Gram through its moving-power parametrization."""
    size = mp.mpf(size)
    epsilon = 1 / size
    log_scale = mp.log(1 / epsilon)
    alpha = RATIONAL_POWER + THETA / log_scale
    delta = NOMINAL_RATIO * epsilon ** alpha
    effective_ratio = delta / epsilon ** RATIONAL_POWER
    slopes = tuple(base.TAIL_CENTER + offset * delta for offset in base.TAIL_OFFSETS)
    exponents = tuple(base.CENTER + epsilon * node for node in base.NODES)
    base_gram = base.covariance_gram(exponents)
    base_means = mp.matrix(
        [base.gaussian_absolute_moment(exponent) for exponent in exponents]
    )
    probabilities = tuple(
        weight * epsilon ** (2 * base.LEVEL)
        * mp.exp(-2 * base.CENTER * slope / epsilon)
        for weight, slope in zip(base.TAIL_WEIGHTS, slopes)
    )
    atoms = tuple(
        mp.matrix([mp.exp(slope * exponent / epsilon) for exponent in exponents])
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
        probability * mp.exp(2 * base.CENTER * slope / epsilon)
        / epsilon ** (2 * base.LEVEL)
        for probability, slope in zip(probabilities, slopes)
    )
    return epsilon, alpha, effective_ratio, gram, rare_scales


def main():
    predicted = predictions()
    scale_offsets = (mp.mpf(1), RATIONAL_POWER, mp.mpf(2), mp.mpf(3), mp.mpf(3))
    rows = []
    for size in (32, 128, 512, 2048, 8192):
        epsilon, alpha, effective_ratio, gram, rare_scales = moving_gram(size)
        eigenvalues = base.descending_eigenvalues(gram)
        values = tuple(
            eigenvalues[base.LEVEL + 2 + index]
            / epsilon ** (2 * (base.LEVEL + scale_offsets[index]))
            for index in range(len(predicted))
        )
        ratios = tuple(value / coefficient
                       for value, coefficient in zip(values, predicted))
        rows.append((size, alpha, effective_ratio, rare_scales, ratios))

    print("Translation-Gram logarithmic rational-resonance certificate")
    print(f"theta {float(THETA):.15f}")
    print(f"nominal ratio {float(NOMINAL_RATIO):.15f}")
    print(f"effective ratio {float(EFFECTIVE_RATIO):.15f}")
    print("predicted coefficients "
          + " ".join(f"{float(value):.15f}" for value in predicted))
    print("n alpha_n effective_ratio normalized_ratios")
    for size, alpha, effective_ratio, _, ratios in rows:
        printed = (alpha, effective_ratio) + ratios
        print(f"{size:5d} " + " ".join(f"{float(value):.12f}" for value in printed))

    ok = True
    ok &= abs(EFFECTIVE_RATIO - 1) < mp.mpf("1e-14")
    ok &= all(abs(row[2] - EFFECTIVE_RATIO) < mp.mpf("1e-14") for row in rows)
    ok &= all(row[1] > RATIONAL_POWER for row in rows)
    ok &= all(rows[index + 1][1] < rows[index][1]
              for index in range(len(rows) - 1))
    ok &= all(
        all(abs(scale - weight) < mp.mpf("1e-14")
            for scale, weight in zip(rare_scales, base.TAIL_WEIGHTS))
        for _, _, _, rare_scales, _ in rows
    )
    ok &= all(value > 0 for value in predicted)
    ok &= all(abs(rows[-1][4][mode] - 1) < abs(rows[1][4][mode] - 1)
              for mode in range(len(predicted)))
    ok &= all(abs(ratio - 1) < mp.mpf("0.04") for ratio in rows[-1][4])
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
