"""Certificate for all stationary moments of a switched CIR variance.

For the row-vector convention used in the paper, the invariant component
moments r[p, i] = E[v**p 1_{Y=i}] obey a triangular sequence of killed-chain
linear systems.  This script checks that recursion against one large
polynomial-semigroup exponential, verifies the constant-coefficient Gamma
special case, and tests the explicit first inverse-speed coefficient on a
nonreversible three-state chain.
"""
import math
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "general"))
from effective_generator import group_inverse, stationary


DEGREE = 6
SPEEDS = (16, 32, 64, 128, 256)


def stationary_moments(speed, q0, pi, c, kappa, variance, degree=DEGREE):
    """Return rows r_p,i = E[v^p 1_{Y=i}], p=0,...,degree."""
    size = len(pi)
    rows = [np.asarray(pi, float)]
    for order in range(1, degree + 1):
        forcing = order * c + order * (order - 1) * variance / 2
        killed = order * np.diag(kappa) - speed * q0
        rows.append(np.linalg.solve(killed.T, rows[-1] * forcing))
        assert np.min(rows[-1]) >= -2e-14
        assert np.linalg.norm(rows[-1] @ killed - rows[-2] * forcing) < 3e-13
    return np.asarray(rows)


def moment_semigroup(speed, q0, initial_regime, v0, c, kappa, variance,
                     degree=DEGREE, maturity=80.0):
    """Evolve every regime-resolved monomial moment in one block exponential."""
    states = len(initial_regime)
    generator = np.zeros(((degree + 1) * states, (degree + 1) * states))
    for order in range(degree + 1):
        block = speed * q0 - order * np.diag(kappa)
        lo, hi = order * states, (order + 1) * states
        generator[lo:hi, lo:hi] = block
        if order < degree:
            forcing = (order + 1) * c + order * (order + 1) * variance / 2
            next_lo, next_hi = hi, hi + states
            generator[lo:hi, next_lo:next_hi] = np.diag(forcing)
    initial = np.concatenate([initial_regime * v0**order for order in range(degree + 1)])
    return (initial @ expm(maturity * generator)).reshape(degree + 1, states)


def fast_coefficients(q0, pi, c, kappa, variance, degree=DEGREE):
    """Leading averaged moments and their exact first inverse-speed coefficient."""
    inverse = group_inverse(q0)
    bar_kappa = pi @ kappa
    leading = [1.0]
    correction = [0.0]
    centered = [np.zeros_like(pi)]
    for order in range(1, degree + 1):
        forcing = order * c + order * (order - 1) * variance / 2
        next_leading = leading[-1] * (pi @ forcing) / (order * bar_kappa)
        source = (
            next_leading * order * pi * kappa
            - leading[-1] * pi * forcing
        )
        next_centered = source @ inverse
        assert abs(source.sum()) < 2e-13
        assert abs(next_centered.sum()) < 2e-13
        next_correction = (
            centered[-1] @ forcing
            + correction[-1] * (pi @ forcing)
            - order * (next_centered @ kappa)
        ) / (order * bar_kappa)
        leading.append(next_leading)
        centered.append(next_centered)
        correction.append(next_correction)
    return np.asarray(leading), np.asarray(correction), np.asarray(centered)


def gamma_moments(c, kappa, variance, degree=DEGREE):
    values = [1.0]
    for order in range(1, degree + 1):
        values.append(values[-1] * (c + (order - 1) * variance / 2) / kappa)
    return np.asarray(values)


def observed_order(errors):
    return math.log(errors[-2] / errors[-1], 2)


def main():
    q0 = np.array(
        [
            [-1.4, 1.0, 0.4],
            [0.2, -1.1, 0.9],
            [0.7, 0.3, -1.0],
        ]
    )
    pi = stationary(q0)
    c = np.array([0.060, 0.120, 0.090])
    kappa = np.array([0.70, 1.40, 0.95])
    variance = np.array([0.035, 0.080, 0.050])
    assert np.min(2 * c - variance) > 0

    exact = stationary_moments(16, q0, pi, c, kappa, variance)
    evolved = moment_semigroup(16, q0, np.array([0.8, 0.1, 0.1]), 0.17,
                               c, kappa, variance)
    semigroup_error = np.max(np.abs(exact - evolved))
    assert semigroup_error < 2e-12

    common_c, common_kappa, common_variance = 0.08, 1.2, 0.04
    common = stationary_moments(
        7, q0, pi,
        np.full(3, common_c),
        np.full(3, common_kappa),
        np.full(3, common_variance),
    ).sum(axis=1)
    gamma_error = np.max(
        np.abs(common - gamma_moments(common_c, common_kappa, common_variance))
    )
    assert gamma_error < 2e-14

    leading, correction, centered = fast_coefficients(
        q0, pi, c, kappa, variance
    )
    residuals = []
    component_residuals = []
    for speed in SPEEDS:
        moments = stationary_moments(speed, q0, pi, c, kappa, variance)
        residuals.append(
            np.abs(moments.sum(axis=1) - leading - correction / speed)
        )
        component_residuals.append(
            np.max(
                np.abs(
                    moments
                    - leading[:, None] * pi
                    - (centered + correction[:, None] * pi) / speed
                )
            )
        )
    residuals = np.asarray(residuals)
    rates = [observed_order(residuals[:, order]) for order in range(1, DEGREE + 1)]
    component_rate = observed_order(component_residuals)
    assert min(rates) > 1.98
    assert component_rate > 1.98

    print("all-order stationary switched-CIR moment certificate")
    print("stationary law:", np.array2string(pi, precision=10))
    print("semigroup discrepancy:", f"{semigroup_error:.3e}")
    print("Gamma reduction discrepancy:", f"{gamma_error:.3e}")
    print("leading moments p=1..6:", np.array2string(leading[1:], precision=10))
    print("first coefficients p=1..6:", np.array2string(correction[1:], precision=10))
    print("scalar residual orders p=1..6:", np.array2string(np.asarray(rates), precision=6))
    print("component residual order:", f"{component_rate:.6f}")
    print("largest m=256 scalar residual:", f"{np.max(residuals[-1, 1:]):.3e}")


if __name__ == "__main__":
    main()
