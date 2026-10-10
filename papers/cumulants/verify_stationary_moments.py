"""Certificate for all stationary moments of a switched CIR variance.

For the row-vector convention used in the paper, the invariant component
moments r[p, i] = E[v**p 1_{Y=i}] obey a triangular sequence of killed-chain
linear systems.  This script checks that recursion against one large
polynomial-semigroup exponential, verifies the constant-coefficient Gamma
special case, and tests the explicit inverse-speed expansion through cubic
order on a nonreversible three-state chain.
"""
import math
import os
import sys

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "general"))
from effective_generator import group_inverse, stationary


DEGREE = 6
SPEEDS = (8, 16, 32, 64, 128)
ASYMPTOTIC_ORDER = 3


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


def fast_coefficients(q0, pi, c, kappa, variance, degree=DEGREE,
                      asymptotic_order=ASYMPTOTIC_ORDER):
    """Return r[p,n] in r_p(m)=sum_n m**(-n) r[p,n]+remainder."""
    inverse = group_inverse(q0)
    bar_kappa = pi @ kappa
    coefficients = np.zeros((degree + 1, asymptotic_order + 1, len(pi)))
    centered = np.zeros_like(coefficients)
    scalar = np.zeros((degree + 1, asymptotic_order + 1))
    coefficients[0, 0] = pi
    scalar[0, 0] = 1.0
    for inverse_order in range(asymptotic_order + 1):
        for moment_order in range(1, degree + 1):
            forcing = (
                moment_order * c
                + moment_order * (moment_order - 1) * variance / 2
            )
            scalar[moment_order, inverse_order] = (
                coefficients[moment_order - 1, inverse_order] @ forcing
                - moment_order
                * (centered[moment_order, inverse_order] @ kappa)
            ) / (moment_order * bar_kappa)
            coefficients[moment_order, inverse_order] = (
                centered[moment_order, inverse_order]
                + scalar[moment_order, inverse_order] * pi
            )
            if inverse_order < asymptotic_order:
                source = (
                    moment_order
                    * coefficients[moment_order, inverse_order]
                    * kappa
                    - coefficients[moment_order - 1, inverse_order] * forcing
                )
                assert abs(source.sum()) < 8e-13
                centered[moment_order, inverse_order + 1] = source @ inverse
                assert abs(centered[moment_order, inverse_order + 1].sum()) < 8e-13
    return coefficients, scalar, centered


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

    coefficients, scalar, centered = fast_coefficients(
        q0, pi, c, kappa, variance
    )
    residuals = {order: [] for order in range(ASYMPTOTIC_ORDER + 1)}
    component_residuals = {order: [] for order in range(ASYMPTOTIC_ORDER + 1)}
    for speed in SPEEDS:
        moments = stationary_moments(speed, q0, pi, c, kappa, variance)
        for inverse_order in range(ASYMPTOTIC_ORDER + 1):
            powers = speed ** -np.arange(inverse_order + 1, dtype=float)
            approximation = np.einsum(
                "pn,n->p", scalar[:, :inverse_order + 1], powers
            )
            component_approximation = np.einsum(
                "pni,n->pi", coefficients[:, :inverse_order + 1], powers
            )
            residuals[inverse_order].append(
                np.max(np.abs(moments.sum(axis=1)[1:] - approximation[1:]))
            )
            component_residuals[inverse_order].append(
                np.max(np.abs(moments[1:] - component_approximation[1:]))
            )
    scalar_rates = np.asarray(
        [observed_order(residuals[order]) for order in range(ASYMPTOTIC_ORDER + 1)]
    )
    component_rates = np.asarray(
        [observed_order(component_residuals[order])
         for order in range(ASYMPTOTIC_ORDER + 1)]
    )
    assert np.min(scalar_rates - np.arange(1, ASYMPTOTIC_ORDER + 2)) > -0.06
    assert np.min(component_rates - np.arange(1, ASYMPTOTIC_ORDER + 2)) > -0.06

    print("all-order stationary switched-CIR moment certificate")
    print("stationary law:", np.array2string(pi, precision=10))
    print("semigroup discrepancy:", f"{semigroup_error:.3e}")
    print("Gamma reduction discrepancy:", f"{gamma_error:.3e}")
    print("leading moments p=1..6:", np.array2string(scalar[1:, 0], precision=10))
    print("first coefficients p=1..6:", np.array2string(scalar[1:, 1], precision=10))
    print("scalar residual orders n=0..3:", np.array2string(scalar_rates, precision=6))
    print("component residual orders n=0..3:", np.array2string(component_rates, precision=6))
    print("largest cubic scalar residual at m=128:", f"{residuals[3][-1]:.3e}")


if __name__ == "__main__":
    main()
