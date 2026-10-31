"""Certificate for the vector finite-chain second-order price expansion.

The exact benchmarks are matrix exponentials for exponential payoffs and the
genuinely mixed payoff x[0]^2*x[1]^2.  Directional contractions are also
compared with the scalar theorem.  Ends with PASS or FAIL.
"""
import math
import os
import sys

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from correlation import (scalar_observable_price_expansion,
                         vector_observable_price_expansion)


Q = np.array([[-3.0, 2.0, 1.0],
              [1.0, -4.0, 3.0],
              [2.0, 1.0, -3.0]])
VALUES = np.array([[-0.4, 0.7],
                   [0.2, -0.3],
                   [0.9, 0.1]])
MATURITY = 0.8
SPEEDS = [4, 8, 16, 32, 64]
THETAS = [np.array([-1.2, 0.4]),
          np.array([0.7, -0.8]),
          np.array([1.1, 0.9]),
          np.array([-0.4, -1.3])]
AFFINE_MAPS = [
    (np.array([[1.7, -0.4], [0.3, 0.8]]), np.array([0.2, -0.6])),
    (np.array([[0.6, 1.1], [-0.9, 0.7]]), np.array([-0.5, 0.4])),
    (np.array([[2.0, 0.3], [0.1, 0.45]]), np.array([1.2, -0.8])),
]
RECTANGULAR_MAPS = [
    (np.array([[1.3], [-0.7]]), np.array([0.4]), np.array([4])),
    (np.array([[1.0, -0.4, 0.8],
               [0.3, 1.2, -0.5]]),
     np.array([0.2, -0.6, 0.9]), np.array([2, 1, 1])),
]


def stationary(Q):
    states = len(Q)
    return np.linalg.solve(
        np.vstack([Q.T[:-1], np.ones(states)]),
        np.r_[np.zeros(states - 1), 1.0]
    )


def tensor_power(vector, order):
    result = np.asarray(vector, float)
    for _ in range(1, order):
        result = np.multiply.outer(result, vector)
    return result


def exponential_derivatives(theta, point):
    value = math.exp(float(theta @ point))
    return [value] + [value * tensor_power(theta, order)
                      for order in range(1, 5)]


def exact_exponential_price(initial, speed, theta):
    pi = stationary(Q)
    mean = pi @ VALUES
    centered = VALUES - mean
    return float(
        math.exp(float(theta @ mean))
        * initial
        @ expm(MATURITY * (
            speed * Q + np.diag(centered @ theta / MATURITY)
        ))
        @ np.ones(len(Q))
    )


def mixed_polynomial_derivatives(point):
    """Derivatives through order four of C(x, y) = x^2 y^2."""
    x, y = point
    derivatives = [x * x * y * y,
                   np.array([2 * x * y * y, 2 * x * x * y]),
                   np.array([[2 * y * y, 4 * x * y],
                             [4 * x * y, 2 * x * x]]),
                   np.zeros((2, 2, 2)), np.zeros((2, 2, 2, 2))]
    for indices in np.ndindex(derivatives[3].shape):
        if sum(indices) == 1:
            derivatives[3][indices] = 4 * y
        elif sum(indices) == 2:
            derivatives[3][indices] = 4 * x
    for indices in np.ndindex(derivatives[4].shape):
        if sum(indices) == 2:
            derivatives[4][indices] = 4.0
    return derivatives


def affine_derivatives(derivatives, matrix):
    """Pull derivative tensors back under y = matrix.T @ x + shift."""
    inverse_jacobian = np.linalg.inv(matrix).T
    transformed = [derivatives[0]]
    for order in range(1, 5):
        old = "abcd"[:order]
        new = "ijkl"[:order]
        rule = ",".join([old] + [a + i for a, i in zip(old, new)])
        transformed.append(np.einsum(
            rule + "->" + new, derivatives[order],
            *([inverse_jacobian] * order)
        ))
    return transformed


def monomial_derivatives(point, powers):
    """Derivative tensors through degree four of prod_j x_j**powers[j]."""
    derivatives = [float(np.prod(point ** powers))]
    for order in range(1, 5):
        tensor = np.zeros((len(point),) * order)
        for indices in np.ndindex(tensor.shape):
            counts = np.bincount(indices, minlength=len(point))
            if np.all(counts <= powers):
                coefficient = np.prod([
                    math.factorial(int(p)) / math.factorial(int(p - count))
                    for p, count in zip(powers, counts)
                ])
                tensor[indices] = coefficient * np.prod(
                    point ** (powers - counts)
                )
        derivatives.append(tensor)
    return derivatives


def pullback_derivatives(derivatives, matrix):
    """Pull a payoff on y = matrix.T @ x + shift back to x-space."""
    pulled = [derivatives[0]]
    for order in range(1, 5):
        old = "ijkl"[:order]
        new = "abcd"[:order]
        rule = ",".join([old] + [a + i for a, i in zip(new, old)])
        pulled.append(np.einsum(
            rule + "->" + new, derivatives[order], *([matrix] * order)
        ))
    return pulled


def exact_mixed_polynomial_price(initial, speed):
    """Exact E[X[0]^2 X[1]^2] from the joint raw-moment ODE."""
    indices = list(np.ndindex(3, 3))
    states = len(Q)
    system = np.zeros((len(indices) * states,) * 2)
    for row, alpha in enumerate(indices):
        block = slice(row * states, (row + 1) * states)
        system[block, block] = speed * Q
        for field in range(2):
            if alpha[field]:
                beta = list(alpha)
                beta[field] -= 1
                column = indices.index(tuple(beta))
                source = slice(column * states, (column + 1) * states)
                system[block, source] = alpha[field] * np.diag(VALUES[:, field])
    initial_moments = np.zeros(len(indices) * states)
    initial_moments[:states] = 1.0
    target = expm(MATURITY * system) @ initial_moments
    block = slice(indices.index((2, 2)) * states,
                  (indices.index((2, 2)) + 1) * states)
    return float(initial @ target[block] / MATURITY ** 4)


def main():
    ok = True
    pi = stationary(Q)
    mean = pi @ VALUES
    priors = [np.eye(3)[j] for j in range(3)]
    priors += [pi, np.array([0.2, 0.3, 0.5])]
    errors = []
    mixed_errors = []
    reduction_error = 0.0
    affine_error = 0.0
    rectangular_error = 0.0
    for speed in SPEEDS:
        maximum = 0.0
        for initial in priors:
            for theta in THETAS:
                derivatives = exponential_derivatives(theta, mean)
                vector_price = vector_observable_price_expansion(
                    Q, VALUES, initial, MATURITY, speed, derivatives
                )
                exact = exact_exponential_price(initial, speed, theta)
                maximum = max(maximum, abs(exact - vector_price))

                scalar_values = VALUES @ theta
                scalar_value = math.exp(float(pi @ scalar_values))
                scalar_derivatives = np.full(5, scalar_value)
                scalar_price = scalar_observable_price_expansion(
                    Q, scalar_values, initial, MATURITY, speed,
                    scalar_derivatives
                )
                reduction_error = max(
                    reduction_error, abs(vector_price - scalar_price)
                )
        errors.append(maximum)
        mixed_maximum = 0.0
        derivatives = mixed_polynomial_derivatives(mean)
        for initial in priors:
            approximate = vector_observable_price_expansion(
                Q, VALUES, initial, MATURITY, speed, derivatives
            )
            exact = exact_mixed_polynomial_price(initial, speed)
            mixed_maximum = max(mixed_maximum, abs(exact - approximate))
            for matrix, shift in AFFINE_MAPS:
                transformed = vector_observable_price_expansion(
                    Q, VALUES @ matrix + shift, initial, MATURITY, speed,
                    affine_derivatives(derivatives, matrix)
                )
                affine_error = max(
                    affine_error, abs(transformed - approximate)
                )
            for matrix, shift, powers in RECTANGULAR_MAPS:
                mapped_values = VALUES @ matrix + shift
                mapped_derivatives = monomial_derivatives(
                    mean @ matrix + shift, powers
                )
                mapped = vector_observable_price_expansion(
                    Q, mapped_values, initial, MATURITY, speed,
                    mapped_derivatives
                )
                pulled = vector_observable_price_expansion(
                    Q, VALUES, initial, MATURITY, speed,
                    pullback_derivatives(mapped_derivatives, matrix)
                )
                rectangular_error = max(
                    rectangular_error, abs(mapped - pulled)
                )
        mixed_errors.append(mixed_maximum)
        print(f"speed {speed:2d}: maximum vector-price error "
              f"{maximum:.8e}; mixed-polynomial error {mixed_maximum:.8e}")
    order = math.log(errors[-2] / errors[-1], 2)
    print(f"last-doubling vector-price order {order:.6f}")
    mixed_order = math.log(mixed_errors[-2] / mixed_errors[-1], 2)
    print(f"last-doubling mixed-polynomial order {mixed_order:.6f}")
    print(f"directional scalar-reduction error {reduction_error:.2e}")
    print(f"affine-coordinate covariance error {affine_error:.2e}")
    print(f"rectangular-map covariance error {rectangular_error:.2e}")
    ok &= order > 2.98
    ok &= errors[-1] < 1.0e-7
    ok &= mixed_order > 2.98
    ok &= mixed_errors[-1] < 2.0e-7
    ok &= reduction_error < 3.0e-13
    ok &= affine_error < 2.0e-12
    ok &= rectangular_error < 2.0e-12

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
