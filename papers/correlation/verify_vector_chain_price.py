"""Certificate for the vector finite-chain second-order price expansion.

The exact benchmarks are matrix exponentials for exponential payoffs and the
genuinely mixed payoff x[0]^2*x[1]^2.  Directional contractions are also
compared with the scalar theorem.  Ends with PASS or FAIL.
"""
import math
import os
import sys
from itertools import product

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


def green_kubo_covariance(values):
    """Two-sided covariance rate for a finite-chain feature matrix."""
    pi = stationary(Q)
    projection = np.outer(np.ones(len(Q)), pi)
    group_inverse = np.linalg.inv(Q + projection) - projection
    centered = values - pi @ values
    one_sided = -centered.T @ np.diag(pi) @ group_inverse @ centered
    return one_sided + one_sided.T


def nonlinear_feature_map(values):
    """A nonlinear map used to delimit affine covariance."""
    values = np.asarray(values)
    return np.stack([values[..., 0] ** 2,
                     values[..., 0] * values[..., 1]], axis=-1)


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


def exact_vector_moments(initial, speed, maximum_powers=(4, 2)):
    """Exact raw moments of the averaged feature vector."""
    indices = list(product(*[range(power + 1)
                             for power in maximum_powers]))
    states = len(Q)
    system = np.zeros((len(indices) * states,) * 2)
    for row, alpha in enumerate(indices):
        block = slice(row * states, (row + 1) * states)
        system[block, block] = speed * Q
        for field, exponent in enumerate(alpha):
            if exponent:
                beta = list(alpha)
                beta[field] -= 1
                column = indices.index(tuple(beta))
                source = slice(column * states, (column + 1) * states)
                system[block, source] = exponent * np.diag(VALUES[:, field])
    initial_moments = np.zeros(len(indices) * states)
    initial_moments[:states] = 1.0
    target = expm(MATURITY * system) @ initial_moments
    return {
        alpha: float(initial @ target[row * states:(row + 1) * states]
                     / MATURITY ** sum(alpha))
        for row, alpha in enumerate(indices)
    }


def centered_moment(moment, powers, center):
    """Convert one bivariate raw moment to the given center."""
    total = 0.0
    for first in range(powers[0] + 1):
        for second in range(powers[1] + 1):
            coefficient = (math.comb(powers[0], first)
                           * math.comb(powers[1], second)
                           * (-center[0]) ** (powers[0] - first)
                           * (-center[1]) ** (powers[1] - second))
            total += coefficient * moment[(first, second)]
    return total


def exact_critical_transform_moments(initial, speed, center):
    """Moments of the old critical map and all quadratic coordinates."""
    moment = exact_vector_moments(initial, speed, maximum_powers=(4, 4))
    first_center = center[0]
    transformed_mean = np.array([
        moment[(2, 0)] - 2 * first_center * moment[(1, 0)]
        + first_center ** 2,
        moment[(0, 1)],
    ])
    transformed_second = np.array([
        [moment[(4, 0)] - 4 * first_center * moment[(3, 0)]
         + 6 * first_center ** 2 * moment[(2, 0)]
         - 4 * first_center ** 3 * moment[(1, 0)] + first_center ** 4,
         moment[(2, 1)] - 2 * first_center * moment[(1, 1)]
         + first_center ** 2 * moment[(0, 1)]],
        [0.0, moment[(0, 2)]],
    ])
    transformed_second[1, 0] = transformed_second[0, 1]
    covariance = transformed_second - np.outer(
        transformed_mean, transformed_mean
    )
    centered = {
        powers: centered_moment(moment, powers, center)
        for powers in [(2, 0), (0, 2), (1, 1), (4, 0), (0, 4),
                       (2, 2), (3, 1), (1, 3)]
    }
    quadratic_mean = np.array([
        centered[(2, 0)], centered[(0, 2)], centered[(1, 1)]
    ])
    quadratic_second = np.array([
        [centered[(4, 0)], centered[(2, 2)], centered[(3, 1)]],
        [centered[(2, 2)], centered[(0, 4)], centered[(1, 3)]],
        [centered[(3, 1)], centered[(1, 3)], centered[(2, 2)]],
    ])
    quadratic_covariance = quadratic_second - np.outer(
        quadratic_mean, quadratic_mean
    )
    return transformed_mean, covariance, quadratic_covariance


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
    rank_map_error = 0.0
    mapped_ranks = []
    nonlinear_mean_errors = []
    nonlinear_covariance_errors = []
    critical_variance_errors = []
    critical_rank_errors = []
    base_covariance = green_kubo_covariance(VALUES)
    critical_hessians = np.array([
        [[2.0, 0.0], [0.0, 0.0]],
        [[0.0, 0.0], [0.0, 2.0]],
        [[0.0, 1.0], [1.0, 0.0]],
    ])
    critical_coefficient = 0.5 * np.einsum(
        "iab,ac,bd,jcd->ij", critical_hessians, base_covariance,
        base_covariance, critical_hessians
    )
    eigenvalues, eigenvectors = np.linalg.eigh(base_covariance)
    covariance_sqrt = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    hessian_images = np.array([
        covariance_sqrt @ hessian @ covariance_sqrt
        for hessian in critical_hessians
    ])
    critical_gram = 0.5 * np.einsum(
        "iab,jba->ij", hessian_images, hessian_images
    )
    critical_gram_error = float(np.max(np.abs(
        critical_coefficient - critical_gram
    )))
    critical_rank = np.linalg.matrix_rank(critical_coefficient, tol=1e-11)
    for matrix, shift, _ in RECTANGULAR_MAPS:
        mapped_values = VALUES @ matrix + shift
        mapped_covariance = green_kubo_covariance(mapped_values)
        rank_map_error = max(
            rank_map_error,
            float(np.max(np.abs(
                mapped_covariance - matrix.T @ base_covariance @ matrix
            )))
        )
        mapped_ranks.append(np.linalg.matrix_rank(mapped_covariance, tol=1e-11))
    nonlinear_gap = (pi @ nonlinear_feature_map(VALUES)
                     - nonlinear_feature_map(pi @ VALUES))
    expected_nonlinear_gap = np.array([5227 / 16900, -17 / 130])
    nonlinear_gap_error = float(np.max(np.abs(
        nonlinear_gap - expected_nonlinear_gap
    )))
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
        nonlinear_mean_error = 0.0
        nonlinear_covariance_error = 0.0
        critical_variance_error = 0.0
        critical_rank_error = 0.0
        tangent_covariance = np.array([
            [0.0, 0.0],
            [0.0, base_covariance[1, 1] / (speed * MATURITY)],
        ])
        for initial in priors:
            transformed_mean, transformed_covariance, quadratic_covariance = (
                exact_critical_transform_moments(initial, speed, mean)
            )
            tangent_mean = np.array([
                base_covariance[0, 0] / (speed * MATURITY),
                transformed_mean[1],
            ])
            nonlinear_mean_error = max(
                nonlinear_mean_error,
                float(np.max(np.abs(transformed_mean - tangent_mean)))
            )
            nonlinear_covariance_error = max(
                nonlinear_covariance_error,
                float(np.max(np.abs(
                    transformed_covariance - tangent_covariance
                )))
            )
            critical_variance = (
                2 * base_covariance[0, 0] ** 2
                / (speed ** 2 * MATURITY ** 2)
            )
            critical_variance_error = max(
                critical_variance_error,
                abs(transformed_covariance[0, 0] - critical_variance)
            )
            predicted_quadratic_covariance = (
                critical_coefficient / (speed ** 2 * MATURITY ** 2)
            )
            critical_rank_error = max(
                critical_rank_error,
                float(np.max(np.abs(
                    quadratic_covariance - predicted_quadratic_covariance
                )))
            )
        nonlinear_mean_errors.append(nonlinear_mean_error)
        nonlinear_covariance_errors.append(nonlinear_covariance_error)
        critical_variance_errors.append(critical_variance_error)
        critical_rank_errors.append(critical_rank_error)
        print(f"speed {speed:2d}: maximum vector-price error "
              f"{maximum:.8e}; mixed-polynomial error {mixed_maximum:.8e}")
        print(f"            nonlinear mean error {nonlinear_mean_error:.8e}; "
              f"tangent-covariance error {nonlinear_covariance_error:.8e}")
        print(f"            critical quadratic-variance error "
              f"{critical_variance_error:.8e}")
        print(f"            critical rank-three covariance error "
              f"{critical_rank_error:.8e}")
    order = math.log(errors[-2] / errors[-1], 2)
    print(f"last-doubling vector-price order {order:.6f}")
    mixed_order = math.log(mixed_errors[-2] / mixed_errors[-1], 2)
    print(f"last-doubling mixed-polynomial order {mixed_order:.6f}")
    print(f"directional scalar-reduction error {reduction_error:.2e}")
    print(f"affine-coordinate covariance error {affine_error:.2e}")
    print(f"rectangular-map covariance error {rectangular_error:.2e}")
    print(f"fixed-feature rank-map error {rank_map_error:.2e}; ranks "
          + " ".join(map(str, mapped_ranks)))
    print("nonlinear stationary Jensen gap [5227/16900, -17/130]; "
          f"rational-check error {nonlinear_gap_error:.2e}")
    nonlinear_mean_order = math.log(
        nonlinear_mean_errors[-2] / nonlinear_mean_errors[-1], 2
    )
    nonlinear_covariance_order = math.log(
        nonlinear_covariance_errors[-2] / nonlinear_covariance_errors[-1], 2
    )
    print(f"nonlinear delta mean/covariance orders "
          f"{nonlinear_mean_order:.6f} {nonlinear_covariance_order:.6f}")
    critical_variance_order = math.log(
        critical_variance_errors[-2] / critical_variance_errors[-1], 2
    )
    print(f"critical quadratic-variance order {critical_variance_order:.6f}")
    critical_rank_order = math.log(
        critical_rank_errors[-2] / critical_rank_errors[-1], 2
    )
    print(f"critical Hessian-Gram discrepancy {critical_gram_error:.2e}; "
          f"coefficient rank {critical_rank}")
    print(f"critical rank-three covariance order {critical_rank_order:.6f}")
    ok &= order > 2.98
    ok &= errors[-1] < 1.0e-7
    ok &= mixed_order > 2.98
    ok &= mixed_errors[-1] < 2.0e-7
    ok &= reduction_error < 3.0e-13
    ok &= affine_error < 2.0e-12
    ok &= rectangular_error < 2.0e-12
    ok &= rank_map_error < 2.0e-12
    ok &= mapped_ranks == [1, 2]
    ok &= nonlinear_gap_error < 2.0e-14
    ok &= nonlinear_mean_order > 1.98
    ok &= nonlinear_covariance_order > 1.98
    ok &= nonlinear_mean_errors[-1] < 3.0e-5
    ok &= nonlinear_covariance_errors[-1] < 2.0e-5
    ok &= critical_variance_order > 2.98
    ok &= critical_variance_errors[-1] < 5.0e-7
    ok &= critical_gram_error < 2.0e-12
    ok &= critical_rank == 3
    ok &= critical_rank_order > 2.98
    ok &= critical_rank_errors[-1] < 2.0e-6

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
