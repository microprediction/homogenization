"""Certificate for the vector finite-chain second-order price expansion.

The exact benchmark is a matrix exponential for exponential payoffs of two
time-averaged observables on a nonreversible three-state chain.  Directional
contractions are also compared with the scalar theorem.  Ends with PASS or
FAIL.
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


def main():
    ok = True
    pi = stationary(Q)
    mean = pi @ VALUES
    priors = [np.eye(3)[j] for j in range(3)]
    priors += [pi, np.array([0.2, 0.3, 0.5])]
    errors = []
    reduction_error = 0.0
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
        print(f"speed {speed:2d}: maximum vector-price error "
              f"{maximum:.8e}")
    order = math.log(errors[-2] / errors[-1], 2)
    print(f"last-doubling vector-price order {order:.6f}")
    print(f"directional scalar-reduction error {reduction_error:.2e}")
    ok &= order > 2.98
    ok &= errors[-1] < 1.0e-7
    ok &= reduction_error < 3.0e-13

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
