"""Certificate for the finite-chain second-order scalar price expansion.

The exact benchmark is a matrix exponential for an exponential payoff on a
nonreversible three-state chain.  A separate two-state check reduces every
spectral coefficient to the closed formula in the arbitrary-prior note.
Ends with PASS or FAIL.
"""
import math
import os
import sys

import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from correlation import (finite_rate_mean_correction,
                         initial_covariance_boundary,
                         initial_layer_coefficient,
                         occupation_covariance_rate,
                         occupation_third_cumulant_rate,
                         scalar_observable_price_expansion)


Q = np.array([[-3.0, 2.0, 1.0],
              [1.0, -4.0, 3.0],
              [2.0, 1.0, -3.0]])
VALUES = np.array([-0.4, 0.2, 0.9])
MATURITY = 0.8
SPEEDS = [4, 8, 16, 32, 64]
THETAS = [-1.3, -0.5, 0.7, 1.4]


def stationary(Q):
    states = len(Q)
    return np.linalg.solve(
        np.vstack([Q.T[:-1], np.ones(states)]),
        np.r_[np.zeros(states - 1), 1.0]
    )


def exact_exponential_price(Q, values, initial, maturity, speed, theta):
    pi = stationary(Q)
    mean = float(pi @ values)
    centered = values - mean
    return float(
        math.exp(theta * mean)
        * initial
        @ expm(maturity * (speed * Q
                           + (theta / maturity) * np.diag(centered)))
        @ np.ones(len(Q))
    )


def main():
    ok = True
    pi = stationary(Q)
    mean = float(pi @ VALUES)
    priors = [np.eye(3)[j] for j in range(3)]
    priors += [pi, np.array([0.2, 0.3, 0.5])]
    errors = []
    for speed in SPEEDS:
        maximum = 0.0
        for initial in priors:
            for theta in THETAS:
                derivatives = np.array([
                    theta ** order * math.exp(theta * mean)
                    for order in range(5)
                ])
                approximation = scalar_observable_price_expansion(
                    Q, VALUES, initial, MATURITY, speed, derivatives
                )
                exact = exact_exponential_price(
                    Q, VALUES, initial, MATURITY, speed, theta
                )
                maximum = max(maximum, abs(exact - approximation))
        errors.append(maximum)
        print(f"speed {speed:2d}: maximum price error {maximum:.8e}")
    order = math.log(errors[-2] / errors[-1], 2)
    print(f"last-doubling price-error order {order:.6f}")
    ok &= order > 2.98
    ok &= errors[-1] < 1.5e-7

    a, b = 1.7, 2.3
    Q2 = np.array([[-a, a], [b, -b]])
    values2 = np.array([-0.55, 0.35])
    r = a + b
    p1, p0 = a / r, b / r
    u, d = p0 * p1, p0 - p1
    delta_value = values2[1] - values2[0]
    covariance_rate = occupation_covariance_rate(Q2)
    third_rate = occupation_third_cumulant_rate(Q2)
    lambda2 = float(delta_value ** 2 * covariance_rate[0, 0])
    lambda3 = float(delta_value ** 3 * third_rate[0, 0, 0])
    coefficient_error = max(
        abs(lambda2 - 2.0 * u * delta_value ** 2 / r),
        abs(lambda3 - 6.0 * u * d * delta_value ** 3 / r ** 2),
    )
    mean_error = 0.0
    for eta in np.linspace(0.0, 1.0, 11):
        initial = np.array([1.0 - eta, eta])
        delta = eta - p1
        b1 = initial_layer_coefficient(Q2, values2, initial)
        b2 = initial_covariance_boundary(Q2, values2, initial)
        coefficient_error = max(
            coefficient_error,
            abs(b1 - delta_value * delta / r),
            abs(b2 + b1 * b1
                - 2.0 * delta_value ** 2 * (delta * d - u) / r ** 2),
        )
        for speed in SPEEDS:
            exact_mean = (
                delta_value * delta * (1.0 - math.exp(-speed * r * MATURITY))
                / (speed * r)
            )
            mean_error = max(mean_error, abs(
                finite_rate_mean_correction(
                    Q2, values2, initial, MATURITY, speed
                ) - exact_mean
            ))
    print(f"two-state spectral-coefficient error {coefficient_error:.2e}")
    print(f"two-state exact-mean-layer error {mean_error:.2e}")
    ok &= coefficient_error < 3.0e-13
    ok &= mean_error < 3.0e-13

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
