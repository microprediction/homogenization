"""Reversible first-event mean: spectral speed law and signed bounds."""

from fractions import Fraction
from math import factorial

import numpy as np
from scipy.integrate import quad
from scipy.linalg import eigh, expm


ROUND_TOL = 4096 * np.finfo(float).eps


def scaled_error(a, b):
    return abs(a - b) / max(1.0, abs(a), abs(b))


def spectrum(q, pi, hazard):
    root_pi = np.sqrt(pi)
    symmetric_l = -root_pi[:, None] * q / root_pi[None, :]
    root_d = np.sqrt(hazard)
    a = symmetric_l / root_d[:, None] / root_d[None, :]
    values, vectors = eigh(a)
    assert abs(values[0]) <= ROUND_TOL * max(1.0, np.linalg.norm(a, 2))
    assert values[1] > 0.0
    weights = (vectors.T @ (root_pi / root_d)) ** 2
    assert scaled_error(weights[0], 1.0 / (pi @ hazard)) <= ROUND_TOL
    return values[1:], weights[1:]


def green_kubo(q, pi, hazard):
    projector = np.outer(np.ones(len(pi)), pi)
    h = np.linalg.inv(-q + projector) - projector
    centered = hazard - pi @ hazard
    return (pi * centered) @ h @ centered / (pi @ hazard) ** 2


def check(q, pi, hazard, errors):
    values, weights = spectrum(q, pi, hazard)
    mean = pi @ hazard
    coefficients = np.array([np.sum(weights / values**j) for j in range(1, 6)])
    errors[0] = max(errors[0], scaled_error(coefficients[0], green_kubo(q, pi, hazard)))
    for speed in np.geomspace(0.05, 30.0, 11):
        r = np.linalg.inv(np.diag(hazard) - speed * q)
        vector = r @ np.ones(len(pi))
        direct = pi @ vector
        spectral = 1.0 / mean + np.sum(weights / (1.0 + speed * values))
        errors[1] = max(errors[1], scaled_error(direct, spectral))
        assert 1.0 / mean - ROUND_TOL <= direct <= pi @ (1.0 / hazard) + ROUND_TOL
        event_prior = pi * hazard / mean
        errors[2] = max(errors[2], scaled_error(event_prior @ vector, 1.0 / mean))
        for k in range(1, 7):
            vector = r @ q @ vector
            direct_derivative = factorial(k) * (pi @ vector)
            spectral_derivative = (-1)**k * factorial(k) * np.sum(
                weights * values**k / (1.0 + speed * values)**(k + 1)
            )
            errors[3] = max(errors[3], scaled_error(direct_derivative, spectral_derivative))
            assert (-1)**k * spectral_derivative > 0.0
        approximation = 1.0 / mean
        for p in range(1, 5):
            approximation += (-1)**(p - 1) * coefficients[p - 1] / speed**p
            signed = (-1)**p * (direct - approximation)
            bound = coefficients[p] / speed**(p + 1)
            scale = max(1.0, abs(direct), abs(approximation), bound)
            violation = max(0.0, -signed, signed - bound) / scale
            errors[4] = max(errors[4], violation)
    return coefficients


def cycle_fifth_taylor():
    """Exact rational fifth derivative at m=1/100, without symbolic libraries."""
    m = Fraction(1, 100)
    numerator = [9*m*m + 18*m + 11, 18*m + 18, Fraction(9)]
    denominator = [3*(6*m*m + 11*m + 6), 3*(12*m + 11), Fraction(18)]
    coefficients = []
    for k in range(6):
        rhs = numerator[k] if k < 3 else Fraction(0)
        rhs -= sum(denominator[j] * coefficients[k-j] for j in range(1, min(k, 2)+1))
        coefficients.append(rhs / denominator[0])
    return factorial(5) * coefficients[5]


def main():
    rng = np.random.default_rng(20261014)
    errors = np.zeros(5)
    cases = 0
    for dimension in range(2, 9):
        for _ in range(8):
            pi = rng.uniform(0.5, 1.5, dimension)
            pi /= pi.sum()
            conductance = rng.uniform(0.01, 0.1, (dimension, dimension))
            conductance = (conductance + conductance.T) / 2.0
            np.fill_diagonal(conductance, 0.0)
            q = conductance / pi[:, None]
            np.fill_diagonal(q, -q.sum(axis=1))
            hazard = rng.uniform(0.3, 3.0, dimension)
            check(q, pi, hazard, errors)
            cases += 1
    benchmark_q = np.array([[-1., 1., 0.], [1., -3., 2.], [0., 2., -2.]])
    pi = np.ones(3) / 3.0
    hazard = np.array([1., 2., 3.])
    coefficients = check(benchmark_q, pi, hazard, errors)
    benchmark = pi @ np.linalg.solve(np.diag(hazard) - benchmark_q, np.ones(3))
    integrated, integration_error = quad(
        lambda t: pi @ expm(t * (benchmark_q - np.diag(hazard))) @ np.ones(3),
        0.0, np.inf, epsabs=1e-11, epsrel=1e-11,
    )
    assert abs(integrated - benchmark) <= max(1e-10, 10 * integration_error)
    cycle_q = np.array([[-1., 1., 0.], [0., -1., 1.], [1., 0., -1.]])
    r = np.linalg.inv(np.diag(hazard) - 0.01 * cycle_q)
    vector = r @ np.ones(3)
    for _ in range(5):
        vector = r @ cycle_q @ vector
    fifth = factorial(5) * (pi @ vector)
    exact_fifth = cycle_fifth_taylor()
    assert exact_fifth > 0 and fifth > 0.8
    assert scaled_error(fifth, float(exact_fifth)) <= ROUND_TOL
    for speed in [0.01, 1.0, 10.0]:
        cycle_mean = pi @ np.linalg.solve(np.diag(hazard) - speed * cycle_q, np.ones(3))
        rational_mean = (9*speed**2 + 18*speed + 11) / (3*(6*speed**2 + 11*speed + 6))
        assert scaled_error(cycle_mean, rational_mean) <= ROUND_TOL
        reversed_mean = pi @ np.linalg.solve(np.diag(hazard) - speed * cycle_q.T, np.ones(3))
        assert scaled_error(cycle_mean, reversed_mean) <= ROUND_TOL
    constant = np.linalg.solve(np.diag(np.full(3, 2.0)) - benchmark_q, np.ones(3))
    assert np.max(abs(constant - 0.5)) <= ROUND_TOL
    assert np.max(errors) <= ROUND_TOL
    print("reversible mean-wait certificate")
    print("random reversible chains", cases, "dimensions 2 through 8; speeds per chain 11")
    for label, error in zip(["Green-Kubo coefficient", "spectral mean", "event-start mean",
                             "derivatives through order 6", "signed bounds through order 4"], errors):
        print("maximum normalized", label, "error/violation", f"{error:.3e}")
    print("reversible benchmark W(1)", f"{benchmark:.10f}")
    print("independent survival-integral discrepancy", f"{abs(integrated-benchmark):.3e}")
    print("reversible benchmark A1, A2", *(f"{x:.10f}" for x in coefficients[:2]))
    print("cycle W^(5)(0.01): matrix, exact rational", f"{fifth:.10f}", exact_fifth)
    print("cycle alternating fifth derivative", f"{-fifth:.10f}")
    print("normalized roundoff tolerance", f"{ROUND_TOL:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
