"""Certificate for exact switching-speed order of a tilted finite chain.

For every irreducible generator Q and real state forcing g, the principal
eigenvalue of m Q + diag(g) is decreasing and convex in m.  In the reversible
case it also has the Rayleigh--Ritz formula

    Lambda(m) = max_{||f||_pi=1} <g f,f>_pi - m E(f,f).

The certificate checks the envelope, strict decrease and convexity,
Hellmann--Feynman derivatives, and the slow- and fast-speed endpoint
coefficients on dense reversible and nonreversible chains.
"""

import numpy as np
from scipy.linalg import eig, eigh


EPS = np.finfo(float).eps
ROUND_TOL = 4096.0 * EPS


def reversible_generator(rng, n):
    """Return pi, Q built from symmetric positive edge conductances."""
    pi = rng.uniform(0.2, 1.8, size=n)
    pi /= pi.sum()
    conductance = rng.uniform(0.05, 1.0, size=(n, n))
    conductance = np.triu(conductance, 1)
    conductance += conductance.T
    q = conductance / pi[:, None]
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))
    return pi, q


def nonreversible_generator(rng, n):
    """Return the stationary law and a dense directed generator."""
    q = rng.uniform(0.05, 1.2, size=(n, n))
    np.fill_diagonal(q, 0.0)
    np.fill_diagonal(q, -q.sum(axis=1))
    values, vectors = eig(q.T)
    pi = vectors[:, np.argmin(np.abs(values))].real
    if pi.sum() < 0.0:
        pi = -pi
    pi /= pi.sum()
    return pi, q


def symmetric_generator(pi, q):
    root = np.sqrt(pi)
    return root[:, None] * q / root[None, :]


def reversible_principal_data(pi, q, g, speed):
    qhat = symmetric_generator(pi, q)
    ahat = speed * qhat + np.diag(g)
    values, vectors = eigh(ahat)
    top = vectors[:, -1]
    first = float(top @ qhat @ top)
    couplings = vectors[:, :-1].T @ qhat @ top
    second = float(2.0 * np.sum(couplings * couplings /
                               (values[-1] - values[:-1])))
    return float(values[-1]), first, second


def principal_data(q, g, speed):
    """Principal eigenvalue and its Hellmann--Feynman speed derivative."""
    values, left, right = eig(speed * q + np.diag(g), left=True, right=True)
    index = int(np.argmax(values.real))
    value = float(values[index].real)
    left_vector = left[:, index].real
    right_vector = right[:, index].real
    pairing = float(left_vector @ right_vector)
    slope = float(left_vector @ q @ right_vector / pairing)
    return value, slope


def group_inverse(q, pi):
    one_pi = np.ones((q.shape[0], 1)) @ pi[None, :]
    return np.linalg.inv(q + one_pi) - one_pi


def scaled_error(lhs, rhs):
    return abs(lhs - rhs) / max(1.0, abs(lhs), abs(rhs))


def main():
    rng = np.random.default_rng(20261012)
    speeds = np.geomspace(0.025, 250.0, 19)
    max_balance_error = 0.0
    max_similarity_error = 0.0
    max_envelope_error = 0.0
    max_derivative_error = 0.0
    min_negative_slope = np.inf
    min_reversible_curvature = np.inf
    max_fast_relative_error = 0.0
    max_slow_relative_error = 0.0

    for case in range(48):
        n = 3 + case % 5
        reversible = case % 2 == 0
        if reversible:
            pi, q = reversible_generator(rng, n)
        else:
            pi, q = nonreversible_generator(rng, n)
        g = rng.normal(size=n)
        g += np.linspace(-0.3, 0.4, n)
        if reversible:
            qhat = symmetric_generator(pi, q)

        scale_q = max(1.0, np.linalg.norm(q, ord=np.inf))
        stationarity_error = np.max(np.abs(pi @ q)) / scale_q
        max_balance_error = max(max_balance_error, stationarity_error)
        if reversible:
            max_balance_error = max(
                max_balance_error,
                np.max(np.abs(pi[:, None] * q - pi[None, :] * q.T)) /
                scale_q,
                np.max(np.abs(qhat - qhat.T)) / scale_q,
            )

        mean_g = float(pi @ g)
        top_g = float(np.max(g))
        previous_value = None
        previous_slope = None
        for speed in speeds:
            value, slope = principal_data(q, g, speed)
            if reversible:
                symmetric_value, symmetric_slope, curvature = (
                    reversible_principal_data(pi, q, g, speed)
                )
                matrix_scale = max(
                    1.0,
                    np.linalg.norm(speed * q + np.diag(g), ord=np.inf),
                )
                max_similarity_error = max(
                    max_similarity_error,
                    abs(value - symmetric_value) / matrix_scale,
                    scaled_error(slope, symmetric_slope),
                )
                min_reversible_curvature = min(
                    min_reversible_curvature, curvature
                )
            max_envelope_error = max(
                max_envelope_error,
                max(0.0, mean_g - value, value - top_g) /
                max(1.0, abs(mean_g), abs(top_g)),
            )
            if previous_value is not None:
                assert value < previous_value
                assert slope > previous_slope
            previous_value = value
            previous_slope = slope
            min_negative_slope = min(min_negative_slope, -slope)

            if speed <= 10.0:
                step = 1.0e-6 * max(1.0, speed)
                plus = principal_data(q, g, speed + step)[0]
                minus = principal_data(q, g, speed - step)[0]
                finite_difference = (plus - minus) / (2.0 * step)
                max_derivative_error = max(
                    max_derivative_error, scaled_error(slope, finite_difference)
                )

        centered = g - mean_g
        qsharp = group_inverse(q, pi)
        green_kubo = -float(pi @ (centered * (qsharp @ centered)))
        assert green_kubo > 0.0
        fast_speed = 2.0e4
        fast_value = principal_data(q, g, fast_speed)[0]
        fast_coefficient = fast_speed * (fast_value - mean_g)
        max_fast_relative_error = max(
            max_fast_relative_error,
            abs(fast_coefficient - green_kubo) /
            max(1.0e-14, abs(green_kubo)),
        )

        maximizer = int(np.argmax(g))
        slow_speed = 1.0e-7
        slow_value = principal_data(q, g, slow_speed)[0]
        slow_slope = (slow_value - top_g) / slow_speed
        max_slow_relative_error = max(
            max_slow_relative_error,
            abs(slow_slope - q[maximizer, maximizer]) /
            max(1.0, abs(q[maximizer, maximizer])),
        )

    assert max_balance_error <= ROUND_TOL
    assert max_similarity_error <= ROUND_TOL
    assert max_envelope_error <= ROUND_TOL
    assert max_derivative_error <= 5.0e-8
    assert min_negative_slope > 0.0
    assert min_reversible_curvature > 0.0
    assert max_fast_relative_error < 5.0e-3
    assert max_slow_relative_error < 2.0e-4

    print("switching-speed order certificate")
    print("  random chains (reversible, nonreversible)", 24, 24)
    print("  maximum stationarity/balance error", f"{max_balance_error:.3e}")
    print("  maximum similarity/eigenvalue error", f"{max_similarity_error:.3e}")
    print("  maximum envelope violation", f"{max_envelope_error:.3e}")
    print("  maximum derivative-check error", f"{max_derivative_error:.3e}")
    print("  minimum magnitude of negative slope", f"{min_negative_slope:.3e}")
    print("  minimum reversible curvature", f"{min_reversible_curvature:.3e}")
    print("  maximum fast-limit coefficient error", f"{max_fast_relative_error:.3e}")
    print("  maximum slow-limit slope error", f"{max_slow_relative_error:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
