"""All-chain mean-wait speed monotonicity; reversibilization and sharp scope."""

from fractions import Fraction

import numpy as np


ROUND_TOL = 4096 * np.finfo(float).eps


def normalized_error(a, b):
    return np.linalg.norm(a-b) / max(1.0, np.linalg.norm(a), np.linalg.norm(b))


def calculate(q, pi, hazard, speed):
    root = np.sqrt(pi)
    l = -root[:, None] * q / root[None, :]
    symmetric, skew = (l+l.T)/2, (l-l.T)/2
    d = np.diag(hazard)
    s = d + speed*symmetric
    inverse_s = np.linalg.inv(s)
    b = s + speed**2 * skew.T @ inverse_s @ skew
    inverse_b = np.linalg.inv(b)
    r = np.linalg.inv(d + speed*l)
    b_prime = symmetric + speed*skew.T @ (
        inverse_s + inverse_s @ d @ inverse_s
    ) @ skew
    vector = inverse_b @ root
    direct_derivative = -root @ r @ l @ r @ root
    energy_derivative = -vector @ b_prime @ vector
    value = root @ r @ root
    reversible_value = root @ inverse_s @ root
    assert np.linalg.eigvalsh(b_prime)[0] >= -ROUND_TOL * max(1.0, np.linalg.norm(b_prime))
    return value, direct_derivative, reversible_value, max(
        normalized_error((r+r.T)/2, inverse_b),
        normalized_error(direct_derivative, energy_derivative),
    )


def exact_counterexample():
    m = Fraction(1, 100)
    n, n1, n2 = 169*m*m + 509*m + 142, 338*m + 509, Fraction(338)
    d, d1, d2 = 26*(34*m*m + 41*m + 10), 26*(68*m + 41), Fraction(26*68)
    first = (n1*d-n*d1)/d**2
    second = (n2*d-n*d2)/d**2 - 2*(n1*d-n*d1)*d1/d**3
    return n/d, first, second


def main():
    rng = np.random.default_rng(20261014)
    max_error = max_order_violation = 0.0
    minimum_decrease = np.inf
    cases = 0
    for dimension in range(3, 9):
        for _ in range(8):
            q = rng.uniform(0.05, 2.0, (dimension, dimension))
            np.fill_diagonal(q, 0.0)
            np.fill_diagonal(q, -q.sum(axis=1))
            system = q.T.copy()
            system[-1] = 1.0
            target = np.zeros(dimension)
            target[-1] = 1.0
            pi = np.linalg.solve(system, target)
            hazard = rng.uniform(0.4, 4.0, dimension)
            for speed in np.geomspace(0.01, 30.0, 11):
                value, derivative, reversible, error = calculate(q, pi, hazard, speed)
                assert derivative < 0.0
                minimum_decrease = min(minimum_decrease, -derivative)
                max_error = max(max_error, error)
                max_order_violation = max(max_order_violation, max(0.0, value-reversible))
                assert value >= 1.0/(pi @ hazard) - ROUND_TOL
            cases += 1
    q = np.array([[-6., 0., 6.], [1., -1., 0.], [0., 1., -1.]])
    pi, hazard = np.array([1., 6., 6.])/13, np.array([2., 1., 10.])
    value, first, reversible, error = calculate(q, pi, hazard, 0.01)
    max_error = max(max_error, error)
    r = np.linalg.inv(np.diag(hazard) - 0.01*q)
    second = 2*pi @ r @ q @ r @ q @ r @ np.ones(3)
    exact = exact_counterexample()
    assert exact[1] < 0 and exact[2] < 0 and second < -0.03
    for actual, rational in zip([value, first, second], exact):
        max_error = max(max_error, normalized_error(actual, float(rational)))
    assert np.max(abs(pi @ q)) <= ROUND_TOL
    # A nonreversible four-cycle attains equality for nonconstant alternating hazards.
    cycle = np.roll(np.eye(4), 1, axis=1) - np.eye(4)
    equality_error = 0.0
    for speed in [0.01, 1.0, 30.0]:
        wait, _, reversible_wait, error = calculate(cycle, np.ones(4)/4, [1., 2., 1., 2.], speed)
        equality_error = max(equality_error, abs(wait-reversible_wait))
        constant, constant_first, _, error_constant = calculate(cycle, np.ones(4)/4, [2.]*4, speed)
        max_error = max(max_error, error, error_constant, abs(constant-0.5), abs(constant_first))
    assert max(max_error, max_order_violation, equality_error) <= ROUND_TOL
    print("nonreversible mean-wait certificate")
    print("random chains", cases, "dimensions 3 through 8; speeds per chain 11")
    print("maximum normalized resolvent/derivative/rational error", f"{max_error:.3e}")
    print("minimum strictly negative derivative magnitude", f"{minimum_decrease:.3e}")
    print("maximum reversibilization-order violation", f"{max_order_violation:.3e}")
    print("nonconstant four-cycle equality discrepancy", f"{equality_error:.3e}")
    print("concave three-cycle W, W', W''", *(f"{float(x):.10f}" for x in exact))
    print("exact curvature fraction", exact[2])
    print("concave three-cycle reversibilized W", f"{reversible:.10f}")
    print("roundoff tolerance", f"{ROUND_TOL:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
