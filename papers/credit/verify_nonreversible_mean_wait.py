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


def check_remainder(q, pi, hazard, speed):
    root = np.sqrt(pi)
    hazard = np.asarray(hazard)
    mean = pi @ hazard
    c = -root[:, None]*q/root[None, :] / np.sqrt(hazard[:, None]*hazard[None, :])
    zero = root*np.sqrt(hazard/mean)
    projection = np.outer(zero, zero)
    h = np.linalg.inv(c+projection)-projection
    z = root/np.sqrt(hazard) - zero/np.sqrt(mean)
    coefficient = z @ h @ z
    constant = np.linalg.norm(h.T @ z)*np.linalg.norm(h @ z)
    contraction = np.linalg.inv(np.eye(len(pi))+h/speed)
    remainder = -(h.T @ z) @ contraction @ (h @ z)/speed**2
    wait = pi @ np.linalg.solve(np.diag(hazard)-speed*q, np.ones(len(pi)))
    residual = wait-1/mean-coefficient/speed
    # Independently identify the first coefficient by the original chain group inverse.
    group = np.linalg.inv(-q+np.outer(np.ones(len(pi)), pi))-np.outer(np.ones(len(pi)), pi)
    f = hazard-mean
    green_kubo = (pi*f) @ group @ f/mean**2
    error = max(normalized_error(coefficient, green_kubo),
                normalized_error(residual, remainder))
    assert coefficient > 0.0
    assert np.linalg.norm(contraction, 2) <= 1+ROUND_TOL
    assert abs(residual) <= constant/speed**2 + ROUND_TOL*max(1.0, abs(wait))
    return error, coefficient, constant


def exact_cycle_wait(speed):
    hazards = [3, 4, 4, 3, 2, 2]
    product = Fraction(1)
    for hazard in hazards:
        product *= hazard+speed
    total = Fraction(0)
    for start in range(6):
        denominator = Fraction(1)
        for length in range(6):
            denominator *= hazards[(start+length) % 6]+speed
            total += speed**length/denominator
    return total/(6*(1-speed**6/product))


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
                remainder_error, _, _ = check_remainder(q, pi, hazard, speed)
                max_error = max(max_error, remainder_error)
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
    # The reversible first-order upper bound fails for an integer-hazard directed six-cycle.
    cycle6 = np.roll(np.eye(6), 1, axis=1)-np.eye(6)
    error, coefficient, constant = check_remainder(cycle6, np.ones(6)/6, [3, 4, 4, 3, 2, 2], 10)
    exact_wait = Fraction(954661, 2827392)
    for m in [Fraction(1, 100), Fraction(1), Fraction(10), Fraction(100)]:
        polynomial = (18*m**5+135*m**4+537*m**3+1176*m**2+1342*m+624) / (
            3*(9*m*m+26*m+24)*(2*m**3+9*m*m+26*m+24))
        assert exact_cycle_wait(m) == polynomial
    leading = Fraction(18, 54)
    first_coefficient = (135-leading*399)/54
    second_coefficient = (537-leading*1548-first_coefficient*399)/54
    assert (leading, first_coefficient, second_coefficient) == (
        Fraction(1, 3), Fraction(1, 27), Fraction(28, 243))
    exact_gap = exact_wait-Fraction(1, 3)-Fraction(1, 270)
    direct_wait = np.mean(np.linalg.solve(np.diag([3, 4, 4, 3, 2, 2])-10*cycle6, np.ones(6)))
    assert exact_gap == Fraction(77633, 127232640) and exact_gap > 0
    assert normalized_error(direct_wait, float(exact_wait)) <= ROUND_TOL
    assert abs(coefficient-1/27) <= ROUND_TOL
    max_error = max(max_error, error)
    assert max_error <= ROUND_TOL
    print("nonreversible mean-wait certificate")
    print("random chains", cases, "dimensions 3 through 8; speeds per chain 11")
    print("maximum normalized resolvent/derivative/rational error", f"{max_error:.3e}")
    print("minimum strictly negative derivative magnitude", f"{minimum_decrease:.3e}")
    print("maximum reversibilization-order violation", f"{max_order_violation:.3e}")
    print("nonconstant four-cycle equality discrepancy", f"{equality_error:.3e}")
    print("concave three-cycle W, W', W''", *(f"{float(x):.10f}" for x in exact))
    print("exact curvature fraction", exact[2])
    print("concave three-cycle reversibilized W", f"{reversible:.10f}")
    print("all-speed unsigned remainder bounds checked on every random case")
    print("six-cycle W(10), A1, C", f"{float(exact_wait):.10f}", f"{coefficient:.10f}", f"{constant:.10f}")
    print("six-cycle excess over first-order upper approximation", f"{float(exact_gap):.10f}", exact_gap)
    print("six-cycle fast expansion coefficients", leading, first_coefficient, second_coefficient)
    print("six-cycle rational formula numerator coefficients", [18, 135, 537, 1176, 1342, 624])
    print("six-cycle denominator factors", "3*(9*m^2+26*m+24)*(2*m^3+9*m^2+26*m+24)")
    print("roundoff tolerance", f"{ROUND_TOL:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
