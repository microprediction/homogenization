"""Exact finite Fourier certificate for the cyclic two-mode exclusion."""

from fractions import Fraction as F
from math import factorial

import numpy as np
from scipy.linalg import null_space


def add(x, y):
    return x[0] + y[0], x[1] + y[1]


def scale(c, x):
    return (c * x[0], c * x[1]) if c >= 0 else (c * x[1], c * x[0])


def mul(x, y):
    values = (x[0] * y[0], x[0] * y[1], x[1] * y[0], x[1] * y[1])
    return min(values), max(values)


def divide_positive(x, y):
    assert 0 < y[0] <= y[1]
    return x[0] / y[1], x[1] / y[0]


def arctan_reciprocal(q, terms):
    total = F(0)
    for k in range(terms):
        total += (-1) ** k * F(1, (2 * k + 1) * q ** (2 * k + 1))
    next_term = F(1, (2 * terms + 1) * q ** (2 * terms + 1))
    return (total, total + next_term) if terms % 2 == 0 else (total - next_term, total)


def pi_interval():
    # Machin: pi/4 = 4 atan(1/5) - atan(1/239).
    a, b = arctan_reciprocal(5, 24), arctan_reciprocal(239, 8)
    return scale(4, add(scale(4, a), scale(-1, b)))


def positive_sin_cos(x, terms=12):
    """Outward rational Taylor bounds for 0 <= x <= pi/2."""
    x2 = mul(x, x)
    sin_terms, cos_terms = [], []
    power = (F(1), F(1))
    for k in range(terms + 1):
        cos_terms.append(scale(F((-1) ** k, factorial(2 * k)), power))
        sin_terms.append(scale(F((-1) ** k, factorial(2 * k + 1)), mul(power, x)))
        power = mul(power, x2)

    def alternating_bounds(items):
        partial = (F(0), F(0))
        for item in items[:-1]:
            partial = add(partial, item)
        extended = add(partial, items[-1])
        return min(partial[0], extended[0]), max(partial[1], extended[1])

    return alternating_bounds(sin_terms), alternating_bounds(cos_terms)


def trig_pi_multiple(numerator, denominator, pi):
    """sin and cos of pi*numerator/denominator, reduced exactly by quadrants."""
    t = F(numerator, denominator) % 2
    quadrant = int(t / F(1, 2))
    local = t - F(quadrant, 2)
    if quadrant % 2:
        local = F(1, 2) - local
    sine, cosine = positive_sin_cos(scale(local, pi))
    sin_sign = 1 if quadrant in (0, 1) else -1
    cos_sign = 1 if quadrant in (0, 3) else -1
    return scale(sin_sign, sine), scale(cos_sign, cosine)


def margin(n, mode, orientation, jump, pi, threshold=(F(1), F(1))):
    sine, cosine = trig_pi_multiple(2 * mode * jump, n, pi)
    dissipation = add((F(1), F(1)), scale(-1, cosine))
    return add(scale(orientation, sine), scale(-1, mul(threshold, dissipation)))


def exact_fourier_table():
    pi = pi_interval()
    assert pi[0] < pi[1] and pi[1] - pi[0] < F(1, 10**30)
    checked, worst = 0, F(-100)

    # For each pair and orientation, alpha*h_k+h_l <= 0 jump by jump.
    tables = {
        5: {(1, 2): lambda e, d: F(1)},
        6: {(1, 2): lambda e, d: F(1)},
        7: {
            (1, 2): lambda e, d: F(1, 5) if e == d else F(1, 2),
            (1, 3): lambda e, d: F(1, 5) if e == d else F(2),
            (2, 3): lambda e, d: F(1, 2) if e == d else F(1, 5),
        },
    }
    for n, pairs in tables.items():
        for (k, ell), alpha_rule in pairs.items():
            for e in (-1, 1):
                for d in (-1, 1):
                    alpha = alpha_rule(e, d)
                    for jump in range(1, n):
                        value = add(scale(alpha, margin(n, k, e, jump, pi)),
                                    margin(n, ell, d, jump, pi))
                        assert value[1] < 0
                        worst = max(worst, value[1])
                        checked += 1

    # At n=8 use exact (a+b*sqrt(2))/2 tables. The second mode never
    # exceeds one; modes one and three have h_1+h_3 <= 0.
    sine = ((0, 0), (0, 1), (2, 0), (0, 1), (0, 0), (0, -1), (-2, 0), (0, -1))
    cosine = ((2, 0), (0, 1), (0, 0), (0, -1), (-2, 0), (0, -1), (0, 0), (0, 1))
    sqrt2 = (F(1414213, 10**6), F(1414214, 10**6))
    assert sqrt2[0] ** 2 < 2 < sqrt2[1] ** 2

    def eighth_margin(mode, orientation, jump):
        index = mode * jump % 8
        a = orientation * sine[index][0] + cosine[index][0] - 2
        b = orientation * sine[index][1] + cosine[index][1]
        return scale(F(1, 2), add((F(a), F(a)), scale(b, sqrt2)))

    for orientation in (-1, 1):
        for jump in range(1, 8):
            assert eighth_margin(2, orientation, jump)[1] <= 0
            checked += 1
    for e in (-1, 1):
        for d in (-1, 1):
            for jump in range(1, 8):
                value = add(eighth_margin(1, e, jump), eighth_margin(3, d, jump))
                assert value[1] <= 0
                checked += 1

    # At n=9 the sharp second-frequency bound is the positive root rho_9 of
    # the determinant for modes (1,+) and (4,-), supported on jumps 1 and 2.
    # A one-trillionth-wide rational bracket isolates that root.
    root_lo, root_hi = F(1245891659322, 10**12), F(1245891659323, 10**12)

    def determinant(threshold):
        h11 = margin(9, 1, 1, 1, pi, threshold)
        h12 = margin(9, 1, 1, 2, pi, threshold)
        h41 = margin(9, 4, -1, 1, pi, threshold)
        h42 = margin(9, 4, -1, 2, pi, threshold)
        return add(mul(h11, h42), scale(-1, mul(h12, h41)))

    assert determinant((root_lo, root_lo))[0] > 0
    assert determinant((root_hi, root_hi))[1] < 0

    def sine_and_dissipation(mode, orientation, jump):
        sine, cosine = trig_pi_multiple(2 * mode * jump, 9, pi)
        return scale(orientation, sine), add((F(1), F(1)), scale(-1, cosine))

    s11, d11 = sine_and_dissipation(1, 1, 1)
    s12, d12 = sine_and_dissipation(1, 1, 2)
    s41, d41 = sine_and_dissipation(4, -1, 1)
    s42, d42 = sine_and_dissipation(4, -1, 2)
    quadratic = (
        add(mul(d11, d42), scale(-1, mul(d12, d41))),
        add(add(scale(-1, mul(s11, d42)), scale(-1, mul(d11, s42))),
            add(mul(s12, d41), mul(d12, s41))),
        add(mul(s11, s42), scale(-1, mul(s12, s41))),
    )
    assert quadratic[0][1] < 0 and quadratic[2][0] > 0
    rho = (root_lo, root_hi)

    # Six sharp configurations form one orbit under multiplication of jumps
    # by units modulo nine and reversal.  In the canonical representative,
    # alpha makes jumps 1 and 2 equalities; the determinant defines the
    # second equality, and all remaining jumps are strictly separated.
    h11 = margin(9, 1, 1, 1, pi, rho)
    h12 = margin(9, 1, 1, 2, pi, rho)
    h41 = margin(9, 4, -1, 1, pi, rho)
    h42 = margin(9, 4, -1, 2, pi, rho)
    assert h11[0] > 0 and h12[1] < 0 and h41[1] < 0 and h42[0] > 0
    alpha = divide_positive(scale(-1, h41), h11)
    ninth_worst = F(-100)
    for jump in range(3, 9):
        value = add(mul(alpha, margin(9, 1, 1, jump, pi, rho)),
                    margin(9, 4, -1, jump, pi, rho))
        assert value[1] < 0
        ninth_worst = max(ninth_worst, value[1])
        checked += 1

    # Every nonsharp pair/orientation is separated by the common multiplier
    # alpha=1, with a wide strict margin.  The omitted six cases are exactly
    # the symmetry orbit of the canonical sharp representative above.
    sharp = {
        (1, 2, -1, -1), (1, 2, 1, 1),
        (1, 4, -1, 1), (1, 4, 1, -1),
        (2, 4, -1, -1), (2, 4, 1, 1),
    }
    for k in range(1, 5):
        for ell in range(k + 1, 5):
            for e in (-1, 1):
                for d in (-1, 1):
                    if (k, ell, e, d) in sharp:
                        continue
                    for jump in range(1, 9):
                        value = add(margin(9, k, e, jump, pi, rho),
                                    margin(9, ell, d, jump, pi, rho))
                        assert value[1] < 0
                        ninth_worst = max(ninth_worst, value[1])
                        checked += 1

    # The canonical two-jump generator has positive weights.  At the exact
    # determinant root its two signed sums vanish simultaneously.
    weight_one = divide_positive(scale(-1, h12), add(h11, scale(-1, h12)))
    assert F(0) < weight_one[0] <= weight_one[1] < F(1)

    # The sharp root lies strictly below the paper's two-mode target.
    sin_fifth, cos_fifth = trig_pi_multiple(1, 5, pi)
    target = divide_positive(cos_fifth, sin_fifth)
    assert target[0] > root_hi > 1
    print(f"Exact rational Fourier inequalities: {checked}; n<=7 worst < {float(worst):.6g}; n=9 non-equality worst < {float(ninth_worst):.6g}")
    print(f"Sharp nine-state root: {float(root_lo):.12f} < rho_9 < {float(root_hi):.12f}; positive jump-1 weight in [{float(weight_one[0]):.12f}, {float(weight_one[1]):.12f}]")


def frequencies(rates):
    n = len(rates) + 1
    result = []
    for k in range(1, (n - 1) // 2 + 1):
        angles = 2 * np.pi * k * np.arange(1, n) / n
        a = rates @ (1 - np.cos(angles))
        b = rates @ np.sin(angles)
        result.append(abs(b) / a)
    return np.sort(result)[::-1]


def sharp_ninth_data():
    def parts(mode, orientation, jump):
        angle = 2 * np.pi * mode * jump / 9
        return orientation * np.sin(angle), 1 - np.cos(angle)

    s11, d11 = parts(1, 1, 1)
    s12, d12 = parts(1, 1, 2)
    s41, d41 = parts(4, -1, 1)
    s42, d42 = parts(4, -1, 2)
    coefficients = (
        d11 * d42 - d12 * d41,
        -s11 * d42 - d11 * s42 + s12 * d41 + d12 * s41,
        s11 * s42 - s12 * s41,
    )
    rho = max(np.roots(coefficients))
    h11, h12 = s11 - rho * d11, s12 - rho * d12
    weight_one = -h12 / (h11 - h12)
    return float(rho), float(weight_one)


def matrix_checks():
    rng, comparisons, maximum = np.random.default_rng(253), 0, 0.0
    rho_nine, weight_one = sharp_ninth_data()
    for n in range(5, 10):
        basis = null_space(np.ones((1, n)))
        for _ in range(40):
            rates = 10 ** rng.uniform(-2, 2, n - 1)
            q = np.zeros((n, n))
            for jump, rate in enumerate(rates, 1):
                q[np.arange(n), (np.arange(n) + jump) % n] = rate
            np.fill_diagonal(q, -q.sum(axis=1))
            form = basis.T @ (-q) @ basis / n
            h, a = (form + form.T) / 2, (form - form.T) / 2
            values, vectors = np.linalg.eigh(h)
            root = vectors @ np.diag(values ** -0.5) @ vectors.T
            observed = np.linalg.svd(root @ a @ root, compute_uv=False)[::2]
            expected = frequencies(rates)
            maximum = max(maximum, np.max(abs(observed[:len(expected)] - expected)))
            if n <= 8:
                assert sum(observed > 1 + 2e-12) <= 1
            else:
                assert observed[1] <= rho_nine + 2e-12
            comparisons += 1
    sharp_rates = np.zeros(8)
    sharp_rates[:2] = weight_one, 1 - weight_one
    sharp = frequencies(sharp_rates)
    assert abs(sharp[0] - rho_nine) < 2e-12
    assert abs(sharp[1] - rho_nine) < 2e-12
    assert maximum < 2e-11
    print(f"{comparisons} dense circulant generators: Fourier/matrix max error {maximum:.3g}")
    print(f"Sharp two-jump realization: rho_9={rho_nine:.12f}, weights=({weight_one:.12f}, {1-weight_one:.12f})")


def main():
    exact_fourier_table()
    matrix_checks()
    print("Compression interlacing: a four-feature block has second frequency at most the full second frequency")
    print("At nine states the second full frequency is at most rho_9 < cot(pi/5), with equality attained")
    print("PASS")


if __name__ == "__main__":
    main()
