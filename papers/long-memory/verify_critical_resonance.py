#!/usr/bin/env python3
"""Exact atomic obstruction to the density-free critical spectral endpoint.

Integer phases avoid evaluating cosine at astronomically large arguments.
Analytic positive-tail bounds are separate from scaled roundoff budgets.
"""
import math
import sys
from fractions import Fraction


def depth(k):
    """max(1, ceil(log2(k+1)/4)), evaluated exactly at powers of 16."""
    return max(1, (k.bit_length() + 3) // 4)


def check_blocks():
    cases = 0
    for n in range(1, 9):
        j = 16 ** n - 1
        assert depth(j) == n and depth(j + 1) == n + 1
        for k in {0, 1, j // 2, j - 1, j, j + 1}:
            d = depth(k)
            size = 2 ** d
            assert sum(2 * l - 1 for l in range(size // 2 + 1, size + 1)) == 3 * size ** 2 // 4
            # Normalize lengths by h_k: no huge powers 2**j are formed.
            for l in range(size // 2, size):
                for q in (0, 1, 3):
                    z = Fraction(4 * l + q, 4)
                    cdf = Fraction(size ** 2, 4) + sum(
                        2 * a - 1 for a in range(size // 2 + 1, l + 1)
                    )
                    assert cdf == l ** 2
                    assert Fraction(size - 2, size) ** 2 <= cdf / z ** 2 <= 1
                    cases += 1
            if k <= j:
                assert k + d <= j + n  # all old phases are integer turns
    return cases


def tail_bound(n, extra):
    # K=j+n+extra; R_j**2 * 4**(-(K+1)).
    return 4 * math.pi ** 2 * 2.0 ** (-2 * (extra + 1))


def resonant_variance(n, extra):
    j = 16 ** n - 1
    exponent = j + n
    terms = []
    for k in range(j + 1, j + n + extra + 1):
        d = depth(k)
        turns_denominator = 2 ** (k + d - exponent)
        for l in range(2 ** (d - 1) + 1, 2 ** d + 1):
            phase = math.pi * (l % turns_denominator) / turns_denominator
            terms.append(4 * (2 * l - 1) / l ** 2 * math.sin(phase) ** 2)
    return math.fsum(terms)


def window_kernel(z, a, c):
    """E[1-cos(Uz)], U uniform on [a,c], avoiding small-z cancellation."""
    if c * abs(z) < 0.1:
        return math.fsum((-1) ** (r + 1) * z ** (2 * r)
                         * (c ** (2 * r + 1) - a ** (2 * r + 1))
                         / ((c - a) * (2 * r + 1) * math.factorial(2 * r))
                         for r in range(1, 7))
    return 1 - (math.sin(c * z) - math.sin(a * z)) / ((c - a) * z)


def averaged_resonant_variance(n, a, c, extra=30):
    """Integer window endpoints cancel all old cosine averages exactly."""
    j = 16 ** n - 1
    value = 0.0
    for d in range(1, n + 1):
        blocks = 16 if d == 1 else 16 ** d - 16 ** (d - 1)
        value += 2 * blocks * math.fsum((2 * l - 1) / l ** 2
                                       for l in range(2 ** (d - 1) + 1, 2 ** d + 1))
    for k in range(j + 1, j + n + extra + 1):
        d = depth(k)
        for l in range(2 ** (d - 1) + 1, 2 ** d + 1):
            z = 2 * math.pi * l * 2.0 ** (j + n - k - d)
            value += 2 * (2 * l - 1) / l ** 2 * window_kernel(z, a, c)
    return value


def check_window_bound():
    from scipy.integrate import quad
    errors = []
    for a, c in ((0, 1), (1, 2), (0.9, 1.1)):
        for scale in (0.25, 1, 4, 16, 64):
            atoms = ((0.03, 0.7), (0.7, 0.2), (2.1, 0.1))
            value = math.fsum(2 * w / x ** 2 * window_kernel(scale * x, a, c) for x, w in atoms)
            reference, quad_error = quad(lambda u: math.fsum(
                4 * w / x ** 2 * math.sin(u * scale * x / 2) ** 2 for x, w in atoms),
                a, c, epsabs=1e-9, epsrel=1e-11)
            errors.append(abs(value - reference / (c - a)))
            budget = 65536 * sys.float_info.epsilon * max(1, value)
            assert errors[-1] <= budget + quad_error / (c - a)
            cut_moment = math.fsum(w / x ** 2 for x, w in atoms if x > 1 / scale)
            bound = ((a * a + a * c + c * c) / 3 * scale ** 2
                     * math.fsum(w for x, w in atoms if x <= 1 / scale)
                     + 4 / ((c - a) * scale)
                     * math.fsum(w / x ** 3 for x, w in atoms if x > 1 / scale))
            assert abs(value - 2 * cut_moment) <= bound + budget
    return max(errors)


def check_bv_smoothing():
    """Check three nonuniform BV maturity laws and the general finite-R bound."""
    from scipy.integrate import quad
    atoms = ((0.03, 0.7), (0.7, 0.2), (2.1, 0.1))
    densities = (
        (0.0, 1.0, lambda u: 2 * (1 - u), Fraction(1, 6), 4.0),
        (0.0, 2.0, lambda u: u if u <= 1 else 2 - u, Fraction(7, 6), 2.0),
        (0.0, 1.0, lambda u: 6 * u * (1 - u), Fraction(3, 10), 3.0),
    )
    errors = []
    for a, c, density, exact_m2, variation in densities:
        mass, mass_error = quad(density, a, c, epsabs=1e-11, epsrel=1e-11)
        m2, m2_error = quad(lambda u: u * u * density(u), a, c,
                            epsabs=1e-11, epsrel=1e-11)
        assert abs(mass - 1) <= 1e-12 + mass_error
        assert abs(m2 - float(exact_m2)) <= 1e-12 + m2_error
        for scale in (0.25, 1, 4, 16, 64):
            value, value_error = quad(lambda u: density(u) * math.fsum(
                4 * w / x ** 2 * math.sin(u * scale * x / 2) ** 2
                for x, w in atoms), a, c, epsabs=1e-9, epsrel=1e-11)
            spectral_terms = []
            spectral_error = 0.0
            for x, w in atoms:
                cosine, cosine_error = quad(
                    lambda u: density(u) * math.cos(scale * x * u),
                    a, c, epsabs=1e-11, epsrel=1e-11)
                spectral_terms.append(2 * w / x ** 2 * (1 - cosine))
                spectral_error += 2 * w / x ** 2 * cosine_error
            spectral = math.fsum(spectral_terms)
            errors.append(abs(value - spectral))
            budget = 65536 * sys.float_info.epsilon * max(1, value, spectral)
            assert errors[-1] <= budget + value_error + spectral_error
            cut_moment = math.fsum(w / x ** 2 for x, w in atoms if x > 1 / scale)
            bound = (m2 * scale ** 2
                     * math.fsum(w for x, w in atoms if x <= 1 / scale)
                     + 2 * variation / scale
                     * math.fsum(w / x ** 3 for x, w in atoms if x > 1 / scale))
            assert abs(value - 2 * cut_moment) <= bound + budget + value_error
    return max(errors)


def main():
    cases = check_blocks()
    print(f"exact CDF bound checks: {cases}; total spectral mass: 1")
    print("n j variance ratio-to-4logR analytic-ratio-upper-bound")
    for n in range(1, 9):
        j = 16 ** n - 1
        value = resonant_variance(n, 30)
        shorter = resonant_variance(n, 12)
        budget = 65536 * sys.float_info.epsilon * max(1.0, value)
        assert -budget <= value - shorter <= tail_bound(n, 12) + budget
        bound = math.pi ** 2 * 4 ** n
        assert value + tail_bound(n, 30) <= bound + budget
        log_r = math.log(2 * math.pi) + (j + n) * math.log(2)
        ratio = value / (4 * log_r)
        upper = bound / (4 * log_r)
        assert ratio <= upper + budget
        print(f"{n} {j} {value:.12f} {ratio:.9e} {upper:.9e}")
    print(f"omitted variance bound after K=j+n+30: {tail_bound(8, 30):.9e}")
    print(f"15 window identity/bound checks; quadrature error: {check_window_bound():.3e}")
    print(f"15 nonuniform BV smoothing checks; quadrature error: {check_bv_smoothing():.3e}")
    print("n averaged ratios for U uniform on [0,1] and [1,2]")
    for n in range(1, 9):
        denominator = 4 * (math.log(2 * math.pi) + (16 ** n + n - 1) * math.log(2))
        values = []
        for a, c in ((0, 1), (1, 2)):
            value = averaged_resonant_variance(n, a, c)
            shorter = averaged_resonant_variance(n, a, c, extra=12)
            budget = 65536 * sys.float_info.epsilon * max(1, value)
            bound = (a * a + a * c + c * c) / 3 * tail_bound(n, 12)
            assert -budget <= value - shorter <= bound + budget
            values.append(value / denominator)
        print(f"{n} {values[0]:.9f} {values[1]:.9f}")
    print(f"[1,2] omitted averaged variance bound: {7 / 3 * tail_bound(8, 30):.9e}")
    print("PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FAIL")
        raise

