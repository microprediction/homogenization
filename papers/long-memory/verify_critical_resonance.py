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
    print("PASS")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("FAIL")
        raise
