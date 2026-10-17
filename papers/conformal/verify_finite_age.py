"""Finite-window exponential ages: exact rank mixtures, not Monte Carlo."""
import math

import numpy as np
import sympy as sp
from scipy.linalg import expm


def check(condition):
    if not condition:
        raise AssertionError("finite-age certificate failed")


def row_exp(initial, matrix, nodes, exponent):
    """Exact exp(c log(2) M) by interpolation at distinct integer eigenvalues."""
    coefficients = [sp.Rational(2) ** (exponent * x) for x in nodes]
    for order in range(1, len(nodes)):
        for j in range(len(nodes) - 1, order - 1, -1):
            coefficients[j] = ((coefficients[j] - coefficients[j - 1])
                               / (nodes[j] - nodes[j - order]))
    result = coefficients[-1] * initial
    identity = sp.eye(matrix.rows)
    for j in range(len(nodes) - 2, -1, -1):
        result = result * (matrix - nodes[j] * identity) + coefficients[j] * initial
    return result


def rank_polynomials(n, k, w):
    u = sp.Symbol("u")
    tail = sum(sp.binomial(n, j) * u**j * (1-u)**(n-j) for j in range(k))
    low = sp.Poly(sp.integrate(sp.expand(tail.subs(u, w*u)), (u, 0, 1)), w)
    high = sp.Poly(sp.integrate(sp.expand(tail.subs(u, w+(1-w)*u)), (u, 0, 1)), w)
    return low, high


def expectation(poly, moments):
    return sum(coefficient * moments[power[0]] for power, coefficient in poly.terms())


def main():
    # Z=1 in state 0, pi=(3/10,7/10); emissions U[0,1] and U[2,3].
    nmax, degree = 8, 9
    q = sp.Matrix([[-7, 7], [3, -3]])
    feature = sp.diag(1, 0)
    matrix = sp.zeros(2*(degree+1))
    for j in range(degree+1):
        matrix[2*j:2*j+2, 2*j:2*j+2] = q - j*sp.eye(2)
        if j:
            matrix[2*j-2:2*j, 2*j:2*j+2] = j*feature
    nodes = list(range(-19, 1))
    # The block spectra are {-j,-j-10}, j=0,...,9: no repeated eigenvalues.
    check(set(nodes) == {-j-offset for j in range(degree+1) for offset in (0, 10)})
    initial = [sp.zeros(1, matrix.rows) for _ in range(2)]
    initial[0][0] = sp.Rational(3, 10)
    initial[1][1] = sp.Rational(7, 10)
    w = sp.Symbol("w")
    polys = {(n, k): rank_polynomials(n, k, w)
             for n in range(1, nmax+1) for k in range(1, n+1)}
    identities, ranks, largest_error = 0, 0, 0.0
    witness = None
    for c in (1, 2, 5):
        delta = sp.Rational(1, 2)**c
        rows = [row_exp(start, matrix, nodes, c) for start in initial]
        numeric = expm(float(c*sp.log(2)) * np.array(matrix).astype(float))
        for start, row in zip(initial, rows):
            expected = np.array(start).astype(float) @ numeric
            actual = np.array(row).astype(float)
            error = np.max(np.abs(actual-expected) / np.maximum(1.0, np.abs(actual)))
            largest_error = max(largest_error, error)
            check(error < 2e-9)
        total, current, boundary = [], [], []
        for j in range(degree+1):
            scale = (1-delta)**(-j)
            total.append(scale * sum(row[2*j]+row[2*j+1] for row in rows))
            current.append(scale * sum(row[2*j] for row in rows))
            boundary.append(scale * (rows[0][2*j]+rows[0][2*j+1]))
        for j in range(degree):
            check(current[j] == (1-delta)*total[j+1] + delta*boundary[j])
            identities += 1
        for (n, k), (low, high) in polys.items():
            difference = low-high
            coverage = expectation(high, total) + expectation(difference, current)
            boundary_rank = expectation(high, total) + expectation(difference, boundary)
            target = sp.Rational(k, n+1)
            check(coverage == (1-delta)*target + delta*boundary_rank)
            check(0 <= boundary_rank <= 1)
            check((1-delta)*target <= coverage <= (1-delta)*target+delta)
            ranks += 1
            if c == 1 and n == k == 2:
                witness = coverage
                check(coverage != target)
    print(f"Exact filter identities: {identities}; exact rank-mixture checks: {ranks}")
    print(f"Independent scipy expm maximum scaled error: {largest_error:.3e}")
    print(f"L=log(2), N=k=2 coverage: {witness} = {float(witness):.12f}; target 2/3")
    delta = math.exp(-10)
    print(f"beta=1, L=10, N=k=9: boundary weight {delta:.12g}")
    print(f"Undercoverage budget {0.9*delta:.12g}; overcoverage budget {0.1*delta:.12g}")
    print(f"Coverage envelope [{0.9*(1-delta):.13f}, {0.9+0.1*delta:.13f}]")
    old = -math.expm1(9*math.log1p(-delta))
    print(f"Full-panel coupling budget 1-(1-delta)^9: {old:.13g}")
    print("PASS")


if __name__ == "__main__":
    main()
