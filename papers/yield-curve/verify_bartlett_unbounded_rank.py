"""Exact rational certificates for the one-cycle Bartlett skew-rank family.

The all-r and all-small-T proof is in bartlett-unbounded-rank.html.
Floating matrix exponentials and quadrature are independent value checks only.
"""
from fractions import Fraction as R
from itertools import combinations
from math import factorial

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm


def family(r):
    n = 2 * r + 1
    Q = np.zeros((n, n), dtype=object)
    for i, j in [(0, r), (r, 2 * r), (2 * r, 0)]:
        Q[i, j] = 1
    for root in (0, r):
        for i in range(root, root + r - 1):
            Q[i, i + 1] = Q[i + 1, i] = 1
    for i in range(n):
        Q[i, i] = -sum(Q[i])
    return Q


def inverse_det(A):
    n = len(A)
    B = np.column_stack((A, np.eye(n, dtype=object))).copy()
    det = R(1)
    for j in range(n):
        p = next(i for i in range(j, n) if B[i, j])
        if p != j:
            B[[j, p]] = B[[p, j]]
            det = -det
        pivot = B[j, j]
        det *= pivot
        B[j] = B[j] / pivot
        for i in range(n):
            if i != j:
                B[i] -= B[i, j] * B[j]
    return B[:, n:], det


def norm(A):
    return max(sum(abs(x) for x in row) for row in A)


def close(A, B):
    assert np.linalg.norm(A - B, np.inf) < 3e-8 * (1 + np.linalg.norm(B, np.inf))


def main():
    degree = 40
    print("Exact rational arithmetic; degree", degree)
    print("Bounds: ||Q||_infinity <= 4; entry remainder <= 54 T; tau denominator 108 r mu")
    for r in range(1, 7):
        Q = family(r)
        n = len(Q)
        assert not any(Q.sum(axis=0)) and not any(Q.sum(axis=1))
        assert norm(Q) <= 4
        C = np.array([[R(1, 2 * factorial(i + j + 3)) for j in range(r)]
                      for i in range(r)], dtype=object)
        Ci, det = inverse_det(C)
        formula = R((-1) ** (r * (r - 1) // 2), 2 ** r)
        for j in range(r):
            formula *= R(factorial(j), factorial(r + j + 2))
        assert det == formula
        mu = norm(Ci)
        tau = min(R(1), 1 / (108 * r * mu))
        assert 54 * r * mu * tau <= R(1, 2)
        power = np.eye(n, dtype=object)
        X = np.zeros((r, r), dtype=object)
        W = X.copy()
        for k in range(degree + 1):
            assert np.array_equal(power[:r, :r], power[:r, :r].T)
            assert np.array_equal(power[r:2*r, r:2*r], power[r:2*r, r:2*r].T)
            delta = power[:r, r:2*r] - power[r:2*r, :r].T
            for i in range(r):
                for j in range(r):
                    d = i + j + 1
                    if k < d:
                        assert delta[i, j] == 0
                    if k == d:
                        assert delta[i, j] == 1
                    X[i, j] += R(delta[i, j], 2 * factorial(k + 2))
                    if k >= d:
                        W[i, j] += R(delta[i, j], 2 * factorial(k + 2)) * tau ** (k-d)
            power = power @ Q
        # Exact Neumann certificates at T=1 and at the theorem's tau.
        tail = R(4 ** 41, factorial(43)) / (1 - R(4, 44))
        Xi, _ = inverse_det(X)
        budget = norm(Xi) * r * tail
        assert budget < R(1, 2)
        scaled_tail = R(4 ** 41, factorial(43)) * tau ** (42-2*r) / (1-4*tau/44)
        assert mu * (norm(W-C) + r * scaled_tail) < R(1, 2)
        # Independent exact long-run computation on the complete centered basis.
        H, _ = inverse_det(-Q + np.full((n, n), R(1, n), dtype=object))
        skewQ = (Q-Q.T) / R(2)
        assert np.array_equal((H-H.T)/R(2), H.T @ skewQ @ H)
        F = np.vstack((np.eye(n-1, dtype=object), -np.ones((1, n-1), dtype=object)))
        K = F.T @ H @ F / R(n)
        A = (K-K.T) / R(2)
        assert any(A.flat)
        for i, j, k, l in combinations(range(n-1), 4):
            assert A[i,j]*A[k,l]-A[i,k]*A[j,l]+A[i,l]*A[j,k] == 0
        q = np.array(Q, dtype=float)
        block = np.zeros((3*n, 3*n))
        block[:n, :n] = q
        block[:n, n:2*n] = block[n:2*n, 2*n:] = np.eye(n)
        M = expm(block)[:n, 2*n:]
        close((M[:r, r:2*r]-M[r:2*r, :r].T)/2, np.array(X, dtype=float))
        if r <= 3:
            integral, _ = quad_vec(lambda t: (1-t)*expm(t*q), 0, 1, epsabs=1e-10)
            close(M, integral)
        print(f"r={r} n={n} mu={mu} tau={tau} long-run skew rank=2 "
              f"Bartlett skew rank={2*r}; T=1 Neumann budget={float(budget):.8g}")
    print("PASS")


if __name__ == "__main__":
    main()
