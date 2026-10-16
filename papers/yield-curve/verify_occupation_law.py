"""Exact five-state equality of all stationary fixed-horizon occupation laws."""
from fractions import Fraction as F
from itertools import combinations, permutations
from math import prod

import numpy as np
from scipy.linalg import expm


def inverse(a):
    n = len(a)
    rows = [[F(a[i, j]) for j in range(n)]+[F(i == j) for j in range(n)] for i in range(n)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        divisor = rows[j][j]; rows[j] = [x/divisor for x in rows[j]]
        for i in range(n):
            if i != j:
                multiplier = rows[i][j]
                rows[i] = [x-multiplier*y for x, y in zip(rows[i], rows[j])]
    return np.array([row[n:] for row in rows], dtype=object)


def determinant(a):
    n = len(a)
    return sum((-1)**sum(p[i] > p[j] for i in range(n) for j in range(i+1, n))
               *prod(a[i, p[i]] for i in range(n)) for p in permutations(range(n)))


def pair(a_block, b_block):
    k, ell = len(a_block), len(b_block)
    outgoing, incoming = -a_block.sum(axis=1)/ell, -a_block.sum(axis=0)/ell
    q = np.block([[a_block, np.outer(outgoing, np.ones(ell, dtype=int))],
                  [np.outer(np.ones(ell, dtype=int), incoming), b_block]])
    rev = np.block([[a_block.T, np.outer(incoming, np.ones(ell, dtype=int))],
                    [np.outer(np.ones(ell, dtype=int), outgoing), b_block]])
    return q, rev, outgoing, incoming


def main():
    a = np.array([[-5, 1], [2, -7]], dtype=object)
    b = np.array([[-11, 4, 4], [2, -10, 5], [6, 3, -12]], dtype=object)
    a = np.vectorize(F)(a); b = np.vectorize(F)(b)
    q, rev, outgoing, incoming = pair(a, b)
    assert list(outgoing) == [F(4, 3), F(5, 3)] and list(incoming) == [F(1), F(2)]
    n = 5; ones = np.ones(n, dtype=int); identity = np.eye(n, dtype=int)
    projection = np.full((n, n), F(1, n)); hs = []
    for generator in (q, rev):
        assert np.all(generator @ ones == 0) and np.all(ones @ generator == 0)
        assert min(generator[~np.eye(n, dtype=bool)]) == 1
        hs.append(inverse(projection-generator)-projection)
    assert len(set(np.diag(q))) == n
    for p in permutations(range(n)):
        assert np.any(q[np.ix_(p, p)] != rev)
        assert np.any(q.T[np.ix_(p, p)] != rev)
    # Both denominator and numerator of the stationary scalar resolvent agree
    # as polynomials in arbitrary diagonal variables, not just at sampled points.
    minors = 0
    for rank_one in (0, 1):
        for size in range(1, n+1):
            for indices in combinations(range(n), size):
                left = (q-rank_one*np.ones((n, n), dtype=int))[np.ix_(indices, indices)]
                right = (rev-rank_one*np.ones((n, n), dtype=int))[np.ix_(indices, indices)]
                assert determinant(left) == determinant(right)
                minors += 1
    assert np.all(hs[0]+hs[0].T == hs[1]+hs[1].T)
    squares = [h @ h for h in hs]
    assert np.all(squares[0]+squares[0].T == squares[1]+squares[1].T)
    rng = np.random.default_rng(20261022)
    for _ in range(24):
        diagonal = np.diag([F(int(v), 10) for v in rng.integers(-9, 10, n)])
        z = F(int(rng.integers(2, 8)))
        r = inverse(z*np.eye(2, dtype=int)-a-diagonal[:2, :2])
        s = inverse(z*np.eye(3, dtype=int)-b-diagonal[2:, 2:])
        u, v = np.ones(2, dtype=int), np.ones(3, dtype=int)
        r0, s0 = u @ r @ u, v @ s @ v
        t, h, w = u @ r @ outgoing, incoming @ r @ u, incoming @ r @ outgoing
        scalar = (r0+s0*(1+t)*(1+h)/(1-s0*w))/n
        for generator in (q, rev):
            resolvent = inverse(z*identity-generator-diagonal)
            assert ones @ resolvent @ ones/n == scalar
    lag_slope = (q[0, 1]-rev[0, 1])/n
    assert lag_slope == F(-1, 5)
    print("five-state blocks: A=((-5,1),(2,-7)); B=((-11,4,4),(2,-10,5),(6,3,-12))")
    print(f"cross vectors: a={list(outgoing)}, d={list(incoming)}; both minimum rates=1")
    print(f"{minors} exact principal-minor equalities for Q and Q-11^T; 240 excluded relabelings/reversals")
    print("24 exact scalar Schur-resolvent identities; full Gram and covariance boundary equal")
    print(f"ordered stationary (1,2) lag-probability difference: slope={lag_slope}")
    pairs = [(np.array(q, dtype=float), np.array(rev, dtype=float))]
    for k, ell in ((2, 3), (3, 3), (2, 4), (3, 4)):
        for _ in range(4):
            a_block = rng.uniform(.1, 1., (k, k)); np.fill_diagonal(a_block, 0.)
            a_block -= np.diag(a_block.sum(axis=1)+ell*rng.uniform(1., 2., k))
            outgoing, incoming = -a_block.sum(1)/ell, -a_block.sum(0)/ell
            rates = rng.uniform(.5, 1., (ell, ell)); rates = (rates+rates.T)/2
            np.fill_diagonal(rates, 0.)
            for i, j in ((0, 1), (1, 2), (2, 0)):
                rates[i, j] += .2; rates[j, i] -= .2
            b_block = rates-np.diag(rates.sum(1)+sum(outgoing))
            left, right, _, _ = pair(a_block, b_block)
            assert min(incoming) > 0
            for generator in (left, right):
                assert np.max(abs(generator.sum(0))) < 8192*np.finfo(float).eps
                assert np.max(abs(generator.sum(1))) < 8192*np.finfo(float).eps
            pairs.append((left, right))
    worst = 0.; count = 0
    for left, right in pairs:
        size = len(left); unit = np.ones(size)
        for _ in range(8):
            diagonal = np.diag(rng.uniform(-.5, .5, size))
            for t in (.001, .05, .5, 2.):
                first = unit @ expm(t*(left+diagonal)) @ unit/size
                second = unit @ expm(t*(right+diagonal)) @ unit/size
                error = abs(first-second)/max(1., abs(first), abs(second))
                worst = max(worst, error)
                assert error < 8192*np.finfo(float).eps
                count += 1
    lag = (expm(.1*pairs[0][0])[0, 1]-expm(.1*pairs[0][1])[0, 1])/5
    assert lag < -.009
    print(f"T=.1 ordered lag-probability difference={lag:.10f}")
    print(f"{len(pairs)} block pairs, {count} finite-horizon occupation transforms; maximum scaled error={worst:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
