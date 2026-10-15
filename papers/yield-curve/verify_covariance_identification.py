"""Generator identification from complete stationary covariance rate and boundary."""
import os
import sys
from fractions import Fraction as F
from itertools import permutations

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm, null_space

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "general"))
from effective_generator import stationary


def sym(a):
    return (a+a.T)/2


def close(a, b):
    assert np.linalg.norm(a-b, 2) <= 4096*np.finfo(float).eps*max(1, np.linalg.norm(b, 2))


def inverse(a):
    n = len(a)
    rows = [[F(a[i, j]) for j in range(n)]+[F(i == j) for j in range(n)] for i in range(n)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if rows[i][j])
        rows[j], rows[pivot] = rows[pivot], rows[j]
        divisor = rows[j][j]
        rows[j] = [x/divisor for x in rows[j]]
        for i in range(n):
            if i != j:
                multiplier = rows[i][j]
                rows[i] = [x-multiplier*y for x, y in zip(rows[i], rows[j])]
    return np.array([row[n:] for row in rows], dtype=object)


def main():
    rng = np.random.default_rng(20261015)
    bounds, quadratures = 0, 0
    for n in (2, 3, 4):
        for trial in range(12):
            q = rng.uniform(0.1, 2, (n, n))
            np.fill_diagonal(q, 0)
            q -= np.diag(q.sum(axis=1))
            pi = stationary(q)
            u = null_space(np.sqrt(pi)[None, :])/np.sqrt(pi)[:, None]
            l = -u.T@(pi[:, None]*q)@u
            h = np.linalg.inv(l)
            g, b = sym(h), sym(h@h)
            k = (h-h.T)/2
            close(b-g@g, k@k)
            if n == 2:
                recovered = np.zeros((1, 1))
            elif n == 3:
                magnitude = np.sqrt(max(0, -np.trace(b-g@g)/2))
                recovered = magnitude*np.array([[0, 1], [-1, 0]])
            else:
                ww = b-g@g-np.trace(b-g@g)*np.eye(3)/2
                eigenvalues, v = np.linalg.eigh(sym(ww))
                w = np.sqrt(max(0, eigenvalues[-1]))*v[:, -1]
                recovered = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
            if np.linalg.norm(recovered-k) > np.linalg.norm(recovered+k):
                recovered = -recovered
            close(recovered, k)
            close(-u@np.linalg.solve(g+recovered, u.T*pi), q)
            close(-u@np.linalg.solve(g-recovered, u.T*pi), q.T*pi[None, :]/pi[:, None])
            gap = np.linalg.eigvalsh(sym(l))[0]
            for t in (0.1, 0.5, 2, 8, 32):
                remainder = sym(h@h@expm(-t*l))
                covariance = 2*t*g-2*b+2*remainder
                rounding = 4096*np.finfo(float).eps*max(1, t*np.linalg.norm(g, 2))
                assert np.linalg.norm(t*g-covariance/2-b, 2) <= np.linalg.norm(h, 2)**2*np.exp(-gap*t)+rounding
                bounds += 1
            if trial < 2:
                t = 1.3
                integral, error = quad_vec(lambda s: 2*(t-s)*sym(u.T@(pi[:, None]*expm(s*q))@u), 0, t)
                exact = 2*t*g-2*b+2*sym(h@h@expm(-t*l))
                assert np.linalg.norm(integral-exact, 2) <= max(1e-10, 4*error)
                quadratures += 1
    print(f"36 low-state reconstructions; {bounds} finite-horizon bounds; {quadratures} independent quadratures")

    p = np.array([1, 1, 1, -3, 0], dtype=object)
    q = np.array([1, 1, 1, 1, -4], dtype=object)
    projection = np.full((5, 5), F(1, 5), dtype=object)
    identity = np.array([[F(i == j) for j in range(5)] for i in range(5)])
    centered = identity-projection
    a = np.zeros((5, 5), dtype=object)
    a[:3, :3] = [[0, 1, -1], [-1, 0, 1], [1, -1, 0]]
    d = (np.outer(p, q)-np.outer(q, p))/F(4)
    s = centered+np.outer(p, p)/F(1000)
    hs = [s+(a+sign*d)/F(100) for sign in (1, -1)]
    qs = [projection-inverse(projection+h) for h in hs]
    assert np.all(a@d == 0) and np.all(d@a == 0)
    assert np.all(a@s == s@a)  # orthogonal invariant planes; s is identity on range(a)
    assert np.all(a@p == 0) and np.all(a@q == 0)
    assert np.all(sym(hs[0]@hs[0]) == sym(hs[1]@hs[1]))
    for generator, h in zip(qs, hs):
        assert np.all(generator.sum(axis=0) == 0) and np.all(generator.sum(axis=1) == 0)
        assert np.all((-generator)@h == centered)
        minimum = min(generator[i, j] for i in range(5) for j in range(5) if i != j)
        assert minimum >= F(172, 1185)
    assert not np.all(qs[0] == qs[1]) and not np.all(qs[0] == qs[1].T)
    assert all(not np.all(qs[0] == qs[1][np.ix_(z, z)]) for z in permutations(range(5)))
    for t in (0.01, 0.1, 0.5, 1, 4, 16):
        close(sym(expm(t*np.array(qs[0], float))), sym(expm(t*np.array(qs[1], float))))
    print("n<=4: (G,B) identifies Q up to reversal; n=5: whole covariance curve need not")
    print("Five-state parameters: 1/1000, 1/100; norm bound=13/250; inverse bound=13/237")
    print("Certified positive-rate lower bound=172/1185; 120 exact relabelings excluded")
    print("Exact rational squared-inverse equality; six symmetric-semigroup comparisons")
    print("PASS")


if __name__ == "__main__":
    main()
