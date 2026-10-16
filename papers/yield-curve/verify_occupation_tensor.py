"""A full long-run third occupation tensor fails to identify four-state chains."""
from fractions import Fraction as F
from itertools import combinations_with_replacement, permutations
from math import factorial

import numpy as np
from scipy.linalg import eigvals, expm


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


def third(h, a, b, c):
    # Polarization of 6 <f,H(f Hf)> under uniform pi.
    return sum(sum(x*(h @ (y*(h @ z))))/4 for x, y, z in permutations((a, b, c)))


def main():
    identity = np.eye(4, dtype=int); ones = np.ones(4, dtype=int)
    pi = np.array([F(1, 4)]*4); projection = np.outer(ones, pi)
    centered = identity-projection; u = np.array([1, 2, 3, -6])
    g = centered+np.outer(u, u)*F(1, 100)
    a = np.array([[0, 1, 0, -1], [-1, 0, 1, 0], [0, -1, 0, 1], [1, 0, -1, 0]])
    b = np.array([[0, -1, 1, 0], [1, 0, 0, -1], [-1, 0, 0, 1], [0, 1, -1, 0]])
    hs = [g+F(1, 10)*k for k in (a, b)]
    qs = [projection-inverse(projection+h) for h in hs]
    for q, h in zip(qs, hs):
        assert np.all(q @ ones == 0) and np.all(pi @ q == 0)
        assert np.all(inverse(projection-q)-projection == h)
        assert np.all((h+h.T)/2 == g)
        assert min(q[~np.eye(4, dtype=bool)]) >= F(2, 31)
    traces = [-np.trace(q) for q in qs]
    assert traces == [F(5050, 1933), F(404, 155)] and traces[0] != traces[1]
    for perm in permutations(range(4)):
        assert np.any(qs[0][np.ix_(perm, perm)] != qs[1])
        assert np.any(qs[0].T[np.ix_(perm, perm)] != qs[1])
    basis = [identity[i]-identity[3] for i in range(3)]
    for i, j, k in combinations_with_replacement(range(3), 3):
        x, y, z = (basis[t] for t in (i, j, k))
        assert third(hs[0], x, y, z) == third(hs[1], x, y, z)
        for h, skew in zip(hs, (a*F(1, 10), b*F(1, 10))):
            polarized = 2*sum(sum(x*((g @ y)*(g @ z)-(skew @ y)*(skew @ z)))
                              for x, y, z in ((x, y, z), (y, x, z), (z, x, y)))/4
            assert third(h, x, y, z) == polarized
    # Exact coefficients of the raw homogeneous cubic identity on all four coordinates.
    for i, j, k in combinations_with_replacement(range(4), 3):
        coefficient = sum(sum(identity[x]*((a @ identity[y])*(a @ identity[z])
                                          -(b @ identity[y])*(b @ identity[z])))
                          for x, y, z in set(permutations((i, j, k))))
        # (-f1+f2)(f3-f4)(f1+f2+f3+f4).
        target = sum((-1, 1, 0, 0)[x]*(0, 0, 1, -1)[y]
                     for x, y, z in set(permutations((i, j, k))))
        assert coefficient == target
    delta = hs[0] @ hs[0]-hs[1] @ hs[1]
    delta = (delta+delta.T)/2  # Only the symmetric boundary enters stationary covariance.
    assert delta[0, 2] == F(1, 50) and np.all(delta == delta.T)
    f = np.array([1, 0, -1, 0]); rate2 = 2*(pi*f) @ g @ f
    rate3 = third(hs[0], f, f, f)
    boundaries = [(pi*f) @ h @ h @ f for h in hs]
    limit = -2*(boundaries[0]-boundaries[1])
    assert rate2 == F(51, 50) and rate3 == F(-153, 625)
    assert boundaries == [F(101, 200), F(103, 200)] and limit == F(1, 50)
    # ||G||=3/2, ||A||=||B||=2, hence ||H||<=17/10; gap>=4 min(q_ij)>=8/31.
    assert sum(u*u) == 50
    for k in (a, b):
        assert np.all(k.T @ k @ k.T @ k == 4*(k.T @ k))
    exp_lower = sum(F(6**j, factorial(j)) for j in range(8))
    assert F(8*24, 31) > 6 and exp_lower > 289
    print(f"positive-rate minima: {[min(q[~np.eye(4, dtype=bool)]) for q in qs]}")
    print(f"exit-rate sums (exclude relabeling and reversal): {traces}")
    print("10 exact symmetric third-tensor components; 20 exact raw cubic coefficients; 48 excluded relabelings/reversals")
    print(f"f=(1,0,-1,0): common variance rate={rate2}, third rate={rate3}; boundaries={boundaries}")
    print(f"variance difference limit={limit}; remainder <= (289/50) exp(-8T/31)")
    print("norm bounds: ||G||=3/2, ||A||=||B||=2, ||H_J||<=17/10; ||f||_pi^2=1/2; gap>=8/31")
    print(f"T>=24 gives positive difference: exp(6) > {exp_lower} > 289")
    rng = np.random.default_rng(20261021); worst = 0.; spectral = 0
    for _ in range(12):
        feature = rng.integers(-3, 4, 4); feature = feature-np.mean(feature)
        if not np.any(feature):
            feature = np.array([1., 0., -1., 0.])
        for q, h in zip(qs, hs):
            qf = np.array(q, dtype=float); gap = sorted(abs(eigvals(qf)))[1]
            truth = float(third(h, feature, feature, feature))
            for fraction in (.07, .1):
                radius = fraction*gap/max(abs(feature)); roots = np.exp(2j*np.pi*np.arange(48)/48)
                vals = [min(eigvals(qf+radius*z*np.diag(feature)), key=abs) for z in roots]
                estimate = 6*np.mean(np.array(vals)/roots**3)/radius**3
                error = abs(estimate-truth)/max(1., abs(truth)); worst = max(worst, error)
                assert error < 1e-7  # Independent floating check, not an interval proof.
                spectral += 1
    moment_error = 0.
    for t in (.1, 1., 4., 12., 24., 32.):
        values = []
        for q, h in zip(qs, hs):
            qf, hf = np.array(q, dtype=float), np.array(h, dtype=float)
            block = np.zeros((12, 12))
            for j in range(3):
                block[4*j:4*j+4, 4*j:4*j+4] = qf
                if j:
                    block[4*j:4*j+4, 4*j-4:4*j] = j*np.diag(f)
            variance = np.full(4, .25) @ (expm(t*block) @ np.r_[ones, np.zeros(8)])[8:]
            closed = float(rate2)*t-2*float((pi*f) @ h @ h @ f)+.5*f @ hf @ hf @ expm(t*qf) @ f
            scale = max(1., abs(variance), abs(closed)); error = abs(variance-closed)/scale
            moment_error = max(moment_error, error)
            assert error < 8192*np.finfo(float).eps
            values.append(variance)
        difference = values[0]-values[1]
        roundoff = 8192*np.finfo(float).eps*max(1., *map(abs, values))
        assert abs(difference-float(limit)) <= 5.78*np.exp(-8*t/31)+roundoff
        if t == 24.:
            assert difference > 0
            print(f"T=24 independent moment-semigroup variance difference={difference:.10f}")
    print(f"{spectral} spectral coefficients: maximum scaled error={worst:.3e}")
    print(f"12 independent moment-semigroup comparisons: maximum scaled error={moment_error:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
