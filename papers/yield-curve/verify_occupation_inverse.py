"""One occupation third-cumulant rate resolves a full three-state GK Gram."""
from fractions import Fraction as F
from itertools import permutations
from math import isqrt

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


def multiply(p, q):
    return [sum(p[i]*q[k-i] for i in range(k+1)) for k in range(4)]


def characteristic_residual(q, f, second, third):
    out = [F(0)]*4
    for perm in permutations(range(3)):
        sign = (-1)**sum(perm[i] > perm[j] for i in range(3) for j in range(i+1, 3))
        term = [F(1), F(0), F(0), F(0)]
        for i, j in enumerate(perm):
            entry = [-q[i, j], -f[i] if i == j else F(0), second if i == j else F(0), third if i == j else F(0)]
            term = multiply(term, entry)
        out = [x+sign*y for x, y in zip(out, term)]
    return out


def check(q):
    identity, ones = np.eye(3, dtype=int), np.ones(3, dtype=int)
    pi = inverse(q.T+np.ones((3, 3), dtype=int)) @ ones
    projection = np.outer(ones, pi)
    h = inverse(projection-q)-projection
    adjoint = np.diag(1/pi) @ h.T @ np.diag(pi)
    g, k = (h+adjoint)/2, (h-adjoint)/2
    square = -np.trace(k @ k)/2
    cycle = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=object)
    amplitude = pi[0]*k[0, 1]
    assert np.all(k == amplitude*np.diag(1/pi) @ cycle)
    assert np.all(k @ k == -square*(identity-projection))
    taus = []
    for i in range(3):
        f = np.array([F(j == i)-pi[i] for j in range(3)])
        tau = 6*(pi*f) @ h @ (f*(h @ f))
        inferred = (tau/(6*pi[i])-g[i, i]**2+pi[i]*(g @ g)[i, i])/(pi[i]*(1-pi[i]))
        assert inferred == square and inferred >= 0
        amplitude2 = np.prod(pi)*inferred
        num, den = isqrt(amplitude2.numerator), isqrt(amplitude2.denominator)
        assert num*num == amplitude2.numerator and den*den == amplitude2.denominator
        skew = F(num, den)*np.diag(1/pi) @ cycle
        candidates = [projection-inverse(projection+g+sign*skew) for sign in (1, -1)]
        reversed_q = np.diag(1/pi) @ q.T @ np.diag(pi)
        assert any(np.all(candidate == q) for candidate in candidates)
        assert any(np.all(candidate == reversed_q) for candidate in candidates)
        assert characteristic_residual(q, f, (pi*f) @ g @ f, tau/6) == [0]*4
        taus.append(tau)
    return pi, g, square, taus


def main():
    rng = np.random.default_rng(20261020)
    chains = []
    for _ in range(32):
        q = np.array([[F(int(x), 10) for x in row] for row in rng.integers(1, 9, (3, 3))])
        np.fill_diagonal(q, F(0)); q -= np.diag(q.sum(axis=1)); chains.append(q)
    projection = np.full((3, 3), F(1, 3)); centered = np.eye(3, dtype=int)-projection
    cycle = np.array([[0, 1, -1], [-1, 0, 1], [1, -1, 0]], dtype=object)
    chains += [np.roll(np.eye(3, dtype=int), 1, axis=1)-np.eye(3, dtype=int)]
    chains += [projection-np.eye(3, dtype=int)+F(1, 10)*np.outer([1, -1, 0], [1, 1, -2])]
    pi = np.array([F(1, 5), F(3, 10), F(1, 2)])
    q = np.diag(1/pi) @ np.array([[0, 1, 2], [1, 0, 3], [2, 3, 0]], dtype=object)
    q -= np.diag(q.sum(axis=1)); chains.append(q)
    for s in (F(1, 4), F(-1, 4), F(1, 2), F(-1, 2)):
        q = (s*cycle-3*centered)/(3+s*s)
        law, gram, square, taus = check(q)
        assert np.all(gram == centered) and square == s*s/3
        assert taus == [F(4, 9)*(1+s*s/3)]*3
        chains.append(q)
        if s > 0:
            print(f"same-Gram cycle s={s}: tau_i={taus[0]} = {float(taus[0]):.10f}; skew-square={square}")
    # Four states: all occupation statistics of the prescribed first state are blind.
    pi4 = np.array([F(1, 4)]*4); projection4 = np.outer(np.ones(4, dtype=int), pi4)
    centered4 = np.eye(4, dtype=int)-projection4
    rotation = np.zeros((4, 4), dtype=object); rotation[1:, 1:] = cycle
    four = []
    for amplitude in (F(0), F(1, 10)):
        h4 = centered4+amplitude*rotation
        q4 = projection4-inverse(projection4+h4)
        assert np.all((h4+h4.T)/2 == centered4)
        assert np.all(q4[0, 1:] == F(1, 4)) and np.all(q4[1:, 0] == F(1, 4))
        f4 = np.array([F(3, 4), F(-1, 4), F(-1, 4), F(-1, 4)])
        assert 6*(pi4*f4) @ h4 @ (f4*(h4 @ f4)) == F(9, 16)
        four.append(q4)
    assert np.all(four[0] == four[0].T) and np.any(four[1] != four[1].T)
    minimum = min(four[1][~np.eye(4, dtype=bool)])
    assert minimum == F(59, 412)
    print(f"four-state obstruction: common tau_1=9/16; minimum off-diagonal rate={minimum}")
    worst, cauchy = 0., 0
    for j, q in enumerate(chains):
        pi, g, square, taus = check(q)
        if j >= 12:
            continue
        qf = np.array(q, dtype=float)
        gap = sorted(abs(eigvals(qf)))[1]
        for i, tau in enumerate(taus):
            f = np.eye(3)[i]-float(pi[i])
            for fraction in (.07, .1):
                radius = fraction*gap/max(abs(f))
                roots = np.exp(2j*np.pi*np.arange(48)/48)
                eigenvalues = [min(eigvals(qf+radius*z*np.diag(f)), key=abs) for z in roots]
                estimate = 6*np.mean(np.array(eigenvalues)/roots**3)/radius**3
                error = abs(estimate-float(tau))/max(1., abs(float(tau)))
                worst = max(worst, error)
                assert error < 1e-7  # Floating spectral cross-check, not an interval proof.
                cauchy += 1
    reversal, reversal_error = 0, 0.
    for n in (2, 3, 4, 5):
        for _ in range(4):
            q = rng.uniform(.1, 1., (n, n)); np.fill_diagonal(q, 0.)
            q -= np.diag(q.sum(axis=1))
            pi = np.linalg.solve(q.T+np.ones((n, n)), np.ones(n))
            rev = q.T*pi[None, :]/pi[:, None]
            diagonal = np.diag(rng.uniform(-.5, .5, n))
            for t in (.01, .2, 1., 4.):
                forward = pi @ expm(t*(q+diagonal)) @ np.ones(n)
                backward = pi @ expm(t*(rev+diagonal)) @ np.ones(n)
                error = abs(forward-backward)/max(1., abs(forward))
                reversal_error = max(reversal_error, error)
                assert error < 8192*np.finfo(float).eps
                reversal += 1
    print(f"{len(chains)} exact rational chains; {3*len(chains)} indicator reconstructions and cubic determinant identities")
    print(f"{cauchy} independent spectral coefficients; maximum scaled error={worst:.3e}")
    print(f"{reversal} finite-horizon reversal-transform comparisons; maximum scaled error={reversal_error:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
