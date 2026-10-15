"""Reversible realization of a complete fixed-feature Green--Kubo Gram."""
import numpy as np
from fractions import Fraction as F
from scipy.integrate import quad_vec
from scipy.linalg import expm

from verify_generator_identified_set import basis, close, ordered, reconstruct
from effective_generator import stationary


def exact_inverse(matrix):
    """Small rational Gauss--Jordan inverse, independent of floating-point solves."""
    n = len(matrix)
    a = [[F(matrix[i, j]) for j in range(n)] + [F(i == j) for j in range(n)]
         for i in range(n)]
    for j in range(n):
        pivot = next(i for i in range(j, n) if a[i][j])
        a[j], a[pivot] = a[pivot], a[j]
        divisor = a[j][j]
        a[j] = [v/divisor for v in a[j]]
        for i in range(n):
            if i != j:
                multiplier = a[i][j]
                a[i] = [v-multiplier*w for v, w in zip(a[i], a[j])]
    return np.array([row[n:] for row in a], dtype=object)


def main():
    rng = np.random.default_rng(20261016)
    # Dense and sparse irreducible triples, including nonuniform pi and zero rates.
    for trial in range(96):
        q = rng.uniform(0.1, 2, (3, 3))
        if trial % 3 == 0:
            q[0, 2] = q[2, 0] = 0  # a path, necessarily reversible
        elif trial % 3 == 1:
            q[1, 0] = q[2, 1] = q[0, 2] = 0  # directed cycle
        np.fill_diagonal(q, 0)
        q -= np.diag(q.sum(axis=1))
        pi = stationary(q)
        u = basis(pi)
        h = ordered(q, pi, u)
        g, k = (h+h.T)/2, (h-h.T)/2
        rev = reconstruct(g, np.zeros((2, 2)), pi, u)
        additive = (q+q.T*pi[None, :]/pi[:, None])/2
        factor = np.linalg.det(g)/(np.linalg.det(g)+k[0, 1]**2)
        close(additive, factor*rev)
        assert np.min(rev[~np.eye(3, dtype=bool)]) >= -2048*np.finfo(float).eps
        close(ordered(rev, pi, u), g)
    print("96 three-state reversible realizations: dense, path and directed-cycle cases")

    q = np.array([[-4, 0, 3, 1], [0, -1, 1, 0],
                  [1, 0, -4, 3], [3, 1, 0, -4]], dtype=object)
    projection = np.full((4, 4), F(1, 4), dtype=object)
    identity = np.array([[F(i == j) for j in range(4)] for i in range(4)])
    expected = np.array([[-192, -8, 100, 100], [-8, -48, 28, 28],
                         [100, 28, -207, 79], [100, 28, 79, -207]], dtype=object)/F(44)
    for epsilon, forbidden in ((F(0), F(-2, 11)), (F(1, 10), F(-43, 280))):
        candidate = q+epsilon*(projection-identity)
        assert np.all(candidate.sum(axis=0) == 0) and np.all(candidate.sum(axis=1) == 0)
        poisson = exact_inverse(projection-candidate)-projection
        assert np.all((-candidate)@poisson == identity-projection)
        assert np.all(poisson@(-candidate) == identity-projection)
        b = (poisson+poisson.T)/2
        rev = projection-exact_inverse(b+projection)
        assert rev[0, 1] == forbidden
        assert np.all(b@rev == projection-identity)
        if not epsilon:
            assert np.all(rev == expected)
        else:
            assert min(candidate[i, j] for i in range(4) for j in range(4) if i != j) == F(1, 40)
        pi = np.full(4, 1/4)
        u = basis(pi)
        q_float = np.array(candidate, float)
        h = ordered(q_float, pi, u)
        g = (h+h.T)/2
        assert np.linalg.eigvalsh(g)[0] > 0
        close(reconstruct(g, np.zeros((3, 3)), pi, u), np.array(rev, float))
        integral, error = quad_vec(lambda t: u.T@expm(t*q_float)@u/4,
                                   0, 40, epsabs=1e-11)
        # ||exp(t Q)|| on centered L2(pi) <= exp(-gap*t); explicit tail.
        gap = np.linalg.eigvalsh(-u.T@(q_float+q_float.T)@u/8)[0]
        tail = np.exp(-40*gap)/gap
        assert np.linalg.norm(integral-h) <= max(1e-10, 4*error)+np.sqrt(3)*tail
        print(f"epsilon={epsilon}: reversible candidate rate (1,2)={forbidden}; exact arithmetic")
    print("Four-state reversible candidate numerator /44:")
    print(np.array(expected*44, int))
    print("Positive-rate example: minimum rate=1/40; two independent semigroup quadratures")
    print("Reversible realization always exists for n<=3, but need not for n=4")
    print("PASS")


if __name__ == "__main__":
    main()
