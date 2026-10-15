"""Full fixed-feature Green--Kubo generator identified set; no maturity loading."""
import math
import os
import sys
from fractions import Fraction as F

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm, null_space

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "general"))
from effective_generator import gk, stationary


def basis(pi):
    root = np.sqrt(pi)
    return null_space(root[None, :])/root[:, None]


def ordered(q, pi, u):
    projection = np.outer(np.ones(len(pi)), pi)
    return u.T @ (pi[:, None]*np.linalg.solve(projection-q, u))


def reconstruct(g, k, pi, u):
    return -u @ np.linalg.solve(g+k, u.T*pi[None, :])


def close(actual, expected):
    scale = max(1, np.linalg.norm(expected))
    assert np.linalg.norm(actual-expected) <= 2048*np.finfo(float).eps*scale


def main():
    rng = np.random.default_rng(20261015)
    cases, quadratures = 0, 0
    for n in range(3, 7):
        for trial in range(8):
            q = rng.uniform(0.1, 2, (n, n))
            np.fill_diagonal(q, 0)
            q -= np.diag(q.sum(axis=1))
            q /= np.max(-np.diag(q))
            pi = stationary(q)
            u = basis(pi)
            h = ordered(q, pi, u)
            g, k = (h+h.T)/2, (h-h.T)/2
            close(reconstruct(g, k, pi, u), q)
            close(reconstruct(g, -k, pi, u), q.T*pi[None, :]/pi[:, None])
            assert np.linalg.eigvalsh(g)[0] > 0
            floor = np.min(q[~np.eye(n, dtype=bool)])/4
            for _ in range(6):
                delta = rng.normal(size=(n-1, n-1))
                delta -= delta.T
                delta *= 0.1*np.linalg.eigvalsh(g)[0]/np.linalg.norm(delta)
                candidate = reconstruct(g, k+delta, pi, u)
                while np.min(candidate[~np.eye(n, dtype=bool)]) < floor:
                    delta /= 2
                    candidate = reconstruct(g, k+delta, pi, u)
                close(candidate@np.ones(n), np.zeros(n))
                close(pi@candidate, np.zeros(n))
                independent = gk(candidate, u.T)
                close((independent+independent.T)/2, g)
                close(ordered(candidate, pi, u), g+k+delta)
                features = u @ rng.normal(size=(n-1, 4))
                coeff = u.T @ (pi[:, None]*features)
                gram = gk(candidate, features.T)
                close((gram+gram.T)/2, coeff.T@g@coeff)
                cases += 1
            if trial < 2:
                reduced = -u.T @ (pi[:, None]*q) @ u
                integral, error = quad_vec(lambda t: expm(-t*reduced),
                                           0, np.inf, epsabs=1e-11)
                assert np.linalg.norm(integral-h) <= max(1e-10, 4*error)
                quadratures += 1
    print(f"32 converse/time-reversal checks; {cases} positive same-Gram perturbations")
    print(f"{quadratures} independent semigroup quadratures; projected feature identities pass")

    # A two-state full centered Gram has no skew degree of freedom.
    q = np.array([[-2, 2], [3, -3]], float)
    pi = stationary(q)
    u = basis(pi)
    h = ordered(q, pi, u)
    close(reconstruct(h, np.zeros((1, 1)), pi, u), q)
    print("Two-state reconstruction: no skew parameter")

    # Exact rational three-state family: -Q#=P+(s/3)C for every s in [-1,1].
    p = np.array([[F(int(i == j))-F(1, 3) for j in range(3)] for i in range(3)])
    circulation = np.array([[F(0), F(1), F(-1)], [F(-1), F(0), F(1)], [F(1), F(-1), F(0)]])
    assert np.all(circulation@circulation == -3*p)
    pi = np.full(3, 1/3)
    u = np.array([[math.sqrt(1.5), 1/math.sqrt(2)],
                  [-math.sqrt(1.5), 1/math.sqrt(2)], [0, -math.sqrt(2)]])
    for s in (F(-1), F(-1, 2), F(0), F(1, 4), F(1, 2), F(1)):
        forward, backward = (1+s)/(3+s*s), (1-s)/(3+s*s)
        q = np.array([[-forward-backward, forward, backward],
                      [backward, -forward-backward, forward],
                      [forward, backward, -forward-backward]])
        inverse = p+s*circulation/3
        assert np.all((-q)@inverse == p)
        assert np.all(inverse+inverse.T == 2*p)
        assert forward >= 0 and backward >= 0 and forward+backward > 0
        h = ordered(np.array(q, float), pi, u)
        close((h+h.T)/2, np.eye(2))
        close(h-h.T, 2*float(s)/math.sqrt(3)*np.array([[0, 1], [-1, 0]]))
        print(f"s={s}: forward={forward}, backward={backward}, full centered Gram=I")
    print("Family interval [-1,1]; -Q#=P+(s/3)C; K=(s/sqrt(3))*[[0,1],[-1,0]]")
    print("Local skew dimension (n-1)(n-2)/2; reversal sends K to -K")
    print("PASS")


if __name__ == "__main__":
    main()
