"""Two-feature affinity frontier; run from any working directory. No Monte Carlo."""
from fractions import Fraction as R
from itertools import combinations, permutations
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm
from effective_generator import gk, stationary


def close(a, b):
    assert np.linalg.norm(np.asarray(a) - b) <= 3e-8 * (1 + np.linalg.norm(b))


def chi(K):
    G = (K + K.T) / 2
    return abs(K[0, 1] - K[1, 0]) / (2 * np.sqrt(np.linalg.det(G)))


def frontier(n, affinity):
    return np.tanh(affinity / (2 * n)) / np.tan(np.pi / n)


def cycle(n, eta, pi):
    theta = 2 * np.pi / n
    total = 2 / (n * (1 - np.cos(theta)))
    Q = np.zeros((n, n))
    for i in range(n):
        Q[i, (i + 1) % n] = total * (1 + eta) / (2 * pi[i])
        Q[i, (i - 1) % n] = total * (1 - eta) / (2 * pi[i])
    Q -= np.diag(Q.sum(axis=1))
    U = np.column_stack((np.cos(theta * np.arange(n)), np.sin(theta * np.arange(n))))
    return Q, U


def affinities(Q):
    """Enumerate all oriented simple cycles, minimum vertex first."""
    result = []
    for length in range(3, len(Q) + 1):
        for vertices in combinations(range(len(Q)), length):
            for rest in permutations(vertices[1:]):
                c = (vertices[0],) + rest
                edges = list(zip(c, c[1:] + c[:1]))
                if all(Q[i, j] > 0 for i, j in edges):
                    result.append((length, abs(sum(np.log(Q[i, j] / Q[j, i]) for i, j in edges))))
    return result


def independent(Q, F, pi):
    P = np.outer(np.ones(len(pi)), pi)
    Fc = F - pi @ F
    return Fc.T @ (pi[:, None] * np.linalg.solve(-Q + P, Fc))


rng = np.random.default_rng(20261019)
count = 0
for n in range(3, 13):
    for eta in (0., .1, -.35, .7, -.95):
        for pi in (np.ones(n) / n, np.arange(1., n + 1) / sum(range(1, n + 1))):
            Q, U = cycle(n, eta, pi)
            C = rng.normal(size=(2, 2)) + 2 * np.eye(2)
            F = -Q @ U @ C
            target = C.T @ np.array([[1, eta / np.tan(np.pi / n)],
                                     [-eta / np.tan(np.pi / n), 1]]) @ C
            close(gk(Q, F.T), target)
            close(independent(Q, F, pi), target)
            close(pi @ Q, np.zeros(n))
            omega = 2 * n * np.arctanh(abs(eta))
            close(chi(target), frontier(n, omega))
            if n <= 6:
                close(max(a for _, a in affinities(Q)), omega)
            count += 1
print(f"cycle attainment: {count} cases, N=3..12, both orientations, two stationary laws")

count = 0
for n in range(3, 7):
    for _ in range(12):
        Q = rng.lognormal(0, 1.2, (n, n))
        np.fill_diagonal(Q, 0)
        Q -= np.diag(Q.sum(axis=1))
        F = rng.normal(size=(n, 2))
        K = gk(Q, F.T)
        close(K, independent(Q, F, stationary(Q)))
        cycles = affinities(Q)
        tight = max(frontier(length, a) for length, a in cycles)
        loose = frontier(n, max(a for _, a in cycles))
        assert chi(K) <= tight + 3e-8 and tight <= loose + 3e-8
        count += 1
print(f"all-simple-cycle necessity: {count} random bidirected chains, N=3..6")

for omega in (.1, 1., 4 * np.log(3), 20.):
    assert np.all(np.diff([frontier(n, omega) for n in range(3, 41)]) > 0)
for x in (.25, .5, 1., 2.):
    coefficient = 2 * np.pi**3 * (x + x**3) / 3
    errors = []
    for n in (32, 64, 128, 256):
        omega = 2 * n * np.arctanh(x * np.tan(np.pi / n))
        assert omega > 2 * np.pi * x
        errors.append(abs(n*n * (omega - 2*np.pi*x) / coefficient - 1))
    assert all(b < a / 3 for a, b in zip(errors, errors[1:])) and errors[-1] < .001
print("strict length monotonicity and fixed-chi N^-2 coefficient: four grids each")

# Exact rational fixture; integer potentials avoid rounded trigonometric zeros.
Q = np.array([[-4, 3, 0, 1], [1, -4, 3, 0], [0, 1, -4, 3], [3, 0, 1, -4]], dtype=object)
U = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=object)
F = -Q @ U
target = np.array([[2, 1], [-1, 2]], dtype=object)
assert np.array_equal(F.T @ U * R(1, 4), target)
print(f"N=4; pi=(1/4,...); rates forward=3 reverse=1; U={U.tolist()}; F={F.tolist()}")
print(f"exact K={target.tolist()}; chi=1/2; Omega/log(3)=4; sigma/log(3)=2")
close(8 * np.arctanh(.5), 4 * np.log(3))
assert frontier(3, 4 * np.log(3)) < .5
for k in range(5):
    scale = R(1, 9**k)
    Qs, Fs, Us = Q * scale, F * R(1, 3**k), U * 3**k
    assert np.array_equal(-Qs @ Us, Fs)
    assert np.array_equal(Fs.T @ Us * R(1, 4), target)
    assert all(sum(Qs[:, j]) == 0 for j in range(4))
    assert all(sum(Fs[:, j]) == 0 for j in range(2))
    # Every edge has flux ratio 3, net flux (1/2)*scale.
    sigma_coefficient = sum((Qs[i, (i+1) % 4] - Qs[(i+1) % 4, i]) * R(1, 4) for i in range(4))
    assert sigma_coefficient == 2 * scale
    close(gk(np.asarray(Qs, float), np.asarray(Fs.T, float)), np.asarray(target, float))
    print(f"s={scale}: exact K unchanged, Omega/log(3)=4, sigma/log(3)={sigma_coefficient}")
Qf, Ff = np.asarray(Q, float), np.asarray(F, float)
integral, error = quad_vec(lambda t: Ff.T @ expm(t * Qf) @ Ff / 4, 0, 10)
# Normal uniform cycle: centered semigroup norm <= exp(-4t).
tail = np.linalg.norm(Ff, 2)**2 * np.exp(-40) / 16
assert np.linalg.norm(integral - np.asarray(target, float)) <= tail + error + 3e-8
print(f"independent correlation quadrature: error={error:.3g}, analytic tail bound={tail:.3g}")
print("PASS")
