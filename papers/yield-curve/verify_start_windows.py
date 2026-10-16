"""All-start raw-window budgets, transient bias and exact block-range checks."""
from fractions import Fraction as F
from itertools import product

import numpy as np
from scipy.linalg import expm, expm_frechet


def coefficient(p):
    return max(sum(abs(a-b) for a, b in zip(x, y))/2 for x in p for y in p)


def geometric(n, alpha):
    c = 0; h = 0
    for _ in range(n):
        h += (1+c)**2; c = 1+alpha*c
    return c, h


def kernels(q, t):
    n = len(q); derivatives = [expm_frechet(t*q, t*np.diag(f), compute_expm=False) for f in np.eye(n)]
    return [np.column_stack([da @ expm(i*t*q) @ db @ np.ones(n)
                             for da in derivatives for db in derivatives]) for i in (0, 1)]


def main():
    k = np.array([[F(1, 2), F(1, 3), F(1, 6)], [F(1, 6), F(1, 2), F(1, 3)],
                  [F(1, 3), F(1, 6), F(1, 2)]], dtype=object)
    p = k @ k @ k; alpha = coefficient(p); nblocks = 12
    ranges = biases = laws = 0; max_mgf = 0.
    for lag in (1, 2):
        paths = [[(end, start, a if lag == 1 else b, k[start, a]*k[a, b]*k[b, end])
                  for a, b, end in product(range(3), repeat=3)] for start in range(3)]
        reward = np.array([[sum(mass for _, a, b, mass in row if a == x and b == y)
                            for x, y in product(range(3), repeat=2)] for row in paths])
        stationary = np.array([sum(reward[:, j])/3 for j in range(9)])
        value = np.full((3, 9), F(0)); c = F(0)
        for length in range(1, nblocks+1):
            for start, row in enumerate(paths):
                for x, y in product(range(3), repeat=2):
                    j = 3*x+y
                    possible = [int(a == x and b == y)+value[end, j] for end, a, b, _ in row]
                    assert max(possible)-min(possible) <= 1+c; ranges += 1
            value = reward+p @ value; c = 1+alpha*c
            for start in range(3):
                error_squared = sum((value[start, j]/length-stationary[j])**2 for j in range(9))
                assert error_squared <= 2*c*c/(length*length); biases += 1
        _, hn = geometric(nblocks, alpha)
        for x, y in ((0, 1), (2, 0)):
            j = 3*x+y
            for start in range(3):
                distribution = {(start, 0): F(1)}
                for _ in range(nblocks):
                    updated = {}
                    for (state, total), probability in distribution.items():
                        for end, a, b, mass in paths[state]:
                            key = end, total+int(a == x and b == y)
                            updated[key] = updated.get(key, F(0))+probability*mass
                    distribution = updated
                mean = value[start, j]
                assert sum(distribution.values()) == 1
                assert sum(total*mass for (_, total), mass in distribution.items()) == mean
                for tilt in (-2., -.5, .5, 2.):
                    mgf = sum(float(mass)*np.exp(tilt*float(total-mean)-tilt*tilt*float(hn)/8)
                              for (_, total), mass in distribution.items())
                    max_mgf = max(max_mgf, mgf)
                    assert mgf <= 1+64*np.finfo(float).eps
                laws += 1
    print(f"exact three-state block model: alpha={alpha}, N={nblocks}")
    print(f"{ranges} exact conditional ranges; {biases} exact all-start squared-bias inequalities")
    print(f"{laws} exact conditional finite laws; 48 MGF cross-checks; maximum normalized MGF={max_mgf:.6f}")
    rng = np.random.default_rng(20261028); chains = []
    for n in range(2, 7):
        for _ in range(2):
            q = rng.uniform(.1, 1., (n, n)); np.fill_diagonal(q, 0.)
            q -= np.diag(q.sum(1)); chains.append(q)
    for n in (3, 5):
        chains.append(np.roll(np.eye(n), 1, axis=1)-np.eye(n))
    chains.append(np.ones((3, 3))/3-np.eye(3)+.1*np.outer([1, -1, 0], [1, 1, -2]))
    count = 0; worst_bias = worst_identity = 0.
    for q in chains:
        n = len(q); pi = np.linalg.solve(q.T+np.ones_like(q), np.ones(n))
        projector = np.outer(np.ones(n), pi)
        for exposure in (.2, 1.):
            t = exposure/max(-np.diag(q)); p = expm(3*t*q); alpha = coefficient(p)
            deviation = np.linalg.inv(np.eye(n)-p+projector)-projector
            for reward in kernels(q, t):
                assert np.max(abs(reward.sum(1)-t*t)) <= 32768*np.finfo(float).eps*max(1., t*t)
                target = pi @ reward
                for length in (1, 3, 16):
                    average = sum(np.linalg.matrix_power(p, j) @ reward for j in range(length))/length
                    identity = (np.eye(n)-np.linalg.matrix_power(p, length)) @ deviation @ reward/length
                    discrepancy = average-np.outer(np.ones(n), target)-identity
                    error = np.max(abs(discrepancy)); worst_identity = max(worst_identity, error/max(1., t*t))
                    roundoff = 32768*np.finfo(float).eps*max(1., t*t)
                    assert error <= roundoff
                    c, _ = geometric(length, alpha); bound = np.sqrt(2)*t*t*c/length
                    for start in range(n):
                        actual = np.linalg.norm(average[start]-target)
                        assert actual <= bound+roundoff
                        worst_bias = max(worst_bias, actual/bound); count += 1
    print(f"{count} continuous-time starting-state/channel cases on {len(chains)} chains")
    print(f"maximum transient bias/bound={worst_bias:.6f}; scaled Poisson-identity discrepancy={worst_identity:.3e}")
    q = np.array([[-.5, .5], [.5, -.5]]); t = np.log(2.); p = expm(3*t*q)
    reward = kernels(q, t)[0]; target = np.ones(2)/2 @ reward; errors = []
    for length in (1, 2, 4, 8, 16, 32):
        average = sum(np.linalg.matrix_power(p, j) @ reward for j in range(length))/length
        law_bias = average[0].reshape(2, 2).sum(1)/(t*t)-.5
        predicted = 2*(1-8.**(-length))/(7*length*t)*np.array([1., -1.])
        assert np.max(abs(law_bias-predicted)) <= 32768*np.finfo(float).eps
        errors.append(np.linalg.norm(average[0]-target))
    order = np.log2(errors[-2]/errors[-1]); assert .99 < order < 1.01
    print(f"two-state flip rate=1/2, T=log(2): N*law bias -> (+2/(7 log(2)),-2/(7 log(2)))")
    print(f"component magnitude={2/(7*t):.9f}; measured full-moment bias order={order:.6f}")
    print("PASS")


if __name__ == '__main__':
    main()
