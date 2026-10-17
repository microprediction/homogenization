"""Reset-chain Dirichlet panels, exact ranks, and an approximate-reset budget."""
import math
from collections import defaultdict
from itertools import product

import numpy as np
import sympy as sp
from scipy.integrate import quad_vec
from scipy.stats import norm


def dirichlet_moment(alpha, powers):
    result = sp.Integer(1)
    for value, power in zip(alpha, powers):
        result *= sp.rf(value, power)
    return result / sp.rf(sum(alpha), sum(powers))


exact = 0
for n in (3, 4):
    pi = sp.Matrix([[sp.Rational(i, n * (n + 1) // 2) for i in range(1, n + 1)]])
    projection = sp.ones(n, 1) * pi
    for clock in (sp.Rational(1, 2), sp.Integer(3)):
        q = clock * (projection - sp.eye(n))
        for beta in (sp.Rational(1, 2), sp.Integer(2)):
            alpha = [clock * value / beta for value in pi]
            for count in range(1, 5):
                kernels = [j * beta * (j * beta * sp.eye(n) - q).inv() for j in range(1, count + 1)]
                joint = defaultdict(lambda: sp.Integer(0))
                for states in product(range(n), repeat=count + 1):
                    weight = pi[states[0]]
                    for j in range(count):
                        weight *= kernels[j][states[j], states[j + 1]]
                    counts = tuple(states[:-1].count(i) for i in range(n))
                    joint[counts, states[-1]] += weight
                for (counts, terminal), weight in joint.items():
                    powers = list(counts)
                    powers[terminal] += 1
                    multinomial = sp.Rational(math.factorial(count), math.prod(math.factorial(r) for r in counts))
                    assert weight == multinomial * dirichlet_moment(alpha, powers)
                    exact += 1
print(f"Exact chronological count/test versus Dirichlet-mixture identities: {exact}; N=1..4, nonuniform 3-/4-state reset chains")


def numerical_masses(q, pi, beta, count):
    n = len(pi)
    kernels = [j * beta * np.linalg.inv(j * beta * np.eye(n) - q) for j in range(1, count + 1)]
    means, scales = np.linspace(-1.2, 1.3, n), np.linspace(.35, 1.1, n)
    def integrand(t):
        cdf, density = norm.cdf((t - means) / scales), norm.pdf((t - means) / scales) / scales
        rows = np.zeros((count + 1, n))
        rows[0] = pi
        for j in range(count):
            updated = rows * (1 - cdf)
            updated[1:] += rows[:-1] * cdf
            rows = updated @ kernels[j] if j < count - 1 else updated
        return rows @ kernels[-1] @ density
    return quad_vec(integrand, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-10)[0]


checks, maximum = 0, 0.
for n in (3, 4):
    pi = np.arange(1, n + 1, dtype=float)
    pi /= pi.sum()
    for clock in (.1, 1., 8.):
        q = clock * (np.ones((n, 1)) @ pi[None, :] - np.eye(n))
        for beta in (.2, 1., 4.):
            for count in (2, 5, 9):
                values = numerical_masses(q, pi, beta, count)
                error = max(abs(values - 1 / (count + 1)))
                assert error <= 2e-9
                maximum = max(maximum, error)
                checks += count + 1
print(f"Arbitrary multi-contrast Gaussian rank-mass checks: {checks}; maximum error={maximum:.3e}")

# Actual non-reset generators retain the same stationary law, but need not be reversible.
bounds, ratio = 0, 0.
for n in (3, 4):
    pi = np.arange(1, n + 1, dtype=float)
    pi /= pi.sum()
    reset = 2 * (np.ones((n, 1)) @ pi[None, :] - np.eye(n))
    cycle = np.zeros((n, n))
    for i in range(n):
        cycle[i, (i + 1) % n] = 1 / pi[i]
        cycle[i, i] = -1 / pi[i]
    for epsilon in (.01, .1, .5):
        q = reset + epsilon * cycle
        assert max(abs(pi @ q)) <= 2e-10
        delta = np.linalg.norm(q - reset, np.inf) / 2
        for count in (2, 5):
            bound = 1 - math.prod(j / (j + delta) for j in range(1, count + 1))
            values = numerical_masses(q, pi, 1., count)
            error = max(abs(np.cumsum(values)[:-1] - np.arange(1, count + 1) / (count + 1)))
            assert error <= bound + 2e-9
            ratio = max(ratio, error / bound)
            bounds += 1
print(f"Approximate-reset rank-error budgets: {bounds}; maximum error/bound={ratio:.9f}")

# A reversible chain need not be a reset chain: three separated continuous score laws.
q = sp.Matrix([[-5, 1, 4], [1, -5, 4], [4, 4, -8]]) / 6
first, second = (sp.eye(3) / 2 - q).inv() / 2, (sp.eye(3) - q).inv()
coverage = 0
for i, j, k in product(range(3), repeat=3):
    equal = int(i == k) + int(j == k)
    rank = 0 if k > max(i, j) else 1 if k < max(i, j) else sp.Rational(equal, equal + 1)
    coverage += first[i, j] * second[j, k] * rank / 3
assert coverage == sp.Rational(1621, 2430)
print('Non-reset reversible counterexample: beta=1/2, N=k=2, separated uniforms; coverage=1621/2430=2/3+1/2430')
print("PASS")
