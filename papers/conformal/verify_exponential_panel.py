"""Exact rank coverage from exponential-age calibration under one CDF contrast."""
import numpy as np
import sympy as sp
from scipy.integrate import quad_vec
from scipy.stats import norm


def stationary(q):
    pi = q.T.nullspace()[0]
    return pi.T / sum(pi)


def resolvents(q, beta, count):
    return [j * beta * (j * beta * sp.eye(q.rows) - q).inv() for j in range(1, count + 1)]


# Exact polynomial rank-count recursion, with independent chronological gaps Exp(j*beta).
score = sp.symbols('s', real=True)
models = [sp.Matrix([[-3, 3], [1, -1]]),
          sp.Matrix([[-1, 1, 0], [0, -1, 1], [4, 0, -4]]),
          sp.Matrix([[-6, 2, 1, 3], [1, -4, 2, 1], [2, 1, -4, 1], [3, 1, 1, -5]])]
signals = [sp.Matrix([1, -1]), sp.Matrix([1, -1, 2]), sp.Matrix([1, -1, 0, 2])]
exact_checks = 0
for q, h in zip(models, signals):
    pi = stationary(q)
    h = h - sp.ones(q.rows, 1) * (pi * h)[0]
    cdf = [score + hi * (score ** 2 - score) / 4 for hi in h]
    density = sp.Matrix([sp.diff(fi, score) for fi in cdf])
    assert max(abs(hi) / 4 for hi in h) < 1
    for beta in (sp.Rational(1, 2), sp.Integer(2)):
        for count in range(1, 6):
            kernels = resolvents(q, beta, count)
            rows = [pi] + [sp.zeros(1, q.rows) for _ in range(count)]
            for observation in range(count):
                updated = []
                for below in range(count + 1):
                    row = [rows[below][i] * (1 - cdf[i]) + (
                        rows[below - 1][i] * cdf[i] if below else 0) for i in range(q.rows)]
                    value = sp.Matrix([row])
                    if observation < count - 1:
                        value *= kernels[observation]
                    updated.append(value.applyfunc(sp.expand))
                rows = updated
            for row in rows:
                integrand = sp.expand((row * kernels[-1] * density)[0])
                assert sp.integrate(integrand, (score, 0, 1)) == sp.Rational(1, count + 1)
                exact_checks += 1
print(f"Exact polynomial rank-mass checks: {exact_checks}; all ranks uniform through N=5 on 2-,3-,4-state chains")


def numerical_rank_masses(q, beta, count):
    n = len(q)
    pi = np.linalg.lstsq(np.vstack([q.T, np.ones(n)]), np.r_[np.zeros(n), 1.], rcond=None)[0]
    kernels = [j * beta * np.linalg.inv(j * beta * np.eye(n) - q) for j in range(1, count + 1)]
    def integrand(t):
        # Arbitrary unequal Gaussian laws, not the polynomial-tilt test family.
        cdf = norm.cdf((t - np.array([.8, -.6])) / np.array([.35, 1.1]))
        density = norm.pdf((t - np.array([.8, -.6])) / np.array([.35, 1.1])) / np.array([.35, 1.1])
        rows = np.zeros((count + 1, n))
        rows[0] = pi
        for observation in range(count):
            updated = rows * (1 - cdf)
            updated[1:] += rows[:-1] * cdf
            rows = updated @ kernels[observation] if observation < count - 1 else updated
        return rows @ kernels[-1] @ density
    return quad_vec(integrand, -np.inf, np.inf, epsabs=1e-10, epsrel=1e-10)[0]


checks, maximum = 0, 0.
q = np.array(models[0], dtype=float)
for count in (2, 5, 9):
    for beta in (.2, 1., 4.):
        for speed in (.1, 1., 8.):
            masses = numerical_rank_masses(speed * q, beta, count)
            error = max(abs(masses - 1 / (count + 1)))
            assert error <= 2e-9
            maximum = max(maximum, error)
            checks += count + 1
print(f"Independent unequal-Gaussian rank-mass comparisons: {checks}; maximum error={maximum:.3e}")
harmonic = sum(sp.Rational(1, j) for j in range(1, 10))
print(f"N=9, beta=1: expected span H9={harmonic}={float(harmonic):.12f}; rank-9 marginal coverage=9/10")
archive_bound = -np.expm1(9 * np.log1p(-np.exp(-10.)))
assert 0 <= archive_bound <= 9 * np.exp(-10.) + 2e-14
print(f"Truncated ages: N=9, beta=1, L=10; marginal rank-error bound={archive_bound:.12e}")


def separated_rank(q, beta):
    # State i emits Uniform[2*i,2*i+1]; rational enumeration of max of two calibrations.
    pi, joint = stationary(q), [sp.Integer(0)] * q.rows
    first, second = resolvents(q, beta, 2)
    for i in range(q.rows):
        for j in range(q.rows):
            for k in range(q.rows):
                if k > max(i, j):
                    rank = 0
                elif k < max(i, j):
                    rank = 1
                else:
                    equal = int(i == k) + int(j == k)
                    rank = sp.Rational(equal, equal + 1)
                joint[k] += pi[i] * first[i, j] * second[j, k] * rank
    return sum(joint), [joint[k] / pi[k] for k in range(q.rows)]


coverage, conditional = separated_rank(models[0], sp.Integer(1))
assert coverage == sp.Rational(2, 3)
assert conditional == [sp.Rational(14, 15), sp.Rational(26, 45)]
print('Not current-state conditional: binary separated uniforms give coverages 14/15 and 26/45; marginal=2/3')
q = sp.Matrix([[-5, 1, 4], [1, -5, 4], [4, 4, -8]]) / 6
coverage, _ = separated_rank(q, sp.Rational(1, 2))
assert coverage == sp.Rational(1621, 2430)
assert coverage - sp.Rational(2, 3) == sp.Rational(1, 2430)
print('Beyond one CDF contrast: three-state separated uniforms give coverage=1621/2430, bias=1/2430')

# N=2 operator identity, valid on every finite generator, not merely one decay mode.
for q in models:
    for beta in (sp.Rational(1, 2), sp.Integer(2)):
        first, second = resolvents(q, beta, 2)
        assert 2 * first == second + first * second
print('Exact two-gap resolvent cancellation checks: 6; schedules are averaged, not conditioned on')
print("PASS")
