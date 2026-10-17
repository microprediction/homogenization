"""CDF misspecification envelopes for exponential-age rank calibration."""
import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.special import betainc
from scipy.stats import binom


def budgets(count, rank, epsilon):
    lower = epsilon + sum(betainc(r + 1, count - r + 1, epsilon) for r in range(rank)) / (count + 1)
    upper = epsilon + sum(betainc(count - r + 1, r + 1, epsilon) for r in range(rank, count + 1)) / (count + 1)
    return lower, upper


# Independently integrate the shifted binomial tails defining both envelopes.
envelopes, error = 0, 0.
for count in (1, 2, 5, 9, 29):
    for rank in range(1, count + 1):
        for epsilon in (.001, .01, .1, .4):
            low, high = budgets(count, rank, epsilon)
            direct_low = epsilon + quad(lambda u: binom.cdf(rank - 1, count, u), 0, epsilon, epsabs=1e-11)[0]
            direct_high = epsilon + quad(lambda u: binom.sf(rank - 1, count, u), 1 - epsilon, 1, epsabs=1e-11)[0]
            error = max(error, abs(low - direct_low), abs(high - direct_high))
            assert epsilon <= min(low, high) and max(low, high) <= 2 * epsilon + 2e-10
            envelopes += 1
assert error <= 2e-9
print(f"Shifted-binomial envelope checks: {envelopes}; maximum quadrature discrepancy={error:.3e}")


def rank_masses(q, beta, count, h, r, delta):
    n = len(q)
    pi = np.linalg.lstsq(np.vstack([q.T, np.ones(n)]), np.r_[np.zeros(n), 1.], rcond=None)[0]
    kernels = [j * beta * np.linalg.inv(j * beta * np.eye(n) - q) for j in range(1, count + 1)]
    # The integrand is a polynomial of degree <= 3*N+2, so this rule integrates it exactly in real arithmetic.
    nodes, weights = leggauss((3 * count + 4) // 2)
    masses = np.zeros(count + 1)
    for s, weight in zip((nodes + 1) / 2, weights / 2):
        cdf = s + h * (s * s - s) / 4 + delta * r * (s ** 3 - s)
        density = 1 + h * (2 * s - 1) / 4 + delta * r * (3 * s * s - 1)
        rows = np.zeros((count + 1, n))
        rows[0] = pi
        for j in range(count):
            updated = rows * (1 - cdf)
            updated[1:] += rows[:-1] * cdf
            rows = updated @ kernels[j] if j < count - 1 else updated
        masses += weight * (rows @ kernels[-1] @ density)
    return masses


models = [np.array([[-5, 1, 4], [1, -5, 4], [4, 4, -8]], dtype=float) / 6,
          np.array([[-6, 2, 1, 3], [1, -4, 2, 1], [2, 1, -4, 1], [3, 1, 1, -5]], dtype=float)]
checks, ratio, zero_error = 0, 0., 0.
for q in models:
    n = len(q)
    h = np.linspace(-1, 1, n)
    r = np.array([1., -2., 1.] if n == 3 else [1., -1., -1., 1.])
    r /= max(abs(r))
    assert max(abs(np.ones(n) @ q)) <= 2e-10  # These examples have uniform invariant law.
    for beta in (.2, 1., 4.):
        for count in (2, 5, 9, 29):
            zero_error = max(zero_error, max(abs(rank_masses(q, beta, count, h, r, 0) - 1 / (count + 1))))
            for delta in (.001, .01, .1):
                assert 1 - max(abs(h)) / 4 - 2 * delta * max(abs(r)) > 0
                epsilon = delta * max(abs(r)) * 2 / (3 * np.sqrt(3))
                coverages = np.cumsum(rank_masses(q, beta, count, h, r, delta))[:-1]
                for rank, coverage in enumerate(coverages, 1):
                    lower, upper = budgets(count, rank, epsilon)
                    gap = coverage - rank / (count + 1)
                    assert -lower - 2e-9 <= gap <= upper + 2e-9
                    ratio = max(ratio, abs(gap) / (2 * epsilon))
                    checks += 1
assert zero_error <= 2e-9
print(f"Chronological multi-contrast coverage budgets: {checks}; N through 29; maximum |error|/(2*epsilon)={ratio:.9f}")
print(f"Zero-perturbation uniform-rank control: maximum error={zero_error:.3e}")

# Exact independent symbolic witness: a second polynomial CDF contrast really breaks exactness.
s = sp.symbols('s')
q = sp.Matrix([[-5, 1, 4], [1, -5, 4], [4, 4, -8]]) / 6
h, r = [-1, 0, 1], [sp.Rational(1, 2), -1, sp.Rational(1, 2)]
f = [s + h[i] * (s ** 2 - s) / 4 + r[i] * (s ** 3 - s) / 10 for i in range(3)]
assert sp.expand(sum(f) / 3 - s) == 0
p1, p2 = (sp.eye(3) - q).inv(), 2 * (2 * sp.eye(3) - q).inv()
coverage = 0
for i in range(3):
    for j in range(3):
        for k in range(3):
            coverage += p1[i, j] * p2[j, k] * sp.integrate(
                sp.expand((1 - f[i] * f[j]) * sp.diff(f[k], s)), (s, 0, 1)) / 3
gap = sp.factor(coverage - sp.Rational(2, 3))
assert gap != 0
direct = sum(rank_masses(models[0], 1., 2, np.array(h), np.array(r, dtype=float), .1)[:2])
assert abs(direct - float(coverage)) <= 2e-9
print(f"Exact N=k=2, beta=1, delta=1/10 witness: coverage={coverage}; bias={gap}")
print("The witness has the same uniform pooled CDF as its one-contrast reference: pooled-CDF closeness alone is insufficient")
low, high = budgets(9, 9, .01)
print(f"N=k=9, epsilon=.01: coverage envelope=[{.9-low:.12f}, {.9+high:.12f}]; simple envelope=[.88,.92]")
print("PASS")
