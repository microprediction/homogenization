"""Finite-chain taper theorem, without diagonalizing the generator.

Feynman--Kac CDF refinement, independent mean quadrature and forward moments.
Floating-point regressions supplement, but do not replace, the analytic proof.
"""
from math import comb, factorial, pi, sqrt

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.stats import norm

from verify_tapered_memory import ALPHA, EPS, MU, SIGMA, cdf_function, stationary_variance


def setup(q):
    n = len(q)
    stationary = np.linalg.solve(q.T + np.ones((n, n)), np.ones(n))
    projection = np.outer(np.ones(n), stationary)
    inverse = np.linalg.inv(-q + projection) - projection
    root = np.sqrt(stationary)
    weighted = -q * root[:, None] / root[None, :]
    gap = np.linalg.eigvalsh((weighted + weighted.T) / 2)[1]
    transition = expm(q)  # Choose dimensionless skeleton step h=1.
    minorization = transition.min(axis=0).sum()
    assert np.all(stationary > 0) and gap > 0 and 0 < minorization <= 1
    moments = [1.]
    for k in range(1, 7):
        moments.append((1 + (1 - minorization) * sum(
            comb(k, j) * moments[j] for j in range(1, k))) / minorization)
    residual = transition - transition.min(axis=0)
    assert residual.min() >= -100 * EPS
    assert np.max(abs(residual.sum(axis=1) - (1 - minorization))) < 1000 * EPS
    return stationary, inverse, gap, moments


def main():
    dense = np.array([[-2., 1.7, .3], [.2, -1.3, 1.1], [.8, .4, -1.2]])
    projection = np.ones((3, 3)) / 3
    nilpotent = .1 * np.outer([1., -1., 0.], [1., 1., -2.])
    defective = projection - np.eye(3) + nilpotent
    assert np.linalg.norm(nilpotent @ nilpotent) < 1000 * EPS
    assert np.linalg.norm(projection @ nilpotent) + np.linalg.norm(nilpotent @ projection) < 1000 * EPS
    assert np.linalg.norm(nilpotent) > .1  # Nonzero Jordan part, not a repeated diagonal block.
    cycle = np.roll(np.eye(4), 1, axis=1) - np.eye(4)
    max_refinement, cases = 0., 0
    m2 = MU**2 / (SIGMA**2 * sqrt(2 * pi * np.e))
    for name, q in (("nonreversible", dense), ("defective", defective), ("sparse-cycle", cycle)):
        stationary, inverse, gap, moments = setup(q)
        f = np.arange(len(q), dtype=float)
        f -= stationary @ f
        norm_f = sqrt(stationary @ f**2)
        span = np.ptp(f)
        for degree in range(3):
            power = np.linalg.matrix_power(inverse, degree + 1)
            limit = -MU / SIGMA * norm.pdf(norm.ppf(1 - ALPHA)) * factorial(degree + 1) * power @ f
            errors = []
            for rate in (8, 16, 32, 64, 128):
                cdf, _ = cdf_function(rate, degree, q=q, feature=f)
                refined, _ = cdf_function(rate, degree, 256, 2e-12, q, f)
                threshold = brentq(lambda x: stationary @ refined(x) - (1 - ALPHA),
                                   SIGMA * norm.ppf(1 - ALPHA) + MU * f.min(),
                                   SIGMA * norm.ppf(1 - ALPHA) + MU * f.max())
                direct, _ = quad_vec(lambda t: (degree + 1) * t**degree * expm(rate*q*t) @ f,
                                     0, 1, epsabs=1e-11, epsrel=1e-11)
                tail = sum(rate**k / factorial(k) * np.linalg.matrix_power(inverse, degree + 1 - k)
                           for k in range(degree + 1)) @ f
                mean = factorial(degree + 1) / rate**(degree + 1) * (power @ f - expm(rate*q) @ tail)
                assert np.max(abs(mean - direct)) < 10000 * EPS * (1 + span)
                d2 = span**2 * min(1., moments[2 * degree + 2] / rate**(2 * degree + 2))
                variance_bound = 2 * norm_f**2 * (degree + 1)**2 / ((2 * degree + 1) * gap * rate)
                remainder = m2 * (sqrt(d2 * variance_bound) + d2 / 2)
                lip = abs(MU) / (SIGMA * sqrt(2*pi)) * span * min(1., moments[degree + 1] / rate**(degree + 1))
                for x in (threshold, 0., 1.5):
                    max_refinement = max(max_refinement, np.max(abs(cdf(x) - refined(x))))
                    delta = refined(x) - stationary @ refined(x)
                    leading = -MU / SIGMA * norm.pdf(x / SIGMA) * mean
                    assert np.max(abs(delta)) <= lip + 2e-9
                    assert np.max(abs(delta - leading)) <= remainder + 2e-9
                # Explicit spectral mean-tail bound, componentwise by weighted L2.
                bound = factorial(degree + 1) * norm_f / (gap * rate)**(degree + 1)
                bound *= np.exp(-gap*rate) * sum((gap*rate)**k / factorial(k) for k in range(degree + 1))
                assert np.linalg.norm(np.sqrt(stationary) * (mean - factorial(degree + 1)*power@f/rate**(degree + 1))) <= bound + 1e-11
                variance = stationary_variance(rate, degree, q, f, stationary)
                assert 0 < variance <= variance_bound + 1e-10
                variance_limit = 2 * (stationary * f) @ inverse @ f * (degree + 1)**2 / (2 * degree + 1)
                delta = refined(threshold) - (1 - ALPHA)
                assert abs(stationary @ delta) < 10000 * EPS
                errors.append(np.linalg.norm(delta))
                cases += 1
            order = np.log2(errors[-2] / errors[-1])
            relative = np.linalg.norm(rate**(degree + 1) * delta - limit) / np.linalg.norm(limit)
            assert abs(order - (degree + 1)) < .12 and relative < .1
            assert abs(rate * variance / variance_limit - 1) < .05
            print(f"{name} d={degree}: order={order:.6f}, scaled-gap relative error={relative:.6f}, "
                  f"variance ratio={rate*variance/variance_limit:.6f}")
            print("  predicted scaled gaps:", np.array2string(limit, precision=9))
    assert max_refinement < 2e-9
    print(f"{cases} model cases, {3*cases} CDF points; max refinement={max_refinement:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
