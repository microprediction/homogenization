"""Start-tapered path scores: exact-transform floating-point regression.

Time-dependent backward Feynman--Kac inversion checks the coupling bounds;
an independent forward moment ODE checks the stationary variance. Gaussian
Fourier-tail control is analytic; ODE/quadrature refinement is not interval proof.
"""
from math import factorial, pi, sqrt

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import gammainc, roots_legendre
from scipy.stats import norm


P, MU, SIGMA, ALPHA = .3, .8, .7, .1
STATIONARY = np.array([1 - P, P])
F = np.array([-P, 1 - P])
EPS = np.finfo(float).eps


def generator(rate):
    return rate * np.array([[-P, P], [1 - P, -(1 - P)]])


def cdf_function(rate, degree, nodes=128, rtol=2e-11):
    """Delta=1; reverse physical time in the backward equation."""
    cutoff = 10 / SIGMA
    x, weights = roots_legendre(nodes)
    frequencies, weights = (x + 1) * cutoff / 2, weights * cutoff / 2
    q = generator(rate)

    def rhs(t, flat):
        v = flat.reshape(2, nodes)
        potential = 1j * MU * (degree + 1) * (1 - t)**degree
        return (q @ v + potential * F[:, None] * frequencies * v).ravel()

    sol = solve_ivp(rhs, (0, 1), np.ones(2 * nodes, complex),
                    method="DOP853", rtol=rtol, atol=rtol / 20)
    assert sol.success
    cf = sol.y[:, -1].reshape(2, nodes)
    if degree == 0:  # Independent constant-potential matrix exponential.
        for j in (0, nodes // 2, nodes - 1):
            exact = expm(q + 1j * MU * frequencies[j] * np.diag(F)) @ np.ones(2)
            assert np.max(abs(cf[:, j] - exact)) < 300 * rtol
    kernel = weights * np.exp(-SIGMA**2 * frequencies**2 / 2) / (pi * frequencies)

    def cdf(value):
        return .5 - np.imag(cf * np.exp(-1j * frequencies * value)) @ kernel

    return cdf, np.exp(-50) / (100 * pi)


def stationary_variance(rate, degree):
    q = generator(rate)

    def rhs(t, moments):
        m1, m2 = moments.reshape(2, 2)
        w = (degree + 1) * t**degree
        return np.concatenate((q.T @ m1 + w * F * STATIONARY,
                               q.T @ m2 + 2 * w * F * m1))

    sol = solve_ivp(rhs, (0, 1), np.zeros(4), method="DOP853",
                    rtol=2e-12, atol=1e-14)
    assert sol.success
    return sum(sol.y[2:, -1])


def main():
    z = norm.ppf(1 - ALPHA)
    m2 = MU**2 / (SIGMA**2 * sqrt(2 * pi * np.e))
    leading_limit = -MU / SIGMA * norm.pdf(z) * (1 - P)
    max_refinement = 0.
    print("Delta=1 p=.3 mu=.8 sigma=.7 alpha=.1; start=1")
    print("degree rate quantile gap gap/L rate*variance variance-factor")
    for degree in range(3):
        gaps = []
        factor = (degree + 1)**2 / (2 * degree + 1)
        for rate in (8, 16, 32, 64, 128):
            cdf, tail = cdf_function(rate, degree)
            refined, _ = cdf_function(rate, degree, 256, 2e-12)
            threshold = brentq(lambda x: STATIONARY @ refined(x) - (1 - ALPHA),
                               SIGMA * z + MU * F[0], SIGMA * z + MU * F[1])
            points = (threshold, -1.5, 0., 1., 2.5)
            refinement = max(np.max(abs(cdf(x) - refined(x))) for x in points)
            max_refinement = max(max_refinement, refinement)
            # An explicit numerical tolerance, not a rigorous ODE error bound.
            numerical = 2e-9 + 100 * EPS * (1 + rate)
            assert refinement < numerical
            laplace = factorial(degree + 1) * gammainc(degree + 1, rate) / rate**(degree + 1)
            independent = quad(lambda t: (degree + 1) * t**degree * np.exp(-rate*t), 0, 1)[0]
            assert abs(laplace - independent) < 1000 * EPS
            variance_bound = 2 * P * (1 - P) * factor / rate
            d2 = abs(F) * factorial(2 * degree + 2) / rate**(2 * degree + 2)
            remainder = m2 * (np.sqrt(d2 * variance_bound) + d2 / 2)
            for x in points:
                delta = refined(x) - STATIONARY @ refined(x)
                leading = -MU / SIGMA * norm.pdf(x / SIGMA) * F * laplace
                bound = MU / (SIGMA * sqrt(2 * pi)) * abs(F) * laplace
                assert np.all(abs(delta) <= bound + numerical)
                assert np.all(abs(delta - leading) <= remainder + numerical)
            gap = refined(threshold) - (1 - ALPHA)
            assert abs(STATIONARY @ gap) < 10000 * EPS
            variance = stationary_variance(rate, degree)
            assert 0 < variance <= variance_bound + numerical
            print(f"{degree} {rate:3d} {threshold:.9f} {gap[1]:.10e} "
                  f"{gap[1]/laplace:.9f} {rate*variance:.9f} {factor:.9f}")
            gaps.append(abs(gap[1]))
        order = np.log2(gaps[-2] / gaps[-1])
        assert abs(order - (degree + 1)) < .12
        assert abs(gap[1] / laplace / leading_limit - 1) < .04
        assert .94 < variance / variance_bound < 1
        print(f"degree={degree} final observed gap order={order:.6f}")
    print(f"limiting gap/L={leading_limit:.9f}; maximum CDF refinement={max_refinement:.3e}")
    print(f"analytic Fourier-tail bound={tail:.3e}; 15 cases, 75 CDF points")
    print("PASS")


if __name__ == "__main__":
    main()
