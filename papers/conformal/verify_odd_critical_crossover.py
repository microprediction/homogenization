"""Odd-power telegraph population-coverage crossover; deterministic quadrature checks.

Analytic bounds, not this finite grid, prove uniformity over every noise level.
Gaussian integrations truncate at 12; polynomial-weighted omitted tails are
below the 3e-8 quadrature allowance for degrees 3, 5, 7 used here.
"""
from math import exp, factorial, gamma, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import gammaincc, ive, ndtr, ndtri
from scipy.stats import beta, poisson


def phi(x):
    return exp(-x*x/2)/sqrt(2*pi)


def root(t, d):
    return np.copysign(abs(t)**(1/d), t)


def integrate(f, lo, hi, point=0):
    points = sorted(set([lo, hi] + [v for v in (0, point) if lo < v < hi]))
    return sum(quad(f, a, b, epsabs=2e-10, epsrel=2e-10, limit=200)[0]
               for a, b in zip(points, points[1:]))


def contrast(x, c):
    if not -sqrt(c) <= x < sqrt(c):
        return 0.0
    z = c*sqrt(max(0, 1-x*x/c))
    return -sqrt(c)/2*exp(z-c)*ive(0, z)


def density(x, c):
    s = sqrt(max(0, 1-x*x/c))
    z = c*s
    return sqrt(c)/2*exp(z-c)*(ive(0, z) + (ive(1, z)/s if s else c/2))


def cdf(u, lam, d, c=None):
    a, b = 1/(1+lam), lam/(1+lam)
    bound = sqrt(c) if c else 12
    dens = (lambda x: density(x, c)) if c else phi
    x = root(u/a, d)
    if not lam:
        if c and abs(x) >= bound:
            return exp(-c)/2 if x == -bound else float(x >= bound)
        return .5 + np.sign(x)*integrate(dens, 0, abs(x))
    value = integrate(lambda x: dens(x)*ndtr((u-a*x**d)/b), -bound, bound, x)
    if c:
        value += exp(-c)/2*sum(ndtr((u-a*x**d)/b) for x in (-bound, bound))
    return value


def quantile(p, lam, d, c=None):
    if p == .5:
        return 0.0
    upper = 2.0
    while cdf(upper, lam, d, c) < p:
        upper *= 2
    return brentq(lambda u: cdf(u, lam, d, c)-p, -upper, upper, xtol=2e-10)


def profile(u, lam, d, c=None):
    """(1+lambda) times the scaled exact contrast or its limiting profile."""
    base = (lambda x: contrast(x, c)) if c else (lambda x: -phi(x)/2)
    if not lam:
        return base(root(u, d))
    if lam <= 1:
        t = (1+lam)*u
        return (1+lam)*integrate(lambda z: base(root(t-lam*z, d))*phi(z),
                                 -12, 12, t/lam)
    a, b = 1/(1+lam), lam/(1+lam)
    bound = sqrt(c) if c else 12
    return integrate(lambda x: base(x)*d*x**(d-1)*phi((u-a*x**d)/b)/b,
                     -bound, bound, root(u/a, d))


def moment(n):
    return 2**(n/2)*gamma((n+1)/2)/sqrt(pi)


def budgets(c, d):
    uniform = 3/c+(sqrt(c)+1)*exp(-c/4)
    weighted = d/(2*c)*sum(k*moment(n) for k, n in
                          ((6, d-1), (1, d+1), (1, d+3)))
    weighted += d/2*(sqrt(2*pi*c)+1)*moment(d-1)*gammaincc(d/2, c/4)
    return uniform, weighted


def main():
    errors = []
    for c in (2, 8, 32):
        for a in (-1, -.8, -.2, 0, .4, .9):
            u = (a+1)/2
            value = -exp(-c)/2
            for j in range(1, int(c+12*sqrt(c)+40)//2):
                value += poisson.pmf(2*j, c)/2*(beta.cdf(u, j+1, j)-beta.cdf(u, j, j+1))
            errors.append(abs(value-contrast(sqrt(c)*a, c)/sqrt(c)))
    assert max(errors) < 3e-8
    z = np.geomspace(1, 10000, 400)
    assert np.all(np.abs(np.sqrt(2*pi*z)*ive(0, z)-1) <= 2/z)
    print(f"Poisson--Beta/Bessel contrast max error: {max(errors):.3g}")
    for c in (2, 16, 64, 256, 1024):
        assert abs(integrate(lambda x: density(x, c), -sqrt(c), sqrt(c))+exp(-c)-1) < 3e-8
    for c in (2, 16, 64, 256):
        for d in (3, 5, 7):
            uniform, weighted = budgets(c, d)
            error = lambda x: abs(contrast(x, c)+phi(x)/2)
            assert max(error(x) for x in np.linspace(-12, 12, 2001)) <= uniform+3e-8
            measured = integrate(lambda x: d*x**(d-1)*error(x), -12, 12, sqrt(c))
            assert measured <= weighted+3e-8
    for d in (3, 5, 7):
        row = []
        for c in (16, 64, 256, 1024):
            maximum = 0.0
            for p in (.5, .9):
                for lam in (0, .02, c**-.5, .5, 2, sqrt(c), .5*c**(d/2), c**(d/2+1)):
                    u, v = quantile(p, lam, d, c), quantile(p, lam, d)
                    assert max(abs(cdf(u, lam, d, c)-p), abs(cdf(v, lam, d)-p)) < 3e-8
                    exact, actual, reference = profile(u, lam, d, c), profile(u, lam, d), profile(v, lam, d)
                    uniform, weighted = budgets(c, d)
                    bound = (1+lam)*min(uniform, weighted/(lam*sqrt(2*pi)) if lam else np.inf)
                    assert exact < 0 and abs(exact-actual) <= bound+3e-8
                    maximum = max(maximum, abs(exact-reference))
                    if not lam:
                        assert abs(root(u, d)-quantile(p, 0, 1, c)) < 3e-8
            row.append(maximum)
        assert row[-1] < row[0]/10
        print(f"degree {d}: max weighted quantile errors at c=16,64,256,1024: "
              + ", ".join(f"{v:.7g}" for v in row))
        p, c, lam = .9, 1024, 1024**(d/2+1)
        value = profile(quantile(p, lam, d, c), lam, d, c)
        limit = -factorial(d)/(2**((d-1)//2)*factorial((d-1)//2))*phi(ndtri(p))/2
        assert abs(value-limit) < .03*abs(limit)
        print(f"degree {d}: large-noise weighted contrast {value:.10g}; limit {limit:.10g}")
    print("PASS")


if __name__ == "__main__":
    main()
