"""Critical quadratic score at noise O(1/c); floating checks, not enclosures."""
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ive, ndtr
from scipy.stats import norm
from verify_path_crossover import asymmetric_occupation_density


def pars(c, a, b):
    r, mean, v = a+b, (b-a)/(a+b), 8*a*b/(a+b)**3
    s = math.sqrt(v/c)
    return r, mean, v, s, (1+mean)/s, (1-mean)/s, (a-b)/(2*b*r)


def integrate(fun, lo, hi, points=()):
    if hi <= lo:
        return 0.
    margin = 1e-10*(hi-lo)
    return quad(fun, lo, hi, points=sorted({x for x in points if lo+margin < x < hi-margin}),
                epsabs=3e-10, epsrel=3e-9, limit=300)[0]


def g(c, x, a, b):
    if not -1 < x < 1:
        return 0., 0.
    radius, ab = math.sqrt(1-x*x), math.sqrt(a*b)
    arg, decay = c*ab*radius, c*(a+b+(a-b)*x)/2
    value = math.exp(arg-decay)*ive(0, arg)
    derivative = -c*math.exp(arg-decay)*((a-b)*ive(0, arg)/2+ab*x*ive(1, arg)/radius)
    return value, derivative


def profile(c, u, a, b):
    r, mean, _, s, _, _, _ = pars(c, a, b)
    left, right = g(c, mean-s*u, a, b), g(c, mean+s*u, a, b)
    return a*c/r*(left[0]-right[0]), -a*c*s/r*(left[1]+right[1])


def psi(u):
    return (u-u**3)*norm.pdf(u)


def derivative_psi(u):
    return (1-4*u*u+u**4)*norm.pdf(u)


def budget(c, a, b):
    r, _, _, _, left, right, coefficient = pars(c, a, b)
    top = max(left, right)
    value = integrate(lambda u: abs(profile(c, u, a, b)[1]-coefficient*derivative_psi(u)),
                      0., top, (left, right, .5, 1., 2., 4.))
    tail = quad(lambda u: abs(coefficient*derivative_psi(u)), top, math.inf, epsabs=3e-10)[0]
    return (value+tail+a*c/r*(math.exp(-a*c)+math.exp(-b*c)))/2


def convolve(fun, t, lam, edges=()):
    if not lam:
        return fun(t) if t >= 0 else 0.
    return integrate(lambda z: fun(t-lam*z)*norm.pdf(z), -12., min(12., t/lam),
                      (0., *((t-x)/lam for x in edges)))


def exact_gap(c, t, lam, a, b):
    _, _, _, _, left, right, _ = pars(c, a, b)
    return convolve(lambda x: profile(c, math.sqrt(max(0., x)), a, b)[0],
                    t, lam, (left*left, right*right))/c


def cdf(c, t, lam, a, b, start):
    r, mean, _, s, left, right, _ = pars(c, a, b)
    endpoint = right if start == 0 else left
    atom = math.exp(-(a if start == 0 else b)*c)
    test = (lambda u: ndtr((t-u*u)/lam)) if lam else (lambda u: float(u*u <= t))
    lo, hi = -left, right
    if not lam:
        lo, hi = max(lo, -math.sqrt(max(0., t))), min(hi, math.sqrt(max(0., t)))
    points = [0.]
    for x in (t-8*lam, t, t+8*lam):
        if x > 0:
            points.extend((-math.sqrt(x), math.sqrt(x)))
    density = lambda u: s/2*asymmetric_occupation_density((mean+s*u+1)/2, c, a, b, start)
    return atom*test(endpoint)+integrate(lambda u: density(u)*test(u), lo, hi, points)


def reference(t, lam):
    if not lam:
        return max(0., 2*ndtr(math.sqrt(max(0., t)))-1)
    points = [math.sqrt(x) for x in (t-8*lam, t, t+8*lam) if x > 0]
    return integrate(lambda u: 2*norm.pdf(u)*ndtr((t-u*u)/lam), 0., 12., points)


def main():
    count, worst = 0, 0.
    for a, b in ((1., 1.), (1.7, .4), (.4, 1.7)):
        for c in (2., 8., 32.):
            bound, coefficient = budget(c, a, b), pars(c, a, b)[-1]
            for lam in (0., .25, 1., 2.):
                for t in (.5, 2., 5.):
                    plus, minus = [cdf(c, t, lam, a, b, i) for i in (0, 1)]
                    actual = a/(a+b)*(plus-minus)
                    exact = exact_gap(c, t, lam, a, b)
                    leading = coefficient/c*convolve(lambda x: psi(math.sqrt(max(0., x))), t, lam)
                    assert abs(actual-exact) < 2e-8
                    assert abs(actual-leading) <= bound/c+2e-8
                    worst = max(worst, abs(actual-leading)/(bound/c+2e-8))
                    count += 1
    print(f'{count} independent CDF/profile and finite-budget checks: PASS; max ratio={worst:.6f}')
    a, b = 1.7, .4
    coefficient, pstar = pars(1., a, b)[-1], 2*ndtr(1.)-1
    print(f'Rates a={a}, b={b}; zero-noise sign threshold pstar={pstar:.9f}')
    assert abs(quad(lambda u: 2*u*psi(u), 0., math.inf)[0]+2) < 2e-9
    median_signs = []
    for lam in (0., .5, 2.):
        for p in (.5, .9):
            target = brentq(lambda t: reference(t, lam)-p, -10*(1+lam), 20+10*lam)
            predicted = coefficient*convolve(lambda x: psi(math.sqrt(max(0., x))), target, lam)
            errors = []
            for c in (32., 128., 512., 2048.):
                pooled = lambda t: (b*cdf(c, t, lam, a, b, 0)+a*cdf(c, t, lam, a, b, 1))/(a+b)
                q = brentq(lambda t: pooled(t)-p, -10*(1+lam), 20+10*lam)
                scaled = c*exact_gap(c, q, lam, a, b)
                errors.append(abs(scaled-predicted))
            assert errors[-1] < .015 and errors[-1] < errors[0]/8
            if p == .5:
                median_signs.append(predicted)
            print(f'lambda={lam}, p={p}: c*gap at c=2048 {scaled:.9f}; limit={predicted:.9f}')
    assert median_signs[0] > 0 > median_signs[-1]
    for c in (32., 128., 512., 2048.):
        pooled = lambda t: (b*cdf(c, t, 0., a, b, 0)+a*cdf(c, t, 0., a, b, 1))/(a+b)
        q = brentq(lambda t: pooled(t)-pstar, 0., 10.)
        scaled = c*exact_gap(c, q, 0., a, b)
    assert abs(scaled) < .002
    print(f'At pstar: c*gap={scaled:.9f}; budget at c=2048={budget(2048., a, b):.9f}')
    print('PASS')


if __name__ == '__main__':
    main()
