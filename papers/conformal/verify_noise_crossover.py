"""Exact telegraph noise crossover; quadrature checks, not interval enclosures."""
import math
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ive, ndtr
from scipy.stats import norm
from verify_path_crossover import conditional_values


def kernels(c, u):
    if abs(u) > math.sqrt(c) or u == math.sqrt(c):
        return 0., 0., 0., 0.
    r = math.sqrt(max(0., 1-u*u/c))
    x = c*r
    b0 = math.exp(x-c)*ive(0, x)
    b1r = math.exp(x-c)*ive(1, x)/r if r else c*math.exp(-c)/2
    k = math.sqrt(c)*b0
    return k, math.sqrt(c)*(b0+b1r)/2, -math.sqrt(c)*u*b1r, u*b1r/2


def integral(fun, lo, hi, points=()):
    if hi <= lo:
        return 0.
    margin = 1e-10*(hi-lo)
    return quad(fun, lo, hi, points=[p for p in points if lo+margin < p < hi-margin],
                epsabs=2e-11, epsrel=2e-10, limit=300)[0]


def cdf(c, t, lam, sign=0):
    root = math.sqrt(c)
    test = (lambda u: ndtr((t-u)/lam)) if lam else (lambda u: float(u <= t))
    atom = math.exp(-c)*((1+sign)*test(root)+(1-sign)*test(-root))/2
    return atom+integral(lambda u: (kernels(c, u)[1]+sign*kernels(c, u)[3])*test(u),
                         -root, root, (0., t))


def gap(c, t, lam):
    if not lam:
        return -kernels(c, t)[0]/(2*math.sqrt(c))
    root = math.sqrt(c)
    if lam < .5:
        value = integral(lambda z: kernels(c, t-lam*z)[0]*norm.pdf(z),
                         max(-12., (t-root)/lam), min(12., (t+root)/lam), (0.,))
    else:
        value = integral(lambda u: kernels(c, u)[0]*norm.pdf((t-u)/lam)/lam,
                         -root, root, (0., t))
    return -value/(2*root)


def budgets(c):
    root = math.sqrt(c)
    tail = 2*norm.sf(root)
    l1 = 2*integral(lambda u: abs(kernels(c, u)[0]-norm.pdf(u)), 0., root)+tail
    derivative = integral(lambda u: abs(kernels(c, u)[2]+u*norm.pdf(u)), 0., root)
    derivative += root*math.exp(-c)+norm.pdf(root)
    eta = (2*integral(lambda u: abs(kernels(c, u)[1]-norm.pdf(u)), 0., root)
           +tail+math.exp(-c))/2
    epsilon = math.sqrt(2)*max(derivative, l1/math.sqrt(2*math.pi))
    return eta, epsilon


def main():
    count, quantile_count, worst = 0, 0, 0.
    for c in (.5, 2., 8.):
        for q in (-.4, 0., .6):
            direct = conditional_values(c, lambda a: ndtr((q-a)/.5))
            for sign, target in zip((1, -1), direct):
                value = cdf(c, q*math.sqrt(c), .5*math.sqrt(c), sign)
                assert abs(value-target) < 2e-9
    print('18 independent Poisson--Beta CDF comparisons: PASS')
    for c in (.5, 2., 8.):
        for t in (-math.sqrt(c), math.sqrt(c)):
            assert abs(cdf(c, t, 0., 1)-cdf(c, t, 0.)-gap(c, t, 0.)) < 2e-9
    print('Six zero-noise endpoint-atom identities: PASS')
    for c in (2., 8., 32., 128.):
        root = math.sqrt(c)
        mass = integral(lambda u: kernels(c, u)[0], -root, root, (0.,))
        assert abs(mass+math.expm1(-2*c)) < 2e-9
        eta, epsilon = budgets(c)
        for lam in (0., .01, .25, 1., 4., 32.):
            for p in (.5, .9):
                t = brentq(lambda x: cdf(c, x, lam)-p,
                           -root-8*lam, root+8*lam, xtol=2e-11)
                actual = cdf(c, t, lam, 1)-p
                exact = gap(c, t, lam)
                assert abs(actual-exact) < 3e-9
                scale = 1/(2*root*math.sqrt(1+lam*lam))
                error = abs(exact+scale*norm.pdf(t/math.sqrt(1+lam*lam)))
                assert error <= scale*epsilon+3e-9
                if eta < min(p, 1-p):
                    quantile_count += 1
                    z = norm.ppf(p)
                    dz = max(z-norm.ppf(p-eta), norm.ppf(p+eta)-z)
                    budget = scale*(epsilon+dz/math.sqrt(2*math.pi*math.e))
                    assert abs(actual+scale*norm.pdf(z)) <= budget+3e-9
                    worst = max(worst, abs(actual+scale*norm.pdf(z))/budget)
                count += 1
    print(f'{count} exact-gap/budget checks, {quantile_count} quantile bounds: PASS; max error/budget={worst:.6f}')
    for beta in (0., .25, .5, 1.):
        values = []
        for c in (64., 256., 1024., 4096.):
            sigma = .5*c**(-beta)
            lam = sigma*math.sqrt(c)
            t = brentq(lambda x: cdf(c, x, lam)-.9, 0., math.sqrt(c)+8*lam)
            value = gap(c, t, lam)
            prediction = -norm.pdf(norm.ppf(.9))/(2*c*math.sqrt(sigma*sigma+1/c))
            assert abs(value/prediction-1) < .04
            values.append(abs(value))
        order = math.log(values[-2]/values[-1])/math.log(4)
        assert abs(order-max(.5, 1-beta)) < .03
        print(f'beta={beta:.2f}: gap at c=4096 {-values[-1]:.9e}; order={order:.6f}')
    values = []
    for c in (64., 256., 1024., 4096.):
        t = brentq(lambda x: cdf(c, x, 0.)-.9, 0., math.sqrt(c))
        values.append(abs(gap(c, t, 0.)))
    order = math.log(values[-2]/values[-1])/math.log(4)
    assert abs(order-.5) < .01
    print(f'Zero noise: gap at c=4096 {-values[-1]:.9e}; order={order:.6f}')
    print('PASS')


if __name__ == '__main__':
    main()
