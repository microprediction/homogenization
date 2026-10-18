"""Exact telegraph noise crossover; quadrature checks, not interval enclosures."""
import math
import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import ive, ndtr, roots_legendre
from scipy.stats import norm
from verify_path_crossover import conditional_values, asymmetric_path_cdfs


def parameters(c, a=1., b=1.):
    r, mean, v = a+b, (b-a)/(a+b), 8*a*b/(a+b)**3
    scale = math.sqrt(c/v)
    return r, mean, v, (-1-mean)*scale, (1-mean)*scale, r*math.sqrt(v*c)/2


def kernels(c, u, a=1., b=1.):
    r, mean, v, lo, hi, factor = parameters(c, a, b)
    if u < lo or u >= hi:
        return 0., 0., 0., 0.
    x = min(1., max(-1., mean+math.sqrt(v/c)*u))
    radius = math.sqrt(max(0., 1-x*x))
    arg, decay = c*math.sqrt(a*b)*radius, c*(r+(a-b)*x)/2
    b0 = math.exp(arg-decay)*ive(0, arg)
    b1r = math.exp(arg-decay)*ive(1, arg)/radius if radius else c*math.sqrt(a*b)*math.exp(-decay)/2
    difference = math.sqrt(v*c)*((a-b)*b0/2+math.sqrt(a*b)*x*b1r)
    j = math.sqrt(v*c)*(2*a*b/r*b0+math.sqrt(a*b)*(1+mean*x)*b1r)/2
    return factor*b0, j, -factor*difference, difference


def integral(fun, lo, hi, points=()):
    if hi <= lo:
        return 0.
    margin = 1e-10*(hi-lo)
    return quad(fun, lo, hi, points=sorted({p for p in points if lo+margin < p < hi-margin}),
                epsabs=2e-11, epsrel=2e-10, limit=300)[0]


def cdf(c, t, lam, sign=0, a=1., b=1.):
    r, _, _, lo, hi, _ = parameters(c, a, b)
    test = (lambda u: ndtr((t-u)/lam)) if lam else (lambda u: float(u <= t))
    weight = b/r if not sign else float(sign == 1)
    atom = weight*math.exp(-a*c)*test(hi)+(1-weight)*math.exp(-b*c)*test(lo)
    return atom+integral(lambda u: (kernels(c, u, a, b)[1]+(weight-b/r)*kernels(c, u, a, b)[3])*test(u),
                         lo, hi, (0., t-8*lam, t, t+8*lam))


def gap(c, t, lam, a=1., b=1.):
    r, _, _, lo, hi, factor = parameters(c, a, b)
    if not lam:
        return -a/(r*factor)*kernels(c, t, a, b)[0]
    if lam < .5:
        value = integral(lambda z: kernels(c, t-lam*z, a, b)[0]*norm.pdf(z),
                         max(-12., (t-hi)/lam), min(12., (t-lo)/lam), (0.,))
    else:
        value = integral(lambda u: kernels(c, u, a, b)[0]*norm.pdf((t-u)/lam)/lam,
                         lo, hi, (0., t))
    return -a/(r*factor)*value


def budgets(c, a=1., b=1.):
    r, _, _, lo, hi, factor = parameters(c, a, b)
    tail, atoms = ndtr(lo)+norm.sf(hi), (b*math.exp(-a*c)+a*math.exp(-b*c))/r
    l1 = integral(lambda u: abs(kernels(c, u, a, b)[0]-norm.pdf(u)), lo, hi, (0.,))+tail
    derivative = integral(lambda u: abs(kernels(c, u, a, b)[2]+u*norm.pdf(u)), lo, hi, (0.,))
    derivative = (derivative+factor*(math.exp(-a*c)+math.exp(-b*c))+norm.pdf(lo)+norm.pdf(hi))/2
    eta = (integral(lambda u: abs(kernels(c, u, a, b)[1]-norm.pdf(u)), lo, hi, (0.,))+tail+atoms)/2
    epsilon = math.sqrt(2)*max(derivative, l1/math.sqrt(2*math.pi))
    return eta, epsilon


def matrix_cdf(c, t, lam, a, b, order):
    _, mean, v, _, _, _ = parameters(c, a, b)
    nodes, weights = roots_legendre(order)
    cutoff = 14/math.sqrt(1+lam*lam)
    freq, weights = (nodes+1)*cutoff/2, weights*cutoff/2
    generator, values = np.array([[-a, a], [b, -b]]), np.diag([1., -1.])
    cf = np.array([expm(c*generator+1j*k*math.sqrt(c/v)*values)@np.ones(2) for k in freq])
    phase = np.exp(-1j*freq*(t+mean*math.sqrt(c/v))-.5*lam*lam*freq*freq)
    return .5-np.sum(weights[:, None]*(phase[:, None]*cf).imag/freq[:, None], axis=0)/math.pi


def main():
    count, quantile_count, worst = 0, 0, 0.
    for c in (.5, 2., 8.):
        for q in (-.4, 0., .6):
            direct = conditional_values(c, lambda a: ndtr((q-a)/.5))
            for sign, target in zip((1, -1), direct):
                value = cdf(c, q*math.sqrt(c), .5*math.sqrt(c), sign)
                assert abs(value-target) < 2e-9
    print('18 independent Poisson--Beta CDF comparisons: PASS')
    for a, b in ((1., 1.), (1.7, .4), (.4, 1.7)):
        for c in (.5, 2., 8.):
            _, _, _, lo, hi, _ = parameters(c, a, b)
            for t in (lo, hi):
                assert abs(cdf(c, t, 0., 1, a, b)-cdf(c, t, 0., 0, a, b)-gap(c, t, 0., a, b)) < 2e-9
    print('18 zero-noise endpoint-atom identities: PASS')
    for a, b in ((1.7, .4), (.4, 1.7)):
        for c in (.5, 2., 8.):
            _, mean, v, _, _, _ = parameters(c, a, b)
            for q in (-.4, 0., .6):
                targets = asymmetric_path_cdfs(q, c, a, b)
                for sign, target in zip((1, -1), targets):
                    assert abs(cdf(c, (q-mean)*math.sqrt(c/v), .5*math.sqrt(c/v), sign, a, b)-target) < 2e-9
    print('36 unequal-rate matrix Feynman--Kac CDF comparisons: PASS')
    for a, b in ((1., 1.), (1.7, .4)):
        for c in (128., 4096.):
            _, _, _, lo, hi, _ = parameters(c, a, b)
            for lam in (0., .01, .25, 1.):
                t = brentq(lambda x: cdf(c, x, lam, 0, a, b)-.9, lo-8*lam, hi+8*lam)
                direct = np.array([cdf(c, t, lam, s, a, b) for s in (1, -1)])
                first, refined = matrix_cdf(c, t, lam, a, b, 192), matrix_cdf(c, t, lam, a, b, 384)
                assert max(abs(first-refined)) < 2e-8 and max(abs(direct-refined)) < 2e-8
    print('32 shrinking/zero-noise matrix CDF comparisons, with refinement: PASS')
    for a, b in ((1., 1.), (1.7, .4), (.4, 1.7)):
        for c in ((2., 8., 32., 128.) if a == b else (8., 32., 128.)):
            r, _, _, lo, hi, factor = parameters(c, a, b)
            mass = integral(lambda u: kernels(c, u, a, b)[0], lo, hi, (0.,))
            assert abs(mass+math.expm1(-r*c)) < 2e-9
            eta, epsilon = budgets(c, a, b)
            for lam in (0., .01, .25, 1., 4., 32.):
                for p in (.5, .9):
                    t = brentq(lambda x: cdf(c, x, lam, 0, a, b)-p, lo-8*lam, hi+8*lam, xtol=2e-11)
                    actual = cdf(c, t, lam, 1, a, b)-p
                    opposite = cdf(c, t, lam, -1, a, b)-p
                    exact = gap(c, t, lam, a, b)
                    assert abs(actual-exact) < 3e-9 and abs(b*actual+a*opposite) < 5e-9
                    scale = a/(r*factor*math.sqrt(1+lam*lam))
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
    for a, b in ((1., 1.), (1.7, .4)):
        for beta in (0., .25, .5, 1., math.inf):
            values = []
            for c in (64., 256., 1024., 4096.):
                r, _, v, lo, hi, _ = parameters(c, a, b)
                sigma = .5*c**(-beta)
                lam = sigma*math.sqrt(c/v)
                t = brentq(lambda x: cdf(c, x, lam, 0, a, b)-.9, lo-8*lam, hi+8*lam)
                value = gap(c, t, lam, a, b)
                prediction = -2*a*norm.pdf(norm.ppf(.9))/(r*r*c*math.sqrt(sigma*sigma+v/c))
                assert abs(value/prediction-1) < .15
                values.append(abs(value))
            order = math.log(values[-2]/values[-1])/math.log(4)
            assert abs(order-max(.5, 1-beta)) < .04
            print(f'rates={a},{b}, beta={beta}: gap at c=4096 {-values[-1]:.9e}; order={order:.6f}')
    print('PASS')


if __name__ == '__main__':
    main()
