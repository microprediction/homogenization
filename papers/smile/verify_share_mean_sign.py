"""Three-state sign preservation and a rational four-state maturity reversal."""
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'cumulants'))
from polymoments import Poly, cir_generator, switched_expectation


def law(q):
    return np.linalg.solve(q.T+np.ones_like(q), np.ones(len(q)))


def mean(q, a, k, t):
    n = len(q)
    matrix = np.zeros((2*n+1, 2*n+1))
    matrix[:n, :n] = q.T
    matrix[n:2*n, :n] = np.diag(a)
    matrix[n:2*n, n:2*n] = q.T-np.diag(k)
    matrix[-1, n:2*n] = 1
    return (expm(t*matrix) @ np.r_[law(q), .04*law(q), 0.])[-1]/2


def kernel(aa):
    """Companion ODE for the inverse characteristic polynomial, not eigenvectors."""
    coefficients = np.poly(aa)
    n = len(aa)
    companion = np.zeros_like(aa)
    companion[:-1, 1:] = np.eye(n-1)
    companion[-1] = -coefficients[:0:-1]
    return lambda t: expm(t*companion)[0, -1]


def positive_series(aa, order=220):
    n = len(aa)
    c = max(0., max(-np.diag(aa)))
    pp = aa+c*np.eye(n)
    rr = np.linalg.norm(pp, np.inf)
    power, traces = np.eye(n), [0.]
    for _ in range(order):
        power = power @ (pp/rr)
        traces.append(np.trace(power))
    hh = [1.]
    for j in range(1, order+1):
        hh.append(sum(traces[l]*hh[j-l] for l in range(1, j+1))/j)
        assert 0 <= hh[-1] <= math.comb(n+j-1, j)*(1+3e-11)
    def evaluate(t):
        term, total = 1., 1.
        for j in range(1, order+1):
            term *= rr*t/(n+j-1)
            total += hh[j]*term
        prefactor = t**(n-1)/math.factorial(n-1)
        value = prefactor*math.exp(-c*t)*total
        tail = prefactor*math.exp((rr-c)*t+(order+1)*math.log(rr*t)-math.lgamma(order+2))
        return value, tail
    return evaluate, c


def rational_four_state():
    q = sp.Matrix(4, 4, lambda i, j: int(j == (i+1) % 4)-int(i == j))
    k = sp.Matrix([sp.Rational(53, 25), sp.Rational(56, 25), sp.Rational(13, 5), sp.Rational(29, 10)])
    a, one = sp.Matrix([2, 1, 5, 5]), sp.ones(4, 1)
    aa = q-sp.diag(*k)
    functional = lambda matrix: (a.T*matrix*one-one.T*matrix*a)[0]/4
    b2, offset = functional(aa**2), functional(aa**3)-sp.trace(aa)*functional(aa**2)
    z = sp.Symbol('z')
    assert sp.expand(functional((z*sp.eye(4)-aa).adjugate())-b2*z-offset) == 0
    assert b2 == -sp.Rational(3, 200) and offset == sp.Rational(3, 40)
    slope = offset/(2*(-aa).det())
    assert slope == sp.Rational(9375, 35231888)
    print(f'Four-cycle: a=(2,1,5,5), k=(2.12,2.24,2.6,2.9); b2={b2}, offset={offset}, long-run gap slope={slope}')
    xi = sp.Matrix([sp.Rational(1, 5), sp.Rational(2, 5), 1, sp.Rational(3, 2)])
    margin = min(2*b-x*x for b, x in zip(a, xi))
    assert margin == sp.Rational(46, 25)
    for r, loading in ((sp.Integer(2), sp.Rational(1, 2)), (-sp.Rational(1, 10), sp.Rational(1, 10))):
        assert all(x*x*loading**2/2+(-sp.Rational(3, 5)*x*r-2)*loading+r*(r-1)/2 < 0 for x in xi)
    print(f'kappa=2, rho=-.6, xi=(.2,.4,1,1.5), m=1, v0=.04; minimum Feller margin={margin}; moment supersolutions (r,b)=(2,.5),(-.1,.1)')
    for t in (sp.Rational(3, 5), sp.Rational(4, 5)):
        values, errors = [], []
        for generator in (q, q.T):
            matrix, initial = sp.zeros(9), sp.zeros(9, 1)
            matrix[:4, :4] = generator.T
            matrix[4:8, :4] = sp.diag(*a)
            matrix[4:8, 4:8] = generator.T-sp.diag(*k)
            for i in range(4):
                matrix[8, 4+i] = 1
                initial[i], initial[4+i] = sp.Rational(1, 4), sp.Rational(1, 100)
            matrix *= t
            term, total = initial.copy(), initial.copy()
            for j in range(1, 161):
                term = matrix*term/j
                total += term
            ll = math.ceil(max(sum(abs(x) for x in matrix.row(i)) for i in range(9)))
            values.append(total[8]/2)
            errors.append(sp.Rational(3**ll*ll**161, 8*math.factorial(161)))
        gap, error = values[0]-values[1], sum(errors)
        lo, hi = sp.floor((gap-error)*10**12), sp.ceiling((gap+error)*10**12)
        assert lo*hi > 0 and error < sp.Rational(1, 10**100)
        assert (hi < 0) == (t == sp.Rational(3, 5))
        qf, af, kf = np.array(q, float), np.array(a, float).ravel(), np.array(k, float).ravel()
        assert abs(mean(qf, af, kf, float(t))-mean(qf.T, af, kf, float(t))-float(gap)) < 3e-11
        pp = Poly(1)
        ls = [cir_generator(pp, float(kk), float(bb/kk), float(xx), -.6) for kk, bb, xx in zip(k, a, xi)]
        legacy = [switched_expectation(g, ls, float(t), pp.monomial(b=1), pp, .04)/2 for g in (qf, qf.T)]
        assert abs(legacy[0]-legacy[1]-float(gap)) < 3e-11
        print(f'Rational T={t}: gap in [{float(lo)/1e12:.12f}, {float(hi)/1e12:.12f}], summed tail < {float(error):.3e}; order=160, dimension=9')


def main():
    rng = np.random.default_rng(206)
    chains = [np.roll(np.eye(3), 1, axis=1)-np.eye(3),
              np.ones((3, 3))/3-np.eye(3)+.1*np.outer([1., -1., 0.], [1., 1., -2.])]
    for _ in range(8):
        qq = rng.uniform(.1, 1., (3, 3))
        np.fill_diagonal(qq, 0.)
        chains.append(qq-np.diag(qq.sum(axis=1)))
    worst, count = 0., 0
    for q in chains:
        pi, a, k = law(q), rng.uniform(.5, 3., 3), rng.uniform(.5, 3., 3)
        reverse = q.T*pi[None, :]/pi[:, None]
        bb = pi @ (k*(q @ a))-pi @ (a*(q @ k))
        for speed in (.5, 2., 8.):
            aa = speed*q-np.diag(k)
            gg, (series, c) = kernel(aa), positive_series(aa)
            for z in (.3, 1., 1+2j):
                resolvent = np.linalg.inv(z*np.eye(3)-aa)
                observed = pi @ (a*(resolvent @ np.ones(3)))-pi @ resolvent @ a
                assert abs(observed-speed*bb/np.linalg.det(z*np.eye(3)-aa)) < 3e-10
            for t in (.2, 1., 2.):
                g, tail = series(t)
                assert g > 0 and abs(gg(t)-g) <= tail+3e-9
                low = speed*abs(bb)*quad(lambda u: (t-u)*u*u*math.exp(-c*u), 0., t)[0]/4
                high = speed*abs(bb)*quad(lambda u: (t-u)*u*u*math.exp(-min(k)*u), 0., t)[0]/4
                predicted = speed*bb*quad(lambda u: (t-u)*gg(u), 0., t, epsabs=2e-11)[0]/2
                observed = mean(speed*q, a, k, t)-mean(speed*reverse, a, k, t)
                worst = max(worst, abs(observed-predicted)/(1+abs(observed)))
                assert low-3e-10 <= abs(observed) <= high+3e-10
                assert observed*bb >= -3e-11
                count += 1
    assert worst < 3e-9
    print(f'{count} independent first-moment/kernel comparisons, resolvent identities, positive-series and two-sided-bound checks; max scaled error={worst:.3e}, tolerance=3e-9')
    rational_four_state()
    print('At least one weighted-strike-area zero lies in (.6,.8); no uniqueness or pointwise call-crossing count claimed')
    print('PASS')


if __name__ == '__main__':
    main()
