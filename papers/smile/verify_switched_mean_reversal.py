"""Exact share-mean cancellation criterion; no global crossing sharpness claim."""
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
    return np.linalg.solve(q.T + np.ones_like(q), np.ones(len(q)))


def integrated_mean(q, a, killing, maturity, v0=.04):
    """Forward probabilities, variance-weighted probabilities, integrated mean."""
    n = len(a)
    matrix = np.zeros((2*n+1, 2*n+1))
    matrix[:n, :n] = q.T
    matrix[n:2*n, :n] = np.diag(a)
    matrix[n:2*n, n:2*n] = q.T - np.diag(killing)
    matrix[-1, n:2*n] = 1
    return (expm(maturity*matrix) @ np.r_[law(q), v0*law(q), 0.])[-1]


def gap_formula(q, a, killing, maturity):
    pi = law(q)
    aa = q - np.diag(killing)
    def integrand(u):
        ee = expm(u*aa)
        return (maturity-u) * (pi @ (a*(ee @ np.ones(len(q)))) - pi @ ee @ a)/2
    return quad(integrand, 0., maturity, epsabs=2e-12, epsrel=2e-11)[0]


def rational_witness(a, target):
    """Direct joint (X,v,Y) moments, with an exact rational Taylor enclosure."""
    pairs = [(p, d-p) for d in range(3) for p in range(d+1)]
    where = {v: j for j, v in enumerate(pairs)}
    xi = [sp.Rational(1, 5), sp.Rational(2, 5), sp.Integer(1)]
    rho = -sp.Rational(3, 5)
    q = sp.Matrix([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
    values, errors, ceilings = [], [], []
    for generator in (q, q.T):
        n = 3*len(pairs) if target == 'second' else 7
        matrix, initial = sp.zeros(n), sp.zeros(n, 1)
        if target == 'second':
            for (p, v), j in where.items():
                for i in range(3):
                    row = 3*j+i
                    if p == 0: initial[row] = sp.Rational(1, 3*25**v)
                    for h in range(3): matrix[row, 3*j+h] += generator.T[i, h]
                    matrix[row, row] -= 2*v
                    if p: matrix[row, 3*where[(p-1, v+1)]+i] -= sp.Rational(p, 2)
                    if p >= 2: matrix[row, 3*where[(p-2, v+1)]+i] += sp.Rational(p*(p-1), 2)
                    if v: matrix[row, 3*where[(p, v-1)]+i] += v*a[i]+v*(v-1)*xi[i]**2/2
                    if p and v: matrix[row, 3*where[(p-1, v)]+i] += p*v*rho*xi[i]
            output = [3*where[(2, 0)]+i for i in range(3)]
            factor = sp.Integer(1)
        else:
            matrix[:3, :3] = generator.T
            matrix[3:6, :3] = sp.diag(*a)
            matrix[3:6, 3:6] = generator.T-sp.diag(*[2-rho*x for x in xi])
            for i in range(3):
                matrix[6, 3+i] = 1
                initial[i], initial[3+i] = sp.Rational(1, 3), sp.Rational(1, 75)
            output, factor = [6], sp.Rational(1, 2)
        term = initial.copy()
        total = term.copy()
        for j in range(1, 161):
            term = matrix*term/j
            total += term
        ceiling = math.ceil(max(sum(abs(x) for x in matrix.row(i)) for i in range(n)))
        ceilings.append(ceiling)
        values.append(factor*sum(total[i] for i in output))
        errors.append(factor*len(output)*max(abs(x) for x in initial)
                      *sp.Rational(3**ceiling*ceiling**161, math.factorial(161)))
    gap, error = values[0]-values[1], sum(errors)
    assert abs(gap)-error > sp.Rational(1, 10**5)
    assert error < sp.Rational(1, 10**90)  # Rational enclosure, not a float tolerance.
    print(f'Rational {target} witness: gap={float(gap):.12g}, total tail < {float(error):.3e}; '
          f'dimension={n}, Taylor order=160, norm ceilings={ceilings}')
    return float(gap)


def main():
    rng = np.random.default_rng(204)
    maximum, cancellation, independent = 0., 0., 0.
    count = 0
    for n in (3, 4, 6):
        for _ in range(3):
            q = rng.uniform(.1, 1.2, (n, n))
            np.fill_diagonal(q, 0.)
            q -= np.diag(q.sum(axis=1))
            pi = law(q)
            reverse = q.T*pi[None, :]/pi[:, None]
            xi, a = rng.uniform(.2, .8, n), rng.uniform(.8, 2., n)
            killing = 2+.6*xi
            for speed in (.5, 3.):
                for maturity in (.2, 1.3):
                    qq, rr = speed*q, speed*reverse
                    gap = (integrated_mean(qq, a, killing, maturity)
                           - integrated_mean(rr, a, killing, maturity))/2
                    maximum = max(maximum, abs(gap-gap_formula(qq, a, killing, maturity)))
                    aa_matrix = qq-np.diag(killing)
                    coefficient = (pi @ (killing*(qq @ a))-pi @ (a*(qq @ killing)))/48
                    ell = np.linalg.norm(aa_matrix, np.inf)
                    remainder = max(a)*math.exp(ell*maturity)*ell**3*maturity**5/120
                    assert abs(gap-coefficient*maturity**4) <= remainder+3e-12
                    if n == 3:
                        determinant = np.linalg.det(np.array([np.ones(n), killing, a]))
                        current = pi[0]*qq[0, 1]-pi[1]*qq[1, 0]
                        assert abs(48*coefficient-current*determinant) < 3e-11
                    for aa, kk in ((1+.7*killing, killing), (a, np.full(n, 2.))):
                        cancellation = max(cancellation, abs(integrated_mean(qq, aa, kk, maturity)
                                                             - integrated_mean(rr, aa, kk, maturity)))
                    pp = Poly(1)
                    ls = [cir_generator(pp, k, b/k, x, -.6) for k, b, x in zip(killing, a, xi)]
                    old = switched_expectation(qq, ls, maturity, pp.monomial(b=1), pp, .04)
                    independent = max(independent, abs(old-integrated_mean(qq, a, killing, maturity)))
                    count += 1
    assert max(maximum, cancellation, independent) < 3e-10
    print(f'{count} semigroup/quadrature and small-maturity bound cases (three-state determinant subset): formula error={maximum:.3e}; '
          f'affine/constant cancellation={cancellation:.3e}; independent moments={independent:.3e}')
    q = sp.Matrix([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
    killing = sp.Matrix([sp.Rational(53, 25), sp.Rational(56, 25), sp.Rational(13, 5)])
    a = sp.Matrix([1, 2, 3])
    coefficient = (killing.T*q*a-a.T*q*killing)[0]/(3*48)
    assert coefficient == -sp.Rational(1, 600)
    currents = (q-q.T)/3
    determinant = sp.Matrix([list(sp.ones(1, 3)), list(killing), list(a)]).det()
    assert (killing.T*currents*a)[0] == determinant/3
    print(f'Unit three-cycle, a=(1,2,3), xi=(.2,.4,1), kappa=2, rho=-.6: '
          f'short-maturity coefficient={coefficient}; determinant={determinant}')
    q4 = sp.Matrix(4, 4, lambda i, j: int(j == (i+1) % 4)-int(i == j))
    k4 = sp.Matrix([killing[0], killing[1], killing[0], killing[1]])
    a4 = sp.Matrix([1, 2, 3, 4])
    matrix = q4-sp.diag(*k4)
    one = sp.ones(4, 1)
    assert sp.Matrix.hstack(one, k4, a4).rank() == 3
    for j in range(4):
        assert (a4.T*matrix**j*one-one.T*matrix**j*a4)[0] == 0
    print('Exact four-state nonaffine cancellation: unit directed cycle, m=1, '
          'k=(2.12,2.24,2.12,2.24), a=(1,2,3,4); power tests j=0..3 all zero')
    obstruction = rational_witness(list(a), 'share')
    af = np.array(killing, dtype=float).ravel()
    nonzero = rational_witness(list(killing), 'second')
    pp = Poly(2)
    qf = np.array(q, dtype=float)
    xi = np.array([.2, .4, 1.])
    for aa in (np.array([1., 2., 3.]), af):
        assert np.all(2*aa > xi**2)
        assert np.all(.5*xi**2*.5**2-(2+1.2*xi)*.5+1 < 0)
        assert np.all(.5*xi**2*.1**2+(-.6*xi*(-.1)-2)*.1+(-.1)*(-1.1)/2 < 0)
    ls = [cir_generator(pp, 2., b/2, x, -.6) for b, x in zip(af, xi)]
    second = [switched_expectation(g, ls, 1., pp.logreturn_power(2), pp, .04) for g in (qf, qf.T)]
    assert abs(nonzero-second[0]+second[1]) < 3e-10
    assert abs(obstruction-gap_formula(qf, np.array([1., 2., 3.]), af, 1.)) < 3e-10
    print('Both witnesses: T=m=1, v0=.04; strict Feller; moment supersolutions '
          '(r,beta)=(2,.5),(-.1,.1) verified')
    print('Affine witness a=k=(2.12,2.24,2.6): equal share means at every maturity, '
          'but unequal second ordinary log moments; at least two crossings, not an exactly-two claim')
    print('PASS')


if __name__ == '__main__':
    main()
