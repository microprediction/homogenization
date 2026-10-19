"""Finite vanilla-strip entropy error; rational CIR quote-separation design."""
import math

import numpy as np
import sympy as sp
from scipy.special import ndtr, xlogy


def entropy(s):
    return xlogy(s, s)-s+1


def grid(lower, upper, intervals):
    h = math.log(upper/lower)/intervals
    d = h/(-math.expm1(-h))-1
    weights = np.full(intervals+1, h)
    weights[0], weights[-1] = d, h-d
    return np.exp(np.linspace(math.log(lower), math.log(upper), intervals+1)), weights, d-math.log1p(d)


def bs(strikes, variance):
    root = math.sqrt(variance)
    d1 = -np.log(strikes)/root+root/2
    return ndtr(d1)-strikes*ndtr(d1-root)


def rational_design():
    q = sp.Matrix([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
    a, xi = sp.Matrix([1, 2, 3]), sp.Matrix([sp.Rational(1, 5), sp.Rational(2, 5), 1])
    k = [2+sp.Rational(3, 5)*x for x in xi]
    values, errors = [], []
    for generator in (q, q.T):
        matrix, initial = sp.zeros(7), sp.zeros(7, 1)
        matrix[:3, :3] = generator.T
        matrix[3:6, :3] = sp.diag(*a)
        matrix[3:6, 3:6] = generator.T-sp.diag(*k)
        for i in range(3):
            matrix[6, 3+i] = 1
            initial[i], initial[3+i] = sp.Rational(1, 3), sp.Rational(1, 75)
        term, total = initial.copy(), initial.copy()
        for j in range(1, 161):
            term = matrix*term/j
            total += term
        ll = math.ceil(max(sum(abs(x) for x in matrix.row(i)) for i in range(7)))
        values.append(total[6]/2)
        errors.append(sp.Rational(3**ll*ll**161, 6*math.factorial(161)))
    difference, error = values[0]-values[1], sum(errors)
    assert difference+error < -sp.Rational(1, 4000) and error < sp.Rational(1, 10**100)
    def exp_bounds(x):
        term = total = sp.Integer(1)
        for j in range(1, 161):
            term *= x/j
            total += term
        return total, total+term*x/161/(1-x/162)
    assert exp_bounds(sp.Rational(2303, 100))[0] > 10**10
    assert exp_bounds(sp.Rational(38, 25))[1] < sp.Rational(23, 5)
    assert exp_bounds(sp.Rational(1, 100))[1] < sp.Rational(10101, 10000)
    assert sp.Rational(101, 10000)**2/8 < sp.Rational(13, 10**6)
    for x in xi:
        assert x*x/8-(2+sp.Rational(6, 5)*x)/2+1 < 0
    assert min(2*b-x*x for b, x in zip(a, xi)) > 0
    bound = sp.Rational(46, 10**6)
    minimum = (sp.Rational(1, 4000)-bound)/24
    assert minimum == sp.Rational(17, 2*10**6)
    assert -sp.Rational(1, 4000)+bound+sp.Rational(48, 10**6) == -sp.Rational(156, 10**6)
    strikes, weights, alpha = grid(1e-5, 1e5, 2303)
    observed_bound = alpha+1e-5+math.exp(1.52)/(2e5)
    assert observed_bound < float(bound)
    print(f'CIR fixture: unit three-cycle, a=(1,2,3), kappa=2, xi=(.2,.4,1), rho=-.6, T=m=1, v0=.04')
    print(f'Rational share-mean gap={float(difference):.12g}; upper bound < -1/4000; total Taylor tail < {float(error):.3e}; order=160')
    print(f'Rational scalar bounds: h<.01, M2=exp(1.52)<4.6, alpha<13e-6, E<46e-6')
    print(f'2304 strikes in [1e-5,1e5]: actual h={weights[1]:.12g}, alpha={alpha:.12g}, E={observed_bound:.12g}')
    print('Guaranteed max exact grid call gap > 8.5e-6; per-curve quote errors <=1e-6 imply observed strip < -156e-6')


def main():
    rng = np.random.default_rng(207)
    maximum, cases = 0., 0
    for _ in range(20):
        laws = []
        for _ in range(2):
            s, p = np.exp(rng.normal(0., 1.8, 7)), rng.dirichlet(np.ones(7))
            s[0] = 0.  # No negative moment exists; the theorem still applies.
            s /= p @ s
            laws.append((s, p))
        for lower, upper, intervals in ((.01, 100., 16), (.001, 1000., 80)):
            strikes, weights, alpha = grid(lower, upper, intervals)
            assert np.all(weights > 0) and abs(weights.sum()-math.log(upper/lower)) < 3e-12
            exact, polygon, calls = [], [], []
            for s, p in laws:
                prices = np.maximum(s[None, :]-strikes[:, None], 0.) @ p
                payoff = entropy(lower)+math.log(lower)*(s-lower)+weights @ np.maximum(s[None, :]-strikes[:, None], 0.)
                interior = np.interp(np.clip(s, lower, upper), strikes, entropy(strikes))
                interior = np.where(s < lower, entropy(lower)+math.log(lower)*(s-lower), interior)
                interior = np.where(s > upper, entropy(upper)+math.log(upper)*(s-upper), interior)
                assert max(abs(payoff-interior)) < 3e-10*(1+max(abs(payoff)))
                exact.append(p @ entropy(s)); polygon.append(p @ payoff); calls.append(prices)
            statistic = weights @ (calls[0]-calls[1])
            gap = exact[0]-exact[1]
            for power in (1.25, 2., 3.5):
                moment = max(p @ s**power for s, p in laws)
                budget = alpha+lower+moment*upper**(1-power)/(power*(power-1))
                maximum = max(maximum, abs(statistic-gap)/budget)
                assert abs(statistic-gap) <= budget+3e-10
            assert abs(statistic-polygon[0]+polygon[1]) < 3e-10
            perturbation = 2e-6*weights.sum()
            assert abs(weights @ ((calls[0]+1e-6)-(calls[1]-1e-6))-statistic-perturbation) < 3e-10
            cases += 1
    for h in (.01, .2, 1., 2.):
        strikes, weights, alpha = grid(1., math.exp(3*h), 3)
        x = h/(-math.expm1(-h))
        chord = entropy(1.)+(x-1)*(entropy(math.exp(h))-entropy(1.))/math.expm1(h)
        assert abs((chord-entropy(x))/x-alpha) < 3e-11
    for power in (1.25, 2., 3.5):
        for target in (.01, .001, .0001):
            lower = target/3
            upper = max(2., (9/(power*(power-1)*target))**(1/(power-1)))
            h0 = math.log1p(math.sqrt(8*target/3))
            intervals = math.ceil(math.log(upper/lower)/h0)
            _, _, alpha = grid(lower, upper, intervals)
            assert alpha+lower+3*upper**(1-power)/(power*(power-1)) <= target*(1+3e-10)
    strikes, weights, alpha = grid(1e-5, 1e5, 2303)
    mixed = (bs(strikes, .01)+bs(strikes, .49))/2
    simple = bs(strikes, .25)
    atm = float((bs(np.array([1.]), .01)[0]+bs(np.array([1.]), .49)[0])/2-bs(np.array([1.]), .25)[0])
    statistic = weights @ (mixed-simple)
    budget = alpha+1e-5+max((math.exp(.01)+math.exp(.49))/2, math.exp(.25))/(2e5)
    assert abs(statistic) < budget and abs(atm) > .03
    print(f'{cases} two-law atom/default cases at powers 1.25,2,3.5, polygon identities and adversarial quote errors; maximum error/budget={maximum:.6g}; four sharp chord controls and nine target-error designs')
    print(f'Zero-entropy negative control: variances (.01,.49) equally mixed versus .25; ATM call gap={atm:.12g}; strip={statistic:.3e}, budget={budget:.3e}')
    rational_design()
    print('No CIR call inversion used: finite quote separation follows from the proved inequality and rational mean gap')
    print('PASS')


if __name__ == '__main__':
    main()
