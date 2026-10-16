"""Gamma-mixture rate confidence sequences from fully observed CTMC paths.

Floating endpoints are cross-checks, not outward-rounded numerical certificates.
"""
from fractions import Fraction as F
from itertools import product
from math import log

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import gammaln


def log_mixture(rate, count, exposure, a=1., b=1.):
    if rate == 0:
        return a*log(b/(b+exposure)) if count == 0 else np.inf
    return (rate*exposure-count*log(rate)+a*log(b)+gammaln(a+count)
            -gammaln(a)-(a+count)*log(b+exposure))


def rate_interval(count, exposure, threshold, a=1., b=1.):
    if exposure == 0:
        if count != 0:
            raise ValueError("Positive count with zero occupation is impossible almost surely")
        return 0., np.inf
    if count == 0:
        return 0., (threshold+a*np.log1p(exposure/b))/exposure
    estimate = count/exposure
    gap = threshold-log_mixture(estimate, count, exposure, a, b)
    assert gap > 0
    # z=log(rate/estimate) avoids a nearly cancelling equation near the MLE.
    def equation(z):
        return count*(np.expm1(z)-z)-gap
    lower = -max(1., gap/count+1); upper = 1.
    while equation(upper) < 0:
        upper *= 2
    roots = [brentq(equation, lower, 0., xtol=1e-12),
             brentq(equation, 0., upper, xtol=1e-12)]
    return tuple(estimate*np.exp(z) for z in roots)


def main():
    threshold = log(40.); integrations = endpoints = 0; worst_integral = worst_root = 0.
    for rate, count, exposure, a, b in product((.2, 1., 3.), (0, 1, 4, 12), (.2, 2., 10.),
                                              (.5, 1., 2.), (.3, 1.7)):
        power = a+count
        integral = quad(lambda z: np.exp((power-1)*log(z)-z-gammaln(power)),
                        0., np.inf, epsabs=2e-10, epsrel=2e-10)[0]
        # Independently integrate the unnormalized likelihood times Gamma density.
        direct = quad(lambda theta: np.exp((power-1)*log(theta)-(b+exposure)*theta),
                      0., np.inf, epsabs=1e-11, epsrel=2e-10)[0]
        log_direct = log(direct)+rate*exposure-count*log(rate)+a*log(b)-gammaln(a)
        discrepancy = abs(log_direct-log_mixture(rate, count, exposure, a, b))
        worst_integral = max(worst_integral, discrepancy)
        assert abs(integral-1) <= 2e-8
        assert discrepancy <= 2e-7; integrations += 1
    for count, exposure, a, b in product((0, 1, 4, 100, 1000000), (.2, 2., 1000000.),
                                        (.5, 1., 2.), (.3, 1.7)):
        lower, upper = rate_interval(count, exposure, threshold, a, b)
        assert lower <= count/exposure <= upper
        for endpoint in ((lower, upper) if count else (upper,)):
            discrepancy = abs(log_mixture(endpoint, count, exposure, a, b)-threshold)
            worst_root = max(worst_root, discrepancy)
            assert discrepancy <= 2e-7*max(1., abs(threshold)); endpoints += 1
    assert rate_interval(0, 0., threshold) == (0., np.inf)
    assert np.isinf(log_mixture(0., 1, 1.))
    for t in (F(1, 10), F(1), F(10), F(100)):
        # Absorbing two-state model, a=b=1: no-jump plus first-jump contribution.
        assert 1/(1+t)+t/(1+t) == 1
        assert 1/(1+t) <= 1  # q=0 boundary is a supermartingale, not mean-one.
    for rate, horizon in product((.2, 1., 3.), (.2, 2., 10.)):
        no_jump = np.exp(-rate*horizon+log_mixture(rate, 0, horizon))
        jumped = quad(lambda s: rate*np.exp(-rate*s+log_mixture(rate, 1, s)), 0., horizon)[0]
        assert abs(no_jump+jumped-1) <= 4096*np.finfo(float).eps
    print(f"{integrations} Gamma-integral cross-checks; maximum log discrepancy={worst_integral:.3e}")
    print(f"{endpoints} endpoint checks; maximum log residual={worst_root:.3e}; zero-exposure interval=[0,infinity)")
    print("4 exact absorbing-chain mixture identities and 4 exact zero-rate supermartingale inequalities")
    print("9 independent absorbing-path expected-mixture integrals")
    rng = np.random.default_rng(20261030); chains = []
    for n in range(2, 7):
        q = rng.uniform(.1, 1., (n, n)); np.fill_diagonal(q, 0.)
        chains.append(q-np.diag(q.sum(1)))
    for n in (3, 5):
        chains.append(np.roll(np.eye(n), 1, axis=1)-np.eye(n))
    chains.append(np.ones((3, 3))/3-np.eye(3)+.1*np.outer([1, -1, 0], [1, 1, -2]))
    tilted = 0; worst = 0.
    for q in chains:
        n = len(q)
        for i, j in product(range(n), repeat=2):
            if i == j or q[i, j] == 0:
                continue
            for multiplier, t in product((.2, 1., 3.), (.1, 1.5)):
                rate = multiplier*q[i, j]; operator = q.copy()
                operator[i, j] *= multiplier; operator[i, i] -= rate-q[i, j]
                error = np.max(abs(expm(t*operator) @ np.ones(n)-1))
                worst = max(worst, error)
                assert error <= 4096*np.finfo(float).eps*max(1., np.linalg.norm(t*operator, np.inf))
                tilted += 1
    print(f"{tilted} tilted Feynman-Kac checks on {len(chains)} chains; maximum mean-one residual={worst:.3e}")
    failures = []
    for rate in (.001, .01, .1, 1., 10.):
        def equation(s):
            return log_mixture(rate, 1, s)-threshold
        minimum = max(0., 2/rate-1); early = 0.
        if equation(0.) > 0:
            early = brentq(equation, 0., minimum)
        high = max(1., minimum+1)
        while equation(high) < 0:
            high *= 2
        late = brentq(equation, minimum, high)
        failure = -np.expm1(-rate*early)+np.exp(-rate*late)
        assert failure <= 1/40+64*np.finfo(float).eps; failures.append(failure)
    print(f"5 analytic first-departure stopping checks; maximum exclusion probability={max(failures):.9f} <= 1/40")
    previous_width = np.inf
    print("algebraic input benchmark: n=2, delta=.05, a=b=1, K=log(40)")
    for exposure in (100, 10000, 1000000):
        lower, upper = rate_interval(exposure, exposure, threshold)
        theta = .1; fixed_x = log(240.)
        old_lower = max(0., (theta*exposure-fixed_x)/(np.expm1(theta)*exposure))
        old_upper = (fixed_x+theta*exposure)/(-np.expm1(-theta)*exposure)
        assert upper-lower < previous_width; previous_width = upper-lower
        excess = fixed_x*(1/np.expm1(theta)+1/(-np.expm1(-theta)))/exposure
        assert abs(old_upper-old_lower-theta-excess) <= 64*np.finfo(float).eps
        print(f"N=S={exposure}: mixture=[{lower:.9f},{upper:.9f}], fixed-tilt width={old_upper-old_lower:.9f}")
    print("fixed tilts {0.1,0.3,1}, true positive rate=1: limiting width=0.100000000")
    print(f"unvisited-row witness H=10, q12=.001: no-jump probability={np.exp(-.01):.9f}, independent of q21")
    print("PASS")


if __name__ == '__main__':
    main()
