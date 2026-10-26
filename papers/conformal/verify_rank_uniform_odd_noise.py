"""CDF-coordinate odd-score profile checks; numerical, not interval bounds."""
import math
import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri, roots_jacobi


def phi(x):
    return np.exp(-np.asarray(x)**2/2)/math.sqrt(2*math.pi)


def root(x, d):
    return np.copysign(abs(x)**(1/d), x)


def integrate(function, points=(), fine=False):
    return quad_vec(function, -12., 12., points=[x for x in points if -12 < x < 12],
                    epsabs=2e-10 if fine else 2e-8,
                    epsrel=2e-10 if fine else 2e-8, limit=300)[0]


def cuts(t, rho, d):
    return [0.]+[root((t+j*rho)/(1-rho), d) for j in (-8, 0, 8)]


def cdf(t, rho, d, fine=False):
    if rho == 0:
        return ndtr(root(t, d))
    if rho == 1:
        return ndtr(t)
    return integrate(lambda g: phi(g)*ndtr((t-(1-rho)*g**d)/rho),
                     cuts(t, rho, d), fine)


def profile(t, rho, d, fine=False):
    if rho == 0:
        return phi(root(t, d))
    if rho == 1:
        return math.prod(range(1, d+1, 2))*phi(t)
    if rho < .5:
        return integrate(lambda z: phi(z)*phi(root((t-rho*z)/(1-rho), d)),
                         [t/rho], fine)/(1-rho)
    return d/rho*integrate(lambda g: g**(d-1)*phi(g)*phi((t-(1-rho)*g**d)/rho),
                           cuts(t, rho, d), fine)


def quantile(u, rho, d, fine=False):
    if rho == 0:
        return ndtri(u)**d
    if rho == 1:
        return ndtri(u)
    radius = max(50, 8**d)
    return brentq(lambda t: cdf(t, rho, d, fine)-u, -radius, radius, xtol=2e-9)


def moment(r):
    return math.prod(range(1, r, 2))  # even r, including r=0


def constant(d, p=2):
    return 2*d*(moment(p*(d-1))*moment(p))**(1/p)


def rank_average(n, k, rho, d, order=24, fine=False):
    nodes, weights = roots_jacobi(order, n-k, k-1)
    levels, weights = (nodes+1)/2, weights/weights.sum()
    return sum(w*profile(quantile(u, rho, d, fine), rho, d, fine)
               for u, w in zip(levels, weights))


def main():
    identity_error = refinement = 0.
    comparisons = transfers = averages = 0
    for d in (1, 3, 5):
        print(f'd={d}: C_d,2={constant(d):.10f}')
        for rho in (0., .1, .5, .8, 1.):
            levels = (.001, .01, .1, .5, .9, .99, .999)
            values = [quantile(u, rho, d) for u in levels]
            heights = [profile(t, rho, d) for t in values]
            for j in range(len(levels)-1):
                delta = levels[j+1]-levels[j]
                bound = min(constant(d, p)*delta**(1-1/p) for p in (2, 4, 8, 16))
                assert abs(heights[j+1]-heights[j]) < bound+3e-7
                comparisons += 1
            if 0 < rho < 1:
                for t, height in zip(values, heights):
                    stein = -integrate(lambda g: g*phi(g)*ndtr((t-(1-rho)*g**d)/rho),
                                       cuts(t, rho, d))/(1-rho)
                    identity_error = max(identity_error, abs(stein-height))
            # Pointwise PIT transfer, independent of any beta density or N.
            for u in (.0001, .01, .5, .99, .9999):
                for delta in (1e-4, .01):
                    v = min(1-1e-8, u+delta)
                    left = profile(quantile(u, rho, d), rho, d)
                    right = profile(quantile(v, rho, d), rho, d)
                    assert abs(right-left) < constant(d)*math.sqrt(delta)+3e-7
                    transfers += 1
        for rho in (0., .5, 1.):
            for n, k in ((9, 9), (99, 50), (999, 1), (999, 999)):
                average = rank_average(n, k, rho, d)
                averages += 1
                probability = k/(n+1)
                center = profile(quantile(probability, rho, d), rho, d)
                variance = probability*(1-probability)/(n+2)
                assert abs(average-center) < constant(d)*variance**.25+3e-7
                refined = rank_average(n, k, rho, d, 36, True)
                refinement = max(refinement, abs(average-refined))
                if d == 1:
                    j = integrate(lambda z: phi(z)**2*np.exp(
                        (k-1)*np.log(np.maximum(ndtr(z), 1e-300))
                        +(n-k)*np.log(np.maximum(ndtr(-z), 1e-300))
                        +math.lgamma(n+1)-math.lgamma(k)-math.lgamma(n+1-k)),
                        [ndtri(probability)], True)
                    assert abs(average-j/math.hypot(1-rho, rho)) < 3e-5
                    if n == 9 and rho == 0:
                        print(f'J_9,9={j:.10f}')
    assert identity_error < 3e-7 and refinement < 3e-5
    # Atomic PIT intervals still lie within the Kolmogorov distance of a continuous reference.
    mass = np.array([.2, .5, .3]); points = np.array([-1., 0., 2.])
    right = np.cumsum(mass); left = right-mass; reference = ndtr(points)
    distance = max(np.max(abs(left-reference)), np.max(abs(right-reference)))
    for v in np.linspace(0, 1, 21):
        assert max(abs(left+v*mass-reference)) <= distance+3e-9
    print(f'{comparisons} probability-modulus checks; {transfers} pointwise PIT transfers; {averages} beta averages')
    print(f'Maximum Stein-identity error {identity_error:.3g}; beta refinement {refinement:.3g}; allowance 3e-5')
    print('Atomic PIT endpoint checks: 63; all rank/noise uniformity is proved analytically, not by this grid')
    print('PASS')


if __name__ == '__main__':
    main()
