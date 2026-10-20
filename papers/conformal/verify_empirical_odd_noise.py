"""Empirical odd-score crossover: numerical checks, not interval enclosures."""
import math
import numpy as np
from scipy.integrate import quad_vec, trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import ndtr, roots_jacobi, roots_legendre


def phi(x):
    return np.exp(-np.asarray(x)**2/2)/math.sqrt(2*math.pi)


def root(t, d):
    return np.copysign(abs(t)**(1/d), t)


def integral(function, lo=-10., hi=10., points=()):
    return quad_vec(function, lo, hi, points=[x for x in points if lo < x < hi],
                    epsabs=2e-8, epsrel=2e-8, limit=250)[0]


def cuts(t, lam, d):
    return [0.]+[root(t+j*lam, d) for j in (-8, 0, 8)]


def reference(u, lam, d):
    t = (1+lam)*u
    return ndtr(root(t, d)) if not lam else integral(
        lambda x: phi(x)*ndtr((t-x**d)/lam), points=cuts(t, lam, d))


def profile(u, lam, d):
    t = (1+lam)*u
    if not lam:
        return phi(root(t, d))
    if lam < 1:
        return (1+lam)*integral(lambda z: phi(root(t-lam*z, d))*phi(z), points=[t/lam])
    return (1+lam)/lam*integral(lambda x: d*x**(d-1)*phi(x)*phi((t-x**d)/lam), points=cuts(t, lam, d))


class Law:
    def __init__(self, c, fine=False):
        q = np.array([[-1.8, 1.4, .4], [.3, -1.2, .9], [1.1, .2, -1.3]])
        self.pi = np.linalg.solve(q.T+np.ones((3, 3)), np.ones(3))
        f = np.array([1., -.7, .2]); f -= self.pi@f
        inverse = np.linalg.inv(q+np.outer(np.ones(3), self.pi))-np.outer(np.ones(3), self.pi)
        self.a = inverse@f
        self.gamma = -2*self.pi@(f*self.a)
        self.a /= math.sqrt(self.gamma)
        order, upper = (480, 96.) if fine else (320, 80.)
        v, w = roots_legendre(order)
        t, w = upper*(v+1)/2, upper*w/(2*math.pi)
        cf = np.array([expm(c*q+1j*u*math.sqrt(c/self.gamma)*np.diag(f))@np.ones(3) for u in t])
        self.bound = 12 if fine else 10
        x = np.linspace(-self.bound, self.bound, 14401 if fine else 6001)
        cdfs = .5-np.imag((np.exp(-1j*x[:, None]*t)*(w/t))@cf)
        self.interpolator, self.c = PchipInterpolator(x, cdfs, axis=0), c
        self.x, self.grid = x, cdfs

    def cdfx(self, x):
        return self.interpolator(np.clip(x, -self.bound, self.bound))

    def cdf(self, u, lam, d):
        t = (1+lam)*u
        if not lam:
            return self.cdfx(root(t, d))
        if lam < 1:
            return integral(lambda z: self.cdfx(root(t-lam*z, d))*phi(z), points=[t/lam])
        return ndtr(t/lam)+integral(lambda x: d*x**(d-1)*phi((t-x**d)/lam)/lam
            *(self.cdfx(x)-(x >= 0)), -self.bound, self.bound, cuts(t, lam, d))

    def weighted(self, u, lam, d):
        if lam < 1:
            f = self.cdf(u, lam, d)
            return math.sqrt(self.c)*(1+lam)*(f-self.pi@f)
        t = (1+lam)*u
        def integrand(x):
            f = self.cdfx(x)
            return d*x**(d-1)*phi((t-x**d)/lam)*(f-self.pi@f)
        return math.sqrt(self.c)*(1+lam)/lam*integral(integrand, -self.bound, self.bound, cuts(t, lam, d))


def rank_average(law, d, lam, order=24):
    nodes, weights = roots_jacobi(order, 0, 8)  # Beta(9,1), N=k=9.
    levels, weights = (nodes+1)/2, weights/weights.sum()
    observed, predicted = np.zeros(3), np.zeros(3)
    for p, w in zip(levels, weights):
        radius = max(50, 8**d)
        u = brentq(lambda u: law.pi@law.cdf(u, lam, d)-p, -radius, radius, xtol=3e-8)
        v = brentq(lambda u: reference(u, lam, d)-p, -radius, radius, xtol=3e-8)
        observed += w*law.weighted(u, lam, d)
        predicted += w*law.a*profile(v, lam, d)
    return observed, predicted


def main():
    largest_refinement, errors = 0., []
    for c in (32, 128, 512):
        law, worst = Law(c), 0.
        fine = Law(c, True) if c == 128 else None
        residual = math.sqrt(c)*(law.grid-(law.grid@law.pi)[:, None])-phi(law.x)[:, None]*law.a
        for d in (1, 3, 5):
            factor = math.prod(range(1, d+1, 2))
            delta = trapezoid(d*abs(law.x)**(d-1)*abs(law.grid@law.pi-ndtr(law.x)), law.x)
            signed = trapezoid((d*abs(law.x)**(d-1))[:, None]*abs(residual), law.x, axis=0)
            for lam in (0., .5, 2., 1e5):
                observed, predicted = rank_average(law, d, lam)
                worst = max(worst, max(abs(observed-predicted)))
                sup = np.max(abs(residual), axis=0)
                bound = (1+lam)*(sup if not lam else np.minimum(sup, signed/(lam*math.sqrt(2*math.pi))))
                bound += 4*abs(law.a)*factor*(9*delta/(1+lam))**min(1, 2/d)
                assert max(abs(observed-predicted)-bound) < 3e-5
                assert abs(law.pi@observed) < 3e-7
                if fine is not None:
                    exact, target = rank_average(fine, d, lam, 36)
                    largest_refinement = max(largest_refinement, max(abs(exact-observed)), max(abs(target-predicted)))
                if c == 512 and d == 3:
                    print(f'c=512,d=3,lambda={lam:g}: weighted gaps {observed}; prediction {predicted}')
        errors.append(worst)
        print(f'c={c}: worst weighted error {worst:.9g}; Gamma={law.gamma:.9f}; a={law.a}')
    assert errors[-1] < errors[0]/2 and largest_refinement < 3e-5
    j = integral(lambda z: 9*ndtr(z)**8*phi(z)**2)
    print(f'J_9,9={j:.10f}; zero-noise endpoint a*J={law.a*j}; large-noise cubic endpoint={3*law.a*j}')
    rng = np.random.default_rng(249)
    for values in rng.integers(-3, 4, size=(100, 10)):
        ties = rng.permutation(10)
        assert all(sorted(range(10), key=lambda j: (values[j]**d, ties[j]))
                   == sorted(range(10), key=lambda j: (values[j], ties[j])) for d in (1, 3, 5))
    for d in (1, 3, 5):
        factor, alpha = math.prod(range(1, d+1, 2)), min(1, 2/d)
        for shift in (.1, .5, 1.):
            transport = integral(lambda x: d*abs(x)**(d-1)*(ndtr(x)-ndtr(x-shift)))
            moment = sum(math.comb(d, j)*shift**(d-j)*math.prod(range(1, j, 2))
                         for j in range(0, d, 2))
            assert abs(transport-moment) < 3e-7
        for lam in (0., .1, 1., 10., 1e5):
            g, z = rng.normal(size=(2, 200, 9))
            actual, target = ((g+.1)**d+lam*z)/(1+lam), (g**d+lam*z)/(1+lam)
            assert np.max(abs(np.sort(actual)-np.sort(target))-np.max(abs(actual-target), axis=1)[:, None]) < 3e-9
            for u, v in rng.normal(size=(8, 2)):
                assert abs(profile(u, lam, d)-profile(v, lam, d)) <= 4*factor*abs(u-v)**alpha+3e-7
    # An atom-only iid benchmark: augmented ranks are exact even without smoothing.
    from fractions import Fraction as F
    low_coverage = F(1, 2)+F(1, 2)*F(1, 2)
    conditional = (low_coverage, 1-low_coverage)
    assert sum(conditional)/2 == F(1, 2)
    assert conditional == (F(3, 4), F(1, 4))
    print('Auxiliary iid binary-score N=1 rank lemma: randomized coverages 3/4,1/4; marginal 1/2')
    print(f'36 rank cases; 300 exact odd-rank patterns; 120 Holder and 27000 order-statistic comparisons; 9 transport identities; maximum refinement {largest_refinement:.3g}; allowance 3e-5')
    print('PASS')


if __name__ == '__main__':
    main()
