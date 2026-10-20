"""Finite-chain odd scores: Fourier/occupation/moment cross-checks, not interval bounds."""
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import gammaincc, ndtr
from verify_path_crossover import asymmetric_occupation_density, finite_chain_path_cdfs


def phi(x):
    return np.exp(-np.asarray(x)**2/2)/math.sqrt(2*math.pi)


def root(t, d):
    return np.copysign(abs(t)**(1/d), t)


def integrate(function, lo, hi, points=()):
    return quad_vec(function, lo, hi, points=[x for x in points if lo < x < hi],
                    epsabs=3e-10, epsrel=3e-10, limit=300)[0]


def cuts(t, lam, d):
    return [0.] + [root(t+j*lam, d) for j in (-8, 0, 8)]


def reference(u, lam, d):
    t = (1+lam)*u
    if not lam:
        return ndtr(root(t, d))
    return integrate(lambda x: phi(x)*ndtr((t-x**d)/lam), -10., 10., cuts(t, lam, d))


def leading(u, lam, d):
    t = (1+lam)*u
    if not lam:
        return phi(root(t, d))
    if lam < 1:
        return (1+lam)*integrate(lambda z: phi(root(t-lam*z, d))*phi(z),
                                 -10., 10., [t/lam])
    return (1+lam)/lam*integrate(lambda x: d*x**(d-1)*phi(x)*phi((t-x**d)/lam),
                                -10., 10., cuts(t, lam, d))


class Law:
    def __init__(self, q, values, c, order=320, upper=80., bound=9.):
        n = len(q)
        self.pi = np.linalg.solve(q.T+np.ones((n, n)), np.ones(n))
        self.f = values-self.pi@values
        inv = np.linalg.inv(q-np.outer(np.ones(n), self.pi))+np.outer(np.ones(n), self.pi)
        self.h = inv@self.f
        self.var = -2*self.pi@(self.f*self.h)
        self.skew = 6*self.pi@(self.f*(inv@(self.f*self.h)))
        v, w = leggauss(order)
        self.u, self.w = upper*(v+1)/2, upper*w/(2*math.pi)
        self.cf = np.array([expm(c*q+1j*u*math.sqrt(c/self.var)*np.diag(self.f))
                            @np.ones(n) for u in self.u])
        self.atom = np.zeros(n)
        for level in np.unique(self.f):
            keep = np.flatnonzero(self.f == level)
            self.atom[keep] = expm(c*q[np.ix_(keep, keep)])@np.ones(len(keep))
        self.location = math.sqrt(c/self.var)*self.f
        self.cont = self.cf-np.exp(1j*np.outer(self.u, self.location))*self.atom
        xx, ww = leggauss(2*order)
        self.x, self.xw, self.bound = bound*xx, bound*ww, bound
        self.grid = np.array([self.cdfx(x) for x in self.x])
        self.gap = self.grid-(self.grid@self.pi)[:, None]
        self.c, self.q = c, q

    def cdfx(self, x):
        return ((1-self.atom)/2-(self.w/self.u)@np.imag(np.exp(-1j*self.u*x)[:, None]*self.cont)
                +self.atom*(self.location <= x))

    def cdf(self, u, lam, d):
        t = (1+lam)*u
        if not lam:
            return self.cdfx(root(t, d))
        if lam < 1:
            return integrate(lambda z: self.cdfx(root(t-lam*z, d))*phi(z), -10., 10., [t/lam])
        kernel = lambda x: d*x**(d-1)*phi((t-x**d)/lam)/lam
        if lam >= 100:
            return ndtr(t/lam)+(self.xw*kernel(self.x))@(self.grid-(self.x >= 0)[:, None])
        return ndtr(t/lam)+integrate(lambda x: kernel(x)*(self.cdfx(x)-(x >= 0)),
                                     -self.bound, self.bound, cuts(t, lam, d))

    def weighted_gap(self, u, lam, d):
        if lam < 1:
            f = self.cdf(u, lam, d)
            return math.sqrt(self.c)*(1+lam)*(f-self.pi@f)
        t = (1+lam)*u
        kernel = lambda x: d*x**(d-1)*phi((t-x**d)/lam)*(1+lam)/lam
        if lam >= 100:
            value = (self.xw*kernel(self.x))@self.gap
        else:
            def integrand(x):
                f = self.cdfx(x)
                return kernel(x)*(f-self.pi@f)
            value = integrate(integrand, -self.bound, self.bound, cuts(t, lam, d))
        return math.sqrt(self.c)*value

    def moments(self, order):
        n = len(self.q)
        block = np.kron(np.eye(order+1), self.c*self.q)
        for j in range(1, order+1):
            block[j*n:(j+1)*n, (j-1)*n:j*n] = j*math.sqrt(self.c/self.var)*np.diag(self.f)
        return (expm(block)@np.r_[np.ones(n), np.zeros(n*order)]).reshape(order+1, n)

    def cdf_budget(self):
        a, b = self.h/math.sqrt(self.var), self.skew/(6*self.var**1.5)
        approx = np.exp(-self.u**2/2)[:, None]*(1+(-1j*np.outer(self.u, a)
                    +b*(1j*self.u[:, None])**3)/math.sqrt(self.c))
        monomial = lambda j: (j/math.e)**(j/2)/math.sqrt(2*math.pi)
        bound = phi(0)+(abs(a)*monomial(1)+abs(b)*(monomial(3)+3*monomial(1)))/math.sqrt(self.c)
        return 2*(self.w/self.u)@abs(self.cf-approx)+24*bound/(math.pi*self.u[-1])


def main():
    q = np.array([[-1.8, 1.4, .4], [.3, -1.2, .9], [1.1, .2, -1.3]])
    raw = np.array([1., -.7, .2])
    errors, refinement, moment_errors, l1_errors = [], 0., [], []
    for c in (32., 128., 512., 2048.):
        law, fine = Law(q, raw, c), Law(q, raw, c, 480, 96., 11.)
        pi, h, var = law.pi, law.h, law.var
        moments, worst, moment_error = law.moments(10), 0., 0.
        for x in (-2., -.5, 0., 1., 2.):
            approximation = ndtr(x)+phi(x)/math.sqrt(c)*(h/math.sqrt(var)
                        -law.skew/(6*var**1.5)*(x*x-1))
            assert np.all(abs(law.cdfx(x)-approximation) <= law.cdf_budget()+3e-8)
            exact = finite_chain_path_cdfs(x, c, q, law.f, math.sqrt(c/var), 2.)
            assert np.max(abs(law.cdf(x/3, 2., 1)-exact)) < 3e-8
        weighted = []
        eps = law.cdf_budget()
        supremum = math.sqrt(c)*(eps+pi@eps)
        for d in (1, 3, 5):
            residual = math.sqrt(c)*law.gap-phi(law.x)[:, None]*h/math.sqrt(var)
            measured = (law.xw*d*abs(law.x)**(d-1))@abs(residual)
            absolute_moment = 2**((d-1)/2)*math.gamma(d/2)/math.sqrt(math.pi)
            budget = (2*3**d*supremum+math.sqrt(c)*3**(d-10)*(moments[10]+pi@moments[10])
                      +d*abs(h)/math.sqrt(var)*absolute_moment*gammaincc(d/2, 4.5))
            assert np.all(measured <= budget+3e-6)
            weighted.append(np.max(measured))
            factor = math.prod(range(1, d+1, 2))
            delta = moments[d]-pi@moments[d]
            moment_error = max(moment_error, np.max(abs(math.sqrt(c)*delta+factor*h/math.sqrt(var))))
            for lam in (0., .02, c**-.5, .5, 2., math.sqrt(c), .5*(c/var)**(d/2), c**(d/2+1)):
                for p in (.5, .9):
                    u = brentq(lambda u: pi@law.cdf(u, lam, d)-p, -20., 20., xtol=2e-10)
                    v = brentq(lambda u: reference(u, lam, d)-p, -20., 20., xtol=2e-10)
                    actual = law.weighted_gap(u, lam, d)
                    prediction = h/math.sqrt(var)*leading(v, lam, d)
                    refinement = max(refinement, np.max(abs(actual-fine.weighted_gap(u, lam, d))))
                    worst = max(worst, np.max(abs(actual-prediction)))
                    assert abs(pi@prediction) < 3e-8
                    if lam == c**(d/2+1):
                        taylor = -math.sqrt(c)*(1+lam)/lam*phi((1+lam)*u/lam)*delta
                        allowance = math.sqrt(c)*(1+lam)/(2*lam**2*math.sqrt(2*math.pi*math.e))*(moments[2*d]+pi@moments[2*d])
                        assert np.all(abs(actual-taylor) <= allowance+3e-6)
        errors.append(worst)
        moment_errors.append(moment_error)
        l1_errors.append(weighted)
        print(f'c={c:.0f}: worst weighted quantile error {worst:.9g}; odd-moment error {moment_error:.9g}; '
              f'weighted L1 errors '+', '.join(f'{x:.7g}' for x in weighted))
    assert errors[-1] < errors[0]/4 and moment_errors[-1] < moment_errors[0]/8 and refinement < 3e-6
    assert np.all(np.array(l1_errors[-1]) < np.array(l1_errors[0])/4)
    print(f'192 quantile cases; maximum weighted refinement {refinement:.3g}; h={h}; Gamma={var:.9f}')
    a, b, c = 1.7, .4, 16.
    binary = Law(np.array([[-a, a], [b, -b]]), np.array([1., -1.]), c, 1280, 320.)
    for x in (-1., 0., 1.):
        threshold = np.clip((x*math.sqrt(binary.var/c)+b/(a+b)*2)/2, 0, 1)
        exact = np.array([integrate(lambda u: asymmetric_occupation_density(u, c, a, b, i), 0., threshold)
                          +math.exp(-b*c)*(i == 1) for i in (0, 1)])
        assert np.max(abs(binary.cdfx(x)-exact)) < 3e-7
    print('Six unequal-rate Bessel CDF comparisons: PASS')
    symmetric = np.array([[-1., 1., 0.], [1., -2., 1.], [0., 1., -1.]])
    atom = Law(symmetric, np.array([-1., 0., 1.]), 1.)
    excess = atom.pi@atom.cdfx(atom.location[1])-.5
    assert abs(excess-math.exp(-2)/6) < 3e-8
    randomized = atom.cdfx(atom.location[1])-.5*atom.atom*(atom.location == atom.location[1])
    assert abs(atom.pi@randomized-.5) < 3e-8
    print(f'Zero-noise pooled median overshoot at c=1: {excess:.11f} = exp(-2)/6')
    defective = Law(np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]]), np.array([1., 1., -8.]), 1.)
    expected = np.array([2/math.e, 1/math.e, math.exp(-4)])
    assert np.max(abs(defective.atom-expected)) < 3e-8
    jump = defective.cdfx(defective.location[0])-defective.cdfx(defective.location[0]-1e-8)
    assert np.max(abs(jump-expected*(defective.location == defective.location[0]))) < 3e-7
    print('Defective generator with repeated-feature killed-block atoms: PASS')
    print('PASS')


if __name__ == '__main__':
    main()
