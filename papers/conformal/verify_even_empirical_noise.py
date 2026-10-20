"""Even-score empirical calibration: floating checks, not interval enclosures."""
import math
import numpy as np
from scipy.integrate import quad_vec, trapezoid
from scipy.interpolate import PchipInterpolator
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri, roots_jacobi, roots_legendre


def phi(x):
    return np.exp(-np.asarray(x)**2/2)/math.sqrt(2*math.pi)


def integral(function, lo=-12., hi=12., points=()):
    return quad_vec(function, lo, hi, points=[x for x in points if lo < x < hi],
                    epsabs=2e-9, epsrel=2e-9, limit=300)[0]


def cuts(t, rho, d):
    return [(x/(1-rho))**(1/d) for x in (t-8*rho, t, t+8*rho) if x > 0]


class Law:
    def __init__(self, c, fine=False):
        q = np.array([[-1.8, 1.4, .4], [.3, -1.2, .9], [1.1, .2, -1.3]])
        self.pi = np.linalg.solve(q.T+np.ones((3, 3)), np.ones(3))
        f = np.array([1., -.7, .2]); f -= self.pi@f
        inverse = np.linalg.inv(q+np.outer(np.ones(3), self.pi))-np.outer(np.ones(3), self.pi)
        h = inverse@f; boundary = 2*inverse@(f*h)
        self.gamma, self.skew = -2*self.pi@(f*h), 3*self.pi@(f*boundary)
        self.a, self.e = -boundary/self.gamma, h*self.skew/(3*self.gamma**2)
        v, w = roots_legendre(480 if fine else 320)
        upper = 96. if fine else 80.
        u, w = upper*(v+1)/2, upper*w/math.pi
        cf = np.array([(expm(c*q+1j*t*math.sqrt(c/self.gamma)*np.diag(f))@np.ones(3)).real for t in u])
        self.x = np.linspace(0., 8., 9601 if fine else 4801)
        self.grid = (np.sin(self.x[:, None]*u)*(w/u))@cf
        self.fold = PchipInterpolator(self.x, self.grid, axis=0)
        self.c = c

    def base(self, t, d):
        if t <= 0:
            return np.zeros(3)
        r = t**(1/d)
        return phi(r)*(self.a*r+self.e*(r**3-3*r))

    def reference(self, t, rho, d):
        if not rho:
            return 2*ndtr(max(t, 0.)**(1/d))-1
        return integral(lambda g: 2*phi(g)*ndtr((t-(1-rho)*g**d)/rho),
                        0., 12., cuts(t, rho, d))

    def profile(self, t, rho, d):
        if not rho:
            return self.base(t, d)
        if rho < .5:
            return integral(lambda z: phi(z)*self.base((t-rho*z)/(1-rho), d),
                            hi=min(12., t/rho))/(1-rho) if t/rho > -12 else np.zeros(3)
        return d/rho*integral(lambda g: phi(g)*phi((t-(1-rho)*g**d)/rho)
            *((self.a-3*self.e)*g**d+self.e*g**(d+2)), 0., 12., cuts(t, rho, d))

    def actual(self, t, rho, d, weighted=False):
        if not rho:
            f = self.fold(np.clip(max(t, 0.)**(1/d), 0., 8.))
            return self.c*(f-self.pi@f) if weighted else f
        if rho < .5:
            f = integral(lambda z: phi(z)*self.fold(np.clip(max((t-rho*z)/(1-rho), 0.)**(1/d), 0., 8.)),
                         hi=min(12., t/rho)) if t/rho > -12 else np.zeros(3)
            return self.c/(1-rho)*(f-self.pi@f) if weighted else f
        def integrand(r):
            f = self.fold(r)
            return d/rho*r**(d-1)*phi((t-(1-rho)*r**d)/rho)*(f-self.pi@f if weighted else f-1)
        value = integral(integrand, 0., 8., cuts(t, rho, d))
        return self.c*value if weighted else ndtr(t/rho)+(1-rho)*value


def rank_average(law, d, rho, n=9, k=9, order=24):
    nodes, weights = roots_jacobi(order, n-k, k-1)
    levels, weights = (nodes+1)/2, weights/weights.sum()
    observed, predicted = np.zeros(3), np.zeros(3)
    for u, w in zip(levels, weights):
        target = brentq(lambda t: law.reference(t, rho, d)-u, -50., 8**d, xtol=2e-8)
        actual = brentq(lambda t: law.pi@law.actual(t, rho, d)-u, -50., 8**d, xtol=2e-8)
        observed += w*law.actual(actual, rho, d, True)
        predicted += w*law.profile(target, rho, d)
    return observed, predicted


def main():
    errors, refinement, identity_error = [], 0., 0.
    for c in (32, 128, 512):
        law, worst = Law(c), 0.
        fine = Law(c, True) if c == 128 else None
        a, e, x = law.a, law.e, law.x
        signed = c*(law.grid-(law.grid@law.pi)[:, None])
        delta = max(abs(law.grid@law.pi-(2*ndtr(x)-1)))
        for d in (2, 4, 6):
            moment = lambda r: math.prod(range(1, r, 2))
            v2 = (a-3*e)**2*moment(2*d)+2*e*(a-3*e)*moment(2*d+2)+e**2*moment(2*d+4)
            constant = np.maximum(np.sqrt(2*a*a+24*e*e), d*np.sqrt(v2))
            endpoint = d/2*moment(d)*(a+(d-2)*e)
            direct = integral(lambda r: d*r**(d-1)*law.base(r**d, d), 0., 12.)
            assert max(abs(direct-endpoint)) < 3e-8
            residual = signed-np.array([law.base(r**d, d) for r in x])
            sup = np.max(abs(residual), axis=0)
            transport = trapezoid((d*x**(d-1))[:, None]*abs(residual), x, axis=0)
            for rho in (0., 1/3, 2/3, 1-1e-5):
                observed, predicted = rank_average(law, d, rho)
                worst = max(worst, max(abs(observed-predicted)))
                bound = sup if not rho else np.minimum(sup/(1-rho), transport/(rho*math.sqrt(2*math.pi)))
                assert max(abs(observed-predicted)-bound-constant*math.sqrt(delta)) < 3e-5
                assert abs(law.pi@observed) < 3e-7
                if fine is not None:
                    exact, target = rank_average(fine, d, rho, order=36)
                    refinement = max(refinement, max(abs(exact-observed)), max(abs(target-predicted)))
                if c == 512 and d == 4:
                    print(f'c=512,d=4,rho={rho:g}: weighted gaps={observed}; prediction={predicted}')
            if c == 128:
                for rho in (.2, .8):
                    for t in (-.5, .5, 3.):
                        direct = -integral(lambda g: phi(g)*(a*(g*g-1)+e*(g**4-6*g*g+3))
                            *ndtr((t-(1-rho)*g**d)/rho), 0., 12., cuts(t, rho, d))/(1-rho)
                        identity_error = max(identity_error, max(abs(direct-law.profile(t, rho, d))))
                    left, right = .25, 2.
                    mass = law.reference(right, rho, d)-law.reference(left, rho, d)
                    assert max(abs(law.profile(right, rho, d)-law.profile(left, rho, d))-constant*math.sqrt(mass)) < 3e-7
            if c == 512:
                print(f'd={d}: probability modulus constants={constant}; large-noise factors={endpoint}')
        errors.append(worst)
        print(f'c={c}: worst weighted error={worst:.9g}; folded Kolmogorov error={delta:.9g}')
    assert errors[-1] < errors[0]/4 and refinement < 3e-5 and identity_error < 3e-7
    rng = np.random.default_rng(251)
    for values in rng.integers(-4, 5, size=(100, 10)):
        ties = rng.permutation(10)
        assert all(sorted(range(10), key=lambda j: (values[j]**d, ties[j]))
                   == sorted(range(10), key=lambda j: (abs(values[j]), ties[j])) for d in (2, 4, 6))
    print(f'36 rank cases; 300 exact even-rank patterns; 18 independent Hermite identities; 6 modulus checks')
    print(f'Maximum joint Fourier/spatial/beta refinement={refinement:.3g}; allowance=3e-5; identity error={identity_error:.3g}')
    print(f'Gamma={law.gamma:.9f}; kappa3={law.skew:.9f}; a={law.a}; e={law.e}')
    print('PASS')


if __name__ == '__main__':
    main()
