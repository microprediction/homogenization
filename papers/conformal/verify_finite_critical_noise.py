"""Finite-chain critical noise: floating Fourier checks, not interval enclosures."""
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, quad_vec
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm
from verify_path_crossover import critical_quadratic_cdfs


def coefficients(q, values):
    n = len(q)
    pi = np.linalg.solve(q.T+np.ones((n, n)), np.ones(n))
    f = values-pi@values
    inverse = np.linalg.inv(q-np.outer(np.ones(n), pi))+np.outer(np.ones(n), pi)
    h = inverse@f
    boundary = 2*inverse@(f*h)
    variance, skew = -2*pi@(f*h), 3*pi@(f*boundary)
    return pi, f, h, boundary, variance, skew


def profile(u, h, boundary, variance, skew):
    return norm.pdf(u)*(-boundary/variance*u+h*skew/(3*variance**2)*(u**3-3*u))


def leading(t, lam, data):
    _, _, h, boundary, variance, skew = data
    return quad_vec(lambda u: 2*u*profile(u, h, boundary, variance, skew)
                    *norm.pdf((t-u*u)/lam)/lam, 0., 12., epsabs=2e-11)[0]


def reference(t, lam):
    return quad(lambda u: 2*norm.pdf(u)*ndtr((t-u*u)/lam), 0., 12., epsabs=2e-11)[0]


class FourierLaw:
    def __init__(self, q, values, c, order=320, upper=80.):
        self.data = coefficients(q, values)
        pi, f, _, _, variance, _ = self.data
        ux, uw = leggauss(order)
        xx, xw = leggauss(2*order)
        self.u, self.w = upper/2*(ux+1), upper/(2*math.pi)*uw
        self.x = 4*(xx+1)
        self.transform = 8*np.cos(np.outer(self.u, self.x))*xw
        self.ch = np.array([(expm(c*q+1j*u*math.sqrt(c/variance)*np.diag(f))
                             @np.ones(len(q))).real for u in self.u])
        self.weights = (self.ch.T*self.w)@self.transform
        self.c, self.upper = c, upper

    def cdf(self, t, lam):
        return self.weights@ndtr((t-self.x**2)/lam)

    def quantile(self, p, lam):
        pi = self.data[0]
        return brentq(lambda t: pi@self.cdf(t, lam)-p, -10., 20.)

    def budget(self, t, lam):
        pi, _, h, boundary, variance, skew = self.data
        transform = self.transform@ndtr((t-self.x**2)/lam)
        approximation = np.exp(-self.u**2/2)[:, None]/self.c*(
            -np.outer(self.u**2, boundary)/(2*variance)
            -np.outer(self.u**4, h)*skew/(6*variance**2))
        residual = self.ch-(self.ch@pi)[:, None]-approximation
        central = (self.w*abs(transform))@abs(residual)
        def fourth(x):
            z = (t-x*x)/lam
            return norm.pdf(z)*(-12*z/lam**2+48*x*x*(1-z*z)/lam**3
                                +16*x**4*(3*z-z**3)/lam**4)
        derivative_norm = 2*quad(lambda x: abs(fourth(x)), 0., np.inf, epsabs=2e-9)[0]
        tail = derivative_norm/math.pi*(2/(3*self.upper**3)+np.exp(-self.upper**2/2)
               /self.upper/self.c*(abs(boundary)/(2*variance*self.upper**2)
                                   +abs(h*skew)/(6*variance**2)))
        return central+tail


def spectral_check(q, values):
    pi, f, h, boundary, variance, skew = coefficients(q, values)
    theta = 2*np.pi*np.arange(64)/64
    radius, eigenvalues, amplitudes = .02, [], []
    for angle in theta:
        z = radius*np.exp(1j*angle)
        eig, right = np.linalg.eig(q+z*np.diag(f))
        j = np.argmin(abs(eig))
        projector = np.outer(right[:, j], np.linalg.inv(right)[j])
        amplitude = projector@np.ones(len(q))
        eigenvalues.append(eig[j])
        amplitudes.append(amplitude-pi@amplitude)
    derivative = lambda data, k: math.factorial(k)/radius**k*np.mean(
        np.asarray(data)*np.exp(-1j*k*theta)[:, None], axis=0)
    assert abs(derivative(np.asarray(eigenvalues)[:, None], 2)[0]-variance) < 2e-8
    assert abs(derivative(np.asarray(eigenvalues)[:, None], 3)[0]-skew) < 2e-7
    assert np.max(abs(derivative(amplitudes, 1)+h)) < 2e-9
    assert np.max(abs(derivative(amplitudes, 2)-boundary)) < 2e-8


def main():
    a, b = 1.7, .4
    q2, f2 = np.array([[-a, a], [b, -b]]), np.array([1., -1.])
    q = np.array([[-1.8, 1.4, .4], [.3, -1.2, .9], [1.1, .2, -1.3]])
    values = np.array([1., -.7, .2])
    for generator, feature in ((q2, f2), (q, values)):
        spectral_check(generator, feature)
    data2 = coefficients(q2, f2)
    for u in (.4, 1.5, 3.):
        assert abs(profile(u, *data2[2:])[0]-(a-b)/(2*b*(a+b))*(u-u**3)*norm.pdf(u)) < 2e-10
    print('Spectral eigenvalue/projector derivatives versus Poisson coefficients: PASS')
    for c in (4., 16.):
        law = FourierLaw(q2, f2, c)
        variance = law.data[4]
        for lam in (.5, 2.):
            for t in (.5, 2., 5.):
                exact = critical_quadratic_cdfs(variance*t/c, c, a, b, signal=1., noise=lam*variance/c)
                assert np.max(abs(law.cdf(t, lam)-exact)) < 3e-8
    print('12 independent two-state Bessel-CDF comparisons: PASS')
    errors, maximum_refinement, checked = [], 0., 0
    for c in (32., 128., 512., 2048.):
        law, refined = FourierLaw(q, values, c), FourierLaw(q, values, c, 480, 96.)
        pi = law.data[0]
        residuals = []
        for lam in (.5, 1., 2.):
            for p in (.5, .9):
                t = law.quantile(p, lam)
                target = brentq(lambda x: reference(x, lam)-p, -10., 20.)
                prediction = leading(target, lam, law.data)
                gap = law.cdf(t, lam)-pi@law.cdf(t, lam)
                discrepancy = np.max(abs(refined.cdf(t, lam)-law.cdf(t, lam)))
                maximum_refinement = max(maximum_refinement, discrepancy)
                assert discrepancy < 3e-8 and abs(pi@prediction) < 2e-10
                assert np.all(abs(gap-leading(t, lam, law.data)/c) <= law.budget(t, lam)+3e-8)
                residuals.append(np.max(abs(c*gap-prediction)))
                checked += 1
                if c == 2048 and lam == 1. and p == .9:
                    print('c-scaled gaps:', np.array2string(c*gap, precision=9))
                    print('predicted limits:', np.array2string(prediction, precision=9))
                    omitted = np.max(abs(prediction-leading(target, lam, (*law.data[:-1], 0.))))
                    assert omitted > .001
                    print(f'Omitting stationary skewness leaves coefficient error={omitted:.9f}')
        errors.append(max(residuals))
        print(f'c={c:.0f}: maximum scaled quantile discrepancy={errors[-1]:.9f}')
    assert errors[-1] < .002 and errors[-1] < errors[0]/8
    pi, f, h, boundary, variance, skew = law.data
    print(f'Nonreversible three-state variance={variance:.9f}; third cumulant rate={skew:.9f}')
    print(f'{checked} three-state quantile/budget checks; maximum refinement discrepancy={maximum_refinement:.3e}')
    print('PASS')


if __name__ == '__main__':
    main()
