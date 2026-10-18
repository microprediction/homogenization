"""Finite-chain critical noise: floating Fourier checks, not interval enclosures."""
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, quad_vec
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm
from verify_path_crossover import critical_quadratic_cdfs, asymmetric_occupation_density


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


def leading(t, lam, data, power=1):
    _, _, h, boundary, variance, skew = data
    if not lam:
        return profile(t**(1/(2*power)), h, boundary, variance, skew) if t > 0 else np.zeros_like(h)
    return quad_vec(lambda u: 2*power*u**(2*power-1)*profile(u, h, boundary, variance, skew)
                    *norm.pdf((t-u**(2*power))/lam)*(1+lam)/lam, 0., 12., epsabs=2e-11,
                    points=[x**(1/(2*power)) for x in (t-8*lam, t, t+8*lam)
                            if 0 < x < 12**(2*power)])[0]/(1+lam)


def reference(t, lam, power=1):
    if not lam:
        return max(0., 2*ndtr(max(0., t)**(1/(2*power)))-1)
    return quad(lambda u: 2*norm.pdf(u)*ndtr((t-u**(2*power))/lam), 0., 12., epsabs=2e-11,
                points=[x**(1/(2*power)) for x in (t-8*lam, t, t+8*lam)
                        if 0 < x < 12**(2*power)])[0]


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
        self.atom = np.zeros(len(q))
        for level in np.unique(f):
            keep = np.flatnonzero(f == level)
            self.atom[keep] = expm(c*q[np.ix_(keep, keep)])@np.ones(len(keep))
        self.atom_location = c/variance*f*f
        self.continuous = self.ch-np.cos(np.outer(self.u, math.sqrt(c/variance)*f))*self.atom
        folded_grid = (2*self.w/self.u*np.sin(np.outer(self.x, self.u)))@self.continuous
        folded_grid += self.atom*(self.atom_location <= self.x[:, None]**2)
        self.gap_grid = folded_grid-(folded_grid@pi)[:, None]
        self.xw = 4*xw
        self.c, self.upper = c, upper

    def cdf(self, t, lam, power=1):
        return self.weights@ndtr((t-self.x**(2*power))/lam)

    def folded(self, t):
        if t < 0:
            return np.zeros(len(self.data[0]))
        return (2*self.w*np.sin(self.u*math.sqrt(t))/self.u)@self.continuous+self.atom*(self.atom_location <= t)

    def across_noise(self, t, lam, power=1):
        if not lam:
            return self.folded(t**(1/power)) if t > 0 else self.folded(t)
        if t/lam <= -12:
            return np.zeros(len(self.data[0]))
        return quad_vec(lambda z: norm.pdf(z)*self.folded(max(0., t-lam*z)**(1/power)), -12., min(12., t/lam),
                        epsabs=2e-10)[0]

    def convolved_gap(self, t, lam, power=1):
        pi = self.data[0]
        if lam >= 100:
            # Fixed Gaussian quadrature avoids adaptive pursuit of the
            # Fourier roundoff floor in polynomially weighted far tails.
            weight = self.xw*2*power*self.x**(2*power-1)*norm.pdf(
                (t-self.x**(2*power))/lam)*(1+lam)/lam
            return weight@self.gap_grid/(1+lam)
        def integrand(u):
            folded = self.folded(u*u)
            return 2*power*u**(2*power-1)*(folded-pi@folded)*norm.pdf(
                (t-u**(2*power))/lam)*(1+lam)/lam
        return quad_vec(integrand, 0., 8., epsabs=2e-12, points=[x**(1/(2*power))
                        for x in (t-8*lam, t, t+8*lam) if 0 < x < 8**(2*power)])[0]/(1+lam)

    def quantile(self, p, lam):
        pi = self.data[0]
        return brentq(lambda t: pi@self.cdf(t, lam)-p, -10., 20.)

    def budget(self, t, lam, power=1):
        pi, _, h, boundary, variance, skew = self.data
        transform = self.transform@ndtr((t-self.x**(2*power))/lam)
        approximation = np.exp(-self.u**2/2)[:, None]/self.c*(
            -np.outer(self.u**2, boundary)/(2*variance)
            -np.outer(self.u**4, h)*skew/(6*variance**2))
        residual = self.ch-(self.ch@pi)[:, None]-approximation
        central = (self.w*abs(transform))@abs(residual)
        def fourth(x):
            degree = 2*power
            z = (t-x**degree)/lam
            d = [0.] + [-math.factorial(degree)/math.factorial(degree-j)*x**(degree-j)/lam
                        if j <= degree else 0. for j in range(1, 5)]
            return norm.pdf(z)*(d[4]-4*z*d[1]*d[3]-3*z*d[2]**2
                +6*(z*z-1)*d[1]**2*d[2]-(z**3-3*z)*d[1]**4)
        cuts = [0.] + sorted({x**(1/(2*power)) for x in
            (t-12*lam, t-3*lam, t, t+3*lam, t+12*lam) if x > 0}) + [np.inf]
        derivative_norm = 2*sum(quad(lambda x: abs(fourth(x)), low, high,
            epsabs=2e-9, limit=200)[0] for low, high in zip(cuts[:-1], cuts[1:]))
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


def zero_noise_check(q, values):
    errors, refinement_error = [], 0.
    for c in (32., 128., 512., 2048.):
        law, refined = FourierLaw(q, values, c), FourierLaw(q, values, c, 480, 96.)
        pi = law.data[0]
        residuals = []
        for lam in (0., .02, c**-.5):
            for p in (.5, .9):
                t = brentq(lambda x: pi@law.across_noise(x, lam)-p, .05, 8.)
                target = brentq(lambda x: reference(x, lam)-p, .05, 8.)
                gap = law.across_noise(t, lam)-p
                refinement_error = max(refinement_error, np.max(abs(refined.across_noise(t, lam)-gap-p)))
                assert refinement_error < 3e-8
                residuals.append(np.max(abs(c*gap-leading(target, lam, law.data))))
        errors.append(max(residuals))
        print(f'c={c:.0f}: zero/shrinking-noise maximum scaled discrepancy={errors[-1]:.9f}')
    assert errors[-1] < errors[0]/8 and errors[-1] < .002
    print(f'24 zero/shrinking-noise checks; maximum refinement discrepancy={refinement_error:.3e}')


def large_noise_check(q, values):
    errors = []
    for c in (32., 128., 512., 2048.):
        law = FourierLaw(q, values, c)
        pi, _, _, boundary, variance, _ = law.data
        residuals = []
        for lam in (math.sqrt(c), .5*c/variance, c*c):
            p = .9
            t = (1+lam)*brentq(lambda s: pi@law.cdf((1+lam)*s, lam)-p, -5., 10.)
            target = (1+lam)*brentq(lambda s: reference((1+lam)*s, lam)-p, -5., 10.)
            gap = law.convolved_gap(t, lam)
            residuals.append((1+lam)*np.max(abs(c*gap-leading(target, lam, law.data))))
            if lam == .5*c/variance:
                fixed_error = np.max(abs(c*c*gap+boundary/.5*norm.pdf(norm.ppf(p))))
        errors.append(max(residuals))
        print(f'c={c:.0f}: growing-noise weighted discrepancy={errors[-1]:.9f}; fixed-noise coefficient error={fixed_error:.9f}')
    assert errors[-1] < errors[0]/8 and fixed_error < .002


def unsmoothed_bessel_check(q, values, a, b):
    for c in (64., 256.):
        law = FourierLaw(q, values, c)
        _, f, _, _, variance, _ = law.data
        mean = values[0]-f[0]
        for t in (.3, 1., 3.):
            radius = math.sqrt(variance*t/c)
            lo, hi = max(-1., mean-radius), min(1., mean+radius)
            for start in (0, 1):
                exact = quad(lambda x: asymmetric_occupation_density((x+1)/2, c, a, b, start)/2,
                             lo, hi, epsabs=2e-11)[0]
                assert abs(law.folded(t)[start]-exact) < 3e-8
    print('12 zero-noise statewise Bessel-CDF checks: PASS')


def even_power_check(q, values):
    errors, refinement, cdf_refinement = [], 0., 0.
    for c in (32., 128., 512., 2048.):
        law, fine = FourierLaw(q, values, c), FourierLaw(q, values, c, 480, 96.)
        pi, _, h, boundary, variance, skew = law.data
        residuals, fixed_errors = [], []
        for power in (2, 3):
            n, order = len(q), 4*power
            block = np.kron(np.eye(order+1), c*q)
            for j in range(1, order+1):
                block[j*n:(j+1)*n, (j-1)*n:j*n] = j*math.sqrt(c/variance)*np.diag(law.data[1])
            moments = (expm(block)@np.r_[np.ones(n), np.zeros(n*order)]).reshape(order+1, n)
            factor = power*math.prod(range(1, 2*power, 2))
            moment = factor*(boundary/variance-2*(power-1)*h*skew/(3*variance**2))
            integral = quad_vec(lambda u: 2*power*u**(2*power-1)*profile(u, *law.data[2:]),
                                0., 12., epsabs=2e-10)[0]
            assert np.max(abs(integral+moment)) < 2e-9
            for lam in (0., .02, c**-.5, .5, 2., math.sqrt(c),
                        .5*(c/variance)**power, c**(power+1)):
                for p in (.5, .9):
                    scale = 1+lam
                    actual = lambda t, obj: obj.across_noise(t, lam, power) if lam < 3 else obj.cdf(t, lam, power)
                    t = scale*brentq(lambda s: pi@actual(scale*s, law)-p, -5., 100.)
                    cdf_refinement = max(cdf_refinement, np.max(abs(actual(t, fine)-actual(t, law))))
                    target = scale*brentq(lambda s: reference(scale*s, lam, power)-p, -5., 100.)
                    gap = law.convolved_gap(t, lam, power) if lam else actual(t, law)-p
                    fine_gap = fine.convolved_gap(t, lam, power) if lam else actual(t, fine)-p
                    refinement = max(refinement, c*scale*np.max(abs(gap-fine_gap)))
                    prediction = leading(target, lam, law.data, power)
                    residuals.append(c*scale*np.max(abs(gap-prediction/c)))
                    assert abs(pi@prediction) < 2e-9
                    if lam == 2.:
                        assert np.all(abs(gap-leading(t, lam, law.data, power)/c)
                                      <= law.budget(t, lam, power)+3e-8)
                    if lam == .5*(c/variance)**power and p == .9:
                        limit = -moment*variance**power/.5*norm.pdf(norm.ppf(p))
                        fixed_errors.append(np.max(abs(c**(power+1)*gap-limit)))
                        if c == 2048:
                            print(f'power={power}: fixed-noise coefficient {limit}')
                    if lam == c**(power+1):
                        delta = moments[2*power]-pi@moments[2*power]
                        taylor = -norm.pdf(t/lam)*delta/lam
                        allowance = (moments[4*power]+pi@moments[4*power])/(2*lam**2*math.pi)
                        assert np.all(c*lam*abs(gap-taylor) <= c*lam*allowance+3e-5)
        errors.append(max(residuals))
        print(f'c={c:.0f}: even-power weighted discrepancy={errors[-1]:.9f}; '
              f'fixed-noise errors='+', '.join(f'{x:.9f}' for x in fixed_errors))
    assert cdf_refinement < 3e-8 and refinement < 2e-5 and errors[-1] < errors[0]/8
    print(f'128 quartic/sextic quantile cases, 16 Fourier and 16 independent moment budgets; '
          f'weighted refinement={refinement:.3e}; CDF refinement={cdf_refinement:.3e}')


def even_bessel_check(q, values, a, b):
    data = coefficients(q, values)
    mean = values[0]-data[1][0]
    error = 0.
    for c in (4., 16.):
        # Steep score maps need more Fourier bandwidth at short horizons.
        law = FourierLaw(q, values, c, 1280, 320.)
        for power in (2, 3):
            for lam in (.5, 2.):
                for t in (.5, 5.):
                    def kernel(raw):
                        return ndtr((t-(math.sqrt(c/data[4])*(raw-mean))**(2*power))/lam)
                    exact = np.array([quad(lambda u: kernel(2*u-1)*asymmetric_occupation_density(
                        u, c, a, b, i), 0., 1., epsabs=2e-11)[0]
                        + math.exp(-rate*c)*kernel(values[i]) for i, rate in enumerate((a, b))])
                    error = max(error, np.max(abs(law.cdf(t, lam, power)-exact)))
    assert error < 5e-8
    print(f'16 independent quartic/sextic Bessel-CDF vectors: max error={error:.3e}')


def main():
    a, b = 1.7, .4
    q2, f2 = np.array([[-a, a], [b, -b]]), np.array([1., -1.])
    q = np.array([[-1.8, 1.4, .4], [.3, -1.2, .9], [1.1, .2, -1.3]])
    values = np.array([1., -.7, .2])
    repeated = FourierLaw(q, np.array([1., 1., -.7]), .5)
    location = repeated.atom_location[0]
    jump = repeated.folded(location)-repeated.folded(location-1e-9)
    assert np.max(abs(jump-repeated.atom*(repeated.atom_location == location))) < 2e-7
    pi, f = repeated.data[:2]
    weighted = np.diag(np.sqrt(pi))@q@np.diag(1/np.sqrt(pi))
    k = 2*len(q)*np.linalg.norm((weighted+weighted.T)/2, 2)
    for theta in (.2, 1., 10., 1000.):
        matrix = weighted+1j*theta*np.diag(f)
        gamma = -np.max(np.linalg.eigvals(matrix).real)
        assert gamma > 0
        for horizon in (.1, 1., 4.):
            bound = math.exp(-gamma*horizon)*sum((k*horizon)**j/math.factorial(j) for j in range(len(q)))
            assert np.linalg.norm(expm(horizon*matrix), 2) <= bound+2e-9
    print('Repeated-feature atoms and 12 high-frequency Schur-bound checks: PASS')
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
    unsmoothed_bessel_check(q2, f2, a, b)
    zero_noise_check(q, values)
    large_noise_check(q, values)
    even_bessel_check(q2, f2, a, b)
    even_power_check(q, values)
    print('PASS')


if __name__ == '__main__':
    main()
