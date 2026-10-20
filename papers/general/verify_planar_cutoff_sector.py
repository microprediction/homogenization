"""Sharp planar hard-cutoff sector: rational constants, OU signs, matrix checks."""
from fractions import Fraction as F
from math import ceil, factorial

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import lambertw


def trig_bounds(x):
    sine = sum((-1)**j*x**(2*j+1)/factorial(2*j+1) for j in range(30))
    cosine = sum((-1)**j*x**(2*j)/factorial(2*j) for j in range(30))
    es, ec = x**61/factorial(61), x**60/factorial(60)
    return sine-es, sine+es, cosine-ec, cosine+ec


def exp_bounds(x):
    n = 80
    total = sum(x**j/factorial(j) for j in range(n+1))
    error = x**(n+1)/factorial(n+1)/(1-x/F(n+2))
    return total, total+error


def constants():
    lo, hi = F('4.493409457909'), F('4.493409457910')
    assert F(22, 7) < lo < hi < F(9, 2)  # Inside (pi,3pi/2), using 3<pi<22/7.
    sl, su, cl, cu = trig_bounds(lo)
    assert sl-lo*cu > 0
    sl, su, cl, cu = trig_bounds(hi)
    assert su-hi*cl < 0
    wl, wu = F('.777917140533'), F('.777917140535')
    assert wl*wl*exp_bounds(2*(wl+1))[1] < 1+lo*lo
    assert wu*wu*exp_bounds(2*(wu+1))[0] > 1+hi*hi
    kl, ku = F('3.1314582802'), F('3.1314582803')
    assert kl*kl < (1+lo*lo)/(wu*(wu+2))
    assert (1+hi*hi)/(wl*(wl+2)) < ku*ku
    theta = brentq(lambda t: np.sin(t)-t*np.cos(t), 4.4, 4.6)
    w = float(lambertw(np.sqrt(1+theta*theta)/np.e).real)
    kappa = np.sqrt(1+theta*theta)/np.sqrt(w*(w+2))
    assert float(kl) < kappa < float(ku)
    print(f'Exact enclosures: {lo} < theta < {hi}; {wl} < w < {wu}; {kl} < kappa_* < {ku}')
    print('Decimal enclosures: 4.493409457909 < theta < 4.493409457910; .777917140533 < w < .777917140535')
    print('3.1314582802 < kappa_* < 3.1314582803')
    print(f'Critical matrix: a=1, d={1/(1+w):.10f}, c={np.sqrt(1+theta*theta)/(1+w):.10f}, T={1+w:.10f}')
    return theta, w, kappa


def rational_witness():
    t, n = F(3, 2), 160
    matrix = [[-F(1, 2), -F(3), F(0), F(0)],
              [F(3), -F(3, 2), F(0), F(0)],
              [F(0), F(1), F(0), F(0)],
              [F(0), F(0), F(1), F(0)]]
    scaled = [[t*x for x in row] for row in matrix]
    bound = ceil(max(sum(map(abs, row)) for row in scaled))
    term = [F(0), F(1), F(0), F(0)]
    total = term[:]
    for j in range(1, n+1):
        term = [sum(x*y for x, y in zip(row, term))/j for row in scaled]
        total = [x+y for x, y in zip(total, term)]
    error = F(3**bound*bound**(n+1), factorial(n+1))
    assert error < F(34, 10**149)
    values = (total[2], total[3]/t, 2*total[3])
    for name, value, err in zip(('R', 'Bartlett', 'integrated variance'), values, (error, error/t, 2*error)):
        scaled_value = value*10**12
        lower = F(scaled_value.numerator//scaled_value.denominator, 10**12)
        upper = lower+F(1, 10**12)
        assert lower < value-err < value+err < upper
        print(f'OU T=3/2: {name} in [{float(lower):.12f},{float(upper):.12f}]')
    assert values[0] < -error and values[1] > error/t
    l = np.array([[.5, 3.], [-3., 1.5]])
    assert np.array_equal(l+l.T, np.diag([1., 3.]))
    integral = np.linalg.solve(l, np.eye(2)-expm(-1.5*l))
    bartlett = np.linalg.inv(l)-np.linalg.solve(l, integral)/1.5
    quadrature, _ = quad_vec(lambda s: expm(-s*l), 0., 1.5, epsabs=1e-10)
    assert max(abs(integral[1, 1]-float(values[0])), abs(bartlett[1, 1]-float(values[1])),
               np.linalg.norm(quadrature-integral)) < 3e-9
    print(f'OU L=[[1/2,3],[-3,3/2]], noise covariance diag(1,3), stationary covariance I; kappa^2=12; K=2/39')
    print(f'Degree-160 rational action: L={bound}; R tail < 3.4e-148; all reported output errors < 6.8e-148; independent quadrature PASS')


def matrix_checks(theta, w, critical):
    phi_star = 1.5*np.pi-theta
    rng, count, maximum = np.random.default_rng(245), 0, 0.
    for kappa in (.2, 1/np.sqrt(3), 2., critical-.03, critical+.03, 4., 8.):
        def criterion(phi):
            return np.log(kappa)+np.arcsinh(kappa*np.sin(phi))-(1.5*np.pi-phi)*np.sqrt(1+kappa*kappa*np.sin(phi)**2)/(kappa*np.cos(phi))
        for phi in (0., .05, .2, phi_star, .4, .7, 1., 1.3):
            c = kappa/np.sqrt(1+kappa*kappa*np.sin(phi)**2)
            d, b = c*np.sin(phi), c*np.cos(phi)
            base = np.array([[1-d, c], [-c, 1+d]])
            minus, plus = (1.5*np.pi-phi)/b, (1.5*np.pi+phi)/b
            first = (1-d-c*np.exp(-minus))/(1+b*b)
            assert criterion(phi) <= criterion(phi_star)+3e-9
            assert (first < 0) == (criterion(phi) > 0)
            angle, speed = rng.uniform(-3, 3), rng.uniform(.2, 3)
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            rotated = speed*rotation@base@rotation.T
            for t in (.2, 1., minus, plus, minus+2*np.pi/b):
                ic = (1+np.exp(-t)*(-np.cos(b*t)+b*np.sin(b*t)))/(1+b*b)
                isine = (b-np.exp(-t)*(np.sin(b*t)+b*np.cos(b*t)))/(1+b*b)
                expected = rotation@np.diag([ic+d*isine/b, ic-d*isine/b])@rotation.T
                observed = speed*np.linalg.solve(rotated, np.eye(2)-expm(-t*rotated/speed))
                observed = (observed+observed.T)/2
                maximum = max(maximum, np.linalg.norm(observed-expected))
                count += 1
    assert count == 280 and maximum < 3e-9
    for d in (0., .3, .8):
        for c in (0., d/2, d):
            l = np.array([[1-d, c], [-c, 1+d]])
            for t in (.1, 1., 5., 20.):
                r = np.linalg.solve(l, np.eye(2)-expm(-t*l))
                assert np.linalg.eigvalsh((r+r.T)/2)[0] > 0
    d, c = 1/(1+w), np.sqrt(1+theta*theta)/(1+w)
    l = np.array([[1-d, c], [-c, 1+d]])
    r = np.linalg.solve(l, np.eye(2)-expm(-(1+w)*l))
    assert abs(r[1, 1]) < 3e-9 and r[0, 0] > 0
    print(f'{count} rotated/scaled complex-spectrum comparisons: max difference {maximum:.3g}; tolerance 3e-9')
    print('36 real-spectrum/Jordan checks and critical singularity check PASS')


def main():
    theta, w, critical = constants()
    rational_witness()
    matrix_checks(theta, w, critical)
    print('PASS')


if __name__ == '__main__':
    main()
