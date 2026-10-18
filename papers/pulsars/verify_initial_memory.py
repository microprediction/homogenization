"""Sharp initial-law memory after continuous and sampled polynomial timing fits.

Run with numpy, scipy, sympy and mpmath. Exact rational identities support the
proof; high-precision and independent quadrature checks are not interval proofs.
"""
import math
import mpmath as mp
import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss
from pulsars import TwoState

mp.mp.dps = 70
s, t = sp.symbols('s t', nonnegative=True)


def exact_polynomials(degree):
    # Orthonormal shifted Legendre basis on [0,1]; retain rational products.
    phi = [sp.legendre(n, 2*t-1) for n in range(degree+1)]
    b = [sp.integrate((s-t)*p, (t, 0, s)) for p in phi]
    kernel = s*t**2/2-t**3/6-sum(
        (2*n+1)*b[n]*b[n].subs(s, t) for n in range(degree+1))
    poly = sp.Poly(sp.integrate(2*kernel, (t, 0, s)), s)
    assert all(poly.nth(j) == 0 for j in range(4))
    assert poly.nth(4) == sp.Rational(1, 4)
    assert poly.nth(5) == -sp.Rational((degree+1)**2, 6)
    return phi, sp.Poly(poly, s), sp.lambdify((s, t), kernel, 'numpy')


def integral_poly(poly, variable, x):
    return sum(mp.mpf(str(c))*mp.gammainc(j+1, 0, x)/x**(j+1)
               for (j,), c in sp.Poly(poly, variable).terms())


def continuous():
    nodes, weights = leggauss(100)
    v, wv = (nodes+1)/2, weights/2
    worst_quad = worst_bound = worst_next = 0.
    for degree in range(1, 7):
        phi, poly, kernel = exact_polynomials(degree)
        for x0 in (1, 4, 16, 64, 256, 1024, 16384):
            x = mp.mpf(x0)
            B = integral_poly(poly.as_expr(), s, x)
            N = -mp.expm1(-2*x)/(2*x)-sum(
                (2*n+1)*integral_poly(p, t, x)**2 for n, p in enumerate(phi))
            a = 6*mp.exp(-x)*sum(x**j/mp.factorial(j) for j in range(5))
            a += mp.mpf(20)*(degree+1)**2/x
            b = mp.exp(-2*x)/2+mp.mpf((degree+1)**2)/x
            assert -mp.mpf('1e-12') <= 6-x**5*B <= a+mp.mpf('1e-12')
            assert -mp.mpf('1e-12') <= mp.mpf('.5')-x*N <= b+mp.mpf('1e-12')
            # Independent triangular impulse-kernel quadrature, scaled at entry.
            cap = min(x0, 100)
            u, wu = (nodes+1)*cap/2, weights*cap/2
            vals = kernel(u[:, None]/x0, u[:, None]*v[None, :]/x0)
            approx = 2*x0**3*np.sum(wu*u*np.exp(-u)*(vals @ wv))
            # |<Rh_s,Rh_t>| <= (st)^(3/2)/3 bounds the omitted triangle.
            tail = 6.4*math.exp(-cap)*sum(cap**j/math.factorial(j) for j in range(5))
            err = abs(float(x**5*B)-approx)
            assert err < 2e-8+tail
            worst_quad = max(worst_quad, err)
            worst_bound = max(worst_bound, float((6-x**5*B)/a))
            if x0 == 16384:
                next_ratio = (6-x**5*B)*x/(20*(degree+1)**2)
                assert abs(next_ratio-1) < mp.mpf('.02')
                worst_next = max(worst_next, abs(float(next_ratio)-1))
        if degree in (1, 2, 6):
            print(f'degree={degree}: x=16384, x^5 B={float(x**5*B):.9f}, '
                  f'x N={float(x*N):.9f}')
    print('Exact degrees 1..6: leading B coefficients 1/4 and -(degree+1)^2/6')
    print(f'42 continuous bounds: maximum deficit/bound={worst_bound:.6f}; '
          f'quadrature error={worst_quad:.3e}; next-coefficient error={worst_next:.3e}')


def sampled():
    times = np.arange(4)/3
    v = np.array([-1., 3., -3., 1.])
    R = np.outer(v, v)/20
    assert np.max(abs(R @ np.column_stack([np.ones(4), times, times**2]))) < 1e-13
    A = np.sum(R[:, 0]**2)/4
    assert abs(A-1/80) < 1e-13
    nodes, weights = leggauss(100)
    worst = 0.
    for x in (8, 16, 32, 64, 128, 256):
        # Integral of 2 <R k_s, integral_0^s R k_u du> exp(-x s).
        B = 0.
        for lo, hi in zip(times[:-1], times[1:]):
            z, wz = lo+(nodes+1)*(hi-lo)/2, weights*(hi-lo)/2
            k = np.maximum(times[None, :]-z[:, None], 0)
            integ = (times[None, :]**2-k**2)/2
            B += np.sum(wz*np.exp(-x*z)*np.sum((k @ R)*(integ @ R), axis=1)/2)
        N = np.sum((R @ np.exp(-x*times))**2)/4
        if x == 256:
            assert abs(x**4*B-6*A) < 1e-12
            assert abs(N-A) < 1e-12
            print(f'Fixed four-point quadratic grid: x^4 B={x**4*B:.9f}, N={N:.9f}')
        if x <= 32:
            m = TwoState(.8, -.2, .8*x, .2*x)
            stationary = m.postfit_ms_sampled(times)
            for q in (0., .2, .5, 1.):
                eta, delta = q-.2, .6
                raw = m.postfit_ms_sampled(times, initial_p1=q)
                var = m.postfit_ms_sampled(times, initial_p1=q, centered=True)
                discrepancy = max(abs(raw-stationary-eta*delta*B),
                                  abs(raw-var-eta**2*N/x**4))
                assert discrepancy < 2e-13
                worst = max(worst, discrepancy)
    print(f'12 sampled raw/variance/mean decompositions: maximum error={worst:.3e}')
    # Exact rational endpoint leverage and the scaled triangular moment.
    vr = sp.Matrix([-1, 3, -3, 1])
    assert (vr*vr.T/20)[0, 0]/4 == sp.Rational(1, 80)
    assert sp.integrate(2*s*sp.integrate(t, (t, 0, s))*sp.exp(-s), (s, 0, sp.oo)) == 6


if __name__ == '__main__':
    continuous()
    sampled()
    print('PASS')
