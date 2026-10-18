"""A capped auxiliary Poisson counter identifies Gamma clocks and finite generators."""
import math

import mpmath as mp
import numpy as np
import sympy as sp
from scipy.integrate import quad_vec
from scipy.linalg import expm, fractional_matrix_power
from scipy.optimize import brentq
from scipy.special import gammaln


def check(condition):
    if not condition:
        raise AssertionError("Poisson-Gamma tag certificate failed")


def norm(a):
    return np.linalg.norm(a, np.inf)


def clock(p0, p1, tick_rate):
    x = p1/p0
    z = -math.log(p0)/x
    check(x > 0 and z > 1)
    def equation(y):
        return (1 if y == 0 else np.inf if y == 1 else -np.log1p(-y)/y)-z
    y = brentq(equation, 0, 1, xtol=1e-13)
    return x/y, tick_rate*(1-y)/y


def conditional_integral(q, shape, rate):
    # t=u^2 removes the endpoint singularity for the smallest tested shape, 1/2.
    def integrand(u):
        if u == 0:
            return np.zeros_like(q)
        log_weight = math.log(2)+shape*math.log(rate)-gammaln(shape)
        log_weight += (2*shape-1)*math.log(u)-rate*u*u
        return np.zeros_like(q) if log_weight < -740 else math.exp(log_weight)*expm(u*u*q)
    return quad_vec(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-11)[0]


def main():
    q = sp.Matrix([[-1, 1, 0], [0, -1, 1], [0, 0, 0]])
    c = 2*sp.eye(3)-q
    m0, m1 = c**-2, 2*c**-3
    check(c*m1 == 2*m0 and 2*sp.eye(3)-2*m0*m1.inv() == q)
    check(m0*sp.ones(3, 1) == sp.ones(3, 1)/4)
    check(m1*sp.ones(3, 1) == sp.ones(3, 1)/4)
    check(4*(2*c)**-2 == m0 and 16*(2*c)**-3 == m1)
    print("Exact rational defective/absorbing inverse, p0=p1=1/4, and unknown-tick-rate scaling: PASS")

    clock_error = 0.
    for shape in (.125, .5, 1., 2.5, 7., 25., 100.):
        for rate in (.1, 1., 10.):
            p0 = (rate/(rate+1))**shape
            fitted_shape, fitted_rate = clock(p0, shape/(rate+1)*p0, 1.)
            clock_error = max(clock_error, abs(fitted_shape-shape)/(1+shape),
                              abs(fitted_rate-rate)/(1+rate))
    check(clock_error < 2e-8)
    print(f"21 real-shape clock fits (shapes 1/8 through 100): max scaled error {clock_error:.3e}")

    nilpotent = np.outer([1, -1, 0, 0], [0, 0, 1, -1])/16
    cases = [np.array([[-1.2, .9, .3], [.2, -.6, .4], [.7, .1, -.8]]),
             np.ones((4, 4))/4-np.eye(4)+nilpotent,
             np.array([[-1., 1., 0.], [0., -1., 1.], [0., 0., 0.]])]
    recovery = quadrature = parameter = 0.
    for q in cases:
        eye = np.eye(len(q))
        for shape in (.5, 1., 2.5, 7.):
            for rate in (.7, 2.3):
                for tick in (.4, 1.5):
                    c = (rate+tick)*eye-q
                    p0 = (rate/(rate+tick))**shape
                    p1 = shape*tick/(rate+tick)*p0
                    observed = []
                    for j, mass in enumerate((p0, p1)):
                        direct = conditional_integral(q, shape+j, rate+tick)
                        target = (rate+tick)**(shape+j)*fractional_matrix_power(c, -shape-j)
                        quadrature = max(quadrature, norm(direct-target))
                        observed.append(mass*direct)
                    m0, m1 = observed
                    alpha, lam = clock(sum(m0[0]), sum(m1[0]), tick)
                    parameter = max(parameter, abs(alpha-shape)/(1+shape), abs(lam-rate)/(1+rate))
                    recovered = (lam+tick)*eye-alpha*tick*np.linalg.solve(m1.T, m0.T).T
                    recovery = max(recovery, norm(recovered-q)/(1+norm(q)))
    check(max(recovery, quadrature, parameter) < 2e-8)
    print(f"48 joint shape/rate/Q recoveries (directed, defective, absorbing): max scaled error {recovery:.3e}, clock error {parameter:.3e}")
    print(f"96 independent normalized tag integrals: max matrix discrepancy {quadrature:.3e}")

    mp.mp.dps = 80
    p, eye = mp.ones(3)/3, mp.eye(3)
    e = eye-p
    j = mp.matrix([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])/mp.sqrt(3)
    theta = mp.pi/7
    a, b = 10*mp.cos(theta)-1, 10*mp.sin(theta)
    generators = [-a*e+b*j, -a*e-b*j]
    minimum_rate = min(q[i, k] for q in generators for i in range(3) for k in range(3) if i != k)
    check(minimum_rate > 0)
    kernels = [(eye-q)**-7 for q in generators]
    tags = [(2*eye-q)**-7 for q in generators]
    mn = lambda v: max(sum(abs(v[i, k]) for k in range(v.cols)) for i in range(v.rows))
    alias_error = max(mn(k-p+mp.mpf('1e-7')*e) for k in kernels)
    m0, m1 = tags[0], 7*(2*eye-generators[0])**-8
    alpha, lam = clock(float(sum(m0[0, k] for k in range(3))),
                       float(sum(m1[0, k] for k in range(3))), 1.)
    recovery = mn((lam+1)*eye-alpha*m0*m1**-1-generators[0])/(1+mn(generators[0]))
    check(max(alias_error, recovery) < mp.mpf('2e-8'))
    print(f"Shape-7 discarded-time alias discrepancy {float(alias_error):.3e}; minimum off-diagonal rate {float(minimum_rate):.12f}; zero-tick matrix gap {float(mn(tags[0]-tags[1])):.12e}")
    print(f"Ternary-tag alias recovery error {float(recovery):.3e}; log10 one-tick infinity-condition number {float(mp.log10(mn(m1)*mn(m1**-1))):.3f}")

    p, eye = mp.ones(2)/2, mp.eye(2)
    tv_error = mp.mpf(0)
    for shape in (mp.mpf('.5'), mp.mpf(1), mp.mpf('2.5'), mp.mpf(7)):
        for speed in (1, 2, 4, 8, 16):
            laws = []
            masses = [mp.mpf(2)**-shape, shape*mp.mpf(2)**(-shape-1)]
            masses += [1-sum(masses)]
            for c in (speed, 2*speed):
                weights = [mp.mpf(c+2)**-shape, shape*mp.mpf(c+2)**(-shape-1)]
                weights += [mp.mpf(c+1)**-shape-sum(weights)]
                check(min(weights) >= 0)
                laws.append([mass*p+v*(eye-p) for mass, v in zip(masses, weights)])
            tv = sum(abs(x-y) for h0, h1 in zip(*laws) for x, y in zip(h0, h1))/4
            target = (mp.mpf(speed+1)**-shape-mp.mpf(2*speed+1)**-shape)/2
            tv_error = max(tv_error, abs(tv-target))
    check(tv_error < mp.mpf('2e-13'))
    print(f"20 fixed-clock reset TV identities: max discrepancy {float(tv_error):.3e}; generator gap equals speed")
    print("PASS")


if __name__ == "__main__":
    main()
