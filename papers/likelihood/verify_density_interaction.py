"""Compact-set density interaction bounds for switched Vasicek.

The theorem and exact-arithmetic discretization budgets are analytic.
Floating matrix products/RK4 are checks, not outward-rounded enclosures.
"""
import math
import numpy as np
from scipy.integrate import quad
from scipy.special import erfc
from verify_filtered_likelihood import matrix_density


def norm(a):
    return np.linalg.norm(a, np.inf)


def constants(kappa, theta, variance, pi, tau, T, lo, hi, A, gap, speed):
    vb = lambda t, s: s*(-math.expm1(-2*kappa*t))/(2*kappa)
    mean, smin = pi@theta, min(variance)
    beta = vb(tau, smin)/2
    J = np.array([math.gamma((j+1)/2)/(2*math.pi*beta**((j+1)/2)) for j in range(5)])
    a, b = kappa*max(abs(theta-mean)), (max(variance)-smin)/2
    H1, H2 = a*J[1]+b*J[2], a*a*J[2]+2*a*b*J[3]+b*b*J[4]
    B = max(hi-min(lo, mean), max(hi, mean)-lo)
    pstar = math.exp(-B*B/(2*vb(tau, pi@variance)))/math.sqrt(2*math.pi*vb(T, pi@variance))
    alpha, z = speed*gap, math.exp(-speed*gap*tau)
    e = A*J[0]*z+A*(2*H1+T*H2)/alpha
    u, bnd = A*H1/alpha, A*(J[0]+A*T*H1)*z+A*A*H2/alpha**2
    delta = e/pstar
    Gamma = (bnd+u*u/(pstar*(1-delta)))/pstar if delta < 1 else math.inf
    m0 = max(1., 8*A*(2*H1+T*H2)/(gap*pstar),
             max(0., math.log(8*A*J[0]/pstar))/(gap*tau))
    return pstar, e, u, bnd, delta, Gamma, m0


def frozen_density(xp, x, speed, t, steps=4096, n=300):
    kappa, theta, variance = .5, np.array([.08, .02]), np.array([.0009, .0001])
    smin, mean, dt = min(variance), np.mean(theta), t/steps
    vmin = smin*(-math.expm1(-2*kappa*t))/(2*kappa)
    vbar = np.mean(variance)*(-math.expm1(-2*kappa*t))/(2*kappa)
    F, h = 13/math.sqrt(vbar), 13/math.sqrt(vbar)/n
    w = np.arange(-n, n+1)*h
    Z = np.broadcast_to(np.eye(2), (len(w), 2, 2)).astype(complex).copy()
    for j in range(steps):
        d = math.exp(-kappa*(j+.5)*dt)
        g = 1j*w[:, None]*kappa*(theta-mean)*d-.5*w[:, None]**2*(variance-smin)*d*d
        c, q = g.mean(axis=1)-speed, (g[:, 0]-g[:, 1])/2
        root = np.sqrt(speed*speed+q*q)
        ep, em = np.exp(dt*(c+root)), np.exp(dt*(c-root))
        ch, sh = (ep+em)/2, (ep-em)/(2*root)
        E = np.empty_like(Z)
        E[:, 0, 0], E[:, 1, 1] = ch+q*sh, ch-q*sh
        E[:, 0, 1] = E[:, 1, 0] = speed*sh
        Z = E@Z
    gaussian = np.exp(-vmin*w*w/2)
    phase = np.exp(1j*w*(math.exp(-kappa*t)*x+mean*(1-math.exp(-kappa*t))-xp))
    K = (h/(2*math.pi)*np.sum((gaussian*phase)[:, None, None]*Z, axis=0)).real
    a, b = kappa*max(abs(theta-mean)), (max(variance)-smin)/2
    time = t*t/(4*steps)*h/(2*math.pi)*np.sum(gaussian*(kappa*a*abs(w)+2*kappa*b*w*w))
    tail = erfc(F*math.sqrt(vmin/2))/math.sqrt(2*math.pi*vmin)
    B = max(abs(x-xp), abs(theta[0]-xp), abs(theta[1]-xp))
    period, vmax = 2*math.pi/h, max(variance)*vbar/np.mean(variance)
    assert period > B
    alias = 2/math.sqrt(2*math.pi*vmin)*math.exp(max(-700., -(period-B)**2/(2*vmax)))
    alias /= -math.expm1(-period*(period-B)/vmax)
    return K, time+tail+alias


def main():
    theta, variance, pi = np.array([.08, .02]), np.array([.0009, .0001]), np.array([.5, .5])
    P, C = np.ones((2, 2))/2, np.eye(2)-np.ones((2, 2))/2
    path, t = np.array([.05, .0555, .047, .061, .052, .0445, .05]), .25
    history, count = [], 0
    for speed in (40., 80., 160., 320.):
        pstar, e, u, bnd, delta, Gamma, m0 = constants(.5, theta, variance, pi, .25, .5, .044, .062, 1., 2., speed)
        actual = []
        for x, xp in zip(path[:-1], path[1:]):
            K = matrix_density(xp, x, speed*np.array([[-1., 1.], [1., -1.]]), .5, theta, variance, t)
            v = (pi@variance)*(-math.expm1(-t))
            mu = math.exp(-t/2)*x+(pi@theta)*(1-math.exp(-t/2))
            p = math.exp(-(xp-mu)**2/(2*v))/math.sqrt(2*math.pi*v)
            D = K-np.outer(K.sum(axis=1), pi@K)/(pi@K).sum()
            assert p >= pstar
            assert norm(K-p*P) <= e+2e-9
            assert norm(C@K@P) <= u+2e-9 and norm(P@K@C) <= u+2e-9
            assert norm(C@K@C) <= bnd+2e-9
            assert norm(D/p) <= (bnd+u*u/(pi@K).sum())/p+2e-9
            actual.append(norm(D/p))
            count += 1
        history.append(max(actual))
        print(f'speed={speed:.0f}: max relative interaction={history[-1]:.6e}')
    order = math.log2(history[-2]/history[-1])
    assert order > 1.8
    print(f'{count} density/block budgets: PASS; last interaction order={order:.6f}')
    th3, v3, pi3 = np.array([.08, .02, .05]), np.array([.0009, .0001, .0006]), np.ones(3)/3
    q3 = np.roll(np.eye(3), 1, axis=1)-np.eye(3)
    K3 = matrix_density(.0555, .05, 160*q3, .5, th3, v3, t, steps=1000)
    C3 = np.eye(3)-np.ones((3, 3))/3
    c3 = constants(.5, th3, v3, pi3, .25, .5, .044, .062, math.sqrt(3), 1.5, 160)
    assert norm(C3@K3@C3) <= c3[3]+2e-9
    print('Nonreversible three-cycle density bound: PASS')
    for j in range(5):
        value = quad(lambda z: z**j*math.exp(-z*z), 0, math.inf)[0]
        assert abs(value-math.gamma((j+1)/2)/2) < 2e-10
    print('Five Gaussian moment formulas: PASS')
    pstar, _, _, _, delta, Gamma, m0 = constants(.5, theta, variance, pi, .25, .5, .044, .062, 1., 2., m0)
    assert delta <= .25+2e-12
    print(f'Compact-set pstar={pstar:.6f}; sufficient speed m0={m0:.6f}')
    print(f'At m0: delta={delta:.6f}, interaction budget={Gamma:.6e}')
    x, xp = .05, .0555
    frozen, budget = frozen_density(xp, x, 160., t)
    refined, smaller = frozen_density(xp, x, 160., t, steps=8192, n=400)
    rk = matrix_density(xp, x, 160*np.array([[-1., 1.], [1., -1.]]), .5, theta, variance, t, nfreq=901, steps=1200)
    assert norm(frozen-refined) <= budget+smaller+2e-9
    assert norm(refined-rk) <= smaller+2e-9
    assert smaller < budget
    print(f'Independent frozen/RK density gap={norm(refined-rk):.6e}; analytic discretization budget={smaller:.6e}')
    print('PASS')


if __name__ == '__main__':
    main()
