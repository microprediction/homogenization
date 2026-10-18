"""Switched-CIR transform envelopes; floating checks, not interval enclosures."""
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def laplace(a, kappa, xi, t, s, v0):
    d = np.sqrt(kappa*kappa+2*xi*xi*np.asarray(s))
    den = kappa+d+(d-kappa)*np.exp(-d*t)
    b = 2*s*(-np.expm1(-d*t))/den
    log_a = 2*a/xi**2*(np.log(2*d/den)+(kappa-d)*t/2)
    return np.exp(log_a-b*v0)


def moment_bound(a, kappa, xi, rho, t, r, v0):
    """Stable-root sufficient case of the positive Riccati supersolution."""
    zeta, source = max(xi)**2/2, r*(r-1)/2
    ell = max(r*rho*xi-kappa)
    assert r > 1 and ell < 0 and ell*ell > 4*zeta*source
    u, v = expm(t*np.array([[0., -zeta], [source, ell]]))@np.array([1., 0.])
    assert u > 0
    return np.exp(-max(a)/zeta*np.log(u)+v/u*v0)


def transform(z, q, speed, t, a, kappa, xi, rho, v0):
    """Exact affine ODE when only the immigration a_i switches."""
    z = np.atleast_1d(z).astype(complex)
    n, count = len(a), len(z)
    def rhs(_, state):
        b, amp = state[:count], state[count:].reshape(n, count)
        return np.r_[xi*xi*b*b/2+(rho*xi*z-kappa)*b+(z*z-z)/2,
                     (speed*q@amp+a[:, None]*b*amp).ravel()]
    sol = solve_ivp(rhs, (0., t), np.r_[np.zeros(count), np.ones(n*count)].astype(complex),
                    method='DOP853', rtol=2e-11, atol=2e-13)
    assert sol.success
    return sol.y[count:, -1].reshape(n, count)*np.exp(sol.y[:count, -1]*v0)


def main():
    a, kappa, xi = np.array([.25, .8, 1.1]), np.array([1.4, .5, 2.]), np.array([.6, 1., .8])
    rho = np.array([-.9, -.9, -.8])
    mr = moment_bound(a,kappa,xi,rho,1.,4.,.04)
    rng, path_error = np.random.default_rng(319), 0.
    for _ in range(16):
        regimes, durations = rng.integers(0, 3, 4), rng.dirichlet(np.ones(4))
        for s in (1., 25., 400.):
            state = np.zeros(2)
            for i, dt in zip(regimes[::-1], durations[::-1]):
                sol = solve_ivp(lambda _, y: [-a[i]*y[1], s-kappa[i]*y[1]-xi[i]**2*y[1]**2/2],
                                (0., dt), state, rtol=2e-11, atol=2e-13, method='DOP853')
                assert sol.success
                state = sol.y[:, -1]
            actual = np.exp(state[0]-state[1]*.04)
            bound = laplace(min(a), max(kappa), max(xi), 1., s, .04)
            path_error = max(path_error, actual-bound)
        for z in (5j, 2+5j):
            state = np.zeros(2,dtype=complex)
            for i, dt in zip(regimes[::-1],durations[::-1]):
                sol = solve_ivp(lambda _, y: [a[i]*y[1], xi[i]**2*y[1]**2/2
                    +(rho[i]*xi[i]*z-kappa[i])*y[1]+(z*z-z)/2], (0.,dt),state,
                    rtol=2e-11,atol=2e-13,method='DOP853')
                assert sol.success
                state = sol.y[:,-1]
            bound = laplace(min(a),max(kappa),max(xi),1.,min(1-rho*rho)*25/2,.04)
            if z.real:
                bound = math.sqrt(mr*laplace(min(a),max(kappa),max(xi),1.,min(1-rho*rho)*25,.04))
            assert abs(np.exp(state[0]+state[1]*.04)) <= bound+3e-9*max(1.,mr)
    assert path_error < 2e-9
    # A constant regime independently checks the closed Laplace Riccati formula.
    riccati_error = 0.
    for t in (.1, 1., 3.):
        for s in (.1, 1., 25., 400.):
            sol = solve_ivp(lambda _, y: [-.25*y[1], s-2*y[1]-y[1]**2/2],
                            (0., t), [0., 0.], rtol=2e-11, atol=2e-13, method='DOP853')
            riccati_error = max(riccati_error, abs(np.exp(sol.y[0,-1]-.04*sol.y[1,-1])
                                                  -laplace(.25, 2., 1., t, s, .04)))
    assert riccati_error < 2e-9
    print(f'48 heterogeneous Laplace and 32 leveraged path bounds; 12 scalar checks: {riccati_error:.3e}')
    q, a = np.array([[-1.7, 1.7], [.4, -.4]]), np.array([.2, .5])
    kappa, xi, rho, v0 = 1.4, .8, -.6, .04
    eta, frequencies, checked = 1-rho*rho, np.array([0., 1., 5., 20., 80.]), 0
    for t in (.25, 1., 3.):
        mr = moment_bound(a, np.full(2,kappa), np.full(2,xi), np.full(2,rho), t, 4., v0)
        # Independent ODE check of the positive Riccati bound.
        sol = solve_ivp(lambda _, y: [max(a)*y[1], xi*xi*y[1]**2/2+(4*rho*xi-kappa)*y[1]+6],
                        (0.,t), [0.,0.], rtol=2e-11, atol=2e-13, method='DOP853')
        assert abs(mr-np.exp(sol.y[0,-1]+sol.y[1,-1]*v0)) < 2e-8*max(1.,mr)
        for speed in (0., .5, 8., 64.):
            for real in (0., 2.):
                actual = abs(transform(real+1j*frequencies,q,speed,t,a,kappa,xi,rho,v0))
                bound = laplace(min(a),kappa,xi,t,eta*frequencies**2/2,v0)
                if real:
                    bound = np.sqrt(mr*laplace(min(a),kappa,xi,t,eta*frequencies**2,v0))
                assert np.all(actual <= bound+3e-9*max(1.,mr))
                checked += actual.size
    print(f'{checked} leveraged statewise real/damped transform bounds: PASS')
    t, pi = 1., np.array([.4,1.7])/2.1
    print(f'Envelope fixture: a={a}, kappa={kappa}, xi={xi}, rho={rho}, '
          f'v0={v0}, T=1, gamma=2, p=2, log-strikes=(-.2,0,.2)')
    constant = 2**(2*min(a)/xi**2)*math.exp(min(a)*kappa*t/xi**2)
    decay = min(a)*t*math.sqrt(eta)/xi
    mr = moment_bound(a,np.full(2,kappa),np.full(2,xi),np.full(2,rho),t,4.,v0)
    damp_constant, damp_decay = math.sqrt(mr*constant), decay/math.sqrt(2)
    def invert(upper, order):
        nodes, weights = leggauss(order)
        u, weights = (nodes+1)*upper/2, weights*upper/(2*math.pi)
        cf = pi@transform(1j*u,q,8.,t,a,kappa,xi,rho,v0)
        mgf = pi@transform(2+1j*u,q,8.,t,a,kappa,xi,rho,v0)
        phase = np.exp(-1j*np.outer(np.array([-.2,0.,.2]),u))
        return np.array([np.real(phase*cf)@weights,
            .5-np.imag(phase*cf)@(weights/u),
            np.exp(-np.array([-.2,0.,.2]))*(np.real(phase*mgf/((1+1j*u)*(2+1j*u)))@weights)])
    values, budgets, refinement = [], [], 0.
    for upper in (40., 80.):
        value, fine = invert(upper,96), invert(upper,192)
        refinement = max(refinement,np.max(abs(value-fine)))
        budget = np.array([constant*np.exp(-decay*upper)/(math.pi*decay),
                           constant*np.exp(-decay*upper)/(math.pi*decay*upper),
                           math.exp(.2)*damp_constant*np.exp(-damp_decay*upper)
                           /(math.pi*damp_decay*upper**2)])
        values.append(fine); budgets.append(budget)
        print(f'U={upper:.0f}: density/CDF/call tail budgets '+', '.join(f'{x:.9g}' for x in budget))
    assert refinement < 3e-8
    assert np.all(np.max(abs(values[0]-values[1]),axis=1) <= budgets[0]+budgets[1]+3e-8)
    print(f'Fourier quadrature refinement: {refinement:.3e}; real decay={decay:.9f}; damped decay={damp_decay:.9f}')
    # Perfect leverage: X_T=v_T-v0-T for a=xi=1, kappa=1/2, rho=1.
    u, h = np.array([1.,5.,20.,80.]), 1-math.exp(-.5)
    exact = np.exp(-1j*u*(v0+1))/(1-1j*h*u)**2*np.exp(1j*u*math.exp(-.5)*v0/(1-1j*h*u))
    actual = transform(1j*u,np.zeros((1,1)),0.,1.,np.array([1.]),.5,1.,1.,v0)[0]
    assert np.max(abs(actual-exact)) < 3e-9
    limit = math.exp(-math.exp(-.5)*v0/h)/h**2
    for frequency in (200.,800.,3200.):
        modulus = math.exp(-h*frequency**2*math.exp(-.5)*v0/(1+h*h*frequency**2))/(1+h*h*frequency**2)
        scaled = frequency**2*modulus
    assert abs(scaled/limit-1) < .001
    print(f'Perfect-correlation a=xi=rho=1, kappa=.5, T=1, v0={v0}, u=3200: '
          f'u^2|CF|={scaled:.9f}; positive limit {limit:.9f}')
    print('PASS')


if __name__ == '__main__':
    main()
