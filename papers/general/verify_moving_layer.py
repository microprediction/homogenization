"""Moving stationary projection: exact majorants and independent ODE checks."""
from fractions import Fraction as F
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def add(a, b):
    return [sum((v[i] if i < len(v) else F(0) for v in (a, b)), F(0))
            for i in range(max(len(a), len(b)))]


def scale(a, x):
    return [x*v for v in a]


def mul(a, b):
    c = [F(0)]*(len(a)+len(b)-1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i+j] += x*y
    return c


def deriv(a):
    return [i*a[i] for i in range(1, len(a))]


def zero(a):
    assert all(x == 0 for x in a)  # Exact rationals, not a floating tolerance.


def exp_upper(x, n=12):
    term, total = F(1), F(1)
    for k in range(1, n+1):
        term *= x/k
        total += term
    return total + term*x/(n+1)/(1-x/(n+2))


def rational_majorants():
    p, b = [F(1, 4), F(1, 4)], [F(0), F(7, 32), F(1, 32), -F(1, 96)]
    h = add(add(b, [F(1, 2)]), scale(p, -1))
    f = [add([F(1)], scale(p, -1)), scale(p, -1)]
    zero(add(deriv(h), [F(1, 32), -F(1, 16), F(1, 32)]))
    zero(add(deriv(b), scale(add(mul(p, add([F(1)], scale(p, -1))), [F(1, 4)]), -F(1, 2))))
    residuals = []
    for gi, fi in zip((1, 0), f):
        v = add(b, scale(fi, F(1, 2)))
        residual = add(deriv(v), mul(add(p, [-F(gi)]), v))
        zero(add(residual, mul(h, fi)))
        residuals.append(residual)
    zero(add(mul(p, residuals[0]), mul(add([F(1)], scale(p, -1)), residuals[1])))
    assert h[0] == F(1, 4) and sum(h) == F(23, 96)
    assert exp_upper(F(3, 8)) < F(3, 2) and exp_upper(F(1)) < 3
    z, zp = F(9, 64), F(69, 512)
    bracket = 2*z+zp+z+F(3, 2)*F(3, 8)*(F(1, 4)+F(1, 2))
    assert bracket == F(501, 512) and 3*bracket == F(1503, 512) < 3
    print("Exact polynomial identities and exponential majorants: PASS")
    print("p in [1/4,1/2]; h in [23/96,1/4]; h'=-(1-t)^2/32")
    print("H=L_Q=1, C0=3/2, gamma0=2, ||w||=3/8, Z=9/64, Z1=69/512")
    print(f"Bracket=501/512; uniform coefficient=1503/512={float(3*bracket):.9f} < 3")


def two_state():
    errors, outer_errors = [], []
    for eps in 2.**-np.arange(1, 10):
        def rhs(t, u):
            p = .25+.25*t
            q = 2*(np.outer(np.ones(2), [p, 1-p])-np.eye(2))
            return q @ u/eps + np.array([u[0], 0.])
        nodes = np.unique(np.r_[np.linspace(0, 1, 1001), eps*np.linspace(0, min(12, 1/eps), 501)])
        sol = solve_ivp(rhs, (0, 1), np.ones(2), method='DOP853', rtol=2e-12, atol=2e-13, dense_output=True)
        assert sol.success
        p = .25+.25*nodes
        f = np.array([1-p, -p])
        a = np.exp(nodes/4+nodes**2/8)
        b = 7*nodes/32+nodes**2/32-nodes**3/96
        outer = a*(1+eps*(b+f/2))
        app = outer+eps*np.exp(-2*nodes/eps)*np.array([[-3/8], [1/8]])
        exact = sol.sol(nodes)
        err = np.max(np.abs(exact-app))
        assert err <= float(F(1503, 512))*eps**2+3e-10
        assert np.max(np.abs(app[:, 0]-1)) < 2e-12
        assert abs(np.max(np.abs(outer[:, 0]-1))/eps-3/8) < 2e-12
        errors.append(err)
        outer_errors.append(np.max(np.abs(exact-outer)))
    implicit = solve_ivp(rhs, (0, 1), np.ones(2), method='Radau', rtol=2e-12, atol=2e-13, dense_output=True)
    assert implicit.success and np.max(np.abs(implicit.sol(nodes)-exact)) < 3e-10
    rates = np.log2(np.array(errors[:-1])/errors[1:])
    assert 1.8 < rates[-1] < 2.2
    print(f"Two-state eps=1/512: sup sampled error={errors[-1]:.12g}, error/eps^2={errors[-1]*512**2:.9f}")
    print(f"Composite last-doubling order={rates[-1]:.6f}; outer initial error/eps=0.375")


def nonreversible_control():
    q0 = np.array([[-1.6, 1., .6], [.5, -1.7, 1.2], [.9, .7, -1.6]])
    dq = np.array([[-.1, .1, 0.], [0., -.1, .1], [-.1, 0., .1]])
    g, one = np.array([.3, -.2, .55]), np.ones(3)
    def data(t):
        q = q0+t*dq
        system = q.T.copy()
        system[-1] = 1
        pi = np.linalg.solve(system, [0., 0., 1.])
        proj = np.outer(one, pi)
        r = np.linalg.inv(q-proj)+proj
        dpi = -pi @ dq @ r
        dr = -r @ dq @ r+proj @ dq @ r @ r
        f, w = g-pi @ g, r @ g
        bp = -pi @ (f*w)-dpi @ w
        return q, pi, r, f, w, dr @ g, bp
    w0 = data(0)[4]
    errors, center = [], 0.
    for eps in 2.**-np.arange(3, 8):
        def rhs(t, y):
            q, pi, _, _, _, _, bp = data(t)
            return np.r_[q @ y[:3]/eps+g*y[:3], (pi @ g)*y[3], bp]
        nodes = np.unique(np.r_[np.linspace(0, 1, 301), eps*np.linspace(0, 8, 101)])
        nodes = nodes[nodes <= 1]
        sol = solve_ivp(rhs, (0, 1), np.r_[one, 1., 0.], method='DOP853', rtol=2e-12, atol=2e-13, dense_output=True)
        assert sol.success
        ys = sol.sol(nodes)
        app = []
        for j, t in enumerate(nodes):
            _, pi, _, f, w, wp, bp = data(t)
            a, b = ys[3:, j]
            residual = -b*f+bp-(pi @ g)*w-wp+g*w
            center = max(center, abs(pi @ residual))
            app.append(a*(one+eps*(b-w))+eps*expm(q0*t/eps) @ w0)
        errors.append(np.max(np.abs(np.array(app).T-ys[:3])))
    assert center < 2e-11
    rate = math.log(errors[-2]/errors[-1], 2)
    assert 1.8 < rate < 2.2
    print(f"Nonreversible moving three-state control: centered residual={center:.3g}, order={rate:.6f}")


if __name__ == '__main__':
    rational_majorants()
    two_state()
    nonreversible_control()
    print('PASS')
