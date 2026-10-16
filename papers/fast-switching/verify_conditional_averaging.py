"""Fixed-start covariance, forward/backward clocks, and a finite-rate log bound."""
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm


def moments(lam, t, forcing, sign):
    q = lam*np.array([[-1., 1.], [1., -1.]])
    start = np.array([(1+sign)/2, (1-sign)/2])
    initial = np.r_[start, np.zeros(4), start]
    def rhs(s, v):
        p, u, w, z = v.reshape(4, 2)
        d = np.array([forcing(s), -forcing(s)])
        return np.r_[p@q, u@q+p*d, w@q+2*u*d, z@q+z*d]
    if t == 0:
        return 0., 0., 0.
    out = solve_ivp(rhs, (0., t), initial, method="DOP853", rtol=2e-12, atol=2e-13)
    assert out.success
    p, u, w, z = out.y[:, -1].reshape(4, 2)
    return u.sum(), w.sum()-u.sum()**2, np.log(z.sum())


# Independent conditional semigroup covariance and double-integral variance.
cov_error, variance_error, cases = 0., 0., 0
for lam in (.25, 1., 4., 16.):
    q = lam*np.array([[-1., 1.], [1., -1.]])
    x = np.array([1., -1.])
    for sign in (-1, 1):
        start = np.array([(1+sign)/2, (1-sign)/2])
        for s in (0., .2, 1.):
            for lag in (0., .3, 1.):
                p = start@expm(s*q)
                cov = p@(x*(expm(lag*q)@x))-(p@x)*(p@expm(lag*q)@x)
                target = np.exp(-2*lam*lag)*(-np.expm1(-4*lam*s))
                cov_error = max(cov_error, abs(cov-target))
                assert abs(cov-target) <= 256*np.finfo(float).eps
                cases += 1
        for t in (.01, .2, 1., 3.):
            a, layer = 2*lam, -np.expm1(-2*lam*t)
            stationary = 2*t/a-2*layer/a**2
            exact = stationary-layer**2/a**2
            integrated = quad(lambda s: 2*(-np.expm1(-a*(t-s)))/a
                              *(-np.expm1(-2*a*s)), 0., t, epsabs=1e-11)[0]
            mean, var, _ = moments(lam, t, lambda s: 1., sign)
            scale = max(1., t*t, abs(exact))
            assert abs(var-exact) <= 2e-10*scale and abs(integrated-exact) <= 2e-10*scale
            assert abs(mean-sign*layer/a) <= 2e-10*max(1., t)
            variance_error = max(variance_error, abs(var-exact)/scale)
print(f"Conditional covariance checks={cases}; max discrepancy={cov_error:.3e}")
print(f"Fixed-start variance ODE max scaled discrepancy={variance_error:.3e}")
layer = -np.expm1(-2.)
print(f"lambda=t=tilde_g=1: stationary variance={1-layer/2:.9f}, fixed-start variance={1-layer/2-layer**2/4:.9f}")
print(f"s=0, lag=lambda=1: conditional covariance=0, stationary covariance={np.exp(-2.):.9f}")

# Forward joint-state equations are independent of the Riccati proof of the bound.
functions = [(lambda s: 1., 1., 0.), (lambda s: s, 3., 1.),
             (lambda s: np.sin(1.7*s), 1., 1.7),
             (lambda s: np.exp(-s), 1., 1.), (lambda s: .7+.4*s, 1.9, .4)]
count, ratio = 0, 0.
for forcing, dmax, lip in functions:
    for lam in (.25, 1., 4., 16., 64.):
        a = 2*lam
        for t in (0., .01, .2, 1., 3.):
            variance_term = quad(lambda s: forcing(s)**2, 0., t)[0]/a
            memory = forcing(0)*(-np.expm1(-a*t))/a
            bound = (lip+dmax*lip*t+2*dmax**2)/a**2+2*dmax**4*t/a**3
            for sign in (-1, 1):
                _, _, actual = moments(lam, t, forcing, sign)
                error = abs(actual-variance_term-sign*memory)
                assert error <= bound+2e-10*max(1., abs(actual))
                ratio = max(ratio, error/bound)
                count += 1
print(f"Uniform finite-rate log-bound checks={count}; max error/bound={ratio:.9f}")

# Explicit ramp witness: forward weighting versus the maturity-clock ODE.
print("Ramp d(s)=s, t=1: lambda, lambda*(log a_plus-log a_minus), reversed-clock counterpart")
for lam in (20., 40., 80., 160.):
    forward = [moments(lam, 1., lambda s: s, sign)[2] for sign in (1, -1)]
    reversed_profile = [moments(lam, 1., lambda s: 1-s, sign)[2] for sign in (1, -1)]
    q = lam*np.array([[-1., 1.], [1., -1.]])
    ode = solve_ivp(lambda s, v: (q+np.diag([s, -s]))@v, (0., 1.), np.ones(2),
                    method="DOP853", rtol=2e-12, atol=2e-13)
    assert ode.success and np.max(abs(np.log(ode.y[:, -1])-reversed_profile)) < 2e-10
    fgap, bgap = lam*np.diff(forward[::-1])[0], lam*np.diff(reversed_profile[::-1])[0]
    assert abs(fgap) < 1/lam and abs(bgap-1) < 1/lam
    print(f"{lam:.0f} {fgap:.9f} {bgap:.9f}")
print("PASS")
