"""Finite-maturity stationary CIR speed classification; no simulation.

Positive scalar quadratures are compared with a polynomial semigroup and
its independent Frechet derivative, including the changing invariant law.
Floating-point checks support the exact proof, not interval certification.
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm_frechet
from scipy.optimize import brentq

from verify_cir_speed import close, formulas
from verify_switching_kappa import (
    full_generator, integrated_variance_operators, stationary_cir_moments,
)


def kernel(kappa, t):
    z = kappa * t
    if z < 0.01:
        return t*t * sum(2 * (-z)**n / math.factorial(n+2) for n in range(9))
    return 2 * (z + math.expm1(-z)) / kappa**2


def profiles(kappa, r, t):
    f = kernel(kappa, t)
    h = quad(lambda s: kernel(kappa, t-s) * math.exp(-r*s),
             0, t, epsabs=1e-14, epsrel=1e-11)[0]
    j = quad(lambda s: kernel(kappa, t-s) * (1+(kappa+r)*s) * math.exp(-r*s),
             0, t, epsabs=1e-14, epsrel=1e-11)[0]
    return f, h, j / f


def semigroup_derivative(alpha, beta, kappa, theta, a, r, t):
    """Differentiate in r, not m; include invariant first/second moments."""
    pi = np.array([beta, alpha]) / (alpha+beta)
    q = np.array([[-alpha, alpha], [beta, -beta]]) / (alpha+beta)
    c, kappas = kappa*theta, np.full(2, kappa)
    first, second = stationary_cir_moments(r, q, pi, c, kappas, a)
    dfirst = np.linalg.solve((r*q-kappa*np.eye(2)).T, -q.T@first)
    dsecond = np.linalg.solve((r*q-2*kappa*np.eye(2)).T,
                             -(2*c+a)*dfirst-q.T@second)
    basis, dc, dk, dvv, dz = integrated_variance_operators(2)
    ops = [dc, dk, dvv/2]
    avg = (pi@c)*dc + kappa*dk + (pi@a)*dvv/2 + dz
    generator = full_generator(r*q, avg, ops, [c, kappas, a])
    initial, dinitial = np.zeros(2*len(basis)), np.zeros(2*len(basis))
    for state in range(2):
        for j, (v, z) in enumerate(basis):
            if z == 0:
                initial[state*len(basis)+j] = (pi[state], first[state], second[state])[v]
                dinitial[state*len(basis)+j] = (0, dfirst[state], dsecond[state])[v]
    payoff = np.zeros(len(basis))
    payoff[basis.index((0, 2))] = 1
    payoff = np.tile(payoff, 2)
    exp, dexp = expm_frechet(t*generator, t*np.kron(q, np.eye(len(basis))))
    raw = initial @ exp @ payoff
    # E[V_T]=bar(theta)*T is independent of r in the joint invariant law.
    return raw-(pi@theta*t)**2, dinitial@exp@payoff+initial@dexp@payoff, raw


def verify():
    rng = np.random.default_rng(20261016)
    models = []
    theta = np.array([0.03, 0.04])
    for a in ([0.01, 0.01], [0.03, 0.01], [0.05, 0.001]):
        models.append((1, 1, 1, theta, np.array(a)))
    models.append((1, 1, 1, np.full(2, 0.04), np.array([0.05, 0.001])))
    for alpha, beta, kappa in ((0.4, 1.7, 0.6), (2, 0.3, 1.8)):
        models.append((alpha, beta, kappa, theta, kappa*np.array([0.05, 0.001])))
    for _ in range(12):
        alpha, beta, kappa = rng.uniform(0.3, 2, 3)
        levels = rng.uniform(0.02, 0.15, 2)
        a = 2*kappa*levels*rng.uniform(0.05, 0.95, 2)
        models.append((alpha, beta, kappa, levels, a))
    checks = 0
    for alpha, beta, kappa, levels, a in models:
        pi = np.array([beta, alpha])/(alpha+beta)
        base = formulas(alpha, beta, kappa, levels, a, 1)
        vt, _, amplitude = base[4:]
        vinf = (pi@a)*(pi@levels)/(2*kappa)
        for t in (0.1, 1, 4, 16):
            previous = math.inf
            for r in (0.1, kappa, 2.0):
                f, h, profile = profiles(kappa, r, t)
                value = vinf*f + (amplitude*f+kappa**2*vt*h)/(kappa+r)
                derivative = -f*(amplitude+kappa**2*vt*profile)/(kappa+r)**2
                actual, dactual, scale = semigroup_derivative(
                    alpha, beta, kappa, levels, a, r, t)
                close(actual, value, scale)
                close(dactual, derivative, scale)
                assert profile < previous
                previous = profile
                if vt == 0:
                    close(derivative, 0)
                elif amplitude >= -1e-14:
                    assert derivative < 0
                if r > 0:
                    limit = (kappa+2*r)/r**2
                    bound = 2*(2*kappa+3*r)/(kappa*f*r**3)
                    assert -1e-14 <= limit-profile <= bound+1e-14
                checks += 1
            close(profiles(kappa, 0, t)[2], t**3/(3*f), t**3/(3*f))
    print(f"{checks} value/Frechet-derivative comparisons and profile enclosures")

    # Benchmark: negative A; classification boundary and minimum-speed curve.
    d = 1.45
    tc = brentq(lambda t: t**3/(3*kernel(1, t))-d, 0.01, 10)
    close(tc**3/(3*kernel(1, tc)), d)
    rinf = (1+math.sqrt(1+d))/d
    print(f"Benchmark D={d:.2f}; critical maturity Tc={tc:.10f}")
    previous = 0
    for t in (3, 4, 8, 16, 64):
        root = brentq(lambda r: profiles(1, r, t)[2]-d, 0, rinf)
        assert previous < root < rinf
        assert profiles(1, root/2, t)[2] > d
        assert profiles(1, (root+rinf)/2, t)[2] < d
        bound = rinf**3*(2+3*root)/(kernel(1, t)*(1+rinf)*root**3)
        assert rinf-root <= bound+1e-14
        previous = root
        print(f"T={t}: minimizing m={root/2:.10f}")
    for t in (1, 2, tc*(1-1e-6), tc):
        assert profiles(1, 0.001, t)[2] < d
    assert profiles(1, 0, tc*(1+1e-6))[2] > d
    for r in (0.1, 1, rinf):
        row = [profiles(1, r, t)[2] for t in (1, 2, 4, 8, 16)]
        assert np.all(np.diff(row) > 0)
    print("Five minimum-speed roots/enclosures; threshold sides and maturity monotonicity")
    print("PASS")


if __name__ == "__main__":
    verify()
