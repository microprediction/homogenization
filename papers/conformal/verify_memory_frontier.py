"""Exact exponential-kernel variance frontier under a positive memory budget."""
import math

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import brentq, minimize
from scipy.special import roots_legendre


def optimum(horizon, budget):
    """Return cutoff, density, endpoint masses, dual multipliers and energy."""
    r, ell = horizon, budget
    if r <= 0 or not math.exp(-r) <= ell <= 1:
        raise ValueError("require positive horizon and exp(-horizon) <= budget <= 1")
    if ell >= 2/(r+2):
        c = 1/(r+2)
        return 0., c, c, c, 2*c, 0., 2*c
    if ell >= 1/(r+1):
        c = (1-ell)/r
        return 0., c, ell-c, c, 2*c, 2*c-ell, ell**2+2*(1-ell)**2/r
    a = brentq(lambda x: -x-math.log1p(r-x)-math.log(ell), 0., r, xtol=1e-13)
    c = 1/(r-a+1)
    return a, c, 0., c, 2*c, c*math.exp(a), (2*(r-a)+1)*c*c


def potential(x, r, a, c, left, right):
    if x < a:
        bulk = math.exp(x-a)*(-math.expm1(-(r-a)))
    else:
        bulk = 2-math.exp(-(x-a))-math.exp(-(r-x))
    return c*bulk + left*math.exp(-x) + right*math.exp(-(r-x))


def moments(r, a, c, left, right, initial, p=.3):
    """Independent six-dimensional regime-resolved forward moment semigroup."""
    q = np.array([[-p, p], [1-p, -(1-p)]])
    f = np.array([-p, 1-p])
    def generator(density):
        g = np.kron(np.eye(3), q.T)
        g[2:4, :2] = density*np.diag(f)
        g[4:6, 2:4] = 2*density*np.diag(f)
        return g
    jet = np.concatenate([initial, left*f*initial, left**2*f*f*initial])
    jet = expm((r-a)*generator(c)) @ (expm(a*generator(0.)) @ jet)
    second = jet[4:6] + 2*right*f*jet[2:4] + right**2*f*f*jet[:2]
    return (jet[2:4]+right*f*jet[:2]).sum(), second.sum()


def coverage(r, ell, order):
    a, c, left, right, *_ = optimum(r, ell)
    p, mu, sigma = .3, .8, .7
    f, q = np.array([-p, 1-p]), np.array([[-p, p], [1-p, -(1-p)]])
    nodes, weights = roots_legendre(order)
    k, weights = (nodes+1)*6/sigma, weights*6/sigma
    cf = np.array([np.exp(1j*t*mu*left*f)*(expm(a*q) @
                   expm((r-a)*(q+1j*t*mu*c*np.diag(f))) @ np.exp(1j*t*mu*right*f)) for t in k])
    def cdf(x):
        return .5-np.sum((weights*np.exp(-sigma*sigma*k*k/2)/k)[:, None]*
                         (np.exp(-1j*k*x)[:, None]*cf).imag, axis=0)/math.pi
    quantile = brentq(lambda x: np.array([1-p, p]) @ cdf(x)-.9, -5., 5.)
    gaps = cdf(quantile)-.9
    actual_memory = min(ell, 2/(r+2))
    bound = abs(mu)*np.array([p, 1-p])*actual_memory/(sigma*math.sqrt(2*math.pi))
    assert np.all(np.abs(gaps) <= bound+1e-9)
    return gaps


def main():
    rng, cases, worst = np.random.default_rng(20261018), 0, 0.
    for r in (.25, 1., 4., 16.):
        l0, l1 = 2/(r+2), 1/(r+1)
        budgets = [1., l0, (l0+l1)/2, l1, math.exp(-r)]
        budgets += [math.exp(-a)/(r-a+1) for a in (.1*r, .5*r, .9*r)]
        previous_energy = 0.
        for ell in sorted(budgets, reverse=True):
            a, c, left, right, eta, lam, energy = optimum(r, ell)
            memory = left+c*math.exp(-a)*(-math.expm1(-(r-a)))+right*math.exp(-r)
            tol = 20000*np.finfo(float).eps*max(1., eta, energy)
            assert min(left, right, c, lam) >= -tol
            assert abs(c*(r-a)+left+right-1) < tol
            assert memory <= ell+tol
            assert abs(lam*(memory-ell)) < tol
            u = lambda x: potential(x, r, a, c, left, right)
            integrated = c*quad(u, a, r, epsabs=1e-12)[0]+left*u(0)+right*u(r)
            assert abs(integrated-energy) < tol
            assert energy >= previous_energy-tol
            previous_energy = energy
            for x in np.linspace(0., r, 101):
                gap = u(x)+lam*math.exp(-x)-eta
                assert gap >= -tol
                if x >= a:
                    assert abs(gap) < tol
            for initial in (np.array([.7, .3]), np.array([1., 0.]), np.array([0., 1.])):
                mean, second = moments(r, a, c, left, right, initial)
                assert abs(mean-(initial[1]-.3)*memory) < tol
                if initial[1] == .3:
                    worst = max(worst, abs(second-.21*energy))
                    assert abs(second-.21*energy) < tol
            # Feasible atomic competitors: blend random measures with delta_R.
            x = np.sort(rng.uniform(0., r, 25))
            weights = rng.dirichlet(np.ones(len(x)))
            l_random = weights @ np.exp(-x)
            fraction = min(1., max(0., (ell-math.exp(-r))/(l_random-math.exp(-r))))
            x, weights = np.r_[x, r], np.r_[fraction*weights, 1-fraction]
            kernel = np.exp(-np.abs(x[:, None]-x[None, :]))
            assert weights @ kernel @ weights >= energy-tol
            cases += 1
    qp_gap, cdf_error = 0., 0.
    for ell in (.25, .15, .05, .005):
        r, x = 8., np.linspace(0., 8., 121)
        kernel, memory = np.exp(-np.abs(x[:, None]-x[None, :])), np.exp(-x)
        start = np.zeros(len(x)); start[-1] = 1.
        fit = minimize(lambda w: w @ kernel @ w, start, jac=lambda w: 2*kernel @ w,
                       method="SLSQP", bounds=[(0., None)]*len(x),
                       constraints=[{"type": "eq", "fun": lambda w: w.sum()-1, "jac": lambda w: np.ones(len(w))},
                                    {"type": "ineq", "fun": lambda w: (ell-memory @ w)/ell, "jac": lambda w: -memory/ell}],
                       options={"ftol": 1e-12, "maxiter": 1000})
        assert fit.success, fit.message
        exact = optimum(r, ell)[-1]
        a, c, left, right, *_ = optimum(r, ell)
        rounded = np.r_[left, c*np.maximum(0., x[1:]-np.maximum(a, x[:-1]))]
        rounded[-1] += right
        assert abs(rounded.sum()-1) < 1e-12 and memory @ rounded <= ell+1e-12
        assert -1e-12 <= rounded @ kernel @ rounded-exact <= max(np.diff(x))+1e-12
        assert abs(fit.x.sum()-1) < 1e-9 and memory @ fit.x <= ell+1e-9
        gap = fit.fun-exact
        assert -1e-9 <= gap < 5e-4
        qp_gap = max(qp_gap, gap)
        coarse, fine = coverage(r, ell, 160), coverage(r, ell, 256)
        cdf_error = max(cdf_error, float(np.max(np.abs(coarse-fine))))
        assert np.max(np.abs(coarse-fine)) < 1e-8
        print(f"R=8 ell={ell:.3f}: exact energy={exact:.10f}; grid excess={gap:.3e}")
    print(f"{cases} frontier cases; {3*cases} independent moment solves; 4 constrained grid optimizations")
    print(f"maximum stationary-variance discrepancy={worst:.3e}; maximum grid excess={qp_gap:.3e}")
    print(f"4 pooled-quantile coverage-bound checks; maximum Fourier-refinement error={cdf_error:.3e}")
    print(f"analytic CDF Fourier-tail bound at K=12/sigma: {math.exp(-72)/(144*math.pi):.3e}")
    print("R=8 transition budgets: L0=0.2000000000, L1=0.1111111111; endpoint floor=", math.exp(-8))
    print("PASS")


if __name__ == "__main__":
    main()
