"""Linear-time, exact-form positive grid design with an exponential-memory cap."""
from fractions import Fraction as F
from itertools import product
from math import pi, sqrt

import numpy as np
from scipy.optimize import brentq, minimize
from scipy.special import ndtr


def design(decay, budget):
    """Return weights, energy, active suffix, potential level and multiplier.

    Input decay[j]=exp(-t[j]), with 0=t[0]<...<t[-1]. Works with
    Fraction inputs as well as floats; no covariance matrix is constructed.
    """
    n = len(decay)
    if n < 2 or decay[0] != 1 or not all(0 < decay[j+1] < decay[j] for j in range(n-1)):
        raise ValueError("need at least two strictly decreasing positive decays, starting at one")
    if not decay[-1] <= budget <= 1:
        raise ValueError("memory budget outside [exp(-R),1]")
    rho = [decay[j+1]/decay[j] for j in range(n-1)]
    tail = [0]*n
    for j in range(n-2, -1, -1):
        tail[j] = tail[j+1] + (1-rho[j])/(1+rho[j])
    information = 1+tail[0]
    if budget == decay[-1]:
        return [0]*(n-1)+[1], 1, n-1, 1+rho[-1], 1/decay[-2]
    if budget >= 1/information:
        k, c, z, energy = 0, 1/information, 0, 1/information
    else:
        k = next(j for j in range(n-1) if budget >= decay[j+1]/((1+tail[j])*(1+rho[j])-1))
        b = budget/decay[k]
        c, z = (1-b)/tail[k], (1-b)/tail[k]-b
        energy = b*b+(1-b)**2/tail[k]
    v = [1/(1+rho[k])]
    v += [1/(1+rho[j-1])-rho[j]/(1+rho[j]) for j in range(k+1, n-1)]
    v += [1/(1+rho[-1])]
    weights = [0]*k+[c*x for x in v]
    weights[k] -= z
    return weights, energy, k, c, z/decay[k]


def kernel_action(decay, weights):
    """Two linear Markov recurrences compute K@weights without a dense K."""
    n, left, right = len(decay), [0]*len(decay), [0]*len(decay)
    for j in range(1, n):
        left[j] = decay[j]/decay[j-1]*(left[j-1]+weights[j-1])
    for j in range(n-2, -1, -1):
        right[j] = decay[j+1]/decay[j]*(right[j+1]+weights[j+1])
    return [left[j]+weights[j]+right[j] for j in range(n)]


def binary_score_check(decay, w, energy):
    """Exact path enumeration, independent of covariance and KKT formulas."""
    p, laws = F(3, 10), []
    stationary = [1-p, p]
    memory = sum(l*x for l, x in zip(decay, w))
    for initial in (stationary, [F(1), F(0)], [F(0), F(1)]):
        atoms, probabilities = [], []
        for path in product((0, 1), repeat=len(w)):
            mass = initial[path[0]]
            for j in range(len(w)-1):
                rho = decay[j+1]/decay[j]
                mass *= (1-rho)*stationary[path[j+1]]+rho*(path[j] == path[j+1])
            atoms.append(sum(x*(y-p) for x, y in zip(w, path))); probabilities.append(mass)
        assert sum(probabilities) == 1
        assert sum(m*a for m, a in zip(probabilities, atoms)) == (initial[1]-p)*memory
        if initial == stationary:
            assert sum(m*a*a for m, a in zip(probabilities, atoms)) == p*(1-p)*energy
        laws.append((np.array(atoms, dtype=float), np.array(probabilities, dtype=float)))
    cdf = lambda q, law: law[1] @ ndtr((q-.8*law[0])/.7)
    quantile = brentq(lambda q: cdf(q, laws[0])-.9, -5., 5.)
    gaps = np.array([cdf(quantile, law)-.9 for law in laws[1:]])
    bound = .8*np.array([float(p), float(1-p)])*float(memory)/(.7*sqrt(2*pi))
    assert np.all(abs(gaps) <= bound+1e-12)
    print("3 exact 32-path score laws; Gaussian-smoothed pooled coverage gaps:", gaps)


def exact_checks():
    decay = [F(1), F(1, 2), F(1, 3), F(1, 5), F(1, 8)]
    rho = [decay[j+1]/decay[j] for j in range(4)]
    information = [1+sum((1-r)/(1+r) for r in rho[j:]) for j in range(4)]
    breaks = [1/information[0]] + [decay[j+1]/(information[j]*(1+rho[j])-1) for j in range(4)]
    assert all(x > y for x, y in zip(breaks, breaks[1:]))
    assert breaks[-1] == decay[-1]
    budgets = [F(1), F(1, 6)] + breaks + [(x+y)/2 for x, y in zip(breaks, breaks[1:])]
    matrix = [[min(x, y)/max(x, y) for y in decay] for x in decay]
    for budget in budgets:
        w, energy, k, eta, lam = design(decay, budget)
        potential = [sum(row[j]*w[j] for j in range(5)) for row in matrix]
        assert potential == kernel_action(decay, w)
        assert min(w) >= 0 and sum(w) == 1 and lam >= 0
        assert sum(x*y for x, y in zip(decay, w)) <= budget
        slack = [u+lam*l-eta for u, l in zip(potential, decay)]
        assert min(slack) >= 0 and sum(x*y for x, y in zip(slack, w)) == 0
        assert lam*(sum(x*y for x, y in zip(decay, w))-budget) == 0
        assert sum(x*y for x, y in zip(potential, w)) == energy
    w, energy, k, *_ = design(decay, F(1, 6))
    binary_score_check(decay, w, energy)
    print("exact grid: t=log(1,2,3,5,8); budget=1/6; weights=", tuple(w))
    print("exact energy=", energy, "=", float(energy), "; first active index=", k)
    print("exact memory breakpoints:", tuple(breaks))
    return len(budgets)


def main():
    exact = exact_checks()
    rng, cases, qp_cases, worst, qp_error = np.random.default_rng(20261019), 0, 0, 0., 0.
    for n in (2, 3, 5, 9, 17):
        for _ in range(4):
            gaps = rng.uniform(.05, 1., n-1)
            times = np.r_[0., np.cumsum(gaps/gaps.sum()*rng.uniform(.2, 10.))]
            decay = np.exp(-times)
            rho = decay[1:]/decay[:-1]
            information = 1+np.cumsum(((1-rho)/(1+rho))[::-1])[::-1]
            breaks = np.r_[1/information[0], decay[1:]/(information*(1+rho)-1)]
            matrix = np.exp(-np.abs(times[:, None]-times[None, :]))
            for budget in np.r_[1., breaks[:-1], (breaks[:-1]+breaks[1:])/2, decay[-1]]:
                w, energy, k, eta, lam = design(list(decay), budget)
                w = np.array(w)
                potential = matrix @ w
                primal_tol = 50000*np.finfo(float).eps
                tol = primal_tol*max(1., eta, lam)
                assert min(w) >= -primal_tol and abs(w.sum()-1) < primal_tol
                assert decay @ w <= budget+primal_tol
                assert min(potential+lam*decay-eta) >= -tol
                assert abs(w @ (potential+lam*decay-eta)) < tol
                assert abs(lam*(decay @ w-budget)) < tol
                error = max(abs(w @ potential-energy), max(abs(potential-kernel_action(decay, w))))
                worst = max(worst, error)
                assert error < primal_tol*max(1., energy)
                cases += 1
            budget = (breaks[0]+decay[-1])/2
            w, energy, *_ = design(list(decay), budget)
            start = np.zeros(n); start[-1] = 1.
            fit = minimize(lambda v: v @ matrix @ v, start, jac=lambda v: 2*matrix @ v,
                           method="SLSQP", bounds=[(0., None)]*n,
                           constraints=[{"type": "eq", "fun": lambda v: v.sum()-1, "jac": lambda v: np.ones(n)},
                                        {"type": "ineq", "fun": lambda v: (budget-decay @ v)/budget, "jac": lambda v: -decay/budget}],
                           options={"ftol": 1e-12, "maxiter": 1000})
            assert fit.success, fit.message
            assert min(fit.x) >= -1e-9 and abs(fit.x.sum()-1) < 1e-9 and decay @ fit.x <= budget+1e-9
            assert abs(fit.fun-energy) < 1e-8
            qp_error = max(qp_error, abs(fit.fun-energy)); qp_cases += 1
    times = np.linspace(0., 8., 100001)
    decay = np.exp(-times)
    w, energy, k, eta, lam = design(list(decay), .005)
    potential = np.array(kernel_action(decay, w))
    assert abs(sum(w)-1) < 1e-9 and min(w) >= 0 and decay @ w <= .005+1e-9
    assert abs(np.dot(w, potential)-energy) < 1e-9
    assert min(potential+lam*decay-eta) > -1e-9
    print(f"{exact} exact-rational cases; {cases} floating KKT cases; {qp_cases} independent QPs")
    print(f"maximum dense/recurrence/energy error={worst:.3e}; maximum QP energy error={qp_error:.3e}")
    print(f"100001-point grid R=8, budget=.005: first active time={times[k]:.8f}; energy={energy:.10f}")
    print("PASS")


if __name__ == "__main__":
    main()
