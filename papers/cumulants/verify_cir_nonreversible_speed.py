"""All-speed stationary CIR variance order without reversibility.

Check the symmetric-resolvent precision identity and its Loewner derivative
against direct inverses and independent invariant-moment/Poisson solves.
Exact fractions certify failure of complete monotonicity on a directed cycle.
"""
from fractions import Fraction as F
from math import factorial

import numpy as np
from scipy.linalg import null_space


EPS = np.finfo(float).eps


def coordinates(q):
    n = len(q)
    pi = np.linalg.solve(q.T + np.ones((n, n)), np.ones(n))
    root = np.sqrt(pi)
    u = null_space(root[None, :]) / root[:, None]
    ell = -u.T @ (pi[:, None] * q) @ u
    return pi, u, ell


def sym(a):
    return (a + a.T) / 2


def exact_cycle_derivative(point, order):
    """Taylor recurrence for (1+3m/2)/(1+3m+3m^2), multiplied by Var(theta)."""
    b0, b1, b2 = 1 + 3*point + 3*point**2, 3 + 6*point, F(3)
    coeff = []
    for k in range(order + 1):
        numerator = 1 + F(3, 2)*point if k == 0 else (F(3, 2) if k == 1 else F(0))
        coeff.append((numerator - (b1*coeff[k-1] if k else 0)
                      - (b2*coeff[k-2] if k >= 2 else 0)) / b0)
    return factorial(order) * coeff[-1] / 15000


def main():
    rng = np.random.default_rng(20261016)
    chains = []
    for n in (2, 3, 4, 6):
        for _ in range(6):
            rates = rng.uniform(.02, 2., (n, n))
            np.fill_diagonal(rates, 0.)
            chains.append(rates - np.diag(rates.sum(axis=1)))
    cycle = np.roll(np.eye(3), 1, axis=1) - np.eye(3)
    nilpotent = .1*np.outer([1., -1., 0.], [1., 1., -2.])
    projection = np.ones((3, 3))/3
    assert np.linalg.norm(nilpotent @ nilpotent) < 1000*EPS
    assert np.linalg.norm(nilpotent @ projection) + np.linalg.norm(projection @ nilpotent) < 1000*EPS
    chains += [cycle, projection - np.eye(3) + nilpotent]
    cases, loewner, worst = 0, 0, 0.
    for q in chains:
        pi, u, ell = coordinates(q)
        s, a = sym(ell), (ell - ell.T)/2
        assert np.linalg.eigvalsh(s)[0] > 0
        theta = .03 + .04*rng.uniform(size=len(q))
        centered = theta - pi @ theta
        feature = u.T @ (pi * centered)
        eye = np.eye(len(ell))
        for kappa in (.5, 2.):
            xi2, previous = .01, None
            assert np.min(2*kappa*theta - xi2) > 0
            for speed in (.01, .1, 1., 10., 100.):
                x = kappa*eye + speed*s
                xi = np.linalg.inv(x)
                precision = x + speed**2 * a.T @ xi @ a
                derivative = s + a.T @ (speed*xi @ (2*kappa*eye + speed*s) @ xi) @ a
                inverse = np.linalg.inv(kappa*eye + speed*ell)
                r = sym(inverse)
                dr = -r @ derivative @ r
                direct_dr = -sym(inverse @ ell @ inverse)
                tolerance = 20000*EPS*max(1., np.linalg.norm(r), np.linalg.norm(dr))
                assert np.linalg.norm(r - np.linalg.inv(precision)) < tolerance
                assert np.linalg.norm(dr - direct_dr) < tolerance
                assert np.linalg.eigvalsh(derivative)[0] > 0
                assert np.linalg.eigvalsh(-dr)[0] > 0
                if previous is not None:
                    assert np.linalg.eigvalsh(previous - r)[0] > 0
                    loewner += 1
                previous = r
                # Independent regime-resolved invariant first/second moments.
                first = np.linalg.solve(kappa*np.eye(len(q)) - speed*q.T, kappa*pi*theta)
                second = np.linalg.solve(2*kappa*np.eye(len(q)) - speed*q.T, (2*kappa*theta + xi2)*first)
                variance = second.sum() - (pi @ theta)**2
                formula = xi2*(pi @ theta)/(2*kappa) + kappa*feature @ r @ feature
                scale = second.sum() + (pi @ theta)**2
                error = abs(variance - formula)/scale
                worst = max(worst, error)
                assert error < 20000*EPS
                # Joint-process Poisson solve phi(v,i)=v/kappa+(H f)_i/speed.
                h = u @ np.linalg.solve(ell, feature)
                poisson_rate = 2*(variance/kappa + first @ h/speed)
                exact_rate = xi2*(pi @ theta)/kappa**2 + 2*(pi*centered) @ h/speed
                assert abs(poisson_rate - exact_rate) < 20000*EPS*max(scale, abs(exact_rate))
                assert feature @ dr @ feature < 0
                cases += 1
    assert exact_cycle_derivative(F(0), 4) == -F(9, 1250)
    positive_witness = exact_cycle_derivative(F(1, 100), 4)
    assert positive_witness < 0
    integer_l = np.eye(3, dtype=int) - np.roll(np.eye(3, dtype=int), 1, axis=1)
    three_p = 3*np.eye(3, dtype=int) - np.ones((3, 3), dtype=int)
    assert np.array_equal(integer_l + integer_l.T, three_p)
    assert np.array_equal(integer_l @ integer_l.T, three_p)
    pi, u, ell = coordinates(cycle)
    theta = np.array([.03, .04, .05])
    f = u.T @ (pi*(theta - pi@theta))
    levels = [F(3, 100), F(4, 100), F(5, 100)]
    mean = sum(levels)/3
    base = F(1, 1000)*mean/2
    var_theta = sum((t - mean)**2 for t in levels)/3
    assert base == F(1, 50000) and var_theta == F(1, 15000)
    for speed in (.01, .1, 1., 10.):
        ratio = (1 + 1.5*speed)/(1 + 3*speed + 3*speed**2)
        assert abs(f @ np.linalg.solve(np.eye(2) + speed*ell, f) - ratio/15000) < 1000*EPS
        first = np.linalg.solve(np.eye(3) - speed*cycle.T, pi*theta)
        second = np.linalg.solve(2*np.eye(3) - speed*cycle.T, (2*theta + .001)*first)
        assert abs(second.sum() - .04**2 - float(base) - ratio*float(var_theta)) < 1000*EPS
    print(f"{cases} precision/derivative/moment/Poisson cases; {loewner} finite Loewner comparisons")
    print(f"maximum invariant-variance relative discrepancy={worst:.3e}")
    print(f"cycle: kappa=1 xi^2=.001 theta=(.03,.04,.05); baseline={base}, Var(theta)={var_theta}")
    print("strict Feller margins:", 2*theta - .001)
    print("fourth derivative at 0 = -9/1250 =", float(exact_cycle_derivative(F(0), 4)))
    print(f"fourth derivative at m=.01 = {positive_witness} = {float(positive_witness):.12f}")
    print("PASS")


if __name__ == "__main__":
    main()
