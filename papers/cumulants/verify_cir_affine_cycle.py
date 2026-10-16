"""Affine CIR loading on a nonreversible cycle: exact two-turning-point proof."""
from fractions import Fraction as F

import numpy as np
from scipy.optimize import brentq

from verify_switching_kappa import integrated_variance_rate


def multiply(p, q):
    out = [F(0)] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        for j, y in enumerate(q):
            out[i+j] += x*y
    return out


def derivative(p):
    return [i*p[i] for i in range(1, len(p))]


def value(p, x):
    out = F(0)
    for a in reversed(p):
        out = out*x + a
    return out


def main():
    q = np.roll(np.eye(6, dtype=int), 1, axis=1) - np.eye(6, dtype=int)
    h = np.array([2, 1, -1, -2, -1, 1])
    assert np.array_equal(q @ q @ h + q @ h + h, np.zeros(6, dtype=int))
    assert h @ h == 12 and h @ q @ h == -6 and h.sum() == 0
    theta = [F(40 + int(x), 1000) for x in h]
    loading = [F(1, 25) - F(9, 5)*(t - F(1, 25)) for t in theta]
    margins = [2*t - a for t, a in zip(theta, loading)]
    s, baseline = F(1, 500000), F(1, 625)
    assert sum((t - F(1, 25))**2 for t in theta)/6 == s
    assert min(margins) == F(81, 2500) and min(loading) > 0
    # Derivative of (m^2-8m+10)/(m*(m^2+m+1)), in ascending powers.
    numerator, denominator = list(map(F, [10, -8, 1])), list(map(F, [0, 1, 1, 1]))
    left, right = multiply(derivative(numerator), denominator), multiply(numerator, derivative(denominator))
    p = [x-y for x, y in zip(left, right)]
    assert p == list(map(F, [-10, -20, -21, 16, -1]))
    signs = [np.sign(int(x)) for x in reversed(p)]
    assert sum(x != y for x, y in zip(signs, signs[1:])) == 2
    assert value(p, F(2)) < 0 < value(p, F(3))
    assert value(p, F(15)) < 0  # Descartes + two disjoint sign brackets.
    # An open interval of affine loadings has the same root count.
    assert F(169, 198) < F(9, 10) < 1
    assert 198*F(169, 198) - 169 == 0
    # Both sides are affine in c, so the c=0,1 identities certify every c.
    for c in (F(0), F(1), F(9, 10)):
        nc = [F(1), 1-2*c, 1-c]
        ld = multiply(derivative(nc), denominator)
        rd = multiply(nc, derivative(denominator))
        pc = [x-y for x, y in zip(ld, rd)]
        assert pc == [-1, -2, c-3, 4*c-2, c-1]
        assert value(pc, F(3)) == 198*c - 169
    gamma = lambda m: float(baseline + s*(m*m - 8*m + 10)/(10*m*(1+m+m*m)))
    roots = [brentq(lambda m: float(value(p, m)), *bracket) for bracket in ((2., 3.), (3., 15.))]
    for m in roots:
        print(f"turning speed={m:.10f}; Gamma={gamma(m):.13f}")
    assert gamma(3) < float(baseline)
    print("cycle theta:", tuple(map(float, theta)), "a:", tuple(map(float, loading)))
    print("minimum strict Feller margin:", min(margins), "=", float(min(margins)))
    print(f"S={s}; Gamma_infinity={float(baseline):.7f}; Gamma(3)={gamma(3):.13f}")
    print("stationary-variance baseline:", baseline/2, "; below-limit interval:", 4-np.sqrt(6), 4+np.sqrt(6))
    print("P(2), P(3), P(15):", *(value(p, F(m)) for m in (2, 3, 15)))
    print("robust two-turning interval: 169/198 < c < 1; witness c=9/10")
    rng, cases, worst = np.random.default_rng(20261017), 0, 0.
    chains = [q.astype(float)]
    for n in (2, 3, 4, 6):
        for _ in range(3):
            rates = rng.uniform(.05, 1., (n, n))
            np.fill_diagonal(rates, 0.)
            chains.append(rates - np.diag(rates.sum(axis=1)))
    chains.append(np.ones((3, 3))/3 - np.eye(3) + .1*np.outer([1., -1., 0.], [1., 1., -2.]))
    for chain in chains:
        n = len(chain)
        pi = np.linalg.solve(chain.T + np.ones((n, n)), np.ones(n))
        levels = .035 + .01*rng.uniform(size=n)
        mean, f = pi @ levels, levels - pi @ levels
        h_inverse = np.linalg.solve(-chain + np.outer(np.ones(n), pi), f)
        green = (pi*f) @ h_inverse
        for kappa in (.5, 2.):
            for beta in (-2.4, -2., -1.8, 0., 1.):
                a = kappa*(.02 + beta*f)
                assert np.min(a) > 0 and np.min(2*kappa*levels - a) > 0
                for m in (.05, .5, 2., 20.):
                    g = (pi*f) @ np.linalg.solve(kappa*np.eye(n) - m*chain, f)
                    expected_v = (pi @ a)*mean/(2*kappa) + kappa*(1+beta/2)*g
                    expected_rate = (pi @ a)*mean/kappa**2 + 2*green/m + beta*g
                    rate, mu, first, second, _, _ = integrated_variance_rate(
                        m, chain, pi, kappa*levels, np.full(n, kappa), a)
                    observed_v = second.sum() - mu**2
                    scale = max(second.sum() + mu**2, abs(rate), abs(expected_rate))
                    error = max(abs(observed_v-expected_v), abs(rate-expected_rate))/scale
                    worst = max(worst, error)
                    assert error < 20000*np.finfo(float).eps
                    cases += 1
    # Check the exact example independently with the general regime-resolved solver.
    for m in (.01, .1, 1., 2., 3., 10., 15., 100.):
        rate, mu, first, second, _, _ = integrated_variance_rate(
            m, q, np.ones(6)/6, np.array(theta, dtype=float), np.ones(6), np.array(loading, dtype=float))
        g = float(s)*(1+m/2)/(1+m+m*m)
        variance = float(baseline)/2 + g/10
        scale = second.sum() + mu**2 + abs(rate)
        assert max(abs(rate-gamma(m)), abs(second.sum()-mu**2-variance)) < 20000*np.finfo(float).eps*scale
    print(f"{cases} affine moment/Poisson cases and 8 exact-example comparisons; maximum scaled error={worst:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
