"""Certificate for the three numbers a fast regime leaves in a Vasicek yield curve.

1. The first-order formula: the residual against the numerical solution shrinks like |Q|^-2.
2. The symmetrized Green-Kubo matrix is positive semidefinite for random non-reversible chains.
3. A two-state chain gives c3^2 = 4 c2 c4 exactly.
4. Two different three-state chains with the same averages and Green-Kubo matrix give yield curves
   that agree to first order and differ only at order |Q|^-2.
5. The long yield: the eigenvalue series converges to the principal eigenvalue even for slow switching, and its
   first two terms are the long-maturity limit of the three-number formula.
"""
import math
import numpy as np
from scipy.optimize import least_squares
from three_numbers import (stationary, gk_matrix, coefficients, int_Bk, log_bond_stationary)

KAPPA = 1.0
QA = np.array([[-3, 2, 1], [1, -2, 1], [0.5, 1.5, -2]], float)
THA = np.array([0.08, 0.04, 0.01])
SA = np.array([0.003, 0.0015, 0.0005])
QB = np.array([[-1, 1, 0], [0.2, -2.2, 2.0], [3.0, 0, -3.0]], float)


def first_order(T, Q, thetas, s, kappa=KAPPA):
    pi = stationary(Q)
    c2, c3, c4 = coefficients(Q, thetas, s, kappa)
    return (-kappa * (pi @ thetas) * int_Bk(1, T, kappa) + (pi @ s / 2 + c2) * int_Bk(2, T, kappa)
            + c3 * int_Bk(3, T, kappa) + c4 * int_Bk(4, T, kappa))


def invariants(Q, thetas, s):
    pi = stationary(Q)
    M = gk_matrix(Q, thetas, s)
    return np.array([pi @ thetas, pi @ s, M[0, 0], M[0, 1], M[1, 1]])


def matched_chain(QB, target):
    """Rates on chain QB reproducing the averages and Green-Kubo matrix in target, with variances kept positive."""
    scale = np.abs(target) + 1e-12
    f = lambda z: (invariants(QB, z[:3], z[3:]) - target) / scale
    best = None
    for seed in range(20):
        z0 = np.concatenate([THA, SA]) * np.random.default_rng(seed).uniform(0.5, 1.5, 6)
        r = least_squares(f, z0, bounds=([-1, -1, -1, 1e-5, 1e-5, 1e-5], [1, 1, 1, 1, 1, 1]),
                          xtol=1e-15, ftol=1e-15, gtol=1e-15)
        if best is None or r.cost < best.cost:
            best = r
    return best.x[:3], best.x[3:], best.cost


def main():
    ok = True
    Ts = [1.0, 3.0, 5.0, 10.0]

    print("1. first-order formula: residual x scale^2 should settle")
    for T in Ts:
        row = []
        for m in [10, 20, 40, 80]:
            r = log_bond_stationary(T, m * QA, THA, SA, KAPPA, dps='double') - first_order(T, m * QA, THA, SA)
            row.append(r * m ** 2)
        drift = abs(row[-1] - row[-2]) / abs(row[-1])
        ok &= drift < 0.05
        print(f"   T={T:5.1f}  " + "  ".join(f"{x: .4e}" for x in row) + f"   drift {drift:.3f}")

    print("2. Green-Kubo matrix PSD on random non-reversible chains")
    rng = np.random.default_rng(1)
    worst = np.inf
    for _ in range(2000):
        n = rng.integers(2, 7)
        Q = rng.exponential(1.0, (n, n)) * (rng.uniform(size=(n, n)) < 0.7)
        np.fill_diagonal(Q, 0)
        Q[np.arange(n), (np.arange(n) + 1) % n] += 0.1   # irreducible
        np.fill_diagonal(Q, -Q.sum(1))
        M = gk_matrix(Q, rng.normal(size=n), rng.normal(size=n))
        worst = min(worst, np.linalg.eigvalsh(M).min() / max(1e-300, np.abs(M).max()))
    ok &= worst > -1e-10
    print(f"   smallest relative eigenvalue {worst:.2e}")

    print("3. two-state chain: c3^2 / (4 c2 c4) = 1")
    for p, q in [(1.0, 1.0), (0.3, 2.0), (5.0, 0.7)]:
        c2, c3, c4 = coefficients(np.array([[-p, p], [q, -q]]), np.array([0.07, 0.01]), np.array([0.002, 0.0004]), KAPPA)
        ratio = c3 ** 2 / (4 * c2 * c4)
        ok &= abs(ratio - 1) < 1e-10
        print(f"   rates ({p}, {q})  ratio {ratio:.12f}")

    print("4. two chains with the same three numbers")
    thB, sB, cost = matched_chain(QB, invariants(QA, THA, SA))
    ok &= cost < 1e-20 and sB.min() > 0
    print(f"   chain B rates theta {np.round(thB, 5)}  s {np.array2string(sB, precision=3)}  match cost {cost:.1e}")
    print("   invariants A", invariants(QA, THA, SA))
    print("   invariants B", invariants(QB, thB, sB))
    for T in Ts:
        row = []
        for m in [10, 20, 40, 80]:
            d = log_bond_stationary(T, m * QA, THA, SA, KAPPA, dps='double') - log_bond_stationary(T, m * QB, thB, sB, KAPPA, dps='double')
            row.append(d * m ** 2)
        drift = abs(row[-1] - row[-2]) / abs(row[-1])
        ok &= drift < 0.05
        print(f"   T={T:5.1f}  diff x scale^2  " + "  ".join(f"{x: .4e}" for x in row) + f"   drift {drift:.3f}")

    print("5. long end: eigenvalue series")
    from long_end import eigen_series, long_yield
    D = -THA + SA / (2 * KAPPA ** 2)
    mu = eigen_series(QA, D, 60)
    c2, c3, c4 = coefficients(QA, THA, SA, KAPPA)
    lim = c2 / KAPPA ** 2 + c3 / KAPPA ** 3 + c4 / KAPPA ** 4
    ok &= abs(mu[2] - lim) < 1e-15
    print(f"   second term {mu[2]:.6e}  long-maturity limit of the formula {lim:.6e}")
    for m in [0.05, 0.5, 5.0]:
        err = abs(-sum(mu[k] * m ** (1 - k) for k in range(1, 61)) - long_yield(m * QA, THA, SA, KAPPA))
        ok &= err < 1e-13
        print(f"   scale {m:5}  error of 60-term series {err:.1e}")

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
