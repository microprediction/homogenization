"""Certificate: factorial cumulants of a Markov-modulated Poisson count equal the cumulants of its integrated rate.

Given the regime path, N_T is Poisson with mean Lambda_T = int_0^T l(y_s) ds, so E[(1+t)^{N_T}] = E[e^{t Lambda_T}] and
every factorial cumulant of the count equals the corresponding cumulant of the integrated rate. Data side: the
factorial moments E[N(N-1)...(N-r+1)] come from the count distribution, computed here from the generating function at
the roots of unity as on the counts page. Rate side: the cumulants of Lambda_T from the polynomial moment system of
polymoments.py. The two agree to the rounding of the count law, for a stationary start and for a start in the busy regime; and
kappa_2(Lambda_T) = 2TK + O(eps^2) uniformly in T, the bound of the cumulants page.
"""
import json, math, os
import numpy as np
from scipy.linalg import expm
from polymoments import Poly, stationary, group_inverse, gk, bs_generator, switched_expectation, cumulants

P = Poly(4)
Q2 = np.array([[-1, 1], [1, -1.0]])
L2 = np.array([8.0, 1.0])


def count_distribution(Q, l, T, nu, M=256):
    """P(N_T = k), k < M, from a(z) = nu . exp(T (Q + (z - 1) diag l)) 1 at the M roots of unity."""
    zs = np.exp(2j * np.pi * np.arange(M) / M)
    a = np.array([nu @ expm(T * (Q + (z - 1) * np.diag(l))) @ np.ones(len(l)) for z in zs])
    p = np.real(np.fft.fft(a)) / M
    return p


def factorial_cumulants(p):
    k = np.arange(len(p))
    fm = [np.sum(k * p), np.sum(k * (k - 1) * p), np.sum(k * (k - 1) * (k - 2) * p), np.sum(k * (k - 1) * (k - 2) * (k - 3) * p)]
    return cumulants(fm)          # cumulants of the factorial moment sequence = factorial cumulants


def main():
    ok, out = True, {}
    print("1. two regimes, rates (8, 1), T = 1: factorial cumulants of the count against cumulants of the integrated rate")
    rows = []
    for lam in (5.0, 10.0, 20.0, 40.0):
        Q = lam * Q2
        for name, nu in (('stationary', stationary(Q)), ('busy start', np.array([1.0, 0.0]))):
            p = count_distribution(Q, L2, 1.0, nu)
            fc = factorial_cumulants(p)
            Ls = [bs_generator(P, li) for li in L2]
            kL = cumulants(np.array([switched_expectation(Q, Ls, 1.0, P.monomial(0, k, 0), P, 0.0, nu) for k in (1, 2, 3, 4)]))
            gap = np.abs(fc - kL).max()
            ok &= gap < 1e-5 and abs(p.sum() - 1) < 1e-12      # k^4 weights amplify rounding in the count law
            rows.append([lam, name] + list(kL) + [gap])
            print(f"   lam={lam:4.0f} {name:11s}: kappa_1..4(Lambda) {kL[0]:.5f} {kL[1]:.5f} {kL[2]:.5f} {kL[3]:.5f}; largest gap to the count's factorial cumulants {gap:.1e}")
    out['identity'] = rows
    print("2. kappa_2(Lambda_T) - 2TK, stationary start, uniformly in T")
    rows = []
    for lam in (5.0, 10.0, 20.0, 40.0):
        Q = lam * Q2
        pi, Qs = stationary(Q), group_inverse(Q)
        K = gk(Q, L2, L2)
        lt = L2 - pi @ L2
        sup = max(abs(-2 * pi @ (lt * (Qs @ Qs @ ((np.eye(2) - expm(Q * T)) @ lt)))) for T in np.geomspace(0.01, 50, 60))
        bound = 2 * pi @ (lt * (Qs @ Qs @ lt))     # C >= 0 for two states, so int u |C| = pi.(l~ (Q#)^2 l~)
        rows.append([lam, 2 * K, sup, bound])
        ok &= sup <= bound * (1 + 1e-9)
        print(f"   lam={lam:4.0f}: 2K = {2 * K:.5f}; sup_T |kappa_2(Lambda_T) - 2TK| = {sup:.3e} <= bound {bound:.3e}")
    out['uniform'] = rows
    ok &= abs(math.log2(rows[-2][2] / rows[-1][2]) - 2) < 0.05
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'counts_results.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == '__main__':
    main()
