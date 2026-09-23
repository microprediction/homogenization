"""The long end of the curve: the long yield is minus the principal eigenvalue of Q + diag g_inf.

With g_inf,i = -theta_i + s_i / (2 kappa^2) (the forcing once B has reached 1/kappa), the long yield is
    y_inf = -Lambda,   Lambda = principal eigenvalue of Q + diag(g_inf).
For a fast chain Q = Q0 / eps, Lambda = mu(eps) / eps with mu the principal eigenvalue of Q0 + eps D.
Rayleigh-Schroedinger with the group inverse gives mu = sum_k eps^k mu_k:
    mu_k = pi . (D v_{k-1}),   v_k = Q0# ( sum_{j=1}^{k-1} mu_j v_{k-j} - D v_{k-1} ),   v_0 = 1.
mu_1 = pi.D is the average and mu_2 = K(D~, D~) is the Green-Kubo term, so the first two terms are the T -> inf
limit of the three-number formula. Unlike the expansion in time, this series converges for small eps.
"""
import numpy as np
from three_numbers import stationary, group_inverse


def eigen_series(Q0, D, order):
    pi, Qs = stationary(Q0), group_inverse(Q0)
    n = len(D)
    v, mu = [np.ones(n)], [0.0]
    for k in range(1, order + 1):
        mu.append(pi @ (D * v[k - 1]))
        rhs = sum((mu[j] * v[k - j] for j in range(1, k)), np.zeros(n)) - D * v[k - 1]
        v.append(Qs @ rhs)
    return np.array(mu)


def long_yield(Q, thetas, s, kappa):
    D = -np.asarray(thetas) + np.asarray(s) / (2 * kappa ** 2)
    return -np.max(np.linalg.eigvals(np.asarray(Q, float) + np.diag(D)).real)


if __name__ == "__main__":
    from verify_three_numbers import QA, THA, SA, KAPPA
    D = -THA + SA / (2 * KAPPA ** 2)
    mu = eigen_series(QA, D, 40)
    for m in [0.02, 0.05, 0.1, 0.5, 2.0]:
        eps = 1 / m
        exact = long_yield(m * QA, THA, SA, KAPPA)
        partial = -np.cumsum([mu[k] * eps ** (k - 1) for k in range(1, 41)])
        errs = [abs(partial[k] - exact) for k in [0, 1, 2, 4, 9, 19, 39]]
        print(f"m={m:5}  y_inf={exact*1e4:9.4f}bp  errors after 1,2,3,5,10,20,40 terms: " + " ".join(f"{e:.1e}" for e in errs))
    r = np.abs(mu[1:]) ** (1 / np.arange(1, 41))
    print("root test |mu_k|^(1/k), k=10,20,30,40:", r[[9, 19, 29, 39]], " -> convergence for m >", r[39])
