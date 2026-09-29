"""Certificate for the cumulant benchmark of integrated variance under a fast regime.

1. Switched Black-Scholes variance s(Y_t), three regimes: Var(V_T) = 2TK - 2 pi.(s~ (Q#)^2 (I - e^{QT}) s~) exactly,
   against the polynomial moment system; kappa_4(M_T) = 3 Var(V_T) for independent Brownian noise; the finite-rate bound
   |Var(V_T) - 2TK| <= 2 int_0^inf u |C(u)| du holds uniformly in T and is O(m^-2); the log return X = M - V/2 has
   kappa_4(X) = 3 kappa_2(V) + 3/2 kappa_3(V) + kappa_4(V)/16, which is not 3 Var(V).
2. CIR variance with a switched mean level (the Heston page's model, rho = 0): Var(V_T) = Var_avg(V_T) + Var(E[V_T | Y])
   exactly, the second term a double integral of the chain covariance with weight h(u) = 1 - e^{-kappa (T-u)};
   replacing it by 2K int h^2 leaves a remainder bounded by 4 int u |C(u)| du uniformly in T.
3. At fixed T the first-order rule L_bar + K A^2 on the polynomial space gives the cumulants of V_T to O(m^-2).
4. With kappa and theta both switched on a one-way cycle, the rule with the product coefficients (kappa theta, kappa)
   reaches O(m^-2); its symmetric part alone does not, because [d_v, v d_v] != 0.
5. Leverage: with rho != 0 the identity for kappa_4(X) fails; exact values for rho = 0 and rho = -0.6.
"""
import json, math, os
import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad
from polymoments import (Poly, stationary, group_inverse, gk, cir_generator, bs_generator, switched_expectation,
                         first_order_expectation, cumulants, cumulant_derivative, logreturn_from_V)

QA = np.array([[-3, 2, 1], [1, -2, 1], [0.5, 1.5, -2]], float)
QC = np.array([[-2.1, 2, 0.1], [0.1, -2.1, 2], [2, 0.1, -2.1]], float)   # strongly one-way cycle 1 -> 2 -> 3 -> 1
Q2 = np.array([[-1, 1], [1, -1.0]])
P = Poly(4)


def chain_cov(Q, f):
    """C(u) = Cov_pi(f(Y_0), f(Y_u)) as a function, and int_0^inf u |C(u)| du."""
    pi = stationary(Q)
    ft = np.asarray(f, float) - pi @ f
    C = lambda u: pi @ (ft * (expm(Q * u) @ ft))
    rate = -max(np.linalg.eigvals(Q).real[np.abs(np.linalg.eigvals(Q)) > 1e-9])
    I = quad(lambda u: u * abs(C(u)), 0, 60 / rate, limit=400)[0]
    return C, I


def var_V_chain(Q, s, T):
    pi, Qs = stationary(Q), group_inverse(Q)
    st = np.asarray(s, float) - pi @ s
    return 2 * T * gk(Q, s, s) - 2 * pi @ (st * (Qs @ Qs @ ((np.eye(len(Q)) - expm(Q * T)) @ st)))


def moments_V(Q, Ls, T, v0, nu=None):
    return np.array([switched_expectation(Q, Ls, T, P.monomial(0, k, 0), P, v0, nu) for k in (1, 2, 3, 4)])


def moments_X(Q, Ls, T, v0, nu=None):
    return np.array([switched_expectation(Q, Ls, T, P.logreturn_power(k), P, v0, nu) for k in (1, 2, 3, 4)])


def var_cond_mean(Q, thetas, kappa, T, nodes=400):
    """Var(E[V_T | Y]) = 2 int_0^T h(u) pi.(th~ * [int_0^u h(u') e^{Q (u-u')} du'] th~) du, h(u) = 1 - e^{-kappa (T-u)}.
    The inner integral is closed form on the centered subspace; the outer one is Gauss-Legendre on two panels."""
    Q = np.asarray(Q, float)
    n, pi, Qs = len(Q), stationary(Q), group_inverse(Q)
    tt = np.asarray(thetas, float) - pi @ thetas
    Qk = np.linalg.inv(Q - kappa * np.eye(n))
    I = np.eye(n)

    def inner(u):  # int_0^u (1 - e^{-kappa (T - u')}) e^{Q (u - u')} du' applied to a centered vector
        return Qs @ (expm(Q * u) - I) @ tt - math.exp(-kappa * (T - u)) * (Qk @ (expm((Q - kappa * I) * u) - I) @ tt)

    f = lambda u: (1 - math.exp(-kappa * (T - u))) * (pi @ (tt * inner(u)))
    rate = -max(np.linalg.eigvals(Q).real[np.abs(np.linalg.eigvals(Q)) > 1e-9])
    split = min(T, 25 / rate)
    x, w = np.polynomial.legendre.leggauss(nodes)
    total = 0.0
    for a, b in ([(0, split), (split, T)] if split < T else [(0, T)]):
        us = 0.5 * (b - a) * x + 0.5 * (b + a)
        total += 0.5 * (b - a) * sum(wi * f(ui) for wi, ui in zip(w, us))
    return 2 * total


def int_h2(kappa, T):
    return T - 2 * (1 - math.exp(-kappa * T)) / kappa + (1 - math.exp(-2 * kappa * T)) / (2 * kappa)


def main():
    ok, out = True, {}

    print("1. switched Black-Scholes variance: three regimes, volatilities 10%, 30%, 20%, chain m Q_A")
    s = np.array([0.01, 0.09, 0.04])
    Ls = [bs_generator(P, si) for si in s]
    rows = []
    for m in (2, 4, 8, 16):
        Q = m * QA
        K = gk(Q, s, s)
        mV = moments_V(Q, Ls, 1.0, 0.0)
        kV = cumulants(mV)
        closed = var_V_chain(Q, s, 1.0)
        M4 = switched_expectation(Q, Ls, 1.0, P.monomial(0, 0, 4), P, 0.0)
        kX = cumulants(moments_X(Q, Ls, 1.0, 0.0))
        ident = logreturn_from_V(kV)
        C, I = chain_cov(Q, s)
        sup = max(abs(var_V_chain(Q, s, T) - 2 * T * K) for T in np.geomspace(0.02, 40, 80))
        ok &= abs(closed - kV[1]) < 1e-14 and abs(M4 - 3 * mV[1]) < 1e-14 and abs(kX[3] - ident[2]) < 1e-14 and sup <= 2 * I * (1 + 1e-9)
        rows.append([m, 6 * K, 3 * kV[1], kX[3], sup, 2 * I])
        print(f"   m={m:2d}  T=1: 6TK {6 * K:.4e}, 3 Var(V) {3 * kV[1]:.4e}, kappa_4(X) {kX[3]:.4e};"
              f" closed form vs moments {abs(closed - kV[1]):.1e}, E[M^4] - 3E[V^2] {abs(M4 - 3 * mV[1]):.1e},"
              f" identity {abs(kX[3] - ident[2]):.1e}; sup_T |Var - 2TK| {sup:.3e} <= bound {2 * I:.3e}")
    r = np.array(rows)
    o = math.log2(r[-2, 4] / r[-1, 4]); ob = math.log2(r[-2, 5] / r[-1, 5])
    print(f"   orders from the last doubling: sup_T |Var(V_T) - 2TK| {o:.3f}, bound {ob:.3f}")
    ok &= abs(o - 2) < 0.1 and abs(ob - 2) < 0.1
    out['chain'] = rows

    print("2. CIR variance with a switched mean level: kappa = 2, theta = (0.09, 0.02), xi = 0.4, v0 = 0.04, rho = 0")
    kappa, th, xi, v0 = 2.0, np.array([0.09, 0.02]), 0.4, 0.04
    rows = []
    Ts = np.geomspace(0.05, 50, 40)
    for lam in (10, 20, 40, 80):
        Q = lam * Q2
        pi = stationary(Q)
        K = gk(Q, th, th)
        Ls = [cir_generator(P, kappa, t, xi, 0.0) for t in th]
        Lavg = cir_generator(P, kappa, pi @ th, xi, 0.0)
        C, I = chain_cov(Q, th)
        worst_dec, worst_gk = 0.0, 0.0
        for T in Ts:
            varV = cumulants(moments_V(Q, Ls, T, v0))[1]
            var_avg = cumulants(moments_V(np.zeros((1, 1)), [Lavg], T, v0))[1]
            vcm = var_cond_mean(Q, th, kappa, T)
            worst_dec = max(worst_dec, abs(varV - var_avg - vcm) / varV)
            worst_gk = max(worst_gk, abs(vcm - 2 * K * int_h2(kappa, T)))
        M4 = switched_expectation(Q, Ls, 1.0, P.monomial(0, 0, 4), P, v0)
        V2 = moments_V(Q, Ls, 1.0, v0)[1]
        rows.append([lam, worst_dec, worst_gk, 4 * I])
        ok &= worst_dec < 1e-9 and worst_gk <= 4 * I * (1 + 1e-9) and abs(M4 - 3 * V2) < 1e-13
        print(f"   lam={lam:3d}: decomposition Var = Var_avg + Var(E[V|Y]) relative error {worst_dec:.1e} over T in [0.05, 50];"
              f" sup_T |Var(E[V|Y]) - 2K int h^2| {worst_gk:.3e} <= bound {4 * I:.3e}; E[M^4] - 3E[V^2] at T=1: {abs(M4 - 3 * V2):.1e}")
    r = np.array(rows)
    o = math.log2(r[-2, 2] / r[-1, 2])
    print(f"   order of the Green-Kubo remainder from the last doubling: {o:.3f}")
    ok &= abs(o - 2) < 0.1
    out['cir_uniform'] = rows

    print("3. fixed T = 1: cumulants of V_T from the first-order rule, switched mean level")
    A_th = kappa * P.d_v()
    rows = []
    for lam in (10, 20, 40, 80):
        Q = lam * Q2
        pi = stationary(Q)
        Ls = [cir_generator(P, kappa, t, xi, 0.0) for t in th]
        Lavg = cir_generator(P, kappa, pi @ th, xi, 0.0)
        K0 = gk(Q2, th, th)
        D = K0 * (A_th @ A_th)
        a, b = zip(*[first_order_expectation(Lavg, D, 1.0, P.monomial(0, k, 0), P, v0) for k in (1, 2, 3, 4)])
        a, b = np.array(a), np.array(b)
        pred = cumulants(a) + cumulant_derivative(a, b) / lam
        exact = cumulants(moments_V(Q, Ls, 1.0, v0))
        avg = cumulants(a)
        rows.append([lam] + list(exact[1:]) + list(pred[1:]) + list(np.abs(exact - pred)[1:]) + list(np.abs(exact - avg)[1:]))
        gkvar = 2 * K0 * int_h2(kappa, 1.0)
        ok &= abs(cumulant_derivative(a, b)[1] - gkvar) < 1e-12
        print(f"   lam={lam:3d}: kappa_2..4 exact {exact[1]:.4e} {exact[2]:.4e} {exact[3]:.4e}; rule residual"
              f" {abs(exact - pred)[1]:.2e} {abs(exact - pred)[2]:.2e} {abs(exact - pred)[3]:.2e}; averaged model residual"
              f" {abs(exact - avg)[1]:.2e} {abs(exact - avg)[2]:.2e} {abs(exact - avg)[3]:.2e};"
              f" rule's variance term vs 2K int h^2: {abs(cumulant_derivative(a, b)[1] - gkvar):.1e}")
    r = np.array(rows)
    orders = [math.log2(r[-2, 7 + j] / r[-1, 7 + j]) for j in range(3)]
    orders0 = [math.log2(r[-2, 10 + j] / r[-1, 10 + j]) for j in range(3)]
    print(f"   orders of the rule's residual: {orders[0]:.3f} {orders[1]:.3f} {orders[2]:.3f}; of the averaged model: {orders0[0]:.3f} {orders0[1]:.3f} {orders0[2]:.3f}")
    ok &= all(abs(o - 2) < 0.15 for o in orders) and all(abs(o - 1) < 0.15 for o in orders0)
    out['cir_fixed_T'] = rows

    print("4. kappa and theta both switched, three regimes on the one-way cycle Q_C: product coefficients c = kappa theta and kappa")
    kap3, th3 = np.array([1.5, 2.5, 2.0]), np.array([0.09, 0.02, 0.05])
    c3 = kap3 * th3
    A_c, A_k = P.d_v(), -P.v_d_v()
    rows = []
    for m in (10, 20, 40, 80):
        Q = m * QC
        pi = stationary(Q)
        Ls = [cir_generator(P, k, t, xi, 0.0) for k, t in zip(kap3, th3)]
        cbar, kbar = pi @ c3, pi @ kap3
        Lavg = cir_generator(P, kbar, cbar / kbar, xi, 0.0)
        feats = [c3, kap3]
        K0 = np.array([[gk(QC, f, g) for g in feats] for f in feats])
        As = [A_c, A_k]
        D = sum(K0[j, k] * (As[j] @ As[k]) for j in range(2) for k in range(2))
        Ksym = 0.5 * (K0 + K0.T)
        Dsym = sum(Ksym[j, k] * (As[j] @ As[k]) for j in range(2) for k in range(2))
        exact = cumulants(moments_V(Q, Ls, 1.0, v0))
        res = {}
        for name, DD in (('rule', D), ('symmetric part only', Dsym)):
            a, b = zip(*[first_order_expectation(Lavg, DD, 1.0, P.monomial(0, k, 0), P, v0) for k in (1, 2, 3, 4)])
            a, b = np.array(a), np.array(b)
            res[name] = np.abs(exact - cumulants(a) - cumulant_derivative(a, b) / m)
        rows.append([m] + list(exact) + list(res['rule']) + list(res['symmetric part only']))
        print(f"   m={m:3d}: kappa_1..4 exact {exact[0]:.4e} {exact[1]:.4e} {exact[2]:.4e} {exact[3]:.4e};"
              f" rule residual {res['rule'][0]:.2e} {res['rule'][1]:.2e} {res['rule'][2]:.2e} {res['rule'][3]:.2e};"
              f" symmetric only {res['symmetric part only'][0]:.2e} {res['symmetric part only'][1]:.2e}")
    r = np.array(rows)
    orders = [math.log2(r[-2, 5 + j] / r[-1, 5 + j]) for j in range(4)]
    osym = math.log2(r[-2, 9] / r[-1, 9])
    print(f"   orders of the rule's residual: {' '.join(f'{o:.3f}' for o in orders)}; symmetric part only, mean: {osym:.3f}")
    ok &= all(abs(o - 2) < 0.15 for o in orders) and osym < 1.5
    out['both'] = rows

    print("5. leverage: the same two-state Heston variance, kappa_4 of the log return at T = 1, lam = 10")
    Q = 10 * Q2
    rows = []
    for rho in (0.0, -0.6):
        Ls = [cir_generator(P, kappa, t, xi, rho) for t in th]
        kV = cumulants(moments_V(Q, Ls, 1.0, v0))
        kX = cumulants(moments_X(Q, Ls, 1.0, v0))
        ident = logreturn_from_V(kV)
        rows.append([rho, kX[2], ident[1], kX[3], ident[2]])
        print(f"   rho={rho:+.1f}: kappa_3(X) exact {kX[2]:+.5e} vs identity {ident[1]:+.5e}; kappa_4(X) exact {kX[3]:.5e} vs identity {ident[2]:.5e}")
    ok &= abs(rows[0][3] - rows[0][4]) < 1e-14 and abs(rows[1][3] / rows[1][4] - 1) > 0.05
    out['leverage'] = rows

    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == '__main__':
    main()
