"""Certificate for the composite that matches the fixed-maturity expansion to the long end, uniformly in maturity.

With Lambda(m) the exact principal eigenvalue of m Q0 + diag g_inf, the x-free log bond price with a stationary start is
    L_m(T) = Lambda(m) T + int_0^T (g_bar - g_bar_inf) dt + (1/m) int_0^T (k - k_inf) dt,   k(t) = K0(g~(t), g~(t)),
which is the fixed-T first-order formula with its secular part g_bar_inf T + k_inf T / m replaced by Lambda(m) T.
1. Symmetric two-state chain: sup_T |log(pi . a(T)) - L_m(T)| over T in [0, m^2] falls like m^-2 (proved on the page).
2. The fixed-T formula alone: its error at T = m^2 settles at |mu_3|, the omitted third eigenvalue term, for chain A;
   for the symmetric two-state chain mu_3 = 0 (the eigenvalue is a function of g~^2) and that error is O(1/m) instead.
3. Three-state non-reversible chain A: the same composite, the same order (checked, not proved).
4. Known starting regime nu: log(nu . a(T)) - L_m(T) - log(nu . r_m(T)), with r_m(T) the instantaneous Perron vector of
   m Q0 + diag g(T) normalized by pi . r = 1, is O(m^-2) uniformly; without the correction the error is O(m^-1).
"""
import math
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from three_numbers import stationary, coefficients, int_Bk
from verify_three_numbers import QA, THA, SA, KAPPA, first_order
from long_end import eigen_series

Q2 = np.array([[-1, 1], [1, -1.0]])
TH2, S2 = np.array([0.08, 0.02]), np.array([0.0009, 0.0001])


def g_vec(t, thetas, s, kappa):
    B = (1 - math.exp(-kappa * t)) / kappa
    return -kappa * np.asarray(thetas) * B + 0.5 * np.asarray(s) * B * B


def perron(M):
    w, V = np.linalg.eig(M)
    i = np.argmax(w.real)
    return w[i].real, np.real(V[:, i])


def exact_log_a(Ts, m, Q0, thetas, s, kappa, T0):
    """log(a(T)) componentwise for each T in Ts: a' = (m Q0 + diag g(t)) a, a(0) = 1. Time-dependent part solved to T0,
    then the constant-coefficient tail by a shifted matrix exponential."""
    Q = m * np.asarray(Q0, float)
    n = len(Q)
    rhs = lambda t, a: Q @ a + g_vec(t, thetas, s, kappa) * a
    sol = solve_ivp(rhs, (0, T0), np.ones(n), method='Radau', rtol=1e-12, atol=1e-16, dense_output=True)
    Minf = Q + np.diag(g_vec(T0, thetas, s, kappa))
    lam, _ = perron(Minf)
    out = []
    for T in Ts:
        if T <= T0:
            a = sol.sol(T)
            out.append(np.log(a))
        else:
            a = expm((Minf - lam * np.eye(n)) * (T - T0)) @ sol.sol(T0)
            out.append(np.log(a) + lam * (T - T0))
    return np.array(out)


def composite(Ts, m, Q0, thetas, s, kappa):
    pi = stationary(Q0)
    ginf = g_vec(1e9, thetas, s, kappa)
    Lam, _ = perron(m * np.asarray(Q0, float) + np.diag(ginf))
    c2, c3, c4 = coefficients(Q0, thetas, s, kappa)
    kinf = c2 / kappa ** 2 + c3 / kappa ** 3 + c4 / kappa ** 4
    return np.array([first_order(T, m * np.asarray(Q0, float), thetas, s, kappa) + T * (Lam - pi @ ginf - kinf / m) for T in Ts])


def grid(m, T0):
    return np.concatenate([np.linspace(0.02, T0, 400), np.geomspace(T0, max(T0 + 1, m * m), 60)[1:]])


def run(Q0, thetas, s, kappa, T0, label):
    ok = True
    pi = stationary(Q0)
    mu3 = eigen_series(Q0, g_vec(1e9, thetas, s, kappa), 3)[3]
    n = len(pi)
    scales = [10, 20, 40, 80]
    errs, errs_fixed, errs_known, errs_known_raw = [], [], [], []
    print(f"   {label}")
    for m in scales:
        Ts = grid(m, T0)
        la = exact_log_a(Ts, m, Q0, thetas, s, kappa, T0)
        stat = np.array([math.log(pi @ np.exp(row)) for row in la])
        L = composite(Ts, m, Q0, thetas, s, kappa)
        err = np.abs(stat - L)
        errs.append(err.max())
        errs_fixed.append(abs(stat[-1] - first_order(Ts[-1], m * np.asarray(Q0, float), thetas, s, kappa)))
        # known start: every regime, correction log(nu . r_m(T))
        worst, worst_raw = 0.0, 0.0
        for i in range(n):
            corr = []
            for T in Ts:
                _, r = perron(m * np.asarray(Q0, float) + np.diag(g_vec(T, thetas, s, kappa)))
                r = r / (pi @ r)
                corr.append(math.log(r[i]))
            worst = max(worst, np.abs(la[:, i] - L - np.array(corr)).max())
            worst_raw = max(worst_raw, np.abs(la[:, i] - L).max())
        errs_known.append(worst); errs_known_raw.append(worst_raw)
        iT = int(np.argmax(err))
        print(f"     m={m:3d}  sup|log(pi.a) - L| = {err.max():.3e} (at T = {Ts[iT]:.3g}, grid to T = {Ts[-1]:.0f});"
              f" fixed-T formula at T = m^2: {errs_fixed[-1]:.3e};"
              f" known start: {worst:.3e} with log(nu.r), {worst_raw:.3e} without")
    o = [math.log2(errs[i] / errs[i + 1]) for i in range(3)]
    ok_ = [math.log2(errs_known[i] / errs_known[i + 1]) for i in range(3)]
    or_ = [math.log2(errs_known_raw[i] / errs_known_raw[i + 1]) for i in range(3)]
    print(f"     orders: stationary composite {o[-1]:.3f}, known start with correction {ok_[-1]:.3f}, without {or_[-1]:.3f};"
          f" fixed-T formula at T = m^2 stays {min(errs_fixed):.2e} to {max(errs_fixed):.2e}")
    ok &= abs(o[-1] - 2) < 0.15 and abs(ok_[-1] - 2) < 0.2 and abs(or_[-1] - 1) < 0.15
    if abs(mu3) > 1e-12:
        print(f"     third eigenvalue coefficient mu_3 = {mu3:.4e}: the fixed-T formula at T = m^2 is off by it, ratio {errs_fixed[-1] / abs(mu3):.4f}")
        ok &= abs(errs_fixed[-1] / abs(mu3) - 1) < 0.05
    else:
        print(f"     mu_3 = {mu3:.1e}: symmetry removes the T/m^2 term; at T = m^2 the fixed-T formula is left with the transient and T mu_4 / m^3, both small here")
        ok &= abs(math.log2(errs_fixed[-2] / errs_fixed[-1]) - 2) < 0.2
    return ok, dict(scales=scales, composite=errs, fixed=errs_fixed, known=errs_known, known_raw=errs_known_raw, mu3=mu3)


def main():
    ok, out = True, {}
    print("1-2. symmetric two-state chain, kappa = 0.5")
    good, out['two_state'] = run(Q2, TH2, S2, 0.5, 80.0, "theta = (0.08, 0.02), s = (0.0009, 0.0001)")
    ok &= good
    print("3-4. three-state non-reversible chain A of the yield-curve page, kappa = 1")
    good, out['chain_A'] = run(QA, THA, SA, KAPPA, 40.0, "Q_A, theta = (0.08, 0.04, 0.01), s = (0.003, 0.0015, 0.0005)")
    ok &= good
    import json, os
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uniform_long_end.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
