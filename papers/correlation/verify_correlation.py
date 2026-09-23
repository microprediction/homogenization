"""Certificate for regime-switching correlation.

Two assets; a two-state chain, calm (correlation 0.2) and crisis (correlation 0.8), calm -> crisis at rate a = 4/3
and back at rate b = 4 (crises last three months on average and occupy a quarter of the time), sped up by m.

Part 1, pricing (volatilities fixed, only the correlation switches):
 1. the exact law of the time spent in crisis (Bessel density) against its known mean and variance;
 2. exchange option: the exact price E[Margrabe(integrated exchange variance)] by quadrature over that law, against
    Lewis's Fourier formula with the matrix-exponential characteristic function, and against chain-path Monte Carlo;
    the first-order rule (Margrabe at the averaged exchange variance plus the volga term) converges at second order;
    implied correlations from the exact price, from the first-order price and from the closed-form parabola;
 3. spread option (S1 - S2 - K)^+ by Gauss-Hermite quadrature at fixed correlation: exact E[C(rho_hat)] against the
    rule C(rho_bar) + (K_rhorho / T) C_rhorho, second-order convergence, Monte Carlo, implied correlations.
Part 2, physical measure (drifts, volatilities and correlation co-switch; bear regime: low drifts, high correlation):
 4. second and third cumulants of the log returns from the rule against the exact cumulant generating function
    (Cauchy formula on exp(t (Q + diag g))), second-order convergence, a cross-check with the engine's ODE solver,
    and exact-in-law Monte Carlo;
 5. the third cumulant of a portfolio, 6 t K(w'mu, w'c w), and the part due to the correlation switch.
Writes results.json for the page. Ends with PASS or FAIL.
"""
import json, math, os, sys
import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'fast-switching'))
sys.path.insert(0, os.path.join(HERE, '..', 'general'))
from correlation import (pi2, K2, Q2, occupation_nodes, simulate_occupation, margrabe, margrabe_greeks, exch_var,
                         margrabe_first_order, implied_corr_margrabe, implied_corr_parabola, margrabe_fourier,
                         spread_price, spread_rho_derivs, implied_corr_spread, cumulants_first_order,
                         cumulants_exact, simulate_returns, cov_entries)

A, B = 4 / 3, 4.0
RHO = [0.2, 0.8]
SPEEDS = [1, 2, 4, 8]
T = 1.0
# part 1
S1, S2, SIG1, SIG2, R = 100.0, 100.0, 0.30, 0.20, 0.02
QS = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6]
KS = [-20, -10, 0, 10, 20, 30, 40]
# part 2
MU1, MU2 = [0.10, -0.20], [0.08, -0.25]
VOL1, VOL2 = [0.15, 0.30], [0.12, 0.25]


def rate(e):
    return math.log(e[-2] / e[-1]) / math.log(2)


def main():
    ok, out = True, {}
    rng = np.random.default_rng(20260923)

    print("1. occupation time of the crisis state")
    rows = []
    for m in SPEEDS:
        a, b = m * A, m * B
        v, w = occupation_nodes(T, a, b)
        p = pi2(a, b)
        lam = a + b
        var = 2 * p[0] * p[1] / lam * (T - (1 - math.exp(-lam * T)) / lam)
        mv = w @ v
        e = max(abs(w.sum() - 1), abs(mv - p[1] * T), abs(w @ (v - mv) ** 2 - var))
        rows.append(e)
        print(f"   m={m}: total mass {w.sum():.12f}, mean {mv:.10f} (want {p[1]*T:.10f}), variance {w @ (v-mv)**2:.10f} (want {var:.10f})")
    ok &= max(rows) < 1e-10

    print("2. exchange option (S1 - q S2)^+")
    out['exchange'] = {'speeds': SPEEDS, 'q': QS, 'rows': []}
    errs, errs0 = [], []
    for m in SPEEDS:
        a, b = m * A, m * B
        p = pi2(a, b)
        rb, Krr = p @ RHO, K2(RHO, RHO, a, b)
        v, w = occupation_nodes(T, a, b)
        rh = RHO[0] + (RHO[1] - RHO[0]) * v / T
        row = {'m': m, 'rhobar': rb, 'Krr': Krr, 'Kss': 4 * SIG1 ** 2 * SIG2 ** 2 * Krr, 'exact': [], 'first': [],
               'averaged': [], 'iv_exact': [], 'iv_first': [], 'iv_parabola': []}
        for q in QS:
            ex = float(w @ margrabe(S1, q * S2, exch_var(SIG1, SIG2, rh) * T))
            fo = margrabe_first_order(S1, q * S2, T, SIG1, SIG2, rb, Krr)
            av = float(margrabe(S1, q * S2, exch_var(SIG1, SIG2, rb) * T))
            row['exact'].append(ex); row['first'].append(fo); row['averaged'].append(av)
            row['iv_exact'].append(implied_corr_margrabe(ex, S1, q * S2, T, SIG1, SIG2))
            row['iv_first'].append(implied_corr_margrabe(fo, S1, q * S2, T, SIG1, SIG2))
            row['iv_parabola'].append(implied_corr_parabola(math.log(S1 / (q * S2)), T, SIG1, SIG2, rb, Krr))
        row['err_first'] = max(abs(x - y) for x, y in zip(row['exact'], row['first']))
        row['err_avg'] = max(abs(x - y) for x, y in zip(row['exact'], row['averaged']))
        row['err_iv_first'] = max(abs(x - y) for x, y in zip(row['iv_exact'], row['iv_first']))
        row['err_iv_parabola'] = max(abs(x - y) for x, y in zip(row['iv_exact'], row['iv_parabola']))
        errs.append(row['err_first']); errs0.append(row['err_avg'])
        out['exchange']['rows'].append(row)
        print(f"   m={m}: rho_bar {rb:.3f}, K_rhorho {Krr:.6f}; max price error averaged {row['err_avg']:.2e}, first order "
              f"{row['err_first']:.2e}; implied-correlation error: first-order price {row['err_iv_first']:.1e}, parabola {row['err_iv_parabola']:.1e}")
    r1, r0 = rate(errs), rate(errs0)
    print(f"   rates over the last doubling: first order {r1:.2f} (want 2), averaged {r0:.2f} (want 1)")
    ok &= r1 > 1.8 and 0.8 < r0 < 1.2
    out['exchange']['rate_first'], out['exchange']['rate_avg'] = r1, r0
    # Fourier (matrix exponential) and Monte Carlo checks at the slowest speed
    a, b = A, B
    v, w = occupation_nodes(T, a, b)
    rh = RHO[0] + (RHO[1] - RHO[0]) * v / T
    fchk = []
    for q in (0.7, 1.0, 1.4):
        ex = float(w @ margrabe(S1, q * S2, exch_var(SIG1, SIG2, rh) * T))
        fr = margrabe_fourier(S1, q * S2, T, SIG1, SIG2, RHO, Q2(a, b), n=600, umax=120.0)
        fchk.append([q, ex, fr])
        print(f"   q={q}: occupation quadrature {ex:.10f}, Fourier with matrix exponential {fr:.10f}")
    ok &= max(abs(x[1] - x[2]) for x in fchk) < 1e-8
    out['exchange']['fourier'] = fchk
    tau = simulate_occupation(T, a, b, 1_000_000, rng)
    rh_mc = RHO[0] + (RHO[1] - RHO[0]) * tau / T
    mc = []
    for q in (0.7, 1.0, 1.4):
        c = margrabe(S1, q * S2, exch_var(SIG1, SIG2, rh_mc) * T)
        ex = float(w @ margrabe(S1, q * S2, exch_var(SIG1, SIG2, rh) * T))
        mc.append([q, ex, float(c.mean()), float(c.std() / math.sqrt(len(c)))])
        print(f"   q={q}: Monte Carlo over chain paths {c.mean():.5f} +- {c.std()/math.sqrt(len(c)):.5f}, exact {ex:.5f}")
    ok &= all(abs(x[1] - x[2]) < 4 * x[3] for x in mc)
    out['exchange']['mc'] = mc

    print("3. spread option (S1 - S2 - K)^+")
    out['spread'] = {'speeds': SPEEDS, 'K': KS, 'rows': []}
    errs, errs0 = [], []
    for m in SPEEDS:
        a, b = m * A, m * B
        p = pi2(a, b)
        rb, Krr = p @ RHO, K2(RHO, RHO, a, b)
        v, w = occupation_nodes(T, a, b)
        rh = RHO[0] + (RHO[1] - RHO[0]) * v / T
        row = {'m': m, 'exact': [], 'first': [], 'averaged': [], 'iv_exact': [], 'iv_first': [], 'crr': []}
        for K in KS:
            ex = float(w @ spread_price(S1, S2, K, T, R, SIG1, SIG2, rh))
            c0 = spread_price(S1, S2, K, T, R, SIG1, SIG2, rb)
            _, crr = spread_rho_derivs(S1, S2, K, T, R, SIG1, SIG2, rb)
            fo = c0 + Krr / T * crr
            row['exact'].append(ex); row['first'].append(fo); row['averaged'].append(c0); row['crr'].append(crr)
            row['iv_exact'].append(implied_corr_spread(ex, S1, S2, K, T, R, SIG1, SIG2))
            row['iv_first'].append(implied_corr_spread(fo, S1, S2, K, T, R, SIG1, SIG2))
        row['err_first'] = max(abs(x - y) for x, y in zip(row['exact'], row['first']))
        row['err_avg'] = max(abs(x - y) for x, y in zip(row['exact'], row['averaged']))
        row['err_iv_first'] = max(abs(x - y) for x, y in zip(row['iv_exact'], row['iv_first']))
        errs.append(row['err_first']); errs0.append(row['err_avg'])
        out['spread']['rows'].append(row)
        print(f"   m={m}: max price error averaged {row['err_avg']:.2e}, first order {row['err_first']:.2e}; "
              f"implied-correlation error {row['err_iv_first']:.1e}")
    r1, r0 = rate(errs), rate(errs0)
    print(f"   rates over the last doubling: first order {r1:.2f} (want 2), averaged {r0:.2f} (want 1)")
    ok &= r1 > 1.8 and 0.8 < r0 < 1.2
    out['spread']['rate_first'], out['spread']['rate_avg'] = r1, r0
    # the constant-correlation quadrature against its own limit K = 0 (Margrabe) and chain-path Monte Carlo
    mg = float(margrabe(S1, S2, exch_var(SIG1, SIG2, 0.35) * T))
    sp = spread_price(S1, S2, 0.0, T, R, SIG1, SIG2, 0.35)
    print(f"   K = 0 against Margrabe at rho = 0.35: quadrature {sp:.12f}, Margrabe {mg:.12f}")
    ok &= abs(sp - mg) < 1e-9
    a, b = A, B
    v, w = occupation_nodes(T, a, b)
    rh = RHO[0] + (RHO[1] - RHO[0]) * v / T
    tau = simulate_occupation(T, a, b, 200_000, rng)
    rh_mc = RHO[0] + (RHO[1] - RHO[0]) * tau / T
    mc = []
    for K in (0, 20):
        c = np.concatenate([spread_price(S1, S2, K, T, R, SIG1, SIG2, chunk) for chunk in np.array_split(rh_mc, 20)])
        ex = float(w @ spread_price(S1, S2, K, T, R, SIG1, SIG2, rh))
        mc.append([K, ex, float(c.mean()), float(c.std() / math.sqrt(len(c)))])
        print(f"   K={K}: Monte Carlo over chain paths {c.mean():.5f} +- {c.std()/math.sqrt(len(c)):.5f}, exact {ex:.5f}")
    ok &= all(abs(x[1] - x[2]) < 4 * x[3] for x in mc)
    # full Monte Carlo of the terminal prices (no conditioning), one strike
    n = 2_000_000
    tau = simulate_occupation(T, a, b, n, rng)
    rbar_path = RHO[0] + (RHO[1] - RHO[0]) * tau / T
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    x2 = (R - SIG2 ** 2 / 2) * T + SIG2 * math.sqrt(T) * z2
    x1 = (R - SIG1 ** 2 / 2) * T + SIG1 * math.sqrt(T) * (rbar_path * z2 + np.sqrt(1 - rbar_path ** 2) * z1)
    pay = math.exp(-R * T) * np.maximum(S1 * np.exp(x1) - S2 * np.exp(x2) - 10.0, 0)
    ex = float(w @ spread_price(S1, S2, 10.0, T, R, SIG1, SIG2, rh))
    mc.append([10, ex, float(pay.mean()), float(pay.std() / math.sqrt(n))])
    print(f"   K=10: Monte Carlo of terminal prices {pay.mean():.4f} +- {pay.std()/math.sqrt(n):.4f}, exact {ex:.4f}")
    ok &= abs(pay.mean() - ex) < 4 * pay.std() / math.sqrt(n)
    out['spread']['mc'] = mc

    print("4. cumulants of log returns under co-switching")
    out['cumulants'] = {'speeds': SPEEDS + [16], 'rows': []}
    keys = ['k11', 'k22', 'k12', 'k111', 'k222', 'k112', 'k122']
    e112, e12 = [], []
    for m in SPEEDS + [16]:
        a, b = m * A, m * B
        fo = cumulants_first_order(T, a, b, MU1, MU2, VOL1, VOL2, RHO)
        ex = cumulants_exact(T, Q2(a, b), MU1, MU2, VOL1, VOL2, RHO)
        c11, c22, c12 = cov_entries(VOL1, VOL2, RHO)
        p = pi2(a, b)
        avg = {'k11': T * p @ c11, 'k22': T * p @ c22, 'k12': T * p @ c12, 'k111': 0, 'k222': 0, 'k112': 0, 'k122': 0}
        out['cumulants']['rows'].append({'m': m, 'first': {k: fo[k] for k in keys}, 'exact': {k: ex[k] for k in keys},
                                         'averaged': avg, 'K_mu1_c12': K2(MU1, c12, a, b), 'K_mu2_c11': K2(MU2, c11, a, b),
                                         'K_mu2_c12': K2(MU2, c12, a, b), 'K_mu1_c22': K2(MU1, c22, a, b)})
        e112.append(max(abs(fo[k] - ex[k]) for k in ['k111', 'k222', 'k112', 'k122']))
        e12.append(max(abs(fo[k] - ex[k]) for k in ['k11', 'k22', 'k12']))
        print(f"   m={m:2d}: kappa_112 rule {fo['k112']:.4e} exact {ex['k112']:.4e};  kappa_122 rule {fo['k122']:.4e} "
              f"exact {ex['k122']:.4e};  kappa_12 rule {fo['k12']:.5f} exact {ex['k12']:.5f}")
    r3, r2 = rate(e112), rate(e12)
    print(f"   rates over the last doubling: third cumulants {r3:.2f}, second cumulants {r2:.2f} (want 2)")
    ok &= r3 > 1.8 and r2 > 1.8
    out['cumulants']['rate3'], out['cumulants']['rate2'] = r3, r2
    out['cumulants']['err3'], out['cumulants']['err2'] = e112, e12
    # cross-check of the exact generating function with the engine's high-precision ODE solver
    from fastswitch import numerical_a, ExpSum
    th1, th2 = 0.7, -0.4
    c11, c22, c12 = cov_entries(VOL1, VOL2, RHO)
    g = [th1 * MU1[i] + th2 * MU2[i] + 0.5 * (th1 ** 2 * c11[i] + 2 * th1 * th2 * c12[i] + th2 ** 2 * c22[i]) for i in range(2)]
    Q = Q2(A, B)
    via_expm = pi2(A, B) @ expm(T * (Q + np.diag(g))) @ np.ones(2)
    via_ode = pi2(A, B) @ np.array(numerical_a(T, Q, [ExpSum.const(gi) for gi in g]))
    print(f"   generating function at theta = (0.7, -0.4): matrix exponential {via_expm:.14f}, engine ODE {via_ode:.14f}")
    ok &= abs(via_expm - via_ode) < 1e-12
    # Monte Carlo, exact in law
    mcrows = []
    for m in (1, 4):
        n = 4_000_000
        x1, x2 = simulate_returns(T, m * A, m * B, MU1, MU2, VOL1, VOL2, RHO, n, rng)
        d1, d2 = x1 - x1.mean(), x2 - x2.mean()
        ex = cumulants_exact(T, Q2(m * A, m * B), MU1, MU2, VOL1, VOL2, RHO)
        fo = cumulants_first_order(T, m * A, m * B, MU1, MU2, VOL1, VOL2, RHO)
        for name, s in [('k12', d1 * d2), ('k112', d1 * d1 * d2), ('k122', d1 * d2 * d2)]:
            est, se = float(s.mean()), float(s.std() / math.sqrt(n))
            mcrows.append([m, name, est, se, ex[name], fo[name]])
            print(f"   m={m}: {name} Monte Carlo {est:.4e} +- {se:.1e}, exact {ex[name]:.4e}, rule {fo[name]:.4e}")
            ok &= abs(est - ex[name]) < 4 * se
        if m == 1:
            # exceedance correlations (both above / both below their means), against a Gaussian with the same covariance
            def exc(u1, u2):
                dn = (u1 < 0) & (u2 < 0); up = (u1 > 0) & (u2 > 0)
                return float(np.corrcoef(u1[dn], u2[dn])[0, 1]), float(np.corrcoef(u1[up], u2[up])[0, 1])
            s1_, s2_ = d1 / d1.std(), d2 / d2.std()
            dn, up = exc(s1_, s2_)
            rr = float(np.corrcoef(d1, d2)[0, 1])
            g1 = rng.standard_normal(n); g2 = rr * g1 + math.sqrt(1 - rr * rr) * rng.standard_normal(n)
            gdn, gup = exc(g1, g2)
            out['exceedance'] = {'corr': rr, 'down': dn, 'up': up, 'gauss_down': gdn, 'gauss_up': gup}
            print(f"   exceedance correlation at the mean: downside {dn:.3f}, upside {up:.3f}; Gaussian with the same "
                  f"correlation {rr:.3f}: {gdn:.3f}, {gup:.3f}")
            ok &= dn > up + 0.05
    out['cumulants']['mc'] = mcrows

    print("5. third cumulant of the equal-weight portfolio")
    w_ = np.array([0.5, 0.5])
    prow = []
    for m in SPEEDS + [16]:
        a, b = m * A, m * B
        c11, c22, c12 = cov_entries(VOL1, VOL2, RHO)
        wm = w_[0] * np.array(MU1) + w_[1] * np.array(MU2)
        wc = w_[0] ** 2 * c11 + w_[1] ** 2 * c22 + 2 * w_[0] * w_[1] * c12
        rule = 6 * T * K2(wm, wc, a, b)
        corr_part = 6 * T * 2 * w_[0] * w_[1] * K2(wm, c12, a, b)
        ex = cumulants_exact(T, Q2(a, b), MU1, MU2, VOL1, VOL2, RHO)
        exact = (w_[0] ** 3 * ex['k111'] + 3 * w_[0] ** 2 * w_[1] * ex['k112'] + 3 * w_[0] * w_[1] ** 2 * ex['k122']
                 + w_[1] ** 3 * ex['k222'])
        var = w_[0] ** 2 * ex['k11'] + 2 * w_[0] * w_[1] * ex['k12'] + w_[1] ** 2 * ex['k22']
        prow.append({'m': m, 'rule': rule, 'exact': exact, 'corr_part': corr_part, 'skew_exact': exact / var ** 1.5,
                     'skew_rule': rule / var ** 1.5})
        print(f"   m={m:2d}: kappa_3 rule {rule:.4e}, exact {exact:.4e}; part from the correlation switch {corr_part:.4e}; "
              f"skewness {exact/var**1.5:.3f}")
    ok &= abs(prow[-1]['rule'] - prow[-1]['exact']) < 0.01 * abs(prow[-1]['exact'])
    out['portfolio'] = prow

    json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
