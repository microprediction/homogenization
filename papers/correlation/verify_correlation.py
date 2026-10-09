"""Certificate for regime-switching correlation.

Two assets; a two-state chain, calm (correlation 0.2) and crisis (correlation 0.8), calm -> crisis at rate a = 4/3
and back at rate b = 4 (crises last three months on average and occupy a quarter of the time), sped up by m.

Part 1, pricing (volatilities fixed, only the correlation switches):
 1. the exact law of the time spent in crisis (Bessel density) against its first four cumulants, including
    nonstationary initial laws and regime relabelling;
 2. exchange option: the exact price E[Margrabe(integrated exchange variance)] by quadrature over that law, against
    Lewis's Fourier formula with the matrix-exponential characteristic function, and against chain-path Monte Carlo;
    the first-order rule (Margrabe at the averaged exchange variance plus the volga term) converges at second order;
    implied correlations from the exact price, from the first-order price and from the closed-form parabola;
 3. spread option (S1 - S2 - K)^+ by Gauss-Hermite quadrature at fixed correlation: exact E[C(rho_hat)] against the
    rule C(rho_bar) + (K_rhorho / T) C_rhorho, second-order convergence, Monte Carlo, implied correlations.
Part 2, physical measure (drifts, volatilities and correlation co-switch; bear regime: low drifts, high correlation):
 4. exact finite-rate cumulants of every order from singleton/pair partitions and occupation-time cumulants, with
    orders two through six checked for two states and orders two through five checked for three states against the exact
    cumulant generating function (Cauchy formula on exp(t (Q + diag g))), the finite-state fast-rate hierarchy checked
    against principal-eigenvalue derivatives, plus second-order convergence of the Green--Kubo rule, a cross-check with
    the engine's ODE solver, and exact-in-law Monte Carlo;
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
                         cumulants_two_state_closed, cumulants_exact, simulate_returns, cov_entries,
                         occupation_cumulants, occupation_cumulants_quadrature,
                         occupation_covariance_rate,
                         occupation_third_cumulant_rate,
                         occupation_joint_cumulants_cauchy,
                         occupation_joint_cumulant_bulk_cauchy,
                         gaussian_occupation_cumulant,
                         gaussian_vector_occupation_cumulant)

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
        oc = occupation_cumulants(T, a, b)
        mv = w @ v
        third = w @ (v - mv) ** 3
        fourth = w @ (v - mv) ** 4 - 3 * (w @ (v - mv) ** 2) ** 2
        e = max(abs(w.sum() - 1), abs(mv - p[1] * T), abs(w @ (v - mv) ** 2 - var),
                abs(third - oc['k3']), abs(fourth - oc['k4']))
        rows.append(e)
        print(f"   m={m}: total mass {w.sum():.12f}, mean {mv:.10f} (want {p[1]*T:.10f}), variance "
              f"{w @ (v-mv)**2:.10f} (want {var:.10f}), third cumulant {third:.10f} (want {oc['k3']:.10f}), "
              f"fourth {fourth:.10f} (want {oc['k4']:.10f})")
    ok &= max(rows) < 1e-10

    print("1b. nonstationary occupation cumulants and initial layer")
    nonstationary = []
    relabel_error = 0.0
    for m in (1, 4, 16):
        a, b = m * A, m * B
        for q in (0.0, 0.1, 0.7, 1.0):
            oc = occupation_cumulants(T, a, b, q)
            v, w = occupation_nodes(T, a, b, initial_p1=q)
            mean = w @ v
            var_q = w @ (v - mean) ** 2
            quad = {'k1': mean, 'k2': var_q, 'k3': w @ (v - mean) ** 3,
                    'k4': w @ (v - mean) ** 4 - 3 * var_q ** 2}
            quad_error = max(abs(quad[k] - oc[k]) for k in quad)
            swapped = occupation_cumulants(T, b, a, 1 - q)
            relabel_error = max(relabel_error, abs(oc['k1'] + swapped['k1'] - T),
                                abs(oc['k2'] - swapped['k2']), abs(oc['k3'] + swapped['k3']),
                                abs(oc['k4'] - swapped['k4']))
            closed = cumulants_two_state_closed(T, a, b, MU1, MU2, VOL1, VOL2, RHO, q)
            exact = cumulants_exact(T, Q2(a, b), MU1, MU2, VOL1, VOL2, RHO,
                                    r=0.15, N=24, initial_p1=q)
            tensor_error = max(abs(closed[k] - exact[k]) for k in exact)
            nonstationary.append({'m': m, 'initial_p1': q, 'occupation': oc,
                                  'quadrature_error': quad_error, 'tensor_error': tensor_error,
                                  'k112': closed['k112']})
            print(f"   m={m:2d}, q={q:.1f}: occupation error {quad_error:.1e}, tensor error {tensor_error:.1e}, "
                  f"kappa_112 {closed['k112']:.6e}")
    frozen = occupation_cumulants(T, 3e-9, 7e-9, 0.8)
    frozen_target = {'k1': 0.8 * T, 'k2': 0.8 * 0.2 * T ** 2,
                     'k3': 0.8 * 0.2 * (1 - 2 * 0.8) * T ** 3,
                     'k4': 0.8 * 0.2 * (1 - 6 * 0.8 * 0.2) * T ** 4}
    frozen_error = max(abs(frozen[k] - frozen_target[k]) for k in frozen)
    print(f"   relabelling error {relabel_error:.1e}; near-frozen Bernoulli error {frozen_error:.1e}")
    ok &= max(row['quadrature_error'] for row in nonstationary) < 1e-10
    ok &= max(row['tensor_error'] for row in nonstationary) < 1e-10
    ok &= relabel_error < 1e-13 and frozen_error < 1e-8
    out['nonstationary'] = {'rows': nonstationary, 'relabel_error': relabel_error,
                            'frozen_error': frozen_error, 'frozen': frozen}

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
        closed = cumulants_two_state_closed(T, a, b, MU1, MU2, VOL1, VOL2, RHO)
        c11, c22, c12 = cov_entries(VOL1, VOL2, RHO)
        p = pi2(a, b)
        avg = {'k11': T * p @ c11, 'k22': T * p @ c22, 'k12': T * p @ c12, 'k111': 0, 'k222': 0, 'k112': 0, 'k122': 0}
        closed_err = max(abs(closed[k] - ex[k]) for k in keys)
        out['cumulants']['rows'].append({'m': m, 'first': {k: fo[k] for k in keys}, 'exact': {k: ex[k] for k in keys},
                                         'closed': {k: closed[k] for k in keys}, 'closed_error': closed_err,
                                         'occupation': closed['occupation'],
                                         'averaged': avg, 'K_mu1_c12': K2(MU1, c12, a, b), 'K_mu2_c11': K2(MU2, c11, a, b),
                                         'K_mu2_c12': K2(MU2, c12, a, b), 'K_mu1_c22': K2(MU1, c22, a, b)})
        e112.append(max(abs(fo[k] - ex[k]) for k in ['k111', 'k222', 'k112', 'k122']))
        e12.append(max(abs(fo[k] - ex[k]) for k in ['k11', 'k22', 'k12']))
        print(f"   m={m:2d}: kappa_112 rule {fo['k112']:.4e} exact {ex['k112']:.4e};  kappa_122 rule {fo['k122']:.4e} "
              f"exact {ex['k122']:.4e};  kappa_12 rule {fo['k12']:.5f} exact {ex['k12']:.5f}; closed error {closed_err:.1e}")
        ok &= closed_err < 2e-11
    r3, r2 = rate(e112), rate(e12)
    print(f"   rates over the last doubling: third cumulants {r3:.2f}, second cumulants {r2:.2f} (want 2)")
    ok &= r3 > 1.8 and r2 > 1.8
    out['cumulants']['rate3'], out['cumulants']['rate2'] = r3, r2
    out['cumulants']['err3'], out['cumulants']['err2'] = e112, e12
    out['cumulants']['max_closed_error'] = max(row['closed_error'] for row in out['cumulants']['rows'])

    print("4b. exact fourth-cumulant tensor and leading variance-switching term")
    keys4 = ['k1111', 'k1112', 'k1122', 'k1222', 'k2222']
    fourth_rows, fourth_errors = [], []
    c11, c22, c12 = cov_entries(VOL1, VOL2, RHO)
    dc11, dc22, dc12 = c11[1] - c11[0], c22[1] - c22[0], c12[1] - c12[0]
    for m in SPEEDS + [16]:
        a, b = m * A, m * B
        p = pi2(a, b)
        lead_scale = 2 * p[0] * p[1] * T / (a + b)
        lead = {
            'k1111': 3 * lead_scale * dc11 ** 2,
            'k1112': 3 * lead_scale * dc11 * dc12,
            'k1122': lead_scale * (dc11 * dc22 + 2 * dc12 ** 2),
            'k1222': 3 * lead_scale * dc22 * dc12,
            'k2222': 3 * lead_scale * dc22 ** 2,
        }
        closed = cumulants_two_state_closed(T, a, b, MU1, MU2, VOL1, VOL2, RHO)
        exact = cumulants_exact(T, Q2(a, b), MU1, MU2, VOL1, VOL2, RHO, r=0.15, N=24)
        closed_error = max(abs(closed[k] - exact[k]) for k in keys4)
        lead_error = max(abs(closed[k] - lead[k]) for k in keys4)
        fourth_errors.append(lead_error)
        fourth_rows.append({'m': m, 'closed': {k: closed[k] for k in keys4},
                            'matrix_exponential': {k: exact[k] for k in keys4}, 'leading': lead,
                            'closed_error': closed_error, 'leading_error': lead_error})
        print(f"   m={m:2d}: kappa_1122 exact {closed['k1122']:.6e}, leading {lead['k1122']:.6e}; "
              f"tensor error {closed_error:.1e}, leading error {lead_error:.1e}")
        ok &= closed_error < 1e-10
    fourth_rate = rate(fourth_errors)
    print(f"   fourth-cumulant leading-error rate {fourth_rate:.2f} (want 2)")
    ok &= fourth_rate > 1.8
    out['fourth_cumulants'] = {'rows': fourth_rows, 'leading_error_rate': fourth_rate,
                               'max_closed_error': max(row['closed_error'] for row in fourth_rows)}

    print("4c. all-order singleton/pair theorem through sixth order")
    regime_means = np.array([MU1, MU2]).T
    regime_covariances = np.array([
        [[c11[z], c12[z]], [c12[z], c22[z]]] for z in range(2)
    ])
    all_order_errors, higher_order_errors = [], []
    examples = {}
    for initial_p1 in (None, 0.0, 0.35, 1.0):
        occupation = occupation_cumulants_quadrature(
            T, A, B, max_order=6, initial_p1=initial_p1, n=160
        )
        matrix = cumulants_exact(
            T, Q2(A, B), MU1, MU2, VOL1, VOL2, RHO,
            r=0.35, N=48, initial_p1=initial_p1, max_order=6
        )
        for order in range(2, 7):
            for number_of_twos in range(order + 1):
                indices = ((0,) * (order - number_of_twos)
                           + (1,) * number_of_twos)
                key = ('k' + '1' * (order - number_of_twos)
                       + '2' * number_of_twos)
                partition_value = gaussian_occupation_cumulant(
                    indices, T, regime_means, regime_covariances, occupation
                )
                error = abs(partition_value - matrix[key])
                all_order_errors.append(error)
                if order >= 5:
                    higher_order_errors.append(error)
        if initial_p1 is None:
            examples = {'k11112': matrix['k11112'],
                        'k111222': matrix['k111222']}
    all_order_error = max(all_order_errors)
    print(f"   100 tensor entries (52 of orders five/six), four initial laws: "
          f"max error {all_order_error:.1e}")
    print(f"   stationary examples: kappa_11112 {examples['k11112']:.6e}, "
          f"kappa_111222 {examples['k111222']:.6e}")
    ok &= all_order_error < 3e-12
    out['all_order_cumulants'] = {
        'orders': [2, 3, 4, 5, 6],
        'initial_p1': ['stationary', 0.0, 0.35, 1.0],
        'checked_entries': len(all_order_errors),
        'checked_higher_entries': len(higher_order_errors),
        'max_matrix_exponential_error': all_order_error,
        'examples': examples,
    }

    print("4d. vector-occupation theorem for a three-state chain")
    q3 = np.array([[-3.0, 2.0, 1.0],
                   [1.0, -4.0, 3.0],
                   [2.0, 1.0, -3.0]])
    mu1_3 = np.array([0.10, -0.15, 0.03])
    mu2_3 = np.array([0.05, 0.08, -0.20])
    vol1_3 = np.array([0.15, 0.28, 0.18])
    vol2_3 = np.array([0.12, 0.22, 0.30])
    rho_3 = np.array([0.10, 0.75, -0.35])
    c11_3, c22_3, c12_3 = cov_entries(vol1_3, vol2_3, rho_3)
    means_3 = np.column_stack([mu1_3, mu2_3])
    covariances_3 = np.array([
        [[c11_3[z], c12_3[z]], [c12_3[z], c22_3[z]]]
        for z in range(3)
    ])
    starts_3 = [None, np.array([1.0, 0.0, 0.0]),
                np.array([0.2, 0.3, 0.5])]
    vector_errors, reference_errors, vector_examples = [], [], {}
    for initial in starts_3:
        occupation_3 = occupation_joint_cumulants_cauchy(
            T, q3, max_order=5, r=0.32, N=32, initial=initial
        )
        matrix_3 = cumulants_exact(
            T, q3, mu1_3, mu2_3, vol1_3, vol2_3, rho_3,
            r=0.32, N=48, max_order=5, initial=initial
        )
        for order in range(2, 6):
            for number_of_twos in range(order + 1):
                indices = ((0,) * (order - number_of_twos)
                           + (1,) * number_of_twos)
                key = ('k' + '1' * (order - number_of_twos)
                       + '2' * number_of_twos)
                partition_value = gaussian_vector_occupation_cumulant(
                    indices, T, means_3[0], covariances_3[0],
                    means_3[1:] - means_3[0],
                    covariances_3[1:] - covariances_3[0], occupation_3
                )
                vector_errors.append(abs(partition_value - matrix_3[key]))
        if initial is None:
            vector_examples['k11122'] = matrix_3['k11122']
            stationary_matrix_3 = matrix_3
    for permutation in (np.array([1, 0, 2]), np.array([2, 0, 1])):
        permuted_q = q3[np.ix_(permutation, permutation)]
        permuted_means = means_3[permutation]
        permuted_covariances = covariances_3[permutation]
        permuted_occupation = occupation_joint_cumulants_cauchy(
            T, permuted_q, max_order=5, r=0.32, N=32
        )
        for order in range(2, 6):
            for number_of_twos in range(order + 1):
                indices = ((0,) * (order - number_of_twos)
                           + (1,) * number_of_twos)
                key = ('k' + '1' * (order - number_of_twos)
                       + '2' * number_of_twos)
                partition_value = gaussian_vector_occupation_cumulant(
                    indices, T, permuted_means[0], permuted_covariances[0],
                    permuted_means[1:] - permuted_means[0],
                    permuted_covariances[1:] - permuted_covariances[0],
                    permuted_occupation
                )
                reference_errors.append(
                    abs(partition_value - stationary_matrix_3[key])
                )
    vector_error = max(vector_errors)
    reference_error = max(reference_errors)
    print(f"   54 tensor entries through order five, three initial laws: "
          f"max error {vector_error:.1e}")
    print(f"   two alternative reference states: max residual "
          f"{reference_error:.1e}")
    print(f"   stationary kappa_11122 {vector_examples['k11122']:.6e}")
    ok &= vector_error < 4e-13 and reference_error < 4e-13
    out['vector_occupation_cumulants'] = {
        'states': 3,
        'orders': [2, 3, 4, 5],
        'initial_laws': ['stationary', 'state_zero', [0.2, 0.3, 0.5]],
        'checked_entries': len(vector_errors),
        'max_matrix_exponential_error': vector_error,
        'reference_state_checks': len(reference_errors),
        'max_reference_state_error': reference_error,
        'examples': vector_examples,
    }

    print("4e. finite-state fast-rate occupation-cumulant theorem")
    fast_speeds = [1, 2, 4, 8, 16]
    bulk_3 = occupation_joint_cumulant_bulk_cauchy(
        q3, max_order=6, r=0.2, N=32
    )
    stationary_3 = np.linalg.solve(
        np.vstack([q3.T[:-1], np.ones(3)]), np.r_[np.zeros(2), 1.0]
    )
    first_derivative_error = max(
        abs(bulk_3[(color,)] - stationary_3[color + 1])
        for color in range(2)
    )
    covariance_rate_3 = occupation_covariance_rate(q3)
    second_derivative_error = max(abs(
        bulk_3[tuple(sorted((a, b)))] - covariance_rate_3[a, b]
    ) for a in range(2) for b in range(2))
    third_rate_3 = occupation_third_cumulant_rate(q3)
    third_derivative_error = max(abs(
        bulk_3[tuple(sorted((a, b, c)))] - third_rate_3[a, b, c]
    ) for a in range(2) for b in range(2) for c in range(2))
    occupation_rows = []
    magnitude_rates, remainder_rates = {}, {}
    exact_occupations = {}
    for speed in fast_speeds:
        exact_occupations[speed] = occupation_joint_cumulants_cauchy(
            T, speed * q3, max_order=6, r=0.32, N=32
        )
    for order in range(2, 7):
        order_keys = [key for key in bulk_3 if len(key) == order]
        magnitudes, remainders = [], []
        for speed in fast_speeds:
            exact_occupation = exact_occupations[speed]
            magnitudes.append(max(abs(exact_occupation[key])
                                  for key in order_keys))
            remainders.append(max(abs(
                exact_occupation[key]
                - T * speed ** (1 - order) * bulk_3[key]
            ) for key in order_keys))
        magnitude_rates[order] = rate(magnitudes)
        remainder_rates[order] = rate(remainders)
        occupation_rows.append({
            'order': order, 'magnitudes': magnitudes,
            'bulk_remainders': remainders,
            'magnitude_rate': magnitude_rates[order],
            'bulk_remainder_rate': remainder_rates[order],
        })
        print(f"   occupation order {order}: magnitude rate "
              f"{magnitude_rates[order]:.3f} (want {order - 1}), "
              f"bulk-remainder rate {remainder_rates[order]:.3f} "
              f"(want {order})")
    return_rates = {}
    return_rows = []
    for speed in fast_speeds:
        tensor = cumulants_exact(
            T, speed * q3, mu1_3, mu2_3, vol1_3, vol2_3, rho_3,
            r=0.32, N=48, max_order=6
        )
        return_rows.append({'m': speed, 'tensor': tensor})
    for order in range(3, 7):
        order_keys = [
            'k' + '1' * (order - number_of_twos) + '2' * number_of_twos
            for number_of_twos in range(order + 1)
        ]
        magnitudes = [max(abs(row['tensor'][key]) for key in order_keys)
                      for row in return_rows]
        return_rates[order] = rate(magnitudes)
        print(f"   return order {order}: magnitude rate "
              f"{return_rates[order]:.3f} "
              f"(want {math.ceil(order / 2) - 1})")
    leading_occupation = {
        tuple(sorted((a, b))): covariance_rate_3[a, b]
        for a in range(2) for b in range(2)
    }
    for order in range(3, 5):
        for colors in np.ndindex(*(2,) * order):
            leading_occupation[tuple(sorted(colors))] = 0.0
    leading_error_rates = {}
    for order in (3, 4):
        leading = {}
        for number_of_twos in range(order + 1):
            indices = ((0,) * (order - number_of_twos)
                       + (1,) * number_of_twos)
            key = ('k' + '1' * (order - number_of_twos)
                   + '2' * number_of_twos)
            leading[key] = gaussian_vector_occupation_cumulant(
                indices, T, means_3[0], covariances_3[0],
                means_3[1:] - means_3[0],
                covariances_3[1:] - covariances_3[0], leading_occupation
            )
        errors = [max(abs(row['tensor'][key] - leading[key] / row['m'])
                      for key in leading) for row in return_rows]
        leading_error_rates[order] = rate(errors)
        print(f"   return order {order}: Green--Kubo leading-error rate "
              f"{leading_error_rates[order]:.3f} (want 2)")
    third_spectral_leading_error_rates = {}
    third_leading_occupation = {}
    for occupation_order in range(1, 7):
        for colors in np.ndindex(*(2,) * occupation_order):
            key = tuple(sorted(colors))
            third_leading_occupation[key] = (
                T * third_rate_3[key] if occupation_order == 3 else 0.0
            )
    for order in (5, 6):
        leading = {}
        for number_of_twos in range(order + 1):
            indices = ((0,) * (order - number_of_twos)
                       + (1,) * number_of_twos)
            key = ('k' + '1' * (order - number_of_twos)
                   + '2' * number_of_twos)
            leading[key] = gaussian_vector_occupation_cumulant(
                indices, T, means_3[0], covariances_3[0],
                means_3[1:] - means_3[0],
                covariances_3[1:] - covariances_3[0],
                third_leading_occupation
            )
        errors = [max(abs(
            row['tensor'][key] - leading[key] / row['m'] ** 2
        ) for key in leading) for row in return_rows]
        third_spectral_leading_error_rates[order] = rate(errors)
        print(f"   return order {order}: third-spectral leading-error rate "
              f"{third_spectral_leading_error_rates[order]:.3f} (want 3)")
    ok &= first_derivative_error < 2e-13
    ok &= second_derivative_error < 2e-13
    ok &= third_derivative_error < 2e-13
    ok &= all(magnitude_rates[order] > order - 1.1
              for order in range(2, 7))
    ok &= all(remainder_rates[order] > order - 0.15
              for order in range(2, 7))
    ok &= all(return_rates[order] > math.ceil(order / 2) - 1.1
              for order in range(3, 7))
    ok &= all(leading_error_rates[order] > 1.9 for order in (3, 4))
    ok &= all(third_spectral_leading_error_rates[order] > 2.85
              for order in (5, 6))
    out['finite_state_fast_rate'] = {
        'speeds': fast_speeds,
        'first_derivative_error': first_derivative_error,
        'group_inverse_second_derivative_error': second_derivative_error,
        'group_inverse_third_derivative_error': third_derivative_error,
        'occupation_rows': occupation_rows,
        'return_rates': return_rates,
        'green_kubo_leading_error_rates': leading_error_rates,
        'third_spectral_leading_error_rates':
            third_spectral_leading_error_rates,
    }

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
        exact4 = (w_[0] ** 4 * ex['k1111'] + 4 * w_[0] ** 3 * w_[1] * ex['k1112']
                  + 6 * w_[0] ** 2 * w_[1] ** 2 * ex['k1122']
                  + 4 * w_[0] * w_[1] ** 3 * ex['k1222'] + w_[1] ** 4 * ex['k2222'])
        var = w_[0] ** 2 * ex['k11'] + 2 * w_[0] * w_[1] * ex['k12'] + w_[1] ** 2 * ex['k22']
        prow.append({'m': m, 'rule': rule, 'exact': exact, 'corr_part': corr_part, 'skew_exact': exact / var ** 1.5,
                     'skew_rule': rule / var ** 1.5, 'fourth_exact': exact4,
                     'excess_kurtosis_exact': exact4 / var ** 2})
        print(f"   m={m:2d}: kappa_3 rule {rule:.4e}, exact {exact:.4e}; part from the correlation switch {corr_part:.4e}; "
              f"skewness {exact/var**1.5:.3f}; excess kurtosis {exact4/var**2:.3f}")
    ok &= abs(prow[-1]['rule'] - prow[-1]['exact']) < 0.01 * abs(prow[-1]['exact'])
    out['portfolio'] = prow

    json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
