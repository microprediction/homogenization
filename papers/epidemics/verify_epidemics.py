"""Certificate: mean and almost-sure growth of linearized SIR and SEIR epidemics with rates switched by a fast chain.

References, all independent of the expansions:
  mean growth         principal eigenvalue of the combined system Q^T (x) I + blockdiag(A_i) (numerical linear algebra),
                      and, for two states, its closed form (SIR) or scalar fixed point (SEIR);
  mean itself         Monte Carlo of E[I_t] over regime paths, against exp(t G) of the combined system;
  almost-sure growth  Monte Carlo of the switched ODE, each holding interval propagated exactly by exp(tau A_i).
Checks:
  1. SIR, beta switched: mean = bbar - gamma + K, almost-sure = bbar - gamma exactly.
  2. SEIR, beta switched: no first-order mean correction (second order, closed form), almost-sure drops by
     K (ybar_E xbar_I)^2 = K sigma^2 / D; errors fall at the expected orders.
  3. SEIR, beta and gamma switched on a three-state cycle, forward and reversed: first-order mean and almost-sure
     growth, and the forward-minus-reversed gap against 2 K^anti sigma / sqrt(D).
  4. A three-compartment structured example for the general formulas.
Writes results.json for the page.
"""
import json, math, os
import numpy as np
from scipy.linalg import expm
from epidemics import (stationary, gk, two_state_K, sir_mean_two_state, seir_vectors, seir_beta_mean_two_state,
                       seir_beta_formulas, seir_beta_gamma_formulas, first_order, mean_growth, lyapunov_mc,
                       cycle_generator, cycle_K, regime_matrices, principal, E_I, I_I)

HERE = os.path.dirname(os.path.abspath(__file__))
SIGMA, GAMMA = 0.5, 0.25                      # rates per week: latent 2 weeks, infectious 4 weeks
BETA2 = (0.455, 0.055)                        # two-state transmission, bbar = 0.255, R0 = 1.02
LAMS = [1, 2, 4, 8, 16]
BETA3 = np.array([0.605, 0.105, 0.055])       # open, testing, caution
GAMMA3 = np.array([0.1, 0.55, 0.1])
A_FWD, B_FWD = 2.0, 0.1                       # cycle 1 -> 2 -> 3 -> 1 at rate 2c, back at 0.1c
CS = [1, 2, 4, 8, 16]
ok = True


def rate(e1, e2, f=2.0):
    return math.log(abs(e1) / abs(e2)) / math.log(f)


def check(cond, msg):
    global ok
    print(("  ok   " if cond else "  FAIL ") + msg)
    ok &= bool(cond)


def two_state(lam):
    return np.array([[-lam, lam], [lam, -lam]], float)


def seir(b, g):
    return np.array([[-SIGMA, b], [SIGMA, -g]])


def mc_jumps(rate_out):
    return int(min(40000, 4000 * max(1.0, rate_out / 2)))


def main():
    out = {'params': {'sigma': SIGMA, 'gamma': GAMMA, 'beta2': BETA2, 'beta3': BETA3.tolist(), 'gamma3': GAMMA3.tolist(),
                      'cycle': [A_FWD, B_FWD]}}
    bb, bt = sum(BETA2) / 2, (BETA2[0] - BETA2[1]) / 2

    # ---------------------------------------------------------------- 1. SIR
    print("1. SIR, beta switched on two states")
    rows = []
    for lam in LAMS:
        Q = two_state(lam)
        num, _ = mean_growth(Q, [np.array([[b - GAMMA]]) for b in BETA2])
        closed = sir_mean_two_state(*BETA2, GAMMA, lam)
        first = bb - GAMMA + two_state_K(bt, lam)
        rows.append({'lam': lam, 'K': two_state_K(bt, lam), 'numerical': num, 'closed': closed, 'first': first,
                     'as': bb - GAMMA, 'err': first - num})
        print(f"  lam {lam:2d}  mean numerical {num:.6f}  closed form {closed:.6f}  first order {first:.6f}"
              f"  error {first - num:.1e}   almost-sure {bb - GAMMA:.6f}")
        check(abs(num - closed) < 1e-12, "closed form equals the combined-system eigenvalue")
    check(rate(rows[-2]['err'], rows[-1]['err']) > 2.8, f"first-order error falls at order 3 for the symmetric chain "
          f"(rate {rate(rows[-2]['err'], rows[-1]['err']):.2f})")
    out['sir'] = rows

    # mean equals E[I_t]: Monte Carlo over regime paths (SIR is exact given the path)
    rng = np.random.default_rng(7)
    lam, t, n = 1.0, 10.0, 200000
    y = rng.integers(0, 2, n)
    clock, integ = np.zeros(n), np.zeros(n)
    alive = np.ones(n, bool)
    while alive.any():
        tau = rng.exponential(1 / lam, n)
        dt = np.minimum(tau, t - clock)
        integ += np.where(alive, (np.array(BETA2)[y] - GAMMA) * dt, 0.0)
        clock += np.where(alive, dt, 0.0)
        alive &= clock < t - 1e-12
        y = np.where(alive, 1 - y, y)
    samp = np.exp(integ)
    _, G = mean_growth(two_state(lam), [np.array([[b - GAMMA]]) for b in BETA2])
    exact_mean = np.array([0.5, 0.5]) @ expm(t * G) @ np.ones(2)
    se = samp.std() / math.sqrt(n)
    print(f"  E[I_10]/I_0 at lam = 1: Monte Carlo {samp.mean():.5f} +- {se:.5f}, combined system {exact_mean:.5f}")
    check(abs(samp.mean() - exact_mean) < 4 * se, "the combined system gives the mean")
    out['sir_mean_check'] = {'lam': lam, 't': t, 'mc': samp.mean(), 'se': se, 'system': exact_mean,
                             'median': float(np.median(samp)), 'as_factor': math.exp((bb - GAMMA) * t)}

    # ---------------------------------------------------------------- 2. SEIR, beta switched
    print("2. SEIR, beta switched on two states")
    r, xb, yb, sD = seir_vectors(bb, SIGMA, GAMMA)
    out['seir_avg'] = {'r': r, 'xbar': xb.tolist(), 'ybar': yb.tolist(), 'sqrtD': sD, 'yE_xI': yb[0] * xb[1],
                       'yI_xI': yb[1] * xb[1]}
    print(f"  r = {r:.6f}, sqrt(D) = {sD:.6f}, ybar_E xbar_I = sigma/sqrt(D) = {yb[0] * xb[1]:.6f}")
    rows = []
    for lam in LAMS:
        Q = two_state(lam)
        As = [seir(b, GAMMA) for b in BETA2]
        num, _ = mean_growth(Q, As)
        fp = seir_beta_mean_two_state(*BETA2, SIGMA, GAMMA, lam)
        f = seir_beta_formulas(*BETA2, SIGMA, GAMMA, lam)
        gen = first_order(Q, seir(bb, GAMMA), [E_I], [list(BETA2)])
        mc, se = lyapunov_mc(Q, As, jumps=mc_jumps(lam), chains=2000, seed=lam)
        rows.append({'lam': lam, 'K': f['K'], 'mean_num': num, 'mean_fp': fp, 'mean2': f['mean2'], 'as_mc': mc, 'as_se': se,
                     'as1': f['as1'], 'mean_err0': num - r, 'mean_err2': f['mean2'] - num, 'as_err': f['as1'] - mc})
        print(f"  lam {lam:2d}  mean: numerical {num:.6f} fixed point {fp:.6f} second order {f['mean2']:.6f}"
              f"  | almost-sure: Monte Carlo {mc:.6f} +- {se:.1e}  first order {f['as1']:.6f}")
        check(abs(num - fp) < 1e-11, "fixed-point formula equals the combined-system eigenvalue")
        check(abs(gen[0] - r) < 1e-14 and abs(gen[1] - f['as1']) < 1e-12, "general formulas: no first-order mean term, "
              "almost-sure term matches")
    e0, e2, ea = [[row[k] for row in rows] for k in ('mean_err0', 'mean_err2', 'as_err')]
    print(f"  (mean - r) lam^2: {', '.join(f'{v * l * l:.5f}' for v, l in zip(e0, LAMS))}"
          f"   predicted bt^2 sigma^2 / (4 sqrt D) = {bt ** 2 * SIGMA ** 2 / (4 * sD):.5f}")
    check(rate(e0[-2], e0[-1]) > 1.9, f"mean moves at second order (rate {rate(e0[-2], e0[-1]):.2f})")
    check(rate(e2[-2], e2[-1]) > 2.7, f"second-order mean error falls at order 3 (rate {rate(e2[-2], e2[-1]):.2f})")
    ra = [rate(ea[i], ea[i + 1]) for i in range(1, 4)]
    print(f"  almost-sure error rates: {', '.join(f'{v:.2f}' for v in ra)}")
    check(all(1.6 < v < 2.4 for v in ra), "almost-sure first-order error falls at order 2")
    out['seir_beta'] = rows

    # ---------------------------------------------------------------- 3. the cycle
    print("3. SEIR, beta and gamma switched on a three-state cycle, forward and reversed")
    rows = []
    for c in CS:
        row = {'c': c}
        for lab, (a, b) in [('fwd', (A_FWD * c, B_FWD * c)), ('rev', (B_FWD * c, A_FWD * c))]:
            Q = cycle_generator(a, b)
            As = [seir(bv, gv) for bv, gv in zip(BETA3, GAMMA3)]
            num, _ = mean_growth(Q, As)
            f = seir_beta_gamma_formulas(Q, BETA3, GAMMA3, SIGMA)
            Kbb, Kbg, Kgb, Kgg = cycle_K(BETA3, BETA3, a, b)[0], cycle_K(BETA3, GAMMA3, a, b)[0], \
                cycle_K(GAMMA3, BETA3, a, b)[0], cycle_K(GAMMA3, GAMMA3, a, b)[0]
            check(max(abs(Kbb - f['Kbb']), abs(Kbg - f['Kbg']), abs(Kgb - f['Kgb']), abs(Kgg - f['Kgg'])) < 1e-13,
                  f"c = {c} {lab}: cycle Green-Kubo closed form equals -pi.(f Q# g)")
            gen = first_order(Q, seir(f['bbar'], f['gbar']), [E_I, I_I], [BETA3, GAMMA3])
            check(abs(gen[0] - f['mean1']) < 1e-13 and abs(gen[1] - f['as1']) < 1e-13 and abs(gen[2] - f['gap']) < 1e-13,
                  f"c = {c} {lab}: SEIR closed forms equal the general formulas")
            mc, se = lyapunov_mc(Q, As, jumps=mc_jumps(a + b), chains=2000, seed=100 + c + (lab == 'rev'))
            row[lab] = {'mean_num': num, 'mean1': f['mean1'], 'as_mc': mc, 'as_se': se, 'as1': f['as1'],
                        'K': [Kbb, Kbg, Kgb, Kgg], 'gap1': f['gap']}
            row['pred'] = f['cycle_pred'] if lab == 'fwd' else row.get('pred')
            row['r'] = f['r']
        row['diff_mean'] = row['fwd']['mean_num'] - row['rev']['mean_num']
        row['diff_as'] = row['fwd']['as_mc'] - row['rev']['as_mc']
        row['diff_as_se'] = math.hypot(row['fwd']['as_se'], row['rev']['as_se'])
        rows.append(row)
        print(f"  c {c:2d}  fwd: mean {row['fwd']['mean_num']:.6f} (first order {row['fwd']['mean1']:.6f})"
              f"  a.s. {row['fwd']['as_mc']:.6f} +- {row['fwd']['as_se']:.0e} (first order {row['fwd']['as1']:.6f})")
        print(f"        rev: mean {row['rev']['mean_num']:.6f} (first order {row['rev']['mean1']:.6f})"
              f"  a.s. {row['rev']['as_mc']:.6f} +- {row['rev']['as_se']:.0e} (first order {row['rev']['as1']:.6f})")
        print(f"        forward minus reversed: mean {row['diff_mean']:.6f}, a.s. {row['diff_as']:.6f},"
              f" predicted {row['pred']:.6f}")
    f0 = seir_beta_gamma_formulas(cycle_generator(A_FWD, B_FWD), BETA3, GAMMA3, SIGMA)
    out['cycle_avg'] = {k: f0[k] for k in ('r', 'bbar', 'gbar', 'sb', 'sg')}
    out['cycle_avg']['W'] = cycle_K(BETA3, GAMMA3, A_FWD, B_FWD)[1]
    dm = [row['diff_mean'] - row['pred'] for row in rows]
    rd = [rate(dm[i], dm[i + 1]) for i in range(len(dm) - 1)]
    print(f"  gap error rates: {', '.join(f'{v:.2f}' for v in rd)}")
    check(rd[-1] > 1.8, "forward-minus-reversed mean gap: error falls at order 2")
    check(abs(rows[-1]['diff_mean'] / rows[-1]['pred'] - 1) < 0.05, "gap ratio tends to 1")
    check(all(abs(row['diff_as'] - row['diff_mean']) < 4 * row['diff_as_se'] + 2 * abs(row['diff_mean'] - row['pred'])
              for row in rows), "almost-sure gap agrees with the mean gap to the accuracy of the expansion")
    for lab in ('fwd', 'rev'):
        em = [row[lab]['mean1'] - row[lab]['mean_num'] for row in rows]
        ea = [row[lab]['as1'] - row[lab]['as_mc'] for row in rows]
        check(rate(em[-2], em[-1]) > 1.8, f"{lab}: first-order mean error falls at order 2 (rate {rate(em[-2], em[-1]):.2f})")
        check(rate(ea[1], ea[3], 4.0) > 1.6, f"{lab}: first-order almost-sure error falls at order 2 "
              f"(rate {rate(ea[1], ea[3], 4.0):.2f} over c = 2 to 8)")
    c2 = rows[1]
    check(c2['fwd']['as_mc'] > 5 * c2['fwd']['as_se'] and c2['rev']['as_mc'] < -5 * c2['rev']['as_se'],
          "at c = 2 the forward cycle invades and the reversed cycle dies out")
    out['cycle'] = rows

    # ---------------------------------------------------------------- 4. general structured example
    print("4. general formulas on a three-compartment model with a non-reversible three-state chain")
    Abar = np.array([[-0.6, 0.3, 0.5], [0.4, -0.5, 0.1], [0.1, 0.3, -0.4]])
    Bs = [np.array([[0, 0.3, 0.6], [0, 0, 0], [0, 0, 0]]), np.array([[-0.2, 0, 0], [0.2, -0.3, 0], [0, 0.3, 0]])]
    phis = [np.array([1.0, -0.5, -0.5]), np.array([-0.4, 1.0, -0.6])]
    Q0 = np.array([[-3, 2, 1], [1, -2, 1], [0.5, 1.5, -2]])
    rows = []
    for c in [2, 4, 8, 16]:
        Q = c * Q0
        pi = stationary(Q)
        As = regime_matrices(Abar, Bs, phis, pi)
        m1, a1, gap = first_order(Q, Abar, Bs, phis)
        num, _ = mean_growth(Q, As)
        mc, se = lyapunov_mc(Q, As, jumps=mc_jumps(3 * c), chains=2000, seed=500 + c)
        r0, xb3, yb3 = principal(Abar)
        rho = np.array([yb3 @ A @ xb3 for A in As])
        Krr = gk(Q, [rho])[0, 0]
        rows.append({'c': c, 'mean_num': num, 'mean1': m1, 'as_mc': mc, 'as_se': se, 'as1': a1, 'gap': gap, 'Krr': Krr})
        print(f"  c {c:2d}  mean {num:.6f} vs {m1:.6f}   a.s. {mc:.6f} +- {se:.0e} vs {a1:.6f}   gap {gap:.6f} = K(rho,rho) {Krr:.6f}")
        check(abs(gap - Krr) < 1e-14 and gap >= 0, "gap equals the Green-Kubo integral of the projected rate")
    em = [row['mean1'] - row['mean_num'] for row in rows]
    ea = [row['as1'] - row['as_mc'] for row in rows]
    check(rate(em[-2], em[-1]) > 1.8, f"mean error order 2 (rate {rate(em[-2], em[-1]):.2f})")
    check(rate(ea[0], ea[2], 4.0) > 1.6, f"almost-sure error order 2 (rate {rate(ea[0], ea[2], 4.0):.2f} over c = 2 to 8)")
    out['general'] = rows

    json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, default=float)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
