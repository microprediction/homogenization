"""Certificate for pulsar spin-down switching (two-state Markov spin-down rate).

Units: days, nHz, nHz per day (1e-15 Hz/s = 0.0864 nHz/day). Phase in nHz day = 8.64e-5 cycles.
 1. Frequency residual variance: closed form vs matrix exponential (Van Loan); Green-Kubo 2KT error ~ 1/(gamma T).
 2. Phase residual variance: closed form vs numerical ODE (moment hierarchy); (2/3) K T^3 error ~ 1/(gamma T).
 3. Characteristic function of the frequency residual: closed form vs matrix exponential; Gaussian (first order)
    error ~ (gamma T)^(-1/2), with layer constant and skewness ~ (gamma T)^(-1).
 4. Characteristic function of the phase residual: engine ODE a' = (Q + i u t diag g) a vs Gaussian / adiabatic.
 5. Cumulant rates lambda_2, lambda_3 against the eigenvalue of Q + s diag g by finite differences.
 6. Exact (discretization-free) Monte Carlo of the switching process: variances, skewness, post-fit residual.
 7. Spectrum: Lorentzian vs numerical Fourier transform of the matrix-exponential covariance.
 8. Quasi-periodic switching: renewal formula vs Erlang-chain Green-Kubo (group inverse) vs Monte Carlo.
"""
import json, math, os, sys
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp, quad
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', 'general'))
from pulsars import TwoState, K_renewal, erlang_chain
from effective_generator import gk

U = 0.0864            # 1e-15 Hz/s in nHz/day
CYC = 8.64e-5         # cycles per nHz day
NU = 1.2289688061     # Hz, PSR B1931+24 (Kramer et al. 2006, Table 1)
ON, OFF = -16.3, -10.8  # 1e-15 Hz/s
M_ON, M_OFF = 7.5, 30.0  # mean on and off durations, days (midpoints of 5-10 and 25-35)


def b1931():
    return TwoState(ON * U, OFF * U, 1 / M_ON, 1 / M_OFF)


def var_nu_vanloan(m, T):
    G = np.diag(m.g)
    Z = np.zeros((2, 2))
    M = np.block([[m.Q, G, Z], [Z, m.Q, G], [Z, Z, m.Q]])
    E = expm(T * M)
    return 2 * m.pi @ E[0:2, 4:6] @ np.ones(2)


def var_phase_ode(m, T):
    Q, g = m.Q, m.g

    def f(t, y):
        m1, m2 = y[:2], y[2:]
        return np.concatenate([Q @ m1 + t * g, Q @ m2 + 2 * t * g * m1])
    sol = solve_ivp(f, (0, T), np.zeros(4), method='DOP853', rtol=1e-12, atol=1e-14 * max(1, T ** 3))
    return m.pi @ sol.y[2:, -1]


def cf_nu_expm(m, u, T):
    return m.pi @ expm(T * (m.Q + 1j * u * np.diag(m.g))) @ np.ones(2)


def cf_phase_ode(m, u, T):
    Q, g = m.Q.astype(complex), m.g

    def f(t, a):
        return Q @ a + 1j * u * t * g * a
    sol = solve_ivp(f, (0, T), np.ones(2, complex), method='DOP853', rtol=1e-11, atol=1e-13)
    return m.pi @ sol.y[:, -1]


def simulate(m, T, N, rng, grid=None):
    """Exact simulation of the stationary two-state chain; returns X(T), Phi(T) (and switch lists if grid)."""
    y = (rng.random(N) < m.p2).astype(int)             # 0 = state 1, 1 = state 2
    rates = np.array([m.a, m.b])
    t = np.zeros(N)
    X = np.zeros(N)
    P = np.zeros(N)
    alive = np.ones(N, bool)
    while alive.any():
        idx = np.where(alive)[0]
        h = rng.exponential(1 / rates[y[idx]])
        t1 = np.minimum(t[idx] + h, T)
        gi = m.g[y[idx]]
        X[idx] += gi * (t1 - t[idx])
        P[idx] += gi * ((T - t[idx]) ** 2 - (T - t1) ** 2) / 2
        t[idx] = t1
        y[idx] = 1 - y[idx]
        alive[idx] = t1 < T
    return X, P


def postfit_mc(m, T, N, rng, n=1001):
    """Mean-square residual of the phase after a least-squares quadratic fit on n equally spaced samples."""
    tg = np.linspace(0, T, n)
    V = np.vander(tg / T, 3)
    H = V @ np.linalg.pinv(V)
    out = np.empty(N)
    for k in range(N):
        s = [0.0]
        y = int(rng.random() < m.p2)
        while s[-1] < T:
            s.append(s[-1] + rng.exponential(1 / (m.a if y == 0 else m.b)))
            y = 1 - y
        y0 = y if (len(s) - 1) % 2 == 0 else 1 - y      # state at time 0
        s = np.array(s)
        s[-1] = T
        states = (y0 + np.arange(len(s) - 1)) % 2
        Xs = np.concatenate([[0.0], np.cumsum(m.g[states] * np.diff(s))])      # X at switch times
        pts = np.union1d(s, tg)
        Xp = np.interp(pts, s, Xs)                                              # X is piecewise linear
        Ph = np.concatenate([[0.0], np.cumsum(0.5 * (Xp[1:] + Xp[:-1]) * np.diff(pts))])
        r = np.interp(tg, pts, Ph)
        r = r - H @ r
        out[k] = np.mean(r ** 2)
    return out


def main():
    ok, out = True, {}
    m = b1931()
    print(f"PSR B1931+24, Markov idealization: pi_on = {m.p1:.3f}, 1/gamma = {1/m.gamma:.2f} d, "
          f"mean spin-down {m.nubar_dot/U:.3f}e-15 Hz/s, K = {m.K:.6e} nHz^2/day")
    out['params'] = dict(nu=NU, on=ON, off=OFF, m_on=M_ON, m_off=M_OFF, p_on=m.p1, corr_days=1 / m.gamma,
                         nubar=m.nubar_dot / U, K_nHz2_per_day=m.K,
                         K_Hz2_per_s=m.K * 1e-18 / 86400, lam3=m.lam3)

    print("1. frequency residual variance")
    rows, errs = [], []
    for T in (30.0, 100.0, 300.0, 1000.0, 3000.0):
        vl, cf, g1 = var_nu_vanloan(m, T), m.var_nu(T), m.var_nu_gk(T)
        rows.append([T, m.gamma * T, vl, cf, abs(cf - vl) / vl, (g1 - vl) / vl])
        errs.append((g1 - vl) / vl)
        print(f"   T {T:6.0f} d (gamma T {m.gamma*T:6.1f}): matrix exp {vl:.8e}, closed form {cf:.8e} "
              f"(rel diff {abs(cf-vl)/vl:.1e}), Green-Kubo rel error {(g1-vl)/vl:+.4%}")
        ok &= abs(cf - vl) / vl < 1e-9
    ok &= all(2.9 < errs[i] / errs[i + 1] < 3.4 for i in range(1, 4))
    out['var_nu'] = rows

    print("2. phase residual variance")
    rows, e1, e2 = [], [], []
    for T in (30.0, 100.0, 300.0, 1000.0, 3000.0):
        vo, cf = var_phase_ode(m, T), m.var_phase(T)
        l, two = m.var_phase_gk(T), m.var_phase_two(T)
        rows.append([T, m.gamma * T, vo, cf, abs(cf - vo) / vo, (l - vo) / vo, (two - vo) / vo])
        e1.append((l - vo) / vo); e2.append((two - vo) / vo)
        print(f"   T {T:6.0f}: ODE {vo:.8e}, closed form {cf:.8e} (rel diff {abs(cf-vo)/vo:.1e}), "
              f"T^3 term rel error {e1[-1]:+.3e}, T^3 + T^2 rel error {e2[-1]:+.3e}")
        ok &= abs(cf - vo) / vo < 1e-7
    # first order: error x gamma T tends to a constant (3/2); with the T^2 term: error x (gamma T)^3 -> -3
    ok &= all(abs(e1[i] * rows[i][1] / 1.5 - 1) < 0.05 for i in range(2, 5))
    ok &= all(abs(e2[i] * rows[i][1] ** 3 / -3 - 1) < 0.02 for i in range(3, 5))
    out['var_phase'] = rows

    print("3. characteristic function of the frequency residual (standardized)")
    rows, eg, es = [], [], []
    for gT in (25.0, 100.0, 400.0, 1600.0):
        T = gT / m.gamma
        sd = math.sqrt(m.var_nu(T))
        us = np.linspace(0, 4, 41) / sd
        ex = np.array([cf_nu_expm(m, u, T) for u in us])
        cl = m.mgf_nu(1j * us, T)
        ga = np.exp(-0.5 * us ** 2 * m.var_nu_gk(T))
        v2 = m.var_nu(T)
        sk = np.exp(-0.5 * us ** 2 * v2 - 1j * us ** 3 * m.kappa3_nu_first(T) / 6)
        d0, d1, d2 = np.abs(cl - ex).max(), np.abs(ga - ex).max(), np.abs(sk - ex).max()
        skew = m.kappa3_nu_first(T) / v2 ** 1.5
        rows.append([gT, T, skew, d0, d1, d2]); eg.append(d1); es.append(d2)
        print(f"   gamma T {gT:6.0f}: closed form vs expm {d0:.1e}; Gaussian err {d1:.2e}; "
              f"+layer+skew err {d2:.2e}; skewness {skew:+.3f}")
        ok &= d0 < 1e-10
    ok &= all(1.8 < eg[i] / eg[i + 1] < 2.2 for i in range(1, 3))
    ok &= all(3.5 < es[i] / es[i + 1] < 4.5 for i in range(1, 3))
    out['cf_nu'] = rows

    print("4. characteristic function of the phase residual via the engine ODE (standardized)")
    rows, eg, es = [], [], []
    for gT in (25.0, 100.0, 400.0):
        T = gT / m.gamma
        v = m.var_phase(T)
        sd = math.sqrt(v)
        us = np.linspace(0, 4, 17) / sd
        ex = np.array([cf_phase_ode(m, u, T) for u in us])
        ga = np.exp(-0.5 * us ** 2 * m.var_phase_gk(T))
        sk = np.exp(-0.5 * us ** 2 * m.var_phase_two(T) - 1j * us ** 3 * m.kappa3_phase_first(T) / 6)
        d1, d2 = np.abs(ga - ex).max(), np.abs(sk - ex).max()
        skew = m.kappa3_phase_first(T) / v ** 1.5
        rows.append([gT, T, skew, d1, d2]); eg.append(d1); es.append(d2)
        print(f"   gamma T {gT:6.0f}: Gaussian err {d1:.2e}; two-term variance + skew err {d2:.2e}; skewness {skew:+.3f}")
    ok &= all(1.8 < eg[i] / eg[i + 1] < 2.3 for i in range(2))
    ok &= 3.5 < es[1] / es[2] < 5.0
    out['cf_phase'] = rows

    print("5. cumulant rates from the dominant eigenvalue")
    h = 1e-2 / (abs(m.Delta) / m.gamma)
    ev = lambda s: max(np.linalg.eigvals(m.Q + s * np.diag(m.g)).real)
    l2 = (ev(h) - 2 * ev(0) + ev(-h)) / (2 * h * h)
    l3 = (ev(2 * h) - 2 * ev(h) + 2 * ev(-h) - ev(-2 * h)) / (12 * h ** 3)
    print(f"   lambda_2: {l2:.6e} vs K {m.K:.6e};  lambda_3: {l3:.6e} vs formula {m.lam3:.6e}")
    ok &= abs(l2 / m.K - 1) < 1e-4 and abs(l3 / m.lam3 - 1) < 1e-3
    ok &= abs(m.K - gk(m.Q, [m.g])[0, 0]) / m.K < 1e-10
    out['rates'] = [l2, m.K, l3, m.lam3]

    print("6. exact Monte Carlo of the switching process")
    rng = np.random.default_rng(7)
    rows = []
    for T in (100.0, 300.0, 1000.0):
        N = 400000
        X, P = simulate(m, T, N, rng)
        vx, vp = X.var(), P.var()
        sk = np.mean((X - X.mean()) ** 3) / vx ** 1.5
        sex, sep = vx * math.sqrt(2 / N), vp * math.sqrt(2 / N)
        v2 = m.var_nu(T)
        rows.append([T, vx, sex, v2, vp, sep, m.var_phase(T), sk, m.kappa3_nu_first(T) / v2 ** 1.5])
        print(f"   T {T:5.0f}: Var X {vx:.5e} +- {sex:.1e} vs {v2:.5e}; Var Phi {vp:.5e} +- {sep:.1e} vs "
              f"{m.var_phase(T):.5e}; skew X {sk:+.3f} vs first order {rows[-1][-1]:+.3f}")
        ok &= abs(vx - v2) < 4 * sex and abs(vp - m.var_phase(T)) < 4 * sep
    out['mc'] = rows
    rows = []
    for T in (300.0, 1000.0, 3000.0):
        N = 4000
        ms = postfit_mc(m, T, N, rng)
        pred = m.postfit_ms(T)
        rows.append([T, ms.mean(), ms.std() / math.sqrt(N), pred])
        print(f"   post-fit mean-square phase residual, T {T:5.0f}: MC {ms.mean():.4e} +- {ms.std()/math.sqrt(N):.1e}"
              f" vs 2KT^3/2520 = {pred:.4e} (ratio {ms.mean()/pred:.3f})")
    ok &= abs(rows[-1][1] / rows[-1][3] - 1) < 0.08
    out['postfit'] = rows

    print("7. spectrum of the spin-down rate")
    C = lambda tau: m.pi @ (m.g * (expm(tau * m.Q) @ m.g))
    rows = []
    for f in (0.0, 0.01, 0.03, 0.1):     # cycles per day
        num = 2 * quad(lambda tau: C(tau) * math.cos(2 * math.pi * f * tau), 0, 400 / m.gamma, limit=400)[0]
        rows.append([f, num, float(m.S_nudot(f))])
        print(f"   f {f:5.2f}/d: numerical FT {num:.8e}, Lorentzian {float(m.S_nudot(f)):.8e}")
        ok &= abs(num / float(m.S_nudot(f)) - 1) < 1e-6
    ok &= abs(float(m.S_nudot(0.0)) - 2 * m.K) < 1e-12 * m.K
    out['spectrum'] = rows

    print("8. quasi-periodic switching")
    s_on, s_off = 5 / math.sqrt(12), 10 / math.sqrt(12)        # uniform on 5-10 and 25-35 days
    Kq = K_renewal(m.Delta, M_ON, s_on, M_OFF, s_off)
    k_on, k_off = round((M_ON / s_on) ** 2), round((M_OFF / s_off) ** 2)
    Qe, ind = erlang_chain(M_ON, k_on, M_OFF, k_off)
    Ke = gk(Qe, [ind * m.Delta])[0, 0]
    Kf = K_renewal(m.Delta, M_ON, M_ON / math.sqrt(k_on), M_OFF, M_OFF / math.sqrt(k_off))
    print(f"   Erlang({k_on},{k_off}) chain: group inverse K {Ke:.8e}, renewal formula {Kf:.8e}")
    ok &= abs(Ke / Kf - 1) < 1e-8
    # Monte Carlo with uniform durations after a burn-in
    N, T, burn = 20000, 20000.0, 2000.0
    t = np.zeros(N); X = np.zeros(N); state = np.zeros(N, int)
    cur = rng.uniform(0, M_ON + M_OFF, N)
    t = -burn + np.zeros(N)
    alive = np.ones(N, bool)
    while alive.any():
        i = np.where(alive)[0]
        d = np.where(state[i] == 0, rng.uniform(5, 10, len(i)), rng.uniform(25, 35, len(i)))
        t0, t1 = t[i], np.minimum(t[i] + d, T)
        lo = np.maximum(t0, 0.0)
        X[i] += np.where(t1 > lo, (t1 - lo), 0.0) * np.where(state[i] == 0, m.g[0], m.g[1])
        t[i] = t1; state[i] = 1 - state[i]; alive[i] = t1 < T
    vq = X.var()
    Kmc, se = vq / (2 * T), vq / (2 * T) * math.sqrt(2 / N)
    print(f"   uniform durations: K renewal {Kq:.5e}, Monte Carlo Var/2T {Kmc:.5e} +- {se:.1e}; "
          f"Markov K {m.K:.5e} (ratio {m.K/Kq:.1f})")
    ok &= abs(Kmc - Kq) < 4 * se + 0.01 * Kq
    out['renewal'] = dict(K_uniform=Kq, K_mc=Kmc, K_mc_se=se, K_markov=m.K, k_on=k_on, k_off=k_off,
                          K_erlang_gi=Ke, K_erlang_formula=Kf)

    print("worked example")
    ex = []
    for mod, name in ((m, 'markov'),):
        for T in (100.0, 300.0, 1000.0, 3000.0):
            sdnu = math.sqrt(mod.var_nu(T))
            sdph = math.sqrt(mod.var_phase(T)) * CYC
            pf = math.sqrt(mod.postfit_ms(T)) * CYC
            pfq = math.sqrt(mod.postfit_ms(T) * Kq / mod.K) * CYC
            ex.append([T, sdnu, math.sqrt(mod.var_nu_gk(T)), sdph, sdph / NU,
                       math.sqrt(mod.var_phase_gk(T)) * CYC / NU, pf / NU * 1e3, pfq / NU * 1e3,
                       mod.kappa3_nu_first(T) / mod.var_nu(T) ** 1.5])
            print(f"   T {T:5.0f} d: sd nu {sdnu:.3f} nHz (GK {ex[-1][2]:.3f}); sd timing {ex[-1][4]:.4f} s "
                  f"(T^3: {ex[-1][5]:.4f}); post-fit rms {ex[-1][6]:.2f} ms (quasi-periodic {ex[-1][7]:.2f} ms); "
                  f"skew {ex[-1][8]:+.3f}")
    out['example'] = ex
    # PSR B1828-11 (Lyne et al. 2010, Table 1): nu 2.469 Hz, nudot -365.68e-15, peak-to-peak 0.71%, F 0.73/yr.
    # Idealization: two rates 0.71% apart, a 500-day mean cycle split 1/3 high |nudot|, 2/3 low (illustrative split).
    nu8, nd8, frac = 2.469, -365.68, 0.0071
    cyc = 365.25 / 0.73
    hi, lo = cyc / 3, 2 * cyc / 3
    d8 = frac * abs(nd8)
    # mean fixed at the published value
    n_hi = nd8 - d8 * (lo / cyc)
    n_lo = n_hi + d8
    m8 = TwoState(n_hi * U, n_lo * U, 1 / hi, 1 / lo)
    ex8 = []
    for yrs in (5.0, 10.0, 20.0):
        T = yrs * 365.25
        ex8.append([yrs, m8.gamma * T, math.sqrt(m8.var_nu(T)), math.sqrt(m8.var_phase(T)) * CYC / nu8,
                    math.sqrt(m8.var_phase_gk(T)) * CYC / nu8, math.sqrt(m8.postfit_ms(T)) * CYC / nu8 * 1e3,
                    m8.kappa3_nu_first(T) / m8.var_nu(T) ** 1.5])
        print(f"   B1828-11 T {yrs:4.0f} yr (gamma T {ex8[-1][1]:.1f}): sd nu {ex8[-1][2]:.2f} nHz, sd timing "
              f"{ex8[-1][3]:.2f} s (T^3: {ex8[-1][4]:.2f}), post-fit rms {ex8[-1][5]:.1f} ms, skew {ex8[-1][6]:+.3f}")
    out['b1828'] = dict(nu=nu8, nudot=nd8, frac=frac, cycle_days=cyc, hi_days=hi, lo_days=lo, n_hi=n_hi, n_lo=n_lo,
                        p_hi=m8.p1, corr_days=1 / m8.gamma, K_nHz2_per_day=m8.K, rows=ex8)
    # a second chain with the same mean spin-down and K but equal occupancy
    m2 = TwoState(m.nubar_dot + 0.5 * 0.8 * m.Delta, m.nubar_dot - 0.5 * 0.8 * m.Delta, m.gamma / 2, m.gamma / 2)
    same = []
    for mod in (m, m2):
        T = 1000.0
        same.append([mod.n1 / U, mod.n2 / U, mod.p1, 1 / mod.gamma, mod.nubar_dot / U, mod.K,
                     math.sqrt(mod.var_nu(T)), math.sqrt(mod.var_phase(T)) * CYC / NU,
                     mod.kappa3_nu_first(T) / mod.var_nu(T) ** 1.5])
        print("   chain", ["%.4g" % x for x in same[-1]])
    ok &= abs(m2.K / m.K - 1) < 1e-12 and abs(m2.nubar_dot / m.nubar_dot - 1) < 1e-12
    out['same_K'] = same
    json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == '__main__':
    main()
