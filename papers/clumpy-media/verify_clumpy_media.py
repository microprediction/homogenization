"""Certificate for light through clumpy media: transmission of a purely absorbing Markovian mixture.

1. The classical two-exponential law equals the matrix exponential pi . expm((Q - diag Sigma) L) . 1.
2. The closed-form coefficients (atomic mix, Green-Kubo K, second-order M3, layer constant C1) for two materials
   and for an N-colour Poisson tessellation equal the general Green-Kubo expressions, and the closed-form log T
   through orders 0, 1, 2 equals the fast-switching engine at those orders.
3. Worked example (dense clumps in thin gas): errors in log T against the matrix exponential fall like
   lam_c^(n+1) at order n, for n = 0, 1, 2, 3, 4, 6, 8.
4. The series in lam_c converges for lam_c |Sigma_A - Sigma_B| < 1 and diverges beyond: Cauchy-integral
   Taylor coefficients of the decay rate grow like |Sigma_A - Sigma_B|^k, and engine partial sums settle at
   lam_c = 0.16 and blow up at lam_c = 0.32.
5. Three nested phases (cores inside envelopes inside gas): the N-material closed form against the matrix
   exponential and the engine, with the same orders of convergence.
6. Monte Carlo: rays traced through realizations of a coloured 3D Poisson plane tessellation, and chord-length
   sampling of the chain, agree with the matrix exponential within three standard errors.
Writes results.json for the page.
"""
import json
import math
import os
import numpy as np
from clumpy import (binary_Q, binary_lams, tessellation_Q, exact_T, two_exponential_T, two_exponential_parts,
                    coefficients, binary_coefficients, log_T_closed, engine_log_T, mc_chain, mc_tessellation,
                    stationary, tau_cumulants)

HERE = os.path.dirname(os.path.abspath(__file__))
SA, SB, PA, L = 5.0, 0.05, 0.1, 10.0          # dense clumps filling 10% of a thin gas, slab of 10 units
LCS = [0.16, 0.08, 0.04, 0.02, 0.01]
ORDERS = [0, 1, 2, 3, 4, 6, 8]
SIG3, PI3 = [0.05, 1.0, 10.0], (0.85, 0.10, 0.05)
ELLS = [0.025, 0.0125, 0.00625, 0.003125]


def nested_Q(ell, pi=PI3):
    """gas <-> envelope <-> core; envelope and core chords have mean ell, envelopes exit either way equally."""
    b = 1 / (2 * ell)
    a, c = pi[1] * b / pi[0], pi[1] * b / pi[2]
    return np.array([[-a, a, 0], [b, -2 * b, b], [0, c, -c]])


def main():
    ok, out = True, {'params': dict(SA=SA, SB=SB, pA=PA, L=L, Sig3=SIG3, pi3=PI3)}

    print("1. two-exponential law against the matrix exponential")
    worst = 0.0
    for SA_, SB_, lA, lB, L_ in [(2, 0.2, 0.1, 0.2, 3), (SA, SB, 0.09, 0.8, L), (1, 0, 0.5, 0.5, 4), (0.3, 7, 2, 0.05, 1)]:
        worst = max(worst, abs(two_exponential_T(SA_, SB_, lA, lB, L_) - exact_T(binary_Q(lA, lB), [SA_, SB_], L_)))
    print(f"   largest difference {worst:.1e}")
    ok &= worst < 1e-13
    out['law_diff'] = worst

    print("2. closed-form coefficients and the engine at orders 0, 1, 2")
    cd, ed = 0.0, 0.0
    for lc in LCS:
        lA, lB = binary_lams(PA, lc)
        Q = binary_Q(lA, lB)
        cg, cb = np.array(coefficients(Q, [SA, SB])), np.array(binary_coefficients(SA, SB, lA, lB))
        cd = max(cd, np.max(np.abs(cg - cb) / np.maximum(1e-300, np.abs(cb))))
        eng = engine_log_T(Q, [SA, SB], L, [0, 1, 2])
        ed = max(ed, max(abs(eng[k] - log_T_closed(cb, L, k)) for k in (0, 1, 2)))
    # N-colour tessellation: K = lam_c Var, M3 = lam_c^2 E[Sigma~^3], C1 = lam_c^2 Var
    p, S, lc = np.array([0.5, 0.3, 0.2]), np.array([0.1, 2.0, 6.0]), 0.07
    St = S - p @ S
    tess = np.array([p @ S, lc * p @ St ** 2, lc ** 2 * p @ St ** 3, lc ** 2 * p @ St ** 2])
    td = np.max(np.abs(np.array(coefficients(tessellation_Q(p, lc), S)) - tess) / np.abs(tess))
    # broken clouds: clear air does not absorb
    c, Sc, lc2 = 0.4, 3.0, 0.05
    lA, lB = binary_lams(c, lc2)
    cloud = np.array([c * Sc, c * (1 - c) * Sc ** 2 * lc2, c * (1 - c) * (1 - 2 * c) * Sc ** 3 * lc2 ** 2,
                      c * (1 - c) * Sc ** 2 * lc2 ** 2])
    kd = np.max(np.abs(np.array(coefficients(binary_Q(lA, lB), [Sc, 0.0])) - cloud) / np.abs(cloud))
    # cumulants of the optical depth: Var tau = 2 K L - 2 C1 (+ exp small), kappa_3 grows like 6 M3 L
    cum = 0.0
    for Q_, S_ in [(binary_Q(*binary_lams(PA, 0.04)), [SA, SB]), (nested_Q(0.0125), SIG3)]:
        m_, K_, M3_, C1_ = coefficients(Q_, S_)
        k1, k2, k3 = tau_cumulants(Q_, S_, L)
        _, _, k3b = tau_cumulants(Q_, S_, 2 * L)
        cum = max(cum, abs(k1 - m_ * L) / (m_ * L), abs(k2 - (2 * K_ * L - 2 * C1_)) / k2,
                  abs((k3b - k3) / (6 * L) - M3_) / abs(M3_))
    print(f"   cumulants of optical depth vs <Sigma> L, 2KL - 2C1, 6 M3 L: largest relative difference {cum:.1e}")
    ok &= cum < 1e-8
    out['cumulant_diff'] = cum
    print(f"   binary closed form vs Green-Kubo form, largest relative difference {cd:.1e}")
    print(f"   N-colour tessellation form {td:.1e}; broken-cloud form {kd:.1e}")
    print(f"   closed-form log T vs engine at orders 0-2, largest difference {ed:.1e}")
    ok &= cd < 1e-10 and td < 1e-10 and kd < 1e-10 and ed < 1e-9
    out.update(coef_diff=cd, tess_diff=td, cloud_diff=kd, engine_closed_diff=ed)

    print("3. worked example: dense clumps in thin gas")
    rows, errs = [], []
    for lc in LCS:
        lA, lB = binary_lams(PA, lc)
        Q = binary_Q(lA, lB)
        T = exact_T(Q, [SA, SB], L)
        cb = binary_coefficients(SA, SB, lA, lB)
        eng = engine_log_T(Q, [SA, SB], L, ORDERS)
        rm, rp, wt = two_exponential_parts(SA, SB, lA, lB)
        rows.append(dict(lam_c=lc, lamA=lA, lamB=lB, clump_depth=SA * lA, T=T, atomic=math.exp(-cb[0] * L),
                         K=cb[1], M3=cb[2], C1=cb[3], r_minus=rm, weight=wt,
                         T_orders={n: math.exp(eng[n]) for n in ORDERS},
                         closed={n: math.exp(log_T_closed(cb, L, n)) for n in (0, 1, 2)}))
        errs.append({n: abs(eng[n] - math.log(T)) for n in ORDERS})
        print(f"   lam_c {lc:<6} T {T:.6f}  atomic {math.exp(-cb[0]*L):.6f}  ratio {T/math.exp(-cb[0]*L):5.2f}  "
              + "  ".join(f"e{n} {errs[-1][n]:.1e}" for n in ORDERS))
    slopes = {n: math.log2(errs[-2][n] / errs[-1][n]) for n in ORDERS}
    print("   observed orders (last halving): " + ", ".join(f"order {n}: {s:.2f}" for n, s in slopes.items()))
    ok &= all(abs(slopes[n] - (n + 1)) < 0.35 for n in ORDERS)
    out['example'] = rows
    out['errors'] = [dict(lam_c=lc, **{str(n): e[n] for n in ORDERS}) for lc, e in zip(LCS, errs)]
    out['slopes'] = {str(n): s for n, s in slopes.items()}

    print("4. radius of convergence lam_c |Sigma_A - Sigma_B| < 1")
    d = SA - SB
    # decay rate as an analytic function of x = lam_c at fixed volume fraction; Taylor coefficients by FFT
    pA, pB = PA, 1 - PA
    m, c0, beta = pA * SA + pB * SB, pB * SA + pA * SB, pA * pB * d * d

    def rate(x):
        return ((m + c0) * x + 1 - np.sqrt((1 + (c0 - m) * x) ** 2 + 4 * beta * x * x + 0j)) / (2 * x)
    r, N = 0.19, 4096
    z = r * np.exp(2j * np.pi * np.arange(N) / N)
    with np.errstate(all='ignore'):
        coef = (np.fft.fft(rate(z)) / N)[:400] / r ** np.arange(400)
    env = lambda a, b: max(abs(coef[j]) * j ** 1.5 for j in range(a, b))
    growth = (env(300, 320) / env(100, 120)) ** (1 / 200)
    print(f"   Taylor coefficients of the decay rate grow like {growth:.3f}^k; |Sigma_A - Sigma_B| = {d:.3f}; "
          f"radius 1/|Sigma_A - Sigma_B| = {1/d:.4f}")
    ok &= abs(growth / d - 1) < 0.005
    rad = []
    for lc in (0.16, 0.32):
        lA, lB = binary_lams(PA, lc)
        Q = binary_Q(lA, lB)
        T = exact_T(Q, [SA, SB], L)
        eng = engine_log_T(Q, [SA, SB], L, list(range(13)))
        e = [abs(eng[n] - math.log(T)) for n in range(13)]
        rad.append(dict(lam_c=lc, x=lc * d, errors=e))
        print(f"   lam_c {lc}: lam_c|d| = {lc*d:.2f}, errors at orders 0, 4, 8, 12: "
              + ", ".join(f"{e[n]:.2e}" for n in (0, 4, 8, 12)))
    ok &= rad[0]['errors'][12] < 0.01 * rad[0]['errors'][0] and rad[1]['errors'][12] > 10 * rad[1]['errors'][0]
    out['radius'] = dict(growth=growth, d=d, rows=rad)

    print("5. three nested phases: gas, envelopes, cores")
    rows3, errs3 = [], []
    for ell in ELLS:
        Q = nested_Q(ell)
        T = exact_T(Q, SIG3, L)
        c3 = coefficients(Q, SIG3)
        eng = engine_log_T(Q, SIG3, L, [0, 1, 2, 4, 6])
        e = {n: abs(eng[n] - math.log(T)) for n in (0, 1, 2, 4, 6)}
        ed3 = max(abs(eng[n] - log_T_closed(c3, L, n)) for n in (0, 1, 2))
        ok &= ed3 < 1e-9
        rows3.append(dict(ell=ell, gas_chord=1 / -Q[0, 0], T=T, atomic=math.exp(-c3[0] * L), K=c3[1], M3=c3[2],
                          C1=c3[3], T_orders={n: math.exp(eng[n]) for n in (0, 1, 2, 4, 6)}, errors=e))
        errs3.append(e)
        print(f"   ell {ell:<8} T {T:.6f}  atomic {math.exp(-c3[0]*L):.6f}  K {c3[1]:.4f}  "
              + "  ".join(f"e{n} {e[n]:.1e}" for n in e) + f"   closed vs engine {ed3:.1e}")
    sl3 = {n: math.log2(errs3[-2][n] / errs3[-1][n]) for n in (0, 1, 2, 4, 6)}
    print("   observed orders: " + ", ".join(f"order {n}: {s:.2f}" for n, s in sl3.items()))
    ok &= all(abs(sl3[n] - (n + 1)) < 0.35 for n in sl3)
    out['nested'] = rows3
    out['nested_slopes'] = {str(n): s for n, s in sl3.items()}

    print("6. Monte Carlo")
    rng = np.random.default_rng(20260923)
    mcs = []
    for lc, media in [(0.16, 300), (0.08, 300), (0.04, 600)]:
        lA, lB = binary_lams(PA, lc)
        Q = binary_Q(lA, lB)
        T = exact_T(Q, [SA, SB], L)
        tm, ts = mc_tessellation([PA, 1 - PA], [SA, SB], lc, L, media, 50, rng)
        cm, cs = mc_chain(Q, [SA, SB], L, 400000, rng)
        mcs.append(dict(case='binary', lam_c=lc, T=T, tess=tm, tess_se=ts, rays=media * 50, chain=cm, chain_se=cs,
                        atomic=math.exp(-(PA * SA + (1 - PA) * SB) * L)))
        print(f"   lam_c {lc}: matrix exponential {T:.5f}; tessellation rays {tm:.5f} +- {ts:.5f}; "
              f"chord sampling {cm:.5f} +- {cs:.5f}")
        ok &= abs(tm - T) < 3 * ts and abs(cm - T) < 3 * cs
    for ell in ELLS[:2]:
        Q = nested_Q(ell)
        T = exact_T(Q, SIG3, L)
        cm, cs = mc_chain(Q, SIG3, L, 400000, rng)
        mcs.append(dict(case='nested', ell=ell, T=T, chain=cm, chain_se=cs))
        print(f"   nested ell {ell}: matrix exponential {T:.6f}; chord sampling {cm:.6f} +- {cs:.6f}")
        ok &= abs(cm - T) < 3 * cs
    out['mc'] = mcs

    out['pass'] = bool(ok)
    with open(os.path.join(HERE, 'results.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == '__main__':
    main()
