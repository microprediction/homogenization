"""Certificate for the first-order likelihood of a discretely observed Vasicek process with fast switching.

1. The corrected transition density approaches the exact one (Fourier inversion) at second order in 1/(lam D).
2. Two different three-state chains with the same averages and Green-Kubo matrix have transition densities that agree
   to first order.
3. On exactly simulated data, residual skewness, excess kurtosis and variance inflation match -6 A3/v^{3/2},
   24 A4/v^2 and 1 + 2 A2/v.
4. The score statistic with the kurtosis part projected onto the positive half-line, LM+ = n (S^2/6 + max(K,0)^2/24),
   has the chi-bar-square null law (chi_1^2 + chi_2^2)/2 mixture: its 95% point is 5.1384, and Monte Carlo under the
   Gaussian null rejects at 5% there, while the unprojected square over-rejects at that point.
5. The composite likelihood (stationary-start transition densities) against the exact filtered likelihood on a
   fixed panel: the gap falls like lam^-2, and a wrong initial prior costs one term of order lam^-1.
"""
import json, math, os, sys
import numpy as np
from density import coefficients, averaged, density_first_order, density_exact, kernel_exact, stationary
from simulate import simulate
from scipy.stats import chi2
from scipy.optimize import brentq
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'yield-curve'))


def main():
    ok, out = True, {}
    kappa, th, s, D, x = 0.5, [0.08, 0.02], [0.0009, 0.0001], 1 / 12, 0.05
    print("1. transition density, two regimes, monthly observations")
    rows, errs = [], []
    for lam in (20.0, 40.0, 80.0):
        Q = lam * np.array([[-1, 1], [1, -1.0]])
        m, v = averaged(Q, kappa, th, s, x, D)
        xs = m + math.sqrt(v) * np.linspace(-4, 4, 17)
        ex = density_exact(xs, Q, kappa, th, s, x, D, n=300)
        f1, f0 = density_first_order(xs, Q, kappa, th, s, x, D)
        e0, e1 = np.abs(ex - f0).max() / f0.max(), np.abs(ex - f1).max() / f0.max()
        rows.append([lam, lam * D, e0, e1]); errs.append(e1)
        print(f"   lam {lam:4.0f} (lam D = {lam*D:.1f}): error of Gaussian {e0:.2%}, of first order {e1:.2%} of peak density")
    ok &= errs[-2] / errs[-1] > 3.0 and errs[-1] < 0.01
    out['density'] = rows
    print("2. matched three-state chains")
    from verify_three_numbers import QA, THA, SA, QB, matched_chain, invariants
    thB, sB, _ = matched_chain(QB, invariants(QA, THA, SA))
    rows, gaps = [], []
    for m_ in (20, 40, 80):
        mA, vA = averaged(m_ * QA, 1.0, THA, SA, 0.04, 0.25)
        xs = mA + math.sqrt(vA) * np.linspace(-4, 4, 13)
        eA = density_exact(xs, m_ * QA, 1.0, THA, SA, 0.04, 0.25, n=260)
        eB = density_exact(xs, m_ * QB, 1.0, thB, sB, 0.04, 0.25, n=260)
        f1, f0 = density_first_order(xs, m_ * QA, 1.0, THA, SA, 0.04, 0.25)
        peak = f0.max()
        rows.append([m_, np.abs(eA - f0).max() / peak, np.abs(eA - eB).max() / peak]); gaps.append(np.abs(eA - eB).max())
        print(f"   scale {m_}: chain A vs Gaussian {rows[-1][1]:.3%}, chain A vs chain B {rows[-1][2]:.4%}")
    ok &= gaps[0] / gaps[1] > 3.5 and gaps[1] / gaps[2] > 3.5
    out['matched'] = rows
    print("3. simulated moments, 200,000 monthly observations")
    rng = np.random.default_rng(3)
    rows = []
    for lam in (60.0, 120.0, 240.0):
        Q = lam * np.array([[-1, 1], [1, -1.0]])
        xs = simulate(200000, D, kappa, th, s, lam, 0.05, rng)
        r = xs[1:] - (xs[:-1] * math.exp(-kappa * D) + np.mean(th) * (1 - math.exp(-kappa * D)))
        A2, A3, A4 = coefficients(Q, kappa, th, s, D)
        v = np.mean(s) * (1 - math.exp(-2 * kappa * D)) / (2 * kappa)
        z = r / r.std()
        row = [lam, float(np.mean(z ** 3)), -6 * A3 / v ** 1.5, float(np.mean(z ** 4) - 3), 24 * A4 / v ** 2,
               float(r.var() / v), 1 + 2 * A2 / v]
        rows.append(row)
        print(f"   lam {lam:4.0f}: skew {row[1]:+.4f} vs {row[2]:+.4f}, excess kurtosis {row[3]:.4f} vs {row[4]:.4f},"
              f" variance ratio {row[5]:.4f} vs {row[6]:.4f}")
        ok &= abs(row[3] - row[4]) < 0.06 and abs(row[1] - row[2]) < 0.03
    out['moments'] = rows

    print("4. the one-sided score statistic: chi-bar-square null")
    cbar = brentq(lambda c: 0.5 * chi2.sf(c, 1) + 0.5 * chi2.sf(c, 2) - 0.05, 3, 8)
    print(f"   95% point of (chi_1^2 + chi_2^2)/2 mixture {cbar:.6f}; of chi_2^2 {chi2.ppf(0.95, 2):.6f}")
    rng = np.random.default_rng(5)
    n, reps, chunk = 4000, 20000, 1000
    rej_plus, rej_sq, rej_jb, neg = 0, 0, 0, 0
    for _ in range(reps // chunk):
        z = rng.standard_normal((chunk, n))
        z = (z - z.mean(1, keepdims=True)) / z.std(1, keepdims=True)
        S, K = (z ** 3).mean(1), (z ** 4).mean(1) - 3
        lm_plus = n * (S ** 2 / 6 + np.maximum(K, 0) ** 2 / 24)
        jb = n * (S ** 2 / 6 + K ** 2 / 24)
        rej_plus += (lm_plus > cbar).sum(); rej_sq += (jb > cbar).sum(); rej_jb += (jb > chi2.ppf(0.95, 2)).sum()
        neg += (K < 0).sum()
    rej_plus, rej_sq, rej_jb, neg = rej_plus / reps, rej_sq / reps, rej_jb / reps, neg / reps
    print(f"   {reps} Gaussian samples of size {n}: LM+ rejects {rej_plus:.4f} at {cbar:.4f};"
          f" the unprojected square rejects {rej_sq:.4f} there and {rej_jb:.4f} at the chi_2^2 point;"
          f" negative excess kurtosis in {neg:.4f} of samples")
    ok &= abs(rej_plus - 0.05) < 0.005 and rej_sq > 0.07
    u = rng.uniform(size=n)
    u = (u - u.mean()) / u.std()
    S, K = (u ** 3).mean(), (u ** 4).mean() - 3
    print(f"   a standardized uniform sample: excess kurtosis {K:+.4f}, Jarque-Bera {n * (S ** 2 / 6 + K ** 2 / 24):.1f},"
          f" LM+ {n * (S ** 2 / 6 + max(K, 0) ** 2 / 24):.2f}")
    out['cone'] = {'critical': cbar, 'size_plus': rej_plus, 'size_square': rej_sq, 'size_jb': rej_jb, 'negative_share': neg,
                   'uniform': [float(K), n * (S ** 2 / 6 + K ** 2 / 24), n * (S ** 2 / 6 + max(K, 0) ** 2 / 24)]}

    print("5. composite likelihood against the exact filtered likelihood, a fixed panel of 8 transitions")
    xs = simulate(8, D, kappa, th, s, 120.0, x, np.random.default_rng(11))
    rows = []
    for lam in (160.0, 320.0, 640.0, 1280.0):
        Q = lam * np.array([[-1, 1], [1, -1.0]])
        pi = stationary(Q)
        Ks = [kernel_exact(xs[k], xs[k + 1], Q, kappa, th, s, D, steps=int(2.5 * lam)) for k in range(8)]
        lC = sum(math.log(pi @ K @ np.ones(2)) for K in Ks)

        def filtered(nu):
            terms = []
            for K in Ks:
                p = nu @ K @ np.ones(2)
                terms.append(math.log(p))
                nu = (nu @ K) / p
            return terms
        fF = filtered(pi)
        fS = filtered(np.array([1.0, 0.0]))
        gap = sum(fF) - lC
        first = fS[0] - math.log(pi @ Ks[0] @ np.ones(2))
        tail = sum(fS[1:]) - sum(math.log(pi @ K @ np.ones(2)) for K in Ks[1:])
        rows.append([lam, gap, first, tail])
        print(f"   lam {lam:4.0f}: filtered - composite {gap:+.3e} (x lam^2: {gap * lam ** 2:+.3f});"
              f" from regime 1: first term {first:+.3e}, later terms {tail:+.3e}")
    r = np.array(rows)
    o_gap = np.log2(np.abs(r[-2, 1] / r[-1, 1])); o_first = np.log2(np.abs(r[-2, 2] / r[-1, 2])); o_tail = np.log2(np.abs(r[-2, 3] / r[-1, 3]))
    print(f"   orders from the last doubling: gap {o_gap:.3f}, first term {o_first:.3f}, later terms {o_tail:.3f}")
    ok &= abs(o_gap - 2) < 0.2 and abs(o_first - 1) < 0.15 and abs(o_tail - 2) < 0.2
    out['filtered'] = rows
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == '__main__':
    main()
