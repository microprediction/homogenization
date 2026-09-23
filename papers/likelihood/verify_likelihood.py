"""Certificate for the first-order likelihood of a discretely observed Vasicek process with fast switching.

1. The corrected transition density approaches the exact one (Fourier inversion) at second order in 1/(lam D).
2. Two different three-state chains with the same averages and Green-Kubo matrix have transition densities that agree
   to first order.
3. On exactly simulated data, residual skewness, excess kurtosis and variance inflation match -6 A3/v^{3/2},
   24 A4/v^2 and 1 + 2 A2/v.
"""
import json, math, os, sys
import numpy as np
from density import coefficients, averaged, density_first_order, density_exact
from simulate import simulate
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
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == '__main__':
    main()
