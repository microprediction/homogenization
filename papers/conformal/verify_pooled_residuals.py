"""Certificate for pooled conformal residuals under a fast hidden regime.

Symmetric two-state chain, flip rate 1/eps each way, horizon Delta, c = Delta / eps, score noise sigma Z, alpha = 0.1.
1. Terminal score S = mu y_Delta + sigma Z: the pooled threshold's conditional coverage error given y_0 is
   +- (e^{-2c}/2) [Phi((q - mu)/sigma) - Phi((q + mu)/sigma)], exponential in c; checked by simulation.
2. Path-average score S = mu A + sigma Z, A = Delta^-1 int y dt: E[A | y_0 = i] = i (1 - e^{-2c}) / (2c), and the
   exact conditional CDF by Fourier inversion of the 2x2 transform agrees with simulation; the coverage error is
   algebraic in c, with first-order form -(mu/sigma) phi(z_{1-alpha}) E[A | i]; the two scores'
   errors are compared: the path score is never better and the ratio grows like e^{2c}/(4c).
3. Slow internal state dX = -kappa (X - y) dt: stationary (X + 1)/2 ~ Beta(lam/kappa, lam/kappa) and
   P(y = +1 | X = x) = (1 + x)/2 (Fokker-Planck), E[X_t | X_0 = x] = x (2 lam e^{-kappa t} - kappa e^{-2 lam t})/(2 lam - kappa),
   so the coverage error given the predictor's state relaxes at rate kappa, not 2 lam; checked by simulation.
4. Calibration dependence at spacing h (terminal score): Cov(1{S_0 <= q}, 1{S_jh <= q}) = e^{-2jh/eps} (F_+ - F_-)^2 / 4 with
   F_+- the CDFs given the regime at the forecast endpoint, hence the variance inflation of the empirical CDF, independent of
   the horizon; checked by simulation.
5. Thresholds: pooled against regime-aware (Mondrian) quantiles, conditional coverage and width.
"""
import json, math, os
import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, beta as beta_dist
from pooled import cdf_terminal, cdf_path, mean_path, pooled_quantile, simulate_occupation, simulate_slow_state, relax

ALPHA, MU, SIGMA = 0.1, 1.0, 1.0


def main():
    ok, out = True, {}
    rng = np.random.default_rng(7)
    z = norm.ppf(1 - ALPHA)

    print("1. terminal score: exponential conditional-coverage error")
    rows = []
    for c in (0.25, 0.5, 1.0, 2.0, 4.0):
        cdf = lambda q, i: cdf_terminal(q, i, MU, SIGMA, c)
        q = pooled_quantile(cdf, ALPHA)
        err = cdf(q, +1) - (1 - ALPHA)
        formula = -0.5 * math.exp(-2 * c) * (norm.cdf((q + MU) / SIGMA) - norm.cdf((q - MU) / SIGMA))
        rows.append([c, q, err, formula])
        ok &= abs(err - formula) < 1e-12 and abs(cdf(q, -1) - (1 - ALPHA) + err) < 1e-12
        print(f"   c={c:4.2f}: pooled q {q:.4f}, coverage given +1 minus 0.9: {err:+.5f} (formula {formula:+.5f})")
    n = 2_000_000
    c = 1.0
    y0 = rng.choice([-1.0, 1.0], n)
    same = rng.uniform(size=n) < 0.5 * (1 + math.exp(-2 * c))
    yD = np.where(same, y0, -y0)
    S = MU * yD + SIGMA * rng.standard_normal(n)
    q = pooled_quantile(lambda q, i: cdf_terminal(q, i, MU, SIGMA, c), ALPHA)
    sim = np.mean(S[y0 > 0] <= q) - (1 - ALPHA)
    exact = cdf_terminal(q, +1, MU, SIGMA, c) - (1 - ALPHA)
    ok &= abs(sim - exact) < 4 * math.sqrt(0.09 / (n / 2))
    print(f"   c=1 simulation {sim:+.5f} vs exact {exact:+.5f} ({n} paths)")
    out['terminal'] = rows

    print("2. path-average score: algebraic conditional-coverage error")
    rows = []
    for c in (0.25, 0.5, 1.0, 2.0, 4.0, 8.0):
        row = [c]
        for mu in (MU, 0.3):
            cdf = lambda q, i, mu=mu: cdf_path(q, i, mu, SIGMA, c)
            q = pooled_quantile(cdf, ALPHA)
            err = cdf(q, +1) - (1 - ALPHA)
            first = -(mu / SIGMA) * norm.pdf(z) * mean_path(+1, c)
            row += [q, err, first]
        qt = pooled_quantile(lambda q, i: cdf_terminal(q, i, MU, SIGMA, c), ALPHA)
        errt = cdf_terminal(qt, +1, MU, SIGMA, c) - (1 - ALPHA)
        row.append(errt)
        rows.append(row)
        print(f"   c={c:4.2f}: mu=1: pooled q {row[1]:.4f}, coverage error given +1 {row[2]:+.5f} (first order in mu {row[3]:+.5f});"
              f" mu=0.3: {row[5]:+.5f} ({row[6]:+.5f}); terminal score, mu=1: {errt:+.5f}; ratio path/terminal {row[2] / errt:.1f}")
    ok &= abs(rows[-1][2] * 2 * 8.0 / (-(MU / SIGMA) * norm.pdf(z)) - 1) < 0.15      # ~ eps / (2 Delta) tail
    ok &= all(abs(r[5] / r[6] - 1) < 0.1 for r in rows)                             # first order accurate at mu = 0.3
    c = 2.0
    y0, A = simulate_occupation(n, c, rng)
    S = MU * A + SIGMA * rng.standard_normal(n)
    q = pooled_quantile(lambda q, i: cdf_path(q, i, MU, SIGMA, c), ALPHA)
    sim = np.mean(S[y0 > 0] <= q)
    exact = cdf_path(q, +1, MU, SIGMA, c)
    ok &= abs(sim - exact) < 4 * math.sqrt(0.09 / (n / 2)) and abs(A[y0 > 0].mean() - mean_path(1, c)) < 4 * A.std() / math.sqrt(n / 2)
    print(f"   c=2 simulation: coverage given +1 {sim:.5f} vs exact {exact:.5f}; E[A | +1] {A[y0 > 0].mean():.5f} vs {mean_path(1, c):.5f}")
    out['path'] = rows

    print("3. a predictor with a slow internal state")
    lam, kappa = 5.0, 1.0
    # stationary law by a long exact simulation of the piecewise-deterministic pair (X, y)
    nS, T = 400_000, 12.0
    x = simulate_slow_state(nS, 0.0, lam, kappa, T, rng)      # from X_0 = 0 with y_0 fair, T >> 1/kappa: near stationary
    a = lam / kappa
    edges = np.linspace(-1, 1, 21)
    hist = np.histogram(x, edges)[0] / nS
    theo = np.diff(beta_dist.cdf((edges + 1) / 2, a, a))
    ok &= np.abs(hist - theo).max() < 0.004
    print(f"   stationary law vs Beta({a:.0f}, {a:.0f}) on 20 bins: largest gap {np.abs(hist - theo).max():.4f}; Var X {x.var():.4f} vs {1 / (2 * a + 1):.4f}")
    rows = []
    for t in (0.5, 1.0, 2.0, 4.0):
        x0 = 0.6
        xt = simulate_slow_state(nS, x0, lam, kappa, t, rng)
        S = MU * xt + SIGMA * rng.standard_normal(nS)
        # pooled threshold: stationary score = mu X + sigma Z, X ~ Beta law; quantile by simulation of the stationary law
        Sst = MU * x + SIGMA * rng.standard_normal(nS)
        q = np.quantile(Sst, 1 - ALPHA)
        cov = np.mean(S <= q)
        first = -(MU / SIGMA) * norm.pdf(z) * x0 * relax(t, lam, kappa)
        rows.append([t, xt.mean(), x0 * relax(t, lam, kappa), cov - (1 - ALPHA), first, math.exp(-2 * lam * t)])
        ok &= abs(xt.mean() - x0 * relax(t, lam, kappa)) < 4 * xt.std() / math.sqrt(nS)
        print(f"   t={t:3.1f}: E[X_t | X_0 = 0.6] {xt.mean():+.4f} vs {x0 * relax(t, lam, kappa):+.4f}; coverage error {cov - (1 - ALPHA):+.4f}"
              f" (first order {first:+.4f}); endpoint memory e^(-2 lam t) = {math.exp(-2 * lam * t):.1e}")
    out['slow_state'] = rows

    print("4. calibration dependence at spacing h, terminal score, c = 1")
    c = 1.0
    cdf = lambda q, i: cdf_terminal(q, i, MU, SIGMA, c)
    q = pooled_quantile(cdf, ALPHA)
    dF = norm.cdf((q - MU) / SIGMA) - norm.cdf((q + MU) / SIGMA)     # CDFs given the regime at the forecast endpoint
    N = 200
    rows = []
    for h in (0.25, 1.0, 4.0):     # in units of eps
        r = math.exp(-2 * h)
        inflation = 1 + 2 * sum((N - j) / N * r ** j for j in range(1, N)) * (dF ** 2 / 4) / (ALPHA * (1 - ALPHA))
        # simulate N terminal scores at spacing h: regimes at the forecast endpoints form a Markov chain with correlation r
        reps = 20000
        y = rng.choice([-1.0, 1.0], reps)
        F = np.zeros(reps)
        for j in range(N):
            if j:
                flip = rng.uniform(size=reps) > 0.5 * (1 + r)
                y = np.where(flip, -y, y)
            S = MU * y + SIGMA * rng.standard_normal(reps)
            F += (S <= q)
        F /= N
        sim = F.var() / (ALPHA * (1 - ALPHA) / N)
        rows.append([h, r, inflation, sim])
        ok &= abs(sim / inflation - 1) < 0.06
        print(f"   h={h:4.2f} eps: endpoint correlation {r:.4f}, variance inflation of the empirical CDF {inflation:.4f}, simulated {sim:.4f}")
    out['calibration'] = rows

    print("5. thresholds: pooled against regime-aware, alpha = 0.1")
    rows = []
    for c in (0.5, 1.0, 2.0):
        for name, cdf in (('terminal', lambda q, i, c=c: cdf_terminal(q, i, MU, SIGMA, c)),
                          ('path', lambda q, i, c=c: cdf_path(q, i, MU, SIGMA, c))):
            q = pooled_quantile(cdf, ALPHA)
            qp = brentq(lambda x: cdf(x, +1) - (1 - ALPHA), -20, 20)
            qm = brentq(lambda x: cdf(x, -1) - (1 - ALPHA), -20, 20)
            rows.append([c, name, q, cdf(q, +1), cdf(q, -1), qp, qm])
            print(f"   c={c:3.1f} {name:8s}: pooled q {q:.4f} covers {cdf(q, +1):.4f} | {cdf(q, -1):.4f}; regime-aware q_+ {qp:.4f}, q_- {qm:.4f}")
    out['thresholds'] = rows

    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == '__main__':
    main()
