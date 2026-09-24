"""Monte Carlo checks of the Poisson-count, Black-Scholes, Heston, bond-option and three-regime pages.

`python3 verify_option_mc.py` (about a minute). The regime path is simulated exactly, by exponential holding
times, with no time grid. Given the path each model has deterministic, piecewise-constant parameters, so each
path is priced in closed form:

  counts          the count is Poisson with mean int_0^T ell(y_s) ds;
  Black-Scholes   the Black-Scholes price at the path's integrated variance;
  Heston          the Heston price with a time-dependent long-run level, whose characteristic function is
                  exp(D(T) v0 + kappa int_0^T theta(s) D(T - s) ds), through Lewis's formula;
  bond option     given the path, int_0^T x and x_T are jointly Gaussian and the bond at expiry is
                  A_j exp(-b x_T) in the regime j at expiry, so the call is a Black formula;
  three regimes   the Vasicek bond, exp(-mean + variance / 2) of the Gaussian int_0^T x.

The Monte Carlo mean and its standard error are compared with the numerical solution of the reduced linear
system a' = (Q + diag g) a. A check passes when they agree within four standard errors.
"""
import cmath
import math
import os
import sys
import numpy as np
from scipy.special import ndtr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastswitch import numerical_a_callable
from options import bs_call, zcb_call
from models import heston_switching_theta

OK = True


def check(name, mc, se, num):
    global OK
    z = abs(mc - num) / se
    OK &= bool(z < 4)
    print(f"{'ok ' if z < 4 else 'FAIL'} {name}: Monte Carlo {mc:.7g} +/- {se:.2g}, numerical {num:.7g}, "
          f"{z:.2f} standard errors")


def regime_segments(lam, T, start, N, rng, accumulate):
    """Simulate N two-state paths switching at rate lam each way, starting in `start`. For every holding
    period [a, b] in regime y, call accumulate(idx, y, T - a, T - b) on the paths idx that are still running."""
    y = np.full(N, start)
    t = np.zeros(N)
    alive = np.ones(N, bool)
    while alive.any():
        idx = np.where(alive)[0]
        tn = np.minimum(t[idx] + rng.exponential(1 / lam, idx.size), T)
        accumulate(idx, y[idx], T - t[idx], T - tn)
        t[idx] = tn
        alive[idx] = tn < T
        y[idx[alive[idx]]] = 1 - y[idx[alive[idx]]]
    return y


def chain_segments(Q, T, start, N, rng, accumulate):
    """As regime_segments, for any finite generator Q."""
    Q = np.asarray(Q, float)
    out = -np.diag(Q)
    jump = np.where(np.eye(len(Q), dtype=bool), 0.0, Q) / out[:, None]
    cum = np.cumsum(jump, axis=1)
    y = np.full(N, start)
    t = np.zeros(N)
    alive = np.ones(N, bool)
    while alive.any():
        idx = np.where(alive)[0]
        tn = np.minimum(t[idx] + rng.exponential(1 / out[y[idx]]), T)
        accumulate(idx, y[idx], T - t[idx], T - tn)
        t[idx] = tn
        alive[idx] = tn < T
        go = idx[alive[idx]]
        y[go] = (rng.random(go.size)[:, None] > cum[y[go]]).sum(axis=1)
    return y


def vasicek_moments(kap, th, sg, x0, T):
    """Accumulators for the path-conditional mean and variance of x_T and int_0^T x, and their covariance."""
    B = lambda u: (1 - np.exp(-kap * u)) / kap
    G1 = lambda u: (u - B(u)) / kap                                      # int_0^u B
    G2 = lambda u: (u - 2 * B(u) + (1 - np.exp(-2 * kap * u)) / (2 * kap)) / kap ** 2   # int_0^u B^2
    G3 = lambda u: (B(u) - (1 - np.exp(-2 * kap * u)) / (2 * kap)) / kap               # int_0^u B e^{-kap s}
    st = {}

    def init(N):
        st.update(mX=np.full(N, x0 * math.exp(-kap * T)), vX=np.zeros(N), mI=np.full(N, x0 * float(B(T))),
                  vI=np.zeros(N), cIX=np.zeros(N))
        return st

    def acc(idx, y, ta, tb):
        st['mX'][idx] += th[y] * (np.exp(-kap * tb) - np.exp(-kap * ta))
        st['vX'][idx] += sg[y] ** 2 / (2 * kap) * (np.exp(-2 * kap * tb) - np.exp(-2 * kap * ta))
        st['mI'][idx] += kap * th[y] * (G1(ta) - G1(tb))
        st['vI'][idx] += sg[y] ** 2 * (G2(ta) - G2(tb))
        st['cIX'][idx] += sg[y] ** 2 * (G3(ta) - G3(tb))
    return init, acc


def mean_se(v):
    return v.mean(), v.std(ddof=1) / math.sqrt(v.size)


# ------------------------------------------------------------------ Black-Scholes with a switching volatility
def black_scholes(N=400000, seed=11):
    sig, r, S0, T, lam = np.array([0.30, 0.15]), 0.03, 100.0, 1.0, 50.0
    rng = np.random.default_rng(seed)
    V = np.zeros(N)

    def acc(idx, y, ta, tb):
        V[idx] += sig[y] ** 2 * (ta - tb)
    regime_segments(lam, T, 0, N, rng, acc)
    sd = np.sqrt(V)
    Q = [[-lam, lam], [lam, -lam]]
    for K in (90.0, 110.0):
        d1 = (math.log(S0 / K) + r * T + V / 2) / sd
        c = S0 * ndtr(d1) - K * math.exp(-r * T) * ndtr(d1 - sd)
        m, se = mean_se(c)
        check(f"Black-Scholes, lambda 50, strike {K:g}", m, se, bs_call(S0, K, T, r, list(sig), Q, 0))


# ------------------------------------------------------------------ Heston with a switching long-run variance
def heston(N=200000, seed=12, chunk=20000):
    kap, ths, xi, rho, T, v0, S0, lam = 2.0, [0.09, 0.02], 0.4, -0.6, 1.0, 0.04, 100.0, 10.0
    Q = [[-lam, lam], [lam, -lam]]
    x, w = np.polynomial.legendre.leggauss(96)
    U = 40.0
    us, ws = (x + 1) * U / 2, w * U / 2
    uc = us - 0.5j
    d = np.sqrt((rho * xi * 1j * uc - kap) ** 2 + xi ** 2 * (1j * uc + uc * uc))
    gm = (kap - rho * xi * 1j * uc - d) / (kap - rho * xi * 1j * uc + d)

    def D(t):
        e = np.exp(-d * t)
        return (kap - rho * xi * 1j * uc - d) / xi ** 2 * (1 - e) / (1 - gm * e)

    def F(t):                                       # int_0^t D, closed form; t has shape (n, 1)
        return ((kap - rho * xi * 1j * uc - d) * t - 2 * np.log((1 - gm * np.exp(-d * t)) / (1 - gm))) / xi ** 2

    # the closed-form integral of D against quadrature, at a few frequencies
    s, sw = np.polynomial.legendre.leggauss(200)
    s, sw = (s + 1) * T / 2, sw * T / 2
    quad = sum(wi * D(si) for si, wi in zip(s, sw))
    err = np.max(np.abs(quad - F(np.array([[T]]))[0]))
    print(f"{'ok ' if err < 1e-10 else 'FAIL'} Heston: closed-form integral of D against quadrature, max error {err:.1e}")

    strikes = (80.0, 90.0, 100.0, 110.0, 120.0)
    kk = np.log(S0 / np.array(strikes))
    rng = np.random.default_rng(seed)
    prices = []
    FT, DT = F(np.array([[T]]))[0], D(T)
    for start in range(0, N, chunk):
        n = min(chunk, N - start)
        acc = np.zeros((n, us.size), complex)       # int over regime-1 periods of D(T - s) ds

        def accumulate(idx, y, ta, tb):
            one = idx[y == 0]
            if one.size:
                acc[one] += F(ta[y == 0][:, None]) - F(tb[y == 0][:, None])
        regime_segments(lam, T, 0, n, rng, accumulate)
        phi = np.exp(DT * v0 + kap * (ths[1] * FT + (ths[0] - ths[1]) * acc))
        tot = np.real(np.exp(1j * np.outer(kk, us))[None, :, :] * phi[:, None, :]) / (us * us + 0.25)
        prices.append(S0 - np.sqrt(S0 * np.array(strikes))[None, :] / math.pi * (tot @ ws))
    prices = np.vstack(prices)

    cfs = [cmath.exp(heston_switching_theta(u - 0.5j, kap, ths, xi, rho, T)[2](T) * v0)
           * numerical_a_callable(T, Q, heston_switching_theta(u - 0.5j, kap, ths, xi, rho, T)[1], rtol=1e-12)[0]
           for u in us]
    for j, K in enumerate(strikes):
        num = S0 - math.sqrt(S0 * K) / math.pi * sum(
            wi * (cmath.exp(1j * u * kk[j]) * c).real / (u * u + 0.25) for c, u, wi in zip(cfs, us, ws))
        m, se = mean_se(prices[:, j])
        check(f"Heston, lambda 10, strike {K:g}", m, se, num)


# ------------------------------------------------------------------ call on a zero-coupon bond, Vasicek
def bond_option(N=400000, seed=13):
    kap, th, sg, x0, T, S, lam = 0.5, np.array([0.05, 0.03]), np.array([0.015, 0.010]), 0.04, 1.0, 4.0, 50.0
    Q = [[-lam, lam], [lam, -lam]]
    b = (1 - math.exp(-kap * (S - T))) / kap
    tau = S - T
    Bf = lambda t: (1 - math.exp(-kap * t)) / kap
    gf = [(lambda i: (lambda t: -kap * th[i] * Bf(t) + 0.5 * sg[i] ** 2 * Bf(t) ** 2))(i) for i in range(2)]
    A = np.real(numerical_a_callable(tau, Q, gf, rtol=1e-12))  # bond factor at expiry, by regime then
    init, acc = vasicek_moments(kap, th, sg, x0, T)
    st = init(N)
    yT = regime_segments(lam, T, 0, N, np.random.default_rng(seed), acc)
    disc = np.exp(-st['mI'] + st['vI'] / 2)                              # E[exp(-int x) | path]
    m = st['mX'] - st['cIX']                                             # mean of x_T under the discounted measure
    sd = b * np.sqrt(st['vX'])
    fwd = A[yT] * np.exp(-b * m + sd * sd / 2)
    for K in (0.88, 0.90):
        d1 = (np.log(fwd / K) + sd * sd / 2) / sd
        c = disc * (fwd * ndtr(d1) - K * ndtr(d1 - sd))
        mc, se = mean_se(c)
        check(f"bond call, lambda 50, strike {K:g}", mc, se, zcb_call(T, S, K, x0, 0, kap, list(th), list(sg), Q))


# ------------------------------------------------------------------ Vasicek bond, three irreversible regimes
def three_regimes(N=400000, seed=15):
    """Given the path, int_0^T x is Gaussian, so the bond price given the path is exp(-mean + variance / 2)."""
    kap, th, s2, T = 2.0, np.array([0.20, 0.08, 0.02]), np.array([0.04, 0.01, 0.0025]), 1.0
    base = np.array([[-3, 2, 1], [1, -2, 1], [0.5, 1.5, -2]], float)
    Bf = lambda t: (1 - math.exp(-kap * t)) / kap
    gf = [(lambda i: (lambda t: -kap * th[i] * Bf(t) + 0.5 * s2[i] * Bf(t) ** 2))(i) for i in range(3)]
    for sc in (4.0, 8.0):
        init, acc = vasicek_moments(kap, th, np.sqrt(s2), 0.0, T)
        st = init(N)
        chain_segments(sc * base, T, 0, N, np.random.default_rng(seed), acc)
        m, se = mean_se(np.exp(-st['mI'] + st['vI'] / 2))
        num = float(np.real(numerical_a_callable(T, sc * base, gf, rtol=1e-12)[0]))
        check(f"three regimes, s = {sc:g}, bond at t = 1 from regime 1", m, se, num)


# ------------------------------------------------------------------ Markov-modulated Poisson counts
def counts(N=400000, seed=14):
    """Given the path the count is Poisson with mean Lambda = int ell(y_s) ds, so P(N = k | path) is exact."""
    from scipy.linalg import expm
    from scipy.stats import poisson
    ell, T, lam, M = np.array([8.0, 1.0]), 1.0, 10.0, 64
    rng = np.random.default_rng(seed)
    Lam = np.zeros(N)

    def acc(idx, y, ta, tb):
        Lam[idx] += ell[y] * (ta - tb)
    regime_segments(lam, T, 0, N, rng, acc)
    Ql = np.array([[-lam, lam], [lam, -lam]])
    zs = np.exp(2j * np.pi * np.arange(M) / M)
    p = np.real(np.fft.fft([(expm((Ql + np.diag((z - 1) * ell)) * T) @ np.ones(2))[0] for z in zs])) / M
    worst = 0.0
    for k in range(16):
        m, se = mean_se(poisson.pmf(k, Lam))
        worst = max(worst, abs(m - p[k]) / se)
    global OK
    OK &= bool(worst < 4)
    print(f"{'ok ' if worst < 4 else 'FAIL'} Poisson counts, lambda 10: P(N = k) for k = 0..15 against the "
          f"generating function, largest gap {worst:.2f} standard errors")
    L = 1 - math.exp(-2 * lam * T)
    eps, lb, lt = 1 / lam, ell.mean(), (ell[0] - ell[1]) / 2
    check("Poisson counts, mean", *mean_se(Lam), lb * T + eps / 2 * lt * L)
    var_formula = lb * T + eps / 2 * lt * L + eps * lt ** 2 * T - eps ** 2 * lt ** 2 * (L / 2 + L * L / 4)
    batches = [Lam[s].mean() + Lam[s].var(ddof=1) for s in np.array_split(np.arange(N), 40)]
    check("Poisson counts, variance", Lam.mean() + Lam.var(ddof=1), np.std(batches, ddof=1) / math.sqrt(40),
          var_formula)


if __name__ == '__main__':
    counts()
    black_scholes()
    heston()
    bond_option()
    three_regimes()
    print("PASS" if OK else "FAIL")
