"""Regime-switching correlation: closed forms and references.

Two assets, log prices x1, x2, covariance entries c_jk(y) switched by a fast Markov chain y.

Part 1 (pricing measure, volatilities fixed, only rho switches). The switched operator sigma1 sigma2 d_12 has
constant coefficients, so it commutes with the rest and the price is exactly E[C(rho_hat_T)], rho_hat_T the
time-average of rho. The first-order rule gives
    C ~ C(rho_bar) + (K_rhorho / T) d^2C/drho^2.
For the exchange option (S1 - q S2)^+ this is Margrabe at the averaged exchange variance plus a volga term:
    C ~ M(Vbar) + T K_ss d^2M/dV^2,   K_ss = 4 sigma1^2 sigma2^2 K_rhorho,
and the implied correlation is, to first order, a parabola in log-moneyness k = log(S1 / (q S2)):
    rho_imp(k) ~ rho_bar + (K_rhorho sigma1 sigma2 / (sbar T)) (1 + sbar T / 4 - k^2 / (sbar T)).

Part 2 (physical measure, drifts, variances and correlation co-switch). Third cumulant of any portfolio w'X_t:
    kappa_3(w) ~ 6 t K(w'mu, w'c w),
and the cross-cumulants kappa_112 ~ 2t [2 K(mu1, c12) + K(mu2, c11)], kappa_122 ~ 2t [2 K(mu2, c12) + K(mu1, c22)].

Two-state chain: calm (state 0) -> crisis (state 1) at rate a, crisis -> calm at rate b.
    pi = (b, a) / (a + b),   K(f, h) = pi0 pi1 (f0 - f1)(h0 - h1) / (a + b).
"""
import math
import numpy as np
from scipy.linalg import expm
from scipy.special import i0e, i1e, ndtr
from scipy.optimize import brentq


# ---------------------------------------------------------------- the chain
def pi2(a, b):
    return np.array([b, a]) / (a + b)


def K2(f, h, a, b):
    """Green-Kubo covariance of two regime functions for the two-state chain."""
    p = pi2(a, b)
    return p[0] * p[1] * (f[0] - f[1]) * (h[0] - h[1]) / (a + b)


def Q2(a, b):
    return np.array([[-a, a], [b, -b]], float)


# ---------------------------------------------------------------- occupation time of the crisis state
def occupation_nodes(T, a, b, n=80):
    """Nodes v and weights w with sum w f(v) = E_pi[f(tau)], tau the time spent in state 1 on [0, T].

    Start in 0, end in 0 (n >= 1 visits to 1):  e^{-a u - b v} sqrt(a b u / v) I1(2 sqrt(a b u v)),
    start in 0, end in 1:                     a e^{-a u - b v} I0(2 sqrt(a b u v)),     u = T - v,
    start in 0, never leave:                  atom e^{-a T} at v = 0,
    and the same with the roles of the states exchanged for a start in 1.
    """
    x, wq = np.polynomial.legendre.leggauss(n)
    v = 0.5 * T * (x + 1)
    wq = 0.5 * T * wq
    u = T - v
    z = 2 * np.sqrt(a * b * u * v)
    # exponentially scaled Bessel functions: I(z) e^{-a u - b v} = Ie(z) e^{z - a u - b v}
    E = np.exp(z - a * u - b * v)
    s = np.sqrt(a * b * u / v) * i1e(z) * E + a * i0e(z) * E            # start calm, tau = v
    s1 = np.sqrt(a * b * v / u) * i1e(z) * E + b * i0e(z) * E           # start crisis, time in calm = u
    p = pi2(a, b)
    nodes = np.concatenate([[0.0], v, [T]])
    w = np.concatenate([[p[0] * math.exp(-a * T)], wq * (p[0] * s + p[1] * s1), [p[1] * math.exp(-b * T)]])
    return nodes, w


def simulate_occupation(T, a, b, n, rng):
    """Exact samples of tau, the time in state 1 on [0, T], from a stationary start."""
    p = pi2(a, b)
    state = (rng.random(n) < p[1]).astype(int)
    t = np.zeros(n)
    tau = np.zeros(n)
    alive = np.ones(n, bool)
    while alive.any():
        rate = np.where(state == 1, b, a)
        hold = rng.exponential(1.0, n) / rate
        dt = np.minimum(hold, T - t)
        tau += np.where(alive & (state == 1), dt, 0.0)
        t = np.where(alive, t + dt, t)
        alive &= t < T
        state = np.where(alive, 1 - state, state)
    return tau


# ---------------------------------------------------------------- Margrabe
def margrabe(S1, qS2, V):
    """Exchange option (S1 - q S2)^+ with exchange variance V = s T (vectorised in V)."""
    V = np.asarray(V, float)
    sv = np.sqrt(V)
    d1 = (np.log(S1 / qS2) + V / 2) / sv
    return S1 * ndtr(d1) - qS2 * ndtr(d1 - sv)


def margrabe_greeks(S1, qS2, V):
    """dM/dV and d^2M/dV^2."""
    sv = math.sqrt(V)
    d1 = (math.log(S1 / qS2) + V / 2) / sv
    d2 = d1 - sv
    phi = math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi)
    return S1 * phi / (2 * sv), S1 * phi * (d1 * d2 - 1) / (4 * V ** 1.5)


def exch_var(s1, s2, rho):
    return s1 * s1 + s2 * s2 - 2 * rho * s1 * s2


def margrabe_first_order(S1, qS2, T, s1, s2, rhobar, Krr):
    V = exch_var(s1, s2, rhobar) * T
    Kss = 4 * s1 * s1 * s2 * s2 * Krr
    return float(margrabe(S1, qS2, V)) + T * Kss * margrabe_greeks(S1, qS2, V)[1]


def implied_corr_margrabe(price, S1, qS2, T, s1, s2):
    f = lambda r: float(margrabe(S1, qS2, exch_var(s1, s2, r) * T)) - price
    return brentq(f, -0.999, 0.999, xtol=1e-14)


def implied_corr_parabola(k, T, s1, s2, rhobar, Krr):
    """First-order implied correlation of the exchange option at log-moneyness k = log(S1 / (q S2))."""
    W = exch_var(s1, s2, rhobar) * T
    return rhobar + Krr * s1 * s2 / W * (1 + W / 4 - k * k / W)


def margrabe_fourier(S1, qS2, T, s1, s2, rhos, Q, n=4000, umax=200.0):
    """Exchange option by Lewis's formula, characteristic function of log(S1/S2) from exp(T (Q + diag g))."""
    pi = np.linalg.solve(np.vstack([Q.T[:-1], np.ones(len(Q))]), np.r_[np.zeros(len(Q) - 1), 1.0])
    s = np.array([exch_var(s1, s2, r) for r in rhos])
    x, w = np.polynomial.legendre.leggauss(n)
    u = 0.5 * umax * (x + 1)
    w = 0.5 * umax * w
    z0 = S1 / qS2
    tot = 0.0
    for uj, wj in zip(u, w):
        uu = uj - 0.5j
        g = -0.5 * s * (1j * uu + uu * uu)
        phi = pi @ expm(T * (Q + np.diag(g))) @ np.ones(len(Q))
        tot += wj * (np.exp(1j * uj * math.log(z0)) * phi).real / (uj * uj + 0.25)
    return qS2 * (z0 - math.sqrt(z0) / math.pi * tot)


# ---------------------------------------------------------------- spread option (S1 - S2 - K)^+
_GH = np.polynomial.hermite_e.hermegauss(120)


def spread_price(S1, S2, K, T, r, s1, s2, rho):
    """Spread call at constant correlation: condition on the second asset (Gauss-Hermite), Black-Scholes in the first."""
    z, w = _GH
    w = w / math.sqrt(2 * math.pi)
    rho = np.atleast_1d(np.asarray(rho, float))[:, None]
    S2T = S2 * np.exp((r - s2 * s2 / 2) * T + s2 * math.sqrt(T) * z)[None, :]
    F1 = S1 * np.exp(r * T - 0.5 * rho * rho * s1 * s1 * T + rho * s1 * math.sqrt(T) * z[None, :])
    v = s1 * s1 * (1 - rho * rho) * T
    Kc = S2T + K
    pos = Kc > 0
    Kp = np.where(pos, Kc, 1.0)
    d1 = (np.log(F1 / Kp) + v / 2) / np.sqrt(v)
    call = np.where(pos, F1 * ndtr(d1) - Kp * ndtr(d1 - np.sqrt(v)), F1 - Kc)
    out = math.exp(-r * T) * (call * w[None, :]).sum(axis=1)
    return out if out.size > 1 else float(out[0])


def spread_rho_derivs(S1, S2, K, T, r, s1, s2, rho, h=0.02):
    """d/drho and d^2/drho^2 of the quadrature price, five-point stencils."""
    c = spread_price(S1, S2, K, T, r, s1, s2, [rho - 2 * h, rho - h, rho, rho + h, rho + 2 * h])
    d1 = (c[0] - 8 * c[1] + 8 * c[3] - c[4]) / (12 * h)
    d2 = (-c[0] + 16 * c[1] - 30 * c[2] + 16 * c[3] - c[4]) / (12 * h * h)
    return d1, d2


def implied_corr_spread(price, S1, S2, K, T, r, s1, s2):
    f = lambda q: spread_price(S1, S2, K, T, r, s1, s2, q) - price
    return brentq(f, -0.99, 0.99, xtol=1e-13)


# ---------------------------------------------------------------- part 2: cumulants of log returns
def cov_entries(sig1, sig2, rho):
    return np.array(sig1) ** 2, np.array(sig2) ** 2, np.array(rho) * np.array(sig1) * np.array(sig2)


def cumulants_first_order(t, a, b, mu1, mu2, sig1, sig2, rho):
    """Second- and third-order cumulants of (X1, X2) at horizon t from the first-order rule."""
    c11, c22, c12 = cov_entries(sig1, sig2, rho)
    p = pi2(a, b)
    K = lambda f, h: K2(f, h, a, b)
    return {
        'k11': t * (p @ c11) + 2 * t * K(mu1, mu1),
        'k22': t * (p @ c22) + 2 * t * K(mu2, mu2),
        'k12': t * (p @ c12) + 2 * t * K(mu1, mu2),
        'k111': 6 * t * K(mu1, c11),
        'k222': 6 * t * K(mu2, c22),
        'k112': 2 * t * (2 * K(mu1, c12) + K(mu2, c11)),
        'k122': 2 * t * (2 * K(mu2, c12) + K(mu1, c22)),
    }


def cumulants_exact(t, Q, mu1, mu2, sig1, sig2, rho, r=0.05, N=16):
    """Exact cumulants from the Markov-modulated cumulant generating function
        log E exp(th1 X1 + th2 X2) = log pi . exp(t (Q + diag g(th))) 1,   g = th.mu + th' c th / 2,
    by Cauchy's formula on a polydisc of radius r (two-dimensional FFT of the exact function)."""
    c11, c22, c12 = cov_entries(sig1, sig2, rho)
    mu1, mu2 = np.array(mu1, float), np.array(mu2, float)
    pi = np.linalg.solve(np.vstack([Q.T[:-1], np.ones(len(Q))]), np.r_[np.zeros(len(Q) - 1), 1.0])
    ang = 2 * np.pi * np.arange(N) / N
    F = np.zeros((N, N), complex)
    for i, al in enumerate(ang):
        for j, be in enumerate(ang):
            t1, t2 = r * np.exp(1j * al), r * np.exp(1j * be)
            g = t1 * mu1 + t2 * mu2 + 0.5 * (t1 * t1 * c11 + 2 * t1 * t2 * c12 + t2 * t2 * c22)
            F[i, j] = np.log(pi @ expm(t * (Q + np.diag(g))) @ np.ones(len(Q)))
    C = np.fft.fft2(F) / N / N          # C[m, n] r^{m+n} = Taylor coefficient of th1^m th2^n
    coef = lambda m, n: (C[m, n] / r ** (m + n)).real * math.factorial(m) * math.factorial(n)
    return {'k11': coef(2, 0), 'k22': coef(0, 2), 'k12': coef(1, 1), 'k111': coef(3, 0), 'k222': coef(0, 3),
            'k112': coef(2, 1), 'k122': coef(1, 2)}


def simulate_returns(t, a, b, mu1, mu2, sig1, sig2, rho, n, rng):
    """Exact joint log returns: given the time tau in crisis, (X1, X2) is Gaussian with integrated mean and covariance."""
    tau = simulate_occupation(t, a, b, n, rng)
    c11, c22, c12 = cov_entries(sig1, sig2, rho)
    lin = lambda f: f[0] * (t - tau) + f[1] * tau
    m1, m2, v11, v22, v12 = lin(mu1), lin(mu2), lin(c11), lin(c22), lin(c12)
    z1, z2 = rng.standard_normal(n), rng.standard_normal(n)
    x1 = m1 + np.sqrt(v11) * z1
    x2 = m2 + v12 / np.sqrt(v11) * z1 + np.sqrt(v22 - v12 * v12 / v11) * z2
    return x1, x2
