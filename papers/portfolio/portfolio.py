"""Portfolio choice when returns and volatilities are switched by a fast Markov chain.

Notation. The chain has generator Q, stationary law pi and group inverse Q#. For regime functions f, h,
    K(f, h) = -pi . (f~ Q# h~) = int_0^inf Cov_pi(f(y_0), h(y_t)) dt,   f~ = f - pi.f.
For two regimes with exit rates q1 (from regime 1) and q2 (from regime 2),
    K(f, h) = pi1 pi2 (f1 - f2)(h1 - h2) / (q1 + q2),   pi1 = q2/(q1+q2),  pi2 = q1/(q1+q2).

1. Observed regime, CRRA utility U(x) = x^(1-g)/(1-g), wealth dX = X[r_y + w (mu_y - r_y)] dt + X w sigma_y dW.
   V_i(t, x) = x^(1-g)/(1-g) Psi_i(T - t) with Psi' = (Q + diag((1-g) psi)) Psi, Psi(0) = 1,
   psi_i = r_i + lam_i^2/(2g), lam_i = (mu_i - r_i)/sigma_i. Optimal weights w_i = lam_i/(g sigma_i) exactly.
   Certainty-equivalent rate CE = log(pi.Psi(T)) / ((1-g) T) ~ psibar + (1-g) K(psi, psi);
   from a known regime i add -(Q# psi~)_i / T.
2. Mean-variance (Zhou-Yin), constant r: P' = (Q + diag(2r - lam^2)) P, beta_i = e^{-2rT} P_i(T)
   = E_i exp(-int_0^T lam^2), frontier Var x_T = beta/(1-beta) (E x_T - x0 e^{rT})^2.
   beta ~ exp(-T <lam^2> + T K(lam^2, lam^2)) for a stationary start, times exp((Q# lam2~)_i) from regime i.
3. Regime-blind constant fraction w: E X_T^(1-g) = x^(1-g) E exp((1-g) int h_y(w)),
   h_y(w) = r + w (mu_y - r) - g w^2 s_y / 2,  s = sigma^2.
   CE(w) ~ hbar(w) + (1-g)[w^2 K_mm - g w^3 K_ms + g^2 w^4 K_ss / 4],
   w0 = (mubar - r)/(g sbar),  dw = (1-g)[2 w0 K_mm - 3 g w0^2 K_ms + g^2 w0^3 K_ss] / (g sbar).
"""
import os, sys
import numpy as np
from scipy.linalg import expm
from scipy.optimize import minimize_scalar, minimize

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fast-switching'))
from fastswitch import FastSwitch, ExpSum  # noqa: E402


# ------------------------------------------------------------------ chain
def stationary(Q):
    w, vl = np.linalg.eig(np.asarray(Q, float).T)
    p = np.real(vl[:, np.argmin(abs(w))])
    return p / p.sum()


def group_inverse(Q):
    Q = np.asarray(Q, float)
    pi = stationary(Q)
    P = np.outer(np.ones(len(pi)), pi)
    return np.linalg.inv(Q - P) + P


def K(Q, f, h):
    pi, Qs = stationary(Q), group_inverse(Q)
    f, h = np.asarray(f, float), np.asarray(h, float)
    return float(-pi @ ((f - pi @ f) * (Qs @ (h - pi @ h))))


def K2(q1, q2, f, h):
    """two-regime closed form"""
    p1, p2 = q2 / (q1 + q2), q1 / (q1 + q2)
    return p1 * p2 * (f[0] - f[1]) * (h[0] - h[1]) / (q1 + q2)


def two_state(q1, q2):
    return np.array([[-q1, q1], [q2, -q2]], float)


def solve_linear(Q, g, T):
    """a(T) for a' = (Q + diag g) a, a(0) = 1, g constant."""
    return expm(T * (np.asarray(Q, float) + np.diag(g))) @ np.ones(len(g))


def engine_log(Q, g, T, orders):
    """log of pi.a(T) and of a(T) from the all-orders engine, for each truncation order."""
    fs = FastSwitch(Q, [ExpSum.const(float(x)) for x in g], order=max(orders))
    pi = fs.pi
    out = {}
    for N in orders:
        a = np.real(fs.a(T, order=N))
        out[N] = (float(np.log(pi @ a)), np.log(a))
    return out


# ------------------------------------------------------------------ 1. observed regime, CRRA
def sharpe(r, mu, sigma):
    return (np.asarray(mu) - np.asarray(r)) / np.asarray(sigma)


def psi(r, mu, sigma, gamma):
    return np.asarray(r, float) + sharpe(r, mu, sigma) ** 2 / (2 * gamma)


def merton_weights(r, mu, sigma, gamma):
    return sharpe(r, mu, sigma) / (gamma * np.asarray(sigma))


def ce_observed_exact(Q, r, mu, sigma, gamma, T, start=None):
    p = psi(r, mu, sigma, gamma)
    a = solve_linear(Q, (1 - gamma) * p, T)
    v = stationary(Q) @ a if start is None else a[start]
    return float(np.log(v) / ((1 - gamma) * T))


def ce_observed_first(Q, r, mu, sigma, gamma, T, start=None):
    p = psi(r, mu, sigma, gamma)
    pi = stationary(Q)
    ce = pi @ p + (1 - gamma) * K(Q, p, p)
    if start is not None:
        ce += -(group_inverse(Q) @ (p - pi @ p))[start] / T
    return float(ce)


def ce_observed_engine(Q, r, mu, sigma, gamma, T, orders, start=None):
    p = psi(r, mu, sigma, gamma)
    res = engine_log(Q, (1 - gamma) * p, T, orders)
    return {N: (v[0] if start is None else v[1][start]) / ((1 - gamma) * T) for N, v in res.items()}


def ce_regime_fractions(Q, r, mu, sigma, gamma, T, w):
    """CE of holding the fraction w_i in regime i (regime observed)."""
    r, mu, s = np.asarray(r, float), np.asarray(mu, float), np.asarray(sigma, float) ** 2
    h = r + np.asarray(w) * (mu - r) - 0.5 * gamma * np.asarray(w) ** 2 * s
    a = solve_linear(Q, (1 - gamma) * h, T)
    return float(np.log(stationary(Q) @ a) / ((1 - gamma) * T))


# ------------------------------------------------------------------ 2. mean-variance frontier
def beta_exact(Q, lam2, T, start=None):
    a = solve_linear(Q, -np.asarray(lam2, float), T)
    return float(stationary(Q) @ a if start is None else a[start])


def beta_riccati(Q, r, lam2, T, start):
    """Zhou-Yin's P system solved as stated: P' = (rho - 2r) P - Q P backward, P(T) = 1; beta = P(0) H(0)^2."""
    Q = np.asarray(Q, float)
    M = Q + np.diag(2 * r - np.asarray(lam2, float))
    P0 = expm(T * M) @ np.ones(len(lam2))
    H0 = np.exp(-r * T)
    return float(P0[start] * H0 ** 2)


def beta_first(Q, lam2, T, start=None):
    lam2 = np.asarray(lam2, float)
    pi = stationary(Q)
    lb = -T * (pi @ lam2) + T * K(Q, lam2, lam2)
    if start is not None:
        lb += (group_inverse(Q) @ (lam2 - pi @ lam2))[start]
    return float(np.exp(lb))


def frontier_ratio(beta):
    """sd(x_T) / (E x_T - x0 e^{rT}) on the efficient frontier"""
    return float(np.sqrt(beta / (1 - beta)))


def zhou_yin_frontier_var(P0, H0, theta, x0, z):
    """Zhou-Yin frontier as restated in Yin & Zhou (2004), eq. (32), with P0 = P(0,i0), H0 = H(0,i0):
    Var x(T) = c/(1-c) [z - P0 H0 x0 / c]^2 + P0 theta x0^2 / c,  c = P0 H0^2 + theta."""
    c = P0 * H0 ** 2 + theta
    return c / (1 - c) * (z - P0 * H0 * x0 / c) ** 2 + P0 * theta * x0 ** 2 / c


def mc_mean_variance(Q, r, lam2, T, x0, d, start, n, rng):
    """Exact Monte Carlo of the LQ feedback u = -(b/sigma^2)(x - d e^{-r(T-t)}), regime observed.
    Given the regime path, z = x - h is a geometric Brownian motion with drift r - lam^2 and volatility -lam,
    so log z_T is Gaussian with mean log z0 + rT - 1.5 L and variance L, L = int lam^2."""
    L = occupation_integral(Q, lam2, T, start, n, rng)
    z0 = x0 - d * np.exp(-r * T)
    zT = z0 * np.exp(r * T - 1.5 * L + np.sqrt(L) * rng.standard_normal(n))
    xT = d + zT
    return float(xT.mean()), float(xT.var()), float(xT.std() / np.sqrt(n))


def occupation_integral(Q, f, T, start, n, rng):
    """int_0^T f(y_t) dt for n exact chain paths from `start` (or from pi when start is None)."""
    Q = np.asarray(Q, float)
    k = len(f)
    rates = -np.diag(Q)
    jump = np.where(np.eye(k, dtype=bool), 0.0, Q) / rates[:, None]
    y = (rng.choice(k, size=n, p=stationary(Q)) if start is None else np.full(n, start))
    t = np.zeros(n)
    out = np.zeros(n)
    alive = np.ones(n, bool)
    f = np.asarray(f, float)
    cum = np.cumsum(jump, axis=1)
    while alive.any():
        idx = np.where(alive)[0]
        hold = rng.exponential(1 / rates[y[idx]])
        dt = np.minimum(hold, T - t[idx])
        out[idx] += f[y[idx]] * dt
        t[idx] += dt
        done = t[idx] >= T - 1e-15
        alive[idx[done]] = False
        mv = idx[~done]
        u = rng.random(len(mv))
        y[mv] = (u[:, None] > cum[y[mv]]).sum(axis=1)
    return out


# ------------------------------------------------------------------ 3. regime-blind constant fraction
def h_blind(w, r, mu, sigma, gamma):
    return r + w * (np.asarray(mu, float) - r) - 0.5 * gamma * w ** 2 * np.asarray(sigma, float) ** 2


def ce_blind_exact(Q, w, r, mu, sigma, gamma, T):
    a = solve_linear(Q, (1 - gamma) * h_blind(w, r, mu, sigma, gamma), T)
    return float(np.log(stationary(Q) @ a) / ((1 - gamma) * T))


def blind_K(Q, mu, sigma):
    s = np.asarray(sigma, float) ** 2
    return K(Q, mu, mu), 0.5 * (K(Q, mu, s) + K(Q, s, mu)), K(Q, s, s)


def ce_blind_first(Q, w, r, mu, sigma, gamma):
    pi = stationary(Q)
    Kmm, Kms, Kss = blind_K(Q, mu, sigma)
    hbar = pi @ h_blind(w, r, mu, sigma, gamma)
    return float(hbar + (1 - gamma) * (w ** 2 * Kmm - gamma * w ** 3 * Kms + 0.25 * gamma ** 2 * w ** 4 * Kss))


def blind_weights(Q, r, mu, sigma, gamma):
    pi = stationary(Q)
    mb, sb = pi @ np.asarray(mu, float), pi @ np.asarray(sigma, float) ** 2
    Kmm, Kms, Kss = blind_K(Q, mu, sigma)
    w0 = (mb - r) / (gamma * sb)
    dw = (1 - gamma) * (2 * w0 * Kmm - 3 * gamma * w0 ** 2 * Kms + gamma ** 2 * w0 ** 3 * Kss) / (gamma * sb)
    return float(w0), float(dw)


def blind_argmax(fun, lo=-3.0, hi=6.0):
    res = minimize_scalar(lambda w: -fun(w), bounds=(lo, hi), method='bounded', options={'xatol': 1e-12})
    return float(res.x), float(-res.fun)


def ce_blind_engine(Q, w, r, mu, sigma, gamma, T, N):
    g = (1 - gamma) * h_blind(w, r, mu, sigma, gamma)
    return engine_log(Q, g, T, [N])[N][0] / ((1 - gamma) * T)
