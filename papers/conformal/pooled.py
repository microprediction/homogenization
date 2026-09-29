"""Conditional coverage of a pooled conformal threshold when the residual depends on a fast hidden two-state regime.

Symmetric chain y_t in {-1, +1}, flipping at rate lam = 1/eps in each direction. Horizon Delta, c = Delta / eps.
Scores:  terminal  S = mu y_Delta + sigma Z;   path average  S = mu A + sigma Z,  A = Delta^-1 int_0^Delta y_t dt;
         slow state S = mu X_Delta + sigma Z,  dX = -kappa (X - y) dt, the predictor's filtered regime.
The pooled threshold q is the (1 - alpha) quantile of the stationary mixture; conditional coverage is P(S <= q | info).
"""
import math
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.stats import norm


def cdf_terminal(q, i, mu, sigma, c):
    """P(mu y_Delta + sigma Z <= q | y_0 = i), i = +-1."""
    p_same = 0.5 * (1 + math.exp(-2 * c))
    return p_same * norm.cdf((q - mu * i) / sigma) + (1 - p_same) * norm.cdf((q + mu * i) / sigma)


def cf_path(us, i, mu, c):
    """E[exp(i u mu A) | y_0 = i] by the 2x2 transform, with Delta = c, eps = 1 (only c matters)."""
    Q = np.array([[-1, 1], [1, -1.0]])
    out = np.empty(len(us), complex)
    for k, u in enumerate(us):
        M = expm(c * Q + 1j * u * mu * np.diag([1.0, -1.0]))     # Delta (Q + i u mu / Delta diag(y))
        out[k] = (M @ np.ones(2))[0 if i > 0 else 1]
    return out


def cdf_path(q, i, mu, sigma, c, n=4000):
    """Gil-Pelaez inversion of the score's characteristic function, trapezoid on [0, U] with the analytic value at 0."""
    U = 40 / sigma
    us = np.linspace(0, U, n + 1)
    phi = cf_path(us[1:], i, mu, c) * np.exp(-0.5 * sigma ** 2 * us[1:] ** 2)
    g = np.empty(n + 1)
    g[0] = mu * mean_path(i, c) - q            # lim_{u -> 0} Im(e^{-iuq} phi(u)) / u
    g[1:] = np.imag(np.exp(-1j * us[1:] * q) * phi) / us[1:]
    return 0.5 - np.trapezoid(g, us) / math.pi


def mean_path(i, c):
    """E[A | y_0 = i] = i (1 - e^{-2c}) / (2c)."""
    return i * (1 - math.exp(-2 * c)) / (2 * c)


def pooled_quantile(cdf, alpha):
    f = lambda q: 0.5 * (cdf(q, +1) + cdf(q, -1)) - (1 - alpha)
    return brentq(f, -20, 20)


def simulate_occupation(n, c, rng):
    """Exact samples of (y_0, A) for the symmetric chain over horizon c with unit rates."""
    y0 = rng.choice([-1.0, 1.0], n)
    y, t, occ = y0.copy(), np.zeros(n), np.zeros(n)
    active = np.ones(n, bool)
    while active.any():
        h = rng.exponential(1.0, n)
        end = np.minimum(t + h, c)
        occ[active] += y[active] * (end - t)[active]
        t = end
        active &= t < c
        y = np.where(active, -y, y)
    return y0, occ / c


def simulate_slow_state(n, x0, lam, kappa, T, rng):
    """X_T given X_0 = x0 (stationary regime posterior P(y_0 = +1 | x0) = (1 + x0) / 2), exact between switches."""
    y = np.where(rng.uniform(size=n) < 0.5 * (1 + x0), 1.0, -1.0)
    x, t = np.full(n, float(x0)), np.zeros(n)
    active = np.ones(n, bool)
    while active.any():
        h = rng.exponential(1 / lam, n)
        end = np.minimum(t + h, T)
        e = np.exp(-kappa * (end - t))
        x = np.where(active, y + (x - y) * e, x)
        t = end
        active &= t < T
        y = np.where(active, -y, y)
    return x


def relax(t, lam, kappa):
    """E[X_t | X_0 = x] / x = (2 lam e^{-kappa t} - kappa e^{-2 lam t}) / (2 lam - kappa)."""
    return (2 * lam * math.exp(-kappa * t) - kappa * math.exp(-2 * lam * t)) / (2 * lam - kappa)
