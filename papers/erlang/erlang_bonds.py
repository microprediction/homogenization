"""Vasicek bonds when the mean level switches between two regimes with Erlang(k) durations.

dr = kappa (theta_X - r) dt + sigma dW, regime X in {1, 2}, each regime lasting an Erlang(k) time with means m1, m2
(k exponential stages of mean m_i / k in a row). The stage chain has 2k states and constant hazards, so the
regime-switching reduction applies unchanged: P_j(T, r) = exp(a_j(T) - B(T) r), with a vector ODE on the stages.

First order in the durations:
    log P = log P_Vasicek(theta_bar) + kappa^2 Dtheta^2 K1 I2(T) - kappa B(T) w,
    K1 = m1^2 m2^2 / (k (m1 + m2)^3),         Green-Kubo number of the regime indicator
    I2(T) = int_0^T B(s)^2 ds,
    w  = expected integrated deviation of theta from theta_bar given the current regime and its age.
"""
import numpy as np
from math import factorial
from scipy.integrate import solve_ivp


def B(T, kappa):
    return (1 - np.exp(-kappa * T)) / kappa


def I1(T, kappa):
    return (T - B(T, kappa)) / kappa


def I2(T, kappa):
    return (T - 2 * B(T, kappa) + (1 - np.exp(-2 * kappa * T)) / (2 * kappa)) / kappa ** 2


def log_vasicek(T, r, kappa, theta, sigma):
    return -B(T, kappa) * r - kappa * theta * I1(T, kappa) + 0.5 * sigma ** 2 * I2(T, kappa)


def K1(m1, m2, k):
    return m1 ** 2 * m2 ** 2 / (k * (m1 + m2) ** 3)


def expected_remaining(age, m, k):
    """Mean remaining duration of an Erlang(k, mean m) regime that has lasted `age`."""
    mu = age * k / m
    num = sum((k - n) * mu ** n / factorial(n) for n in range(k))
    den = sum(mu ** n / factorial(n) for n in range(k))
    return m / k * num / den


def w_first(regime, R, m1, m2, k, dtheta):
    """E int_0^inf (theta_X - theta_bar) ds given the regime and its expected remaining duration R."""
    M = m1 + m2; p1, p2 = m1 / M, m2 / M
    c = (m1 - m2) * (k + 1) / (2 * k)
    if regime == 1:
        return dtheta * p2 * (R - p1 * m2 - p1 * c)
    return -dtheta * p1 * (R + p2 * c - p1 * m2)


def log_price_first(T, r, regime, age, kappa, th1, th2, sigma, m1, m2, k):
    tbar = (m1 * th1 + m2 * th2) / (m1 + m2); d = th1 - th2
    R = expected_remaining(age, m1 if regime == 1 else m2, k)
    return (log_vasicek(T, r, kappa, tbar, sigma) + kappa ** 2 * d ** 2 * K1(m1, m2, k) * I2(T, kappa)
            - kappa * B(T, kappa) * w_first(regime, R, m1, m2, k, d))


def stage_chain(m1, m2, k):
    n = 2 * k; Q = np.zeros((n, n))
    for i in range(n):
        rate = k / m1 if i < k else k / m2
        Q[i, i] = -rate; Q[i, (i + 1) % n] = rate
    return Q


def log_price_numerical(T, r, regime, age, kappa, th1, th2, sigma, m1, m2, k):
    """Numerical solution of the stage-chain system, mixed over the stage posterior given the age."""
    Q = stage_chain(m1, m2, k); th = np.r_[np.full(k, th1), np.full(k, th2)]
    def rhs(t, a):
        b = B(t, kappa); g = -kappa * th * b + 0.5 * sigma ** 2 * b ** 2
        return Q @ a + g * a
    sol = solve_ivp(rhs, (0, T), np.ones(2 * k), rtol=1e-11, atol=1e-13)
    a = sol.y[:, -1]
    m = m1 if regime == 1 else m2; mu = age * k / m
    post = np.array([mu ** n / factorial(n) for n in range(k)]); post /= post.sum()
    idx = np.arange(k) + (0 if regime == 1 else k)
    return np.log(post @ a[idx]) - B(T, kappa) * r
