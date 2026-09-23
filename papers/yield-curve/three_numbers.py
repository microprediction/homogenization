"""The three numbers a fast regime leaves in a yield curve.

Vasicek short rate with mean level theta_y and variance s_y = sigma_y^2 switched by a fast chain with generator Q,
started from its stationary law pi. Bond price P(T) = exp(-B(T) x) pi . a(T), a' = (Q + diag g) a, a(0) = 1,
g_i = -kappa theta_i B + s_i B^2 / 2. With u = theta - theta_bar, v = s - s_bar and the Green-Kubo form
    K(f, h) = -pi . (f * Q# h)     (symmetric part = int_0^inf Cov_pi(f(y_0), h(y_tau)) dtau, positive semidefinite)
the first-order yield curve is
    log P(T) + B x = -kappa theta_bar int B + (s_bar/2) int B^2 + c2 int B^2 + c3 int B^3 + c4 int B^4 + O(|Q|^-2),
    c2 = kappa^2 K(u,u),  c3 = -kappa Ksym(u,v),  c4 = K(v,v) / 4,
so c2 >= 0, c4 >= 0 and c3^2 <= 4 c2 c4, with equality when the symmetrized 2x2 matrix has rank one.
"""
import math
import os
import sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fast-switching'))
from fastswitch import FastSwitch, ExpSum, numerical_a, numerical_a_callable


def stationary(Q):
    w, vl = np.linalg.eig(np.asarray(Q, float).T)
    p = np.real(vl[:, np.argmin(abs(w))])
    return p / p.sum()


def group_inverse(Q):
    Q = np.asarray(Q, float)
    n, pi = Q.shape[0], stationary(Q)
    P = np.outer(np.ones(n), pi)
    return np.linalg.inv(Q - P) + P


def gk_matrix(Q, thetas, s):
    """Symmetrized Green-Kubo matrix of (u, v)."""
    pi, Qs = stationary(Q), group_inverse(Q)
    u = np.asarray(thetas) - pi @ np.asarray(thetas)
    v = np.asarray(s) - pi @ np.asarray(s)
    K = lambda f, h: -pi @ (f * (Qs @ h))
    kuv = 0.5 * (K(u, v) + K(v, u))
    return np.array([[K(u, u), kuv], [kuv, K(v, v)]])


def coefficients(Q, thetas, s, kappa):
    M = gk_matrix(Q, thetas, s)
    return kappa ** 2 * M[0, 0], -kappa * M[0, 1], M[1, 1] / 4


def int_Bk(k, T, kappa):
    E = lambda n: T + sum(math.comb(n, j) * (-1) ** j * (1 - math.exp(-j * kappa * T)) / (j * kappa) for j in range(1, n + 1))
    return E(k) / kappa ** k


def g_funcs(thetas, s, kappa):
    return [ExpSum({0: -th + v / (2 * kappa ** 2), kappa: th - 2 * v / (2 * kappa ** 2), 2 * kappa: v / (2 * kappa ** 2)})
            for th, v in zip(thetas, s)]


def log_bond_stationary(T, Q, thetas, s, kappa, order=None, dps=None):
    """log(pi . a(T)): the x-free part of the log bond price with a stationary start."""
    pi = stationary(Q)
    g = g_funcs(thetas, s, kappa)
    if dps == 'double':
        a = numerical_a_callable(T, Q, [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in g], rtol=1e-13)
    elif dps:
        a = np.array(numerical_a(T, Q, g, dps=dps), float)
    else:
        a = FastSwitch(Q, g, order=order).a(T, order)
    return math.log(pi @ a)
