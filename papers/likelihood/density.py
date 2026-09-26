"""Transition density of a discretely observed Vasicek process whose level and variance switch with a fast chain.

x' given x after time D, regime drawn from pi at the start and summed at the end (the regimes at successive
observations are independent up to terms of order exp(-2 lam D)). With c the Laplace variable, B(t) = c e^{-kappa t},
g_i = -kappa theta_i B + s_i B^2 / 2 and gt = -kappa u B + v_ B^2 / 2 (u, v_ the level and variance fluctuations),
    E[exp(-c x') | x] = exp(-c m + c^2 v / 2) * exp( int_0^D K(gt, gt) dt ) + O(|Q|^-2)
                      = exp(-c m + c^2 v / 2) * exp(A2 c^2 + A3 c^3 + A4 c^4) + O(|Q|^-2),
    A2 = kappa^2 K_uu (1 - e^{-2 kappa D}) / (2 kappa),  A3 = -kappa K_uv^sym (1 - e^{-3 kappa D}) / (3 kappa),
    A4 = K_vv / 4 (1 - e^{-4 kappa D}) / (4 kappa).
A factor c^n acts as the n-th derivative of the N(m, v) density, f^{(n)} = (-1)^n v^{-n/2} He_n(z) f, so
    p(x' | x) = f(x') (1 + A2 He_2(z)/v - A3 He_3(z)/v^{3/2} + A4 He_4(z)/v^2) + O(|Q|^-2).
"""
import math
import numpy as np
from numpy.polynomial.hermite_e import hermeval


def stationary(Q):
    w, vl = np.linalg.eig(np.asarray(Q, float).T)
    p = np.real(vl[:, np.argmin(abs(w))])
    return p / p.sum()


def gk(Q, f, h):
    Q = np.asarray(Q, float)
    pi = stationary(Q)
    one = np.ones(len(pi))
    Qs = np.linalg.inv(Q - np.outer(one, pi)) + np.outer(one, pi)
    ft, ht = f - pi @ f, h - pi @ h
    return -pi @ (ft * (Qs @ ht))


def coefficients(Q, kappa, thetas, s, D):
    th, s = np.asarray(thetas, float), np.asarray(s, float)
    Kuu, Kvv = gk(Q, th, th), gk(Q, s, s)
    Kuv = 0.5 * (gk(Q, th, s) + gk(Q, s, th))
    A2 = kappa ** 2 * Kuu * (1 - math.exp(-2 * kappa * D)) / (2 * kappa)
    A3 = -kappa * Kuv * (1 - math.exp(-3 * kappa * D)) / (3 * kappa)
    A4 = Kvv / 4 * (1 - math.exp(-4 * kappa * D)) / (4 * kappa)
    return A2, A3, A4


def averaged(Q, kappa, thetas, s, x, D):
    pi = stationary(Q)
    m = x * math.exp(-kappa * D) + (pi @ thetas) * (1 - math.exp(-kappa * D))
    v = (pi @ s) * (1 - math.exp(-2 * kappa * D)) / (2 * kappa)
    return m, v


def density_first_order(xp, Q, kappa, thetas, s, x, D):
    m, v = averaged(Q, kappa, thetas, s, x, D)
    A2, A3, A4 = coefficients(Q, kappa, thetas, s, D)
    z = (np.asarray(xp) - m) / math.sqrt(v)
    f = np.exp(-z * z / 2) / math.sqrt(2 * math.pi * v)
    # c^n on e^{-c x'} transform <-> f^{(n)} = (-1)^n v^{-n/2} He_n(z) f
    corr = A2 * hermeval(z, [0, 0, 1]) / v - A3 * hermeval(z, [0, 0, 0, 1]) / v ** 1.5 + A4 * hermeval(z, [0, 0, 0, 0, 1]) / v ** 2
    return f * (1 + corr), f


def density_exact(xp, Q, kappa, thetas, s, x, D, U=None, n=400):
    """Fourier inversion of E[e^{iu x'}] = e^{iu e^{-kD} x} pi . a(D), a' = (Q + diag g) a, a(0) = 1,
    g_i(t) = i u kappa theta_i e^{-kappa t} - u^2 s_i e^{-2 kappa t} / 2 (B = -iu e^{-kappa t})."""
    from scipy.linalg import expm
    Q = np.asarray(Q, float)
    pi = stationary(Q)
    m, v = averaged(Q, kappa, thetas, s, x, D)
    if U is None:
        U = 12 / math.sqrt(v)
    us = np.linspace(-U, U, n + 1)
    phis = []
    steps = 200
    for u in us:
        a = np.ones(len(pi), complex)
        dt = D / steps
        # integrate backwards in time-to-go: a'(t) = (Q + diag g(t)) a, t from 0 to D
        for k in range(steps):
            t = (k + 0.5) * dt
            g = 1j * u * kappa * np.asarray(thetas) * math.exp(-kappa * t) - 0.5 * u * u * np.asarray(s) * math.exp(-2 * kappa * t)
            a = expm((Q + np.diag(g)) * dt) @ a
        phis.append(np.exp(1j * u * math.exp(-kappa * D) * x) * (pi @ a))
    phis = np.array(phis)
    du = us[1] - us[0]
    xp = np.atleast_1d(xp)
    return np.array([np.real(np.sum(phis * np.exp(-1j * us * y)) * du / (2 * math.pi)) for y in xp])


def kernel_transform(us, Q, kappa, thetas, s, D, steps=800):
    """A(u) with A_ij(u) = E[exp(i u (x' - e^{-kappa D} x)) 1{y' = j} | x, y = i]: the regime-resolved transform.

    A solves A' = (Q + diag g(t)) A, A(0) = I, integrated by RK4 for all u at once."""
    Q = np.asarray(Q, float)
    n = len(Q)
    us = np.asarray(us, float)
    A = np.broadcast_to(np.eye(n, dtype=complex), (len(us), n, n)).copy()
    dt = D / steps

    def M(t):
        g = 1j * us[:, None] * kappa * np.asarray(thetas) * math.exp(-kappa * t) \
            - 0.5 * us[:, None] ** 2 * np.asarray(s) * math.exp(-2 * kappa * t)
        out = np.broadcast_to(Q, (len(us), n, n)).astype(complex)
        out[:, np.arange(n), np.arange(n)] += g
        return out

    for k in range(steps):
        t = k * dt
        M0, M1, M2 = M(t), M(t + dt / 2), M(t + dt)
        k1 = M0 @ A
        k2 = M1 @ (A + dt / 2 * k1)
        k3 = M1 @ (A + dt / 2 * k2)
        k4 = M2 @ (A + dt * k3)
        A = A + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    return A


def kernel_exact(x, xp, Q, kappa, thetas, s, D, n=600, steps=800):
    """The matrix transition density K_ij(x, x') = density of (x' , y' = j) given (x, y = i), by Fourier inversion."""
    m, v = averaged(Q, kappa, thetas, s, x, D)
    U = 12 / math.sqrt(v)
    us = np.linspace(-U, U, n + 1)
    A = kernel_transform(us, Q, kappa, thetas, s, D, steps)
    du = us[1] - us[0]
    ph = np.exp(-1j * us * (xp - math.exp(-kappa * D) * x))
    return np.real(np.einsum('u,uij->ij', ph, A)) * du / (2 * math.pi)
