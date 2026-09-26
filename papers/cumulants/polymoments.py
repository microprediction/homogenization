"""Exact moments of integrated variance, and of the log return, under a fast regime, from polynomial moment systems.

Two models. (a) A switched Black-Scholes variance s(Y_t): V_T = int s(Y) dt, M_T = int sqrt(s(Y)) dW.
(b) A CIR variance dv = kappa_y (theta_y - v) dt + xi sqrt(v) dB with switched coefficients (product coefficients
c_y = kappa_y theta_y and kappa_y), V_T = int v dt, M_T = int sqrt(v) dW, d<W, B> = rho dt.
In both, the generator on (regime) x (polynomials in v, z = int v, M of total degree <= deg) is a finite matrix, and
E[f(Y_T, v_T, z_T, M_T)] = [exp(T G) f](y_0, v_0, 0, 0) exactly. The log return is X = M - z/2.
"""
import itertools
import numpy as np
from scipy.linalg import expm


def stationary(Q):
    w, vl = np.linalg.eig(np.asarray(Q, float).T)
    p = np.real(vl[:, np.argmin(abs(w))])
    return p / p.sum()


def group_inverse(Q):
    Q = np.asarray(Q, float)
    pi = stationary(Q)
    P = np.outer(np.ones(len(pi)), pi)
    return np.linalg.inv(Q - P) + P


def gk(Q, f, h):
    """K(f, h) = -pi . (f~ Q# h~)."""
    Q = np.asarray(Q, float)
    pi, Qs = stationary(Q), group_inverse(Q)
    ft, ht = np.asarray(f, float) - pi @ f, np.asarray(h, float) - pi @ h
    return -pi @ (ft * (Qs @ ht))


class Poly:
    """Monomials v^a z^b M^c with a + b + c <= deg, and the operators of the two models as matrices on them."""

    def __init__(self, deg):
        self.deg = deg
        self.mons = [(a, b, c) for a in range(deg + 1) for b in range(deg + 1) for c in range(deg + 1) if a + b + c <= deg]
        self.idx = {m: i for i, m in enumerate(self.mons)}
        self.n = len(self.mons)

    def op(self, rule):
        """Matrix of a first-order-in-degree operator given as rule(a, b, c) -> list of (coef, (a', b', c'))."""
        A = np.zeros((self.n, self.n))
        for j, (a, b, c) in enumerate(self.mons):
            for coef, m in rule(a, b, c):
                if coef != 0 and sum(m) <= self.deg:
                    A[self.idx[m], j] += coef
        return A

    # ---- operators (act on the function of (v, z, M), returned as matrices)
    def d_v(self):            # d/dv
        return self.op(lambda a, b, c: [(a, (a - 1, b, c))] if a else [])

    def v_d_v(self):          # v d/dv
        return self.op(lambda a, b, c: [(a, (a, b, c))])

    def v_d_vv(self):         # v d^2/dv^2
        return self.op(lambda a, b, c: [(a * (a - 1), (a - 1, b, c))] if a > 1 else [])

    def v_d_z(self):          # v d/dz   (z' = v)
        return self.op(lambda a, b, c: [(b, (a + 1, b - 1, c))] if b else [])

    def d_z(self):            # d/dz    (z' = s, a constant per regime)
        return self.op(lambda a, b, c: [(b, (a, b - 1, c))] if b else [])

    def v_d_MM(self):         # v d^2/dM^2
        return self.op(lambda a, b, c: [(c * (c - 1), (a + 1, b, c - 2))] if c > 1 else [])

    def d_MM(self):           # d^2/dM^2
        return self.op(lambda a, b, c: [(c * (c - 1), (a, b, c - 2))] if c > 1 else [])

    def v_d_Mv(self):         # v d^2/dM dv
        return self.op(lambda a, b, c: [(a * c, (a, b, c - 1))] if a and c else [])

    def vec(self, coeffs):
        """Coefficient vector of a polynomial given as {(a, b, c): coef}."""
        f = np.zeros(self.n)
        for m, x in coeffs.items():
            f[self.idx[m]] += x
        return f

    def monomial(self, a=0, b=0, c=0):
        return self.vec({(a, b, c): 1.0})

    def logreturn_power(self, k):
        """(M - z/2)^k."""
        from math import comb
        return self.vec({(0, k - j, j): comb(k, j) * (-0.5) ** (k - j) for j in range(k + 1)})

    def evaluate(self, f, v0):
        """f(v0, 0, 0)."""
        return sum(f[self.idx[(a, 0, 0)]] * v0 ** a for a in range(self.deg + 1))


def cir_generator(P, kappa, theta, xi, rho):
    """L = kappa (theta - v) d_v + xi^2/2 v d_vv + v d_z + v/2 d_MM + rho xi v d_Mv on the monomial space."""
    return (kappa * theta * P.d_v() - kappa * P.v_d_v() + 0.5 * xi ** 2 * P.v_d_vv() + P.v_d_z()
            + 0.5 * P.v_d_MM() + rho * xi * P.v_d_Mv())


def bs_generator(P, s):
    """L = s d_z + s/2 d_MM: a constant variance s."""
    return s * P.d_z() + 0.5 * s * P.d_MM()


def switched_expectation(Q, Ls, T, f, P, v0, nu=None):
    """E_nu[f(Y_T, v_T, z_T, M_T)] with regime generator Q and per-regime generators Ls (list of matrices)."""
    Q = np.asarray(Q, float)
    n = len(Q)
    G = np.kron(Q, np.eye(P.n))
    for i, L in enumerate(Ls):
        G[i * P.n:(i + 1) * P.n, i * P.n:(i + 1) * P.n] += L
    F = expm(T * G) @ np.tile(f, n)
    vals = np.array([P.evaluate(F[i * P.n:(i + 1) * P.n], v0) for i in range(n)])
    nu = stationary(Q) if nu is None else np.asarray(nu, float)
    return nu @ vals


def first_order_expectation(Lbar, D, T, f, P, v0):
    """[exp(T Lbar) f + int_0^T exp((T-s) Lbar) D exp(s Lbar) f ds](v0, 0, 0), by one block exponential."""
    n = Lbar.shape[0]
    B = np.zeros((2 * n, 2 * n))
    B[:n, :n], B[n:, n:], B[:n, n:] = Lbar, Lbar, D
    E = expm(T * B)
    return P.evaluate(E[:n, :n] @ f, v0), P.evaluate(E[:n, n:] @ f, v0)


def cumulants(m):
    """kappa_1..kappa_4 from raw moments m[0..3] = E[X], E[X^2], E[X^3], E[X^4]."""
    m1, m2, m3, m4 = m
    return np.array([m1, m2 - m1 ** 2, m3 - 3 * m1 * m2 + 2 * m1 ** 3,
                     m4 - 4 * m1 * m3 - 3 * m2 ** 2 + 12 * m1 ** 2 * m2 - 6 * m1 ** 4])


def cumulant_derivative(a, b):
    """d/d(eps) of the cumulants of moments a + eps b at eps = 0."""
    a1, a2, a3, a4 = a
    b1, b2, b3, b4 = b
    return np.array([b1, b2 - 2 * a1 * b1, b3 - 3 * (a1 * b2 + a2 * b1) + 6 * a1 ** 2 * b1,
                     b4 - 4 * (a1 * b3 + a3 * b1) - 6 * a2 * b2 + 12 * (a1 ** 2 * b2 + 2 * a1 * a2 * b1) - 24 * a1 ** 3 * b1])


def logreturn_from_V(kV):
    """Cumulants of X = M - V/2 given those of V, when M | V ~ N(0, V): kappa_2, kappa_3, kappa_4 of X."""
    k1, k2, k3, k4 = kV
    return np.array([k1 + k2 / 4, -1.5 * k2 - k3 / 8, 3 * k2 + 1.5 * k3 + k4 / 16])
