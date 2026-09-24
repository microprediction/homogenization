"""Heston with the variance level theta and the squared vol-of-vol xi^2 switched by a fast chain, rho = 0.

Operators on functions of v (after x has been Fourier transformed):
    A_theta = kappa d_v,     A_xi2 = (1/2) v d_vv,     [A_theta, A_xi2] = (kappa/2) d_vv.
The first-order rule adds sum_jk K_jk A_j A_k; its antisymmetric part is K^anti (kappa/2) d_vv, a constant
diffusion in the variance that a reversible driver cannot produce. Everything is computed on one grid in v so the
rule and the numerical solution share discretisation error.
"""
import math, cmath, sys, os
import numpy as np
from scipy.linalg import expm
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'general'))
from effective_generator import stationary, gk


class Grid:
    def __init__(self, vmax=1.0, n=161):
        self.v = np.linspace(0, vmax, n)
        h = self.v[1] - self.v[0]
        D1 = (np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)) / (2 * h)
        D1[0, :3] = np.array([-3, 4, -1]) / (2 * h)                  # one-sided at v = 0
        D1[-1, -3:] = np.array([1, -4, 3]) / (2 * h)
        D2 = (np.diag(np.ones(n - 1), 1) - 2 * np.eye(n) + np.diag(np.ones(n - 1), -1)) / h ** 2
        D2[0, :] = 0                                                 # v d_vv vanishes at v = 0 anyway
        D2[-1, :] = D2[-2, :]
        self.D1, self.D2, self.V = D1, D2, np.diag(self.v)

    def index(self, v0):
        """The node at v0. The grid must contain it: a value is read off at a node, never at the nearest one."""
        i0 = int(np.argmin(abs(self.v - v0)))
        if abs(self.v[i0] - v0) > 1e-12:
            raise ValueError(f"v0 = {v0} is not a grid node; the nearest is {self.v[i0]:.10g}. Choose n so that v0 is a node.")
        return i0


def operators(grid, u, kappa):
    """Averaged-part builder and the two switched operators, for Fourier frequency u (complex)."""
    k = -0.5 * (u * u + 1j * u)                                      # x-part: -(1/2) v (u^2 + i u)
    A_th = kappa * grid.D1
    A_xi = 0.5 * grid.V @ grid.D2
    base = -kappa * grid.V @ grid.D1 + k * grid.V                    # the parts that do not switch
    return base.astype(complex), [A_th.astype(complex), A_xi.astype(complex)]


def first_order(L, D, T, f):
    n = L.shape[0]
    M = np.zeros((2 * n, 2 * n), dtype=complex)
    M[:n, :n], M[n:, n:], M[:n, n:] = L, L, D
    E = expm(T * M)
    return E[:n, :n] @ f + E[:n, n:] @ f


def char_fns(u, T, Q, thetas, xi2s, kappa, grid, v0):
    """phi(u) for the return, stationary start: numerical, averaged, symmetric-only rule, full rule."""
    base, As = operators(grid, u, kappa)
    phis = [np.asarray(thetas, float), np.asarray(xi2s, float)]
    pi = stationary(Q)
    avg = [pi @ p for p in phis]
    Lbar = base + avg[0] * As[0] + avg[1] * As[1]
    n, m = len(pi), len(grid.v)
    f = np.ones(m, complex)
    G = np.kron(Q, np.eye(m)).astype(complex)
    for i in range(n):
        Li = base + phis[0][i] * As[0] + phis[1][i] * As[1]
        G[i * m:(i + 1) * m, i * m:(i + 1) * m] += Li
    full = np.kron(pi, np.eye(m)) @ (expm(T * G) @ np.kron(np.ones(n), f))
    K = gk(Q, phis)
    Ks = 0.5 * (K + K.T)
    corr = lambda KK: sum(KK[j, k] * As[j] @ As[k] for j in range(2) for k in range(2))
    i0 = grid.index(v0)
    return {'numerical': full[i0], 'averaged': (expm(T * Lbar) @ f)[i0],
            'symmetric part only': first_order(Lbar, corr(Ks), T, f)[i0], 'rule': first_order(Lbar, corr(K), T, f)[i0]}


def calls(strikes, T, Q, thetas, xi2s, kappa, grid, v0, S0=100.0, U=None, n=48):
    from numpy.polynomial.legendre import leggauss
    if U is None:
        U = math.sqrt(80 / (stationary(Q) @ np.asarray(thetas) * T))
    x, w = leggauss(n)
    us, ws = 0.5 * U * (x + 1), 0.5 * U * w
    cf = [char_fns(u - 0.5j, T, Q, thetas, xi2s, kappa, grid, v0) for u in us]
    out = {}
    for name in cf[0]:
        row = []
        for K_ in strikes:
            k = math.log(S0 / K_)
            tot = sum(wi * (cmath.exp(1j * u * k) * c[name]).real / (u * u + 0.25) for u, wi, c in zip(us, ws, cf))
            row.append(S0 - math.sqrt(S0 * K_) / math.pi * tot)
        out[name] = row
    return out


def implied_vol(C, S0, K, T):
    N = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))
    lo, hi = 1e-4, 3.0
    for _ in range(200):
        s = 0.5 * (lo + hi)
        d1 = (math.log(S0 / K) + 0.5 * s * s * T) / (s * math.sqrt(T))
        if S0 * N(d1) - K * N(d1 - s * math.sqrt(T)) > C: hi = s
        else: lo = s
    return 0.5 * (lo + hi)
