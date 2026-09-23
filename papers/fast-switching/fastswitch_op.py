"""Fast-switching expansion to all orders with a general coupling operator:

    a' = (Q0 / eps + G(t)) a,   a(0) = one,   G(t) = sum_k c_k(t) M_k,

where Q0 has a left null vector pi and a right null vector one with pi . one = 1 (a Markov generator with
one = (1, ..., 1), or the Hermite form of an Ornstein-Uhlenbeck generator with one = e_0), the c_k are
ExpSum or Cheb time functions and the M_k are constant matrices. With s = pi . a and w = a / s - one:

    s' / s = pi G (one + w),
    eps w' = Q0 w + eps F(w),   F(w) = G (one + w) - (one + w) pi G (one + w).

The outer series and the initial layer are as in fastswitch.py, with G in place of diag(g).
"""
import math
import numpy as np
from fastswitch import ExpPoly, ExpSum


class Op:
    """G(t) = sum_k c_k(t) M_k acting on vectors of time functions."""

    def __init__(self, terms):
        self.terms = [(c, np.asarray(M)) for c, M in terms]
        self.n = self.terms[0][1].shape[0]
        self.zero = self.terms[0][0].scale(0.0)

    def apply(self, v):
        out = [self.zero for _ in range(self.n)]
        for c, M in self.terms:
            for i in range(self.n):
                row = M[i]
                acc = None
                for j in np.nonzero(row)[0]:
                    x = v[j].scale(row[j])
                    acc = x if acc is None else acc + x
                if acc is not None:
                    out[i] = out[i] + c * acc
        return out

    def apply_const(self, u):
        """G applied to a constant vector u"""
        out = [self.zero for _ in range(self.n)]
        for c, M in self.terms:
            mu = M @ u
            for i in range(self.n):
                if mu[i]:
                    out[i] = out[i] + c.scale(mu[i])
        return out


class FastSwitchOp:
    def __init__(self, Q0, eps, G, pi, one, order=6):
        Q0 = np.asarray(Q0, float)
        n = self.n = Q0.shape[0]
        self.eps, self.Q0, self.N = eps, Q0, order
        self.pi, self.one = pi, one = np.asarray(pi, float), np.asarray(one, float)
        P = np.outer(one, pi)
        self.Qs = Qs = np.linalg.inv(Q0 - P) + P  # group inverse: Q0 Qs = I - one pi, pi Qs = 0
        self.G = G
        zero = G.zero
        N = order

        def pidot(v):
            acc = zero
            for i in range(n):
                if pi[i]:
                    acc = acc + v[i].scale(pi[i])
            return acc

        def matvec(M, v):
            out = []
            for i in range(n):
                acc = zero
                for j in np.nonzero(M[i])[0]:
                    acc = acc + v[j].scale(M[i, j])
                out.append(acc)
            return out

        def outer_const(u, f):  # constant vector u times scalar function f
            return [f.scale(u[i]) if u[i] else zero for i in range(n)]

        G1 = G.apply_const(one)
        self.gbar = gbar = pidot(G1)
        w = [[zero for _ in range(n)]]
        F = [[G1[i] - gbar.scale(one[i]) for i in range(n)]]
        for m in range(1, N + 1):
            rhs = [w[m - 1][i].deriv() - F[m - 1][i] for i in range(n)]
            w.append(matvec(Qs, rhs))
            Gw = G.apply(w[m])
            piGw = pidot(Gw)
            Fm = [Gw[i] - w[m][i] * gbar - piGw.scale(one[i]) for i in range(n)]
            for i1 in range(1, m):
                pg = pidot(G.apply(w[m - i1]))
                Fm = [Fm[i] - w[i1][i] * pg for i in range(n)]
            F.append(Fm)
        self.w = w
        self.log_terms = [None] + [pidot(G.apply(w[m])) for m in range(1, N + 1)]

        # inner layer: Taylor data at t = 0
        K = N + 2
        cT = [np.array(c.taylor(K)) for c, _ in G.terms]
        Ms = [M for _, M in G.terms]
        WT = [None] + [np.array([w[b][i].taylor(K) for i in range(n)]) for b in range(1, N + 1)]  # n x (K+1)
        GoneT = [sum(cT[k][a] * (Ms[k] @ one) for k in range(len(Ms))) for a in range(K + 1)]  # vectors
        lam, V = np.linalg.eig(Q0)
        Vi = np.linalg.inv(V)
        self.lam, self.cT, self.Ms = lam, cT, Ms

        def Ga_apply(a, eta_vec):
            """(Taylor order a of G) applied to a vector of ExpPoly"""
            out = [ExpPoly() for _ in range(n)]
            for k, M in enumerate(Ms):
                ca = cT[k][a]
                if not ca:
                    continue
                for i in range(n):
                    for j in np.nonzero(M[i])[0]:
                        out[i] = out[i] + eta_vec[j].times_poly([0] * a + [ca * M[i, j]])
            return out

        def pi_ep(v):
            acc = ExpPoly()
            for i in range(n):
                if pi[i]:
                    acc = acc + v[i].scale(pi[i])
            return acc

        eta = [[ExpPoly() for _ in range(n)] for _ in range(N + 1)]
        for m in range(1, N + 1):
            f = [ExpPoly() for _ in range(n)]
            for a in range(0, m):
                j = m - 1 - a
                if j < 1:
                    continue
                Ge = Ga_apply(a, eta[j])                        # G eta
                pGe = pi_ep(Ge)
                pGone = float(pi @ GoneT[a])                    # pi G one at Taylor order a
                for i in range(n):
                    f[i] = f[i] + Ge[i] + eta[j][i].times_poly([0] * a + [-pGone]) + pGe.scale(-one[i])
            for a in range(0, m):
                for b in range(1, m):
                    for c in range(0, m):
                        j = m - 1 - a - b - c
                        if j < 1:
                            continue
                        # pi G w_o at orders (a, b, c): coefficient of tau^(a+c)
                        pGw = sum(cT[k][a] * float(pi @ (Ms[k] @ WT[b][:, c])) for k in range(len(Ms)))
                        pGe = pi_ep(Ga_apply(a, eta[j]))
                        for i in range(n):
                            f[i] = f[i] + eta[j][i].times_poly([0] * (a + c) + [-pGw])
                            f[i] = f[i] + pGe.times_poly([0] * c + [-WT[b][i, c]])
            for a in range(0, m):
                for j1 in range(1, m):
                    j2 = m - 1 - a - j1
                    if j2 < 1:
                        continue
                    pGe = pi_ep(Ga_apply(a, eta[j2]))
                    for i in range(n):
                        f[i] = f[i] + (eta[j1][i] * pGe).scale(-1)
            y0 = Vi @ np.array([-w[m][i].value(0.0) for i in range(n)], complex)
            fe = [sum((f[i].scale(Vi[j, i]) for i in range(n) if Vi[j, i]), ExpPoly()) for j in range(n)]
            ye = []
            for j in range(n):
                if abs(lam[j]) < 1e-10:
                    ye.append(ExpPoly())
                else:
                    ye.append(ExpPoly.solve(lam[j], fe[j], y0[j]))
            eta[m] = [sum((ye[j].scale(V[i, j]) for j in range(n) if V[i, j]), ExpPoly()) for i in range(n)]
        self.eta = eta
        self._Ga_apply, self._pi_ep = Ga_apply, pi_ep

    def a(self, t, order=None):
        """The vector a(t) (coefficients in the basis of Q0) through eps^order."""
        N = self.N if order is None else order
        n, e, T = self.n, self.eps, t / self.eps
        log_s = self.gbar.integral(t) + sum(e ** m * self.log_terms[m].integral(t) for m in range(1, N + 1))
        for a in range(0, N + 1):
            for j in range(1, N + 1):
                if 1 + a + j <= N:
                    log_s = log_s + e ** (1 + a + j) * self._pi_ep(self._Ga_apply(a, self.eta[j])).integral(T)
        w = np.array([sum(e ** m * self.w[m][i].value(t) for m in range(1, N + 1)) for i in range(n)], complex)
        w += np.array([sum(e ** m * self.eta[m][i].value(T) for m in range(1, N + 1)) for i in range(n)], complex)
        out = np.exp(log_s) * (self.one + w)
        return out.real if np.all(np.abs(out.imag) < 1e-300) or not np.iscomplexobj(log_s) else out


# ------------------------------------------------------------------ Hermite form of a fast OU factor
def hermite_ops(M):
    """Probabilists' Hermite basis He_0..He_{M-1}: generator of dY = -Y dt + sqrt(2) dZ is diag(0,-1,...);
    multiplication by y is Y with y He_n = He_{n+1} + n He_{n-1}; E[.] picks the He_0 coefficient."""
    L = -np.diag(np.arange(M, dtype=float))
    Y = np.zeros((M, M))
    for k in range(M):
        if k + 1 < M:
            Y[k + 1, k] = 1.0      # He_k -> He_{k+1}
        if k >= 1:
            Y[k - 1, k] = float(k)  # He_k -> k He_{k-1}
    pi = np.zeros(M); pi[0] = 1.0
    one = np.zeros(M); one[0] = 1.0
    return L, Y, pi, one


def hermite_eval(coef, y):
    """sum_n coef[n] He_n(y)"""
    h0, h1, tot = 1.0, y, coef[0]
    if len(coef) > 1:
        tot += coef[1] * y
    for n in range(2, len(coef)):
        h0, h1 = h1, y * h1 - (n - 1) * h0
        tot += coef[n] * h1
    return tot
