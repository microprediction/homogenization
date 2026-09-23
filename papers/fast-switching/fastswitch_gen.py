"""Fast-switching expansion to all orders with several time scales:

    a' = ( Q0 / delta^q + sum_p delta^p G_p(t) ) a,   a(0) = one,

Q0 with left null vector pi and right null vector one (pi . one = 1), G_p = Op (sums of time function x matrix),
p + q >= 1. q = 1 is averaging (delta = eps, a fast chain); q = 2 with p in {-1, 0} is the diffusive scaling of a
fast mean-reverting factor with correlation (delta = sqrt(eps)). With s = pi . a, v = a / s = one + w:

    s' / s = sum_p delta^p pi G_p v,
    delta^q w' = Q0 w + sum_p delta^(p+q) F_p(w),   F_p(w) = G_p v - v (pi G_p v).

Outer: w_n = Q0# ( [w']_{n-q} - sum_p [F_p]_{n-p-q} ), w_0 = 0. Inner layer in tau = t / delta^q as before.
"""
import numpy as np
from collections import defaultdict
from fastswitch import ExpPoly


class FastSwitchGen:
    def __init__(self, Q0, delta, q, Gs, pi, one, order=6):
        """Gs: dict p -> Op."""
        Q0 = np.asarray(Q0, float)
        n = self.n = Q0.shape[0]
        self.delta, self.q, self.N, self.Gs = delta, q, order, Gs
        self.pi, self.one = pi, one = np.asarray(pi, float), np.asarray(one, float)
        Pr = np.outer(one, pi)
        Qs = np.linalg.inv(Q0 - Pr) + Pr
        ps = sorted(Gs)
        assert all(p + q >= 1 for p in ps)
        anyop = Gs[ps[0]]
        zero = anyop.zero
        N = order

        def pidot(v):
            acc = zero
            for i in range(n):
                if pi[i]:
                    acc = acc + v[i].scale(pi[i])
            return acc

        def matvec(M, v):
            return [sum((v[j].scale(M[i, j]) for j in np.nonzero(M[i])[0]), zero) for i in range(n)]

        v0 = [_const_like(zero, one[i]) if one[i] else zero for i in range(n)]
        w = [[zero for _ in range(n)]]
        v = [v0]                       # v_k: v_0 = one, v_k = w_k
        Gv = {p: [Gs[p].apply(v0)] for p in ps}           # Gv[p][k] = G_p v_k
        piGv = {p: [pidot(Gv[p][0])] for p in ps}         # pi G_p v_k

        def Fp(p, k):
            """[F_p]_k = G_p v_k - sum_{i+j=k} v_i (pi G_p v_j)"""
            out = list(Gv[p][k])
            for i in range(k + 1):
                j = k - i
                out = [out[r] - v[i][r] * piGv[p][j] for r in range(n)]
            return out

        for m in range(1, N + 1):
            rhs = [zero for _ in range(n)]
            if m - q >= 0:
                rhs = [w[m - q][i].deriv() for i in range(n)]
            for p in ps:
                k = m - p - q
                if 0 <= k < m:
                    f = Fp(p, k)
                    rhs = [rhs[i] - f[i] for i in range(n)]
            wm = matvec(Qs, rhs)
            w.append(wm)
            v.append(wm)
            for p in ps:
                Gv[p].append(Gs[p].apply(wm))
                piGv[p].append(pidot(Gv[p][m]))
        self.w, self.piGv, self.ps = w, piGv, ps

        # inner layer
        K = N + 2
        cT = {p: [np.array(c.taylor(K)) for c, _ in Gs[p].terms] for p in ps}
        Ms = {p: [M for _, M in Gs[p].terms] for p in ps}
        WT = [None] + [np.array([w[b][i].taylor(K) for i in range(n)]) for b in range(1, N + 1)]
        lam, V = np.linalg.eig(Q0)
        Vi = np.linalg.inv(V)

        def Ga(p, a, vec):
            out = [ExpPoly() for _ in range(n)]
            for k, M in enumerate(Ms[p]):
                ca = cT[p][k][a]
                if not ca:
                    continue
                for i in range(n):
                    for j in np.nonzero(M[i])[0]:
                        out[i] = out[i] + vec[j].times_poly([0] * a + [ca * M[i, j]])
            return out

        def pi_ep(vec):
            acc = ExpPoly()
            for i in range(n):
                if pi[i]:
                    acc = acc + vec[i].scale(pi[i])
            return acc

        def voT(b, c):
            """coefficient vector of delta^b (b >= 0) and tau^c in v_outer(delta^q tau), excluding time scaling"""
            if b == 0:
                return one if c == 0 else np.zeros(n)
            return WT[b][:, c]

        # pi G_p v_o at (Taylor a of G, order b of v, Taylor c of v): scalar
        def pGv(p, a, b, c):
            return sum(cT[p][k][a] * float(pi @ (Ms[p][k] @ voT(b, c))) for k in range(len(Ms[p])))

        eta = [[ExpPoly() for _ in range(n)] for _ in range(N + 1)]
        for m in range(1, N + 1):
            f = [ExpPoly() for _ in range(n)]
            for p in ps:
                # delta^(p+q) [ G_p eta - eta pi G_p v_o - v_o pi G_p eta - eta pi G_p eta ], Taylor a: delta^(q a)
                for a in range(0, N + 1):
                    base = p + q + q * a
                    if base >= m:
                        break
                    j = m - base  # G_p eta_j and v_o pi G eta_j with b = 0
                    if 1 <= j:
                        Ge = Ga(p, a, eta[j])
                        pGe = pi_ep(Ge)
                        for i in range(n):
                            f[i] = f[i] + Ge[i] + pGe.scale(-one[i])
                    for b in range(0, m):
                        for c in range(0, m):
                            j = m - base - b - q * c
                            if j < 1:
                                continue
                            s_ = pGv(p, a, b, c)
                            if s_:
                                for i in range(n):
                                    f[i] = f[i] + eta[j][i].times_poly([0] * (a + c) + [-s_])
                            if b >= 1:
                                pGe = pi_ep(Ga(p, a, eta[j]))
                                for i in range(n):
                                    if WT[b][i, c]:
                                        f[i] = f[i] + pGe.times_poly([0] * c + [-WT[b][i, c]])
                    for j1 in range(1, m):
                        j2 = m - base - j1
                        if j2 < 1:
                            continue
                        pGe = pi_ep(Ga(p, a, eta[j2]))
                        for i in range(n):
                            f[i] = f[i] + (eta[j1][i] * pGe).scale(-1)
            y0 = Vi @ np.array([-w[m][i].value(0.0) for i in range(n)], complex)
            fe = [sum((f[i].scale(Vi[jj, i]) for i in range(n) if Vi[jj, i]), ExpPoly()) for jj in range(n)]
            ye = [ExpPoly() if abs(lam[jj]) < 1e-10 else ExpPoly.solve(lam[jj], fe[jj], y0[jj]) for jj in range(n)]
            eta[m] = [sum((ye[jj].scale(V[i, jj]) for jj in range(n) if V[i, jj]), ExpPoly()) for i in range(n)]
        self.eta, self._Ga, self._pi_ep = eta, Ga, pi_ep

    def a(self, t, order=None):
        N = self.N if order is None else order
        d, q, T, n = self.delta, self.q, t / self.delta ** self.q, self.n
        log_s = 0.0
        for p in self.ps:
            for k in range(0, N - p + 1):
                if k < len(self.piGv[p]):
                    log_s = log_s + d ** (p + k) * self.piGv[p][k].integral(t)
        # layer: int_0^t sum_p delta^p pi G_p eta = delta^q int_0^T sum_p delta^p pi G_p(delta^q tau) eta(tau)
        for p in self.ps:
            for a in range(0, N + 1):
                for j in range(1, N + 1):
                    o = q + p + q * a + j
                    if o <= N:
                        log_s = log_s + d ** o * self._pi_ep(self._Ga(p, a, self.eta[j])).integral(T)
        w = np.array([sum(d ** m * self.w[m][i].value(t) for m in range(1, N + 1)) for i in range(n)], complex)
        w += np.array([sum(d ** m * self.eta[m][i].value(T) for m in range(1, N + 1)) for i in range(n)], complex)
        out = np.exp(log_s) * (self.one + w)
        return out.real if not np.iscomplexobj(log_s) or abs(np.imag(log_s)) == 0 else out


def _const_like(zero, c):
    """A constant time function of the same class as zero."""
    from fastswitch import ExpSum, Cheb
    if isinstance(zero, ExpSum):
        return ExpSum({0.0: c})
    s = zero.s.copy()
    s.coef = np.zeros(1, dtype=s.coef.dtype) + c
    return Cheb(s)
