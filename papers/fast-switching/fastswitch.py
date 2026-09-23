"""Fast-switching expansion to all orders for a' = (Q/eps + diag g(t)) a, a(0) = 1, on any finite chain.

Q0 = eps Q is a fixed generator with stationary row vector pi. Write s = pi . a and w = a / s - 1 (so pi . w = 0):

    s' / s = gbar + pi . (g * w),  gbar = pi . g,
    eps w' = Q0 w + eps F(w),      F(w) = g * (1 + w) - (1 + w) (gbar + pi . (g * w)).

Outer series w = sum_{n>=1} eps^n w_n:  w_n = Q0# (w_{n-1}' - F_{n-1}),  Q0# the group inverse, w_0 = 0.
Every g_i is an ExpSum (sum of c exp(-alpha t)), a class closed under products and derivatives, so every
w_n is an ExpSum and every term of log s integrates in closed form.

Initial layer: eta = w - w_outer, tau = t / eps:
    d eta / d tau = Q0 eta + eps [F(w_outer + eta) - F(w_outer)],   eta(0) = -w_outer(0),
solved order by order in the eigenbasis of Q0, in the span of tau^k exp(mu tau).

a_i(t) = s(t) (1 + w_i(t) + eta_i(t / eps)), exact to O(eps^(N+1)) for t > 0.
"""
import math
import cmath
from collections import defaultdict
import numpy as np


def _key(z):
    z = complex(z)
    return (round(z.real, 12), round(z.imag, 12))


# ------------------------------------------------------------------ outer functions of t
class ExpSum:
    """sum_alpha c_alpha exp(-alpha t), alpha >= 0 real."""

    def __init__(self, terms=None):
        self.t = defaultdict(float)
        for a, c in (terms or {}).items():
            self.t[round(float(a), 12)] += c

    @staticmethod
    def const(c):
        return ExpSum({0.0: c})

    def __add__(self, o):
        r = ExpSum(self.t)
        for a, c in o.t.items():
            r.t[a] += c
        return r

    def __sub__(self, o):
        return self + o.scale(-1.0)

    def scale(self, s):
        return ExpSum({a: s * c for a, c in self.t.items()})

    def __mul__(self, o):
        r = ExpSum()
        for a1, c1 in self.t.items():
            for a2, c2 in o.t.items():
                r.t[round(a1 + a2, 12)] += c1 * c2
        return r

    def deriv(self):
        return ExpSum({a: -a * c for a, c in self.t.items() if a})

    def value(self, t):
        return sum(c * math.exp(-a * t) for a, c in self.t.items())

    def integral(self, t):
        return sum(c * t if a == 0 else c * (1 - math.exp(-a * t)) / a for a, c in self.t.items())

    def taylor(self, K):
        """coefficients of t^k, k = 0..K"""
        return [sum(c * (-a) ** k for a, c in self.t.items()) / math.factorial(k) for k in range(K + 1)]


def _vec(n):
    return [ExpSum() for _ in range(n)]


# ------------------------------------------------------------------ inner functions of tau
class ExpPoly:
    """sum c[(k, mu)] tau^k exp(mu tau), mu complex."""

    def __init__(self, terms=None):
        self.t = defaultdict(complex)
        for (k, m), c in (terms or {}).items():
            self.t[(k, _key(m))] += c

    def __add__(self, o):
        r = ExpPoly()
        for d in (self.t, o.t):
            for key, c in d.items():
                r.t[key] += c
        return r

    def scale(self, s):
        r = ExpPoly()
        for key, c in self.t.items():
            r.t[key] = s * c
        return r

    def times_poly(self, coeffs):
        r = ExpPoly()
        for (k, m), c in self.t.items():
            for i, a in enumerate(coeffs):
                if a:
                    r.t[(k + i, m)] += c * a
        return r

    def __mul__(self, o):
        r = ExpPoly()
        for (k1, m1), c1 in self.t.items():
            for (k2, m2), c2 in o.t.items():
                r.t[(k1 + k2, _key(complex(*m1) + complex(*m2)))] += c1 * c2
        return r

    def value(self, tau):
        return sum(c * tau ** k * cmath.exp(complex(*m) * tau) for (k, m), c in self.t.items())

    def integral(self, T):
        tot = 0j
        for (k, m), c in self.t.items():
            mu = complex(*m)
            nm = -mu  # Re(nm) > 0 for decaying terms
            tail = cmath.exp(mu * T) * sum((nm * T) ** i / math.factorial(i) for i in range(k + 1))
            tot += c * math.factorial(k) / nm ** (k + 1) * (1 - tail)
        return tot

    @staticmethod
    def solve(lam, f, y0):
        """y' = lam y + f, y(0) = y0."""
        y = ExpPoly()
        for (k, m), c in f.t.items():
            mu = complex(*m)
            if _key(mu) == _key(lam):
                y.t[(k + 1, _key(lam))] += c / (k + 1)
            else:
                cc = lam - mu  # y = p exp(mu tau), p' - cc p = tau^k
                for i in range(k + 1):
                    y.t[(k - i, _key(mu))] += -c * math.factorial(k) / math.factorial(k - i) / cc ** (i + 1)
        y.t[(0, _key(lam))] += y0 - y.value(0.0)
        return y


def _vpoly(n):
    return [ExpPoly() for _ in range(n)]


# ------------------------------------------------------------------ the engine
class FastSwitch:
    def __init__(self, Q, g, order=6):
        """Q: generator (rows sum to zero). g: list of ExpSum, one per state. eps = 1 / (-trace(Q) / n)."""
        Q = np.asarray(Q, float)
        n = self.n = Q.shape[0]
        self.eps = eps = n / -np.trace(Q)
        Q0 = self.Q0 = eps * Q
        w_, vl = np.linalg.eig(Q0.T)
        pi = np.real(vl[:, np.argmin(abs(w_))])
        self.pi = pi = pi / pi.sum()
        one = np.ones(n)
        self.Qs = Qs = np.linalg.inv(Q0 - np.outer(one, pi)) + np.outer(one, pi)  # group inverse
        self.g, self.N = g, order
        N = order
        zero = self._zero = g[0].scale(0.0)
        self.gbar = gbar = sum((g[i].scale(pi[i]) for i in range(n)), zero)

        def matvec(M, v):
            return [sum((v[j].scale(M[i, j]) for j in range(n)), zero) for i in range(n)]

        def pidot(v):
            return sum((v[i].scale(pi[i]) for i in range(n)), zero)

        # outer series
        w = [[zero for _ in range(n)]]
        F = [[g[i] - gbar for i in range(n)]]
        for m in range(1, N + 1):
            rhs = [w[m - 1][i].deriv() - F[m - 1][i] for i in range(n)]
            w.append(matvec(Qs, rhs))
            gw = pidot([g[i] * w[m][i] for i in range(n)])
            Fm = [g[i] * w[m][i] - w[m][i] * gbar - gw for i in range(n)]
            for i1 in range(1, m):
                pg = pidot([g[k] * w[m - i1][k] for k in range(n)])
                Fm = [Fm[i] - w[i1][i] * pg for i in range(n)]
            F.append(Fm)
        self.w = w
        self.log_terms = [None] + [pidot([g[i] * w[m][i] for i in range(n)]) for m in range(1, N + 1)]

        # inner layer
        lam, V = np.linalg.eig(Q0)
        Vi = np.linalg.inv(V)
        self.lam, self.V, self.Vi = lam, V, Vi
        K = N + 2
        gT = [np.array(g[i].taylor(K)) for i in range(n)]            # g_i(eps tau) = sum_a eps^a tau^a gT[i][a]
        gbT = np.array(gbar.taylor(K))
        WT = [None] + [[np.array(w[b][i].taylor(K)) for i in range(n)] for b in range(1, N + 1)]
        self.gT, self.gbT = gT, gbT

        def poly(a, c):
            return [0] * a + [c]

        eta = [_vpoly(n) for _ in range(N + 1)]
        for m in range(1, N + 1):
            f = _vpoly(n)
            # eps * [ g*eta - eta*gbar - eta*pi(g*w_o) - (1 + w_o) pi(g*eta) - eta pi(g*eta) ] at eps^m
            for a in range(0, m):
                j = m - 1 - a
                if j < 1:
                    continue
                # pi(g*eta_j) at Taylor order a
                pge = sum((eta[j][k].times_poly(poly(a, pi[k] * gT[k][a])) for k in range(n)), ExpPoly())
                for i in range(n):
                    f[i] = f[i] + eta[j][i].times_poly(poly(a, gT[i][a] - gbT[a])) + pge.scale(-1)
            for a in range(0, m):
                for b in range(1, m):
                    for c in range(0, m):
                        j = m - 1 - a - b - c
                        if j < 1:
                            continue
                        pgw = sum(pi[k] * gT[k][a] * WT[b][k][c] for k in range(n))  # coefficient of tau^(a+c)
                        pge = sum((eta[j][k].times_poly(poly(a, pi[k] * gT[k][a])) for k in range(n)), ExpPoly())
                        for i in range(n):
                            f[i] = f[i] + eta[j][i].times_poly(poly(a + c, -pgw))
                            f[i] = f[i] + pge.times_poly(poly(c, -WT[b][i][c]))
            for a in range(0, m):
                for j1 in range(1, m):
                    j2 = m - 1 - a - j1
                    if j2 < 1:
                        continue
                    pge = sum((eta[j2][k].times_poly(poly(a, pi[k] * gT[k][a])) for k in range(n)), ExpPoly())
                    for i in range(n):
                        f[i] = f[i] + (eta[j1][i] * pge).scale(-1)
            # solve d eta_m / d tau = Q0 eta_m + f in the eigenbasis, eta_m(0) = -w_m(0)
            y0 = Vi @ np.array([-w[m][i].value(0.0) for i in range(n)], complex)
            fe = [sum((f[i].scale(Vi[j, i]) for i in range(n)), ExpPoly()) for j in range(n)]
            ye = []
            for j in range(n):
                if abs(lam[j]) < 1e-10:  # stationary direction: pi . eta = 0, so this component vanishes
                    assert abs(y0[j]) < 1e-9, y0[j]
                    ye.append(ExpPoly())
                else:
                    ye.append(ExpPoly.solve(lam[j], fe[j], y0[j]))
            eta[m] = [sum((ye[j].scale(V[i, j]) for j in range(n)), ExpPoly()) for i in range(n)]
        self.eta = eta
        self.is_complex = any(np.iscomplexobj(np.asarray(gt)) and np.any(np.abs(np.imag(gt)) > 0) for gt in gT)

    def a(self, t, order=None):
        """The vector a(t) through eps^order (complex when g is complex)."""
        N = self.N if order is None else order
        n, e, T, pi = self.n, self.eps, t / self.eps, self.pi
        log_s = self.gbar.integral(t) + sum(e ** m * self.log_terms[m].integral(t) for m in range(1, N + 1))
        # layer contribution to log s: eps int_0^T pi(g(eps tau) * eta(tau)) dtau
        for a in range(0, N + 1):
            for j in range(1, N + 1):
                if 1 + a + j <= N:
                    integrand = sum((self.eta[j][k].times_poly([0] * a + [pi[k] * self.gT[k][a]]) for k in range(n)), ExpPoly())
                    log_s = log_s + e ** (1 + a + j) * integrand.integral(T)
        w = np.array([sum(e ** m * self.w[m][i].value(t) for m in range(1, N + 1)) for i in range(n)], complex)
        w += np.array([sum(e ** m * self.eta[m][i].value(T) for m in range(1, N + 1)) for i in range(n)], complex)
        out = cmath.exp(log_s) * (1 + w)
        return out if self.is_complex else out.real

def numerical_a(t, Q, g, dps=30):
    """a(t) for a' = (Q + diag g(t)) a, a(0) = 1, by mpmath's Taylor-series ODE solver."""
    import mpmath as mp
    mp.mp.dps = dps
    Qm = mp.matrix(np.asarray(Q, float).tolist())
    n = Qm.rows
    terms = [[(mp.mpf(a), mp.mpf(c)) for a, c in gi.t.items()] for gi in g]

    def f(r, a):
        out = []
        for i in range(n):
            gi = sum(c * mp.e ** (-al * r) for al, c in terms[i])
            out.append(gi * a[i] + sum(Qm[i, j] * a[j] for j in range(n)))
        return out
    return mp.odefun(f, 0, [1] * n)(mp.mpf(t))


# ------------------------------------------------------------------ smooth functions that are not exponential sums
class Cheb:
    """A smooth function on [0, T] held as a Chebyshev series; same interface as ExpSum."""
    MAXDEG = 120

    def __init__(self, series):
        self.s = series

    @classmethod
    def fit(cls, f, T, deg=80):
        from numpy.polynomial import Chebyshev
        xs = np.cos(np.pi * (np.arange(deg + 1) + 0.5) / (deg + 1)) * T / 2 + T / 2
        ys = np.array([f(x) for x in xs])
        if np.iscomplexobj(ys):
            re = Chebyshev.fit(xs, ys.real, deg, domain=[0, T])
            im = Chebyshev.fit(xs, ys.imag, deg, domain=[0, T])
            return cls(Chebyshev(re.coef + 1j * im.coef, domain=[0, T]))
        return cls(Chebyshev.fit(xs, ys, deg, domain=[0, T]))

    def _wrap(self, s):
        if len(s.coef) > self.MAXDEG + 1:
            s = s.truncate(self.MAXDEG + 1)
        return Cheb(s)

    def __add__(self, o):
        return self._wrap(self.s + o.s)

    def __sub__(self, o):
        return self._wrap(self.s - o.s)

    def scale(self, c):
        return Cheb(self.s * c)

    def __mul__(self, o):
        return self._wrap(self.s * o.s)

    def deriv(self):
        return Cheb(self.s.deriv())

    def value(self, t):
        return self.s(t)

    def integral(self, t):
        return self.s.integ(lbnd=0)(t)

    def taylor(self, K):
        out, d = [], self.s
        for k in range(K + 1):
            out.append(d(0.0) / math.factorial(k))
            d = d.deriv()
        return out


def numerical_a_callable(t, Q, gfuncs, rtol=1e-12):
    """a(t) for a' = (Q + diag g(t)) a with g given as callables (scipy DOP853, complex allowed)."""
    from scipy.integrate import solve_ivp
    Q = np.asarray(Q, float)
    n = Q.shape[0]
    probe = np.array([f(0.3) for f in gfuncs])
    cplx = np.iscomplexobj(probe) and np.any(probe.imag != 0)

    def rhs(r, y):
        a = y[:n] + 1j * y[n:] if cplx else y
        d = np.array([f(r) for f in gfuncs]) * a + Q @ a
        return np.concatenate([d.real, d.imag]) if cplx else d
    y0 = np.concatenate([np.ones(n), np.zeros(n)]) if cplx else np.ones(n)
    sol = solve_ivp(rhs, (0, t), y0, method='DOP853', rtol=rtol, atol=1e-14)
    y = sol.y[:, -1]
    return y[:n] + 1j * y[n:] if cplx else y
