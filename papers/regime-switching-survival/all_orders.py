"""All-orders fast-switching expansion for survival under a two-regime OU hazard.

rho = d/m solves the Riccati equation  rho' = g (1 - rho^2) - 2 lam rho,  rho(0) = 0,
log m(t) = int_0^t (gbar + g rho),  u_y = exp(-B x) m (1 +/- rho),  + for regime 0.

Outer series  rho_o = sum_{n>=1} eps^n rho_n(E),  E = 1 - exp(-k t), rho_n polynomials in E:
    rho_1 = g/2,   rho_{n+1} = -(1/2) (d/dt rho_n + g sum_{i+j=n} rho_i rho_j).
Inner layer  eta = rho - rho_o, tau = lam t:  d eta/d tau = -2 eta - eps g(eps tau) (2 rho_o(eps tau) eta + eta^2),
    eta(0) = -rho_o(0), solved order by order in the algebra spanned by tau^k exp(-2 j tau).
"""
import math
from collections import defaultdict
import numpy as np
from numpy.polynomial import polynomial as P


def _taylor_of_poly_in_E(p, kappa, K):
    """Taylor coefficients in r (degree <= K) of p(E(r)), E = 1 - exp(-kappa r)."""
    e = np.zeros(K + 1)
    for i in range(1, K + 1):
        e[i] = -(-kappa) ** i / math.factorial(i)
    out = np.zeros(K + 1)
    power = np.zeros(K + 1); power[0] = 1.0
    for c in p:
        out += c * power
        power = np.convolve(power, e)[:K + 1]
    return out


class ExpPoly:
    """sum c[(k, j)] tau^k exp(-2 j tau), j >= 1."""
    def __init__(self, terms=None):
        self.t = defaultdict(float)
        if terms:
            for key, c in terms.items():
                self.t[key] += c

    def __add__(self, o):
        r = ExpPoly(self.t)
        for key, c in o.t.items():
            r.t[key] += c
        return r

    def scale(self, a):
        return ExpPoly({key: a * c for key, c in self.t.items()})

    def times_poly(self, coeffs):  # multiply by sum coeffs[i] tau^i
        r = ExpPoly()
        for (k, j), c in self.t.items():
            for i, a in enumerate(coeffs):
                if a:
                    r.t[(k + i, j)] += c * a
        return r

    def __mul__(self, o):
        r = ExpPoly()
        for (k1, j1), c1 in self.t.items():
            for (k2, j2), c2 in o.t.items():
                r.t[(k1 + k2, j1 + j2)] += c1 * c2
        return r

    def value(self, tau):
        return sum(c * tau ** k * math.exp(-2 * j * tau) for (k, j), c in self.t.items())

    def integral(self, T):  # int_0^T
        tot = 0.0
        for (k, j), c in self.t.items():
            a = 2 * j
            tail = math.exp(-a * T) * sum((a * T) ** i / math.factorial(i) for i in range(k + 1)) if math.isfinite(T) else 0.0
            tot += c * math.factorial(k) / a ** (k + 1) * (1 - tail)
        return tot

    @staticmethod
    def solve(f, y0):
        """y' + 2 y = f, y(0) = y0."""
        y = ExpPoly()
        for (k, j), c in f.t.items():
            if j == 1:
                y.t[(k + 1, 1)] += c / (k + 1)
            else:
                b = 2 * (j - 1)  # p' - b p = tau^k  =>  p = -sum_i k!/(k-i)! tau^(k-i) / b^(i+1)
                for i in range(k + 1):
                    y.t[(k - i, j)] += -c * math.factorial(k) / math.factorial(k - i) / b ** (i + 1)
        y.t[(0, 1)] += y0 - y.value(0.0)
        return y


class FullSeries:
    def __init__(self, kappa, thetas, sigmas, lmbd, order=6):
        k = self.k = kappa
        self.lam, self.eps, self.N = lmbd, 1 / lmbd, order
        thb, tht = (thetas[0] + thetas[1]) / 2, (thetas[0] - thetas[1]) / 2
        ssb, sst = (sigmas[0] ** 2 + sigmas[1] ** 2) / 2, (sigmas[0] ** 2 - sigmas[1] ** 2) / 2
        self.gbar = np.array([0.0, -thb, ssb / (2 * k * k)])
        self.g = np.array([0.0, -tht, sst / (2 * k * k)])
        N = order
        # outer
        rho = [np.zeros(1), self.g / 2]
        for n in range(1, N):
            conv = np.zeros(1)
            for i in range(1, n):
                conv = P.polyadd(conv, P.polymul(rho[i], rho[n - i]))
            rho.append(-0.5 * P.polyadd(self._ddt(rho[n]), P.polymul(self.g, conv)))
        self.rho = rho
        self.log_outer = [None] + [P.polymul(self.g, rho[n]) for n in range(1, N + 1)]
        # inner: Taylor data at r = 0
        K = N + 2
        G = _taylor_of_poly_in_E(self.g, k, K)                       # g(r) = sum G[i] r^i
        R = [None] + [_taylor_of_poly_in_E(rho[n], k, K) for n in range(1, N + 1)]
        eta = [ExpPoly() for _ in range(N + 1)]
        for n in range(2, N + 1):
            f = ExpPoly()
            # -eps g(eps tau) 2 rho_o(eps tau) eta: eps^(1 + a + b + c + m), G[a] tau^a, R[b][c] tau^c, eta_m
            for a in range(1, n):
                for b in range(1, n):
                    for c in range(0, n):
                        m = n - 1 - a - b - c
                        if m >= 2 and G[a] and R[b][c]:
                            f = f + eta[m].times_poly([0] * (a + c) + [-2 * G[a] * R[b][c]])
            # -eps g(eps tau) eta^2: eps^(1 + a + m1 + m2)
            for a in range(1, n):
                for m1 in range(2, n):
                    m2 = n - 1 - a - m1
                    if m2 >= 2 and G[a]:
                        f = f + (eta[m1] * eta[m2]).times_poly([0] * a + [-G[a]])
            eta[n] = ExpPoly.solve(f, -P.polyval(0.0, rho[n]))
        self.eta, self.G = eta, G

    def _ddt(self, p):
        return P.polymul(P.polyder(p) if len(p) > 1 else np.zeros(1), np.array([self.k, -self.k]))

    def _J(self, p, t):
        k, tot = self.k, 0.0
        for n, c in enumerate(p):
            if c:
                tot += c * (t + sum(math.comb(n, j) * (-1) ** j * (1 - math.exp(-j * k * t)) / (j * k) for j in range(1, n + 1)))
        return tot

    def u(self, t, x, s, order=None):
        """Survival probability with all terms through eps^order (outer and initial layer)."""
        N = self.N if order is None else order
        e, E = self.eps, 1 - math.exp(-self.k * t)
        T = t / e
        logm = self._J(self.gbar, t) + sum(e ** n * self._J(self.log_outer[n], t) for n in range(1, N + 1))
        # layer contribution to log m: int_0^t g eta dr = eps int_0^T g(eps tau) eta(tau) dtau
        for a in range(1, N + 1):
            for m in range(2, N + 1):
                if 1 + a + m <= N and self.G[a]:
                    logm += e ** (1 + a + m) * self.G[a] * self.eta[m].times_poly([0] * a + [1]).integral(T)
        rho = sum(e ** n * P.polyval(E, self.rho[n]) for n in range(1, N + 1))
        rho += sum(e ** n * self.eta[n].value(T) for n in range(2, N + 1))
        pm = 1 if s == 0 else -1
        return math.exp(-E / self.k * x + logm) * (1 + pm * rho)


def exact_mp(t, x, s, kappa, thetas, sigmas, lmbd, dps=30):
    """High-precision exact solution of the Riccati system."""
    import mpmath as mp
    mp.mp.dps = dps
    k = mp.mpf(kappa)
    thb, tht = (mp.mpf(thetas[0]) + thetas[1]) / 2, (mp.mpf(thetas[0]) - thetas[1]) / 2
    s0, s1 = mp.mpf(sigmas[0]) ** 2, mp.mpf(sigmas[1]) ** 2
    ssb, sst = (s0 + s1) / 2, (s0 - s1) / 2
    lam = mp.mpf(lmbd)
    def B(r): return (1 - mp.e ** (-k * r)) / k
    def gb(r): b = B(r); return -k * thb * b + ssb * b * b / 2
    def gt(r): b = B(r); return -k * tht * b + sst * b * b / 2
    f = mp.odefun(lambda r, y: [gt(r) * (1 - y[0] ** 2) - 2 * lam * y[0], gb(r) + gt(r) * y[0]], 0, [0, 0])
    rho, logm = f(mp.mpf(t))
    pm = 1 if s == 0 else -1
    return mp.e ** (-B(mp.mpf(t)) * x + logm) * (1 + pm * rho)
