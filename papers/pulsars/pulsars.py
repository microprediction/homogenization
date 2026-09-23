"""Pulsar spin-down switching: closed forms for the two-state model.

The spin-down rate nudot(y_t) switches between nudot_1 and nudot_2 with a hidden Markov chain y_t that leaves
state 1 at rate a and state 2 at rate b, started from its stationary law. With the averaged spin-down removed,
    X(T)   = nu(T) - nu(0) - nubar_dot T        = int_0^T g(y_s) ds,          g = nudot - nubar_dot,
    Phi(T) = phi(T) - phi(0) - nu(0)T - nubar_dot T^2/2 = int_0^T (T - s) g(y_s) ds.
Any consistent units; the certificate uses days, nHz and nHz per day.
"""
import math
import numpy as np

DAY = 86400.0


class TwoState:
    def __init__(self, nudot1, nudot2, a, b):
        self.n1, self.n2, self.a, self.b = float(nudot1), float(nudot2), float(a), float(b)
        self.gamma = self.a + self.b
        self.p1, self.p2 = self.b / self.gamma, self.a / self.gamma
        self.Delta = self.n1 - self.n2
        self.Sigma = (self.p2 - self.p1) * self.Delta
        self.nubar_dot = self.p1 * self.n1 + self.p2 * self.n2
        self.g = np.array([self.p2 * self.Delta, -self.p1 * self.Delta])   # fluctuations of the rate
        self.Q = np.array([[-self.a, self.a], [self.b, -self.b]])
        self.pi = np.array([self.p1, self.p2])

    # ---- Green-Kubo and cumulant rates -------------------------------------------------------------------
    @property
    def K(self):
        """K = int_0^inf Cov(nudot(y_0), nudot(y_t)) dt = p1 p2 Delta^2 / gamma."""
        return self.p1 * self.p2 * self.Delta ** 2 / self.gamma

    @property
    def lam3(self):
        """Coefficient of s^3 in the growth rate lambda(s): p1 p2 (p2 - p1) Delta^3 / gamma^2."""
        return self.p1 * self.p2 * (self.p2 - self.p1) * self.Delta ** 3 / self.gamma ** 2

    @property
    def lam4(self):
        return self.p1 * self.p2 * (1 - 5 * self.p1 * self.p2) * self.Delta ** 4 / self.gamma ** 3

    def lam(self, s):
        """Growth rate of E exp(s X_T): the dominant eigenvalue of Q + s diag(g)."""
        s = np.asarray(s, complex)
        return 0.5 * (s * self.Sigma - self.gamma + np.sqrt(self.gamma ** 2 - 2 * self.gamma * s * self.Sigma
                                                             + s ** 2 * self.Delta ** 2))

    # ---- frequency residual --------------------------------------------------------------------------------
    def var_nu(self, T):
        g = self.gamma
        return 2 * self.K * (T - (1 - math.exp(-g * T)) / g)

    def var_nu_gk(self, T):
        return 2 * self.K * T

    def kappa3_nu_first(self, T):
        """Leading third cumulant of X(T): 6 lam3 T."""
        return 6 * self.lam3 * T

    def mgf_nu(self, s, T):
        """E exp(s X_T), closed form (s complex)."""
        s = np.asarray(s, complex)
        D = 0.5 * np.sqrt(self.gamma ** 2 - 2 * self.gamma * s * self.Sigma + s ** 2 * self.Delta ** 2)
        tr = s * self.Sigma - self.gamma
        D = np.where(D.real < 0, -D, D)                      # the formula is even in D; take Re D >= 0
        E = np.exp(-2 * D * T)
        return np.exp((0.5 * tr + D) * T) * (0.5 * (1 + E) - 0.25 * tr * (1 - E) / D)

    # ---- phase residual ------------------------------------------------------------------------------------
    def var_phase(self, T):
        g = self.gamma
        return 2 * self.K * (T ** 3 / 3 - T ** 2 / (2 * g) + 1 / g ** 3 - math.exp(-g * T) * (1 / g ** 3 + T / g ** 2))

    def var_phase_gk(self, T):
        return 2 * self.K * T ** 3 / 3

    def var_phase_two(self, T):
        return 2 * self.K * (T ** 3 / 3 - T ** 2 / (2 * self.gamma))

    def kappa3_phase_first(self, T):
        return 1.5 * self.lam3 * T ** 4

    def postfit_ms(self, T):
        """First-order mean-square phase residual over [0, T] after a fitted quadratic: 2 K T^3 / 2520."""
        return 2 * self.K * T ** 3 / 2520

    # ---- spectra (two-sided, per Hz) ----------------------------------------------------------------------
    def S_nudot(self, f):
        w = 2 * math.pi * np.asarray(f, float)
        return 2 * self.p1 * self.p2 * self.Delta ** 2 * self.gamma / (self.gamma ** 2 + w ** 2)

    def S_phase(self, f):
        w = 2 * math.pi * np.asarray(f, float)
        return self.S_nudot(f) / w ** 4


def K_renewal(Delta, m1, s1, m2, s2):
    """Green-Kubo coefficient for alternating renewal switching: state 1 durations mean m1, sd s1; state 2 m2, s2.
    K = Delta^2 (m2^2 s1^2 + m1^2 s2^2) / (2 (m1 + m2)^3)."""
    return Delta ** 2 * (m2 ** 2 * s1 ** 2 + m1 ** 2 * s2 ** 2) / (2 * (m1 + m2) ** 3)


def erlang_chain(m1, k1, m2, k2):
    """Generator of an alternating chain with Erlang(k1) durations of mean m1 in state 1 and Erlang(k2), mean m2,
    in state 2; returns (Q, indicator of state 1)."""
    n = k1 + k2
    Q = np.zeros((n, n))
    r1, r2 = k1 / m1, k2 / m2
    for i in range(n):
        r = r1 if i < k1 else r2
        Q[i, i] = -r
        Q[i, (i + 1) % n] = r
    ind = np.array([1.0] * k1 + [0.0] * k2)
    return Q, ind
