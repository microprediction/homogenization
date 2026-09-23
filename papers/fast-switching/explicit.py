"""Closed-form terms for the two-state expansion used on the example pages.

For a two-state chain switching at rate lam each way, eps = 1/lam, gbar = (g1+g2)/2, gt = (g1-g2)/2 and gt(0) = 0:
    log m(t) = int gbar + (eps/2) int gt^2 - (eps^2/8) gt(t)^2 + O(eps^3)
    omega(t) = (eps/2) gt(t) - (eps^2/4) gt'(t) + O(eps^3),       a_{1,2} = m (1 +/- omega).
(The initial layer adds (eps^2/4) gt'(0) exp(-2 lam t) to omega, and first enters log m at eps^4.)
For constant g the two-state system is solved exactly:
    a_{1,2}(t) = exp((gbar - lam) t) [cosh(s t) + (lam +/- gt) sinh(s t) / s],   s = sqrt(lam^2 + gt^2).
"""
import cmath, math


# ---------------------------------------------------------------- Vasicek B = (1 - e^{-kappa t}) / kappa
def I_k(k, t, kappa):
    """int_0^t B^k."""
    s = t + sum(math.comb(k, j) * (-1) ** j * (1 - math.exp(-j * kappa * t)) / (j * kappa) for j in range(1, k + 1))
    return s / kappa ** k


def J_1(t, kappa, m):
    """int_0^t 1/(1 + m B)."""
    c, q = 1 + m / kappa, m / kappa
    return t / c + math.log((c - q * math.exp(-kappa * t)) / (c - q)) / (c * kappa)


def J_2(t, kappa, m):
    """int_0^t 1/(1 + m B)^2."""
    c, q = 1 + m / kappa, m / kappa
    return J_1(t, kappa, m) / c - (1 / (c - q * math.exp(-kappa * t)) - 1 / (c - q)) / (c * kappa)


def vasicek_jump_integrals(t, kappa, a, b, c, m):
    """int_0^t f^2 for f = a B + b B^2 + c (r - 1), r = 1/(1 + m B)  (c = 0 or m = 0: no jump part)."""
    I1, I2, I3, I4 = (I_k(k, t, kappa) for k in (1, 2, 3, 4))
    out = a * a * I2 + 2 * a * b * I3 + b * b * I4
    if c:
        J1, J2 = J_1(t, kappa, m), J_2(t, kappa, m)
        int_B_rm1 = (t - J1) / m - I1                       # int B (r - 1)
        int_B2_rm1 = (I1 - (t - J1) / m) / m - I2           # int B^2 (r - 1)
        int_rm1_sq = J2 - 2 * J1 + t                        # int (r - 1)^2
        out += c * c * int_rm1_sq + 2 * a * c * int_B_rm1 + 2 * b * c * int_B2_rm1
    return out


def vasicek_jump_mean(t, kappa, a, b, c, m):
    """int_0^t (a B + b B^2 + c (r - 1))."""
    out = a * I_k(1, t, kappa) + b * I_k(2, t, kappa)
    if c:
        out += c * (J_1(t, kappa, m) - t)
    return out


# ---------------------------------------------------------------- CIR B = 2(e^{ht}-1) / ((h+kappa)(e^{ht}-1) + 2h)
def cir_B(t, kappa, sigma):
    h = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    e = math.exp(h * t) - 1
    return 2 * e / ((h + kappa) * e + 2 * h)


def cir_int_B(t, kappa, sigma):
    """int_0^t B = -(2/sigma^2) log(2h e^{(kappa+h)t/2} / ((h+kappa)(e^{ht}-1) + 2h))."""
    h = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    return -2 / sigma ** 2 * (math.log(2 * h) + (kappa + h) * t / 2 - math.log((h + kappa) * (math.exp(h * t) - 1) + 2 * h))


def cir_int_B2(t, kappa, sigma):
    """int_0^t B^2 by E = e^{hs} and partial fractions: B = 2(E-1)/(cE + e0), c = h+kappa, e0 = h-kappa."""
    h = math.sqrt(kappa ** 2 + 2 * sigma ** 2)
    c, e0 = h + kappa, h - kappa
    A1 = 1 / e0 ** 2
    A2 = (1 - A1 * c * c) / c
    A3 = -2 - 2 * A1 * c * e0 - A2 * e0
    F = lambda E: A1 * math.log(E) + A2 / c * math.log(c * E + e0) - A3 / (c * (c * E + e0))
    return 4 / h * (F(math.exp(h * t)) - F(1.0))


# ---------------------------------------------------------------- assembling
def two_state_second_order(int_gbar, int_gt2, gt_T, gtp_T, eps, sign=+1):
    """a_{1 or 2}(T) through eps^2 from the closed-form pieces (gt(0) = 0)."""
    logm = int_gbar + eps / 2 * int_gt2 - eps ** 2 / 8 * gt_T ** 2
    om = eps / 2 * gt_T - eps ** 2 / 4 * gtp_T
    return cmath.exp(logm) * (1 + sign * om), logm, om


def two_state_constant_exact(gbar, gt, lam, t, sign=+1):
    s = cmath.sqrt(lam * lam + gt * gt)
    return cmath.exp((gbar - lam) * t) * (cmath.cosh(s * t) + (lam + sign * gt) * cmath.sinh(s * t) / s)
