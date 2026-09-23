"""Option prices under fast regime switching: Black-Scholes with a switching volatility (Lewis's formula) and
calls on zero-coupon bonds under Vasicek with a switching mean level and volatility (Gil-Pelaez inversion)."""
import math
import cmath
import numpy as np
from fastswitch import FastSwitch, numerical_a_callable
from models import bs_switching, vasicek_terminal


def _nodes(U, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return (x + 1) * U / 2, w * U / 2


def bs_call(S0, K, T, r, sigmas, Q, start, order=None, U=None, n=96):
    """European call; order=None uses the numerical solution of the reduced system. The Fourier integral stops
    where the characteristic function falls below exp(-40); beyond that the regimes' exponents differ by more
    than the switching rate and the expansion does not apply."""
    if U is None:
        Qa = np.asarray(Q, float)
        wv, vl = np.linalg.eig(Qa.T)
        pi = np.real(vl[:, np.argmin(abs(wv))])
        pi = pi / pi.sum()
        U = math.sqrt(80 / (float(pi @ np.asarray(sigmas) ** 2) * T))
    us, ws = _nodes(U, n)
    k, tot = math.log(S0 / K), 0.0
    for u, w in zip(us, ws):
        g, gf = bs_switching(u - 0.5j, r, sigmas)
        phi = numerical_a_callable(T, Q, gf, rtol=1e-12)[start] if order is None else FastSwitch(Q, g, order=order).a(T, order)[start]
        tot += w * (cmath.exp(1j * u * k) * phi).real / (u * u + 0.25)
    return S0 - math.sqrt(S0 * K) * math.exp(-r * T) / math.pi * tot


def zcb_call(T, S, K, x0, start, kappa, thetas, sigmas, Q, order=None, U=None, n=160):
    """Call expiring at T on the zero-coupon bond maturing at S. The bond at T in regime j is A_j exp(-b x_T).
    The Gil-Pelaez integrals stop at eight standard deviations of x_T, in frequency, under the averaged model."""
    Q = np.asarray(Q, float)
    m = len(thetas)
    if U is None:
        wv, vl = np.linalg.eig(Q.T)
        pi = np.real(vl[:, np.argmin(abs(wv))])
        pi = pi / pi.sum()
        var = float(pi @ np.asarray(sigmas) ** 2) / (2 * kappa) * (1 - math.exp(-2 * kappa * T))
        U = 8 / math.sqrt(var)
    b = (1 - math.exp(-kappa * (S - T))) / kappa

    def a_vec(t, c, a0):
        g, gf, Bc = vasicek_terminal(kappa, thetas, sigmas, c)
        if order is None:
            a = numerical_a_callable(t, Q, gf, rtol=1e-12, a0=a0)
        else:
            a = FastSwitch(Q, g, order=order, a0=a0).a(t, order)
        return a, Bc.value(t)
    A = a_vec(S - T, 0.0, np.ones(m))[0].real       # regime bond factors at expiry
    us, ws = _nodes(U, n)
    price = 0.0
    for j in range(m):
        xstar = math.log(A[j] / K) / b
        ej = np.eye(m)[j]
        for c0, weight in ((b, A[j]), (0.0, -K)):
            a, Bv = a_vec(T, c0, ej)
            total = (a[start] * cmath.exp(-Bv * x0)).real
            integral = 0.0
            for u, w in zip(us, ws):
                a, Bv = a_vec(T, c0 - 1j * u, ej)
                phi = a[start] * cmath.exp(-Bv * x0)
                integral += w * (cmath.exp(-1j * u * xstar) * phi).imag / u
            price += weight * (0.5 * total - integral / math.pi)
    return price
