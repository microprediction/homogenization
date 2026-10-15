"""Reversible multi-state CIR speed laws and an exact two-turning-point example."""
import math
from fractions import Fraction as F

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq

from verify_switching_kappa import integrated_variance_rate


def modes(q, pi, theta, a):
    root = np.sqrt(pi)
    lam, vectors = eigh(-root[:, None]*q/root[None, :])
    assert abs(lam[0]) < 1e-13 and np.all(lam[1:] > 0)
    th = vectors[:, 1:].T @ (root*(theta-pi@theta))
    aa = vectors[:, 1:].T @ (root*(a-pi@a))
    return lam[1:], th**2, th*aa


def spectral(m, k, lam, s, c, order=0):
    """Order-n derivatives of the centered variance and long-run rate."""
    weight = k*s+c/2
    factor = (-1)**order * math.factorial(order)
    variance = factor*np.sum(weight*lam**order/(k+m*lam)**(order+1))
    rate = factor*(2*np.sum(s/lam)/m**(order+1)
                   + np.sum(c*lam**order/(k*(k+m*lam)**(order+1))))
    return variance, rate


def close(x, y, scale=1):
    tol = max(1e-14, 1024*np.finfo(float).eps*max(scale, abs(y)))
    assert abs(x-y) <= tol, (x, y, tol)


def check(q, pi, theta, a, k, m):
    assert np.all(a > 0) and np.all(2*k*theta > a)
    lam, s, c = modes(q, pi, theta, a)
    variance, rate = spectral(m, k, lam, s, c)
    exact, mean, _, second, _, _ = integrated_variance_rate(
        m, q, pi, k*theta, np.full(len(pi), k), a)
    base = (pi@a)*(pi@theta)
    close(mean, pi@theta)
    close(second.sum()-mean**2, base/(2*k)+variance, second.sum())
    close(exact, base/k**2+rate)


def polynomial(m):
    return -8600*m**4+9580*m**3+5799*m**2-242*m-11


def main():
    rng = np.random.default_rng(20261015)
    models = []
    for n in range(2, 7):
        for _ in range(8):
            pi = rng.uniform(0.5, 1.5, n)
            pi /= pi.sum()
            conductance = rng.uniform(0.1, 1, (n, n))
            conductance = (conductance+conductance.T)/2
            np.fill_diagonal(conductance, 0)
            q = conductance/pi[:, None]
            q -= np.diag(q.sum(axis=1))
            q /= np.max(-np.diag(q))
            models.append((q, pi))
    # Repeated nonzero eigenvalue: sums are invariant to its basis choice.
    models.append((np.ones((5, 5))/5-np.eye(5), np.full(5, 0.2)))
    checks = 0
    for q, pi in models:
        k = rng.uniform(0.5, 2)
        theta = rng.uniform(0.02, 0.1, len(pi))
        a = 2*k*theta*rng.uniform(0.05, 0.95, len(pi))
        for m in np.geomspace(0.05, 32, 8):
            check(q, pi, theta, a, k, m)
            checks += 1
    print(f"{checks} spectral/moment/Poisson checks, including a repeated eigenvalue")

    cone_checks, minima = 0, 0
    for q, pi in models[::5]:
        k = 1.3
        theta = 0.05+rng.uniform(-0.002, 0.002, len(pi))
        for slope in (-3*k, -2*k, -k, 0, k):
            a = 0.04*k+slope*(theta-0.05)
            lam, s, c = modes(q, pi, theta, a)
            for m in (0.1, 1, 10):
                check(q, pi, theta, a, k, m)
                if slope >= -2*k:
                    for order in range(7):
                        v, rate = spectral(m, k, lam, s, c, order)
                        assert (-1)**order*v >= -1e-14
                        assert (-1)**order*rate > 0
                        cone_checks += 1
                else:
                    assert spectral(m, k, lam, s, c, 1)[0] > 0
            if slope < -2*k:
                scaled = lambda m: m*m*spectral(m, k, lam, s, c, 1)[1]
                upper = 1.0
                while scaled(upper) <= 0:
                    upper *= 2
                root = brentq(scaled, 1e-8, upper)
                assert scaled(root/2) < 0 < scaled(root*2)
                minima += 1
    print(f"{cone_checks} alternating-derivative checks; {minima} affine-loading minima")

    q = np.array([[-31, 29, 2], [29, -31, 2], [2, 2, -4]])/6
    pi, theta, a = np.full(3, 1/3), np.array([128, 123, 124])/2500, np.array([3, 9, 3])/100
    lam, s, c = modes(q, pi, theta, a)
    close(np.max(np.abs(lam-[1, 10])), 0)
    for actual, expected in zip(s, (F(1, 12500000), F(1, 1500000))):
        close(actual, float(expected))
    for actual, expected in zip(c, (F(1, 250000), -F(1, 50000))):
        close(actual, float(expected))
    # Exact rational identity, not an eigenvalue-grid/root-count argument.
    for m in (F(1, 20), F(1, 10), F(1), F(2), F(3)):
        derivative = -F(11, 37500000)/m**2-F(1, 250000)/(1+m)**2+F(1, 5000)/(1+10*m)**2
        assert derivative*37500000*m*m*(1+m)**2*(1+10*m)**2 == polynomial(m)
    signs = [-8600, 9580, 5799, -242, -11]
    assert sum(x*y < 0 for x, y in zip(signs, signs[1:])) == 2
    assert polynomial(F(1, 20)) < 0 < polynomial(F(1, 10))
    assert polynomial(F(1)) > 0 > polynomial(F(2))
    roots = [brentq(polynomial, 0.05, 0.1), brentq(polynomial, 1, 2)]
    print("Benchmark: Q=[[-31,29,2],[29,-31,2],[2,2,-4]]/6; kappa=1; pi=(1,1,1)/3")
    print("theta=(128,123,124)/2500; a=(3,9,3)/100; Feller margins:", 2*theta-a)
    print("lambda=(1,10); S=(1/12500000,1/1500000); C=(1/250000,-1/50000)")
    print("P(m)=-8600m^4+9580m^3+5799m^2-242m-11; denominator=37500000m^2(1+m)^2(1+10m)^2")
    for m in roots:
        check(q, pi, theta, a, 1, m)
        close(spectral(m, 1, lam, s, c, 1)[1], 0)
        print(f"turning speed {m:.10f}; rate {0.0025+spectral(m, 1, lam, s, c)[1]:.13f}")
    ratio = math.sqrt(1750/39)
    vmax = (ratio-1)/(10-ratio)
    assert 1 < ratio < 10
    close(spectral(vmax, 1, lam, s, c, 1)[0], 0)
    assert spectral(vmax/2, 1, lam, s, c, 1)[0] > 0
    assert spectral(vmax*2, 1, lam, s, c, 1)[0] < 0
    print(f"R=sqrt(1750/39); stationary-variance maximum speed {vmax:.10f}; averaged rate=0.0025")
    print("Exact Descartes/sign proof: two simple positive roots, in (1/20,1/10) and (1,2)")
    print("PASS")


if __name__ == "__main__":
    main()
