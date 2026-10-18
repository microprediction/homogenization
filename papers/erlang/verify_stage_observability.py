"""Exact Erlang-stage bond jets and a short-maturity stage-prior inverse."""
import math
from fractions import Fraction as F

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp

from erlang_bonds import B, log_price_numerical, stage_chain

KAP, TH, SIG, MEANS = F(1, 2), (F(7, 100), F(1, 100)), F(1, 100), (F(3, 10), F(3, 5))


def check(condition):
    if not condition:
        raise AssertionError("stage-observability certificate failed")


def jets(k, order):
    b = [F(0)]+[(-KAP)**(j-1)/math.factorial(j) for j in range(1, order+1)]
    g = [[-KAP*theta*b[j]+SIG**2/2*sum(b[h]*b[j-h] for h in range(1, j))
          for theta in TH] for j in range(order+1)]
    a, frozen, ratio = [[F(1)]*(2*k)], [[F(1)]*2], [[F(0)]*(2*k)]
    for n in range(1, order+1):
        frozen.append([sum((g[h][i]*frozen[n-1-h][i] for h in range(1, n)), F(0))/n for i in range(2)])
        row = []
        for j in range(2*k):
            i = j//k
            value = k/MEANS[i]*(a[n-1][(j+1) % (2*k)]-a[n-1][j])
            value += sum(g[h][i]*a[n-1-h][j] for h in range(1, n))
            row.append(value/n)
        a.append(row)
        ratio.append([a[n][j]-frozen[n][j//k]
                      -sum(frozen[h][j//k]*ratio[n-h][j] for h in range(1, n+1))
                      for j in range(2*k)])
    return ratio


def leading(k, regime, remaining):
    return KAP*(TH[regime]-TH[1-regime])*(k/MEANS[regime])**remaining/math.factorial(remaining+2)


def main():
    zeros = coefficients = 0
    for k in (1, 2, 3, 4, 7):
        ratio = jets(k, k+2)
        for i in range(2):
            for remaining in range(1, k+1):
                stage = i*k+k-remaining
                for n in range(remaining+2):
                    check(ratio[n][stage] == 0)
                    zeros += 1
                check(ratio[remaining+2][stage] == leading(k, i, remaining))
                coefficients += 1
    print(f"Exact rational bond jets: {zeros} vanishing coefficients and {coefficients} first nonzero coefficients (k=1,2,3,4,7; both regimes)")

    zero_error = 0.
    for k in (1, 2, 4, 16):
        for i in (1, 2):
            for age in (0., .3, .6):
                zero_error = max(zero_error, abs(np.expm1(log_price_numerical(
                    0., .03, i, age, .5, .07, .01, .01, .3, .6, k))))
    check(zero_error < 2e-12)
    print(f"24 production zero-maturity unit-payoff checks: max error {zero_error:.3e}")

    k, nodes = 4, (1, 2, 3)
    determinant_constant = math.prod(leading(k, 0, ell) for ell in range(1, k))
    determinant_constant *= math.prod(t**3 for t in nodes)
    determinant_constant *= math.prod(nodes[j]-nodes[i] for i in range(k-1) for j in range(i+1, k-1))
    print(f"k=4, nodes 1,2,3: determinant power 12; exact leading coefficient {determinant_constant}")

    mp.mp.dps = 70
    to_mp = lambda x: mp.mpf(x.numerator)/x.denominator
    kap, sigma = to_mp(KAP), to_mp(SIG)
    theta, mu = list(map(to_mp, TH)), [to_mp(k/m) for m in MEANS]
    def rhs(t, a):
        b = -mp.expm1(-kap*t)/kap
        return [mu[j//k]*(a[(j+1) % (2*k)]-a[j])
                +(-kap*theta[j//k]*b+sigma**2*b*b/2)*a[j] for j in range(2*k)]
    solution = mp.odefun(rhs, 0, [mp.mpf(1)]*(2*k), tol=mp.mpf('1e-45'), degree=30)
    def frozen(t):
        b = -mp.expm1(-kap*t)/kap
        i1 = (t-b)/kap
        i2 = (t-2*b-mp.expm1(-2*kap*t)/(2*kap))/kap**2
        return mp.exp(-kap*theta[0]*i1+sigma**2*i2/2)
    previous = None
    recovery_error = mp.mpf(0)
    prior = [mp.mpf(j)/10 for j in (1, 2, 3, 4)]
    for denominator in (128, 256, 512, 1024, 2048):
        eps = mp.mpf(1)/denominator
        h, observed = mp.matrix(k-1), mp.matrix(k-1, 1)
        for row, node in enumerate(nodes):
            t = eps*node
            a, f = solution(t), frozen(t)
            d = [a[k-ell]/f-1 for ell in range(1, k+1)]
            for col in range(k-1):
                h[row, col] = d[col]-d[-1]
            observed[row] = sum(prior[ell-1]*a[k-ell] for ell in range(1, k+1))/f-1-d[-1]
        singular = mp.svd(h, compute_uv=False)[k-2]
        normalized = mp.det(h)/(to_mp(determinant_constant)*eps**12)
        fitted = list(mp.lu_solve(h, observed))
        fitted.append(1-sum(fitted))
        recovery_error = max(recovery_error, max(abs(a-b) for a, b in zip(fitted, prior)))
        if previous is not None:
            measured_power = mp.log(previous/singular)/mp.log(2)
        previous = singular
    check(abs(normalized-1) < mp.mpf('.1') and abs(measured_power-5) < mp.mpf('.1'))
    check(recovery_error < mp.mpf('2e-8'))
    print(f"High-precision ODE inverse at epsilon=1/2048: normalized determinant {float(normalized):.9f}; least-singular-value power {float(measured_power):.9f}")
    print(f"Known-high-regime prior (0.1,0.2,0.3,0.4), five designs: max recovery error {float(recovery_error):.3e}")

    q = stage_chain(.3, .6, k)
    levels = np.repeat([.07, .01], k)
    def binary_rhs(t, a):
        b = B(t, .5)
        return q@a+(-.5*levels*b+.5*.01**2*b*b)*a
    a = solve_ivp(binary_rhs, (0, 1), np.ones(2*k), rtol=1e-11, atol=1e-13).y[:, -1]
    yields = (-np.log(a[:k])+B(1, .5)*.03)*1e4
    check(yields[0]-yields[-1] > 1)
    print("Pure high-regime stages, remaining 4,3,2,1; one-year yields (bp): "
          +", ".join(f"{value:.6f}" for value in yields))
    print("PASS")


if __name__ == "__main__":
    main()
