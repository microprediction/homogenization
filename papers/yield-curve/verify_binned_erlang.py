"""Finite-bin Erlang-clock inversion without a matrix-logarithm branch."""
import math
from fractions import Fraction

import mpmath as mp
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm


def check(condition):
    if not condition:
        raise AssertionError("binned-Erlang certificate failed")


def norm(a):
    return max(sum(abs(a[i, j]) for j in range(a.cols)) for i in range(a.rows))


def bin_matrices(q, shape, rate, width, count=None):
    n, d = q.rows, shape-1
    a = q-rate*mp.eye(n)
    b = mp.expm(width*a)
    ai = a**-1
    integrals = [ai*(b-mp.eye(n))]
    for power in range(1, shape):
        integrals.append(ai*(width**power*b-power*integrals[-1]))
    coefficients = [rate**shape*width**k*integrals[d-k]
                    /(math.factorial(k)*math.factorial(d-k)) for k in range(shape)]
    bins = []
    for k in range(2*shape if count is None else count):
        polynomial = sum((k**r*c for r, c in enumerate(coefficients)), mp.zeros(n))
        bins.append(b**k*polynomial)
    return bins, b


def cdf(shape, x):
    return mp.gammainc(shape, 0, x, regularized=True)


def fit_rate(shape, probability, width):
    low, high = mp.mpf(0), mp.mpf(1)
    while cdf(shape, high*width) < probability:
        high *= 2
    for _ in range(500):
        middle = (low+high)/2
        if cdf(shape, middle*width) < probability:
            low = middle
        else:
            high = middle
    return (low+high)/2


def fit_clock(probability, second_cdf, width, bound):
    candidates = []
    for shape in range(1, bound+1):
        rate = fit_rate(shape, probability, width)
        candidates.append((abs(cdf(shape, 2*rate*width)-second_cdf), shape, rate))
    candidates.sort()
    return candidates[0][1], candidates[0][2], candidates[1][0] if bound > 1 else mp.inf


def reconstruct(bins, shape, width, rate):
    n, d = bins[0].rows, shape-1
    h0 = mp.matrix(n*shape)
    h1 = mp.matrix(n*shape)
    for i in range(n*shape):
        for j in range(n*shape):
            index = i//n+j//n
            h0[i, j], h1[i, j] = bins[index][i % n, j % n], bins[index+1][i % n, j % n]
    inverse = h0**-1
    companion = h1*inverse
    b = companion[d*n:shape*n, d*n:shape*n]/shape
    difference = sum(((-1)**(d-k)*math.comb(d, k)*b**(-k)*bins[k]
                      for k in range(shape)), mp.zeros(n))
    q = rate*mp.eye(n)+rate**shape*width**d*(b-mp.eye(n))*difference**-1
    return q, rate, b, mp.log10(norm(h0)*norm(inverse))


def main():
    exact = 0
    for d in range(8):
        for i in range(d+1):
            for j in range(d+1):
                value = sum((-1)**(i+j-r-s)*math.comb(i, r)*math.comb(j, s)*(r+s)**d
                            for r in range(i+1) for s in range(j+1))
                check(value == sum((-1)**(i+j-k)*math.comb(i+j, k)*k**d for k in range(i+j+1)))
                if i+j >= d:
                    check(value == (math.factorial(d) if i+j == d else 0))
                exact += 1
    print(f"{exact} exact finite-difference/Hankel antidiagonal identities")
    mp.mp.dps = 80
    separation = mp.inf
    for shape in range(1, 8):
        for x in (mp.mpf('.1'), mp.mpf(1), mp.mpf(10)):
            fitted, rate, gap = fit_clock(cdf(shape, x), cdf(shape, 2*x), mp.mpf(1), 7)
            check(fitted == shape and abs(rate-x) < mp.mpf('2e-8'))
            separation = min(separation, gap)
    print(f"21 two-cutpoint shape/rate fits (bound 7): minimum runner-up CDF gap {float(separation):.3e}")
    nilpotent = np.outer([1, -1, 0, 0], [0, 0, 1, -1])/16
    cases = [np.array([[-1.2, .9, .3], [.2, -.6, .4], [.7, .1, -.8]]),
             np.ones((4, 4))/4-np.eye(4)+nilpotent,
             np.array([[-1., 1., 0.], [0., -.5, .5], [0., 0., 0.]])]
    max_recovery = max_rate = max_quadrature = 0.
    count = 0
    for qf in cases:
        q = mp.matrix(qf.tolist())
        # Remove binary64 row-sum noise before asserting an exact generator model.
        for i in range(q.rows):
            q[i, i] = -sum(q[i, j] for j in range(q.cols) if i != j)
        for shape in (1, 2, 3, 7):
            rate, width = mp.mpf('1.3'), mp.mpf('.4')
            bins, true_b = bin_matrices(q, shape, rate, width, count=14)
            fitted_shape, fitted_rate, _ = fit_clock(sum(bins[0][0, j] for j in range(q.rows)),
                sum(bins[k][0, j] for k in (0, 1) for j in range(q.rows)), width, 7)
            check(fitted_shape == shape)
            recovered, fitted_rate, b, _ = reconstruct(bins, fitted_shape, width, fitted_rate)
            max_recovery = max(max_recovery, float(norm(recovered-q)/(1+norm(q))))
            max_rate = max(max_rate, float(abs(fitted_rate-rate)))
            check(norm(b-true_b) < mp.mpf('2e-8'))
            for k, target in enumerate(bins):
                integral = quad_vec(lambda t: float(rate)**shape*t**(shape-1)*np.exp(-float(rate)*t)
                                    /math.factorial(shape-1)*expm(t*np.array(q.tolist(), dtype=float)),
                                    k*float(width), (k+1)*float(width), epsabs=1e-11)[0]
                max_quadrature = max(max_quadrature, np.max(np.abs(integral-np.array(target.tolist(), dtype=float))))
                count += 1
    check(max(max_recovery, max_rate, max_quadrature) < 2e-8)
    print(f"12 joint shape/rate/Q recoveries (shapes 1,2,3,7; bound 7; directed, defective, absorbing): max scaled Q error {max_recovery:.3e}, rate error {max_rate:.3e}")
    print(f"{count} independent bin integrals: max matrix discrepancy {max_quadrature:.3e}")

    reset_p = mp.ones(2)/2
    tv_error = mp.mpf(0)
    for shape in (1, 2, 3, 7):
        for speed in (1, 2, 4, 8, 16):
            laws = []
            for c in (speed, 2*speed):
                q = c*(reset_p-mp.eye(2))
                bins, _ = bin_matrices(q, shape, mp.mpf(1), mp.mpf(1))
                total = (mp.eye(2)-q)**(-shape)
                laws.append(bins+[total-sum(bins, mp.zeros(2))])
            tv = sum(abs(x-y) for h0, h1 in zip(*laws) for x, y in zip(h0, h1))/4
            target = (mp.mpf(1)/(1+speed))**shape/2-(mp.mpf(1)/(1+2*speed))**shape/2
            tv_error = max(tv_error, abs(tv-target))
    check(tv_error < mp.mpf('2e-13'))
    print(f"20 fixed-clock reset TV identities (speeds 1,2,4,8,16): max discrepancy {float(tv_error):.3e}")

    mp.mp.dps = 180
    p, eye = mp.ones(3)/3, mp.eye(3)
    e = eye-p
    j = mp.matrix([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])/mp.sqrt(3)
    theta = mp.pi/7
    a, b = 10*mp.cos(theta)-1, 10*mp.sin(theta)
    forward, backward = -a*e+b*j, -a*e-b*j
    check(min(forward[i, k] for i in range(3) for k in range(3) if i != k) > 0)
    kernels = [(eye-q)**-7 for q in (forward, backward)]
    alias_error = max(norm(k-p+mp.mpf('1e-7')*e) for k in kernels)
    bins0, _ = bin_matrices(forward, 7, mp.mpf(1), mp.mpf(1))
    bins1, _ = bin_matrices(backward, 7, mp.mpf(1), mp.mpf(1))
    fitted_shape, fitted_rate, _ = fit_clock(sum(bins0[0][0, j] for j in range(3)),
        sum(bins0[k][0, j] for k in (0, 1) for j in range(3)), mp.mpf(1), 7)
    check(fitted_shape == 7)
    recovered, fitted_rate, _, log_condition = reconstruct(bins0, fitted_shape, mp.mpf(1), fitted_rate)
    recovery_error = norm(recovered-forward)/(1+norm(forward))
    check(max(alias_error, recovery_error, abs(fitted_rate-1)) < mp.mpf('2e-8'))
    print(f"Shape-7 known-clock alias: R=P-10^-7(I-P); discrepancy {float(alias_error):.3e}; minimum rate {float(min(forward[i,k] for i in range(3) for k in range(3) if i!=k)):.12f}")
    print(f"15-valued duration category separates it: first-bin matrix gap {float(norm(bins0[0]-bins1[0])):.12e}; recovery error {float(recovery_error):.3e}")
    print(f"Alias recovery uses 180 digits; log10 Hankel infinity-condition number {float(log_condition):.3f}")
    check([2*Fraction(1, 8), 2*Fraction(1, 6), 4*Fraction(1, 8), 4*Fraction(1, 6)]
          == [Fraction(1, 4), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)])
    print("Unknown-family scaling obstruction: two-step totals in [1/4,1/3] and [1/2,2/3], both in bin zero")
    print("PASS")


if __name__ == "__main__":
    main()
