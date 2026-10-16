"""Exact rational-interval witness: finite-maturity variance rises with speed.

All interval endpoints are Fractions, rounded outward to a rational lattice
after arithmetic. Square roots use integer isqrt; Taylor remainders enclose
the exponentials and trig functions. A separate floating polynomial semigroup
checks the closed formula but is not used to certify its strict inequality.
"""
from fractions import Fraction as F
from math import factorial, isqrt

import numpy as np

from verify_switching_kappa import stationary_integrated_variance


GRID = 10**30


def down(x):
    return F(x.numerator*GRID // x.denominator, GRID)


class I:
    def __init__(self, lo, hi=None):
        self.lo, self.hi = F(lo), F(lo if hi is None else hi)
        assert self.lo <= self.hi

    @staticmethod
    def cast(x):
        return x if isinstance(x, I) else I(x)

    @staticmethod
    def rounded(lo, hi):
        return I(down(lo), -down(-hi))

    def __add__(self, x):
        x = I.cast(x)
        return I.rounded(self.lo+x.lo, self.hi+x.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, x):
        return self + -I.cast(x)

    def __rsub__(self, x):
        return I.cast(x) + -self

    def __mul__(self, x):
        x = I.cast(x)
        products = [a*b for a in (self.lo, self.hi) for b in (x.lo, x.hi)]
        return I.rounded(min(products), max(products))

    __rmul__ = __mul__

    def __truediv__(self, x):
        x = I.cast(x)
        assert x.lo > 0 or x.hi < 0
        return self * I(1/x.hi, 1/x.lo)

    def __pow__(self, n):
        assert isinstance(n, int) and n >= 0
        result, base = I(1), self
        while n:
            if n % 2:
                result = result*base
            base, n = base*base, n//2
        return result

    def sqrt(self):
        assert self.lo >= 0
        lo = isqrt(self.lo.numerator*GRID**2 // self.lo.denominator)
        hi = isqrt(self.hi.numerator*GRID**2 // self.hi.denominator) + 1
        return I(F(lo, GRID), F(hi, GRID))

    def midpoint(self):
        return float((self.lo+self.hi)/2)


def exp_interval(x):
    y = x/128
    assert max(abs(y.lo), abs(y.hi)) <= 1
    polynomial = sum((y**j)/factorial(j) for j in range(41))
    # Lagrange remainder, e^|y| < 3 for |y|<=1; then raise to power 128.
    error = F(3, factorial(41))
    small = polynomial + I(-error, error)
    assert small.lo > 0
    return small**128


def trig_interval(x):
    assert max(abs(x.lo), abs(x.hi)) <= 8
    error = F(8**121, factorial(121))  # Degree-120 Taylor; derivatives bounded by 1.
    cosine = sum((-1)**j * (x**(2*j))/factorial(2*j) for j in range(61))
    sine = sum((-1)**j * (x**(2*j+1))/factorial(2*j+1) for j in range(60))
    return cosine + I(-error, error), sine + I(-error, error)


def add(z, w):
    return z[0]+w[0], z[1]+w[1]


def mul(z, w):
    return z[0]*w[0]-z[1]*w[1], z[0]*w[1]+z[1]*w[0]


def scale(x, z):
    return x*z[0], x*z[1]


def divide(z, w):
    denominator = w[0]*w[0]+w[1]*w[1]
    product = mul(z, (w[0], -w[1]))
    return product[0]/denominator, product[1]/denominator


def evaluate(speed):
    # cos(pi/16), sin(pi/16) via exact half-angle radicals, no numerical pi.
    radical = (2+I(2).sqrt()).sqrt()
    cosine, sine = (2+radical).sqrt()/2, (2-radical).sqrt()/2
    z = 1-cosine, sine
    a = scale(speed, z)
    co, si = trig_interval(a[1])
    exponential = scale(exp_interval(-a[0]), (co, -si))
    fa = scale(2, divide(add(add(a, (I(-1), I(0))), exponential), mul(a, a)))
    fk = 2*(I(99)+exp_interval(I(-100)))/10000
    numerator = add(scale(100, fa), scale(-fk, a))
    denominator = add((I(10000), I(0)), scale(-1, mul(a, a)))
    integrated = F(1, 5000000)*fk + F(1, 200)*divide(numerator, denominator)[0]
    variance = I(F(1, 5000000)) + F(1, 200)*divide((I(1), I(0)), add((I(100), I(0)), a))[0]
    rate = F(1, 250000000) + F(1, 20000*speed)
    return integrated, variance, rate


def main():
    q = np.roll(np.eye(32), 1, axis=1)-np.eye(32)
    pi = np.ones(32)/32
    theta = .04+.01*np.cos(2*np.pi*np.arange(32)/32)
    enclosed = {}
    brackets = {36: (F(".00000249554"), F(".00000249556")),
                40: (F(".00000265526"), F(".00000265529"))}
    print("32-cycle; T=1 kappa=100 xi^2=.001 theta_j=.04+.01 cos(2 pi j/32)")
    print("strict minimum Feller margin =", F(5999, 1000))
    print("speed  Var(v)          Var(V_1)        long-run rate   semigroup discrepancy")
    for speed in (36, 40):
        integrated, variance, rate = evaluate(speed)
        lower, upper = brackets[speed]
        assert lower < integrated.lo <= integrated.hi < upper
        direct = stationary_integrated_variance(speed, 1., q, pi, 100*theta,
                                               np.full(32, 100.), np.full(32, .001))
        tolerance = max(1e-14, 10000*np.finfo(float).eps*(.04**2+abs(direct)))
        assert abs(direct-integrated.midpoint()) < tolerance
        enclosed[speed] = integrated, variance, rate
        print(f"{speed} {variance.midpoint():.12e} {integrated.midpoint():.12e} "
              f"{float(rate):.12e} {abs(direct-integrated.midpoint()):.3e}")
        print(f"  certified: {float(lower):.8e} < Var(V_1) < {float(upper):.8e}")
    earlier, later = enclosed[36], enclosed[40]
    assert later[0].lo > F(106, 100)*earlier[0].hi
    assert later[0].lo-earlier[0].hi > F(".00000015970")
    assert later[1].hi < earlier[1].lo and later[2] < earlier[2]
    print("certified increase > 1.5970e-7 and > 6%; one-time variance and long-run rate decrease")
    print("PASS")


if __name__ == "__main__":
    main()
