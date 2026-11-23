"""Sharp mixing-unrestricted stationary-window confidence obstruction.

Fraction identities certify the geometry and finite-record example; floating
Feynman-Kac checks are independent cross-checks, not proofs of coverage.
"""
from fractions import Fraction as F
from itertools import product

import numpy as np
from scipy.linalg import expm, expm_frechet


def generator(p, rate):
    return rate*np.array([[-(1-p), 1-p], [p, -p]])


def target(p, rate, t, channel):
    law = np.array([p, 1-p]); v = p*(1-p)
    a = (-np.expm1(-rate*t)/(rate*t))**2*np.exp(-channel*rate*t)
    return t*t*(np.outer(law, law)+a*v*np.array([[1., -1.], [-1., 1.]]))


def independent_target(p, rate, t, channel):
    q = generator(p, rate); law = np.array([p, 1-p]); one = np.ones(2)
    derivatives = [expm_frechet(t*q, t*np.diag(e), compute_expm=False) for e in np.eye(2)]
    return np.array([[law @ da @ expm(channel*t*q) @ db @ one
                      for db in derivatives] for da in derivatives])


def main():
    c = np.array([[F(1), F(-1)], [F(-1), F(1)]], dtype=object)
    center = np.array([[F(1, 2), F(0)], [F(0), F(1, 2)]], dtype=object)
    count = 0; worst_squared = F(0)
    for p, a in product((F(j, 32) for j in range(33)), (F(j, 16) for j in range(17))):
        law = np.array([p, 1-p]); reverse = law[::-1]; v = p*(1-p); z = p-F(1, 2)
        moment = np.outer(law, law)+a*v*c
        mirrored = np.outer(reverse, reverse)+a*v*c
        gap = (2*p-1)*np.array([[1, 0], [0, -1]])
        assert np.all(moment-mirrored == gap)
        assert sum(((moment-mirrored)**2).flat) == 2*(1-2*p)**2
        squared = sum(((moment-center)**2).flat)
        assert squared == 2*z*z+4*v*v*(1-a)**2
        assert squared <= F(1, 4)+4*z**4 <= F(1, 2)
        assert all(x >= 0 for x in moment.flat) and sum(moment.flat) == 1
        worst_squared = max(worst_squared, squared); count += 1
    assert worst_squared == F(1, 2)
    print(f"{count} exact rational gap, positivity and matching-center checks (including closure points)")
    print(f"sharp squared no-data radius/T^4={worst_squared}; radius/T^2={np.sqrt(.5):.9f}")
    cases = 0; worst_formula = worst_gap = 0.
    for p, exposure, t, channel in product((.01, .25, .5, .9), (1e-7, .01, .3, 2., 20.),
                                          (.2, 1., 4.), (0, 1)):
        rate = exposure/t; moment = target(p, rate, t, channel)
        independent = independent_target(p, rate, t, channel)
        scale = max(1., t*t); error = np.max(abs(moment-independent))/scale
        worst_formula = max(worst_formula, error)
        assert error <= 32768*np.finfo(float).eps
        predicted = t*t*(2*p-1)*np.diag([1., -1.])
        gap_error = np.max(abs(moment-target(1-p, rate, t, channel)-predicted))/scale
        worst_gap = max(worst_gap, gap_error)
        assert gap_error <= 4096*np.finfo(float).eps; cases += 1
    print(f"{cases} independent Feynman-Kac target checks; maximum scaled formula error={worst_formula:.3e}")
    print(f"maximum scaled mirrored-gap discrepancy={worst_gap:.3e}")
    # Exact finite-record witness: all rates are below the known cap 1.
    blocks = 1000000; t = F(1); horizon = 3*blocks*t
    p = F(1, 100); rate = 1/(100*horizon); radius = F(69, 100); delta = F(1, 20)
    assert rate*max(p, 1-p) <= 1
    gap_squared = 2*t**4*(1-2*p)**2
    assert gap_squared > 4*radius*radius
    exponent = rate*(1-p)*horizon
    common_mass_floor = 1-exponent  # exp(-x)>=1-x, x>=0.
    failure_floor = common_mass_floor/2
    assert failure_floor > delta
    print(f"finite witness: N={blocks}, T={t}, H={horizon}, p={p}, r={rate}, known exit cap=1")
    print(f"radius={radius} < half target gap={float(np.sqrt(float(gap_squared))/2):.9f}")
    print(f"common-atom exponent={exponent}; mass >= {common_mass_floor}; minimax failure >= {failure_floor}={float(failure_floor):.5f}")
    print(f"nominal delta={delta}; actual common mass/2={np.exp(-float(exponent))/2:.9f}")
    # Exact finite randomization check: disjoint success sets on the shared atom.
    tests = 0
    for weight_left, weight_right in product((F(j, 20) for j in range(21)), repeat=2):
        if weight_left+weight_right <= 1:
            left_failure = common_mass_floor*(1-weight_left)
            right_failure = common_mass_floor*(1-weight_right)
            assert max(left_failure, right_failure) >= failure_floor; tests += 1
    print(f"{tests} exact shared-atom randomized-decision checks")
    for inverse_p in (10, 100, 10000, 1000000):
        tiny_p = 1/inverse_p; q = generator(tiny_p, .001)
        error = np.linalg.norm(np.diag([1., 0.])-target(tiny_p, .001, 1., 0))
        assert error <= np.sqrt(2)+64*np.finfo(float).eps
        assert np.max(abs(q.sum(1))) <= 64*np.finfo(float).eps
        print(f"constant-path raw-estimator error, p=1/{inverse_p}: {error:.9f}")
    print(f"sharp constant-path raw budget/T^2={np.sqrt(2):.9f}")
    print("PASS")


if __name__ == '__main__':
    main()
