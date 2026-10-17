"""One elapsed-time total: conditional kernels, grid aliases and near-grid TV."""
from collections import defaultdict
from fractions import Fraction
from itertools import product
import math

import numpy as np
import mpmath as mp
import sympy as sp
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.special import gammaln


def check(condition):
    if not condition:
        raise AssertionError("total-time certificate failed")


def observed_law(q, m, durations, weights, retain_path=False, kernels=None):
    n = len(q)
    if kernels is None:
        kernels = [expm(float(t)*q) for t in durations]
    law = defaultdict(float)
    for choices in product(range(len(durations)), repeat=m):
        # Two-atom totals can instead be keyed by their count, avoiding float keys.
        total = sum(choices) if retain_path else sum((durations[c] for c in choices), Fraction(0))
        weight = math.prod(weights[c] for c in choices)/n
        for path in product(range(n), repeat=m+1):
            value = weight*math.prod(kernels[c][path[j], path[j+1]]
                                     for j, c in enumerate(choices))
            key = (total, path) if retain_path else (total, path[0], path[-1])
            law[key] += value
    return law


def main():
    p = sp.ones(3)/3
    e = sp.eye(3)-p
    j = sp.Matrix([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])/sp.sqrt(3)
    check(j*j == -e and j*p == sp.zeros(3) and p*j == sp.zeros(3))
    a, b = 4*sp.pi, 2*sp.pi
    q0, q1 = -a*e, -a*e+b*j
    for q in (q0, q1):
        check(q*sp.ones(3, 1) == sp.zeros(3, 1))
        check(min(q[r, s] for r in range(3) for s in range(3) if r != s) > 0)
    # Algebraic rotation formula proves equality on every positive integer time.
    for k in range(1, 7):
        check(sp.sin(k*b) == 0 and sp.cos(k*b) == 1)
    f = sp.Matrix([1, -1, 0])
    strengths = [sp.simplify(2*(f.T*(-q+p).inv()*f)[0]/3) for q in (q0, q1)]
    check(strengths == [1/(3*sp.pi), 4/(15*sp.pi)])
    check(sp.simplify(strengths[1]/strengths[0]) == sp.Rational(4, 5))
    for n in range(3, 7):
        jn = sp.zeros(n)
        jn[:3, :3] = j
        en = sp.eye(n)-sp.ones(n)/n
        qn = -2*n*b*en+b*jn
        check(jn**3 == -jn and jn*sp.ones(n, 1) == sp.zeros(n, 1))
        check(min(qn[r, s] for r in range(n) for s in range(n) if r != s) > 0)
    q0f, q1f = np.array(q0).astype(float), np.array(q1).astype(float)
    alias_error = max(np.max(np.abs(expm(k*q0f)-expm(k*q1f))) for k in range(1, 7))
    check(alias_error < 2e-9)
    print("Exact J^2=-(I-P), positivity, integer-time rotation and fixed-feature GK identities checked")
    print("Higher-dimensional rotation and strict-positivity checks: n=3,4,5,6")
    print(f"Grid alias: max kernel discrepancy {alias_error:.3e}; GK strengths {float(strengths[0]):.12f}, {float(strengths[1]):.12f}; ratio 4/5")

    nilpotent = np.outer([1, -1, 0, 0], [0, 0, 1, -1])/16
    defective = np.ones((4, 4))/4-np.eye(4)+nilpotent
    check(np.array_equal(nilpotent@nilpotent, np.zeros((4, 4))))
    cases = [np.array([[-1.2, .9, .3], [.2, -.6, .4], [.7, .1, -.8]]),
             defective, np.array([[-1., 1., 0.], [0., -.5, .5], [0., 0., 0.]])]
    durations, weights = [Fraction(1, 10), Fraction(3, 20)], [1/3, 2/3]
    records = 0
    conditional_error = 0.
    for q in cases:
        for m in (1, 2, 3):
            law = observed_law(q, m, durations, weights)
            mass = defaultdict(float)
            for choices in product(range(2), repeat=m):
                total = sum((durations[c] for c in choices), Fraction(0))
                mass[total] += math.prod(weights[c] for c in choices)
            expected = {t: expm(float(t)*q) for t in mass}
            for (t, row, col), value in law.items():
                conditional_error = max(conditional_error,
                    abs(value*len(q)/mass[t]-expected[t][row, col]))
                records += 1
            check(abs(sum(law.values())-1) < 2e-9)
    check(conditional_error < 2e-9)
    print(f"{records} endpoint/total conditional-kernel entries, including defective and absorbing chains: max error {conditional_error:.3e}")

    # The smooth erasing clock from the preceding note: sum two intervals only.
    theta = np.pi/12
    qa = -(2*np.cos(theta)-1)*np.array(e).astype(float) + 2*np.sin(theta)*np.array(j).astype(float)
    w = 1/4097
    shapes, mixing = np.array([2, 14, 26]), np.array([w*w, 2*w*(1-w), (1-w)**2])
    tilted_error = 0.
    for s in (0., .1, 1.):
        c = np.linalg.inv((1+s)*np.eye(3)-qa)
        transform = w*c+(1-w)*np.linalg.matrix_power(c, 13)
        target = transform@transform
        def integrand(t):
            density = 0. if t == 0 else np.dot(mixing, np.exp((shapes-1)*np.log(t)-t-gammaln(shapes)))
            return density*np.exp(-s*t)*expm(t*qa)
        actual = quad_vec(integrand, 0, np.inf, points=[14, 26, 52], epsabs=1e-10)[0]
        tilted_error = max(tilted_error, np.max(np.abs(actual-target)))
        if s == 0:
            check(np.max(np.abs(target-np.ones((3, 3))/3)) < 2e-9)
    check(tilted_error < 2e-9)
    print(f"3 independent total-time Laplace integrals for the smooth erasing clock: max error {tilted_error:.3e}")

    gap = np.linalg.norm(q1f-q0f, np.inf)
    # Grid kernels nearly coincide: high precision avoids subtracting .333... in binary64.
    mp.mp.dps = 50
    def accurate_kernels(times, rotated):
        result = []
        for t in times:
            decay, angle = mp.exp(-4*mp.pi*t), 2*mp.pi*t
            kernel = np.empty((3, 3), dtype=object)
            for row in range(3):
                for col in range(3):
                    ev = mp.mpf(int(row == col))-mp.mpf(1)/3
                    jv = mp.mpf(int(sp.sign(j[row, col])))/mp.sqrt(3)
                    kernel[row, col] = mp.mpf(1)/3+decay*(mp.cos(angle)*ev+mp.sin(angle)*jv if rotated else ev)
            result.append(kernel)
        return result
    max_ratio = 0.
    checks = 0
    for power in range(1, 7):
        eps = mp.sqrt(2)*mp.mpf(10)**(-power)  # irrational by construction, not a numerical test
        kernels0, kernels1 = [accurate_kernels([mp.mpf(1), 1+eps], rotated) for rotated in (False, True)]
        for m in range(1, 5):
            l0 = observed_law(q0f, m, [mp.mpf(1), 1+eps], [mp.mpf('.5')]*2, True, kernels0)
            l1 = observed_law(q1f, m, [mp.mpf(1), 1+eps], [mp.mpf('.5')]*2, True, kernels1)
            tv = .5*sum(abs(l0[key]-l1[key]) for key in l0)
            bound = min(1., m*eps*mp.exp(-4*mp.pi)*mp.pi/mp.sqrt(3))
            check(tv <= bound+2e-13)
            max_ratio = max(max_ratio, tv/bound)
            checks += 1
    print(f"{checks} near-grid full-label-path/total TV bounds (50 digits): max observed TV/budget {float(max_ratio):.9f}; generator gap {gap:.12f}")
    print("PASS")


if __name__ == "__main__":
    main()
