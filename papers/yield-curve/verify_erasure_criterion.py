"""Positive-polynomial clocks: exact finite-chain erasure, including a Jordan pair."""
import math
from fractions import Fraction

import numpy as np
import sympy as sp
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.special import gammaln, logsumexp


def check(condition):
    if not condition:
        raise AssertionError("erasure-criterion certificate failed")


def cycle(n):
    q = -sp.eye(n)
    for j in range(n):
        q[j, (j+1) % n] = 1
    return q


def defective():
    basis = sp.ones(5, 1).row_join(sp.eye(5)[:, :4]
                                - sp.Matrix([0, 0, 0, 0, 1])*sp.ones(1, 4))
    rotation = sp.Matrix([[0, -1, 0, 0], [1, 0, 0, 0],
                          [0, 0, 0, -1], [0, 0, 1, 0]])
    nilpotent = sp.zeros(4)
    nilpotent[:2, 2:] = sp.eye(2)
    block = sp.zeros(5)
    block[1:, 1:] = -sp.eye(4)+sp.Rational(9, 50)*rotation+nilpotent/100
    return basis*block*basis.inv()


def positive_multiplier(coefficients):
    coefficients = list(coefficients)
    for power in range(10001):
        if min(coefficients) > 0:
            return power, coefficients
        coefficients = [coefficients[0]] + [a+b for a, b in zip(coefficients, coefficients[1:])] + [coefficients[-1]]
    raise AssertionError("fixed example exceeded the certificate search budget")


def polynomial_at(coefficients, matrix, centered):
    result = sp.zeros(matrix.rows)
    for coefficient in reversed(coefficients):
        result = result*matrix+coefficient*centered
    return result


def main():
    x = sp.Symbol("x")
    max_error = integral_error = 0.0
    total_weights = 0
    cases = [("three-cycle", cycle(3), sp.Integer(1)),
             ("five-cycle", cycle(5), sp.Integer(1)),
             ("defective-five", defective(), sp.Rational(1, 10))]
    for name, q, rate in cases:
        n = q.rows
        p = sp.ones(n)/n
        e = sp.eye(n)-p
        check(q*sp.ones(n, 1) == sp.zeros(n, 1))
        check(sp.ones(1, n)*q == sp.zeros(1, n))
        check(min(q[i, j] for i in range(n) for j in range(n) if i != j) >= 0)
        c = rate*(rate*sp.eye(n)-q).inv()
        quotient, remainder = sp.div(c.charpoly(x).as_expr(), x-1, x)
        check(remainder == 0)
        poly = sp.Poly(quotient, x).clear_denoms()[1]
        coefficients = [int(a) for a in reversed(poly.all_coeffs())]
        check(polynomial_at(coefficients, c, e) == sp.zeros(n))
        if name == "defective-five":
            quadratic = sp.Poly(3106*x*x-550*x+25, x)
            check(sp.rem(poly, quadratic**2) == 0)
            check(polynomial_at([25, -550, 3106], c, e) != sp.zeros(n))
            check(min(q[i, j] for i in range(n) for j in range(n) if i != j) > 0)
            check(sp.expand(q.charpoly(x).as_expr()/x)
                  == sp.expand(((x+1)**2+sp.Rational(9, 50)**2)**2))
        power, positive = positive_multiplier(coefficients)
        # Exact coefficient sum and divisibility certify the mixture, no eigentolerance.
        check(sum(positive) == 2**power*sum(coefficients))
        check(min(positive) > 0)
        weights = np.array([float(Fraction(a, sum(positive))) for a in positive])
        total_weights += len(weights)
        cf = np.array(c).astype(float)
        kernel, current = np.zeros((n, n)), cf.copy()
        for weight in weights:
            kernel += weight*current
            current = current@cf
        max_error = max(max_error, np.max(np.abs(kernel-np.ones((n, n))/n)))
        print(f"{name}: M(x)={poly.as_expr()}; N={power}; Erlang shapes 1..{len(weights)}")
        print(f"  all {len(weights)} coefficients positive exactly; rate={rate}; minimum off-diagonal={min(q[i,j] for i in range(n) for j in range(n) if i!=j)}")
        if name == "defective-five":
            print("  eigenvalues -1 +/- 9i/50, each with a nontrivial Jordan block")
        if name != "defective-five":
            shapes = np.arange(1, len(weights)+1)
            logs = np.array([math.log(a)-math.log(sum(positive)) for a in positive])
            rf, qf = float(rate), np.array(q).astype(float)
            mean = float(np.dot(weights, shapes)/rf)
            def density(t):
                if t == 0:
                    return rf*weights[0]
                return np.exp(logsumexp(logs+np.log(rf)+(shapes-1)*np.log(rf*t)-rf*t-gammaln(shapes)))
            integral = quad_vec(lambda t: density(t)*expm(t*qf),
                                0, np.inf, points=[mean/2, mean, 2*mean],
                                epsabs=1e-10, epsrel=1e-10)[0]
            integral_error = max(integral_error, np.max(np.abs(integral-np.ones((n, n))/n)))
    check(max(max_error, integral_error) < 2e-8)
    for n in (2, 4, 6):
        alternating = sp.Matrix([(-1)**j for j in range(n)])
        check(cycle(n)*alternating == -2*alternating)
        check(sum(alternating) == 0)
    reset = sp.ones(5)/5-sp.eye(5)
    contrast = sp.Matrix([1, -1, 0, 0, 0])
    check(reset*contrast == -contrast)
    print(f"Total exactly positive clock weights: {total_weights}; maximum kernel error {max_error:.3e}")
    print(f"Two independent semigroup integrals: maximum error {integral_error:.3e}")
    print("3 even-cycle and 1 reversible-reset exact real-eigenvector obstructions checked")
    print("PASS")


if __name__ == "__main__":
    main()
