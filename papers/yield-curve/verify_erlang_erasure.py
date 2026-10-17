"""A fixed smooth Coxian clock erases a continuum of three-state generators."""
import math

import numpy as np
import sympy as sp
from scipy.integrate import quad_vec
from scipy.linalg import expm


def check(condition):
    if not condition:
        raise AssertionError("Erlang-erasure certificate failed")


def zero(matrix):
    check(all(sp.simplify(x) == 0 for x in matrix))


def main():
    identity, projector = sp.eye(3), sp.ones(3)/3
    centered = identity-projector
    cosine = (sp.sqrt(6)+sp.sqrt(2))/4
    sine = (sp.sqrt(6)-sp.sqrt(2))/4
    a, b = 2*cosine-1, 2*sine
    forward, backward = a/3+b/sp.sqrt(3), a/3-b/sp.sqrt(3)
    q = sp.Matrix([[-forward-backward, forward, backward],
                   [backward, -forward-backward, forward],
                   [forward, backward, -forward-backward]])
    j = sp.simplify((q+a*centered)/b)
    zero(j.T+j)
    zero(j*j+centered)
    c = projector+cosine*centered/2+sine*j/2
    zero((identity-q)*c-identity)
    # Complex multiplication in span{I,J}, with J^2=-I on the centered space.
    real, imag = sp.Integer(1), sp.Integer(0)
    for _ in range(12):
        real, imag = (sp.simplify((real*cosine-imag*sine)/2),
                      sp.simplify((imag*cosine+real*sine)/2))
    check(real == -sp.Rational(1, 4096) and imag == 0)
    c12 = projector+real*centered+imag*j
    zero((c+4096*c*c12)/4097-projector)
    weight = sp.Rational(1, 4097)
    phases = -sp.eye(13)
    for k in range(12):
        phases[k, k+1] = 1-weight if k == 0 else 1
    start = sp.zeros(1, 13)
    start[0] = 1
    exits = -phases*sp.ones(13, 1)
    mean = (start*(-phases).inv()*sp.ones(13, 1))[0]
    second = (2*start*(-phases).inv()**2*sp.ones(13, 1))[0]
    check(mean == sp.Rational(53249, 4097))
    check(second == sp.Rational(745474, 4097))
    lower = sp.simplify(backward-b/sp.sqrt(3)*(sp.Rational(1, 100)+sp.Rational(1, 99)))
    check(lower > 0)  # Uniform bound for every |epsilon|<=1/100.
    h = sp.Matrix([1, 0, -1])
    projection = h*h.T/2
    zero(projection*j*projection)
    g = sp.Matrix([1, -2, 1])
    zero(projection*j+h*g.T/(2*sp.sqrt(3)))
    zero(j*projection-g*h.T/(2*sp.sqrt(3)))
    eps = sp.Symbol("epsilon", real=True)
    sf, si = identity+eps*projection, identity-eps/(1+eps)*projection
    f = sp.Matrix([1, -1, 0])
    exact_strength = (2*f.T*sf*(a*centered+b*j)*si*f)[0]/(3*(a*a+b*b))
    formula = (4*a+sp.sqrt(3)*b*(1/(1+eps)-(1+eps)))/(3*(a*a+b*b))
    check(sp.simplify(exact_strength-formula) == 0)
    feature = np.array([1.0, -1.0, 0.0])
    q0, p = np.array(q).astype(float), np.ones((3, 3))/3
    hp = np.array(projection).astype(float)
    t = np.array(phases).astype(float)
    alpha, exit_float = np.array(start).astype(float), np.array(exits).astype(float)
    erasure_error = phase_error = variance_error = perturbation_excess = 0.0
    values, kernels, generators = {}, {}, {}
    cases = np.linspace(-0.01, 0.01, 21)
    for epsilon in cases:
        s = np.eye(3)+epsilon*hp
        qe = s@q0@np.linalg.inv(s)
        check(np.min(qe[~np.eye(3, dtype=bool)]) >= float(lower)-2e-9)
        check(np.max(np.abs(qe.sum(axis=1))) < 2e-9)
        check(np.max(np.abs(qe.sum(axis=0))) < 2e-9)
        ce = np.linalg.inv(np.eye(3)-qe)
        r = (ce+4096*np.linalg.matrix_power(ce, 13))/4097
        erasure_error = max(erasure_error, np.max(np.abs(r-p)))
        augmented = np.kron(t, np.eye(3))+np.kron(np.eye(13), qe)
        other = -np.kron(alpha, np.eye(3))@np.linalg.solve(augmented, np.kron(exit_float, np.eye(3)))
        phase_error = max(phase_error, np.max(np.abs(other-p)))
        potential = np.linalg.inv(p-qe)-p
        strength = 2*feature@potential@feature/3
        analytic = float((4*a+sp.sqrt(3)*b*(1/(1+epsilon)-(1+epsilon)))/(3*(a*a+b*b)))
        variance_error = max(variance_error, abs(strength-analytic))
        for eta in (-1e-5, 1e-5, 1e-3):
            rw = (float(weight)+eta)*ce+(1-float(weight)-eta)*np.linalg.matrix_power(ce, 13)
            perturbation_excess = max(perturbation_excess, np.linalg.norm(rw-p, np.inf)-2*abs(eta))
        values[round(epsilon, 3)], kernels[round(epsilon, 3)] = strength, ce
        generators[round(epsilon, 3)] = qe
    integral_error = 0.0
    for epsilon in (-0.01, 0.0, 0.01):
        def density(x):
            return math.exp(-x)*(1+4096*x**12/math.factorial(12))/4097
        integral = quad_vec(lambda x: density(x)*expm(x*generators[epsilon]),
                            0, np.inf, epsabs=1e-10, epsrel=1e-10)[0]
        integral_error = max(integral_error, np.max(np.abs(integral-p)))
    check(max(erasure_error, phase_error, variance_error, integral_error, perturbation_excess) < 2e-9)
    check(values[-0.01] > values[0.0] > values[0.01])
    tagged_gap = np.linalg.norm(kernels[0.01]-kernels[0.0], np.inf)
    check(tagged_gap > 1e-4)
    print("Exact centered rotation/resolvent cancellation: C^12=P-(I-P)/4096; R=P")
    print(f"Clock Erlang weights: {weight}, {1-weight}; mean {mean}; second moment {second}")
    print(f"Base forward/backward rates: {float(forward):.12f}, {float(backward):.12f}")
    print(f"All |epsilon|<=.01: off-diagonal lower bound {float(lower):.12f}")
    print(f"21 kernels; max erasure error {erasure_error:.3e}; phase-chain error {phase_error:.3e}")
    print(f"3 independent semigroup integrals: max error {integral_error:.3e}")
    print(f"63 weight perturbations: max bound excess {perturbation_excess:.3e}")
    print(f"Green-Kubo strengths at epsilon=-.01,0,.01: {values[-.01]:.12f}, {values[0.0]:.12f}, {values[.01]:.12f}")
    print(f"Strength formula max error {variance_error:.3e}; tagged-kernel gap {tagged_gap:.12f}")
    print("PASS")


if __name__ == "__main__":
    main()
