"""Certificate for the null-space amplitudes in the solvability hierarchy.

The page uses the two-state system

    d_t u = eps^{-1} Q u + G u,
    Q = [[-1, 1], [1, -1]],  G = diag(1, 0).

It checks the reported obstruction to centering the *whole* first
coefficient, verifies the corrected average/shape recursion, and compares
the resulting first-order outer approximation with the exact matrix
exponential.  It also checks an exact slow/fast modal split and the uniform
first-order composite obtained by restoring the leading initial layer.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm


Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
G = np.diag([1.0, 0.0])
ONE = np.ones(2)
SHAPE = np.array([1.0, -1.0])
PI = np.array([0.5, 0.5])
P = np.outer(ONE, PI)
QSHARP = np.linalg.inv(Q - P) + P


def average(v: np.ndarray) -> float:
    return float(PI @ v)


def coefficients(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return u_0, u_1 and the matched u_2 for initial data u(0)=1."""
    a0 = np.exp(t / 2.0)
    chi1 = (a0 / 4.0) * SHAPE
    a1 = (t * a0 / 8.0)
    chi2 = (t * a0 / 32.0) * SHAPE

    # The -1/16 is fixed by matching the exponentially decaying initial
    # layer.  It is not needed for the first-order result, but lets the
    # certificate check one order further.
    a2 = a0 * (t * t / 128.0 - 1.0 / 16.0)
    return a0 * ONE, a1 * ONE + chi1, a2 * ONE + chi2


def exact(t: float, eps: float) -> np.ndarray:
    return expm((Q / eps + G) * t) @ ONE


def exact_modal(t: float, eps: float) -> np.ndarray:
    """Exact slow/fast decomposition in mean and centered coordinates."""
    d = np.sqrt(1.0 + eps**2 / 4.0)
    slow = 0.5 + (d - 1.0) / eps
    fast = 0.5 - (d + 1.0) / eps
    mean = ((d + 1.0) * np.exp(slow * t) + (d - 1.0) * np.exp(fast * t)) / (2.0 * d)
    shape = eps * (np.exp(slow * t) - np.exp(fast * t)) / (4.0 * d)
    return mean * ONE + shape * SHAPE


def first_outer(t: float, eps: float) -> np.ndarray:
    """First-order outer approximation for initial data u(0)=1."""
    a0 = np.exp(t / 2.0)
    return a0 * ONE + eps * (t * a0 * ONE / 8.0 + a0 * SHAPE / 4.0)


def first_composite(t: float, eps: float) -> np.ndarray:
    """First outer approximation plus the leading fast initial layer."""
    return first_outer(t, eps) - eps * np.exp(-2.0 * t / eps) * SHAPE / 4.0


def uniform_constant(t_max: float) -> float:
    """Explicit coefficient in the proved uniform O(eps^2) bound."""
    return (
        np.exp(5.0 * t_max / 8.0)
        * (3.0 / 32.0 + t_max / 32.0 + t_max**2 / 128.0)
        + t_max * np.exp(t_max / 2.0) / 128.0
        + 17.0 / 96.0
    )


def observed_order(errors: list[float], epsilons: np.ndarray) -> float:
    return float(np.polyfit(np.log(epsilons[-4:]), np.log(errors[-4:]), 1)[0])


def stationary(q: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eig(q.T)
    pi = np.real(vectors[:, np.argmin(np.abs(values))])
    return pi / pi.sum()


def group_inverse(q: np.ndarray, pi: np.ndarray) -> np.ndarray:
    p = np.outer(np.ones(len(pi)), pi)
    return np.linalg.inv(q - p) + p


def main() -> None:
    ident = np.eye(2)
    assert np.max(np.abs(Q @ QSHARP - (ident - P))) < 2e-15
    assert np.max(np.abs(QSHARP @ Q - (ident - P))) < 2e-15
    assert np.max(np.abs(PI @ QSHARP)) < 2e-15

    t = 1.25
    a0 = np.exp(t / 2.0)
    a0_prime = a0 / 2.0
    chi1 = (a0 / 4.0) * SHAPE
    chi1_prime = (a0 / 8.0) * SHAPE

    rhs1 = a0_prime * ONE - G @ (a0 * ONE)
    assert abs(average(rhs1)) < 2e-15
    assert np.max(np.abs(Q @ chi1 - rhs1)) < 2e-15
    assert np.max(np.abs(QSHARP @ rhs1 - chi1)) < 2e-15

    # Centering the whole u_1 means u_1=chi_1.  Its next right-hand side
    # is not solvable: its stationary mean is exactly -a_0/8.
    centered_rhs2 = chi1_prime - G @ chi1
    centered_obstruction = average(centered_rhs2)
    assert abs(centered_obstruction + a0 / 8.0) < 2e-15

    # Retaining u_1=a_1 1+chi_1 and imposing the next solvability
    # condition gives a_1' - a_1/2 = a_0/8.
    a1 = t * a0 / 8.0
    a1_prime = a0 / 8.0 + a1 / 2.0
    u1 = a1 * ONE + chi1
    u1_prime = a1_prime * ONE + chi1_prime
    rhs2 = u1_prime - G @ u1
    assert abs(average(rhs2)) < 2e-15
    chi2 = (t * a0 / 32.0) * SHAPE
    assert np.max(np.abs(Q @ chi2 - rhs2)) < 2e-15

    epsilons = np.array([0.20, 0.14, 0.10, 0.07, 0.05, 0.035, 0.025, 0.018])
    u0, u1_coeff, u2_coeff = coefficients(t)
    centered_errors: list[float] = []
    first_errors: list[float] = []
    second_errors: list[float] = []
    for eps in epsilons:
        target = exact(t, eps)
        centered_errors.append(float(np.linalg.norm(target - (u0 + eps * chi1), np.inf)))
        first_errors.append(float(np.linalg.norm(target - (u0 + eps * u1_coeff), np.inf)))
        second_errors.append(float(np.linalg.norm(target - (u0 + eps * u1_coeff + eps**2 * u2_coeff), np.inf)))

    centered_order = observed_order(centered_errors, epsilons)
    first_order = observed_order(first_errors, epsilons)
    second_order = observed_order(second_errors, epsilons)
    assert 0.96 < centered_order < 1.04
    assert 1.94 < first_order < 2.06
    assert 2.88 < second_order < 3.12

    # The outer approximation alone cannot be uniform at t=0: its initial
    # centered mismatch is exactly eps/4.  Adding the leading layer cancels
    # that mismatch and gives a uniform O(eps^2) approximation on [0,T].
    t_max = 2.0
    times = np.linspace(0.0, t_max, 2001)
    modal_error = max(
        float(np.linalg.norm(exact_modal(s, 0.137) - exact(s, 0.137), np.inf))
        for s in times
    )
    assert modal_error < 3e-14

    uniform_outer_errors: list[float] = []
    uniform_composite_errors: list[float] = []
    bound_ratios: list[float] = []
    c_t = uniform_constant(t_max)
    for eps in epsilons:
        outer_error = max(
            float(np.linalg.norm(exact_modal(s, eps) - first_outer(s, eps), np.inf))
            for s in times
        )
        composite_error = max(
            float(np.linalg.norm(exact_modal(s, eps) - first_composite(s, eps), np.inf))
            for s in times
        )
        uniform_outer_errors.append(outer_error)
        uniform_composite_errors.append(composite_error)
        bound_ratios.append(composite_error / (c_t * eps**2))
        assert abs(outer_error - eps / 4.0) < 3e-14
        assert composite_error <= c_t * eps**2

    uniform_outer_order = observed_order(uniform_outer_errors, epsilons)
    uniform_composite_order = observed_order(uniform_composite_errors, epsilons)
    assert 0.99 < uniform_outer_order < 1.01
    assert 1.98 < uniform_composite_order < 2.03

    # A nonreversible three-state check of the general constant-forcing
    # corollary.  For A_eps=Q/eps+diag(g), the first slow-eigenvalue
    # correction and the null-space amplitude have coefficient
    # K=-pi.(g~ Q# g~).
    q3 = np.array([[-2.1, 2.0, 0.1], [0.1, -2.1, 2.0], [2.0, 0.1, -2.1]])
    g3 = np.array([1.1, -0.4, 0.6])
    pi3 = stationary(q3)
    qs3 = group_inverse(q3, pi3)
    centered3 = g3 - pi3 @ g3
    k3 = -float(pi3 @ (centered3 * (qs3 @ centered3)))
    a3 = np.exp(float(pi3 @ g3) * t)
    u03 = a3 * np.ones(3)
    chi13 = -a3 * (qs3 @ centered3)
    u13 = t * k3 * a3 * np.ones(3) + chi13
    centered3_errors: list[float] = []
    full3_errors: list[float] = []
    eigen3_errors: list[float] = []
    for eps in epsilons:
        generator = q3 / eps + np.diag(g3)
        target = expm(generator * t) @ np.ones(3)
        centered3_errors.append(float(np.linalg.norm(target - (u03 + eps * chi13), np.inf)))
        full3_errors.append(float(np.linalg.norm(target - (u03 + eps * u13), np.inf)))
        eigenvalues = np.linalg.eigvals(generator)
        slow_eigenvalue = eigenvalues[np.argmax(np.real(eigenvalues))]
        eigen3_errors.append(abs(float(np.real(slow_eigenvalue) - pi3 @ g3) / eps - k3))

    centered3_order = observed_order(centered3_errors, epsilons)
    full3_order = observed_order(full3_errors, epsilons)
    eigen3_order = observed_order(eigen3_errors, epsilons)
    assert 0.96 < centered3_order < 1.04
    assert 1.94 < full3_order < 2.08
    assert 0.96 < eigen3_order < 1.04

    print("null-space solvability hierarchy certificate")
    print(f"centered-only next-order obstruction  {centered_obstruction:.12e}")
    print(f"exact obstruction -a0/8            {-a0 / 8.0:.12e}")
    print(f"corrected next stationary residual {average(rhs2):.3e}")
    print(f"centered-only approximation order  {centered_order:.6f}")
    print(f"full first-corrector order          {first_order:.6f}")
    print(f"matched second-corrector order      {second_order:.6f}")
    print(f"smallest-eps centered error         {centered_errors[-1]:.12e}")
    print(f"smallest-eps first-order error      {first_errors[-1]:.12e}")
    print(f"smallest-eps second-order error     {second_errors[-1]:.12e}")
    print(f"modal formula maximum error         {modal_error:.3e}")
    print(f"uniform outer order on [0,2]        {uniform_outer_order:.6f}")
    print(f"uniform composite order on [0,2]    {uniform_composite_order:.6f}")
    print(f"proved uniform coefficient C_2      {c_t:.12e}")
    print(f"largest error / proved bound        {max(bound_ratios):.6f}")
    print(f"three-state Green-Kubo coefficient  {k3:.12e}")
    print(f"three-state centered-only order     {centered3_order:.6f}")
    print(f"three-state full first order        {full3_order:.6f}")
    print(f"three-state eigen-coefficient order {eigen3_order:.6f}")
    print("ok")


if __name__ == "__main__":
    main()
