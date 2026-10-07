"""Certificate for the null-space amplitudes in the solvability hierarchy.

The page uses the two-state system

    d_t u = eps^{-1} Q u + G u,
    Q = [[-1, 1], [1, -1]],  G = diag(1, 0).

It checks the reported obstruction to centering the *whole* first
coefficient, verifies the corrected average/shape recursion, and compares
the resulting first-order outer approximation with the exact matrix
exponential.  The exact comparison starts away from the initial layer.
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


def observed_order(errors: list[float], epsilons: np.ndarray) -> float:
    return float(np.polyfit(np.log(epsilons[-4:]), np.log(errors[-4:]), 1)[0])


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
    print("ok")


if __name__ == "__main__":
    main()
