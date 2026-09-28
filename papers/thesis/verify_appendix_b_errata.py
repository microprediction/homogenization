"""Certificate for the Appendix B deterministic-shift pricing errata.

The 2001 thesis omits one deterministic discount factor in its shifted
caplet/floorlet formula, applies one maturity's shift factor to both legs of a
swaplet, and uses the wrong discount-exponent sign in a futures recursion.
This script checks the corrected identities on exact finite scenario trees and
reproduces the constant-rate counterexamples.
"""

from __future__ import annotations

import math

import numpy as np


def deterministic_discount(shifts: np.ndarray) -> float:
    """Return exp(-sum shifts) for unit-length intervals."""
    return math.exp(-float(np.sum(shifts)))


def zero_bonds(
    transition: np.ndarray, rates: np.ndarray, steps: int
) -> np.ndarray:
    """Risk-neutral zero bonds on a finite-state, unit-step short-rate tree."""
    bonds = np.ones(len(rates))
    one_step_discount = np.exp(-rates)
    for _ in range(steps):
        bonds = one_step_discount * (transition @ bonds)
    return bonds


def option_identity(
    transition: np.ndarray,
    rates: np.ndarray,
    shifts: np.ndarray,
    raw_strike: float,
    is_caplet: bool,
) -> tuple[float, float, float]:
    """Direct, corrected-factorized, and printed cap/floor values.

    Dates are t=0, T=1, tau=2 and tau+Delta=3.  The state at t=0 is
    state zero.  The fixing-date payoff is a positive part of two shifted
    zero bonds.  The corrected formula factors A(0,T)A(T,tau)=A(0,tau);
    the printed formula retains only A(T,tau).
    """
    start = 0
    base_discount_to_fixing = math.exp(-rates[start])
    probabilities_at_fixing = transition[start]
    bond_to_tau = zero_bonds(transition, rates, 1)
    bond_to_payment = zero_bonds(transition, rates, 2)

    a_t_T = deterministic_discount(shifts[:1])
    a_T_tau = deterministic_discount(shifts[1:2])
    a_T_payment = deterministic_discount(shifts[1:3])
    adjusted_strike = raw_strike * a_T_payment / a_T_tau

    if is_caplet:
        shifted_payoff = np.maximum(
            a_T_tau * bond_to_tau
            - raw_strike * a_T_payment * bond_to_payment,
            0.0,
        )
        unshifted_generalized_payoff = np.maximum(
            bond_to_tau - adjusted_strike * bond_to_payment, 0.0
        )
    else:
        shifted_payoff = np.maximum(
            raw_strike * a_T_payment * bond_to_payment
            - a_T_tau * bond_to_tau,
            0.0,
        )
        unshifted_generalized_payoff = np.maximum(
            adjusted_strike * bond_to_payment - bond_to_tau, 0.0
        )

    direct = (
        a_t_T
        * base_discount_to_fixing
        * probabilities_at_fixing.dot(shifted_payoff)
    )
    base_generalized_option = (
        base_discount_to_fixing
        * probabilities_at_fixing.dot(unshifted_generalized_payoff)
    )
    corrected = a_t_T * a_T_tau * base_generalized_option
    printed = a_T_tau * base_generalized_option
    return direct, corrected, printed


def swaplet_identity(
    transition: np.ndarray,
    rates: np.ndarray,
    shifts: np.ndarray,
    raw_strike: float,
) -> tuple[float, float]:
    """Correct and printed shifted swaplet values at t=0."""
    start = 0
    first_step = math.exp(-rates[start])
    base_to_tau = (
        first_step
        * transition[start].dot(zero_bonds(transition, rates, 1))
    )
    base_to_payment = (
        first_step
        * transition[start].dot(zero_bonds(transition, rates, 2))
    )
    a_t_tau = deterministic_discount(shifts[:2])
    a_t_payment = deterministic_discount(shifts[:3])
    correct = (
        a_t_tau * base_to_tau
        - raw_strike * a_t_payment * base_to_payment
    )
    printed = a_t_tau * (
        base_to_tau - raw_strike * base_to_payment
    )
    return correct, printed


def forward_measure_identity(
    transition: np.ndarray,
    next_rates: np.ndarray,
    shift: float,
    payoff: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explicit forward expectation, negative-sign formula, printed formula."""
    discount = np.exp(-(next_rates + shift))
    bond = transition @ discount
    forward_weights = transition * discount[np.newaxis, :] / bond[:, None]
    explicit = forward_weights @ payoff
    risk_neutral = (transition @ (discount * payoff)) / bond
    printed = (
        transition @ (np.exp(next_rates + shift) * payoff)
    ) / bond
    return explicit, risk_neutral, printed


def main() -> None:
    rng = np.random.default_rng(20260928)
    largest_option_error = 0.0
    largest_forward_error = 0.0
    largest_printed_ratio_error = 0.0
    positive_option_cases = 0

    for _ in range(500):
        transition = rng.dirichlet(np.ones(3), size=3)
        rates = rng.uniform(-0.01, 0.08, size=3)
        shifts = rng.uniform(-0.005, 0.04, size=3)
        raw_strike = rng.uniform(0.92, 1.12)

        for is_caplet in (True, False):
            direct, corrected, printed = option_identity(
                transition,
                rates,
                shifts,
                raw_strike,
                is_caplet,
            )
            largest_option_error = max(
                largest_option_error, abs(direct - corrected)
            )
            if direct > 1e-10:
                expected_ratio = math.exp(shifts[0])
                largest_printed_ratio_error = max(
                    largest_printed_ratio_error,
                    abs(printed / direct - expected_ratio),
                )
                positive_option_cases += 1

        payoff = rng.uniform(0.2, 1.8, size=3)
        explicit, corrected_forward, _ = forward_measure_identity(
            transition,
            rates,
            shifts[0],
            payoff,
        )
        largest_forward_error = max(
            largest_forward_error,
            float(np.max(np.abs(explicit - corrected_forward))),
        )

    assert largest_option_error < 3e-15
    assert largest_forward_error < 3e-15
    assert positive_option_cases > 100
    assert largest_printed_ratio_error < 5e-13

    # The constant-rate example from Issue #69: t=0, T=tau=1,
    # tau+Delta=2, K=0, beta X=0, and deterministic shift 5%.
    shift = 0.05
    a_01 = math.exp(-shift)
    a_02 = math.exp(-2.0 * shift)
    caplet_correct = a_01 - a_02
    caplet_printed = 1.0 - a_01
    swaplet_correct = a_01 - a_02
    swaplet_printed = a_01 * (1.0 - 1.0)
    assert abs(caplet_correct - 0.0463920064647545) < 2e-16
    assert abs(caplet_printed - 0.0487705754992860) < 2e-16
    assert swaplet_printed == 0.0

    # Use an in-the-money floorlet to show that the same omitted A(0,T)
    # affects the reverse positive part.
    raw_strike = 1.1
    floorlet_correct = raw_strike * a_02 - a_01
    floorlet_printed = raw_strike * a_01 - 1.0
    assert floorlet_correct > 0.0
    assert abs(floorlet_printed / floorlet_correct - math.exp(shift)) < 2e-14

    # A nontrivial finite-state check of the futures recursion.  The negative
    # discount exponent is exactly the forward-measure change of numeraire.
    transition = np.array([[0.82, 0.18], [0.27, 0.73]])
    next_rates = np.array([0.02, 0.06])
    payoff = np.array([0.7, 1.4])
    explicit, corrected_forward, printed_forward = forward_measure_identity(
        transition, next_rates, 0.025, payoff
    )
    forward_error = float(np.max(np.abs(explicit - corrected_forward)))
    printed_forward_error = float(
        np.max(np.abs(explicit - printed_forward))
    )
    assert forward_error < 3e-16
    assert printed_forward_error > 0.1

    # With a constant payoff, the corrected recursion preserves the constant.
    # The printed positive exponent does not.
    explicit_one, corrected_one, printed_one = forward_measure_identity(
        transition, next_rates, 0.025, np.ones(2)
    )
    assert np.max(np.abs(explicit_one - 1.0)) < 3e-16
    assert np.max(np.abs(corrected_one - 1.0)) < 3e-16
    assert np.min(printed_one - 1.0) > 0.08

    print("Appendix B deterministic-shift pricing errata")
    print(f"maximum caplet/floorlet identity error: {largest_option_error:.3e}")
    print(
        f"printed-factor ratio error over {positive_option_cases} positive "
        f"cases: {largest_printed_ratio_error:.3e}"
    )
    print(f"maximum forward-measure identity error: {largest_forward_error:.3e}")
    print(
        "constant-rate caplet: "
        f"correct {caplet_correct:.10f}, printed {caplet_printed:.10f}"
    )
    print(
        "constant-rate floorlet: "
        f"correct {floorlet_correct:.10f}, printed {floorlet_printed:.10f}"
    )
    print(
        "constant-rate swaplet: "
        f"correct {swaplet_correct:.10f}, printed {swaplet_printed:.10f}"
    )
    print(
        "nontrivial futures recursion: "
        f"correct error {forward_error:.3e}, "
        f"printed error {printed_forward_error:.6f}"
    )
    print(
        "PASS: maturity-specific shift factors and negative-exponent "
        "change of numeraire"
    )


if __name__ == "__main__":
    main()
