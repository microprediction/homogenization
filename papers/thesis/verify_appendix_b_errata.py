"""Certificate for the Appendix B deterministic-shift pricing errata.

The 2001 thesis omits one deterministic discount factor in its shifted
caplet/floorlet formula, applies one maturity's shift factor to both legs of a
swaplet, and uses the wrong discount-exponent sign in a futures recursion.
This script checks the corrected identities on exact finite scenario trees and
reproduces the constant-rate counterexamples.  It also checks the master
transport identity for an arbitrary nonlinear payoff on several bond
maturities, which shows why different payment dates cannot share one shift
factor.  Finally, it checks that deterministic shifts leave every bond-forward
measure unchanged and therefore preserve Black implied volatility after the
corresponding deterministic rescaling of forward and strike.  It also checks
the sharp limitation: a positive bond-basket (in particular annuity) measure
is shift-invariant only when all of its maturity factors coincide.
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


def forward_state_distribution(
    transition: np.ndarray,
    rates: np.ndarray,
    shifts: np.ndarray,
    horizon: int,
    split: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Base and shifted U-forward state laws at an intermediate date.

    The terminal numeraire date is ``horizon`` and the distribution is
    evaluated at ``split``.  The deterministic shift may vary by interval.
    Prefix, suffix and total shift factors are deliberately retained here so
    the certificate tests their exact cancellation rather than simplifying it
    in advance.
    """
    if not 0 <= split <= horizon <= len(shifts):
        raise ValueError("require 0 <= split <= horizon <= len(shifts)")
    kernel = np.diag(np.exp(-rates)) @ transition
    initial = np.zeros(len(rates))
    initial[0] = 1.0
    prefix = initial @ np.linalg.matrix_power(kernel, split)
    suffix = np.linalg.matrix_power(kernel, horizon - split) @ np.ones(
        len(rates)
    )
    terminal_bond = float(
        initial @ np.linalg.matrix_power(kernel, horizon) @ np.ones(len(rates))
    )
    base = prefix * suffix / terminal_bond

    prefix_shift = math.exp(-float(np.sum(shifts[:split])))
    suffix_shift = math.exp(-float(np.sum(shifts[split:horizon])))
    total_shift = math.exp(-float(np.sum(shifts[:horizon])))
    shifted = (
        prefix_shift * prefix * suffix_shift * suffix
        / (total_shift * terminal_bond)
    )
    return base, shifted


def basket_measure_state_distribution(
    transition: np.ndarray,
    rates: np.ndarray,
    shifts: np.ndarray,
    split: int,
    maturities: tuple[int, ...],
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Base and shifted state laws under a positive bond-basket numeraire."""
    if split < 0 or any(maturity < split for maturity in maturities):
        raise ValueError("payment maturities must not precede the split")
    if max(maturities) > len(shifts) or np.min(weights) <= 0.0:
        raise ValueError("need enough shifts and strictly positive weights")

    kernel = np.diag(np.exp(-rates)) @ transition
    initial = np.zeros(len(rates))
    initial[0] = 1.0
    prefix = initial @ np.linalg.matrix_power(kernel, split)
    bonds = np.column_stack(
        [
            np.linalg.matrix_power(kernel, maturity - split)
            @ np.ones(len(rates))
            for maturity in maturities
        ]
    )
    base_numeraire = bonds @ weights
    base = prefix * base_numeraire / float(prefix @ base_numeraire)

    maturity_factors = np.array(
        [
            deterministic_discount(shifts[split:maturity])
            for maturity in maturities
        ]
    )
    shifted_numeraire = bonds @ (weights * maturity_factors)
    shifted = prefix * shifted_numeraire / float(prefix @ shifted_numeraire)
    return base, shifted, maturity_factors


def normal_cdf(value: float) -> float:
    """Standard normal distribution function."""
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def black_call(forward: float, strike: float, volatility: float) -> float:
    """Unit-expiry undiscounted Black call."""
    if volatility <= 0.0:
        return max(forward - strike, 0.0)
    d1 = math.log(forward / strike) / volatility + 0.5 * volatility
    d2 = d1 - volatility
    return forward * normal_cdf(d1) - strike * normal_cdf(d2)


def black_implied_vol(forward: float, strike: float, price: float) -> float:
    """Invert the unit-expiry Black call by monotone bisection."""
    intrinsic = max(forward - strike, 0.0)
    if price < intrinsic - 2e-14 or price > forward + 2e-14:
        raise ValueError("price violates Black bounds")
    if price <= intrinsic + 1e-15:
        return 0.0
    low, high = 0.0, 8.0
    for _ in range(90):
        middle = 0.5 * (low + high)
        if black_call(forward, strike, middle) < price:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def caplet_forward_invariance(
    transition: np.ndarray,
    rates: np.ndarray,
    shifts: np.ndarray,
) -> tuple[float, float, float]:
    """Forward-law, normalized-price and implied-vol shift invariance.

    Dates are t=0, fixing T=1, first bond maturity tau=2 and payment
    maturity U=3.  The contractual strike is chosen at the shifted forward,
    ensuring a nondegenerate implied volatility in every random case.
    """
    start = 0
    bond_to_tau = zero_bonds(transition, rates, 1)
    bond_to_payment = zero_bonds(transition, rates, 2)
    discount_to_fixing = math.exp(-rates[start])
    payment_bond = discount_to_fixing * transition[start].dot(
        bond_to_payment
    )
    forward_weights = (
        discount_to_fixing
        * transition[start]
        * bond_to_payment
        / payment_bond
    )
    base_ratio = bond_to_tau / bond_to_payment
    base_forward = float(forward_weights.dot(base_ratio))

    a_T_tau = deterministic_discount(shifts[1:2])
    a_T_payment = deterministic_discount(shifts[1:3])
    ratio_scale = a_T_tau / a_T_payment
    shifted_ratio = ratio_scale * base_ratio
    shifted_forward = ratio_scale * base_forward
    shifted_strike = shifted_forward
    base_strike = shifted_strike / ratio_scale

    base_normalized_call = float(
        forward_weights.dot(np.maximum(base_ratio - base_strike, 0.0))
    )
    shifted_normalized_call = float(
        forward_weights.dot(
            np.maximum(shifted_ratio - shifted_strike, 0.0)
        )
    )
    price_error = abs(
        shifted_normalized_call - ratio_scale * base_normalized_call
    )
    base_iv = black_implied_vol(
        base_forward, base_strike, base_normalized_call
    )
    shifted_iv = black_implied_vol(
        shifted_forward, shifted_strike, shifted_normalized_call
    )
    return (
        float(np.max(np.abs(np.sum(forward_weights) - 1.0))),
        price_error,
        abs(base_iv - shifted_iv),
    )


def shifted_zero_bonds(
    transition: np.ndarray,
    rates: np.ndarray,
    future_shifts: np.ndarray,
) -> np.ndarray:
    """Zero bonds with one deterministic shift for each future interval."""
    bonds = np.ones(len(rates))
    for shift in future_shifts[::-1]:
        bonds = np.exp(-(rates + shift)) * (transition @ bonds)
    return bonds


def multibond_transport_identity(
    transition: np.ndarray,
    rates: np.ndarray,
    shifts: np.ndarray,
) -> tuple[float, float, float]:
    """Direct and transported prices for a nonlinear three-bond payoff.

    Dates are t=0 and T=1, and the underlying bonds mature one, two and
    three intervals after T.  The direct calculation builds each shifted
    bond independently.  The transported calculation rescales the three
    unshifted bonds by their maturity-specific deterministic factors before
    applying the same nonlinear payoff.
    """
    start = 0
    base_bonds = np.column_stack(
        [zero_bonds(transition, rates, steps) for steps in (1, 2, 3)]
    )
    direct_shifted_bonds = np.column_stack(
        [
            shifted_zero_bonds(
                transition, rates, shifts[1 : 1 + steps]
            )
            for steps in (1, 2, 3)
        ]
    )
    maturity_factors = np.exp(-np.cumsum(shifts[1:4]))
    transported_bonds = base_bonds * maturity_factors[np.newaxis, :]

    weights = np.array([0.80, -0.25, 0.15])
    strike = 0.55

    def payoff(bonds: np.ndarray) -> np.ndarray:
        return np.maximum(bonds @ weights - strike, 0.0)

    shifted_discount_to_fixing = math.exp(-(rates[start] + shifts[0]))
    direct = shifted_discount_to_fixing * transition[start].dot(
        payoff(direct_shifted_bonds)
    )
    transported = (
        math.exp(-shifts[0])
        * math.exp(-rates[start])
        * transition[start].dot(payoff(transported_bonds))
    )
    bond_error = float(
        np.max(np.abs(direct_shifted_bonds - transported_bonds))
    )
    return direct, transported, bond_error


def main() -> None:
    rng = np.random.default_rng(20260928)
    largest_option_error = 0.0
    largest_forward_error = 0.0
    largest_printed_ratio_error = 0.0
    largest_multibond_error = 0.0
    largest_shifted_bond_error = 0.0
    largest_forward_law_error = 0.0
    largest_normalized_call_error = 0.0
    largest_implied_vol_error = 0.0
    largest_equal_factor_basket_error = 0.0
    largest_basket_measure_change = 0.0
    positive_option_cases = 0

    for _ in range(500):
        transition = rng.dirichlet(np.ones(3), size=3)
        rates = rng.uniform(-0.01, 0.08, size=3)
        shifts = rng.uniform(-0.005, 0.04, size=3)
        multibond_shifts = np.append(
            shifts, 0.5 * (shifts[1] + shifts[2])
        )
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

        direct_basket, transported_basket, bond_error = (
            multibond_transport_identity(
                transition,
                rates,
                multibond_shifts,
            )
        )
        largest_multibond_error = max(
            largest_multibond_error,
            abs(direct_basket - transported_basket),
        )
        largest_shifted_bond_error = max(
            largest_shifted_bond_error,
            bond_error,
        )

        for horizon in (2, 3):
            for split in range(horizon + 1):
                base_law, shifted_law = forward_state_distribution(
                    transition,
                    rates,
                    multibond_shifts,
                    horizon,
                    split,
                )
                largest_forward_law_error = max(
                    largest_forward_law_error,
                    float(np.max(np.abs(base_law - shifted_law))),
                    abs(float(np.sum(base_law)) - 1.0),
                    abs(float(np.sum(shifted_law)) - 1.0),
                )

        weight_error, call_error, implied_vol_error = (
            caplet_forward_invariance(
                transition,
                rates,
                shifts,
            )
        )
        largest_forward_law_error = max(
            largest_forward_law_error,
            weight_error,
        )
        largest_normalized_call_error = max(
            largest_normalized_call_error,
            call_error,
        )
        largest_implied_vol_error = max(
            largest_implied_vol_error,
            implied_vol_error,
        )

        # A bond basket is not a single bond: its numeraire measure changes
        # unless all maturity-specific shift factors coincide.  Setting the
        # last two interval shifts to zero forces equality for maturities
        # 2, 3 and 4 as viewed at split 1, and recovers exact invariance.
        maturities = (2, 3, 4)
        weights = np.array([0.7, 1.1, 0.9])
        base_basket_law, shifted_basket_law, maturity_factors = (
            basket_measure_state_distribution(
                transition,
                rates,
                multibond_shifts,
                1,
                maturities,
                weights,
            )
        )
        largest_basket_measure_change = max(
            largest_basket_measure_change,
            0.5 * float(np.sum(np.abs(base_basket_law - shifted_basket_law))),
        )
        if not np.allclose(maturity_factors, maturity_factors[0]):
            assert np.max(np.abs(base_basket_law - shifted_basket_law)) > 1e-12

        equal_factor_shifts = multibond_shifts.copy()
        equal_factor_shifts[2:] = 0.0
        base_equal, shifted_equal, equal_factors = (
            basket_measure_state_distribution(
                transition,
                rates,
                equal_factor_shifts,
                1,
                maturities,
                weights,
            )
        )
        assert np.max(np.abs(equal_factors - equal_factors[0])) < 2e-16
        largest_equal_factor_basket_error = max(
            largest_equal_factor_basket_error,
            float(np.max(np.abs(base_equal - shifted_equal))),
        )

    assert largest_option_error < 3e-15
    assert largest_forward_error < 3e-15
    assert positive_option_cases > 100
    assert largest_printed_ratio_error < 5e-13
    assert largest_multibond_error < 3e-15
    assert largest_shifted_bond_error < 3e-15
    assert largest_forward_law_error < 3e-15
    assert largest_normalized_call_error < 3e-15
    assert largest_implied_vol_error < 3e-12
    assert largest_equal_factor_basket_error < 3e-15
    assert largest_basket_measure_change > 1e-5

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

    # A reproducible two-state annuity-measure counterexample.
    annuity_transition = np.array([[0.50, 0.50], [0.05, 0.95]])
    annuity_rates = np.array([0.00, 0.50])
    annuity_shifts = np.array([0.02, 0.01, 0.30, -0.20])
    annuity_base, annuity_shifted, annuity_factors = (
        basket_measure_state_distribution(
            annuity_transition,
            annuity_rates,
            annuity_shifts,
            1,
            (2, 3, 4),
            np.ones(3),
        )
    )
    annuity_tv = 0.5 * float(np.sum(np.abs(annuity_base - annuity_shifted)))
    assert annuity_tv > 0.002

    print("Appendix B deterministic-shift pricing errata")
    print(f"maximum caplet/floorlet identity error: {largest_option_error:.3e}")
    print(
        f"printed-factor ratio error over {positive_option_cases} positive "
        f"cases: {largest_printed_ratio_error:.3e}"
    )
    print(f"maximum forward-measure identity error: {largest_forward_error:.3e}")
    print(
        "maximum nonlinear multibond transport error: "
        f"{largest_multibond_error:.3e}"
    )
    print(
        "maximum independently shifted bond error: "
        f"{largest_shifted_bond_error:.3e}"
    )
    print(
        "maximum multi-step forward-law shift error: "
        f"{largest_forward_law_error:.3e}"
    )
    print(
        "maximum normalized-call scaling error: "
        f"{largest_normalized_call_error:.3e}"
    )
    print(
        "maximum shifted Black implied-vol error: "
        f"{largest_implied_vol_error:.3e}"
    )
    print(
        "maximum equal-factor basket-measure error: "
        f"{largest_equal_factor_basket_error:.3e}"
    )
    print(
        "largest random basket-measure change (TV): "
        f"{largest_basket_measure_change:.9f}"
    )
    print(
        "annuity counterexample: factors "
        f"{annuity_factors}, base law {annuity_base}, "
        f"shifted law {annuity_shifted}, TV {annuity_tv:.9f}"
    )
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
        "change of numeraire, including nonlinear multibond payoffs and "
        "forward-measure/implied-vol invariance, with the exact limitation "
        "for bond-basket numeraires"
    )


if __name__ == "__main__":
    main()
