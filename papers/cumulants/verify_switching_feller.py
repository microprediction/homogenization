"""Certificate for the statewise Feller condition under regime switching.

For a bad regime, the probability of hitting zero before its exponential
holding time expires is the Laplace transform of the frozen CIR hitting time.
The closed Tricomi-U formula is checked against its integral representation.
For a reducible chain, the exact condition depends only on states reachable
from the support of the initial regime law.  Vol-of-vol is allowed to vary by
regime; the correct statewise ratio is then 2 c_i / sigma_i**2, whereas the
fast averaged ratio uses E[sigma_i**2].
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma, hyperu


def reachable_states(generator, initial_support):
    """Return states reachable by positive-rate directed paths."""
    generator = np.asarray(generator, dtype=float)
    reached = set(int(state) for state in initial_support)
    frontier = list(reached)
    while frontier:
        state = frontier.pop()
        for target, rate in enumerate(generator[state]):
            if target != state and rate > 0.0 and target not in reached:
                reached.add(target)
                frontier.append(target)
    return np.array(sorted(reached), dtype=int)


def frozen_hit_before_switch(c, kappa, sigma, value, exit_rate):
    """Return P(tau_0 < Exp(exit_rate)) for a sub-Feller CIR regime."""
    beta = 2.0 * c / sigma**2
    assert 0.0 < beta < 1.0
    alpha = exit_rate / kappa
    z = 2.0 * kappa * value / sigma**2
    closed = gamma(1.0 + alpha - beta) / gamma(1.0 - beta) * hyperu(
        alpha, beta, z
    )

    integral, quadrature_error = quad(
        lambda t: math.exp(-z * t)
        * t ** (alpha - 1.0)
        * (1.0 + t) ** (beta - alpha - 1.0),
        0.0,
        np.inf,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=400,
    )
    independent = (
        gamma(1.0 + alpha - beta)
        / (gamma(1.0 - beta) * gamma(alpha))
        * integral
    )
    return closed, independent, quadrature_error


def main():
    sigma = np.array([0.4, 0.3])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    stationary = np.array([0.5, 0.5])
    value = 0.04

    statewise_ratios = 2.0 * c / sigma**2
    averaged_variance = stationary @ sigma**2
    averaged_ratio = 2.0 * (stationary @ c) / averaged_variance
    assert averaged_ratio > 1.0
    assert statewise_ratios[0] < 1.0 <= statewise_ratios[1]
    # After the exact integrating-factor and quadratic-variation time change,
    # these are half the dimensions of the time-inhomogeneous BESQ process.
    bessel_dimensions = 4.0 * c / sigma**2
    assert np.allclose(bessel_dimensions, 2.0 * statewise_ratios)

    print("1. averaged versus statewise Feller tests")
    print(f"   statewise ratios: {statewise_ratios[0]:.6f}, {statewise_ratios[1]:.6f}")
    print(f"   regime vol-of-vol: {sigma[0]:.6f}, {sigma[1]:.6f}")
    print(f"   averaged squared vol-of-vol: {averaged_variance:.6f}")
    print(f"   correctly averaged ratio: {averaged_ratio:.6f}")
    print(
        "   time-changed BESQ dimensions: "
        f"{bessel_dimensions[0]:.6f}, {bessel_dimensions[1]:.6f}"
    )

    probabilities = []
    discrepancies = []
    print("2. zero before the next regime switch")
    for exit_rate in (0.75, 1.5, 3.0, 6.0):
        closed, independent, quadrature_error = frozen_hit_before_switch(
            c[0], kappa[0], sigma[0], value, exit_rate
        )
        probabilities.append(closed)
        discrepancies.append(abs(closed - independent))
        print(
            f"   q={exit_rate:4.2f}: probability {closed:.10f}; "
            f"quadrature discrepancy {abs(closed - independent):.3e}; "
            f"reported quadrature error {quadrature_error:.3e}"
        )

    assert all(0.0 < probability < 1.0 for probability in probabilities)
    assert all(left > right for left, right in zip(probabilities, probabilities[1:]))
    assert max(discrepancies) < 5e-13
    assert abs(probabilities[1] - 0.234192768272051) < 5e-15

    # A deficient state in a closed class disjoint from the initial support is
    # irrelevant.  Adding one positive-rate edge makes it reachable, at which
    # point the frozen positive hitting probability proves necessity.
    reducible_q = np.array(
        [
            [-1.0, 1.0, 0.0, 0.0],
            [1.0, -1.0, 0.0, 0.0],
            [0.0, 0.0, -1.5, 1.5],
            [0.0, 0.0, 1.0, -1.0],
        ]
    )
    reachable_q = reducible_q.copy()
    reachable_q[1] = np.array([1.0, -1.5, 0.5, 0.0])
    reducible_sigma = np.array([0.2, 0.3, 0.4, 0.25])
    reducible_c = np.array([0.08, 0.08, 0.04, 0.10])
    reducible_kappa = np.array([1.0, 1.2, 0.8, 1.4])
    reducible_ratios = 2.0 * reducible_c / reducible_sigma**2
    initial_support = [0]
    closed_reachable = reachable_states(reducible_q, initial_support)
    opened_reachable = reachable_states(reachable_q, initial_support)
    assert np.array_equal(closed_reachable, np.array([0, 1]))
    assert np.array_equal(opened_reachable, np.arange(4))
    assert np.min(reducible_ratios[closed_reachable]) >= 1.0
    assert reducible_ratios[2] < 1.0
    reachable_probability, independent, _ = frozen_hit_before_switch(
        reducible_c[2],
        reducible_kappa[2],
        reducible_sigma[2],
        value,
        -reachable_q[2, 2],
    )
    assert abs(reachable_probability - independent) < 5e-13
    assert reachable_probability > 0.2
    print("3. reducible-chain reachable-set criterion")
    print(
        "   initial support {0}; reachable before edge "
        f"{closed_reachable.tolist()}, after edge {opened_reachable.tolist()}"
    )
    print(
        f"   unreachable deficient ratio {reducible_ratios[2]:.6f}; "
        "once reachable, conditional boundary probability before exit "
        f"{reachable_probability:.10f}"
    )
    print(
        "PASS: the correctly averaged variable-volatility model passes while "
        "a reachable bad regime has boundary probability "
        f"{probabilities[1]:.10f}; the reducible reachable-set test passes; "
        "independent error "
        f"{max(discrepancies):.3e}"
    )


if __name__ == "__main__":
    main()
