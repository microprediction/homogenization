"""Certificate for the maturity-uniform two-state initial-layer formulas.

The checks distinguish seven statements which are easy to conflate:

1. For arbitrary q(0), adding the first layer makes the first-order error
   uniformly O(eps**2), while adding both layer orders makes the error
   uniformly O(eps**3).
2. If q(0) = 0, the second-order outer approximation has only O(eps**2)
   maximum error.  Adding the second-order layer makes it uniformly
   O(eps**3).
3. The omitted layer becomes as small as the nominal outer remainder at
   t = (eps / 2) log(1 / eps), not at a fixed multiple of eps.
4. With unequal positive transition rates, the stationary weights determine
   the amplitude and the nonzero eigenvalue determines the layer width.  The
   first and second composites remain uniformly O(eps**2) and O(eps**3).
5. At second order, deleting the unequal-rate layer without retaining its
   permanent amplitude shift leaves O(eps**2) fixed-maturity error.  The
   matched outer formula restores O(eps**3), uniformly after the sharp
   logarithmic crossover.
6. A regime-dependent initial vector creates an order-zero layer.  Retaining
   it gives O(eps) error, and its first nonlinear Duhamel correction restores
   a uniform O(eps**2) approximation.
7. The exact Volterra map for that nonlinear ratio is an O(eps) contraction.
   Starting from the order-zero layer, k Picard iterates therefore give a
   uniform O(eps**(k+1)) component approximation without differentiating the
   forcing.
8. A direct slow/fast-coordinate Volterra map removes the ratio singularity.
   It covers arbitrary signed or complex initial vectors, including zero
   stationary mean, with the same gain of one power per Picard iterate and a
   computable a posteriori component bound.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from uniform_layer import (
    first_order,
    first_order_asymmetric,
    first_order_asymmetric_initial,
    second_order,
    second_order_asymmetric,
    second_order_outer_asymmetric,
)


HERE = Path(__file__).resolve().parent
T_MAX = 1.5
EPSILONS = np.array([1 / 20, 1 / 40, 1 / 80, 1 / 160], float)


def order(errors):
    return float(np.polyfit(np.log(EPSILONS), np.log(errors), 1)[0])


def exact_curve(eps, b, q, grid):
    def rhs(t, a):
        bp, qp = b(t), q(t)
        return [(bp + qp - 1 / eps) * a[0] + a[1] / eps,
                a[0] / eps + (bp - qp - 1 / eps) * a[1]]

    sol = solve_ivp(rhs, (0, float(grid[-1])), [1.0, 1.0], method='DOP853',
                    rtol=2e-12, atol=2e-14, dense_output=True,
                    max_step=min(0.002, eps / 8))
    assert sol.success
    return sol.sol(grid)[0]


def exact_curve_asymmetric(eps, g1, g2, rate12, rate21, grid, initial=(1.0, 1.0)):
    """Both components for an unequal-rate two-state generator."""
    def rhs(t, a):
        return [
            (g1(t) - rate12 / eps) * a[0] + rate12 * a[1] / eps,
            rate21 * a[0] / eps + (g2(t) - rate21 / eps) * a[1],
        ]

    sol = solve_ivp(rhs, (0, float(grid[-1])), initial, method='DOP853',
                    rtol=2e-12, atol=2e-14, dense_output=True,
                    max_step=min(0.002, eps / (8 * (rate12 + rate21))))
    assert sol.success
    return sol.sol(grid)


def picard_components_asymmetric(
    eps, g_bar, delta, rate12, rate21, initial, grid, iterations,
    return_details=False,
):
    """Components from exact nonlinear Volterra/Picard ratio iterates.

    This is deliberately independent of the original two-component solve.
    Each scalar iterate solves the stable convolution equation driven by the
    previous iterate.  The mean is then reconstructed from its exact scalar
    equation.
    """
    rho = rate12 + rate21
    pi1, pi2 = rate21 / rho, rate12 / rho
    skew_weight = pi2 - pi1
    variance_weight = pi1 * pi2
    m0 = float(pi1 * initial[0] + pi2 * initial[1])
    r0 = float((initial[0] - initial[1]) / m0)

    ratio = lambda t: r0 * np.exp(-rho * np.asarray(t) / eps)
    for _ in range(iterations):
        previous = ratio

        def rhs(t, value):
            old = float(previous(t))
            source = delta(t) * (
                1 + skew_weight * old - variance_weight * old ** 2)
            return [source - rho * value[0] / eps]

        solution = solve_ivp(
            rhs, (0, float(grid[-1])), [r0], method='DOP853',
            rtol=4e-13, atol=4e-15, dense_output=True,
            max_step=min(0.001, eps / (12 * rho)),
        )
        assert solution.success
        ratio = lambda t, solution=solution: solution.sol(t)[0]

    def mean_rhs(t, log_mean):
        return [g_bar(t) + variance_weight * delta(t) * float(ratio(t))]

    mean_solution = solve_ivp(
        mean_rhs, (0, float(grid[-1])), [math.log(m0)], method='DOP853',
        rtol=4e-13, atol=4e-15, dense_output=True,
        max_step=min(0.001, eps / (12 * rho)),
    )
    assert mean_solution.success
    ratio_values = ratio(grid)
    mean_values = np.exp(mean_solution.sol(grid)[0])
    components = np.vstack((
        mean_values * (1 + pi2 * ratio_values),
        mean_values * (1 - pi1 * ratio_values),
    ))
    if return_details:
        return components, ratio_values, mean_values
    return components


def direct_picard_components_asymmetric(
    eps, g_bar, delta, rate12, rate21, initial, grid, iterations,
    return_details=False,
):
    """Components from the nonsingular slow/fast Volterra iteration.

    After removing ``exp(int g_bar)``, write ``M`` for the stationary mean
    coordinate and ``D`` for the component difference.  Then

        M(t) = m0 + v * integral_0^t delta(u) D(u) du,

    while ``D`` is obtained from a stable convolution.  No division by ``M``
    occurs, so this chart remains valid when ``m0`` or ``M(t)`` is zero.
    """
    rho = rate12 + rate21
    pi1, pi2 = rate21 / rho, rate12 / rho
    skew_weight = pi2 - pi1
    variance_weight = pi1 * pi2
    initial = np.asarray(initial)
    m0 = float(pi1 * initial[0] + pi2 * initial[1])
    d0 = float(initial[0] - initial[1])

    difference = lambda t: d0 * np.exp(-rho * np.asarray(t) / eps)
    for _ in range(iterations):
        previous = difference

        def integral_rhs(t, value):
            return [delta(t) * float(previous(t))]

        with np.errstate(invalid='ignore', divide='ignore'):
            integral_solution = solve_ivp(
                integral_rhs, (0, float(grid[-1])), [0.0], method='DOP853',
                rtol=4e-13, atol=4e-15, dense_output=True,
                max_step=min(0.001, eps / (12 * rho)),
            )
        assert integral_solution.success

        def difference_rhs(t, value):
            old = float(previous(t))
            mean = m0 + variance_weight * float(integral_solution.sol(t)[0])
            source = delta(t) * (mean + skew_weight * old)
            return [source - rho * value[0] / eps]

        with np.errstate(invalid='ignore', divide='ignore'):
            difference_solution = solve_ivp(
                difference_rhs, (0, float(grid[-1])), [d0], method='DOP853',
                rtol=4e-13, atol=4e-15, dense_output=True,
                max_step=min(0.001, eps / (12 * rho)),
            )
        assert difference_solution.success
        difference = lambda t, solution=difference_solution: solution.sol(t)[0]

    def mean_rhs(t, value):
        return [variance_weight * delta(t) * float(difference(t))]

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_solution = solve_ivp(
            mean_rhs, (0, float(grid[-1])), [m0], method='DOP853',
            rtol=4e-13, atol=4e-15, dense_output=True,
            max_step=min(0.001, eps / (12 * rho)),
        )
    assert mean_solution.success

    def common_rhs(t, value):
        return [g_bar(t)]

    with np.errstate(invalid='ignore', divide='ignore'):
        common_solution = solve_ivp(
            common_rhs, (0, float(grid[-1])), [0.0], method='DOP853',
            rtol=4e-13, atol=4e-15, dense_output=True, max_step=0.001,
        )
    assert common_solution.success
    difference_values = difference(grid)
    mean_values = mean_solution.sol(grid)[0]
    common_factor = np.exp(common_solution.sol(grid)[0])
    components = np.vstack((
        common_factor * (mean_values + pi2 * difference_values),
        common_factor * (mean_values - pi1 * difference_values),
    ))
    if return_details:
        return components, difference_values, mean_values
    return components


def integrals(grid, eps, b_values, q_values):
    zeros = np.zeros(1)
    int_b = np.r_[zeros, cumulative_trapezoid(b_values, grid)]
    int_q2 = np.r_[zeros, cumulative_trapezoid(q_values ** 2, grid)]
    int_q_layer = np.r_[zeros, cumulative_trapezoid(q_values * np.exp(-2 * grid / eps), grid)]
    return int_b, int_q2, int_q_layer


def nonzero_start():
    b = lambda t: -0.04 - 0.015 * math.exp(-0.7 * t)
    q = lambda t: 0.65 + 0.35 * (1 - math.exp(-1.3 * t))
    qp = lambda t: 0.455 * math.exp(-1.3 * t)
    q0 = q(0.0)
    qp0 = qp(0.0)
    outer_errors, first_errors, second_errors, fixed_errors = [], [], [], []
    rows = []
    for eps in EPSILONS:
        # Resolve the fastest layer uniformly as eps decreases.
        grid = np.unique(np.r_[np.linspace(0, T_MAX, 3001), eps * np.linspace(0, 8, 1001)])
        grid = grid[grid <= T_MAX]
        bv = np.array([b(t) for t in grid])
        qv = np.array([q(t) for t in grid])
        qpv = np.array([qp(t) for t in grid])
        ib, iq2, iql = integrals(grid, eps, bv, qv)
        exact = exact_curve(eps, b, q, grid)
        outer = np.exp(ib + eps * iq2 / 2) * (1 + eps * qv / 2)
        comp1 = first_order(grid, eps, ib, iq2, qv, q0, iql)
        comp2 = second_order(
            grid, eps, ib, iq2, qv, qpv, q0, qp0, iql
        )
        oe = float(np.max(np.abs(outer - exact)))
        c1e = float(np.max(np.abs(comp1 - exact)))
        c2e = float(np.max(np.abs(comp2 - exact)))
        fe = float(abs(outer[-1] - exact[-1]))
        outer_errors.append(oe)
        first_errors.append(c1e)
        second_errors.append(c2e)
        fixed_errors.append(fe)
        rows.append(dict(
            epsilon=float(eps), outer_sup=oe, first_composite_sup=c1e,
            second_composite_sup=c2e, outer_fixed_T=fe,
        ))
    return (
        rows, order(outer_errors), order(first_errors), order(second_errors),
        order(fixed_errors),
    )


def zero_start():
    b = lambda t: -0.03 - 0.01 * math.exp(-0.9 * t)
    alpha, kappa = -0.8, 1.6
    q = lambda t: alpha * (1 - math.exp(-kappa * t))
    qp = lambda t: alpha * kappa * math.exp(-kappa * t)
    qp0 = qp(0.0)
    outer_errors, composite_errors, crossover_ratios = [], [], []
    rows = []
    for eps in EPSILONS:
        grid = np.unique(np.r_[np.linspace(0, T_MAX, 3001), eps * np.linspace(0, 8, 1001)])
        grid = grid[grid <= T_MAX]
        bv = np.array([b(t) for t in grid])
        qv = np.array([q(t) for t in grid])
        qpv = np.array([qp(t) for t in grid])
        ib, iq2, iql = integrals(grid, eps, bv, qv)
        exact = exact_curve(eps, b, q, grid)
        omega_outer = eps * qv / 2 - eps ** 2 * qpv / 4
        outer = np.exp(ib + eps * iq2 / 2 - eps ** 2 * qv ** 2 / 8) * (1 + omega_outer)
        comp = second_order(
            grid, eps, ib, iq2, qv, qpv, 0.0, qp0, iql
        )
        oe = float(np.max(np.abs(outer - exact)))
        ce = float(np.max(np.abs(comp - exact)))
        fe = float(abs(outer[-1] - exact[-1]))
        t_cross = eps * math.log(1 / eps) / 2
        layer_at_cross = eps ** 2 * abs(qp0) * math.exp(-2 * t_cross / eps) / 4
        crossover_ratios.append(layer_at_cross / eps ** 3)
        outer_errors.append(oe)
        composite_errors.append(ce)
        rows.append(dict(epsilon=float(eps), outer_sup=oe, composite_sup=ce,
                         outer_fixed_T=fe, crossover_ratio=float(crossover_ratios[-1])))
    return rows, order(outer_errors), order(composite_errors), crossover_ratios


def asymmetric_rates():
    """Check the general stationary weights and gap independently."""
    rate12, rate21 = 1.7, 0.4
    rho = rate12 + rate21
    pi1, pi2 = rate21 / rho, rate12 / rho
    variance_weight = pi1 * pi2
    g_bar = lambda t: -0.04 - 0.015 * math.exp(-0.7 * t)
    delta = lambda t: 0.9 + 0.4 * (1 - math.exp(-1.2 * t))
    delta_prime = lambda t: 0.48 * math.exp(-1.2 * t)
    g1 = lambda t: g_bar(t) + pi2 * delta(t)
    g2 = lambda t: g_bar(t) - pi1 * delta(t)
    delta0 = delta(0.0)
    delta_prime0 = delta_prime(0.0)
    outer_errors, first_errors, second_errors, fixed_errors = [], [], [], []
    naive_second_fixed_errors, corrected_second_fixed_errors = [], []
    corrected_post_layer_errors = []
    rows = []
    for eps in EPSILONS:
        crossover = 2 * eps * math.log(1 / eps) / rho
        grid = np.unique(np.r_[
            np.linspace(0, T_MAX, 3001),
            eps * np.linspace(0, 8 / rho, 1001),
            crossover,
        ])
        grid = grid[grid <= T_MAX]
        gbar_values = np.array([g_bar(t) for t in grid])
        delta_values = np.array([delta(t) for t in grid])
        delta_prime_values = np.array([delta_prime(t) for t in grid])
        int_gbar = np.r_[0.0, cumulative_trapezoid(gbar_values, grid)]
        int_delta = np.r_[0.0, cumulative_trapezoid(delta_values, grid)]
        int_delta2 = np.r_[0.0, cumulative_trapezoid(delta_values ** 2, grid)]
        int_delta3 = np.r_[0.0, cumulative_trapezoid(delta_values ** 3, grid)]
        int_delta_layer = np.r_[
            0.0,
            cumulative_trapezoid(
                delta_values * np.exp(-rho * grid / eps), grid),
        ]
        exact = exact_curve_asymmetric(
            eps, g1, g2, rate12, rate21, grid)
        ratio_outer = eps * delta_values / rho
        log_m_outer = int_gbar + eps * variance_weight * int_delta2 / rho
        mean_outer = np.exp(log_m_outer)
        outer = np.vstack((
            mean_outer * (1 + pi2 * ratio_outer),
            mean_outer * (1 - pi1 * ratio_outer),
        ))
        skew_weight = pi2 - pi1
        ratio_outer2 = ratio_outer + eps ** 2 * (
            skew_weight * delta_values ** 2 - delta_prime_values
        ) / rho ** 2
        log_m_outer2 = log_m_outer + eps ** 2 * variance_weight / rho ** 2 * (
            skew_weight * int_delta3
            - (delta_values ** 2 - delta0 ** 2) / 2
        )
        naive_mean_outer2 = np.exp(log_m_outer2)
        naive_outer2 = np.vstack((
            naive_mean_outer2 * (1 + pi2 * ratio_outer2),
            naive_mean_outer2 * (1 - pi1 * ratio_outer2),
        ))
        corrected_outer2 = np.vstack([
            second_order_outer_asymmetric(
                eps, int_gbar, int_delta2, int_delta3, delta_values,
                delta_prime_values, delta0, rate12, rate21, state=state)
            for state in (0, 1)
        ])
        first_composite = np.vstack([
            first_order_asymmetric(
                grid, eps, int_gbar, int_delta2, delta_values, delta0,
                int_delta_layer, rate12, rate21, state=state)
            for state in (0, 1)
        ])
        second_composite = np.vstack([
            second_order_asymmetric(
                grid, eps, int_gbar, int_delta, int_delta2, int_delta3,
                delta_values, delta_prime_values, delta0, delta_prime0,
                int_delta_layer, rate12, rate21, state=state)
            for state in (0, 1)
        ])
        outer_error = float(np.max(np.abs(outer - exact)))
        first_error = float(np.max(np.abs(first_composite - exact)))
        second_error = float(np.max(np.abs(second_composite - exact)))
        fixed_error = float(np.max(np.abs(outer[:, -1] - exact[:, -1])))
        naive_second_fixed_error = float(np.max(
            np.abs(naive_outer2[:, -1] - exact[:, -1])))
        predicted_naive_coefficient = (
            math.exp(float(int_gbar[-1])) * variance_weight * delta0 ** 2 / rho ** 2
        )
        naive_shift_ratio = (
            naive_second_fixed_error / (eps ** 2 * predicted_naive_coefficient)
        )
        corrected_second_fixed_error = float(np.max(
            np.abs(corrected_outer2[:, -1] - exact[:, -1])))
        post_layer = grid >= crossover
        corrected_post_layer_error = float(np.max(
            np.abs(corrected_outer2[:, post_layer] - exact[:, post_layer])))
        outer_errors.append(outer_error)
        first_errors.append(first_error)
        second_errors.append(second_error)
        fixed_errors.append(fixed_error)
        naive_second_fixed_errors.append(naive_second_fixed_error)
        corrected_second_fixed_errors.append(corrected_second_fixed_error)
        corrected_post_layer_errors.append(corrected_post_layer_error)
        rows.append(dict(
            epsilon=float(eps), outer_sup=outer_error,
            first_composite_sup=first_error,
            second_composite_sup=second_error, outer_fixed_T=fixed_error,
            naive_second_outer_fixed_T=naive_second_fixed_error,
            corrected_second_outer_fixed_T=corrected_second_fixed_error,
            corrected_second_outer_post_layer_sup=corrected_post_layer_error,
            second_outer_crossover=float(crossover),
            naive_shift_leading_ratio=float(naive_shift_ratio),
        ))
    return (
        rows, order(outer_errors), order(first_errors), order(second_errors),
        order(fixed_errors), order(naive_second_fixed_errors),
        order(corrected_second_fixed_errors),
        order(corrected_post_layer_errors), rate12, rate21,
    )


def regime_dependent_initial_data():
    """Check the order-zero layer generated by a nonconstant initial vector."""
    rate12, rate21 = 1.7, 0.4
    rho = rate12 + rate21
    pi1, pi2 = rate21 / rho, rate12 / rho
    initial = np.array([1.4, 0.7])
    m0 = float(pi1 * initial[0] + pi2 * initial[1])
    r0 = float((initial[0] - initial[1]) / m0)
    g_bar = lambda t: -0.04 - 0.015 * math.exp(-0.7 * t)
    delta = lambda t: 0.9 + 0.4 * (1 - math.exp(-1.2 * t))
    g1 = lambda t: g_bar(t) + pi2 * delta(t)
    g2 = lambda t: g_bar(t) - pi1 * delta(t)
    delta0 = delta(0.0)
    projection_errors, leading_errors, first_errors = [], [], []
    picard0_errors, picard1_errors, picard2_errors = [], [], []
    picard1_bounds, picard2_bounds = [], []
    rows = []
    for eps in EPSILONS:
        grid = np.unique(np.r_[
            np.linspace(0, T_MAX, 3001),
            eps * np.linspace(0, 8 / rho, 1001),
        ])
        grid = grid[grid <= T_MAX]
        gbar_values = np.array([g_bar(t) for t in grid])
        delta_values = np.array([delta(t) for t in grid])
        int_gbar = np.r_[0.0, cumulative_trapezoid(gbar_values, grid)]
        int_delta = np.r_[0.0, cumulative_trapezoid(delta_values, grid)]
        int_delta2 = np.r_[0.0, cumulative_trapezoid(delta_values ** 2, grid)]
        int_delta_layer = np.r_[
            0.0,
            cumulative_trapezoid(
                delta_values * np.exp(-rho * grid / eps), grid),
        ]
        exact = exact_curve_asymmetric(
            eps, g1, g2, rate12, rate21, grid, initial=initial)
        projection = np.vstack((
            m0 * np.exp(int_gbar), m0 * np.exp(int_gbar)))
        ratio0 = r0 * np.exp(-rho * grid / eps)
        leading_mean = m0 * np.exp(int_gbar)
        leading = np.vstack((
            leading_mean * (1 + pi2 * ratio0),
            leading_mean * (1 - pi1 * ratio0),
        ))
        first = np.vstack([
            first_order_asymmetric_initial(
                grid, eps, int_gbar, int_delta, int_delta2,
                delta_values, delta0, int_delta_layer,
                rate12, rate21, initial, state=state)
            for state in (0, 1)
        ])
        picard_details = [
            picard_components_asymmetric(
                eps, g_bar, delta, rate12, rate21, initial, grid, iterations,
                return_details=True)
            for iterations in range(3)
        ]
        picard = [value[0] for value in picard_details]
        pe = float(np.max(np.abs(projection - exact)))
        le = float(np.max(np.abs(leading - exact)))
        fe = float(np.max(np.abs(first - exact)))
        picard_errors = [float(np.max(np.abs(value - exact))) for value in picard]
        # Completely computable Banach bounds: R=2 is invariant for every
        # tested eps, and q is the certified contraction factor on that ball.
        ratio_radius = 2.0
        delta_bound = 1.3
        gbar_bound = 0.055
        skew_weight = pi2 - pi1
        variance_weight = pi1 * pi2
        source_bound = delta_bound * (
            1 + abs(skew_weight) * ratio_radius
            + variance_weight * ratio_radius ** 2)
        invariant_margin = (
            ratio_radius - abs(r0) - eps * source_bound / rho)
        contraction = eps * delta_bound * (
            abs(skew_weight) + 2 * variance_weight * ratio_radius) / rho
        assert invariant_margin > 0 and contraction < 1
        mean_bound = m0 * math.exp(T_MAX * (
            gbar_bound + variance_weight * delta_bound * ratio_radius))
        component_factor = mean_bound * (
            (1 + max(pi1, pi2) * ratio_radius)
            * variance_weight * delta_bound * T_MAX
            + max(pi1, pi2))
        posterior_bounds = []
        for k in (1, 2):
            ratio_increment = float(np.max(np.abs(
                picard_details[k][1] - picard_details[k - 1][1])))
            ratio_bound = contraction * ratio_increment / (1 - contraction)
            posterior_bounds.append(component_factor * ratio_bound)
        picard1_bounds.append(posterior_bounds[0])
        picard2_bounds.append(posterior_bounds[1])
        projection_errors.append(pe)
        leading_errors.append(le)
        first_errors.append(fe)
        picard0_errors.append(picard_errors[0])
        picard1_errors.append(picard_errors[1])
        picard2_errors.append(picard_errors[2])
        rows.append(dict(
            epsilon=float(eps), projection_sup=pe,
            leading_layer_sup=le, first_composite_sup=fe,
            picard_0_sup=picard_errors[0],
            picard_1_sup=picard_errors[1],
            picard_2_sup=picard_errors[2],
            picard_1_aposteriori_bound=posterior_bounds[0],
            picard_2_aposteriori_bound=posterior_bounds[1],
            contraction_factor=float(contraction),
            invariant_margin=float(invariant_margin),
            initial_match=float(np.max(np.abs(first[:, 0] - initial))),
            picard_initial_match=float(max(
                np.max(np.abs(value[:, 0] - initial)) for value in picard)),
        ))
    return (
        rows, order(projection_errors), order(leading_errors),
        order(first_errors), order(picard0_errors), order(picard1_errors),
        order(picard2_errors), order(picard1_bounds), order(picard2_bounds),
        initial.tolist(), m0, r0,
    )


def zero_mean_initial_data():
    """Check the nonsingular direct iteration when the ratio chart fails."""
    rate12, rate21 = 1.7, 0.4
    rho = rate12 + rate21
    pi1, pi2 = rate21 / rho, rate12 / rho
    skew_weight = pi2 - pi1
    variance_weight = pi1 * pi2
    initial = np.array([1.0, -pi1 / pi2])
    m0 = float(pi1 * initial[0] + pi2 * initial[1])
    d0 = float(initial[0] - initial[1])
    assert abs(m0) < 1e-15
    g_bar = lambda t: -0.04 - 0.015 * math.exp(-0.7 * t)
    delta = lambda t: 0.9 + 0.4 * (1 - math.exp(-1.2 * t))
    g1 = lambda t: g_bar(t) + pi2 * delta(t)
    g2 = lambda t: g_bar(t) - pi1 * delta(t)
    direct_errors = [[], [], []]
    direct_bounds = [[], []]
    rows = []
    for eps in EPSILONS:
        grid = np.unique(np.r_[
            np.linspace(0, T_MAX, 3001),
            eps * np.linspace(0, 8 / rho, 1001),
        ])
        grid = grid[grid <= T_MAX]
        exact = exact_curve_asymmetric(
            eps, g1, g2, rate12, rate21, grid, initial=initial)
        details = [
            direct_picard_components_asymmetric(
                eps, g_bar, delta, rate12, rate21, initial, grid,
                iterations, return_details=True)
            for iterations in range(3)
        ]
        errors = [float(np.max(np.abs(value[0] - exact))) for value in details]

        # Direct-coordinate invariant ball and Banach bound.  Unlike the
        # ratio certificate, these constants remain finite at m0 = 0.
        difference_radius = 2.0
        delta_bound = 1.3
        gbar_bound = 0.055
        direct_lipschitz = abs(skew_weight) + variance_weight * delta_bound * T_MAX
        contraction = eps * delta_bound * direct_lipschitz / rho
        invariant_margin = difference_radius - abs(d0) - eps * delta_bound / rho * (
            abs(m0) + direct_lipschitz * difference_radius)
        assert invariant_margin > 0 and contraction < 1
        component_factor = math.exp(T_MAX * gbar_bound) * (
            max(pi1, pi2) + variance_weight * delta_bound * T_MAX)
        bounds = []
        for k in (1, 2):
            increment = float(np.max(np.abs(
                details[k][1] - details[k - 1][1])))
            difference_bound = contraction * increment / (1 - contraction)
            bounds.append(component_factor * difference_bound)
        for k in range(3):
            direct_errors[k].append(errors[k])
        for k in range(2):
            direct_bounds[k].append(bounds[k])
        rows.append(dict(
            epsilon=float(eps), direct_0_sup=errors[0],
            direct_1_sup=errors[1], direct_2_sup=errors[2],
            direct_1_aposteriori_bound=bounds[0],
            direct_2_aposteriori_bound=bounds[1],
            contraction_factor=float(contraction),
            invariant_margin=float(invariant_margin),
            initial_match=float(max(
                np.max(np.abs(value[0][:, 0] - initial)) for value in details)),
        ))
    return (
        rows, *(order(values) for values in direct_errors),
        *(order(values) for values in direct_bounds), initial.tolist(), m0, d0,
    )


def main():
    nz, nz_outer, nz_first, nz_second, nz_fixed = nonzero_start()
    z, z_outer, z_comp, cross = zero_start()
    (asym, asym_outer, asym_first, asym_second, asym_fixed,
     asym_naive_second_fixed, asym_corrected_second_fixed,
     asym_corrected_post_layer, rate12, rate21) = asymmetric_rates()
    (initial_rows, initial_projection, initial_leading, initial_first,
     initial_picard0, initial_picard1, initial_picard2,
     initial_picard1_bound, initial_picard2_bound,
     initial, initial_m0, initial_r0) = regime_dependent_initial_data()
    (zero_mean_rows, zero_mean_direct0, zero_mean_direct1,
     zero_mean_direct2, zero_mean_bound1, zero_mean_bound2,
     zero_mean_initial, zero_mean_m0, zero_mean_d0) = zero_mean_initial_data()
    print('1. q(0) != 0: outer, first composite, and second composite')
    for row in nz:
        print(f"   eps={row['epsilon']:.6f} outer sup={row['outer_sup']:.3e} "
              f"first comp={row['first_composite_sup']:.3e} "
              f"second comp={row['second_composite_sup']:.3e} "
              f"fixed-T outer={row['outer_fixed_T']:.3e}")
    print(f'   observed orders: outer {nz_outer:.6f}, first composite '
          f'{nz_first:.6f}, second composite {nz_second:.6f}, '
          f'fixed-T outer {nz_fixed:.6f}')
    print('2. q(0) = 0: second-order outer versus uniform composite')
    for row in z:
        print(f"   eps={row['epsilon']:.6f} outer sup={row['outer_sup']:.3e} "
              f"composite sup={row['composite_sup']:.3e} fixed-T outer={row['outer_fixed_T']:.3e}")
    print(f'   observed uniform orders: outer sup {z_outer:.6f}, composite sup {z_comp:.6f}')
    print(f'3. crossover layer / eps^3: min={min(cross):.12f}, max={max(cross):.12f}')
    print(f'4. asymmetric rates q12={rate12:.2f}, q21={rate21:.2f}')
    for row in asym:
        print(f"   eps={row['epsilon']:.6f} outer sup={row['outer_sup']:.3e} "
              f"first comp={row['first_composite_sup']:.3e} "
              f"second comp={row['second_composite_sup']:.3e} "
              f"naive/corrected second fixed={row['naive_second_outer_fixed_T']:.3e}/"
              f"{row['corrected_second_outer_fixed_T']:.3e} "
              f"post-layer corrected={row['corrected_second_outer_post_layer_sup']:.3e} "
              f"naive/leading shift={row['naive_shift_leading_ratio']:.6f}")
    print(f'   observed orders: outer {asym_outer:.6f}, first composite '
          f'{asym_first:.6f}, second composite {asym_second:.6f}, '
          f'fixed-T first outer {asym_fixed:.6f}, naive second fixed '
          f'{asym_naive_second_fixed:.6f}, corrected second fixed '
          f'{asym_corrected_second_fixed:.6f}, corrected post-layer '
          f'{asym_corrected_post_layer:.6f}')
    print(f'5. regime-dependent initial vector {initial}, m0={initial_m0:.6f}, '
          f'r0={initial_r0:.6f}')
    for row in initial_rows:
        print(f"   eps={row['epsilon']:.6f} projection={row['projection_sup']:.3e} "
              f"leading layer={row['leading_layer_sup']:.3e} "
              f"first composite={row['first_composite_sup']:.3e} "
              f"Picard 0/1/2={row['picard_0_sup']:.3e}/"
              f"{row['picard_1_sup']:.3e}/{row['picard_2_sup']:.3e} "
              f"bounds 1/2={row['picard_1_aposteriori_bound']:.3e}/"
              f"{row['picard_2_aposteriori_bound']:.3e} "
              f"initial match={row['initial_match']:.1e}")
    print(f'   observed orders: projection {initial_projection:.6f}, '
          f'leading layer {initial_leading:.6f}, first composite {initial_first:.6f}, '
          f'Picard 0/1/2 {initial_picard0:.6f}/{initial_picard1:.6f}/'
          f'{initial_picard2:.6f}, bounds 1/2 '
          f'{initial_picard1_bound:.6f}/{initial_picard2_bound:.6f}')
    print(f'6. zero-mean signed initial vector {zero_mean_initial}, '
          f'm0={zero_mean_m0:.1e}, d0={zero_mean_d0:.6f}')
    for row in zero_mean_rows:
        print(f"   eps={row['epsilon']:.6f} direct 0/1/2="
              f"{row['direct_0_sup']:.3e}/{row['direct_1_sup']:.3e}/"
              f"{row['direct_2_sup']:.3e} bounds 1/2="
              f"{row['direct_1_aposteriori_bound']:.3e}/"
              f"{row['direct_2_aposteriori_bound']:.3e} "
              f"q={row['contraction_factor']:.6f} "
              f"margin={row['invariant_margin']:.6f}")
    print(f'   observed orders: direct 0/1/2 '
          f'{zero_mean_direct0:.6f}/{zero_mean_direct1:.6f}/'
          f'{zero_mean_direct2:.6f}, bounds 1/2 '
          f'{zero_mean_bound1:.6f}/{zero_mean_bound2:.6f}')
    assert 0.9 < nz_outer < 1.1
    assert 1.85 < nz_first < 2.15
    assert 2.8 < nz_second < 3.2
    assert 1.85 < nz_fixed < 2.15
    assert 1.85 < z_outer < 2.15
    assert 2.8 < z_comp < 3.2
    assert max(cross) - min(cross) < 1e-12
    assert 0.9 < asym_outer < 1.1
    assert 1.85 < asym_first < 2.15
    assert 2.8 < asym_second < 3.2
    assert 1.85 < asym_fixed < 2.15
    assert 1.85 < asym_naive_second_fixed < 2.15
    assert 2.8 < asym_corrected_second_fixed < 3.2
    assert 2.7 < asym_corrected_post_layer < 3.3
    assert abs(asym[-1]['naive_shift_leading_ratio'] - 1) < 0.02
    assert -0.1 < initial_projection < 0.1
    assert 0.85 < initial_leading < 1.15
    assert 1.8 < initial_first < 2.2
    assert 0.85 < initial_picard0 < 1.15
    assert 1.8 < initial_picard1 < 2.2
    assert 2.75 < initial_picard2 < 3.25
    assert 1.8 < initial_picard1_bound < 2.2
    assert 2.75 < initial_picard2_bound < 3.25
    assert all(
        row['picard_1_sup'] <= row['picard_1_aposteriori_bound']
        and row['picard_2_sup'] <= row['picard_2_aposteriori_bound']
        for row in initial_rows)
    assert max(row['initial_match'] for row in initial_rows) < 1e-14
    assert max(row['picard_initial_match'] for row in initial_rows) < 1e-13
    assert 0.85 < zero_mean_direct0 < 1.15
    assert 1.8 < zero_mean_direct1 < 2.2
    assert 2.75 < zero_mean_direct2 < 3.25
    assert 1.8 < zero_mean_bound1 < 2.2
    assert 2.75 < zero_mean_bound2 < 3.25
    assert all(
        row['direct_1_sup'] <= row['direct_1_aposteriori_bound']
        and row['direct_2_sup'] <= row['direct_2_aposteriori_bound']
        for row in zero_mean_rows)
    assert max(row['initial_match'] for row in zero_mean_rows) < 1e-13
    out = dict(nonzero_start=dict(
                   rows=nz, outer_sup_order=nz_outer,
                   first_composite_sup_order=nz_first,
                   second_composite_sup_order=nz_second,
                   outer_fixed_order=nz_fixed),
               zero_start=dict(rows=z, outer_sup_order=z_outer,
                               composite_sup_order=z_comp,
                               crossover_ratio=cross[0]),
               asymmetric_rates=dict(
                   rows=asym, rate12=rate12, rate21=rate21,
                   outer_sup_order=asym_outer,
                   first_composite_sup_order=asym_first,
                   second_composite_sup_order=asym_second,
                   outer_fixed_order=asym_fixed,
                   naive_second_outer_fixed_order=asym_naive_second_fixed,
                   corrected_second_outer_fixed_order=asym_corrected_second_fixed,
                   corrected_second_outer_post_layer_order=asym_corrected_post_layer))
    out['regime_dependent_initial_data'] = dict(
        rows=initial_rows, initial=initial, m0=initial_m0, r0=initial_r0,
        projection_sup_order=initial_projection,
        leading_layer_sup_order=initial_leading,
        first_composite_sup_order=initial_first,
        picard_0_sup_order=initial_picard0,
        picard_1_sup_order=initial_picard1,
        picard_2_sup_order=initial_picard2,
        picard_1_bound_order=initial_picard1_bound,
        picard_2_bound_order=initial_picard2_bound)
    out['zero_mean_initial_data'] = dict(
        rows=zero_mean_rows, initial=zero_mean_initial,
        weighted_mean=zero_mean_m0, initial_difference=zero_mean_d0,
        direct_0_sup_order=zero_mean_direct0,
        direct_1_sup_order=zero_mean_direct1,
        direct_2_sup_order=zero_mean_direct2,
        direct_1_bound_order=zero_mean_bound1,
        direct_2_bound_order=zero_mean_bound2)
    (HERE / 'uniform_layer_results.json').write_text(json.dumps(out, indent=2) + '\n')
    print('PASS')


if __name__ == '__main__':
    main()
