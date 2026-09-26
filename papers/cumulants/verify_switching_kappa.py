"""Certificate for simultaneous switching of CIR drift and vol-of-vol.

The correct regime features are c_y=kappa_y theta_y, kappa_y, and
a_y=xi_y^2.  Under regime-dependent leverage, eta_y=rho_y*xi_y is the
additional feature because it is the coefficient of the cross derivative.
Polynomial closure gives an exact finite-dimensional check of the general
Green--Kubo rule.
"""
import math
import os
import sys

import numpy as np
from scipy.linalg import eig, expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "general"))
from effective_generator import (
    effective_generator,
    full_generator,
    gk,
    group_inverse,
    stationary,
)


SPEEDS = (8, 16, 32, 64)


def polynomial_operators(degree=5):
    """Matrices for d_v, -v d_v, and (v/2)d_vv in (1,v,...)."""
    derivative = np.zeros((degree + 1, degree + 1))
    multiply_v = np.zeros_like(derivative)
    for power in range(1, degree + 1):
        derivative[power - 1, power] = power
        multiply_v[power, power - 1] = 1.0
    a_c = derivative
    a_kappa = -multiply_v @ derivative
    a_variance = 0.5 * multiply_v @ derivative @ derivative
    return a_c, a_kappa, a_variance, multiply_v


def integrated_variance_operators(degree=4):
    """Polynomial operators on monomials v^i z^j, i+j <= degree, with dz=v dt."""
    basis = [(i, total - i) for total in range(degree + 1) for i in range(total + 1)]
    index = {powers: position for position, powers in enumerate(basis)}
    size = len(basis)

    def operator(action):
        matrix = np.zeros((size, size))
        for column, powers in enumerate(basis):
            for coefficient, target in action(*powers):
                if coefficient and target in index:
                    matrix[index[target], column] += coefficient
        return matrix

    d_v = operator(lambda i, j: [(i, (i - 1, j))] if i else [])
    minus_v_d_v = operator(lambda i, j: [(-i, (i, j))] if i else [])
    v_d_vv = operator(lambda i, j: [(i * (i - 1), (i - 1, j))] if i >= 2 else [])
    v_d_z = operator(lambda i, j: [(j, (i + 1, j - 1))] if j else [])
    return basis, d_v, minus_v_d_v, v_d_vv, v_d_z


def leveraged_operators(degree=4):
    """Polynomial operators on monomials M^i v^j z^k for Heston leverage."""
    basis = [
        (i, j, total - i - j)
        for total in range(degree + 1)
        for i in range(total + 1)
        for j in range(total - i + 1)
    ]
    index = {powers: position for position, powers in enumerate(basis)}
    size = len(basis)

    def operator(action):
        matrix = np.zeros((size, size))
        for column, powers in enumerate(basis):
            for coefficient, target in action(*powers):
                if coefficient and target in index:
                    matrix[index[target], column] += coefficient
        return matrix

    d_v = operator(lambda i, j, k: [(j, (i, j - 1, k))] if j else [])
    minus_v_d_v = operator(lambda i, j, k: [(-j, (i, j, k))] if j else [])
    v_d_vv = operator(
        lambda i, j, k: [(j * (j - 1), (i, j - 1, k))] if j >= 2 else []
    )
    v_d_mm = operator(
        lambda i, j, k: [(i * (i - 1), (i - 2, j + 1, k))] if i >= 2 else []
    )
    v_d_mdv = operator(
        lambda i, j, k: [(i * j, (i - 1, j, k))] if i and j else []
    )
    v_d_z = operator(
        lambda i, j, k: [(k, (i, j + 1, k - 1))] if k else []
    )
    return basis, d_v, minus_v_d_v, v_d_vv, v_d_mm, v_d_mdv, v_d_z


def first_order(lbar, correction, maturity, payoff):
    """Duhamel term for a first-order generator perturbation."""
    size = len(payoff)
    block = np.zeros((2 * size, 2 * size))
    block[:size, :size] = lbar
    block[size:, size:] = lbar
    block[:size, size:] = correction
    semigroup = expm(maturity * block)
    return semigroup[:size, :size] @ payoff + semigroup[:size, size:] @ payoff


def reversed_generator(q, pi):
    return np.diag(1.0 / pi) @ q.T @ np.diag(pi)


def rate(errors):
    return math.log(errors[-2] / errors[-1], 2)


def cumulants(raw):
    """First four cumulants from raw moments E[Z],...,E[Z^4]."""
    m1, m2, m3, m4 = raw
    return np.array(
        [
            m1,
            m2 - m1**2,
            m3 - 3 * m2 * m1 + 2 * m1**3,
            m4 - 4 * m3 * m1 - 3 * m2**2 + 12 * m2 * m1**2 - 6 * m1**4,
        ]
    )


def cumulant_correction(raw, correction):
    """Directional derivative of the first four cumulants in the moment correction."""
    m1, m2, m3, m4 = raw
    d1, d2, d3, d4 = correction
    return np.array(
        [
            d1,
            d2 - 2 * m1 * d1,
            d3 - 3 * (d2 * m1 + m2 * d1) + 6 * m1**2 * d1,
            d4
            - 4 * (d3 * m1 + m3 * d1)
            - 6 * m2 * d2
            + 12 * (d2 * m1**2 + 2 * m2 * m1 * d1)
            - 24 * m1**3 * d1,
        ]
    )


def exact_integrated_mean(speed, maturity, v0, q0, pi, c, kappa):
    """Exact E[int_0^T v_s ds] from the closed regime-resolved first moment."""
    drift = speed * q0 - np.diag(kappa)
    stationary_components = np.linalg.solve(drift.T, -(pi * c))
    initial_components = pi * v0
    transient_integral = (initial_components - stationary_components) @ np.linalg.solve(
        drift, expm(maturity * drift) - np.eye(len(pi))
    )
    return (
        maturity * stationary_components.sum() + transient_integral.sum(),
        stationary_components.sum(),
    )


def spectral_mean_composite(speed, maturity, initial_components, q0, pi, c, kappa):
    """Exact mean and its exact-slope/principal-mode uniform composite."""
    drift = speed * q0 - np.diag(kappa)
    stationary_components = np.linalg.solve(drift.T, -(pi * c))
    difference = initial_components - stationary_components
    values, left, right = eig(drift, left=True, right=True)
    principal = np.argmax(values.real)
    eigenvalue = values[principal].real
    right_vector = right[:, principal].real
    left_vector = left[:, principal].real
    projection = np.outer(right_vector, left_vector) / (left_vector @ right_vector)
    exact = maturity * stationary_components.sum() + (
        difference @ np.linalg.solve(
            drift, expm(maturity * drift) - np.eye(len(pi))
        )
    ).sum()
    slow_integral = np.expm1(eigenvalue * maturity) / eigenvalue
    composite = (
        maturity * stationary_components.sum()
        + (difference @ projection @ np.ones(len(pi))) * slow_integral
    )
    return exact, composite, eigenvalue


def green_kubo_mean_composite(
    speed, maturity, initial_components, q0, pi, c, kappa
):
    """Uniform mean composite using the exact slope but explicit slow-mode data."""
    drift = speed * q0 - np.diag(kappa)
    stationary_components = np.linalg.solve(drift.T, -(pi * c))
    difference = initial_components - stationary_components
    centered_kappa = kappa - pi @ kappa
    corrector = group_inverse(q0) @ centered_kappa
    kappa_gk = -pi @ (centered_kappa * corrector)
    approximate_eigenvalue = -(pi @ kappa) + kappa_gk / speed
    approximate_right_vector = np.ones(len(pi)) + corrector / speed
    exact = maturity * stationary_components.sum() + (
        difference @ np.linalg.solve(
            drift, expm(maturity * drift) - np.eye(len(pi))
        )
    ).sum()
    slow_integral = np.expm1(approximate_eigenvalue * maturity) / (
        approximate_eigenvalue
    )
    composite = (
        maturity * stationary_components.sum()
        + (difference @ approximate_right_vector) * slow_integral
    )
    return exact, composite, approximate_eigenvalue, kappa_gk


def stationary_mean_coefficients(order, q0, pi, c, kappa):
    """Poisson recursion for the stationary first-moment expansion."""
    group = group_inverse(q0)
    kappa_matrix = np.diag(kappa)
    average_kappa = pi @ kappa
    rows = [(pi @ c / average_kappa) * pi]

    # At first order the affine forcing enters once.  At later orders the
    # preceding coefficient is simply propagated through the killing rate.
    centered = (rows[0] @ kappa_matrix - pi * c) @ group
    rows.append(centered - (centered @ kappa / average_kappa) * pi)
    for index in range(1, order):
        centered = rows[index] @ kappa_matrix @ group
        rows.append(centered - (centered @ kappa / average_kappa) * pi)

    assert abs(rows[0] @ kappa - pi @ c) < 2e-14
    assert all(abs(row @ kappa) < 2e-13 for row in rows[1:])
    coefficients = np.array([row.sum() for row in rows])
    return rows, coefficients


def truncated_stationary_mean_composite(
    speed, maturity, initial_components, q0, pi, c, kappa, order
):
    """Explicit slow-mode composite with a finite stationary expansion."""
    drift = speed * q0 - np.diag(kappa)
    exact_stationary = np.linalg.solve(drift.T, -(pi * c))
    exact = maturity * exact_stationary.sum() + (
        (initial_components - exact_stationary)
        @ np.linalg.solve(drift, expm(maturity * drift) - np.eye(len(pi)))
    ).sum()

    rows, coefficients = stationary_mean_coefficients(order, q0, pi, c, kappa)
    powers = speed ** np.arange(order + 1)
    stationary_approximation = sum(
        (row / powers[index] for index, row in enumerate(rows)),
        start=np.zeros(len(pi)),
    )
    slope_approximation = np.sum(coefficients / powers)
    centered_kappa = kappa - pi @ kappa
    corrector = group_inverse(q0) @ centered_kappa
    kappa_gk = -pi @ (centered_kappa * corrector)
    approximate_eigenvalue = -(pi @ kappa) + kappa_gk / speed
    slow_integral = np.expm1(approximate_eigenvalue * maturity) / (
        approximate_eigenvalue
    )
    composite = (
        maturity * slope_approximation
        + (initial_components - stationary_approximation)
        @ (np.ones(len(pi)) + corrector / speed)
        * slow_integral
    )
    return exact, composite, exact_stationary.sum(), slope_approximation


def verify_stationary_mean_expansion():
    """Check the Poisson recursion and its sharp growing-horizon scope."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    speeds = np.array([8, 16, 32, 64, 128], dtype=float)
    rows, coefficients = stationary_mean_coefficients(3, q0, pi, c, kappa)
    expected = np.array([1 / 14, -9 / 2450, 18 / 8575])
    assert np.max(np.abs(coefficients[:3] - expected)) < 2e-15

    slope_errors = {order: [] for order in (1, 2, 3)}
    window_errors = {order: [] for order in (1, 2, 3)}
    initial_components = pi * np.array([0.02, 0.08])
    for speed in speeds:
        drift = speed * q0 - np.diag(kappa)
        exact_stationary = np.linalg.solve(drift.T, -(pi * c))
        for order in (1, 2, 3):
            approximation = sum(
                coefficients[index] / speed**index
                for index in range(order + 1)
            )
            slope_errors[order].append(abs(exact_stationary.sum() - approximation))
            horizon = speed ** (order - 1)
            maturities = np.r_[0.0, np.geomspace(1e-8, horizon, 900)]
            errors = []
            for maturity in maturities:
                exact, composite, _, _ = truncated_stationary_mean_composite(
                    speed,
                    maturity,
                    initial_components,
                    q0,
                    pi,
                    c,
                    kappa,
                    order,
                )
                errors.append(abs(exact - composite))
            window_errors[order].append(max(errors))

    slope_rates = {order: rate(values) for order, values in slope_errors.items()}
    window_rates = {order: rate(values) for order, values in window_errors.items()}
    for order in (1, 2, 3):
        assert order + 0.75 < slope_rates[order] < order + 1.25
        assert 1.75 < window_rates[order] < 2.25

    # Repeat the recursion itself on a nonreversible chain, independently
    # checking the stationary solve and the predicted p+1 slope orders.
    q3 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi3 = stationary(q3)
    kappa3 = np.array([1.1, 2.3, 3.0])
    c3 = kappa3 * np.array([0.035, 0.080, 0.050])
    _, coefficients3 = stationary_mean_coefficients(3, q3, pi3, c3, kappa3)
    three_state_rates = []
    for order in (1, 2, 3):
        errors = []
        for speed in speeds:
            drift = speed * q3 - np.diag(kappa3)
            exact_slope = np.linalg.solve(drift.T, -(pi3 * c3)).sum()
            approximation = sum(
                coefficients3[index] / speed**index
                for index in range(order + 1)
            )
            errors.append(abs(exact_slope - approximation))
        three_state_rates.append(rate(errors))
    assert all(
        order + 0.7 < value < order + 1.3
        for order, value in zip((1, 2, 3), three_state_rates)
    )

    print("3. Poisson-recursive stationary mean on growing horizons")
    print(
        "   two-state coefficients mu_0,mu_1,mu_2,mu_3: "
        + " ".join(f"{value:.10f}" for value in coefficients)
    )
    print(
        "   slope remainder rates p=1,2,3: "
        + "/".join(f"{slope_rates[order]:.3f}" for order in (1, 2, 3))
    )
    print(
        "   composite rates on T<=m^(p-1): "
        + "/".join(f"{window_rates[order]:.3f}" for order in (1, 2, 3))
    )
    print(
        "   nonreversible slope remainder rates: "
        + "/".join(f"{value:.3f}" for value in three_state_rates)
    )
    return slope_rates, window_rates, three_state_rates


def verify_uniform_spectral_mean():
    """Check uniform O(m^-2), and O(m^-3) without initial regime dependence."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    speeds = (8, 16, 32, 64, 128)
    independent_errors = []
    conditional_errors = []
    explicit_independent_errors = []
    explicit_conditional_errors = []
    eigenvalue_errors = []
    eigenvalues = []
    for speed in speeds:
        maturities = np.r_[0.0, np.geomspace(1e-8, float(speed**2), 1600)]
        errors = {"independent": [], "conditional": []}
        explicit_errors = {"independent": [], "conditional": []}
        initial = {
            "independent": pi * 0.05,
            "conditional": pi * np.array([0.02, 0.08]),
        }
        for maturity in maturities:
            for label, initial_components in initial.items():
                exact, composite, eigenvalue = spectral_mean_composite(
                    speed,
                    maturity,
                    initial_components,
                    q0,
                    pi,
                    c,
                    kappa,
                )
                errors[label].append(abs(exact - composite))
                exact_explicit, explicit, approximate_eigenvalue, kappa_gk = (
                    green_kubo_mean_composite(
                        speed,
                        maturity,
                        initial_components,
                        q0,
                        pi,
                        c,
                        kappa,
                    )
                )
                assert abs(exact - exact_explicit) < 2e-14
                explicit_errors[label].append(abs(exact - explicit))
        independent_errors.append(max(errors["independent"]))
        conditional_errors.append(max(errors["conditional"]))
        explicit_independent_errors.append(max(explicit_errors["independent"]))
        explicit_conditional_errors.append(max(explicit_errors["conditional"]))
        eigenvalues.append(eigenvalue)
        eigenvalue_errors.append(abs(eigenvalue - approximate_eigenvalue))

    independent_rate = rate(independent_errors)
    conditional_rate = rate(conditional_errors)
    assert 2.8 < independent_rate < 3.2
    assert 1.8 < conditional_rate < 2.2
    explicit_independent_rate = rate(explicit_independent_errors)
    explicit_conditional_rate = rate(explicit_conditional_errors)
    assert 1.8 < explicit_independent_rate < 2.2
    assert 1.8 < explicit_conditional_rate < 2.2
    # This symmetric example has an additional cancellation: the eigenvalue
    # remainder is third rather than merely second order.
    assert 2.8 < rate(eigenvalue_errors) < 3.2
    print("2. slope-and-principal-mode uniform mean composite")
    print(
        "   independent initial variance: "
        + " ".join(f"{value:.3e}" for value in independent_errors)
        + f"  rate {independent_rate:.3f}"
    )
    print(
        "   regime-dependent initial mean: "
        + " ".join(f"{value:.3e}" for value in conditional_errors)
        + f"  rate {conditional_rate:.3f}"
    )
    print(f"   slow eigenvalue at m=128: {eigenvalues[-1]:.10f}")
    print("   exact slope plus explicit Green--Kubo slow mode")
    print(
        "      independent initial variance: "
        + " ".join(f"{value:.3e}" for value in explicit_independent_errors)
        + f"  rate {explicit_independent_rate:.3f}"
    )
    print(
        "      regime-dependent initial mean: "
        + " ".join(f"{value:.3e}" for value in explicit_conditional_errors)
        + f"  rate {explicit_conditional_rate:.3f}"
    )
    print(
        f"      lambda_hat=-bar(kappa)+K_kappa,kappa/m; "
        f"K_kappa,kappa={kappa_gk:.10f}; eigenvalue rate "
        f"{rate(eigenvalue_errors):.3f}"
    )

    # Recheck the explicit formula on a nonreversible three-state chain.
    q3 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi3 = stationary(q3)
    kappa3 = np.array([1.1, 2.3, 3.0])
    c3 = kappa3 * np.array([0.035, 0.080, 0.050])
    three_state_errors = {"independent": [], "conditional": []}
    initial3 = {
        "independent": pi3 * 0.05,
        "conditional": pi3 * np.array([0.02, 0.08, 0.05]),
    }
    for speed in speeds:
        maturities = np.r_[0.0, np.geomspace(1e-8, float(speed**2), 1600)]
        for label, initial_components in initial3.items():
            errors3 = []
            for maturity in maturities:
                exact, composite, _, _ = green_kubo_mean_composite(
                    speed,
                    maturity,
                    initial_components,
                    q3,
                    pi3,
                    c3,
                    kappa3,
                )
                errors3.append(abs(exact - composite))
            three_state_errors[label].append(max(errors3))
    three_state_rates = {
        label: rate(values) for label, values in three_state_errors.items()
    }
    assert 1.8 < three_state_rates["independent"] < 2.2
    assert 1.8 < three_state_rates["conditional"] < 2.2
    print(
        "      nonreversible three-state rates independent/conditional: "
        f"{three_state_rates['independent']:.3f}/"
        f"{three_state_rates['conditional']:.3f}"
    )
    return (
        independent_rate,
        conditional_rate,
        explicit_independent_rate,
        explicit_conditional_rate,
    )


def verify_long_maturity_nonuniformity():
    """Certify the sharp failure of an absolute maturity-uniform O(m^-2) bound."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    # xi=(0.2,0.2) makes both statewise Feller inequalities strict.  Vol-of-vol
    # does not enter this first-moment obstruction.
    assert np.all(2 * c > np.array([0.2, 0.2]) ** 2)

    mu0 = 1.0 / 14.0
    mu1 = -9.0 / 2450.0
    joint_scale_limit = 18.0 / 8575.0
    speeds = (8, 16, 32, 64, 128)
    residuals = []
    slope_errors = []
    for speed in speeds:
        maturity = float(speed**2)
        integrated_mean, stationary_mean = exact_integrated_mean(
            speed, maturity, mu0, q0, pi, c, kappa
        )
        closed_mean = (25.0 * speed + 13.0) / (50.0 * (7.0 * speed + 4.0))
        slope_remainder = 18.0 / (1225.0 * speed * (7.0 * speed + 4.0))
        assert abs(stationary_mean - closed_mean) < 1e-15
        assert abs(stationary_mean - mu0 - mu1 / speed - slope_remainder) < 1e-15
        residuals.append(integrated_mean - mu0 * maturity - mu1 * maturity / speed)
        slope_errors.append(abs(speed**2 * slope_remainder - joint_scale_limit))

    assert residuals[-1] > joint_scale_limit
    assert abs(residuals[-1] - joint_scale_limit) < 1.2e-5
    assert 0.9 < rate(slope_errors) < 1.1
    print("1. long-maturity obstruction under simultaneous switching")
    print("   mu_m=(25m+13)/(50(7m+4))")
    print("   mu_m-mu_0-mu_1/m=18/(1225m(7m+4)) > 0")
    print(
        "   T=m^2 corrected residual: "
        + " ".join(f"{value:.10f}" for value in residuals)
        + f"  limit {joint_scale_limit:.10f}"
    )
    return residuals


def main():
    long_maturity_residuals = verify_long_maturity_nonuniformity()
    uniform_mean_rates = verify_uniform_spectral_mean()
    expansion_rates = verify_stationary_mean_expansion()
    q0 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi = stationary(q0)
    kappa = np.array([1.1, 2.3, 3.0])
    theta = np.array([0.035, 0.080, 0.050])
    c = kappa * theta
    xi = np.array([0.20, 0.25, 0.22])
    variance = xi**2
    assert np.all(2 * c > variance)

    a_c, a_kappa, a_variance, multiply_v = polynomial_operators()
    operators = [a_c, a_kappa, a_variance]
    lbar = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
    )
    payoff = np.array([0.7, -0.2, 0.3, -0.1, 0.04, -0.005])
    maturity = 0.7

    k0 = gk(q0, [c, kappa, variance])
    identity = np.eye(len(payoff))
    derivative = a_c
    derivative2 = derivative @ derivative
    derivative3 = derivative2 @ derivative
    derivative4 = derivative3 @ derivative
    explicit = (
        (k0[1, 1] * multiply_v - k0[0, 1] * identity) @ derivative
        + (
            (k0[0, 0] + 0.5 * k0[0, 2]) * identity
            - (k0[0, 1] + k0[1, 0]) * multiply_v
            - (0.5 * k0[1, 2] + k0[2, 1]) * multiply_v
            + k0[1, 1] * (multiply_v @ multiply_v)
        )
        @ derivative2
        + (
            0.5 * (k0[0, 2] + k0[2, 0] + k0[2, 2]) * multiply_v
            - 0.5 * (k0[1, 2] + k0[2, 1]) * (multiply_v @ multiply_v)
        )
        @ derivative3
        + 0.25 * k0[2, 2] * (multiply_v @ multiply_v) @ derivative4
    )
    abstract = effective_generator(np.zeros_like(lbar), operators, k0)
    formula_error = np.max(abs(explicit - abstract))
    assert formula_error < 2e-15

    print("4. switched coordinates and the explicit CIR corrector")
    print(f"   stationary law {pi}")
    print(f"   kappa*theta {c}; average theta is {(pi @ c) / (pi @ kappa):.8f}")
    print(f"   xi^2 {variance}; effective xi is {math.sqrt(pi @ variance):.8f}")
    print(f"   operator formula error {formula_error:.2e}")
    print(f"   K(c,kappa,xi^2)=\n{k0}")

    errors = {"averaged": [], "symmetric": [], "full": []}
    forward_values = []
    for speed in SPEEDS:
        generator = full_generator(speed * q0, lbar, operators, [c, kappa, variance])
        truth = np.kron(pi, np.eye(len(payoff))) @ (
            expm(maturity * generator) @ np.kron(np.ones(len(pi)), payoff)
        )
        full_correction = abstract / speed
        symmetric_correction = effective_generator(
            np.zeros_like(lbar), operators, k0 / speed, "sym"
        )
        averaged = expm(maturity * lbar) @ payoff
        full = first_order(lbar, full_correction, maturity, payoff)
        symmetric = first_order(lbar, symmetric_correction, maturity, payoff)
        errors["averaged"].append(np.linalg.norm(truth - averaged))
        errors["symmetric"].append(np.linalg.norm(truth - symmetric))
        errors["full"].append(np.linalg.norm(truth - full))
        forward_values.append(truth)

    print("5. polynomial semigroup check")
    for name, values in errors.items():
        print(
            f"   {name:9s}: "
            + " ".join(f"{value:.3e}" for value in values)
            + f"  rate {rate(values):.3f}"
        )
    assert 0.8 < rate(errors["averaged"]) < 1.2
    assert 0.8 < rate(errors["symmetric"]) < 1.2
    assert 1.8 < rate(errors["full"]) < 2.2

    q_reverse = reversed_generator(q0, pi)
    k_reverse = gk(q_reverse, [c, kappa, variance])
    assert np.max(abs(k_reverse - k0.T)) < 2e-14
    reverse_values = []
    direction_errors = []
    for speed, forward_value in zip(SPEEDS, forward_values):
        generator = full_generator(
            speed * q_reverse, lbar, operators, [c, kappa, variance]
        )
        reverse_value = np.kron(pi, np.eye(len(payoff))) @ (
            expm(maturity * generator) @ np.kron(np.ones(len(pi)), payoff)
        )
        reverse_values.append(reverse_value)
        direction_correction = effective_generator(
            np.zeros_like(lbar), operators, (k0 - k0.T) / speed
        )
        predicted = (
            first_order(lbar, direction_correction, maturity, payoff)
            - expm(maturity * lbar) @ payoff
        )
        direction_errors.append(
            np.linalg.norm((forward_value - reverse_value) - predicted)
        )
    direction_rate = rate(direction_errors)
    assert 1.8 < direction_rate < 2.2
    anti = 0.5 * (k0 - k0.T)
    print("6. cycle reversal")
    print(
        "   antisymmetric coefficients "
        f"c/kappa={anti[0, 1]:+.10f}, c/xi^2={anti[0, 2]:+.10f}, "
        f"kappa/xi^2={anti[1, 2]:+.10f}"
    )
    print(
        "   corrected forward-minus-reverse residual: "
        + " ".join(f"{value:.3e}" for value in direction_errors)
        + f"  rate {direction_rate:.3f}"
    )
    basis, a_c_z, a_kappa_z, v_d_vv, v_d_z = integrated_variance_operators()
    a_variance_z = 0.5 * v_d_vv
    lbar_z = (
        (pi @ c) * a_c_z
        + (pi @ kappa) * a_kappa_z
        + (pi @ variance) * a_variance_z
        + v_d_z
    )
    correction_z = effective_generator(
        np.zeros_like(lbar_z), [a_c_z, a_kappa_z, a_variance_z], k0
    )
    initial = np.array([0.06**i * 0.0**j for i, j in basis])
    moment_payoffs = []
    for power in range(1, 5):
        payoff_z = np.zeros(len(basis))
        payoff_z[basis.index((0, power))] = 1.0
        moment_payoffs.append(payoff_z)

    averaged_moments = np.array(
        [initial @ (expm(maturity * lbar_z) @ payoff_z) for payoff_z in moment_payoffs]
    )
    corrected_unit = np.array(
        [
            initial @ first_order(lbar_z, correction_z, maturity, payoff_z)
            for payoff_z in moment_payoffs
        ]
    )
    moment_correction = corrected_unit - averaged_moments
    averaged_cumulants = cumulants(averaged_moments)
    cumulant_first = cumulant_correction(averaged_moments, moment_correction)
    martingale_average = 3 * averaged_cumulants[1]
    martingale_first = 3 * cumulant_first[1]
    logreturn_average = (
        3 * averaged_cumulants[1]
        + 1.5 * averaged_cumulants[2]
        + averaged_cumulants[3] / 16
    )
    logreturn_first = (
        3 * cumulant_first[1] + 1.5 * cumulant_first[2] + cumulant_first[3] / 16
    )

    cumulant_errors = {"martingale": [], "log return": []}
    averaged_errors = {"martingale": [], "log return": []}
    for speed in SPEEDS:
        generator = full_generator(
            speed * q0,
            lbar_z,
            [a_c_z, a_kappa_z, a_variance_z],
            [c, kappa, variance],
        )
        exact_moments = []
        for payoff_z in moment_payoffs:
            coefficients = np.kron(pi, np.eye(len(basis))) @ (
                expm(maturity * generator) @ np.kron(np.ones(len(pi)), payoff_z)
            )
            exact_moments.append(initial @ coefficients)
        exact_cumulants = cumulants(np.array(exact_moments))
        exact_martingale = 3 * exact_cumulants[1]
        exact_logreturn = (
            3 * exact_cumulants[1]
            + 1.5 * exact_cumulants[2]
            + exact_cumulants[3] / 16
        )
        averaged_errors["martingale"].append(abs(exact_martingale - martingale_average))
        averaged_errors["log return"].append(abs(exact_logreturn - logreturn_average))
        cumulant_errors["martingale"].append(
            abs(exact_martingale - martingale_average - martingale_first / speed)
        )
        cumulant_errors["log return"].append(
            abs(exact_logreturn - logreturn_average - logreturn_first / speed)
        )

    print("7. integrated-variance cumulants")
    print(
        f"   kappa4(M): averaged {martingale_average:.10f}, coefficient {martingale_first:+.10f}"
    )
    print(
        f"   kappa4(X): averaged {logreturn_average:.10f}, coefficient {logreturn_first:+.10f}"
    )
    for name in cumulant_errors:
        print(
            f"   {name:10s}: averaged rate {rate(averaged_errors[name]):.3f}; "
            f"corrected residual "
            + " ".join(f"{value:.3e}" for value in cumulant_errors[name])
            + f"  rate {rate(cumulant_errors[name]):.3f}"
        )
        assert 0.8 < rate(averaged_errors[name]) < 1.2
        assert 1.8 < rate(cumulant_errors[name]) < 2.2

    (
        basis_x,
        a_c_x,
        a_kappa_x,
        v_d_vv_x,
        v_d_mm,
        v_d_mdv,
        v_d_z_x,
    ) = leveraged_operators()
    a_variance_x = 0.5 * v_d_vv_x
    operators_x = [a_c_x, a_kappa_x, a_variance_x, v_d_mdv]
    initial_x = np.array([0.0**i * 0.06**j * 0.0**k for i, j, k in basis_x])
    payoff_x = []
    for power in range(1, 5):
        payoff = np.zeros(len(basis_x))
        for z_power in range(power + 1):
            m_power = power - z_power
            payoff[basis_x.index((m_power, 0, z_power))] = (
                math.comb(power, z_power) * (-0.5) ** z_power
            )
        payoff_x.append(payoff)

    def leverage_case(eta):
        """Fourth-cumulant expansion for cross coefficient eta=rho*xi."""
        features_x = [c, kappa, variance, eta]
        k0_x = gk(q0, features_x)
        lbar_x = (
            (pi @ c) * a_c_x
            + (pi @ kappa) * a_kappa_x
            + (pi @ variance) * a_variance_x
            + 0.5 * v_d_mm
            + (pi @ eta) * v_d_mdv
            + v_d_z_x
        )
        correction_x = effective_generator(
            np.zeros_like(lbar_x), operators_x, k0_x
        )
        averaged_raw_x = np.array(
            [initial_x @ (expm(maturity * lbar_x) @ payoff) for payoff in payoff_x]
        )
        corrected_unit_x = np.array(
            [
                initial_x @ first_order(lbar_x, correction_x, maturity, payoff)
                for payoff in payoff_x
            ]
        )
        correction_raw_x = corrected_unit_x - averaged_raw_x
        averaged_kappa_x = cumulants(averaged_raw_x)[3]
        correction_kappa_x = cumulant_correction(
            averaged_raw_x, correction_raw_x
        )[3]

        averaged_x_errors = []
        corrected_x_errors = []
        for speed in SPEEDS:
            generator_x = full_generator(
                speed * q0, lbar_x, operators_x, features_x
            )
            exact_raw_x = []
            for payoff in payoff_x:
                coefficients = np.kron(pi, np.eye(len(basis_x))) @ (
                    expm(maturity * generator_x)
                    @ np.kron(np.ones(len(pi)), payoff)
                )
                exact_raw_x.append(initial_x @ coefficients)
            exact_kappa_x = cumulants(np.array(exact_raw_x))[3]
            averaged_x_errors.append(abs(exact_kappa_x - averaged_kappa_x))
            corrected_x_errors.append(
                abs(exact_kappa_x - averaged_kappa_x - correction_kappa_x / speed)
            )
        result = (
            averaged_kappa_x,
            correction_kappa_x,
            averaged_x_errors,
            corrected_x_errors,
        )
        assert 0.8 < rate(averaged_x_errors) < 1.2
        assert 1.8 < rate(corrected_x_errors) < 2.2
        return result, k0_x

    leverage_results = {}
    for rho in (0.0, -0.7):
        leverage_results[f"rho={rho:+.1f}"] = leverage_case(rho * xi)[0]

    # For fixed rho, the minimal eta feature is exactly equivalent to using
    # xi as the feature and absorbing rho into the operator.
    rho_fixed = -0.7
    correction_eta_fixed = effective_generator(
        np.zeros_like(a_c_x),
        operators_x,
        gk(q0, [c, kappa, variance, rho_fixed * xi]),
    )
    correction_xi_fixed = effective_generator(
        np.zeros_like(a_c_x),
        [a_c_x, a_kappa_x, a_variance_x, rho_fixed * v_d_mdv],
        gk(q0, [c, kappa, variance, xi]),
    )
    fixed_coordinate_error = np.max(
        abs(correction_eta_fixed - correction_xi_fixed)
    )
    assert fixed_coordinate_error < 2e-15

    rho_switch = np.array([-0.80, -0.25, 0.35])
    eta_switch = rho_switch * xi
    switched_result, k0_eta = leverage_case(eta_switch)
    leverage_results["switched rho"] = switched_result
    effective_xi = math.sqrt(pi @ variance)
    effective_rho = (pi @ eta_switch) / effective_xi
    assert np.all(np.abs(rho_switch) <= 1.0)
    assert abs(effective_rho) <= 1.0

    # Reversal transposes the four-feature Green--Kubo matrix.  The new
    # directional terms are K^anti_{c,eta}[A_c,A_eta] and
    # K^anti_{a,eta}[A_a,A_eta]; A_kappa commutes with A_eta.
    k0_eta_reverse = gk(q_reverse, [c, kappa, variance, eta_switch])
    eta_reversal_error = np.max(abs(k0_eta_reverse - k0_eta.T))
    assert eta_reversal_error < 2e-14
    commutator_c_eta = a_c_x @ v_d_mdv - v_d_mdv @ a_c_x
    commutator_kappa_eta = a_kappa_x @ v_d_mdv - v_d_mdv @ a_kappa_x
    commutator_a_eta = a_variance_x @ v_d_mdv - v_d_mdv @ a_variance_x
    index_x = {powers: position for position, powers in enumerate(basis_x)}
    expected_c_eta = np.zeros_like(commutator_c_eta)
    expected_a_eta = np.zeros_like(commutator_a_eta)
    for column, (m_power, v_power, z_power) in enumerate(basis_x):
        if m_power and v_power:
            target = (m_power - 1, v_power - 1, z_power)
            expected_c_eta[index_x[target], column] = m_power * v_power
        if m_power and v_power >= 2:
            target = (m_power - 1, v_power - 1, z_power)
            expected_a_eta[index_x[target], column] = (
                0.5 * m_power * v_power * (v_power - 1)
            )
    assert np.max(abs(commutator_c_eta - expected_c_eta)) < 2e-14
    assert np.linalg.norm(commutator_kappa_eta) < 2e-14
    assert np.max(abs(commutator_a_eta - expected_a_eta)) < 2e-14
    commutator_error = max(
        np.max(abs(commutator_c_eta - expected_c_eta)),
        np.linalg.norm(commutator_kappa_eta),
        np.max(abs(commutator_a_eta - expected_a_eta)),
    )

    independent = leverage_results["rho=+0.0"]
    assert abs(independent[0] - logreturn_average) < 2e-15
    assert abs(independent[1] - logreturn_first) < 2e-15
    print("8. simultaneous switching with regime-dependent leverage")
    print(
        f"   rho states {rho_switch}; eta=rho*xi {eta_switch}; "
        f"effective rho {effective_rho:+.8f}"
    )
    print(
        f"   fixed-rho coordinate error {fixed_coordinate_error:.2e}; "
        f"commutator error {commutator_error:.2e}; "
        f"reversal error {eta_reversal_error:.2e}"
    )
    for label, result in leverage_results.items():
        average, coefficient, average_errors, corrected_errors = result
        print(
            f"   {label:12s}: averaged {average:.10f}, coefficient {coefficient:+.10f}; "
            f"rates {rate(average_errors):.3f}/{rate(corrected_errors):.3f}"
        )
        print("      corrected residual " + " ".join(f"{x:.3e}" for x in corrected_errors))

    print(
        f"PASS: full rate {rate(errors['full']):.3f}, direction rate {direction_rate:.3f}, "
        f"cumulant rates {rate(cumulant_errors['martingale']):.3f}/"
        f"{rate(cumulant_errors['log return']):.3f}, leverage rate "
        f"{rate(leverage_results['switched rho'][3]):.3f}, long-maturity residual "
        f"{long_maturity_residuals[-1]:.10f}, uniform mean rates "
        f"{uniform_mean_rates[0]:.3f}/{uniform_mean_rates[1]:.3f}, "
        f"growing-window rate {expansion_rates[1][3]:.3f}"
    )


if __name__ == "__main__":
    main()
