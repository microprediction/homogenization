"""Certificate for the maturity-uniform two-regime approximation.

The reference calculation integrates the exact scalar equations induced by the
original two-by-two pricing system.  A direct integration of that system on a
finite interval and a constant-forcing closed form provide separate checks.
The final calculation lets the switching generator itself vary and isolates
the loss caused by its moving invariant distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import eig, expm
from scipy.optimize import root_scalar


KAPPA = 1.0
THETA = np.array([0.08, 0.02])
VARIANCE = np.array([0.004, 0.001])
THETA_BAR, THETA_DELTA = np.mean(THETA), (THETA[0] - THETA[1]) / 2
VAR_BAR, VAR_DELTA = np.mean(VARIANCE), (VARIANCE[0] - VARIANCE[1]) / 2


def forcing(t: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the stationary mean and half-difference of the Vasicek forcing."""
    b = -np.expm1(-KAPPA * np.asarray(t)) / KAPPA
    mean = -KAPPA * THETA_BAR * b + 0.5 * VAR_BAR * b**2
    delta = -KAPPA * THETA_DELTA * b + 0.5 * VAR_DELTA * b**2
    return mean, delta


G_BAR_INF = -THETA_BAR + VAR_BAR / (2 * KAPPA**2)
DELTA_INF = -THETA_DELTA + VAR_DELTA / (2 * KAPPA**2)


def solve_log_system(m: float, times: np.ndarray) -> np.ndarray:
    """Solve for log stationary amplitude, ratio, and the two transient integrals."""

    def rhs(t: float, y: np.ndarray) -> list[float]:
        mean, delta = forcing(t)
        _, ratio, _, _ = y
        return [
            mean + delta * ratio,
            delta - 2 * m * ratio - delta * ratio**2,
            mean - G_BAR_INF,
            delta**2 - DELTA_INF**2,
        ]

    ans = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        np.zeros(4),
        t_eval=times,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert ans.success
    return ans.y


def composite(m: float, times: np.ndarray, solution: np.ndarray, eta: float = 0.0) -> np.ndarray:
    """Uniform log-amplitude approximation for start weight eta=2p-1."""
    omega = np.sqrt(m**2 + DELTA_INF**2)
    principal = G_BAR_INF + omega - m
    _, delta = forcing(times)
    boundary_ratio = delta / (2 * m)  # delta(0)=0 in the Vasicek example
    return (
        principal * times
        + solution[2]
        + solution[3] / (2 * m)
        + np.log1p(eta * boundary_ratio)
    )


def exact_log(solution: np.ndarray, eta: float = 0.0) -> np.ndarray:
    return solution[0] + np.log1p(eta * solution[1])


def theorem_bound(m: float, eta: float = 0.0) -> float:
    """Explicit bound from the theorem, specialized to this monotone forcing."""
    g = abs(DELTA_INF)
    variation = g
    l1_tail = THETA_DELTA - VAR_DELTA / KAPPA**2 + VAR_DELTA / (4 * KAPPA**2)
    stationary = g * (g + variation) / (4 * m * (m - g)) + g**3 * l1_tail / (2 * m**3)
    if eta == 0:
        return stationary
    derivative_bound = KAPPA * THETA_DELTA
    start = (derivative_bound / (4 * m**2) + g**3 / (8 * m**3)) / (1 - g / m)
    return stationary + abs(eta) * start


def check_direct_system() -> None:
    """Check the Riccati variables against the original two-component system."""
    m = 3.0
    times = np.linspace(0, 20, 401)
    reduced = solve_log_system(m, times)

    def rhs(t: float, a: np.ndarray) -> np.ndarray:
        mean, delta = forcing(t)
        matrix = np.array([[mean + delta - m, m], [m, mean - delta - m]])
        return matrix @ a

    direct = solve_ivp(
        rhs,
        (0.0, 20.0),
        np.ones(2),
        t_eval=times,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    ).y
    direct_log = np.log(np.mean(direct, axis=0))
    direct_ratio = (direct[0] - direct[1]) / (direct[0] + direct[1])
    assert np.max(np.abs(direct_log - reduced[0])) < 2e-11
    # The subtraction in the direct ratio loses digits once the two components
    # become nearly equal; this tolerance is still three orders below the
    # approximation errors measured below.
    assert np.max(np.abs(direct_ratio - reduced[1])) < 3e-9


def check_constant_projection() -> None:
    """Verify the exact far-end spectral projection for constant forcing."""
    m, delta, mean, t = 2.3, -0.071, -0.04, 40.0
    omega = np.sqrt(m**2 + delta**2)
    slow = mean + omega - m
    exact = np.exp((mean - m) * t) * (
        np.cosh(omega * t) + (m / omega) * np.sinh(omega * t)
    )
    projection = (m + omega) / (2 * omega)
    fast_remainder = (omega - m) / (2 * omega) * np.exp(-2 * omega * t)
    reconstructed = np.exp(slow * t) * (projection + fast_remainder)
    assert abs(exact / reconstructed - 1) < 3e-14


def defective_generator() -> np.ndarray:
    """Irreducible generator with a Jordan block at eigenvalue -1."""
    projection = np.ones((3, 3)) / 3
    u = np.array([1.0, -1.0, 0.0])
    v = np.array([1.0, 1.0, -2.0])
    return projection - np.eye(3) + 0.1 * np.outer(u, v)


def dominant_projection(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    """Perron eigenvalue and Riesz projection of an irreducible Metzler matrix."""
    values, left, right = eig(matrix, left=True, right=True)
    index = int(np.argmax(values.real))
    assert abs(values[index].imag) < 2e-12
    eigenvalue = float(values[index].real)
    left_vector = left[:, index].real
    right_vector = right[:, index].real
    projection = np.outer(right_vector, left_vector) / (left_vector @ right_vector)
    return eigenvalue, projection


def check_general_constant_forcing() -> None:
    """Maturity-uniform stationary theorem on a defective three-state chain."""
    Q = defective_generator()
    assert np.min(Q - np.diag(np.diag(Q))) >= 0
    assert np.max(np.abs(Q.sum(axis=1))) < 2e-15
    # Q+I has rank two but its square has rank one: eigenvalue -1 has
    # algebraic multiplicity two and geometric multiplicity one.
    assert np.linalg.matrix_rank(Q + np.eye(3), tol=2e-13) == 2
    assert np.linalg.matrix_rank((Q + np.eye(3)) @ (Q + np.eye(3)), tol=2e-13) == 1

    pi = np.ones(3) / 3
    one = np.ones(3)
    forcing_values = np.array([-0.2, 0.3, 0.6])
    forcing_matrix = np.diag(forcing_values)
    stationary_projection = np.outer(one, pi)
    group_inverse = np.linalg.inv(Q - stationary_projection) + stationary_projection
    centered = forcing_values - pi @ forcing_values
    green_kubo = -pi @ (centered * (group_inverse @ centered))

    print("\nConstant forcing on a defective three-state chain")
    print(" m      principal eigenvalue   projection weight   max |log s-Lambda*T|   m^2 error")
    errors = []
    eigenvalue_errors = []
    for m in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        matrix = m * Q + forcing_matrix
        principal, projection = dominant_projection(matrix)
        weight = float(pi @ projection @ one)
        assert weight > 0
        times = np.unique(
            np.r_[0.0, 0.05 / m, 0.2 / m, 1 / m, 5 / m, 0.1, 1.0, m, m**2]
        )
        log_residuals = []
        projection_residuals = []
        for time in times:
            scaled = float(
                pi @ expm((matrix - principal * np.eye(3)) * time) @ one
            )
            assert scaled > 0
            log_residuals.append(abs(np.log(scaled)))
            projection_residuals.append(abs(np.log(scaled / weight)))
        error = max(log_residuals)
        errors.append(error)
        eigenvalue_errors.append(
            abs(principal - (pi @ forcing_values + green_kubo / m))
        )
        assert max(projection_residuals) < 0.16 / m**2
        print(
            f"{m:3.0f}      {principal:16.12f}    {weight:16.12f}"
            f"       {error:16.9e}   {m*m*error:12.8f}"
        )

    rates = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    eigenvalue_rates = np.log2(
        np.asarray(eigenvalue_errors[:-1]) / np.asarray(eigenvalue_errors[1:])
    )
    assert rates[-1] > 1.98
    assert eigenvalue_rates[-1] > 1.95
    print(f"stationary uniform-error rate: {rates[-1]:.6f}")
    print(f"first-order eigenvalue residual rate: {eigenvalue_rates[-1]:.6f}")


def moving_spectral_data(
    m: float,
    time: float,
    limiting_forcing: np.ndarray,
    initial_forcing: np.ndarray | None = None,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Instantaneous Perron data in the gauge pi @ right == 1.

    The connection is left @ right_derivative.  The derivative is obtained
    from the bordered eigenvector equation, independently of finite
    differences or an eigenvector phase convention.
    """
    Q = defective_generator()
    pi = np.ones(3) / 3
    if initial_forcing is None:
        initial_forcing = np.zeros(3)
    alpha = -np.expm1(-time)
    alpha_derivative = np.exp(-time)
    forcing_values = initial_forcing + alpha * (
        limiting_forcing - initial_forcing
    )
    forcing_derivative = alpha_derivative * (
        limiting_forcing - initial_forcing
    )
    matrix = m * Q + np.diag(forcing_values)
    matrix_derivative = np.diag(forcing_derivative)

    values, left, right = eig(matrix, left=True, right=True)
    index = int(np.argmax(values.real))
    assert abs(values[index].imag) < 2e-12
    eigenvalue = float(values[index].real)
    right_vector = right[:, index].real
    right_vector /= pi @ right_vector
    left_vector = left[:, index].real
    left_vector /= left_vector @ right_vector

    eigenvalue_derivative = float(
        left_vector @ matrix_derivative @ right_vector
    )
    derivative_rhs = -(
        matrix_derivative - eigenvalue_derivative * np.eye(3)
    ) @ right_vector
    bordered = np.block(
        [
            [matrix - eigenvalue * np.eye(3), right_vector[:, None]],
            [pi[None, :], np.zeros((1, 1))],
        ]
    )
    right_derivative = np.linalg.solve(
        bordered, np.r_[derivative_rhs, 0.0]
    )[:3]
    connection = float(left_vector @ right_derivative)
    return eigenvalue, connection, right_vector, left_vector


def check_general_time_dependent_forcing() -> None:
    """Moving-projection theorem on the defective three-state chain."""
    Q = defective_generator()
    pi = np.ones(3) / 3
    one = np.ones(3)
    limiting_forcing = np.array([-0.6, -0.2, -0.4])

    # Check the bordered derivative against a phase-fixed centered difference.
    m_check, time_check, step = 7.0, 0.8, 2e-5
    _, connection, right_vector, _ = moving_spectral_data(
        m_check, time_check, limiting_forcing
    )
    right_minus = moving_spectral_data(
        m_check, time_check - step, limiting_forcing
    )[2]
    right_plus = moving_spectral_data(
        m_check, time_check + step, limiting_forcing
    )[2]
    right_derivative_fd = (right_plus - right_minus) / (2 * step)
    matrix = m_check * Q + np.diag(
        -np.expm1(-time_check) * limiting_forcing
    )
    _, left, right = eig(matrix, left=True, right=True)
    index = int(np.argmax(_.real))
    left_vector = left[:, index].real
    left_vector /= left_vector @ right_vector
    # Rescale the left vector to pair with the pi-normalized right vector.
    left_vector *= pi @ right_vector
    assert np.max(np.abs(pi @ right_derivative_fd)) < 2e-11
    assert abs(connection - left_vector @ right_derivative_fd) < 2e-10
    assert np.max(np.abs(pi @ right_vector - 1.0)) < 2e-14

    # Independent check of the normalized equations against the original
    # nonautonomous three-component pricing system.
    direct_times = np.linspace(0.0, 5.0, 101)
    direct_m = 3.0

    def direct_rhs(time: float, amplitude: np.ndarray) -> np.ndarray:
        alpha = -np.expm1(-time)
        return (
            direct_m * Q + np.diag(alpha * limiting_forcing)
        ) @ amplitude

    direct = solve_ivp(
        direct_rhs,
        (0.0, 5.0),
        one,
        t_eval=direct_times,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert direct.success

    def normalized_rhs(time: float, state: np.ndarray) -> np.ndarray:
        normalized = np.r_[state[:2], 3.0 - state[0] - state[1]]
        alpha = -np.expm1(-time)
        matrix = direct_m * Q + np.diag(alpha * limiting_forcing)
        log_derivative = float(pi @ matrix @ normalized)
        derivative = matrix @ normalized - log_derivative * normalized
        return np.r_[derivative[:2], log_derivative]

    normalized = solve_ivp(
        normalized_rhs,
        (0.0, 5.0),
        np.array([1.0, 1.0, 0.0]),
        t_eval=direct_times,
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert normalized.success
    reconstructed = np.vstack(
        [
            normalized.y[0],
            normalized.y[1],
            3.0 - normalized.y[0] - normalized.y[1],
        ]
    ) * np.exp(normalized.y[2])
    direct_error = float(np.max(np.abs(reconstructed - direct.y)))
    assert direct_error < 1e-11
    print(f"moving-system direct ODE error: {direct_error:.3e}")

    print("\nTime-dependent forcing on a defective three-state chain")
    print(
        " m      stationary max error   m^3 error"
        "      specified-start max error   m^2 error"
    )
    stationary_errors = []
    specified_errors = []
    for m in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        horizon = max(20.0, m**2)
        times = np.unique(
            np.r_[np.linspace(0.0, min(20.0, horizon), 401), m, horizon]
        )

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            # Enforce pi @ normalized_amplitude == 1 algebraically.  Directly
            # integrating all three normalized coordinates makes that exact
            # invariant numerically unstable when the price decays.
            normalized = np.r_[
                state[:2], 3.0 - state[0] - state[1]
            ]
            alpha = -np.expm1(-time)
            matrix = m * Q + np.diag(alpha * limiting_forcing)
            log_derivative = float(pi @ matrix @ normalized)
            normalized_derivative = (
                matrix @ normalized - log_derivative * normalized
            )
            eigenvalue, connection_value, _, _ = moving_spectral_data(
                m, time, limiting_forcing
            )
            return np.r_[
                normalized_derivative[:2],
                log_derivative,
                eigenvalue - connection_value,
            ]

        answer = solve_ivp(
            rhs,
            (0.0, horizon),
            np.array([1.0, 1.0, 0.0, 0.0]),
            t_eval=times,
            method="Radau",
            rtol=2e-10,
            atol=2e-12,
        )
        assert answer.success
        stationary_error = float(
            np.max(np.abs(answer.y[2] - answer.y[3]))
        )
        specified_error = 0.0
        for index, time in enumerate(times):
            normalized = np.r_[
                answer.y[:2, index],
                3.0 - answer.y[0, index] - answer.y[1, index],
            ]
            right_at_time = moving_spectral_data(
                m, float(time), limiting_forcing
            )[2]
            for state_index in range(3):
                exact = answer.y[2, index] + np.log(
                    normalized[state_index]
                )
                approximation = answer.y[3, index] + np.log(
                    right_at_time[state_index]
                )
                specified_error = max(
                    specified_error, abs(exact - approximation)
                )

        stationary_errors.append(stationary_error)
        specified_errors.append(specified_error)
        print(
            f"{m:3.0f}      {stationary_error:18.10e}"
            f"   {m**3 * stationary_error:12.8f}"
            f"      {specified_error:18.10e}"
            f"   {m**2 * specified_error:12.8f}"
        )

    stationary_rates = np.log2(
        np.asarray(stationary_errors[:-1])
        / np.asarray(stationary_errors[1:])
    )
    specified_rates = np.log2(
        np.asarray(specified_errors[:-1])
        / np.asarray(specified_errors[1:])
    )
    assert stationary_rates[-1] > 2.9
    assert specified_rates[-1] > 1.9
    print(
        "stationary moving-projection residual rate: "
        f"{stationary_rates[-1]:.6f}"
    )
    print(
        "specified-start moving-projection residual rate: "
        f"{specified_rates[-1]:.6f}"
    )


def check_nonzero_initial_forcing() -> None:
    """Uniform orders when the slow eigenspace is mismatched at time zero."""
    Q = defective_generator()
    pi = np.ones(3) / 3
    one = np.ones(3)
    initial_forcing = np.array([-0.25, 0.12, -0.02])
    limiting_forcing = np.array([-0.6, -0.2, -0.4])

    print("\nNonzero near-end forcing on a defective three-state chain")
    print(
        " m      stationary max error   m^2 error"
        "      specified-start max error   m error"
    )
    stationary_errors = []
    specified_errors = []
    stationary_slip_errors = []
    specified_slip_errors = []
    for m in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        horizon = max(20.0, m**2)
        times = np.unique(
            np.r_[np.linspace(0.0, min(20.0, horizon), 401), m, horizon]
        )
        eigenvalue_zero, _, right_zero, left_zero = moving_spectral_data(
            m, 0.0, limiting_forcing, initial_forcing
        )
        initial_slow_amplitude = float(left_zero @ one)
        assert initial_slow_amplitude > 0
        initial_fast_amplitude = one - initial_slow_amplitude * right_zero
        initial_matrix = m * Q + np.diag(initial_forcing)
        assert abs(left_zero @ initial_fast_amplitude) < 2e-14

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            normalized = np.r_[
                state[:2], 3.0 - state[0] - state[1]
            ]
            alpha = -np.expm1(-time)
            forcing_values = initial_forcing + alpha * (
                limiting_forcing - initial_forcing
            )
            matrix = m * Q + np.diag(forcing_values)
            log_derivative = float(pi @ matrix @ normalized)
            normalized_derivative = (
                matrix @ normalized - log_derivative * normalized
            )
            eigenvalue, connection_value, _, _ = moving_spectral_data(
                m, time, limiting_forcing, initial_forcing
            )
            return np.r_[
                normalized_derivative[:2],
                log_derivative,
                eigenvalue - connection_value,
            ]

        answer = solve_ivp(
            rhs,
            (0.0, horizon),
            np.array([1.0, 1.0, 0.0, np.log(initial_slow_amplitude)]),
            t_eval=times,
            method="Radau",
            rtol=2e-10,
            atol=2e-12,
        )
        assert answer.success
        stationary_error = float(
            np.max(np.abs(answer.y[2] - answer.y[3]))
        )
        stationary_slip_error = 0.0
        specified_error = 0.0
        specified_slip_error = 0.0
        for index, time in enumerate(times):
            normalized = np.r_[
                answer.y[:2, index],
                3.0 - answer.y[0, index] - answer.y[1, index],
            ]
            right_at_time = moving_spectral_data(
                m, float(time), limiting_forcing, initial_forcing
            )[2]
            frozen_slip = expm(
                (initial_matrix - eigenvalue_zero * np.eye(3)) * time
            ) @ initial_fast_amplitude
            stationary_slip = answer.y[3, index] + np.log1p(
                pi @ frozen_slip / initial_slow_amplitude
            )
            stationary_slip_error = max(
                stationary_slip_error,
                abs(answer.y[2, index] - stationary_slip),
            )
            for state_index in range(3):
                exact = answer.y[2, index] + np.log(
                    normalized[state_index]
                )
                approximation = answer.y[3, index] + np.log(
                    right_at_time[state_index]
                )
                specified_error = max(
                    specified_error, abs(exact - approximation)
                )
                slip_approximation = answer.y[3, index] + np.log(
                    right_at_time[state_index]
                    + frozen_slip[state_index] / initial_slow_amplitude
                )
                specified_slip_error = max(
                    specified_slip_error,
                    abs(exact - slip_approximation),
                )

        stationary_errors.append(stationary_error)
        specified_errors.append(specified_error)
        stationary_slip_errors.append(stationary_slip_error)
        specified_slip_errors.append(specified_slip_error)
        print(
            f"{m:3.0f}      {stationary_error:18.10e}"
            f"   {m**2 * stationary_error:12.8f}"
            f"      {specified_error:18.10e}"
            f"   {m * specified_error:12.8f}"
        )

    print(
        "\nFrozen initial-slip correction on the same chain\n"
        " m      stationary max error   m^3 error"
        "      specified-start max error   m^2 error"
    )
    for m, stationary_error, specified_error in zip(
        (2.0, 4.0, 8.0, 16.0, 32.0, 64.0),
        stationary_slip_errors,
        specified_slip_errors,
    ):
        print(
            f"{m:3.0f}      {stationary_error:18.10e}"
            f"   {m**3 * stationary_error:12.8f}"
            f"      {specified_error:18.10e}"
            f"   {m**2 * specified_error:12.8f}"
        )

    stationary_rates = np.log2(
        np.asarray(stationary_errors[:-1])
        / np.asarray(stationary_errors[1:])
    )
    specified_rates = np.log2(
        np.asarray(specified_errors[:-1])
        / np.asarray(specified_errors[1:])
    )
    stationary_slip_rates = np.log2(
        np.asarray(stationary_slip_errors[:-1])
        / np.asarray(stationary_slip_errors[1:])
    )
    specified_slip_rates = np.log2(
        np.asarray(specified_slip_errors[:-1])
        / np.asarray(specified_slip_errors[1:])
    )
    assert stationary_rates[-1] > 1.9
    assert specified_rates[-1] > 0.95
    assert stationary_slip_rates[-1] > 2.9
    assert specified_slip_rates[-1] > 1.9
    print(
        "nonzero-endpoint stationary residual rate: "
        f"{stationary_rates[-1]:.6f}"
    )
    print(
        "nonzero-endpoint specified-start residual rate: "
        f"{specified_rates[-1]:.6f}"
    )
    print(
        "initial-slip stationary residual rate: "
        f"{stationary_slip_rates[-1]:.6f}"
    )
    print(
        "initial-slip specified-start residual rate: "
        f"{specified_slip_rates[-1]:.6f}"
    )


def varying_generator_data(
    m: float, time: float, moving: bool = True
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Perron data when the two-state fast generator itself moves."""
    alpha = -np.expm1(-time)
    alpha_derivative = np.exp(-time)
    generator_alpha = alpha if moving else 0.0
    generator_alpha_derivative = alpha_derivative if moving else 0.0
    rate_12 = 1.0 + 0.4 * generator_alpha
    rate_21 = 0.7 + 0.5 * generator_alpha
    rate_12_derivative = 0.4 * generator_alpha_derivative
    rate_21_derivative = 0.5 * generator_alpha_derivative
    generator = np.array(
        [[-rate_12, rate_12], [rate_21, -rate_21]]
    )
    generator_derivative = np.array(
        [
            [-rate_12_derivative, rate_12_derivative],
            [rate_21_derivative, -rate_21_derivative],
        ]
    )
    forcing = alpha * np.array([-0.6, -0.2])
    forcing_derivative = alpha_derivative * np.array([-0.6, -0.2])
    matrix = m * generator + np.diag(forcing)
    matrix_derivative = m * generator_derivative + np.diag(
        forcing_derivative
    )
    gauge = np.array([0.7, 1.0]) / 1.7

    values, left, right = eig(matrix, left=True, right=True)
    index = int(np.argmax(values.real))
    assert abs(values[index].imag) < 2e-12
    eigenvalue = float(values[index].real)
    right_vector = right[:, index].real
    right_vector /= gauge @ right_vector
    left_vector = left[:, index].real
    left_vector /= left_vector @ right_vector

    eigenvalue_derivative = float(
        left_vector @ matrix_derivative @ right_vector
    )
    derivative_rhs = -(
        matrix_derivative - eigenvalue_derivative * np.eye(2)
    ) @ right_vector
    bordered = np.block(
        [
            [matrix - eigenvalue * np.eye(2), right_vector[:, None]],
            [gauge[None, :], np.zeros((1, 1))],
        ]
    )
    right_derivative = np.linalg.solve(
        bordered, np.r_[derivative_rhs, 0.0]
    )[:2]
    connection = float(left_vector @ right_derivative)
    return eigenvalue, connection, right_vector, left_vector


def check_time_dependent_generator() -> None:
    """A moving invariant law sharply loses the stationary extra order."""
    gauge = np.array([0.7, 1.0]) / 1.7
    one = np.ones(2)

    # Check the bordered derivative independently of the eigensolver's phase.
    m_check, time_check, step = 7.0, 0.8, 2e-5
    _, connection, _, left_vector = varying_generator_data(
        m_check, time_check
    )
    right_minus = varying_generator_data(
        m_check, time_check - step
    )[2]
    right_plus = varying_generator_data(
        m_check, time_check + step
    )[2]
    right_derivative_fd = (right_plus - right_minus) / (2 * step)
    assert abs(connection - left_vector @ right_derivative_fd) < 2e-10
    assert abs(gauge @ right_derivative_fd) < 2e-11

    print("\nTime-dependent two-state generator")
    print(
        " m      moving-Q max error   m^2 error"
        "      fixed-Q max error   m^3 error"
    )
    moving_errors = []
    fixed_errors = []
    for m in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        horizon = max(20.0, m**2)
        times = np.unique(
            np.r_[np.linspace(0.0, min(20.0, horizon), 401), m, horizon]
        )
        errors = []
        for moving in (True, False):

            def rhs(time: float, state: np.ndarray) -> np.ndarray:
                normalized = np.array(
                    [
                        state[0],
                        (1.0 - gauge[0] * state[0]) / gauge[1],
                    ]
                )
                alpha = -np.expm1(-time)
                rate_12 = 1.0 + (0.4 * alpha if moving else 0.0)
                rate_21 = 0.7 + (0.5 * alpha if moving else 0.0)
                generator = np.array(
                    [[-rate_12, rate_12], [rate_21, -rate_21]]
                )
                matrix = m * generator + np.diag(
                    alpha * np.array([-0.6, -0.2])
                )
                log_derivative = float(gauge @ matrix @ normalized)
                normalized_derivative = (
                    matrix @ normalized - log_derivative * normalized
                )
                eigenvalue, connection, _, _ = varying_generator_data(
                    m, time, moving
                )
                return np.array(
                    [
                        normalized_derivative[0],
                        log_derivative,
                        eigenvalue - connection,
                    ]
                )

            answer = solve_ivp(
                rhs,
                (0.0, horizon),
                np.array([one[0], 0.0, 0.0]),
                t_eval=times,
                method="Radau",
                rtol=2e-10,
                atol=2e-12,
            )
            assert answer.success
            errors.append(float(np.max(np.abs(answer.y[1] - answer.y[2]))))

        moving_errors.append(errors[0])
        fixed_errors.append(errors[1])
        print(
            f"{m:3.0f}      {errors[0]:18.10e}   {m**2 * errors[0]:12.8f}"
            f"      {errors[1]:18.10e}   {m**3 * errors[1]:12.8f}"
        )

    moving_rates = np.log2(
        np.asarray(moving_errors[:-1]) / np.asarray(moving_errors[1:])
    )
    fixed_rates = np.log2(
        np.asarray(fixed_errors[:-1]) / np.asarray(fixed_errors[1:])
    )
    assert 1.9 < moving_rates[-1] < 2.1
    assert fixed_rates[-1] > 2.9
    print(f"moving-generator residual rate: {moving_rates[-1]:.6f}")
    print(f"fixed-generator comparison rate: {fixed_rates[-1]:.6f}")


PERIOD = 2 * np.pi
PERIODIC_MEAN = -0.35
PERIODIC_DELTA_SQUARE_MEAN = 0.45**2 + (0.22**2 + 0.08**2) / 2


def periodic_forcing(time: float | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A smooth periodic forcing with nonzero mean contrast."""
    time = np.asarray(time)
    mean = PERIODIC_MEAN + 0.08 * np.cos(time) + 0.03 * np.sin(2 * time)
    delta = 0.45 + 0.22 * np.sin(time) - 0.08 * np.cos(2 * time)
    return mean, delta


def periodic_ratio_solution(m: float) -> tuple[float, object]:
    """Return the attracting periodic ratio and its one-period dense solution."""

    def ratio_rhs(time: float, state: np.ndarray) -> np.ndarray:
        delta = float(periodic_forcing(time)[1])
        return np.array([delta - 2 * m * state[0] - delta * state[0] ** 2])

    def period_displacement(initial: float) -> float:
        answer = solve_ivp(
            ratio_rhs,
            (0.0, PERIOD),
            np.array([initial]),
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert answer.success
        return float(answer.y[0, -1] - initial)

    fixed_point = root_scalar(
        period_displacement, bracket=(-0.99, 0.99), xtol=2e-14
    )
    assert fixed_point.converged

    def augmented_rhs(time: float, state: np.ndarray) -> np.ndarray:
        mean, delta = periodic_forcing(time)
        ratio = state[0]
        phase_derivative = (
            mean
            - PERIODIC_MEAN
            + (delta**2 - PERIODIC_DELTA_SQUARE_MEAN) / (2 * m)
        )
        return np.array(
            [
                delta - 2 * m * ratio - delta * ratio**2,
                mean + delta * ratio,
                phase_derivative,
            ]
        )

    orbit = solve_ivp(
        augmented_rhs,
        (0.0, PERIOD),
        np.array([fixed_point.root, 0.0, 0.0]),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
        dense_output=True,
    )
    assert orbit.success
    assert abs(orbit.y[0, -1] - fixed_point.root) < 2e-11
    return float(fixed_point.root), orbit


def periodic_floquet_exponent(m: float) -> tuple[float, object, float]:
    """Compute the exponent by Riccati reduction and check the monodromy."""
    _, orbit = periodic_ratio_solution(m)
    exponent = float(orbit.y[1, -1] / PERIOD)

    def fundamental_rhs(time: float, state: np.ndarray) -> np.ndarray:
        mean, delta = periodic_forcing(time)
        matrix = np.array(
            [[mean + delta - m, m], [m, mean - delta - m]]
        )
        return (matrix @ state.reshape(2, 2)).ravel()

    fundamental = solve_ivp(
        fundamental_rhs,
        (0.0, PERIOD),
        np.eye(2).ravel(),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert fundamental.success
    multipliers = np.linalg.eigvals(fundamental.y[:, -1].reshape(2, 2))
    multiplier = float(np.max(multipliers.real))
    assert multiplier > 0 and np.max(np.abs(multipliers.imag)) < 2e-12
    monodromy_exponent = np.log(multiplier) / PERIOD
    monodromy_error = abs(exponent - monodromy_exponent)
    assert monodromy_error < 3e-11
    return exponent, orbit, monodromy_error


def check_periodic_forcing() -> None:
    """Floquet renormalization gives a uniform error despite infinite variation."""
    print("\nPeriodic forcing with infinite total variation")
    print(
        " m      max Floquet-phase error   m^2 error"
        "      m^3 exponent truncation"
    )
    errors = []
    drift_constants = []
    monodromy_errors = []
    for m in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        exponent, orbit, monodromy_error = periodic_floquet_exponent(m)
        monodromy_errors.append(monodromy_error)
        first_order_exponent = (
            PERIODIC_MEAN + PERIODIC_DELTA_SQUARE_MEAN / (2 * m)
        )
        drift_constant = m**3 * abs(exponent - first_order_exponent)
        drift_constants.append(drift_constant)

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            mean, delta = periodic_forcing(time)
            ratio = state[0]
            phase_derivative = (
                mean
                - PERIODIC_MEAN
                + (delta**2 - PERIODIC_DELTA_SQUARE_MEAN) / (2 * m)
            )
            return np.array(
                [
                    delta - 2 * m * ratio - delta * ratio**2,
                    mean + delta * ratio - exponent - phase_derivative,
                ]
            )

        times = np.linspace(0.0, 4 * PERIOD, 4001)
        exact = solve_ivp(
            rhs,
            (0.0, 4 * PERIOD),
            np.zeros(2),
            t_eval=times,
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert exact.success
        error = float(np.max(np.abs(exact.y[1])))
        errors.append(error)

        # Independently check that the periodic-orbit part has zero secular
        # increment after subtracting the exact exponent.
        periodic_increment = float(orbit.y[1, -1] - exponent * PERIOD)
        assert abs(periodic_increment) < 2e-13
        assert abs(orbit.y[2, -1]) < 2e-11
        print(
            f"{m:3.0f}      {error:20.11e}   {m**2 * error:12.8f}"
            f"      {drift_constant:18.10e}"
        )

    rates = np.log2(np.asarray(errors[:-1]) / np.asarray(errors[1:]))
    assert rates[-1] > 1.95

    # The omitted third-order mean is
    # -(<delta'^2>+<delta^4>)/(8m^3).  Sampling a full period without its
    # duplicate endpoint evaluates that coefficient independently.
    grid = np.linspace(0.0, PERIOD, 200000, endpoint=False)
    _, delta = periodic_forcing(grid)
    delta_derivative = 0.22 * np.cos(grid) + 0.16 * np.sin(2 * grid)
    predicted_drift = np.mean(delta_derivative**2 + delta**4) / 8
    assert abs(drift_constants[-1] / predicted_drift - 1) < 0.03
    print(f"periodic uniform-error rate: {rates[-1]:.6f}")
    print(f"maximum Riccati/monodromy discrepancy: {max(monodromy_errors):.3e}")
    print(
        "third-order drift: numerical "
        f"{drift_constants[-1]:.10e}, predicted {predicted_drift:.10e}"
    )


ASYM_ALPHA = 1.7
ASYM_BETA = 0.4
ASYM_RHO = ASYM_ALPHA + ASYM_BETA
ASYM_PI = np.array([ASYM_BETA, ASYM_ALPHA]) / ASYM_RHO
ASYM_V = float(np.prod(ASYM_PI))
ASYM_S = float(ASYM_PI[1] - ASYM_PI[0])


def asymmetric_periodic_orbit(m: float) -> tuple[float, object]:
    """Periodic slow/fast ratio for unequal two-state switching rates."""

    def ratio_rhs(time: float, state: np.ndarray) -> np.ndarray:
        delta = float(periodic_forcing(time)[1])
        ratio = state[0]
        return np.array(
            [
                delta
                + (ASYM_S * delta - m * ASYM_RHO) * ratio
                - ASYM_V * delta * ratio**2
            ]
        )

    def period_displacement(initial: float) -> float:
        answer = solve_ivp(
            ratio_rhs,
            (0.0, PERIOD),
            np.array([initial]),
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert answer.success
        return float(answer.y[0, -1] - initial)

    fixed_point = root_scalar(
        period_displacement, bracket=(-0.99, 0.99), xtol=2e-14
    )
    assert fixed_point.converged

    def augmented_rhs(time: float, state: np.ndarray) -> np.ndarray:
        mean, delta = periodic_forcing(time)
        ratio = state[0]
        return np.array(
            [
                delta
                + (ASYM_S * delta - m * ASYM_RHO) * ratio
                - ASYM_V * delta * ratio**2,
                mean + ASYM_V * delta * ratio,
            ]
        )

    orbit = solve_ivp(
        augmented_rhs,
        (0.0, PERIOD),
        np.array([fixed_point.root, 0.0]),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
        dense_output=True,
    )
    assert orbit.success
    assert abs(orbit.y[0, -1] - fixed_point.root) < 2e-11
    return float(fixed_point.root), orbit


def check_asymmetric_periodic_forcing() -> None:
    """Second-order phase and initial slip restore a uniform third order."""
    grid = np.linspace(0.0, PERIOD, 200000, endpoint=False)
    _, delta_grid = periodic_forcing(grid)
    delta_two_mean = float(np.mean(delta_grid**2))
    delta_three_mean = float(np.mean(delta_grid**3))
    predicted_second_drift = (
        ASYM_V * ASYM_S * delta_three_mean / ASYM_RHO**2
    )
    delta_zero = float(periodic_forcing(0.0)[1])

    print("\nPeriodic forcing with unequal switching rates")
    print(
        " m      first-phase error   m^2 error"
        "      phase+slip error   m^3 error      m^2 slope drift"
    )
    first_errors = []
    slip_errors = []
    drift_constants = []
    monodromy_errors = []
    for m in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        _, orbit = asymmetric_periodic_orbit(m)
        exponent = float(orbit.y[1, -1] / PERIOD)
        first_exponent = (
            PERIODIC_MEAN
            + ASYM_V * delta_two_mean / (m * ASYM_RHO)
        )
        drift_constants.append(m**2 * (exponent - first_exponent))

        def fundamental_rhs(time: float, state: np.ndarray) -> np.ndarray:
            mean, delta = periodic_forcing(time)
            forcing_values = np.array(
                [mean + ASYM_PI[1] * delta, mean - ASYM_PI[0] * delta]
            )
            generator = np.array(
                [[-ASYM_ALPHA, ASYM_ALPHA], [ASYM_BETA, -ASYM_BETA]]
            )
            matrix = m * generator + np.diag(forcing_values)
            return (matrix @ state.reshape(2, 2)).ravel()

        fundamental = solve_ivp(
            fundamental_rhs,
            (0.0, PERIOD),
            np.eye(2).ravel(),
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert fundamental.success
        multiplier = float(
            np.max(
                np.linalg.eigvals(
                    fundamental.y[:, -1].reshape(2, 2)
                ).real
            )
        )
        monodromy_error = abs(exponent - np.log(multiplier) / PERIOD)
        monodromy_errors.append(monodromy_error)
        assert monodromy_error < 3e-11

        slip_scale = ASYM_V * delta_zero**2 / (m**2 * ASYM_RHO**2)

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            mean, delta = periodic_forcing(time)
            delta_derivative = 0.22 * np.cos(time) + 0.16 * np.sin(2 * time)
            ratio = state[0]
            log_derivative = mean + ASYM_V * delta * ratio
            phase_one_derivative = (
                mean
                - PERIODIC_MEAN
                + ASYM_V
                * (delta**2 - delta_two_mean)
                / (m * ASYM_RHO)
            )
            phase_two_derivative = phase_one_derivative + (
                ASYM_V
                / (m**2 * ASYM_RHO**2)
                * (
                    ASYM_S * (delta**3 - delta_three_mean)
                    - delta * delta_derivative
                )
            )
            slip_derivative = (
                -slip_scale * m * ASYM_RHO * np.exp(-m * ASYM_RHO * time)
            )
            return np.array(
                [
                    delta
                    + (ASYM_S * delta - m * ASYM_RHO) * ratio
                    - ASYM_V * delta * ratio**2,
                    log_derivative - exponent - phase_one_derivative,
                    log_derivative
                    - exponent
                    - phase_two_derivative
                    - slip_derivative,
                ]
            )

        times = np.linspace(0.0, 4 * PERIOD, 4001)
        exact = solve_ivp(
            rhs,
            (0.0, 4 * PERIOD),
            np.zeros(3),
            t_eval=times,
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert exact.success
        first_error = float(np.max(np.abs(exact.y[1])))
        slip_error = float(np.max(np.abs(exact.y[2])))
        first_errors.append(first_error)
        slip_errors.append(slip_error)
        print(
            f"{m:3.0f}      {first_error:16.9e}   {m**2 * first_error:11.7f}"
            f"      {slip_error:16.9e}   {m**3 * slip_error:11.7f}"
            f"      {drift_constants[-1]:14.9e}"
        )

    first_rates = np.log2(
        np.asarray(first_errors[:-1]) / np.asarray(first_errors[1:])
    )
    slip_rates = np.log2(
        np.asarray(slip_errors[:-1]) / np.asarray(slip_errors[1:])
    )
    assert first_rates[-1] > 1.95
    assert slip_rates[-1] > 2.9
    assert abs(drift_constants[-1] / predicted_second_drift - 1) < 0.03
    print(f"unequal-rate first-phase order: {first_rates[-1]:.6f}")
    print(f"unequal-rate phase-plus-slip order: {slip_rates[-1]:.6f}")
    print(
        "second-order Floquet drift: numerical "
        f"{drift_constants[-1]:.10e}, predicted {predicted_second_drift:.10e}"
    )
    print(
        "maximum unequal-rate Riccati/monodromy discrepancy: "
        f"{max(monodromy_errors):.3e}"
    )


def moving_periodic_coefficients(
    time: float | np.ndarray,
) -> tuple[np.ndarray, ...]:
    """Rates and slow/fast coefficients for a periodic moving generator."""
    alpha = 1.4 + 0.25 * np.sin(time) + 0.08 * np.cos(2 * time)
    beta = 0.7 + 0.18 * np.cos(time) - 0.06 * np.sin(2 * time)
    alpha_derivative = 0.25 * np.cos(time) - 0.16 * np.sin(2 * time)
    beta_derivative = -0.18 * np.sin(time) - 0.12 * np.cos(2 * time)
    rho = alpha + beta
    rho_derivative = alpha_derivative + beta_derivative
    pi_one = beta / rho
    pi_two = alpha / rho
    pi_one_derivative = (beta_derivative * rho - beta * rho_derivative) / rho**2
    variance = pi_one * pi_two
    skew = pi_two - pi_one
    return (
        alpha,
        beta,
        rho,
        rho_derivative,
        pi_one,
        pi_two,
        pi_one_derivative,
        variance,
        skew,
    )


def moving_periodic_orbit(m: float) -> tuple[float, object]:
    """Periodic Riccati orbit when both switching rates move."""

    def ratio_rhs(time: float, state: np.ndarray) -> np.ndarray:
        _, delta = periodic_forcing(time)
        *_, rho, _rho_derivative, _pi_one, _pi_two, pi_derivative, variance, skew = (
            moving_periodic_coefficients(time)
        )
        coupling = variance * delta + pi_derivative
        ratio = state[0]
        return np.array(
            [
                delta
                + (skew * delta - m * rho) * ratio
                - coupling * ratio**2
            ]
        )

    def period_displacement(initial: float) -> float:
        answer = solve_ivp(
            ratio_rhs,
            (0.0, PERIOD),
            np.array([initial]),
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert answer.success
        return float(answer.y[0, -1] - initial)

    fixed_point = root_scalar(
        period_displacement, bracket=(-0.99, 0.99), xtol=2e-14
    )
    assert fixed_point.converged

    def augmented_rhs(time: float, state: np.ndarray) -> np.ndarray:
        mean, delta = periodic_forcing(time)
        *_, rho, _rho_derivative, _pi_one, _pi_two, pi_derivative, variance, skew = (
            moving_periodic_coefficients(time)
        )
        coupling = variance * delta + pi_derivative
        ratio = state[0]
        return np.array(
            [
                delta
                + (skew * delta - m * rho) * ratio
                - coupling * ratio**2,
                mean + coupling * ratio,
            ]
        )

    orbit = solve_ivp(
        augmented_rhs,
        (0.0, PERIOD),
        np.array([fixed_point.root, 0.0]),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
        dense_output=True,
    )
    assert orbit.success
    assert abs(orbit.y[0, -1] - fixed_point.root) < 2e-11
    return float(fixed_point.root), orbit


def check_moving_periodic_generator() -> None:
    """Floquet transport remains uniform for periodically moving rates."""
    arbitrary_prior_one = 0.86
    grid = np.linspace(0.0, PERIOD, 200000, endpoint=False)
    _, delta_grid = periodic_forcing(grid)
    delta_derivative_grid = 0.22 * np.cos(grid) + 0.16 * np.sin(2 * grid)
    (
        _,
        _,
        rho_grid,
        rho_derivative_grid,
        _,
        _,
        pi_derivative_grid,
        variance_grid,
        skew_grid,
    ) = moving_periodic_coefficients(grid)
    coupling_grid = variance_grid * delta_grid + pi_derivative_grid
    outer_one_grid = delta_grid / rho_grid
    outer_one_derivative_grid = (
        delta_derivative_grid / rho_grid
        - delta_grid * rho_derivative_grid / rho_grid**2
    )
    outer_two_grid = (
        skew_grid * delta_grid * outer_one_grid - outer_one_derivative_grid
    ) / rho_grid
    dynamic_first_drift = float(
        np.mean(variance_grid * delta_grid * outer_one_grid)
    )
    geometric_first_drift = float(
        np.mean(pi_derivative_grid * outer_one_grid)
    )
    first_drift = dynamic_first_drift + geometric_first_drift
    predicted_second_drift = float(np.mean(coupling_grid * outer_two_grid))

    mean_zero, delta_zero = periodic_forcing(0.0)
    del mean_zero
    (
        alpha_zero,
        beta_zero,
        rho_zero,
        _rho_derivative_zero,
        pi_one_zero,
        pi_two_zero,
        pi_derivative_zero,
        variance_zero,
        _skew_zero,
    ) = moving_periodic_coefficients(0.0)
    coupling_zero = float(variance_zero * delta_zero + pi_derivative_zero)

    print("\nPeriodic forcing with a moving fast generator")
    print(
        " m      first-phase error   m^2 error"
        "      phase+slip error   m^3 error      m^2 slope drift"
    )
    first_errors = []
    slip_errors = []
    arbitrary_no_slip_errors = []
    arbitrary_slip_errors = []
    drift_constants = []
    first_drift_constants = []
    monodromy_errors = []
    direct_errors = []
    for m in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        _, orbit = moving_periodic_orbit(m)
        exponent = float(orbit.y[1, -1] / PERIOD)
        first_exponent = PERIODIC_MEAN + first_drift / m
        first_drift_constants.append(m * (exponent - PERIODIC_MEAN))
        drift_constants.append(m**2 * (exponent - first_exponent))

        def fundamental_rhs(time: float, state: np.ndarray) -> np.ndarray:
            mean, delta = periodic_forcing(time)
            (
                alpha,
                beta,
                _rho,
                _rho_derivative,
                pi_one,
                pi_two,
                _pi_derivative,
                _variance,
                _skew,
            ) = moving_periodic_coefficients(time)
            forcing_values = np.array(
                [mean + pi_two * delta, mean - pi_one * delta]
            )
            generator = np.array([[-alpha, alpha], [beta, -beta]])
            matrix = m * generator + np.diag(forcing_values)
            if state.size == 2:
                return matrix @ state
            return (matrix @ state.reshape(2, 2)).ravel()

        fundamental = solve_ivp(
            fundamental_rhs,
            (0.0, PERIOD),
            np.eye(2).ravel(),
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert fundamental.success
        multiplier = float(
            np.max(
                np.linalg.eigvals(
                    fundamental.y[:, -1].reshape(2, 2)
                ).real
            )
        )
        monodromy_error = abs(exponent - np.log(multiplier) / PERIOD)
        monodromy_errors.append(monodromy_error)
        assert monodromy_error < 3e-11

        slip_scale = coupling_zero * delta_zero / (m**2 * rho_zero**2)

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            mean, delta = periodic_forcing(time)
            delta_derivative = 0.22 * np.cos(time) + 0.16 * np.sin(2 * time)
            (
                _alpha,
                _beta,
                rho,
                rho_derivative,
                _pi_one,
                _pi_two,
                pi_derivative,
                variance,
                skew,
            ) = moving_periodic_coefficients(time)
            coupling = variance * delta + pi_derivative
            outer_one = delta / rho
            outer_one_derivative = (
                delta_derivative / rho - delta * rho_derivative / rho**2
            )
            outer_two = (
                skew * delta * outer_one - outer_one_derivative
            ) / rho
            ratio = state[0]
            log_derivative = mean + coupling * ratio
            phase_one_derivative = (
                mean - PERIODIC_MEAN
                + (coupling * outer_one - first_drift) / m
            )
            phase_two_derivative = phase_one_derivative + (
                coupling * outer_two - predicted_second_drift
            ) / m**2
            slip_derivative = (
                -slip_scale * m * rho_zero * np.exp(-m * rho_zero * time)
            )
            return np.array(
                [
                    delta
                    + (skew * delta - m * rho) * ratio
                    - coupling * ratio**2,
                    log_derivative,
                    log_derivative - exponent - phase_one_derivative,
                    log_derivative
                    - exponent
                    - phase_two_derivative
                    - slip_derivative,
                ]
            )

        times = np.linspace(0.0, 4 * PERIOD, 4001)
        exact = solve_ivp(
            rhs,
            (0.0, 4 * PERIOD),
            np.zeros(4),
            t_eval=times,
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert exact.success
        (
            _alpha_times,
            _beta_times,
            rho_times,
            rho_derivative_times,
            pi_one_times,
            _pi_two_times,
            _pi_derivative_times,
            _variance_times,
            skew_times,
        ) = moving_periodic_coefficients(times)
        _, delta_times = periodic_forcing(times)
        delta_derivative_times = (
            0.22 * np.cos(times) + 0.16 * np.sin(2 * times)
        )
        outer_one_times = delta_times / rho_times
        outer_one_derivative_times = (
            delta_derivative_times / rho_times
            - delta_times * rho_derivative_times / rho_times**2
        )
        outer_two_times = (
            skew_times * delta_times * outer_one_times
            - outer_one_derivative_times
        ) / rho_times
        observation_weight = float(pi_one_zero) - pi_one_times
        exact_observation = np.log1p(observation_weight * exact.y[0])
        observation_phase_one = observation_weight * outer_one_times / m
        observation_phase_two = observation_phase_one + (
            observation_weight * outer_two_times
            - 0.5 * observation_weight**2 * outer_one_times**2
        ) / m**2
        observation_slip = (
            -observation_weight
            * delta_zero
            / (m * rho_zero)
            * np.exp(-m * rho_zero * times)
        )
        first_error = float(
            np.max(
                np.abs(
                    exact.y[2]
                    + exact_observation
                    - observation_phase_one
                )
            )
        )
        slip_error = float(
            np.max(
                np.abs(
                    exact.y[3]
                    + exact_observation
                    - observation_phase_two
                    - observation_slip
                )
            )
        )
        arbitrary_observation_weight = arbitrary_prior_one - pi_one_times
        arbitrary_exact_observation = np.log1p(
            arbitrary_observation_weight * exact.y[0]
        )
        arbitrary_observation_phase = (
            arbitrary_observation_weight * outer_one_times / m
        )
        arbitrary_observation_slip = (
            -arbitrary_observation_weight
            * delta_zero
            / (m * rho_zero)
            * np.exp(-m * rho_zero * times)
        )
        arbitrary_no_slip_error = float(
            np.max(
                np.abs(
                    exact.y[2]
                    + arbitrary_exact_observation
                    - arbitrary_observation_phase
                )
            )
        )
        arbitrary_slip_error = float(
            np.max(
                np.abs(
                    exact.y[2]
                    + arbitrary_exact_observation
                    - arbitrary_observation_phase
                    - arbitrary_observation_slip
                )
            )
        )
        full = solve_ivp(
            fundamental_rhs,
            (0.0, 4 * PERIOD),
            np.ones(2),
            t_eval=times,
            method="Radau",
            rtol=2e-13,
            atol=2e-15,
        )
        assert full.success
        direct_price = pi_one_zero * full.y[0] + pi_two_zero * full.y[1]
        direct_errors.append(
            float(
                np.max(
                    np.abs(
                        np.log(direct_price)
                        - exact.y[1]
                        - exact_observation
                    )
                )
            )
        )
        arbitrary_direct_price = (
            arbitrary_prior_one * full.y[0]
            + (1.0 - arbitrary_prior_one) * full.y[1]
        )
        direct_errors.append(
            float(
                np.max(
                    np.abs(
                        np.log(arbitrary_direct_price)
                        - exact.y[1]
                        - arbitrary_exact_observation
                    )
                )
            )
        )
        assert direct_errors[-1] < 3e-11
        first_errors.append(first_error)
        slip_errors.append(slip_error)
        arbitrary_no_slip_errors.append(arbitrary_no_slip_error)
        arbitrary_slip_errors.append(arbitrary_slip_error)
        print(
            f"{m:3.0f}      {first_error:16.9e}   {m**2 * first_error:11.7f}"
            f"      {slip_error:16.9e}   {m**3 * slip_error:11.7f}"
            f"      {drift_constants[-1]:14.9e}"
        )

    first_rates = np.log2(
        np.asarray(first_errors[:-1]) / np.asarray(first_errors[1:])
    )
    slip_rates = np.log2(
        np.asarray(slip_errors[:-1]) / np.asarray(slip_errors[1:])
    )
    arbitrary_no_slip_rates = np.log2(
        np.asarray(arbitrary_no_slip_errors[:-1])
        / np.asarray(arbitrary_no_slip_errors[1:])
    )
    arbitrary_slip_rates = np.log2(
        np.asarray(arbitrary_slip_errors[:-1])
        / np.asarray(arbitrary_slip_errors[1:])
    )
    assert first_rates[-1] > 1.95
    assert slip_rates[-1] > 2.9
    assert 0.95 < arbitrary_no_slip_rates[-1] < 1.05
    assert arbitrary_slip_rates[-1] > 1.95
    predicted_arbitrary_layer = abs(
        (arbitrary_prior_one - float(pi_one_zero))
        * delta_zero
        / rho_zero
    )
    measured_arbitrary_layer = 64.0 * arbitrary_no_slip_errors[-1]
    assert abs(measured_arbitrary_layer / predicted_arbitrary_layer - 1) < 0.01
    assert abs(first_drift_constants[-1] / first_drift - 1) < 0.01
    assert abs(drift_constants[-1] / predicted_second_drift - 1) < 0.03
    assert abs(geometric_first_drift) > 1e-4
    assert min(float(alpha_zero), float(beta_zero)) > 0
    assert abs(float(pi_one_zero + pi_two_zero) - 1) < 1e-14
    print(f"moving-periodic first-phase order: {first_rates[-1]:.6f}")
    print(f"moving-periodic phase-plus-slip order: {slip_rates[-1]:.6f}")
    print(
        "moving-periodic arbitrary-prior order without observation slip: "
        f"{arbitrary_no_slip_rates[-1]:.6f}"
    )
    print(
        "moving-periodic arbitrary-prior order with observation slip: "
        f"{arbitrary_slip_rates[-1]:.6f}"
    )
    print(
        "moving-periodic arbitrary-prior leading layer: numerical "
        f"{measured_arbitrary_layer:.10e}, predicted "
        f"{predicted_arbitrary_layer:.10e}; corrected m=64 error "
        f"{arbitrary_slip_errors[-1]:.10e}"
    )
    print(
        "moving-periodic first-order drift: numerical "
        f"{first_drift_constants[-1]:.10e}, predicted {first_drift:.10e}; "
        f"geometric part {geometric_first_drift:.10e}"
    )
    print(
        "moving-periodic second-order drift: numerical "
        f"{drift_constants[-1]:.10e}, predicted {predicted_second_drift:.10e}"
    )
    print(
        "maximum moving-periodic Riccati/monodromy discrepancy: "
        f"{max(monodromy_errors):.3e}"
    )
    print(
        "maximum moving-periodic reduced/full-system discrepancy: "
        f"{max(direct_errors):.3e}"
    )


def periodic_three_state_generator(time: float) -> np.ndarray:
    """A smooth periodic irreducible generator with a moving invariant law."""
    rate_01 = 0.8 + 0.1 * np.sin(time)
    rate_02 = 0.3 + 0.05 * np.cos(2 * time)
    rate_10 = 0.4 + 0.05 * np.cos(time)
    rate_12 = 1.0 + 0.1 * np.sin(2 * time)
    rate_20 = 0.7 + 0.08 * np.sin(time)
    rate_21 = 0.5 + 0.06 * np.cos(2 * time)
    return np.array(
        [
            [-rate_01 - rate_02, rate_01, rate_02],
            [rate_10, -rate_10 - rate_12, rate_12],
            [rate_20, rate_21, -rate_20 - rate_21],
        ]
    )


def periodic_three_state_generator_derivative(time: float) -> np.ndarray:
    """Derivative of the moving three-state generator."""
    derivative_01 = 0.1 * np.cos(time)
    derivative_02 = -0.1 * np.sin(2 * time)
    derivative_10 = -0.05 * np.sin(time)
    derivative_12 = 0.2 * np.cos(2 * time)
    derivative_20 = 0.08 * np.cos(time)
    derivative_21 = -0.12 * np.sin(2 * time)
    return np.array(
        [
            [
                -derivative_01 - derivative_02,
                derivative_01,
                derivative_02,
            ],
            [
                derivative_10,
                -derivative_10 - derivative_12,
                derivative_12,
            ],
            [
                derivative_20,
                derivative_21,
                -derivative_20 - derivative_21,
            ],
        ]
    )


def periodic_three_state_forcing(time: float) -> np.ndarray:
    """Periodic diagonal forcing for the finite-chain Floquet certificate."""
    return np.diag(
        [
            -0.35 + 0.22 * np.sin(time),
            -0.20 + 0.17 * np.cos(time),
            -0.46 + 0.13 * np.sin(2 * time),
        ]
    )


def stationary_row(generator: np.ndarray) -> np.ndarray:
    """Stationary row of an irreducible finite-state generator."""
    values, vectors = np.linalg.eig(generator.T)
    index = int(np.argmin(np.abs(values)))
    row = vectors[:, index].real
    row /= np.sum(row)
    return row


def finite_chain_first_order_drift() -> tuple[float, float, float]:
    """Mean, dynamic, and geometric terms in the first Floquet correction."""
    integrands = []
    grid = np.linspace(0.0, PERIOD, 20000, endpoint=False)
    one = np.ones(3)
    for time in grid:
        generator = periodic_three_state_generator(time)
        generator_derivative = periodic_three_state_generator_derivative(time)
        pi = stationary_row(generator)
        forcing_values = np.diag(periodic_three_state_forcing(time))
        forcing_mean = float(pi @ forcing_values)
        centered = forcing_values - forcing_mean
        projection = np.outer(one, pi)
        group_inverse = np.linalg.inv(generator - projection) + projection
        first_profile = -group_inverse @ centered
        assert abs(pi @ first_profile) < 2e-13
        assert np.max(np.abs(generator @ first_profile + centered)) < 3e-13

        bordered = generator.T.copy()
        right_hand_side = -generator_derivative.T @ pi
        bordered[-1, :] = one
        right_hand_side[-1] = 0.0
        pi_derivative = np.linalg.solve(bordered, right_hand_side)
        assert abs(np.sum(pi_derivative)) < 2e-13
        assert np.max(
            np.abs(pi_derivative @ generator + pi @ generator_derivative)
        ) < 3e-13

        dynamic = float(pi @ (centered * first_profile))
        geometric = float(pi_derivative @ first_profile)
        integrands.append((forcing_mean, dynamic, geometric))
    averages = np.mean(np.asarray(integrands), axis=0)
    return float(averages[0]), float(averages[1]), float(averages[2])


def finite_chain_periodic_errors(
    m: float, moving_generator: bool
) -> dict[str, float]:
    """Exact Floquet profile and frozen-slip errors for a three-state chain."""
    fixed_generator = defective_generator()

    def generator(time: float) -> np.ndarray:
        if moving_generator:
            return periodic_three_state_generator(time)
        return fixed_generator

    generator_zero = generator(0.0)
    pi_zero = stationary_row(generator_zero)
    one = np.ones(3)
    arbitrary_prior = np.array([0.8, 0.1, 0.1])
    assert np.max(np.abs(generator_zero.sum(axis=1))) < 2e-15
    assert np.min(generator_zero - np.diag(np.diag(generator_zero))) >= 0

    def matrix_rhs(time: float, state: np.ndarray) -> np.ndarray:
        matrix = m * generator(time) + periodic_three_state_forcing(time)
        return (matrix @ state.reshape(3, -1)).ravel()

    fundamental = solve_ivp(
        matrix_rhs,
        (0.0, PERIOD),
        np.eye(3).ravel(),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert fundamental.success
    monodromy = fundamental.y[:, -1].reshape(3, 3)
    values, left, right = eig(monodromy, left=True, right=True)
    index = int(np.argmax(values.real))
    assert abs(values[index].imag) < 2e-11
    multiplier = float(values[index].real)
    right_vector = right[:, index].real
    left_vector = left[:, index].real
    if pi_zero @ right_vector < 0:
        right_vector *= -1
        left_vector *= -1
    right_vector /= pi_zero @ right_vector
    left_vector /= left_vector @ right_vector
    exponent = float(np.log(multiplier) / PERIOD)
    principal_weight = float(left_vector @ one)
    fast_initial = one - principal_weight * right_vector
    assert abs(left_vector @ fast_initial) < 2e-12

    frozen_matrix = (
        m * generator_zero
        + periodic_three_state_forcing(0.0)
        - exponent * np.eye(3)
    )
    times = np.linspace(0.0, 4 * PERIOD, 4001)
    exact = solve_ivp(
        matrix_rhs,
        (0.0, 4 * PERIOD),
        one,
        t_eval=times,
        method="Radau",
        rtol=2e-12,
        atol=2e-14,
    )
    principal = solve_ivp(
        matrix_rhs,
        (0.0, 4 * PERIOD),
        right_vector,
        t_eval=times,
        method="Radau",
        rtol=2e-12,
        atol=2e-14,
    )
    assert exact.success and principal.success
    periodicity_error = np.max(
        np.abs(
            np.exp(-exponent * PERIOD)
            * principal.y[:, 1000]
            - right_vector
        )
    )
    assert periodicity_error < 3e-11

    frozen_fast = np.column_stack(
        [expm(frozen_matrix * time) @ fast_initial for time in times]
    )
    exponential = np.exp(exponent * times)

    def observation_errors(prior: np.ndarray) -> tuple[float, float]:
        exact_price = prior @ exact.y
        principal_price = principal_weight * (prior @ principal.y)
        slip_price = principal_price + exponential * (prior @ frozen_fast)
        assert np.min(exact_price) > 0
        assert np.min(principal_price) > 0
        assert np.min(slip_price) > 0
        plain_error = float(
            np.max(np.abs(np.log(exact_price) - np.log(principal_price)))
        )
        slip_error = float(
            np.max(np.abs(np.log(exact_price) - np.log(slip_price)))
        )
        assert abs(exact_price[0] - slip_price[0]) < 2e-13
        return plain_error, slip_error

    stationary_plain, stationary_slip = observation_errors(pi_zero)
    arbitrary_plain, arbitrary_slip = observation_errors(arbitrary_prior)
    return {
        "stationary_plain": stationary_plain,
        "stationary_slip": stationary_slip,
        "arbitrary_plain": arbitrary_plain,
        "arbitrary_slip": arbitrary_slip,
        "periodicity_error": float(periodicity_error),
        "exponent": exponent,
    }


def check_general_periodic_generator() -> None:
    """Finite-chain Floquet profile and frozen boundary slip."""
    leading_mean, dynamic_drift, geometric_drift = (
        finite_chain_first_order_drift()
    )
    predicted_drift = dynamic_drift + geometric_drift
    moving_results = []
    fixed_results = []
    print("\nFinite-chain periodic Floquet profile")
    print(
        " m      moving arbitrary raw   moving arbitrary slip"
        "   moving stationary raw   fixed stationary slip"
    )
    for m in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
        moving = finite_chain_periodic_errors(m, moving_generator=True)
        fixed = finite_chain_periodic_errors(m, moving_generator=False)
        moving_results.append(moving)
        fixed_results.append(fixed)
        print(
            f"{m:3.0f}      {moving['arbitrary_plain']:18.9e}"
            f"   {moving['arbitrary_slip']:18.9e}"
            f"   {moving['stationary_plain']:18.9e}"
            f"   {fixed['stationary_slip']:18.9e}"
        )

    def final_rate(results: list[dict[str, float]], key: str) -> float:
        return float(np.log2(results[-2][key] / results[-1][key]))

    moving_arbitrary_raw_rate = final_rate(moving_results, "arbitrary_plain")
    moving_arbitrary_slip_rate = final_rate(moving_results, "arbitrary_slip")
    moving_stationary_rate = final_rate(moving_results, "stationary_plain")
    fixed_stationary_raw_rate = final_rate(fixed_results, "stationary_plain")
    fixed_stationary_slip_rate = final_rate(fixed_results, "stationary_slip")
    measured_drift = 64.0 * (
        moving_results[-1]["exponent"] - leading_mean
    )
    assert 0.95 < moving_arbitrary_raw_rate < 1.1
    assert moving_arbitrary_slip_rate > 1.95
    assert moving_stationary_rate > 1.95
    assert fixed_stationary_raw_rate > 1.95
    assert fixed_stationary_slip_rate > 2.9
    assert abs(measured_drift / predicted_drift - 1) < 0.02
    assert abs(geometric_drift) > 1e-4
    assert max(result["periodicity_error"] for result in moving_results) < 3e-11
    assert max(result["periodicity_error"] for result in fixed_results) < 3e-11
    print(
        "moving three-state arbitrary-prior rate without slip: "
        f"{moving_arbitrary_raw_rate:.6f}"
    )
    print(
        "moving three-state arbitrary-prior rate with slip: "
        f"{moving_arbitrary_slip_rate:.6f}"
    )
    print(
        "moving three-state stationary-prior rate: "
        f"{moving_stationary_rate:.6f}"
    )
    print(
        "fixed defective three-state stationary rate without slip: "
        f"{fixed_stationary_raw_rate:.6f}"
    )
    print(
        "fixed defective three-state stationary rate with slip: "
        f"{fixed_stationary_slip_rate:.6f}"
    )
    print(
        "maximum finite-chain Floquet periodicity discrepancy: "
        f"{max(result['periodicity_error'] for result in moving_results + fixed_results):.3e}"
    )
    print(
        "moving three-state first-order Floquet drift: numerical "
        f"{measured_drift:.10e}, predicted {predicted_drift:.10e}; "
        f"dynamic {dynamic_drift:.10e}, geometric {geometric_drift:.10e}"
    )


def main() -> None:
    check_direct_system()
    check_constant_projection()
    check_general_constant_forcing()
    check_general_time_dependent_forcing()
    check_nonzero_initial_forcing()
    check_time_dependent_generator()
    check_periodic_forcing()
    check_asymmetric_periodic_forcing()
    check_moving_periodic_generator()
    check_general_periodic_generator()

    print("Uniform stationary-start error (all sampled T in [0,max(20,m^2)])")
    print(" m       max error       theorem bound     m^2 max error")
    errors = []
    for m in (1.0, 2.0, 4.0, 8.0, 16.0, 32.0):
        horizon = max(20.0, m**2)
        times = np.unique(
            np.r_[np.linspace(0, min(20.0, horizon), 801), m, min(m**2, horizon), horizon]
        )
        solution = solve_log_system(m, times)
        error = float(np.max(np.abs(exact_log(solution) - composite(m, times, solution))))
        bound = theorem_bound(m)
        assert error <= bound * (1 + 2e-8)
        errors.append(error)
        print(f"{m:3.0f}   {error:14.7e}   {bound:14.7e}   {m*m*error:14.7e}")

        for eta in (-1.0, 1.0):
            start_error = float(
                np.max(np.abs(exact_log(solution, eta) - composite(m, times, solution, eta)))
            )
            assert start_error <= theorem_bound(m, eta) * (1 + 2e-8)

    rates = np.log2(np.array(errors[:-1]) / np.array(errors[1:]))
    assert rates[-1] > 1.95
    print(f"Last observed convergence rate: {rates[-1]:.6f}")
    print("Direct two-component ODE check: passed")
    print("Constant-forcing spectral projection check: passed")
    print("Specified-start boundary-layer bounds: passed")
    print("Defective finite-chain constant-forcing theorem: passed")
    print("Defective finite-chain moving-projection theorem: passed")
    print("Nonzero-endpoint mismatch and frozen-slip orders: passed")
    print("Time-dependent-generator sharp-loss theorem: passed")
    print("Periodic Floquet-phase theorem: passed")
    print("Unequal-rate periodic phase-and-slip theorem: passed")
    print("Moving-generator periodic phase-and-slip theorem: passed")
    print("Finite-chain periodic Floquet-profile theorem: passed")


if __name__ == "__main__":
    main()
