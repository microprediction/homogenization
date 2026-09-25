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


def main() -> None:
    check_direct_system()
    check_constant_projection()
    check_general_constant_forcing()
    check_general_time_dependent_forcing()
    check_nonzero_initial_forcing()
    check_time_dependent_generator()

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


if __name__ == "__main__":
    main()
