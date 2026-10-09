"""Certificate for the null-space amplitudes in the solvability hierarchy.

The page uses the two-state system

    d_t u = eps^{-1} Q u + G u,
    Q = [[-1, 1], [1, -1]],  G = diag(1, 0).

It checks the reported obstruction to centering the *whole* first
coefficient, verifies the corrected average/shape recursion, and compares
the resulting first-order outer approximation with the exact matrix
exponential.  It also checks an exact slow/fast modal split and the uniform
first-order composite obtained by restoring the leading initial layer.  A
nonreversible three-state example additionally certifies the analytic slow
spectral-projector series, including its first derivative and a computable
Cauchy remainder for the initial-data amplitude.  A second resolvent contour
enclosing the fast spectrum certifies its exponentially decaying semigroup,
including any nonnormal transient amplification.  Finally the eigenvalue,
projector, and fast estimates are combined into an all-time semigroup bound
and checked on a growing t=eps^{-2}/4 maturity window.

The certificate also treats a smoothly time-dependent two-state generator.
It verifies the moving-centering identity, including the geometric term
pi'(t) chi_m, and checks at one period that retaining this term gives a
second-order outer error while dropping it leaves a first-order error.  For
the two-state model it further identifies the accumulated geometric term as
the oriented protocol integral int (delta/s) dp and checks invariance under
an orientation-preserving time change, sign reversal under path reversal,
and vanishing when the closed protocol is confined to a graph h=H(p).
For a nonreversible three-state family it then verifies the general
parameter-space connection and its curvature, including Stokes' theorem on
a two-parameter loop.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad, quad_vec, solve_ivp
from scipy.linalg import eig, expm


Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
G = np.diag([1.0, 0.0])
ONE = np.ones(2)
SHAPE = np.array([1.0, -1.0])
PI = np.array([0.5, 0.5])
P = np.outer(ONE, PI)
QSHARP = np.linalg.inv(Q - P) + P


def zero_row_sum_matrix(off_diagonal: list[list[float]]) -> np.ndarray:
    """Complete the diagonal so that every row sums to zero."""
    matrix = np.array(off_diagonal, dtype=float)
    np.fill_diagonal(matrix, 0.0)
    np.fill_diagonal(matrix, -matrix.sum(axis=1))
    return matrix


PROTOCOL_Q0 = zero_row_sum_matrix(
    [[0.0, 1.0, 0.6], [0.5, 0.0, 1.2], [0.9, 0.7, 0.0]]
)
PROTOCOL_QX = zero_row_sum_matrix(
    [[0.0, 0.2, 0.0], [0.0, 0.0, 0.1], [-0.12, 0.0, 0.0]]
)
PROTOCOL_QY = zero_row_sum_matrix(
    [[0.0, 0.0, -0.1], [0.15, 0.0, 0.0], [0.0, 0.16, 0.0]]
)


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


def moving_data(t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return Q(t), pi(t), pi'(t), and diagonal forcing g(t)."""
    alpha = 1.4 + 0.25 * np.sin(t) + 0.08 * np.cos(2.0 * t)
    beta = 0.7 + 0.18 * np.cos(t) - 0.06 * np.sin(2.0 * t)
    alpha_prime = 0.25 * np.cos(t) - 0.16 * np.sin(2.0 * t)
    beta_prime = -0.18 * np.sin(t) - 0.12 * np.cos(2.0 * t)
    rate_sum = alpha + beta
    p = beta / rate_sum
    p_prime = (beta_prime * rate_sum - beta * (alpha_prime + beta_prime)) / rate_sum**2

    q = np.array([[-alpha, alpha], [beta, -beta]])
    pi = np.array([p, 1.0 - p])
    pi_prime = np.array([p_prime, -p_prime])

    bar_g = -0.35 + 0.08 * np.cos(t) + 0.03 * np.sin(2.0 * t)
    delta = 0.45 + 0.22 * np.sin(t) - 0.08 * np.cos(2.0 * t)
    g = np.array([bar_g + (1.0 - p) * delta, bar_g - p * delta])
    return q, pi, pi_prime, g


def moving_coefficients(t: float) -> tuple[float, float, float, np.ndarray]:
    """Return bar g, dynamic/geometric coefficients, and the first shape/a0."""
    q, pi, pi_prime, g = moving_data(t)
    bar_g = float(pi @ g)
    f = g - bar_g * np.ones(2)
    r = group_inverse(q, pi)
    shape_per_a0 = -r @ f
    dynamic = -float(pi @ (f * (r @ f)))
    geometric = -float(pi_prime @ (r @ f))
    return bar_g, dynamic, geometric, shape_per_a0


def two_state_protocol_data(t: float) -> tuple[float, float, float, float]:
    """Return p, p', rate sum s, and forcing contrast delta."""
    q, pi, pi_prime, g = moving_data(t)
    rate_sum = -float(np.trace(q))
    return float(pi[0]), float(pi_prime[0]), rate_sum, float(g[0] - g[1])


def stationary_linear_solve(q: np.ndarray) -> np.ndarray:
    """Stationary row law via a linear solve, retaining complex perturbations."""
    system = q.T.copy()
    system[-1, :] = 1.0
    rhs = np.zeros(q.shape[0], dtype=q.dtype)
    rhs[-1] = 1.0
    return np.linalg.solve(system, rhs)


def protocol_connection(x: complex, y: complex) -> np.ndarray:
    """Connection coefficients A_x,A_y for a three-state parameter family."""
    q = PROTOCOL_Q0 + x * PROTOCOL_QX + y * PROTOCOL_QY
    pi = stationary_linear_solve(q)
    p = np.outer(np.ones(3), pi)
    r = np.linalg.inv(q - p) + p
    g = np.array(
        [0.3 + 0.1 * y, -0.2 + 0.15 * x, 0.55 - 0.08 * x + 0.05 * y],
        dtype=q.dtype,
    )
    f = g - (pi @ g) * np.ones(3)
    return np.array(
        [pi @ PROTOCOL_QX @ r @ r @ f, pi @ PROTOCOL_QY @ r @ r @ f]
    )


def protocol_curvature(x: float, y: float) -> float:
    """Return d_x A_y-d_y A_x by complex-step differentiation."""
    step = 1e-20
    d_x_a_y = np.imag(protocol_connection(x + 1j * step, y)[1]) / step
    d_y_a_x = np.imag(protocol_connection(x, y + 1j * step)[0]) / step
    return float(d_x_a_y - d_y_a_x)


def check_protocol_curvature() -> dict[str, float]:
    """Check the finite-state connection identity and Stokes' theorem."""
    x0, x1 = -0.6, 0.7
    y0, y1 = -0.5, 0.8
    identity_residuals: list[float] = []
    minimum_rate = np.inf

    for x in np.linspace(x0, x1, 9):
        for y in np.linspace(y0, y1, 9):
            q = PROTOCOL_Q0 + x * PROTOCOL_QX + y * PROTOCOL_QY
            pi = stationary_linear_solve(q)
            p = np.outer(np.ones(3), pi)
            r = np.linalg.inv(q - p) + p
            g = np.array([0.3 + 0.1 * y, -0.2 + 0.15 * x, 0.55 - 0.08 * x + 0.05 * y])
            f = g - float(pi @ g) * np.ones(3)
            pi_x = -pi @ PROTOCOL_QX @ r
            pi_y = -pi @ PROTOCOL_QY @ r
            direct = np.array([-pi_x @ r @ f, -pi_y @ r @ f])
            identity_residuals.append(float(np.max(np.abs(direct - protocol_connection(x, y)))))
            minimum_rate = min(
                minimum_rate,
                *(q[i, j] for i in range(3) for j in range(3) if i != j),
            )

    # Counterclockwise boundary integral of A_x dx+A_y dy.
    line_integral = (
        quad(lambda x: float(protocol_connection(x, y0)[0]), x0, x1, epsabs=2e-13)[0]
        + quad(lambda y: float(protocol_connection(x1, y)[1]), y0, y1, epsabs=2e-13)[0]
        - quad(lambda x: float(protocol_connection(x, y1)[0]), x0, x1, epsabs=2e-13)[0]
        - quad(lambda y: float(protocol_connection(x0, y)[1]), y0, y1, epsabs=2e-13)[0]
    )

    nodes, weights = np.polynomial.legendre.leggauss(40)
    area_integral = 0.0
    for i, node_x in enumerate(nodes):
        x = 0.5 * (x0 + x1) + 0.5 * (x1 - x0) * node_x
        for j, node_y in enumerate(nodes):
            y = 0.5 * (y0 + y1) + 0.5 * (y1 - y0) * node_y
            area_integral += (
                weights[i]
                * weights[j]
                * protocol_curvature(x, y)
                * (x1 - x0)
                * (y1 - y0)
                / 4.0
            )

    center_q = PROTOCOL_Q0
    cycle_affinity = float(
        np.log(
            center_q[0, 1]
            * center_q[1, 2]
            * center_q[2, 0]
            / (center_q[1, 0] * center_q[2, 1] * center_q[0, 2])
        )
    )
    stokes_error = abs(line_integral - area_integral)
    assert max(identity_residuals) < 3e-17
    assert minimum_rate > 0.4
    assert abs(cycle_affinity) > 1.0
    assert abs(line_integral) > 1e-3
    assert stokes_error < 2e-13

    return {
        "identity_residual": max(identity_residuals),
        "minimum_rate": float(minimum_rate),
        "cycle_affinity": cycle_affinity,
        "center_curvature": protocol_curvature(0.0, 0.0),
        "line_integral": float(line_integral),
        "area_integral": float(area_integral),
        "stokes_error": float(stokes_error),
    }


def check_moving_generator_hierarchy() -> dict[str, float]:
    """Certify the moving-pi hierarchy and its first-order endpoint error."""
    period = 2.0 * np.pi
    times = np.linspace(0.0, period, 401)
    algebraic_residuals: list[float] = []
    gauge_derivative_residuals: list[float] = []
    dynamic_formula_residuals: list[float] = []
    geometric_formula_residuals: list[float] = []

    # The derivative of pi(t) chi_1(t)=0 must contain pi'(t) chi_1(t).
    step = 2e-6
    for t in times:
        q, pi, pi_prime, g = moving_data(t)
        _, dynamic, geometric, shape = moving_coefficients(t)
        p, p_prime, rate_sum, delta = two_state_protocol_data(t)
        dynamic_formula_residuals.append(
            abs(dynamic - p * (1.0 - p) * delta**2 / rate_sum)
        )
        geometric_formula_residuals.append(abs(geometric - p_prime * delta / rate_sum))
        algebraic_residuals.append(abs(float(pi @ shape)))
        _, pi_minus, _, _ = moving_data(t - step)
        _, pi_plus, _, _ = moving_data(t + step)
        _, _, _, shape_minus = moving_coefficients(t - step)
        _, _, _, shape_plus = moving_coefficients(t + step)
        shape_prime = (shape_plus - shape_minus) / (2.0 * step)
        numerical_gauge_derivative = float(pi @ shape_prime + pi_prime @ shape)
        direct_gauge_derivative = float(
            ((pi_plus @ shape_plus) - (pi_minus @ shape_minus)) / (2.0 * step)
        )
        gauge_derivative_residuals.append(
            abs(numerical_gauge_derivative - direct_gauge_derivative)
        )
        assert np.linalg.norm(q @ shape + (g - float(pi @ g) * np.ones(2))) < 3e-14

    assert max(algebraic_residuals) < 2e-15
    assert max(gauge_derivative_residuals) < 2e-9
    assert max(dynamic_formula_residuals) < 2e-15
    assert max(geometric_formula_residuals) < 2e-15

    def hierarchy_rhs(t: float, state: np.ndarray) -> np.ndarray:
        a0, a1, a1_dynamic = state
        bar_g, dynamic, geometric, _ = moving_coefficients(t)
        return np.array(
            [
                bar_g * a0,
                bar_g * a1 + a0 * (dynamic + geometric),
                bar_g * a1_dynamic + a0 * dynamic,
            ]
        )

    hierarchy = solve_ivp(
        hierarchy_rhs,
        (0.0, period),
        np.array([1.0, 0.0, 0.0]),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    assert hierarchy.success
    a0, a1, a1_dynamic = hierarchy.y[:, -1]
    _, _, _, final_shape = moving_coefficients(period)
    chi1 = a0 * final_shape

    epsilons = np.array([0.08, 0.06, 0.045, 0.034, 0.025, 0.019, 0.014, 0.01])
    full_errors: list[float] = []
    dynamic_only_errors: list[float] = []
    for eps in epsilons:
        def exact_rhs(t: float, value: np.ndarray) -> np.ndarray:
            q, _, _, g = moving_data(t)
            return (q / eps + np.diag(g)) @ value

        solution = solve_ivp(
            exact_rhs,
            (0.0, period),
            np.ones(2),
            method="DOP853",
            rtol=2e-12,
            atol=2e-14,
        )
        assert solution.success
        endpoint = solution.y[:, -1]
        full_outer = a0 * np.ones(2) + eps * (a1 * np.ones(2) + chi1)
        dynamic_outer = a0 * np.ones(2) + eps * (a1_dynamic * np.ones(2) + chi1)
        full_errors.append(float(np.linalg.norm(endpoint - full_outer, np.inf)))
        dynamic_only_errors.append(float(np.linalg.norm(endpoint - dynamic_outer, np.inf)))

    full_order = observed_order(full_errors, epsilons)
    dynamic_only_order = observed_order(dynamic_only_errors, epsilons)
    geometric_integral = float((a1 - a1_dynamic) / a0)
    assert 1.90 < full_order < 2.10
    assert 0.94 < dynamic_only_order < 1.06
    assert abs(geometric_integral) > 1e-3

    def geometric_integrand(t: float) -> float:
        _, p_prime, rate_sum, delta = two_state_protocol_data(t)
        return p_prime * delta / rate_sum

    protocol_integral = quad(
        geometric_integrand, 0.0, period, epsabs=2e-13, epsrel=2e-13, limit=200
    )[0]

    # The map phi is a nonuniform, orientation-preserving traversal of the
    # same closed protocol.  Its derivative remains at least 0.65.
    def phi(u: float) -> float:
        return u + 0.35 * np.sin(u)

    def phi_prime(u: float) -> float:
        return 1.0 + 0.35 * np.cos(u)

    reparameterized_integral = quad(
        lambda u: geometric_integrand(phi(u)) * phi_prime(u),
        0.0,
        period,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=200,
    )[0]
    reversed_integral = quad(
        lambda u: -geometric_integrand(period - u),
        0.0,
        period,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=200,
    )[0]

    # If delta/s is a function of p alone, the one-form (delta/s) dp is
    # exact on a closed protocol and the geometric contribution vanishes.
    def graph_protocol_integrand(t: float) -> float:
        p, p_prime, _, _ = two_state_protocol_data(t)
        h_of_p = 0.4 - 0.2 * p + 0.3 * p**2
        return p_prime * h_of_p

    graph_protocol_integral = quad(
        graph_protocol_integrand,
        0.0,
        period,
        epsabs=2e-13,
        epsrel=2e-13,
        limit=200,
    )[0]

    assert abs(protocol_integral - geometric_integral) < 3e-11
    assert abs(reparameterized_integral - protocol_integral) < 3e-13
    assert abs(reversed_integral + protocol_integral) < 3e-13
    assert abs(graph_protocol_integral) < 3e-13

    return {
        "gauge_residual": max(gauge_derivative_residuals),
        "coefficient_formula_residual": max(
            max(dynamic_formula_residuals), max(geometric_formula_residuals)
        ),
        "geometric_integral": geometric_integral,
        "reparameterization_error": abs(reparameterized_integral - protocol_integral),
        "reversal_error": abs(reversed_integral + protocol_integral),
        "graph_protocol_integral": graph_protocol_integral,
        "full_order": full_order,
        "dynamic_only_order": dynamic_only_order,
        "full_error": full_errors[-1],
        "dynamic_only_error": dynamic_only_errors[-1],
    }


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
    # corollary.  For A_eps=Q/eps+diag(g), the first three slow-eigenvalue
    # corrections are K, L and M.  The fourth-cumulant coefficient M has
    # the connected subtraction term that is absent from the raw ordered
    # four-time moment.
    q3 = np.array([[-2.1, 2.0, 0.1], [0.1, -2.1, 2.0], [2.0, 0.1, -2.1]])
    g3 = np.array([1.1, -0.4, 0.6])
    pi3 = stationary(q3)
    qs3 = group_inverse(q3, pi3)
    centered3 = g3 - pi3 @ g3
    k3 = -float(pi3 @ (centered3 * (qs3 @ centered3)))
    h13 = -(qs3 @ centered3)
    h23 = qs3 @ (centered3 * (qs3 @ centered3))
    l3 = float(pi3 @ (centered3 * h23))
    h33 = qs3 @ (-centered3 * h23 + k3 * h13 + l3 * np.ones(3))
    m3 = float(pi3 @ (centered3 * h33))
    m3_direct = -float(
        pi3 @ (centered3 * (qs3 @ (centered3 * (qs3 @ (centered3 * (qs3 @ centered3))))))
        + k3 * pi3 @ (centered3 * (qs3 @ (qs3 @ centered3)))
    )
    assert abs(pi3 @ h13) < 2e-15
    assert abs(pi3 @ h23) < 2e-15
    assert abs(pi3 @ h33) < 2e-15
    assert abs(m3 - m3_direct) < 2e-15
    assert np.max(np.abs(q3 @ h13 + centered3)) < 2e-15
    assert np.max(np.abs(q3 @ h23 - centered3 * (qs3 @ centered3) - k3 * np.ones(3))) < 2e-15
    assert np.max(np.abs(q3 @ h33 - (-centered3 * h23 + k3 * h13 + l3 * np.ones(3)))) < 2e-15

    # Independent ordered-correlation quadrature.  The nonzero eigenvalues
    # have real part -3.15, so truncation at 12 makes the omitted tail far
    # smaller than the displayed tolerance.
    correlation_cutoff = 12.0
    integrated_future, _ = quad_vec(
        lambda s: expm(q3 * s) @ centered3,
        0.0,
        correlation_cutoff,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    l3_quadrature, _ = quad_vec(
        lambda s: (pi3 * centered3) @ (expm(q3 * s) @ (centered3 * integrated_future)),
        0.0,
        correlation_cutoff,
        epsabs=1e-13,
        epsrel=1e-13,
    )
    l3_quadrature_error = abs(float(l3_quadrature) - l3)
    assert l3_quadrature_error < 2e-15
    a3 = np.exp(float(pi3 @ g3) * t)
    u03 = a3 * np.ones(3)
    chi13 = -a3 * (qs3 @ centered3)
    u13 = t * k3 * a3 * np.ones(3) + chi13
    centered3_errors: list[float] = []
    full3_errors: list[float] = []
    eigen3_errors: list[float] = []
    eigen_k_errors: list[float] = []
    eigen_kl_errors: list[float] = []
    eigen_klm_errors: list[float] = []
    l_coefficient_errors: list[float] = []
    m_coefficient_errors: list[float] = []
    for eps in epsilons:
        generator = q3 / eps + np.diag(g3)
        target = expm(generator * t) @ np.ones(3)
        centered3_errors.append(float(np.linalg.norm(target - (u03 + eps * chi13), np.inf)))
        full3_errors.append(float(np.linalg.norm(target - (u03 + eps * u13), np.inf)))
        eigenvalues = np.linalg.eigvals(generator)
        slow_eigenvalue = eigenvalues[np.argmax(np.real(eigenvalues))]
        slow_eigenvalue = float(np.real(slow_eigenvalue))
        eigen3_errors.append(abs((slow_eigenvalue - pi3 @ g3) / eps - k3))
        eigen_k_errors.append(abs(slow_eigenvalue - pi3 @ g3 - eps * k3))
        eigen_kl_errors.append(abs(slow_eigenvalue - pi3 @ g3 - eps * k3 - eps**2 * l3))
        eigen_klm_errors.append(
            abs(slow_eigenvalue - pi3 @ g3 - eps * k3 - eps**2 * l3 - eps**3 * m3)
        )
        l_coefficient_errors.append(abs((slow_eigenvalue - pi3 @ g3 - eps * k3) / eps**2 - l3))
        m_coefficient_errors.append(
            abs((slow_eigenvalue - pi3 @ g3 - eps * k3 - eps**2 * l3) / eps**3 - m3)
        )

    centered3_order = observed_order(centered3_errors, epsilons)
    full3_order = observed_order(full3_errors, epsilons)
    eigen3_order = observed_order(eigen3_errors, epsilons)
    eigen_k_order = observed_order(eigen_k_errors, epsilons)
    eigen_kl_order = observed_order(eigen_kl_errors, epsilons)
    eigen_klm_order = observed_order(eigen_klm_errors, epsilons)
    l_coefficient_order = observed_order(l_coefficient_errors, epsilons)
    m_coefficient_order = observed_order(m_coefficient_errors, epsilons)
    assert 0.96 < centered3_order < 1.04
    assert 1.94 < full3_order < 2.08
    assert 0.96 < eigen3_order < 1.04
    assert 1.96 < eigen_k_order < 2.04
    assert 2.85 < eigen_kl_order < 3.08
    assert 3.85 < eigen_klm_order < 4.08
    assert 0.85 < l_coefficient_order < 1.08
    assert 0.85 < m_coefficient_order < 1.08

    # A certified analytic disk from a resolvent contour.  Singular values
    # are 1-Lipschitz in the scalar spectral parameter.  Thus a regular
    # N-point grid on |w|=r gives a rigorous lower bound after subtracting
    # the maximum distance r*pi/N to the nearest sampled point.
    contour_radius = 1.8
    contour_points = 8192
    angles = 2.0 * np.pi * np.arange(contour_points) / contour_points
    sampled_smin = min(
        np.linalg.svd(
            contour_radius * np.exp(1j * angle) * np.eye(3) - q3,
            compute_uv=False,
        )[-1]
        for angle in angles
    )
    certified_smin = sampled_smin - contour_radius * np.pi / contour_points
    certified_epsilon_radius = certified_smin / np.linalg.norm(np.diag(centered3), 2)
    assert certified_smin > 0.0
    assert certified_epsilon_radius > 2.10

    # Use a strict interior circle for Cauchy's estimate and verify the
    # resulting fourth-order eigenvalue remainder at every tested epsilon.
    cauchy_radius = 0.99 * certified_epsilon_radius
    cauchy_bounds = [
        contour_radius * eps**4
        / (cauchy_radius**5 * (1.0 - eps / cauchy_radius))
        for eps in epsilons
    ]
    cauchy_ratios = [
        error / bound for error, bound in zip(eigen_klm_errors, cauchy_bounds)
    ]
    assert max(cauchy_ratios) < 0.009

    # The same contour also controls the slow spectral projector, hence the
    # amplitude selected by arbitrary initial data.  The exact projector is
    # computed independently from paired left/right eigenvectors.  Its first
    # derivative at zero is the standard reduced-resolvent expression
    #
    #     P_1 = -Q# D_f P_0 - P_0 D_f Q#.
    #
    # On |z|=rho, the contour resolvent is bounded by
    # 1/(s_r-rho||D_f||), so ||P(z)|| <= r/(s_r-rho||D_f||).
    p0 = np.outer(np.ones(3), pi3)
    df3 = np.diag(centered3)
    p1 = -qs3 @ df3 @ p0 - p0 @ df3 @ qs3
    projector_errors: list[float] = []
    for eps in epsilons:
        scipy_values, left, right = eig(q3 + eps * df3, left=True, right=True)
        slow_index = int(np.argmin(np.abs(scipy_values)))
        left_vector = left[:, slow_index]
        right_vector = right[:, slow_index]
        exact_projector = np.outer(right_vector, left_vector.conj()) / np.vdot(
            left_vector, right_vector
        )
        projector_errors.append(
            float(np.linalg.norm(exact_projector - p0 - eps * p1, 2))
        )

    projector_order = observed_order(projector_errors, epsilons)
    assert 1.94 < projector_order < 2.06
    projector_cauchy_radius = 0.5 * certified_epsilon_radius
    projector_sup_bound = contour_radius / (
        certified_smin
        - projector_cauchy_radius * np.linalg.norm(df3, 2)
    )
    projector_bounds = [
        projector_sup_bound
        * (eps / projector_cauchy_radius) ** 2
        / (1.0 - eps / projector_cauchy_radius)
        for eps in epsilons
    ]
    projector_bound_ratios = [
        error / bound
        for error, bound in zip(projector_errors, projector_bounds)
    ]
    assert max(projector_bound_ratios) < 1.0

    # A separate contour enclosing the two fast eigenvalues controls the
    # complementary semigroup.  If Gamma lies in Re(w) <= -gamma and
    #
    #   s_Gamma = min_Gamma sigma_min(w I - Q),
    #
    # then, for eps ||D_f|| < s_Gamma, the Dunford integral gives
    #
    # ||exp(t(Q+eps D_f)/eps)(I-P(eps))||
    #   <= length(Gamma) exp(-gamma t/eps)
    #      / (2 pi (s_Gamma-eps||D_f||)).
    #
    # For this example a circle centered at -3.15 with radius 2.4 encloses
    # both fast eigenvalues, excludes zero, and has gamma=0.75.  As above,
    # singular-value Lipschitz continuity turns the sampled minimum into a
    # rigorous lower bound after subtracting the half-mesh chord bound.
    fast_center = -3.15
    fast_radius = 2.4
    fast_gamma = -(fast_center + fast_radius)
    fast_angles = 2.0 * np.pi * np.arange(contour_points) / contour_points
    fast_sampled_smin = min(
        np.linalg.svd(
            (fast_center + fast_radius * np.exp(1j * angle)) * np.eye(3) - q3,
            compute_uv=False,
        )[-1]
        for angle in fast_angles
    )
    fast_certified_smin = (
        fast_sampled_smin - fast_radius * np.pi / contour_points
    )
    fast_epsilon_radius = fast_certified_smin / np.linalg.norm(df3, 2)
    assert fast_gamma > 0.0
    assert fast_certified_smin > 0.0
    assert fast_epsilon_radius > 0.89

    fast_bound_ratios: list[float] = []
    scaled_times = np.linspace(0.0, 5.0, 1001)
    for eps in epsilons:
        perturbed = q3 + eps * df3
        scipy_values, left, right = eig(perturbed, left=True, right=True)
        slow_index = int(np.argmin(np.abs(scipy_values)))
        left_vector = left[:, slow_index]
        right_vector = right[:, slow_index]
        exact_projector = np.outer(right_vector, left_vector.conj()) / np.vdot(
            left_vector, right_vector
        )
        fast_constant = fast_radius / (
            fast_certified_smin - eps * np.linalg.norm(df3, 2)
        )
        for scaled_time in scaled_times:
            exact_fast = expm(perturbed * scaled_time) @ (
                np.eye(3) - exact_projector
            )
            proved_bound = fast_constant * np.exp(-fast_gamma * scaled_time)
            ratio = float(np.linalg.norm(exact_fast, 2) / proved_bound)
            fast_bound_ratios.append(ratio)
            assert ratio < 1.0
    assert max(fast_bound_ratios) < 0.31

    # Combining the two Cauchy remainders with the fast-contour estimate
    # gives an end-to-end bound for the whole semigroup.  Write
    #
    #   lambda_3 = bar(g) + eps K + eps^2 L + eps^3 M,
    #   P_1(eps) = P_0 + eps P_1.
    #
    # After division by exp(t lambda_3), the exact decomposition and the
    # elementary exponential inequality imply
    #
    # ||exp(-t lambda_3) exp(t A_eps) - P_1(eps)||
    # <= exp(t E_lambda) E_P
    #    + ||P_1(eps)|| (exp(t E_lambda)-1)
    #    + exp(t(bar(g)-lambda_3)) C_f exp(-gamma t/eps).
    #
    # Here E_lambda and E_P are precisely the certified Cauchy bounds
    # above.  The formula is valid for every t >= 0 and exposes, rather
    # than hides, the accumulation t E_lambda of the eigenvalue remainder.
    # At t=eps^{-2}/4 the first-order projector and cubic eigenvalue
    # truncations are both O(eps^2) on this normalized slow scale.
    long_maturity_errors: list[float] = []
    long_maturity_bounds: list[float] = []
    identity_residuals: list[float] = []
    bar_g3 = float(pi3 @ g3)
    for eps, eigen_bound, projector_bound in zip(
        epsilons, cauchy_bounds, projector_bounds
    ):
        perturbed = q3 + eps * df3
        scipy_values, left, right = eig(perturbed, left=True, right=True)
        slow_index = int(np.argmin(np.abs(scipy_values)))
        spectral_projectors: list[np.ndarray] = []
        for index in range(3):
            spectral_projectors.append(
                np.outer(right[:, index], left[:, index].conj())
                / np.vdot(left[:, index], right[:, index])
            )
        exact_projector = spectral_projectors[slow_index]
        exact_lambda = bar_g3 + float(np.real(scipy_values[slow_index])) / eps
        approximate_lambda = (
            bar_g3 + eps * k3 + eps**2 * l3 + eps**3 * m3
        )
        approximate_projector = p0 + eps * p1
        long_maturity = 0.25 / eps**2

        # Evaluate the normalized spectral decomposition mode by mode.  It
        # is algebraically the full matrix exponential, but avoids overflow
        # in its large common slow factor at the longest tested maturities.
        normalized_exact = (
            np.exp(long_maturity * (exact_lambda - approximate_lambda))
            * exact_projector
        )
        for index, value in enumerate(scipy_values):
            if index != slow_index:
                normalized_exact += (
                    np.exp(
                        long_maturity
                        * (bar_g3 + value / eps - approximate_lambda)
                    )
                    * spectral_projectors[index]
                )
        long_maturity_errors.append(
            float(np.linalg.norm(normalized_exact - approximate_projector, 2))
        )

        fast_constant = fast_radius / (
            fast_certified_smin - eps * np.linalg.norm(df3, 2)
        )
        proved_bound = (
            np.exp(long_maturity * eigen_bound) * projector_bound
            + np.linalg.norm(approximate_projector, 2)
            * np.expm1(long_maturity * eigen_bound)
            + np.exp(long_maturity * (bar_g3 - approximate_lambda))
            * fast_constant
            * np.exp(-fast_gamma * long_maturity / eps)
        )
        long_maturity_bounds.append(float(proved_bound))
        assert long_maturity_errors[-1] <= proved_bound

        # The paired projectors must resolve the identity; this separately
        # guards the spectral evaluation used above.
        identity_residuals.append(
            float(np.linalg.norm(sum(spectral_projectors) - np.eye(3), 2))
        )

    long_maturity_order = observed_order(long_maturity_errors, epsilons)
    long_maturity_bound_order = observed_order(long_maturity_bounds, epsilons)
    long_maturity_bound_ratios = [
        error / bound
        for error, bound in zip(long_maturity_errors, long_maturity_bounds)
    ]
    assert max(identity_residuals) < 3e-15
    assert 1.94 < long_maturity_order < 2.06
    assert 1.94 < long_maturity_bound_order < 2.06
    assert max(long_maturity_bound_ratios) < 1.0

    moving = check_moving_generator_hierarchy()
    protocol = check_protocol_curvature()

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
    print(f"three-state third-cumulant coeff L  {l3:.12e}")
    print(f"three-state fourth-cumulant coeff M {m3:.12e}")
    print(f"ordered-correlation quadrature err  {l3_quadrature_error:.3e}")
    print(f"three-state centered-only order     {centered3_order:.6f}")
    print(f"three-state full first order        {full3_order:.6f}")
    print(f"three-state eigen-coefficient order {eigen3_order:.6f}")
    print(f"GK-only eigenvalue residual order   {eigen_k_order:.6f}")
    print(f"K+L eigenvalue residual order       {eigen_kl_order:.6f}")
    print(f"K+L+M eigenvalue residual order     {eigen_klm_order:.6f}")
    print(f"L-coefficient convergence order     {l_coefficient_order:.6f}")
    print(f"M-coefficient convergence order     {m_coefficient_order:.6f}")
    print(f"certified eigen-series radius       {certified_epsilon_radius:.12e}")
    print(f"largest cubic error / Cauchy bound  {max(cauchy_ratios):.6f}")
    print(f"first-order projector error order   {projector_order:.6f}")
    print(f"largest projector error / bound     {max(projector_bound_ratios):.6f}")
    print(f"certified fast-contour eps radius   {fast_epsilon_radius:.12e}")
    print(f"certified fast decay exponent       {fast_gamma:.12e}")
    print(f"largest fast semigroup / bound      {max(fast_bound_ratios):.6f}")
    print(f"normalized t=eps^-2/4 error order   {long_maturity_order:.6f}")
    print(f"composite proved-bound order        {long_maturity_bound_order:.6f}")
    print(f"largest composite error / bound     {max(long_maturity_bound_ratios):.6f}")
    print(f"moving-gauge derivative residual    {moving['gauge_residual']:.3e}")
    print(f"two-state coefficient formula resid {moving['coefficient_formula_residual']:.3e}")
    print(f"integrated geometric coefficient    {moving['geometric_integral']:.12e}")
    print(f"protocol reparameterization error   {moving['reparameterization_error']:.3e}")
    print(f"protocol reversal sign error        {moving['reversal_error']:.3e}")
    print(f"graph-valued loop integral          {moving['graph_protocol_integral']:.3e}")
    print(f"moving full first-order error order {moving['full_order']:.6f}")
    print(f"without geometric term error order  {moving['dynamic_only_order']:.6f}")
    print(f"smallest-eps moving full error       {moving['full_error']:.12e}")
    print(f"smallest-eps dynamic-only error      {moving['dynamic_only_error']:.12e}")
    print(f"protocol connection identity resid. {protocol['identity_residual']:.3e}")
    print(f"three-state minimum jump rate       {protocol['minimum_rate']:.6f}")
    print(f"three-state cycle affinity          {protocol['cycle_affinity']:.12e}")
    print(f"protocol curvature at center        {protocol['center_curvature']:.12e}")
    print(f"protocol boundary integral          {protocol['line_integral']:.12e}")
    print(f"protocol curvature-area integral    {protocol['area_integral']:.12e}")
    print(f"protocol Stokes residual            {protocol['stokes_error']:.3e}")
    print("ok")


if __name__ == "__main__":
    main()
