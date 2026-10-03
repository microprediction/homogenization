"""Independent checks for the finite-rate and known-start identifiability note.

The proof is in tools/pages/identifiability.html. This certificate compares the
first-order expressions to an ODE solve; it does not substitute for that proof.
"""
import math

import numpy as np
from scipy.integrate import quad_vec, solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import gammaincc, ndtr

from three_numbers import (
    coefficients, g_funcs, group_inverse, int_Bk, stationary,
)
from verify_three_numbers import KAPPA, QA, SA, THA
from fastswitch import numerical_a_callable


def first_order(T, m, start=None):
    pi = stationary(QA)
    c2, c3, c4 = coefficients(QA, THA, SA, KAPPA)
    result = (-KAPPA * (pi @ THA) * int_Bk(1, T, KAPPA)
              + (pi @ SA / 2 + c2 / m) * int_Bk(2, T, KAPPA)
              + c3 / m * int_Bk(3, T, KAPPA)
              + c4 / m * int_Bk(4, T, KAPPA))
    if start is not None:
        B = -math.expm1(-KAPPA * T) / KAPPA
        u, v = THA - pi @ THA, SA - pi @ SA
        H = -group_inverse(QA)
        result += (-KAPPA * (H @ u)[start] * B
                   + (H @ v)[start] * B * B / 2) / m
    return result


def exact(T, m):
    g = g_funcs(THA, SA, KAPPA)
    a = numerical_a_callable(
        T, m * QA, [(lambda gi: lambda t: gi.value(t))(gi) for gi in g],
        rtol=1e-13,
    )
    return np.log(a)


def mixing_constants(Q):
    """A constructive (generally conservative) C and gamma for ||e^{Qt}R||."""
    pi = stationary(Q)
    R = np.eye(len(pi)) - np.outer(np.ones(len(pi)), pi)
    t0 = 1.0
    while True:
        rho = np.linalg.norm(expm(Q * t0) @ R, np.inf)
        if 0 < rho < 1:
            return 2 / rho, -math.log(rho) / t0
        t0 *= 2
        assert t0 < 10000, "Could not certify exponential mixing"


def error_bounds(T, m):
    """The explicit log bounds in the note; returns stationary and known-start."""
    pi = stationary(QA)
    u, v = THA - pi @ THA, SA - pi @ SA
    G = np.max(np.abs(KAPPA * u)) / KAPPA + np.max(np.abs(v)) / (2 * KAPPA ** 2)
    L = KAPPA * np.max(np.abs(u)) + np.max(np.abs(v)) / KAPPA
    C, gamma = mixing_constants(QA)
    alpha = C / (m * gamma)
    assert 4 * alpha * G * math.exp(2 * G * T) <= 1, "Rate outside the proven bound"
    W = alpha ** 2 * math.exp(G * T) * (4 * G ** 2 + L + 2 * alpha * G ** 3)
    stationary_bound = T * G * math.exp(G * T) * W
    known_bound = stationary_bound + math.exp(G * T) * W + 4 * alpha ** 2 * G ** 2 * math.exp(4 * G * T)
    return stationary_bound, known_bound


def random_generator(size, rng):
    """Dense irreducible generator, generally nonreversible."""
    rates = rng.uniform(0.1, 2.0, size=(size, size))
    np.fill_diagonal(rates, 0.0)
    return rates - np.diag(rates.sum(axis=1))


def feature_gram(Q, features):
    """Fixed-feature symmetric Green--Kubo Gram matrix."""
    pi = stationary(Q)
    centered = features - np.outer(np.ones(len(pi)), pi @ features)
    H = -group_inverse(Q)
    raw = centered.T @ np.diag(pi) @ H @ centered
    return centered, (raw + raw.T) / 2


def finite_horizon_covariance(Q0, features, T, m=1.0):
    """Exact covariance of integral_0^T F(Y_s) ds under stationarity."""
    pi = stationary(Q0)
    centered = features - np.outer(np.ones(len(pi)), pi @ features)
    H = -group_inverse(Q0)
    transient = np.eye(len(pi)) - expm(m * Q0 * T)
    kernel = T * H / m - H @ H @ transient / m ** 2
    raw = centered.T @ np.diag(pi) @ kernel @ centered
    return centered, raw + raw.T


def arbitrary_prior_covariance(Q, features, T, prior):
    """Exact additive-functional covariance from the joint moment ODE."""
    Q = np.asarray(Q, float)
    features = np.asarray(features, float)
    prior = np.asarray(prior, float)
    states, feature_count = features.shape
    block_m = feature_count * states
    block_s = feature_count * feature_count * states
    initial = np.concatenate([prior, np.zeros(block_m + block_s)])
    feature_rows = features.T

    def rhs(_, value):
        p = value[:states]
        moments = value[states:states + block_m].reshape(feature_count, states)
        second = value[states + block_m:].reshape(feature_count, feature_count, states)
        dp = p @ Q
        dm = moments @ Q + feature_rows * p
        ds = np.einsum("abj,jk->abk", second, Q)
        ds += moments[:, None, :] * feature_rows[None, :, :]
        ds += moments[None, :, :] * feature_rows[:, None, :]
        return np.concatenate([dp, dm.ravel(), ds.ravel()])

    solution = solve_ivp(rhs, (0.0, T), initial, method="DOP853",
                         rtol=2e-12, atol=2e-14)
    assert solution.success
    value = solution.y[:, -1]
    moments = value[states:states + block_m].reshape(feature_count, states)
    second = value[states + block_m:].reshape(feature_count, feature_count, states)
    mean = moments.sum(axis=1)
    covariance = second.sum(axis=2) - np.outer(mean, mean)
    return (covariance + covariance.T) / 2


def initial_mean_response(Q, features, T, m, prior):
    """Exact mean of the centered additive functional from an initial prior."""
    pi = stationary(Q)
    centered = features - np.outer(np.ones(len(pi)), pi @ features)
    H = -group_inverse(Q)
    return prior @ H @ (np.eye(len(pi)) - expm(m * Q * T)) @ centered / m


def krylov_matrix(Q, centered):
    """[F, QF, ..., Q^(n-2)F] on the centered state space."""
    blocks = []
    current = np.asarray(centered, float)
    for _ in range(len(Q) - 1):
        blocks.append(current)
        current = Q @ current
    return np.column_stack(blocks)


def transient_response_matrix(Q, centered, taus):
    """Columns H(I-exp(Q tau))F for the supplied scaled maturities."""
    H = -group_inverse(Q)
    identity = np.eye(len(Q))
    return np.column_stack([
        H @ (identity - expm(Q * tau)) @ centered
        for tau in taus
    ])


def project_simplex(values):
    """Euclidean projection of one or many row vectors onto the simplex."""
    values = np.asarray(values, dtype=float)
    was_vector = values.ndim == 1
    rows = values[None, :] if was_vector else values
    ordered = np.sort(rows, axis=1)[:, ::-1]
    cumulative = np.cumsum(ordered, axis=1) - 1.0
    indices = np.arange(1, rows.shape[1] + 1)
    active = ordered - cumulative / indices > 0.0
    rho = np.sum(active, axis=1) - 1
    thresholds = cumulative[np.arange(len(rows)), rho] / (rho + 1.0)
    projected = np.maximum(rows - thresholds[:, None], 0.0)
    return projected[0] if was_vector else projected


def capped_fekete_nodes(order):
    """Positive nodes after fixing zero in the Fekete design on [0, 1]."""
    legendre = np.polynomial.legendre.Legendre.basis(order)
    interior = legendre.deriv().roots()
    return (np.concatenate((interior, [1.0])) + 1.0) / 2.0


def log_vandermonde_with_zero(nodes):
    """Log of prod_j c_j prod_{i<j}(c_j-c_i) for ordered nodes."""
    augmented = np.concatenate(([0.0], np.asarray(nodes, float)))
    return sum(
        math.log(augmented[j] - augmented[i])
        for i in range(len(augmented))
        for j in range(i + 1, len(augmented))
    )


def verify_krylov_observability():
    """The entire scaled-maturity curve spans exactly the Krylov space."""
    rng = np.random.default_rng(26092026)
    checked = 0
    for states, feature_count in ((3, 1), (4, 2), (6, 1), (7, 3)):
        for _ in range(8):
            Q = random_generator(states, rng)
            pi = stationary(Q)
            features = rng.normal(size=(states, feature_count))
            centered = features - np.outer(np.ones(states), pi @ features)
            krylov = krylov_matrix(Q, centered)
            sampled = transient_response_matrix(
                Q, centered, np.linspace(0.13, 2.3, states - 1))
            assert (np.linalg.matrix_rank(krylov, tol=1e-9)
                    == np.linalg.matrix_rank(sampled, tol=1e-9))
            checked += 1

    # A scalar cyclic feature on four states.  Three scaled maturities recover
    # all three prior degrees of freedom although the fixed-feature Gram rank
    # is only one.
    Q = np.array([
        [-1.6, 0.8, 0.5, 0.3],
        [0.2, -1.3, 0.7, 0.4],
        [0.6, 0.1, -1.5, 0.8],
        [0.3, 0.9, 0.2, -1.4],
    ])
    pi = stationary(Q)
    feature = np.array([[1.1], [-0.7], [0.2], [1.6]])
    centered = feature - np.outer(np.ones(4), pi @ feature)
    krylov = krylov_matrix(Q, centered)
    augmented_krylov = np.column_stack([np.ones(4), krylov])
    assert np.linalg.matrix_rank(augmented_krylov, tol=2e-12) == 4

    taus = (0.25, 0.8, 1.9)
    response = transient_response_matrix(Q, centered, taus)
    augmented = np.column_stack([np.ones(4), response])
    prior = np.array([0.10, 0.25, 0.40, 0.25])
    recovered = np.linalg.solve(
        augmented.T, np.array([1.0, *(prior @ response)]))
    recovery_error = np.max(np.abs(recovered - prior))
    assert recovery_error < 2e-13

    # For tau_j=epsilon*c_j, the determinant has a Vandermonde leading
    # coefficient and exponent 1+...+(n-1).  This both proves that distinct
    # sufficiently small maturities work and quantifies their ill-conditioning.
    scales = np.array([0.7, 1.3, 2.2])
    vandermonde = np.prod([
        scales[j] - scales[i]
        for i in range(len(scales))
        for j in range(i + 1, len(scales))
    ])
    leading = (np.linalg.det(augmented_krylov) * np.prod(scales)
               * vandermonde / math.prod(math.factorial(k)
                                          for k in range(1, 4)))
    exponent = 6
    determinant_errors = []
    determinant_ratios = []
    augmented_singular_values = []
    response_singular_values = []
    for epsilon in (0.2, 0.1, 0.05, 0.025, 0.0125, 0.00625):
        small_response = transient_response_matrix(
            Q, centered, epsilon * scales)
        small_augmented = np.column_stack([np.ones(4), small_response])
        determinant = np.linalg.det(small_augmented)
        ratio = determinant / (epsilon ** exponent * leading)
        determinant_ratios.append(ratio)
        determinant_errors.append(abs(ratio - 1.0))
        augmented_singular_values.append(
            np.linalg.svd(small_augmented, compute_uv=False))
        response_singular_values.append(
            np.linalg.svd(small_response, compute_uv=False))
    determinant_rate = math.log(
        determinant_errors[-2] / determinant_errors[-1], 2)
    assert determinant_errors[-1] < 0.075
    assert determinant_rate > 0.9

    # The determinant is only the product.  The analytic Krylov-Vandermonde
    # factorization gives the individual augmented orders 1, eps, ..., eps^r
    # and response-only orders eps, ..., eps^r.  Thus normalization-inclusive
    # conditioning grows as eps^-r, response-relative conditioning as
    # eps^-(r-1), and absolute prior-recovery noise amplification as eps^-r.
    augmented_rates = np.log2(
        augmented_singular_values[-2] / augmented_singular_values[-1])
    response_rates = np.log2(
        response_singular_values[-2] / response_singular_values[-1])
    expected_augmented_rates = np.arange(4)
    expected_response_rates = np.arange(1, 4)
    assert np.max(abs(augmented_rates - expected_augmented_rates)) < 0.03
    assert np.max(abs(response_rates - expected_response_rates)) < 0.03
    augmented_condition_rate = math.log(
        (augmented_singular_values[-1][0]
         / augmented_singular_values[-1][-1])
        / (augmented_singular_values[-2][0]
           / augmented_singular_values[-2][-1]),
        2,
    )
    response_condition_rate = math.log(
        (response_singular_values[-1][0]
         / response_singular_values[-1][-1])
        / (response_singular_values[-2][0]
           / response_singular_values[-2][-1]),
        2,
    )
    assert abs(augmented_condition_rate - 3) < 0.03
    assert abs(response_condition_rate - 2) < 0.03

    # The leading constants are explicit, not merely bounded.  If K=U R is
    # a QR factorization and V=L W is an LQ factorization of the leading
    # coefficient matrix, then sigma_{k+1}(K D_eps V)/eps^k tends to
    # |R_kk L_kk|.  The same construction applies after deleting the
    # normalization direction.
    coefficient_block = np.vstack([
        scales ** k / math.factorial(k) for k in range(1, 4)
    ])
    coefficient_matrix = np.zeros((4, 4))
    coefficient_matrix[0, 0] = 1.0
    coefficient_matrix[1:, 1:] = coefficient_block

    _, augmented_krylov_triangular = np.linalg.qr(augmented_krylov)
    _, augmented_coefficient_transpose_triangular = np.linalg.qr(
        coefficient_matrix.T)
    augmented_coefficient_lower = (
        augmented_coefficient_transpose_triangular.T)
    predicted_augmented_constants = abs(
        np.diag(augmented_krylov_triangular)
        * np.diag(augmented_coefficient_lower))

    _, krylov_triangular = np.linalg.qr(krylov)
    _, coefficient_transpose_triangular = np.linalg.qr(
        coefficient_block.T)
    coefficient_lower = coefficient_transpose_triangular.T
    predicted_response_constants = abs(
        np.diag(krylov_triangular) * np.diag(coefficient_lower))

    measured_augmented_constants = (
        augmented_singular_values[-1]
        / (0.00625 ** np.arange(4)))
    measured_response_constants = (
        response_singular_values[-1]
        / (0.00625 ** np.arange(1, 4)))
    augmented_constant_relative_errors = abs(
        measured_augmented_constants / predicted_augmented_constants - 1.0)
    response_constant_relative_errors = abs(
        measured_response_constants / predicted_response_constants - 1.0)
    assert np.max(augmented_constant_relative_errors) < 0.015
    assert np.max(response_constant_relative_errors) < 0.015

    # If the largest scaled maturity is capped, the leading determinant is a
    # Vandermonde product on the nodes {0,c_1,...,c_r}.  Its unique maximizer
    # is the Legendre--Gauss--Lobatto design: the endpoints together with the
    # roots of P_r'.  Check the equilibrium equations, strict concavity, and
    # the improvement over equally spaced nodes for dimensions 3 through 9.
    design_ratios = []
    maximum_gradient_residual = 0.0
    least_hessian_gap = math.inf
    for order in range(2, 9):
        optimal_nodes = capped_fekete_nodes(order)
        all_nodes = np.concatenate(([0.0], optimal_nodes))
        free_nodes = optimal_nodes[:-1]
        gradient = np.array([
            sum(1.0 / (node - other)
                for other in all_nodes if other != node)
            for node in free_nodes
        ])
        maximum_gradient_residual = max(
            maximum_gradient_residual, np.max(np.abs(gradient)))

        hessian = np.empty((order - 1, order - 1))
        for i, node_i in enumerate(free_nodes):
            for j, node_j in enumerate(free_nodes):
                if i == j:
                    hessian[i, j] = -sum(
                        1.0 / (node_i - other) ** 2
                        for other in all_nodes if other != node_i)
                else:
                    hessian[i, j] = 1.0 / (node_i - node_j) ** 2
        least_hessian_gap = min(
            least_hessian_gap, -np.linalg.eigvalsh(hessian)[-1])
        assert np.linalg.eigvalsh(hessian)[-1] < 0.0

        equally_spaced = np.arange(1, order + 1) / order
        design_ratios.append(math.exp(
            log_vandermonde_with_zero(optimal_nodes)
            - log_vandermonde_with_zero(equally_spaced)))
        assert design_ratios[-1] >= 1.0 - 2e-14

    assert maximum_gradient_residual < 3e-11
    assert least_hessian_gap > 1.0

    # In the four-state example, retain the previous cap 2.2 and compare the
    # existing nodes with the exact capped optimum.  The actual determinant
    # ratio at small epsilon converges to the ratio of leading coefficients.
    maturity_cap = scales[-1]
    optimal_scales = maturity_cap * capped_fekete_nodes(3)
    optimal_design_gain = math.exp(
        log_vandermonde_with_zero(optimal_scales / maturity_cap)
        - log_vandermonde_with_zero(scales / maturity_cap))
    epsilon = 0.0015625
    current_small_augmented = np.column_stack([
        np.ones(4),
        transient_response_matrix(Q, centered, epsilon * scales),
    ])
    optimal_small_augmented = np.column_stack([
        np.ones(4),
        transient_response_matrix(Q, centered, epsilon * optimal_scales),
    ])
    measured_design_gain = (
        abs(np.linalg.det(optimal_small_augmented))
        / abs(np.linalg.det(current_small_augmented)))
    assert abs(measured_design_gain / optimal_design_gain - 1.0) < 0.004

    # Determinant optimality, weakest-direction optimality, and condition
    # optimality are genuinely different.  For three states the two maturity
    # nodes can be written (c,1).  The last LQ diagonal of
    # [[c,1],[c^2/2,1/2]] is
    # c(1-c)/(2 sqrt(1+c^2)).  It is maximized at the unique root of
    # c^3+2c-1=0, while the asymptotic condition coefficient is minimized at
    # sqrt(2)-1.  The Vandermonde determinant is maximized at c=1/2.
    e_optimal_node = next(
        root.real for root in np.roots([1.0, 0.0, 2.0, -1.0])
        if abs(root.imag) < 1e-12 and 0.0 < root.real < 1.0)
    condition_optimal_node = math.sqrt(2.0) - 1.0
    determinant_optimal_node = 0.5

    def coefficient_constants(node):
        first = math.sqrt(1.0 + node * node)
        second = node * (1.0 - node) / (2.0 * first)
        return first, second

    d_constants = coefficient_constants(determinant_optimal_node)
    e_constants = coefficient_constants(e_optimal_node)
    c_constants = coefficient_constants(condition_optimal_node)
    assert abs(e_optimal_node ** 3 + 2.0 * e_optimal_node - 1.0) < 1e-14
    assert e_constants[1] > c_constants[1] > d_constants[1]
    assert (c_constants[0] / c_constants[1]
            < e_constants[0] / e_constants[1]
            < d_constants[0] / d_constants[1])

    # Pull the coefficient comparison back through a genuine cyclic
    # three-state response.  The model QR factor is common to all designs, so
    # the exact small-epsilon rankings must reproduce the analytic ones.
    design_Q = np.array([
        [-1.3, 1.0, 0.3],
        [0.2, -0.9, 0.7],
        [0.6, 0.4, -1.0],
    ])
    design_pi = stationary(design_Q)
    design_feature = np.array([[1.0], [-0.5], [0.2]])
    design_centered = (
        design_feature - np.outer(np.ones(3), design_pi @ design_feature))
    design_krylov = np.column_stack([
        design_centered, design_Q @ design_centered])
    assert np.linalg.matrix_rank(design_krylov, tol=1e-12) == 2
    _, design_krylov_triangular = np.linalg.qr(design_krylov)
    design_epsilon = 0.001
    measured_minimum_constants = []
    measured_condition_constants = []
    predicted_minimum_constants = []
    predicted_condition_constants = []
    for node in (determinant_optimal_node,
                 e_optimal_node, condition_optimal_node):
        first, second = coefficient_constants(node)
        predicted_minimum_constants.append(
            abs(design_krylov_triangular[1, 1]) * second)
        predicted_condition_constants.append(
            abs(design_krylov_triangular[0, 0]) * first
            / predicted_minimum_constants[-1])
        design_response = transient_response_matrix(
            design_Q, design_centered, design_epsilon * np.array([node, 1.0]))
        singular_values = np.linalg.svd(design_response, compute_uv=False)
        measured_minimum_constants.append(
            singular_values[-1] / design_epsilon ** 2)
        measured_condition_constants.append(
            design_epsilon * singular_values[0] / singular_values[-1])
    assert np.max(np.abs(
        np.array(measured_minimum_constants)
        / np.array(predicted_minimum_constants) - 1.0)) < 0.003
    assert np.max(np.abs(
        np.array(measured_condition_constants)
        / np.array(predicted_condition_constants) - 1.0)) < 0.003
    assert measured_minimum_constants[1] > measured_minimum_constants[0]
    assert measured_condition_constants[2] < measured_condition_constants[1]

    # A repeated nonzero eigenvalue is a genuine scalar-feature obstruction.
    # The complete-graph generator has a three-dimensional eigenspace at -4;
    # every scalar Krylov iterate is therefore collinear with F.
    repeated_Q = np.ones((4, 4)) - 4 * np.eye(4)
    repeated_feature = np.array([[1.0], [-1.0], [2.0], [-2.0]])
    repeated_krylov = krylov_matrix(repeated_Q, repeated_feature)
    repeated_response = transient_response_matrix(
        repeated_Q, repeated_feature, (0.2, 0.7, 1.4))
    assert np.linalg.matrix_rank(repeated_krylov, tol=2e-12) == 1
    assert np.linalg.matrix_rank(repeated_response, tol=2e-12) == 1

    print("\nKrylov characterization of transient observability")
    print(f"random rank identities checked: {checked}")
    print(f"four-state augmented Krylov determinant: "
          f"{np.linalg.det(augmented_krylov):.9f}")
    print(f"three-maturity response determinant: {np.linalg.det(augmented):.9e}")
    print(f"recovered four-state prior: {recovered}")
    print(f"maximum recovery error: {recovery_error:.3e}")
    print(f"small-maturity normalized determinant: "
          f"{determinant_ratios[-1]:.8f}")
    print(f"normalized determinant convergence rate: {determinant_rate:.6f}")
    print("small-maturity augmented singular-value rates: "
          + " ".join(f"{rate:.6f}" for rate in augmented_rates))
    print("small-maturity response singular-value rates: "
          + " ".join(f"{rate:.6f}" for rate in response_rates))
    print(f"augmented/response condition-number rates: "
          f"{augmented_condition_rate:.6f}/{response_condition_rate:.6f}")
    print("predicted augmented singular-value constants: "
          + " ".join(f"{value:.9g}" for value in predicted_augmented_constants))
    print("predicted response singular-value constants: "
          + " ".join(f"{value:.9g}" for value in predicted_response_constants))
    print("maximum leading-constant relative errors: "
          f"{np.max(augmented_constant_relative_errors):.3e}/"
          f"{np.max(response_constant_relative_errors):.3e}")
    print("capped determinant-optimal nodes for four states: "
          + " ".join(f"{value:.9f}" for value in capped_fekete_nodes(3)))
    print(f"leading determinant gain over the existing capped design: "
          f"{optimal_design_gain:.9f}")
    print(f"measured determinant gain at epsilon={epsilon:g}: "
          f"{measured_design_gain:.9f}")
    print("optimal/equispaced determinant gains for 3--9 states: "
          + " ".join(f"{value:.6f}" for value in design_ratios))
    print(f"maximum Fekete equilibrium residual: "
          f"{maximum_gradient_residual:.3e}")
    print(f"smallest strict-concavity eigenvalue gap: "
          f"{least_hessian_gap:.6f}")
    print("three-state D/E/condition-optimal interior nodes: "
          f"{determinant_optimal_node:.9f} {e_optimal_node:.9f} "
          f"{condition_optimal_node:.9f}")
    print(f"E-optimal weakest-direction gain over D-optimal: "
          f"{e_constants[1] / d_constants[1]:.9f}")
    print("measured weakest-direction constants (D/E/condition): "
          + " ".join(f"{value:.9g}"
                     for value in measured_minimum_constants))
    print("measured condition coefficients (D/E/condition): "
          + " ".join(f"{value:.9g}"
                     for value in measured_condition_constants))
    print("repeated-eigenvalue scalar observability rank: 1")


def verify_real_spectrum_all_maturities():
    """A real spectrum is enough, and is necessary in three states."""
    # Start from a reversible weighted path plus a complete-graph component,
    # then add a nonzero circulation.  The chain is not reversible, but its
    # three nonzero eigenvalues remain real and simple.
    reversible_part = np.array([
        [-1.0, 1.0, 0.0, 0.0],
        [1.0, -3.0, 2.0, 0.0],
        [0.0, 2.0, -5.0, 3.0],
        [0.0, 0.0, 3.0, -3.0],
    ]) + 0.2 * (np.ones((4, 4)) - 4.0 * np.eye(4))
    circulation = 0.1 * np.array([
        [0.0, 1.0, -1.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    Q = reversible_part + circulation
    pi = stationary(Q)
    detailed_balance_residual = np.max(np.abs(
        pi[:, None] * Q - pi[None, :] * Q.T))
    assert detailed_balance_residual > 0.04

    eigenvalues, modes = np.linalg.eig(Q)
    ordering = np.argsort(-eigenvalues.real)
    eigenvalues = eigenvalues[ordering]
    modes = modes[:, ordering]
    assert np.max(np.abs(eigenvalues.imag)) < 2e-13
    assert np.max(np.abs(modes.imag)) < 2e-13
    eigenvalues = eigenvalues.real
    modes = modes.real
    modes[:, 0] = 1.0
    assert np.max(np.abs(Q @ modes - modes @ np.diag(eigenvalues))) < 3e-13

    rates = -eigenvalues[1:]
    chosen_coefficients = np.array([1.0, 0.7, -0.4])
    feature = modes[:, 1:] @ chosen_coefficients
    modal_coefficients = np.linalg.solve(modes, feature)[1:]
    assert np.max(np.abs(modal_coefficients - chosen_coefficients)) < 2e-13

    rng = np.random.default_rng(27092026)
    maximum_factorization_error = 0.0
    minimum_response_determinant = math.inf
    minimum_singular_value = math.inf
    expected_kernel_sign = (-1) ** (len(rates) * (len(rates) - 1) // 2)
    for _ in range(200):
        taus = np.cumsum(rng.uniform(0.05, 0.35, len(rates)))
        response = transient_response_matrix(Q, feature[:, None], taus)
        augmented = np.column_stack([np.ones(len(Q)), response])
        kernel = 1.0 - np.exp(-np.outer(rates, taus))
        kernel_determinant = np.linalg.det(kernel)
        assert np.sign(kernel_determinant) == expected_kernel_sign
        predicted = (np.linalg.det(modes)
                     * np.prod(modal_coefficients / rates)
                     * kernel_determinant)
        observed = np.linalg.det(augmented)
        relative_error = abs(observed - predicted) / abs(observed)
        maximum_factorization_error = max(
            maximum_factorization_error, relative_error)
        minimum_response_determinant = min(
            minimum_response_determinant, abs(observed))
        minimum_singular_value = min(
            minimum_singular_value,
            np.linalg.svd(augmented, compute_uv=False)[-1])
    assert maximum_factorization_error < 3e-12
    assert minimum_response_determinant > 4e-7

    taus = np.array([0.2, 0.7, 1.5])
    response = transient_response_matrix(Q, feature[:, None], taus)
    augmented = np.column_stack([np.ones(len(Q)), response])
    prior = np.array([0.10, 0.25, 0.40, 0.25])
    recovered = np.linalg.solve(
        augmented.T, np.array([1.0, *(prior @ response)]))
    recovery_error = np.max(np.abs(recovered - prior))
    assert recovery_error < 3e-14

    # A genuinely defective irreducible generator.  In the basis (1, v, w),
    # Q has a size-two Jordan block at -1:
    # Qv=-v and Qw=(1/10)v-w.  Taking F=w makes (1,F,QF) cyclic.
    defective_Q = np.array([
        [-8.0 / 15.0, 1.0 / 15.0, 7.0 / 15.0],
        [1.0 / 3.0, -2.0 / 3.0, 1.0 / 3.0],
        [1.0 / 5.0, 3.0 / 5.0, -4.0 / 5.0],
    ])
    generalized_eigenvector = np.array([-4.0, 0.0, 4.0])
    cyclic_vector = np.array([-4.0, 1.0, 3.0])
    jordan_strength = 0.1
    defective_feature = cyclic_vector[:, None]
    defective_basis = np.column_stack([
        np.ones(3), generalized_eigenvector, cyclic_vector])
    assert np.min(defective_Q[~np.eye(3, dtype=bool)]) > 0.06
    assert np.max(np.abs(defective_Q.sum(axis=1))) < 2e-16
    assert np.max(np.abs(
        defective_Q @ generalized_eigenvector
        + generalized_eigenvector)) < 3e-16
    assert np.max(np.abs(
        defective_Q @ cyclic_vector
        - jordan_strength * generalized_eigenvector
        + cyclic_vector)) < 3e-16
    assert np.linalg.matrix_rank(defective_Q + np.eye(3), tol=2e-13) == 2
    defective_krylov_determinant = np.linalg.det(np.column_stack([
        np.ones(3), defective_feature, defective_Q @ defective_feature]))
    assert abs(defective_krylov_determinant) > 1.1

    # The response is g0(t)w + eta*g1(t)v, where
    # g0=1-exp(-t) and g1=1-(1+t)exp(-t).  Since g1/g0 is strictly
    # increasing, every ordered positive pair has a nonzero determinant.
    defective_maximum_factorization_error = 0.0
    defective_minimum_response_determinant = math.inf
    defective_minimum_singular_value = math.inf
    for _ in range(200):
        defective_taus = np.cumsum(rng.uniform(0.05, 0.35, 2))
        defective_response = transient_response_matrix(
            defective_Q, defective_feature, defective_taus)
        defective_augmented = np.column_stack([
            np.ones(3), defective_response])
        g0 = 1.0 - np.exp(-defective_taus)
        g1 = 1.0 - (1.0 + defective_taus) * np.exp(-defective_taus)
        predicted = (np.linalg.det(defective_basis) * jordan_strength
                     * (g1[0] * g0[1] - g1[1] * g0[0]))
        observed = np.linalg.det(defective_augmented)
        defective_maximum_factorization_error = max(
            defective_maximum_factorization_error,
            abs(observed - predicted) / abs(observed))
        defective_minimum_response_determinant = min(
            defective_minimum_response_determinant, abs(observed))
        defective_minimum_singular_value = min(
            defective_minimum_singular_value,
            np.linalg.svd(defective_augmented, compute_uv=False)[-1])
    assert defective_maximum_factorization_error < 8e-13
    assert defective_minimum_response_determinant > 2e-4

    defective_taus = np.array([0.4, 1.7])
    defective_response = transient_response_matrix(
        defective_Q, defective_feature, defective_taus)
    defective_augmented = np.column_stack([
        np.ones(3), defective_response])
    defective_prior = np.array([0.15, 0.55, 0.30])
    defective_recovered = np.linalg.solve(
        defective_augmented.T,
        np.array([1.0, *(defective_prior @ defective_response)]))
    defective_recovery_error = np.max(np.abs(
        defective_recovered - defective_prior))
    assert defective_recovery_error < 8e-15

    # A directed three-cycle has the complex spectrum
    # 0, -3/2 +/- i sqrt(3)/2.  Although the scalar feature is cyclic, at
    # tau=pi/b and 2pi/b the two complex numerators 1-exp((-a+ib)tau)
    # are real.  The two real response vectors are therefore collinear.
    cycle_Q = np.array([
        [-1.0, 1.0, 0.0],
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
    ])
    cycle_feature = np.array([[1.0], [-1.0], [0.0]])
    cycle_krylov = np.column_stack([
        np.ones(3), cycle_feature, cycle_Q @ cycle_feature])
    cycle_krylov_determinant = np.linalg.det(cycle_krylov)
    assert abs(cycle_krylov_determinant) > 2.9
    frequency = math.sqrt(3.0) / 2.0
    exceptional_taus = (math.pi / frequency, 2.0 * math.pi / frequency)
    exceptional_response = transient_response_matrix(
        cycle_Q, cycle_feature, exceptional_taus)
    exceptional_augmented = np.column_stack([
        np.ones(3), exceptional_response])
    exceptional_determinant = np.linalg.det(exceptional_augmented)
    exceptional_singular_value = np.linalg.svd(
        exceptional_augmented, compute_uv=False)[-1]
    assert abs(exceptional_determinant) < 8e-15
    assert exceptional_singular_value < 8e-15

    # The exceptional pair is a simple analytic singularity.  Moving only
    # tau_2 by delta differentiates its response column to exp(Q tau_2)F.
    # For a rank-two 3 x 3 matrix A_0, det(A_delta) = d*delta+O(delta^2)
    # implies sigma_min(A_delta) = |d|*|delta|/(sigma_1 sigma_2)
    # +O(delta^2).  The condition number therefore blows up like 1/|delta|.
    exceptional_singular_values = np.linalg.svd(
        exceptional_augmented, compute_uv=False)
    exceptional_derivative_column = (
        expm(cycle_Q * exceptional_taus[1]) @ cycle_feature[:, 0])
    exceptional_determinant_slope = np.linalg.det(np.column_stack([
        np.ones(3), exceptional_response[:, 0],
        exceptional_derivative_column,
    ]))
    predicted_singular_slope = (
        abs(exceptional_determinant_slope)
        / np.prod(exceptional_singular_values[:2]))
    predicted_condition_coefficient = (
        exceptional_singular_values[0] / predicted_singular_slope)
    assert abs(exceptional_determinant_slope) > 1e-5

    transverse_deltas = np.array([
        0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125,
    ])
    transverse_determinants = []
    transverse_singular_values = []
    transverse_condition_numbers = []
    for delta in transverse_deltas:
        transverse_response = transient_response_matrix(
            cycle_Q, cycle_feature,
            (exceptional_taus[0], exceptional_taus[1] + delta))
        transverse_augmented = np.column_stack([
            np.ones(3), transverse_response])
        transverse_spectrum = np.linalg.svd(
            transverse_augmented, compute_uv=False)
        transverse_determinants.append(np.linalg.det(
            transverse_augmented))
        transverse_singular_values.append(transverse_spectrum[-1])
        transverse_condition_numbers.append(
            transverse_spectrum[0] / transverse_spectrum[-1])
    transverse_determinants = np.array(transverse_determinants)
    transverse_singular_values = np.array(transverse_singular_values)
    transverse_condition_numbers = np.array(transverse_condition_numbers)
    determinant_slope_errors = np.abs(
        transverse_determinants / transverse_deltas
        - exceptional_determinant_slope)
    determinant_slope_orders = np.log2(
        determinant_slope_errors[:-1] / determinant_slope_errors[1:])
    assert abs(
        transverse_determinants[-1] / transverse_deltas[-1]
        / exceptional_determinant_slope - 1.0) < 0.002
    assert abs(
        transverse_singular_values[-1] / transverse_deltas[-1]
        / predicted_singular_slope - 1.0) < 0.002
    assert abs(
        transverse_condition_numbers[-1] * transverse_deltas[-1]
        / predicted_condition_coefficient - 1.0) < 0.002
    assert determinant_slope_orders[-1] > 0.99

    generic_response = transient_response_matrix(
        cycle_Q, cycle_feature, (0.4, 1.7))
    generic_determinant = np.linalg.det(np.column_stack([
        np.ones(3), generic_response]))
    assert abs(generic_determinant) > 0.15

    # In the three-state complex-spectrum case the complete exceptional set
    # is the zero set of one explicit scalar kernel.  If the centered
    # eigenvalues are -a +/- ib, the oriented area of the two complex response
    # coordinates is, up to one nonzero model constant,
    #
    # e^{-a(t1+t2)} sin(b(t2-t1))
    # - e^{-a t2} sin(b t2) + e^{-a t1} sin(b t1).
    #
    # For the directed cycle and chosen feature the constant is -2/sqrt(3).
    decay = 1.5

    def aliasing_kernel(t1, t2, a=decay, b=frequency):
        return (
            math.exp(-a * (t1 + t2)) * math.sin(b * (t2 - t1))
            - math.exp(-a * t2) * math.sin(b * t2)
            + math.exp(-a * t1) * math.sin(b * t1)
        )

    cycle_aliasing_constant = (
        generic_determinant / aliasing_kernel(0.4, 1.7))
    assert abs(cycle_aliasing_constant + 2.0 / math.sqrt(3.0)) < 2e-14
    aliasing_rng = np.random.default_rng(29092026)
    maximum_aliasing_formula_error = 0.0
    for _ in range(200):
        t1, t2 = np.sort(aliasing_rng.uniform(0.05, 8.0, size=2))
        response = transient_response_matrix(
            cycle_Q, cycle_feature, (t1, t2))
        observed = np.linalg.det(np.column_stack([
            np.ones(3), response]))
        predicted = cycle_aliasing_constant * aliasing_kernel(t1, t2)
        maximum_aliasing_formula_error = max(
            maximum_aliasing_formula_error, abs(observed - predicted))
    assert maximum_aliasing_formula_error < 2e-14

    # Every pair (k*pi/b,l*pi/b), 1 <= k < l, is exceptional.  The scalar
    # kernel derivative in the second maturity is nonzero there:
    # b*(-1)^l*exp(-a*t_l)*((-1)^k*exp(-a*t_k)-1).
    lattice_pairs = [
        (k, ell) for k in range(1, 4) for ell in range(k + 1, 5)]
    maximum_lattice_derivative_error = 0.0
    for k, ell in ((1, 2), (1, 3), (2, 3)):
        t1 = k * math.pi / frequency
        t2 = ell * math.pi / frequency
        response = transient_response_matrix(
            cycle_Q, cycle_feature, (t1, t2))
        derivative_column = expm(cycle_Q * t2) @ cycle_feature[:, 0]
        observed_slope = np.linalg.det(np.column_stack([
            np.ones(3), response[:, 0], derivative_column]))
        kernel_slope = (
            frequency * (-1) ** ell * math.exp(-decay * t2)
            * ((-1) ** k * math.exp(-decay * t1) - 1.0))
        predicted_slope = cycle_aliasing_constant * kernel_slope
        maximum_lattice_derivative_error = max(
            maximum_lattice_derivative_error,
            abs(observed_slope / predicted_slope - 1.0))
    assert maximum_lattice_derivative_error < 1e-8

    # The directed cycle is not exceptional: every three-state generator
    # with a complex centered pair -a +/- ib has an infinite exceptional
    # lattice at k*pi/b and l*pi/b.  In complex coordinates the response is
    # (exp((-a+ib)t)-1)/(-a+ib), so the second response is exactly
    # a real multiple of the first on that lattice.  Verify six pairs on each
    # of 100 random irreducible generators with positive off-diagonal rates.
    converse_rng = np.random.default_rng(27092027)
    converse_cases = 0
    converse_attempts = 0
    maximum_collinearity_error = 0.0
    maximum_relative_exceptional_determinant = 0.0
    maximum_relative_exceptional_singular_value = 0.0
    minimum_relative_generic_determinant = math.inf
    while converse_cases < 100 and converse_attempts < 10000:
        converse_attempts += 1
        random_rates = converse_rng.uniform(0.1, 2.0, (3, 3))
        np.fill_diagonal(random_rates, 0.0)
        random_Q = random_rates.copy()
        random_Q[np.diag_indices(3)] = -random_rates.sum(axis=1)
        random_eigenvalues = np.linalg.eigvals(random_Q)
        complex_eigenvalue = random_eigenvalues[
            np.argmax(random_eigenvalues.imag)]
        a = -complex_eigenvalue.real
        b = complex_eigenvalue.imag
        if b < 0.2 or a / b > 5.0:
            continue
        random_pi = stationary(random_Q)
        raw_feature = converse_rng.normal(size=3)
        random_feature = (raw_feature
                          - (random_pi @ raw_feature) * np.ones(3))[:, None]
        random_krylov = np.column_stack([
            np.ones(3), random_feature, random_Q @ random_feature])
        if abs(np.linalg.det(random_krylov)) < 1e-5:
            continue

        for k, ell in lattice_pairs:
            random_exceptional_taus = (
                k * math.pi / b, ell * math.pi / b)
            random_exceptional_response = transient_response_matrix(
                random_Q, random_feature, random_exceptional_taus)
            exact_ratio = (
                ((-1) ** ell * math.exp(-a * random_exceptional_taus[1])
                 - 1.0)
                / ((-1) ** k * math.exp(-a * random_exceptional_taus[0])
                   - 1.0))
            collinearity_error = np.linalg.norm(
                random_exceptional_response[:, 1]
                - exact_ratio * random_exceptional_response[:, 0]
            ) / np.linalg.norm(random_exceptional_response[:, 1])
            random_exceptional_augmented = np.column_stack([
                np.ones(3), random_exceptional_response])
            exceptional_column_norms = np.linalg.norm(
                random_exceptional_augmented, axis=0)
            relative_exceptional_determinant = abs(np.linalg.det(
                random_exceptional_augmented)) / np.prod(
                    exceptional_column_norms)
            exceptional_singular_values = np.linalg.svd(
                random_exceptional_augmented, compute_uv=False)
            relative_exceptional_singular_value = (
                exceptional_singular_values[-1]
                / exceptional_singular_values[0])

            maximum_collinearity_error = max(
                maximum_collinearity_error, collinearity_error)
            maximum_relative_exceptional_determinant = max(
                maximum_relative_exceptional_determinant,
                relative_exceptional_determinant)
            maximum_relative_exceptional_singular_value = max(
                maximum_relative_exceptional_singular_value,
                relative_exceptional_singular_value)

        random_generic_response = transient_response_matrix(
            random_Q, random_feature, (0.4 / b, 1.7 / b))
        random_generic_augmented = np.column_stack([
            np.ones(3), random_generic_response])
        relative_generic_determinant = abs(np.linalg.det(
            random_generic_augmented)) / np.prod(np.linalg.norm(
                random_generic_augmented, axis=0))

        minimum_relative_generic_determinant = min(
            minimum_relative_generic_determinant,
            relative_generic_determinant)
        converse_cases += 1
    assert converse_cases == 100
    assert maximum_collinearity_error < 8e-15
    assert maximum_relative_exceptional_determinant < 8e-15
    assert maximum_relative_exceptional_singular_value < 8e-15
    assert minimum_relative_generic_determinant > 0.02

    print("\nReal-spectrum scalar observability at every maturity tuple")
    print(f"detailed-balance residual: {detailed_balance_residual:.9f}")
    print(f"nonzero spectral rates: {rates}")
    print(f"modal feature coefficients: {modal_coefficients}")
    print(f"maturity tuples checked: 200")
    print(f"maximum determinant factorization error: "
          f"{maximum_factorization_error:.3e}")
    print(f"minimum sampled response determinant: "
          f"{minimum_response_determinant:.3e}")
    print(f"minimum sampled singular value: {minimum_singular_value:.3e}")
    print(f"prior recovery error: {recovery_error:.3e}")
    print("\nDefective real-spectrum certificate")
    print(f"defective Krylov determinant: "
          f"{defective_krylov_determinant:.9f}")
    print("Jordan relations: Qv=-v, Qw=0.1v-w")
    print("defective maturity pairs checked: 200")
    print(f"maximum defective factorization error: "
          f"{defective_maximum_factorization_error:.3e}")
    print(f"minimum defective response determinant: "
          f"{defective_minimum_response_determinant:.3e}")
    print(f"minimum defective singular value: "
          f"{defective_minimum_singular_value:.3e}")
    print(f"defective prior recovery error: "
          f"{defective_recovery_error:.3e}")
    print("\nComplex-spectrum exceptional maturities")
    print(f"directed-cycle Krylov determinant: {cycle_krylov_determinant:.9f}")
    print(f"exceptional maturities: {exceptional_taus}")
    print(f"exceptional response determinant: {exceptional_determinant:.3e}")
    print(f"exceptional smallest singular value: "
          f"{exceptional_singular_value:.3e}")
    print(f"transverse determinant slope: "
          f"{exceptional_determinant_slope:.12e}")
    print(f"predicted smallest-singular-value slope: "
          f"{predicted_singular_slope:.12e}")
    print(f"predicted condition-number coefficient: "
          f"{predicted_condition_coefficient:.9f}")
    print("transverse determinant slopes: "
          f"{transverse_determinants / transverse_deltas}")
    print("transverse smallest-singular-value slopes: "
          f"{transverse_singular_values / transverse_deltas}")
    print("transverse condition-number coefficients: "
          f"{transverse_condition_numbers * transverse_deltas}")
    print("determinant-slope convergence orders: "
          f"{determinant_slope_orders}")
    print(f"generic response determinant: {generic_determinant:.9f}")
    print(f"directed-cycle aliasing constant: "
          f"{cycle_aliasing_constant:.12f}")
    print(f"maximum exact aliasing-kernel error: "
          f"{maximum_aliasing_formula_error:.3e}")
    print(f"maximum lattice derivative relative error: "
          f"{maximum_lattice_derivative_error:.3e}")
    print("\nSharp three-state complex-spectrum converse")
    print(f"random complex-spectrum chains checked: {converse_cases}")
    print(f"exceptional lattice pairs checked: "
          f"{converse_cases * len(lattice_pairs)}")
    print(f"maximum exact-ratio residual: "
          f"{maximum_collinearity_error:.3e}")
    print(f"maximum relative exceptional determinant: "
          f"{maximum_relative_exceptional_determinant:.3e}")
    print(f"maximum relative exceptional singular value: "
          f"{maximum_relative_exceptional_singular_value:.3e}")
    print(f"minimum relative generic determinant: "
          f"{minimum_relative_generic_determinant:.3e}")


def verify_long_end_conditioning():
    """Sharp singular-value rates when a maturity design is translated right."""
    # Scale the real-spectrum nonreversible example so that all asymptotic
    # directions remain resolvable in ordinary double precision.  The feature
    # has a nonzero coefficient in each nonconstant eigenmode.
    reversible_part = np.array([
        [-1.0, 1.0, 0.0, 0.0],
        [1.0, -3.0, 2.0, 0.0],
        [0.0, 2.0, -5.0, 3.0],
        [0.0, 0.0, 3.0, -3.0],
    ]) + 0.2 * (np.ones((4, 4)) - 4.0 * np.eye(4))
    circulation = 0.1 * np.array([
        [0.0, 1.0, -1.0, 0.0],
        [-1.0, 0.0, 1.0, 0.0],
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ])
    Q = 0.25 * (reversible_part + circulation)
    eigenvalues, modes = np.linalg.eig(Q)
    ordering = np.argsort(-eigenvalues.real)
    eigenvalues = eigenvalues[ordering].real
    modes = modes[:, ordering].real
    modes[:, 0] = 1.0
    rates = -eigenvalues[1:]
    assert np.all(np.diff(rates) > 0.5)
    feature = modes[:, 1:] @ np.array([1.0, 0.7, -0.4])

    offsets = np.array([0.2, 0.8, 1.5])
    translations = np.arange(4.0, 17.0, 2.0)
    quote_standard_deviations = np.array([1.0, 1.4, 0.8])
    quote_correlation = 0.55 ** np.abs(
        np.subtract.outer(np.arange(len(offsets)), np.arange(len(offsets))))
    quote_covariance = (
        quote_standard_deviations[:, None]
        * quote_correlation
        * quote_standard_deviations[None, :])
    quote_covariance_root = np.linalg.cholesky(quote_covariance)
    quote_covariance_eigenvalues = np.linalg.eigvalsh(quote_covariance)
    assert quote_covariance_eigenvalues[0] > 0.0
    centered_basis = np.linalg.qr(np.column_stack([
        np.ones(4), np.eye(4)[:, 1:]
    ]))[0][:, 1:]
    spectra = []
    statistical_spectra = []
    correlated_statistical_spectra = []
    determinants = []
    for translation in translations:
        response = transient_response_matrix(
            Q, feature[:, None], translation + offsets)
        augmented = np.column_stack([np.ones(4), response])
        spectra.append(np.linalg.svd(augmented, compute_uv=False))
        statistical_spectra.append(np.linalg.svd(
            centered_basis.T @ response, compute_uv=False))
        correlated_statistical_spectra.append(np.linalg.svd(
            np.linalg.solve(
                quote_covariance_root, response.T @ centered_basis),
            compute_uv=False))
        determinants.append(abs(np.linalg.det(augmented)))
        assert np.linalg.matrix_rank(augmented, tol=1e-13) == 4
    spectra = np.asarray(spectra)
    statistical_spectra = np.asarray(statistical_spectra)
    correlated_statistical_spectra = np.asarray(
        correlated_statistical_spectra)
    determinants = np.asarray(determinants)

    # There are two O(1) directions: normalization and the common limiting
    # response.  The remaining singular values have rates a_1 and a_2, not
    # a_2 and a_3; the fastest mode is omitted from the leading determinant.
    fitted_rates = -np.array([
        np.polyfit(translations[-4:], np.log(spectra[-4:, index]), 1)[0]
        for index in range(4)
    ])
    determinant_rate = -np.polyfit(
        translations[-4:], np.log(determinants[-4:]), 1)[0]
    condition_rate = np.polyfit(
        translations[-4:],
        np.log(spectra[-4:, 0] / spectra[-4:, -1]), 1)[0]
    assert np.max(np.abs(fitted_rates[:2])) < 0.003
    assert abs(fitted_rates[2] - rates[0]) < 0.002
    assert abs(fitted_rates[3] - rates[1]) < 0.002
    assert abs(determinant_rate - rates[:2].sum()) < 0.002
    assert abs(condition_rate - rates[1]) < 0.003

    # The relevant inverse problem for an unknown prior is the restriction of
    # C_M' to {h: h'1=0}.  Its least singular value has the same second-fastest
    # exponential rate as the augmented matrix.  This turns exact noiseless
    # rank into a statistical impossibility result under Gaussian quote noise.
    statistical_rate = -np.polyfit(
        translations[-4:],
        np.log(statistical_spectra[-4:, -1]), 1)[0]
    assert abs(statistical_rate - rates[1]) < 0.003
    correlated_statistical_rate = -np.polyfit(
        translations[-4:],
        np.log(correlated_statistical_spectra[-4:, -1]), 1)[0]
    assert abs(correlated_statistical_rate - rates[1]) < 0.003

    # A known nonspherical quote covariance changes the relevant geometry from
    # Euclidean to Mahalanobis.  Uniform eigenvalue bounds compare the whitened
    # and unwhitened weakest signals and therefore preserve the long-end rate.
    covariance_lower = quote_covariance_eigenvalues[0]
    covariance_upper = quote_covariance_eigenvalues[-1]
    assert np.all(
        correlated_statistical_spectra[:, -1]
        >= statistical_spectra[:, -1] / math.sqrt(covariance_upper)
        - 5e-15)
    assert np.all(
        correlated_statistical_spectra[:, -1]
        <= statistical_spectra[:, -1] / math.sqrt(covariance_lower)
        + 5e-15)

    pi = stationary(Q)
    prior_radius = 0.4 * np.min(pi)
    quote_noise = 1e-5
    testing_bounds = ndtr(
        -prior_radius * statistical_spectra[:, -1] / quote_noise)
    assert testing_bounds[-1] > 0.49999
    assert np.all(np.diff(testing_bounds) > 0)

    # Check the exact equal-prior Gaussian testing error independently by
    # simulating the full vector observation at the first translation.
    response = transient_response_matrix(
        Q, feature[:, None], translations[0] + offsets)
    left, _, _ = np.linalg.svd(centered_basis.T @ response)
    weakest_prior_direction = centered_basis @ left[:, -1]
    prior_plus = pi + prior_radius * weakest_prior_direction
    prior_minus = pi - prior_radius * weakest_prior_direction
    assert min(prior_plus.min(), prior_minus.min()) > 0
    midpoint = response.T @ pi
    half_difference = prior_radius * response.T @ weakest_prior_direction
    rng = np.random.default_rng(2101973)
    simulations = 200_000
    plus_observations = (
        response.T @ prior_plus
        + quote_noise * rng.normal(size=(simulations, len(offsets))))
    minus_observations = (
        response.T @ prior_minus
        + quote_noise * rng.normal(size=(simulations, len(offsets))))
    plus_errors = ((plus_observations - midpoint) @ half_difference <= 0)
    minus_errors = ((minus_observations - midpoint) @ half_difference >= 0)
    empirical_testing_error = 0.5 * (
        np.mean(plus_errors) + np.mean(minus_errors))
    exact_testing_error = ndtr(
        -np.linalg.norm(half_difference) / quote_noise)
    assert abs(empirical_testing_error - exact_testing_error) < 0.002

    # Invert on the centered subspace, then project onto the probability
    # simplex.  Projection cannot increase distance from the true prior, so
    # this supplies the constructive upper bound matching the testing rate.
    inverse_problem = response.T @ centered_basis
    singular_values = np.linalg.svd(inverse_problem, compute_uv=False)
    recovery_noise = quote_noise * rng.normal(
        size=(simulations, len(offsets)))
    coordinate_errors = np.linalg.solve(
        inverse_problem, recovery_noise.T).T
    unconstrained = pi + coordinate_errors @ centered_basis.T
    projected = project_simplex(unconstrained)
    assert np.min(projected) >= 0.0
    assert np.max(np.abs(np.sum(projected, axis=1) - 1.0)) < 5e-15
    unconstrained_errors = np.linalg.norm(unconstrained - pi, axis=1)
    projected_errors = np.linalg.norm(projected - pi, axis=1)
    maximum_projection_inflation = float(np.max(
        projected_errors - unconstrained_errors))
    assert maximum_projection_inflation < 2e-15

    empirical_unconstrained_mse = float(np.mean(
        unconstrained_errors ** 2))
    exact_unconstrained_mse = quote_noise ** 2 * np.sum(
        singular_values ** -2)
    mse_relative_error = abs(
        empirical_unconstrained_mse / exact_unconstrained_mse - 1.0)
    assert mse_relative_error < 0.01

    least_signal = singular_values[-1]
    chi_threshold = prior_radius * least_signal / quote_noise
    chi_tail_bound = gammaincc(
        len(offsets) / 2.0, chi_threshold ** 2 / 2.0)
    empirical_projected_tail = float(np.mean(
        projected_errors >= prior_radius))
    assert empirical_projected_tail <= chi_tail_bound + 0.002
    rms_upper_bound = math.sqrt(len(offsets)) * quote_noise / least_signal

    # Repeat the two-point and constructive calculations with correlated and
    # heteroskedastic Gaussian quote noise.  The weakest prior direction is now
    # the right singular vector of the whitened inverse problem.
    whitened_inverse_problem = np.linalg.solve(
        quote_covariance_root, inverse_problem)
    _, correlated_singular_values, correlated_right = np.linalg.svd(
        whitened_inverse_problem)
    correlated_weakest_coordinate = correlated_right[-1]
    correlated_weakest_direction = (
        centered_basis @ correlated_weakest_coordinate)
    correlated_prior_plus = (
        pi + prior_radius * correlated_weakest_direction)
    correlated_prior_minus = (
        pi - prior_radius * correlated_weakest_direction)
    assert min(
        correlated_prior_plus.min(), correlated_prior_minus.min()) > 0
    correlated_half_difference = (
        prior_radius * response.T @ correlated_weakest_direction)
    correlated_noise = (
        quote_noise
        * rng.normal(size=(simulations, len(offsets)))
        @ quote_covariance_root.T)
    correlated_plus_observations = (
        response.T @ correlated_prior_plus + correlated_noise)
    correlated_minus_observations = (
        response.T @ correlated_prior_minus
        + quote_noise
        * rng.normal(size=(simulations, len(offsets)))
        @ quote_covariance_root.T)
    discriminant_direction = np.linalg.solve(
        quote_covariance, correlated_half_difference)
    correlated_plus_errors = (
        (correlated_plus_observations - midpoint)
        @ discriminant_direction <= 0)
    correlated_minus_errors = (
        (correlated_minus_observations - midpoint)
        @ discriminant_direction >= 0)
    correlated_empirical_testing_error = 0.5 * (
        np.mean(correlated_plus_errors)
        + np.mean(correlated_minus_errors))
    correlated_exact_testing_error = ndtr(
        -prior_radius * correlated_singular_values[-1] / quote_noise)
    assert abs(
        correlated_empirical_testing_error
        - correlated_exact_testing_error) < 0.002

    whitened_noise = np.linalg.solve(
        quote_covariance_root, correlated_noise.T).T
    correlated_coordinate_errors = np.linalg.solve(
        whitened_inverse_problem, whitened_noise.T).T
    correlated_unconstrained = (
        pi + correlated_coordinate_errors @ centered_basis.T)
    correlated_projected = project_simplex(correlated_unconstrained)
    correlated_unconstrained_errors = np.linalg.norm(
        correlated_unconstrained - pi, axis=1)
    correlated_projected_errors = np.linalg.norm(
        correlated_projected - pi, axis=1)
    assert np.max(
        correlated_projected_errors
        - correlated_unconstrained_errors) < 2e-15
    correlated_exact_mse = quote_noise ** 2 * np.sum(
        correlated_singular_values ** -2)
    correlated_empirical_mse = float(np.mean(
        correlated_unconstrained_errors ** 2))
    assert abs(
        correlated_empirical_mse / correlated_exact_mse - 1.0) < 0.01
    correlated_chi_threshold = (
        prior_radius * correlated_singular_values[-1] / quote_noise)
    correlated_chi_tail_bound = gammaincc(
        len(offsets) / 2.0, correlated_chi_threshold ** 2 / 2.0)
    correlated_empirical_tail = float(np.mean(
        correlated_projected_errors >= prior_radius))
    assert correlated_empirical_tail <= correlated_chi_tail_bound + 0.002

    scaled_weak = spectra[:, -2] * np.exp(rates[0] * translations)
    scaled_weakest = spectra[:, -1] * np.exp(rates[1] * translations)
    scaled_determinants = determinants * np.exp(
        rates[:2].sum() * translations)
    assert abs(scaled_weak[-1] / scaled_weak[-2] - 1.0) < 0.002
    assert abs(scaled_weakest[-1] / scaled_weakest[-2] - 1.0) < 0.002
    assert abs(scaled_determinants[-1] / scaled_determinants[-2] - 1.0) < 1e-5

    print("\nSharp long-end observability conditioning")
    print(f"nonzero decay rates: {rates}")
    print(f"measured singular-value rates: {fitted_rates}")
    print(f"measured determinant rate: {determinant_rate:.9f}")
    print(f"measured condition-number rate: {condition_rate:.9f}")
    print(f"centered-prior least-singular-value rate: "
          f"{statistical_rate:.9f}")
    print(f"correlated-noise whitened rate: "
          f"{correlated_statistical_rate:.9f}")
    print("terminal scaled weak singular values: "
          f"{scaled_weak[-1]:.9f}, {scaled_weakest[-1]:.9f}")
    print(f"terminal scaled determinant: {scaled_determinants[-1]:.9f}")
    print("Gaussian two-prior testing lower bounds: "
          + ", ".join(f"{value:.9f}" for value in testing_bounds))
    print(f"Gaussian testing error, exact vs Monte Carlo: "
          f"{exact_testing_error:.9f}, {empirical_testing_error:.9f}")
    print("projected inverse exact/empirical unconstrained MSE: "
          f"{exact_unconstrained_mse:.9e}, "
          f"{empirical_unconstrained_mse:.9e}")
    print("projected recovery tail and chi upper bound: "
          f"{empirical_projected_tail:.9f}, {chi_tail_bound:.9f}")
    print(f"projected recovery RMS upper bound: {rms_upper_bound:.9f}")
    print(f"maximum simplex-projection inflation: "
          f"{maximum_projection_inflation:.3e}")
    print("correlated quote-covariance eigenvalues: "
          f"{quote_covariance_eigenvalues}")
    print("correlated Gaussian testing error, exact vs Monte Carlo: "
          f"{correlated_exact_testing_error:.9f}, "
          f"{correlated_empirical_testing_error:.9f}")
    print("correlated GLS exact/empirical unconstrained MSE: "
          f"{correlated_exact_mse:.9e}, {correlated_empirical_mse:.9e}")
    print("correlated projected tail and chi upper bound: "
          f"{correlated_empirical_tail:.9f}, "
          f"{correlated_chi_tail_bound:.9f}")


def verify_complex_long_end_aliasing():
    """Oscillatory modes create infinitely many translated rank losses."""
    cycle_Q = np.array([
        [-1.0, 1.0, 0.0],
        [0.0, -1.0, 1.0],
        [1.0, 0.0, -1.0],
    ])
    cycle_feature = np.array([[1.0], [-1.0], [0.0]])
    decay = 1.5
    frequency = math.sqrt(3.0) / 2.0
    offsets = np.array([0.4, 1.7])
    c1, c2 = offsets

    # Substituting t_j=M+c_j into the exact three-state aliasing kernel gives
    #
    # K(M+c1,M+c2) = exp(-a M) L(M) + C exp(-2a M),
    # L(M) = exp(-a c1) sin(b(M+c1))
    #        -exp(-a c2) sin(b(M+c2)).
    #
    # Since c1 != c2, L is a nonzero sinusoid.  Its simple zeros therefore
    # perturb to infinitely many exact zeros of the full determinant.
    sine_coefficient = (
        math.exp(-decay * c1) * math.cos(frequency * c1)
        - math.exp(-decay * c2) * math.cos(frequency * c2)
    )
    cosine_coefficient = (
        math.exp(-decay * c1) * math.sin(frequency * c1)
        - math.exp(-decay * c2) * math.sin(frequency * c2)
    )
    amplitude = math.hypot(sine_coefficient, cosine_coefficient)
    phase = math.atan2(cosine_coefficient, sine_coefficient)
    correction = (
        math.exp(-decay * (c1 + c2))
        * math.sin(frequency * (c2 - c1))
    )
    assert amplitude > 0.5

    def leading(translation):
        return amplitude * math.sin(frequency * translation + phase)

    def scaled_kernel(translation):
        return leading(translation) + correction * math.exp(
            -decay * translation)

    def aliasing_kernel(t1, t2):
        return (
            math.exp(-decay * (t1 + t2))
            * math.sin(frequency * (t2 - t1))
            - math.exp(-decay * t2) * math.sin(frequency * t2)
            + math.exp(-decay * t1) * math.sin(frequency * t1)
        )

    cycle_constant = -2.0 / math.sqrt(3.0)
    maximum_factorization_error = 0.0
    for translation in np.linspace(0.0, 10.0, 41):
        taus = translation + offsets
        response = transient_response_matrix(
            cycle_Q, cycle_feature, taus)
        observed = np.linalg.det(np.column_stack([
            np.ones(3), response]))
        predicted = cycle_constant * aliasing_kernel(*taus)
        maximum_factorization_error = max(
            maximum_factorization_error, abs(observed - predicted))
        scaled_direct = aliasing_kernel(*taus) * math.exp(
            decay * translation)
        assert abs(scaled_direct - scaled_kernel(translation)) < 2e-14
    assert maximum_factorization_error < 1e-14

    # Locate four consecutive long-end singular translations.  If m_k is a
    # zero of L, then M_k-m_k=O(exp(-a m_k)); the normalized shifts converge
    # in magnitude to |C|/(b*amplitude).
    leading_zeros = []
    exact_zeros = []
    normalized_shifts = []
    for k in range(1, 5):
        leading_zero = (k * math.pi - phase) / frequency
        radius = math.pi / (4.0 * frequency)
        exact_zero = brentq(
            scaled_kernel,
            leading_zero - radius,
            leading_zero + radius,
            xtol=1e-14,
        )
        leading_zeros.append(leading_zero)
        exact_zeros.append(exact_zero)
        normalized_shifts.append(
            (exact_zero - leading_zero)
            * math.exp(decay * leading_zero))
        assert abs(scaled_kernel(exact_zero)) < 2e-15
    leading_zeros = np.asarray(leading_zeros)
    exact_zeros = np.asarray(exact_zeros)
    normalized_shifts = np.asarray(normalized_shifts)
    predicted_shift_magnitude = abs(correction) / (
        frequency * amplitude)
    assert abs(
        abs(normalized_shifts[-2]) / predicted_shift_magnitude - 1.0
    ) < 5e-5

    # At a singular translation the response columns are exactly collinear.
    # Direct matrix arithmetic remains reliable for the first two roots.
    root_singular_values = []
    root_relative_determinants = []
    for exact_zero in exact_zeros[:2]:
        response = transient_response_matrix(
            cycle_Q, cycle_feature, exact_zero + offsets)
        augmented = np.column_stack([np.ones(3), response])
        spectrum = np.linalg.svd(augmented, compute_uv=False)
        root_singular_values.append(spectrum[-1])
        root_relative_determinants.append(
            abs(np.linalg.det(augmented))
            / np.prod(np.linalg.norm(augmented, axis=0)))
    assert max(root_singular_values) < 2e-14
    assert max(root_relative_determinants) < 2e-14

    print("\nComplex-spectrum translated long-end aliasing")
    print(f"offsets: {offsets}")
    print(f"leading sinusoid amplitude: {amplitude:.12f}")
    print(f"maximum exact factorization error: "
          f"{maximum_factorization_error:.3e}")
    print(f"leading zeros: {leading_zeros}")
    print(f"exact singular translations: {exact_zeros}")
    print(f"normalized root shifts: {normalized_shifts}")
    print(f"predicted shift magnitude: {predicted_shift_magnitude:.12f}")
    print(f"maximum root relative determinant: "
          f"{max(root_relative_determinants):.3e}")


def verify_initial_mixture_observability():
    """Boundary-layer maturities can identify more than the outer projection."""
    Q = np.array([
        [-1.4, 1.1, 0.3],
        [0.2, -1.1, 0.9],
        [0.8, 0.4, -1.2],
    ])
    pi = stationary(Q)
    feature = np.array([[1.0], [-0.4], [0.7]])
    centered = feature - np.outer(np.ones(3), pi @ feature)
    H = -group_inverse(Q)
    taus = (0.4, 1.7)
    observability = np.column_stack([
        H @ (np.eye(3) - expm(Q * tau)) @ centered[:, 0]
        for tau in taus
    ])
    augmented = np.column_stack([np.ones(3), observability])
    singular_values = np.linalg.svd(observability, compute_uv=False)
    assert np.linalg.matrix_rank(observability, tol=2e-12) == 2
    assert abs(np.linalg.det(augmented)) > 0.04

    prior = np.array([0.15, 0.55, 0.30])
    maximum_quadrature_error = 0.0
    for m in (3.0, 11.0, 37.0):
        scaled_responses = []
        for tau in taus:
            T = tau / m
            closed = initial_mean_response(Q, feature, T, m, prior)[0]
            numerical = quad_vec(
                lambda s: prior @ expm(m * Q * s) @ centered[:, 0],
                0.0,
                T,
                epsabs=2e-13,
                epsrel=2e-13,
            )[0]
            maximum_quadrature_error = max(
                maximum_quadrature_error, abs(closed - numerical))
            scaled_responses.append(m * closed)
        recovered = np.linalg.solve(
            augmented.T, np.array([1.0, *scaled_responses]))
        assert np.max(np.abs(recovered - prior)) < 3e-13

    # The same phenomenon occurs in the yield curve itself on T=tau/m.
    # Since B(r/m)=r/m+O(m^-2), the initial-prior difference is
    # -kappa/m^2 times nu integral_0^tau (tau-s)e^{Qs}u ds, with an
    # O(m^-3) remainder.  Two tau values identify this three-state prior.
    kappa = 0.7
    theta = feature[:, 0]
    variance = np.array([0.12, 0.30, 0.20])
    yield_observability = np.column_stack([
        quad_vec(
            lambda s: (tau - s) * expm(Q * s) @ centered[:, 0],
            0.0,
            tau,
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        for tau in taus
    ])
    yield_augmented = np.column_stack([np.ones(3), yield_observability])
    yield_singular_values = np.linalg.svd(
        yield_observability, compute_uv=False)
    assert np.linalg.matrix_rank(yield_observability, tol=2e-12) == 2
    yield_coefficients = (prior - pi) @ yield_observability
    yield_recovered = np.linalg.solve(
        yield_augmented.T, np.array([1.0, *yield_coefficients]))
    assert np.max(np.abs(yield_recovered - prior)) < 3e-13

    def boundary_log_price(tau, m, initial, theta_values=theta,
                           variance_values=variance):
        T = tau / m

        def rhs(t, value):
            B = -math.expm1(-kappa * t) / kappa
            forcing = (-kappa * theta_values * B
                       + variance_values * B * B / 2)
            return (m * Q + np.diag(forcing)) @ value

        solution = solve_ivp(
            rhs, (0.0, T), np.ones(3), method="DOP853",
            rtol=2e-13, atol=2e-15)
        assert solution.success
        return math.log(initial @ solution.y[:, -1])

    combined_second_feature = (kappa ** 2 * centered[:, 0]
                               + variance - pi @ variance)
    second_yield_observability = np.column_stack([
        quad_vec(
            lambda s: ((tau - s) ** 2 * expm(Q * s)
                       @ combined_second_feature / 2),
            0.0,
            tau,
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        for tau in taus
    ])
    first_boundary_rates = []
    second_boundary_rates = []
    for tau, first_column, second_column in zip(
            taus, yield_observability.T, second_yield_observability.T):
        first_coefficient = -kappa * (prior - pi) @ first_column
        second_coefficient = (prior - pi) @ second_column
        first_errors = []
        second_errors = []
        for m in (20.0, 40.0, 80.0, 160.0):
            exact_difference = (
                boundary_log_price(tau, m, prior)
                - boundary_log_price(tau, m, pi))
            first_errors.append(abs(
                exact_difference - first_coefficient / m ** 2))
            second_errors.append(abs(
                exact_difference - first_coefficient / m ** 2
                - second_coefficient / m ** 3))
        first_boundary_rates.append(math.log(
            first_errors[-2] / first_errors[-1], 2))
        second_boundary_rates.append(math.log(
            second_errors[-2] / second_errors[-1], 2))
        assert first_boundary_rates[-1] > 2.99
        assert second_boundary_rates[-1] > 3.99

    # If theta is constant, the m^-2 initial-prior signal vanishes.  A
    # switched positive variance becomes the leading m^-3 signal and two
    # scaled maturities can still identify this three-state prior.
    constant_theta = np.full(3, 0.6)
    switched_variance = np.array([2.0, 0.6, 1.7])
    centered_variance = switched_variance - pi @ switched_variance
    volatility_observability = np.column_stack([
        quad_vec(
            lambda s: ((tau - s) ** 2 * expm(Q * s)
                       @ centered_variance / 2),
            0.0,
            tau,
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
        for tau in taus
    ])
    volatility_augmented = np.column_stack(
        [np.ones(3), volatility_observability])
    assert np.linalg.matrix_rank(volatility_observability, tol=2e-12) == 2
    volatility_coefficients = (prior - pi) @ volatility_observability
    volatility_recovered = np.linalg.solve(
        volatility_augmented.T,
        np.array([1.0, *volatility_coefficients]))
    assert np.max(np.abs(volatility_recovered - prior)) < 3e-13
    volatility_rates = []
    for tau, coefficient in zip(taus, volatility_coefficients):
        errors = []
        for m in (20.0, 40.0, 80.0, 160.0):
            exact_difference = (
                boundary_log_price(
                    tau, m, prior, constant_theta, switched_variance)
                - boundary_log_price(
                    tau, m, pi, constant_theta, switched_variance))
            errors.append(abs(exact_difference - coefficient / m ** 3))
        volatility_rates.append(math.log(errors[-2] / errors[-1], 2))
        assert volatility_rates[-1] > 3.99

    # With fixed positive maturities the two columns converge to the same
    # Poisson vector.  Exact finite-m rank survives, but its second singular
    # value becomes exponentially small and the extra direction is unstable.
    fixed_maturities = (0.4, 1.7)
    small_singular_values = []
    for m in (2.0, 4.0, 8.0, 16.0):
        fixed = np.column_stack([
            H @ (np.eye(3) - expm(m * Q * T)) @ centered[:, 0]
            for T in fixed_maturities
        ])
        small_singular_values.append(np.linalg.svd(fixed, compute_uv=False)[-1])
    assert small_singular_values[-1] < 1.1e-6
    assert small_singular_values[-1] < small_singular_values[0] / 40000

    print("\nInitial-mixture observability")
    print(f"boundary-layer observability rank: {np.linalg.matrix_rank(observability)}")
    print(f"boundary-layer singular values: {singular_values}")
    print(f"recovered initial prior: {recovered}")
    print(f"maximum exact formula vs quadrature error: {maximum_quadrature_error:.3e}")
    print(f"yield boundary-layer observability rank: "
          f"{np.linalg.matrix_rank(yield_observability)}")
    print(f"yield boundary-layer singular values: {yield_singular_values}")
    print(f"yield first boundary-layer remainder rates: {first_boundary_rates}")
    print(f"yield second boundary-layer remainder rates: {second_boundary_rates}")
    print(f"volatility-only observability rank: "
          f"{np.linalg.matrix_rank(volatility_observability)}")
    print(f"volatility-only recovered prior: {volatility_recovered}")
    print(f"volatility-only remainder rates: {volatility_rates}")
    print("fixed-maturity second singular values: "
          + ", ".join(f"{value:.8e}" for value in small_singular_values))


def verify_finite_horizon_covariance():
    """Exact operator formula, rank theorem, and fast-rate remainder."""
    rng = np.random.default_rng(31012026)
    rates = []
    for states, feature_count in ((3, 2), (5, 7)):
        Q0 = random_generator(states, rng)
        features = rng.normal(size=(states, feature_count))
        pi = stationary(Q0)
        centered, exact = finite_horizon_covariance(Q0, features, T=0.73, m=2.4)

        # Independent quadrature of integral_0^T (T-s){C(s)+C(s)^T} ds.
        D = np.diag(pi)
        numerical, _ = quad_vec(
            lambda s: (0.73 - s) * (
                centered.T @ D @ expm(2.4 * Q0 * s) @ centered
                + centered.T @ expm(2.4 * Q0.T * s) @ D @ centered
            ),
            0.0,
            0.73,
            epsabs=2e-12,
            epsrel=2e-12,
        )
        assert np.max(np.abs(exact - numerical)) < 2e-11
        assert np.linalg.eigvalsh(exact)[0] > -2e-11
        assert np.linalg.matrix_rank(exact, tol=2e-10) == np.linalg.matrix_rank(centered, tol=2e-10)

        _, gram = feature_gram(Q0, features)
        errors = []
        for m in (8.0, 16.0, 32.0, 64.0):
            _, covariance = finite_horizon_covariance(Q0, features, T=0.73, m=m)
            errors.append(np.linalg.norm(covariance - 2 * 0.73 * gram / m, ord=2))
        rates.append(math.log(errors[-2] / errors[-1], 2))
        assert rates[-1] > 1.98

    # Reversible chains have the stronger Loewner upper bound Sigma_T <= 2TG/m.
    conductance = rng.uniform(0.2, 1.5, size=(5, 5))
    conductance = (conductance + conductance.T) / 2
    np.fill_diagonal(conductance, 0.0)
    Q0 = conductance - np.diag(conductance.sum(axis=1))
    features = rng.normal(size=(5, 4))
    _, covariance = finite_horizon_covariance(Q0, features, T=1.1, m=3.0)
    _, gram = feature_gram(Q0, features)
    assert np.linalg.eigvalsh(2 * 1.1 * gram / 3.0 - covariance)[0] > -2e-12

    # Closed two-state example from Issue #31.
    Q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    _, covariance = finite_horizon_covariance(Q0, np.array([[1.0], [-1.0]]), T=1.0)
    expected = 1.0 - (1.0 - math.exp(-2.0)) / 2.0
    assert abs(covariance[0, 0] - expected) < 2e-14
    print("\nExact finite-horizon covariance")
    print(f"two-state exact variance: {covariance[0, 0]:.10f}")
    print(f"leading Green--Kubo value: {1.0:.10f}")
    print(f"last two measured remainder rates: {rates[0]:.6f}, {rates[1]:.6f}")


def verify_arbitrary_prior_rank():
    """Exact fixed-feature rank persists for every initial regime prior."""
    rng = np.random.default_rng(24092026)
    smallest_positive = math.inf
    maximum_stationary_error = 0.0
    tested = 0
    for states, feature_count in ((3, 2), (5, 7)):
        for _ in range(6):
            Q = random_generator(states, rng)
            features = rng.normal(size=(states, feature_count))
            pi = stationary(Q)
            centered = features - np.outer(np.ones(states), pi @ features)
            target_rank = np.linalg.matrix_rank(centered, tol=2e-10)
            priors = [pi, np.eye(states)[0], np.eye(states)[-1],
                      np.arange(1, states + 1) / sum(range(1, states + 1))]
            for prior in priors:
                covariance = arbitrary_prior_covariance(2.3 * Q, features, 0.73, prior)
                eigenvalues = np.linalg.eigvalsh(covariance)
                assert eigenvalues[0] > -3e-11
                assert np.linalg.matrix_rank(covariance, tol=2e-9) == target_rank
                positive = eigenvalues[eigenvalues > 2e-9]
                smallest_positive = min(smallest_positive, positive[0])
                tested += 1

            _, closed = finite_horizon_covariance(Q, features, T=0.73, m=2.3)
            ode = arbitrary_prior_covariance(2.3 * Q, features, 0.73, pi)
            maximum_stationary_error = max(
                maximum_stationary_error, float(np.max(np.abs(closed - ode))))

    assert maximum_stationary_error < 3e-11
    print("\nArbitrary-prior finite-horizon rank")
    print(f"prior/case combinations checked: {tested}")
    print(f"smallest positive covariance eigenvalue: {smallest_positive:.6e}")
    print(f"maximum stationary formula vs moment-ODE error: {maximum_stationary_error:.3e}")


def verify_general_gram_theorem():
    """PSD and exact rank for fixed features; integrated rank can be larger."""
    rng = np.random.default_rng(20260924)
    for states, feature_count in ((3, 2), (4, 6), (7, 5)):
        for _ in range(10):
            Q = random_generator(states, rng)
            features = rng.normal(size=(states, feature_count))
            centered, gram = feature_gram(Q, features)
            eigenvalues = np.linalg.eigvalsh(gram)
            assert eigenvalues[0] > -2e-12
            feature_rank = np.linalg.matrix_rank(centered, tol=2e-10)
            gram_rank = np.linalg.matrix_rank(gram, tol=2e-10)
            assert gram_rank == feature_rank
            assert gram_rank <= states - 1

    # A two-state chain has one centered feature direction.  The fixed Gram
    # matrix therefore has rank one, but integrating a loading vector whose
    # range rotates with maturity produces a full-rank 3x3 moment matrix.
    Q = np.array([[-1.3, 1.3], [0.7, -0.7]])
    centered, gram = feature_gram(Q, np.array([[-1.0], [1.0]]))
    assert np.linalg.matrix_rank(centered) == np.linalg.matrix_rank(gram) == 1
    powers = np.arange(3)
    hilbert = 1 / (powers[:, None] + powers[None, :] + 1)
    integrated = gram[0, 0] * hilbert
    assert np.linalg.eigvalsh(integrated)[0] > 1e-5
    assert np.linalg.matrix_rank(integrated) == 3

    # Exact loading-rank theorem.  Let G=RR' have fixed rank r and let
    # L(t)=sum_j t^j L_j.  The integrated matrix factors as
    # B (Gamma kron I_r) B', B=[L_0 R ... L_{q-1} R], where Gamma is the
    # positive-definite moment Gram matrix.  Thus its rank is rank(B), which
    # may reach q*r rather than r.
    rng = np.random.default_rng(20260925)
    feature_count, feature_rank = 4, 2
    contracts, basis_count = 7, 3
    R = rng.normal(size=(feature_count, feature_rank))
    fixed_gram = R @ R.T
    loading_blocks = rng.normal(
        size=(basis_count, contracts, feature_count))
    block_matrix = np.concatenate(
        [loading_blocks[j] @ R for j in range(basis_count)], axis=1)
    powers = np.arange(basis_count)
    basis_gram = 1 / (powers[:, None] + powers[None, :] + 1)
    factored = block_matrix @ np.kron(
        basis_gram, np.eye(feature_rank)) @ block_matrix.T

    def integrand(t):
        loading = sum(t ** j * loading_blocks[j]
                      for j in range(basis_count))
        return (loading @ fixed_gram @ loading.T).ravel()

    quadrature = quad_vec(integrand, 0.0, 1.0, epsabs=1e-13)[0].reshape(
        contracts, contracts)
    loading_error = np.max(np.abs(quadrature - factored))
    integrated_rank = np.linalg.matrix_rank(factored, tol=2e-10)
    span_rank = np.linalg.matrix_rank(block_matrix, tol=2e-10)
    assert loading_error < 2e-12
    assert integrated_rank == span_rank == basis_count * feature_rank
    null_vector = np.linalg.svd(block_matrix, full_matrices=True)[0][:, -1]
    assert np.linalg.norm(factored @ null_vector) < 2e-12

    print("\nGeneral fixed-feature Gram theorem")
    print(f"two-state fixed Gram rank: {np.linalg.matrix_rank(gram)}")
    print(f"maturity-integrated loading rank: {np.linalg.matrix_rank(integrated)}")
    print(f"integrated eigenvalues: {np.linalg.eigvalsh(integrated)}")
    print(f"finite-basis fixed rank -> integrated rank: "
          f"{feature_rank} -> {integrated_rank}")
    print(f"loading factorization vs quadrature error: {loading_error:.3e}")


def main():
    verify_general_gram_theorem()
    verify_finite_horizon_covariance()
    verify_arbitrary_prior_rank()
    verify_krylov_observability()
    verify_real_spectrum_all_maturities()
    verify_long_end_conditioning()
    verify_complex_long_end_aliasing()
    verify_initial_mixture_observability()

    for T in (0.2, 1.0, 3.0, 5.0):
        B = -math.expm1(-KAPPA * T) / KAPPA
        assert abs(B - (T - KAPPA * int_Bk(1, T, KAPPA))) < 2e-13
        assert abs(B ** 2 - (2 * int_Bk(1, T, KAPPA) - 2 * KAPPA * int_Bk(2, T, KAPPA))) < 2e-13

    print("m   T  m^2 * stationary error   m^2 * max known-start error")
    for T in (1.0, 3.0):
        pi = stationary(QA)
        for m in (10, 20, 40, 80):
            result = exact(T, m)
            stat_error = abs(math.log(pi @ np.exp(result)) - first_order(T, m))
            known_error = max(abs(result[i] - first_order(T, m, i)) for i in range(len(pi)))
            print(f"{m:2}  {T:.0f}  {m * m * stat_error:.8g}  {m * m * known_error:.8g}")
            stat_bound, known_bound = error_bounds(T, m)
            assert stat_error <= stat_bound + 5e-13
            assert known_error <= known_bound + 5e-13
            assert stat_error < 0.005 / m ** 2
            assert known_error < 0.05 / m ** 2

    print(
        "PASS: general and arbitrary-prior finite-horizon Gram rank, exact covariance, "
        "exact integrated-loading rank, Krylov, real-spectrum, generic-rank, and "
        "three-state exceptional-set observability, sharp real-spectrum and "
        "oscillatory long-end conditioning and whitened noisy recovery, "
        "shape identities, "
        "known-start expansion, and explicit bounds"
    )


if __name__ == "__main__":
    main()
