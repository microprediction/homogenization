"""Certificate for positive CIR intensities and the maturity-loading rank trap.

The two-state chain has an instantaneous rank-one Green-Kubo matrix, but two
different CIR Riccati loadings give a rank-two integrated correction matrix.
More generally, the Hadamard-rank bound is checked at every feasible pair of
ranks through dimension eight, and positive-volatility CIR loadings make the
integrated correction full rank through eight names despite only two regimes.
For four nearly coalescing mean-reversion rates, a high-precision certificate
checks the confluent-Vandermonde determinant constant, the eigenvalue powers
0, 2, 4, 6 and their leading constants, and the resulting sixth-power
condition-number blow-up.
For partially coalescing rates in clusters of sizes three, two, and one, a
second high-precision check verifies the Hermite-jet exponent multiset
0, 0, 0, 2, 2, 4, the eighth-power determinant law, and the fourth-power
condition-number blow-up.  An exact rational Taylor-coefficient calculation
proves that the joint jet independence assumed by the generic cluster theorem
is automatic for the CIR Riccati loading at every set of distinct centers.
At the opposite, long-maturity limit, a four-name certificate checks that the
loading Gram matrix has one eigenvalue growing linearly in maturity while the
other three converge to positive transient-Gram limits, so its condition
number grows linearly despite retaining full algebraic rank.  It also checks
the rank-one-corrected inverse expansion in operator norm, including its
second-order remainder and the finite inverse-information floor on the
transient subspace.  A rank-two fixed Green--Kubo certificate then checks the
general law: exactly r information eigenvalues grow linearly when the fixed
matrix has rank r, while maturity integration can make every finite-maturity
matrix full rank through a positive transient complement.  An additional
certificate checks the exact full-rank criterion: for distinct CIR
mean-reversion rates, K Hadamard J(T) is positive definite at every positive
maturity if and only if every diagonal entry of the positive semidefinite K
is positive, irrespective of the rank of K.  For repeated loading shapes it
checks the exact cluster formula: the integrated rank is the sum of the ranks
of the within-cluster principal blocks of K.  A separate heterogeneous-CIR
check gives the complete two-name criterion: equal mean reversion alone does
not collapse rank when vol-of-vol differs; the loading Gramian is singular
exactly when both Riccati parameter pairs coincide.
The direct pricing ODE independently checks the pairwise first-order formula
and the endpoint-memory correction for an arbitrary initial regime prior.  A
separate constant-hazard example verifies that two competing default channels
already identify the antisymmetric Green--Kubo component.  The multi-cause
check proves that d-1 independent exposure designs are necessary and
sufficient to identify a d-by-d antisymmetric component.  It also verifies
the sharp square-root-two conditioning infimum and the fixed-exposure minimax
bound for minimally identifying designs.
"""
import math
from fractions import Fraction

import mpmath as mp
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

KAPPA = np.array([1.0, 2.0])
SIGMA = np.array([0.1, 0.1])
THETA = np.array([[0.06, 0.02], [0.05, 0.015]])
X0 = np.array([0.03, 0.03])
PI = np.array([0.5, 0.5])
Q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
T = 5.0


def loadings(T):
    Bs = []
    for kappa, sigma in zip(KAPPA, SIGMA):
        result = solve_ivp(
            lambda t, b: 1 - kappa * b - 0.5 * sigma ** 2 * b ** 2,
            (0, T), [0.0], method="DOP853", rtol=1e-12, atol=1e-14,
            dense_output=True,
        )
        assert result.success
        Bs.append(lambda t, sol=result.sol: float(sol(t)[0]))
    return Bs


def exact_log_survival(names, m, Bs, prior=PI):
    def rhs(t, a):
        g = -sum(KAPPA[j] * THETA[j] * Bs[j](t) for j in names)
        return m * Q0 @ a + g * a

    result = solve_ivp(rhs, (0, T), np.ones(2), method="BDF",
                       rtol=2e-12, atol=1e-14)
    assert result.success
    return math.log(prior @ result.y[:, -1]) - sum(Bs[j](T) * X0[j] for j in names)


def group_inverse(Q):
    P = np.ones((len(PI), 1)) @ PI[None, :]
    return np.linalg.inv(Q + P) - P


def stationary(Q):
    values, vectors = np.linalg.eig(np.asarray(Q, float).T)
    pi = np.real(vectors[:, np.argmin(abs(values))])
    return pi / pi.sum()


def general_group_inverse(Q):
    pi = stationary(Q)
    projection = np.outer(np.ones(len(pi)), pi)
    return np.linalg.inv(Q + projection) - projection


def first_event_probabilities(Q, hazards, weights=None):
    """Exact cause probabilities in a stationary competing-risks race."""
    pi = stationary(Q)
    hazards = np.asarray(hazards, float)
    if weights is None:
        weights = np.ones(len(hazards))
    scaled = np.asarray(weights, float)[:, None] * hazards
    total = scaled.sum(axis=0)
    return pi @ np.linalg.solve(np.diag(total) - Q, scaled.T)


def first_event_correction(Q0, hazards, weights=None):
    """Vector coefficients in p=leading+coefficient/m+O(m^-2)."""
    pi = stationary(Q0)
    hazards = np.asarray(hazards, float)
    if weights is None:
        weights = np.ones(len(hazards))
    scaled = np.asarray(weights, float)[:, None] * hazards
    means = scaled @ pi
    centered = scaled - means[:, None]
    group = general_group_inverse(Q0)
    green_kubo = np.array(
        [[-pi @ (centered[j] * (group @ centered[k]))
          for k in range(len(hazards))]
         for j in range(len(hazards))]
    )
    mean_total = means.sum()
    k_total_total = green_kubo.sum()
    coefficient = (
        (means / mean_total) * k_total_total - green_kubo.sum(axis=0)
    ) / mean_total
    return means / mean_total, coefficient, green_kubo


def skew_measurement_matrix(designs):
    """Matrix sending the upper triangle of skew A to stacked A @ designs."""
    designs = [np.asarray(w, float) for w in designs]
    d = len(designs[0])
    columns = []
    pairs = []
    for i in range(d):
        for j in range(i + 1, d):
            basis = np.zeros((d, d))
            basis[i, j] = 1
            basis[j, i] = -1
            columns.append(np.concatenate([basis @ w for w in designs]))
            pairs.append((i, j))
    return np.column_stack(columns), pairs


def predicted_skew_singular_values(designs):
    """Singular values for the upper-triangle coefficient norm on skew A."""
    design_matrix = np.column_stack(designs)
    eigenvalues = np.linalg.eigvalsh(design_matrix @ design_matrix.T)
    eigenvalues[eigenvalues < 1e-12 * eigenvalues[-1]] = 0.0
    values = [
        math.sqrt(max(0.0, eigenvalues[i] + eigenvalues[j]))
        for i in range(len(eigenvalues))
        for j in range(i + 1, len(eigenvalues))
    ]
    return np.sort(values)[::-1]


def near_simplex_designs(d, epsilon):
    """Explicit positive d-1 design family 1+epsilon*e_r."""
    return [
        np.ones(d) + epsilon * np.eye(d)[r]
        for r in range(d - 1)
    ]


def recover_skew_near_simplex(responses, epsilon):
    """Closed-form recovery for the near-simplex positive designs."""
    d = len(responses) + 1
    b = np.zeros(d)
    for r, response in enumerate(responses):
        b[r] = response[r]
    b[-1] = -b[:-1].sum()
    recovered = np.zeros((d, d))
    for r, response in enumerate(responses):
        recovered[:, r] = (response - b) / epsilon
    recovered[:, -1] = b - recovered[:, :-1].sum(axis=1)
    return recovered


def recover_skew(designs, responses):
    """Recover a skew matrix from exact products A w by least squares."""
    measurement, pairs = skew_measurement_matrix(designs)
    rhs = np.concatenate(responses)
    coefficients, residuals, rank, _ = np.linalg.lstsq(measurement, rhs, rcond=None)
    assert rank == len(pairs)
    assert residuals.size == 0 or residuals.max() < 1e-28
    recovered = np.zeros((len(designs[0]), len(designs[0])))
    for value, (i, j) in zip(coefficients, pairs):
        recovered[i, j] = value
        recovered[j, i] = -value
    return recovered


def first_default_probability(Q, h1, h2):
    """Exact probability that channel 1 fires before channel 2."""
    return float(first_event_probabilities(Q, [h1, h2])[0])


def first_default_correction(Q0, h1, h2):
    """Leading value and coefficient in p1=leading+coefficient/m+O(m^-2)."""
    leading, coefficient, green_kubo = first_event_correction(Q0, [h1, h2])
    return leading[0], coefficient[0], green_kubo


def first_log_survival(names, m, Bs, D, prior=PI):
    leading = -sum(
        Bs[j](T) * X0[j]
        + KAPPA[j] * (PI @ THETA[j])
        * quad(Bs[j], 0, T, epsabs=1e-12)[0]
        for j in names
    )
    slow = sum(D[j, k] for j in names for k in names) / m
    H = -group_inverse(Q0)
    endpoint = sum(
        -KAPPA[j] * (THETA[j] - (PI @ THETA[j])) * Bs[j](T)
        for j in names
    )
    memory = prior @ H @ endpoint / m
    return leading + slow + memory


def default_correlation(m, Bs, prior):
    s1 = math.exp(exact_log_survival((0,), m, Bs, prior))
    s2 = math.exp(exact_log_survival((1,), m, Bs, prior))
    s12 = math.exp(exact_log_survival((0, 1), m, Bs, prior))
    return ((s12 - s1 * s2)
            / math.sqrt(s1 * (1 - s1) * s2 * (1 - s2)))


def page_prior_benchmark():
    """Reproduce the arbitrary-prior numbers on the two-name credit page."""
    kappa, sigma, horizon, m = 2.0, 0.2, 3.0, 2.0
    theta = np.array([[0.8, 0.005], [0.6, 0.005]])
    x0 = np.array([0.05, 0.05])
    result = solve_ivp(
        lambda t, b: 1 - kappa * b - 0.5 * sigma ** 2 * b ** 2,
        (0, horizon), [0.0], method="DOP853", rtol=1e-12, atol=1e-14,
        dense_output=True,
    )
    assert result.success
    B = lambda t: float(result.sol(t)[0])

    def survival(names, prior):
        def rhs(t, a):
            g = -kappa * sum(theta[j] for j in names) * B(t)
            return m * Q0 @ a + g * a

        solved = solve_ivp(rhs, (0, horizon), np.ones(2), method="BDF",
                           rtol=2e-12, atol=1e-14)
        assert solved.success
        return (math.exp(-B(horizon) * sum(x0[j] for j in names))
                * (prior @ solved.y[:, -1]))

    correlations = []
    for q in (0.1, 0.5, 0.9):
        prior = np.array([q, 1 - q])
        s1, s2 = survival((0,), prior), survival((1,), prior)
        s12 = survival((0, 1), prior)
        correlations.append(
            (s12 - s1 * s2)
            / math.sqrt(s1 * (1 - s1) * s2 * (1 - s2)))
    expected = np.array([0.101692, 0.095799, 0.083268])
    assert np.allclose(correlations, expected, atol=5e-7)
    return correlations


def rank_amplification_checks():
    """Check the sharp Hadamard-rank bound and many-name CIR amplification."""
    rng = np.random.default_rng(20260928)
    maximum_factor_error = 0.0
    sharp_regime_bounds = []
    for d in range(2, 9):
        for rank_k in range(1, min(3, d) + 1):
            for rank_j in range(1, min(3, d) + 1):
                u = rng.normal(size=(d, rank_k))
                v = rng.normal(size=(d, rank_j))
                k_matrix = u @ u.T
                j_matrix = v @ v.T
                integrated = k_matrix * j_matrix
                tensor_rows = np.einsum("ir,is->irs", u, v).reshape(
                    d, rank_k * rank_j
                )
                maximum_factor_error = max(
                    maximum_factor_error,
                    np.max(abs(integrated - tensor_rows @ tensor_rows.T)),
                )
                assert np.linalg.matrix_rank(integrated, tol=1e-10) == min(
                    d, rank_k * rank_j
                )

    # The corrected regime-count inference is
    # n >= 1 + ceil(rank(K Hadamard J) / rank(J)).  It is algebraically sharp
    # for every pair (rank(K), rank(J)): choose d >= rank(K) rank(J) and
    # generic row factors, so the rowwise tensor products have full rank.
    for rank_k in range(1, 5):
        for rank_j in range(1, 5):
            d = rank_k * rank_j
            u = rng.normal(size=(d, rank_k))
            v = rng.normal(size=(d, rank_j))
            integrated = (u @ u.T) * (v @ v.T)
            observed_rank = np.linalg.matrix_rank(integrated, tol=1e-10)
            inferred_states = 1 + math.ceil(observed_rank / rank_j)
            assert observed_rank == d
            assert inferred_states == rank_k + 1
            sharp_regime_bounds.append(
                (rank_k, rank_j, observed_rank, inferred_states)
            )

    # With two regimes, choose centered hazard contrasts so that K=11'/2.
    # Distinct mean-reversion rates make the zero-volatility loadings
    # (1-exp(-kappa*t))/kappa linearly independent.  Small positive
    # volatilities preserve the Gram determinant by continuity.
    rows = []
    horizon = 10.0
    for d in range(2, 9):
        kappas = np.geomspace(0.2, 5.0, d)
        sigmas = np.full(d, 0.02)
        contrasts = 1 / kappas
        mean_levels = 1.2 * contrasts
        theta = np.column_stack(
            [mean_levels + contrasts, mean_levels - contrasts]
        )
        assert np.min(2 * kappas[:, None] * theta - sigmas[:, None] ** 2) > 0

        loading_functions = []
        for kappa, sigma in zip(kappas, sigmas):
            solved = solve_ivp(
                lambda t, b, k=kappa, s=sigma:
                    1 - k * b - 0.5 * s**2 * b**2,
                (0, horizon), [0.0], method="DOP853",
                rtol=1e-12, atol=1e-14, dense_output=True,
            )
            assert solved.success
            loading_functions.append(
                lambda t, sol=solved.sol: float(sol(t)[0])
            )
        gram = np.array([
            [quad(lambda t, fj=loading_functions[j], fk=loading_functions[k]:
                  fj(t) * fk(t), 0, horizon,
                  epsabs=1e-13, epsrel=1e-13)[0]
             for k in range(d)]
            for j in range(d)
        ])
        instantaneous = np.ones((d, d)) / 2
        integrated = instantaneous * gram
        eigenvalues = np.linalg.eigvalsh(integrated)
        assert np.linalg.matrix_rank(instantaneous) == 1
        assert np.linalg.matrix_rank(integrated, tol=1e-10) == d
        assert eigenvalues[0] > 1e-8
        rows.append((d, eigenvalues[0], eigenvalues[-1] / eigenvalues[0]))

    assert maximum_factor_error < 5e-13
    print("\nsharp maturity-loading rank amplification")
    print(f"  tensor-factor identity error {maximum_factor_error:.2e}")
    print(
        "  corrected regime bound sharp in "
        f"{len(sharp_regime_bounds)} rank pairs through r=s=4"
    )
    for d, smallest, condition in rows:
        print(f"  d={d}: smallest eigenvalue {smallest:.9e}, "
              f"condition {condition:.6e}")
    return rows


def long_maturity_loading_checks():
    """Certify the long-maturity rank-one collapse of CIR loading Gramians.

    If b is the vector of limiting Riccati loadings and r(t)=B(t)-b, then
    J(T)=T b b' + C + exponentially small terms.  On b-perp the cross terms
    in C vanish, leaving the positive transient Gramian int r(t)r(t)'dt.
    Block inversion further gives J(T)^-1=H^+ + ww'/(|b|^2 T)+O(T^-2).
    """
    kappas = np.array([0.35, 0.8, 1.7, 3.2])
    sigma = 0.18
    gammas = np.sqrt(kappas**2 + 2 * sigma**2)
    limiting_loadings = 2 / (gammas + kappas)

    def loading_vector(time):
        decay = np.exp(-gammas * time)
        return 2 * (1 - decay) / (
            (gammas + kappas) * (1 - decay) + 2 * gammas * decay
        )

    def gram(horizon):
        return np.array([
            [quad(
                lambda time, j=j, k=k:
                loading_vector(time)[j] * loading_vector(time)[k],
                0,
                horizon,
                epsabs=1e-11,
                epsrel=1e-12,
                limit=300,
            )[0] for k in range(len(kappas))]
            for j in range(len(kappas))
        ])

    def transient(time):
        return loading_vector(time) - limiting_loadings

    transient_gram = np.array([
        [quad(
            lambda time, j=j, k=k:
            transient(time)[j] * transient(time)[k],
            0,
            np.inf,
            epsabs=1e-12,
            epsrel=1e-12,
            limit=300,
        )[0] for k in range(len(kappas))]
        for j in range(len(kappas))
    ])
    transient_integral = np.array([
        quad(
            lambda time, j=j: transient(time)[j],
            0,
            np.inf,
            epsabs=1e-12,
            epsrel=1e-12,
            limit=300,
        )[0] for j in range(len(kappas))
    ])
    constant_matrix = (
        np.outer(limiting_loadings, transient_integral)
        + np.outer(transient_integral, limiting_loadings)
        + transient_gram
    )
    loading_norm_squared = limiting_loadings @ limiting_loadings
    unit_loading = limiting_loadings / np.sqrt(loading_norm_squared)
    projector = np.eye(len(kappas)) - np.outer(unit_loading, unit_loading)
    _, _, right_vectors = np.linalg.svd(limiting_loadings[None, :])
    complement = right_vectors[1:].T
    compressed = complement.T @ transient_gram @ complement
    limiting_small_eigenvalues = np.linalg.eigvalsh(compressed)
    assert limiting_small_eigenvalues[0] > 0
    transient_inverse = complement @ np.linalg.inv(compressed) @ complement.T
    coupling = projector @ constant_matrix @ unit_loading
    inverse_direction = unit_loading - transient_inverse @ coupling
    inverse_floor_trace = np.trace(np.linalg.inv(compressed))

    horizons = 2.0 ** np.arange(5, 12)
    values = []
    conditions = []
    determinants = []
    inverse_errors = []
    stationary_variances = []
    transient_traces = []
    for horizon in horizons:
        matrix = gram(horizon)
        eigenvalues = np.linalg.eigvalsh(matrix)
        values.append(np.r_[eigenvalues[:-1], eigenvalues[-1] / horizon])
        conditions.append(eigenvalues[-1] / eigenvalues[0] / horizon)
        determinants.append(np.linalg.det(matrix) / horizon)
        matrix_inverse = np.linalg.inv(matrix)
        inverse_approximation = (
            transient_inverse
            + np.outer(inverse_direction, inverse_direction)
            / (loading_norm_squared * horizon)
        )
        inverse_errors.append(np.linalg.norm(
            matrix_inverse - inverse_approximation, ord=2
        ))
        stationary_variances.append(
            horizon * limiting_loadings @ matrix_inverse @ limiting_loadings
        )
        transient_traces.append(np.trace(
            complement.T @ matrix_inverse @ complement
        ))

    values = np.asarray(values)
    targets = np.r_[
        limiting_small_eigenvalues,
        loading_norm_squared,
    ]
    errors = np.abs(values - targets)
    final_orders = np.log2(errors[-2] / errors[-1])
    assert np.min(final_orders) > 0.98
    assert np.max(np.abs(values[-1] / targets - 1)) < 0.004

    condition_target = (
        limiting_loadings @ limiting_loadings
        / limiting_small_eigenvalues[0]
    )
    determinant_target = (
        (limiting_loadings @ limiting_loadings) * np.linalg.det(compressed)
    )
    assert abs(conditions[-1] / condition_target - 1) < 0.003
    assert abs(determinants[-1] / determinant_target - 1) < 0.01

    inverse_orders = np.log2(
        np.asarray(inverse_errors[:-1]) / np.asarray(inverse_errors[1:])
    )
    assert inverse_orders[-1] > 1.99
    assert inverse_errors[-1] < 0.01
    assert abs(stationary_variances[-1] - 1) < 0.005
    assert abs(transient_traces[-1] / inverse_floor_trace - 1) < 0.001

    print("\nlong-maturity CIR loading certificate")
    print("  limiting bounded eigenvalues "
          + " ".join(f"{value:.10e}"
                     for value in limiting_small_eigenvalues))
    print("  T=2048 bounded eigenvalues "
          + " ".join(f"{value:.10e}" for value in values[-1, :-1]))
    print("  convergence orders "
          + " ".join(f"{order:.6f}" for order in final_orders))
    print(f"  condition/T {conditions[-1]:.10e}, "
          f"limit {condition_target:.10e}")
    print(f"  determinant/T {determinants[-1]:.10e}, "
          f"limit {determinant_target:.10e}")
    print(f"  inverse remainder {inverse_errors[-1]:.10e}, "
          f"observed order {inverse_orders[-1]:.6f}")
    print(f"  T b'J^-1b {stationary_variances[-1]:.10f}, limit 1")
    print(f"  transient inverse trace {transient_traces[-1]:.10f}, "
          f"limit {inverse_floor_trace:.10f}")
    return (limiting_small_eigenvalues, values, final_orders,
            conditions, determinants, inverse_errors, inverse_orders,
            stationary_variances, transient_traces)


def finite_rank_long_maturity_checks():
    """Certify the rank-r long-maturity law for K Hadamard J(T).

    The fixed positive semidefinite Green--Kubo matrix K has rank two, while
    distinct CIR maturity loadings make D(T)=K Hadamard J(T) positive
    definite.  The stationary matrix A=diag(b)Kdiag(b) therefore supplies
    two order-T eigenvalues.  Compression of K Hadamard C to ker(A) supplies
    the other three finite limits and the limiting inverse-information floor.
    """
    kappas = np.array([0.25, 0.6, 1.2, 2.4, 5.0])
    sigma = 0.18
    features = np.column_stack((
        np.ones(len(kappas)),
        np.array([-2.0, -0.7, 0.2, 1.1, 2.3]),
    ))
    green_kubo = features @ features.T
    assert np.linalg.matrix_rank(green_kubo, tol=1e-11) == 2

    gammas = np.sqrt(kappas**2 + 2 * sigma**2)
    limiting_loadings = 2 / (gammas + kappas)

    def loading_vector(time):
        decay = np.exp(-gammas * time)
        return 2 * (1 - decay) / (
            (gammas + kappas) * (1 - decay) + 2 * gammas * decay
        )

    def loading_gram(horizon):
        return np.array([
            [quad(
                lambda time, j=j, k=k:
                loading_vector(time)[j] * loading_vector(time)[k],
                0,
                horizon,
                epsabs=1e-11,
                epsrel=1e-12,
                limit=300,
            )[0] for k in range(len(kappas))]
            for j in range(len(kappas))
        ])

    def transient(time):
        return loading_vector(time) - limiting_loadings

    transient_integral = np.array([
        quad(
            lambda time, j=j: transient(time)[j],
            0,
            np.inf,
            epsabs=1e-12,
            epsrel=1e-12,
            limit=300,
        )[0] for j in range(len(kappas))
    ])
    transient_gram = np.array([
        [quad(
            lambda time, j=j, k=k:
            transient(time)[j] * transient(time)[k],
            0,
            np.inf,
            epsabs=1e-12,
            epsrel=1e-12,
            limit=300,
        )[0] for k in range(len(kappas))]
        for j in range(len(kappas))
    ])
    loading_constant = (
        np.outer(limiting_loadings, transient_integral)
        + np.outer(transient_integral, limiting_loadings)
        + transient_gram
    )
    stationary_matrix = green_kubo * np.outer(
        limiting_loadings, limiting_loadings
    )
    constant_matrix = green_kubo * loading_constant

    stationary_values, stationary_vectors = np.linalg.eigh(stationary_matrix)
    rank = np.count_nonzero(stationary_values > 1e-10)
    assert rank == 2
    null_basis = stationary_vectors[:, :-rank]
    range_basis = stationary_vectors[:, -rank:]
    compressed = null_basis.T @ constant_matrix @ null_basis
    transient_values = np.linalg.eigvalsh(compressed)
    assert transient_values[0] > 0

    null_inverse = null_basis @ np.linalg.inv(compressed) @ null_basis.T
    range_inverse = (
        range_basis
        @ np.linalg.inv(range_basis.T @ stationary_matrix @ range_basis)
        @ range_basis.T
    )
    null_projector = null_basis @ null_basis.T
    range_projector = range_basis @ range_basis.T
    inverse_lift = (
        range_projector
        - null_inverse @ null_projector @ constant_matrix @ range_projector
    )

    horizons = 2.0 ** np.arange(5, 12)
    values = []
    conditions = []
    determinants = []
    inverse_errors = []
    range_traces = []
    null_traces = []
    for horizon in horizons:
        matrix = green_kubo * loading_gram(horizon)
        eigenvalues = np.linalg.eigvalsh(matrix)
        assert eigenvalues[0] > 0
        values.append(np.r_[eigenvalues[:-rank], eigenvalues[-rank:] / horizon])
        conditions.append(eigenvalues[-1] / eigenvalues[0] / horizon)
        determinants.append(np.linalg.det(matrix) / horizon**rank)
        matrix_inverse = np.linalg.inv(matrix)
        inverse_approximation = (
            null_inverse
            + inverse_lift @ range_inverse @ inverse_lift.T / horizon
        )
        inverse_errors.append(np.linalg.norm(
            matrix_inverse - inverse_approximation, ord=2
        ))
        range_traces.append(horizon * np.trace(
            range_basis.T @ matrix_inverse @ range_basis
        ))
        null_traces.append(np.trace(
            null_basis.T @ matrix_inverse @ null_basis
        ))

    values = np.asarray(values)
    targets = np.r_[transient_values, stationary_values[-rank:]]
    final_orders = np.log2(
        np.abs(values[-2] - targets) / np.abs(values[-1] - targets)
    )
    assert np.min(final_orders) > 0.99
    assert np.max(np.abs(values[-1] / targets - 1)) < 0.003

    condition_target = stationary_values[-1] / transient_values[0]
    determinant_target = (
        np.prod(stationary_values[-rank:]) * np.linalg.det(compressed)
    )
    assert abs(conditions[-1] / condition_target - 1) < 0.002
    assert abs(determinants[-1] / determinant_target - 1) < 0.006

    inverse_orders = np.log2(
        np.asarray(inverse_errors[:-1]) / np.asarray(inverse_errors[1:])
    )
    range_trace_target = np.trace(
        np.linalg.inv(range_basis.T @ stationary_matrix @ range_basis)
    )
    null_trace_target = np.trace(np.linalg.inv(compressed))
    assert inverse_orders[-1] > 1.99
    assert inverse_errors[-1] < 0.004
    assert abs(range_traces[-1] / range_trace_target - 1) < 0.002
    assert abs(null_traces[-1] / null_trace_target - 1) < 0.001

    print("\nrank-r long-maturity CIR loading certificate")
    print(f"  fixed Green-Kubo rank {rank}, integrated rank {len(kappas)}")
    print("  limiting transient eigenvalues "
          + " ".join(f"{value:.10e}" for value in transient_values))
    print("  stationary eigenvalues "
          + " ".join(f"{value:.10e}"
                     for value in stationary_values[-rank:]))
    print("  T=2048 scaled spectrum "
          + " ".join(f"{value:.10e}" for value in values[-1]))
    print("  convergence orders "
          + " ".join(f"{order:.6f}" for order in final_orders))
    print(f"  condition/T {conditions[-1]:.10e}, "
          f"limit {condition_target:.10e}")
    print(f"  determinant/T^{rank} {determinants[-1]:.10e}, "
          f"limit {determinant_target:.10e}")
    print(f"  inverse remainder {inverse_errors[-1]:.10e}, "
          f"observed order {inverse_orders[-1]:.6f}")
    print(f"  range inverse trace {range_traces[-1]:.10f}, "
          f"limit {range_trace_target:.10f}")
    print(f"  null inverse trace {null_traces[-1]:.10f}, "
          f"limit {null_trace_target:.10f}")
    return (transient_values, stationary_values[-rank:], values,
            final_orders, inverse_errors, inverse_orders)


def integrated_full_rank_criterion_checks():
    """Check the sharp finite-maturity full-rank criterion.

    Distinct CIR mean-reversion rates make the loading Gramian J(T) positive
    definite for every T > 0.  If K = F F' is positive semidefinite, then
    K Hadamard J is the sum over feature columns f of
    diag(f) J diag(f).  It is therefore positive definite exactly when every
    row of F is nonzero, equivalently when every diagonal entry of K is
    positive.  The check uses fixed matrices of ranks one, two, and three,
    followed by a zero-diagonal counterexample.
    """
    kappas = np.array([0.3, 0.6, 1.1, 2.0, 3.7])
    sigma = 0.25
    horizon = 4.0
    gammas = np.sqrt(kappas**2 + 2 * sigma**2)

    def loadings_at(time):
        decay = np.exp(-gammas * time)
        return 2 * (1 - decay) / (
            (gammas + kappas) * (1 - decay) + 2 * gammas * decay
        )

    loading_gram = np.array([
        [quad(
            lambda time, j=j, k=k:
            loadings_at(time)[j] * loadings_at(time)[k],
            0,
            horizon,
            epsabs=1e-13,
            epsrel=1e-13,
            limit=300,
        )[0] for k in range(len(kappas))]
        for j in range(len(kappas))
    ])
    loading_eigenvalues = np.linalg.eigvalsh(loading_gram)
    assert loading_eigenvalues[0] > 8e-7

    features = [
        np.array([[1.0], [-0.7], [0.3], [1.2], [-2.0]]),
        np.column_stack((
            np.ones(5),
            np.array([-2.0, -0.7, 0.2, 1.1, 2.3]),
        )),
        np.column_stack((
            np.ones(5),
            np.array([-2.0, -0.7, 0.2, 1.1, 2.3]),
            np.array([0.5, -1.3, 0.8, 2.2, -0.4]),
        )),
    ]
    minimum_eigenvalues = []
    for rank, feature in enumerate(features, start=1):
        green_kubo = feature @ feature.T
        assert np.linalg.matrix_rank(green_kubo, tol=1e-11) == rank
        integrated = green_kubo * loading_gram
        decomposition = sum(
            np.diag(feature[:, column])
            @ loading_gram
            @ np.diag(feature[:, column])
            for column in range(rank)
        )
        assert np.max(np.abs(integrated - decomposition)) < 2e-15
        eigenvalues = np.linalg.eigvalsh(integrated)
        assert eigenvalues[0] > 1e-7
        assert np.linalg.matrix_rank(integrated, tol=1e-10) == 5
        minimum_eigenvalues.append(eigenvalues[0])

    zero_row_feature = features[-1].copy()
    zero_row_feature[0] = 0.0
    zero_diagonal_matrix = zero_row_feature @ zero_row_feature.T
    assert zero_diagonal_matrix[0, 0] == 0.0
    singular_integrated = zero_diagonal_matrix * loading_gram
    singular_eigenvalues = np.linalg.eigvalsh(singular_integrated)
    assert singular_eigenvalues[0] == 0.0
    assert np.linalg.matrix_rank(singular_integrated, tol=1e-10) == 4

    print("\nfinite-maturity full-rank criterion")
    print(f"  loading-Gram minimum eigenvalue "
          f"{loading_eigenvalues[0]:.10e}")
    print("  integrated minimum eigenvalues at fixed ranks 1, 2, 3 "
          + " ".join(f"{value:.10e}" for value in minimum_eigenvalues))
    print("  one zero Green-Kubo diagonal gives integrated rank 4 of 5")
    return loading_eigenvalues, np.asarray(minimum_eigenvalues)


def repeated_loading_rank_checks():
    """Check the exact rank formula when CIR loading shapes repeat.

    Partition names by equal mean-reversion rates.  Factoring K = F F' and
    the loading Gramian shows that K Hadamard J is the Gram matrix of the
    rowwise tensors f_i tensor g_{c(i)}.  Distinct CIR loading functions
    g_c are linearly independent, so the tensor spans for different clusters
    form a direct sum.  The integrated rank is therefore the sum of the
    ranks of the within-cluster principal blocks of K.
    """
    cluster_sizes = (3, 2, 1)
    cluster_ids = np.repeat(np.arange(len(cluster_sizes)), cluster_sizes)
    centers = np.array([0.4, 1.1, 2.3])
    kappas = centers[cluster_ids]
    sigma = 0.25
    horizon = 4.0
    gammas = np.sqrt(kappas**2 + 2 * sigma**2)

    def loadings_at(time):
        decay = np.exp(-gammas * time)
        return 2 * (1 - decay) / (
            (gammas + kappas) * (1 - decay) + 2 * gammas * decay
        )

    loading_gram = np.array([
        [quad(
            lambda time, j=j, k=k:
            loadings_at(time)[j] * loadings_at(time)[k],
            0,
            horizon,
            epsabs=1e-13,
            epsrel=1e-13,
            limit=300,
        )[0] for k in range(len(kappas))]
        for j in range(len(kappas))
    ])
    assert np.linalg.matrix_rank(loading_gram, tol=1e-10) == len(centers)

    features = [
        np.array([[1.0], [-0.7], [0.3], [1.2], [-2.0], [0.5]]),
        np.column_stack((
            np.ones(6),
            np.array([-2.0, -0.7, 0.2, 1.1, 2.3, -1.4]),
        )),
        np.column_stack((
            np.ones(6),
            np.array([-2.0, -0.7, 0.2, 1.1, 2.3, -1.4]),
            np.array([0.5, -1.3, 0.8, 2.2, -0.4, 1.7]),
        )),
    ]
    observed_ranks = []
    predicted_ranks = []
    block_rank_rows = []
    for feature in features:
        green_kubo = feature @ feature.T
        block_ranks = []
        start = 0
        for size in cluster_sizes:
            indices = slice(start, start + size)
            block_ranks.append(np.linalg.matrix_rank(
                green_kubo[indices, indices], tol=1e-11
            ))
            start += size
        integrated = green_kubo * loading_gram
        observed = np.linalg.matrix_rank(integrated, tol=1e-10)
        predicted = sum(block_ranks)
        assert observed == predicted
        observed_ranks.append(observed)
        predicted_ranks.append(predicted)
        block_rank_rows.append(tuple(block_ranks))

    assert observed_ranks == [3, 5, 6]
    assert block_rank_rows == [(1, 1, 1), (2, 2, 1), (3, 2, 1)]
    # Positive diagonal entries alone cease to imply full rank when loading
    # shapes collide: the rank-one K has positive diagonal but integrated
    # rank equal only to the number of distinct loading clusters.
    assert np.all(np.diag(features[0] @ features[0].T) > 0)
    assert observed_ranks[0] < len(kappas)

    print("\nfinite-maturity rank with repeated CIR loadings")
    print("  cluster sizes " + " ".join(map(str, cluster_sizes)))
    print("  fixed Green-Kubo ranks 1 2 3")
    print("  within-cluster ranks " + "; ".join(
        "+".join(map(str, row)) for row in block_rank_rows
    ))
    print("  predicted/observed integrated ranks "
          + " ".join(map(str, observed_ranks)))
    return tuple(observed_ranks), tuple(block_rank_rows)


def heterogeneous_two_name_rank_checks():
    """Check the complete two-name criterion with heterogeneous CIR inputs.

    The Riccati loading B_{kappa,sigma} satisfies

        B' = 1 - kappa B - sigma^2 B^2 / 2,  B(0) = 0.

    Hence B'(0)=1, B''(0)=-kappa, and
    B'''(0)=kappa^2-sigma^2.  Two loadings are proportional only if the
    proportionality constant is one, and they are then equal only if both
    (kappa, sigma) pairs coincide.  Their two-by-two Gramian is therefore
    positive definite for every positive horizon exactly when the pairs are
    distinct.

    The certificate deliberately keeps kappa equal and changes only sigma.
    A rank-one positive-diagonal K then acquires rank two after maturity
    integration.  Repeating the full pair collapses the loading Gramian and
    leaves the rank of K unchanged.
    """
    horizon = 4.0
    kappas = np.array([1.1, 1.1])
    sigmas = np.array([0.18, 0.46])

    def loading_gram(volatilities):
        gammas = np.sqrt(kappas**2 + 2 * volatilities**2)

        def loadings_at(time):
            decay = np.exp(-gammas * time)
            return 2 * (1 - decay) / (
                (gammas + kappas) * (1 - decay) + 2 * gammas * decay
            )

        return np.array([
            [quad(
                lambda time, j=j, k=k:
                loadings_at(time)[j] * loadings_at(time)[k],
                0,
                horizon,
                epsabs=1e-13,
                epsrel=1e-13,
                limit=300,
            )[0] for k in range(2)]
            for j in range(2)
        ])

    distinct_gram = loading_gram(sigmas)
    gram_determinant = np.linalg.det(distinct_gram)
    assert gram_determinant > 9e-4
    assert np.linalg.eigvalsh(distinct_gram)[0] > 2e-4

    feature = np.array([1.0, -0.8])
    rank_one_k = np.outer(feature, feature)
    distinct_integrated = rank_one_k * distinct_gram
    distinct_eigenvalues = np.linalg.eigvalsh(distinct_integrated)
    assert distinct_eigenvalues[0] > 1.7e-4
    assert np.linalg.matrix_rank(distinct_integrated, tol=1e-10) == 2

    repeated_gram = loading_gram(np.array([0.18, 0.18]))
    assert np.max(np.abs(repeated_gram - repeated_gram[0, 0])) < 2e-15
    repeated_integrated = rank_one_k * repeated_gram
    assert np.linalg.matrix_rank(repeated_integrated, tol=1e-10) == 1

    full_rank_k = np.array([[1.0, 0.3], [0.3, 0.7]])
    repeated_full_integrated = full_rank_k * repeated_gram
    assert np.linalg.matrix_rank(repeated_full_integrated, tol=1e-10) == 2
    assert np.max(np.abs(
        repeated_full_integrated - repeated_gram[0, 0] * full_rank_k
    )) < 2e-15

    print("\nheterogeneous two-name CIR loading criterion")
    print(f"  equal-kappa, unequal-sigma Gram determinant "
          f"{gram_determinant:.10e}")
    print("  rank-one K integrated eigenvalues "
          + " ".join(f"{value:.10e}" for value in distinct_eigenvalues))
    print("  identical-pair integrated ranks for rank-one/full-rank K: 1/2")
    return gram_determinant, distinct_eigenvalues


def coalescing_loading_checks():
    """Certify the confluent-Vandermonde law for nearly equal loadings.

    High precision is essential here: the smallest eigenvalue of a four-name
    loading Gram matrix is order epsilon^6, while its determinant is order
    epsilon^12.
    """
    mp.mp.dps = 70
    dimension = 4
    horizon = mp.mpf("4")
    kappa0 = mp.mpf("2")
    sigma = mp.mpf("0.02")
    nodes = [mp.mpf(value) for value in ("-1.5", "-0.5", "0.5", "1.5")]

    def loading(t, kappa):
        gamma = mp.sqrt(kappa * kappa + 2 * sigma * sigma)
        growth = mp.expm1(gamma * t)
        return 2 * growth / ((gamma + kappa) * growth + 2 * gamma)

    derivatives = [
        lambda t, order=order: mp.diff(
            lambda kappa: loading(t, kappa), kappa0, order
        )
        for order in range(dimension)
    ]
    derivative_gram = mp.matrix([
        [mp.quad(lambda t, r=r, s=s:
                 derivatives[r](t) * derivatives[s](t), [0, horizon])
         for s in range(dimension)]
        for r in range(dimension)
    ])
    vandermonde = mp.mpf(1)
    for j in range(dimension):
        for i in range(j):
            vandermonde *= nodes[j] - nodes[i]
    coefficient_matrix = mp.matrix([
        [nodes[j] ** order / mp.factorial(order)
         for j in range(dimension)]
        for order in range(dimension)
    ])
    derivative_cholesky = mp.cholesky(derivative_gram)
    coefficient_cholesky = mp.cholesky(
        coefficient_matrix * coefficient_matrix.T)
    eigenvalue_constants = [
        (derivative_cholesky[order, order]
         * coefficient_cholesky[order, order]) ** 2
        for order in range(dimension)
    ]

    determinant_constant = (
        vandermonde**2 * mp.det(derivative_gram)
        / mp.fprod(mp.factorial(r)**2 for r in range(dimension))
    )
    assert determinant_constant > 0
    assert (abs(mp.fprod(eigenvalue_constants) / determinant_constant - 1)
            < mp.mpf("1e-60"))

    epsilons = [mp.mpf(2) ** (-power) for power in range(3, 9)]
    spectra = []
    determinant_ratios = []
    for epsilon in epsilons:
        kappas = [kappa0 + epsilon * node for node in nodes]
        gram = mp.matrix([
            [mp.quad(lambda t, j=j, k=k:
                     loading(t, kappas[j]) * loading(t, kappas[k]),
                     [0, horizon])
             for k in range(dimension)]
            for j in range(dimension)
        ])
        eigenvalues = sorted(mp.eigsy(gram, eigvals_only=True), reverse=True)
        assert eigenvalues[-1] > 0
        spectra.append(eigenvalues)
        determinant_ratios.append(
            mp.det(gram) /
            (epsilon ** (dimension * (dimension - 1))
             * determinant_constant)
        )

    log_eps = np.log(np.array([float(value) for value in epsilons[-4:]]))
    slopes = []
    for index in range(dimension):
        log_eigenvalue = np.log(np.array([
            float(spectrum[index]) for spectrum in spectra[-4:]
        ]))
        slopes.append(float(np.polyfit(log_eps, log_eigenvalue, 1)[0]))
    expected = 2 * np.arange(dimension)
    assert np.max(abs(np.array(slopes) - expected)) < 0.08
    assert abs(float(determinant_ratios[-1]) - 1) < 2e-4

    measured_constants = [
        spectra[-1][order] / epsilons[-1] ** (2 * order)
        for order in range(dimension)
    ]
    constant_relative_errors = [
        abs(measured / predicted - 1)
        for measured, predicted in zip(measured_constants,
                                       eigenvalue_constants)
    ]
    assert max(constant_relative_errors) < mp.mpf("2e-5")

    condition_slopes = []
    for left, right, epsilon_left, epsilon_right in zip(
            spectra[:-1], spectra[1:], epsilons[:-1], epsilons[1:]):
        condition_left = left[0] / left[-1]
        condition_right = right[0] / right[-1]
        condition_slopes.append(
            mp.log(condition_right / condition_left)
            / mp.log(epsilon_right / epsilon_left)
        )
    assert abs(float(condition_slopes[-1]) + 2 * (dimension - 1)) < 0.08

    print("\ncoalescing CIR loading certificate")
    print("  eigenvalue log-log slopes "
          + " ".join(f"{slope:.6f}" for slope in slopes))
    print("  predicted eigenvalue constants "
          + " ".join(mp.nstr(value, 10) for value in eigenvalue_constants))
    print("  maximum eigenvalue-constant relative error "
          + mp.nstr(max(constant_relative_errors), 8))
    print(f"  determinant ratio {float(determinant_ratios[-1]):.12f}")
    print(f"  condition-number slope {float(condition_slopes[-1]):.6f}")
    return (slopes, determinant_ratios, condition_slopes,
            eigenvalue_constants, constant_relative_errors)


def clustered_loading_checks():
    """Certify the partial-collision law for several distinct clusters.

    The cluster sizes (3, 2, 1) predict squared singular-value powers
    (0, 0, 0, 2, 2, 4), determinant power 8, and condition-number power -4.
    The determinant coefficient is the jet Gram determinant times one
    squared, factorial-scaled Vandermonde factor for each cluster.
    """
    mp.mp.dps = 80
    horizon = mp.mpf("4")
    sigma = mp.mpf("0.02")
    centers = [mp.mpf(value) for value in ("0.8", "2.0", "4.0")]
    nodes = [
        [mp.mpf(value) for value in ("-1", "0", "1")],
        [mp.mpf(value) for value in ("-0.75", "0.75")],
        [mp.mpf("0")],
    ]
    dimension = sum(map(len, nodes))

    def loading(t, kappa):
        gamma = mp.sqrt(kappa * kappa + 2 * sigma * sigma)
        growth = mp.expm1(gamma * t)
        return 2 * growth / ((gamma + kappa) * growth + 2 * gamma)

    jets = [
        (center, order)
        for center, cluster_nodes in zip(centers, nodes)
        for order in range(len(cluster_nodes))
    ]
    jet_gram = mp.matrix([
        [mp.quad(
            lambda t, center_j=center_j, order_j=order_j,
            center_k=center_k, order_k=order_k:
            mp.diff(lambda kappa: loading(t, kappa), center_j, order_j)
            * mp.diff(lambda kappa: loading(t, kappa), center_k, order_k),
            [0, horizon],
        ) for center_k, order_k in jets]
        for center_j, order_j in jets
    ])
    jet_determinant = mp.det(jet_gram)
    assert jet_determinant > mp.mpf("1e-19")

    determinant_constant = jet_determinant
    determinant_power = 0
    expected_powers = []
    for cluster_nodes in nodes:
        cluster_size = len(cluster_nodes)
        determinant_power += cluster_size * (cluster_size - 1)
        expected_powers.extend(2 * order for order in range(cluster_size))
        for j in range(cluster_size):
            for i in range(j):
                determinant_constant *= (cluster_nodes[j]
                                         - cluster_nodes[i]) ** 2
        determinant_constant /= mp.fprod(
            mp.factorial(order) ** 2 for order in range(cluster_size)
        )
    expected_powers.sort()

    epsilons = [mp.mpf(10) ** (-power) for power in range(2, 8)]
    spectra = []
    determinant_ratios = []
    for epsilon in epsilons:
        kappas = [
            center + epsilon * node
            for center, cluster_nodes in zip(centers, nodes)
            for node in cluster_nodes
        ]
        gram = mp.matrix([
            [mp.quad(lambda t, kappa_j=kappa_j, kappa_k=kappa_k:
                     loading(t, kappa_j) * loading(t, kappa_k),
                     [0, horizon])
             for kappa_k in kappas]
            for kappa_j in kappas
        ])
        eigenvalues = sorted(mp.eigsy(gram, eigvals_only=True), reverse=True)
        assert eigenvalues[-1] > 0
        spectra.append(eigenvalues)
        determinant_ratios.append(
            mp.det(gram)
            / (epsilon ** determinant_power * determinant_constant)
        )

    log_eps = np.log(np.array([float(value) for value in epsilons]))
    slopes = []
    for index in range(dimension):
        log_eigenvalue = np.log(np.array([
            float(spectrum[index]) for spectrum in spectra
        ]))
        slopes.append(float(np.polyfit(log_eps, log_eigenvalue, 1)[0]))
    determinant_slope = float(np.polyfit(
        log_eps,
        np.log(np.array([float(mp.fprod(spectrum))
                         for spectrum in spectra])),
        1,
    )[0])
    condition_slope = float(np.polyfit(
        log_eps,
        np.log(np.array([float(spectrum[0] / spectrum[-1])
                         for spectrum in spectra])),
        1,
    )[0])

    assert np.max(abs(np.array(slopes) - expected_powers)) < 5e-4
    assert abs(determinant_slope - determinant_power) < 5e-4
    assert abs(condition_slope + max(expected_powers)) < 5e-4
    assert abs(float(determinant_ratios[-1]) - 1) < 1e-11

    print("\nclustered CIR loading certificate")
    print("  cluster sizes " + " ".join(str(len(cluster))
                                          for cluster in nodes))
    print("  eigenvalue log-log slopes "
          + " ".join(f"{slope:.6f}" for slope in slopes))
    print(f"  determinant slope {determinant_slope:.6f}")
    print(f"  determinant ratio {float(determinant_ratios[-1]):.12f}")
    print(f"  condition-number slope {condition_slope:.6f}")
    print("  jet Gram determinant " + mp.nstr(jet_determinant, 12))
    return (slopes, determinant_slope, determinant_ratios,
            condition_slope, jet_determinant)


def cir_jet_independence_checks():
    """Certify the exact confluent determinant behind CIR jet independence.

    If B_kappa(t)=sum_{n>=1} b_n(kappa)t^n, the Riccati recurrence makes
    b_n a degree-(n-1) polynomial with leading coefficient
    (-1)^(n-1)/n!.  Evaluation of these polynomials and their derivatives at
    distinct cluster centers is therefore a nonsingular confluent
    Vandermonde system.  Rational arithmetic checks both the recurrence and
    its closed determinant for the (3,2,1) certificate.
    """

    def polynomial_add(left, right):
        result = [Fraction(0)] * max(len(left), len(right))
        for index, value in enumerate(left):
            result[index] += value
        for index, value in enumerate(right):
            result[index] += value
        return result

    def polynomial_scale(polynomial, scalar):
        return [scalar * value for value in polynomial]

    def polynomial_product(left, right):
        result = [Fraction(0)] * (len(left) + len(right) - 1)
        for left_index, left_value in enumerate(left):
            for right_index, right_value in enumerate(right):
                result[left_index + right_index] += left_value * right_value
        return result

    def derivative_evaluation(polynomial, order, center):
        return sum(
            coefficient
            * Fraction(math.factorial(degree),
                       math.factorial(degree - order))
            * center ** (degree - order)
            for degree, coefficient in enumerate(polynomial)
            if degree >= order
        )

    def exact_determinant(matrix):
        matrix = [row[:] for row in matrix]
        determinant = Fraction(1)
        for column in range(len(matrix)):
            pivot = next(row for row in range(column, len(matrix))
                         if matrix[row][column])
            if pivot != column:
                matrix[column], matrix[pivot] = matrix[pivot], matrix[column]
                determinant *= -1
            pivot_value = matrix[column][column]
            determinant *= pivot_value
            for row in range(column + 1, len(matrix)):
                multiplier = matrix[row][column] / pivot_value
                for entry in range(column, len(matrix)):
                    matrix[row][entry] -= multiplier * matrix[column][entry]
        return determinant

    centers = [Fraction(4, 5), Fraction(2), Fraction(4)]
    multiplicities = [3, 2, 1]
    dimension = sum(multiplicities)
    sigma_squared = Fraction(1, 2500)

    # coefficients[n] stores b_n(kappa) in increasing powers of kappa.
    coefficients = [None, [Fraction(1)]]
    for n in range(1, dimension):
        quadratic = [Fraction(0)]
        for left in range(1, n):
            quadratic = polynomial_add(
                quadratic,
                polynomial_product(coefficients[left],
                                   coefficients[n - left]),
            )
        recurrence = polynomial_add(
            polynomial_scale([Fraction(0)] + coefficients[n], -1),
            polynomial_scale(quadratic, -sigma_squared / 2),
        )
        coefficients.append(polynomial_scale(recurrence, Fraction(1, n + 1)))

    for n in range(1, dimension + 1):
        assert len(coefficients[n]) == n
        assert coefficients[n][-1] == Fraction(
            (-1) ** (n - 1), math.factorial(n))

    jets = [
        (center, order)
        for center, multiplicity in zip(centers, multiplicities)
        for order in range(multiplicity)
    ]
    coefficient_jet_matrix = [
        [derivative_evaluation(coefficients[n], order, center)
         for center, order in jets]
        for n in range(1, dimension + 1)
    ]
    exact = exact_determinant(coefficient_jet_matrix)

    predicted = Fraction(1)
    for n in range(1, dimension + 1):
        predicted *= Fraction((-1) ** (n - 1), math.factorial(n))
    for left, left_center in enumerate(centers):
        for right in range(left + 1, len(centers)):
            predicted *= (centers[right] - left_center) ** (
                multiplicities[left] * multiplicities[right])
    for multiplicity in multiplicities:
        for order in range(multiplicity):
            predicted *= math.factorial(order)

    assert exact == predicted == Fraction(-1536, 48828125)
    print("\nCIR joint-jet independence certificate")
    print("  cluster sizes " + " ".join(map(str, multiplicities)))
    print(f"  exact Taylor-jet determinant {exact}")
    print(f"  decimal determinant {float(exact):.12e}")
    return exact


def main():
    rank_amplification_checks()
    long_maturity_loading_checks()
    finite_rank_long_maturity_checks()
    integrated_full_rank_criterion_checks()
    repeated_loading_rank_checks()
    heterogeneous_two_name_rank_checks()
    coalescing_loading_checks()
    cir_jet_independence_checks()
    clustered_loading_checks()
    assert np.all(2 * KAPPA[:, None] * THETA > SIGMA[:, None] ** 2)
    Bs = loadings(T)
    J = np.array([[quad(lambda t: Bs[j](t) * Bs[k](t), 0, T,
                        epsabs=1e-12)[0] for k in range(2)] for j in range(2)])
    d = (THETA[:, 0] - THETA[:, 1]) / 2
    # Q0(1,-1) = -2(1,-1), so its Green-Kubo coefficient is 1/2.
    K = np.outer(KAPPA * d, KAPPA * d) / 2
    D = K * J

    assert abs(np.linalg.det(K)) < 1e-22
    assert np.linalg.eigvalsh(D)[0] > 1e-6
    assert np.allclose(D / J, K, rtol=1e-13, atol=1e-16)
    print("Feller margin:", np.min(2 * KAPPA[:, None] * THETA - SIGMA[:, None] ** 2))
    print("eigenvalues: instantaneous K", np.linalg.eigvalsh(K))
    print("eigenvalues: integrated D(5)", np.linalg.eigvalsh(D))
    print("m   exact pairwise log ratio   first order   m^2 residual")

    previous = None
    for m in (10, 20, 40, 80):
        pair = (exact_log_survival((0, 1), m, Bs)
                - exact_log_survival((0,), m, Bs)
                - exact_log_survival((1,), m, Bs))
        first = 2 * D[0, 1] / m
        residual = (pair - first) * m ** 2
        print(f"{m:2}  {pair:.11g}  {first:.11g}  {residual:.11g}")
        if previous is not None:
            assert abs(residual - previous) / abs(residual) < 0.13
        previous = residual

    # A nonstationary prior produces a first-order endpoint corrector.  Because
    # that corrector is additive over names, it cancels from the pairwise log
    # ratio.  The following is an independent ODE check of both statements.
    print("\nnonstationary prior: max survival error and pair residual")
    errors = []
    pair_errors = []
    prior = np.array([0.9, 0.1])
    for m in (10, 20, 40, 80):
        errs = [abs(exact_log_survival(names, m, Bs, prior)
                    - first_log_survival(names, m, Bs, D, prior))
                for names in ((0,), (1,), (0, 1))]
        pair = (exact_log_survival((0, 1), m, Bs, prior)
                - exact_log_survival((0,), m, Bs, prior)
                - exact_log_survival((1,), m, Bs, prior))
        pair_error = abs(pair - 2 * D[0, 1] / m)
        errors.append(max(errs))
        pair_errors.append(pair_error)
        print(f"{m:2}  {max(errs):.11g}  {pair_error * m ** 2:.11g}")
    survival_rate = math.log(errors[-2] / errors[-1], 2)
    pair_rate = math.log(pair_errors[-2] / pair_errors[-1], 2)
    assert 1.8 < survival_rate < 2.2
    assert 1.8 < pair_rate < 2.2

    # Correlation is O(1/m); changing the prior affects it only at O(1/m^2).
    correlation_differences = []
    for m in (10, 20, 40, 80):
        difference = abs(default_correlation(m, Bs, prior)
                         - default_correlation(m, Bs, PI))
        correlation_differences.append(difference)
    correlation_rate = math.log(
        correlation_differences[-2] / correlation_differences[-1], 2)
    assert 1.8 < correlation_rate < 2.2
    print(f"rates: survival {survival_rate:.6f}, pair {pair_rate:.6f}, "
          f"prior effect on correlation {correlation_rate:.6f}")
    correlations = page_prior_benchmark()
    print("credit-page correlations q=0.1,0.5,0.9:",
          " ".join(f"{x:.6f}" for x in correlations))

    # Two default channels suffice to expose cycle direction.  Unordered
    # survival uses a diagonal killing rate and is invariant under time
    # reversal, but the absorbing channel label retains an oriented K entry.
    print("\ntwo-name first-default race on a directed three-state cycle")
    q_cycle = np.array([[-2.1, 2.0, 0.1],
                        [0.1, -2.1, 2.0],
                        [2.0, 0.1, -2.1]])
    hazard1 = np.array([0.07, 0.01, 0.01])
    hazard2 = np.array([0.01, 0.07, 0.01])
    leading, coefficient, race_k = first_default_correction(
        q_cycle, hazard1, hazard2)
    reverse_leading, reverse_coefficient, reverse_k = first_default_correction(
        q_cycle.T, hazard1, hazard2)
    assert abs(leading - 0.5) < 1e-14
    assert abs(reverse_leading - leading) < 1e-14
    assert np.max(abs(reverse_k - race_k.T)) < 2e-15
    mean_total = (hazard1 + hazard2).mean()
    gap_coefficient = (race_k[0, 1] - race_k[1, 0]) / mean_total
    assert abs((coefficient - reverse_coefficient) - gap_coefficient) < 2e-15

    race_errors = []
    gap_errors = []
    for m in (5, 10, 20, 40):
        clockwise = first_default_probability(m * q_cycle, hazard1, hazard2)
        counterclockwise = first_default_probability(
            m * q_cycle.T, hazard1, hazard2)
        approximation = leading + coefficient / m
        race_errors.append(abs(clockwise - approximation))
        gap_errors.append(abs(
            (clockwise - counterclockwise) - gap_coefficient / m))
        print(
            f"{m:2d}  p1 clockwise {clockwise:.10f}  reverse {counterclockwise:.10f}"
            f"  gap {clockwise-counterclockwise:+.8f}"
        )
    race_rate = math.log(race_errors[-2] / race_errors[-1], 2)
    gap_rate = math.log(gap_errors[-2] / gap_errors[-1], 2)
    assert 1.8 < race_rate < 2.2
    assert 1.8 < gap_rate < 2.2

    # Stationary Feynman--Kac survival cannot distinguish the two directions.
    pi3 = np.ones(3) / 3
    for hazard in (hazard1, hazard2, hazard1 + hazard2):
        clockwise_survival = pi3 @ expm(
            3.0 * (10 * q_cycle - np.diag(hazard))) @ np.ones(3)
        reverse_survival = pi3 @ expm(
            3.0 * (10 * q_cycle.T - np.diag(hazard))) @ np.ones(3)
        assert abs(clockwise_survival - reverse_survival) < 2e-14
    print(
        f"K12={race_k[0, 1]:+.8e}, K21={race_k[1, 0]:+.8e}; "
        f"gap coefficient {gap_coefficient:.10f}; rates {race_rate:.3f}, {gap_rate:.3f}"
    )

    # With three or more causes an unweighted race need not see the full
    # antisymmetric matrix.  Cyclically permuted hazards give A 1 = 0 even
    # though A is nonzero.  Positive exposure weights turn the observed
    # first-order residual into diag(w) A w / bar(h(w)); varying w recovers A.
    print("\nweighted three-cause races recover the full oriented component")
    hazards3 = np.array([
        [0.07, 0.01, 0.01],
        [0.01, 0.07, 0.01],
        [0.01, 0.01, 0.07],
    ])
    leading3, coefficient3, k3 = first_event_correction(q_cycle, hazards3)
    anti3 = 0.5 * (k3 - k3.T)
    assert np.linalg.norm(anti3) > 1e-5
    assert np.linalg.norm(anti3 @ np.ones(3)) < 2e-18
    assert np.max(abs(coefficient3)) < 2e-16
    equal_forward = first_event_probabilities(10 * q_cycle, hazards3)
    equal_reverse = first_event_probabilities(10 * q_cycle.T, hazards3)
    assert np.max(abs(equal_forward - equal_reverse)) < 2e-15
    assert np.max(abs(equal_forward - leading3)) < 2e-15

    weights = np.array([1.0, 2.0, 4.0])
    weighted_leading, weighted_coefficient, _ = first_event_correction(
        q_cycle, hazards3, weights
    )
    _, reverse_weighted_coefficient, _ = first_event_correction(
        q_cycle.T, hazards3, weights
    )
    pi3 = stationary(q_cycle)
    weighted_mean = (weights[:, None] * hazards3) @ pi3
    mean_total3 = weighted_mean.sum()
    predicted_gap = 2 * weights * (anti3 @ weights) / mean_total3
    assert np.max(
        abs(weighted_coefficient - reverse_weighted_coefficient - predicted_gap)
    ) < 3e-15

    weighted_errors = []
    weighted_gap_errors = []
    for m in (5, 10, 20, 40):
        forward = first_event_probabilities(m * q_cycle, hazards3, weights)
        reverse = first_event_probabilities(m * q_cycle.T, hazards3, weights)
        weighted_errors.append(
            np.max(abs(forward - weighted_leading - weighted_coefficient / m))
        )
        weighted_gap_errors.append(
            np.max(abs(forward - reverse - predicted_gap / m))
        )
    weighted_rate = math.log(weighted_errors[-2] / weighted_errors[-1], 2)
    weighted_gap_rate = math.log(
        weighted_gap_errors[-2] / weighted_gap_errors[-1], 2
    )
    assert 1.8 < weighted_rate < 2.2
    assert 1.8 < weighted_gap_rate < 2.2

    # Unordered survival identifies sym(K), and each weighted race supplies
    # A w.  For skew A, d-1 independent designs are necessary and sufficient:
    # the kernel consists of skew maps supported on the designs' orthogonal
    # complement and has dimension choose(d-q, 2).
    rng = np.random.default_rng(20260926)
    maximum_spectrum_error = 0.0
    for d in range(3, 8):
        for q in range(1, d):
            rank_q = 0
            while rank_q < q:
                generic_designs = [rng.uniform(0.5, 2.0, d) for _ in range(q)]
                rank_q = np.linalg.matrix_rank(np.column_stack(generic_designs))
            measurement, _ = skew_measurement_matrix(generic_designs)
            expected_rank = math.comb(d, 2) - math.comb(d - q, 2)
            assert np.linalg.matrix_rank(measurement) == expected_rank
            observed = np.linalg.svd(measurement, compute_uv=False)
            observed = observed[observed > 1e-11]
            predicted = predicted_skew_singular_values(generic_designs)
            predicted = predicted[predicted > 1e-11]
            assert len(observed) == len(predicted) == expected_rank
            maximum_spectrum_error = max(
                maximum_spectrum_error, np.max(abs(observed - predicted))
            )

    # The explicit positive family w_r=1+epsilon*e_r has smallest singular
    # value epsilon and a closed-form reconstruction. Its condition number
    # quantifies the instability as the designs collapse toward one vector.
    maximum_condition_error = 0.0
    maximum_recovery_error = 0.0
    maximum_limit_error = 0.0
    maximum_budget_error = 0.0
    example_condition = None
    for d in range(3, 9):
        random_matrix = rng.normal(size=(d, d))
        random_skew = random_matrix - random_matrix.T
        for epsilon in (0.2, 0.5, 1.0):
            explicit_designs = near_simplex_designs(d, epsilon)
            measurement, _ = skew_measurement_matrix(explicit_designs)
            singular_values = np.linalg.svd(measurement, compute_uv=False)
            condition = singular_values[0] / singular_values[-1]
            predicted_condition = math.sqrt(
                2 * epsilon**2 + (d - 1) * (d + 2 * epsilon)
            ) / epsilon
            maximum_condition_error = max(
                maximum_condition_error, abs(condition - predicted_condition)
            )
            responses = [random_skew @ w for w in explicit_designs]
            reconstructed = recover_skew_near_simplex(responses, epsilon)
            maximum_recovery_error = max(
                maximum_recovery_error,
                np.max(abs(reconstructed - random_skew)),
            )
            if d == 6 and epsilon == 0.5:
                example_condition = condition

        # For q=d-1, kappa >= sqrt(2). Equality requires equal nonzero
        # frame eigenvalues, hence equal-norm orthogonal design columns. No
        # strictly positive columns can attain it, but the explicit family
        # approaches it as epsilon grows. Under ||W||_F^2=1, the matching
        # minimax bound is sigma_min <= 1/sqrt(d-1).
        previous_condition = math.inf
        previous_budget_sigma = 0.0
        for epsilon in (1.0, 10.0, 100.0, 1000.0):
            explicit_designs = near_simplex_designs(d, epsilon)
            measurement, _ = skew_measurement_matrix(explicit_designs)
            singular_values = np.linalg.svd(measurement, compute_uv=False)
            condition = singular_values[0] / singular_values[-1]
            assert condition > math.sqrt(2)
            assert condition < previous_condition
            previous_condition = condition
            predicted_square = (
                2
                + 2 * (d - 1) / epsilon
                + d * (d - 1) / epsilon**2
            )
            maximum_limit_error = max(
                maximum_limit_error, abs(condition**2 - predicted_square)
            )

            design_matrix = np.column_stack(explicit_designs)
            normalized_designs = [
                w / np.linalg.norm(design_matrix) for w in explicit_designs
            ]
            normalized_measurement, _ = skew_measurement_matrix(
                normalized_designs
            )
            budget_sigma = np.linalg.svd(
                normalized_measurement, compute_uv=False
            )[-1]
            assert budget_sigma < 1 / math.sqrt(d - 1)
            assert budget_sigma > previous_budget_sigma
            previous_budget_sigma = budget_sigma
            predicted_budget_sigma = epsilon / math.sqrt(
                (d - 1) * (epsilon**2 + 2 * epsilon + d)
            )
            maximum_budget_error = max(
                maximum_budget_error,
                abs(budget_sigma - predicted_budget_sigma),
            )

        # The nonnegative boundary design e_1,...,e_{d-1} attains both
        # bounds exactly. It is excluded only by strict positivity.
        boundary_designs = [np.eye(d)[r] for r in range(d - 1)]
        boundary_measurement, _ = skew_measurement_matrix(boundary_designs)
        boundary_singular_values = np.linalg.svd(
            boundary_measurement, compute_uv=False
        )
        assert abs(
            boundary_singular_values[0] / boundary_singular_values[-1]
            - math.sqrt(2)
        ) < 2e-15
        assert abs(boundary_singular_values[-1] - 1.0) < 2e-15
    assert maximum_spectrum_error < 2e-13
    assert maximum_condition_error < 2e-12
    assert maximum_recovery_error < 2e-13
    assert maximum_limit_error < 2e-12
    assert maximum_budget_error < 2e-13

    # In dimension three, two positive independent designs recover A.  The
    # single equal-weight design is already the sharp d-2 counterexample:
    # anti3 is nonzero but anti3 @ 1 = 0.
    designs = [
        np.array([1.0, 1.0, 1.0]),
        np.array([2.0, 1.0, 1.0]),
    ]
    symmetric3 = 0.5 * (k3 + k3.T)
    responses = []
    for design in designs:
        leading_d, coefficient_d, _ = first_event_correction(
            q_cycle, hazards3, design
        )
        means_d = design * (hazards3 @ pi3)
        total_d = means_d.sum()
        symmetric_d = np.outer(design, design) * symmetric3
        symmetric_coefficient = (
            leading_d * symmetric_d.sum() - symmetric_d.sum(axis=0)
        ) / total_d
        responses.append(total_d * (coefficient_d - symmetric_coefficient) / design)
    recovered_anti = recover_skew(designs, responses)
    recovery_error = np.max(abs(recovered_anti - anti3))
    assert recovery_error < 3e-18
    print(
        f"   ||A||_F={np.linalg.norm(anti3):.8e}; equal-weight gap "
        f"{np.max(abs(equal_forward-equal_reverse)):.2e}"
    )
    print(
        "   weighted gap coefficient "
        + " ".join(f"{value:+.10f}" for value in predicted_gap)
    )
    print(
        f"   weighted probability/gap rates {weighted_rate:.3f}/"
        f"{weighted_gap_rate:.3f}; two-design recovery error "
        f"{recovery_error:.2e}"
    )
    print(
        f"   frame-spectrum error {maximum_spectrum_error:.2e}; "
        f"near-simplex recovery error {maximum_recovery_error:.2e}; "
        f"d=6, epsilon=0.5 condition {example_condition:.6f}"
    )
    print(
        f"   optimal-design errors: condition {maximum_limit_error:.2e}; "
        f"fixed-budget {maximum_budget_error:.2e}; infimum sqrt(2)"
    )

    print("PASS: positivity, sharp rank amplification, distinct, repeated, "
          "and heterogeneous two-name loading rank criteria, automatic CIR "
          "jet independence, prior memory, pair cancellation, and ordered "
          "default")


if __name__ == "__main__":
    main()
