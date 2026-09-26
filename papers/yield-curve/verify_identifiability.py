"""Independent checks for the finite-rate and known-start identifiability note.

The proof is in tools/pages/identifiability.html. This certificate compares the
first-order expressions to an ODE solve; it does not substitute for that proof.
"""
import math

import numpy as np
from scipy.integrate import quad_vec, solve_ivp
from scipy.linalg import expm

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
    for epsilon in (0.2, 0.1, 0.05, 0.025, 0.0125):
        small_response = transient_response_matrix(
            Q, centered, epsilon * scales)
        determinant = np.linalg.det(
            np.column_stack([np.ones(4), small_response]))
        ratio = determinant / (epsilon ** exponent * leading)
        determinant_ratios.append(ratio)
        determinant_errors.append(abs(ratio - 1.0))
    determinant_rate = math.log(
        determinant_errors[-2] / determinant_errors[-1], 2)
    assert determinant_errors[-1] < 0.075
    assert determinant_rate > 0.9

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
    print("repeated-eigenvalue scalar observability rank: 1")


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
        "exact integrated-loading rank, Krylov and initial-mixture observability, shape identities, "
        "known-start expansion, and explicit bounds"
    )


if __name__ == "__main__":
    main()
