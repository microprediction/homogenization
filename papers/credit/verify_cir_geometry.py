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
    for d, smallest, condition in rows:
        print(f"  d={d}: smallest eigenvalue {smallest:.9e}, "
              f"condition {condition:.6e}")
    return rows


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


def main():
    rank_amplification_checks()
    coalescing_loading_checks()
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

    print("PASS: positivity, sharp rank amplification, prior memory, pair "
          "cancellation, and ordered default")


if __name__ == "__main__":
    main()
