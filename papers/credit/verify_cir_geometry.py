"""Certificate for positive CIR intensities and the maturity-loading rank trap.

The two-state chain has an instantaneous rank-one Green-Kubo matrix, but two
different CIR Riccati loadings give a rank-two integrated correction matrix.
The direct pricing ODE independently checks the pairwise first-order formula
and the endpoint-memory correction for an arbitrary initial regime prior.  A
separate constant-hazard example verifies that two competing default channels
already identify the antisymmetric Green--Kubo component.  The multi-cause
check proves that d-1 independent exposure designs are necessary and
sufficient to identify a d-by-d antisymmetric component.
"""
import math

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


def main():
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
    for d in range(3, 8):
        for q in range(1, d):
            rank_q = 0
            while rank_q < q:
                generic_designs = [rng.uniform(0.5, 2.0, d) for _ in range(q)]
                rank_q = np.linalg.matrix_rank(np.column_stack(generic_designs))
            measurement, _ = skew_measurement_matrix(generic_designs)
            expected_rank = math.comb(d, 2) - math.comb(d - q, 2)
            assert np.linalg.matrix_rank(measurement) == expected_rank

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

    print("PASS: positivity, rank, prior memory, pair cancellation, and ordered default")


if __name__ == "__main__":
    main()
