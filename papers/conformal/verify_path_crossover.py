"""Certificate for conditional coverage of a path-dependent Markov score.

The score is S = mu * A_c + sigma * Z, where A_c is the average of a
symmetric two-state chain over c switching times.  Exact CDFs are computed
from a Poisson--Beta occupation-time mixture.  Direct path simulation and the
independent Feynman--Kac characteristic function check that representation.
The certificate also checks the exact covariance of overlapping occupation
windows, its fast-switching effective-sample limit, and the exact finite-sample
coverage of a sliding-window order-statistic threshold.  It additionally
verifies when stride thinning restores iid rank coverage and an explicit
finite-sample coupling bound for residual Markov dependence.  A separate
nonreversible-chain calculation checks the corresponding absolute-regularity
bound beyond the symmetric binary example, including discrete scores with
deterministic or randomized tie handling.  It also checks the exact beta law
of iid coverage conditional on the realized calibration sample and the sharp
one-sided tolerance-limit/PAC choice of calibration order statistic.  Finally,
it checks a total-variation transfer from that iid beta law to a conservative
training-conditional tail bound under absolute regularity and the slack-free
rearrangement bound available when the actual panel order-statistic law is
known.
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ndtr, roots_hermite, roots_jacobi, roots_legendre
from scipy.stats import beta as beta_distribution
from scipy.stats import norm


TARGET = 0.90
MU = 1.0
SIGMA = 0.5


def window_covariance(
    lag: float, horizon: float, switching_rate: float
) -> float:
    """Covariance of two stationary occupation averages separated by ``lag``.

    The chain takes values +/-1 and flips at ``switching_rate`` in either
    direction, so Cov(Y_s,Y_t)=exp(-2*switching_rate*abs(t-s)).
    """
    if lag < 0 or horizon <= 0 or switching_rate <= 0:
        raise ValueError("lag must be nonnegative; horizon and rate positive")
    decay = 2 * switching_rate
    if lag <= horizon:
        integral = 2 * (horizon - lag) / decay + (
            math.exp(-decay * (horizon - lag))
            + math.exp(-decay * (horizon + lag))
            - 2 * math.exp(-decay * lag)
        ) / decay**2
    else:
        integral = (
            math.exp(-decay * (lag - horizon))
            * (1 - math.exp(-decay * horizon)) ** 2
            / decay**2
        )
    return integral / horizon**2


def window_covariance_quadrature(
    lag: float, horizon: float, switching_rate: float
) -> float:
    """Independent adaptive quadrature check of ``window_covariance``."""
    integrand = lambda difference: (
        horizon - abs(difference)
    ) * math.exp(-2 * switching_rate * abs(lag - difference))
    points = [point for point in (0.0, lag) if -horizon < point < horizon]
    integral = quad(
        integrand, -horizon, horizon, points=points, epsabs=2e-13, epsrel=2e-13
    )[0]
    return integral / horizon**2


def linear_window_effective_size(
    count: int, spacing: float, horizon: float, switching_rate: float
) -> tuple[float, float]:
    """Finite-sample variance inflation and ESS for the mean of window averages."""
    variance = window_covariance(0.0, horizon, switching_rate)
    inflation = 1.0 + 2 * sum(
        (1 - lag / count)
        * window_covariance(lag * spacing, horizon, switching_rate)
        / variance
        for lag in range(1, count)
    )
    return inflation, count / inflation


def binomial_mass(count: int, successes: int, probability: float) -> float:
    """Binomial probability, extended by zero outside its natural range."""
    if successes < 0 or successes > count:
        return 0.0
    return (
        math.comb(count, successes)
        * probability**successes
        * (1.0 - probability) ** (count - successes)
    )


def iid_training_confidence(
    calibration_count: int, order: int, target: float
) -> float:
    """Probability that iid calibration-conditional coverage exceeds target."""
    return sum(
        binomial_mass(calibration_count, count, target)
        for count in range(order)
    )


def pac_calibration_order(
    calibration_count: int, target: float, failure_probability: float
) -> int | None:
    """Smallest finite order statistic meeting a training-conditional target."""
    for order in range(1, calibration_count + 1):
        if iid_training_confidence(calibration_count, order, target) >= (
            1.0 - failure_probability
        ):
            return order
    return None


def integer_beta_cdf(value: float, alpha: int, beta: int) -> float:
    """CDF of Beta(alpha, beta) for positive integer parameters."""
    if value <= 0.0:
        return 0.0
    if value >= 1.0:
        return 1.0
    degree = alpha + beta - 1
    return sum(
        binomial_mass(degree, count, value)
        for count in range(alpha, degree + 1)
    )


def training_conditional_transfer_bound(
    calibration_count: int,
    order: int,
    target: float,
    panel_total_variation: float,
    test_decoupling_total_variation: float,
    slack: float,
) -> float:
    """Separated-TV bound for failure of training-conditional coverage.

    ``panel_total_variation`` compares the calibration-panel law with its iid
    product.  ``test_decoupling_total_variation`` compares the joint law with
    the product of the *actual* panel law and an independent test score.  The
    latter has the same panel marginal, so its conditional-coverage loss costs
    one, rather than two, total-variation terms.
    """
    iid_tail = integer_beta_cdf(
        target + slack, order, calibration_count + 1 - order
    )
    return min(
        1.0,
        iid_tail
        + panel_total_variation
        + test_decoupling_total_variation / slack,
    )


def exact_binary_training_failure(
    calibration_count: int,
    order: int,
    target: float,
    correlation: float,
) -> float:
    """Exact training-conditional failure for a randomized binary score.

    The stationary state is 0 or 1 and has lag-one correlation
    ``correlation``.  Given state y and an independent U(0,1), the score is
    (y + U)/2, whose stationary marginal is U(0,1).  Conditional on the
    calibration scores, the next-state transition and the kth order statistic
    give an elementary beta integral.  Enumerating only the calibration state
    path therefore evaluates the training-conditional failure probability
    exactly, without simulating the uniforms.
    """
    import itertools

    if not (1 <= order <= calibration_count):
        raise ValueError("order must be between one and calibration_count")
    if not (-1.0 < correlation < 1.0):
        raise ValueError("correlation must lie strictly between -1 and 1")

    stay = (1.0 + correlation) / 2.0
    failure = 0.0
    for states in itertools.product((0, 1), repeat=calibration_count):
        path_probability = 0.5
        for previous, current in zip(states[:-1], states[1:]):
            path_probability *= stay if previous == current else 1.0 - stay

        zero_count = states.count(0)
        probability_zero = (
            stay if states[-1] == 0 else 1.0 - stay
        )
        probability_one = 1.0 - probability_zero
        if zero_count >= order:
            # The threshold is U_(order)/2 among the zero-state uniforms and
            # conditional coverage is probability_zero * U_(order).
            alpha = order
            beta = zero_count + 1 - order
            cutoff = target / probability_zero
        else:
            # Every zero-state score lies below the threshold.  The remaining
            # order statistic is among the one-state uniforms.
            alpha = order - zero_count
            one_count = calibration_count - zero_count
            beta = one_count + 1 - alpha
            cutoff = (target - probability_zero) / probability_one
        failure += path_probability * integer_beta_cdf(cutoff, alpha, beta)
    return failure


@lru_cache(maxsize=None)
def binary_panel_order_components(
    calibration_count: int,
    order: int,
    correlation: float,
) -> list[tuple[float, float, float, int, int]]:
    """Mixture law of F(Q_k) for the randomized binary score.

    Each tuple is ``(weight, shift, scale, alpha, beta)`` and represents
    ``shift + scale * Beta(alpha, beta)``.  The stationary marginal CDF is
    F(s)=s because S=(Y+U)/2 is uniform on (0,1).
    """
    import itertools

    stay = (1.0 + correlation) / 2.0
    components: dict[tuple[float, float, int, int], float] = {}
    for states in itertools.product((0, 1), repeat=calibration_count):
        path_probability = 0.5
        for previous, current in zip(states[:-1], states[1:]):
            path_probability *= stay if previous == current else 1.0 - stay
        zero_count = states.count(0)
        if zero_count >= order:
            key = (0.0, 0.5, order, zero_count + 1 - order)
        else:
            one_count = calibration_count - zero_count
            alpha = order - zero_count
            key = (0.5, 0.5, alpha, one_count + 1 - alpha)
        components[key] = components.get(key, 0.0) + path_probability
    return [
        (weight, shift, scale, alpha, beta)
        for (shift, scale, alpha, beta), weight in components.items()
    ]


def binary_panel_order_cdf(
    value: float,
    calibration_count: int,
    order: int,
    correlation: float,
) -> float:
    """Exact CDF of F(Q_k) under the dependent calibration-panel law."""
    answer = 0.0
    for weight, shift, scale, alpha, beta in binary_panel_order_components(
        calibration_count, order, correlation
    ):
        standardized = (value - shift) / scale
        answer += weight * beta_distribution.cdf(
            standardized, alpha, beta
        )
    return float(answer)


def binary_panel_order_lower_cost(
    target: float,
    upper: float,
    calibration_count: int,
    order: int,
    correlation: float,
) -> float:
    """Integral from target to upper of (u-target) against F(Q_k)."""
    if upper <= target:
        return 0.0
    answer = 0.0
    for weight, shift, scale, alpha, beta in binary_panel_order_components(
        calibration_count, order, correlation
    ):
        left = max(target, shift)
        right = min(upper, shift + scale)
        if right <= left:
            continue
        answer += weight * quad(
            lambda value: (value - target)
            * beta_distribution.pdf(
                (value - shift) / scale, alpha, beta
            )
            / scale,
            left,
            right,
            epsabs=2e-14,
            epsrel=2e-13,
        )[0]
    return float(answer)


def binary_panel_rearrangement_bound(
    calibration_count: int,
    order: int,
    target: float,
    correlation: float,
    test_decoupling_total_variation: float,
) -> tuple[float, float]:
    """Slack-free conditional-failure bound from the actual panel law."""
    maximum_cost = binary_panel_order_lower_cost(
        target, 1.0, calibration_count, order, correlation
    )
    if test_decoupling_total_variation >= maximum_cost:
        return 1.0, 1.0
    cutoff = brentq(
        lambda upper: binary_panel_order_lower_cost(
            target, upper, calibration_count, order, correlation
        )
        - test_decoupling_total_variation,
        target,
        1.0,
        xtol=2e-14,
    )
    return (
        binary_panel_order_cdf(
            cutoff, calibration_count, order, correlation
        ),
        cutoff,
    )


def exact_split_coverage(
    calibration_count: int,
    order: int,
    correlation: float,
    signal: float = 1.0,
    noise: float = 0.5,
    quadrature_order: int = 80,
) -> float:
    """Exact marginal coverage of a Markov-dependent split threshold.

    The latent states are a stationary symmetric +/-1 chain with adjacent
    correlation ``correlation``.  Calibration and test scores have Gaussian
    emissions ``signal * state + noise * Z``.  The threshold is calibration
    order statistic ``order``; the test point immediately follows the sample.
    """
    if not (1 <= order <= calibration_count):
        raise ValueError("order must be between one and calibration_count")
    if not (-1.0 < correlation < 1.0):
        raise ValueError("correlation must lie strictly between -1 and 1")

    transition = np.array([
        [(1.0 + correlation) / 2.0, (1.0 - correlation) / 2.0],
        [(1.0 - correlation) / 2.0, (1.0 + correlation) / 2.0],
    ])
    states = np.array([-1.0, 1.0])
    nodes, weights = roots_hermite(quadrature_order)
    coverage = 0.0

    for test_index, test_state in enumerate(states):
        for node, weight in zip(nodes, weights):
            test_score = signal * test_state + noise * math.sqrt(2.0) * node
            emission_cdfs = ndtr((test_score - signal * states) / noise)

            # Joint law of the number of calibration scores below the fixed
            # test score and the current latent state.  Emission comes before
            # the transition, leaving the state at the test time after N steps.
            distribution = np.zeros((calibration_count + 1, 2))
            distribution[0] = 0.5
            for _ in range(calibration_count):
                emitted = distribution * (1.0 - emission_cdfs)[None, :]
                emitted[1:] += distribution[:-1] * emission_cdfs[None, :]
                distribution = emitted @ transition

            coverage += (
                weight / math.sqrt(math.pi)
                * distribution[:order, test_index].sum()
            )
    return float(coverage)


def stationary_distribution(transition: np.ndarray) -> np.ndarray:
    """Stationary row distribution of an irreducible finite transition matrix."""
    size = transition.shape[0]
    system = np.vstack((transition.T - np.eye(size), np.ones(size)))
    target = np.append(np.zeros(size), 1.0)
    stationary = np.linalg.lstsq(system, target, rcond=None)[0]
    assert np.max(np.abs(stationary @ transition - stationary)) < 2e-14
    return stationary


def markov_beta(transition: np.ndarray, stationary: np.ndarray, lag: int) -> float:
    """Absolute-regularity coefficient of a stationary finite Markov chain.

    Total variation uses the half-L1 convention.  Because the future path
    includes the state at ``lag``, the data-processing upper bound is exact.
    """
    propagated = np.linalg.matrix_power(transition, lag)
    return float(
        np.sum(
            stationary
            * 0.5
            * np.abs(propagated - stationary[None, :]).sum(axis=1)
        )
    )


def finite_markov_rank_coverage(
    transition: np.ndarray,
    stationary: np.ndarray,
    means: np.ndarray,
    noise: float,
    calibration_count: int,
    order: int,
    stride: int,
    quadrature_order: int = 100,
) -> float:
    """Exact marginal split-rank coverage for a general finite Markov chain."""
    sampled = np.linalg.matrix_power(transition, stride)
    nodes, weights = roots_hermite(quadrature_order)
    state_count = len(stationary)
    coverage = 0.0
    for test_index in range(state_count):
        for node, weight in zip(nodes, weights):
            test_score = means[test_index] + noise * math.sqrt(2.0) * node
            emission_cdfs = ndtr((test_score - means) / noise)
            distribution = np.zeros((calibration_count + 1, state_count))
            distribution[0] = stationary
            for _ in range(calibration_count):
                emitted = distribution * (1.0 - emission_cdfs)[None, :]
                emitted[1:] += distribution[:-1] * emission_cdfs[None, :]
                distribution = emitted @ sampled
            coverage += (
                weight / math.sqrt(math.pi)
                * distribution[:order, test_index].sum()
            )
    return float(coverage)


def finite_markov_discrete_coverage(
    transition: np.ndarray,
    stationary: np.ndarray,
    scores: np.ndarray,
    calibration_count: int,
    order: int,
    stride: int,
    randomized_ties: bool,
) -> float:
    """Exact rank coverage for deterministic finite-state scores.

    With randomized ties, each score is augmented by an independent uniform
    variable and pairs are ordered lexicographically.  Gauss--Legendre
    quadrature is exact here because the integrand is a polynomial of degree
    at most ``calibration_count`` in the test-point uniform.
    """
    sampled = np.linalg.matrix_power(transition, stride)
    state_count = len(stationary)
    if randomized_ties:
        nodes, weights = roots_legendre(calibration_count + 1)
        uniforms = (nodes + 1.0) / 2.0
        weights = weights / 2.0
    else:
        uniforms = np.array([0.0])
        weights = np.array([1.0])

    coverage = 0.0
    for test_index in range(state_count):
        for uniform, weight in zip(uniforms, weights):
            less = (scores < scores[test_index]).astype(float)
            if randomized_ties:
                less += uniform * (scores == scores[test_index])
            distribution = np.zeros((calibration_count + 1, state_count))
            distribution[0] = stationary
            for _ in range(calibration_count):
                emitted = distribution * (1.0 - less)[None, :]
                emitted[1:] += distribution[:-1] * less[None, :]
                distribution = emitted @ sampled
            coverage += weight * distribution[:order, test_index].sum()
    return float(coverage)


def deterministic_iid_discrete_coverage(
    probabilities: np.ndarray,
    scores: np.ndarray,
    calibration_count: int,
    order: int,
) -> float:
    """IID coverage of ``test <= kth calibration score`` with atoms."""
    coverage = 0.0
    for probability, score in zip(probabilities, scores):
        strictly_below = float(probabilities[scores < score].sum())
        coverage += probability * sum(
            math.comb(calibration_count, count)
            * strictly_below**count
            * (1.0 - strictly_below) ** (calibration_count - count)
            for count in range(order)
        )
    return float(coverage)


def sampled_path_total_variation(
    transition: np.ndarray,
    stationary: np.ndarray,
    stride: int,
    sample_count: int,
) -> float:
    """Brute-force TV from a sampled Markov path to iid stationary draws."""
    import itertools

    sampled = np.linalg.matrix_power(transition, stride)
    state_count = len(stationary)
    distance = 0.0
    for path in itertools.product(range(state_count), repeat=sample_count):
        joint = stationary[path[0]]
        independent = stationary[path[0]]
        for previous, current in zip(path[:-1], path[1:]):
            joint *= sampled[previous, current]
            independent *= stationary[current]
        distance += abs(joint - independent)
    return 0.5 * distance


def sampled_test_decoupling_total_variation(
    transition: np.ndarray,
    stationary: np.ndarray,
    stride: int,
    calibration_count: int,
) -> float:
    """TV between a sampled path and actual-panel x stationary-test law."""
    import itertools

    sampled = np.linalg.matrix_power(transition, stride)
    state_count = len(stationary)
    distance = 0.0
    for path in itertools.product(
        range(state_count), repeat=calibration_count + 1
    ):
        panel_probability = stationary[path[0]]
        for previous, current in zip(
            path[: calibration_count - 1], path[1:calibration_count]
        ):
            panel_probability *= sampled[previous, current]
        joint = panel_probability * sampled[path[-2], path[-1]]
        decoupled = panel_probability * stationary[path[-1]]
        distance += abs(joint - decoupled)
    return 0.5 * distance


def split_coverage_first_order(
    calibration_count: int,
    order: int,
    signal: float = 1.0,
    noise: float = 0.5,
) -> tuple[float, float, float]:
    """First derivative at zero adjacent-state correlation.

    Returns the calibration--calibration contribution, the final
    calibration--test contribution, and their combined coefficient.
    """
    def emission_cdf(score: float, state: int) -> float:
        return norm.cdf((score - signal * state) / noise)

    def emission_density(score: float, state: int) -> float:
        return norm.pdf((score - signal * state) / noise) / noise

    def mixture_cdf(score: float) -> float:
        return 0.5 * (emission_cdf(score, 1) + emission_cdf(score, -1))

    def mixture_density(score: float) -> float:
        return 0.5 * (emission_density(score, 1) + emission_density(score, -1))

    def integrated_posterior_contrast(score: float) -> float:
        return 0.5 * (emission_cdf(score, 1) - emission_cdf(score, -1))

    calibration_pair = quad(
        lambda score: integrated_posterior_contrast(score) ** 2
        * (
            binomial_mass(
                calibration_count - 2, order - 1, mixture_cdf(score)
            )
            - binomial_mass(
                calibration_count - 2, order - 2, mixture_cdf(score)
            )
        )
        * mixture_density(score),
        -np.inf,
        np.inf,
        epsabs=2e-13,
        epsrel=2e-12,
    )[0]

    final_pair = -quad(
        lambda score: 0.5
        * (emission_density(score, 1) - emission_density(score, -1))
        * integrated_posterior_contrast(score)
        * binomial_mass(
            calibration_count - 1, order - 1, mixture_cdf(score)
        ),
        -np.inf,
        np.inf,
        epsabs=2e-13,
        epsrel=2e-12,
    )[0]
    combined = (calibration_count - 1) * calibration_pair + final_pair
    return calibration_pair, final_pair, combined


def simulate_split_coverage(
    calibration_count: int,
    order: int,
    correlation: float,
    size: int,
    rng: np.random.Generator,
    signal: float = 1.0,
    noise: float = 0.5,
) -> float:
    """Direct simulation of the Markov-dependent calibration rank event."""
    state = np.where(rng.random(size) < 0.5, -1.0, 1.0)
    calibration = np.empty((size, calibration_count))
    stay_probability = (1.0 + correlation) / 2.0
    for index in range(calibration_count):
        calibration[:, index] = signal * state + noise * rng.normal(size=size)
        state *= np.where(rng.random(size) < stay_probability, 1.0, -1.0)
    test = signal * state + noise * rng.normal(size=size)
    threshold_value = np.partition(calibration, order - 1, axis=1)[:, order - 1]
    return float(np.mean(test <= threshold_value))


def sliding_block_model(
    window_length: int, correlation: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """States, stationary law, and transition matrix for sliding blocks.

    Each block contains ``window_length`` consecutive +/-1 states.  A legal
    transition drops its first coordinate and appends one new state.
    """
    if window_length < 1:
        raise ValueError("window_length must be positive")
    if not (-1.0 < correlation < 1.0):
        raise ValueError("correlation must lie strictly between -1 and 1")

    block_count = 2**window_length
    states = np.empty((block_count, window_length), dtype=float)
    for index in range(block_count):
        states[index] = [
            1.0 if index & (1 << (window_length - 1 - bit)) else -1.0
            for bit in range(window_length)
        ]

    stationary = np.full(block_count, 0.5)
    if window_length > 1:
        stationary *= np.prod(
            (1.0 + correlation * states[:, :-1] * states[:, 1:]) / 2.0,
            axis=1,
        )

    transition = np.zeros((block_count, block_count))
    for source, block in enumerate(states):
        for appended in (-1.0, 1.0):
            target_block = np.r_[block[1:], appended]
            target = int(np.flatnonzero(np.all(states == target_block, axis=1))[0])
            transition[source, target] = (
                1.0 + correlation * block[-1] * appended
            ) / 2.0

    assert abs(stationary.sum() - 1.0) < 2e-14
    assert np.max(np.abs(stationary @ transition - stationary)) < 2e-14
    return states, stationary, transition


def exact_sliding_window_coverage(
    calibration_count: int,
    order: int,
    window_length: int,
    correlation: float,
    signal: float = 1.0,
    noise: float = 0.5,
    quadrature_order: int = 80,
    stride: int = 1,
) -> float:
    """Exact marginal coverage for strided sliding path scores.

    The latent mean of score ``j`` is the average of the length-L block
    ``(Y_{j*stride},...,Y_{j*stride+L-1})``.  Conditional score noises are
    independent.  The dynamic program keeps the full sliding block and the
    count of calibration scores below a fixed test score, then integrates the
    test score.  Its sampled block kernel is the ``stride``th power of the
    one-step sliding-block kernel.
    """
    if not (1 <= order <= calibration_count):
        raise ValueError("order must be between one and calibration_count")
    if stride < 1:
        raise ValueError("stride must be positive")

    blocks, stationary, transition = sliding_block_model(
        window_length, correlation
    )
    sampled_transition = np.linalg.matrix_power(transition, stride)
    means = signal * blocks.mean(axis=1)
    nodes, weights = roots_hermite(quadrature_order)
    coverage = 0.0

    for test_index, test_mean in enumerate(means):
        for node, weight in zip(nodes, weights):
            test_score = test_mean + noise * math.sqrt(2.0) * node
            probabilities = ndtr((test_score - means) / noise)
            distribution = np.zeros((calibration_count + 1, len(blocks)))
            distribution[0] = stationary
            for _ in range(calibration_count):
                emitted = distribution * (1.0 - probabilities)[None, :]
                emitted[1:] += distribution[:-1] * probabilities[None, :]
                distribution = emitted @ sampled_transition
            coverage += (
                weight / math.sqrt(math.pi)
                * distribution[:order, test_index].sum()
            )
    return float(coverage)


def simulate_sliding_window_coverage(
    calibration_count: int,
    order: int,
    window_length: int,
    correlation: float,
    size: int,
    rng: np.random.Generator,
    signal: float = 1.0,
    noise: float = 0.5,
    stride: int = 1,
) -> float:
    """Direct simulation check for ``exact_sliding_window_coverage``."""
    if stride < 1:
        raise ValueError("stride must be positive")
    path_length = calibration_count * stride + window_length
    paths = np.empty((size, path_length))
    paths[:, 0] = np.where(rng.random(size) < 0.5, -1.0, 1.0)
    stay_probability = (1.0 + correlation) / 2.0
    for time in range(1, path_length):
        paths[:, time] = paths[:, time - 1] * np.where(
            rng.random(size) < stay_probability, 1.0, -1.0
        )
    cumulative = np.cumsum(paths, axis=1)
    padded = np.pad(cumulative, ((0, 0), (1, 0)))
    all_averages = (
        padded[:, window_length:] - padded[:, :-window_length]
    ) / window_length
    averages = all_averages[:, ::stride]
    assert averages.shape[1] == calibration_count + 1
    scores = signal * averages + noise * rng.normal(size=averages.shape)
    threshold = np.partition(
        scores[:, :calibration_count], order - 1, axis=1
    )[:, order - 1]
    return float(np.mean(scores[:, calibration_count] <= threshold))


@lru_cache(maxsize=None)
def beta_rule(alpha: int, beta: int, order: int = 72) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and normalized weights for a Beta(alpha,beta) expectation."""
    x, weights = roots_jacobi(order, beta - 1, alpha - 1)
    return (x + 1) / 2, weights / weights.sum()


def beta_expect(alpha: int, beta: int, function) -> complex:
    nodes, weights = beta_rule(alpha, beta)
    return np.dot(weights, function(nodes))


def poisson_terms(c: float):
    """Yield (jump count, probability) until less than 2e-14 mass remains."""
    probability = math.exp(-c)
    cumulative = probability
    yield 0, probability
    n = 0
    minimum = max(10, int(c + 10 * math.sqrt(max(c, 1))))
    while 1 - cumulative > 2e-14 or n < minimum:
        n += 1
        probability *= c / n
        cumulative += probability
        yield n, probability


def conditional_values(c: float, function) -> tuple[complex, complex]:
    """Exact E[function(A_c)] for starts +1 and -1."""
    plus = minus = 0.0
    for jumps, weight in poisson_terms(c):
        if jumps == 0:
            plus += weight * function(np.array([1.0]))[0]
            minus += weight * function(np.array([-1.0]))[0]
        elif jumps % 2:
            k = (jumps - 1) // 2
            common = beta_expect(k + 1, k + 1, lambda u: function(2 * u - 1))
            plus += weight * common
            minus += weight * common
        else:
            k = jumps // 2
            plus_term = beta_expect(k + 1, k, lambda u: function(2 * u - 1))
            minus_term = beta_expect(k, k + 1, lambda u: function(2 * u - 1))
            plus += weight * plus_term
            minus += weight * minus_term
    return plus, minus


def conditional_cdfs(q: float, c: float) -> tuple[float, float]:
    values = conditional_values(c, lambda a: ndtr((q - MU * a) / SIGMA))
    return float(values[0]), float(values[1])


def threshold(c: float, start_probability: float = 0.5) -> float:
    def equation(q: float) -> float:
        plus, minus = conditional_cdfs(q, c)
        return start_probability * plus + (1 - start_probability) * minus - TARGET

    return brentq(equation, -4.0, 4.0, xtol=2e-13)


def mixture_characteristic(k: float, c: float, start: int) -> complex:
    values = conditional_values(c, lambda a: np.exp(1j * k * MU * a))
    return math.exp(-0.5 * SIGMA**2 * k**2) * values[0 if start == 1 else 1]


def feynman_kac_characteristic(k: float, c: float, start: int) -> complex:
    """Closed two-by-two matrix exponential, independent of the Beta mixture."""
    z = 1j * k * MU / c
    omega = cmath.sqrt(1 + z**2)
    sign = 1 if start == 1 else -1
    transform = math.exp(-c) * (
        cmath.cosh(c * omega) + (1 + sign * z) * cmath.sinh(c * omega) / omega
    )
    return math.exp(-0.5 * SIGMA**2 * k**2) * transform


def simulate_scores(c: float, start: int, size: int, rng: np.random.Generator) -> np.ndarray:
    """Simulate exponential holding times, not the Poisson--Beta representation."""
    remaining = np.full(size, c)
    state = np.full(size, float(start))
    integral = np.zeros(size)
    active = np.ones(size, dtype=bool)
    while np.any(active):
        indices = np.flatnonzero(active)
        waits = rng.exponential(size=len(indices))
        used = np.minimum(waits, remaining[indices])
        integral[indices] += state[indices] * used
        remaining[indices] -= used
        flipped = waits < used + 1e-15
        state[indices[flipped]] *= -1
        active[indices[~flipped]] = False
    return MU * integral / c + SIGMA * rng.normal(size=size)


def terminal_gap(c: float) -> float:
    """Coverage gap for the terminal analogue MU*Y_c + SIGMA*Z."""
    pooled = lambda q: 0.5 * (
        ndtr((q - MU) / SIGMA) + ndtr((q + MU) / SIGMA)
    )
    q = brentq(lambda x: pooled(x) - TARGET, -4, 4)
    initial_gap = ndtr((q - MU) / SIGMA) - TARGET
    return math.exp(-2 * c) * initial_gap


def predictor_pooled_threshold(
    switching_rate: float, predictor_rate: float, signal: float, noise: float
) -> float:
    """Stationary score quantile for the exponentially filtered predictor."""
    nu = switching_rate / predictor_rate
    nodes, weights = roots_jacobi(100, nu - 1, nu - 1)
    weights = weights / weights.sum()
    return brentq(
        lambda q: np.dot(weights, ndtr((q - signal * nodes) / noise)) - TARGET,
        -4.0,
        4.0,
        xtol=2e-13,
    )


def predictor_cdfs(
    times: np.ndarray,
    x0: float,
    q: float,
    switching_rate: float,
    predictor_rate: float,
    signal: float,
    noise: float,
    quadrature_order: int = 180,
) -> np.ndarray:
    """Exact CDFs from a vectorized time-ordered Feynman--Kac solve."""
    nodes, weights = roots_legendre(quadrature_order)
    cutoff = 14 / noise
    frequencies = (nodes + 1) * cutoff / 2
    weights = weights * cutoff / 2
    count = len(frequencies)

    def rhs(tau: float, values: np.ndarray) -> np.ndarray:
        plus, minus = values[:count], values[count:]
        potential = 1j * frequencies * signal * predictor_rate * np.exp(-predictor_rate * tau)
        return np.r_[
            (-switching_rate + potential) * plus + switching_rate * minus,
            switching_rate * plus + (-switching_rate - potential) * minus,
        ]

    solution = solve_ivp(
        rhs,
        (0.0, float(times[-1])),
        np.ones(2 * count, dtype=complex),
        t_eval=times,
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    ).y
    start_plus_probability = (1 + x0) / 2
    answer = []
    for column, time in enumerate(times):
        transform = (
            start_plus_probability * solution[:count, column]
            + (1 - start_plus_probability) * solution[count:, column]
        )
        characteristic = np.exp(
            -0.5 * noise**2 * frequencies**2
            + 1j * frequencies * signal * np.exp(-predictor_rate * time) * x0
        ) * transform
        integral = np.dot(
            weights,
            np.imag(np.exp(-1j * frequencies * q) * characteristic) / frequencies,
        )
        answer.append(0.5 - integral / np.pi)
    return np.asarray(answer)


def predictor_memory(
    time: float | np.ndarray, switching_rate: float, predictor_rate: float
) -> np.ndarray:
    """E[X_t | X_0=x]/x under the stationary hidden-state posterior."""
    time = np.asarray(time)
    relaxation = 2 * switching_rate
    if abs(relaxation - predictor_rate) < 1e-12:
        return (1 + predictor_rate * time) * np.exp(-predictor_rate * time)
    return (
        relaxation * np.exp(-predictor_rate * time)
        - predictor_rate * np.exp(-relaxation * time)
    ) / (relaxation - predictor_rate)


def simulate_predictor_score(
    time: float,
    x0: float,
    size: int,
    switching_rate: float,
    predictor_rate: float,
    signal: float,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Direct piecewise-deterministic simulation from stationary P(Y0|X0)."""
    state = np.where(rng.random(size) < (1 + x0) / 2, 1.0, -1.0)
    predictor = np.full(size, x0)
    remaining = np.full(size, time)
    active = np.ones(size, dtype=bool)
    while np.any(active):
        indices = np.flatnonzero(active)
        waits = rng.exponential(1 / switching_rate, size=len(indices))
        used = np.minimum(waits, remaining[indices])
        predictor[indices] = state[indices] + (
            predictor[indices] - state[indices]
        ) * np.exp(-predictor_rate * used)
        remaining[indices] -= used
        flipped = waits < used + 1e-15
        state[indices[flipped]] *= -1
        active[indices[~flipped]] = False
    return signal * predictor + noise * rng.normal(size=size)


def main() -> None:
    rng = np.random.default_rng(20260923)

    # Independent transform check over both oscillatory and non-oscillatory cases.
    for c in (0.4, 1.0, 4.0, 16.0):
        for k in (0.3, 1.7, 4.2):
            for start in (-1, 1):
                error = abs(
                    mixture_characteristic(k, c, start)
                    - feynman_kac_characteristic(k, c, start)
                )
                assert error < 3e-13

    print("Path-score crossover at the pooled 90% threshold")
    print(" c       q_pool       coverage +     coverage -     c*(gap +)     terminal gap +")
    gaps = []
    for c in (0.5, 1, 2, 4, 8, 16, 32, 64):
        q = threshold(c)
        plus, minus = conditional_cdfs(q, c)
        assert abs(0.5 * (plus + minus) - TARGET) < 3e-12
        gap = plus - TARGET
        gaps.append(gap)
        print(
            f"{c:4.1f}   {q:11.8f}   {plus:12.8f}   {minus:12.8f}"
            f"   {c*gap:12.8f}   {terminal_gap(c):14.8e}"
        )

    limit = -MU * norm.pdf(norm.ppf(TARGET)) / (2 * SIGMA)
    assert abs(64 * gaps[-1] - limit) < 0.006
    print(f"Predicted limit of c times the + coverage gap: {limit:.8f}")

    # The state-aware thresholds differ from the pooled threshold by +/-mu/(2c).
    print("\nPopulation thresholds at c=4")
    print(" P(Y0=+)     threshold       conditional/model coverage")
    for probability in (0.0, 0.25, 0.5, 0.75, 1.0):
        q = threshold(4.0, probability)
        plus, minus = conditional_cdfs(q, 4.0)
        coverage = probability * plus + (1 - probability) * minus
        assert abs(coverage - TARGET) < 3e-12
        print(f"   {probability:4.2f}       {q:11.8f}          {coverage:11.8f}")

    for c in (16.0, 32.0, 64.0):
        pooled = threshold(c)
        plus_q = threshold(c, 1.0)
        minus_q = threshold(c, 0.0)
        assert abs(c * (plus_q - pooled) - MU / 2) < 0.02
        assert abs(c * (minus_q - pooled) + MU / 2) < 0.02

    # Direct holding-time simulation at two crossover scales.
    for c in (1.0, 4.0):
        q = threshold(c)
        exact = conditional_cdfs(q, c)
        observed = tuple(
            np.mean(simulate_scores(c, start, 120_000, rng) <= q)
            for start in (1, -1)
        )
        assert np.max(np.abs(np.asarray(observed) - exact)) < 0.004
        print(f"Direct path simulation c={c:g}: exact={np.round(exact, 6)}, observed={np.round(observed, 6)}")

    # Exact first moment of the path average.
    for c in (0.3, 1.0, 7.0):
        plus_mean, minus_mean = conditional_values(c, lambda a: a)
        expected = (1 - math.exp(-2 * c)) / (2 * c)
        assert abs(plus_mean - expected) < 3e-13
        assert abs(minus_mean + expected) < 3e-13

    # Calibration windows inherit dependence from their shared path.  Check
    # the closed covariance against independent tensor quadrature on both
    # sides of the overlap boundary.
    horizon, switching_rate = 1.0, 8.0
    for lag in (0.0, 0.1, 0.5, 0.999, 1.0, 1.4, 2.0):
        closed = window_covariance(lag, horizon, switching_rate)
        numerical = window_covariance_quadrature(lag, horizon, switching_rate)
        assert abs(closed - numerical) < 2e-12

    # The covariance is continuous at the point where the windows cease to
    # overlap.  At fixed lag below the horizon, its correlation tends to the
    # triangular overlap fraction as switching becomes fast.
    boundary = (
        1 - math.exp(-2 * switching_rate * horizon)
    ) ** 2 / (4 * switching_rate**2 * horizon**2)
    assert abs(window_covariance(horizon, horizon, switching_rate) - boundary) < 1e-15
    print("\nOverlapping calibration windows")
    print(" rate     Corr(A_0,A_0.5)    variance inflation    effective N (N=200)")
    for rate in (2.0, 8.0, 32.0, 128.0):
        variance = window_covariance(0.0, horizon, rate)
        correlation = window_covariance(0.5, horizon, rate) / variance
        inflation, effective = linear_window_effective_size(
            200, 0.1, horizon, rate
        )
        print(
            f" {rate:4.0f}        {correlation:12.8f}"
            f"          {inflation:12.8f}       {effective:10.5f}"
        )
    fast_inflation = 1 + 2 * sum(
        (1 - lag / 200) * max(1 - lag * 0.1 / horizon, 0)
        for lag in range(1, 200)
    )
    inflation, effective = linear_window_effective_size(200, 0.1, horizon, 128.0)
    assert abs(inflation - fast_inflation) < 0.04
    assert abs(effective / 200 - 1 / fast_inflation) < 5e-4
    print(
        "Fast-limit effective fraction: "
        f"exact(rate=128)={effective/200:.8f}, triangular={1/fast_inflation:.8f}"
    )

    # The random calibration order statistic is a separate finite-sample
    # effect.  For N=9 and k=9 the iid rank coverage is exactly 0.9.  The
    # dynamic program propagates the full Markov-binomial count law up to the
    # immediately following test score.
    calibration_count, order = 9, 9
    iid_coverage = order / (calibration_count + 1)
    pair_cc, pair_ct, coefficient = split_coverage_first_order(
        calibration_count, order
    )
    correlations = np.array([0.1, 0.05, 0.025, 0.0125])
    finite_coverages = np.array([
        exact_split_coverage(calibration_count, order, rho)
        for rho in correlations
    ])
    residuals = np.abs(
        finite_coverages - iid_coverage - correlations * coefficient
    )
    residual_rates = np.log2(residuals[:-1] / residuals[1:])
    assert abs(exact_split_coverage(calibration_count, order, 0.0) - iid_coverage) < 2e-14
    assert abs(
        (exact_split_coverage(calibration_count, order, 1e-4) - iid_coverage)
        / 1e-4
        - coefficient
    ) < 4e-6
    assert residual_rates[-1] > 1.98
    persistent_coverage = exact_split_coverage(
        calibration_count, order, 0.8
    )
    simulated_coverage = simulate_split_coverage(
        calibration_count, order, 0.8, 300_000, rng
    )
    assert abs(simulated_coverage - persistent_coverage) < 0.002
    print("\nFinite-sample split threshold")
    print(f"iid rank coverage: {iid_coverage:.8f}")
    print(
        "first-order adjacent-pair terms: "
        f"CC={pair_cc:.10f}, CT={pair_ct:.10f}, total={coefficient:.10f}"
    )
    print(
        "last measured corrected-residual rate: "
        f"{residual_rates[-1]:.6f}"
    )
    print(
        "exact coverage at adjacent-state correlation 0.8: "
        f"{persistent_coverage:.8f}"
    )
    print(
        "direct simulation at correlation 0.8: "
        f"{simulated_coverage:.8f}"
    )

    # Marginal iid rank coverage averages over the random calibration sample.
    # Conditional on that sample, the next-score coverage is F(Q_k).  The PIT
    # turns this into the kth order statistic of N independent uniforms, hence
    # Beta(k, N+1-k).  Verify its moments and a lower tail by independent
    # adaptive quadrature, then reproduce them from fresh calibration panels.
    beta_a = order
    beta_b = calibration_count + 1 - order
    beta_normalizer = math.factorial(calibration_count) / (
        math.factorial(beta_a - 1) * math.factorial(beta_b - 1)
    )
    beta_density = lambda coverage: (
        beta_normalizer
        * coverage ** (beta_a - 1)
        * (1.0 - coverage) ** (beta_b - 1)
    )
    beta_mass = quad(beta_density, 0.0, 1.0, epsabs=2e-14)[0]
    beta_mean = quad(
        lambda coverage: coverage * beta_density(coverage),
        0.0,
        1.0,
        epsabs=2e-14,
    )[0]
    beta_second = quad(
        lambda coverage: coverage**2 * beta_density(coverage),
        0.0,
        1.0,
        epsabs=2e-14,
    )[0]
    beta_variance = (
        beta_a * beta_b
        / ((calibration_count + 1) ** 2 * (calibration_count + 2))
    )
    beta_lower_tail = quad(beta_density, 0.0, 0.8, epsabs=2e-14)[0]
    assert abs(beta_mass - 1.0) < 2e-14
    assert abs(beta_mean - iid_coverage) < 2e-14
    assert abs(beta_second - beta_mean**2 - beta_variance) < 2e-14
    assert abs(beta_lower_tail - 0.8**9) < 2e-14

    calibration_uniforms = rng.random((400_000, calibration_count))
    conditional_coverages = np.partition(
        calibration_uniforms, order - 1, axis=1
    )[:, order - 1]
    empirical_tail = np.mean(conditional_coverages < 0.8)
    assert abs(np.mean(conditional_coverages) - beta_mean) < 4e-4
    assert abs(np.var(conditional_coverages) - beta_variance) < 5e-5
    assert abs(empirical_tail - beta_lower_tail) < 0.0015
    beta_quantiles = np.array([0.05, 0.10, 0.50]) ** (1.0 / order)
    empirical_quantiles = np.quantile(
        conditional_coverages, [0.05, 0.10, 0.50]
    )
    assert np.max(np.abs(empirical_quantiles - beta_quantiles)) < 0.0015
    print("\nIid coverage conditional on the calibration sample")
    print(
        f"law: Beta({beta_a},{beta_b}); mean={beta_mean:.8f}; "
        f"sd={math.sqrt(beta_variance):.8f}"
    )
    print(
        "exact 5%, 10%, 50% quantiles: "
        + ", ".join(f"{quantile:.8f}" for quantile in beta_quantiles)
    )
    print(
        "P(conditional coverage < 0.8): "
        f"exact={beta_lower_tail:.9f}, simulation={empirical_tail:.9f}"
    )

    # The beta law gives the sharp one-sided tolerance-limit design.  To have
    # conditional coverage at least p with calibration-sample probability at
    # least 1-delta, the kth order statistic must satisfy a binomial CDF
    # inequality.  If even the sample maximum fails, no finite order statistic
    # can meet the requested pair (p, delta).
    target = 0.9
    failure_probabilities = (0.10, 0.05, 0.01)
    minimum_counts = []
    for failure_probability in failure_probabilities:
        count = 1
        while target**count > failure_probability:
            count += 1
        minimum_counts.append(count)
        assert pac_calibration_order(
            count - 1, target, failure_probability
        ) is None
        assert pac_calibration_order(
            count, target, failure_probability
        ) == count
    assert minimum_counts == [22, 29, 44]

    conventional_order = math.ceil(target * (100 + 1))
    conventional_confidence = iid_training_confidence(
        100, conventional_order, target
    )
    pac_orders = [
        pac_calibration_order(100, target, failure_probability)
        for failure_probability in failure_probabilities
    ]
    assert pac_orders == [95, 96, 97]
    assert abs(conventional_confidence - 0.5487098345579959) < 2e-14

    # Independently integrate the beta density and compare it with the
    # binomial identity for several nontrivial designs.
    pac_quadrature_error = 0.0
    for check_count, check_order in (
        (9, 9),
        (22, 22),
        (50, 49),
        (100, 96),
        (200, 188),
    ):
        a, b = check_order, check_count + 1 - check_order
        log_beta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)

        def density(coverage):
            if coverage <= 0.0 or coverage >= 1.0:
                return 0.0
            return math.exp(
                (a - 1) * math.log(coverage)
                + (b - 1) * math.log1p(-coverage)
                - log_beta
            )

        beta_success = quad(
            density, target, 1.0, epsabs=1e-13, epsrel=1e-13
        )[0]
        binomial_success = iid_training_confidence(
            check_count, check_order, target
        )
        pac_quadrature_error = max(
            pac_quadrature_error, abs(beta_success - binomial_success)
        )
    assert pac_quadrature_error < 5e-13

    print("\nSharp iid training-conditional calibration design")
    print("target coverage 0.9")
    print(" delta      minimum N     required k     marginal coverage")
    for failure_probability, count in zip(
        failure_probabilities, minimum_counts
    ):
        print(
            f" {failure_probability:5.2f}       {count:5d}          {count:5d}"
            f"          {count / (count + 1):.8f}"
        )
    print(
        "N=100 conventional k=91: training confidence "
        f"{conventional_confidence:.8f}"
    )
    print("N=100 PAC orders for delta=.10,.05,.01: " + str(pac_orders))
    print(
        "maximum beta-integral/binomial discrepancy: "
        f"{pac_quadrature_error:.3e}"
    )

    # Transfer the iid beta tail to a dependent training-conditional
    # statement.  The score (Y + U)/2 has an exactly uniform stationary
    # marginal but reveals the binary state.  Its conditional coverage law is
    # therefore non-iid and still admits the exact state-path/beta calculation
    # above.  Separate the calibration-panel distance from iid from the cost
    # of decoupling only the test score from the actual panel.  This uses
    # (N-1) beta + beta/slack rather than coupling all N+1 scores twice.
    transfer_target = 0.8
    transfer_strides = (28, 40, 48)
    transfer_rows = []
    for stride in transfer_strides:
        sampled_correlation = 0.8**stride
        sampled_transition = np.array([
            [
                (1.0 + sampled_correlation) / 2.0,
                (1.0 - sampled_correlation) / 2.0,
            ],
            [
                (1.0 - sampled_correlation) / 2.0,
                (1.0 + sampled_correlation) / 2.0,
            ],
        ])
        exact_failure = exact_binary_training_failure(
            calibration_count,
            order,
            transfer_target,
            sampled_correlation,
        )
        exact_panel_tv = sampled_path_total_variation(
            sampled_transition,
            np.array([0.5, 0.5]),
            1,
            calibration_count,
        )
        exact_test_tv = sampled_test_decoupling_total_variation(
            sampled_transition,
            np.array([0.5, 0.5]),
            1,
            calibration_count,
        )
        beta = sampled_correlation / 2.0
        berbee_panel_tv = min(1.0, (calibration_count - 1) * beta)
        berbee_test_tv = beta

        def optimized_bound(panel_tv, test_tv):
            result = minimize_scalar(
                lambda slack: training_conditional_transfer_bound(
                    calibration_count,
                    order,
                    transfer_target,
                    panel_tv,
                    test_tv,
                    slack,
                ),
                bounds=(1e-10, 1.0 - transfer_target - 1e-10),
                method="bounded",
                options={"xatol": 1e-14},
            )
            return result.fun, result.x

        exact_tv_bound, exact_tv_slack = optimized_bound(
            exact_panel_tv, exact_test_tv
        )
        berbee_bound, berbee_slack = optimized_bound(
            berbee_panel_tv, berbee_test_tv
        )
        rearrangement_bound, rearrangement_cutoff = (
            binary_panel_rearrangement_bound(
                calibration_count,
                order,
                transfer_target,
                sampled_correlation,
                exact_test_tv,
            )
        )
        assert exact_failure <= exact_tv_bound + 2e-14
        assert exact_failure <= rearrangement_bound + 2e-14
        assert rearrangement_bound <= exact_tv_bound + 2e-14
        assert exact_tv_bound <= berbee_bound + 2e-14
        assert exact_panel_tv <= berbee_panel_tv + 2e-14
        assert abs(exact_test_tv - beta) < 2e-14
        assert abs(
            binary_panel_order_cdf(
                1.0, calibration_count, order, sampled_correlation
            )
            - 1.0
        ) < 2e-14
        transfer_rows.append((
            stride,
            exact_failure,
            exact_panel_tv,
            exact_test_tv,
            berbee_panel_tv,
            exact_tv_bound,
            exact_tv_slack,
            berbee_bound,
            berbee_slack,
            rearrangement_bound,
            rearrangement_cutoff,
        ))

    expected_transfer_failures = (
        0.13428284941340723,
        0.1342221891772618,
        0.13421847631778047,
    )
    assert max(
        abs(row[1] - expected)
        for row, expected in zip(transfer_rows, expected_transfer_failures)
    ) < 2e-14
    expected_rearrangement_bounds = (
        0.1914916332883245,
        0.14861291518720113,
        0.1400590709815776,
    )
    assert max(
        abs(row[9] - expected)
        for row, expected in zip(
            transfer_rows, expected_rearrangement_bounds
        )
    ) < 2e-12
    print("\nTraining-conditional total-variation transfer")
    print("randomized binary score, N=k=9, target 0.8")
    print(
        "stride   exact failure   panel TV   test TV   "
        "rearranged   exact-TV   separated-beta"
    )
    for row in transfer_rows:
        print(
            f" {row[0]:3d}      {row[1]:.8f}   {row[2]:.8f} "
            f"{row[3]:.8f}   {row[9]:.8f}  {row[5]:.8f}     {row[7]:.8f}"
        )
    print(
        "iid beta failure at target 0.8: "
        f"{integer_beta_cdf(0.8, order, calibration_count + 1 - order):.9f}"
    )

    # Sliding path scores share latent states even when the sampled states are
    # independent (rho=0).  The lifted-block dynamic program is exact up to
    # Gauss--Hermite quadrature and makes this geometric dependence explicit.
    window_length = 4
    overlap_independent = exact_sliding_window_coverage(
        calibration_count, order, window_length, 0.0, quadrature_order=100
    )
    overlap_persistent = exact_sliding_window_coverage(
        calibration_count, order, window_length, 0.8, quadrature_order=100
    )
    strides = (1, 2, 4, 12, 20, 28)
    independent_stride_coverages = np.array([
        exact_sliding_window_coverage(
            calibration_count, order, window_length, 0.0,
            quadrature_order=100, stride=stride,
        )
        for stride in strides
    ])
    persistent_stride_coverages = np.array([
        exact_sliding_window_coverage(
            calibration_count, order, window_length, 0.8,
            quadrature_order=100, stride=stride,
        )
        for stride in strides
    ])
    _, block_stationary, block_transition = sliding_block_model(
        window_length, 0.8
    )
    tv_errors = []
    for stride in strides[2:]:
        sampled = np.linalg.matrix_power(block_transition, stride)
        row_tv = 0.5 * np.abs(sampled - block_stationary[None, :]).sum(axis=1)
        predicted_tv = 0.8 ** (stride - window_length + 1) / 2
        tv_errors.append(np.max(np.abs(row_tv - predicted_tv)))
    assert max(tv_errors) < 2e-15
    coarse_overlap = exact_sliding_window_coverage(
        calibration_count, order, window_length, 0.8, quadrature_order=60
    )
    assert abs(coarse_overlap - overlap_persistent) < 3e-8
    assert overlap_independent < iid_coverage - 0.01
    assert np.max(np.abs(independent_stride_coverages[2:] - iid_coverage)) < 3e-14
    # For d >= L, the exact one-step block mixing distance is
    # |rho|^(d-L+1)/2.  A sequential maximal coupling over N transitions
    # bounds the rank-event error by N times this distance.
    for stride, coverage in zip(strides[2:], persistent_stride_coverages[2:]):
        bound = min(
            1.0,
            calibration_count * 0.8 ** (stride - window_length + 1) / 2,
        )
        assert abs(coverage - iid_coverage) <= bound + 2e-14
    overlap_simulated_independent = simulate_sliding_window_coverage(
        calibration_count, order, window_length, 0.0, 300_000, rng
    )
    overlap_simulated_persistent = simulate_sliding_window_coverage(
        calibration_count, order, window_length, 0.8, 300_000, rng
    )
    disjoint_simulated = simulate_sliding_window_coverage(
        calibration_count, order, window_length, 0.0, 200_000, rng,
        stride=window_length,
    )
    assert abs(overlap_simulated_independent - overlap_independent) < 0.002
    assert abs(overlap_simulated_persistent - overlap_persistent) < 0.002
    assert abs(disjoint_simulated - iid_coverage) < 0.002
    # L=1 is exactly the terminal-emission dynamic program above.
    assert abs(
        exact_sliding_window_coverage(
            calibration_count, order, 1, 0.8, quadrature_order=100
        )
        - persistent_coverage
    ) < 3e-11
    print("\nFinite-sample overlapping path threshold")
    print(
        "four-point windows, independent latent samples: "
        f"exact={overlap_independent:.8f}, "
        f"simulation={overlap_simulated_independent:.8f}"
    )
    print(
        "four-point windows, adjacent correlation 0.8: "
        f"exact={overlap_persistent:.8f}, "
        f"simulation={overlap_simulated_persistent:.8f}"
    )
    print("stride     rho=0 coverage     rho=0.8 coverage     coupling bound")
    for stride, independent, persistent in zip(
        strides, independent_stride_coverages, persistent_stride_coverages
    ):
        bound = (
            min(
                1.0,
                calibration_count * 0.8 ** (stride - window_length + 1) / 2,
            )
            if stride >= window_length else float("nan")
        )
        print(
            f" {stride:3d}       {independent:12.8f}       "
            f"{persistent:12.8f}       {bound:12.8f}"
        )
    print(
        "disjoint-window simulation at rho=0: "
        f"{disjoint_simulated:.8f}"
    )
    print(
        "maximum error in exact block total-variation formula: "
        f"{max(tv_errors):.3e}"
    )

    # The same rank-event argument is not specific to a binary chain.  For a
    # stationary process, Berbee's sequential coupling bounds the TV distance
    # of N+1 separated blocks from iid blocks by N beta(g), where g is the
    # number of transitions from the end of one block to the start of the
    # next.  A nonreversible three-state chain checks both the finite-chain
    # beta formula and the induced rank-coverage inequality.
    general_transition = np.array([
        [0.74, 0.20, 0.06],
        [0.08, 0.67, 0.25],
        [0.31, 0.09, 0.60],
    ])
    general_stationary = stationary_distribution(general_transition)
    general_means = np.array([-1.0, 0.25, 1.4])
    general_strides = (1, 2, 4, 8, 12)
    general_betas = np.array([
        markov_beta(general_transition, general_stationary, stride)
        for stride in general_strides
    ])
    general_coverages = np.array([
        finite_markov_rank_coverage(
            general_transition,
            general_stationary,
            general_means,
            0.65,
            calibration_count,
            order,
            stride,
        )
        for stride in general_strides
    ])
    path_tvs = np.array([
        sampled_path_total_variation(
            general_transition, general_stationary, stride, 5
        )
        for stride in general_strides
    ])
    assert (
        np.max(
            np.abs(general_stationary @ general_transition - general_stationary)
        )
        < 2e-14
    )
    flux = general_stationary[:, None] * general_transition
    assert np.max(np.abs(flux - flux.T)) > 0.02
    assert np.all(path_tvs <= 4 * general_betas + 3e-15)
    assert np.all(
        np.abs(general_coverages - iid_coverage)
        <= np.minimum(1.0, calibration_count * general_betas) + 3e-14
    )
    assert general_betas[-1] < general_betas[0] / 1000
    print("\nAbsolute-regularity certificate: nonreversible three-state chain")
    print("stride       beta       five-draw TV     exact coverage     N beta")
    for stride, beta, path_tv, coverage in zip(
        general_strides, general_betas, path_tvs, general_coverages
    ):
        print(
            f" {stride:3d}     {beta:10.8f}     {path_tv:12.8f}"
            f"      {coverage:12.8f}     {calibration_count * beta:10.8f}"
        )

    # Continuous scores are not needed if ties are randomized.  Attach an
    # independent uniform to each score and order the pairs lexicographically.
    # For deterministic discrete scores the same coupling argument is valid,
    # but its iid target is conservative and generally not k/(N+1).
    discrete_scores = np.array([0.0, 0.0, 1.0])
    iid_transition = np.tile(general_stationary, (3, 1))
    deterministic_iid = deterministic_iid_discrete_coverage(
        general_stationary,
        discrete_scores,
        calibration_count,
        order,
    )
    deterministic_iid_dp = finite_markov_discrete_coverage(
        iid_transition,
        general_stationary,
        discrete_scores,
        calibration_count,
        order,
        1,
        False,
    )
    randomized_iid = finite_markov_discrete_coverage(
        iid_transition,
        general_stationary,
        discrete_scores,
        calibration_count,
        order,
        1,
        True,
    )
    discrete_strides = (1, 4, 12)
    deterministic_discrete = np.array([
        finite_markov_discrete_coverage(
            general_transition,
            general_stationary,
            discrete_scores,
            calibration_count,
            order,
            stride,
            False,
        )
        for stride in discrete_strides
    ])
    randomized_discrete = np.array([
        finite_markov_discrete_coverage(
            general_transition,
            general_stationary,
            discrete_scores,
            calibration_count,
            order,
            stride,
            True,
        )
        for stride in discrete_strides
    ])
    discrete_betas = np.array([
        markov_beta(general_transition, general_stationary, stride)
        for stride in discrete_strides
    ])
    assert abs(deterministic_iid - deterministic_iid_dp) < 4e-15
    assert deterministic_iid > iid_coverage + 0.08
    assert abs(randomized_iid - iid_coverage) < 4e-15
    assert np.all(
        np.abs(deterministic_discrete - deterministic_iid)
        <= np.minimum(1.0, calibration_count * discrete_betas) + 3e-14
    )
    assert np.all(
        np.abs(randomized_discrete - iid_coverage)
        <= np.minimum(1.0, calibration_count * discrete_betas) + 3e-14
    )
    print("\nDiscrete-score tie certificate")
    print(
        "iid deterministic baseline: "
        f"{deterministic_iid:.8f}; randomized baseline: {randomized_iid:.8f}"
    )
    print("stride       beta       deterministic       randomized       N beta")
    for stride, beta, deterministic, randomized in zip(
        discrete_strides,
        discrete_betas,
        deterministic_discrete,
        randomized_discrete,
    ):
        print(
            f" {stride:3d}     {beta:10.8f}       {deterministic:12.8f}"
            f"     {randomized:12.8f}     {calibration_count * beta:10.8f}"
        )

    # A predictor with an exponentially weighted internal state retains memory
    # at its own rate, even after the endpoint regime has mixed.
    switching_rate, predictor_rate = 1.0, 0.2
    signal, noise, x0 = 0.1, 0.5, 0.8
    nu = switching_rate / predictor_rate
    stationary_variance = 1 / (2 * nu + 1)
    assert abs(stationary_variance - predictor_rate / (predictor_rate + 2 * switching_rate)) < 1e-15
    predictor_q = predictor_pooled_threshold(
        switching_rate, predictor_rate, signal, noise
    )
    times = np.array([0.0, 1.0, 2.0, 5.0, 10.0, 20.0])
    exact = predictor_cdfs(
        times, x0, predictor_q, switching_rate, predictor_rate, signal, noise
    )
    memory = predictor_memory(times, switching_rate, predictor_rate)
    first_order = TARGET - signal / noise * norm.pdf(norm.ppf(TARGET)) * x0 * memory
    print("\nExponentially filtered predictor")
    print(" t       memory factor    exact coverage    first-order coverage    endpoint memory")
    for time, retain, coverage, approximation in zip(times, memory, exact, first_order):
        print(
            f"{time:4.0f}      {retain:12.8f}     {coverage:12.8f}"
            f"       {approximation:12.8f}       {math.exp(-2*switching_rate*time):12.4e}"
        )
    assert np.max(np.abs(exact - first_order)) < 0.0025
    observed = np.mean(
        simulate_predictor_score(
            5.0, x0, 180_000, switching_rate, predictor_rate, signal, noise, rng
        )
        <= predictor_q
    )
    assert abs(observed - exact[3]) < 0.003
    print(f"Direct predictor simulation t=5: exact={exact[3]:.8f}, observed={observed:.8f}")

    print(
        "PASS: path and predictor crossovers, overlap covariance, "
        "finite-sample terminal and strided-window rank coverage, general "
        "absolute-regularity coupling with discrete-score tie handling, "
        "iid training-conditional beta law, sharp PAC design, and dependent "
        "total-variation transfer with a slack-free rearrangement bound, "
        "exact transforms, and simulations"
    )


if __name__ == "__main__":
    main()
