"""Certificate for conditional coverage of a path-dependent Markov score.

The score is S = mu * A_c + sigma * Z, where A_c is the average of a
finite-state chain over c switching times.  In the symmetric two-state case, exact CDFs are
computed from a Poisson--Beta occupation-time mixture.  Smooth nonlinear
functions of that occupation average check both the gradient/Hessian
crossover and the second-order boundary term at a critical score point.
Direct path simulation
and the independent Feynman--Kac characteristic function check that
representation.  A second matrix Feynman--Kac calculation treats unequal
transition rates and checks the stationary-weighted cancellation of unequal
conditional-coverage errors.  A nonreversible three-state calculation then
checks the general Poisson-equation coefficients through second order and the
resulting third-order residual.  Three- and four-state examples also check the
affine dimension of the posterior-cancellation sets: first-order cancellation
need not imply stationarity, and in four states even the first two terms can
vanish at a nonstationary full-support posterior.
The certificate also checks the exact covariance of overlapping occupation
windows, its fast-switching effective-sample limit, and the exact finite-sample
coverage of a sliding-window order-statistic threshold.  It additionally
verifies when stride thinning restores iid rank coverage and an explicit
finite-sample coupling bound for residual Markov dependence.  A separate
nonreversible-chain calculation checks the corresponding absolute-regularity
bound beyond the symmetric binary example, including discrete scores with
deterministic or randomized tie handling.  It also checks the exact beta law
of iid coverage conditional on the realized, tie-augmented calibration sample
for arbitrary score distributions, and the sharp one-sided tolerance-limit/PAC
choice of calibration order statistic.  It also checks the exact beta-spacing
law and sharp PAC rank-width design for an interval with two finite calibration
order-statistic endpoints.  Finally,
it checks total-variation transfers from that iid beta law to conservative
training-conditional tail bounds under absolute regularity, both for regular
and irregular deterministic block spacing.  It checks the exact convex
allocation of a fixed spacing budget for marginal and separated-PAC mixing
envelopes, and an exact counterexample where the calibration panel converges
to iid but fixed final-gap memory leaves a non-iid conditional-coverage law,
including unequal stationary masses for the two latent states.
It also checks the slack-free rearrangement
bound obtained from the iid beta baseline and the two separated TV budgets,
with a sharper version when the actual panel order-statistic law is known.
The complete-law Wasserstein certificate includes the exact beta mean absolute
deviation and its resulting finite calibration-size design.
It finally checks the sharp distinction between a predictor fitted on an
independent training sample and a memorizing predictor fitted on the
calibration labels, as well as the exact total-variation cost of separating a
past training sigma-field from a future Markov panel.
"""

from __future__ import annotations

import cmath
import math
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, minimize_scalar
from scipy.special import ive, ndtr, roots_hermite, roots_jacobi, roots_legendre
from scipy.stats import beta as beta_distribution
from scipy.stats import binom as binomial_distribution
from scipy.stats import norm


TARGET = 0.90
MU = 1.0
SIGMA = 0.5
BERRY_ESSEEN_CONSTANT = 0.4748


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


def pac_interval_rank_width(
    calibration_count: int, target: float, failure_probability: float
) -> int | None:
    """Smallest rank width for two finite order-statistic endpoints.

    If the endpoints have ranks ``lower < upper``, their conditional coverage
    has the Beta(upper-lower, N+1-upper+lower) law.  Only widths 1,...,N-1
    are available because both endpoints must be calibration observations.
    """
    for width in range(1, calibration_count):
        if iid_training_confidence(calibration_count, width, target) >= (
            1.0 - failure_probability
        ):
            return width
    return None


def minimum_two_endpoint_panel(
    target: float, failure_probability: float
) -> tuple[int, int]:
    """Smallest panel size and rank width for a finite two-endpoint interval."""
    calibration_count = 2
    while True:
        width = pac_interval_rank_width(
            calibration_count, target, failure_probability
        )
        if width is not None:
            return calibration_count, width
        calibration_count += 1


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


def beta_mean_absolute_deviation(alpha: int, beta: int) -> float:
    """Exact mean absolute deviation of a beta law about its mean.

    If ``B`` is Beta(alpha, beta) and ``m = alpha / (alpha + beta)``,
    integration of the derivative of ``x**alpha * (1-x)**beta`` gives

        E|B-m| = 2 m**alpha (1-m)**beta
                 / ((alpha+beta) Beta(alpha,beta)).

    The logarithmic evaluation remains stable for large calibration panels.
    """
    if alpha <= 0 or beta <= 0:
        raise ValueError("beta parameters must be positive")
    total = alpha + beta
    mean = alpha / total
    log_value = (
        math.log(2.0)
        + alpha * math.log(mean)
        + beta * math.log1p(-mean)
        - math.log(total)
        - math.lgamma(alpha)
        - math.lgamma(beta)
        + math.lgamma(total)
    )
    return math.exp(log_value)


def least_conditional_mad_panel(
    target: float,
    tolerance: float,
    dependence_budget: float = 0.0,
    maximum_count: int = 1_000_000,
) -> tuple[int, int, float]:
    """Least upper order-statistic design meeting an L1 coverage target.

    The chosen order is ``ceil(target * (N+1))``.  The returned certificate
    is the beta mean absolute deviation, plus the upward rank rounding error,
    plus the separated dependence budget from the Wasserstein theorem.
    """
    if not (0.0 < target < 1.0):
        raise ValueError("target must lie strictly between zero and one")
    if tolerance <= dependence_budget:
        raise ValueError("tolerance must exceed the dependence budget")
    for calibration_count in range(1, maximum_count + 1):
        total = calibration_count + 1
        order = math.ceil(target * total)
        if order > calibration_count:
            continue
        nominal = order / total
        certificate = (
            nominal
            - target
            + beta_mean_absolute_deviation(order, total - order)
            + dependence_budget
        )
        if certificate <= tolerance:
            return calibration_count, order, certificate
    raise ValueError("no design found below maximum_count")


def atomic_lex_order_cdf(
    value: float,
    probabilities: tuple[float, ...],
    calibration_count: int,
    order: int,
) -> float:
    """CDF of randomized conditional coverage for an atomic iid score.

    Attach an independent uniform tie variable to each score and order the
    pairs lexicographically.  If the atom masses are ``probabilities``, the
    randomized distributional transform maps atom j onto its cumulative-mass
    interval.  This routine computes the probability below ``value`` by first
    locating that interval and then applying the order-statistic binomial
    formula.  It deliberately works atom by atom rather than calling a beta
    CDF, providing a certificate of the arbitrary-score extension.
    """
    if not 1 <= order <= calibration_count:
        raise ValueError("order must be between one and calibration_count")
    if any(probability <= 0.0 for probability in probabilities):
        raise ValueError("atom probabilities must be positive")
    if abs(sum(probabilities) - 1.0) > 1e-13:
        raise ValueError("atom probabilities must sum to one")
    if value <= 0.0:
        transformed_cdf = 0.0
    elif value >= 1.0:
        transformed_cdf = 1.0
    else:
        transformed_cdf = 0.0
        for probability in probabilities:
            interval_right = transformed_cdf + probability
            if value >= interval_right:
                transformed_cdf = interval_right
                continue
            tie_fraction = (value - transformed_cdf) / probability
            transformed_cdf += probability * tie_fraction
            break
    return sum(
        binomial_mass(calibration_count, count, transformed_cdf)
        for count in range(order, calibration_count + 1)
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


def optimal_exponential_gap_allocation(
    weights: np.ndarray, excess_gap_budget: float, decay_rate: float
) -> np.ndarray:
    """Minimize sum_i weights_i exp(-decay_rate*x_i).

    The constraints are x_i >= 0 and sum_i x_i = excess_gap_budget.  The
    logarithmic water level implements the exact KKT solution and also covers
    coordinates that optimally remain at their minimum gap.
    """
    weights = np.asarray(weights, dtype=float)
    if (
        weights.ndim != 1
        or len(weights) == 0
        or np.any(weights <= 0.0)
        or excess_gap_budget < 0.0
        or decay_rate <= 0.0
    ):
        raise ValueError("positive weights/rate and nonnegative budget required")
    if excess_gap_budget == 0.0:
        return np.zeros_like(weights)

    log_weights = np.log(weights)
    target = decay_rate * excess_gap_budget
    log_water = brentq(
        lambda level: np.maximum(log_weights - level, 0.0).sum() - target,
        float(log_weights.min() - target - 1.0),
        float(log_weights.max()),
        xtol=1e-14,
    )
    allocation = np.maximum(log_weights - log_water, 0.0) / decay_rate
    assert abs(allocation.sum() - excess_gap_budget) < 2e-12
    return allocation


def iid_panel_order_lower_cost(
    target: float,
    upper: float,
    calibration_count: int,
    order: int,
) -> float:
    """Lower-tail deficit cost for the iid beta order-statistic law."""
    if upper <= target:
        return 0.0
    alpha = order
    beta = calibration_count + 1 - order
    right = min(upper, 1.0)
    return float(
        quad(
            lambda value: (value - target)
            * beta_distribution.pdf(value, alpha, beta),
            target,
            right,
            epsabs=2e-14,
            epsrel=2e-13,
        )[0]
    )


def iid_robust_rearrangement_bound(
    calibration_count: int,
    order: int,
    target: float,
    panel_total_variation: float,
    test_decoupling_total_variation: float,
) -> tuple[float, float]:
    """Slack-free conditional-failure bound from the iid beta baseline.

    A maximal coupling matches the actual and iid calibration panels outside
    a set of mass ``panel_total_variation``.  On the matched part, the lower
    rearrangement cost is computed under the exact iid beta law.  The result
    therefore needs only the two separated TV budgets, not the actual panel
    order-statistic distribution.
    """
    alpha = order
    beta = calibration_count + 1 - order
    maximum_cost = iid_panel_order_lower_cost(
        target, 1.0, calibration_count, order
    )
    if test_decoupling_total_variation >= maximum_cost:
        return 1.0, 1.0
    cutoff = brentq(
        lambda upper: iid_panel_order_lower_cost(
            target, upper, calibration_count, order
        )
        - test_decoupling_total_variation,
        target,
        1.0,
        xtol=2e-14,
    )
    iid_mass = beta_distribution.cdf(cutoff, alpha, beta)
    return min(1.0, panel_total_variation + iid_mass), cutoff


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


def exact_binary_training_failure_irregular(
    calibration_count: int,
    order: int,
    target: float,
    correlation: float,
    strides: tuple[int, ...],
) -> float:
    """Exact training-conditional failure on an irregular sampling grid.

    ``strides`` contains the ``calibration_count - 1`` successive distances
    inside the calibration panel and the final calibration--test distance.
    This is the irregular-grid analogue of
    :func:`exact_binary_training_failure`.
    """
    import itertools

    if not (1 <= order <= calibration_count):
        raise ValueError("order must be between one and calibration_count")
    if not (-1.0 < correlation < 1.0):
        raise ValueError("correlation must lie strictly between -1 and 1")
    if len(strides) != calibration_count or min(strides) < 1:
        raise ValueError("strides must contain calibration_count positive gaps")

    sampled_correlations = [correlation**stride for stride in strides]
    failure = 0.0
    for states in itertools.product((0, 1), repeat=calibration_count):
        path_probability = 0.5
        for index, (previous, current) in enumerate(
            zip(states[:-1], states[1:])
        ):
            stay = (1.0 + sampled_correlations[index]) / 2.0
            path_probability *= stay if previous == current else 1.0 - stay

        test_stay = (1.0 + sampled_correlations[-1]) / 2.0
        probability_zero = (
            test_stay if states[-1] == 0 else 1.0 - test_stay
        )
        probability_one = 1.0 - probability_zero
        zero_count = states.count(0)
        if zero_count >= order:
            alpha = order
            beta = zero_count + 1 - order
            cutoff = target / probability_zero
        else:
            alpha = order - zero_count
            one_count = calibration_count - zero_count
            beta = one_count + 1 - alpha
            cutoff = (target - probability_zero) / probability_one
        failure += path_probability * integer_beta_cdf(cutoff, alpha, beta)
    return failure


def exact_binary_conditional_coverage_wasserstein(
    calibration_count: int,
    order: int,
    correlation: float,
) -> tuple[float, float]:
    """W1 distance to the iid beta law and deviation from its mean.

    ``exact_binary_training_failure(..., target, ...)`` is the CDF of the
    random training-conditional coverage.  In one dimension W1 is the
    integral of the absolute CDF difference.  The second return value is
    E|C-k/(N+1)|, recovered from the same exact CDF without Monte Carlo.
    """
    nominal = order / (calibration_count + 1.0)
    alpha = order
    beta = calibration_count + 1 - order
    breakpoints = sorted({
        (1.0 - correlation) / 2.0,
        (1.0 + correlation) / 2.0,
        nominal,
    })

    def dependent_cdf(value: float) -> float:
        return exact_binary_training_failure(
            calibration_count, order, value, correlation
        )

    wasserstein = quad(
        lambda value: abs(
            dependent_cdf(value) - integer_beta_cdf(value, alpha, beta)
        ),
        0.0,
        1.0,
        points=breakpoints,
        epsabs=2e-12,
        epsrel=2e-11,
        limit=150,
    )[0]
    mean_absolute_deviation = (
        quad(
            dependent_cdf,
            0.0,
            nominal,
            points=[point for point in breakpoints if point < nominal],
            epsabs=2e-12,
            epsrel=2e-11,
            limit=150,
        )[0]
        + quad(
            lambda value: 1.0 - dependent_cdf(value),
            nominal,
            1.0,
            points=[point for point in breakpoints if point > nominal],
            epsabs=2e-12,
            epsrel=2e-11,
            limit=150,
        )[0]
    )
    return float(wasserstein), float(mean_absolute_deviation)


def exact_partition_training_failure_irregular(
    calibration_count: int,
    order: int,
    target: float,
    transition: np.ndarray,
    strides: tuple[int, ...],
) -> float:
    """Exact irregular-grid failure for an unequal stationary partition.

    State zero has stationary mass ``r`` and emits a uniform score on
    ``[0,r]``; state one emits uniformly on ``[r,1]``.  The stationary score
    is therefore uniform even when the two state masses are unequal.  Direct
    state-path enumeration is independent of the iid-panel mixture below.
    """
    import itertools

    if not (1 <= order <= calibration_count):
        raise ValueError("order must be between one and calibration_count")
    if len(strides) != calibration_count or min(strides) < 1:
        raise ValueError("strides must contain calibration_count positive gaps")
    transition = np.asarray(transition, dtype=float)
    if transition.shape != (2, 2):
        raise ValueError("transition must be two by two")
    if np.min(transition) < 0 or not np.allclose(transition.sum(axis=1), 1.0):
        raise ValueError("transition must be row stochastic")
    lower_mass = transition[1, 0] / (transition[0, 1] + transition[1, 0])
    stationary = np.array([lower_mass, 1.0 - lower_mass])
    sampled_transitions = [
        np.linalg.matrix_power(transition, stride) for stride in strides
    ]
    failure = 0.0
    for states in itertools.product((0, 1), repeat=calibration_count):
        path_probability = stationary[states[0]]
        for index, (previous, current) in enumerate(
            zip(states[:-1], states[1:])
        ):
            path_probability *= sampled_transitions[index][previous, current]
        probability_lower = sampled_transitions[-1][states[-1], 0]
        probability_upper = 1.0 - probability_lower
        lower_count = states.count(0)
        if lower_count >= order:
            failure += path_probability * integer_beta_cdf(
                target / probability_lower,
                order,
                lower_count + 1 - order,
            )
        else:
            failure += path_probability * integer_beta_cdf(
                (target - probability_lower) / probability_upper,
                order - lower_count,
                calibration_count + 1 - order,
            )
    return failure


def iid_panel_fixed_test_failure_limit(
    calibration_count: int,
    order: int,
    target: float,
    test_correlation: float,
    lower_mass: float = 0.5,
) -> float:
    """Failure limit for any order statistic with fixed test memory.

    Calibration scores are iid U(0,1), but the partition ``[0,r]`` or
    ``[r,1]`` containing the last score reveals its binary state, where
    ``r=lower_mass``.  Conditional on that state, the next state retains the
    nontrivial transition eigenvalue ``test_correlation``.  This is the limit of
    :func:`exact_binary_training_failure_irregular` when all within-panel
    correlations vanish while the final calibration--test correlation stays
    fixed.

    If ``j`` of the first ``N-1`` scores are in the lower interval and the
    last state is ``y``, the total lower count is ``z=j+1-y``.  For ``z>=k``,
    the kth order statistic divided by ``r`` is Beta(k,z+1-k); otherwise its
    rescaled excess above ``r`` is Beta(k-z,N+1-k).  The binomial-beta
    mixture evaluates the complete training-conditional failure law without
    simulation or path enumeration.
    """
    if not (1 <= order <= calibration_count):
        raise ValueError("order must be between one and calibration_count")
    if not (0.0 < target < 1.0):
        raise ValueError("target must lie strictly between zero and one")
    if not (0.0 <= test_correlation < 1.0):
        raise ValueError("test_correlation must lie in [0,1)")
    if not (0.0 < lower_mass < 1.0):
        raise ValueError("lower_mass must lie strictly between zero and one")

    failure = 0.0
    for last_state in (0, 1):
        last_is_lower = 1 - last_state
        probability_zero = lower_mass + test_correlation * (
            last_is_lower - lower_mass
        )
        probability_one = 1.0 - probability_zero
        last_weight = lower_mass if last_is_lower else 1.0 - lower_mass
        for first_panel_zeros in range(calibration_count):
            zero_count = first_panel_zeros + last_is_lower
            weight = (
                last_weight
                * math.comb(calibration_count - 1, first_panel_zeros)
                * lower_mass**first_panel_zeros
                * (1.0 - lower_mass) ** (
                    calibration_count - 1 - first_panel_zeros
                )
            )
            if zero_count >= order:
                failure += weight * integer_beta_cdf(
                    target / probability_zero,
                    order,
                    zero_count + 1 - order,
                )
            else:
                failure += weight * integer_beta_cdf(
                    (target - probability_zero) / probability_one,
                    order - zero_count,
                    calibration_count + 1 - order,
                )
    return failure


def iid_panel_fixed_test_mean_coverage(
    calibration_count: int,
    order: int,
    test_correlation: float,
    lower_mass: float = 0.5,
) -> float:
    """Mean coverage in the iid-panel, fixed-test-memory limit."""
    if not (1 <= order <= calibration_count):
        raise ValueError("order must be between one and calibration_count")
    if not (0.0 <= test_correlation < 1.0):
        raise ValueError("test_correlation must lie in [0,1)")
    if not (0.0 < lower_mass < 1.0):
        raise ValueError("lower_mass must lie strictly between zero and one")

    coverage = 0.0
    for last_state in (0, 1):
        last_is_lower = 1 - last_state
        probability_zero = lower_mass + test_correlation * (
            last_is_lower - lower_mass
        )
        probability_one = 1.0 - probability_zero
        last_weight = lower_mass if last_is_lower else 1.0 - lower_mass
        for first_panel_zeros in range(calibration_count):
            zero_count = first_panel_zeros + last_is_lower
            weight = (
                last_weight
                * math.comb(calibration_count - 1, first_panel_zeros)
                * lower_mass**first_panel_zeros
                * (1.0 - lower_mass) ** (
                    calibration_count - 1 - first_panel_zeros
                )
            )
            if zero_count >= order:
                coverage += (
                    weight
                    * probability_zero
                    * order
                    / (zero_count + 1)
                )
            else:
                coverage += weight * (
                    probability_zero
                    + probability_one
                    * (order - zero_count)
                    / (calibration_count - zero_count + 1)
                )
    return coverage


def iid_panel_fixed_test_marginal_shift(
    calibration_count: int,
    order: int,
    lower_mass: float = 0.5,
) -> float:
    """Coefficient of final-memory correlation in marginal coverage.

    For an arbitrary lower-state mass ``r``, if ``Z`` is Binomial(N,r),
    the coefficient is

    E[(Z/N-r) d_k(Z)],

    where ``d_k(z)=k/(z+1)`` for ``z>=k`` and
    ``d_k(z)=(N+1-k)/(N+1-z)`` otherwise.  At ``r=1/2`` this reduces to

    ((N-k+1) P(B <= k-1) - k P(B >= k+1)) / (N(N+1)).

    This is obtained independently by conditioning on the iid kth order
    statistic and integrating the last score's lower/upper-half sign.
    """
    if not (1 <= order <= calibration_count):
        raise ValueError("order must be between one and calibration_count")
    if not (0.0 < lower_mass < 1.0):
        raise ValueError("lower_mass must lie strictly between zero and one")
    if lower_mass != 0.5:
        counts = np.arange(calibration_count + 1)
        weights = binomial_distribution.pmf(
            counts, calibration_count, lower_mass
        )
        conditional_profile = np.where(
            counts >= order,
            order / (counts + 1.0),
            (calibration_count + 1 - order)
            / (calibration_count + 1.0 - counts),
        )
        return float(np.sum(
            weights
            * (counts / calibration_count - lower_mass)
            * conditional_profile
        ))
    trials = calibration_count + 1
    denominator = 2.0**trials
    lower_tail = sum(
        math.comb(trials, count) for count in range(order)
    ) / denominator
    upper_tail = sum(
        math.comb(trials, count) for count in range(order + 1, trials + 1)
    ) / denominator
    return (
        (calibration_count - order + 1) * lower_tail
        - order * upper_tail
    ) / (calibration_count * trials)


def binary_relative_entropy_from_half(probability: float) -> float:
    """Bernoulli relative entropy D(probability || 1/2)."""
    if not (0.0 < probability < 1.0):
        raise ValueError("probability must lie strictly between zero and one")
    return (
        probability * math.log(2.0 * probability)
        + (1.0 - probability) * math.log(2.0 * (1.0 - probability))
    )


def iid_panel_fixed_test_crossover_bound(
    calibration_count: int,
    order: int,
) -> float:
    """Berry--Esseen error bound for the central marginal-shift scaling."""
    if not (1 <= order <= calibration_count):
        raise ValueError("order must be between one and calibration_count")
    trials = calibration_count + 1
    root_trials = math.sqrt(trials)
    standardized_rank = (2 * order - trials) / root_trials
    order_fraction = order / trials
    return (
        trials
        / calibration_count
        * (
            BERRY_ESSEEN_CONSTANT / root_trials
            + (1.0 - order_fraction)
            * math.sqrt(2.0 / math.pi)
            / root_trials
            + abs(standardized_rank) / (2.0 * root_trials)
        )
        + 1.0 / (2.0 * calibration_count)
    )


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


def sampled_path_total_variation_irregular(
    transition: np.ndarray,
    stationary: np.ndarray,
    strides: tuple[int, ...],
) -> float:
    """Brute-force TV from an irregular Markov sample to iid draws."""
    import itertools

    if not strides or min(strides) < 1:
        raise ValueError("strides must be nonempty and positive")
    sampled = [
        np.linalg.matrix_power(transition, stride) for stride in strides
    ]
    state_count = len(stationary)
    distance = 0.0
    for path in itertools.product(range(state_count), repeat=len(strides) + 1):
        joint = stationary[path[0]]
        independent = stationary[path[0]]
        for index, (previous, current) in enumerate(
            zip(path[:-1], path[1:])
        ):
            joint *= sampled[index][previous, current]
            independent *= stationary[current]
        distance += abs(joint - independent)
    return 0.5 * distance


def training_future_total_variation(
    transition: np.ndarray,
    stationary: np.ndarray,
    training_gap: int,
    future_strides: tuple[int, ...],
) -> float:
    """TV cost of replacing a future Markov panel after training by a copy.

    The first coordinate is the state available to training at time zero.
    The future panel begins ``training_gap`` transitions later and then uses
    ``future_strides``.  The comparison law preserves the complete Markov law
    within that future panel, changing only its dependence on the training
    state.  Because the first future state is retained, this TV distance is
    exactly ``markov_beta(..., training_gap)``.
    """
    import itertools

    if training_gap < 1 or min(future_strides, default=1) < 1:
        raise ValueError("training gap and future strides must be positive")
    training_transition = np.linalg.matrix_power(transition, training_gap)
    panel_transitions = [
        np.linalg.matrix_power(transition, stride)
        for stride in future_strides
    ]
    state_count = len(stationary)
    distance = 0.0
    for path in itertools.product(
        range(state_count), repeat=len(future_strides) + 2
    ):
        joint = (
            stationary[path[0]]
            * training_transition[path[0], path[1]]
        )
        decoupled = stationary[path[0]] * stationary[path[1]]
        for index, (previous, current) in enumerate(
            zip(path[1:-1], path[2:])
        ):
            probability = panel_transitions[index][previous, current]
            joint *= probability
            decoupled *= probability
        distance += abs(joint - decoupled)
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


def exact_irregular_window_coverage(
    calibration_count: int,
    order: int,
    window_length: int,
    correlation: float,
    strides: tuple[int, ...],
    signal: float = 1.0,
    noise: float = 0.5,
    quadrature_order: int = 80,
) -> float:
    """Exact marginal coverage for irregularly spaced sliding-block scores.

    ``strides[j]`` is the distance between the start of block ``j`` and the
    start of block ``j+1``.  The tuple therefore includes the final distance
    from the last calibration block to the test block.
    """
    if not (1 <= order <= calibration_count):
        raise ValueError("order must be between one and calibration_count")
    if len(strides) != calibration_count or min(strides) < 1:
        raise ValueError("strides must contain calibration_count positive gaps")

    blocks, stationary, transition = sliding_block_model(
        window_length, correlation
    )
    sampled_transitions = [
        np.linalg.matrix_power(transition, stride) for stride in strides
    ]
    means = signal * blocks.mean(axis=1)
    nodes, weights = roots_hermite(quadrature_order)
    coverage = 0.0

    for test_index, test_mean in enumerate(means):
        for node, weight in zip(nodes, weights):
            test_score = test_mean + noise * math.sqrt(2.0) * node
            probabilities = ndtr((test_score - means) / noise)
            distribution = np.zeros((calibration_count + 1, len(blocks)))
            distribution[0] = stationary
            for sampled_transition in sampled_transitions:
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


def nonlinear_path_map(average: np.ndarray) -> np.ndarray:
    """Smooth nonlinear path functional used by the exact certificate."""
    return 0.4 * average + 0.6 * average**2


def nonlinear_conditional_cdfs(q: float, c: float) -> tuple[float, float]:
    """Exact CDFs for MU * nonlinear_path_map(A_c) + SIGMA * Z."""
    values = conditional_values(
        c,
        lambda average: ndtr(
            (q - MU * nonlinear_path_map(average)) / SIGMA
        ),
    )
    return float(values[0]), float(values[1])


def nonlinear_threshold(c: float) -> float:
    """Stationary pooled threshold for the nonlinear path score."""
    return brentq(
        lambda q: 0.5 * sum(nonlinear_conditional_cdfs(q, c)) - TARGET,
        -4.0,
        4.0,
        xtol=2e-13,
    )


def asymmetric_occupation_density(
    fraction_plus: float,
    c: float,
    rate_plus_to_minus: float,
    rate_minus_to_plus: float,
    start: int,
) -> float:
    """Continuous occupation-fraction density for an unequal two-state CTMC.

    ``start`` is zero for the plus state and one for the minus state.  The
    endpoint atoms are handled separately by
    :func:`asymmetric_occupation_expectation`.  Exponentially scaled Bessel
    functions keep the formula stable for the long horizons used below.
    """
    u = float(fraction_plus)
    alpha, beta = rate_plus_to_minus, rate_minus_to_plus
    if not 0.0 < u < 1.0:
        return 0.0
    root = math.sqrt(alpha * beta * u * (1.0 - u))
    argument = 2.0 * c * root
    exponent = -c * (
        alpha * u + beta * (1.0 - u) - 2.0 * root
    )
    common = c * math.exp(exponent)
    if start == 0:
        return common * (
            alpha * ive(0, argument)
            + math.sqrt(alpha * beta * u / (1.0 - u))
            * ive(1, argument)
        )
    if start == 1:
        return common * (
            beta * ive(0, argument)
            + math.sqrt(alpha * beta * (1.0 - u) / u)
            * ive(1, argument)
        )
    raise ValueError("start must be zero (plus) or one (minus)")


def asymmetric_occupation_expectation(
    function,
    c: float,
    rate_plus_to_minus: float,
    rate_minus_to_plus: float,
    start: int,
) -> float:
    """Exact expectation of a function of the plus occupation fraction."""
    alpha, beta = rate_plus_to_minus, rate_minus_to_plus
    if min(c, alpha, beta) <= 0.0:
        raise ValueError("c and both transition rates must be positive")
    if start == 0:
        atom = math.exp(-alpha * c) * function(1.0)
    elif start == 1:
        atom = math.exp(-beta * c) * function(0.0)
    else:
        raise ValueError("start must be zero (plus) or one (minus)")
    stationary_plus = beta / (alpha + beta)
    continuous = quad(
        lambda u: function(u)
        * asymmetric_occupation_density(u, c, alpha, beta, start),
        0.0,
        1.0,
        points=[stationary_plus],
        epsabs=2e-11,
        epsrel=2e-11,
        limit=300,
    )[0]
    return float(atom + continuous)


def critical_quadratic_cdfs(
    q: float,
    c: float,
    rate_plus_to_minus: float,
    rate_minus_to_plus: float,
    signal: float = MU,
    noise: float = SIGMA,
) -> np.ndarray:
    """Exact CDFs when psi(A)=(A-E_pi A)^2 has zero gradient."""
    alpha, beta = rate_plus_to_minus, rate_minus_to_plus
    stationary_mean = (beta - alpha) / (alpha + beta)
    answer = []
    for start in (0, 1):
        answer.append(
            asymmetric_occupation_expectation(
                lambda u: ndtr(
                    (
                        q
                        - signal * ((2.0 * u - 1.0) - stationary_mean) ** 2
                    )
                    / noise
                ),
                c,
                alpha,
                beta,
                start,
            )
        )
    return np.asarray(answer)


def critical_quadratic_threshold(
    c: float,
    rate_plus_to_minus: float,
    rate_minus_to_plus: float,
    signal: float = MU,
    noise: float = SIGMA,
) -> float:
    """Stationary pooled quantile for the critical quadratic score."""
    stationary = np.array([
        rate_minus_to_plus,
        rate_plus_to_minus,
    ]) / (rate_plus_to_minus + rate_minus_to_plus)
    return brentq(
        lambda q: stationary
        @ critical_quadratic_cdfs(
            q,
            c,
            rate_plus_to_minus,
            rate_minus_to_plus,
            signal,
            noise,
        )
        - TARGET,
        -4.0 * noise,
        4.0 * (abs(signal) + noise),
        xtol=2e-13,
    )


def critical_score_coefficients(
    generator: np.ndarray,
    state_observables: np.ndarray,
    hessian: np.ndarray,
    third_derivative: np.ndarray,
    signal: float = MU,
    noise: float = SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Second-order fixed-start coefficients at a critical score point.

    The gradient of the score map at the stationary mean is assumed to be
    zero.  Returns the coverage coefficients, boundary-covariance tensors,
    Green--Kubo covariance, and Poisson corrector.
    """
    generator = np.asarray(generator, dtype=float)
    state_observables = np.atleast_2d(state_observables).astype(float)
    if state_observables.shape[0] != generator.shape[0]:
        state_observables = state_observables.T
    stationary = generator_stationary_distribution(generator)
    projection = np.outer(np.ones(len(stationary)), stationary)
    group_inverse = np.linalg.inv(generator - projection) + projection
    centered = state_observables - np.outer(
        np.ones(len(stationary)), stationary @ state_observables
    )
    corrector = group_inverse @ centered
    weighted = stationary[:, None] * centered
    covariance_rate = -(
        centered.T @ (stationary[:, None] * corrector)
        + corrector.T @ weighted
    )
    dimension = centered.shape[1]
    boundary_covariance = np.empty(
        (len(stationary), dimension, dimension)
    )
    for left in range(dimension):
        for right in range(dimension):
            forcing = (
                centered[:, left] * corrector[:, right]
                + centered[:, right] * corrector[:, left]
            )
            boundary_covariance[:, left, right] = group_inverse @ forcing
    curvature = np.einsum(
        "ab,iab->i", np.asarray(hessian), boundary_covariance
    )
    cubic = np.einsum(
        "abc,ia,bc->i",
        np.asarray(third_derivative),
        corrector,
        covariance_rate,
    )
    coefficients = -(
        signal * norm.pdf(norm.ppf(TARGET)) / (2.0 * noise)
    ) * (curvature - cubic)
    assert np.max(
        np.abs(np.einsum("i,iab->ab", stationary, boundary_covariance))
    ) < 2e-14
    assert np.max(np.abs(stationary @ coefficients)) < 2e-14
    return (
        coefficients,
        boundary_covariance,
        covariance_rate,
        corrector,
    )


def asymmetric_path_cdfs(
    q: float,
    c: float,
    rate_plus_to_minus: float,
    rate_minus_to_plus: float,
    signal: float = MU,
    noise: float = SIGMA,
    quadrature_order: int = 240,
) -> np.ndarray:
    """Exact unequal-rate path-score CDFs by matrix Feynman--Kac inversion.

    Time is measured in units for which the generator is
    [[-alpha, alpha], [beta, -beta]], and ``c`` is the elapsed time.  The
    additive score uses the average of the state values +1 and -1.
    """
    alpha, beta = rate_plus_to_minus, rate_minus_to_plus
    if min(c, alpha, beta, noise) <= 0:
        raise ValueError("c, transition rates, and noise must be positive")
    nodes, weights = roots_legendre(quadrature_order)
    cutoff = 14.0 / noise
    frequencies = (nodes + 1.0) * cutoff / 2.0
    weights = weights * cutoff / 2.0
    generator = np.array([[-alpha, alpha], [beta, -beta]])
    state_values = np.diag([1.0, -1.0])
    one = np.ones(2)
    transforms = np.stack(
        [
            expm(c * generator + 1j * frequency * signal * state_values)
            @ one
            for frequency in frequencies
        ],
        axis=1,
    )
    characteristics = transforms * np.exp(
        -0.5 * noise**2 * frequencies**2
    )
    inversion = np.imag(
        np.exp(-1j * frequencies * q)[None, :] * characteristics
    ) / frequencies
    return 0.5 - inversion @ weights / math.pi


def asymmetric_path_threshold(
    c: float,
    rate_plus_to_minus: float,
    rate_minus_to_plus: float,
    signal: float = MU,
    noise: float = SIGMA,
) -> float:
    """Stationary pooled threshold for the unequal-rate path score."""
    alpha, beta = rate_plus_to_minus, rate_minus_to_plus
    stationary = np.array([beta, alpha]) / (alpha + beta)
    return brentq(
        lambda q: stationary
        @ asymmetric_path_cdfs(q, c, alpha, beta, signal, noise)
        - TARGET,
        -4.0,
        4.0,
        xtol=2e-13,
    )


def generator_stationary_distribution(generator: np.ndarray) -> np.ndarray:
    """Stationary row distribution of a finite irreducible generator."""
    generator = np.asarray(generator, dtype=float)
    states = generator.shape[0]
    if generator.shape != (states, states):
        raise ValueError("generator must be square")
    if np.max(np.abs(generator @ np.ones(states))) > 2e-13:
        raise ValueError("generator rows must sum to zero")
    if np.min(generator - np.diag(np.diag(generator))) < -2e-13:
        raise ValueError("generator off-diagonal entries must be nonnegative")
    system = generator.T.copy()
    system[-1] = 1.0
    target = np.zeros(states)
    target[-1] = 1.0
    stationary = np.linalg.solve(system, target)
    if np.min(stationary) <= 0:
        raise ValueError("generator must have a strictly positive stationary law")
    return stationary


def finite_chain_path_cdfs(
    q: float,
    c: float,
    generator: np.ndarray,
    state_values: np.ndarray,
    signal: float = MU,
    noise: float = SIGMA,
    quadrature_order: int = 300,
) -> np.ndarray:
    """Exact finite-chain path-score CDFs by Feynman--Kac inversion."""
    generator = np.asarray(generator, dtype=float)
    state_values = np.asarray(state_values, dtype=float)
    states = len(state_values)
    if generator.shape != (states, states):
        raise ValueError("generator and state_values have incompatible sizes")
    if min(c, noise) <= 0:
        raise ValueError("c and noise must be positive")
    generator_stationary_distribution(generator)
    nodes, weights = roots_legendre(quadrature_order)
    cutoff = 14.0 / noise
    frequencies = (nodes + 1.0) * cutoff / 2.0
    weights = weights * cutoff / 2.0
    observable = np.diag(state_values)
    one = np.ones(states)
    transforms = np.stack(
        [
            expm(c * generator + 1j * frequency * signal * observable) @ one
            for frequency in frequencies
        ],
        axis=1,
    )
    characteristics = transforms * np.exp(-0.5 * noise**2 * frequencies**2)
    inversion = np.imag(
        np.exp(-1j * frequencies * q)[None, :] * characteristics
    ) / frequencies
    return 0.5 - inversion @ weights / math.pi


def finite_chain_path_threshold(
    c: float,
    generator: np.ndarray,
    state_values: np.ndarray,
    signal: float = MU,
    noise: float = SIGMA,
) -> float:
    """Stationary pooled threshold for a finite-chain path score."""
    stationary = generator_stationary_distribution(generator)
    radius = abs(signal) * np.max(np.abs(state_values)) + 10.0 * noise
    return brentq(
        lambda q: stationary
        @ finite_chain_path_cdfs(
            q, c, generator, state_values, signal, noise
        )
        - TARGET,
        -radius,
        radius,
        xtol=2e-13,
    )


def finite_chain_path_coefficients(
    generator: np.ndarray,
    state_values: np.ndarray,
    signal: float = MU,
    noise: float = SIGMA,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """First two Poisson-equation coefficients of the coverage gaps.

    Returns the first- and second-order coefficient vectors, stationary law,
    group inverse Q#, and asymptotic variance rate.  The convention is
    Q Q# = Q# Q = I - 1 pi.
    """
    generator = np.asarray(generator, dtype=float)
    state_values = np.asarray(state_values, dtype=float)
    stationary = generator_stationary_distribution(generator)
    one = np.ones(len(stationary))
    projection = np.outer(one, stationary)
    group_inverse = np.linalg.inv(generator - projection) + projection
    centered = state_values - stationary @ state_values
    h = group_inverse @ centered
    variance_rate = -2.0 * stationary @ (centered * h)
    second_geometry = group_inverse @ (centered * h)
    standardized_signal = signal / noise
    z = norm.ppf(TARGET)
    density = norm.pdf(z)
    first_coefficients = standardized_signal * density * h
    second_coefficients = -density * (
        standardized_signal**2 * z * second_geometry
        + 0.5 * standardized_signal**3 * variance_rate * h
    )
    assert variance_rate >= -1e-14
    assert abs(stationary @ first_coefficients) < 2e-15
    assert abs(stationary @ second_coefficients) < 2e-15
    return (
        first_coefficients,
        second_coefficients,
        stationary,
        group_inverse,
        variance_rate,
    )


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

    # A smooth nonlinear function of the same occupation average has the
    # identical start-memory coefficient multiplied by its derivative at the
    # stationary mean.  The quadratic term changes the pooled threshold at
    # order 1/c, but that common shift cancels from the leading conditional
    # coverage gap.
    nonlinear_scales = np.array([16.0, 32.0, 64.0, 128.0])
    nonlinear_gaps = []
    nonlinear_thresholds = []
    for c in nonlinear_scales:
        q = nonlinear_threshold(c)
        plus, minus = nonlinear_conditional_cdfs(q, c)
        assert abs(0.5 * (plus + minus) - TARGET) < 3e-12
        nonlinear_thresholds.append(q)
        nonlinear_gaps.append(plus - TARGET)
    nonlinear_gaps = np.asarray(nonlinear_gaps)
    nonlinear_limit = 0.4 * limit
    nonlinear_threshold_coefficient = (
        MU * 1.2 / 2.0
        + MU**2 * 0.4**2 * norm.ppf(TARGET) / (2.0 * SIGMA)
    )
    nonlinear_residuals = np.abs(
        nonlinear_gaps - nonlinear_limit / nonlinear_scales
    )
    nonlinear_rates = np.log2(
        nonlinear_residuals[:-1] / nonlinear_residuals[1:]
    )
    assert nonlinear_rates[-1] > 1.9
    assert abs(
        nonlinear_scales[-1] * nonlinear_gaps[-1] - nonlinear_limit
    ) < 0.002
    nonlinear_scaled_threshold = nonlinear_scales[-1] * (
        nonlinear_thresholds[-1] - SIGMA * norm.ppf(TARGET)
    )
    assert abs(
        nonlinear_scaled_threshold - nonlinear_threshold_coefficient
    ) < 0.008
    print("\nSmooth nonlinear path-score crossover")
    print("psi(a)=0.4a+0.6a^2; psi'(0)=0.4")
    print(" c       q_pool       c*(gap +)")
    for c, q, gap in zip(
        nonlinear_scales, nonlinear_thresholds, nonlinear_gaps
    ):
        print(f"{c:4.0f}   {q:11.8f}   {c * gap:12.8f}")
    print(
        f"predicted limit {nonlinear_limit:.8f}; "
        f"last post-leading residual rate {nonlinear_rates[-1]:.6f}; "
        f"scaled threshold shift {nonlinear_scaled_threshold:.8f} "
        f"versus {nonlinear_threshold_coefficient:.8f}"
    )

    # At a critical point of the score map the gradient coefficient vanishes.
    # The first nonzero fixed-start discrepancy is then order c^-2.  An
    # unequal-rate chain is essential here: the symmetric quadratic example
    # would make the two conditional laws identical and conceal the boundary
    # covariance.  The exact Bessel occupation law independently checks the
    # Poisson-equation coefficient.
    critical_alpha, critical_beta = 1.7, 0.4
    critical_generator = np.array(
        [
            [-critical_alpha, critical_alpha],
            [critical_beta, -critical_beta],
        ]
    )
    critical_values = np.array([1.0, -1.0])
    critical_stationary = np.array(
        [critical_beta, critical_alpha]
    ) / (critical_alpha + critical_beta)
    (
        critical_coefficients,
        critical_boundary_covariance,
        critical_covariance_rate,
        critical_corrector,
    ) = critical_score_coefficients(
        critical_generator,
        critical_values,
        np.array([[2.0]]),
        np.zeros((1, 1, 1)),
    )
    critical_scales = np.array([32.0, 64.0, 128.0, 256.0, 512.0])
    critical_thresholds = []
    critical_gaps = []
    critical_mass_errors = []
    for c in critical_scales:
        q = critical_quadratic_threshold(
            c, critical_alpha, critical_beta
        )
        coverages = critical_quadratic_cdfs(
            q, c, critical_alpha, critical_beta
        )
        assert abs(critical_stationary @ coverages - TARGET) < 4e-11
        critical_thresholds.append(q)
        critical_gaps.append(coverages - TARGET)
        critical_mass_errors.append(
            max(
                abs(
                    asymmetric_occupation_expectation(
                        lambda _: 1.0,
                        c,
                        critical_alpha,
                        critical_beta,
                        start,
                    )
                    - 1.0
                )
                for start in (0, 1)
            )
        )
    critical_thresholds = np.asarray(critical_thresholds)
    critical_gaps = np.asarray(critical_gaps)
    critical_residuals = np.abs(
        critical_gaps
        - critical_coefficients[None, :] / critical_scales[:, None] ** 2
    )
    critical_rates = np.log2(
        critical_residuals[:-1] / critical_residuals[1:]
    )
    assert np.min(critical_rates[-1]) > 2.8
    assert abs(critical_stationary @ critical_coefficients) < 2e-15
    assert max(critical_mass_errors) < 2e-10
    critical_threshold_scaled = critical_scales[-1] * (
        critical_thresholds[-1] - SIGMA * norm.ppf(TARGET)
    )
    critical_threshold_limit = MU * critical_covariance_rate[0, 0]
    assert abs(
        critical_threshold_scaled - critical_threshold_limit
    ) < 0.002
    print("\nCritical nonlinear path-score crossover")
    print("psi(a)=(a-E_pi A)^2; psi'(E_pi A)=0")
    print(" c       q_pool       c^2*gap +      c^2*gap -")
    for c, q, gap in zip(
        critical_scales, critical_thresholds, critical_gaps
    ):
        print(
            f"{c:4.0f}   {q:11.8f}   {c**2 * gap[0]:12.8f}"
            f"   {c**2 * gap[1]:12.8f}"
        )
    print(
        "predicted c^2 limits: "
        f"{critical_coefficients[0]:.8f}, "
        f"{critical_coefficients[1]:.8f}; "
        f"last residual rates {critical_rates[-1, 0]:.6f}, "
        f"{critical_rates[-1, 1]:.6f}"
    )
    print(
        "boundary covariance: "
        f"{critical_boundary_covariance[:, 0, 0]}; "
        "corrector: "
        f"{critical_corrector[:, 0]}; "
        "Green--Kubo variance: "
        f"{critical_covariance_rate[0, 0]:.8f}"
    )
    print(
        f"scaled threshold shift {critical_threshold_scaled:.8f} "
        f"versus {critical_threshold_limit:.8f}; "
        f"maximum occupation-mass error {max(critical_mass_errors):.3e}"
    )

    # Unequal rates destroy the equal-and-opposite symmetry.  The conditional
    # errors instead cancel with the stationary weights.  Matrix
    # Feynman--Kac inversion is independent of the Poisson--Beta formula above.
    alpha, beta = 1.7, 0.4
    rho = alpha + beta
    stationary = np.array([beta, alpha]) / rho
    state_values = np.array([1.0, -1.0])
    stationary_mean = float(stationary @ state_values)
    unequal_coefficients = (
        -MU
        / SIGMA
        * norm.pdf(norm.ppf(TARGET))
        * (state_values - stationary_mean)
        / rho
    )
    unequal_scales = np.array([32.0, 64.0, 128.0, 256.0, 512.0])
    unequal_gaps = []
    unequal_thresholds = []
    for c in unequal_scales:
        q = asymmetric_path_threshold(c, alpha, beta)
        coverages = asymmetric_path_cdfs(q, c, alpha, beta)
        assert abs(stationary @ coverages - TARGET) < 4e-12
        unequal_thresholds.append(q)
        unequal_gaps.append(coverages - TARGET)
    unequal_gaps = np.asarray(unequal_gaps)
    unequal_residuals = np.abs(
        unequal_gaps - unequal_coefficients[None, :] / unequal_scales[:, None]
    )
    unequal_rates = np.log2(
        unequal_residuals[:-1] / unequal_residuals[1:]
    )
    assert np.min(unequal_rates[-1]) > 1.95
    assert abs(stationary @ unequal_coefficients) < 2e-16
    assert abs(
        unequal_coefficients[0] / unequal_coefficients[1] + alpha / beta
    ) < 2e-14
    print("\nUnequal-rate path-score crossover")
    print(
        "rates alpha=1.7, beta=0.4; stationary mean "
        f"{stationary_mean:.8f}"
    )
    print(" c       q_pool        c*gap +        c*gap -")
    for c, q, gap in zip(unequal_scales, unequal_thresholds, unequal_gaps):
        print(
            f"{c:4.0f}   {q:11.8f}   {c * gap[0]:12.8f}"
            f"   {c * gap[1]:12.8f}"
        )
    print(
        "predicted limits: "
        f"{unequal_coefficients[0]:.8f}, "
        f"{unequal_coefficients[1]:.8f}; "
        f"last residual rates {unequal_rates[-1, 0]:.6f}, "
        f"{unequal_rates[-1, 1]:.6f}"
    )

    # The same leading coefficient is a Poisson-equation object for every
    # finite irreducible chain.  This deliberately nonreversible example
    # checks all three state-specific coefficients by an exact matrix
    # Feynman--Kac calculation and Fourier inversion.
    finite_generator = np.array(
        [
            [-1.1, 0.8, 0.3],
            [0.4, -1.4, 1.0],
            [0.7, 0.5, -1.2],
        ]
    )
    finite_values = np.array([-1.0, 0.35, 1.4])
    (
        finite_coefficients,
        finite_second_coefficients,
        finite_stationary,
        finite_group_inverse,
        finite_variance_rate,
    ) = (
        finite_chain_path_coefficients(finite_generator, finite_values)
    )
    projection = np.outer(np.ones(3), finite_stationary)
    assert np.max(
        np.abs(finite_generator @ finite_group_inverse - (np.eye(3) - projection))
    ) < 2e-15
    assert np.max(
        np.abs(finite_group_inverse @ finite_generator - (np.eye(3) - projection))
    ) < 2e-15
    assert abs(finite_stationary @ finite_coefficients) < 2e-15
    assert abs(finite_stationary @ finite_second_coefficients) < 2e-15
    stationary_flux = finite_stationary[:, None] * finite_generator
    assert np.max(np.abs(stationary_flux - stationary_flux.T)) > 0.1

    # The general formula must reduce exactly to the unequal two-state result.
    two_state_coefficients, _, _, _, _ = finite_chain_path_coefficients(
        np.array([[-alpha, alpha], [beta, -beta]]), state_values
    )
    assert np.max(np.abs(two_state_coefficients - unequal_coefficients)) < 3e-15

    finite_scales = np.array([32.0, 64.0, 128.0, 256.0, 512.0, 1024.0])
    finite_thresholds = []
    finite_gaps = []
    for c in finite_scales:
        q = finite_chain_path_threshold(c, finite_generator, finite_values)
        coverages = finite_chain_path_cdfs(
            q, c, finite_generator, finite_values
        )
        assert abs(finite_stationary @ coverages - TARGET) < 5e-12
        finite_thresholds.append(q)
        finite_gaps.append(coverages - TARGET)
    finite_gaps = np.asarray(finite_gaps)
    finite_residuals = np.abs(
        finite_gaps - finite_coefficients[None, :] / finite_scales[:, None]
    )
    finite_rates = np.log2(finite_residuals[:-1] / finite_residuals[1:])
    assert np.min(finite_rates[-1]) > 1.95
    finite_second_residuals = np.abs(
        finite_gaps
        - finite_coefficients[None, :] / finite_scales[:, None]
        - finite_second_coefficients[None, :] / finite_scales[:, None] ** 2
    )
    finite_second_rates = np.log2(
        finite_second_residuals[:-1] / finite_second_residuals[1:]
    )
    assert np.min(finite_second_rates[-1]) > 2.95
    scaled_second_limit = finite_scales[-1] ** 2 * (
        finite_gaps[-1] - finite_coefficients / finite_scales[-1]
    )
    assert np.max(
        np.abs(scaled_second_limit - finite_second_coefficients)
    ) < 0.005
    limiting_threshold = (
        MU * (finite_stationary @ finite_values)
        + SIGMA * norm.ppf(TARGET)
    )
    threshold_coefficient = (
        MU**2
        * finite_variance_rate
        * norm.ppf(TARGET)
        / (2.0 * SIGMA)
    )
    scaled_threshold_shift = finite_scales[-1] * (
        finite_thresholds[-1] - limiting_threshold
    )
    assert abs(scaled_threshold_shift - threshold_coefficient) < 0.005
    print("\nFinite-chain path-score crossover (nonreversible three-state example)")
    print("stationary law: " + ", ".join(f"{x:.8f}" for x in finite_stationary))
    print(" c       q_pool        c*gap 1        c*gap 2        c*gap 3")
    for c, q, gap in zip(finite_scales, finite_thresholds, finite_gaps):
        print(
            f"{c:4.0f}   {q:11.8f}   {c * gap[0]:12.8f}"
            f"   {c * gap[1]:12.8f}   {c * gap[2]:12.8f}"
        )
    print(
        "predicted limits: "
        + ", ".join(f"{x:.8f}" for x in finite_coefficients)
        + "; last residual rates "
        + ", ".join(f"{x:.6f}" for x in finite_rates[-1])
    )
    print(
        "variance rate: "
        f"{finite_variance_rate:.8f}; predicted second coefficients: "
        + ", ".join(f"{x:.8f}" for x in finite_second_coefficients)
    )
    print(
        "scaled pooled-threshold shift at c=1024: "
        f"{scaled_threshold_shift:.8f}; predicted {threshold_coefficient:.8f}"
    )
    print(
        "scaled second-order residual at c=1024: "
        + ", ".join(f"{x:.8f}" for x in scaled_second_limit)
        + "; last third-order residual rates "
        + ", ".join(f"{x:.6f}" for x in finite_second_rates[-1])
    )

    # A coarsening of the initial state mixes the state-specific gaps.  In
    # three or more states a nonstationary posterior can therefore annihilate
    # the first coefficient without reproducing stationary-start coverage.
    # Mix states 1 and 3 in the unique proportion that kills that coefficient.
    posterior_weight = -finite_coefficients[2] / (
        finite_coefficients[0] - finite_coefficients[2]
    )
    cancelling_posterior = np.array(
        [posterior_weight, 0.0, 1.0 - posterior_weight]
    )
    assert np.max(np.abs(cancelling_posterior - finite_stationary)) > 0.2
    assert abs(cancelling_posterior @ finite_coefficients) < 2e-17
    posterior_second_coefficient = (
        cancelling_posterior @ finite_second_coefficients
    )
    posterior_gaps = finite_gaps @ cancelling_posterior
    posterior_second_residuals = np.abs(
        posterior_gaps
        - posterior_second_coefficient / finite_scales**2
    )
    posterior_third_rates = np.log2(
        posterior_second_residuals[:-1] / posterior_second_residuals[1:]
    )
    assert posterior_second_coefficient < -0.1
    assert posterior_third_rates[-1] > 2.95
    print("\nNonstationary posterior with first-order cancellation")
    print(
        "posterior: "
        + ", ".join(f"{x:.8f}" for x in cancelling_posterior)
        + f"; predicted c^2 limit {posterior_second_coefficient:.8f}"
    )
    print(" c       c^2 mixed gap")
    for c, gap in zip(finite_scales, posterior_gaps):
        print(f"{c:4.0f}   {c**2 * gap:14.8f}")
    print(
        "last residual rate after subtracting c^-2 term: "
        f"{posterior_third_rates[-1]:.6f}"
    )

    # If coefficient vectors c_1,...,c_k have stationary mean zero, the
    # posterior slice annihilating them has affine dimension
    # n-rank[1,c_1,...,c_k].  In four states the first two independent
    # coefficient vectors therefore leave a one-dimensional cancellation
    # family through the stationary posterior.  Choose a full-support
    # nonstationary point on that line and verify the resulting O(c^-3) gap.
    four_generator = np.array(
        [
            [-1.5, 0.7, 0.5, 0.3],
            [0.2, -1.3, 0.6, 0.5],
            [0.4, 0.3, -1.2, 0.5],
            [0.8, 0.2, 0.4, -1.4],
        ]
    )
    four_values = np.array([-1.2, -0.1, 0.7, 1.6])
    (
        four_first,
        four_second,
        four_stationary,
        _,
        _,
    ) = finite_chain_path_coefficients(four_generator, four_values)
    constraints = np.vstack([np.ones(4), four_first, four_second])
    assert np.linalg.matrix_rank(constraints, tol=1e-12) == 3
    _, _, right_vectors = np.linalg.svd(constraints)
    cancellation_direction = right_vectors[-1]
    if cancellation_direction[0] > 0.0:
        cancellation_direction *= -1.0
    negative = cancellation_direction < 0.0
    boundary_step = np.min(
        four_stationary[negative] / -cancellation_direction[negative]
    )
    double_cancelling_posterior = (
        four_stationary + 0.7 * boundary_step * cancellation_direction
    )
    assert np.min(double_cancelling_posterior) > 0.08
    assert abs(np.sum(double_cancelling_posterior) - 1.0) < 2e-15
    assert abs(double_cancelling_posterior @ four_first) < 2e-15
    assert abs(double_cancelling_posterior @ four_second) < 2e-15
    assert np.max(
        np.abs(double_cancelling_posterior - four_stationary)
    ) > 0.2

    four_scales = np.array([32.0, 64.0, 128.0, 256.0, 512.0, 1024.0])
    double_cancellation_gaps = []
    for c in four_scales:
        q = finite_chain_path_threshold(c, four_generator, four_values)
        coverages = finite_chain_path_cdfs(
            q, c, four_generator, four_values
        )
        assert abs(four_stationary @ coverages - TARGET) < 5e-12
        double_cancellation_gaps.append(
            double_cancelling_posterior @ (coverages - TARGET)
        )
    double_cancellation_gaps = np.asarray(double_cancellation_gaps)
    double_cancellation_rates = np.log2(
        np.abs(
            double_cancellation_gaps[:-1]
            / double_cancellation_gaps[1:]
        )
    )
    assert double_cancellation_rates[-1] > 2.98
    print("\nFour-state posterior with first- and second-order cancellation")
    print(
        "stationary: "
        + ", ".join(f"{x:.8f}" for x in four_stationary)
    )
    print(
        "posterior:  "
        + ", ".join(f"{x:.8f}" for x in double_cancelling_posterior)
    )
    print(" c       c^3 mixed gap")
    for c, gap in zip(four_scales, double_cancellation_gaps):
        print(f"{c:4.0f}   {c**3 * gap:14.8f}")
    print(
        "last mixed-gap order after two cancellations: "
        f"{double_cancellation_rates[-1]:.6f}"
    )

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

    # Continuity is unnecessary if ties are resolved by independent uniforms.
    # For an atomic score, H(s,v)=F(s-)+v P(S=s) maps each atom onto its own
    # cumulative-mass interval and the mixture of those intervals is exactly
    # uniform.  Evaluate the resulting kth-order CDF atom by atom for an
    # asymmetric three-atom law and compare with the beta formula.
    atomic_probabilities = (0.17, 0.46, 0.37)
    atomic_grid = np.unique(np.concatenate([
        np.linspace(0.0, 1.0, 101),
        np.cumsum(atomic_probabilities),
    ]))
    atomic_designs = ((9, 9), (9, 5), (20, 7))
    atomic_beta_error = max(
        abs(
            atomic_lex_order_cdf(
                float(value),
                atomic_probabilities,
                atomic_count,
                atomic_order,
            )
            - integer_beta_cdf(
                float(value), atomic_order, atomic_count + 1 - atomic_order
            )
        )
        for atomic_count, atomic_order in atomic_designs
        for value in atomic_grid
    )
    assert atomic_beta_error < 3e-15
    print(
        "randomized three-atom beta-CDF maximum grid error: "
        f"{atomic_beta_error:.3e}"
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

    # Two finite calibration endpoints give a beta *spacing*, not the beta
    # order statistic above.  For lower < upper, the PIT transforms their
    # conditional interval coverage into U_(upper)-U_(lower).  Uniform
    # spacings are Dirichlet, so this difference is Beta(r,N+1-r), where
    # r=upper-lower.  Its law depends only on rank width, not interval
    # location.  Check the law by quadrature and simulation for the sample
    # range of nine observations (lower=1, upper=9, r=8).
    lower_order, upper_order = 1, 9
    rank_width = upper_order - lower_order
    spacing_rng = np.random.default_rng(20261010)
    ordered_uniforms = np.sort(
        spacing_rng.random((400_000, calibration_count)), axis=1
    )
    interval_coverages = (
        ordered_uniforms[:, upper_order - 1]
        - ordered_uniforms[:, lower_order - 1]
    )
    spacing_a = rank_width
    spacing_b = calibration_count + 1 - rank_width
    spacing_mean = rank_width / (calibration_count + 1)
    spacing_variance = (
        spacing_a
        * spacing_b
        / ((calibration_count + 1) ** 2 * (calibration_count + 2))
    )
    spacing_target = 0.7
    spacing_lower_tail = integer_beta_cdf(
        spacing_target, spacing_a, spacing_b
    )
    spacing_log_beta = (
        math.lgamma(spacing_a)
        + math.lgamma(spacing_b)
        - math.lgamma(spacing_a + spacing_b)
    )

    def spacing_density(coverage):
        if coverage <= 0.0 or coverage >= 1.0:
            return 0.0
        return math.exp(
            (spacing_a - 1) * math.log(coverage)
            + (spacing_b - 1) * math.log1p(-coverage)
            - spacing_log_beta
        )

    spacing_quadrature_tail = quad(
        spacing_density,
        0.0,
        spacing_target,
        epsabs=2e-14,
        epsrel=2e-13,
    )[0]
    empirical_spacing_tail = np.mean(interval_coverages < spacing_target)
    assert abs(spacing_quadrature_tail - spacing_lower_tail) < 2e-14
    assert abs(np.mean(interval_coverages) - spacing_mean) < 5e-4
    assert abs(np.var(interval_coverages) - spacing_variance) < 5e-5
    assert abs(empirical_spacing_tail - spacing_lower_tail) < 0.0015

    # The exact PAC rule is the same binomial-tail inversion with rank width
    # in place of the one-sided order.  The finite-endpoint constraint r<=N-1
    # changes feasibility: the sample range succeeds iff
    # p^(N-1) [N-(N-1)p] <= delta.
    two_endpoint_minima = [
        minimum_two_endpoint_panel(target, failure_probability)
        for failure_probability in failure_probabilities
    ]
    assert two_endpoint_minima == [(38, 37), (46, 45), (64, 63)]
    for failure_probability, (count, width) in zip(
        failure_probabilities, two_endpoint_minima
    ):
        range_failure = target ** (count - 1) * (
            count - (count - 1) * target
        )
        assert abs(
            range_failure
            - (1.0 - iid_training_confidence(count, count - 1, target))
        ) < 3e-15
        assert range_failure <= failure_probability
        previous_failure = target ** (count - 2) * (
            count - 1 - (count - 2) * target
        )
        assert previous_failure > failure_probability
        assert width == count - 1

    conventional_width = math.ceil(target * (100 + 1))
    conventional_interval_confidence = iid_training_confidence(
        100, conventional_width, target
    )
    pac_widths = [
        pac_interval_rank_width(100, target, failure_probability)
        for failure_probability in failure_probabilities
    ]
    assert pac_widths == [95, 96, 97]
    assert abs(
        conventional_interval_confidence - 0.5487098345579959
    ) < 2e-14
    print("\nTwo-finite-endpoint iid calibration design")
    print(
        f"N=9 ranks 1,9: law=Beta({spacing_a},{spacing_b}); "
        f"mean={spacing_mean:.8f}; sd={math.sqrt(spacing_variance):.8f}"
    )
    print(
        f"P(conditional interval coverage < {spacing_target:.1f}): "
        f"exact={spacing_lower_tail:.9f}, "
        f"simulation={empirical_spacing_tail:.9f}"
    )
    print("target coverage 0.9")
    print(" delta      minimum N     rank width     marginal coverage")
    for failure_probability, (count, width) in zip(
        failure_probabilities, two_endpoint_minima
    ):
        print(
            f" {failure_probability:5.2f}       {count:5d}"
            f"          {width:5d}          {width / (count + 1):.8f}"
        )
    print(
        "N=100 conventional width=91: training confidence "
        f"{conventional_interval_confidence:.8f}"
    )
    print("N=100 PAC widths for delta=.10,.05,.01: " + str(pac_widths))

    # Transfer the iid beta tail to a dependent training-conditional
    # statement.  The score (Y + U)/2 has an exactly uniform stationary
    # marginal but reveals the binary state.  Its conditional coverage law is
    # therefore non-iid and still admits the exact state-path/beta calculation
    # above.  Separate the calibration-panel distance from iid from the cost
    # of decoupling only the test score from the actual panel.  This uses
    # (N-1) beta + beta/slack rather than coupling all N+1 scores twice.
    transfer_target = 0.8
    transfer_strides = (28, 40, 48)
    nominal_conditional_coverage = order / (calibration_count + 1.0)
    iid_conditional_deviation = beta_mean_absolute_deviation(
        order, calibration_count + 1 - order
    )
    iid_conditional_deviation_quadrature = quad(
        lambda value: abs(value - nominal_conditional_coverage)
        * beta_distribution.pdf(
            value, order, calibration_count + 1 - order
        ),
        0.0,
        1.0,
        points=[nominal_conditional_coverage],
        epsabs=2e-14,
        epsrel=2e-13,
    )[0]
    assert abs(
        iid_conditional_deviation
        - iid_conditional_deviation_quadrature
    ) < 2e-14
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
        robust_exact_tv_bound, robust_exact_tv_cutoff = (
            iid_robust_rearrangement_bound(
                calibration_count,
                order,
                transfer_target,
                exact_panel_tv,
                exact_test_tv,
            )
        )
        robust_berbee_bound, robust_berbee_cutoff = (
            iid_robust_rearrangement_bound(
                calibration_count,
                order,
                transfer_target,
                berbee_panel_tv,
                berbee_test_tv,
            )
        )
        assert exact_failure <= exact_tv_bound + 2e-14
        assert exact_failure <= rearrangement_bound + 2e-14
        assert exact_failure <= robust_exact_tv_bound + 2e-14
        assert exact_failure <= robust_berbee_bound + 2e-14
        assert rearrangement_bound <= exact_tv_bound + 2e-14
        assert rearrangement_bound <= robust_exact_tv_bound + 2e-14
        assert robust_exact_tv_bound <= exact_tv_bound + 2e-14
        assert robust_berbee_bound <= berbee_bound + 2e-14
        assert exact_tv_bound <= berbee_bound + 2e-14
        assert exact_panel_tv <= berbee_panel_tv + 2e-14
        assert abs(exact_test_tv - beta) < 2e-14
        conditional_wasserstein, conditional_deviation = (
            exact_binary_conditional_coverage_wasserstein(
                calibration_count, order, sampled_correlation
            )
        )
        distributional_tv_budget = exact_panel_tv + exact_test_tv
        assert (
            conditional_wasserstein
            <= distributional_tv_budget + 2e-11
        )
        assert abs(
            conditional_deviation - iid_conditional_deviation
        ) <= distributional_tv_budget + 2e-11
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
            robust_exact_tv_bound,
            robust_exact_tv_cutoff,
            robust_berbee_bound,
            robust_berbee_cutoff,
            conditional_wasserstein,
            conditional_deviation,
            distributional_tv_budget,
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
    expected_conditional_wasserstein = (
        0.0000214995,
        0.0000014743,
        0.0000002473,
    )
    expected_conditional_deviation = (
        0.0697455929,
        0.0697363574,
        0.0697358002,
    )
    assert max(
        abs(row[15] - expected)
        for row, expected in zip(
            transfer_rows, expected_conditional_wasserstein
        )
    ) < 5e-11
    assert max(
        abs(row[16] - expected)
        for row, expected in zip(
            transfer_rows, expected_conditional_deviation
        )
    ) < 5e-11
    print("\nTraining-conditional total-variation transfer")
    print("randomized binary score, N=k=9, target 0.8")
    print(
        "stride   exact failure   panel TV   test TV   "
        "actual-law   iid-robust   exact-TV   robust-beta   slack-beta"
    )
    for row in transfer_rows:
        print(
            f" {row[0]:3d}      {row[1]:.8f}   {row[2]:.8f} "
            f"{row[3]:.8f}   {row[9]:.8f}    {row[11]:.8f}  "
            f"{row[5]:.8f}    {row[13]:.8f}    {row[7]:.8f}"
        )
    print(
        "iid beta failure at target 0.8: "
        f"{integer_beta_cdf(0.8, order, calibration_count + 1 - order):.9f}"
    )
    print("\nWhole conditional-coverage law transfer")
    print("randomized binary score, N=k=9; iid law Beta(9,1)")
    print("stride       W1 exact       TV budget       E|C-.9|")
    for row in transfer_rows:
        print(
            f" {row[0]:3d}       {row[15]:.10f}    "
            f"{row[17]:.10f}    {row[16]:.10f}"
        )
    print(
        "iid Beta(9,1) E|B-.9|: "
        f"{iid_conditional_deviation:.11f}"
    )

    # The irreducible iid conditional fluctuation has an exact beta formula
    # and a square-root calibration-size law.  Check the identity against
    # independent quadrature on an interior parameter grid and its uniform
    # Stirling approximation on five fixed rank fractions.
    beta_mad_identity_error = 0.0
    for total in (5, 10, 20, 40, 80):
        for beta_alpha in range(1, total):
            beta_beta = total - beta_alpha
            beta_mean = beta_alpha / total
            quadrature_value = quad(
                lambda value: abs(value - beta_mean)
                * beta_distribution.pdf(value, beta_alpha, beta_beta),
                0.0,
                1.0,
                points=[beta_mean],
                epsabs=2e-13,
                epsrel=2e-13,
            )[0]
            beta_mad_identity_error = max(
                beta_mad_identity_error,
                abs(
                    beta_mean_absolute_deviation(beta_alpha, beta_beta)
                    - quadrature_value
                ),
            )
    assert beta_mad_identity_error < 8e-14

    beta_mad_rows = []
    beta_mad_relative_error = 0.0
    for rank_fraction in (0.1, 0.25, 0.5, 0.75, 0.9):
        ratios = []
        for total in (160, 640, 2560, 10240, 20480):
            beta_alpha = round(rank_fraction * total)
            beta_beta = total - beta_alpha
            exact_mad = beta_mean_absolute_deviation(beta_alpha, beta_beta)
            leading_mad = math.sqrt(
                2.0 * rank_fraction * (1.0 - rank_fraction)
                / (math.pi * total)
            )
            ratios.append(exact_mad / leading_mad)
        assert abs(ratios[-1] - 1.0) < 5e-5
        beta_mad_relative_error = max(
            beta_mad_relative_error, abs(ratios[-1] - 1.0)
        )
        beta_mad_rows.append((rank_fraction, ratios[-1]))

    conditional_design_rows = []
    for tolerance, dependence_budget in (
        (0.05, 0.0),
        (0.03, 0.0),
        (0.03, 0.005),
        (0.02, 0.0),
    ):
        conditional_design_rows.append(
            (
                tolerance,
                dependence_budget,
                *least_conditional_mad_panel(
                    0.9, tolerance, dependence_budget
                ),
            )
        )
    expected_designs = (
        (29, 27),
        (69, 63),
        (89, 81),
        (149, 135),
    )
    assert tuple((row[2], row[3]) for row in conditional_design_rows) == (
        expected_designs
    )
    print("\nExact beta conditional-dispersion design")
    print(
        "maximum identity error: "
        f"{beta_mad_identity_error:.3e}; maximum final asymptotic "
        f"relative error: {beta_mad_relative_error:.3e}"
    )
    print(" target  budget      N      k     certified E|C-.9|")
    for (
        tolerance,
        dependence_budget,
        design_count,
        design_order,
        design_certificate,
    ) in conditional_design_rows:
        print(
            f" {tolerance:.3f}   {dependence_budget:.3f}    "
            f"{design_count:4d}   {design_order:4d}       "
            f"{design_certificate:.9f}"
        )

    # The calibration panel and final test gap have genuinely separate
    # asymptotic roles.  Let the eight within-panel gaps grow while the final
    # gap stays at one transition.  The panel law converges to iid, but the
    # fixed final correlation 0.8 leaves a non-iid training-conditional
    # failure law for every order statistic.  For N=k=9 and p=0.8, the
    # binomial-beta mixture in iid_panel_fixed_test_failure_limit reduces to
    # (4/9)^9 + (7/18)(8/9)^8 = 6553600/43046721.
    separated_strides = (4, 8, 16, 32, 64)
    separated_rows = []
    base_transition = np.array([[0.9, 0.1], [0.1, 0.9]])
    stationary_binary = np.array([0.5, 0.5])
    separated_limit = iid_panel_fixed_test_failure_limit(
        calibration_count, order, transfer_target, 0.8
    )
    exact_separated_limit = 6553600 / 43046721
    assert abs(separated_limit - exact_separated_limit) < 2e-16
    for stride in separated_strides:
        exact_failure = exact_binary_training_failure_irregular(
            calibration_count,
            order,
            transfer_target,
            0.8,
            (stride,) * (calibration_count - 1) + (1,),
        )
        panel_tv = sampled_path_total_variation_irregular(
            base_transition,
            stationary_binary,
            (stride,) * (calibration_count - 1),
        )
        separated_rows.append((stride, panel_tv, exact_failure))
    assert abs(separated_rows[-1][2] - separated_limit) < 2e-9
    assert separated_rows[-1][1] < 7e-7
    assert abs(
        separated_limit
        - integer_beta_cdf(
            transfer_target, order, calibration_count + 1 - order
        )
    ) > 0.018

    print("\nFinal test-gap necessity")
    print("within-panel stride   panel TV from iid   exact failure")
    for stride, panel_tv, exact_failure in separated_rows:
        print(f" {stride:6d}              {panel_tv:.9f}       {exact_failure:.9f}")
    print(f"fixed-test-memory limit: {separated_limit:.12f}")

    # Check the full binomial-beta mixture.  With zero final correlation it
    # collapses to the iid Beta(k,N+1-k) law.  At correlation 0.8 it agrees,
    # for all nine order statistics, with exact path enumeration after the
    # within-panel stride has removed calibration dependence.  The same
    # mixture gives the marginal mean coverage; unlike the iid benchmark,
    # fixed test memory can change this mean as well as its panelwise tail.
    fixed_memory_rows = []
    zero_memory_error = 0.0
    stride_limit_error = 0.0
    for candidate_order in range(1, calibration_count + 1):
        iid_failure = integer_beta_cdf(
            transfer_target,
            candidate_order,
            calibration_count + 1 - candidate_order,
        )
        zero_memory_error = max(
            zero_memory_error,
            abs(
                iid_panel_fixed_test_failure_limit(
                    calibration_count,
                    candidate_order,
                    transfer_target,
                    0.0,
                )
                - iid_failure
            ),
        )
        fixed_memory_failure = iid_panel_fixed_test_failure_limit(
            calibration_count,
            candidate_order,
            transfer_target,
            0.8,
        )
        enumerated_failure = exact_binary_training_failure_irregular(
            calibration_count,
            candidate_order,
            transfer_target,
            0.8,
            (128,) * (calibration_count - 1) + (1,),
        )
        stride_limit_error = max(
            stride_limit_error,
            abs(enumerated_failure - fixed_memory_failure),
        )
        fixed_memory_rows.append(
            (
                candidate_order,
                iid_failure,
                fixed_memory_failure,
                iid_panel_fixed_test_mean_coverage(
                    calibration_count,
                    candidate_order,
                    0.8,
                ),
            )
        )
    assert zero_memory_error < 3e-16
    assert stride_limit_error < 1e-13
    expected_fixed_memory_rows = {
        5: (0.98041856, 0.7394770904850105, 0.5),
        8: (0.43620761600000024, 0.3539670303807809, 0.8160416666666668),
        9: (0.13421772800000006, 0.1522438840347445, 0.9087152777777779),
    }
    for candidate_order, iid_failure, fixed_failure, mean_coverage in (
        fixed_memory_rows
    ):
        if candidate_order in expected_fixed_memory_rows:
            expected = expected_fixed_memory_rows[candidate_order]
            assert max(
                abs(actual - target_value)
                for actual, target_value in zip(
                    (iid_failure, fixed_failure, mean_coverage), expected
                )
            ) < 3e-15

    marginal_shift_error = 0.0
    marginal_symmetry_error = 0.0
    for candidate_count in range(2, 21):
        for candidate_order in range(1, candidate_count + 1):
            shift = iid_panel_fixed_test_marginal_shift(
                candidate_count, candidate_order
            )
            mixture_mean = iid_panel_fixed_test_mean_coverage(
                candidate_count, candidate_order, 0.37
            )
            marginal_shift_error = max(
                marginal_shift_error,
                abs(
                    mixture_mean
                    - candidate_order / (candidate_count + 1)
                    - 0.37 * shift
                ),
            )
            reflected_shift = iid_panel_fixed_test_marginal_shift(
                candidate_count,
                candidate_count + 1 - candidate_order,
            )
            marginal_symmetry_error = max(
                marginal_symmetry_error,
                abs(shift + reflected_shift),
            )
    assert marginal_shift_error < 4e-16
    assert marginal_symmetry_error < 2e-17
    assert abs(
        iid_panel_fixed_test_marginal_shift(9, 9) - 251 / 23040
    ) < 2e-17

    # Away from the median rank, the exact binomial-tail coefficient has an
    # exponentially accurate outer approximation.  At the median scale its
    # transition is Gaussian.  These checks cover every admissible order for
    # N=2,...,200 and a three-level central-limit sequence independently of
    # the mixture-mean identity above.
    outer_bound_excess = 0.0
    outer_bound_ratio = 0.0
    for candidate_count in range(2, 201):
        trials = candidate_count + 1
        for candidate_order in range(1, candidate_count + 1):
            if 2 * candidate_order == trials:
                assert abs(
                    iid_panel_fixed_test_marginal_shift(
                        candidate_count, candidate_order
                    )
                ) < 2e-17
                continue
            shift = iid_panel_fixed_test_marginal_shift(
                candidate_count, candidate_order
            )
            if 2 * candidate_order > trials:
                outer_value = (
                    trials - candidate_order
                ) / (candidate_count * trials)
                outer_error = outer_value - shift
            else:
                outer_value = -candidate_order / (
                    candidate_count * trials
                )
                outer_error = shift - outer_value
            entropy = binary_relative_entropy_from_half(
                candidate_order / trials
            )
            chernoff_bound = math.exp(-trials * entropy) / candidate_count
            assert outer_error >= -2e-17
            outer_bound_excess = max(
                outer_bound_excess, outer_error - chernoff_bound
            )
            if chernoff_bound > 1e-15:
                outer_bound_ratio = max(
                    outer_bound_ratio,
                    max(0.0, outer_error) / chernoff_bound,
                )
    assert outer_bound_excess < 2e-16
    assert outer_bound_ratio < 1.0

    berry_esseen_excess = -math.inf
    central_berry_esseen_ratio = 0.0
    for candidate_count in range(2, 201):
        trials = candidate_count + 1
        for candidate_order in range(1, candidate_count + 1):
            standardized_rank = (
                2 * candidate_order - trials
            ) / math.sqrt(trials)
            scaled_shift = trials * iid_panel_fixed_test_marginal_shift(
                candidate_count, candidate_order
            )
            gaussian_limit = norm.cdf(standardized_rank) - 0.5
            approximation_error = abs(scaled_shift - gaussian_limit)
            berry_esseen_bound = iid_panel_fixed_test_crossover_bound(
                candidate_count, candidate_order
            )
            berry_esseen_excess = max(
                berry_esseen_excess,
                approximation_error - berry_esseen_bound,
            )
            if abs(standardized_rank) <= 2.1:
                central_berry_esseen_ratio = max(
                    central_berry_esseen_ratio,
                    approximation_error / berry_esseen_bound,
                )
    assert berry_esseen_excess < 1e-15
    assert central_berry_esseen_ratio < 1.0

    crossover_errors = []
    crossover_rows = []
    for trials in (32, 128, 512):
        level_errors = []
        for target_x in (-2.0, -1.0, 0.0, 1.0, 2.0):
            candidate_order = round(
                (trials + target_x * math.sqrt(trials)) / 2.0
            )
            standardized_rank = (
                2 * candidate_order - trials
            ) / math.sqrt(trials)
            scaled_shift = trials * iid_panel_fixed_test_marginal_shift(
                trials - 1, candidate_order
            )
            gaussian_limit = norm.cdf(standardized_rank) - 0.5
            level_errors.append(abs(scaled_shift - gaussian_limit))
            crossover_rows.append(
                (
                    trials,
                    standardized_rank,
                    scaled_shift,
                    gaussian_limit,
                )
            )
        crossover_errors.append(max(level_errors))
    assert crossover_errors[0] > crossover_errors[1] > crossover_errors[2]
    assert crossover_errors[-1] < 0.045

    print("\nAll-order-statistic fixed-test-memory law")
    print(
        f"zero-memory beta-law max error: {zero_memory_error:.3e}; "
        f"stride-128 enumeration max error: {stride_limit_error:.3e}"
    )
    print(
        "closed marginal-shift max error over N=2,...,20: "
        f"{marginal_shift_error:.3e}; reflection error: "
        f"{marginal_symmetry_error:.3e}"
    )
    print(
        "outer Chernoff envelope over N=2,...,200: max excess "
        f"{outer_bound_excess:.3e}, max error/bound "
        f"{outer_bound_ratio:.9f}"
    )
    print(
        "central Gaussian crossover max errors at n=32,128,512: "
        + ", ".join(f"{error:.9f}" for error in crossover_errors)
    )
    print(
        "Berry--Esseen crossover envelope over N=2,...,200: "
        f"max error-minus-bound {berry_esseen_excess:.9f}; "
        "max central error/bound "
        f"{central_berry_esseen_ratio:.9f}"
    )
    print(" n      x_n        n D_N,k      Phi(x_n)-1/2")
    for trials, standardized_rank, scaled_shift, gaussian_limit in (
        crossover_rows
    ):
        if trials == 512:
            print(
                f" {trials:3d}  {standardized_rank: .6f}  "
                f"{scaled_shift: .9f}    {gaussian_limit: .9f}"
            )
    print(" k    iid failure   fixed-memory failure   mean coverage")
    for candidate_order, iid_failure, fixed_failure, mean_coverage in (
        fixed_memory_rows
    ):
        print(
            f" {candidate_order:1d}    {iid_failure:.9f}       "
            f"{fixed_failure:.9f}          {mean_coverage:.9f}"
        )

    # The half-interval construction is not essential.  Let the lower state
    # have stationary mass r and emit uniformly on [0,r], with the upper
    # state uniform on [r,1].  The stationary score remains exactly uniform.
    # A transition with nontrivial eigenvalue a has lower-state probability
    # r+a(1{last lower}-r) at the test point.  Check the resulting exact
    # binomial-beta law against a separate Markov-path enumeration, and the
    # marginal coefficient against the independent component mean.
    unequal_lower_mass = 0.3
    unequal_test_memory = 0.8
    unequal_transition = np.array([
        [
            unequal_lower_mass
            + unequal_test_memory * (1.0 - unequal_lower_mass),
            (1.0 - unequal_lower_mass) * (1.0 - unequal_test_memory),
        ],
        [
            unequal_lower_mass * (1.0 - unequal_test_memory),
            1.0 - unequal_lower_mass
            + unequal_test_memory * unequal_lower_mass,
        ],
    ])
    unequal_rows = []
    unequal_zero_memory_error = 0.0
    unequal_enumeration_error = 0.0
    for candidate_order in range(1, calibration_count + 1):
        iid_failure = integer_beta_cdf(
            transfer_target,
            candidate_order,
            calibration_count + 1 - candidate_order,
        )
        zero_memory_failure = iid_panel_fixed_test_failure_limit(
            calibration_count,
            candidate_order,
            transfer_target,
            0.0,
            unequal_lower_mass,
        )
        unequal_zero_memory_error = max(
            unequal_zero_memory_error,
            abs(zero_memory_failure - iid_failure),
        )
        fixed_failure = iid_panel_fixed_test_failure_limit(
            calibration_count,
            candidate_order,
            transfer_target,
            unequal_test_memory,
            unequal_lower_mass,
        )
        enumerated_failure = exact_partition_training_failure_irregular(
            calibration_count,
            candidate_order,
            transfer_target,
            unequal_transition,
            (128,) * (calibration_count - 1) + (1,),
        )
        unequal_enumeration_error = max(
            unequal_enumeration_error,
            abs(fixed_failure - enumerated_failure),
        )
        mean_coverage = iid_panel_fixed_test_mean_coverage(
            calibration_count,
            candidate_order,
            unequal_test_memory,
            unequal_lower_mass,
        )
        marginal_shift = iid_panel_fixed_test_marginal_shift(
            calibration_count, candidate_order, unequal_lower_mass
        )
        assert abs(
            mean_coverage
            - candidate_order / (calibration_count + 1)
            - unequal_test_memory * marginal_shift
        ) < 7e-16
        unequal_rows.append(
            (candidate_order, iid_failure, fixed_failure, mean_coverage)
        )
    assert unequal_zero_memory_error < 8e-16
    assert unequal_enumeration_error < 3e-13

    unequal_shift_error = 0.0
    unequal_reflection_error = 0.0
    for candidate_count in range(2, 21):
        for lower_mass in (0.2, 0.3, 0.65, 0.8):
            for candidate_order in range(1, candidate_count + 1):
                shift = iid_panel_fixed_test_marginal_shift(
                    candidate_count, candidate_order, lower_mass
                )
                mixture_mean = iid_panel_fixed_test_mean_coverage(
                    candidate_count, candidate_order, 0.37, lower_mass
                )
                unequal_shift_error = max(
                    unequal_shift_error,
                    abs(
                        mixture_mean
                        - candidate_order / (candidate_count + 1)
                        - 0.37 * shift
                    ),
                )
                reflected_shift = iid_panel_fixed_test_marginal_shift(
                    candidate_count,
                    candidate_count + 1 - candidate_order,
                    1.0 - lower_mass,
                )
                unequal_reflection_error = max(
                    unequal_reflection_error,
                    abs(shift + reflected_shift),
                )
    assert unequal_shift_error < 1.5e-15
    assert unequal_reflection_error < 2e-16

    # Away from the partition mass r, binomial concentration and the exact
    # covariance identity give N D -> -tau(1-r)/r below r and
    # N D -> r(1-tau)/(1-r) above r.  The three sample sizes certify the
    # approach without using the component-mixture mean.
    unequal_outer_cases = (
        (0.3, 0.2),
        (0.3, 0.8),
        (0.7, 0.4),
        (0.7, 0.8),
    )
    unequal_outer_rows = []
    for lower_mass, rank_fraction in unequal_outer_cases:
        limit = (
            -rank_fraction * (1.0 - lower_mass) / lower_mass
            if rank_fraction < lower_mass
            else lower_mass * (1.0 - rank_fraction) / (1.0 - lower_mass)
        )
        approximations = []
        for candidate_count in (256, 1024, 4096):
            candidate_order = round(
                (candidate_count + 1) * rank_fraction
            )
            approximations.append(
                candidate_count
                * iid_panel_fixed_test_marginal_shift(
                    candidate_count, candidate_order, lower_mass
                )
            )
        errors = [abs(value - limit) for value in approximations]
        assert errors[-1] < 2.5e-4
        unequal_outer_rows.append(
            (lower_mass, rank_fraction, approximations[-1], limit)
        )

    # At the omitted transition k/(N+1) -> r, the two smooth branches are
    # selected with asymptotically Gaussian probabilities.  If
    # x_N=(k-1-(N-1)r)/sqrt((N-1)r(1-r)) -> x, then
    #
    #     N D_{N,k}(r) -> Phi(x)+r-1.
    #
    # This interpolates between -(1-r) and r, the two one-sided outer
    # limits.  The crossing increment has binomial mass O(N^{-1/2}) and
    # therefore does not contribute to the limit.
    unequal_critical_rows = []
    unequal_critical_scaled_error = 0.0
    for lower_mass in (0.3, 0.7):
        for target_x in (-1.0, 0.0, 1.0):
            errors = []
            row = None
            for candidate_count in (256, 1024, 4096):
                variance = (
                    (candidate_count - 1)
                    * lower_mass
                    * (1.0 - lower_mass)
                )
                candidate_order = int(round(
                    1.0
                    + (candidate_count - 1) * lower_mass
                    + target_x * math.sqrt(variance)
                ))
                standardized_rank = (
                    candidate_order
                    - 1.0
                    - (candidate_count - 1) * lower_mass
                ) / math.sqrt(variance)
                scaled_shift = candidate_count * (
                    iid_panel_fixed_test_marginal_shift(
                        candidate_count, candidate_order, lower_mass
                    )
                )
                gaussian_limit = float(ndtr(standardized_rank)) + lower_mass - 1.0
                errors.append(abs(scaled_shift - gaussian_limit))
                unequal_critical_scaled_error = max(
                    unequal_critical_scaled_error,
                    math.sqrt(candidate_count) * errors[-1],
                )
                row = (
                    lower_mass,
                    standardized_rank,
                    scaled_shift,
                    gaussian_limit,
                )
            assert errors[-1] < 0.021
            assert row is not None
            unequal_critical_rows.append(row)
    assert unequal_critical_scaled_error < 1.3

    print("\nUnequal-mass fixed-test-memory law")
    print(
        f"lower-state mass {unequal_lower_mass:.1f}; zero-memory beta error "
        f"{unequal_zero_memory_error:.3e}; stride-128 enumeration error "
        f"{unequal_enumeration_error:.3e}"
    )
    print(
        "general marginal-shift max error: "
        f"{unequal_shift_error:.3e}; reflection error: "
        f"{unequal_reflection_error:.3e}"
    )
    print(" r     tau       N D at N=4096       outer limit")
    for lower_mass, rank_fraction, approximation, limit in unequal_outer_rows:
        print(
            f" {lower_mass:.1f}   {rank_fraction:.1f}       "
            f"{approximation: .9f}       {limit: .9f}"
        )
    print(" r      x_N       N D at N=4096       Phi(x_N)+r-1")
    for lower_mass, standardized_rank, approximation, limit in (
        unequal_critical_rows
    ):
        print(
            f" {lower_mass:.1f}   {standardized_rank: .5f}       "
            f"{approximation: .9f}       {limit: .9f}"
        )
    print(
        "critical-window max sqrt(N) error: "
        f"{unequal_critical_scaled_error:.9f}"
    )
    print(" k    iid failure   fixed-memory failure   mean coverage")
    for candidate_order, iid_failure, fixed_failure, mean_coverage in (
        unequal_rows
    ):
        if candidate_order in (3, 5, 8, 9):
            print(
                f" {candidate_order:1d}    {iid_failure:.9f}       "
                f"{fixed_failure:.9f}          {mean_coverage:.9f}"
            )

    # The iid-baseline rearrangement has a sharp square-root small-budget
    # law.  If g is the beta density at p, its excess above the iid failure
    # probability is sqrt(2*g*eta)+O(eta), whereas optimizing the slack bound
    # gives 2*sqrt(g*eta)+O(eta).
    beta_density_at_target = beta_distribution.pdf(
        transfer_target, order, calibration_count + 1 - order
    )
    small_budgets = np.array([1e-8, 1e-10, 1e-12])
    iid_failure = integer_beta_cdf(
        transfer_target, order, calibration_count + 1 - order
    )
    robust_ratios = []
    slack_ratios = []
    for budget in small_budgets:
        robust_bound, _ = iid_robust_rearrangement_bound(
            calibration_count,
            order,
            transfer_target,
            0.0,
            budget,
        )
        slack_result = minimize_scalar(
            lambda slack: training_conditional_transfer_bound(
                calibration_count,
                order,
                transfer_target,
                0.0,
                budget,
                slack,
            ),
            bounds=(1e-14, 1.0 - transfer_target - 1e-14),
            method="bounded",
            options={"xatol": 1e-15},
        )
        robust_ratios.append(
            (robust_bound - iid_failure) / math.sqrt(budget)
        )
        slack_ratios.append(
            (slack_result.fun - iid_failure) / math.sqrt(budget)
        )
    robust_limit = math.sqrt(2.0 * beta_density_at_target)
    slack_limit = 2.0 * math.sqrt(beta_density_at_target)
    assert abs(robust_ratios[-1] - robust_limit) < 4e-6
    assert abs(slack_ratios[-1] - slack_limit) < 6e-6
    print(
        "small-TV excess constants (robust/slack): "
        f"{robust_ratios[-1]:.9f}/{slack_ratios[-1]:.9f}; "
        f"limits {robust_limit:.9f}/{slack_limit:.9f}"
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

    # Deterministic nonuniform spacing costs the sum of the individual
    # absolute-regularity coefficients, not N times a coefficient evaluated
    # at the average gap.  Here start_strides are distances between block
    # starts and mixing_gaps count transitions from the end of one block to
    # the start of the next.  The same sequence, viewed with L=1, gives an
    # exact binary training-conditional calculation.
    irregular_start_strides = (8, 12, 16, 20, 24, 28, 32, 36, 40)
    irregular_mixing_gaps = tuple(
        stride - window_length + 1 for stride in irregular_start_strides
    )
    irregular_coverage = exact_irregular_window_coverage(
        calibration_count,
        order,
        window_length,
        0.8,
        irregular_start_strides,
        quadrature_order=100,
    )
    uniform_irregular = exact_irregular_window_coverage(
        calibration_count,
        order,
        window_length,
        0.8,
        (8,) * calibration_count,
        quadrature_order=100,
    )
    assert abs(
        uniform_irregular
        - exact_sliding_window_coverage(
            calibration_count,
            order,
            window_length,
            0.8,
            stride=8,
            quadrature_order=100,
        )
    ) < 2e-14

    binary_transition = np.array([
        [0.9, 0.1],
        [0.1, 0.9],
    ])
    irregular_path_tv = sampled_path_total_variation_irregular(
        binary_transition,
        np.array([0.5, 0.5]),
        irregular_mixing_gaps,
    )
    irregular_beta_sum = 0.5 * sum(
        0.8**gap for gap in irregular_mixing_gaps
    )
    average_gap_beta = (
        calibration_count
        * 0.5
        * 0.8 ** (sum(irregular_mixing_gaps) / calibration_count)
    )
    worst_gap_beta = (
        calibration_count * 0.5 * 0.8 ** min(irregular_mixing_gaps)
    )
    assert abs(irregular_coverage - iid_coverage) <= irregular_beta_sum
    assert irregular_path_tv <= irregular_beta_sum + 2e-14
    # Convexity of h -> rho^h makes an average-gap substitution go in the
    # wrong direction.  In this example it is smaller even than the actual
    # joint total variation, so it cannot be a valid coupling bound.
    assert irregular_path_tv > 3.9 * average_gap_beta

    irregular_training_failure = exact_binary_training_failure_irregular(
        calibration_count,
        order,
        transfer_target,
        0.8,
        irregular_mixing_gaps,
    )
    irregular_panel_budget = 0.5 * sum(
        0.8**gap for gap in irregular_mixing_gaps[:-1]
    )
    irregular_test_budget = 0.5 * 0.8 ** irregular_mixing_gaps[-1]
    irregular_transfer = minimize_scalar(
        lambda slack: training_conditional_transfer_bound(
            calibration_count,
            order,
            transfer_target,
            irregular_panel_budget,
            irregular_test_budget,
            slack,
        ),
        bounds=(1e-10, 1.0 - transfer_target - 1e-10),
        method="bounded",
        options={"xatol": 1e-14},
    )
    assert irregular_training_failure <= irregular_transfer.fun + 2e-14
    assert abs(irregular_coverage - 0.8987479129344994) < 2e-13
    assert abs(irregular_path_tv - 0.16384) < 2e-14
    assert abs(irregular_training_failure - 0.13934751885262905) < 2e-14

    # With an exponential beta envelope and a fixed total spacing budget,
    # equal gaps minimize the marginal coupling envelope.  The separated PAC
    # penalty instead assigns weight 1/gamma to the final test gap and has the
    # exact logarithmic water-filling solution.
    gap_floor = 1.0
    excess_gap_budget = (
        sum(irregular_mixing_gaps)
        - calibration_count * gap_floor
    )
    exponential_decay = -math.log(0.8)
    equal_gaps = np.full(
        calibration_count,
        gap_floor + excess_gap_budget / calibration_count,
    )
    equal_marginal_envelope = 0.5 * np.sum(
        np.exp(-exponential_decay * equal_gaps)
    )
    assert abs(equal_marginal_envelope - average_gap_beta) < 2e-15
    assert equal_marginal_envelope < irregular_beta_sum

    design_slack = 0.05
    design_weights = np.ones(calibration_count)
    design_weights[-1] = 1.0 / design_slack
    equal_excess = optimal_exponential_gap_allocation(
        np.ones(calibration_count), excess_gap_budget, exponential_decay
    )
    assert np.max(np.abs(
        equal_excess - excess_gap_budget / calibration_count
    )) < 2e-13
    low_budget = 0.5 * math.log(1.0 / design_slack) / exponential_decay
    low_budget_allocation = optimal_exponential_gap_allocation(
        design_weights, low_budget, exponential_decay
    )
    assert np.max(np.abs(low_budget_allocation[:-1])) < 2e-13
    assert abs(low_budget_allocation[-1] - low_budget) < 2e-13
    optimal_excess = optimal_exponential_gap_allocation(
        design_weights, excess_gap_budget, exponential_decay
    )
    optimal_gaps = gap_floor + optimal_excess
    weighted_terms = design_weights * np.exp(
        -exponential_decay * optimal_excess
    )
    assert np.max(weighted_terms) - np.min(weighted_terms) < 2e-13
    optimal_penalty = 0.5 * math.exp(
        -exponential_decay * gap_floor
    ) * weighted_terms.sum()
    equal_weighted_penalty = 0.5 * np.sum(
        design_weights * np.exp(-exponential_decay * equal_gaps)
    )
    assert optimal_penalty < 0.45 * equal_weighted_penalty

    # An independent random feasible-design check guards the KKT
    # implementation without treating numerical optimization as the proof.
    random_allocations = rng.dirichlet(
        np.ones(calibration_count), size=2000
    ) * excess_gap_budget
    random_objectives = 0.5 * math.exp(
        -exponential_decay * gap_floor
    ) * np.sum(
        design_weights[None, :]
        * np.exp(-exponential_decay * random_allocations),
        axis=1,
    )
    assert float(random_objectives.min()) > optimal_penalty
    assert abs(optimal_gaps[0] - 19.50831835) < 5e-9
    assert abs(optimal_gaps[-1] - 32.93345322) < 5e-9
    assert abs(optimal_penalty - 0.05789767565619619) < 2e-14
    assert abs(equal_weighted_penalty - 0.12912720851596698) < 2e-14

    print("\nIrregularly spaced nonoverlapping path scores")
    print("block-start strides: " + str(irregular_start_strides))
    print("mixing gaps: " + str(irregular_mixing_gaps))
    print(
        f"exact path-score coverage={irregular_coverage:.8f}; "
        f"absolute error={abs(irregular_coverage-iid_coverage):.8f}"
    )
    print(
        f"exact joint TV={irregular_path_tv:.8f}; "
        f"sum-beta={irregular_beta_sum:.8f}; "
        f"average-gap surrogate={average_gap_beta:.8f}; "
        f"worst-gap bound={worst_gap_beta:.8f}"
    )
    print(
        f"training failure={irregular_training_failure:.8f}; "
        f"optimized PAC bound={irregular_transfer.fun:.8f}; "
        f"slack={irregular_transfer.x:.8f}"
    )
    print(
        "fixed-span marginal envelope, irregular/equal: "
        f"{irregular_beta_sum:.8f}, {equal_marginal_envelope:.8f}"
    )
    print(
        "fixed-span PAC-weighted optimal calibration/test gaps: "
        f"{optimal_gaps[0]:.8f}, {optimal_gaps[-1]:.8f}"
    )
    print(
        "fixed-span PAC-weighted penalty, optimal/equal: "
        f"{optimal_penalty:.8f}, {equal_weighted_penalty:.8f}"
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

    # Fitting on an independent sample makes the learned score map fixed after
    # conditioning on that sample.  In contrast, a learner that memorizes the
    # calibration labels has zero calibration residuals and positive test
    # residual almost surely.  Continuous uniforms give a direct certificate:
    # the honest residual ranks retain exact iid coverage, whereas leakage
    # drives coverage to zero despite complete independence across examples.
    fitted_draws = 400_000
    fitted_calibration = rng.random((fitted_draws, calibration_count))
    fitted_test = rng.random(fitted_draws)
    fitted_thresholds = np.partition(
        fitted_calibration, order - 1, axis=1
    )[:, order - 1]
    honest_fitted_coverage = float(np.mean(fitted_test <= fitted_thresholds))
    leaky_fitted_coverage = float(np.mean(fitted_test <= 0.0))
    assert abs(honest_fitted_coverage - iid_coverage) < 0.0015
    assert leaky_fitted_coverage == 0.0

    # The coefficient for a remote same-series training sample is one beta
    # term, with no multiplier by the number of future blocks.  Enumerating the
    # entire future path confirms that later within-panel dependence is kept in
    # both laws and cancels from this training/future TV comparison.
    training_gap = 7
    future_strides = (3, 5, 2, 6)
    training_tv = training_future_total_variation(
        binary_transition,
        np.array([0.5, 0.5]),
        training_gap,
        future_strides,
    )
    training_beta = markov_beta(
        binary_transition, np.array([0.5, 0.5]), training_gap
    )
    assert abs(training_tv - training_beta) < 2e-15
    assert abs(training_beta - 0.5 * 0.8**training_gap) < 2e-15
    print("\nFitted-predictor split certificate")
    print(
        "independent training: empirical coverage="
        f"{honest_fitted_coverage:.8f}; iid target={iid_coverage:.8f}"
    )
    print(
        "calibration-label memorizer: exact coverage="
        f"{leaky_fitted_coverage:.8f}"
    )
    print(
        "remote training/future-panel TV: exact="
        f"{training_tv:.8f}; beta(g)={training_beta:.8f}"
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
        "PASS: symmetric and unequal-rate path crossovers, predictor "
        "crossover, overlap covariance, "
        "finite-sample terminal and strided-window rank coverage, general "
        "finite-chain path coefficients through second order, smooth "
        "nonlinear path maps including the critical-gradient second order, "
        "posterior cancellation geometry through second order, "
        "regular and irregular absolute-regularity coupling with optimal "
        "fixed-span gap allocation and discrete-score tie handling, "
        "iid training-conditional beta law including randomized atoms, "
        "sharp one-sided and two-finite-endpoint PAC designs, "
        "symmetric and unequal-mass all-order "
        "fixed-test-memory laws, and dependent "
        "total-variation transfer with coefficient-only and actual-law "
        "slack-free rearrangement bounds plus whole-law Wasserstein control "
        "and exact beta conditional-dispersion calibration design, "
        "independent-training validity, calibration-leakage failure, "
        "remote-training total-variation separation, "
        "exact transforms, and simulations"
    )


if __name__ == "__main__":
    main()
