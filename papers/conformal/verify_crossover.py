"""Checks finite-chain conditional-coverage crossover and calibration variance.

Scores conditional on terminal regime are exponential with scales 1 and 3.
The Monte Carlo uses CTMC endpoint transitions, not the analytic coverage
or covariance formula used by the note.  Nonreversible three-state examples
check the exact semigroup identity and the additive-reversibilization bounds;
a reversible three-state example checks the corresponding specialization.
An exact hidden-state enumeration checks the burn-in transfer from an
arbitrary initial regime law to a stationary calibration panel.  It also
checks exact transient mean and variance identities for the empirical CDF.
Irregular observation grids check the pairwise-kernel variance bound and
show that substituting the average spacing can be anti-conservative.  The
same exponential kernel gives the exact minimum-variance nonnegative weights
and their closed-form information for every deterministic grid.  With fixed
endpoints it also verifies that equal spacing uniquely maximizes this
information, together with the endpoint-atom limit of the optimal weights on
a dense grid and a sharp global observation-budget error bound.
Finally, an exact polynomial Feynman--Kac recursion computes the full count
law, checks the one-sided Cantelli conversion from the spectral variance bound
to a training-panel PAC bound, and supplies both a finite-panel Chernoff
certificate and the exact Perron large-deviation rate for long regular panels.
It verifies the associated sharp lattice tail prefactor and its first two
relative saddle-point corrections against exact coefficient tails through 960
observations.  A moderate-deviation certificate then lets the success fraction
approach its mean while its standardized distance still diverges, checking the
boundary where the lattice amplitude itself becomes singular.
A lattice Edgeworth certificate covers the complementary central zone.  It
retains both the half-integer continuity correction and the arbitrary-start
mean and variance hidden in the Perron boundary amplitude.  Its explicit
second-order lattice term includes the midpoint Euler--Maclaurin correction
and leaves an order-n^{-3/2} residual against exact coefficient tails.  An
all-vertices check certifies uniformity over every initial regime law.
The curvature of that Perron eigenvalue is checked against the discrete
Green--Kubo variance, including its exact finite-panel intercept and remainder.
The exact finite-panel variance factor is also followed through a joint limit
in which the panel size grows while the per-observation mixing exposure may
vanish.  This certifies the critical finite-total-exposure profile and its
dense- and resolved-grid limits without making a conditional-coverage claim.
For every bounded moving total exposure it also checks the first two
finite-panel Euler--Maclaurin corrections, the cancellation of the nominal
order-n^{-1} term, and a computable order-n^{-6} remainder bound.
An all-order eigenvector recursion supplies every dependent-panel cumulant
rate and the local expansion of the large-deviation rate function.  A
spectral-projector calculation also identifies the complete order-one
boundary correction for every fixed cumulant order and every initial law.
An exact i.i.d. split-conformal calculation finally separates all of these
dependence results from conditional validity: a rare group's conditional
coverage can tend to zero while the ordinary finite-sample marginal guarantee
remains conservative.
A relative-entropy comparison with the iid pooled panel gives explicit
joint panel-size/fast-switching total-variation bounds on regular and
irregular observation grids.  Their exact mutual-information and Renyi-2/
weighted-Hilbert--Schmidt refinements retain every nonconstant singular mode
instead of replacing them all by the slowest relaxation rate.  The same
chain rule gives one exact arbitrary-start path entropy, combining residual
initial memory and all within-panel transitions before Pinsker is applied.
The resulting emitted-panel comparison is a marginal law over random panels,
not a pointwise, feature-conditional, or training-conditional guarantee.  The
same
data-processing argument transfers the entire realized pooled
coverage law to its finite-sample Beta order-statistic benchmark, controlling
all bounded diagnostics and moments rather than only one failure event.  A
succeeding block of future predictions has a Beta-binomial iid benchmark,
not an independent-binomial law, because every indicator shares the random
calibration threshold.  Applying the joint score-law comparison before the
threshold/count map transfers this whole repeated-coverage law as well.  A
symmetric two-state example reduces
exactly to homogeneous or heterogeneous biased-versus-fair Bernoulli products
and proves that the resulting square-root information scale is sharp,
including its critical local-asymptotic-normal and persistent-plus-diffuse
limits.  A complementary overlapping-emission calculation identifies the
observed Fisher information lost through the score channel, proves an exact
finite-panel chi-square identity, and checks the attenuated critical TV limit.
The same factorization
with an arbitrary burned-in start gives the exact two-parameter limit in which
residual initial memory and the Gaussian transition experiment coexist.
"""
import itertools
import math
from decimal import Decimal, localcontext

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import brentq, minimize_scalar
from scipy.special import betainc, logsumexp, ndtr
from scipy.stats import binom

TARGET = 0.9
EPS = 0.2
SCALES = np.array([1.0, 3.0])
ROUND_TOL = 64 * np.finfo(float).eps


def pooled_cdf(x):
    return 1 - np.exp(-x / SCALES).mean()


def stationary(Q):
    A = Q.T.copy()
    A[-1] = 1
    b = np.zeros(len(Q))
    b[-1] = 1
    return np.linalg.solve(A, b)


def dependence_hilbert_schmidt_squared(P, pi):
    """Weighted Hilbert--Schmidt dependence energy of a Markov kernel.

    This is simultaneously the stationary average row chi-square divergence
    from ``pi`` and the squared Frobenius norm of the L2(pi) similarity of
    ``P - 1*pi``.
    """
    centered = P - pi[None, :]
    row_chi_square = np.sum(
        pi[:, None] * centered ** 2 / pi[None, :]
    )
    similarity = (
        np.diag(np.sqrt(pi))
        @ centered
        @ np.diag(1 / np.sqrt(pi))
    )
    singular_energy = np.linalg.norm(similarity, ord="fro") ** 2
    assert abs(row_chi_square - singular_energy) < 3e-13
    return row_chi_square


def nonstationary_path_relative_entropy(initial, transitions, pi):
    """Exact KL of a nonstationary Markov path from ``pi`` product measure.

    The initial marginal is ``initial``.  ``transitions[t]`` carries the
    marginal at time t to time t+1.  The returned transition terms retain the
    changing pre-transition marginal rather than replacing it by stationarity.
    """
    initial = np.asarray(initial, dtype=float)
    pi = np.asarray(pi, dtype=float)
    assert initial.shape == pi.shape and np.all(pi > 0)

    positive = initial > 0
    initial_kl = float(np.sum(
        initial[positive] * np.log(initial[positive] / pi[positive])
    ))
    marginal = initial.copy()
    transition_terms = []
    for transition in transitions:
        transition = np.asarray(transition, dtype=float)
        row_kl = np.zeros(len(pi))
        positive_entries = transition > 0
        log_density = np.zeros_like(transition)
        log_density[positive_entries] = np.log(
            transition[positive_entries]
            / np.broadcast_to(pi, transition.shape)[positive_entries]
        )
        row_kl = np.sum(transition * log_density, axis=1)
        transition_terms.append(float(marginal @ row_kl))
        marginal = marginal @ transition
    return initial_kl + sum(transition_terms), np.asarray(transition_terms)


def cdf_vector(x, scales):
    return 1 - np.exp(-x / scales)


def pooled_quantile(scales, pi):
    return brentq(lambda x: pi @ cdf_vector(x, scales) - TARGET,
                  0.0, 50.0)


def endpoint_simulation(rng, P, scales, q, draws=120000):
    observed = []
    for start in range(len(scales)):
        final = rng.choice(len(scales), size=draws, p=P[start])
        scores = rng.exponential(scale=scales[final])
        observed.append(np.mean(scores <= q))
    return np.asarray(observed)


def additive_gap(Q, pi):
    """Gap of -(Q+Q*)/2 on centered L2(pi)."""
    D = np.diag(pi)
    Q_star = np.diag(1 / pi) @ Q.T @ D
    Q_symmetric = (Q + Q_star) / 2
    similarity = (np.diag(np.sqrt(pi)) @ Q_symmetric
                  @ np.diag(1 / np.sqrt(pi)))
    assert np.allclose(similarity, similarity.T, atol=ROUND_TOL)
    eigenvalues = np.linalg.eigvalsh(-similarity)
    assert abs(eigenvalues[0]) < 2e-14 and eigenvalues[1] > 0
    return eigenvalues[1]


def additive_gap_witness(Q, pi):
    """Return the gap and a centered unit eigenfunction attaining its slope."""
    D_sqrt = np.diag(np.sqrt(pi))
    D_inv_sqrt = np.diag(1 / np.sqrt(pi))
    Q_star = np.diag(1 / pi) @ Q.T @ np.diag(pi)
    similarity = D_sqrt @ ((Q + Q_star) / 2) @ D_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(-similarity)
    witness = D_inv_sqrt @ eigenvectors[:, 1]
    assert abs(pi @ witness) < 2e-14
    assert abs(pi @ witness ** 2 - 1) < 2e-14
    return eigenvalues[1], witness


def finite_variance_factor(a, n):
    """Exact geometric factor for n observations with correlation envelope a^lag."""
    direct = 1 + 2 / n * sum((n - lag) * a ** lag
                             for lag in range(1, n))
    closed = ((1 + a) / (1 - a)
              - 2 * a * (1 - a ** n) / (n * (1 - a) ** 2))
    assert abs(direct - closed) < 2e-14
    return closed


def critical_variance_profile(total_exposure):
    """Limit of V_n(exp(-total_exposure/n))/n."""
    if abs(total_exposure) < 1e-6:
        # Stable continuation of 2 (lambda - 1 + exp(-lambda)) / lambda**2.
        return (1 - total_exposure / 3
                + total_exposure ** 2 / 12
                - total_exposure ** 3 / 60)
    return (2 * (total_exposure - 1 + math.exp(-total_exposure))
            / total_exposure ** 2)


def normalized_variance_factor(step_exposure, n):
    """Return V_n(exp(-step_exposure))/n without near-one cancellation."""
    return ((1 + 2 / n * sum(
        (n - lag) * math.exp(-step_exposure * lag)
        for lag in range(1, n)
    )) / n)


def critical_factor_decimal(total_exposure, n):
    """High-precision W_n for x_n=total_exposure/n.

    The fourth- and sixth-order residuals in the critical expansion are much
    smaller than double precision at the largest panel sizes, so the
    certificate evaluates the defining finite sum with Decimal arithmetic.
    """
    lam = Decimal(total_exposure)
    nd = Decimal(n)
    dependence = sum(
        Decimal(n - lag) * (-lam * Decimal(lag) / nd).exp()
        for lag in range(1, n)
    )
    return (Decimal(1) + Decimal(2) * dependence / nd) / nd


def joint_panel_switching_checks():
    """Certify the sharp joint panel-size/per-step-mixing regimes."""
    n_critical = 4096
    critical_rows = []
    for total_exposure in (0.05, 0.5, 2.0, 8.0):
        normalized = normalized_variance_factor(
            total_exposure / n_critical, n_critical
        )
        limit = critical_variance_profile(total_exposure)
        assert abs(normalized - limit) < 2e-7
        # The symmetric binary two-state construction attains the envelope.
        a = math.exp(-total_exposure / n_critical)
        exact_binary_variance = (
            0.25 / n_critical
            + 0.5 / n_critical ** 2 * sum(
                (n_critical - lag) * a ** lag
                for lag in range(1, n_critical)
            )
        )
        assert abs(exact_binary_variance - 0.25 * normalized) < 2e-14
        critical_rows.append((total_exposure, normalized, limit))

    # Since lambda_n=n*x_n identically, Euler--Maclaurin applies pointwise at
    # every moving total exposure lambda_n.  Its periodic-B_6 remainder gives
    # an explicit finite-n error bound c6(lambda_n)/n^6.
    # The explicit 1/n term in W_n cancels the endpoint correction of the
    # Riemann sum.  High precision exposes all three successive rates.
    expansion_rows = []
    with localcontext() as context:
        context.prec = 80
        for lam_text in ("0.05", "0.5", "2", "8"):
            lam = Decimal(lam_text)
            exp_minus = (-lam).exp()
            profile = Decimal(2) * (lam - 1 + exp_minus) / lam ** 2
            c2 = (1 + lam - exp_minus) / Decimal(6)
            c4 = -lam ** 2 * (lam + 3 - 3 * exp_minus) / Decimal(360)
            c6 = lam ** 4 * (lam + 5 - 5 * exp_minus) / Decimal(15120)
            errors = [[], [], []]
            bound_ratios = []
            for n in (32, 64, 128, 256):
                value = critical_factor_decimal(lam_text, n)
                errors[0].append(abs(value - profile))
                errors[1].append(abs(value - profile - c2 / Decimal(n) ** 2))
                errors[2].append(abs(
                    value - profile - c2 / Decimal(n) ** 2
                    - c4 / Decimal(n) ** 4
                ))
                bound_ratios.append(errors[2][-1] / (c6 / Decimal(n) ** 6))
            rates = [
                math.log2(float(level[-2] / level[-1]))
                for level in errors
            ]
            assert abs(rates[0] - 2) < 1e-4
            assert abs(rates[1] - 4) < 2e-4
            assert abs(rates[2] - 6) < 2e-4
            assert max(bound_ratios) <= 1
            n = 256
            value = critical_factor_decimal(lam_text, n)
            scaled_c2 = (value - profile) * Decimal(n) ** 2
            scaled_c4 = (
                value - profile - c2 / Decimal(n) ** 2
            ) * Decimal(n) ** 4
            assert abs(scaled_c2 - c2) < Decimal("3e-5")
            assert abs(scaled_c4 - c4) < Decimal("6e-5")
            expansion_rows.append((
                float(lam), rates, float(c2), float(c4),
                float(c6), float(max(bound_ratios))
            ))

    # Dense observations with diverging total exposure have only
    # total_exposure/2 variance-equivalent observations to first order.
    n_dense = 16384
    dense_exposure = math.sqrt(n_dense)
    dense_normalized = normalized_variance_factor(
        dense_exposure / n_dense, n_dense
    )
    dense_ratio = dense_exposure * dense_normalized / 2
    assert abs(dense_ratio - 1) < 0.008

    # A resolved grid instead supplies a fixed inflation per observation.
    n_resolved = 4096
    step_exposure = 0.7
    resolved_factor = n_resolved * normalized_variance_factor(
        step_exposure, n_resolved
    )
    resolved_limit = 1 / math.tanh(step_exposure / 2)
    assert abs(resolved_factor - resolved_limit) < 0.001

    print("joint panel/switching critical profile:",
          "; ".join(
              f"lambda={lam:g}: {value:.9f} (limit {limit:.9f})"
              for lam, value, limit in critical_rows
          ))
    print("critical finite-panel expansion:",
          "; ".join(
              f"lambda={lam:g}: rates {rates[0]:.6f},"
              f" {rates[1]:.6f}, {rates[2]:.6f};"
              f" c2={c2:.9f}, c4={c4:.9f};"
              f" max remainder/bound={bound_ratio:.9f}"
              for lam, rates, c2, c4, c6, bound_ratio in expansion_rows
          ))
    print("joint panel/switching outer regimes:",
          f"dense ratio {dense_ratio:.9f},",
          f"resolved factor {resolved_factor:.9f}",
          f"(limit {resolved_limit:.9f})")


def cantelli_order_statistic_checks(Q, pi, gamma_s, scales):
    """Check the one-sided PAC bound for a dependent order statistic.

    At the population p-quantile, the event that the kth calibration order
    statistic lies below that quantile is the event that at least k binary
    indicators are one.  The polynomial Feynman--Kac recursion therefore
    supplies the failure probability without simulating scores or ranks and
    without enumerating all 2**n binary paths.
    """
    target = 0.8
    n = k = 9
    q = brentq(
        lambda x: pi @ cdf_vector(x, scales) - target,
        0.0,
        50.0,
    )
    success = cdf_vector(q, scales)
    rows = []
    for spacing in (0.05, 0.1, 0.2, 0.4, 0.8, 1.5):
        transition = expm(spacing * Q)
        transitions = [transition] * (n - 1)
        count_law = binary_count_distribution(pi, transitions, success)
        exact_failure = count_law[k:].sum()

        # Independent path enumeration certifies every coefficient of the
        # dynamic program at this deliberately small n.
        law = binary_panel_law(pi, transition, success, n)
        enumerated_count_law = aggregate_binary_count_law(law, n)
        assert np.max(abs(count_law - enumerated_count_law)) < ROUND_TOL
        z = 1.17
        pgf_matrix = binary_count_pgf(pi, transitions, success, z)
        pgf_polynomial = count_law @ z ** np.arange(n + 1)
        assert abs(pgf_matrix - pgf_polynomial) < 2e-14
        panel_mean, panel_variance = panel_moments_from_law(law, n)
        assert abs(panel_mean - target) < ROUND_TOL

        a = math.exp(-gamma_s * spacing)
        factor = finite_variance_factor(a, n)
        variance_bound = target * (1 - target) * factor / n
        assert panel_variance <= variance_bound + 2e-14
        rank_gap = k / n - target
        cantelli = variance_bound / (variance_bound + rank_gap ** 2)
        chebyshev = min(1.0, variance_bound / rank_gap ** 2)
        assert exact_failure <= cantelli + 2e-14
        assert cantelli < chebyshev
        rows.append((spacing, factor, exact_failure, cantelli, chebyshev))

    iid_failure = target ** n
    assert abs(rows[-1][2] - iid_failure) < 2e-4
    print("one-sided dependent order-statistic PAC bound:")
    print("  spacing   V_n       exact failure   Cantelli    Chebyshev")
    for spacing, factor, exact_failure, cantelli, chebyshev in rows:
        print(
            f"   {spacing:4.2f}   {factor:8.6f}    {exact_failure:10.8f}"
            f"    {cantelli:8.6f}    {chebyshev:8.6f}"
        )
    print(f"  iid Beta(9,1) failure: {iid_failure:.9f}")

    # A non-extreme order statistic gives a genuinely interior Chernoff
    # optimizer.  This larger panel is cheap for the O(n^2 d^2) coefficient
    # recursion but would require more than a billion binary paths.
    large_n, large_k, spacing = 30, 27, 0.1
    transition = expm(spacing * Q)
    transitions = [transition] * (large_n - 1)
    count_law = binary_count_distribution(pi, transitions, success)
    exact_failure = count_law[large_k:].sum()
    chernoff, tilt = chernoff_tail_bound(count_law, large_k)
    a = math.exp(-gamma_s * spacing)
    factor = finite_variance_factor(a, large_n)
    variance_bound = target * (1 - target) * factor / large_n
    rank_gap = large_k / large_n - target
    cantelli = variance_bound / (variance_bound + rank_gap ** 2)
    assert exact_failure <= chernoff <= cantelli
    print("polynomial Feynman--Kac count law:")
    print(f"  n={large_n}, k={large_k}, spacing={spacing:.2f},"
          f" exact {exact_failure:.9f}, Chernoff {chernoff:.9f},"
          f" Cantelli {cantelli:.9f}, tilt {tilt:.8f}")

    # The same tilted matrix has a Perron eigenvalue that gives the exact
    # long-panel logarithmic moment-generating function and large-deviation
    # rate.  Coefficient recursion supplies independent finite-n tails.
    tail_level = 0.9
    asymptotic_rate, asymptotic_tilt = perron_tail_rate(
        transition, success, tail_level)
    assert abs(perron_log_cgf(transition, success, 0.0)) < ROUND_TOL
    derivative = (perron_log_cgf(transition, success, 1e-5)
                  - perron_log_cgf(transition, success, -1e-5)) / 2e-5
    assert abs(derivative - target) < 2e-10
    (sharp_prefactor, correction, second_correction, curvature,
     boundary_factor) = perron_sharp_tail_approx(
        transition, pi, success, tail_level)
    _, fine_correction, fine_second_correction, _, _ = perron_sharp_tail_approx(
        transition, pi, success, tail_level, radius=0.07)
    assert abs(correction - fine_correction) < 2e-6
    assert abs(second_correction - fine_second_correction) < 2e-4
    rate_rows = []
    for panel_size in (30, 60, 120, 240, 480, 960):
        law = binary_count_distribution(
            pi, [transition] * (panel_size - 1), success)
        threshold = math.ceil(tail_level * panel_size)
        tail = law[threshold:].sum()
        finite_rate = -math.log(tail) / panel_size
        sharp_tail = (sharp_prefactor
                      * math.exp(-panel_size * asymptotic_rate)
                      / math.sqrt(panel_size))
        corrected_tail = sharp_tail * (1 + correction / panel_size)
        second_corrected_tail = sharp_tail * (
            1 + correction / panel_size
            + second_correction / panel_size ** 2)
        rate_rows.append((panel_size, tail, finite_rate,
                          sharp_tail, tail / sharp_tail,
                          tail / corrected_tail,
                          tail / second_corrected_tail))
    assert abs(rate_rows[-1][2] - asymptotic_rate) < 0.005
    print("Perron large-deviation rate at success fraction 0.9:")
    for (panel_size, tail, finite_rate, sharp_tail,
         sharp_ratio, corrected_ratio, second_corrected_ratio) in rate_rows:
        print(f"  n={panel_size:3d} tail {tail:.10e},"
              f" -log(tail)/n {finite_rate:.9f},"
              f" sharp/corrected/second ratios"
              f" {sharp_ratio:.9f}/{corrected_ratio:.9f}/"
              f"{second_corrected_ratio:.9f}")
    print(f"  limiting rate {asymptotic_rate:.9f},"
          f" optimizing tilt {asymptotic_tilt:.9f},"
          f" tilted variance {curvature:.9f},"
          f" boundary factor {boundary_factor:.9f},"
          f" 1/n coefficient {correction:.9f},"
          f" 1/n^2 coefficient {second_correction:.9f}")
    sharp_ratios = np.asarray([row[4] for row in rate_rows])
    corrected_ratios = np.asarray([row[5] for row in rate_rows])
    second_corrected_ratios = np.asarray([row[6] for row in rate_rows])
    assert np.all(np.diff(sharp_ratios) > 0)
    assert abs(sharp_ratios[-1] - 1) < 0.02
    assert np.all(np.diff(corrected_ratios[1:]) < 0)
    assert np.max(abs(corrected_ratios[-2:] - 1)) < 0.004
    assert abs(second_corrected_ratios[-1] - 1) < 2e-4

    # The compact-interior saddle theorem has a nontrivial boundary at the
    # mean because (1-exp(-theta))^{-1} diverges.  In the moderate-deviation
    # zone theta -> 0 but sqrt(n)*theta -> infinity, the same exact amplitude
    # remains valid and collapses to the universal sigma/(delta*sqrt(2*pi*n))
    # prefactor.  The n^(3/4) displacement makes delta asymptotic to n^(-1/4).
    moderate_target = 0.5
    moderate_quantile = brentq(
        lambda x: pi @ cdf_vector(x, scales) - moderate_target,
        0.0,
        50.0,
    )
    moderate_success = cdf_vector(moderate_quantile, scales)
    moderate_variance = perron_variance_terms(
        transition, pi, moderate_success)[0]
    moderate_rows = []
    for panel_size in (120, 240, 480, 960, 1920, 3840):
        threshold = math.ceil(
            panel_size * moderate_target + 0.5 * panel_size ** 0.75)
        level = threshold / panel_size
        delta = level - moderate_target
        law = binary_count_distribution(
            pi, [transition] * (panel_size - 1), moderate_success,
            renormalize=True)
        exact_tail = law[threshold:].sum()
        exact_rate, tilt = perron_tail_rate(
            transition, moderate_success, level)
        prefactor, _, _, _, _ = perron_sharp_tail_approx(
            transition, pi, moderate_success, level,
            radius=min(0.025, tilt / 4))
        perron_tail = (
            prefactor * math.exp(-panel_size * exact_rate)
            / math.sqrt(panel_size))
        gaussian_tail = (
            math.sqrt(moderate_variance)
            / (delta * math.sqrt(2 * math.pi * panel_size))
            * math.exp(-panel_size * exact_rate))
        moderate_rows.append((
            panel_size, delta, panel_size * tilt ** 2,
            exact_tail / perron_tail, exact_tail / gaussian_tail))

    perron_ratios = np.asarray([row[3] for row in moderate_rows])
    gaussian_ratios = np.asarray([row[4] for row in moderate_rows])
    scaled_tilts = np.asarray([row[2] for row in moderate_rows])
    assert np.all(np.diff(scaled_tilts) > 0)
    assert np.all(np.diff(perron_ratios) > 0)
    assert np.all(np.diff(gaussian_ratios) < 0)
    assert abs(perron_ratios[-1] - 1) < 0.025
    assert abs(gaussian_ratios[-1] - 1) < 0.065
    print("moderate-deviation bridge to the mean:")
    print("  n      delta       n*theta^2   exact/Perron   exact/Gaussian")
    for (panel_size, delta, scaled_tilt,
         perron_ratio, gaussian_ratio) in moderate_rows:
        print(
            f"  {panel_size:4d}  {delta:.8f}  {scaled_tilt:10.6f}"
            f"    {perron_ratio:.9f}      {gaussian_ratio:.9f}")

    # In the central zone the lattice pole can no longer be separated from
    # the Gaussian saddle.  For P(S_n >= k), summing the local expansion puts
    # the normal coordinate at k-1/2.  The first derivative of log B_nu is an
    # order-one mean shift and therefore contributes at the same n^{-1/2}
    # order as skewness.  At the next order, the ordinary second derivative
    # B_nu''(0), the fourth cumulant, cross-products and the midpoint
    # Euler--Maclaurin correction all contribute.  A nonstationary point mass
    # makes these boundary terms visible; exact coefficient tails certify the
    # signs and the O(n^{-3/2}) remainder on a fixed compact set of
    # standardized thresholds.
    central_initial = np.array([1.0, 0.0, 0.0])
    (central_mean, central_variance, central_third,
     central_fourth) = perron_cumulant_rates(
        transition, pi, success, 4)
    central_sigma = math.sqrt(central_variance)
    (central_boundary_first,
     central_boundary_log_second) = cauchy_derivatives(
        lambda theta: perron_boundary_log_cgf(
            transition, central_initial, success, theta), 2)
    central_boundary_second = (
        central_boundary_log_second + central_boundary_first ** 2
    )
    central_rows = []
    for panel_size in (120, 240, 480, 960, 1920, 3840):
        law = binary_count_distribution(
            central_initial, [transition] * (panel_size - 1), success,
            renormalize=True)
        gaussian_errors = []
        skew_only_errors = []
        corrected_errors = []
        second_corrected_errors = []
        for target_x in (-1.0, 0.0, 1.0, 2.0):
            threshold = math.ceil(
                panel_size * central_mean
                + central_sigma * math.sqrt(panel_size) * target_x + 0.5)
            x = ((threshold - 0.5 - panel_size * central_mean)
                 / (central_sigma * math.sqrt(panel_size)))
            exact_tail = law[threshold:].sum()
            gaussian_tail = ndtr(-x)
            density = math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
            skew_only = (
                gaussian_tail
                + density / math.sqrt(panel_size)
                * central_third / (6 * central_sigma ** 3)
                * (x ** 2 - 1))
            corrected = (
                skew_only
                + density / math.sqrt(panel_size)
                * central_boundary_first / central_sigma)
            hermite_1 = x
            hermite_3 = x ** 3 - 3 * x
            hermite_5 = x ** 5 - 10 * x ** 3 + 15 * x
            second_coefficient = (
                (central_boundary_second / (2 * central_sigma ** 2)
                 - 1 / (24 * central_sigma ** 2)) * hermite_1
                + (central_fourth / (24 * central_sigma ** 4)
                   + central_boundary_first * central_third
                   / (6 * central_sigma ** 4)) * hermite_3
                + central_third ** 2 / (72 * central_sigma ** 6)
                * hermite_5)
            second_corrected = (
                corrected + density * second_coefficient / panel_size)
            gaussian_errors.append(abs(exact_tail - gaussian_tail))
            skew_only_errors.append(abs(exact_tail - skew_only))
            corrected_errors.append(abs(exact_tail - corrected))
            second_corrected_errors.append(
                abs(exact_tail - second_corrected))
        central_rows.append((
            panel_size, max(gaussian_errors), max(skew_only_errors),
            max(corrected_errors), max(second_corrected_errors)))

    scaled_gaussian_errors = np.asarray([
        math.sqrt(row[0]) * row[1] for row in central_rows])
    scaled_corrected_errors = np.asarray([
        row[0] * row[3] for row in central_rows])
    scaled_second_errors = np.asarray([
        row[0] ** 1.5 * row[4] for row in central_rows])
    assert abs(central_mean - target) < 2e-14
    assert central_variance > 0
    assert central_boundary_first > 0.6
    assert central_boundary_log_second < -0.9
    assert abs(
        central_boundary_second
        - (central_boundary_log_second + central_boundary_first ** 2)
    ) < ROUND_TOL
    assert np.max(abs(
        scaled_gaussian_errors - scaled_gaussian_errors[-1])) < 0.005
    assert np.max(scaled_corrected_errors) < 0.31
    assert central_rows[-1][3] < central_rows[-1][2] / 50
    assert np.max(scaled_second_errors) < 0.58
    assert central_rows[-1][4] < central_rows[-1][3] / 25
    print("second-order central lattice Edgeworth bridge (state-zero start):")
    print("  n      first-order err   second-order err   n^(3/2)*second")
    for (panel_size, gaussian_error, skew_only_error,
         corrected_error, second_corrected_error) in central_rows:
        print(
            f"  {panel_size:4d}      {corrected_error:.9f}"
            f"       {second_corrected_error:.9f}"
            f"          {panel_size ** 1.5 * second_corrected_error:.9f}")
    print(
        f"  sigma^2 {central_variance:.12f},"
        f" kappa_3 {central_third:.12f},"
        f" kappa_4 {central_fourth:.12f},"
        f" B'(0) {central_boundary_first:.12f},"
        f" B''(0) {central_boundary_second:.12f},"
        f" (log B)''(0) {central_boundary_log_second:.12f}")

    # The exact tail is linear in the initial law.  The approximation is also
    # linear once its boundary coefficients are written as B_nu'(0) and
    # B_nu''(0)=d_nu+b_nu^2.  Hence the largest absolute remainder over the
    # entire probability simplex is attained at a point-mass start.  Check
    # every vertex independently at three panel sizes.
    uniform_initial_rows = []
    for panel_size in (120, 480, 1920):
        vertex_errors = []
        for initial_state in range(len(pi)):
            initial = np.eye(len(pi))[initial_state]
            boundary_first, boundary_log_second = cauchy_derivatives(
                lambda theta: perron_boundary_log_cgf(
                    transition, initial, success, theta), 2)
            boundary_second = (
                boundary_log_second + boundary_first ** 2
            )
            law = binary_count_distribution(
                initial, [transition] * (panel_size - 1), success,
                renormalize=True)
            for target_x in (-1.0, 0.0, 1.0, 2.0):
                threshold = math.ceil(
                    panel_size * central_mean
                    + central_sigma * math.sqrt(panel_size) * target_x
                    + 0.5)
                x = ((threshold - 0.5 - panel_size * central_mean)
                     / (central_sigma * math.sqrt(panel_size)))
                exact_tail = law[threshold:].sum()
                density = math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)
                hermite_1 = x
                hermite_3 = x ** 3 - 3 * x
                hermite_5 = x ** 5 - 10 * x ** 3 + 15 * x
                first_coefficient = (
                    boundary_first / central_sigma
                    + central_third / (6 * central_sigma ** 3)
                    * (x ** 2 - 1))
                second_coefficient = (
                    (boundary_second / (2 * central_sigma ** 2)
                     - 1 / (24 * central_sigma ** 2)) * hermite_1
                    + (central_fourth / (24 * central_sigma ** 4)
                       + boundary_first * central_third
                       / (6 * central_sigma ** 4)) * hermite_3
                    + central_third ** 2 / (72 * central_sigma ** 6)
                    * hermite_5)
                approximation = (
                    ndtr(-x)
                    + density * first_coefficient / math.sqrt(panel_size)
                    + density * second_coefficient / panel_size)
                vertex_errors.append(abs(exact_tail - approximation))
        uniform_initial_rows.append((panel_size, max(vertex_errors)))

    uniform_scaled_errors = np.asarray([
        panel_size ** 1.5 * error
        for panel_size, error in uniform_initial_rows
    ])
    assert np.all(np.isfinite(uniform_scaled_errors))
    assert uniform_initial_rows[-1][1] < uniform_initial_rows[0][1] / 12
    print("uniform-in-initial-law second-order Edgeworth certificate:")
    print("  n      worst vertex error   n^(3/2)*worst")
    for panel_size, error in uniform_initial_rows:
        print(
            f"  {panel_size:4d}       {error:.9f}"
            f"          {panel_size ** 1.5 * error:.9f}")

    variance_rate, variance_intercept, fundamental, centered = (
        perron_variance_terms(transition, pi, success))
    step = 0.003
    log_cgfs = [perron_log_cgf(transition, success, j * step)
                for j in (-2, -1, 0, 1, 2)]
    numerical_curvature = (
        -log_cgfs[4] + 16 * log_cgfs[3] - 30 * log_cgfs[2]
        + 16 * log_cgfs[1] - log_cgfs[0]) / (12 * step ** 2)
    assert abs(numerical_curvature - variance_rate) < 5e-9
    variance_rows = []
    for panel_size in (1, 2, 4, 8, 16, 30, 60):
        law = binary_count_distribution(
            pi, [transition] * (panel_size - 1), success)
        counts = np.arange(panel_size + 1)
        exact_variance = law @ counts ** 2 - (law @ counts) ** 2
        remainder = 2 * pi @ (
            centered * (
                np.linalg.matrix_power(transition, panel_size + 1)
                @ fundamental @ fundamental @ centered))
        reconstructed = (
            panel_size * variance_rate + variance_intercept + remainder)
        # exact_variance is a difference of moments of size panel_size**2, so
        # its rounding error grows with the panel.
        assert abs(exact_variance - reconstructed) < 2e-13 * panel_size
        variance_rows.append(
            (panel_size, exact_variance / panel_size, remainder))
    print("Perron curvature and discrete Green--Kubo variance:")
    print(f"  Lambda''(0) {numerical_curvature:.12f},"
          f" Green--Kubo rate {variance_rate:.12f},"
          f" intercept {variance_intercept:.12f}")
    for panel_size, variance_per_score, remainder in variance_rows[-2:]:
        print(f"  n={panel_size:2d} variance/n {variance_per_score:.12f},"
              f" exact remainder {remainder:.3e}")

    cumulant_rates = perron_cumulant_rates(transition, pi, success, 4)
    third_rate, fourth_rate = cumulant_rates[2:4]
    iid_order = 8
    iid_transition = np.outer(np.ones(len(pi)), pi)
    iid_rates = perron_cumulant_rates(
        iid_transition, pi, success, iid_order)
    iid_panel_size = 12
    iid_law = binary_count_distribution(
        pi, [iid_transition] * (iid_panel_size - 1), success)
    iid_counts = np.arange(iid_panel_size + 1, dtype=float)
    iid_moments = [1.0] + [iid_law @ iid_counts ** k
                           for k in range(1, iid_order + 1)]
    iid_cumulants = [0.0]
    for k in range(1, iid_order + 1):
        iid_cumulants.append(iid_moments[k] - sum(
            math.comb(k - 1, j - 1)
            * iid_cumulants[j] * iid_moments[k - j]
            for j in range(1, k)))
    iid_error = np.max(abs(
        iid_rates - np.asarray(iid_cumulants[1:]) / iid_panel_size))
    assert iid_error < 5e-8

    # The Perron projector gives the entire O(1) boundary correction, not
    # merely the stationary variance intercept.  Cauchy differentiation of
    # that projector amplitude is independent of the coefficient recursion.
    boundary_order = 4
    boundary_rows = []
    for label, initial in (("stationary", pi),
                           ("state zero", np.array([1.0, 0.0, 0.0]))):
        boundary = cauchy_derivatives(
            lambda theta: perron_boundary_log_cgf(
                transition, initial, success, theta),
            boundary_order)
        panel_size = 60
        law = binary_count_distribution(
            initial, [transition] * (panel_size - 1), success)
        exact = discrete_cumulants(law, boundary_order)
        finite_intercept = exact - panel_size * cumulant_rates
        error = np.max(abs(boundary - finite_intercept))
        assert error < 3e-8
        boundary_rows.append((label, boundary, error))
    assert abs(boundary_rows[0][1][0]) < 2e-11
    assert abs(boundary_rows[0][1][1] - variance_intercept) < 2e-10

    third_step = 0.005

    def third_difference(step_size):
        values = {j: perron_log_cgf(
            transition, success, j * step_size) for j in (-2, -1, 1, 2)}
        return (values[2] - 2 * values[1] + 2 * values[-1] - values[-2]
                ) / (2 * step_size ** 3)

    third_fine = third_difference(third_step)
    third_coarse = third_difference(2 * third_step)
    numerical_third = third_fine + (third_fine - third_coarse) / 3
    assert abs(numerical_third - third_rate) < 2e-8
    fourth_step = 0.03

    def fourth_difference(step_size):
        values = [perron_log_cgf(
            transition, success, j * step_size) for j in (-2, -1, 0, 1, 2)]
        return (values[0] - 4 * values[1] + 6 * values[2]
                - 4 * values[3] + values[4]) / step_size ** 4

    fourth_fine = fourth_difference(fourth_step)
    fourth_coarse = fourth_difference(2 * fourth_step)
    numerical_fourth = fourth_fine + (fourth_fine - fourth_coarse) / 3
    assert abs(numerical_fourth - fourth_rate) < 2e-6
    finite_third_rows = []
    finite_fourth_rows = []
    for panel_size in (30, 60, 120, 240, 480):
        law = binary_count_distribution(
            pi, [transition] * (panel_size - 1), success)
        counts = np.arange(panel_size + 1)
        mean = law @ counts
        centered_counts = counts - mean
        second_moment = law @ centered_counts ** 2
        finite_third = law @ centered_counts ** 3 / panel_size
        finite_fourth = (
            law @ centered_counts ** 4 - 3 * second_moment ** 2
        ) / panel_size
        finite_third_rows.append((panel_size, finite_third))
        finite_fourth_rows.append((panel_size, finite_fourth))
    assert abs(finite_third_rows[-1][1] - third_rate) < 0.002
    assert abs(finite_fourth_rows[-1][1] - fourth_rate) < 0.004

    local_rows = []
    for delta in (0.04, 0.02, 0.01, 0.005, 0.0025):
        exact_rate, _ = perron_tail_rate(
            transition, success, target + delta)
        quadratic = delta ** 2 / (2 * variance_rate)
        cubic = (quadratic
                 - third_rate * delta ** 3 / (6 * variance_rate ** 3))
        quartic = (
            cubic
            + (3 * third_rate ** 2 - variance_rate * fourth_rate)
            * delta ** 4 / (24 * variance_rate ** 5))
        local_rows.append(
            (delta, exact_rate, abs(exact_rate - quadratic),
             abs(exact_rate - cubic), abs(exact_rate - quartic)))
    deltas = np.array([row[0] for row in local_rows])
    quadratic_errors = np.array([row[2] for row in local_rows])
    cubic_errors = np.array([row[3] for row in local_rows])
    quartic_errors = np.array([row[4] for row in local_rows])
    quadratic_order = np.polyfit(
        np.log(deltas), np.log(quadratic_errors), 1)[0]
    cubic_order = np.polyfit(
        np.log(deltas), np.log(cubic_errors), 1)[0]
    quartic_order = np.polyfit(
        np.log(deltas), np.log(quartic_errors), 1)[0]
    assert (quadratic_order > 3.0 and cubic_order > 4.0
            and quartic_order > 5.0)
    print("Perron cumulant recursion and local rate expansion:")
    print(f"  iid recursion through order {iid_order}:"
          f" maximum error {iid_error:.3e}")
    print(f"  Lambda'''(0) {numerical_third:.12f},"
          f" perturbation rate {third_rate:.12f},"
          f" n=480 rate {finite_third_rows[-1][1]:.12f}")
    print(f"  Lambda''''(0) {numerical_fourth:.12f},"
          f" perturbation rate {fourth_rate:.12f},"
          f" n=480 rate {finite_fourth_rows[-1][1]:.12f}")
    print(f"  quadratic/cubic/quartic residual orders"
          f" {quadratic_order:.6f}/{cubic_order:.6f}/{quartic_order:.6f}")
    print(f"  delta=0.02 exact {local_rows[1][1]:.12f},"
          f" quadratic error {local_rows[1][2]:.3e},"
          f" cubic error {local_rows[1][3]:.3e},"
          f" quartic error {local_rows[1][4]:.3e}")
    print("Perron boundary cumulants through order four:")
    for label, boundary, error in boundary_rows:
        formatted = ", ".join(f"{value:.12f}" for value in boundary)
        print(f"  {label}: ({formatted}), n=60 max error {error:.3e}")


def binary_count_distribution(initial, transitions, success, renormalize=False):
    """Count law for finite-state binary emissions on any deterministic grid.

    If ``D(z)=diag(1-success+z*success)``, the returned coefficients are those
    of ``initial D(z) P_1 D(z) ... P_{n-1} D(z) 1``.  The recursion stores one
    state row for each possible count and so avoids binary-path enumeration.
    """
    initial = np.asarray(initial)
    success = np.asarray(success)
    n = len(transitions) + 1
    states = np.zeros((n + 1, len(initial)))
    states[0] = initial * (1 - success)
    states[1] = initial * success
    for observation, transition in enumerate(transitions, start=1):
        predicted = states @ transition
        updated = np.zeros_like(states)
        updated[:observation + 1] += (
            predicted[:observation + 1] * (1 - success))
        updated[1:observation + 2] += (
            predicted[:observation + 1] * success)
        states = updated
    count_law = states.sum(axis=1)
    total_mass = count_law.sum()
    if renormalize:
        assert abs(total_mass - 1) < 2e-12
        count_law /= total_mass
    else:
        assert abs(total_mass - 1) < 2e-14
    assert count_law.min() > -ROUND_TOL
    return count_law


def binary_count_pgf(initial, transitions, success, z):
    """Evaluate the finite-state count probability-generating matrix."""
    emission = 1 - success + z * success
    forward = initial * emission
    for transition in transitions:
        forward = (forward @ transition) * emission
    return forward.sum()


def perron_log_cgf(transition, success, theta):
    """Long-panel log MGF per observation from the tilted Perron root."""
    return float(perron_log_cgf_analytic(transition, success, theta).real)


def perron_log_cgf_analytic(transition, success, theta):
    """Analytic Perron branch near a real tilt, including complex arguments."""
    emission = 1 - success + np.exp(theta) * success
    tilted = transition @ np.diag(emission)
    eigenvalues = np.linalg.eigvals(tilted)
    rho = eigenvalues[np.argmax(abs(eigenvalues))]
    return np.log(rho)


def perron_boundary_log_cgf(transition, initial, success, theta):
    """Order-one log-MGF boundary term for a regular binary panel.

    The transpose, rather than the conjugate transpose, is intentional:
    this function is evaluated at complex arguments for Cauchy derivatives.
    The returned invariant is ``log(A_nu(theta))-log(rho(theta))`` where
    ``A_nu=(nu D r)(ell^T 1)`` and ``ell^T r=1``.
    """
    emission = 1 - success + np.exp(theta) * success
    tilted = transition @ np.diag(emission)
    eigenvalues, right_vectors = np.linalg.eig(tilted)
    index = np.argmax(abs(eigenvalues))
    rho = eigenvalues[index]
    right = right_vectors[:, index]
    left_values, left_vectors = np.linalg.eig(tilted.T)
    left = left_vectors[:, np.argmin(abs(left_values - rho))]
    left /= left @ right
    amplitude = ((initial * emission) @ right) * (left @ np.ones(len(initial)))
    return np.log(amplitude) - np.log(rho)


def cauchy_derivatives(function, order, center=0.0,
                       radius=0.025, points=128):
    """Derivatives at ``center`` of an analytic scalar by Cauchy FFT."""
    angles = 2 * np.pi * np.arange(points) / points
    values = np.asarray([function(center + radius * np.exp(1j * angle))
                         for angle in angles])
    derivatives = []
    for k in range(1, order + 1):
        coefficient = np.mean(values * np.exp(-1j * k * angles)) / radius ** k
        derivatives.append(math.factorial(k) * coefficient.real)
    return np.asarray(derivatives)


def discrete_cumulants(probabilities, order):
    """Cumulants through ``order`` of an integer-valued coefficient law."""
    values = np.arange(len(probabilities), dtype=float)
    moments = [1.0] + [probabilities @ values ** k
                       for k in range(1, order + 1)]
    cumulants = [0.0]
    for k in range(1, order + 1):
        cumulants.append(moments[k] - sum(
            math.comb(k - 1, j - 1) * cumulants[j] * moments[k - j]
            for j in range(1, k)))
    return np.asarray(cumulants[1:])


def perron_variance_terms(transition, pi, success):
    """Green--Kubo variance rate and intercept for regular binary panels."""
    dimension = len(pi)
    projector = np.ones((dimension, 1)) @ pi.reshape(1, dimension)
    fundamental = np.linalg.inv(
        np.eye(dimension) - transition + projector)
    probability = pi @ success
    centered = success - probability
    variance_rate = (
        probability * (1 - probability)
        + 2 * pi @ (centered * (transition @ fundamental @ centered)))
    variance_intercept = -2 * pi @ (
        centered * (
            transition @ fundamental @ fundamental @ centered))
    return variance_rate, variance_intercept, fundamental, centered


def perron_cumulant_rates(transition, pi, success, order):
    """Cumulants per observation through ``order`` by Perron recursion."""
    assert order >= 1
    dimension = len(pi)
    identity = np.eye(dimension)
    one = np.ones(dimension)
    projector = np.outer(one, pi)
    fundamental = np.linalg.inv(identity - transition + projector)
    derivative = transition @ np.diag(success)
    vectors = [one]
    eigenvalue_derivatives = [1.0]
    cumulants = [0.0]
    for k in range(1, order + 1):
        eigenvalue_derivative = sum(
            math.comb(k, j) * pi @ (derivative @ vectors[k - j])
            for j in range(1, k + 1))
        eigenvalue_derivatives.append(eigenvalue_derivative)
        rhs = sum(
            math.comb(k, j)
            * (derivative - eigenvalue_derivatives[j] * identity)
            @ vectors[k - j]
            for j in range(1, k + 1))
        vector = fundamental @ rhs
        vector -= one * (pi @ vector)
        vectors.append(vector)
        cumulant = eigenvalue_derivatives[k] - sum(
            math.comb(k - 1, j - 1)
            * cumulants[j] * eigenvalue_derivatives[k - j]
            for j in range(1, k))
        cumulants.append(cumulant)
    return np.asarray(cumulants[1:])


def perron_tail_rate(transition, success, level):
    """Legendre rate for an upper-tail success fraction above its mean."""
    assert 0 < level < 1

    def negative_rate(theta):
        return perron_log_cgf(transition, success, theta) - theta * level

    result = minimize_scalar(negative_rate, bounds=(0.0, 50.0),
                             method="bounded", options={"xatol": 1e-13})
    assert result.success and result.x > 0
    return -result.fun, result.x


def perron_sharp_tail_approx(transition, initial, success, level,
                             radius=0.05):
    """Return the lattice prefactor and its first two relative corrections."""
    _, theta = perron_tail_rate(transition, success, level)
    lambda_derivatives = cauchy_derivatives(
        lambda z: perron_log_cgf_analytic(transition, success, z),
        6, center=theta, radius=radius, points=256)
    curvature, third, fourth, fifth, sixth = lambda_derivatives[1:6]
    log_boundary = perron_boundary_log_cgf(
        transition, initial, success, theta).real
    boundary_factor = math.exp(log_boundary)

    def tail_amplitude(z):
        return (np.exp(perron_boundary_log_cgf(
            transition, initial, success, z)) / (1 - np.exp(-z)))

    amplitude = tail_amplitude(theta).real
    amplitude_derivatives = cauchy_derivatives(
        tail_amplitude, 4, center=theta, radius=radius, points=256)
    first, second, third_amplitude, fourth_amplitude = amplitude_derivatives
    correction = (
        -second / (2 * curvature * amplitude)
        + first * third / (2 * curvature ** 2 * amplitude)
        + fourth / (8 * curvature ** 2)
        - 5 * third ** 2 / (24 * curvature ** 3))
    second_correction = (
        385 * third ** 4
        + 144 * fourth_amplitude * curvature ** 4 / amplitude
        - 210 * third ** 2 * curvature
        * (3 * fourth + 4 * first * third / amplitude)
        - 24 * curvature ** 3
        * (sixth + 6 * first * fifth / amplitude
           + 15 * second * fourth / amplitude
           + 20 * third_amplitude * third / amplitude)
        + 21 * curvature ** 2
        * (8 * third * fifth + 5 * fourth ** 2
           + 40 * first * third * fourth / amplitude
           + 40 * second * third ** 2 / amplitude)
    ) / (1152 * curvature ** 6)
    prefactor = amplitude / math.sqrt(2 * math.pi * curvature)
    return (prefactor, correction, second_correction, curvature,
            boundary_factor)


def chernoff_tail_bound(count_law, k):
    """Return inf_{theta>0} E exp(theta(S-k)) and its optimizing tilt."""
    count_law = np.asarray(count_law)
    n = len(count_law) - 1
    assert 0 <= k <= n
    if k == 0:
        return 1.0, 0.0
    if k == n:
        return count_law[-1], math.inf
    positive = count_law > 0
    counts = np.arange(n + 1)[positive]
    log_probabilities = np.log(count_law[positive])

    def log_bound(theta):
        return logsumexp(log_probabilities + theta * (counts - k))

    result = minimize_scalar(log_bound, bounds=(0.0, 50.0), method="bounded",
                             options={"xatol": 1e-13})
    assert result.success
    return min(1.0, math.exp(result.fun)), result.x


def aggregate_binary_count_law(law, n):
    """Aggregate a lexicographic 2**n binary-path law by success count."""
    counts = np.asarray([
        sum(pattern) for pattern in itertools.product((0, 1), repeat=n)
    ])
    return np.bincount(counts, weights=law, minlength=n + 1)


def binary_panel_law(initial, P, success, n):
    """Exact law of n binary emissions from a finite-state hidden chain."""
    law = []
    for pattern in itertools.product((0, 1), repeat=n):
        forward = initial.copy()
        for t, bit in enumerate(pattern):
            emission = success if bit else 1 - success
            forward = forward * emission
            if t + 1 < n:
                forward = forward @ P
        law.append(forward.sum())
    law = np.asarray(law)
    assert abs(law.sum() - 1) < 2e-14
    return law


def binary_irregular_panel_law(initial, Q, success, times):
    """Exact binary-emission law on a strictly increasing time grid."""
    times = np.asarray(times)
    assert times.ndim == 1 and len(times) >= 1
    assert abs(times[0]) < ROUND_TOL and np.all(np.diff(times) > 0)
    transitions = [expm((times[j + 1] - times[j]) * Q)
                   for j in range(len(times) - 1)]
    law = []
    for pattern in itertools.product((0, 1), repeat=len(times)):
        forward = initial.copy()
        for j, bit in enumerate(pattern):
            emission = success if bit else 1 - success
            forward = forward * emission
            if j + 1 < len(times):
                forward = forward @ transitions[j]
        law.append(forward.sum())
    law = np.asarray(law)
    assert abs(law.sum() - 1) < 2e-14
    return law


def panel_moments_from_law(law, n):
    """Mean and variance of the empirical success rate from a panel law."""
    values = np.asarray([sum(pattern) / n
                         for pattern in itertools.product((0, 1), repeat=n)])
    mean = law @ values
    variance = law @ values ** 2 - mean ** 2
    return mean, variance


def transient_panel_moments(initial, P, success, n):
    """Exact empirical-CDF moments from transient Markov reward sums."""
    marginals = []
    second_moment = 0.0
    distribution = initial.copy()
    powers = [np.linalg.matrix_power(P, lag) for lag in range(n)]
    for left in range(n):
        marginal = distribution @ success
        marginals.append(marginal)
        second_moment += marginal
        for right in range(left + 1, n):
            joint = distribution @ (success * (powers[right - left] @ success))
            second_moment += 2 * joint
        distribution = distribution @ P
    mean = sum(marginals) / n
    variance = second_moment / n ** 2 - mean ** 2
    return mean, variance


def stationary_irregular_variance(Q, pi, success, times):
    """Exact empirical-CDF variance on an arbitrary stationary time grid."""
    times = np.asarray(times)
    n = len(times)
    p = pi @ success
    centered = success - p
    variance = p * (1 - p) / n
    for left in range(n):
        for right in range(left + 1, n):
            transition = expm((times[right] - times[left]) * Q)
            covariance = pi @ (centered * (transition @ centered))
            variance += 2 * covariance / n ** 2
    return variance


def stationary_weighted_irregular_variance(Q, pi, success, times, weights):
    """Exact variance of a weighted empirical CDF on a stationary grid."""
    times = np.asarray(times)
    weights = np.asarray(weights)
    p = pi @ success
    centered = success - p
    variance = p * (1 - p) * (weights @ weights)
    for left in range(len(times)):
        for right in range(left + 1, len(times)):
            transition = expm((times[right] - times[left]) * Q)
            covariance = pi @ (centered * (transition @ centered))
            variance += 2 * weights[left] * weights[right] * covariance
    return variance


def optimal_exponential_weights(times, rate):
    """BLUE weights for covariance exp(-rate*abs(t_i-t_j))."""
    times = np.asarray(times)
    rho = np.exp(-rate * np.diff(times))
    precision_ones = np.empty(len(times))
    precision_ones[0] = 1 / (1 + rho[0])
    precision_ones[-1] = 1 / (1 + rho[-1])
    precision_ones[1:-1] = (1 / (1 + rho[:-1])
                             - rho[1:] / (1 + rho[1:]))
    information = 1 + np.sum((1 - rho) / (1 + rho))
    weights = precision_ones / information
    return weights, information


def irregular_panel_checks(Q, pi, gamma_s, success):
    """Check arbitrary-grid variance and start-up-bias bounds."""
    times = np.array([0.0, 0.07, 0.31, 0.9, 1.8])
    n = len(times)
    p = pi @ success
    exact_variance = stationary_irregular_variance(
        Q, pi, success, times)
    pairwise_factor = (1 + 2 / n * sum(
        math.exp(-gamma_s * (times[right] - times[left]))
        for left in range(n) for right in range(left + 1, n)))
    variance_bound = p * (1 - p) / n * pairwise_factor
    assert exact_variance <= variance_bound + 2e-14
    enumerated = binary_irregular_panel_law(pi, Q, success, times)
    transitions = [expm((times[j + 1] - times[j]) * Q)
                   for j in range(n - 1)]
    count_law = binary_count_distribution(pi, transitions, success)
    assert np.max(abs(count_law - aggregate_binary_count_law(
        enumerated, n))) < ROUND_TOL
    enumerated_mean, enumerated_variance = panel_moments_from_law(
        enumerated, n)
    assert abs(enumerated_mean - p) < ROUND_TOL
    assert abs(enumerated_variance - exact_variance) < ROUND_TOL

    initial = np.zeros(len(pi))
    initial[0] = 1
    burnin = 0.2
    exact_mean = sum(initial @ expm((burnin + t) * Q) @ success
                     for t in times) / n
    density_norm = math.sqrt(1 / pi[0] - 1)
    score_norm = math.sqrt(pi @ (success - p) ** 2)
    bias_bound = (density_norm * score_norm / n
                  * sum(math.exp(-gamma_s * (burnin + t))
                        for t in times))
    assert abs(exact_mean - p) <= bias_bound + 2e-14

    # A symmetric binary chain attains the pairwise bound and the start-up
    # bias bound on every grid.  It also supplies a counterexample to replacing
    # an irregular grid by its average adjacent spacing.
    sharp_Q = np.array([[-0.5, 0.5], [0.5, -0.5]])
    sharp_pi = np.array([0.5, 0.5])
    sharp_success = np.array([0.0, 1.0])
    clustered_times = np.array([0.0, 0.01, 0.02, 0.03, 6.0])
    sharp_n = len(clustered_times)
    sharp_exact = stationary_irregular_variance(
        sharp_Q, sharp_pi, sharp_success, clustered_times)
    sharp_factor = (1 + 2 / sharp_n * sum(
        math.exp(-(clustered_times[right] - clustered_times[left]))
        for left in range(sharp_n)
        for right in range(left + 1, sharp_n)))
    sharp_bound = 0.25 / sharp_n * sharp_factor
    assert abs(sharp_exact - sharp_bound) < ROUND_TOL
    sharp_law = binary_irregular_panel_law(
        sharp_pi, sharp_Q, sharp_success, clustered_times)
    _, sharp_enumerated_variance = panel_moments_from_law(
        sharp_law, sharp_n)
    assert abs(sharp_enumerated_variance - sharp_exact) < ROUND_TOL

    average_spacing = ((clustered_times[-1] - clustered_times[0])
                       / (sharp_n - 1))
    average_factor = finite_variance_factor(
        math.exp(-average_spacing), sharp_n)
    average_surrogate = 0.25 / sharp_n * average_factor
    assert sharp_exact > 2 * average_surrogate

    sharp_burnin = 0.2
    sharp_initial = np.array([1.0, 0.0])
    sharp_mean = sum(
        sharp_initial @ expm((sharp_burnin + t) * sharp_Q)
        @ sharp_success for t in clustered_times) / sharp_n
    sharp_bias_bound = (0.5 / sharp_n * sum(
        math.exp(-(sharp_burnin + t)) for t in clustered_times))
    assert abs(abs(sharp_mean - 0.5) - sharp_bias_bound) < ROUND_TOL

    # The sharp exponential envelope is itself a covariance matrix.  Its
    # inverse is tridiagonal, so the minimum-variance weights and information
    # are explicit and remain strictly positive on every ordered grid.
    optimal_weights, information = optimal_exponential_weights(
        clustered_times, rate=1.0)
    sharp_kernel = np.exp(-np.abs(
        clustered_times[:, None] - clustered_times[None, :]))
    direct_weights = (np.linalg.solve(sharp_kernel,
                                      np.ones(sharp_n)))
    direct_weights /= direct_weights.sum()
    assert np.max(abs(optimal_weights - direct_weights)) < 2e-14
    assert np.min(optimal_weights) > 0
    assert abs(optimal_weights.sum() - 1) < ROUND_TOL
    optimal_factor = optimal_weights @ sharp_kernel @ optimal_weights
    assert abs(optimal_factor - 1 / information) < ROUND_TOL
    sharp_weighted_variance = stationary_weighted_irregular_variance(
        sharp_Q, sharp_pi, sharp_success, clustered_times, optimal_weights)
    assert abs(sharp_weighted_variance - 0.25 / information) < ROUND_TOL
    equal_weight_variance = sharp_exact
    assert sharp_weighted_variance < equal_weight_variance

    nonrev_weights, nonrev_information = optimal_exponential_weights(
        times, rate=gamma_s)
    nonrev_weighted_variance = stationary_weighted_irregular_variance(
        Q, pi, success, times, nonrev_weights)
    assert nonrev_weighted_variance <= (
        p * (1 - p) / nonrev_information + 2e-14)

    weight_formula_error = 0.0
    grid_rng = np.random.default_rng(20261010)
    for grid_size in range(2, 13):
        for _ in range(8):
            random_times = np.r_[0.0, np.cumsum(
                grid_rng.uniform(0.005, 2.0, size=grid_size - 1))]
            random_rate = grid_rng.uniform(0.1, 4.0)
            formula_weights, formula_information = (
                optimal_exponential_weights(random_times, random_rate))
            random_kernel = np.exp(-random_rate * np.abs(
                random_times[:, None] - random_times[None, :]))
            solved = np.linalg.solve(random_kernel, np.ones(grid_size))
            solved_information = solved.sum()
            solved /= solved_information
            weight_formula_error = max(
                weight_formula_error,
                np.max(abs(formula_weights - solved)),
                abs(formula_information - solved_information))
            assert np.min(formula_weights) > 0
    assert weight_formula_error < 2e-13

    # Once the weights have been optimized, observation-time design reduces
    # to maximizing 1 + sum tanh(rate * gap / 2).  Strict concavity of tanh on
    # the positive half-line makes equal spacing the unique fixed-span
    # optimum.  Random designs check the formula, and a dense equal grid
    # checks the resulting endpoint atoms and continuous interior mass.
    design_rate = 1.0
    total_span = 6.0
    design_n = 8
    equal_times = np.linspace(0.0, total_span, design_n)
    equal_weights, equal_information = optimal_exponential_weights(
        equal_times, design_rate)
    equal_information_formula = 1 + (design_n - 1) * math.tanh(
        design_rate * total_span / (2 * (design_n - 1)))
    assert abs(equal_information - equal_information_formula) < ROUND_TOL

    design_rng = np.random.default_rng(20261011)
    random_gaps = design_rng.dirichlet(
        np.ones(design_n - 1), size=4096) * total_span
    random_information = 1 + np.tanh(
        design_rate * random_gaps / 2).sum(axis=1)
    assert np.max(random_information) < equal_information

    dense_n = 4097
    dense_times = np.linspace(0.0, total_span, dense_n)
    dense_weights, dense_information = optimal_exponential_weights(
        dense_times, design_rate)
    exposure = design_rate * total_span
    limiting_information = 1 + exposure / 2
    limiting_endpoint_weight = 1 / (2 + exposure)
    limiting_interior_mass = exposure / (2 + exposure)
    assert abs(dense_information - limiting_information) < 6e-7
    assert abs(dense_weights[0] - limiting_endpoint_weight) < 1e-4
    assert abs(dense_weights[-1] - limiting_endpoint_weight) < 1e-4
    assert abs(dense_weights[1:-1].sum()
               - limiting_interior_mass) < 2e-4
    assert abs(equal_weights.sum() - 1) < ROUND_TOL

    # The continuous-observation information is a genuine finite-sample
    # ceiling.  If N=n-1 is the number of equal subintervals and L is total
    # mixing exposure, then I_N=1+N*tanh(L/(2N)) increases to 1+L/2.  The
    # elementary global inequality 0 <= x-tanh(x) <= x^3/3 gives an
    # asymptotically sharp error certificate for every L and N.
    budget_ratios = []
    variance_ratios = []
    for budget_exposure in (0.1, 0.5, 2.0, 8.0, 32.0):
        continuous_information = 1 + budget_exposure / 2
        previous_information = 0.0
        for intervals in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                          1024):
            budget_x = np.longdouble(budget_exposure) / (2 * intervals)
            finite_information = float(
                1 + intervals * np.tanh(budget_x))
            assert finite_information > previous_information
            assert finite_information <= continuous_information + 2e-14
            information_gap = float(
                intervals * (budget_x - np.tanh(budget_x)))
            information_bound = (budget_exposure ** 3
                                 / (24 * intervals ** 2))
            assert information_gap <= information_bound + 2e-14
            finite_variance = 1 / finite_information
            continuous_variance = 1 / continuous_information
            variance_gap = (information_gap
                            / (finite_information
                               * continuous_information))
            assert abs(variance_gap - (finite_variance
                                       - continuous_variance)) < 2e-14
            budget_variance_bound = (information_bound
                                     / (finite_information
                                        * continuous_information))
            assert 0 <= variance_gap <= budget_variance_bound + 2e-14
            previous_information = finite_information
        budget_ratios.append(information_gap / information_bound)
        variance_ratios.append(variance_gap / budget_variance_bound)
    assert min(budget_ratios) > 0.999
    assert min(variance_ratios) > 0.999

    # Sampling cost selects a finite optimum.  The continuous extension
    # f(N)=N*tanh(L/(2N)) is strictly increasing and strictly concave, since
    # f''(N)=-2*x**2*sech(x)**2*tanh(x)/N for x=L/(2N).  Hence the information
    # increments decrease, as do the normalized variance reductions
    # d_N=1/I_N-1/I_(N+1).  The exact cost-optimal interval count is the first
    # N with d_N <= c.  Its small-cost cube-root law follows from
    # d_N ~ L**3 / (12*I_infinity**2*N**3).
    cost_rows = []
    maximum_threshold_error = 0.0
    maximum_asymptotic_error = 0.0
    for cost_exposure in (0.5, 2.0, 8.0, 32.0):
        cost_limit_information = np.longdouble(1 + cost_exposure / 2)

        def cost_information(intervals):
            cost_x = np.longdouble(cost_exposure) / (2 * intervals)
            return np.longdouble(1) + intervals * np.tanh(cost_x)

        previous_reduction = np.longdouble("inf")
        for intervals in range(1, 128):
            reduction = (1 / cost_information(intervals)
                         - 1 / cost_information(intervals + 1))
            assert 0 < reduction < previous_reduction
            previous_reduction = reduction

        for observation_cost in (1e-6, 1e-8, 1e-10):
            intervals = 1
            while (1 / cost_information(intervals)
                   - 1 / cost_information(intervals + 1)
                   > observation_cost):
                intervals += 1
            next_reduction = (1 / cost_information(intervals)
                              - 1 / cost_information(intervals + 1))
            if intervals > 1:
                previous_reduction = (1 / cost_information(intervals - 1)
                                      - 1 / cost_information(intervals))
                maximum_threshold_error = max(
                    maximum_threshold_error,
                    float(max(observation_cost - previous_reduction,
                              next_reduction - observation_cost, 0)))
                assert previous_reduction >= observation_cost
            assert next_reduction <= observation_cost

            asymptotic_intervals = (
                cost_exposure ** 3
                / (12 * observation_cost * float(cost_limit_information) ** 2)
            ) ** (1 / 3)
            relative_error = abs(intervals / asymptotic_intervals - 1)
            if observation_cost == 1e-10:
                maximum_asymptotic_error = max(
                    maximum_asymptotic_error, relative_error)
                cost_rows.append((cost_exposure, intervals,
                                  asymptotic_intervals, next_reduction))
        assert intervals < 10000
    assert maximum_threshold_error == 0
    assert maximum_asymptotic_error < 0.002

    print("irregular nonreversible panel:",
          f"exact variance {exact_variance:.8f},",
          f"bound {variance_bound:.8f}, factor {pairwise_factor:.8f}")
    print("irregular-grid saturation and average-spacing counterexample:",
          f"exact {sharp_exact:.8f}, pairwise bound {sharp_bound:.8f},",
          f"average-spacing surrogate {average_surrogate:.8f},",
          f"ratio {sharp_exact / average_surrogate:.8f}")
    print("irregular-grid start-up bias:",
          f"nonreversible {exact_mean - p:+.8f} <= {bias_bound:.8f};",
          f"sharp two-state {sharp_mean - 0.5:+.8f}")
    print("optimal irregular-grid weights:",
          np.array2string(optimal_weights, precision=8),
          f"information {information:.8f},",
          f"variance {sharp_weighted_variance:.8f} vs",
          f"equal {equal_weight_variance:.8f},",
          f"random-grid formula error {weight_formula_error:.2e}")
    print("joint time-and-weight design:",
          f"equal-grid information {equal_information:.8f} vs",
          f"best of 4096 random grids {np.max(random_information):.8f};",
          f"dense information {dense_information:.8f} ->",
          f"{limiting_information:.8f}, endpoint weights",
          f"{dense_weights[0]:.8f} -> {limiting_endpoint_weight:.8f},",
          f"interior mass {dense_weights[1:-1].sum():.8f} ->",
          f"{limiting_interior_mass:.8f}")
    print("observation-budget information bound:",
          "last-gap/bound ratios",
          np.array2string(np.array(budget_ratios), precision=9),
          "for exposures [0.1, 0.5, 2, 8, 32];",
          f"minimum variance-gap ratio {min(variance_ratios):.9f}")
    print("cost-optimal observation counts at cost 1e-10:",
          "; ".join(
              f"L={cost_exposure:g}: {intervals} "
              f"(cube-root {asymptotic_intervals:.3f}, "
              f"next gain {float(next_reduction):.3e})"
              for (cost_exposure, intervals, asymptotic_intervals,
                   next_reduction) in cost_rows),
          f"; maximum relative count error {maximum_asymptotic_error:.3e}")


def transient_panel_checks(Q, pi, gamma_s, success):
    """Check transient panel moments and the within-panel relaxation bound."""
    initial = np.array([1.0, 0.0, 0.0])
    density_norm = math.sqrt(1 / pi[0] - 1)
    centered = success - pi @ success
    score_norm = math.sqrt(pi @ centered ** 2)
    spacing = 0.3
    n = 4
    P = expm(spacing * Q)
    a = math.exp(-gamma_s * spacing)
    averaging_factor = (1 - a ** n) / (n * (1 - a))
    rows = []
    for scaled_burnin in (0.0, 0.25, 0.5, 1.0):
        burned = initial @ expm(scaled_burnin * Q)
        law = binary_panel_law(burned, P, success, n)
        enumerated_mean, enumerated_variance = panel_moments_from_law(law, n)
        exact_mean, exact_variance = transient_panel_moments(
            burned, P, success, n)
        assert abs(enumerated_mean - exact_mean) < ROUND_TOL
        assert abs(enumerated_variance - exact_variance) < ROUND_TOL
        bias = exact_mean - pi @ success
        score_bound = (density_norm * score_norm
                       * math.exp(-gamma_s * scaled_burnin)
                       * averaging_factor)
        assert abs(bias) <= score_bound + 2e-14
        rows.append((scaled_burnin, bias, exact_variance, score_bound))

    # A symmetric two-state chain and success vector (0,1) attain the
    # universal 1/2-prefactor bound for every n, spacing and burn-in.
    sharp_Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
    sharp_pi = np.array([0.5, 0.5])
    sharp_initial = np.array([1.0, 0.0])
    sharp_success = np.array([0.0, 1.0])
    sharp_n, sharp_spacing, sharp_burnin = 7, 0.35, 0.2
    sharp_P = expm(sharp_spacing * sharp_Q)
    sharp_burned = sharp_initial @ expm(sharp_burnin * sharp_Q)
    sharp_mean, sharp_variance = transient_panel_moments(
        sharp_burned, sharp_P, sharp_success, sharp_n)
    sharp_a = math.exp(-2 * sharp_spacing)
    sharp_factor = (1 - sharp_a ** sharp_n) / (sharp_n * (1 - sharp_a))
    sharp_bound = 0.5 * math.exp(-2 * sharp_burnin) * sharp_factor
    assert abs(abs(sharp_mean - sharp_pi @ sharp_success) - sharp_bound) < ROUND_TOL
    sharp_law = binary_panel_law(sharp_burned, sharp_P, sharp_success, sharp_n)
    law_mean, law_variance = panel_moments_from_law(sharp_law, sharp_n)
    assert abs(law_mean - sharp_mean) < ROUND_TOL
    assert abs(law_variance - sharp_variance) < ROUND_TOL

    print("transient empirical-CDF moments:")
    for scaled_burnin, bias, variance, score_bound in rows:
        print(f"  mb={scaled_burnin:4.2f} bias {bias:+.8f},"
              f" variance {variance:.8f}, bound {score_bound:.8f}")
    print("  sharp two-state bias:",
          f"exact {sharp_mean - 0.5:+.8f}, bound {sharp_bound:.8f}")


def burnin_transfer_checks(Q, pi, gamma_s, success):
    """Check TV transfer from a nonstationary start to a score panel."""
    initial = np.array([1.0, 0.0, 0.0])
    density_norm = math.sqrt(1 / pi[0] - 1)
    panel_transition = expm(0.3 * Q)
    stationary_panel = binary_panel_law(pi, panel_transition, success, 4)
    rows = []
    for scaled_burnin in (0.0, 0.25, 0.5, 1.0):
        burned = initial @ expm(scaled_burnin * Q)
        state_tv = 0.5 * np.abs(burned - pi).sum()
        panel = binary_panel_law(burned, panel_transition, success, 4)
        panel_tv = 0.5 * np.abs(panel - stationary_panel).sum()
        l2_bound = 0.5 * density_norm * math.exp(-gamma_s * scaled_burnin)
        assert panel_tv <= state_tv + 2e-14
        assert state_tv <= l2_bound + 2e-14
        rows.append((scaled_burnin, panel_tv, state_tv, l2_bound))

    tolerance = 0.01
    required_scaled_burnin = max(
        0.0, math.log(density_norm / (2 * tolerance)) / gamma_s)
    burned = initial @ expm(required_scaled_burnin * Q)
    required_state_tv = 0.5 * np.abs(burned - pi).sum()
    assert required_state_tv <= tolerance + 2e-14
    print("nonstationary panel burn-in transfer:")
    for scaled_burnin, panel_tv, state_tv, l2_bound in rows:
        print(f"  mb={scaled_burnin:4.2f} panel TV {panel_tv:.8f},"
              f" state TV {state_tv:.8f}, L2 bound {l2_bound:.8f}")
    print("  1% L2 burn-in certificate:",
          f"mb >= {required_scaled_burnin:.8f},"
          f" exact state TV {required_state_tv:.8f}")
    transient_panel_checks(Q, pi, gamma_s, success)


def bernoulli_product_tv(trials, correlation):
    """Exact TV between biased and fair Bernoulli product laws.

    The likelihood ratio is increasing in the number of successes, so total
    variation is the difference of the two upper tails at its crossing.
    """
    log_plus = math.log1p(correlation)
    log_minus = math.log1p(-correlation)
    crossing = -trials * log_minus / (log_plus - log_minus)
    threshold = math.ceil(crossing - 2e-14)
    biased_tail = binom.sf(threshold - 1, trials,
                           (1 + correlation) / 2)
    fair_tail = binom.sf(threshold - 1, trials, 0.5)
    return biased_tail - fair_tail


def two_level_bernoulli_product_tv(trials_a, correlation_a,
                                   trials_b, correlation_b):
    """Exact TV for two groups of biased signs against fair signs.

    Aggregating each group by its success count evaluates the heterogeneous
    product experiment without enumerating 2**(trials_a + trials_b) signs.
    """
    counts_a = np.arange(trials_a + 1)
    counts_b = np.arange(trials_b + 1)
    fair_a = binom.pmf(counts_a, trials_a, 0.5)
    fair_b = binom.pmf(counts_b, trials_b, 0.5)
    log_likelihood_a = (
        counts_a * math.log1p(correlation_a)
        + (trials_a - counts_a) * math.log1p(-correlation_a)
    )
    log_likelihood_b = (
        counts_b * math.log1p(correlation_b)
        + (trials_b - counts_b) * math.log1p(-correlation_b)
    )
    reference = fair_a[:, None] * fair_b[None, :]
    likelihood = np.exp(
        log_likelihood_a[:, None] + log_likelihood_b[None, :]
    )
    return 0.5 * np.sum(reference * np.abs(likelihood - 1))


def biased_sign_product_affinity(correlations):
    """Hellinger affinity of biased independent signs against fair signs."""
    correlations = np.asarray(correlations)
    assert np.all(np.abs(correlations) <= 1)
    factors = (
        np.sqrt(1 + correlations) + np.sqrt(1 - correlations)
    ) / 2
    return math.exp(np.log(factors).sum())


def burned_bernoulli_product_tv(trials, correlation, initial_bias):
    """Exact TV for one biased initial sign and biased transition signs.

    Under the iid stationary reference law, the initial sign and all
    transition signs are independent and fair.  Under the burned-in Markov
    law they remain independent, with means initial_bias and correlation.
    Aggregating the transition signs by their success count avoids enumerating
    2**(trials + 1) paths.
    """
    counts = np.arange(trials + 1)
    biased = binom.pmf(counts, trials, (1 + correlation) / 2)
    fair = binom.pmf(counts, trials, 0.5)
    return 0.25 * sum(
        np.abs((1 + sign * initial_bias) * biased - fair).sum()
        for sign in (-1, 1)
    )


def absolute_scaled_lognormal_deviation(scale, information):
    """E|scale*exp(sqrt(c)G-c/2)-1| for c=information."""
    if information == 0:
        return abs(scale - 1)
    if scale == 0:
        return 1.0
    root_information = math.sqrt(information)
    log_scale = math.log(scale)
    upper = ndtr(root_information / 2 + log_scale / root_information)
    crossing = ndtr(-root_information / 2
                    + log_scale / root_information)
    return 2 * (scale * upper - crossing) - scale + 1


def burned_panel_lan_limit(initial_bias, information):
    """TV limit for a fixed initial bias and Bernoulli LAN information."""
    return 0.25 * sum(
        absolute_scaled_lognormal_deviation(
            1 + sign * initial_bias, information
        )
        for sign in (-1, 1)
    )


def joint_panel_limit_surface_checks():
    """Check the closed joint burn-in/panel LAN phase surface.

    The limit is a two-point mixture of scaled lognormal likelihood ratios.
    Splitting each absolute value at its unique crossing gives a closed
    Gaussian-CDF formula, implemented by ``burned_panel_lan_limit``.  An
    independent quadrature checks that formula.  Data processing in the
    initial sign and the exponential-Brownian likelihood martingale imply
    monotonicity in the absolute initial bias and in the LAN information.
    Marginalization and the product-TV inequality give matching two-source
    lower and upper bounds.
    """
    normalizer = math.sqrt(2.0 * math.pi)

    def quadrature_absolute_deviation(scale, information):
        if information == 0:
            return abs(scale - 1.0)
        if scale == 0:
            return 1.0
        root = math.sqrt(information)
        log_scale = math.log(scale)

        def integrand(gaussian):
            log_density = -0.5 * gaussian ** 2 - math.log(normalizer)
            log_weighted_density = (
                log_scale
                - 0.5 * (gaussian - root) ** 2
                - math.log(normalizer)
            )
            if log_scale + root * gaussian - information / 2.0 >= 0:
                return (math.exp(log_weighted_density)
                        - math.exp(log_density))
            return (math.exp(log_density)
                    - math.exp(log_weighted_density))

        return quad(
            integrand, -np.inf, np.inf,
            epsabs=2e-12, epsrel=2e-12, limit=500,
        )[0]

    maximum_quadrature_error = 0.0
    for initial_bias in (0.0, 0.2, 0.6, 1.0):
        for information in (0.05, 0.5, 2.0, 8.0):
            quadrature = 0.25 * sum(
                quadrature_absolute_deviation(
                    1 + sign * initial_bias, information
                )
                for sign in (-1, 1)
            )
            closed = burned_panel_lan_limit(initial_bias, information)
            maximum_quadrature_error = max(
                maximum_quadrature_error, abs(quadrature - closed)
            )
    assert maximum_quadrature_error < 2e-11

    biases = np.linspace(0.0, 1.0, 81)
    informations = np.r_[0.0, np.logspace(-4, 2, 121)]
    surface = np.array([
        [burned_panel_lan_limit(bias, information)
         for information in informations]
        for bias in biases
    ])
    minimum_bias_increment = float(np.min(np.diff(surface, axis=0)))
    minimum_information_increment = float(np.min(np.diff(surface, axis=1)))
    assert minimum_bias_increment > -2e-14
    assert minimum_information_increment > -2e-14

    maximum_bound_violation = 0.0
    for bias_index, bias in enumerate(biases):
        initial_tv = bias / 2.0
        for information_index, information in enumerate(informations):
            panel_tv = (2.0 * ndtr(math.sqrt(information) / 2.0) - 1.0)
            lower = max(initial_tv, panel_tv)
            upper = min(1.0, initial_tv + panel_tv)
            value = surface[bias_index, information_index]
            maximum_bound_violation = max(
                maximum_bound_violation, lower - value, value - upper
            )
    assert maximum_bound_violation < 2e-14

    benchmark_bias = 0.6
    benchmark_information = 1.0
    benchmark_value = burned_panel_lan_limit(
        benchmark_bias, benchmark_information
    )
    benchmark_initial_lower = benchmark_bias / 2.0
    benchmark_panel_lower = (
        2.0 * ndtr(math.sqrt(benchmark_information) / 2.0) - 1.0
    )
    benchmark_upper = benchmark_initial_lower + benchmark_panel_lower

    print("closed joint burn-in/panel phase surface:")
    print(f"  Gaussian-CDF / independent quadrature error "
          f"{maximum_quadrature_error:.3e}")
    print(f"  minimum bias/information increments "
          f"{minimum_bias_increment:.3e}, "
          f"{minimum_information_increment:.3e}")
    print(f"  largest two-source bound violation "
          f"{maximum_bound_violation:.3e}")
    print(f"  delta=0.6, c=1: lower bounds "
          f"{benchmark_initial_lower:.9f}, {benchmark_panel_lower:.9f}; "
          f"exact {benchmark_value:.9f}; upper {benchmark_upper:.9f}")

    return (
        maximum_quadrature_error,
        minimum_bias_increment,
        minimum_information_increment,
        maximum_bound_violation,
        benchmark_value,
        benchmark_initial_lower,
        benchmark_panel_lower,
        benchmark_upper,
    )


def persistent_diffuse_product_tv(persistent_correlations, trials,
                                  diffuse_correlation):
    """Exact TV for fixed biased signs followed by equal diffuse signs."""
    counts = np.arange(trials + 1)
    fair = binom.pmf(counts, trials, 0.5)
    log_likelihood = (
        counts * math.log1p(diffuse_correlation)
        + (trials - counts) * math.log1p(-diffuse_correlation)
    )
    diffuse_likelihood = np.exp(log_likelihood)
    persistent_correlations = np.asarray(persistent_correlations)
    total = 0.0
    for signs in itertools.product(
            (-1, 1), repeat=len(persistent_correlations)):
        scale = np.prod(
            1 + persistent_correlations * np.asarray(signs)
        )
        total += np.sum(fair * np.abs(scale * diffuse_likelihood - 1))
    return 0.5 * total / 2 ** len(persistent_correlations)


def persistent_diffuse_lan_limit(persistent_correlations, information):
    """TV limit for fixed biased signs times diffuse Bernoulli LAN noise."""
    persistent_correlations = np.asarray(persistent_correlations)
    total = 0.0
    for signs in itertools.product(
            (-1, 1), repeat=len(persistent_correlations)):
        scale = np.prod(
            1 + persistent_correlations * np.asarray(signs)
        )
        total += absolute_scaled_lognormal_deviation(scale, information)
    return 0.5 * total / 2 ** len(persistent_correlations)


def symmetric_binary_channel_law(panel_size, correlation, fidelity):
    """Exact binary-score law from a symmetric hidden binary chain.

    Hidden and observed states are encoded by signs.  The hidden transition
    correlation is ``correlation`` and the binary symmetric emission channel
    has E[R | X] = fidelity * X.  The pooled score law is therefore fair.
    A vectorized forward recursion returns the probabilities in the same
    prefix order as the accompanying observation-sign array.
    """
    transition = np.array([
        [(1 + correlation) / 2, (1 - correlation) / 2],
        [(1 - correlation) / 2, (1 + correlation) / 2],
    ])
    emit_minus = np.array([(1 + fidelity) / 2, (1 - fidelity) / 2])
    emit_plus = emit_minus[::-1]
    forward = np.stack([0.5 * emit_minus, 0.5 * emit_plus])
    observations = np.array([[-1], [1]], dtype=np.int8)
    for _ in range(1, panel_size):
        predicted = forward @ transition
        forward = np.concatenate([
            predicted * emit_minus,
            predicted * emit_plus,
        ])
        observations = np.concatenate([
            np.column_stack([observations,
                             -np.ones(len(observations), dtype=np.int8)]),
            np.column_stack([observations,
                             np.ones(len(observations), dtype=np.int8)]),
        ])
    return forward.sum(axis=1), observations


def symmetric_emission_second_moment(panel_size, correlation,
                                     channel_information):
    """Exact iid-reference second moment of the emitted likelihood ratio.

    If ``g=dK_+/dKbar-1`` and tau=E_Kbar[g^2], two hidden replicas reduce the
    second moment to a two-state Feynman--Kac recursion.  Their product sign
    has transition correlation ``correlation**2`` and vertex weight 1+tau*u.
    """
    replica_correlation = correlation ** 2
    transition = np.array([
        [(1 + replica_correlation) / 2,
         (1 - replica_correlation) / 2],
        [(1 - replica_correlation) / 2,
         (1 + replica_correlation) / 2],
    ])
    weight = np.array([1 - channel_information,
                       1 + channel_information])
    forward = 0.5 * weight
    for _ in range(1, panel_size):
        forward = (forward @ transition) * weight
    return forward.sum()


def overlapping_emission_lan_checks():
    """Certify emission attenuation of the two-state panel threshold.

    For a dominated symmetric channel, dK_x/dKbar=1+x*g and
    tau=E_Kbar[g^2].  At hidden independence the observed score for the
    transition correlation is sum g(R_j)g(R_{j+1}), with exact information
    (n-1)tau^2.  The replicated-chain recursion independently checks the
    complete finite-n likelihood second moment.  A binary symmetric channel
    permits exact enumeration of the emitted TV distance and its LAN limit.
    """
    fidelity = 0.8
    tau = fidelity ** 2

    derivative_size = 8
    step = 1e-5
    law_plus, observations = symmetric_binary_channel_law(
        derivative_size, step, fidelity
    )
    law_minus, observations_minus = symmetric_binary_channel_law(
        derivative_size, -step, fidelity
    )
    assert np.array_equal(observations, observations_minus)
    iid_probability = 2.0 ** -derivative_size
    numerical_score = (
        (law_plus - law_minus) / (2 * step * iid_probability)
    )
    exact_score = tau * np.sum(
        observations[:, :-1] * observations[:, 1:], axis=1
    )
    score_error = float(np.max(np.abs(numerical_score - exact_score)))
    exact_information = float(
        iid_probability * np.sum(exact_score ** 2)
    )
    assert score_error < 2e-8
    assert abs(exact_information - (derivative_size - 1) * tau ** 2) < 2e-13

    moment_size = 10
    moment_correlation = 0.23
    moment_law, _ = symmetric_binary_channel_law(
        moment_size, moment_correlation, fidelity
    )
    enumerated_second_moment = (
        2.0 ** moment_size * np.sum(moment_law ** 2)
    )
    replicated_second_moment = symmetric_emission_second_moment(
        moment_size, moment_correlation, tau
    )
    assert abs(enumerated_second_moment
               - replicated_second_moment) < 3e-14

    critical_rows = []
    for panel_size in (6, 10, 14, 18):
        correlation = 1 / math.sqrt(panel_size - 1)
        law, _ = symmetric_binary_channel_law(
            panel_size, correlation, fidelity
        )
        exact_tv = 0.5 * np.abs(
            law - 2.0 ** -panel_size
        ).sum()
        critical_rows.append((panel_size, correlation, exact_tv))
    critical_limit = 2 * ndtr(tau / 2) - 1
    hidden_limit = 2 * ndtr(0.5) - 1
    assert abs(critical_rows[-1][2] - critical_limit) < 0.002
    assert critical_limit < hidden_limit

    second_moment_rows = []
    for panel_size in (64, 256, 1024, 4096):
        correlation = 1 / math.sqrt(panel_size - 1)
        value = symmetric_emission_second_moment(
            panel_size, correlation, tau
        )
        second_moment_rows.append((panel_size, value))
    second_moment_limit = math.exp(tau ** 2)
    assert abs(second_moment_rows[-1][1]
               - second_moment_limit) < 3e-4

    print("overlapping-emission LAN attenuation:")
    print(f"  score finite-difference error {score_error:.3e}; "
          f"information {(derivative_size - 1) * tau ** 2:.12f}")
    print(f"  exact/enumerated likelihood second moment "
          f"{replicated_second_moment:.12f}")
    print("  n       a=(n-1)^-1/2    exact emitted TV")
    for panel_size, correlation, exact_tv in critical_rows:
        print(f"  {panel_size:2d}      {correlation:.9f}       {exact_tv:.9f}")
    print(f"  emitted LAN limit {critical_limit:.9f}; "
          f"hidden-path limit {hidden_limit:.9f}")
    print(f"  local second moment n=4096 "
          f"{second_moment_rows[-1][1]:.9f}; limit "
          f"{second_moment_limit:.9f}")

    return critical_rows, critical_limit, second_moment_rows


def joint_panel_mixing_checks(Q, pi, gamma_s):
    """Check the joint long-panel/fast-switching TV theorem.

    For a stationary hidden path, the KL divergence from an iid stationary
    path is (n-1) times the stationary average one-step KL.  The exact
    weighted Hilbert--Schmidt norm retains all singular modes.
    Bounding every mode by the additive-gap contraction, then applying
    Pinsker, gives the simpler dimension-only bound uniform over every
    emission kernel and event.
    """
    dimension = len(pi)
    panel_size = 200
    rows = []
    for scaled_spacing in (0.25, 0.5, 1.0, 1.5, 2.0):
        transition = expm(scaled_spacing * Q)
        density = transition / pi[None, :]
        one_step_kl = np.sum(
            pi[:, None] * transition * np.log(density))
        one_step_chi = np.sum(
            pi[:, None] * (transition - pi[None, :]) ** 2
            / pi[None, :])
        hilbert_schmidt = dependence_hilbert_schmidt_squared(
            transition, pi
        )
        spectral_chi = ((dimension - 1)
                        * math.exp(-2 * gamma_s * scaled_spacing))
        assert 0 <= one_step_kl <= one_step_chi + 2e-14
        assert abs(one_step_chi - hilbert_schmidt) < 3e-13
        assert one_step_chi <= spectral_chi + 3e-14
        one_step_renyi = math.log1p(one_step_chi)
        assert one_step_kl <= one_step_renyi + 2e-14
        assert one_step_renyi <= one_step_chi + 2e-14
        exact_pinsker = min(
            1.0, math.sqrt((panel_size - 1) * one_step_kl / 2))
        renyi_pinsker = min(
            1.0,
            math.sqrt((panel_size - 1) * one_step_renyi / 2),
        )
        hs_pinsker = min(
            1.0, math.sqrt((panel_size - 1) * one_step_chi / 2))
        spectral_pinsker = min(
            1.0,
            math.sqrt((panel_size - 1) * (dimension - 1) / 2)
            * math.exp(-gamma_s * scaled_spacing),
        )
        assert exact_pinsker <= renyi_pinsker + 2e-14
        assert renyi_pinsker <= hs_pinsker + 2e-14
        assert hs_pinsker <= spectral_pinsker + 2e-14
        rows.append((scaled_spacing, one_step_kl, one_step_chi,
                     spectral_chi, spectral_pinsker))

    print("joint panel-size/switching-rate TV certificate:")
    print("  mh     row KL       row chi2     spectral chi2   panel TV bound")
    for spacing, kl, chi, spectral_chi, panel_tv in rows:
        print(f" {spacing:4.2f}  {kl:11.8f}  {chi:11.8f}"
              f"    {spectral_chi:11.8f}      {panel_tv:11.8f}")

    # Effective-rank example.  Two five-state clusters mix rapidly within
    # themselves and slowly across a matched bridge.  Only one nonconstant
    # mode remains visible at the chosen spacing, so replacing all nine modes
    # by the gap mode loses a factor close to nine in squared dependence.
    cluster_dimension = 10
    clustered_Q = np.zeros((cluster_dimension, cluster_dimension))
    for cluster in (range(5), range(5, 10)):
        for left in cluster:
            for right in cluster:
                if left != right:
                    clustered_Q[left, right] = 2.0
    for left in range(5):
        clustered_Q[left, left + 5] = 0.1
        clustered_Q[left + 5, left] = 0.1
    clustered_Q[np.diag_indices(cluster_dimension)] = (
        -clustered_Q.sum(axis=1)
    )
    clustered_pi = np.ones(cluster_dimension) / cluster_dimension
    clustered_time = 10.0
    clustered_transition = expm(clustered_time * clustered_Q)
    relaxation_rates = np.linalg.eigvalsh(-clustered_Q)
    assert abs(relaxation_rates[0]) < 4e-14
    exact_energy = dependence_hilbert_schmidt_squared(
        clustered_transition, clustered_pi
    )
    modal_energy = np.exp(
        -2 * clustered_time * relaxation_rates[1:]
    ).sum()
    gap_energy = (
        (cluster_dimension - 1)
        * math.exp(-2 * clustered_time * relaxation_rates[1])
    )
    assert abs(exact_energy - modal_energy) < 3e-14
    effective_rank = (
        exact_energy
        / math.exp(-2 * clustered_time * relaxation_rates[1])
    )
    effective_panel_size = 20
    exact_hs_bound = min(
        1.0,
        math.sqrt((effective_panel_size - 1) * exact_energy / 2),
    )
    gap_only_bound = min(
        1.0,
        math.sqrt((effective_panel_size - 1) * gap_energy / 2),
    )
    assert abs(effective_rank - 1.0) < 2e-13
    assert exact_hs_bound < 0.42 and gap_only_bound == 1.0
    print("Hilbert--Schmidt effective-rank refinement:")
    print(
        f"  exact energy {exact_energy:.12f}, gap envelope "
        f"{gap_energy:.12f}, effective rank {effective_rank:.12f}"
    )
    print(
        f"  n={effective_panel_size} exact-HS TV bound "
        f"{exact_hs_bound:.12f}, gap-only bound {gap_only_bound:.1f}"
    )

    # At moderate dependence, D_1 <= D_2 = log(1+chi^2) is visibly sharper
    # than the direct D_1 <= chi^2 substitution.
    moderate_time = 0.2
    moderate_transition = expm(moderate_time * clustered_Q)
    moderate_density = moderate_transition / clustered_pi[None, :]
    moderate_kl = np.sum(
        clustered_pi[:, None] * moderate_transition
        * np.log(moderate_density)
    )
    moderate_energy = dependence_hilbert_schmidt_squared(
        moderate_transition, clustered_pi
    )
    moderate_renyi = math.log1p(moderate_energy)
    moderate_bounds = (
        math.sqrt(moderate_kl / 2),
        math.sqrt(moderate_renyi / 2),
        math.sqrt(moderate_energy / 2),
    )
    assert moderate_kl <= moderate_renyi <= moderate_energy
    assert abs(moderate_bounds[0] - 0.5609965594516808) < 2e-14
    assert abs(moderate_bounds[1] - 0.6019344208474726) < 2e-14
    assert abs(moderate_bounds[2] - 0.7293863052431115) < 2e-14
    print("moderate-dependence information refinements (n=2):")
    print(
        f"  exact-KL {moderate_bounds[0]:.12f}, Renyi-2 "
        f"{moderate_bounds[1]:.12f}, HS {moderate_bounds[2]:.12f}"
    )

    # Sharpness.  For a stationary symmetric two-state chain, write the
    # states as signs X_j and set Z_j=X_j X_{j+1}.  The Z_j are iid with
    # P(Z_j=1)=(1+a)/2, while under an iid state panel they are fair.  Hence
    # path TV is exactly the TV between these Bernoulli product laws.
    # Disjoint-support continuous emissions preserve this TV exactly.
    brute_trials = 7
    brute_a = 0.23
    markov_path = []
    iid_path = []
    for signs in itertools.product((-1, 1), repeat=brute_trials + 1):
        probability = 0.5
        for left, right in zip(signs[:-1], signs[1:]):
            probability *= (1 + brute_a * left * right) / 2
        markov_path.append(probability)
        iid_path.append(2 ** -(brute_trials + 1))
    brute_tv = 0.5 * np.abs(np.asarray(markov_path)
                            - np.asarray(iid_path)).sum()
    formula_tv = bernoulli_product_tv(brute_trials, brute_a)
    assert abs(brute_tv - formula_tv) < ROUND_TOL

    critical_rows = []
    for exponent in (3, 4, 5, 6, 7):
        a = 2.0 ** -exponent
        subcritical_trials = round(1 / a)
        critical_trials = round(1 / a ** 2)
        supercritical_trials = round(1 / a ** 3)
        critical_rows.append((
            a,
            bernoulli_product_tv(subcritical_trials, a),
            bernoulli_product_tv(critical_trials, a),
            bernoulli_product_tv(supercritical_trials, a),
        ))
    critical_limit = 2 * ndtr(0.5) - 1
    assert critical_rows[-1][1] < 0.04
    assert abs(critical_rows[-1][2] - critical_limit) < 0.003
    assert critical_rows[-1][3] > 0.999
    print("sharp symmetric two-state threshold:")
    print("    a       TV(n=a^-1)  TV(n=a^-2)  TV(n=a^-3)")
    for a, subcritical, critical, supercritical in critical_rows:
        print(f" {a:8.6f}    {subcritical:10.8f}  {critical:10.8f}"
              f"  {supercritical:10.8f}")
    print(f"  critical LAN limit: {critical_limit:.8f}")

    overlapping_emission_lan_checks()

    # Joint burn-in/panel sharpness.  For a symmetric two-state chain, a
    # pre-panel imbalance eta becomes delta=eta*exp(-2*m*b).  The bijection
    # from paths to (X_0, Z_1, ..., Z_N) turns the exact likelihood ratio into
    # (1+delta*X_0) times the Bernoulli-product likelihood ratio.  If
    # N*a^2 -> c, LAN sends the latter to exp(sqrt(c)G-c/2), independently of
    # X_0.  The following checks both the finite factorization and its limit.
    brute_initial_bias = 0.37
    burned_path = []
    iid_path = []
    for signs in itertools.product((-1, 1), repeat=brute_trials + 1):
        probability = (1 + brute_initial_bias * signs[0]) / 2
        for left, right in zip(signs[:-1], signs[1:]):
            probability *= (1 + brute_a * left * right) / 2
        burned_path.append(probability)
        iid_path.append(2 ** -(brute_trials + 1))
    burned_brute_tv = 0.5 * np.abs(np.asarray(burned_path)
                                   - np.asarray(iid_path)).sum()
    burned_formula_tv = burned_bernoulli_product_tv(
        brute_trials, brute_a, brute_initial_bias)
    assert abs(burned_brute_tv - burned_formula_tv) < ROUND_TOL

    limiting_bias = 0.6
    limiting_information = 1.0
    joint_limit = burned_panel_lan_limit(
        limiting_bias, limiting_information)
    joint_rows = []
    for exponent in (3, 4, 5, 6, 7):
        a = 2.0 ** -exponent
        trials = round(limiting_information / a ** 2)
        joint_rows.append((
            a,
            burned_bernoulli_product_tv(trials, a, limiting_bias),
        ))
    assert abs(joint_rows[-1][1] - joint_limit) < 3e-5
    assert abs(burned_panel_lan_limit(limiting_bias, 0)
               - abs(limiting_bias) / 2) < ROUND_TOL
    assert abs(burned_panel_lan_limit(0, limiting_information)
               - critical_limit) < ROUND_TOL
    print("sharp joint burn-in/panel limit (delta=0.6, c=1):")
    for a, exact_tv in joint_rows:
        print(f"  a={a:8.6f} exact TV {exact_tv:.9f}")
    print(f"  mixed Bernoulli/lognormal limit: {joint_limit:.9f}")


def nonstationary_joint_entropy_checks(Q, pi):
    """Enumerate the exact arbitrary-start path-entropy comparison.

    This deliberately nonreversible example checks the relative-entropy
    chain rule independently by enumerating every hidden path.  A binary
    emission then checks data processing and shows that applying Pinsker once
    to the joint entropy can improve the usual burn-in-plus-stationary-panel
    triangle bound.
    """
    initial = np.array([1.0, 0.0, 0.0])
    burn_in = 0.2
    spacing = 0.3
    panel_size = 8
    initial_after_burn_in = initial @ expm(burn_in * Q)
    transition = expm(spacing * Q)
    transitions = [transition] * (panel_size - 1)

    path_kl, transition_terms = nonstationary_path_relative_entropy(
        initial_after_burn_in, transitions, pi
    )

    path_law = []
    iid_path_law = []
    for path in itertools.product(range(len(pi)), repeat=panel_size):
        probability = initial_after_burn_in[path[0]]
        iid_probability = pi[path[0]]
        for t in range(panel_size - 1):
            probability *= transition[path[t], path[t + 1]]
            iid_probability *= pi[path[t + 1]]
        path_law.append(probability)
        iid_path_law.append(iid_probability)
    path_law = np.asarray(path_law)
    iid_path_law = np.asarray(iid_path_law)
    enumerated_kl = float(np.sum(path_law * np.log(
        path_law / iid_path_law
    )))
    path_tv = 0.5 * np.sum(abs(path_law - iid_path_law))
    joint_pinsker = min(1.0, math.sqrt(path_kl / 2))
    assert abs(path_law.sum() - 1) < 3e-14
    assert abs(iid_path_law.sum() - 1) < 3e-14
    assert abs(path_kl - enumerated_kl) < 3e-14
    assert path_tv <= joint_pinsker + 2e-14

    success = np.array([0.15, 0.65, 0.90])
    emitted_law = binary_panel_law(
        initial_after_burn_in, transition, success, panel_size
    )
    pooled_success = float(pi @ success)
    iid_emitted_law = np.asarray([
        np.prod([
            pooled_success if bit else 1 - pooled_success
            for bit in pattern
        ])
        for pattern in itertools.product((0, 1), repeat=panel_size)
    ])
    emitted_tv = 0.5 * np.sum(abs(emitted_law - iid_emitted_law))
    assert emitted_tv <= path_tv + 2e-14

    density = transition / pi[None, :]
    stationary_one_step_kl = float(np.sum(
        pi[:, None] * transition * np.log(density)
    ))
    state_tv = 0.5 * np.sum(abs(initial_after_burn_in - pi))
    additive_bound = min(
        1.0,
        state_tv
        + math.sqrt((panel_size - 1) * stationary_one_step_kl / 2),
    )
    assert joint_pinsker < additive_bound

    print("nonstationary joint path-entropy certificate:")
    print(f"  initial KL {path_kl - transition_terms.sum():.12f},"
          f" transition KL {transition_terms.sum():.12f},"
          f" joint KL {path_kl:.12f}")
    print(f"  enumerated hidden-path KL {enumerated_kl:.12f},"
          f" hidden-path TV {path_tv:.12f}")
    print(f"  emitted binary-panel TV {emitted_tv:.12f},"
          f" joint Pinsker {joint_pinsker:.12f}")
    print(f"  burn-in plus stationary-panel triangle bound"
          f" {additive_bound:.12f}")


def beta_order_statistic_transfer_checks(Q, pi, scales):
    """Check the full coverage-law transfer, including atomic scores.

    Under the iid pooled comparator, applying the continuous pooled CDF to
    the kth score order statistic gives Beta(k,n+1-k).  Data processing
    therefore carries any panel-law TV bound to the entire distribution of
    the random coverage C_0, not only to a selected lower-tail event.
    The dependent CDF below is computed exactly from the finite-state count
    recursion at each pooled probability; quadrature then checks the first
    three moment consequences independently.  A second exact recursion checks
    the randomized distributional transform for a three-atom score law and
    the sharper iid atomic comparator for deterministic tie retention.
    """
    panel_size = 12
    rank = 10
    scaled_spacing = 1.0
    transition = expm(scaled_spacing * Q)
    density = transition / pi[None, :]
    one_step_kl = np.sum(
        pi[:, None] * transition * np.log(density)
    )
    panel_tv_bound = math.sqrt(
        (panel_size - 1) * one_step_kl / 2
    )

    def dependent_coverage_cdf(probability):
        if probability <= 0:
            return 0.0
        if probability >= 1:
            return 1.0
        quantile = brentq(
            lambda x: pi @ cdf_vector(x, scales) - probability,
            0.0,
            250.0,
        )
        success = cdf_vector(quantile, scales)
        count_law = binary_count_distribution(
            pi, [transition] * (panel_size - 1), success
        )
        return count_law[rank:].sum()

    grid = np.linspace(0.02, 0.98, 97)
    discrepancies = np.array([
        abs(
            dependent_coverage_cdf(probability)
            - betainc(rank, panel_size + 1 - rank, probability)
        )
        for probability in grid
    ])
    maximum_grid_discrepancy = discrepancies.max()
    maximizing_probability = grid[discrepancies.argmax()]
    assert maximum_grid_discrepancy <= panel_tv_bound + 2e-14

    moment_rows = []
    for order in (1, 2, 3):
        dependent_moment = quad(
            lambda probability: order * probability ** (order - 1)
            * (1 - dependent_coverage_cdf(probability)),
            0.0,
            1.0,
            epsabs=2e-10,
        )[0]
        beta_moment = np.prod([
            rank + offset for offset in range(order)
        ]) / np.prod([
            panel_size + 1 + offset for offset in range(order)
        ])
        discrepancy = abs(dependent_moment - beta_moment)
        assert discrepancy <= panel_tv_bound + 2e-10
        moment_rows.append((order, dependent_moment, beta_moment,
                            discrepancy))

    # Atomic-score certificate.  Randomized PIT makes the pooled transform
    # exactly uniform even at atoms.  Conditional on each hidden state its CDF
    # is piecewise linear; the same finite-state count recursion therefore
    # gives the dependent kth-order CDF without simulation.
    atom_probabilities = np.array([
        [0.65, 0.25, 0.10],
        [0.20, 0.55, 0.25],
        [0.05, 0.25, 0.70],
    ])
    pooled_masses = pi @ atom_probabilities
    pooled_right = np.cumsum(pooled_masses)
    pooled_left = pooled_right - pooled_masses

    def randomized_atomic_cdf(probability):
        fractions = np.clip(
            (probability - pooled_left) / pooled_masses, 0.0, 1.0
        )
        state_success = atom_probabilities @ fractions
        count_law = binary_count_distribution(
            pi, [transition] * (panel_size - 1), state_success
        )
        return count_law[rank:].sum()

    atomic_grid = np.linspace(0.01, 0.99, 99)
    atomic_discrepancies = np.array([
        abs(
            randomized_atomic_cdf(probability)
            - betainc(rank, panel_size + 1 - rank, probability)
        )
        for probability in atomic_grid
    ])
    maximum_atomic_discrepancy = atomic_discrepancies.max()
    maximizing_atomic_probability = atomic_grid[
        atomic_discrepancies.argmax()
    ]
    assert maximum_atomic_discrepancy <= panel_tv_bound + 2e-14

    # With deterministic ties, C_0=F(R_(k)) is discrete.  At a reported level
    # c, its exact iid comparator is I_{H_F(c)}, where
    # H_F(c)=P{F(R)<=c}; replacing H_F(c) by c is valid but often very loose.
    deterministic_rows = []
    for atom_index, coverage_level in enumerate(pooled_right):
        state_success = atom_probabilities[:, :atom_index + 1].sum(axis=1)
        count_law = binary_count_distribution(
            pi, [transition] * (panel_size - 1), state_success
        )
        dependent_probability = count_law[rank:].sum()
        iid_probability = betainc(
            rank, panel_size + 1 - rank, coverage_level
        )
        deterministic_rows.append((
            coverage_level, dependent_probability, iid_probability,
            abs(dependent_probability - iid_probability),
        ))
    maximum_deterministic_discrepancy = max(
        row[3] for row in deterministic_rows
    )
    assert maximum_deterministic_discrepancy <= panel_tv_bound + 2e-14

    demonstration_level = 0.80
    included_atoms = pooled_right <= demonstration_level
    atomic_sublevel_probability = pooled_masses[included_atoms].sum()
    state_success = atom_probabilities[:, included_atoms].sum(axis=1)
    count_law = binary_count_distribution(
        pi, [transition] * (panel_size - 1), state_success
    )
    dependent_sublevel = count_law[rank:].sum()
    exact_iid_sublevel = betainc(
        rank, panel_size + 1 - rank, atomic_sublevel_probability
    )
    continuous_beta_envelope = betainc(
        rank, panel_size + 1 - rank, demonstration_level
    )
    assert atomic_sublevel_probability <= demonstration_level
    assert dependent_sublevel <= continuous_beta_envelope + panel_tv_bound

    print("full Beta order-statistic transfer certificate:")
    print(
        f"  n={panel_size}, k={rank}, mh={scaled_spacing:.1f}: "
        f"KL-TV bound {panel_tv_bound:.12f}"
    )
    print(
        f"  maximum exact CDF discrepancy on 97 points: "
        f"{maximum_grid_discrepancy:.12f} at p={maximizing_probability:.2f}"
    )
    print("  order   dependent moment   Beta moment       discrepancy")
    for order, dependent, beta, discrepancy in moment_rows:
        print(
            f"    {order:d}       {dependent:.12f}   {beta:.12f}"
            f"   {discrepancy:.3e}"
        )
    print("  atomic randomized-PIT check:")
    print(
        f"    maximum exact Beta-CDF discrepancy on 99 points: "
        f"{maximum_atomic_discrepancy:.12f} "
        f"at u={maximizing_atomic_probability:.2f}"
    )
    print(
        f"    maximum deterministic atomic-comparator discrepancy: "
        f"{maximum_deterministic_discrepancy:.12f}"
    )
    print(
        f"    at c={demonstration_level:.2f}, H_F(c)="
        f"{atomic_sublevel_probability:.12f}: dependent "
        f"{dependent_sublevel:.12f}, exact iid {exact_iid_sublevel:.12f}, "
        f"continuous-Beta envelope {continuous_beta_envelope:.12f}"
    )


def beta_binomial_law(trials, alpha, beta):
    """Return the beta-binomial probability vector without SciPy fitting."""
    law = np.empty(trials + 1)
    log_beta_denominator = (
        math.lgamma(alpha) + math.lgamma(beta)
        - math.lgamma(alpha + beta)
    )
    for successes in range(trials + 1):
        log_probability = (
            math.log(math.comb(trials, successes))
            + math.lgamma(alpha + successes)
            + math.lgamma(beta + trials - successes)
            - math.lgamma(alpha + beta + trials)
            - log_beta_denominator
        )
        law[successes] = math.exp(log_probability)
    assert abs(law.sum() - 1.0) < 3e-14
    return law


def three_category_compositions(total):
    """Return three-category count vectors in a fixed lexicographic order."""
    return [
        (left, middle, total - left - middle)
        for left in range(total + 1)
        for middle in range(total - left + 1)
    ]


def dirichlet_multinomial_law(trials, alpha):
    """Return the three-category Dirichlet-multinomial law exactly."""
    alpha = np.asarray(alpha, dtype=float)
    assert alpha.shape == (3,) and np.all(alpha > 0)
    total_alpha = float(alpha.sum())
    law = []
    for counts in three_category_compositions(trials):
        log_probability = math.lgamma(trials + 1)
        log_probability -= sum(math.lgamma(count + 1) for count in counts)
        log_probability += math.lgamma(total_alpha)
        log_probability -= math.lgamma(total_alpha + trials)
        log_probability += sum(
            math.lgamma(parameter + count) - math.lgamma(parameter)
            for parameter, count in zip(alpha, counts)
        )
        law.append(math.exp(log_probability))
    law = np.asarray(law)
    assert abs(law.sum() - 1.0) < 3e-14
    return law


def conditional_multilevel_count(path, calibration_size, ranks, future_size):
    """Exact three-bin future-count law conditional on a binary state path.

    The two calibration thresholds split future observations into three
    nested-coverage categories. Disjoint uniform supports reduce each case
    to one Dirichlet-multinomial law or two independent beta-binomial splits.
    """
    lower_rank, upper_rank = ranks
    assert 1 <= lower_rank < upper_rank <= calibration_size
    calibration = path[:calibration_size]
    future = path[calibration_size:]
    calibration_low = sum(state == 0 for state in calibration)
    future_low = sum(state == 0 for state in future)
    calibration_high = calibration_size - calibration_low
    future_high = future_size - future_low
    compositions = three_category_compositions(future_size)
    index = {counts: position for position, counts in enumerate(compositions)}
    law = np.zeros(len(compositions))

    if calibration_low >= upper_rank:
        component = dirichlet_multinomial_law(
            future_low,
            (lower_rank, upper_rank - lower_rank,
             calibration_low + 1 - upper_rank),
        )
        for probability, counts in zip(
                component, three_category_compositions(future_low)):
            augmented = (counts[0], counts[1], counts[2] + future_high)
            law[index[augmented]] += probability
    elif calibration_low < lower_rank:
        shifted_lower = lower_rank - calibration_low
        shifted_upper = upper_rank - calibration_low
        component = dirichlet_multinomial_law(
            future_high,
            (shifted_lower, shifted_upper - shifted_lower,
             calibration_high + 1 - shifted_upper),
        )
        for probability, counts in zip(
                component, three_category_compositions(future_high)):
            augmented = (counts[0] + future_low, counts[1], counts[2])
            law[index[augmented]] += probability
    else:
        low_split = beta_binomial_law(
            future_low, lower_rank, calibration_low + 1 - lower_rank
        )
        shifted_upper = upper_rank - calibration_low
        high_split = beta_binomial_law(
            future_high, shifted_upper,
            calibration_high + 1 - shifted_upper
        )
        for below_lower, low_probability in enumerate(low_split):
            low_middle = future_low - below_lower
            for high_middle, high_probability in enumerate(high_split):
                counts = (
                    below_lower,
                    low_middle + high_middle,
                    future_high - high_middle,
                )
                law[index[counts]] += low_probability * high_probability
    assert abs(law.sum() - 1.0) < 3e-14
    return law


def conditional_disjoint_support_count(path, calibration_size, rank, future_size):
    """Exact future-success count law conditional on a binary state path.

    State zero emits uniformly on (0, 1/2), and state one emits uniformly on
    (1/2, 1).  If the calibration order statistic lies in one support, the
    relevant within-support quantile is beta distributed.  Mixing the future
    Bernoulli trials over that quantile gives a beta-binomial law.
    """
    calibration = path[:calibration_size]
    future = path[calibration_size:]
    calibration_low = sum(state == 0 for state in calibration)
    future_low = sum(state == 0 for state in future)
    law = np.zeros(future_size + 1)

    if calibration_low >= rank:
        within_rank = rank
        beta_tail = calibration_low + 1 - rank
        component = beta_binomial_law(
            future_low, within_rank, beta_tail
        )
        law[:future_low + 1] = component
    else:
        within_rank = rank - calibration_low
        calibration_high = calibration_size - calibration_low
        future_high = future_size - future_low
        beta_tail = calibration_high + 1 - within_rank
        component = beta_binomial_law(
            future_high, within_rank, beta_tail
        )
        law[future_low:future_low + future_high + 1] = component
    assert abs(law.sum() - 1.0) < 3e-14
    return law


def repeated_future_coverage_checks():
    """Check the calibration-plus-future beta-binomial transfer exactly."""
    calibration_size = 7
    rank = 6
    future_size = 5
    total_size = calibration_size + future_size
    correlation = 0.2
    transition = np.array([
        [(1.0 + correlation) / 2.0, (1.0 - correlation) / 2.0],
        [(1.0 - correlation) / 2.0, (1.0 + correlation) / 2.0],
    ])

    dependent_law = np.zeros(future_size + 1)
    enumerated_iid_law = np.zeros(future_size + 1)
    for path in itertools.product((0, 1), repeat=total_size):
        conditional_law = conditional_disjoint_support_count(
            path, calibration_size, rank, future_size
        )
        path_probability = 0.5
        for index in range(total_size - 1):
            path_probability *= transition[path[index], path[index + 1]]
        dependent_law += path_probability * conditional_law
        enumerated_iid_law += 2.0 ** (-total_size) * conditional_law

    beta_parameter = calibration_size + 1 - rank
    iid_law = beta_binomial_law(
        future_size, rank, beta_parameter
    )
    assert np.max(abs(enumerated_iid_law - iid_law)) < 3e-14

    pooled_coverage = rank / (calibration_size + 1)
    independent_binomial = binom.pmf(
        np.arange(future_size + 1), future_size, pooled_coverage
    )
    iid_binomial_tv = 0.5 * np.sum(abs(iid_law - independent_binomial))

    count_values = np.arange(future_size + 1)
    iid_mean = float(iid_law @ count_values)
    iid_variance = float(iid_law @ (count_values - iid_mean) ** 2)
    beta_binomial_variance = (
        future_size * pooled_coverage * (1.0 - pooled_coverage)
        * (calibration_size + 1 + future_size)
        / (calibration_size + 2)
    )
    assert abs(iid_mean - future_size * pooled_coverage) < 3e-14
    assert abs(iid_variance - beta_binomial_variance) < 3e-14

    one_step_information = float(np.sum(
        0.5 * transition * np.log(transition / 0.5)
    ))
    joint_tv_bound = math.sqrt(
        (total_size - 1) * one_step_information / 2.0
    )
    exact_count_tv = 0.5 * np.sum(abs(dependent_law - iid_law))
    assert exact_count_tv <= joint_tv_bound + 3e-14

    print("repeated future-coverage law certificate:")
    print(
        f"  calibration n={calibration_size}, rank k={rank}, "
        f"future L={future_size}, correlation a={correlation:.1f}"
    )
    print(
        f"  exact dependent/Beta-binomial count TV {exact_count_tv:.12f}, "
        f"joint score-law bound {joint_tv_bound:.12f}"
    )
    print(
        f"  iid Beta-binomial/independent-binomial TV "
        f"{iid_binomial_tv:.12f}"
    )
    print(
        f"  iid future-count variance {iid_variance:.12f}, "
        f"naive binomial variance "
        f"{future_size * pooled_coverage * (1.0 - pooled_coverage):.12f}"
    )


def multilevel_future_coverage_checks():
    """Check the nested-threshold Dirichlet-multinomial transfer exactly."""
    calibration_size = 7
    ranks = (3, 6)
    future_size = 5
    total_size = calibration_size + future_size
    correlation = 0.2
    transition = np.array([
        [(1.0 + correlation) / 2.0, (1.0 - correlation) / 2.0],
        [(1.0 - correlation) / 2.0, (1.0 + correlation) / 2.0],
    ])

    compositions = three_category_compositions(future_size)
    dependent_law = np.zeros(len(compositions))
    enumerated_iid_law = np.zeros(len(compositions))
    for path in itertools.product((0, 1), repeat=total_size):
        conditional_law = conditional_multilevel_count(
            path, calibration_size, ranks, future_size
        )
        path_probability = 0.5
        for position in range(total_size - 1):
            path_probability *= transition[path[position], path[position + 1]]
        dependent_law += path_probability * conditional_law
        enumerated_iid_law += 2.0 ** (-total_size) * conditional_law

    alpha = np.array([
        ranks[0], ranks[1] - ranks[0], calibration_size + 1 - ranks[1]
    ])
    iid_law = dirichlet_multinomial_law(future_size, alpha)
    assert np.max(abs(enumerated_iid_law - iid_law)) < 3e-14

    probabilities = alpha / alpha.sum()
    fixed_multinomial = np.array([
        math.factorial(future_size)
        / math.prod(math.factorial(count) for count in counts)
        * math.prod(probability ** count
                    for probability, count in zip(probabilities, counts))
        for counts in compositions
    ])
    iid_multinomial_tv = 0.5 * np.sum(abs(iid_law - fixed_multinomial))
    exact_count_tv = 0.5 * np.sum(abs(dependent_law - iid_law))
    one_step_information = float(np.sum(
        0.5 * transition * np.log(transition / 0.5)
    ))
    joint_tv_bound = math.sqrt(
        (total_size - 1) * one_step_information / 2.0
    )
    assert exact_count_tv <= joint_tv_bound + 3e-14

    count_matrix = np.asarray(compositions, dtype=float)
    iid_mean = iid_law @ count_matrix
    centered = count_matrix - iid_mean
    iid_covariance = centered.T @ (iid_law[:, None] * centered)
    naive_covariance = future_size * (
        np.diag(probabilities) - np.outer(probabilities, probabilities)
    )
    inflation = (calibration_size + 1 + future_size) / (calibration_size + 2)
    assert np.max(abs(iid_covariance - inflation * naive_covariance)) < 3e-14

    print("multilevel future-coverage law certificate:")
    print(
        f"  calibration n={calibration_size}, ranks={ranks}, "
        f"future L={future_size}, correlation a={correlation:.1f}"
    )
    print(
        f"  exact dependent/Dirichlet-multinomial TV "
        f"{exact_count_tv:.12f}, joint score-law bound "
        f"{joint_tv_bound:.12f}"
    )
    print(
        f"  iid Dirichlet-multinomial/fixed-multinomial TV "
        f"{iid_multinomial_tv:.12f}, covariance inflation "
        f"{inflation:.12f}"
    )


def irregular_joint_panel_mixing_checks(Q, pi, gamma_s):
    """Check the irregular-grid path-TV theorem and its sharp scale.

    Stationarity makes relative entropy additive over unequal transitions.
    Pinsker and rowwise D <= chi-square therefore depend on the sum of the
    individual Hilbert--Schmidt energies, not on the average spacing.
    """
    dimension = len(pi)
    spacings = np.array([0.35, 0.50, 0.75, 1.10, 1.70])
    transitions = [expm(spacing * Q) for spacing in spacings]
    row_kl_sum = 0.0
    energy_sum = 0.0
    renyi_sum = 0.0
    for transition in transitions:
        density = transition / pi[None, :]
        row_kl_sum += np.sum(
            pi[:, None] * transition * np.log(density)
        )
        energy = dependence_hilbert_schmidt_squared(transition, pi)
        energy_sum += energy
        renyi_sum += math.log1p(energy)
    gap_energy_sum = (dimension - 1) * np.exp(
        -2 * gamma_s * spacings
    ).sum()

    markov_path = []
    iid_path = []
    for path in itertools.product(range(dimension), repeat=len(spacings) + 1):
        probability = pi[path[0]]
        for transition, left, right in zip(
                transitions, path[:-1], path[1:]):
            probability *= transition[left, right]
        markov_path.append(probability)
        iid_path.append(np.prod(pi[list(path)]))
    markov_path = np.asarray(markov_path)
    iid_path = np.asarray(iid_path)
    path_kl = np.sum(markov_path * np.log(markov_path / iid_path))
    path_tv = 0.5 * np.abs(markov_path - iid_path).sum()
    kl_bound = min(1.0, math.sqrt(row_kl_sum / 2))
    renyi_bound = min(1.0, math.sqrt(renyi_sum / 2))
    energy_bound = min(1.0, math.sqrt(energy_sum / 2))
    gap_bound = min(1.0, math.sqrt(gap_energy_sum / 2))
    assert abs(path_kl - row_kl_sum) < 5e-14
    assert path_tv <= kl_bound + 2e-14
    assert kl_bound <= renyi_bound + 2e-14
    assert renyi_bound <= energy_bound + 2e-14
    assert energy_bound <= gap_bound + 2e-14

    # The path-to-transition-sign bijection remains exact when each gap has
    # a different correlation.  This verifies it independently by enumerating
    # all state paths and all transition signs.
    correlations = np.array([0.10, 0.20, 0.05, 0.30, 0.12])
    two_state_markov = []
    two_state_iid = []
    for signs in itertools.product((-1, 1), repeat=len(correlations) + 1):
        probability = 0.5
        for correlation, left, right in zip(
                correlations, signs[:-1], signs[1:]):
            probability *= (1 + correlation * left * right) / 2
        two_state_markov.append(probability)
        two_state_iid.append(2 ** -(len(correlations) + 1))
    two_state_path_tv = 0.5 * np.abs(
        np.asarray(two_state_markov) - np.asarray(two_state_iid)
    ).sum()
    two_state_affinity = np.sqrt(
        np.asarray(two_state_markov) * np.asarray(two_state_iid)
    ).sum()
    transition_sign_tv = 0.0
    for signs in itertools.product((-1, 1), repeat=len(correlations)):
        likelihood = np.prod(1 + correlations * np.asarray(signs))
        transition_sign_tv += 0.5 * 2 ** -len(correlations) * abs(
            likelihood - 1
        )
    assert abs(two_state_path_tv - transition_sign_tv) < ROUND_TOL
    product_affinity = biased_sign_product_affinity(correlations)
    assert abs(two_state_affinity - product_affinity) < ROUND_TOL
    assert 1 - product_affinity <= two_state_path_tv + ROUND_TOL
    assert two_state_path_tv <= math.sqrt(
        1 - product_affinity ** 2
    ) + ROUND_TOL

    # Heterogeneous triangular array: half the correlations are a, half 2a,
    # with the common group size chosen so sum_j a_j^2 -> 1.  The exact
    # product TV converges to the same LAN limit 2 Phi(1/2)-1.
    critical_rows = []
    for exponent in (3, 4, 5, 6):
        correlation_a = 2.0 ** -exponent
        correlation_b = 2 * correlation_a
        trials = round(1 / (5 * correlation_a ** 2))
        information = trials * (correlation_a ** 2 + correlation_b ** 2)
        critical_rows.append((
            correlation_a,
            trials,
            information,
            two_level_bernoulli_product_tv(
                trials, correlation_a, trials, correlation_b
            ),
        ))
    critical_limit = 2 * ndtr(0.5) - 1
    assert abs(critical_rows[-1][3] - critical_limit) < 1e-4

    # Sum of squared correlations determines the zero and one TV phases, but
    # not an interior critical value unless the largest correlation vanishes.
    # A single persistent correlation and a diffuse triangular array can have
    # the same squared information and different limiting TV distances.
    profile_information = 0.25
    persistent_tv = bernoulli_product_tv(
        1, math.sqrt(profile_information)
    )
    diffuse_trials = 2 ** 16
    diffuse_correlation = math.sqrt(
        profile_information / diffuse_trials
    )
    diffuse_tv = bernoulli_product_tv(
        diffuse_trials, diffuse_correlation
    )
    profile_lan_limit = 2 * ndtr(
        math.sqrt(profile_information) / 2
    ) - 1
    assert abs(persistent_tv - 0.25) < ROUND_TOL
    assert abs(diffuse_tv - profile_lan_limit) < 1e-6
    assert persistent_tv - diffuse_tv > 0.05

    # General critical profile: finitely many correlations can persist while
    # the remaining infinitesimal correlations converge to Gaussian
    # likelihood noise.  The limit is their independent likelihood product,
    # not a function of total squared correlation alone.
    persistent_correlations = np.array([0.5, 0.3])
    diffuse_information = 0.25
    persistent_diffuse_limit = persistent_diffuse_lan_limit(
        persistent_correlations, diffuse_information
    )
    persistent_diffuse_rows = []
    for tail_trials in (64, 256, 1024, 4096, 16384):
        tail_correlation = math.sqrt(
            diffuse_information / tail_trials
        )
        persistent_diffuse_rows.append((
            tail_trials,
            persistent_diffuse_product_tv(
                persistent_correlations,
                tail_trials,
                tail_correlation,
            ),
        ))
    assert abs(
        persistent_diffuse_rows[-1][1] - persistent_diffuse_limit
    ) < 1e-6

    print("irregular-grid joint path-TV certificate:")
    print(f"  exact path KL {path_kl:.12f}, exact path TV {path_tv:.12f}")
    print(f"  Pinsker(KL) {kl_bound:.12f}, Renyi-2 {renyi_bound:.12f},"
          f" HS bound {energy_bound:.12f}, gap bound {gap_bound:.12f}")
    print("  unequal two-state path/sign TV identity:"
          f" {two_state_path_tv:.12f}")
    print("  exact product Hellinger affinity:"
          f" {product_affinity:.12f}, TV interval"
          f" [{1 - product_affinity:.12f},"
          f" {math.sqrt(1 - product_affinity ** 2):.12f}]")
    print("heterogeneous critical scale (correlations a and 2a):")
    print("    a       trials/group  sum correlation^2    exact TV")
    for correlation_a, trials, information, exact_tv in critical_rows:
        print(f" {correlation_a:8.6f}   {trials:7d}"
              f"         {information:10.8f}      {exact_tv:10.8f}")
    print(f"  heterogeneous LAN limit: {critical_limit:.8f}")
    print("critical-profile counterexample at sum correlation^2 = 0.25:")
    print(f"  one persistent correlation: TV {persistent_tv:.9f}")
    print(f"  {diffuse_trials} diffuse correlations: TV {diffuse_tv:.9f},"
          f" LAN limit {profile_lan_limit:.9f}")
    print("persistent-plus-diffuse critical profile"
          " (persistent correlations 0.5, 0.3; diffuse c=0.25):")
    for diffuse_trials, exact_tv in persistent_diffuse_rows:
        print(f"  {diffuse_trials:5d} diffuse signs: exact TV {exact_tv:.9f}")
    print(f"  product Bernoulli/lognormal limit:"
          f" {persistent_diffuse_limit:.9f}")


def nonreversible_contraction_checks():
    # This chain has a nonuniform invariant law and violates detailed balance.
    # Its additive reversibilization nevertheless gives an L2(pi) contraction
    # rate for the original, nonnormal semigroup.
    Q = np.array([[-3.0, 2.7, 0.3],
                  [0.2, -2.2, 2.0],
                  [2.4, 0.4, -2.8]])
    pi = stationary(Q)
    assert not np.allclose(np.diag(pi) @ Q, Q.T @ np.diag(pi))
    gamma_s, witness = additive_gap_witness(Q, pi)
    scales = np.array([0.7, 1.5, 4.0])
    q = pooled_quantile(scales, pi)
    f = cdf_vector(q, scales)
    centered = f - TARGET
    norm = math.sqrt(pi @ centered ** 2)

    largest_ratio = 0.0
    for c in np.linspace(0.05, 2.0, 40):
        evolved = expm(c * Q) @ centered
        ratio = math.sqrt(pi @ evolved ** 2) / (norm * math.exp(-gamma_s * c))
        largest_ratio = max(largest_ratio, ratio)
        assert ratio <= 1 + 2e-14
        statewise_bound = np.sqrt(1 / pi - 1) * norm * math.exp(-gamma_s * c)
        assert np.all(abs(evolved) <= statewise_bound + 2e-14)

    # The additive gap is not merely sufficient.  Its eigenfunction attains
    # the logarithmic norm derivative at zero, so every larger prefactor-one
    # exponential rate fails immediately.
    sharp_time = 1e-3
    witness_norm = math.sqrt(pi @ witness ** 2)
    witness_evolved = expm(sharp_time * Q) @ witness
    sharp_ratio = (math.sqrt(pi @ witness_evolved ** 2)
                   / (witness_norm * math.exp(-gamma_s * sharp_time)))
    faster_ratio = (math.sqrt(pi @ witness_evolved ** 2)
                    / (witness_norm * math.exp(-1.01 * gamma_s * sharp_time)))
    assert sharp_ratio <= 1 + 2e-14
    assert faster_ratio > 1

    m, h, n = 5.0, 0.1, 100
    P = expm(m * h * Q)
    exact_var = TARGET * (1 - TARGET) / n
    Pk = np.eye(3)
    for lag in range(1, n):
        Pk = Pk @ P
        covariance = pi @ (centered * (Pk @ centered))
        exact_var += 2 * (n - lag) * covariance / n ** 2
    a = math.exp(-m * gamma_s * h)
    finite_factor = finite_variance_factor(a, n)
    contraction_bound = TARGET * (1 - TARGET) / n * finite_factor
    assert exact_var <= contraction_bound + ROUND_TOL
    print("nonreversible additive-reversibilization bound:",
          f"gap {gamma_s:.8f}, largest score ratio {largest_ratio:.8f},")
    print("sharp exponent witness:",
          f"envelope ratio {sharp_ratio:.12f},",
          f"1%-faster ratio {faster_ratio:.12f}")
    print("nonreversible calibration variance:",
          f"exact {exact_var:.8f}, finite-n bound {contraction_bound:.8f},",
          f"factor {finite_factor:.8f}")

    # Sharpness of the finite-n variance factor.  A symmetric two-state chain
    # and conditional CDF values (0, 1) attain every covariance envelope
    # exactly; continuous score laws with disjoint supports realize this case.
    saturation_n = 17
    saturation_a = math.exp(-0.7)
    saturation_factor = finite_variance_factor(saturation_a, saturation_n)
    saturation_bound = 0.25 / saturation_n * saturation_factor
    saturation_exact = (0.25 / saturation_n
                        + 0.5 / saturation_n ** 2
                        * sum((saturation_n - lag) * saturation_a ** lag
                              for lag in range(1, saturation_n)))
    assert abs(saturation_exact - saturation_bound) < ROUND_TOL
    print("finite-n factor saturation:",
          f"n {saturation_n}, a {saturation_a:.8f},",
          f"variance {saturation_exact:.8f}")
    joint_panel_switching_checks()
    joint_panel_mixing_checks(Q, pi, gamma_s)
    joint_panel_limit_surface_checks()
    nonstationary_joint_entropy_checks(Q, pi)
    beta_order_statistic_transfer_checks(Q, pi, scales)
    repeated_future_coverage_checks()
    multilevel_future_coverage_checks()
    irregular_joint_panel_mixing_checks(Q, pi, gamma_s)
    cantelli_order_statistic_checks(Q, pi, gamma_s, scales)
    irregular_panel_checks(Q, pi, gamma_s, f)
    burnin_transfer_checks(Q, pi, gamma_s, f)


def finite_chain_checks(rng):
    # A directed cycle is irreducible and nonreversible.  Its conditional
    # coverage is a matrix-exponential mixture, not one scalar exponential.
    Q_cycle = np.array([[-1.7, 1.5, 0.2],
                        [0.2, -1.7, 1.5],
                        [1.5, 0.2, -1.7]])
    scales = np.array([0.7, 1.5, 4.0])
    pi = stationary(Q_cycle)
    q = pooled_quantile(scales, pi)
    f = cdf_vector(q, scales)
    print("\nnonreversible finite-chain endpoint theorem")
    for c in (0.0, 0.5, 1.0, 2.0):
        P = expm(c * Q_cycle)
        theory = P @ f
        observed = endpoint_simulation(rng, P, scales, q)
        assert max(abs(observed - theory)) < 0.004
        print(f"c={c:3.1f} theory {theory.round(5)} simulation {observed.round(5)}")
    # Delta=c/m leaves the same nonzero crossover at every switching rate.
    assert np.allclose(expm(5 * Q_cycle * (1 / 5)),
                       expm(20 * Q_cycle * (1 / 20)), atol=ROUND_TOL)

    nonreversible_contraction_checks()

    # Symmetric Q is reversible with uniform pi.  Check the L2 spectral-gap
    # bound and the general finite-chain covariance identity.
    Q_rev = np.array([[-1.0, 1.0, 0.0],
                      [1.0, -3.0, 2.0],
                      [0.0, 2.0, -2.0]])
    pi = stationary(Q_rev)
    assert np.allclose(pi, np.ones(3) / 3)
    gap = additive_gap(Q_rev, pi)
    q = pooled_quantile(scales, pi)
    f = cdf_vector(q, scales)
    centered = f - TARGET
    f_norm = math.sqrt(pi @ centered ** 2)
    for c in (0.25, 0.5, 1.0, 2.0):
        errors = abs(expm(c * Q_rev) @ f - TARGET)
        bounds = np.sqrt(1 / pi - 1) * f_norm * math.exp(-gap * c)
        assert np.all(errors <= bounds + 1e-14)

    m, h, n, reps = 5.0, 0.1, 100, 6000
    P = expm(m * h * Q_rev)
    exact_var = TARGET * (1 - TARGET) / n
    Pk = np.eye(3)
    for lag in range(1, n):
        Pk = Pk @ P
        covariance = pi @ (centered * (Pk @ centered))
        exact_var += 2 * (n - lag) * covariance / n ** 2
    a = math.exp(-m * gap * h)
    spectral_bound = finite_variance_factor(a, n) / (4 * n)
    assert exact_var <= spectral_bound

    states = rng.choice(3, size=reps, p=pi)
    samples = np.empty((reps, n), dtype=np.int8)
    for t in range(n):
        u = rng.random(reps)
        cumulative = np.cumsum(P[states], axis=1)
        states = (u[:, None] > cumulative).sum(axis=1)
        samples[:, t] = (rng.exponential(scale=scales[states]) <= q)
    empirical_var = samples.mean(axis=1).var(ddof=1)
    assert abs(empirical_var - exact_var) / exact_var < 0.08
    print("reversible calibration variance:",
          f"empirical {empirical_var:.8f}, exact {exact_var:.8f},",
          f"spectral bound {spectral_bound:.8f}, gap {gap:.6f}")


def rare_group_split_conformal_check():
    """Certify conditional failure in an i.i.d. two-group example.

    Let both the group label and score be Bernoulli(delta), so common scores
    are zero and rare scores are one. With the conventional nonrandomized
    split-conformal rule, q is the k-th calibration order statistic, where
    k=ceil((n+1)(1-alpha)). A rare test point is covered exactly when at
    least n-k+1 calibration points are rare. Ties only make the usual
    marginal guarantee conservative; they do not weaken the counterexample.
    """
    alpha = 0.1
    delta = 0.01
    expected = {
        19: 0.015273761488896613,
        49: 0.00013221002446730005,
        99: 6.930983962862892e-08,
        199: 2.6281101520802502e-14,
    }
    rare_coverages = []
    for n, target in expected.items():
        k = math.ceil((n + 1) * (1 - alpha))
        required_rare = n - k + 1
        rare_coverage = sum(
            math.comb(n, j) * delta ** j * (1 - delta) ** (n - j)
            for j in range(required_rare, n + 1)
        )
        marginal_coverage = 1 - delta + delta * rare_coverage
        assert abs(rare_coverage - target) < ROUND_TOL
        assert marginal_coverage >= k / (n + 1) >= 1 - alpha
        rare_coverages.append(rare_coverage)
        print(
            "iid rare-group split conformal:",
            f"n={n}, k={k}, required rare={required_rare},",
            f"conditional={rare_coverage:.12g},",
            f"marginal={marginal_coverage:.12g}",
        )
    assert all(x > y for x, y in zip(rare_coverages, rare_coverages[1:]))
    assert rare_coverages[2] < 7e-8


def main():
    rng = np.random.default_rng(20260923)
    q = brentq(lambda x: pooled_cdf(x) - TARGET, 0.0, 30.0)
    conditional = 1 - np.exp(-q / SCALES)
    print(f"pooled 90% threshold: {q:.8f}")
    print("Delta/epsilon  theory in states 1,2  independent simulation in states 1,2")
    for ratio in (0.0, 0.5, 1.0, 2.0, 4.0):
        decay = math.exp(-2 * ratio)
        theory = TARGET + decay * (conditional - TARGET)
        observed = []
        for start in (0, 1):
            # Symmetric two-state CTMC endpoint transition, then fresh emission.
            final = np.where(rng.random(150000) < (1 + decay) / 2, start, 1 - start)
            score = rng.exponential(scale=SCALES[final])
            observed.append(np.mean(score <= q))
        assert max(abs(np.array(observed) - theory)) < 0.004
        print(f"{ratio:5.1f}          {theory.round(5)}            {np.round(observed, 5)}")

    n, h, reps = 100, 0.1, 6000
    decay = math.exp(-2 * h / EPS)
    states = rng.integers(0, 2, size=reps)
    samples = np.empty((reps, n), dtype=np.int8)
    for t in range(n):
        states ^= (rng.random(reps) < (1 - decay) / 2)
        samples[:, t] = (rng.exponential(scale=SCALES[states]) <= q)
    empirical_var = samples.mean(axis=1).var(ddof=1)
    d = (conditional[0] - conditional[1]) / 2
    predicted_var = (TARGET * (1 - TARGET) / n
                     + 2 * d ** 2 / n ** 2
                     * sum((n - lag) * decay ** lag for lag in range(1, n)))
    print(f"empirical calibration-CDF variance: {empirical_var:.8f}")
    print(f"exact calibration-CDF variance:     {predicted_var:.8f}")
    assert abs(empirical_var - predicted_var) / predicted_var < 0.07
    finite_chain_checks(rng)
    rare_group_split_conformal_check()
    print("PASS: finite-chain crossover, additive/reversible spectral bounds,"
          " regular and irregular calibration variance, exact polynomial"
          " count law, optimal calibration design and its cost rule, the"
          " sharp joint panel/switching variance profile,"
          " full Beta order-statistic, repeated-future"
          " Beta-binomial, and multilevel Dirichlet-multinomial transfers,"
          " regular- and"
          " irregular-grid and arbitrary-start joint panel-size/"
          "fast-switching TV bounds and sharp stationary, heterogeneous,"
          " persistent-plus-diffuse, overlapping-emission, and burned-start"
          " two-state limits,"
          " Perron tail rate,"
          " sharp prefactor and first two"
          " relative saddle-point corrections, the moderate-deviation"
          " bridge to the mean, the second-order central lattice Edgeworth"
          " correction, Green--Kubo"
          " curvature, all-order"
          " cumulants, arbitrary-start boundary constants and local tail"
          " correction,"
          " one-sided order-statistic PAC"
          " conversion,"
          " iid rare-group conditional failure,"
          " nonstationary burn-in transfer, and transient panel moments")
    print("PASS")


if __name__ == "__main__":
    main()
