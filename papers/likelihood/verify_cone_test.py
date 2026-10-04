"""Certificate for the cone-corrected skewness--kurtosis score test.

The fast-switching alternative has unrestricted first-order skewness but
nonnegative first-order excess kurtosis after the variance correction is
profiled out.  The normalized score therefore lies in R x R_+, so its
Gaussian limit is projected by replacing the kurtosis coordinate by its
positive part.  The null law is 1/2 chi^2_1 + 1/2 chi^2_2.  Under a local
Gaussian score shift, one-dimensional quadrature gives exact limiting power.
For a general Fisher covariance the projection must first residualize and
standardize the unrestricted scores against the one-sided score.  The same
construction is checked below for three unrestricted coordinates.
It is also checked for several Fisher-orthogonal one-sided coordinates and
for two and three correlated one-sided coordinates with dense unrestricted
blocks.  In three dimensions the exact weights use orthant probabilities of
both the constrained correlation and its precision correlation.
"""
import json
import math
import os

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2, norm


def chibar_cdf(x):
    return 0.5 * chi2.cdf(x, 1) + 0.5 * chi2.cdf(x, 2)


def orthant_chibar_cdf(x, unrestricted_dimension, one_sided_dimension):
    """Null CDF for Fisher-orthogonal one-sided canonical coordinates."""
    return sum(
        math.comb(one_sided_dimension, active)
        * chi2.cdf(x, unrestricted_dimension + active)
        / 2.0 ** one_sided_dimension
        for active in range(one_sided_dimension + 1)
    )


def project_correlated_orthant_2d(scores, correlation):
    """Metric projection onto R_+^2 for unit variances and correlation rho.

    The four candidates are the vertex, the two axis faces, and the
    interior.  Selecting the candidate nearest in the inverse-covariance
    metric also records the dimension of the face hit by the projection.
    """
    scores = np.asarray(scores, dtype=float)
    covariance = np.array([
        [1.0, correlation],
        [correlation, 1.0],
    ])
    precision = np.linalg.inv(covariance)
    candidates = np.zeros((scores.shape[0], 4, 2))
    candidates[:, 1, 0] = np.maximum(
        scores[:, 0] - correlation * scores[:, 1], 0.0
    )
    candidates[:, 2, 1] = np.maximum(
        scores[:, 1] - correlation * scores[:, 0], 0.0
    )
    candidates[:, 3, :] = scores
    feasible = np.ones((scores.shape[0], 4), dtype=bool)
    feasible[:, 3] = np.all(scores >= 0.0, axis=1)
    displacement = candidates - scores[:, None, :]
    distances = np.einsum(
        "nki,ij,nkj->nk", displacement, precision, displacement
    )
    distances[~feasible] = np.inf
    selected = np.argmin(distances, axis=1)
    projection = candidates[np.arange(scores.shape[0]), selected]
    statistic = np.einsum(
        "ni,ij,nj->n", projection, precision, projection
    )
    face_dimensions = np.array([0, 1, 1, 2])[selected]
    return statistic, face_dimensions


def project_correlated_orthant(scores, correlation):
    """Metric projection onto a low-dimensional correlated orthant.

    Each subset of positive coordinates defines a face.  On that face the
    inverse-correlation metric has an explicit unconstrained minimizer.  We
    enumerate the faces, discard minimizers outside their relative closures,
    and retain the nearest candidate.  This is exponential in q and is used
    here only as a transparent q=3 certificate, not as a production solver.
    """
    scores = np.asarray(scores, dtype=float)
    correlation = np.asarray(correlation, dtype=float)
    q = correlation.shape[0]
    precision = np.linalg.inv(correlation)
    best_distance = np.full(scores.shape[0], np.inf)
    best_projection = np.zeros_like(scores)
    best_dimension = np.zeros(scores.shape[0], dtype=int)
    indices = np.arange(q)
    for mask in range(1 << q):
        active = indices[(mask & (1 << indices)) != 0]
        inactive = indices[(mask & (1 << indices)) == 0]
        candidate = np.zeros_like(scores)
        if active.size:
            candidate[:, active] = scores[:, active]
            if inactive.size:
                candidate[:, active] += scores[:, inactive] @ (
                    np.linalg.solve(
                        precision[np.ix_(active, active)],
                        precision[np.ix_(active, inactive)],
                    ).T
                )
            feasible = np.all(candidate[:, active] >= -1e-12, axis=1)
        else:
            feasible = np.ones(scores.shape[0], dtype=bool)
        displacement = candidate - scores
        distance = np.einsum(
            "ni,ij,nj->n", displacement, precision, displacement
        )
        improve = feasible & (distance < best_distance)
        best_distance[improve] = distance[improve]
        best_projection[improve] = candidate[improve]
        best_dimension[improve] = active.size
    statistic = np.einsum(
        "ni,ij,nj->n", best_projection, precision, best_projection
    )
    return statistic, best_dimension


def trivariate_orthant_weights(correlation):
    """Exact chi-bar weights for a three-dimensional Gaussian orthant."""
    correlation = np.asarray(correlation, dtype=float)
    precision = np.linalg.inv(correlation)
    scales = np.sqrt(np.diag(precision))
    precision_correlation = precision / np.outer(scales, scales)

    def positive_orthant_probability(matrix):
        return 0.125 + sum(
            math.asin(matrix[i, j])
            for i, j in ((0, 1), (0, 2), (1, 2))
        ) / (4.0 * math.pi)

    w3 = positive_orthant_probability(correlation)
    w0 = positive_orthant_probability(precision_correlation)
    weights = np.array([w0, 0.5 - w3, 0.5 - w0, w3])
    return weights, precision_correlation


def normal_square_tail(delta, threshold):
    """P((Z + delta)^2 > threshold^2), for Z standard normal."""
    return norm.cdf(-threshold - delta) + norm.sf(threshold - delta)


def local_power(delta_skew, delta_kurt, critical):
    """Exact limiting power under a Gaussian shift of the two scores.

    If (Z3, Z4) converges to N((delta_skew, delta_kurt), I), the limiting
    statistic is (Z3)^2 + max(Z4, 0)^2.  Integrating over the positive
    projected coordinate leaves a one-dimensional quadrature.
    """
    root = math.sqrt(critical)
    negative_face = norm.cdf(-delta_kurt) * normal_square_tail(delta_skew, root)
    curved = quad(
        lambda y: normal_square_tail(delta_skew, math.sqrt(max(0.0, critical - y * y)))
        * norm.pdf(y - delta_kurt),
        0.0,
        root,
        epsabs=2e-13,
        epsrel=2e-13,
    )[0]
    outside = norm.sf(root - delta_kurt)
    return negative_face + curved + outside


def halfspace_power(eta, delta, critical):
    """Power for p unrestricted canonical scores and one one-sided score.

    If U ~ N_p(eta, I), V ~ N(delta, 1), and U is independent of V, the
    projected statistic is ||U||^2 + max(V, 0)^2.  Conditional on V, the
    first term is noncentral chi-square with p degrees of freedom.
    """
    eta = np.asarray(eta, dtype=float)
    degrees = eta.size
    noncentrality = float(eta @ eta)
    root = math.sqrt(critical)
    negative_face = norm.cdf(-delta) * ncx2.sf(
        critical, degrees, noncentrality
    )
    curved = quad(
        lambda y: ncx2.sf(
            max(0.0, critical - y * y), degrees, noncentrality
        ) * norm.pdf(y - delta),
        0.0,
        root,
        epsabs=2e-13,
        epsrel=2e-13,
    )[0]
    outside = norm.sf(root - delta)
    return negative_face + curved + outside


def statistics(z):
    """Jarque--Bera coordinates after profiling Gaussian mean and variance."""
    z = z - z.mean(axis=1, keepdims=True)
    z = z / np.sqrt(np.mean(z * z, axis=1, keepdims=True))
    skew = np.mean(z ** 3, axis=1)
    excess = np.mean(z ** 4, axis=1) - 3.0
    n = z.shape[1]
    ordinary = n * (skew * skew / 6.0 + excess * excess / 24.0)
    cone = n * (skew * skew / 6.0 + np.maximum(excess, 0.0) ** 2 / 24.0)
    return skew, excess, ordinary, cone


def canonical_scores(scores, covariance):
    """Independent coordinates adapted to R x R_+ in the Fisher metric."""
    sigma_2 = math.sqrt(covariance[1, 1])
    conditional_variance = (
        covariance[0, 0] - covariance[0, 1] ** 2 / covariance[1, 1]
    )
    unrestricted = (
        scores[:, 0] - covariance[0, 1] * scores[:, 1] / covariance[1, 1]
    ) / math.sqrt(conditional_variance)
    one_sided = scores[:, 1] / sigma_2
    return unrestricted, one_sided


def symmetric_root(matrix, inverse=False):
    """Symmetric positive-definite square root or inverse square root."""
    values, vectors = np.linalg.eigh(matrix)
    if np.min(values) <= 0.0:
        raise ValueError("matrix must be positive definite")
    powers = values ** (-0.5 if inverse else 0.5)
    return (vectors * powers) @ vectors.T


def canonical_halfspace_scores(scores, covariance):
    """Canonical coordinates for R^p x R_+ under a Fisher metric."""
    scores = np.asarray(scores, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    covariance_x = covariance[:-1, :-1]
    cross = covariance[:-1, -1]
    variance_v = covariance[-1, -1]
    schur = covariance_x - np.outer(cross, cross) / variance_v
    residual = scores[:, :-1] - np.outer(scores[:, -1], cross / variance_v)
    unrestricted = residual @ symmetric_root(schur, inverse=True)
    one_sided = scores[:, -1] / math.sqrt(variance_v)
    return unrestricted, one_sided, schur


def main():
    rng = np.random.default_rng(20260924)
    alpha = 0.05
    critical = brentq(lambda x: chibar_cdf(x) - (1.0 - alpha), 0.0, 20.0)
    n, reps, batch = 4000, 20000, 100
    cone_reject = ordinary_at_chibar = negative_excess = changed = 0
    for start in range(0, reps, batch):
        b = min(batch, reps - start)
        _, excess, ordinary, cone = statistics(rng.standard_normal((b, n)))
        cone_reject += int(np.sum(cone > critical))
        ordinary_at_chibar += int(np.sum(ordinary > critical))
        negative_excess += int(np.sum(excess < 0.0))
        changed += int(np.sum(np.abs(ordinary - cone) > 1e-14))

    # A symmetric platykurtic law is evidence against unrestricted normality,
    # but not in the nonnegative-kurtosis direction generated here.
    z = rng.uniform(-math.sqrt(3.0), math.sqrt(3.0), size=(1, 200000))
    skew, excess, ordinary, cone = statistics(z)

    # Independent check of the local-asymptotic power formula.  These are
    # draws from the limiting Gaussian score experiment, not finite-n data.
    local_reps = 2000000
    base_scores = rng.standard_normal((local_reps, 2))
    scenarios = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0), (0.0, 2.0)]
    local_rows = []
    max_power_error = 0.0
    jb_critical = chi2.ppf(1.0 - alpha, 2)
    for delta_skew, delta_kurt in scenarios:
        shifted = base_scores + np.array([delta_skew, delta_kurt])
        projected = shifted[:, 0] ** 2 + np.maximum(shifted[:, 1], 0.0) ** 2
        exact_power = local_power(delta_skew, delta_kurt, critical)
        simulated_power = float(np.mean(projected > critical))
        max_power_error = max(max_power_error, abs(exact_power - simulated_power))
        jb_power = float(ncx2.sf(jb_critical, 2, delta_skew ** 2 + delta_kurt ** 2))
        local_rows.append({
            "delta_skew": delta_skew,
            "delta_kurt": delta_kurt,
            "exact_cone_power": exact_power,
            "simulated_cone_power": simulated_power,
            "ordinary_jb_power": jb_power,
        })

    # General Fisher covariance.  Projection onto R x R_+ is metric, not
    # coordinatewise: residualize the unrestricted score against the
    # one-sided score, then standardize both.  The canonical coordinates are
    # independent N(0,1), so the null chi-bar law is covariance-invariant.
    rho = 0.75
    covariance = np.array([[1.7, rho * math.sqrt(1.7 * 0.8)],
                           [rho * math.sqrt(1.7 * 0.8), 0.8]])
    correlated_reps = 2000000
    correlated = rng.multivariate_normal(np.zeros(2), covariance, size=correlated_reps)
    canonical_u, canonical_v = canonical_scores(correlated, covariance)
    metric_statistic = canonical_u ** 2 + np.maximum(canonical_v, 0.0) ** 2
    precision = np.linalg.inv(covariance)
    interior_statistic = np.einsum("ni,ij,nj->n", correlated, precision, correlated)
    piecewise_statistic = np.where(
        correlated[:, 1] >= 0.0, interior_statistic, canonical_u ** 2
    )
    metric_identity_error = float(np.max(np.abs(metric_statistic - piecewise_statistic)))
    correlated_null_rejection = float(np.mean(metric_statistic > critical))

    # The superficially natural rule that marginally standardizes and clips
    # the second coordinate omits the Fisher-metric residualization.
    naive_u = correlated[:, 0] / math.sqrt(covariance[0, 0])
    naive_v = correlated[:, 1] / math.sqrt(covariance[1, 1])
    naive_statistic = naive_u ** 2 + np.maximum(naive_v, 0.0) ** 2
    naive_null_rejection = float(np.mean(naive_statistic > critical))

    # Choose a raw mean shift whose canonical coordinates are (1,1).  This
    # verifies that the existing power formula remains valid after the same
    # covariance transformation.
    eta_unrestricted, eta_one_sided = 1.0, 1.0
    delta_2 = eta_one_sided * math.sqrt(covariance[1, 1])
    delta_1 = (
        covariance[0, 1] * delta_2 / covariance[1, 1]
        + eta_unrestricted
        * math.sqrt(covariance[0, 0] - covariance[0, 1] ** 2 / covariance[1, 1])
    )
    shifted_correlated = correlated + np.array([delta_1, delta_2])
    shifted_u, shifted_v = canonical_scores(shifted_correlated, covariance)
    correlated_local_power = float(
        np.mean(shifted_u ** 2 + np.maximum(shifted_v, 0.0) ** 2 > critical)
    )
    correlated_local_power_exact = local_power(
        eta_unrestricted, eta_one_sided, critical
    )

    # Dimension-free half-space theorem.  Here p=3 unrestricted coordinates
    # are correlated with each other and with the one-sided coordinate.  The
    # Schur residualization whitens them, leaving the universal null law
    # 1/2 chi^2_p + 1/2 chi^2_{p+1} and a one-dimensional power integral.
    p = 3
    covariance_p = np.array([
        [1.50, 0.35, -0.15, 0.40],
        [0.35, 1.20, 0.25, -0.20],
        [-0.15, 0.25, 0.90, 0.30],
        [0.40, -0.20, 0.30, 0.80],
    ])
    p_reps = 1500000
    general = rng.multivariate_normal(
        np.zeros(p + 1), covariance_p, size=p_reps
    )
    general_u, general_v, general_schur = canonical_halfspace_scores(
        general, covariance_p
    )
    general_statistic = np.sum(general_u ** 2, axis=1) + np.maximum(
        general_v, 0.0
    ) ** 2
    general_chibar_cdf = lambda x: (
        0.5 * chi2.cdf(x, p) + 0.5 * chi2.cdf(x, p + 1)
    )
    general_critical = brentq(
        lambda x: general_chibar_cdf(x) - (1.0 - alpha), 0.0, 30.0
    )
    general_null_rejection = float(np.mean(general_statistic > general_critical))

    precision_p = np.linalg.inv(covariance_p)
    interior_p = np.einsum("ni,ij,nj->n", general, precision_p, general)
    residual_p = general[:, :-1] - np.outer(
        general[:, -1], covariance_p[:-1, -1] / covariance_p[-1, -1]
    )
    face_p = np.einsum(
        "ni,ij,nj->n", residual_p, np.linalg.inv(general_schur), residual_p
    )
    piecewise_p = np.where(general[:, -1] >= 0.0, interior_p, face_p)
    general_identity_error = float(
        np.max(np.abs(general_statistic - piecewise_p))
    )

    eta_p = np.array([0.6, -0.8, 0.4])
    delta_p = 0.9
    raw_delta_v = delta_p * math.sqrt(covariance_p[-1, -1])
    raw_delta_x = (
        covariance_p[:-1, -1] * raw_delta_v / covariance_p[-1, -1]
        + symmetric_root(general_schur) @ eta_p
    )
    shifted_general = general + np.r_[raw_delta_x, raw_delta_v]
    shifted_general_u, shifted_general_v, _ = canonical_halfspace_scores(
        shifted_general, covariance_p
    )
    general_power_simulated = float(np.mean(
        np.sum(shifted_general_u ** 2, axis=1)
        + np.maximum(shifted_general_v, 0.0) ** 2
        > general_critical
    ))
    general_power_exact = halfspace_power(
        eta_p, delta_p, general_critical
    )
    general_null_size_exact = halfspace_power(
        np.zeros(p), 0.0, general_critical
    )
    p1_reduction_error = max(
        abs(
            halfspace_power(np.array([delta_skew]), delta_kurt, critical)
            - local_power(delta_skew, delta_kurt, critical)
        )
        for delta_skew, delta_kurt in scenarios
    )

    # The summaries above are scalar.  Release the large Monte Carlo arrays
    # before allocating the two additional orthant experiments below.
    del base_scores, correlated, shifted_correlated
    del canonical_u, canonical_v, shifted_u, shifted_v
    del general, shifted_general, general_u, general_v
    del shifted_general_u, shifted_general_v

    # Several one-sided scores.  After residualizing the unrestricted block,
    # suppose the q constrained canonical coordinates are Fisher-orthogonal.
    # Their signs are independent fair coins under the null.  Conditional on
    # exactly j positive coordinates, the statistic is chi-square_(p+j), so
    # the chi-bar weights are binomial.
    orthant_p, orthant_q = 2, 3
    orthant_c = np.array([[1.30, 0.25], [0.25, 0.90]])
    orthant_s = np.diag([0.80, 1.10, 0.60])
    orthant_b = np.array([
        [0.25, -0.15, 0.20],
        [-0.10, 0.18, 0.12],
    ])
    orthant_a = (
        orthant_c
        + orthant_b @ np.linalg.solve(orthant_s, orthant_b.T)
    )
    orthant_covariance = np.block([
        [orthant_a, orthant_b],
        [orthant_b.T, orthant_s],
    ])
    orthant_reps = 1000000
    orthant_raw = rng.multivariate_normal(
        np.zeros(orthant_p + orthant_q),
        orthant_covariance,
        size=orthant_reps,
    )
    orthant_y = orthant_raw[:, orthant_p:]
    orthant_residual = (
        orthant_raw[:, :orthant_p]
        - orthant_y @ np.linalg.solve(orthant_s, orthant_b.T)
    )
    orthant_u = orthant_residual @ symmetric_root(
        orthant_c, inverse=True
    )
    orthant_v = orthant_y @ symmetric_root(orthant_s, inverse=True)
    orthant_statistic = (
        np.sum(orthant_u ** 2, axis=1)
        + np.sum(np.maximum(orthant_v, 0.0) ** 2, axis=1)
    )
    orthant_critical = brentq(
        lambda x: orthant_chibar_cdf(
            x, orthant_p, orthant_q
        ) - (1.0 - alpha),
        0.0,
        40.0,
    )
    orthant_null_rejection = float(np.mean(
        orthant_statistic > orthant_critical
    ))
    orthant_positive_count = np.sum(orthant_v > 0.0, axis=1)
    orthant_face_frequencies = np.bincount(
        orthant_positive_count, minlength=orthant_q + 1
    ) / orthant_reps
    orthant_weights = np.array([
        math.comb(orthant_q, active) / 2.0 ** orthant_q
        for active in range(orthant_q + 1)
    ])
    orthant_weight_error = float(np.max(np.abs(
        orthant_face_frequencies - orthant_weights
    )))

    # Correlation among two constrained coordinates changes the cone angles
    # and hence the chi-bar weights.  A dense unrestricted block contributes
    # an independent chi-square_p term after regression on the constrained
    # block.  Importantly, rho is the marginal correlation in that block,
    # not the conditional correlation after regressing it on X.
    # The exact weights are
    # (1/4-asin(rho)/(2pi), 1/2, 1/4+asin(rho)/(2pi)).
    constrained_p = 2
    constrained_rho = 0.70
    constrained_reps = 1000000
    constrained_s = np.array([
        [1.20, constrained_rho * math.sqrt(1.20 * 0.80)],
        [constrained_rho * math.sqrt(1.20 * 0.80), 0.80],
    ])
    constrained_c = np.array([[1.10, 0.20], [0.20, 0.90]])
    constrained_b = np.array([[1.40, 1.30], [-1.40, -1.30]])
    constrained_a = (
        constrained_c
        + constrained_b @ np.linalg.solve(
            constrained_s, constrained_b.T
        )
    )
    constrained_covariance = np.block([
        [constrained_a, constrained_b],
        [constrained_b.T, constrained_s],
    ])
    constrained_raw = rng.multivariate_normal(
        np.zeros(constrained_p + 2),
        constrained_covariance,
        size=constrained_reps,
    )
    constrained_y = constrained_raw[:, constrained_p:]
    constrained_residual = (
        constrained_raw[:, :constrained_p]
        - constrained_y @ np.linalg.solve(
            constrained_s, constrained_b.T
        )
    )
    constrained_u = constrained_residual @ symmetric_root(
        constrained_c, inverse=True
    )
    constrained_v = constrained_y / np.sqrt(np.diag(constrained_s))
    constrained_orthant_statistic, constrained_faces = (
        project_correlated_orthant_2d(
            constrained_v, constrained_rho
        )
    )
    constrained_statistic = (
        np.sum(constrained_u ** 2, axis=1)
        + constrained_orthant_statistic
    )
    angle_term = math.asin(constrained_rho) / (2.0 * math.pi)
    constrained_weights = np.array([
        0.25 - angle_term,
        0.50,
        0.25 + angle_term,
    ])
    constrained_face_frequencies = np.bincount(
        constrained_faces, minlength=3
    ) / constrained_reps
    constrained_weight_error = float(np.max(np.abs(
        constrained_face_frequencies - constrained_weights
    )))

    constrained_conditional_s = (
        constrained_s
        - constrained_b.T @ np.linalg.solve(
            constrained_a, constrained_b
        )
    )
    constrained_conditional_rho = (
        constrained_conditional_s[0, 1]
        / math.sqrt(
            constrained_conditional_s[0, 0]
            * constrained_conditional_s[1, 1]
        )
    )

    def constrained_chibar_cdf(x, weights):
        return sum(
            weight * chi2.cdf(x, constrained_p + face_dimension)
            for face_dimension, weight in enumerate(weights)
        )

    constrained_critical = brentq(
        lambda x: constrained_chibar_cdf(
            x, constrained_weights
        ) - (1.0 - alpha),
        0.0,
        30.0,
    )
    constrained_null_rejection = float(np.mean(
        constrained_statistic > constrained_critical
    ))
    binomial_q2_weights = np.array([0.25, 0.50, 0.25])
    binomial_q2_critical = brentq(
        lambda x: constrained_chibar_cdf(
            x, binomial_q2_weights
        ) - (1.0 - alpha),
        0.0,
        30.0,
    )
    constrained_rejection_at_binomial = float(np.mean(
        constrained_statistic > binomial_q2_critical
    ))
    conditional_angle_term = (
        math.asin(constrained_conditional_rho) / (2.0 * math.pi)
    )
    conditional_q2_weights = np.array([
        0.25 - conditional_angle_term,
        0.50,
        0.25 + conditional_angle_term,
    ])
    conditional_q2_critical = brentq(
        lambda x: constrained_chibar_cdf(
            x, conditional_q2_weights
        ) - (1.0 - alpha),
        0.0,
        40.0,
    )
    constrained_rejection_at_conditional = float(np.mean(
        constrained_statistic > conditional_q2_critical
    ))

    # Three correlated constraints admit another closed form.  The weight of
    # the full-dimensional face is the positive-orthant probability under R;
    # the vertex weight is the same probability under corr(R^{-1}).  Conic
    # Gauss--Bonnet then gives w1=1/2-w3 and w2=1/2-w0.  Enumerating all eight
    # faces supplies an independent metric-projection check.
    del constrained_raw, constrained_y, constrained_residual
    del constrained_u, constrained_v, constrained_statistic
    trivariate_p, trivariate_q = 2, 3
    trivariate_reps = 1000000
    trivariate_correlation = np.array([
        [1.00, 0.55, -0.25],
        [0.55, 1.00, 0.35],
        [-0.25, 0.35, 1.00],
    ])
    trivariate_scales = np.sqrt(np.array([1.20, 0.80, 1.50]))
    trivariate_s = (
        trivariate_correlation
        * np.outer(trivariate_scales, trivariate_scales)
    )
    trivariate_c = np.array([[1.10, 0.20], [0.20, 0.90]])
    trivariate_b = np.array([
        [0.55, -0.35, 0.40],
        [-0.45, 0.30, 0.25],
    ])
    trivariate_a = (
        trivariate_c
        + trivariate_b @ np.linalg.solve(trivariate_s, trivariate_b.T)
    )
    trivariate_covariance = np.block([
        [trivariate_a, trivariate_b],
        [trivariate_b.T, trivariate_s],
    ])
    trivariate_raw = rng.multivariate_normal(
        np.zeros(trivariate_p + trivariate_q),
        trivariate_covariance,
        size=trivariate_reps,
    )
    trivariate_y = trivariate_raw[:, trivariate_p:]
    trivariate_residual = (
        trivariate_raw[:, :trivariate_p]
        - trivariate_y @ np.linalg.solve(trivariate_s, trivariate_b.T)
    )
    trivariate_u = trivariate_residual @ symmetric_root(
        trivariate_c, inverse=True
    )
    trivariate_v = trivariate_y / trivariate_scales
    trivariate_orthant_statistic, trivariate_faces = (
        project_correlated_orthant(
            trivariate_v, trivariate_correlation
        )
    )
    trivariate_statistic = (
        np.sum(trivariate_u ** 2, axis=1)
        + trivariate_orthant_statistic
    )
    trivariate_weights, trivariate_precision_correlation = (
        trivariate_orthant_weights(trivariate_correlation)
    )
    trivariate_face_frequencies = np.bincount(
        trivariate_faces, minlength=trivariate_q + 1
    ) / trivariate_reps
    trivariate_weight_error = float(np.max(np.abs(
        trivariate_face_frequencies - trivariate_weights
    )))

    def trivariate_chibar_cdf(x, weights):
        return sum(
            weight * chi2.cdf(x, trivariate_p + face_dimension)
            for face_dimension, weight in enumerate(weights)
        )

    trivariate_critical = brentq(
        lambda x: trivariate_chibar_cdf(
            x, trivariate_weights
        ) - (1.0 - alpha),
        0.0,
        40.0,
    )
    trivariate_null_rejection = float(np.mean(
        trivariate_statistic > trivariate_critical
    ))
    binomial_q3_weights = np.array([0.125, 0.375, 0.375, 0.125])
    binomial_q3_critical = brentq(
        lambda x: trivariate_chibar_cdf(
            x, binomial_q3_weights
        ) - (1.0 - alpha),
        0.0,
        40.0,
    )
    trivariate_rejection_at_binomial = float(np.mean(
        trivariate_statistic > binomial_q3_critical
    ))

    out = {
        "n": n,
        "replications": reps,
        "chibar_95": critical,
        "null_rejection_cone": cone_reject / reps,
        "null_rejection_unprojected_at_chibar": ordinary_at_chibar / reps,
        "null_negative_excess_fraction": negative_excess / reps,
        "null_projection_changes_fraction": changed / reps,
        "uniform_skewness": float(skew[0]),
        "uniform_excess_kurtosis": float(excess[0]),
        "uniform_ordinary_statistic": float(ordinary[0]),
        "uniform_cone_statistic": float(cone[0]),
        "local_score_replications": local_reps,
        "local_power": local_rows,
        "local_power_max_simulation_error": max_power_error,
        "correlated_score_covariance": covariance.tolist(),
        "correlated_score_correlation": rho,
        "correlated_score_replications": correlated_reps,
        "fisher_projection_identity_max_error": metric_identity_error,
        "correlated_null_rejection_metric_projection": correlated_null_rejection,
        "correlated_null_rejection_naive_clipping": naive_null_rejection,
        "correlated_local_raw_mean": [delta_1, delta_2],
        "correlated_local_canonical_mean": [eta_unrestricted, eta_one_sided],
        "correlated_local_power_exact": correlated_local_power_exact,
        "correlated_local_power_simulated": correlated_local_power,
        "general_halfspace_unrestricted_dimension": p,
        "general_halfspace_covariance": covariance_p.tolist(),
        "general_halfspace_replications": p_reps,
        "general_halfspace_chibar_95": general_critical,
        "general_halfspace_exact_null_size": general_null_size_exact,
        "general_halfspace_null_rejection": general_null_rejection,
        "general_halfspace_projection_identity_max_error": general_identity_error,
        "general_halfspace_p1_reduction_max_error": p1_reduction_error,
        "general_halfspace_canonical_mean": [*eta_p.tolist(), delta_p],
        "general_halfspace_power_exact": general_power_exact,
        "general_halfspace_power_simulated": general_power_simulated,
        "orthant_unrestricted_dimension": orthant_p,
        "orthant_one_sided_dimension": orthant_q,
        "orthant_covariance": orthant_covariance.tolist(),
        "orthant_replications": orthant_reps,
        "orthant_chibar_weights": orthant_weights.tolist(),
        "orthant_face_frequencies": orthant_face_frequencies.tolist(),
        "orthant_weight_max_error": orthant_weight_error,
        "orthant_chibar_95": orthant_critical,
        "orthant_null_rejection": orthant_null_rejection,
        "correlated_constrained_correlation": constrained_rho,
        "correlated_constrained_unrestricted_dimension": constrained_p,
        "correlated_constrained_covariance": constrained_covariance.tolist(),
        "correlated_constrained_conditional_correlation":
            constrained_conditional_rho,
        "correlated_constrained_replications": constrained_reps,
        "correlated_constrained_chibar_weights": constrained_weights.tolist(),
        "correlated_constrained_face_frequencies":
            constrained_face_frequencies.tolist(),
        "correlated_constrained_weight_max_error":
            constrained_weight_error,
        "correlated_constrained_chibar_95": constrained_critical,
        "correlated_constrained_null_rejection":
            constrained_null_rejection,
        "correlated_constrained_binomial_95": binomial_q2_critical,
        "correlated_constrained_rejection_at_binomial":
            constrained_rejection_at_binomial,
        "correlated_constrained_conditional_95": conditional_q2_critical,
        "correlated_constrained_rejection_at_conditional":
            constrained_rejection_at_conditional,
        "trivariate_constrained_correlation":
            trivariate_correlation.tolist(),
        "trivariate_constrained_precision_correlation":
            trivariate_precision_correlation.tolist(),
        "trivariate_constrained_unrestricted_dimension": trivariate_p,
        "trivariate_constrained_covariance": trivariate_covariance.tolist(),
        "trivariate_constrained_replications": trivariate_reps,
        "trivariate_constrained_chibar_weights":
            trivariate_weights.tolist(),
        "trivariate_constrained_face_frequencies":
            trivariate_face_frequencies.tolist(),
        "trivariate_constrained_weight_max_error":
            trivariate_weight_error,
        "trivariate_constrained_chibar_95": trivariate_critical,
        "trivariate_constrained_null_rejection":
            trivariate_null_rejection,
        "trivariate_constrained_binomial_95": binomial_q3_critical,
        "trivariate_constrained_rejection_at_binomial":
            trivariate_rejection_at_binomial,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cone_test_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"chi-bar-square 95% critical value: {critical:.6f}")
    print(f"null rejection, cone statistic: {out['null_rejection_cone']:.4f}")
    print(f"null rejection, unprojected statistic at same cutoff: {out['null_rejection_unprojected_at_chibar']:.4f}")
    print(f"negative sample excess kurtosis: {out['null_negative_excess_fraction']:.4f}")
    print(f"uniform: skew {skew[0]:+.5f}, excess {excess[0]:+.5f}, "
          f"ordinary {ordinary[0]:.2f}, cone {cone[0]:.2f}")
    print("local asymptotic power: delta_S delta_K exact cone simulated cone ordinary JB")
    for row in local_rows:
        print(f"  {row['delta_skew']:7.2f} {row['delta_kurt']:7.2f} "
              f"{row['exact_cone_power']:.6f} {row['simulated_cone_power']:.6f} "
              f"{row['ordinary_jb_power']:.6f}")
    print(f"correlated scores (rho={rho:.2f}), metric null rejection: "
          f"{correlated_null_rejection:.6f}")
    print(f"correlated scores, naive clipping null rejection: {naive_null_rejection:.6f}")
    print(f"metric/canonical projection identity error: {metric_identity_error:.3e}")
    print("correlated local power, exact/simulated: "
          f"{correlated_local_power_exact:.6f}/{correlated_local_power:.6f}")
    print(f"general half-space (p={p}) 95% critical value: {general_critical:.6f}")
    print("general half-space null size, exact/simulated: "
          f"{general_null_size_exact:.6f}/{general_null_rejection:.6f}")
    print(f"general half-space projection identity error: {general_identity_error:.3e}")
    print(f"general formula p=1 reduction error: {p1_reduction_error:.3e}")
    print("general half-space local power, exact/simulated: "
          f"{general_power_exact:.6f}/{general_power_simulated:.6f}")
    print(f"orthant p={orthant_p}, q={orthant_q} critical/null rejection: "
          f"{orthant_critical:.6f}/{orthant_null_rejection:.6f}")
    print("orthant exact/simulated face weights and max error: "
          f"{orthant_weights}, {orthant_face_frequencies}, "
          f"{orthant_weight_error:.3e}")
    print("correlated constrained weights exact/simulated, max error: "
          f"{constrained_weights}, {constrained_face_frequencies}, "
          f"{constrained_weight_error:.3e}")
    print("correlated constrained p+2 critical/correct rejection: "
          f"{constrained_critical:.6f}/{constrained_null_rejection:.6f}")
    print("marginal/conditional constrained correlations and wrong rejections: "
          f"{constrained_rho:.6f}/{constrained_conditional_rho:.6f}, "
          f"binomial {constrained_rejection_at_binomial:.6f}, "
          f"conditional {constrained_rejection_at_conditional:.6f}")
    print("three-constraint weights exact/simulated, max error: "
          f"{trivariate_weights}, {trivariate_face_frequencies}, "
          f"{trivariate_weight_error:.3e}")
    print("three-constraint critical/correct/binomial rejection: "
          f"{trivariate_critical:.6f}/{trivariate_null_rejection:.6f}/"
          f"{trivariate_rejection_at_binomial:.6f}")

    ok = (
        abs(out["null_rejection_cone"] - alpha) < 0.006
        and abs(out["null_negative_excess_fraction"] - 0.5) < 0.03
        and out["null_rejection_unprojected_at_chibar"] > 0.070
        and out["uniform_ordinary_statistic"] > 1000.0
        and out["uniform_cone_statistic"] < 2.0
        and abs(local_rows[0]["exact_cone_power"] - alpha) < 1e-12
        and max_power_error < 0.0015
        and all(row["exact_cone_power"] > row["ordinary_jb_power"]
                for row in local_rows[1:])
        and metric_identity_error < 5e-14
        and abs(correlated_null_rejection - alpha) < 0.0015
        and abs(naive_null_rejection - alpha) > 0.005
        and abs(correlated_local_power - correlated_local_power_exact) < 0.0015
        and general_identity_error < 8e-14
        and abs(general_null_size_exact - alpha) < 1e-12
        and abs(general_null_rejection - alpha) < 0.0015
        and p1_reduction_error < 1e-12
        and abs(general_power_simulated - general_power_exact) < 0.0015
        and abs(orthant_null_rejection - alpha) < 0.0015
        and orthant_weight_error < 0.0015
        and abs(constrained_null_rejection - alpha) < 0.0015
        and constrained_weight_error < 0.0015
        and abs(constrained_rejection_at_binomial - alpha) > 0.008
        and abs(constrained_rejection_at_conditional - alpha) > 0.008
        and abs(trivariate_null_rejection - alpha) < 0.0015
        and trivariate_weight_error < 0.0015
        and abs(trivariate_rejection_at_binomial - alpha) > 0.005
    )
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
