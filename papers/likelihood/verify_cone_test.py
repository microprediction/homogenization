"""Certificate for the cone-corrected skewness--kurtosis score test.

The fast-switching alternative has unrestricted first-order skewness but
nonnegative first-order excess kurtosis after the variance correction is
profiled out.  The normalized score therefore lies in R x R_+, so its
Gaussian limit is projected by replacing the kurtosis coordinate by its
positive part.  The null law is 1/2 chi^2_1 + 1/2 chi^2_2.  Under a local
Gaussian score shift, one-dimensional quadrature gives exact limiting power.
For a general Fisher covariance the projection must first residualize and
standardize the unrestricted score against the one-sided score.
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
    )
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
