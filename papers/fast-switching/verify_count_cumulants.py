"""Certificate for the exact Cox-count and factorial-cumulant identities.

Two independent finite-dimensional calculations are compared:

1. Polynomial moments of the integrated intensity Lambda_T are propagated by
   the backward Feynman--Kac moment hierarchy.
2. The joint regime/count master equation is truncated at a count so remote
   that its remaining mass is below rounding error, and ordinary cumulants of
   N_T are computed from that distribution.

For a Cox count, factorial cumulants of N_T must equal ordinary cumulants of
Lambda_T.  The certificate checks this through order four for one stream and
through bidegree (2,2) for two conditionally independent streams driven by a
nonreversible three-state chain.  A third calculation adds a regime-dependent
common-shock stream and verifies the all-order correction through bidegree
(3,3), first for unit jumps and then for arbitrary integer-valued bivariate
marks.  It then verifies the closed two-state finite-horizon formula used on
the counts page and its O(lambda^-2) first-order residual.  Finally, an exact
finite-difference construction gives two positive mixing laws with identical
first four cumulants but different mixed-Poisson count laws.
The last calculation turns the inverse discontinuity into a finite-sample
minimax obstruction: unrestricted mixing laws cannot be recovered uniformly
in total variation, even on a compact intensity interval.  A local Poisson
experiment gives an optimized parametric-scale Wasserstein-1 lower bound, and
the sample mean gives a matching n^-1/2 upper rate on the point-mass submodel.
For the unrestricted compact class, a parity-conditioned binomial construction
instead proves the genuinely nonparametric lower obstruction
``(log log n)/(log n)``.  A normalized-factorial-moment estimator and Jackson
approximation give the matching upper order.  The certificate checks the
factorial-moment second-moment identity, the shifted-Chebyshev coefficient
bound, and decay of the resulting stochastic remainder.  The parametric and
unrestricted rates are deliberately kept distinct.
"""

import itertools
import math
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import expm_multiply
from scipy.stats import norm, poisson


def cumulants4(raw):
    """First four cumulants from the first four raw moments."""
    m1, m2, m3, m4 = raw
    return np.array([
        m1,
        m2 - m1 * m1,
        m3 - 3 * m2 * m1 + 2 * m1 ** 3,
        m4 - 4 * m3 * m1 - 3 * m2 * m2 + 12 * m2 * m1 * m1 - 6 * m1 ** 4,
    ])


def factorial_cumulants4(ordinary):
    """Invert kappa_r = sum_k S(r,k) factorial_k through order four."""
    k1, k2, k3, k4 = ordinary
    return np.array([
        k1,
        k2 - k1,
        k3 - 3 * k2 + 2 * k1,
        k4 - 6 * k3 + 11 * k2 - 6 * k1,
    ])


def finite_cumulant_twins(order=4, relative_perturbation=0.4):
    """Positive mixing laws agreeing through ``order`` but not in law.

    The certificate is specialized to four cumulants.  The construction on
    the counts page works for every finite order by replacing four below with
    the desired order.
    """
    if order != 4:
        raise ValueError("the numerical certificate is specialized to order four")
    n = order + 1
    indices = np.arange(n + 1)
    support = indices.astype(float) + 1.0
    coefficients = np.array([math.comb(n, int(j)) for j in indices], float)
    base = coefficients / 2.0 ** n
    delta = relative_perturbation / 2.0 ** n
    signed = coefficients * (-1.0) ** indices
    plus = base + delta * signed
    minus = base - delta * signed

    raw_plus = np.array([plus @ support ** r for r in range(1, order + 1)])
    raw_minus = np.array([minus @ support ** r for r in range(1, order + 1)])
    intensity_gap = np.max(np.abs(cumulants4(raw_plus) - cumulants4(raw_minus)))

    count_grid = np.arange(80, dtype=float)
    count_plus = np.array([plus @ poisson.pmf(k, support) for k in count_grid])
    count_minus = np.array([minus @ poisson.pmf(k, support) for k in count_grid])
    count_raw_plus = np.array([
        count_grid ** r @ count_plus for r in range(1, order + 1)])
    count_raw_minus = np.array([
        count_grid ** r @ count_minus for r in range(1, order + 1)])
    factorial_plus = factorial_cumulants4(cumulants4(count_raw_plus))
    factorial_minus = factorial_cumulants4(cumulants4(count_raw_minus))
    factorial_gap = np.max(np.abs(factorial_plus - factorial_minus))

    zero_gap = count_plus[0] - count_minus[0]
    exact_zero_gap = (2.0 * delta * math.exp(-1.0)
                      * (1.0 - math.exp(-1.0)) ** n)
    total_variation = 0.5 * np.sum(np.abs(count_plus - count_minus))
    total_variation_tail_bound = poisson.sf(count_grid[-1], support.max())

    assert np.all(plus > 0.0) and np.all(minus > 0.0)
    assert abs(plus.sum() - 1.0) < 1e-15
    assert abs(minus.sum() - 1.0) < 1e-15
    assert np.max(np.abs(raw_plus - raw_minus)) < 1e-12
    assert intensity_gap < 1e-10
    assert factorial_gap < 1e-9
    assert abs(zero_gap - exact_zero_gap) < 1e-15
    assert total_variation > 0.0
    assert total_variation_tail_bound < 1e-50
    return {
        "support": support,
        "plus": plus,
        "minus": minus,
        "intensity_gap": intensity_gap,
        "factorial_gap": factorial_gap,
        "zero_gap": zero_gap,
        "total_variation": total_variation,
        "total_variation_tail_bound": total_variation_tail_bound,
    }


def poisson_inverse_instability():
    """Quantify why mixed-Poisson identifiability is not TV stability.

    Point masses at distinct intensities have total-variation distance one,
    although their Poisson images become arbitrarily close when the two
    intensities coalesce. Consecutive large intensities additionally keep
    Wasserstein-1 distance one while their count laws converge in TV.
    """
    center = 4.5
    steps = np.array([1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
    local_tv = []
    local_cutoffs = []
    for step in steps:
        upper = center + step
        cutoff = math.floor(step / math.log(upper / center))
        local_cutoffs.append(cutoff)
        local_tv.append(
            poisson.cdf(cutoff, center) - poisson.cdf(cutoff, upper))
    local_tv = np.array(local_tv)
    local_limit = poisson.pmf(math.floor(center), center)

    means = np.array([10, 100, 1000, 10000, 100000, 1000000], int)
    consecutive_tv = np.array([
        poisson.cdf(int(mean), mean)
        - poisson.cdf(int(mean), mean + 1.0)
        for mean in means
    ])
    scaled_tv = np.sqrt(means) * consecutive_tv
    asymptotic_constant = 1.0 / math.sqrt(2.0 * math.pi)
    hellinger_bounds = np.array([
        math.sqrt(-math.expm1(
            -(math.sqrt(mean + 1.0) - math.sqrt(mean)) ** 2))
        for mean in means
    ])

    assert set(local_cutoffs) == {4}
    assert abs(local_tv[-1] / steps[-1] - local_limit) < 2e-7
    assert abs(scaled_tv[-1] - asymptotic_constant) < 2e-7
    assert np.all(consecutive_tv <= hellinger_bounds)
    return {
        "center": center,
        "steps": steps,
        "local_tv": local_tv,
        "local_limit": local_limit,
        "means": means,
        "consecutive_tv": consecutive_tv,
        "scaled_tv": scaled_tv,
        "asymptotic_constant": asymptotic_constant,
        "hellinger_bounds": hellinger_bounds,
    }


def poisson_mixture_tv_minimax():
    """Exact two-point lower bound for estimating a Poisson mixing law.

    Under point-mass mixing at ``a`` or ``b``, ``n`` iid mixed-Poisson
    observations are iid Poisson.  Their likelihood ratio depends only on
    the sum, and the conditional allocation given the sum is the same under
    both hypotheses.  Consequently their product total variation is exactly
    that between Pois(n*a) and Pois(n*b).  We take b=a+1/n inside [4,5].

    A maximal-coupling proof gives, for every mixing-law estimator,

        max_i E_i TV(mu_hat, mu_i)
          >= TV(mu_0,mu_1) * (1-TV(P_0^n,P_1^n)) / 2.

    The mixing-law distance is one for the two distinct point masses.
    """
    intensity = 4.5
    sample_sizes = np.array(
        [10, 100, 1000, 10000, 100000, 1000000], dtype=int
    )
    product_tv = []
    cutoffs = []
    hellinger_bounds = []
    for sample_size in sample_sizes:
        lower_mean = sample_size * intensity
        upper_intensity = intensity + 1.0 / sample_size
        upper_mean = sample_size * upper_intensity
        cutoff = math.floor(
            (upper_mean - lower_mean) / math.log(upper_mean / lower_mean)
        )
        cutoffs.append(cutoff)
        product_tv.append(
            poisson.cdf(cutoff, lower_mean)
            - poisson.cdf(cutoff, upper_mean)
        )
        hellinger_bounds.append(
            math.sqrt(
                -math.expm1(
                    -sample_size
                    * (math.sqrt(upper_intensity) - math.sqrt(intensity)) ** 2
                )
            )
        )
    product_tv = np.asarray(product_tv)
    hellinger_bounds = np.asarray(hellinger_bounds)
    risk_lower_bounds = 0.5 * (1.0 - product_tv)
    scaled_tv = np.sqrt(sample_sizes) * product_tv
    asymptotic_constant = 1.0 / math.sqrt(2.0 * math.pi * intensity)

    assert np.all(np.asarray(cutoffs) == intensity * sample_sizes)
    assert np.all(product_tv <= hellinger_bounds)
    assert np.all(np.diff(product_tv) < 0.0)
    assert np.all(np.diff(risk_lower_bounds) > 0.0)
    assert abs(scaled_tv[-1] - asymptotic_constant) < 2e-8
    assert risk_lower_bounds[-1] > 0.4999
    return {
        "intensity": intensity,
        "sample_sizes": sample_sizes,
        "product_tv": product_tv,
        "hellinger_bounds": hellinger_bounds,
        "risk_lower_bounds": risk_lower_bounds,
        "scaled_tv": scaled_tv,
        "asymptotic_constant": asymptotic_constant,
    }


def poisson_mixture_w1_local_minimax():
    """Exact local two-point certificate for Wasserstein-1 risk.

    Restrict the unknown mixing law to point masses on [4, 5].  At the two
    local alternatives ``delta_a`` and ``delta_(a+h/sqrt(n))``, Wasserstein-1
    distance is ``h/sqrt(n)`` and the sufficient statistic in the count
    experiment is Poisson with means ``n*a`` and ``n*a+h*sqrt(n)``.

    The two-point metric-loss inequality therefore gives

        sqrt(n) R_n >= h/2 * (1 - TV(Pois(n*a), Pois(n*a+h*sqrt(n)))).

    If ``z=h/(2*sqrt(a))``, the exact one-crossing formula and the normal
    limit give TV -> 1-2*Phi(-z).  Thus the limiting lower-bound constant is
    ``h*Phi(-z)``.  It is maximized by the unique positive solution of
    ``Phi(-z)=z*phi(z)``.
    """
    intensity = 4.5
    z_star = brentq(
        lambda z: norm.cdf(-z) - z * norm.pdf(z), 0.01, 3.0
    )
    h_star = 2.0 * math.sqrt(intensity) * z_star
    sample_sizes = np.array(
        [100, 1000, 10000, 100000, 1000000], dtype=int
    )
    product_tv = []
    cutoffs = []
    upper_intensities = []
    for sample_size in sample_sizes:
        lower_mean = sample_size * intensity
        upper_intensity = intensity + h_star / math.sqrt(sample_size)
        upper_mean = sample_size * upper_intensity
        cutoff = math.floor(
            (upper_mean - lower_mean) / math.log(upper_mean / lower_mean)
        )
        cutoffs.append(cutoff)
        upper_intensities.append(upper_intensity)
        product_tv.append(
            poisson.cdf(cutoff, lower_mean)
            - poisson.cdf(cutoff, upper_mean)
        )
    product_tv = np.asarray(product_tv)
    upper_intensities = np.asarray(upper_intensities)
    scaled_risk_lower_bounds = 0.5 * h_star * (1.0 - product_tv)
    asymptotic_product_tv = 1.0 - 2.0 * norm.cdf(-z_star)
    asymptotic_constant = h_star * norm.cdf(-z_star)

    assert np.all(upper_intensities <= 5.0)
    assert np.all(np.diff(product_tv) > 0.0)
    assert np.all(np.diff(scaled_risk_lower_bounds) < 0.0)
    assert abs(norm.cdf(-z_star) - z_star * norm.pdf(z_star)) < 1e-14
    assert abs(product_tv[-1] - asymptotic_product_tv) < 2e-4
    assert abs(scaled_risk_lower_bounds[-1] - asymptotic_constant) < 2e-4
    return {
        "intensity": intensity,
        "z_star": z_star,
        "h_star": h_star,
        "sample_sizes": sample_sizes,
        "upper_intensities": upper_intensities,
        "cutoffs": np.asarray(cutoffs),
        "product_tv": product_tv,
        "scaled_risk_lower_bounds": scaled_risk_lower_bounds,
        "asymptotic_product_tv": asymptotic_product_tv,
        "asymptotic_constant": asymptotic_constant,
    }


def poisson_point_mass_w1_upper():
    """Constructive Wasserstein-1 upper bound on the point-mass submodel.

    If the mixing law is ``delta_lambda``, the observations are iid
    ``Pois(lambda)`` and their sum S is ``Pois(n*lambda)``.  The estimator
    projecting ``S/n`` to [4,5] and taking the point mass there cannot
    increase its Wasserstein loss.  The unprojected benchmark satisfies

        E abs(S/n-lambda)
          = 2*lambda*P(Pois(n*lambda)=floor(n*lambda))
          <= sqrt(lambda/n),

    Equality follows by splitting the centered Poisson variable at its
    mean and using k p_mu(k) = mu p_mu(k-1); the inequality is Cauchy--Schwarz.
    On lambda in [4,5] this gives the uniform upper bound sqrt(5/n).
    """
    intensity = 4.5
    sample_sizes = np.array(
        [10, 100, 1000, 10000, 100000, 1000000], dtype=int
    )
    means = sample_sizes * intensity
    exact_risks = np.array([
        2.0 * intensity * poisson.pmf(math.floor(mean), mean)
        for mean in means
    ])
    scaled_exact_risks = np.sqrt(sample_sizes) * exact_risks
    pointwise_limit = math.sqrt(2.0 * intensity / math.pi)
    uniform_upper_bounds = np.sqrt(5.0 / sample_sizes)

    assert np.all(exact_risks <= np.sqrt(intensity / sample_sizes))
    assert np.all(exact_risks <= uniform_upper_bounds)
    assert abs(scaled_exact_risks[-1] - pointwise_limit) < 2e-7
    return {
        "intensity": intensity,
        "sample_sizes": sample_sizes,
        "exact_risks": exact_risks,
        "scaled_exact_risks": scaled_exact_risks,
        "pointwise_limit": pointwise_limit,
        "uniform_upper_bounds": uniform_upper_bounds,
        "uniform_scaled_bound": math.sqrt(5.0),
    }


def poisson_mixture_w1_nonparametric_lower(lower=4.0, width=1.0):
    """Parity-binomial lower bound for unrestricted compact Poisson mixtures.

    Let J have a Binomial(L, 1/2) law and condition it on even or odd parity.
    The two mixing laws of ``lower + width*J/L`` agree in their first L-1
    moments, while the alternating-binomial CDF identity gives their exact
    Wasserstein distance ``width/L``.

    A useful probabilistic proof controls the induced count laws.  Write the
    count as ``Z + sum_i B_i X_i``, where ``Z`` is Pois(lower), the B_i are
    iid Bernoulli(1/2) conditioned on parity, and the X_i are iid
    Pois(width/L).  If any X_i is zero, flipping the first such B_i changes
    parity without changing the count.  Hence

        TV(P_even, P_odd) <= (1-exp(-width/L))**L.

    Tensorization and the two-point metric-loss inequality then yield

        R_n >= width/(2L) * (1-n*(1-exp(-width/L))**L).

    Taking L just above log(n)/log(log(n)) proves
    liminf log(n)/log(log(n)) R_n >= width/2.
    """
    if lower < 0.0 or width <= 0.0:
        raise ValueError("Poisson intensities require lower >= 0 and width > 0")
    # At much larger orders the exact count-law difference falls below
    # double-precision resolution, so the direct pmf check stops at 12.
    check_orders = np.array([4, 8, 12], dtype=int)
    exact_w1 = []
    count_tv = []
    coupling_bounds = []
    moment_gaps = []
    count_grid = np.arange(80, dtype=float)

    for order in check_orders:
        indices = np.arange(order + 1, dtype=int)
        binomial = np.array(
            [math.comb(int(order), int(j)) for j in indices], dtype=float
        )
        parity_weights = binomial / 2.0 ** (order - 1)
        even = np.where(indices % 2 == 0, parity_weights, 0.0)
        odd = np.where(indices % 2 == 1, parity_weights, 0.0)
        support = lower + width * indices / order

        # The integer finite-difference identity is exact before the affine
        # change of support, and hence proves equality of all lower moments.
        exact_differences = [
            sum((-1) ** int(j) * math.comb(int(order), int(j))
                * int(j) ** degree for j in indices)
            for degree in range(order)
        ]
        moment_gap = max(abs(x) for x in exact_differences)
        moment_gaps.append(moment_gap)

        cdf_difference = np.cumsum(even - odd)
        w1 = width / order * np.sum(np.abs(cdf_difference[:-1]))
        exact_w1.append(w1)

        count_even = np.array([
            even @ poisson.pmf(k, support) for k in count_grid
        ])
        count_odd = np.array([
            odd @ poisson.pmf(k, support) for k in count_grid
        ])
        tv = 0.5 * np.sum(np.abs(count_even - count_odd))
        coupling = (-math.expm1(-width / order)) ** order
        count_tv.append(tv)
        coupling_bounds.append(coupling)

        assert abs(even.sum() - 1.0) < 1e-15
        assert abs(odd.sum() - 1.0) < 1e-15
        assert moment_gap == 0
        assert abs(w1 - width / order) < 2e-15
        assert tv <= coupling * (1.0 + 2e-12)

    # Optimize the explicit finite-n lower bound over integer L.  The last
    # rows use log(n) directly so that the asymptotic regime can be certified
    # without overflowing floating-point sample sizes.
    log_sample_sizes = np.array([
        math.log(1e3), math.log(1e6), math.log(1e12),
        math.log(1e24), math.log(1e48), 1000.0,
    ])
    sample_labels = ("10^3", "10^6", "10^12", "10^24", "10^48", "e^1000")
    optimal_orders = []
    risk_lower_bounds = []
    scaled_lower_bounds = []
    product_tv_bounds = []
    for log_n in log_sample_sizes:
        best = (-1.0, None, None)
        order = 2
        while True:
            log_one_count_bound = (
                order * math.log(-math.expm1(-width / order))
            )
            log_product_bound = log_n + log_one_count_bound
            product_bound = (
                math.exp(log_product_bound)
                if log_product_bound < 0.0 else 1.0
            )
            lower_bound = width / (2.0 * order) * (1.0 - product_bound)
            if lower_bound > best[0]:
                best = (lower_bound, order, product_bound)
            # Every later candidate is at most width/(2*(order+1)), so this
            # certifies that the integer optimizer has been found globally.
            if width / (2.0 * (order + 1)) <= best[0]:
                break
            order += 1
        lower_bound, order, product_bound = best
        optimal_orders.append(order)
        risk_lower_bounds.append(lower_bound)
        product_tv_bounds.append(product_bound)
        scaled_lower_bounds.append(
            log_n / math.log(log_n) * lower_bound
        )

    risk_lower_bounds = np.asarray(risk_lower_bounds)
    scaled_lower_bounds = np.asarray(scaled_lower_bounds)
    product_tv_bounds = np.asarray(product_tv_bounds)
    assert np.all(risk_lower_bounds > 0.0)
    assert scaled_lower_bounds[-1] > 0.37
    assert product_tv_bounds[-1] < 0.2
    return {
        "lower": lower,
        "width": width,
        "check_orders": check_orders,
        "moment_gaps": np.asarray(moment_gaps),
        "exact_w1": np.asarray(exact_w1),
        "count_tv": np.asarray(count_tv),
        "coupling_bounds": np.asarray(coupling_bounds),
        "sample_labels": sample_labels,
        "log_sample_sizes": log_sample_sizes,
        "optimal_orders": np.asarray(optimal_orders),
        "risk_lower_bounds": risk_lower_bounds,
        "scaled_lower_bounds": scaled_lower_bounds,
        "product_tv_bounds": product_tv_bounds,
        "asymptotic_scaled_lower_bound": width / 2.0,
    }


def poisson_mixture_w1_moment_upper(max_intensity=5.0, degree_fraction=0.5):
    """Certificate for the ingredients of the moment-estimator upper bound.

    If ``N | theta`` is Poisson, the unbiased normalized moment statistic is
    ``(N)_k / M^k``.  The exact product identity

        (N)_k^2 = sum_j binom(k,j)^2 j! (N)_{2k-j}

    gives the uniform variance bound used on the page.  Shifted Chebyshev
    polynomials obey ``q_{j+1}=(4x-2)q_j-q_{j-1}``, so their monomial
    coefficient l1 norms are at most ``7^j``.  Finally we report the logarithm
    of the stochastic factor for ``L=floor(c log(n)/loglog(n))``; this avoids
    constructing astronomically large sample sizes.
    """
    if max_intensity <= 0.0:
        raise ValueError("max_intensity must be positive")
    if not 0.0 < degree_fraction < 1.0:
        raise ValueError("degree_fraction must lie in (0,1)")

    orders = np.arange(1, 9, dtype=int)
    count_grid = np.arange(180, dtype=int)
    pmf = poisson.pmf(count_grid, max_intensity)
    direct_second_moments = []
    formula_second_moments = []
    for order in orders:
        falling = np.zeros_like(count_grid, dtype=float)
        eligible = count_grid >= order
        falling[eligible] = np.array([
            math.prod(range(int(n - order + 1), int(n + 1)))
            for n in count_grid[eligible]
        ], dtype=float)
        direct_second_moments.append(float((falling * falling) @ pmf))
        formula_second_moments.append(sum(
            math.comb(int(order), j) ** 2 * math.factorial(j)
            * max_intensity ** (2 * int(order) - j)
            for j in range(int(order) + 1)
        ))
    direct_second_moments = np.asarray(direct_second_moments)
    formula_second_moments = np.asarray(formula_second_moments)
    relative_second_moment_error = np.max(np.abs(
        direct_second_moments / formula_second_moments - 1.0))

    # Coefficients are stored in ascending monomial order.
    chebyshev_l1 = [1, 3]
    q_previous = np.array([1], dtype=object)
    q_current = np.array([-1, 2], dtype=object)
    for degree in range(1, 12):
        scaled = np.pad(-2 * q_current, (0, 1))
        shifted = np.pad(4 * q_current, (1, 0))
        previous = np.pad(q_previous, (0, len(q_current) + 1 - len(q_previous)))
        q_next = scaled + shifted - previous
        chebyshev_l1.append(sum(abs(int(x)) for x in q_next))
        q_previous, q_current = q_current, q_next
    chebyshev_l1 = np.asarray(chebyshev_l1, dtype=float)
    degrees = np.arange(len(chebyshev_l1))

    log_sample_sizes = np.array([100.0, 300.0, 1000.0, 3000.0])
    moment_degrees = np.maximum(1, np.floor(
        degree_fraction * log_sample_sizes / np.log(log_sample_sizes)
    ).astype(int))
    inverse_scale = max(1.0, 1.0 / max_intensity)
    log_stochastic_factors = (
        -0.5 * log_sample_sizes
        + 1.5 * np.log(moment_degrees + 1.0)
        + moment_degrees * np.log(
            14.0 * np.sqrt(inverse_scale * moment_degrees)
        )
    )

    assert relative_second_moment_error < 3e-13
    assert np.all(chebyshev_l1 <= 7.0 ** degrees)
    assert np.all(np.diff(log_stochastic_factors) < 0.0)
    assert log_stochastic_factors[-1] < -400.0
    return {
        "orders": orders,
        "relative_second_moment_error": relative_second_moment_error,
        "chebyshev_degrees": degrees,
        "chebyshev_l1": chebyshev_l1,
        "log_sample_sizes": log_sample_sizes,
        "moment_degrees": moment_degrees,
        "log_stochastic_factors": log_stochastic_factors,
        "degree_fraction": degree_fraction,
        "max_intensity": max_intensity,
    }


def mixed_cumulants22(raw):
    """Mixed cumulants kappa_11, kappa_21, kappa_12 and kappa_22.

    ``raw[a, b]`` is E[X^a Y^b], with 0 <= a,b <= 2.  The last
    expression is the joint-cumulant partition formula for (X,X,Y,Y).
    """
    mx, my = raw[1, 0], raw[0, 1]
    m20, m02, m11 = raw[2, 0], raw[0, 2], raw[1, 1]
    m21, m12, m22 = raw[2, 1], raw[1, 2], raw[2, 2]
    k11 = m11 - mx * my
    k21 = m21 - m20 * my - 2.0 * m11 * mx + 2.0 * mx ** 2 * my
    k12 = m12 - m02 * mx - 2.0 * m11 * my + 2.0 * my ** 2 * mx
    k22 = (m22 - m20 * m02 - 2.0 * m11 ** 2
           - 2.0 * m21 * my - 2.0 * m12 * mx
           + 2.0 * m20 * my ** 2 + 2.0 * m02 * mx ** 2
           + 8.0 * m11 * mx * my - 6.0 * mx ** 2 * my ** 2)
    return np.array([k11, k21, k12, k22])


def mixed_factorial_cumulants22(ordinary):
    """Convert mixed ordinary cumulants by signed Stirling transforms."""
    k11, k21, k12, k22 = ordinary
    return np.array([k11, k21 - k11, k12 - k11,
                     k22 - k21 - k12 + k11])


def _set_partitions(labels):
    """Yield every set partition of a short labelled tuple."""
    if not labels:
        yield []
        return
    first, rest = labels[0], labels[1:]
    for partition in _set_partitions(rest):
        yield [(first,)] + partition
        for j in range(len(partition)):
            yield (partition[:j]
                   + [(first,) + partition[j]]
                   + partition[j + 1:])


def joint_cumulant(repeats, moment):
    """Joint cumulant from a callable returning multivariate raw moments.

    ``repeats[j]`` is the number of copies of coordinate ``j``.  The usual
    partition formula is practical here because the certificate stops at six
    arguments (Bell number 203).
    """
    labels = tuple(axis for axis, count in enumerate(repeats)
                   for _ in range(count))
    total = 0.0
    for partition in _set_partitions(labels):
        coefficient = ((-1.0) ** (len(partition) - 1)
                       * math.factorial(len(partition) - 1))
        term = coefficient
        for block in partition:
            degrees = tuple(block.count(axis) for axis in range(len(repeats)))
            term *= moment(degrees)
        total += term
    return total


def bivariate_factorial_cumulant_grid(p, degree=3):
    """Mixed factorial cumulants from a bivariate count distribution."""
    grid = np.arange(p.shape[0], dtype=float)
    falling = np.ones((degree + 1, len(grid)))
    for r in range(1, degree + 1):
        falling[r] = falling[r - 1] * (grid - r + 1.0)

    def factorial_moment(degrees):
        a, b = degrees
        return np.einsum("i,j,ij->", falling[a], falling[b], p)

    answer = np.empty((degree, degree))
    for r in range(1, degree + 1):
        for s in range(1, degree + 1):
            answer[r - 1, s - 1] = joint_cumulant(
                (r, s), factorial_moment)
    return answer


def integrated_intensity_cumulants(Q, rates, T, prior, degree=4):
    """Cumulants of integral_0^T rates[Y_s] ds from a backward moment ODE."""
    Q = np.asarray(Q, float)
    rates = np.asarray(rates, float)
    prior = np.asarray(prior, float)
    n = len(rates)
    A = np.zeros(((degree + 1) * n, (degree + 1) * n))
    D = np.diag(rates)
    for k in range(degree + 1):
        sl = slice(k * n, (k + 1) * n)
        A[sl, sl] = Q
        if k:
            A[sl, slice((k - 1) * n, k * n)] = k * D
    y0 = np.zeros((degree + 1) * n)
    y0[:n] = 1.0
    y = expm(A * T) @ y0
    raw = np.array([prior @ y[k * n:(k + 1) * n] for k in range(1, degree + 1)])
    return cumulants4(raw)


def count_cumulants(Q, rates, T, prior, max_count=80):
    """Cumulants from the forward regime/count master equation."""
    Q = np.asarray(Q, float)
    rates = np.asarray(rates, float)
    prior = np.asarray(prior, float)
    n = len(rates)
    D = np.diag(rates)
    A = np.zeros(((max_count + 1) * n, (max_count + 1) * n))
    for k in range(max_count + 1):
        sl = slice(k * n, (k + 1) * n)
        A[sl, sl] = Q.T - D
        if k:
            A[sl, slice((k - 1) * n, k * n)] = D
    y0 = np.zeros((max_count + 1) * n)
    y0[:n] = prior
    y = expm(A * T) @ y0
    p = np.array([y[k * n:(k + 1) * n].sum() for k in range(max_count + 1)])
    k = np.arange(max_count + 1, dtype=float)
    raw = np.array([np.dot(k ** r, p) for r in range(1, 5)])
    return cumulants4(raw), p


def integrated_intensity_mixed_cumulants(Q, rates1, rates2, T, prior):
    """Mixed cumulants of two integrated rates, through bidegree (2,2)."""
    Q = np.asarray(Q, float)
    rates1, rates2 = np.asarray(rates1, float), np.asarray(rates2, float)
    prior = np.asarray(prior, float)
    n = len(prior)
    pairs = [(a, b) for a in range(3) for b in range(3)]
    pos = {pair: j for j, pair in enumerate(pairs)}
    A = np.zeros((len(pairs) * n, len(pairs) * n))
    for j, (a, b) in enumerate(pairs):
        sl = slice(j * n, (j + 1) * n)
        A[sl, sl] = Q
        if a:
            lo = pos[a - 1, b]
            A[sl, slice(lo * n, (lo + 1) * n)] = a * np.diag(rates1)
        if b:
            lo = pos[a, b - 1]
            A[sl, slice(lo * n, (lo + 1) * n)] = b * np.diag(rates2)
    y0 = np.zeros(len(pairs) * n)
    y0[:n] = 1.0
    y = expm(A * T) @ y0
    raw = np.empty((3, 3))
    for j, (a, b) in enumerate(pairs):
        raw[a, b] = prior @ y[j * n:(j + 1) * n]
    return mixed_cumulants22(raw)


def integrated_intensity_joint_cumulant111(Q, rates1, rates2, rates3,
                                             T, prior):
    """Third joint cumulant of three integrated rates."""
    Q = np.asarray(Q, float)
    rates = [np.asarray(item, float) for item in (rates1, rates2, rates3)]
    prior = np.asarray(prior, float)
    n = len(prior)
    triples = [(a, b, c) for a in range(2) for b in range(2)
               for c in range(2)]
    pos = {triple: j for j, triple in enumerate(triples)}
    A = np.zeros((len(triples) * n, len(triples) * n))
    for j, triple in enumerate(triples):
        sl = slice(j * n, (j + 1) * n)
        A[sl, sl] = Q
        for axis, degree in enumerate(triple):
            if degree:
                lower = list(triple)
                lower[axis] -= 1
                lo = pos[tuple(lower)]
                A[sl, slice(lo * n, (lo + 1) * n)] = np.diag(rates[axis])
    y0 = np.zeros(len(triples) * n)
    y0[:n] = 1.0
    y = expm(A * T) @ y0
    raw = {}
    for j, triple in enumerate(triples):
        raw[triple] = prior @ y[j * n:(j + 1) * n]
    ma, mb, mc = raw[1, 0, 0], raw[0, 1, 0], raw[0, 0, 1]
    return (raw[1, 1, 1] - ma * raw[0, 1, 1]
            - mb * raw[1, 0, 1] - mc * raw[1, 1, 0]
            + 2.0 * ma * mb * mc)


def integrated_intensity_joint_cumulant(Q, rates, repeats, T, prior):
    """Joint cumulant of repeated integrated-rate coordinates."""
    Q = np.asarray(Q, float)
    rates = [np.asarray(item, float) for item in rates]
    repeats = tuple(repeats)
    prior = np.asarray(prior, float)
    n = len(prior)
    degrees = list(itertools.product(*[range(r + 1) for r in repeats]))
    pos = {degree: j for j, degree in enumerate(degrees)}
    generator = np.zeros((len(degrees) * n, len(degrees) * n))
    for j, degree in enumerate(degrees):
        sl = slice(j * n, (j + 1) * n)
        generator[sl, sl] = Q
        for axis, power in enumerate(degree):
            if power:
                lower = list(degree)
                lower[axis] -= 1
                lo = pos[tuple(lower)]
                generator[sl, slice(lo * n, (lo + 1) * n)] = (
                    power * np.diag(rates[axis]))
    y0 = np.zeros(len(degrees) * n)
    y0[:n] = 1.0
    y = expm(generator * T) @ y0
    raw = {
        degree: prior @ y[j * n:(j + 1) * n]
        for j, degree in enumerate(degrees)
    }
    return joint_cumulant(repeats, raw.__getitem__)


def common_shock_factorial_cumulant(Q, rates1, rates2, common_rates,
                                      r, s, T, prior):
    """All-order common-shock formula for mixed factorial cumulants."""
    rates_a = np.asarray(rates1) + np.asarray(common_rates)
    rates_b = np.asarray(rates2) + np.asarray(common_rates)
    answer = 0.0
    for k in range(min(r, s) + 1):
        coefficient = (math.comb(r, k) * math.comb(s, k)
                       * math.factorial(k))
        answer += coefficient * integrated_intensity_joint_cumulant(
            Q, (rates_a, rates_b, common_rates), (r - k, s - k, k),
            T, prior)
    return answer


def mark_factorial_moments(marks, probabilities, degree):
    """Return E[(J1)_a (J2)_b], optionally for each hidden state."""
    marks = np.asarray(marks, int)
    probabilities = np.asarray(probabilities, float)
    if marks.ndim != 2 or marks.shape[1] != 2:
        raise ValueError("marks must have shape (number of marks, 2)")
    if np.any(marks < 0) or np.any(probabilities < 0):
        raise ValueError("marks and probabilities must be nonnegative")
    if probabilities.ndim == 1:
        if len(probabilities) != len(marks) or not np.isclose(
                probabilities.sum(), 1.0):
            raise ValueError("mark probabilities must match and sum to one")
    elif probabilities.ndim == 2:
        if probabilities.shape[1] != len(marks) or not np.allclose(
                probabilities.sum(axis=1), 1.0):
            raise ValueError("each statewise mark law must sum to one")
    else:
        raise ValueError("mark probabilities must be a vector or matrix")
    falling = np.ones((2, degree + 1, len(marks)))
    for axis in range(2):
        for order in range(1, degree + 1):
            falling[axis, order] = (
                falling[axis, order - 1] * (marks[:, axis] - order + 1))
    if probabilities.ndim == 1:
        return np.einsum("ai,bi,i->ab", falling[0], falling[1], probabilities)
    return np.einsum("ai,bi,ki->kab", falling[0], falling[1], probabilities)


def marked_common_shock_factorial_cumulant(
        Q, rates1, rates2, common_rates, mark_moments, r, s, T, prior):
    """Partition formula for an arbitrarily marked common Poisson stream.

    A block containing ``a`` derivatives in the first coordinate and ``b``
    in the second contributes the integrated-rate variable W_ab.  Singleton
    variables are A=Lambda1+E[J1]C and B=Lambda2+E[J2]C; every larger block is
    E[(J1)_a (J2)_b] C.  Summing their joint cumulants over set partitions is
    the multivariate logarithmic Faa di Bruno formula.
    """
    rates1, rates2 = np.asarray(rates1), np.asarray(rates2)
    common_rates = np.asarray(common_rates)
    mark_moments = np.asarray(mark_moments)
    if mark_moments.ndim not in (2, 3):
        raise ValueError("mark moments must be common or state dependent")

    def marked_rates(block_type):
        if mark_moments.ndim == 2:
            coefficient = mark_moments[block_type]
        else:
            coefficient = mark_moments[(slice(None),) + block_type]
        return coefficient * common_rates

    variables = {
        (1, 0): rates1 + marked_rates((1, 0)),
        (0, 1): rates2 + marked_rates((0, 1)),
    }
    labels = tuple([0] * r + [1] * s)
    answer = 0.0
    for partition in _set_partitions(labels):
        multiplicities = {}
        block_rates = {}
        vanishes = False
        for block in partition:
            block_type = (block.count(0), block.count(1))
            if block_type not in variables:
                rates = marked_rates(block_type)
                if np.all(rates == 0.0):
                    vanishes = True
                    break
                variables[block_type] = rates
            multiplicities[block_type] = multiplicities.get(block_type, 0) + 1
            block_rates[block_type] = variables[block_type]
        if vanishes:
            continue
        types = tuple(multiplicities)
        answer += integrated_intensity_joint_cumulant(
            Q,
            tuple(block_rates[item] for item in types),
            tuple(multiplicities[item] for item in types),
            T,
            prior,
        )
    return answer


def common_shock_factorial_cumulants22(Q, rates1, rates2, common_rates,
                                        T, prior):
    """Predicted mixed factorial cumulants with a shared Poisson component.

    Conditional on the regime path, N1=I1+C and N2=I2+C, where the three
    Poisson components have cumulative intensities Lambda1, Lambda2 and Cbar.
    Put A=Lambda1+Cbar and B=Lambda2+Cbar.  Substitution of v=t1*t2 in the
    joint cumulant generating function gives the four returned terms.
    """
    rates_a = np.asarray(rates1) + np.asarray(common_rates)
    rates_b = np.asarray(rates2) + np.asarray(common_rates)
    base = integrated_intensity_mixed_cumulants(
        Q, rates_a, rates_b, T, prior)
    cov_ac = integrated_intensity_mixed_cumulants(
        Q, rates_a, common_rates, T, prior)[0]
    cov_bc = integrated_intensity_mixed_cumulants(
        Q, rates_b, common_rates, T, prior)[0]
    common_kappa = integrated_intensity_cumulants(
        Q, common_rates, T, prior)
    joint_abc = integrated_intensity_joint_cumulant111(
        Q, rates_a, rates_b, common_rates, T, prior)
    return base + np.array([
        common_kappa[0],
        2.0 * cov_ac,
        2.0 * cov_bc,
        4.0 * joint_abc + 2.0 * common_kappa[1],
    ])


def bivariate_count_mixed_cumulants(
        Q, rates1, rates2, T, prior, max_count=75, common_rates=None,
        common_marks=None, common_mark_probabilities=None):
    """Mixed count cumulants from a sparse two-count forward master equation."""
    Q = np.asarray(Q, float)
    rates1, rates2 = np.asarray(rates1, float), np.asarray(rates2, float)
    prior = np.asarray(prior, float)
    common_rates = (np.zeros_like(rates1) if common_rates is None
                    else np.asarray(common_rates, float))
    if common_marks is None:
        common_marks = np.array([[1, 1]])
        common_mark_probabilities = np.array([1.0])
    else:
        if common_mark_probabilities is None:
            raise ValueError("common mark probabilities are required")
        common_marks = np.asarray(common_marks, int)
        common_mark_probabilities = np.asarray(
            common_mark_probabilities, float)
    n = len(prior)
    if common_mark_probabilities.ndim == 1:
        probabilities_by_state = np.broadcast_to(
            common_mark_probabilities, (n, len(common_marks)))
    elif common_mark_probabilities.ndim == 2:
        probabilities_by_state = common_mark_probabilities
    else:
        probabilities_by_state = np.empty((0, 0))
    if (common_marks.ndim != 2 or common_marks.shape != (len(common_marks), 2)
            or len(common_marks) == 0
            or probabilities_by_state.shape != (n, len(common_marks))
            or np.any(common_marks < 0)
            or np.any(probabilities_by_state < 0)
            or not np.allclose(probabilities_by_state.sum(axis=1), 1.0)):
        raise ValueError("invalid common mark distribution")
    m = max_count + 1
    shift = diags(np.ones(m - 1), -1, shape=(m, m), format="csr")
    count_eye = eye(m, format="csr")
    regime_eye = eye(m * m, format="csr")
    within = Q.T - np.diag(rates1 + rates2 + common_rates)
    A = (kron(regime_eye, within, format="csr")
         + kron(kron(shift, count_eye), np.diag(rates1), format="csr")
         + kron(kron(count_eye, shift), np.diag(rates2), format="csr"))
    shift_powers = {
        0: count_eye,
        **{
            jump: diags(np.ones(m - jump), -jump, shape=(m, m), format="csr")
            for jump in range(1, common_marks.max() + 1)
        },
    }
    for mark, probabilities in zip(common_marks, probabilities_by_state.T):
        A += kron(
            kron(shift_powers[mark[0]], shift_powers[mark[1]]),
            np.diag(probabilities * common_rates),
            format="csr",
        )
    y0 = np.zeros(m * m * n)
    y0[:n] = prior
    p = np.asarray(expm_multiply(A * T, y0)).reshape(m, m, n).sum(axis=2)
    grid = np.arange(m, dtype=float)
    raw = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            raw[a, b] = np.einsum("i,j,ij->", grid ** a, grid ** b, p)
    return mixed_cumulants22(raw), p


def compound_poisson_tail_bound(
        threshold, horizon, idiosyncratic_rate, common_rate,
        marks, probabilities, coordinate):
    """Chernoff bound for a coordinate of the dominating marked count."""
    marks = np.asarray(marks, float)[:, coordinate]
    probabilities = np.asarray(probabilities, float)
    candidates = np.linspace(0.02, 4.0, 2000)
    mark_exponentials = np.exp(np.outer(candidates, marks))
    if probabilities.ndim == 1:
        mark_mgf = mark_exponentials @ probabilities
    else:
        mark_mgf = np.max(mark_exponentials @ probabilities.T, axis=1)
    exponents = (
        horizon * idiosyncratic_rate * np.expm1(candidates)
        + horizon * common_rate
        * (mark_mgf - 1.0)
        - threshold * candidates
    )
    return float(np.exp(exponents.min()))


def two_state_exact(rates, T, lam, sign=1.0):
    """Exact known-start mean and overdispersion for the symmetric chain."""
    mean_rate = 0.5 * (rates[0] + rates[1])
    half_difference = 0.5 * (rates[0] - rates[1])
    eps = 1.0 / lam
    L = 1.0 - math.exp(-2.0 * lam * T)
    mean = mean_rate * T + sign * 0.5 * eps * half_difference * L
    overdispersion = (eps * half_difference ** 2 * T
                      - eps ** 2 * half_difference ** 2 * (0.5 * L + 0.25 * L ** 2))
    return mean, overdispersion


def main():
    twins = finite_cumulant_twins()
    instability = poisson_inverse_instability()
    minimax = poisson_mixture_tv_minimax()
    w1_minimax = poisson_mixture_w1_local_minimax()
    w1_point_mass = poisson_point_mass_w1_upper()
    w1_nonparametric = poisson_mixture_w1_nonparametric_lower()
    w1_moment_upper = poisson_mixture_w1_moment_upper()

    # A genuinely nonreversible chain: all three stationary edge currents are nonzero.
    Q = np.array([[-3.0, 2.0, 1.0],
                  [1.0, -4.0, 3.0],
                  [2.0, 1.0, -3.0]])
    rates = np.array([0.5, 3.0, 6.0])
    prior = np.array([0.2, 0.5, 0.3])
    intensity_kappa = integrated_intensity_cumulants(Q, rates, 1.3, prior)
    count_kappa, p = count_cumulants(Q, rates, 1.3, prior)
    factorial_kappa = factorial_cumulants4(count_kappa)
    identity_error = np.max(np.abs(factorial_kappa - intensity_kappa))
    # Conditional on every path, Lambda_T <= max(rates) T, so this Poisson
    # survival probability bounds all mass omitted by the count truncation.
    tail_bound = poisson.sf(len(p) - 1, rates.max() * 1.3)

    # Two conditionally independent count streams share the same hidden chain.
    # Mixed factorial cumulants should remove each stream's own shot noise and
    # recover the joint cumulants of the two cumulative intensities.
    rates_b = np.array([4.0, 0.75, 2.5])
    mixed_intensity = integrated_intensity_mixed_cumulants(
        Q, rates, rates_b, 1.3, prior)
    mixed_count, p_bi = bivariate_count_mixed_cumulants(
        Q, rates, rates_b, 1.3, prior)
    mixed_factorial = mixed_factorial_cumulants22(mixed_count)
    mixed_error = np.max(np.abs(mixed_factorial - mixed_intensity))
    bivariate_tail_bound = (poisson.sf(p_bi.shape[0] - 1, rates.max() * 1.3)
                            + poisson.sf(p_bi.shape[1] - 1, rates_b.max() * 1.3))

    # Shared events are a different observation model.  Conditional on the
    # path, a third Poisson stream increments both observed counts at once.
    # Its shot noise survives the mixed factorial transform.  The predicted
    # terms follow by substituting v=t1*t2 in K(A*t1+B*t2+Cbar*v).
    common_rates = np.array([0.4, 1.2, 0.7])
    common_count, p_common = bivariate_count_mixed_cumulants(
        Q, rates, rates_b, 1.3, prior, max_count=80,
        common_rates=common_rates)
    common_factorial = mixed_factorial_cumulants22(common_count)
    common_predicted = common_shock_factorial_cumulants22(
        Q, rates, rates_b, common_rates, 1.3, prior)
    common_error = np.max(np.abs(common_factorial - common_predicted))
    common_factorial_grid = bivariate_factorial_cumulant_grid(
        p_common, degree=3)
    common_predicted_grid = np.array([
        [common_shock_factorial_cumulant(
            Q, rates, rates_b, common_rates, r, s, 1.3, prior)
         for s in range(1, 4)]
        for r in range(1, 4)
    ])
    common_all_order_error = np.max(
        np.abs(common_factorial_grid - common_predicted_grid))
    common_tail_bound = (
        poisson.sf(p_common.shape[0] - 1,
                   (rates + common_rates).max() * 1.3)
        + poisson.sf(p_common.shape[1] - 1,
                     (rates_b + common_rates).max() * 1.3)
    )

    # Marked common events replace the unit increment (1,1) by an iid
    # integer-valued vector J.  The independent forward equation shifts by
    # each possible mark.  The partition formula uses only the falling-
    # factorial moments E[(J1)_a (J2)_b] and integrated-rate cumulants.
    common_marks = np.array([
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [3, 1],
        [1, 3],
    ])
    common_mark_probabilities = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
    mark_moments = mark_factorial_moments(
        common_marks, common_mark_probabilities, degree=3)
    marked_count, p_marked = bivariate_count_mixed_cumulants(
        Q,
        rates,
        rates_b,
        1.3,
        prior,
        max_count=110,
        common_rates=common_rates,
        common_marks=common_marks,
        common_mark_probabilities=common_mark_probabilities,
    )
    marked_factorial_grid = bivariate_factorial_cumulant_grid(
        p_marked, degree=3)
    marked_predicted_grid = np.array([
        [marked_common_shock_factorial_cumulant(
            Q,
            rates,
            rates_b,
            common_rates,
            mark_moments,
            r,
            s,
            1.3,
            prior,
        ) for s in range(1, 4)]
        for r in range(1, 4)
    ])
    marked_error = np.max(
        np.abs(marked_factorial_grid - marked_predicted_grid))
    marked_tail_bound = sum(
        compound_poisson_tail_bound(
            p_marked.shape[axis] - 1,
            1.3,
            (rates, rates_b)[axis].max(),
            common_rates.max(),
            common_marks,
            common_mark_probabilities,
            axis,
        )
        for axis in range(2)
    )

    # The mark law may itself depend on the hidden state.  Conditional on
    # the regime path this is still a marked Poisson random measure, now with
    # statewise falling-factorial moments.  The same partition theorem uses
    # the integrated rates c_i E[(J1)_a(J2)_b | Y=i].
    state_mark_probabilities = np.array([
        [0.35, 0.20, 0.15, 0.10, 0.10, 0.10],
        [0.10, 0.25, 0.25, 0.20, 0.10, 0.10],
        [0.15, 0.10, 0.20, 0.15, 0.20, 0.20],
    ])
    state_mark_moments = mark_factorial_moments(
        common_marks, state_mark_probabilities, degree=3)
    _, p_state_marked = bivariate_count_mixed_cumulants(
        Q,
        rates,
        rates_b,
        1.3,
        prior,
        max_count=110,
        common_rates=common_rates,
        common_marks=common_marks,
        common_mark_probabilities=state_mark_probabilities,
    )
    state_marked_factorial_grid = bivariate_factorial_cumulant_grid(
        p_state_marked, degree=3)
    state_marked_predicted_grid = np.array([
        [marked_common_shock_factorial_cumulant(
            Q,
            rates,
            rates_b,
            common_rates,
            state_mark_moments,
            r,
            s,
            1.3,
            prior,
        ) for s in range(1, 4)]
        for r in range(1, 4)
    ])
    state_marked_error = np.max(np.abs(
        state_marked_factorial_grid - state_marked_predicted_grid))
    state_marked_tail_bound = sum(
        compound_poisson_tail_bound(
            p_state_marked.shape[axis] - 1,
            1.3,
            (rates, rates_b)[axis].max(),
            common_rates.max(),
            common_marks,
            state_mark_probabilities,
            axis,
        )
        for axis in range(2)
    )

    # The marked partition theorem must reduce exactly to the previous
    # matching formula when every common mark is (1,1).
    unit_mark_moments = mark_factorial_moments(
        np.array([[1, 1]]), np.array([1.0]), degree=3)
    unit_partition_grid = np.array([
        [marked_common_shock_factorial_cumulant(
            Q,
            rates,
            rates_b,
            common_rates,
            unit_mark_moments,
            r,
            s,
            1.3,
            prior,
        ) for s in range(1, 4)]
        for r in range(1, 4)
    ])
    unit_reduction_error = np.max(
        np.abs(unit_partition_grid - common_predicted_grid))

    # With only a deterministic common component, marginal cumulative rates
    # have no stochastic covariance, yet kappa_11^(F) equals its mean.  This
    # is the sharp counterexample to identification without conditional
    # independence.
    common_only_rate = 0.8
    zero = np.zeros(3)
    deterministic_common = np.full(3, common_only_rate)
    common_only_count, _ = bivariate_count_mixed_cumulants(
        Q, zero, zero, 1.3, prior, max_count=45,
        common_rates=deterministic_common)
    common_only_factorial = mixed_factorial_cumulants22(common_only_count)
    common_only_expected = np.array([common_only_rate * 1.3, 0.0, 0.0, 0.0])
    common_only_error = np.max(
        np.abs(common_only_factorial - common_only_expected))

    # The page's symmetric two-state example, checked for both known starts.
    rates2 = np.array([8.0, 1.0])
    closed_errors = []
    for lam in (5.0, 10.0, 20.0, 40.0):
        Q2 = lam * np.array([[-1.0, 1.0], [1.0, -1.0]])
        for start, sign in ((0, 1.0), (1, -1.0)):
            prior2 = np.eye(2)[start]
            kap = integrated_intensity_cumulants(Q2, rates2, 1.0, prior2)
            mean, overdispersion = two_state_exact(rates2, 1.0, lam, sign)
            closed_errors.extend((abs(kap[0] - mean), abs(kap[1] - overdispersion)))

    # The omitted boundary term is second order at fixed positive maturity.
    lams = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    residuals = []
    half_difference = 3.5
    for lam in lams:
        _, exact_gap = two_state_exact(rates2, 1.0, lam)
        residuals.append(abs(exact_gap - half_difference ** 2 / lam))
    residual_order = -np.polyfit(np.log(lams), np.log(residuals), 1)[0]

    # Numbers printed on the page, obtained here by independent constructions.
    Q10 = 10.0 * np.array([[-1.0, 1.0], [1.0, -1.0]])
    intensity10 = integrated_intensity_cumulants(Q10, rates2, 1.0, [1.0, 0.0])
    count10, _ = count_cumulants(Q10, rates2, 1.0, [1.0, 0.0])

    print("finite-order twin intensity-cumulant gap",
          f"{twins['intensity_gap']:.3e}")
    print("finite-order twin factorial-cumulant gap",
          f"{twins['factorial_gap']:.3e}")
    print("finite-order twin zero-count gap", f"{twins['zero_gap']:.12f}")
    print("finite-order twin count total variation",
          f"{twins['total_variation']:.12f}")
    print("finite-order twin total-variation tail bound",
          f"{twins['total_variation_tail_bound']:.3e}")
    print("local inverse-instability TV / step",
          f"{instability['local_tv'][-1] / instability['steps'][-1]:.12f}",
          "limit", f"{instability['local_limit']:.12f}")
    print("large-intensity sqrt(m) TV",
          f"{instability['scaled_tv'][-1]:.12f}",
          "limit", f"{instability['asymptotic_constant']:.12f}")
    print("large-intensity TV and Hellinger upper bound",
          f"{instability['consecutive_tv'][-1]:.12e}",
          f"{instability['hellinger_bounds'][-1]:.12e}")
    print("unrestricted mixing-law TV minimax certificate")
    print(" n       product TV       Le Cam risk lower bound")
    for sample_size, product_tv, risk_bound in zip(
            minimax["sample_sizes"], minimax["product_tv"],
            minimax["risk_lower_bounds"]):
        print(f"{sample_size:7d}   {product_tv:.12f}       {risk_bound:.12f}")
    print("sqrt(n) product TV",
          f"{minimax['scaled_tv'][-1]:.12f}",
          "limit", f"{minimax['asymptotic_constant']:.12f}")
    print("local Wasserstein-1 minimax certificate")
    print(" n       product TV       sqrt(n) risk lower bound")
    for sample_size, product_tv, risk_bound in zip(
            w1_minimax["sample_sizes"], w1_minimax["product_tv"],
            w1_minimax["scaled_risk_lower_bounds"]):
        print(f"{sample_size:7d}   {product_tv:.12f}       {risk_bound:.12f}")
    print("optimized z and h",
          f"{w1_minimax['z_star']:.12f}", f"{w1_minimax['h_star']:.12f}")
    print("sqrt(n) W1 risk lower-bound limit",
          f"{w1_minimax['asymptotic_constant']:.12f}")
    print("point-mass Wasserstein-1 sample-mean certificate")
    print(" n       exact risk       sqrt(n) exact risk")
    for sample_size, risk, scaled_risk in zip(
            w1_point_mass["sample_sizes"], w1_point_mass["exact_risks"],
            w1_point_mass["scaled_exact_risks"]):
        print(f"{sample_size:7d}   {risk:.12f}       {scaled_risk:.12f}")
    print("sqrt(n) point-mass risk limit",
          f"{w1_point_mass['pointwise_limit']:.12f}",
          "uniform upper bound", f"{w1_point_mass['uniform_scaled_bound']:.12f}")
    print("unrestricted compact-mixture Wasserstein-1 lower certificate")
    print(" L       exact W1         count TV          coupling bound")
    for order, w1, tv, coupling in zip(
            w1_nonparametric["check_orders"], w1_nonparametric["exact_w1"],
            w1_nonparametric["count_tv"],
            w1_nonparametric["coupling_bounds"]):
        print(f"{order:2d}   {w1:.12f}   {tv:.3e}   {coupling:.3e}")
    print(" n       optimal L       risk lower bound   scaled lower bound")
    for label, order, risk, scaled in zip(
            w1_nonparametric["sample_labels"],
            w1_nonparametric["optimal_orders"],
            w1_nonparametric["risk_lower_bounds"],
            w1_nonparametric["scaled_lower_bounds"]):
        print(f"{label:>7s}   {order:5d}           {risk:.12e}   {scaled:.12f}")
    print("asymptotic log(n)/loglog(n) lower constant",
          f"{w1_nonparametric['asymptotic_scaled_lower_bound']:.12f}")
    print("factorial-moment W1 upper-bound certificate")
    print("maximum relative second-moment identity error",
          f"{w1_moment_upper['relative_second_moment_error']:.3e}")
    print(" log(n)   degree L   log stochastic factor")
    for log_n, degree, log_factor in zip(
            w1_moment_upper["log_sample_sizes"],
            w1_moment_upper["moment_degrees"],
            w1_moment_upper["log_stochastic_factors"]):
        print(f"{log_n:7.0f}   {degree:8d}   {log_factor:21.6f}")
    print("nonreversible factorial identity max error", f"{identity_error:.3e}")
    print("count-truncation tail bound", f"{tail_bound:.3e}")
    print("mixed factorial identity max error", f"{mixed_error:.3e}")
    print("bivariate count-truncation tail bound", f"{bivariate_tail_bound:.3e}")
    print("mixed factorial/intensity cumulants (11, 21, 12, 22)",
          " ".join(f"{x:.8f}" for x in mixed_intensity))
    print("common-shock formula max error", f"{common_error:.3e}")
    print("common-shock all-order formula max error through (3,3)",
          f"{common_all_order_error:.3e}")
    print("common-shock count-truncation tail bound", f"{common_tail_bound:.3e}")
    print("common-shock mixed factorial cumulants (11, 21, 12, 22)",
          " ".join(f"{x:.8f}" for x in common_factorial))
    print("common-shock (3,3) factorial cumulant",
          f"{common_factorial_grid[2, 2]:.8f}")
    print("marked common-shock formula max error through (3,3)",
          f"{marked_error:.3e}")
    print("marked common-shock count-truncation tail bound",
          f"{marked_tail_bound:.3e}")
    print("marked common-shock (3,3) factorial cumulant",
          f"{marked_factorial_grid[2, 2]:.8f}")
    print("state-dependent marked formula max error through (3,3)",
          f"{state_marked_error:.3e}")
    print("state-dependent marked count-truncation tail bound",
          f"{state_marked_tail_bound:.3e}")
    print("state-dependent marked (3,3) factorial cumulant",
          f"{state_marked_factorial_grid[2, 2]:.8f}")
    print("unit-mark reduction max error", f"{unit_reduction_error:.3e}")
    print("common-only mixed factorial cumulants",
          " ".join(f"{x:.8f}" for x in common_only_factorial))
    print("two-state closed-form max error", f"{max(closed_errors):.3e}")
    print("first-order overdispersion residual order", f"{residual_order:.6f}")
    print("lambda=10 ordinary count cumulants", " ".join(f"{x:.8f}" for x in count10))
    print("lambda=10 factorial/intensity cumulants", " ".join(f"{x:.8f}" for x in intensity10))

    assert identity_error < 2e-9
    assert tail_bound < 1e-45
    assert mixed_error < 2e-9
    assert bivariate_tail_bound < 1e-45
    assert common_error < 2e-9
    assert common_all_order_error < 2e-8
    assert common_tail_bound < 1e-45
    assert marked_error < 2e-7
    assert marked_tail_bound < 1e-35
    assert state_marked_error < 2e-7
    assert state_marked_tail_bound < 1e-35
    assert unit_reduction_error < 2e-10
    assert common_only_error < 2e-9
    assert max(closed_errors) < 2e-12
    assert abs(residual_order - 2.0) < 0.01
    assert abs(count10[1] - 5.808125) < 2e-10
    print("PASS")


if __name__ == "__main__":
    main()
