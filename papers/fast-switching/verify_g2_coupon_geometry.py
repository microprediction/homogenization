"""Numerical certificate for the G2++ coupon-bond exercise geometry.

A single zero-coupon bond at option expiry depends on one Gaussian linear
combination.  A coupon bond generally does not: distinct cash-flow maturities
have non-collinear G2++ loading vectors.  This script certifies the determinant,
the strict convexity, curvature mass and wing asymptotics of the exercise
boundary, an exact conditional-Gaussian one-dimensional pricing formula, and
its reduction to the one-factor formula.  It also certifies a closed-form
tail bound for truncating the remaining Gaussian integral and the general
d-factor-to-(d-1)-factor conditioning identity with its boundary Hessian.
It also certifies an oblique conditioning direction in a case where no
original coordinate has cash-flow loadings of one sign.
"""

from itertools import product
from math import exp, expm1, log, pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr, roots_hermitenorm


def B(kappa, tau):
    """Ornstein--Uhlenbeck bond loading."""
    return -expm1(-kappa * tau) / kappa


def loading(a, b, tau):
    return B(a, tau), B(b, tau)


def determinant(u, v):
    return u[0] * v[1] - u[1] * v[0]


def coupon(x, z, loadings, weights=None):
    if weights is None:
        weights = [1.0] * len(loadings)
    return sum(w * exp(-p * x - q * z)
               for w, (p, q) in zip(weights, loadings))


def boundary_z(x, loadings, weights=None, strike=2.0):
    """Solve the unique z with coupon(x,z)=strike."""
    return brentq(
        lambda z: coupon(x, z, loadings, weights) - strike,
        -100.0,
        100.0,
        xtol=1e-14,
        rtol=1e-14,
    )


def boundary_derivatives(x, z, loadings, weights=None):
    """Implicit slope and curvature of the coupon=strike boundary."""
    if weights is None:
        weights = [1.0] * len(loadings)
    terms = [w * exp(-p * x - q * z)
             for w, (p, q) in zip(weights, loadings)]
    px = sum(term * p for term, (p, _) in zip(terms, loadings))
    qz = sum(term * q for term, (_, q) in zip(terms, loadings))
    slope = -px / qz
    curvature = sum(
        term * (p + q * slope) ** 2
        for term, (p, q) in zip(terms, loadings)
    ) / qz
    return slope, curvature


def curvature_variance_identity(x, z, loadings, weights=None):
    """Return the slope and the weighted-variance form of zeta''(x)."""
    if weights is None:
        weights = [1.0] * len(loadings)
    terms = [w * exp(-p * x - q * z)
             for w, (p, q) in zip(weights, loadings)]
    denominator = sum(term * q
                      for term, (_, q) in zip(terms, loadings))
    probabilities = [term * q / denominator
                     for term, (_, q) in zip(terms, loadings)]
    ratios = [p / q for p, q in loadings]
    ratio_mean = sum(probability * ratio
                     for probability, ratio in zip(probabilities, ratios))
    curvature = sum(
        probability * q * (ratio - ratio_mean) ** 2
        for probability, (_, q), ratio
        in zip(probabilities, loadings, ratios)
    )
    return -ratio_mean, curvature


def unique_wing_parameters(loadings, weights, strike):
    """Return the unique extreme-loading asymptotes and exponential rates.

    The right wing x -> +infinity is selected by the smallest p/q ratio; the
    left wing x -> -infinity is selected by the largest.  This certificate has
    distinct ratios.  The accompanying page states the grouped-extreme result
    when a minimum or maximum is attained more than once.
    """
    ratios = [p / q for p, q in loadings]
    right = min(range(len(ratios)), key=ratios.__getitem__)
    left = max(range(len(ratios)), key=ratios.__getitem__)
    assert sum(r == ratios[right] for r in ratios) == 1
    assert sum(r == ratios[left] for r in ratios) == 1

    right_intercept = log(weights[right] / strike) / loadings[right][1]
    left_intercept = log(weights[left] / strike) / loadings[left][1]
    right_rate = min(
        q * (r - ratios[right])
        for r, (_, q) in zip(ratios, loadings) if r > ratios[right]
    )
    left_rate = min(
        q * (ratios[left] - r)
        for r, (_, q) in zip(ratios, loadings) if r < ratios[left]
    )
    right_competitor_mass = sum(
        weight * exp(-q * right_intercept)
        for k, (weight, (_, q)) in enumerate(zip(weights, loadings))
        if k != right
    )
    left_competitor_mass = sum(
        weight * exp(-q * left_intercept)
        for k, (weight, (_, q)) in enumerate(zip(weights, loadings))
        if k != left
    )
    return {
        "right": (right, ratios[right], right_intercept, right_rate,
                  right_competitor_mass),
        "left": (left, ratios[left], left_intercept, left_rate,
                 left_competitor_mass),
    }


def wing_error_bound(q, competitor_mass, strike, rate, distance):
    """Explicit dominant-wing bound once its logarithm has positive input."""
    relative_competitor = competitor_mass / strike * exp(-rate * distance)
    assert 0 <= relative_competitor < 1
    return -log(1 - relative_competitor) / q


def conditional_gaussian_price(
        loadings, weights, strike, mean_x, mean_z, sigma_x, sigma_z, rho,
        cutoff=10.0):
    """Exact receiver price under a bivariate Gaussian expiry law.

    The discount-to-expiry prefactor is omitted.  It can be restored by
    multiplying the returned forward-measure expectation by P(0,T).
    """
    conditional_sigma = sigma_z * sqrt(1 - rho * rho)

    def conditional_value(standard_x):
        x = mean_x + sigma_x * standard_x
        mean = mean_z + rho * sigma_z * standard_x
        z_star = boundary_z(x, loadings, weights, strike)
        strike_probability = ndtr((z_star - mean) / conditional_sigma)
        value = -strike * strike_probability
        for weight, (p, q) in zip(weights, loadings):
            tilted_probability = ndtr(
                (z_star - mean + q * conditional_sigma ** 2)
                / conditional_sigma
            )
            value += weight * exp(
                -p * x - q * mean + 0.5 * q * q * conditional_sigma ** 2
            ) * tilted_probability
        return value * exp(-0.5 * standard_x ** 2) / sqrt(2 * pi)

    return quad(conditional_value, -cutoff, cutoff,
                epsabs=2e-13, epsrel=2e-13, limit=250)[0]


def receiver_truncation_bound(
        loadings, weights, mean_x, mean_z, sigma_x, sigma_z, rho, cutoff):
    """Bound the receiver-price mass outside a standardized X window.

    Since (C-K)^+ <= C, each omitted cash-flow term is an exponentially
    tilted Gaussian tail.  The returned bound is for the undiscounted
    forward-measure value; multiply it by P(0,T) for the time-zero price.
    """
    conditional_variance = sigma_z * sigma_z * (1 - rho * rho)
    bound = 0.0
    for weight, (p, q) in zip(weights, loadings):
        tilt = p * sigma_x + q * rho * sigma_z
        moment = weight * exp(
            -p * mean_x - q * mean_z
            + 0.5 * q * q * conditional_variance
            + 0.5 * tilt * tilt
        )
        bound += moment * (
            ndtr(tilt - cutoff) + ndtr(-tilt - cutoff)
        )
    return bound


def multifactor_coupon(y, z, loadings, weights):
    """Coupon value with the last Gaussian coordinate singled out."""
    return sum(
        weight * exp(-float(np.dot(p, y)) - q * z)
        for weight, (p, q) in zip(weights, loadings)
    )


def multifactor_boundary(y, loadings, weights, strike):
    """Unique last-coordinate root when every singled-out loading is positive."""
    assert all(q > 0 for _, q in loadings)
    return brentq(
        lambda z: multifactor_coupon(y, z, loadings, weights) - strike,
        -100.0,
        100.0,
        xtol=1e-14,
        rtol=1e-14,
    )


def multifactor_boundary_derivatives(y, z, loadings, weights):
    """Gradient and Hessian of the codimension-one coupon boundary."""
    terms = np.array([
        weight * exp(-float(np.dot(p, y)) - q * z)
        for weight, (p, q) in zip(weights, loadings)
    ])
    denominator = sum(
        term * q for term, (_, q) in zip(terms, loadings)
    )
    gradient = -sum(
        (term * p for term, (p, _) in zip(terms, loadings)),
        start=np.zeros_like(y),
    ) / denominator
    hessian = sum(
        (term * np.outer(p + q * gradient, p + q * gradient)
         for term, (p, q) in zip(terms, loadings)),
        start=np.zeros((len(y), len(y))),
    ) / denominator
    return gradient, hessian


def rotate_conditioning_problem(loadings, mean, covariance, direction):
    """Rotate a Gaussian problem so ``direction`` is the last coordinate."""
    loadings = np.asarray(loadings, dtype=float)
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    _, _, right_vectors = np.linalg.svd(direction.reshape(1, -1))
    complement = right_vectors[1:].T
    frame = np.column_stack((complement, direction))
    assert np.max(np.abs(frame.T @ frame - np.eye(len(direction)))) < 1e-14
    rotated_loadings = [
        (complement.T @ loading, float(direction @ loading))
        for loading in loadings
    ]
    assert all(q > 0 for _, q in rotated_loadings)
    return (
        rotated_loadings,
        frame.T @ mean,
        frame.T @ covariance @ frame,
    )


def multifactor_conditional_price(
        loadings, weights, strike, mean, covariance, order=32,
        numerical_inner=False):
    """Gaussian-Hermite certificate for the general dimension reduction.

    The final factor is integrated analytically unless ``numerical_inner`` is
    true, in which case an independent adaptive integral checks the same
    conditional expectation.  The remaining d-1 Gaussian coordinates use a
    tensor Gauss-Hermite rule; this is a certificate, not a general-purpose
    high-dimensional integration algorithm.
    """
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    covariance_y = covariance[:-1, :-1]
    covariance_yz = covariance[:-1, -1]
    regression = np.linalg.solve(covariance_y, covariance_yz)
    conditional_variance = (
        covariance[-1, -1] - covariance_yz @ regression
    )
    assert conditional_variance > 0
    conditional_sigma = sqrt(conditional_variance)
    cholesky_y = np.linalg.cholesky(covariance_y)
    nodes, quadrature_weights = roots_hermitenorm(order)
    quadrature_weights = quadrature_weights / sqrt(2 * pi)

    value = 0.0
    for indices in product(range(order), repeat=len(mean) - 1):
        standard_y = np.array([nodes[index] for index in indices])
        weight_y = float(np.prod([
            quadrature_weights[index] for index in indices
        ]))
        y = mean[:-1] + cholesky_y @ standard_y
        conditional_mean = mean[-1] + regression @ (y - mean[:-1])
        z_star = multifactor_boundary(y, loadings, weights, strike)
        upper = (z_star - conditional_mean) / conditional_sigma

        if numerical_inner:
            def integrand(standard_z):
                z = conditional_mean + conditional_sigma * standard_z
                payoff = multifactor_coupon(y, z, loadings, weights) - strike
                return payoff * exp(-0.5 * standard_z ** 2) / sqrt(2 * pi)

            conditional = quad(
                integrand, -12.0, upper,
                epsabs=3e-13, epsrel=3e-13, limit=200,
            )[0]
        else:
            conditional = -strike * ndtr(upper)
            for cash_weight, (p, q) in zip(weights, loadings):
                tilted_probability = ndtr(
                    upper + q * conditional_sigma
                )
                conditional += cash_weight * exp(
                    -float(np.dot(p, y)) - q * conditional_mean
                    + 0.5 * q * q * conditional_variance
                ) * tilted_probability
        value += weight_y * conditional
    return value


def nested_quadrature_price(
        loadings, weights, strike, mean_x, mean_z, sigma_x, sigma_z, rho):
    """Independent two-dimensional numerical expectation for the certificate."""
    conditional_sigma = sigma_z * sqrt(1 - rho * rho)

    def outer(standard_x):
        x = mean_x + sigma_x * standard_x
        mean = mean_z + rho * sigma_z * standard_x
        z_star = boundary_z(x, loadings, weights, strike)
        upper = (z_star - mean) / conditional_sigma

        def inner(standard_z):
            z = mean + conditional_sigma * standard_z
            payoff = coupon(x, z, loadings, weights) - strike
            return payoff * exp(-0.5 * standard_z ** 2) / sqrt(2 * pi)

        conditional = quad(inner, -12.0, upper,
                           epsabs=2e-12, epsrel=2e-12, limit=250)[0]
        return conditional * exp(-0.5 * standard_x ** 2) / sqrt(2 * pi)

    return quad(outer, -9.0, 9.0,
                epsabs=2e-11, epsrel=2e-11, limit=250)[0]


def one_factor_price(loadings, weights, strike, mean, variance):
    """Jamshidian-limit price when every loading is (p_k,p_k)."""
    scalar_loadings = [p for p, q in loadings if abs(p - q) < 1e-14]
    assert len(scalar_loadings) == len(loadings)
    root = brentq(
        lambda y: sum(w * exp(-p * y)
                      for w, p in zip(weights, scalar_loadings)) - strike,
        -100.0,
        100.0,
    )
    standard_deviation = sqrt(variance)
    value = -strike * ndtr((root - mean) / standard_deviation)
    for weight, p in zip(weights, scalar_loadings):
        value += weight * exp(-p * mean + 0.5 * p * p * variance) * ndtr(
            (root - mean + p * variance) / standard_deviation
        )
    return value


def log_ratio_derivative(a, b, tau):
    return a / (exp(a * tau) - 1) - b / (exp(b * tau) - 1)


def main():
    a, b = 0.5, 0.1
    tau1, tau2 = 1.0, 4.0
    loadings = [loading(a, b, tau1), loading(a, b, tau2)]
    det = determinant(*loadings)

    # R(tau)=B_a(tau)/B_b(tau) is strictly decreasing here (a>b), so
    # distinct maturities have distinct loading directions.
    for tau in (0.1, 0.5, 1.0, 2.0, 4.0, 10.0):
        assert log_ratio_derivative(a, b, tau) < 0
    assert abs(det - 0.9487045402383338) < 1e-14

    qsum = loadings[0][1] + loadings[1][1]
    curvature = 2 * det * det / qsum**3
    assert abs(curvature - 0.023475128385069474) < 1e-14
    slope, implicit_curvature = boundary_derivatives(0.0, 0.0, loadings)
    assert slope < 0
    assert abs(implicit_curvature - curvature) < 1e-14
    variance_slope, variance_curvature = curvature_variance_identity(
        0.0, 0.0, loadings
    )
    assert abs(variance_slope - slope) < 1e-14
    assert abs(variance_curvature - curvature) < 1e-14

    xs = (-0.1, 0.0, 0.1)
    zs = tuple(boundary_z(x, loadings) for x in xs)
    slopes = ((zs[1] - zs[0]) / 0.1, (zs[2] - zs[1]) / 0.1)
    assert abs(zs[1]) < 1e-14
    assert slopes[1] - slopes[0] > 0.002

    # If the two reversion speeds coincide, every loading is collinear and the
    # determinant (hence the curvature) vanishes: the one-factor reduction.
    one_factor_loadings = [loading(a, a, tau1), loading(a, a, tau2)]
    assert abs(determinant(*one_factor_loadings)) < 1e-15

    weights = [0.65, 1.35]
    strike = sum(weights)

    # The exact boundary wings are selected by the extreme ratios r_k=p_k/q_k.
    # For a unique extreme i, zeta(x)=-r_i*x+c_i+O(exp(-delta*|x|)), where
    # c_i=log(w_i/K)/q_i and delta is the smallest competing exponent gap.
    wings = unique_wing_parameters(loadings, weights, strike)
    (right_i, right_ratio, right_intercept, right_rate,
     right_competitor_mass) = wings["right"]
    (left_i, left_ratio, left_intercept, left_rate,
     left_competitor_mass) = wings["left"]
    assert right_i == 1 and left_i == 0
    assert abs(right_rate - 0.28776530960717517) < 1e-14
    assert abs(left_rate - 0.9969302226339624) < 1e-14

    right_xs = (20.0, 30.0, 40.0)
    right_errors = tuple(
        boundary_z(x, loadings, weights, strike)
        + right_ratio * x - right_intercept
        for x in right_xs
    )
    left_xs = (-10.0, -15.0, -20.0)
    left_errors = tuple(
        boundary_z(x, loadings, weights, strike)
        + left_ratio * x - left_intercept
        for x in left_xs
    )
    right_bounds = tuple(
        wing_error_bound(loadings[right_i][1], right_competitor_mass,
                         strike, right_rate, x)
        for x in right_xs
    )
    left_bounds = tuple(
        wing_error_bound(loadings[left_i][1], left_competitor_mass,
                         strike, left_rate, abs(x))
        for x in left_xs
    )
    assert all(a > b > 0 for a, b in zip(right_errors, right_errors[1:]))
    assert all(a > b > 0 for a, b in zip(left_errors, left_errors[1:]))
    assert all(error <= bound + 2e-14
               for error, bound in zip(right_errors, right_bounds))
    assert all(error <= bound + 2e-14
               for error, bound in zip(left_errors, left_bounds))
    observed_right_rate = -log(right_errors[2] / right_errors[1]) / 10.0
    observed_left_rate = -log(left_errors[2] / left_errors[1]) / 5.0
    assert abs(observed_right_rate - right_rate) < 5e-6
    assert abs(observed_left_rate - left_rate) < 2e-5

    # The exact wings turn strict convexity into a global invariant.  Since
    # zeta' tends to -r_max on the left and -r_min on the right,
    # integral_R zeta'' = r_max-r_min.  The finite interval below captures the
    # total mass to machine precision and independently checks the fundamental
    # theorem identity against its endpoint slopes.
    curvature_left, curvature_right = -50.0, 100.0

    def curvature_integrand(x):
        z = boundary_z(x, loadings, weights, strike)
        return boundary_derivatives(x, z, loadings, weights)[1]

    curvature_mass, curvature_quadrature_error = quad(
        curvature_integrand,
        curvature_left,
        curvature_right,
        epsabs=1e-13,
        epsrel=1e-13,
        limit=300,
    )
    left_slope = boundary_derivatives(
        curvature_left,
        boundary_z(curvature_left, loadings, weights, strike),
        loadings,
        weights,
    )[0]
    right_slope = boundary_derivatives(
        curvature_right,
        boundary_z(curvature_right, loadings, weights, strike),
        loadings,
        weights,
    )[0]
    ratio_spread = left_ratio - right_ratio
    assert abs(curvature_mass - (right_slope - left_slope)) < 1e-13
    assert abs(curvature_mass - ratio_spread) < 2e-14

    mean_x, mean_z = 0.01, -0.015
    sigma_x, sigma_z, rho = 0.25, 0.20, -0.35
    conditional_price = conditional_gaussian_price(
        loadings, weights, strike,
        mean_x, mean_z, sigma_x, sigma_z, rho,
    )
    truncated_cutoff = 6.0
    truncated_price = conditional_gaussian_price(
        loadings, weights, strike,
        mean_x, mean_z, sigma_x, sigma_z, rho,
        cutoff=truncated_cutoff,
    )
    truncation_bound = receiver_truncation_bound(
        loadings, weights,
        mean_x, mean_z, sigma_x, sigma_z, rho,
        truncated_cutoff,
    )
    truncation_error = conditional_price - truncated_price
    assert -2e-15 <= truncation_error <= truncation_bound
    numerical_price = nested_quadrature_price(
        loadings, weights, strike,
        mean_x, mean_z, sigma_x, sigma_z, rho,
    )
    assert abs(conditional_price - numerical_price) < 2e-12

    conditional_one_factor = conditional_gaussian_price(
        one_factor_loadings, weights, strike,
        mean_x, mean_z, sigma_x, sigma_z, rho,
    )
    scalar_one_factor = one_factor_price(
        one_factor_loadings,
        weights,
        strike,
        mean_x + mean_z,
        sigma_x ** 2 + sigma_z ** 2 + 2 * rho * sigma_x * sigma_z,
    )
    assert abs(conditional_one_factor - scalar_one_factor) < 2e-12

    # The same monotone conditioning argument reduces a d-factor Gaussian
    # coupon problem to d-1 dimensions.  Here three distinct maturities under
    # three mean-reversion speeds give a genuinely curved two-dimensional
    # boundary.  Its Hessian is positive definite because the three loading
    # ratios p_k/q_k affinely span R^2.
    speeds = (0.50, 0.20, 0.08)
    maturities = (1.0, 3.0, 6.0)
    three_factor_loadings = [
        (np.array([B(speeds[0], tau), B(speeds[1], tau)]),
         B(speeds[2], tau))
        for tau in maturities
    ]
    loading_ratios = [p / q for p, q in three_factor_loadings]
    affine_ratio_determinant = np.linalg.det(np.vstack((
        loading_ratios[1] - loading_ratios[0],
        loading_ratios[2] - loading_ratios[0],
    )))
    assert abs(affine_ratio_determinant) > 1e-3
    three_factor_weights = [0.35, 0.70, 1.20]
    three_factor_strike = sum(three_factor_weights)
    boundary_point = np.array([0.03, -0.02])
    boundary_height = multifactor_boundary(
        boundary_point, three_factor_loadings,
        three_factor_weights, three_factor_strike,
    )
    _, boundary_hessian = multifactor_boundary_derivatives(
        boundary_point, boundary_height, three_factor_loadings,
        three_factor_weights,
    )
    hessian_eigenvalues = np.linalg.eigvalsh(boundary_hessian)
    assert hessian_eigenvalues[0] > 0
    finite_difference_step = 1e-3
    finite_difference_hessian = np.zeros((2, 2))
    for row in range(2):
        direction = np.zeros(2)
        direction[row] = finite_difference_step
        finite_difference_hessian[row, row] = (
            multifactor_boundary(
                boundary_point + direction, three_factor_loadings,
                three_factor_weights, three_factor_strike,
            )
            - 2 * boundary_height
            + multifactor_boundary(
                boundary_point - direction, three_factor_loadings,
                three_factor_weights, three_factor_strike,
            )
        ) / finite_difference_step ** 2
    first = np.array([finite_difference_step, finite_difference_step])
    second = np.array([finite_difference_step, -finite_difference_step])
    finite_difference_hessian[0, 1] = finite_difference_hessian[1, 0] = (
        multifactor_boundary(
            boundary_point + first, three_factor_loadings,
            three_factor_weights, three_factor_strike,
        )
        - multifactor_boundary(
            boundary_point + second, three_factor_loadings,
            three_factor_weights, three_factor_strike,
        )
        - multifactor_boundary(
            boundary_point - second, three_factor_loadings,
            three_factor_weights, three_factor_strike,
        )
        + multifactor_boundary(
            boundary_point - first, three_factor_loadings,
            three_factor_weights, three_factor_strike,
        )
    ) / (4 * finite_difference_step ** 2)
    hessian_discrepancy = np.max(np.abs(
        boundary_hessian - finite_difference_hessian
    ))
    assert hessian_discrepancy < 1e-8

    three_factor_mean = np.array([0.01, -0.02, 0.015])
    three_factor_covariance = np.array([
        [0.040, 0.006, -0.004],
        [0.006, 0.025, 0.003],
        [-0.004, 0.003, 0.016],
    ])
    assert np.linalg.eigvalsh(three_factor_covariance)[0] > 0
    reduced_price_28 = multifactor_conditional_price(
        three_factor_loadings, three_factor_weights, three_factor_strike,
        three_factor_mean, three_factor_covariance, order=28,
    )
    reduced_price_36 = multifactor_conditional_price(
        three_factor_loadings, three_factor_weights, three_factor_strike,
        three_factor_mean, three_factor_covariance, order=36,
    )
    independent_inner_price = multifactor_conditional_price(
        three_factor_loadings, three_factor_weights, three_factor_strike,
        three_factor_mean, three_factor_covariance, order=36,
        numerical_inner=True,
    )
    assert abs(reduced_price_36 - reduced_price_28) < 2e-10
    assert abs(reduced_price_36 - independent_inner_price) < 2e-11

    # No original coordinate works in this example: every column contains
    # both positive and negative loadings.  But the loading hull lies in the
    # plane b.u=1.2/sqrt(3), so u=(1,1,1)/sqrt(3) strictly separates it from
    # the origin.  The barycenter is the closest hull point, proving the
    # separation margin (and hull distance) exactly for this symmetric case.
    oblique_loadings = np.array([
        [2.0, -0.4, -0.4],
        [-0.4, 2.0, -0.4],
        [-0.4, -0.4, 2.0],
    ])
    coordinate_minima = np.min(oblique_loadings, axis=0)
    coordinate_maxima = np.max(oblique_loadings, axis=0)
    assert np.all(coordinate_minima < 0)
    assert np.all(coordinate_maxima > 0)
    first_direction = np.ones(3) / sqrt(3)
    first_margins = oblique_loadings @ first_direction
    separation_margin = float(np.min(first_margins))
    hull_nearest_point = np.mean(oblique_loadings, axis=0)
    hull_distance = float(np.linalg.norm(hull_nearest_point))
    assert np.max(first_margins) - separation_margin < 1e-14
    assert np.linalg.norm(
        hull_nearest_point - separation_margin * first_direction
    ) < 1e-14
    assert abs(hull_distance - separation_margin) < 1e-14
    second_direction = np.array([1.2, 1.0, 1.0])
    second_direction /= np.linalg.norm(second_direction)
    second_margin = float(np.min(oblique_loadings @ second_direction))
    assert second_margin > 0

    oblique_weights = [0.7, 0.9, 1.1]
    oblique_strike = sum(oblique_weights)

    def oblique_price(direction, order, numerical_inner=False):
        rotated = rotate_conditioning_problem(
            oblique_loadings, three_factor_mean,
            three_factor_covariance, direction,
        )
        return multifactor_conditional_price(
            rotated[0], oblique_weights, oblique_strike,
            rotated[1], rotated[2], order=order,
            numerical_inner=numerical_inner,
        )

    oblique_prices_36 = [
        oblique_price(direction, 36)
        for direction in (first_direction, second_direction)
    ]
    oblique_prices_44 = [
        oblique_price(direction, 44)
        for direction in (first_direction, second_direction)
    ]
    oblique_inner_28 = [
        oblique_price(direction, 28, numerical_inner=True)
        for direction in (first_direction, second_direction)
    ]
    oblique_analytic_28 = [
        oblique_price(direction, 28)
        for direction in (first_direction, second_direction)
    ]
    oblique_direction_discrepancy = abs(
        oblique_prices_44[0] - oblique_prices_44[1]
    )
    oblique_inner_discrepancy = max(
        abs(analytic - numerical)
        for analytic, numerical in zip(oblique_analytic_28, oblique_inner_28)
    )
    assert max(
        abs(coarse - fine)
        for coarse, fine in zip(oblique_prices_36, oblique_prices_44)
    ) < 1e-12
    assert oblique_direction_discrepancy < 1e-12
    assert oblique_inner_discrepancy < 2e-12

    print("loadings:", loadings)
    print("determinant:", f"{det:.12f}")
    print("boundary z(-0.1), z(0), z(0.1):", tuple(f"{z:.12f}" for z in zs))
    print("secant slopes:", tuple(f"{s:.12f}" for s in slopes))
    print("analytic curvature z''(0):", f"{curvature:.12f}")
    print("right-wing maturity index, ratio, intercept:",
          right_i, f"{right_ratio:.12f}", f"{right_intercept:.12f}")
    print("right-wing predicted/observed exponential rate:",
          f"{right_rate:.12f}", f"{observed_right_rate:.12f}")
    print("right-wing error/bound at x=20:",
          f"{right_errors[0]:.12e}", f"{right_bounds[0]:.12e}")
    print("left-wing maturity index, ratio, intercept:",
          left_i, f"{left_ratio:.12f}", f"{left_intercept:.12f}")
    print("left-wing predicted/observed exponential rate:",
          f"{left_rate:.12f}", f"{observed_left_rate:.12f}")
    print("left-wing error/bound at x=-10:",
          f"{left_errors[0]:.12e}", f"{left_bounds[0]:.12e}")
    print("curvature mass / extreme-ratio spread / quadrature error:",
          f"{curvature_mass:.12f}", f"{ratio_spread:.12f}",
          f"{curvature_quadrature_error:.3e}")
    print("conditional one-dimensional price:", f"{conditional_price:.12f}")
    print("independent nested-quadrature price:", f"{numerical_price:.12f}")
    print("pricing discrepancy:", f"{abs(conditional_price - numerical_price):.3e}")
    print("cutoff-6 truncation error / rigorous bound:",
          f"{truncation_error:.12e}", f"{truncation_bound:.12e}")
    print("one-factor reduction discrepancy:",
          f"{abs(conditional_one_factor - scalar_one_factor):.3e}")
    print("three-factor boundary Hessian eigenvalues:",
          tuple(f"{value:.12e}" for value in hessian_eigenvalues))
    print("three-factor loading-ratio affine determinant:",
          f"{affine_ratio_determinant:.12e}")
    print("three-factor Hessian finite-difference discrepancy:",
          f"{hessian_discrepancy:.3e}")
    print("three-factor reduced prices (orders 28, 36):",
          f"{reduced_price_28:.12f}", f"{reduced_price_36:.12f}")
    print("three-factor analytic/numerical-inner discrepancy:",
          f"{abs(reduced_price_36 - independent_inner_price):.3e}")
    print("oblique coordinate loading ranges:", tuple(
        (f"{minimum:.1f}", f"{maximum:.1f}")
        for minimum, maximum in zip(coordinate_minima, coordinate_maxima)
    ))
    print("oblique separation margin / hull distance / second margin:",
          f"{separation_margin:.12f}", f"{hull_distance:.12f}",
          f"{second_margin:.12f}")
    print("oblique prices along two directions:",
          *(f"{price:.15f}" for price in oblique_prices_44))
    print("oblique direction / numerical-inner discrepancies:",
          f"{oblique_direction_discrepancy:.3e}",
          f"{oblique_inner_discrepancy:.3e}")
    print("PASS: curved Gaussian boundaries and exact dimension reduction")


if __name__ == "__main__":
    main()
