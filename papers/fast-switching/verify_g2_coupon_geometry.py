"""Numerical certificate for the G2++ coupon-bond exercise geometry.

A single zero-coupon bond at option expiry depends on one Gaussian linear
combination.  A coupon bond generally does not: distinct cash-flow maturities
have non-collinear G2++ loading vectors.  This script certifies the determinant,
the strict convexity of the exercise boundary, an exact conditional-Gaussian
one-dimensional pricing formula, and its reduction to the one-factor formula.
"""

from math import exp, expm1, pi, sqrt

from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr


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


def conditional_gaussian_price(
        loadings, weights, strike, mean_x, mean_z, sigma_x, sigma_z, rho):
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

    return quad(conditional_value, -10.0, 10.0,
                epsabs=2e-13, epsrel=2e-13, limit=250)[0]


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
    mean_x, mean_z = 0.01, -0.015
    sigma_x, sigma_z, rho = 0.25, 0.20, -0.35
    conditional_price = conditional_gaussian_price(
        loadings, weights, strike,
        mean_x, mean_z, sigma_x, sigma_z, rho,
    )
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

    print("loadings:", loadings)
    print("determinant:", f"{det:.12f}")
    print("boundary z(-0.1), z(0), z(0.1):", tuple(f"{z:.12f}" for z in zs))
    print("secant slopes:", tuple(f"{s:.12f}" for s in slopes))
    print("analytic curvature z''(0):", f"{curvature:.12f}")
    print("conditional one-dimensional price:", f"{conditional_price:.12f}")
    print("independent nested-quadrature price:", f"{numerical_price:.12f}")
    print("pricing discrepancy:", f"{abs(conditional_price - numerical_price):.3e}")
    print("one-factor reduction discrepancy:",
          f"{abs(conditional_one_factor - scalar_one_factor):.3e}")
    print("PASS: curved G2++ boundary and exact conditional-Gaussian price")


if __name__ == "__main__":
    main()
