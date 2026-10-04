"""Numerical certificate for the G2++ coupon-bond exercise geometry.

A single zero-coupon bond at option expiry depends on one Gaussian linear
combination.  A coupon bond generally does not: distinct cash-flow maturities
have non-collinear G2++ loading vectors.  This script certifies the determinant,
the curvature of a two-cash-flow exercise boundary, and the one-factor limit.
"""

from math import exp, expm1


def B(kappa, tau):
    """Ornstein--Uhlenbeck bond loading."""
    return -expm1(-kappa * tau) / kappa


def loading(a, b, tau):
    return B(a, tau), B(b, tau)


def determinant(u, v):
    return u[0] * v[1] - u[1] * v[0]


def coupon(x, z, loadings):
    return sum(exp(-p * x - q * z) for p, q in loadings)


def boundary_z(x, loadings, strike=2.0):
    """Solve the unique z with coupon(x,z)=strike by bisection."""
    lo, hi = -10.0, 10.0
    assert coupon(x, lo, loadings) > strike > coupon(x, hi, loadings)
    for _ in range(100):
        mid = (lo + hi) / 2
        if coupon(x, mid, loadings) > strike:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


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

    xs = (-0.1, 0.0, 0.1)
    zs = tuple(boundary_z(x, loadings) for x in xs)
    slopes = ((zs[1] - zs[0]) / 0.1, (zs[2] - zs[1]) / 0.1)
    assert abs(zs[1]) < 1e-14
    assert slopes[1] - slopes[0] > 0.002

    # If the two reversion speeds coincide, every loading is collinear and the
    # determinant (hence the curvature) vanishes: the one-factor reduction.
    one_factor_loadings = [loading(a, a, tau1), loading(a, a, tau2)]
    assert abs(determinant(*one_factor_loadings)) < 1e-15

    print("loadings:", loadings)
    print("determinant:", f"{det:.12f}")
    print("boundary z(-0.1), z(0), z(0.1):", tuple(f"{z:.12f}" for z in zs))
    print("secant slopes:", tuple(f"{s:.12f}" for s in slopes))
    print("analytic curvature z''(0):", f"{curvature:.12f}")
    print("PASS: distinct G2++ cash-flow loadings give a curved exercise boundary")


if __name__ == "__main__":
    main()
