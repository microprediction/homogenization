"""Certificate for the sharp maturity-dependent likelihood coefficient cone."""

import math

import numpy as np


def integral(j, a):
    """Integral of exp(-j t) from zero to a, evaluated stably."""
    return -math.expm1(-j * a) / j


def rho_closed(a):
    x = math.exp(-a)
    return 8.0 * (1.0 + x + x * x) ** 2 / (
        9.0 * (1.0 + x) ** 2 * (1.0 + x * x)
    )


def main():
    ok = True
    grid = (1.0e-3, 1.0 / 24.0, 0.25, 1.0, 10.0)
    values = []

    print("sharp loading factor rho(a) = I_3^2/(I_2 I_4)")
    for a in grid:
        quadrature_ratio = integral(3, a) ** 2 / (integral(2, a) * integral(4, a))
        closed_ratio = rho_closed(a)
        residual = abs(quadrature_ratio - closed_ratio)
        values.append(closed_ratio)
        print(
            f"a={a:.9f} rho={closed_ratio:.12f} "
            f"closed/integral residual={residual:.3e}"
        )
        ok &= residual < 2.0e-14
        ok &= 8.0 / 9.0 < closed_ratio < 1.0

    ok &= all(values[j + 1] < values[j] for j in range(len(values) - 1))

    # A rank-one Gram matrix attains the bound.  These arbitrary positive
    # scales also check that the normalization of A2, A3, and A4 cancels.
    a = 0.7
    kuu, kww = 2.3, 0.4
    kusym = math.sqrt(kuu * kww)
    i2, i3, i4 = (integral(j, a) for j in (2, 3, 4))
    a2 = kuu * i2
    a3 = -kusym * i3
    a4 = 0.25 * kww * i4
    attained = a3 * a3 / (4.0 * a2 * a4)
    print(
        f"rank-one attainment at a={a:.1f}: ratio={attained:.12f}, "
        f"rho={rho_closed(a):.12f}"
    )
    ok &= abs(attained - rho_closed(a)) < 2.0e-14

    # Random positive-semidefinite Gram matrices must remain inside the cone.
    rng = np.random.default_rng(20261012)
    worst = 0.0
    for _ in range(1000):
        z = rng.normal(size=(2, 4))
        gram = z @ z.T
        kuu, kww, kusym = gram[0, 0], gram[1, 1], gram[0, 1]
        ratio = (kusym * i3) ** 2 / (4.0 * (kuu * i2) * (0.25 * kww * i4))
        worst = max(worst, ratio / rho_closed(a))
    print(f"largest random ratio divided by rho: {worst:.12f}")
    ok &= worst <= 1.0 + 2.0e-14

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
