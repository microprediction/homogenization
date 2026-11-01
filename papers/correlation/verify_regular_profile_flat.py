#!/usr/bin/env python3
"""Certificate for regularly varying flat-profile covariance."""

import math

from scipy.optimize import brentq, minimize_scalar

from verify_gauge_flat import chain_data, contracted_gauge_rate


GAUGE_CONSTANT = 4.60437900568596


def profile(value):
    """A non-power regularly varying profile of index one."""
    return value * math.log1p(value)


def inverse_profile(level):
    """Invert x log(1+x) at a positive level."""
    return float(brentq(
        lambda value: profile(value) - level,
        1e-12,
        max(2.0, level),
        xtol=1e-13,
    ))


def saddle(log_speed, power, scores, green_kubo):
    """Minimize the exact Legendre/profile exponent per unit speed."""
    def objective(profile_level):
        radius = 1.0 / inverse_profile(profile_level)
        value, _ = contracted_gauge_rate(scores, green_kubo, radius)
        return power * math.exp(profile_level - log_speed) + value

    result = minimize_scalar(
        objective,
        bounds=(10.0, log_speed),
        method="bounded",
        options={"xatol": 1e-7},
    )
    return float(result.x), float(result.fun)


def main():
    stationary, scores, green_kubo = chain_data()
    rows = []
    for log_speed in (128.0, 512.0, 2048.0, 8192.0):
        inverse = inverse_profile(log_speed)
        radius = 1.0 / inverse
        boundary_rate, angle = contracted_gauge_rate(
            scores, green_kubo, radius
        )
        quadratic = radius ** 2 / (2.0 * GAUGE_CONSTANT)
        boundary_ratio = boundary_rate / quadratic

        _, second = saddle(log_speed, 2.0, scores, green_kubo)
        _, first = saddle(log_speed, 1.0, scores, green_kubo)
        leading = 1.0 / (2.0 * GAUGE_CONSTANT * inverse ** 2)
        variance_rate = second / leading
        mean_separation = (2.0 * first - second) / leading
        rows.append((
            log_speed,
            inverse,
            boundary_ratio,
            angle,
            variance_rate,
            mean_separation,
        ))

    regular_variation_checks = []
    for scale in (128.0, 512.0, 2048.0, 8192.0):
        ratio = profile(2.0 * scale) / profile(scale)
        regular_variation_checks.append(ratio)

    print("Regularly varying flat-profile covariance certificate")
    print(f"stationary law {stationary}")
    print("profile H(x) = x log(1+x), regular-variation index 1")
    print(f"tangent gauge constant {GAUGE_CONSTANT:.9f}")
    for scale, ratio in zip(
            (128.0, 512.0, 2048.0, 8192.0),
            regular_variation_checks):
        print(f"H(2x)/H(x) at x={scale:7.0f}: {ratio:.9f}")
    for row in rows:
        log_speed, inverse, boundary_ratio, angle, variance, separation = row
        print(
            f"log speed {log_speed:7.0f}: inverse {inverse:.9f}, "
            f"boundary {boundary_ratio:.9f}, angle {angle:.9f}, "
            f"variance {variance:.9f}, separation {separation:.9f}"
        )

    ok = True
    ok &= all(ratio > regular_variation_checks[i + 1] > 2.0
              for i, ratio in enumerate(regular_variation_checks[:-1]))
    ok &= regular_variation_checks[-1] < 2.16
    ok &= abs(rows[-1][2] - 1.0) < 3e-5
    ok &= all(rows[i][4] > rows[i + 1][4] > 1.0
              for i in range(len(rows) - 1))
    ok &= all(rows[i][5] > rows[i + 1][5] > 1.0
              for i in range(len(rows) - 1))
    ok &= rows[-1][4] < 1.006
    ok &= rows[-1][5] < 1.006
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
