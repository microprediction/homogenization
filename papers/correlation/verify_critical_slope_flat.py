#!/usr/bin/env python3
"""Certificate for the finite-slope boundary of slow-profile covariance."""

import math

from verify_flat_map import continuous_integral


def gaussian_absolute_moment(power):
    """Return E|Z|^power for a standard Gaussian Z."""
    return (
        2.0 ** (0.5 * power)
        * math.gamma(0.5 * (power + 1.0))
        / math.sqrt(math.pi)
    )


def telegraph_absolute_moment(speed, power):
    """Exact absolute moment of the symmetric telegraph average."""
    continuous = continuous_integral(
        speed,
        lambda position: position ** power,
        absolute_tolerance=1e-14,
    )
    return continuous + math.exp(-speed)


def normalized_rows(slope):
    """Compare exact telegraph moments with the Gaussian boundary law."""
    first_limit = gaussian_absolute_moment(slope)
    second_limit = gaussian_absolute_moment(2.0 * slope)
    variance_limit = second_limit - first_limit ** 2
    rows = []
    for speed in (16.0, 64.0, 256.0, 1024.0, 4096.0):
        first = telegraph_absolute_moment(speed, slope)
        second = telegraph_absolute_moment(speed, 2.0 * slope)
        variance = second - first ** 2
        rows.append((
            speed,
            speed ** (0.5 * slope) * first / first_limit,
            speed ** slope * second / second_limit,
            speed ** slope * variance / variance_limit,
        ))
    return first_limit, second_limit, variance_limit, rows


def main():
    slope = 1.0
    first_limit, second_limit, variance_limit, rows = normalized_rows(slope)
    exact_log_constant = -math.log(variance_limit)
    naive_log_constant = 1.0 - math.log(2.0)

    _, _, quadratic_variance, quadratic_rows = normalized_rows(2.0)

    print("Finite-slope flat-profile covariance certificate")
    print(f"critical slope {slope:.1f}")
    print(f"Gaussian first moment {first_limit:.9f}")
    print(f"Gaussian second moment {second_limit:.9f}")
    print(f"Gaussian variance constant {variance_limit:.9f}")
    print(f"exact additive log constant {exact_log_constant:.9f}")
    print(f"naive slow-profile additive constant {naive_log_constant:.9f}")
    print(f"additive constant gap {exact_log_constant - naive_log_constant:.9f}")
    for speed, first, second, variance in rows:
        print(
            f"speed {speed:4.0f}: first moment {first:.9f}, "
            f"second moment {second:.9f}, variance {variance:.9f}"
        )
    print(f"quadratic-slope Gaussian variance {quadratic_variance:.9f}")
    print(
        "quadratic-slope final normalized variance "
        f"{quadratic_rows[-1][3]:.9f}"
    )

    ok = True
    ok &= abs(first_limit - math.sqrt(2.0 / math.pi)) < 5e-14
    ok &= abs(second_limit - 1.0) < 5e-14
    ok &= abs(variance_limit - (1.0 - 2.0 / math.pi)) < 5e-14
    ok &= all(
        0.9 < rows[index][3] < rows[index + 1][3] < 1.0
        for index in range(len(rows) - 1)
    )
    ok &= rows[-1][1] > 0.9999
    ok &= rows[-1][2] > 0.9998
    ok &= rows[-1][3] > 0.9997
    ok &= abs(exact_log_constant - naive_log_constant) > 0.7
    ok &= abs(quadratic_variance - 2.0) < 5e-14
    ok &= quadratic_rows[-1][3] > 0.9993
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
