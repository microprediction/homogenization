#!/usr/bin/env python3
"""Certificate for the vanishing-slope logarithmic covariance phase."""

import math

from scipy.integrate import quad
from scipy.special import digamma, polygamma

from verify_flat_map import continuous_integral


PROFILE_SCALES = (32.0, 128.0, 512.0, 2048.0, 8192.0)
CHAIN_SPEEDS = (64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0)


def log_gaussian_density(log_radius):
    """Density of log|Z| for a standard Gaussian Z."""
    return math.sqrt(2.0 / math.pi) * math.exp(
        log_radius - 0.5 * math.exp(2.0 * log_radius)
    )


def normalized_profile(logarithmic_scale):
    """Moments of the normalized square-root profile under its Gaussian law."""
    derivative = 0.5 / math.sqrt(logarithmic_scale)

    def normalized_increment(log_radius):
        increment = (
            math.sqrt(logarithmic_scale)
            - math.sqrt(max(0.0, logarithmic_scale - log_radius))
        )
        return math.expm1(increment) / derivative

    mean = quad(
        lambda value: normalized_increment(value) * log_gaussian_density(value),
        -60.0,
        8.0,
        epsabs=2e-12,
        epsrel=2e-12,
        limit=300,
    )[0]
    second = quad(
        lambda value: normalized_increment(value) ** 2
        * log_gaussian_density(value),
        -60.0,
        8.0,
        epsabs=2e-12,
        epsrel=2e-12,
        limit=300,
    )[0]
    return mean, second - mean ** 2


def exact_telegraph_variance(speed):
    """Variance of the rescaled square-root profile under the exact law."""
    logarithmic_scale = 0.5 * math.log(speed)
    normalization = math.exp(math.sqrt(logarithmic_scale))

    def transformed(position):
        if position == 0.0:
            return 0.0
        return normalization * math.exp(
            -math.sqrt(math.log(1.0 / position))
        )

    first = continuous_integral(
        speed, transformed, absolute_tolerance=2e-13
    ) + math.exp(-speed) * normalization
    second = continuous_integral(
        speed, lambda position: transformed(position) ** 2,
        absolute_tolerance=2e-13,
    ) + math.exp(-speed) * normalization ** 2
    return second - first ** 2


def main():
    target_mean = 0.5 * (digamma(0.5) + math.log(2.0))
    target_variance = 0.25 * polygamma(1, 0.5)
    profile_rows = []
    for scale in PROFILE_SCALES:
        mean, variance = normalized_profile(scale)
        profile_rows.append((scale, mean / target_mean, variance / target_variance))

    chain_rows = []
    for speed in CHAIN_SPEEDS:
        scale = 0.5 * math.log(speed)
        _, profile_variance = normalized_profile(scale)
        exact_variance = exact_telegraph_variance(speed)
        derivative = 0.5 / math.sqrt(scale)
        chain_rows.append((
            speed,
            exact_variance / (derivative ** 2 * profile_variance),
        ))

    print("Vanishing-slope flat-profile covariance certificate")
    print("profile h(t) = sqrt(t), gauge rho(x) = |x|")
    print(f"Gaussian log-radius mean {target_mean:.12f}")
    print(f"Gaussian log-radius variance {target_variance:.12f}")
    for scale, mean_ratio, variance_ratio in profile_rows:
        print(
            f"log scale {scale:4.0f}: normalized mean {mean_ratio:.9f}, "
            f"normalized variance {variance_ratio:.9f}"
        )
    for speed, ratio in chain_rows:
        print(f"speed {speed:5.0f}: exact/profile variance {ratio:.9f}")

    ok = True
    euler_gamma = -digamma(1.0)
    ok &= abs(target_mean + 0.5 * (euler_gamma + math.log(2.0))) < 5e-13
    ok &= abs(target_variance - math.pi ** 2 / 8.0) < 5e-13
    ok &= all(
        0.8 < profile_rows[index][1] < profile_rows[index + 1][1] < 1.0
        for index in range(len(profile_rows) - 1)
    )
    ok &= all(
        0.7 < profile_rows[index][2] < profile_rows[index + 1][2] < 1.0
        for index in range(len(profile_rows) - 1)
    )
    ok &= profile_rows[-1][1] > 0.992
    ok &= profile_rows[-1][2] > 0.983
    ok &= all(abs(ratio - 1.0) < 0.01 for _, ratio in chain_rows)
    ok &= abs(chain_rows[-1][1] - 1.0) < 1e-5
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
