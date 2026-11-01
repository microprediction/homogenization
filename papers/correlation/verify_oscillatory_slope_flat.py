#!/usr/bin/env python3
"""Certificate for bounded, nonconvergent logarithmic slopes."""

import math

from scipy.integrate import quad
from scipy.special import i0e, i1e


FREQUENCY = 3.0
PHASES = (
    ("minimum", 1.5 * math.pi),
    ("maximum", 2.5 * math.pi),
    ("minimum", 3.5 * math.pi),
    ("maximum", 4.5 * math.pi),
    ("minimum", 5.5 * math.pi),
    ("maximum", 6.5 * math.pi),
)


def profile(value):
    """Increasing profile with derivative 1 + sin(3(sqrt(1+t)-1))/2."""
    root = math.sqrt(1.0 + value)
    phase = FREQUENCY * (root - 1.0)
    return (
        value
        + (1.0 - root * math.cos(phase)) / FREQUENCY
        + math.sin(phase) / FREQUENCY ** 2
    )


def derivative(value):
    """Logarithmic slope of the profile."""
    phase = FREQUENCY * (math.sqrt(1.0 + value) - 1.0)
    return 1.0 + 0.5 * math.sin(phase)


def gaussian_absolute_moment(power):
    """Return E|G|^power for a standard Gaussian G."""
    return (
        2.0 ** (0.5 * power)
        * math.gamma(0.5 * (power + 1.0))
        / math.sqrt(math.pi)
    )


def moving_variance(slope):
    """Variance of |G|^slope."""
    first = gaussian_absolute_moment(slope)
    second = gaussian_absolute_moment(2.0 * slope)
    return second - first ** 2


def normalized_map(logarithmic_scale, radius):
    """Return exp(h(L)-h(L-log radius)), with value zero at radius zero."""
    if radius == 0.0:
        return 0.0
    return math.exp(
        profile(logarithmic_scale)
        - profile(logarithmic_scale - math.log(radius))
    )


def gaussian_profile_moment(logarithmic_scale, power):
    """Moment of the finite profile under the limiting Gaussian law."""
    def integrand(radius):
        return (
            normalized_map(logarithmic_scale, radius) ** power
            * math.exp(-0.5 * radius ** 2)
            / math.sqrt(2.0 * math.pi)
        )

    value, _ = quad(
        integrand, 0.0, 12.0, epsabs=2e-13, epsrel=2e-12, limit=300
    )
    return 2.0 * value


def exact_profile_moment(logarithmic_scale, power):
    """Moment under the exact atom-plus-Bessel telegraph law."""
    speed = math.exp(2.0 * logarithmic_scale)
    square_root_speed = math.sqrt(speed)

    def integrand(rescaled_position):
        if rescaled_position == 0.0:
            return 0.0
        position = rescaled_position / square_root_speed
        radius = math.sqrt(1.0 - position ** 2)
        argument = speed * radius
        density = (
            0.5 * square_root_speed
            * (i0e(argument) + i1e(argument) / radius)
            * math.exp(-rescaled_position ** 2 / (1.0 + radius))
        )
        return (
            normalized_map(logarithmic_scale, rescaled_position) ** power
            * density
        )

    value, _ = quad(
        integrand, 0.0, 12.0, epsabs=2e-13, epsrel=2e-12, limit=300
    )
    atom = math.exp(-speed + power * profile(logarithmic_scale))
    return 2.0 * value + atom


def phase_scale(phase):
    """Return L for which the derivative phase equals the requested value."""
    root = 1.0 + phase / FREQUENCY
    return root ** 2 - 1.0


def row(label, phase):
    """Compare moving-constant, finite-profile and exact-law variances."""
    logarithmic_scale = phase_scale(phase)
    slope = derivative(logarithmic_scale)
    target = moving_variance(slope)
    gaussian_first = gaussian_profile_moment(logarithmic_scale, 1.0)
    gaussian_second = gaussian_profile_moment(logarithmic_scale, 2.0)
    gaussian_variance = gaussian_second - gaussian_first ** 2
    exact_first = exact_profile_moment(logarithmic_scale, 1.0)
    exact_second = exact_profile_moment(logarithmic_scale, 2.0)
    exact_variance = exact_second - exact_first ** 2
    return (
        label,
        logarithmic_scale,
        slope,
        gaussian_variance / target,
        exact_variance / gaussian_variance,
    )


def main():
    rows = [row(*phase) for phase in PHASES]
    lower_constant = moving_variance(0.5)
    upper_constant = moving_variance(1.5)

    print("Oscillatory bounded-slope covariance certificate")
    print("h'(t) = 1 + sin(3(sqrt(1+t)-1))/2")
    print(f"slope 0.5 Gaussian variance {lower_constant:.12f}")
    print(f"slope 1.5 Gaussian variance {upper_constant:.12f}")
    for label, scale, slope, profile_ratio, exact_ratio in rows:
        print(
            f"{label:7s} at L {scale:.9f}: slope {slope:.1f}, "
            f"profile/moving {profile_ratio:.9f}, "
            f"exact/profile {exact_ratio:.9f}"
        )

    minimum_rows = rows[0::2]
    maximum_rows = rows[1::2]
    ok = True
    ok &= abs(derivative(0.0) - 1.0) < 5e-14
    ok &= abs(profile(0.0)) < 5e-14
    ok &= 0.1 < lower_constant < 0.2
    ok &= 0.7 < upper_constant < 1.0
    ok &= upper_constant - lower_constant > 0.6
    ok &= all(abs(values[2] - 0.5) < 5e-14 for values in minimum_rows)
    ok &= all(abs(values[2] - 1.5) < 5e-14 for values in maximum_rows)
    ok &= all(
        abs(minimum_rows[index + 1][3] - 1.0)
        < abs(minimum_rows[index][3] - 1.0)
        for index in range(len(minimum_rows) - 1)
    )
    ok &= all(
        abs(maximum_rows[index + 1][3] - 1.0)
        < abs(maximum_rows[index][3] - 1.0)
        for index in range(len(maximum_rows) - 1)
    )
    ok &= abs(minimum_rows[-1][3] - 1.0) < 0.08
    ok &= abs(maximum_rows[-1][3] - 1.0) < 0.08
    ok &= all(abs(values[4] - 1.0) < 2e-5 for values in rows)
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
