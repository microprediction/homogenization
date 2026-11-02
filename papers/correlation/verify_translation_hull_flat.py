#!/usr/bin/env python3
"""Certificate for the bounded-slope translation-hull covariance theorem."""

import math

from scipy.integrate import quad
from scipy.special import i0e, i1e


AMPLITUDE = 0.8
PHASES = (0.0, 0.5 * math.pi, math.pi, 1.5 * math.pi)


def profile(value):
    """Periodic perturbation h(t)=t+a sin(t)."""
    return value + AMPLITUDE * math.sin(value)


def derivative(value):
    """Derivative of the periodic profile."""
    return 1.0 + AMPLITUDE * math.cos(value)


def local_increment(phase, shift):
    """Translation-hull element q_phase(shift)."""
    return (
        shift
        + AMPLITUDE * (math.sin(phase + shift) - math.sin(phase))
    )


def limit_map(phase, radius):
    """Limiting normalized map exp(-q_phase(-log radius))."""
    if radius == 0.0:
        return 0.0
    shift = -math.log(radius)
    return math.exp(-local_increment(phase, shift))


def gaussian_moment(phase, power):
    """Moment of the translation-hull functional under N(0,1)."""
    def integrand(radius):
        return (
            limit_map(phase, radius) ** power
            * math.exp(-0.5 * radius ** 2)
            / math.sqrt(2.0 * math.pi)
        )

    value, _ = quad(
        integrand, 0.0, 12.0, epsabs=2e-13, epsrel=2e-12, limit=300
    )
    return 2.0 * value


def gaussian_variance(phase):
    """Variance of the translation-hull Gaussian functional."""
    first = gaussian_moment(phase, 1.0)
    second = gaussian_moment(phase, 2.0)
    return second - first ** 2


def exact_moment(logarithmic_scale, power):
    """Normalized moment under the exact telegraph occupation law."""
    speed = math.exp(2.0 * logarithmic_scale)
    square_root_speed = math.sqrt(speed)

    def integrand(rescaled_position):
        if rescaled_position == 0.0:
            return 0.0
        position = rescaled_position / square_root_speed
        radius = math.sqrt(1.0 - position ** 2)
        argument = speed * radius
        log_radius = math.log(1.0 / position)
        normalized_weight = math.exp(
            power * (profile(logarithmic_scale) - profile(log_radius))
        )
        density = (
            0.5 * square_root_speed
            * (i0e(argument) + i1e(argument) / radius)
            * math.exp(-rescaled_position ** 2 / (1.0 + radius))
        )
        return normalized_weight * density

    continuous_half, _ = quad(
        integrand, 0.0, 12.0,
        epsabs=2e-13, epsrel=2e-12, limit=300,
    )
    atom = math.exp(
        -speed + power * (profile(logarithmic_scale) - profile(0.0))
    )
    return 2.0 * continuous_half + atom


def exact_variance(logarithmic_scale):
    """Normalized variance under the exact telegraph law."""
    first = exact_moment(logarithmic_scale, 1.0)
    second = exact_moment(logarithmic_scale, 2.0)
    return second - first ** 2


def phase_name(phase):
    """Short label for the four certificate phases."""
    names = {
        0.0: "0",
        0.5 * math.pi: "pi/2",
        math.pi: "pi",
        1.5 * math.pi: "3pi/2",
    }
    return names[phase]


def main():
    constants = [(phase, derivative(phase), gaussian_variance(phase))
                 for phase in PHASES]
    equal_slope_rows = []
    phase_cycles = (
        (0.5 * math.pi, (1, 2, 3)),
        (1.5 * math.pi, (0, 1, 2)),
    )
    for phase, cycles in phase_cycles:
        target = gaussian_variance(phase)
        for cycle in cycles:
            logarithmic_scale = phase + 2.0 * math.pi * cycle
            equal_slope_rows.append((
                phase,
                cycle,
                logarithmic_scale,
                exact_variance(logarithmic_scale) / target,
            ))

    power_variance = 1.0 - 2.0 / math.pi
    lower_equal_slope = gaussian_variance(0.5 * math.pi)
    upper_equal_slope = gaussian_variance(1.5 * math.pi)

    print("Bounded-slope translation-hull covariance certificate")
    print("profile h(t) = t + 0.8 sin(t)")
    for phase, slope, variance in constants:
        print(
            f"phase {phase_name(phase):5s}: slope {slope:.9f}, "
            f"Gaussian variance {variance:.12f}"
        )
    print(f"instantaneous-slope power variance {power_variance:.12f}")
    print(
        "equal-slope phase gap "
        f"{abs(upper_equal_slope - lower_equal_slope):.12f}"
    )
    for phase, cycle, scale, ratio in equal_slope_rows:
        print(
            f"phase {phase_name(phase):5s}, cycle {cycle}: "
            f"L {scale:.9f}, exact/profile {ratio:.9f}"
        )

    first_phase_rows = equal_slope_rows[:3]
    second_phase_rows = equal_slope_rows[3:]
    ok = True
    ok &= abs(profile(0.0)) < 5e-14
    ok &= abs(derivative(0.5 * math.pi) - 1.0) < 5e-14
    ok &= abs(derivative(1.5 * math.pi) - 1.0) < 5e-14
    ok &= abs(upper_equal_slope - lower_equal_slope) > 0.19
    ok &= abs(lower_equal_slope - power_variance) > 0.1
    ok &= abs(upper_equal_slope - power_variance) > 0.05
    ok &= all(
        abs(first_phase_rows[index + 1][3] - 1.0)
        < abs(first_phase_rows[index][3] - 1.0)
        for index in range(2)
    )
    ok &= all(
        abs(second_phase_rows[index + 1][3] - 1.0)
        < abs(second_phase_rows[index][3] - 1.0)
        for index in range(2)
    )
    ok &= abs(first_phase_rows[-1][3] - 1.0) < 2e-10
    ok &= abs(second_phase_rows[-1][3] - 1.0) < 2e-10
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
