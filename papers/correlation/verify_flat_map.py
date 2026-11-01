"""Exact telegraph certificate for a smooth flat post-averaging map.

The stationary symmetric telegraph process has velocities +/-1 and switches
at rate m on [0, 1].  Its integrated position has the classical atom-plus-
Bessel law.  We apply C-infinity flat maps with value zero at the origin and
verify both stretched-exponential and slower log-flat covariance scales.  Ends
with PASS or FAIL.
"""
import math

from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import i0e, i1e, ive


SPEEDS = [16, 64, 256, 1024, 4096]
FAMILY_SPEED = 262144
FLAT_POWERS = [1.0, 2.0, 4.0]
LOG_FLAT_SPEEDS = [1e8, 1e32, 1e128, 1e256]
BORDERLINE_SPEEDS = LOG_FLAT_SPEEDS


def continuous_integral(speed, weight, absolute_tolerance=1e-13):
    """Integrate a weight against the continuous half of the exact law."""
    turning_point = math.asin(min(0.99, speed ** -0.25))

    def integrand(theta):
        position = math.sin(theta)
        radius = math.cos(theta)
        argument = speed * radius
        density_dx = (
            0.5 * speed * math.exp(argument - speed)
            * (radius * ive(0, argument) + ive(1, argument))
        )
        return weight(position) * density_dx

    value, _ = quad(
        integrand, 0.0, math.pi / 2,
        epsabs=absolute_tolerance, epsrel=2e-11, limit=300,
        points=[turning_point],
    )
    return 2.0 * value


def flat_moment(speed, power):
    """Return E[exp(-power / X_m^2)], defining the value at zero as zero."""
    def weight(position):
        if position == 0.0:
            return 0.0
        return math.exp(-power / position ** 2)

    continuous = continuous_integral(speed, weight, absolute_tolerance=0.0)
    atoms = math.exp(-speed - power)
    return continuous + atoms


def logarithmic_constant(alpha, power):
    """Laplace exponent for E exp(-power / |X_m|**alpha)."""
    return (
        0.5 * (alpha + 2.0)
        * alpha ** (-alpha / (alpha + 2.0))
        * power ** (2.0 / (alpha + 2.0))
    )


def log_flat_moment(speed, alpha, power):
    """Log moment with its leading stretched-exponential cost removed."""
    exponent = alpha / (alpha + 2.0)
    shift = logarithmic_constant(alpha, power) * speed ** exponent
    optimizer = (alpha * power / speed) ** (1.0 / (alpha + 2.0))
    turning_point = math.asin(min(0.99, optimizer))

    def integrand(theta):
        position = math.sin(theta)
        if position == 0.0:
            return 0.0
        radius = math.cos(theta)
        argument = speed * radius
        return (
            math.exp(
                -power / position ** alpha + argument - speed + shift
            )
            * 0.5 * speed
            * (radius * ive(0, argument) + ive(1, argument))
        )

    value, _ = quad(
        integrand, 0.0, math.pi / 2,
        epsabs=1e-13, epsrel=2e-11, limit=300,
        points=[turning_point],
    )
    scaled_atoms = math.exp(-speed - power + shift)
    return math.log(2.0 * value + scaled_atoms) - shift


def slow_flat_saddle(speed, gamma, power):
    """Quadratic-rate saddle for exp(-power log(e/|X_m|)**gamma)."""
    right_side = math.log(speed) + 2.0 - math.log(power * gamma)

    def equation(radius_log):
        return 2.0 * radius_log + (gamma - 1.0) * math.log(radius_log) - right_side

    radius_log = brentq(equation, 0.5, right_side)
    cost = power * (
        radius_log ** gamma
        + 0.5 * gamma * radius_log ** (gamma - 1.0)
    )
    return radius_log, cost


def log_profile_moment(speed, power, profile, profile_derivative, saddle):
    """Exact-law log moment for a profile of log(e/|X_m|)."""
    radius_log, shift = saddle(speed, power)
    optimizer = math.sqrt(power * profile_derivative(radius_log))
    upper = max(30.0, 5.0 * optimizer + 20.0)

    def integrand(rescaled_position):
        if rescaled_position == 0.0:
            return 0.0
        position = rescaled_position / math.sqrt(speed)
        radius = math.sqrt(1.0 - position ** 2)
        argument = speed * radius
        log_radius = math.log(math.e / position)
        exponent = (
            -power * profile(log_radius) + shift
            - rescaled_position ** 2 / (1.0 + radius)
        )
        return (
            math.exp(exponent) * 0.5 * math.sqrt(speed)
            * (i0e(argument) + i1e(argument) / radius)
        )

    value, _ = quad(
        integrand, 0.0, upper, epsabs=1e-12, epsrel=2e-11,
        limit=400, points=[optimizer],
    )
    scaled_atoms = math.exp(-speed - power * profile(1.0) + shift)
    return math.log(2.0 * value + scaled_atoms) - shift


def log_slow_flat_moment(speed, gamma, power):
    """Exact-law log moment for the log-power flat family."""
    return log_profile_moment(
        speed,
        power,
        lambda radius_log: radius_log ** gamma,
        lambda radius_log: gamma * radius_log ** (gamma - 1.0),
        lambda local_speed, local_power: slow_flat_saddle(
            local_speed, gamma, local_power
        ),
    )


def borderline_saddle(speed, power):
    """Saddle for the profile H(r)=r log(r)**2."""
    right_side = math.log(speed) + 2.0 - math.log(power)

    def equation(radius_log):
        log_radius = math.log(radius_log)
        derivative = log_radius ** 2 + 2.0 * log_radius
        return (
            2.0 * radius_log + math.log(derivative)
            - right_side
        )

    radius_log = brentq(equation, 1.01, right_side)
    log_radius = math.log(radius_log)
    derivative = log_radius ** 2 + 2.0 * log_radius
    cost = power * (
        radius_log * log_radius ** 2 + 0.5 * derivative
    )
    return radius_log, cost


def log_borderline_moment(speed, power):
    """Exact-law log moment for H(r)=r log(r)**2."""
    return log_profile_moment(
        speed,
        power,
        lambda radius_log: radius_log * math.log(radius_log) ** 2,
        lambda radius_log: (
            math.log(radius_log) ** 2 + 2.0 * math.log(radius_log)
        ),
        borderline_saddle,
    )


def polynomial_moment(speed, power):
    """Return an ordinary even moment, including the two endpoint atoms."""
    continuous = continuous_integral(
        speed, lambda position: position ** power
    )
    return continuous + math.exp(-speed)


def main():
    ok = True
    normalization_errors = []
    second_moment_errors = []
    variances = []
    mean_rates = []
    variance_rates = []

    for speed in SPEEDS:
        mass = polynomial_moment(speed, 0)
        exact_second = (
            1.0 / speed
            - (1.0 - math.exp(-2.0 * speed)) / (2.0 * speed ** 2)
        )
        second = polynomial_moment(speed, 2)
        mean = flat_moment(speed, 1)
        second_flat = flat_moment(speed, 2)
        variance = second_flat - mean ** 2
        mean_rate = -math.log(mean) / math.sqrt(speed)
        variance_rate = -math.log(variance) / math.sqrt(speed)

        normalization_errors.append(abs(mass - 1.0))
        second_moment_errors.append(abs(second - exact_second))
        variances.append(variance)
        mean_rates.append(mean_rate)
        variance_rates.append(variance_rate)

        print(
            f"speed {speed:4d}: flat mean {mean:.8e}; "
            f"variance {variance:.8e}; log rates "
            f"{mean_rate:.6f} {variance_rate:.6f}"
        )

    maximum_mass_error = max(normalization_errors)
    maximum_second_error = max(second_moment_errors)
    print(f"maximum density normalization error {maximum_mass_error:.2e}")
    print(f"maximum exact second-moment error {maximum_second_error:.2e}")
    print(
        "limiting log rates sqrt(2) and 2; final errors "
        f"{abs(mean_rates[-1] - math.sqrt(2.0)):.6f} "
        f"{abs(variance_rates[-1] - 2.0):.6f}"
    )

    family_errors = []
    for alpha in FLAT_POWERS:
        exponent = alpha / (alpha + 2.0)
        log_mean = log_flat_moment(FAMILY_SPEED, alpha, 1.0)
        log_second = log_flat_moment(FAMILY_SPEED, alpha, 2.0)
        log_variance = log_second + math.log1p(
            -math.exp(2.0 * log_mean - log_second)
        )
        target = logarithmic_constant(alpha, 2.0)
        observed = -log_variance / FAMILY_SPEED ** exponent
        error = abs(observed - target)
        family_errors.append(error)
        print(
            f"flat power {alpha:.0f} at speed {FAMILY_SPEED}: "
            f"stretch exponent {exponent:.6f}; "
            f"variance log rate {observed:.6f}; target {target:.6f}"
        )

    slow_rates = []
    separation_rates = []
    saddle_errors = []
    gamma = 2.0
    for speed in LOG_FLAT_SPEEDS:
        log_speed = math.log(speed)
        log_mean = log_slow_flat_moment(speed, gamma, 1.0)
        log_second = log_slow_flat_moment(speed, gamma, 2.0)
        log_variance = log_second + math.log1p(
            -math.exp(2.0 * log_mean - log_second)
        )
        target = 2.0 ** (1.0 - gamma)
        observed = -log_variance / log_speed ** gamma
        separation = (-2.0 * log_mean + log_second) / log_speed
        separation_target = gamma * math.log(2.0) / 2.0 ** (gamma - 1.0)
        _, mean_cost = slow_flat_saddle(speed, gamma, 1.0)
        _, second_cost = slow_flat_saddle(speed, gamma, 2.0)
        saddle_error = max(
            abs(-log_mean - mean_cost), abs(-log_second - second_cost)
        )
        slow_rates.append(observed)
        separation_rates.append(separation)
        saddle_errors.append(saddle_error)
        print(
            f"log-flat power {gamma:.0f} at speed {speed:.0e}: "
            f"variance log rate {observed:.6f}; target {target:.6f}; "
            f"separation {separation:.6f}; target {separation_target:.6f}; "
            f"saddle error {saddle_error:.6f}"
        )
    print(f"maximum log-flat saddle error {max(saddle_errors):.6f}")

    borderline_rates = []
    borderline_separations = []
    borderline_saddle_errors = []
    for speed in BORDERLINE_SPEEDS:
        log_speed = math.log(speed)
        profile_log = math.log(0.5 * log_speed)
        profile_derivative = profile_log ** 2 + 2.0 * profile_log
        log_mean = log_borderline_moment(speed, 1.0)
        log_second = log_borderline_moment(speed, 2.0)
        log_variance = log_second + math.log1p(
            -math.exp(2.0 * log_mean - log_second)
        )
        observed = -log_variance / (log_speed * profile_log ** 2)
        separation = (
            (-2.0 * log_mean + log_second) / profile_derivative
        )
        _, mean_cost = borderline_saddle(speed, 1.0)
        _, second_cost = borderline_saddle(speed, 2.0)
        saddle_error = max(
            abs(-log_mean - mean_cost), abs(-log_second - second_cost)
        )
        borderline_rates.append(observed)
        borderline_separations.append(separation)
        borderline_saddle_errors.append(saddle_error)
        print(
            f"borderline log-flat at speed {speed:.0e}: "
            f"variance log rate {observed:.6f}; target 1.000000; "
            f"separation {separation:.6f}; target {math.log(2.0):.6f}; "
            f"saddle error {saddle_error:.6f}"
        )
    print(
        "maximum borderline saddle error "
        f"{max(borderline_saddle_errors):.6f}"
    )

    ok &= maximum_mass_error < 2e-12
    ok &= maximum_second_error < 2e-12
    ok &= all(left > right for left, right in zip(mean_rates, mean_rates[1:]))
    ok &= all(left > right for left, right in zip(
        variance_rates, variance_rates[1:]
    ))
    ok &= abs(mean_rates[-1] - math.sqrt(2.0)) < 0.005
    ok &= abs(variance_rates[-1] - 2.0) < 0.01
    ok &= variances[-1] < 2e-56
    ok &= max(family_errors) < 0.01
    ok &= abs(slow_rates[-1] - 0.5) < 0.01
    ok &= abs(separation_rates[-1] - math.log(2.0)) < 0.01
    ok &= max(saddle_errors) < 0.5
    ok &= abs(borderline_rates[-1] - 1.0) < 0.02
    ok &= abs(borderline_separations[-1] - math.log(2.0)) < 0.02
    ok &= max(borderline_saddle_errors) < 0.5

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
