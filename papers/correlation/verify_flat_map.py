"""Exact telegraph certificate for a smooth flat post-averaging map.

The stationary symmetric telegraph process has velocities +/-1 and switches
at rate m on [0, 1].  Its integrated position has the classical atom-plus-
Bessel law.  We apply the C-infinity flat map exp(-1/x^2), with value zero at
the origin, and verify its stretched-exponential covariance scale.  Ends with
PASS or FAIL.
"""
import math

from scipy.integrate import quad
from scipy.special import ive


SPEEDS = [16, 64, 256, 1024, 4096]
FAMILY_SPEED = 262144
FLAT_POWERS = [1.0, 2.0, 4.0]


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

    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
