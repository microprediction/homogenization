"""First-order rank-error cancellation by calibration/test-gap design."""
from fractions import Fraction as F
from itertools import product
from math import log, prod

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import roots_hermitenorm
from scipy.stats import binom, norm


def exact_separated_polynomial(size, rank, final_multiplier, calibration_multipliers=None):
    """Uniform(- state) on [0,1], Uniform(+ state) on [2,3]."""
    coefficients = [F(0)] * (size + 1)
    multipliers = list(calibration_multipliers or [1] * (size - 1)) + [final_multiplier]
    for states in product((-1, 1), repeat=size + 1):
        lower = states[:-1].count(-1)
        conditional = (min(F(1), F(rank, lower + 1)) if states[-1] == -1 else
                       min(F(1), max(F(0), F(rank - lower, size - lower + 1))))
        polynomial = [1]
        for edge in range(size):
            weight = states[edge] * states[edge + 1] * multipliers[edge]
            updated = polynomial + [0]
            for degree, coefficient in enumerate(polynomial):
                updated[degree + 1] += weight * coefficient
            polynomial = updated
        for degree, coefficient in enumerate(polynomial):
            coefficients[degree] += conditional * F(coefficient, 2 ** (size + 1))
    return coefficients


exact_cases = bounds = 0
for size in range(2, 10):
    for rank in range(1, size + 1):
        polynomial = exact_separated_polynomial(size, rank, 2)
        assert polynomial[0] == F(rank, size + 1) and polynomial[1] == 0
        exact_cases += 1
        for rho in (F(1, 100), F(1, 10), F(1, 2)):
            error = abs(sum(coefficient * rho ** degree for degree, coefficient in enumerate(polynomial[2:], 2)))
            bound = F(min(rank, size + 1 - rank), size + 1) * ((1 + rho) ** (size - 1) * (1 + 2 * rho) - 1 - (size + 1) * rho)
            assert error <= bound
            bounds += 1
irregular_cases = 0
for size in range(3, 7):
    multipliers = list(range(1, size)) + [size]
    for rank in range(1, size + 1):
        polynomial = exact_separated_polynomial(size, rank, size, multipliers[:-1])
        assert polynomial[1] == 0
        irregular_cases += 1
        for rho in (F(1, 10 * size), F(1, 2 * size), F(1, size)):
            error = abs(sum(coefficient * rho ** degree for degree, coefficient in enumerate(polynomial[2:], 2)))
            bound = F(min(rank, size + 1 - rank), size + 1) * (prod(1 + rho * a for a in multipliers) - 1 - rho * sum(multipliers))
            assert error <= bound
            bounds += 1
print(f"Exact separated-emission polynomial cancellations: {exact_cases} regular, {irregular_cases} irregular; finite remainder checks: {bounds}")
assert exact_separated_polynomial(2, 2, 2) == [F(2, 3), F(0), F(1, 12)]
print("Sharp-order witness, N=k=2: coverage = 2/3 + rho^2/12")


def coverage(size, rank, correlations, means, scales, order=240):
    nodes, weights = roots_hermitenorm(order)
    weights /= np.sqrt(2 * np.pi)

    def count_mass(x):
        cdf = norm.cdf((np.atleast_1d(x)[:, None] - means) / scales)
        mass = np.zeros((len(cdf), rank, 2))
        mass[:, 0] = .5
        for rho in correlations:
            updated = mass * (1 - cdf[:, None, :])
            updated[:, 1:] += mass[:, :-1] * cdf[:, None, :]
            transition = np.array([[1 + rho, 1 - rho], [1 - rho, 1 + rho]]) / 2
            mass = updated @ transition
        return mass.sum(axis=1)

    value = sum(weights @ count_mass(means[state] + scales[state] * nodes)[:, state] for state in range(2))
    return value, count_mass


def coefficients(size, rank, means, scales):
    def integrand(x):
        cdf = norm.cdf((x - means) / scales)
        density = norm.pdf((x - means) / scales) / scales
        u, pooled_density = cdf.mean(), density.mean()
        primitive = (cdf[1] - cdf[0]) / 2
        cc = primitive ** 2 * (binom.pmf(rank - 1, size - 2, u) - binom.pmf(rank - 2, size - 2, u)) * pooled_density
        ct = -(density[1] - density[0]) / 2 * primitive * binom.pmf(rank - 1, size - 1, u)
        return np.array([cc, ct])
    return quad_vec(integrand, -np.inf, np.inf, epsabs=2e-12)[0]


checked = 0
for size in (2, 3, 5, 9):
    for rank in range(1, size + 1):
        for means, scales in ((np.array([-1., 1.]), np.array([.5, .5])),
                              (np.array([-.3, 1.2]), np.array([.7, 1.3]))):
            cc, ct = coefficients(size, rank, means, scales)
            assert abs(ct + (size - 1) * cc / 2) < 2e-10
            epsilon = 1e-4
            plus, _ = coverage(size, rank, [epsilon] * (size - 1) + [2 * epsilon], means, scales)
            minus, _ = coverage(size, rank, [-epsilon] * (size - 1) + [-2 * epsilon], means, scales)
            assert abs((plus - minus) / (2 * epsilon)) < 2e-7
            checked += 1
print(f"Gaussian integration-by-parts and zero-slope cases: {checked}")
size = rank = 9
means, scales = np.array([-1., 1.]), np.array([.5, .5])
cc, ct = coefficients(size, rank, means, scales)
print(f"N=k=9: A_CC={cc:.10f}; A_CT={ct:.10f}; equal-gap coefficient={(size - 1) * cc + ct:.10f}")
errors = []
print("rho   equal-span equal-gap error   balanced-gap error   finite bound")
for rho in (.1, .05, .025, .0125):
    equal_rho = rho * 2 ** (1 / size)
    ordinary, _ = coverage(size, rank, [equal_rho] * size, means, scales)
    balanced, mass = coverage(size, rank, [rho] * (size - 1) + [2 * rho], means, scales)
    independent = sum(quad(lambda x: norm.pdf((x - means[state]) / scales[state]) / scales[state] * mass(x)[0, state],
                           -np.inf, np.inf, epsabs=2e-11)[0] for state in range(2))
    assert abs(independent - balanced) < 2e-10
    bound = min(rank, size + 1 - rank) / (size + 1) * ((1 + rho) ** (size - 1) * (1 + 2 * rho) - 1 - (size + 1) * rho)
    error = balanced - rank / (size + 1)
    assert abs(error) <= bound + 1e-10
    errors.append(abs(error))
    print(f"{rho:.4f} {ordinary - .9:+.10f} {error:+.10f} {bound:.10f}")
order = log(errors[-2] / errors[-1]) / log(2)
assert 1.9 < order < 2.1
print(f"Last balanced-gap error order: {order:.6f}")
equal, _ = coverage(9, 9, [.2] * 9, means, scales)
balanced, _ = coverage(9, 9, [.2] * 8 + [.4], means, scales)
print(f"Equal calibration gaps, rho=.2: ordinary coverage={equal:.10f}; balanced coverage={balanced:.10f}")
print("PASS")
