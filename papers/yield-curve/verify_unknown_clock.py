"""Unknown positive Cox-clock calibration from labels and bounded waiting transforms."""
from collections import defaultdict
from fractions import Fraction as F
from math import exp, log, sqrt

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm


def radius(visits, size, delta):
    if not visits:
        return 1.
    variance = 1 + visits / 4
    return min(1., sqrt(2 * variance * log(size * (size + 1) * sqrt(variance) / delta)) / visits)


def reconstruct(kernel, transform, eta, zeta, discount):
    try:
        inverse = np.linalg.inv(kernel)
    except np.linalg.LinAlgError:
        return None
    kappa = np.linalg.norm(inverse, np.inf)
    if kappa * eta >= 1:
        return None
    inverse_error = kappa ** 2 * eta / (1 - kappa * eta)
    denominator = 1 - inverse @ transform
    error = inverse_error + kappa * zeta
    if np.min(denominator) <= error or np.min(transform) <= zeta:
        return None
    rates = discount * transform / denominator
    lower = discount * np.maximum(0, transform - zeta) / (denominator + error)
    upper = discount * np.minimum(1, transform + zeta) / (denominator - error)
    rate_error = max(np.max(rates - lower), np.max(upper - rates))
    generator = np.diag(rates) @ (np.eye(len(kernel)) - inverse)
    budget = rate_error * np.linalg.norm(np.eye(len(kernel)) - inverse, np.inf) + max(upper) * inverse_error
    return generator, rates, lower, upper, budget, kappa * eta


means = crossings = 0
max_mean = max_failure = 0.
weights = (F(9, 10), F(1, 20), F(1, 20))
reward = np.array([[F(i + j + 1, 6) for j in range(3)] for i in range(3)])
for persistence in (F(1, 10), F(9, 10)):
    kernel = tuple(tuple(persistence * (i == j) + (1 - persistence) * weights[j] for j in range(3)) for i in range(3))
    targets = [[float(x) for x in row] + [float(sum(p * w for p, w in zip(row, reward[i])))] for i, row in enumerate(kernel)]
    for start in range(3):
        paths = {(start, (0,) * 9): F(1)}
        surviving, failed = paths.copy(), F(0)
        for horizon in range(1, 8):
            def advance(law):
                result = defaultdict(F)
                for (state, counts), mass in law.items():
                    for target in range(3):
                        updated = list(counts)
                        updated[3 * state + target] += 1
                        result[target, tuple(updated)] += mass * kernel[state][target]
                return result
            def values(counts):
                output = []
                for i in range(3):
                    row = counts[3 * i:3 * i + 3]
                    visits = sum(row)
                    sums = list(row) + [float(sum(c * w for c, w in zip(row, reward[i])))]
                    variance = 1 + visits / 4
                    output.extend(exp((total - visits * target) ** 2 / (2 * variance)) / sqrt(variance)
                                  for total, target in zip(sums, targets[i]))
                return np.array(output)
            paths, next_surviving = advance(paths), advance(surviving)
            expectations = sum(float(mass) * values(counts) for (_, counts), mass in paths.items())
            assert np.max(expectations) <= 1 + 1e-10
            max_mean = max(max_mean, np.max(expectations))
            means += len(expectations)
            surviving = {}
            for key, mass in next_surviving.items():
                if max(values(key[1])) > 60:
                    failed += mass
                else:
                    surviving[key] = mass
            assert failed <= F(1, 5)
            max_failure = max(max_failure, float(failed))
            crossings += 1
print(f"Enumerated bounded-reward mixture means: {means}; maximum={max_mean:.9f}")
print(f"First-crossing laws: {crossings}; maximum failure={max_failure:.9f} at delta=0.2")

rng = np.random.default_rng(20261017)
chains = []
for size in range(2, 6):
    off = rng.uniform(.05, .8, (size, size))
    np.fill_diagonal(off, 0)
    chains.append(off - np.diag(off.sum(axis=1)))
for size in (3, 5):
    chains.append(.7 * (np.roll(np.eye(size), 1, axis=1) - np.eye(size)))
chains.append(np.array([[-.25, .25, 0], [0, -.25, .25], [1, 0, -1]]))
checked = passed = abstained = clipped = 0
max_residual = max_ratio = 0.
for generator in chains:
    size = len(generator)
    identity = np.eye(size)
    for rates in (np.full(size, .3), np.ones(size), np.full(size, 5), np.linspace(.5, 2, size)):
        killed = np.diag(rates) - generator
        kernel = np.linalg.solve(killed, np.diag(rates))
        row_speeds = np.diag(np.linspace(.4, 3, size))
        aliased = np.linalg.solve(row_speeds @ killed, row_speeds @ np.diag(rates))
        assert np.linalg.norm(aliased - kernel, np.inf) < 1e-10
        for discount in (.2, 1., 5.):
            transform = np.linalg.solve(killed + discount * identity, rates)
            integral, _ = quad_vec(lambda t: exp(-discount * t) * (expm(-t * killed) @ rates), 0, np.inf, epsabs=1e-11)
            assert np.linalg.norm(integral - transform, np.inf) < 1e-9
            result = reconstruct(kernel, transform, 0., 0., discount)
            residual = max(np.linalg.norm(result[0] - generator, np.inf), max(abs(result[1] - rates))) / max(1, max(rates), np.linalg.norm(generator, np.inf))
            assert residual < 1e-9
            max_residual = max(max_residual, residual)
            checked += 1
            noise = rng.dirichlet(np.ones(size), size)
            for amplitude in (1e-5, 1e-4, .001, .03):
                estimate = (1 - amplitude) * kernel + amplitude * noise
                moment = (1 - amplitude) * transform + amplitude * rng.uniform(0, 1, size)
                result = reconstruct(estimate, moment, np.linalg.norm(estimate - kernel, np.inf), max(abs(moment - transform)), discount)
                if result is None:
                    abstained += 1
                    continue
                candidate, inferred, lower, upper, budget, _ = result
                slack = 2048 * np.finfo(float).eps * max(1, max(rates), np.linalg.norm(generator, np.inf))
                assert np.all(lower <= rates + slack) and np.all(rates <= upper + slack)
                error = np.linalg.norm(candidate - generator, np.inf)
                assert error <= budget + slack
                projected = np.maximum(candidate, 0)
                np.fill_diagonal(projected, 0)
                projected -= np.diag(projected.sum(axis=1))
                cost = np.linalg.norm(projected - candidate, np.inf)
                assert np.linalg.norm(projected - generator, np.inf) <= budget + cost + slack
                clipped += cost > 1e-10
                max_ratio = max(max_ratio, error / budget)
                passed += 1
print(f"Inverse/quadrature cases: {checked}; maximum scaled reconstruction error={max_residual:.3e}")
print(f"Perturbations certified/abstained: {passed}/{abstained}; clipping costs: {clipped}; maximum error/budget={max_ratio:.9f}")
kernel = np.array([[.9, .08, .02], [.02, .9, .08], [.08, .02, .9]])
truth = np.diag([1., 2., 4.]) @ (np.eye(3) - np.linalg.inv(kernel))
transform = np.linalg.solve(np.eye(3) + np.diag([1., 2., 4.]) - truth, np.array([1., 2., 4.]))
zeta = radius(1000000, 3, .05)
result = reconstruct(kernel, transform, 3 * zeta, zeta, 1.)
assert result is not None
print(f"Million-visits bounded budget: zeta={zeta:.9f}; eta={3 * zeta:.9f}; gate={result[5]:.9f}")
print("Clock intervals: " + "; ".join(f"[{a:.9f},{b:.9f}]" for a, b in zip(result[2], result[3])))
print(f"Generator radius={result[4]:.9f}; transform=" + ",".join(f"{x:.9f}" for x in transform))
assert reconstruct(np.eye(3), np.zeros(3), 2., 1., 1.) is None
print("PASS")
