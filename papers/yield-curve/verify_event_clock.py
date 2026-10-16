"""Known Cox-clock inverse and anytime Dirichlet-row confidence certificate."""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from math import factorial, lgamma, log, sqrt

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm


@lru_cache(None)
def mixture(counts, probabilities):
    visits = sum(counts)
    value = F(factorial(len(counts) - 1), factorial(visits + len(counts) - 1))
    for count, probability in zip(counts, probabilities):
        value *= factorial(count) / probability ** count
    return value


means = crossings = 0
row_budgets = 0
worst_crossing = F(0)
for weights in ((F(1, 2), F(9, 20), F(1, 20)), (F(1, 3),) * 3):
    kernel = tuple(tuple(F(9, 10) * (i == j) + weights[j] / 10 for j in range(3)) for i in range(3))
    for start in range(3):
        paths = {(start, (0,) * 9): F(1)}
        survivors = paths.copy()
        failed = F(0)
        for horizon in range(1, 8):
            def advance(law):
                result = defaultdict(F)
                for (state, counts), mass in law.items():
                    for target in range(3):
                        updated = list(counts)
                        updated[3 * state + target] += 1
                        result[target, tuple(updated)] += mass * kernel[state][target]
                return result
            paths, next_survivors = advance(paths), advance(survivors)
            for row in range(3):
                assert sum(mass * mixture(counts[3 * row:3 * row + 3], kernel[row])
                           for (_, counts), mass in paths.items()) == 1
                means += 1
                for (_, counts), mass in paths.items():
                    row_counts = counts[3 * row:3 * row + 3]
                    visits = sum(row_counts)
                    if visits and mixture(row_counts, kernel[row]) <= 15:
                        empirical = np.array(row_counts) / visits
                        log_at_mle = lgamma(3) + sum(lgamma(x + 1) for x in row_counts) - lgamma(visits + 3)
                        log_at_mle -= sum(x * log(x / visits) for x in row_counts if x)
                        budget = min(2, sqrt(2 * (log(15) - log_at_mle) / visits))
                        assert np.linalg.norm(empirical - np.array(kernel[row], dtype=float), 1) <= budget + 1e-10
                        row_budgets += 1
            survivors = {}
            for key, mass in next_survivors.items():
                counts = key[1]
                if any(mixture(counts[3 * row:3 * row + 3], kernel[row]) > 15 for row in range(3)):
                    failed += mass
                else:
                    survivors[key] = mass
            assert failed <= F(1, 5)
            worst_crossing = max(worst_crossing, failed)
            crossings += 1
print(f"Exact row means: {means}; first-crossing laws: {crossings}; maximum failure: {float(worst_crossing):.9f}")
print(f"Accepted-row numerical budgets: {row_budgets}")

rng = np.random.default_rng(20261016)
chains = []
for size in range(2, 6):
    off = rng.uniform(.05, .8, (size, size))
    np.fill_diagonal(off, 0)
    chains.append(off - np.diag(off.sum(axis=1)))
for size in (3, 5):
    chains.append(.7 * (np.roll(np.eye(size), 1, axis=1) - np.eye(size)))
defective = np.array([[-.25, .25, 0], [0, -.25, .25], [1, 0, -1]])
assert np.array_equal(defective @ np.linalg.matrix_power(defective + .75 * np.eye(3), 2), np.zeros((3, 3)))
assert np.linalg.norm(defective @ (defective + .75 * np.eye(3)), np.inf) > .1
chains.append(defective)
checked = perturbed = clipped = 0
worst_residual = worst_ratio = 0.
for generator in chains:
    size = len(generator)
    identity = np.eye(size)
    system = generator.T.copy()
    system[-1] = 1
    stationary = np.linalg.solve(system, np.r_[np.zeros(size - 1), 1])
    for rates in (np.full(size, .3), np.ones(size), np.full(size, 5), np.linspace(.5, 2, size)):
        clock = np.diag(rates)
        kernel = np.linalg.solve(clock - generator, clock)
        integral, _ = quad_vec(lambda t: expm(t * (generator - clock)) @ clock, 0, np.inf, epsabs=1e-11)
        reconstructed = clock @ (identity - np.linalg.inv(kernel))
        residual = np.linalg.norm(reconstructed - generator, np.inf) / max(1, np.linalg.norm(generator, np.inf))
        worst_residual = max(worst_residual, residual)
        assert residual < 1e-9 and np.linalg.norm(integral - kernel, np.inf) < 1e-9
        event_law = stationary * rates / (stationary @ rates)
        assert np.linalg.norm(event_law @ kernel - event_law, 1) < 1e-10
        assert np.linalg.norm(np.linalg.solve(7 * clock - 7 * generator, 7 * clock) - kernel, np.inf) < 1e-10
        checked += 1
        noise = rng.dirichlet(np.ones(size), size)
        for amplitude in (1e-4, 1e-3, .01):
            estimate = (1 - amplitude) * kernel + amplitude * noise
            eta = np.linalg.norm(estimate - kernel, np.inf)
            inverse = np.linalg.inv(estimate)
            kappa = np.linalg.norm(inverse, np.inf)
            assert kappa * eta < 1
            radius = max(rates) * kappa ** 2 * eta / (1 - kappa * eta)
            candidate = clock @ (identity - inverse)
            projected = np.maximum(candidate, 0)
            np.fill_diagonal(projected, 0)
            projected -= np.diag(projected.sum(axis=1))
            cost = np.linalg.norm(projected - candidate, np.inf)
            slack = 1024 * np.finfo(float).eps * max(1, np.linalg.norm(generator, np.inf))
            error = np.linalg.norm(candidate - generator, np.inf)
            assert error <= radius + slack
            assert np.linalg.norm(projected - generator, np.inf) <= radius + cost + slack
            worst_ratio = max(worst_ratio, error / radius)
            clipped += cost > 1e-10
            perturbed += 1
print(f"Killed-kernel checks: {checked}; scaled inverse residual: {worst_residual:.3e}")
print(f"Perturbations: {perturbed}; nonzero clipping costs: {clipped}; maximum error/radius: {worst_ratio:.9f}")

counts = np.array([[900000, 80000, 20000], [20000, 900000, 80000], [80000, 20000, 900000]])
assert np.array_equal(counts.sum(axis=0), counts.sum(axis=1))
estimate = counts / counts.sum(axis=1)[:, None]
log_m = lgamma(3) + sum(lgamma(int(x) + 1) for x in counts[0]) - lgamma(1000003)
log_m -= sum(x * log(x / 1000000) for x in counts[0])
eta = min(2, sqrt(2 * (log(3 / .05) - log_m) / 1000000))
kappa = np.linalg.norm(np.linalg.inv(estimate), np.inf)
radius = 4 * kappa ** 2 * eta / (1 - kappa * eta)
assert kappa * eta < 1
candidate = np.diag([1, 2, 4]) @ (np.eye(3) - np.linalg.inv(estimate))
assert np.min(candidate + np.diag(-np.diag(candidate))) >= 0
physical = np.array([4, 2, 1]) / 7
assert np.linalg.norm(physical @ candidate, 1) < 1e-10
print(f"Balanced million-visits example: log M={log_m:.9f}; eta={eta:.9f}; kappa={kappa:.9f}")
print(f"Gate={kappa * eta:.9f}; generator radius={radius:.9f}; physical law=(4/7,2/7,1/7), event law=(1/3,1/3,1/3)")
cycle = np.roll(np.eye(3), 1, axis=1)
rate = 4 * np.pi / sqrt(3)
forward, backward = rate * (cycle - np.eye(3)), rate * (cycle.T - np.eye(3))
snapshot_gap = np.linalg.norm(expm(forward) - expm(backward), np.inf)
clock_gap = np.linalg.norm(np.linalg.inv(np.eye(3) - forward) - np.linalg.inv(np.eye(3) - backward), np.inf)
assert snapshot_gap < 1e-10
assert abs(clock_gap - 2 * rate / (1 + 3 * rate + 3 * rate ** 2)) < 1e-10
print(f"Alias audit: unit-spacing gap={snapshot_gap:.3e}; unit-rate Poisson gap={clock_gap:.9f}")
print("PASS")
