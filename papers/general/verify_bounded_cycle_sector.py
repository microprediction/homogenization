"""Numerical checks for the bounded-cycle strong-sector theorem."""

from itertools import combinations, permutations

import numpy as np


STATE_COUNT = 6
TOLERANCE = 2e-11


def directed_cycle_forms(cycle, state_count):
    """Return the symmetric and skew parts of a directed-cycle Laplacian."""
    flow = np.zeros((state_count, state_count), dtype=float)
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        flow[source, target] += 1.0
    laplacian = np.diag(np.sum(flow, axis=1)) - flow
    return (laplacian + laplacian.T) / 2, (laplacian - laplacian.T) / 2


def whitened_norm(h_form, d_form):
    """Ground the last state and return ||H^{-1/2} D H^{-1/2}||_2."""
    h_form = h_form[:-1, :-1]
    d_form = d_form[:-1, :-1]
    eigenvalues, eigenvectors = np.linalg.eigh(h_form)
    assert eigenvalues[0] > 1e-12
    inverse_root = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    return float(np.linalg.svd(inverse_root @ d_form @ inverse_root,
                               compute_uv=False)[0])


def simple_cycles(state_count):
    """List labeled simple directed cycles, modulo cyclic rotation."""
    cycles = []
    for length in range(2, state_count + 1):
        for support in combinations(range(state_count), length):
            start = min(support)
            tail_states = tuple(state for state in support if state != start)
            for tail in permutations(tail_states):
                cycles.append((start,) + tail)
    return cycles


# A single length-l cycle has Fourier ratios
# |sin(2*pi*k/l)|/(1-cos(2*pi*k/l)) = |cot(pi*k/l)|.
fourier_lengths = tuple(range(3, 10))
largest_fourier_error = 0.0
for length in fourier_lengths:
    h_form, d_form = directed_cycle_forms(tuple(range(length)), length)
    observed = whitened_norm(h_form, d_form)
    predicted = 1 / np.tan(np.pi / length)
    largest_fourier_error = max(largest_fourier_error,
                                abs(observed - predicted))
assert largest_fourier_error <= TOLERANCE

# Six-state counts provide a small independent check of the cycle generator.
cycles = simple_cycles(STATE_COUNT)
cycles_by_length = {
    length: [cycle for cycle in cycles if len(cycle) == length]
    for length in range(2, STATE_COUNT + 1)
}
cycle_counts = tuple(len(cycles_by_length[length])
                     for length in range(2, STATE_COUNT + 1))
assert cycle_counts == (15, 40, 90, 144, 120)
assert len(cycles) == 409
cycle_forms = {
    cycle: directed_cycle_forms(cycle, STATE_COUNT)
    for cycle in cycles
}

# Random nonnegative mixtures test addition closure for several length caps.
# Complete reversible flow makes the grounded symmetric form positive definite.
rng = np.random.default_rng(20261022)
random_cases = 600
largest_excess = -np.inf
for case in range(random_cases):
    length_cap = 3 + case % 4
    eligible = [cycle for cycle in cycles if len(cycle) <= length_cap]
    selected = rng.choice(len(eligible), size=rng.integers(1, 21), replace=True)
    weights = np.exp(rng.uniform(-3.0, 3.0, size=len(selected)))
    h_form = np.zeros((STATE_COUNT, STATE_COUNT), dtype=float)
    d_form = np.zeros((STATE_COUNT, STATE_COUNT), dtype=float)
    for index, weight in zip(selected, weights):
        cycle_h, cycle_d = cycle_forms[eligible[index]]
        h_form += weight * cycle_h
        d_form += weight * cycle_d
    for first, second in combinations(range(STATE_COUNT), 2):
        weight = np.exp(rng.uniform(-2.0, 2.0))
        h_form[first, first] += weight
        h_form[second, second] += weight
        h_form[first, second] -= weight
        h_form[second, first] -= weight
    observed = whitened_norm(h_form, d_form)
    ceiling = 1 / np.tan(np.pi / length_cap)
    largest_excess = max(largest_excess, observed - ceiling)

assert largest_excess <= TOLERANCE

print("single-cycle Fourier lengths checked: 3 through 9")
print(f"largest single-cycle Fourier discrepancy: {largest_fourier_error:.3e}")
print("six-state simple cycles by lengths 2 through 6: "
      + ", ".join(str(count) for count in cycle_counts))
print(f"total six-state simple cycles: {len(cycles)}")
print(f"random bounded-cycle mixtures with reversible flow: {random_cases}")
print(f"largest observed excess over cot(pi/L): {largest_excess:.3e}")
print("portable norm tolerance: 2e-11")
print("PASS")
