"""Certificate for the directed-triangle strong-sector theorem."""

from fractions import Fraction
from itertools import combinations, permutations

import numpy as np


STATE_COUNT = 6
SECTOR_SQUARED = Fraction(1, 3)


def directed_triangle_forms(cycle, state_count=STATE_COUNT):
    """Return full integer symmetric/skew doubled Laplacian forms."""
    flow = np.zeros((state_count, state_count), dtype=np.int64)
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        flow[source, target] += 1
    laplacian = np.diag(np.sum(flow, axis=1)) - flow
    return laplacian + laplacian.T, laplacian - laplacian.T


def is_connected(cycles):
    adjacency = [set() for _ in range(STATE_COUNT)]
    for cycle in cycles:
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            adjacency[source].add(target)
            adjacency[target].add(source)
    seen = {0}
    stack = [0]
    while stack:
        state = stack.pop()
        for neighbor in adjacency[state] - seen:
            seen.add(neighbor)
            stack.append(neighbor)
    return len(seen) == STATE_COUNT


# Ground a canonical triangle.  The doubled forms avoid fractions:
# H2=2H and D2=2D.  Exact arithmetic verifies
# D2^T H2^{-1} D2 = H2/3 on its two-dimensional active quotient.
canonical_h2 = np.array([[2, -1], [-1, 2]], dtype=object)
canonical_d2 = np.array([[0, -1], [1, 0]], dtype=object)
canonical_h2_inverse = np.array([
    [Fraction(2, 3), Fraction(1, 3)],
    [Fraction(1, 3), Fraction(2, 3)],
], dtype=object)
assert np.array_equal(
    canonical_d2.T @ canonical_h2_inverse @ canonical_d2,
    canonical_h2 * SECTOR_SQUARED,
)

# Enumerate every labeled orientation, counted up to cyclic rotation, and
# independently check all equal-weight four-triangle supports on six states.
triangles = []
for support in combinations(range(STATE_COUNT), 3):
    start = min(support)
    tail_states = tuple(state for state in support if state != start)
    for tail in permutations(tail_states):
        triangles.append((start,) + tail)
assert len(triangles) == 40

triangle_forms = [directed_triangle_forms(cycle) for cycle in triangles]
quadruple_count = 0
connected_quadruple_count = 0
largest_equal_weight_frequency = 0.0
for indices in combinations(range(len(triangles)), 4):
    quadruple_count += 1
    cycles = [triangles[index] for index in indices]
    if not is_connected(cycles):
        continue
    connected_quadruple_count += 1
    h2 = sum((triangle_forms[index][0] for index in indices))[:-1, :-1]
    d2 = sum((triangle_forms[index][1] for index in indices))[:-1, :-1]
    eigenvalues, eigenvectors = np.linalg.eigh(h2.astype(float))
    assert eigenvalues[0] > 1e-12
    inverse_root = (
        eigenvectors * eigenvalues ** -0.5
    ) @ eigenvectors.T
    frequencies = np.linalg.svd(
        inverse_root @ d2.astype(float) @ inverse_root,
        compute_uv=False,
    )
    largest_equal_weight_frequency = max(
        largest_equal_weight_frequency, float(frequencies[0])
    )

assert quadruple_count == 91390
assert connected_quadruple_count == 63360
assert largest_equal_weight_frequency <= 1 / np.sqrt(3) + 2e-11

# Arbitrary positive triangle weights and arbitrary reversible edges test the
# addition and contraction steps independently of the equal-weight sweep.
rng = np.random.default_rng(20261022)
random_cases = 500
largest_random_frequency = 0.0
for _ in range(random_cases):
    selected = rng.choice(len(triangles), size=rng.integers(1, 16), replace=True)
    weights = np.exp(rng.uniform(-4.0, 4.0, size=len(selected)))
    h2 = np.zeros((STATE_COUNT, STATE_COUNT), dtype=float)
    d2 = np.zeros((STATE_COUNT, STATE_COUNT), dtype=float)
    for index, weight in zip(selected, weights):
        h2 += weight * triangle_forms[index][0]
        d2 += weight * triangle_forms[index][1]
    for first, second in combinations(range(STATE_COUNT), 2):
        weight = np.exp(rng.uniform(-5.0, 3.0))
        edge = np.zeros_like(h2)
        edge[first, first] = edge[second, second] = 2 * weight
        edge[first, second] = edge[second, first] = -2 * weight
        h2 += edge
    h2 = h2[:-1, :-1]
    d2 = d2[:-1, :-1]
    eigenvalues, eigenvectors = np.linalg.eigh(h2)
    assert eigenvalues[0] > 1e-12
    inverse_root = (
        eigenvectors * eigenvalues ** -0.5
    ) @ eigenvectors.T
    frequencies = np.linalg.svd(
        inverse_root @ d2 @ inverse_root,
        compute_uv=False,
    )
    largest_random_frequency = max(
        largest_random_frequency, float(frequencies[0])
    )

assert largest_random_frequency <= 1 / np.sqrt(3) + 2e-11

print(f"labeled directed triangles on six states: {len(triangles)}")
print(f"four-triangle supports: {quadruple_count}")
print(f"connected four-triangle supports: {connected_quadruple_count}")
print("exact one-triangle sector constant squared: 1/3")
print("largest equal-weight four-triangle frequency: "
      f"{largest_equal_weight_frequency:.12f}")
print(f"random weighted attachment cases: {random_cases}")
print("largest random weighted attachment frequency: "
      f"{largest_random_frequency:.12f}")
print("portable frequency ceiling: 1/sqrt(3) + 2e-11")
print("PASS")
