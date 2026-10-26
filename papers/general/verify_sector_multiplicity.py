"""Checks for sharp-sector multiplicity under a bounded cycle length."""

from itertools import combinations, permutations

import numpy as np


LENGTH = 5
SECTOR = 1 / np.tan(np.pi / LENGTH)
ZERO_TOLERANCE = 2e-10
NORM_TOLERANCE = 2e-11


def cycle_forms(cycle, state_count):
    """Return the full symmetric and skew Laplacian forms of one cycle."""
    flow = np.zeros((state_count, state_count), dtype=float)
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        flow[source, target] += 1.0
    laplacian = np.diag(np.sum(flow, axis=1)) - flow
    return (laplacian + laplacian.T) / 2, (laplacian - laplacian.T) / 2


def nullity(matrix):
    eigenvalues = np.linalg.eigvalsh(matrix)
    assert eigenvalues[0] >= -ZERO_TOLERANCE
    return int(np.sum(eigenvalues < ZERO_TOLERANCE)), eigenvalues


def whitened_singular_values(h_form, d_form):
    """Ground the final state and compute the normalized skew singular values."""
    h_form = h_form[:-1, :-1]
    d_form = d_form[:-1, :-1]
    eigenvalues, eigenvectors = np.linalg.eigh(h_form)
    assert eigenvalues[0] > 1e-12
    inverse_root = (eigenvectors * eigenvalues ** -0.5) @ eigenvectors.T
    return np.linalg.svd(inverse_root @ d_form @ inverse_root,
                         compute_uv=False)


# Each single q-cycle has exactly two null directions in
# cot(pi/q) H - iD: constants and its oriented first Fourier mode.
single_lengths = tuple(range(3, 10))
for length in single_lengths:
    h_form, d_form = cycle_forms(tuple(range(length)), length)
    sector = 1 / np.tan(np.pi / length)
    kernel_dimension, _ = nullity(sector * h_form - 1j * d_form)
    assert kernel_dimension == 2

# Fix one five-cycle and enumerate every relative second five-cycle by overlap.
# One shared vertex leaves two independent extremal amplitudes; two or more
# shared vertices determine the second affine Fourier mode from the first.
pair_counts = {}
maximum_nullities = {}
minimum_positive_eigenvalue = np.inf
for overlap in range(1, LENGTH + 1):
    state_count = 2 * LENGTH - overlap
    first_cycle = tuple(range(LENGTH))
    first_h, first_d = cycle_forms(first_cycle, state_count)
    pair_count = 0
    maximum_nullity = 0
    for shared in combinations(range(LENGTH), overlap):
        second_support = shared + tuple(range(LENGTH, state_count))
        start = min(second_support)
        tail_states = tuple(state for state in second_support if state != start)
        for tail in permutations(tail_states):
            second_h, second_d = cycle_forms((start,) + tail, state_count)
            matrix = SECTOR * (first_h + second_h) - 1j * (first_d + second_d)
            kernel_dimension, eigenvalues = nullity(matrix)
            positive = eigenvalues[eigenvalues >= ZERO_TOLERANCE]
            minimum_positive_eigenvalue = min(
                minimum_positive_eigenvalue, float(positive[0])
            )
            maximum_nullity = max(maximum_nullity, kernel_dimension)
            pair_count += 1
    pair_counts[overlap] = pair_count
    maximum_nullities[overlap] = maximum_nullity

assert pair_counts == {1: 120, 2: 240, 3: 240, 4: 120, 5: 24}
assert maximum_nullities == {1: 3, 2: 2, 3: 2, 4: 2, 5: 2}
assert minimum_positive_eigenvalue > 0.03

# The one-vertex overlap is the sharp nine-state shared-hub construction.
state_count = 9
h_form = np.zeros((state_count, state_count), dtype=float)
d_form = np.zeros((state_count, state_count), dtype=float)
for cycle in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8)):
    cycle_h, cycle_d = cycle_forms(cycle, state_count)
    h_form += cycle_h
    d_form += cycle_d
singular_values = whitened_singular_values(h_form, d_form)
assert np.max(np.abs(singular_values[:4] - SECTOR)) <= NORM_TOLERANCE
assert singular_values[4] < SECTOR - 0.5

print("single-cycle equality kernels checked for lengths: 3 through 9")
print("five-cycle pairs by overlap 1 through 5: "
      + ", ".join(str(pair_counts[index]) for index in range(1, 6)))
print(f"total relative five-cycle pairs: {sum(pair_counts.values())}")
print("maximum Hermitian kernel dimensions by overlap: "
      + ", ".join(str(maximum_nullities[index]) for index in range(1, 6)))
print("smallest separated positive eigenvalue: "
      f"{minimum_positive_eigenvalue:.12f}")
print("shared-hub maximal singular values: "
      + ", ".join(f"{value:.12f}" for value in singular_values[:4]))
print("portable zero/norm tolerances: 2e-10, 2e-11")
print("PASS")
