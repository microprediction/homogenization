"""Exact equal-weight certificate for four distinct directed four-cycles."""

from collections import Counter
from fractions import Fraction
from itertools import combinations, islice, permutations
from math import comb

import numpy as np


N = 6
GROUND = N - 1
BATCH_SIZE = 20_000


def canonical(cycle):
    cycle = tuple(cycle)
    return min(cycle[index:] + cycle[:index] for index in range(len(cycle)))


def permutation_sign(permutation):
    inversions = sum(
        permutation[first] > permutation[second]
        for first in range(len(permutation))
        for second in range(first + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


GROUND_PERMUTATIONS = np.array(
    list(permutations(range(GROUND))), dtype=np.int64
)
GROUND_SIGNS = np.array([
    permutation_sign(permutation)
    for permutation in GROUND_PERMUTATIONS
], dtype=np.int64)


def integer_forms(cycle):
    """Return the integer matrices 2H and 2D for a unit cycle flow."""
    weights = np.zeros((N, N), dtype=np.int64)
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        weights[source, target] += 1
    laplacian = np.diag(np.sum(weights, axis=1)) - weights
    grounded = laplacian[:GROUND, :GROUND]
    return grounded + grounded.T, grounded - grounded.T


def batch_determinants(matrices):
    terms = np.ones(
        (len(matrices), len(GROUND_PERMUTATIONS)), dtype=np.int64
    )
    for row in range(GROUND):
        terms *= matrices[:, row, GROUND_PERMUTATIONS[:, row]]
    return terms @ GROUND_SIGNS


def coefficient_batch(indices):
    symmetric = np.sum(SYMMETRIC_FORMS[indices], axis=1)
    skew = np.sum(SKEW_FORMS[indices], axis=1)
    maximum_entry = max(
        int(np.max(np.abs(3 * symmetric - skew))),
        int(np.max(np.abs(2 * symmetric - skew))),
        int(np.max(np.abs(symmetric - skew))),
    )
    assert len(GROUND_PERMUTATIONS) * maximum_entry ** GROUND < 2 ** 63
    value_one, value_two, value_three = [
        batch_determinants(scale * symmetric - skew)
        for scale in (1, 2, 3)
    ]
    numerator_five = value_three - 4 * value_two + 5 * value_one
    numerator_three = value_two - 2 * value_one
    assert np.all(numerator_five % 120 == 0)
    assert np.all(numerator_three % 6 == 0)
    coefficient_five = numerator_five // 120
    coefficient_three = numerator_three // 6 - 5 * coefficient_five
    coefficient_one = value_one - coefficient_five - coefficient_three
    return coefficient_five, coefficient_three, coefficient_one


def combination_batches(size, choose, batch_size):
    iterator = combinations(range(size), choose)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield batch


FOUR_CYCLES = sorted({
    canonical(cycle)
    for support in combinations(range(N), 4)
    for cycle in (
        (min(support),) + ordering
        for ordering in permutations(
            tuple(state for state in support if state != min(support))
        )
    )
})
assert len(FOUR_CYCLES) == 90

FORMS = [integer_forms(cycle) for cycle in FOUR_CYCLES]
SYMMETRIC_FORMS = np.stack([forms[0] for forms in FORMS])
SKEW_FORMS = np.stack([forms[1] for forms in FORMS])

total_supports = connected_supports = 0
sum_barrier_supports = root_only_supports = violations = 0
equality_supports = set()
equality_coefficients = {}
equality_type_counts = Counter()

for batch in combination_batches(len(FOUR_CYCLES), 4, BATCH_SIZE):
    indices = np.array(batch, dtype=np.int64)
    a5, a3, a1 = coefficient_batch(indices)
    assert np.all(a5 >= 0)
    connected = a5 > 0
    # If A <= 0, the sum of the two squared frequencies is at most 2/3.
    # If B <= 0, their monic quadratic is nonpositive at 1/3.
    sum_barrier = 3 * a3 - 2 * a5
    root_barrier = a5 - 3 * a3 + 9 * a1
    total_supports += len(batch)
    connected_supports += int(np.sum(connected))
    sum_barrier_supports += int(np.sum(connected & (sum_barrier <= 0)))
    root_only_supports += int(np.sum(
        connected & (sum_barrier > 0) & (root_barrier <= 0)
    ))
    violations += int(np.sum(
        connected & (sum_barrier > 0) & (root_barrier > 0)
    ))
    equality = connected & (root_barrier == 0) & (sum_barrier >= 0)
    for position in np.flatnonzero(equality):
        support = tuple(FOUR_CYCLES[index] for index in batch[position])
        coefficients = (
            int(a5[position]), int(a3[position]), int(a1[position])
        )
        equality_supports.add(support)
        equality_coefficients[support] = coefficients
        equality_type_counts[coefficients] += 1

assert total_supports == comb(90, 4) == 2_555_190
assert connected_supports == 2_390_985
assert total_supports - connected_supports == 164_205
assert sum_barrier_supports == 2_144_115
assert root_only_supports == 246_870
assert sum_barrier_supports + root_only_supports == connected_supports
assert violations == 0
assert len(equality_supports) == 1_980
assert equality_type_counts == Counter({
    (1104, 800, 144): 720,
    (1008, 696, 120): 720,
    (1152, 1152, 256): 360,
    (864, 576, 96): 180,
})


def transform_cycle(cycle, relabeling, reverse):
    directed = tuple(reversed(cycle)) if reverse else cycle
    return canonical(tuple(relabeling[state] for state in directed))


state_relabelings = tuple(permutations(range(N)))
unseen = set(equality_supports)
equality_orbits = []
while unseen:
    representative = min(unseen)
    orbit = {
        tuple(sorted(
            transform_cycle(cycle, relabeling, reverse)
            for cycle in representative
        ))
        for relabeling in state_relabelings
        for reverse in (False, True)
    }
    assert orbit <= equality_supports
    equality_orbits.append((
        representative, len(orbit), equality_coefficients[representative]
    ))
    unseen -= orbit

expected_orbits = (
    (
        (
            (0, 1, 2, 3), (0, 1, 2, 4),
            (0, 3, 5, 1), (0, 3, 5, 4),
        ),
        720,
        (1104, 800, 144),
    ),
    (
        (
            (0, 1, 2, 3), (0, 1, 2, 4),
            (0, 3, 5, 4), (2, 3, 5, 4),
        ),
        360,
        (1152, 1152, 256),
    ),
    (
        (
            (0, 1, 2, 3), (0, 1, 3, 2),
            (2, 3, 4, 5), (2, 4, 5, 3),
        ),
        180,
        (864, 576, 96),
    ),
    (
        (
            (0, 1, 2, 3), (0, 1, 3, 4),
            (0, 1, 4, 2), (2, 3, 4, 5),
        ),
        720,
        (1008, 696, 120),
    ),
)
assert tuple(equality_orbits) == expected_orbits
equality_other_roots = (
    Fraction(9, 23), Fraction(2, 3), Fraction(1, 3), Fraction(5, 14)
)
for record, other_root in zip(equality_orbits, equality_other_roots):
    a5, a3, a1 = record[2]
    assert Fraction(a3, a5) == Fraction(1, 3) + other_root
    assert Fraction(a1, a5) == Fraction(1, 3) * other_root

print(f"directed four-cycles: {len(FOUR_CYCLES)}")
print(f"four-distinct-cycle supports: {total_supports}")
print(f"connected supports: {connected_supports}")
print(f"disconnected supports: {total_supports - connected_supports}")
print(f"sum-barrier supports: {sum_barrier_supports}")
print(f"root-only supports: {root_only_supports}")
print(f"violations of lower squared frequency <= 1/3: {violations}")
print(f"equality supports: {len(equality_supports)}")
print(f"equality orbits: {len(equality_orbits)}")
print(f"equality orbit sizes: {tuple(record[1] for record in equality_orbits)}")
print(f"equality pencil coefficients: "
      f"{tuple(record[2] for record in equality_orbits)}")
print(f"other equality squared frequencies: {equality_other_roots}")
print("equal weights on four distinct directed four-cycles: sigma_2^2 <= 1/3")
print("PASS")
