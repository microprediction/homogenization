"""Exact local response of an eight-cycle to one added directed cycle.

For unit flow on the standard directed eight-cycle, let s(t) be the middle
squared intrinsic frequency after adding weight t on another simple directed
cycle.  The base value s(0)=1 is simple.  This certificate recovers the first
three Taylor derivatives from exact determinant expansions, then exhausts every
simple directed cycle on the same eight labeled states.
"""

from fractions import Fraction
from itertools import combinations, permutations


ZERO = Fraction(0)
ONE = Fraction(1)
STATE_COUNT = 8
GROUND_SIZE = 7


def cycle_forms(cycle):
    flow = [[ZERO for _ in range(STATE_COUNT)] for _ in range(STATE_COUNT)]
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        flow[source][target] += ONE
    laplacian = [
        [
            (sum(flow[i], ZERO) if i == j else ZERO) - flow[i][j]
            for j in range(STATE_COUNT)
        ]
        for i in range(STATE_COUNT)
    ]
    symmetric = [
        [
            (laplacian[i][j] + laplacian[j][i]) / 2
            for j in range(1, STATE_COUNT)
        ]
        for i in range(1, STATE_COUNT)
    ]
    skew = [
        [
            (laplacian[i][j] - laplacian[j][i]) / 2
            for j in range(1, STATE_COUNT)
        ]
        for i in range(1, STATE_COUNT)
    ]
    return symmetric, skew


def inverse_and_determinant(matrix):
    size = len(matrix)
    augmented = [
        list(row) + [Fraction(i == j) for j in range(size)]
        for i, row in enumerate(matrix)
    ]
    determinant = ONE
    for column in range(size):
        pivot = next(
            row for row in range(column, size) if augmented[row][column]
        )
        if pivot != column:
            augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
            determinant *= -1
        value = augmented[column][column]
        determinant *= value
        augmented[column] = [entry / value for entry in augmented[column]]
        for row in range(size):
            if row != column and augmented[row][column]:
                value = augmented[row][column]
                augmented[row] = [
                    entry - value * pivot_entry
                    for entry, pivot_entry in zip(
                        augmented[row], augmented[column]
                    )
                ]
    return [row[size:] for row in augmented], determinant


def matrix_product(left, right):
    return [
        [
            sum(
                (left[i][k] * right[k][j] for k in range(len(right))),
                ZERO,
            )
            for j in range(len(right[0]))
        ]
        for i in range(len(left))
    ]


# Interpolate a cubic in x=u^2 from u=0,1,2,3.
vandermonde = [[Fraction(x) ** power for power in range(4)] for x in (0, 1, 4, 9)]
inverse_vandermonde, _ = inverse_and_determinant(vandermonde)


def interpolate_even(values):
    return [
        sum(
            (inverse_vandermonde[power][node] * values[node] for node in range(4)),
            ZERO,
        )
        for power in range(4)
    ]


base_symmetric, base_skew = cycle_forms(tuple(range(STATE_COUNT)))
base_data = []
for u in range(4):
    base_matrix = [
        [
            base_symmetric[i][j] + u * base_skew[i][j]
            for j in range(GROUND_SIZE)
        ]
        for i in range(GROUND_SIZE)
    ]
    inverse, determinant = inverse_and_determinant(base_matrix)
    base_data.append((u, inverse, determinant))

base_coefficients = interpolate_even(
    [determinant for _, _, determinant in base_data]
)
assert base_coefficients == [
    Fraction(1, 16), Fraction(7, 16), Fraction(7, 16), Fraction(1, 16)
]


def frequency_derivatives(cycle):
    """Return exact s'(0), s''(0), s'''(0) for the middle frequency."""

    symmetric, skew = cycle_forms(cycle)
    first_determinant_coefficients = []
    second_determinant_coefficients = []
    third_determinant_coefficients = []
    for u, inverse, base_determinant in base_data:
        perturbation = [
            [
                symmetric[i][j] + u * skew[i][j]
                for j in range(GROUND_SIZE)
            ]
            for i in range(GROUND_SIZE)
        ]
        product = matrix_product(inverse, perturbation)
        trace = sum((product[i][i] for i in range(GROUND_SIZE)), ZERO)
        trace_square = sum(
            (
                product[i][j] * product[j][i]
                for i in range(GROUND_SIZE)
                for j in range(GROUND_SIZE)
            ),
            ZERO,
        )
        product_square = matrix_product(product, product)
        trace_cube = sum(
            (
                product_square[i][j] * product[j][i]
                for i in range(GROUND_SIZE)
                for j in range(GROUND_SIZE)
            ),
            ZERO,
        )
        first_determinant_coefficients.append(base_determinant * trace)
        second_determinant_coefficients.append(
            base_determinant * (trace * trace - trace_square) / 2
        )
        third_determinant_coefficients.append(
            base_determinant
            * (trace ** 3 - 3 * trace * trace_square + 2 * trace_cube) / 6
        )

    first = interpolate_even(first_determinant_coefficients)
    second = interpolate_even(second_determinant_coefficients)
    third = interpolate_even(third_determinant_coefficients)

    # Q(t,s)=a0(t)s^3-a1(t)s^2+a2(t)s-a3(t).  The determinant trace
    # identity gives coefficients through t^3; implicit differentiation at
    # (t,s)=(0,1) gives the three derivatives below.
    q_s = 3 * base_coefficients[0] - 2 * base_coefficients[1] + base_coefficients[2]
    q_ss = 6 * base_coefficients[0] - 2 * base_coefficients[1]
    q_t = first[0] - first[1] + first[2] - first[3]
    q_ts = 3 * first[0] - 2 * first[1] + first[2]
    q_tt = 2 * (second[0] - second[1] + second[2] - second[3])
    q_ttt = 6 * (third[0] - third[1] + third[2] - third[3])
    first_derivative = -q_t / q_s
    second_derivative = -(
        q_tt + 2 * q_ts * first_derivative
        + q_ss * first_derivative * first_derivative
    ) / q_s
    q_sss = 6 * base_coefficients[0]
    q_tss = 6 * first[0] - 2 * first[1]
    q_tts = 2 * (3 * second[0] - 2 * second[1] + second[2])
    third_derivative = -(
        q_ttt + 3 * q_tts * first_derivative
        + 3 * q_tss * first_derivative ** 2
        + q_sss * first_derivative ** 3
        + 3 * (q_ts + q_ss * first_derivative) * second_derivative
    ) / q_s
    return first_derivative, second_derivative, third_derivative


def normalize_cycle(cycle):
    minimum = min(cycle)
    index = cycle.index(minimum)
    return cycle[index:] + cycle[:index]


def rotation_orbit(cycle):
    return {
        normalize_cycle(tuple((vertex + shift) % STATE_COUNT for vertex in cycle))
        for shift in range(STATE_COUNT)
    }


cycles = set()
for length in range(2, STATE_COUNT + 1):
    for support in combinations(range(STATE_COUNT), length):
        start = min(support)
        for tail in permutations(vertex for vertex in support if vertex != start):
            cycles.add((start,) + tail)

representatives = {min(rotation_orbit(cycle)) for cycle in cycles}
covered = set()
orbit_sizes = {}
first_distribution = {}
neutral_curvature_distribution = {}
neutral_curvature_lengths = {}
maximal_curvature_thirds = {}
maximal_curvature_length_thirds = {}
for cycle in representatives:
    orbit = rotation_orbit(cycle)
    assert orbit <= cycles
    covered.update(orbit)
    multiplicity = len(orbit)
    orbit_sizes[multiplicity] = orbit_sizes.get(multiplicity, 0) + 1
    first, second, third = frequency_derivatives(cycle)
    first_distribution[first] = first_distribution.get(first, 0) + multiplicity
    if first == 0:
        neutral_curvature_distribution[second] = (
            neutral_curvature_distribution.get(second, 0) + multiplicity
        )
        if second == 2:
            neutral_curvature_lengths[len(cycle)] = (
                neutral_curvature_lengths.get(len(cycle), 0) + multiplicity
            )
            maximal_curvature_thirds[third] = (
                maximal_curvature_thirds.get(third, 0) + multiplicity
            )
            key = (len(cycle), third)
            maximal_curvature_length_thirds[key] = (
                maximal_curvature_length_thirds.get(key, 0) + multiplicity
            )

assert len(cycles) == 16064
assert len(representatives) == 2025
assert covered == cycles
assert orbit_sizes == {1: 4, 2: 4, 4: 21, 8: 1996}
assert first_distribution == {
    Fraction(0): 268,
    Fraction(-1, 2): 704,
    Fraction(-1): 1880,
    Fraction(-3, 2): 3200,
    Fraction(-2): 4724,
    Fraction(-5, 2): 2752,
    Fraction(-3): 1936,
    Fraction(-7, 2): 512,
    Fraction(-4): 88,
}
assert neutral_curvature_distribution == {
    Fraction(2): 24,
    Fraction(1): 32,
    Fraction(0): 156,
    Fraction(-1): 32,
    Fraction(-2): 24,
}
assert neutral_curvature_lengths == {7: 16, 8: 8}
assert maximal_curvature_thirds == {Fraction(-12): 16, Fraction(-18): 8}
assert maximal_curvature_length_thirds == {
    (7, Fraction(-12)): 16,
    (8, Fraction(-18)): 8,
}
assert frequency_derivatives((1, 2, 6, 7, 3, 4, 5)) == (
    ZERO, Fraction(2), Fraction(-12)
)
print("maximal-curvature third-derivative distribution:", maximal_curvature_thirds)
print("strong seven-cycle derivatives:", frequency_derivatives((1, 2, 6, 7, 3, 4, 5)))

print("simple directed cycles checked:", len(cycles))
print("rotation-orbit representatives:", len(representatives))
print("first-derivative distribution:", first_distribution)
print("first-order-neutral cycles:", first_distribution[ZERO])
print("neutral second-derivative distribution:", neutral_curvature_distribution)
print("curvature-two cycles by length:", neutral_curvature_lengths)
print("PASS")
