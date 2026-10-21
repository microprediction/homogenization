"""Certificate for every six-state two-Hamilton-cycle overlap cone."""

from fractions import Fraction
from itertools import permutations
import math
from math import comb

import numpy as np


N = 6
REFERENCE_CYCLE = tuple(range(N))
CYCLE_A = (0, 5, 2, 3, 4, 1)
CYCLE_B = (0, 3, 4, 5, 2, 1)


def sign(perm):
    inversions = sum(
        perm[i] > perm[j]
        for i in range(len(perm))
        for j in range(i + 1, len(perm))
    )
    return -1 if inversions % 2 else 1


def poly_add(left, right):
    size = max(len(left), len(right))
    out = [Fraction(0) for _ in range(size)]
    for i, value in enumerate(left):
        out[i] += value
    for i, value in enumerate(right):
        out[i] += value
    return out


def poly_mul(left, right):
    out = [Fraction(0) for _ in range(len(left) + len(right) - 1)]
    for i, value in enumerate(left):
        for j, other in enumerate(right):
            out[i + j] += value * other
    return out


def determinant_polynomial(matrix):
    size = len(matrix)
    out = [Fraction(0)]
    for perm in permutations(range(size)):
        term = [Fraction(sign(perm))]
        for row, column in enumerate(perm):
            term = poly_mul(term, matrix[row][column])
        out = poly_add(out, term)
    return out + [Fraction(0)] * (size + 1 - len(out))


def sparse_add(left, right):
    out = dict(left)
    for degree, value in right.items():
        out[degree] = out.get(degree, Fraction(0)) + value
    return {degree: value for degree, value in out.items() if value}


def sparse_scale(poly, scalar):
    return {degree: scalar * value for degree, value in poly.items()
            if scalar * value}


def sparse_mul(left, right):
    out = {}
    for (t_left, r_left), value_left in left.items():
        for (t_right, r_right), value_right in right.items():
            degree = (t_left + t_right, r_left + r_right)
            out[degree] = out.get(degree, Fraction(0)) + value_left * value_right
    return {degree: value for degree, value in out.items() if value}


def sparse_determinant(matrix):
    out = {}
    for perm in permutations(range(len(matrix))):
        term = {(0, 0): Fraction(sign(perm))}
        for row, column in enumerate(perm):
            term = sparse_mul(term, matrix[row][column])
        out = sparse_add(out, term)
    return out


def symbolic_pencil_coefficients(other_cycle):
    """Return det(H), and the t^3 and t coefficients, as polynomials in r."""
    weights = [[{} for _ in range(N)] for _ in range(N)]
    for cycle, r_degree in ((REFERENCE_CYCLE, 0), (other_cycle, 1)):
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            weights[source][target] = sparse_add(
                weights[source][target], {(0, r_degree): Fraction(1)}
            )
    laplacian = [[{} for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                for entry in weights[i]:
                    laplacian[i][j] = sparse_add(laplacian[i][j], entry)
            else:
                laplacian[i][j] = sparse_scale(weights[i][j], -1)
    grounded = [row[:5] for row in laplacian[:5]]
    half = Fraction(1, 2)
    symmetric = [[sparse_scale(sparse_add(grounded[i][j], grounded[j][i]), half)
                  for j in range(5)] for i in range(5)]
    skew = [[sparse_scale(sparse_add(grounded[i][j],
                                     sparse_scale(grounded[j][i], -1)), half)
             for j in range(5)] for i in range(5)]
    pencil = [[sparse_add({(t_degree + 1, r_degree): value
                           for (t_degree, r_degree), value in symmetric[i][j].items()},
                          sparse_scale(skew[i][j], -1))
               for j in range(5)] for i in range(5)]
    numerator = sparse_determinant(pencil)
    det_h = sparse_determinant(symmetric)
    coefficient = lambda poly, t_degree: tuple(
        poly.get((t_degree, r_degree), Fraction(0)) for r_degree in range(6)
    )
    denominator = coefficient(det_h, 0)
    assert coefficient(numerator, 5) == denominator
    return denominator, coefficient(numerator, 3), coefficient(numerator, 1)


def compactify(poly):
    """Power coefficients of (1-s)^5 poly(s/(1-s)) on 0 <= s <= 1."""
    out = [Fraction(0) for _ in range(6)]
    for degree, value in enumerate(poly):
        for extra in range(6 - degree):
            out[degree + extra] += value * comb(5 - degree, extra) * (-1) ** extra
    return out


def bernstein_max(poly, left, right):
    """Exact upper bound from degree-five Bernstein coefficients on an interval."""
    shifted = [Fraction(0) for _ in range(6)]
    width = right - left
    for degree, value in enumerate(poly):
        for power in range(degree + 1):
            shifted[power] += (value * comb(degree, power)
                               * left ** (degree - power) * width ** power)
    coefficients = [
        sum(shifted[power] * Fraction(comb(index, power), comb(5, power))
            for power in range(index + 1))
        for index in range(6)
    ]
    return max(coefficients)


def exact_cover(sum_test, root_test, left=Fraction(0), right=Fraction(1), depth=0):
    """Cover by intervals where S <= 6/5 or f(3/5) <= 0."""
    if bernstein_max(sum_test, left, right) <= 0:
        return 1, depth
    if bernstein_max(root_test, left, right) <= 0:
        return 1, depth
    assert depth < 8
    midpoint = (left + right) / 2
    left_count, left_depth = exact_cover(sum_test, root_test, left, midpoint, depth + 1)
    right_count, right_depth = exact_cover(sum_test, root_test, midpoint, right, depth + 1)
    return left_count + right_count, max(left_depth, right_depth)


def flow(a, b, dtype=object):
    out = np.zeros((N, N), dtype=dtype)
    for weight, cycle in ((a, CYCLE_A), (b, CYCLE_B)):
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            out[source, target] += weight
    return out


def grounded_forms(a, b):
    stationary_flow = flow(a, b)
    laplacian = np.diag(np.sum(stationary_flow, axis=1)) - stationary_flow
    grounded = laplacian[:5, :5]
    half = Fraction(1, 2)
    symmetric = np.empty((5, 5), dtype=object)
    skew = np.empty((5, 5), dtype=object)
    for i in range(5):
        for j in range(5):
            symmetric[i, j] = half * (grounded[i, j] + grounded[j, i])
            skew[i, j] = half * (grounded[i, j] - grounded[j, i])
    return symmetric, skew


def exact_coefficients(a, b):
    symmetric, skew = grounded_forms(a, b)
    pencil = [
        [[-skew[i, j], symmetric[i, j]] for j in range(5)]
        for i in range(5)
    ]
    return determinant_polynomial(pencil)


def expected_coefficients(a, b):
    denominator = a * a + 3 * a * b + b * b
    det_h = Fraction(3, 16) * (a + b) * denominator * denominator
    sum_sq = (
        2
        * (
            5 * a**4
            + 12 * a**3 * b
            + 11 * a * a * b * b
            + 12 * a * b**3
            + 5 * b**4
        )
        / (3 * denominator * denominator)
    )
    product_sq = ((a * a + a * b + b * b) / denominator) ** 2
    return [Fraction(0), det_h * product_sq, Fraction(0), det_h * sum_sq,
            Fraction(0), det_h]


def frequencies(a, b):
    x = a * b / (a + b) ** 2
    sum_sq = 2 * (5 - 8 * x - 3 * x * x) / (3 * (1 + x) ** 2)
    product_sq = ((1 - x) / (1 + x)) ** 2
    discriminant = max(0.0, sum_sq * sum_sq - 4 * product_sq)
    roots = ((sum_sq + math.sqrt(discriminant)) / 2,
             (sum_sq - math.sqrt(discriminant)) / 2)
    return math.sqrt(roots[0]), math.sqrt(roots[1])


exact_cases = [
    (Fraction(1), Fraction(0)),
    (Fraction(0), Fraction(1)),
    (Fraction(1), Fraction(1)),
    (Fraction(1), Fraction(2)),
    (Fraction(2), Fraction(5)),
    (Fraction(7), Fraction(3)),
]
for exact_a, exact_b in exact_cases:
    assert exact_coefficients(exact_a, exact_b) == expected_coefficients(exact_a, exact_b)

equal = expected_coefficients(Fraction(1), Fraction(1))
det_equal = equal[5]
assert [value / det_equal for value in equal] == [
    Fraction(0), Fraction(9, 25), Fraction(0), Fraction(6, 5),
    Fraction(0), Fraction(1)
]

# Every ordered pair of Hamilton cycles is a relabeling of REFERENCE_CYCLE
# and one of these 120 cycles.  Write r=s/(1-s), compactifying r in [0,infinity]
# to s in [0,1].  If S <= 6/5, the smaller squared frequency is at most 3/5.
# Otherwise f(3/5) <= 0 puts 3/5 between the two squared frequencies.
relative_cycles = [(0,) + tail for tail in permutations(range(1, N))]
pencil_types = set()
largest_cover = 0
deepest_cover = 0
for other_cycle in relative_cycles:
    denominator, sum_numerator, product_numerator = symbolic_pencil_coefficients(
        other_cycle
    )
    pencil_types.add((denominator, sum_numerator, product_numerator))
    sum_test = compactify([
        sum_numerator[index] - Fraction(6, 5) * denominator[index]
        for index in range(6)
    ])
    root_test = compactify([
        product_numerator[index]
        - Fraction(3, 5) * sum_numerator[index]
        + Fraction(9, 25) * denominator[index]
        for index in range(6)
    ])
    cover_size, cover_depth = exact_cover(sum_test, root_test)
    largest_cover = max(largest_cover, cover_size)
    deepest_cover = max(deepest_cover, cover_depth)

assert len(relative_cycles) == 120
assert len(pencil_types) == 20
assert largest_cover == 6
assert deepest_cover == 4

rng = np.random.default_rng(20261021)
max_spectrum_error = 0.0
max_stationarity_error = 0.0
largest_second = 0.0
for _ in range(300):
    a_float, b_float = np.exp(rng.uniform(-6.0, 6.0, 2))
    stationary_flow = np.asarray(flow(a_float, b_float, dtype=float), dtype=float)
    pi = rng.dirichlet(np.ones(N))
    generator = stationary_flow / pi[:, None]
    generator[np.diag_indices(N)] = 0.0
    generator[np.diag_indices(N)] = -np.sum(generator, axis=1)
    max_stationarity_error = max(
        max_stationarity_error,
        float(np.max(np.abs(pi @ generator))),
        float(np.max(np.abs(np.sum(generator, axis=1)))),
    )

    laplacian = np.diag(np.sum(stationary_flow, axis=1)) - stationary_flow
    grounded = laplacian[:5, :5]
    symmetric = (grounded + grounded.T) / 2
    skew = (grounded - grounded.T) / 2
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    assert eigenvalues[0] > 0
    whitening = (vectors / np.sqrt(eigenvalues)) @ vectors.T
    singular = np.linalg.svd(whitening @ skew @ whitening, compute_uv=False)
    observed = (singular[0], singular[2])
    predicted = frequencies(a_float, b_float)
    max_spectrum_error = max(
        max_spectrum_error,
        abs(observed[0] - predicted[0]),
        abs(observed[1] - predicted[1]),
    )
    largest_second = max(largest_second, observed[1])

target = math.sqrt(1 + 2 / math.sqrt(5))
sharp_second = math.sqrt(Fraction(3, 5))
assert sharp_second < target
assert max_spectrum_error < 2e-11
assert max_stationarity_error < 2e-10
assert largest_second <= sharp_second + 2e-11

print(f"exact pencil identities: {len(exact_cases)} rational rate pairs")
print("equal-weight normalized pencil: t (t^2 + 3/5)^2")
print(f"relative Hamilton cycles classified exactly: {len(relative_cycles)}")
print(f"distinct exact pencils: {len(pencil_types)}")
print(f"largest Bernstein cover: {largest_cover} intervals at depth {deepest_cover}")
print("random stationary-law cases: 300")
print(f"maximum spectrum/formula discrepancy: {max_spectrum_error:.3e}")
print(f"maximum stationarity residual: {max_stationarity_error:.3e}")
print("portable assertion ceilings: spectrum 2e-11; stationarity 2e-10")
print(f"largest sampled second frequency: {largest_second:.12f}")
print(f"sharp two-Hamilton-cycle bound sqrt(3/5): {sharp_second:.12f}")
print(f"target cot(pi/5): {target:.12f}")
print("PASS")
