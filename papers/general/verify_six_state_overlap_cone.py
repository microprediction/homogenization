"""Certificate for sparse six-state cycle mixtures and their sharp skew bound."""

from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations, product
import math
from math import comb

import numpy as np


N = 6
GROUND = N - 1
CYCLE_A = (0, 5, 2, 3, 4, 1)
CYCLE_B = (0, 3, 4, 5, 2, 1)
THREE_CYCLES = (
    (0, 1, 2, 3),
    (0, 1, 4, 5),
    (2, 3, 4, 5),
)
HAMILTON_TRIPLE_ORBIT_REPRESENTATIVES = (
    ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4), (0, 1, 2, 4, 5, 3)),
    ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4), (0, 1, 2, 4, 3, 5)),
    ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4), (0, 1, 3, 2, 4, 5)),
    ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4), (0, 1, 4, 2, 3, 5)),
    ((0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3), (0, 1, 4, 2, 3, 5)),
    ((0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 5, 4), (0, 2, 1, 3, 4, 5)),
    ((0, 1, 2, 3, 4, 5), (0, 1, 2, 4, 5, 3), (0, 1, 2, 5, 3, 4)),
)


def sign(perm):
    inversions = sum(
        perm[i] > perm[j]
        for i in range(len(perm))
        for j in range(i + 1, len(perm))
    )
    return -1 if inversions % 2 else 1


GROUND_PERMUTATIONS = np.array(
    list(permutations(range(GROUND))), dtype=np.int64
)
GROUND_SIGNS = np.array(
    [sign(tuple(perm)) for perm in GROUND_PERMUTATIONS], dtype=np.int64
)


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


def multivariate_mul(left, right):
    """Multiply sparse polynomials with arbitrary exponent tuples."""
    out = {}
    for degree_left, value_left in left.items():
        for degree_right, value_right in right.items():
            degree = tuple(
                left_power + right_power
                for left_power, right_power in zip(degree_left, degree_right)
            )
            out[degree] = out.get(degree, Fraction(0)) + value_left * value_right
    return {degree: value for degree, value in out.items() if value}


def multivariate_power(poly, power):
    out = {(0,) * len(next(iter(poly))): Fraction(1)}
    for _ in range(power):
        out = multivariate_mul(out, poly)
    return out


def multivariate_compose(poly, substitutions):
    """Compose a sparse polynomial with sparse polynomial substitutions."""
    variables = len(next(iter(substitutions[0])))
    out = {}
    for exponent, coefficient in poly.items():
        term = {(0,) * variables: coefficient}
        for substitution, power in zip(substitutions, exponent):
            term = multivariate_mul(
                term, multivariate_power(substitution, power)
            )
        out = sparse_add(out, term)
    return out


def tensor_bernstein_controls(poly):
    """Exact tensor-product Bernstein controls on the unit box."""
    variables = len(next(iter(poly)))
    degrees = tuple(
        max(exponent[variable] for exponent in poly)
        for variable in range(variables)
    )
    controls = []
    for index in product(*(range(degree + 1) for degree in degrees)):
        control = Fraction(0)
        for exponent, coefficient in poly.items():
            if all(power <= bound for power, bound in zip(exponent, index)):
                factor = Fraction(1)
                for bound, power, degree in zip(index, exponent, degrees):
                    factor *= Fraction(comb(bound, power), comb(degree, power))
                control += coefficient * factor
        controls.append(control)
    return controls


def extremal_pair_polynomial(coefficients, exponents, first, second):
    """Substitute pair weights 1+u, 1-u and third weight v."""
    third = 3 - first - second
    out = {}
    for coefficient, exponent in zip(coefficients, exponents):
        first_power = exponent[first]
        second_power = exponent[second]
        third_power = exponent[third]
        for left_power in range(first_power + 1):
            for right_power in range(second_power + 1):
                power = (left_power + right_power, third_power)
                value = (
                    Fraction(int(coefficient))
                    * comb(first_power, left_power)
                    * comb(second_power, right_power)
                    * (-1) ** right_power
                )
                out[power] = out.get(power, Fraction(0)) + value
    return {power: value for power, value in out.items() if value}


def reverse_first_variable(poly):
    """Apply u -> -u to a bivariate sparse polynomial."""
    return {
        exponent: (-1) ** exponent[0] * coefficient
        for exponent, coefficient in poly.items()
    }


def multivariate_determinant(matrix):
    out = {}
    variables = len(next(iter(matrix[0][0])))
    for perm in permutations(range(len(matrix))):
        term = {(0,) * variables: Fraction(sign(perm))}
        for row, column in enumerate(perm):
            term = multivariate_mul(term, matrix[row][column])
        out = sparse_add(out, term)
    return out


def sparse_determinant(matrix):
    out = {}
    for perm in permutations(range(len(matrix))):
        term = {(0, 0): Fraction(sign(perm))}
        for row, column in enumerate(perm):
            term = sparse_mul(term, matrix[row][column])
        out = sparse_add(out, term)
    return out


def symbolic_pencil_coefficients_n(first_cycle, other_cycle, state_count):
    """Return the top three pencil coefficients as polynomials in r."""
    grounded_size = state_count - 1
    weights = [[{} for _ in range(state_count)] for _ in range(state_count)]
    for cycle, r_degree in ((first_cycle, 0), (other_cycle, 1)):
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            weights[source][target] = sparse_add(
                weights[source][target], {(0, r_degree): Fraction(1)}
            )
    laplacian = [[{} for _ in range(state_count)] for _ in range(state_count)]
    for i in range(state_count):
        for j in range(state_count):
            if i == j:
                for entry in weights[i]:
                    laplacian[i][j] = sparse_add(laplacian[i][j], entry)
            else:
                laplacian[i][j] = sparse_scale(weights[i][j], -1)
    grounded = [row[:grounded_size] for row in laplacian[:grounded_size]]
    half = Fraction(1, 2)
    symmetric = [[sparse_scale(sparse_add(grounded[i][j], grounded[j][i]), half)
                  for j in range(grounded_size)] for i in range(grounded_size)]
    skew = [[sparse_scale(sparse_add(grounded[i][j],
                                     sparse_scale(grounded[j][i], -1)), half)
             for j in range(grounded_size)] for i in range(grounded_size)]
    pencil = [[sparse_add({(t_degree + 1, r_degree): value
                           for (t_degree, r_degree), value in symmetric[i][j].items()},
                          sparse_scale(skew[i][j], -1))
               for j in range(grounded_size)] for i in range(grounded_size)]
    numerator = sparse_determinant(pencil)
    det_h = sparse_determinant(symmetric)
    coefficient = lambda poly, t_degree: tuple(
        poly.get((t_degree, r_degree), Fraction(0))
        for r_degree in range(state_count)
    )
    denominator = coefficient(det_h, 0)
    assert coefficient(numerator, grounded_size) == denominator
    return (
        denominator,
        coefficient(numerator, grounded_size - 2),
        coefficient(numerator, grounded_size - 4),
    )


def symbolic_pencil_coefficients(first_cycle, other_cycle):
    """Six-state wrapper: det(H), t^3 and t coefficients in r."""
    return symbolic_pencil_coefficients_n(first_cycle, other_cycle, N)


def symbolic_three_cycle_coefficients(cycles=THREE_CYCLES):
    """Exact coefficient polynomials for weights 1, r, s on three cycles."""
    weights = [[{} for _ in range(N)] for _ in range(N)]
    weight_polynomials = (
        {(0, 0, 0): Fraction(1)},
        {(0, 1, 0): Fraction(1)},
        {(0, 0, 1): Fraction(1)},
    )
    for cycle, weight in zip(cycles, weight_polynomials):
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            weights[source][target] = sparse_add(weights[source][target], weight)
    laplacian = [[{} for _ in range(N)] for _ in range(N)]
    for i in range(N):
        for j in range(N):
            if i == j:
                for entry in weights[i]:
                    laplacian[i][j] = sparse_add(laplacian[i][j], entry)
            else:
                laplacian[i][j] = sparse_scale(weights[i][j], -1)
    grounded = [row[:GROUND] for row in laplacian[:GROUND]]
    half = Fraction(1, 2)
    symmetric = [[
        sparse_scale(sparse_add(grounded[i][j], grounded[j][i]), half)
        for j in range(GROUND)
    ] for i in range(GROUND)]
    skew = [[
        sparse_scale(
            sparse_add(grounded[i][j], sparse_scale(grounded[j][i], -1)),
            half,
        )
        for j in range(GROUND)
    ] for i in range(GROUND)]
    pencil = [[
        sparse_add(
            {(t_degree + 1, r_degree, s_degree): value
             for (t_degree, r_degree, s_degree), value in symmetric[i][j].items()},
            sparse_scale(skew[i][j], -1),
        )
        for j in range(GROUND)
    ] for i in range(GROUND)]
    numerator = multivariate_determinant(pencil)
    det_h = multivariate_determinant(symmetric)

    def coefficient(poly, t_degree):
        return {
            (r_degree, s_degree): value
            for (degree, r_degree, s_degree), value in poly.items()
            if degree == t_degree
        }

    denominator = coefficient(det_h, 0)
    assert coefficient(numerator, 5) == denominator
    return denominator, coefficient(numerator, 3), coefficient(numerator, 1)


def compactify(poly):
    """Power coefficients of (1-s)^d poly(s/(1-s)) on 0 <= s <= 1."""
    degree_bound = len(poly) - 1
    out = [Fraction(0) for _ in range(degree_bound + 1)]
    for degree, value in enumerate(poly):
        for extra in range(degree_bound + 1 - degree):
            out[degree + extra] += (
                value
                * comb(degree_bound - degree, extra)
                * (-1) ** extra
            )
    return out


def bernstein_max(poly, left, right):
    """Exact upper bound from Bernstein coefficients on an interval."""
    degree_bound = len(poly) - 1
    shifted = [Fraction(0) for _ in range(degree_bound + 1)]
    width = right - left
    for degree, value in enumerate(poly):
        for power in range(degree + 1):
            shifted[power] += (value * comb(degree, power)
                               * left ** (degree - power) * width ** power)
    coefficients = [
        sum(shifted[power]
            * Fraction(comb(index, power), comb(degree_bound, power))
            for power in range(index + 1))
        for index in range(degree_bound + 1)
    ]
    return max(coefficients)


def exact_cover(sum_test, root_test, left=Fraction(0), right=Fraction(1), depth=0):
    """Cover by intervals where S <= 6/5 or f(3/5) <= 0."""
    if bernstein_max(sum_test, left, right) <= 0:
        return 1, depth
    if bernstein_max(root_test, left, right) <= 0:
        return 1, depth
    assert depth < 12
    midpoint = (left + right) / 2
    left_count, left_depth = exact_cover(sum_test, root_test, left, midpoint, depth + 1)
    right_count, right_depth = exact_cover(sum_test, root_test, midpoint, right, depth + 1)
    return left_count + right_count, max(left_depth, right_depth)


def fraction_matrix_inverse(matrix):
    """Exact Gauss-Jordan inverse of a small rational matrix."""
    size = len(matrix)
    augmented = [
        [Fraction(value) for value in row]
        + [Fraction(index == column) for column in range(size)]
        for index, row in enumerate(matrix)
    ]
    for column in range(size):
        pivot = next(
            row for row in range(column, size) if augmented[row][column]
        )
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        pivot_value = augmented[column][column]
        augmented[column] = [value / pivot_value for value in augmented[column]]
        for row in range(size):
            if row == column or not augmented[row][column]:
                continue
            multiple = augmented[row][column]
            augmented[row] = [
                value - multiple * pivot_entry
                for value, pivot_entry in zip(
                    augmented[row], augmented[column]
                )
            ]
    return [row[size:] for row in augmented]


def homogeneous_midpoint_transforms(exponents):
    """Integer coefficient maps for the four midpoint subtriangles."""
    half = Fraction(1, 2)
    vertices = (
        ((1, 0, 0), (half, half, 0), (half, 0, half)),
        ((half, half, 0), (0, 1, 0), (0, half, half)),
        ((half, 0, half), (0, half, half), (0, 0, 1)),
        ((half, half, 0), (0, half, half), (half, 0, half)),
    )
    exponent_index = {exponent: index for index, exponent in enumerate(exponents)}
    transforms = []
    for child_vertices in vertices:
        linear_forms = [
            {
                tuple(int(coordinate == child) for coordinate in range(3)):
                    Fraction(child_vertices[child][parent])
                for child in range(3)
                if child_vertices[child][parent]
            }
            for parent in range(3)
        ]
        transform = [
            [Fraction(0) for _ in exponents] for _ in exponents
        ]
        for column, exponent in enumerate(exponents):
            term = {(0, 0, 0): Fraction(1)}
            for variable, power in enumerate(exponent):
                term = multivariate_mul(
                    term,
                    multivariate_power(linear_forms[variable], power),
                )
            for child_exponent, value in term.items():
                transform[exponent_index[child_exponent]][column] = value
        assert all(
            (32 * value).denominator == 1
            for row in transform for value in row
        )
        transforms.append(np.array([
            [int(32 * value) for value in row] for row in transform
        ], dtype=object))
    return transforms


def simplex_child_vertices(vertices):
    """The four midpoint children of a barycentric triangle."""
    first, second, third = vertices
    midpoint_12 = tuple((a + b) / 2 for a, b in zip(first, second))
    midpoint_13 = tuple((a + b) / 2 for a, b in zip(first, third))
    midpoint_23 = tuple((a + b) / 2 for a, b in zip(second, third))
    return (
        (first, midpoint_12, midpoint_13),
        (midpoint_12, second, midpoint_23),
        (midpoint_13, midpoint_23, third),
        (midpoint_12, midpoint_23, midpoint_13),
    )


def inside_extremal_box(vertices, pair):
    """Whether a triangle lies in the certified extremal-pair box."""
    first, second = pair
    third = 3 - first - second
    for vertex in vertices:
        pair_sum = vertex[first] + vertex[second]
        if 64 * abs(vertex[first] - vertex[second]) > 3 * pair_sum:
            return False
        if 128 * vertex[third] > pair_sum:
            return False
    return True


def exact_simplex_cover(
    sum_test,
    root_test,
    transforms,
    local_pairs=(),
    vertices=((Fraction(1), Fraction(0), Fraction(0)),
              (Fraction(0), Fraction(1), Fraction(0)),
              (Fraction(0), Fraction(0), Fraction(1))),
    depth=0,
):
    """Cover a weight simplex by global or local exact certificates."""
    if max(sum_test) <= 0 or max(root_test) <= 0:
        return 1, depth, 0
    if any(inside_extremal_box(vertices, pair) for pair in local_pairs):
        return 1, depth, 1
    assert depth < 12
    child_vertices = simplex_child_vertices(vertices)
    covers = [
        exact_simplex_cover(
            transform @ sum_test,
            transform @ root_test,
            transforms,
            local_pairs,
            child,
            depth + 1,
        )
        for transform, child in zip(transforms, child_vertices)
    ]
    return (
        sum(cover[0] for cover in covers),
        max(cover[1] for cover in covers),
        sum(cover[2] for cover in covers),
    )


def inside_short_cycle_corner(vertices, corner, height):
    """Whether a triangle lies in a proved short-cycle corner box."""
    other = [index for index in range(3) if index != corner]
    return all(
        vertex[other[0]] + vertex[other[1]] <= height * vertex[corner]
        for vertex in vertices
    )


def exact_simplex_cover_with_corner(
    sum_test,
    root_test,
    transforms,
    corner,
    corner_height,
    local_pairs=(),
    vertices=((Fraction(1), Fraction(0), Fraction(0)),
              (Fraction(0), Fraction(1), Fraction(0)),
              (Fraction(0), Fraction(0), Fraction(1))),
    depth=0,
):
    """Cover a simplex, recording corner and equality-edge local cells."""
    if max(sum_test) <= 0 or max(root_test) <= 0:
        return 1, depth, 0, 0
    if inside_short_cycle_corner(vertices, corner, corner_height):
        return 1, depth, 1, 0
    if any(inside_extremal_box(vertices, pair) for pair in local_pairs):
        return 1, depth, 0, 1
    assert depth < 12
    covers = [
        exact_simplex_cover_with_corner(
            transform @ sum_test,
            transform @ root_test,
            transforms,
            corner,
            corner_height,
            local_pairs,
            child,
            depth + 1,
        )
        for transform, child in zip(
            transforms, simplex_child_vertices(vertices)
        )
    ]
    return (
        sum(cover[0] for cover in covers),
        max(cover[1] for cover in covers),
        sum(cover[2] for cover in covers),
        sum(cover[3] for cover in covers),
    )


def exact_simplex_cover_with_two_corners(
    sum_test,
    root_test,
    transforms,
    corners,
    corner_height,
    vertices=((Fraction(1), Fraction(0), Fraction(0)),
              (Fraction(0), Fraction(1), Fraction(0)),
              (Fraction(0), Fraction(0), Fraction(1))),
    depth=0,
):
    """Cover a simplex, recording cells in either proved corner box."""
    if max(sum_test) <= 0 or max(root_test) <= 0:
        return 1, depth, 0
    if any(
        inside_short_cycle_corner(vertices, corner, corner_height)
        for corner in corners
    ):
        return 1, depth, 1
    assert depth < 12
    covers = [
        exact_simplex_cover_with_two_corners(
            transform @ sum_test,
            transform @ root_test,
            transforms,
            corners,
            corner_height,
            child,
            depth + 1,
        )
        for transform, child in zip(
            transforms, simplex_child_vertices(vertices)
        )
    ]
    return (
        sum(cover[0] for cover in covers),
        max(cover[1] for cover in covers),
        sum(cover[2] for cover in covers),
    )


def canonical_cycle(cycle):
    start = min(range(len(cycle)), key=lambda index: cycle[index])
    return cycle[start:] + cycle[:start]


def reverse_cycle(cycle):
    """Reverse orientation and retain the canonical cyclic representative."""
    return canonical_cycle((cycle[0],) + tuple(reversed(cycle[1:])))


def connected_union(first_cycle, second_cycle, state_count=N):
    neighbors = [set() for _ in range(state_count)]
    for cycle in (first_cycle, second_cycle):
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            neighbors[source].add(target)
            neighbors[target].add(source)
    reached = {0}
    frontier = [0]
    while frontier:
        state = frontier.pop()
        for target in neighbors[state] - reached:
            reached.add(target)
            frontier.append(target)
    return len(reached) == state_count


def integer_grounded_forms(cycle):
    """Return 2H and 2D as integer matrices for a unit cycle flow."""
    weights = np.zeros((N, N), dtype=np.int64)
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        weights[source, target] += 1
    laplacian = np.diag(np.sum(weights, axis=1)) - weights
    grounded = laplacian[:GROUND, :GROUND]
    return grounded + grounded.T, grounded - grounded.T


def grounded_forms_from_flow(stationary_flow):
    laplacian = np.diag(np.sum(stationary_flow, axis=1)) - stationary_flow
    grounded = laplacian[:GROUND, :GROUND]
    return (grounded + grounded.T) / 2, (grounded - grounded.T) / 2


def intrinsic_frequencies(symmetric, skew, positive_tolerance=1e-12):
    """Two positive skew frequencies, allowing semidefinite symmetric forms."""
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    positive = eigenvalues > positive_tolerance
    whitening = (
        vectors[:, positive] / np.sqrt(eigenvalues[positive])
    ) @ vectors[:, positive].T
    singular = np.linalg.svd(whitening @ skew @ whitening, compute_uv=False)
    return singular[0], singular[2]


def batch_integer_determinants(matrices):
    """Exact 5-by-5 determinants for an int64 matrix batch."""
    terms = np.ones(
        (len(matrices), len(GROUND_PERMUTATIONS)), dtype=np.int64
    )
    for row in range(GROUND):
        terms *= matrices[:, row, GROUND_PERMUTATIONS[:, row]]
    return terms @ GROUND_SIGNS


def batch_integer_pencil_coefficients(symmetric, skew):
    """Exact t^5, t^3 and t coefficients for batches of 5D pencils."""
    value_one = batch_integer_determinants(symmetric - skew)
    value_two = batch_integer_determinants(2 * symmetric - skew)
    value_three = batch_integer_determinants(3 * symmetric - skew)
    coefficient_five = (
        value_three - 4 * value_two + 5 * value_one
    ) // 120
    coefficient_three = (
        (value_two - 2 * value_one) // 6 - 5 * coefficient_five
    )
    coefficient_one = value_one - coefficient_five - coefficient_three
    assert np.array_equal(
        120 * coefficient_five,
        value_three - 4 * value_two + 5 * value_one,
    )
    assert np.array_equal(
        6 * (coefficient_three + 5 * coefficient_five),
        value_two - 2 * value_one,
    )
    return coefficient_five, coefficient_three, coefficient_one


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

# On the displayed three-cycle architecture, homogeneity lets us normalize the
# first weight to one and write the other two as r and s.  Verify the complete
# coefficient identities as exact bivariate polynomials.  In symmetric
# notation e1=1+r+s, e2=r+s+rs, e3=rs they imply
#   9*a5 - 15*a3 + 25*a1 = -e2*(e1*e2 - 9*e3).
# The final factor is nonnegative by AM-GM, with equality only at r=s=1.
one = {(0, 0): Fraction(1)}
r_variable = {(1, 0): Fraction(1)}
s_variable = {(0, 1): Fraction(1)}
e1 = sparse_add(sparse_add(one, r_variable), s_variable)
e2 = sparse_add(
    sparse_add(r_variable, s_variable),
    multivariate_mul(r_variable, s_variable),
)
e3 = multivariate_mul(r_variable, s_variable)
e1_e2 = multivariate_mul(e1, e2)

three_a5, three_a3, three_a1 = symbolic_three_cycle_coefficients()
expected_a5 = sparse_scale(
    multivariate_mul(
        e2,
        sparse_add(sparse_scale(e1_e2, 3), sparse_scale(e3, -2)),
    ),
    Fraction(1, 8),
)
expected_a3 = sparse_scale(
    multivariate_mul(
        e2,
        sparse_add(sparse_scale(e1_e2, 2), sparse_scale(e3, -3)),
    ),
    Fraction(1, 4),
)
expected_a1 = sparse_scale(
    multivariate_mul(multivariate_mul(e1, e2), e2),
    Fraction(1, 8),
)
assert (three_a5, three_a3, three_a1) == (
    expected_a5,
    expected_a3,
    expected_a1,
)
root_barrier = sparse_add(
    sparse_add(sparse_scale(three_a5, 9), sparse_scale(three_a3, -15)),
    sparse_scale(three_a1, 25),
)
am_gm_gap = sparse_add(e1_e2, sparse_scale(e3, -9))
assert root_barrier == sparse_scale(multivariate_mul(e2, am_gm_gap), -1)

# Seven symmetry orbits of genuine Hamilton triples admit an even simpler
# arbitrary-weight proof.  For every representative, every coefficient of
# 9*a5 - 15*a3 + 25*a1 is nonpositive as a polynomial in r,s, and at least
# one is negative.  Hence the root barrier is strictly negative for r,s>0.
hamilton_barriers = []
for representative in HAMILTON_TRIPLE_ORBIT_REPRESENTATIVES:
    hamilton_a5, hamilton_a3, hamilton_a1 = (
        symbolic_three_cycle_coefficients(representative)
    )
    barrier = sparse_add(
        sparse_add(
            sparse_scale(hamilton_a5, 9),
            sparse_scale(hamilton_a3, -15),
        ),
        sparse_scale(hamilton_a1, 25),
    )
    assert barrier and all(value <= 0 for value in barrier.values())
    assert any(value < 0 for value in barrier.values())
    hamilton_barriers.append(barrier)

# Count the supports represented by those seven orbits after fixing one
# Hamilton cycle to (0,1,2,3,4,5).  Relabeling and global reversal preserve
# the singular frequencies; the orbit sets are disjoint.
standard_hamilton_cycle = tuple(range(N))
hamilton_orbit_supports = []
for representative in HAMILTON_TRIPLE_ORBIT_REPRESENTATIVES:
    supports = set()
    for relabeling in permutations(range(N)):
        relabeled = [
            canonical_cycle(tuple(relabeling[state] for state in cycle))
            for cycle in representative
        ]
        for reverse_all in (False, True):
            transformed = (
                [reverse_cycle(cycle) for cycle in relabeled]
                if reverse_all else relabeled
            )
            support = tuple(sorted(transformed))
            if standard_hamilton_cycle in support:
                supports.add(support)
    hamilton_orbit_supports.append(supports)
assert sorted(map(len, hamilton_orbit_supports)) == [6, 9, 18, 18, 18, 18, 36]
assert len(set().union(*hamilton_orbit_supports)) == 123
assert sum(map(len, hamilton_orbit_supports)) == 123

# Relabel the first simple cycle canonically by its length; enumerate every
# possible second simple cycle.  Connected union is equivalent to irreducibility
# for these balanced flows.  Write r=s/(1-s), compactifying r in [0,infinity]
# to s in [0,1].  If S <= 6/5, the smaller squared frequency is at most 3/5;
# otherwise f(3/5) <= 0 puts 3/5 between the squared frequencies.
all_cycles = []
for cycle_length in range(2, N + 1):
    all_cycles.extend(sorted({
        canonical_cycle(cycle)
        for cycle in permutations(range(N), cycle_length)
    }))
cycle_pairs = [
    (tuple(range(cycle_length)), other_cycle)
    for cycle_length in range(2, N + 1)
    for other_cycle in all_cycles
    if connected_union(tuple(range(cycle_length)), other_cycle)
]
pencil_types = set()
largest_cover = 0
deepest_cover = 0
for first_cycle, other_cycle in cycle_pairs:
    denominator, sum_numerator, product_numerator = symbolic_pencil_coefficients(
        first_cycle, other_cycle
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

assert len(all_cycles) == 409
assert len(cycle_pairs) == 1372
assert len(pencil_types) == 133
assert largest_cover == 6
assert deepest_cover == 4

# A two-cycle directed base need not itself use all six states before reversible
# edges connect the final flow.  On a connected active support of at most four
# states there is at most one positive skew frequency.  The only new case is
# therefore a connected five-state active support, which the same exact pencil
# argument classifies here.  Disconnected two-cycle supports are two disjoint
# cycle components; with at most six active states their second frequency is at
# most the 3+3 value cot(pi/3)=1/sqrt(3).
five_state_count = N - 1
five_state_cycles = []
for cycle_length in range(2, five_state_count + 1):
    five_state_cycles.extend(sorted({
        canonical_cycle(cycle)
        for cycle in permutations(range(five_state_count), cycle_length)
    }))
five_state_pairs = [
    (tuple(range(cycle_length)), other_cycle)
    for cycle_length in range(2, five_state_count + 1)
    for other_cycle in five_state_cycles
    if connected_union(
        tuple(range(cycle_length)), other_cycle, five_state_count
    )
]
five_state_pencil_types = set()
five_state_largest_cover = 0
five_state_deepest_cover = 0
for first_cycle, other_cycle in five_state_pairs:
    denominator, sum_numerator, product_numerator = (
        symbolic_pencil_coefficients_n(
            first_cycle, other_cycle, five_state_count
        )
    )
    five_state_pencil_types.add(
        (denominator, sum_numerator, product_numerator)
    )
    sum_test = compactify([
        sum_numerator[index] - Fraction(6, 5) * denominator[index]
        for index in range(five_state_count)
    ])
    root_test = compactify([
        product_numerator[index]
        - Fraction(3, 5) * sum_numerator[index]
        + Fraction(9, 25) * denominator[index]
        for index in range(five_state_count)
    ])
    cover_size, cover_depth = exact_cover(sum_test, root_test)
    five_state_largest_cover = max(five_state_largest_cover, cover_size)
    five_state_deepest_cover = max(five_state_deepest_cover, cover_depth)

assert len(five_state_cycles) == 84
assert len(five_state_pairs) == 232
assert len(five_state_pencil_types) == 41
assert five_state_largest_cover == 6
assert five_state_deepest_cover == 3

# For three distinct cycles at equal weight, det(t(2H)-2D) has the form
# a5*t^5 + a3*t^3 + a1*t.  Evaluate it at t=1,2,3 and recover the integer
# coefficients.  Repeated cycles merge to an already covered two-cycle flow.
integer_forms = [integer_grounded_forms(cycle) for cycle in all_cycles]
integer_symmetric = np.stack([form[0] for form in integer_forms])
integer_skew = np.stack([form[1] for form in integer_forms])
cycle_index = {cycle: index for index, cycle in enumerate(all_cycles)}
connected_triples = 0
triple_coefficient_pairs = set()
equality_triples = []
for cycle_length in range(2, N + 1):
    first_cycle = tuple(range(cycle_length))
    first_index = cycle_index[first_cycle]
    other_indices = [
        index for index in range(len(all_cycles)) if index != first_index
    ]
    pairs = np.array(list(combinations(other_indices, 2)), dtype=np.int64)
    for start in range(0, len(pairs), 4096):
        batch = pairs[start:start + 4096]
        symmetric = (
            integer_symmetric[first_index]
            + integer_symmetric[batch[:, 0]]
            + integer_symmetric[batch[:, 1]]
        )
        skew = (
            integer_skew[first_index]
            + integer_skew[batch[:, 0]]
            + integer_skew[batch[:, 1]]
        )
        # The largest evaluated entry is 21, so the 120-term Leibniz sums
        # are far below the int64 ceiling and remain exact.
        assert np.max(np.abs(3 * symmetric - skew)) <= 21
        coefficient_five, coefficient_three, coefficient_one = (
            batch_integer_pencil_coefficients(symmetric, skew)
        )

        connected = coefficient_five > 0
        connected_triples += int(np.sum(connected))
        sum_test = 5 * coefficient_three <= 6 * coefficient_five
        root_test = (
            9 * coefficient_five
            - 15 * coefficient_three
            + 25 * coefficient_one
            <= 0
        )
        assert np.all((sum_test | root_test)[connected])
        for denominator, sum_numerator, product_numerator in zip(
            coefficient_five[connected],
            coefficient_three[connected],
            coefficient_one[connected],
        ):
            triple_coefficient_pairs.add(
                (
                    Fraction(int(sum_numerator), int(denominator)),
                    Fraction(int(product_numerator), int(denominator)),
                )
            )

        equality = (
            connected
            & (5 * coefficient_three == 6 * coefficient_five)
            & (25 * coefficient_one == 9 * coefficient_five)
        )
        for local_index in np.flatnonzero(equality):
            second_index, third_index = batch[local_index]
            equality_triples.append(
                (
                    first_cycle,
                    all_cycles[int(second_index)],
                    all_cycles[int(third_index)],
                )
            )

expected_equality_triples = [
    ((0, 1, 2, 3), (0, 1, 4, 5), (2, 3, 4, 5)),
    ((0, 1, 2, 3), (0, 1, 5, 4), (2, 3, 5, 4)),
    ((0, 1, 2, 3), (0, 4, 5, 3), (1, 2, 4, 5)),
    ((0, 1, 2, 3), (0, 5, 4, 3), (1, 2, 5, 4)),
]
assert connected_triples == 382466
assert len(triple_coefficient_pairs) == 3207
assert equality_triples == expected_equality_triples

# The four representatives above are one support orbit under state relabeling
# and simultaneous reversal of all cycles.  The latter changes D to -D and
# therefore leaves the intrinsic singular frequencies unchanged.
equality_orbit_maps = [
    ((0, 1, 2, 3, 4, 5), False),
    ((0, 1, 2, 3, 5, 4), False),
    ((0, 3, 2, 1, 5, 4), True),
    ((0, 3, 2, 1, 4, 5), True),
]
orbit_representative = expected_equality_triples[0]
for target, (relabeling, reverse_all) in zip(
    expected_equality_triples, equality_orbit_maps
):
    transformed = [
        canonical_cycle(tuple(relabeling[state] for state in cycle))
        for cycle in orbit_representative
    ]
    if reverse_all:
        transformed = [reverse_cycle(cycle) for cycle in transformed]
    assert tuple(sorted(transformed)) == tuple(sorted(target))

# Prove the arbitrary-weight bound for every triple of Hamilton cycles whose
# three boundary pairs are nonextremal.  There are 7,021 normalized Hamilton
# triples.  Exactly 120 labeled Hamilton pairs attain the two-cycle bound at
# equal weights, with degree two at every cycle; excluding triples containing
# one of those pairs leaves 6,670 supports.
hamilton_indices = np.array([
    index for index, cycle in enumerate(all_cycles) if len(cycle) == N
], dtype=np.int64)
hamilton_pairs = np.array(
    list(combinations(hamilton_indices, 2)), dtype=np.int64
)
pair_symmetric = (
    integer_symmetric[hamilton_pairs[:, 0]]
    + integer_symmetric[hamilton_pairs[:, 1]]
)
pair_skew = (
    integer_skew[hamilton_pairs[:, 0]]
    + integer_skew[hamilton_pairs[:, 1]]
)
pair_a5, pair_a3, pair_a1 = batch_integer_pencil_coefficients(
    pair_symmetric, pair_skew
)
extremal_pair_mask = (
    (5 * pair_a3 == 6 * pair_a5)
    & (25 * pair_a1 == 9 * pair_a5)
)
extremal_hamilton_pairs = {
    tuple(pair) for pair in hamilton_pairs[extremal_pair_mask]
}
assert len(extremal_hamilton_pairs) == 120
assert {
    sum(
        tuple(sorted((index, other))) in extremal_hamilton_pairs
        for other in hamilton_indices if other != index
    )
    for index in hamilton_indices
} == {2}

standard_hamilton_index = cycle_index[standard_hamilton_cycle]
other_hamilton_indices = [
    index for index in hamilton_indices if index != standard_hamilton_index
]
normalized_hamilton_triples = np.array(
    list(combinations(other_hamilton_indices, 2)), dtype=np.int64
)
nonextremal_boundary = np.array([
    all(
        tuple(sorted(pair)) not in extremal_hamilton_pairs
        for pair in (
            (standard_hamilton_index, second),
            (standard_hamilton_index, third),
            (second, third),
        )
    )
    for second, third in normalized_hamilton_triples
], dtype=bool)
assert len(normalized_hamilton_triples) == 7021
assert int(np.sum(nonextremal_boundary)) == 6670

# Recover each homogeneous degree-five barrier polynomial exactly from its
# values at the 21 integer barycentric nodes.  The interpolation and all
# determinants use integer arithmetic.
homogeneous_exponents = [
    exponent
    for exponent in product(range(6), repeat=3)
    if sum(exponent) == 5
]
vandermonde = [
    [
        a ** i * b ** j * c ** k
        for i, j, k in homogeneous_exponents
    ]
    for a, b, c in homogeneous_exponents
]
inverse_vandermonde = fraction_matrix_inverse(vandermonde)
interpolation_denominator = 1
for row in inverse_vandermonde:
    for value in row:
        interpolation_denominator = math.lcm(
            interpolation_denominator, value.denominator
        )
interpolation_matrix = np.array([
    [int(interpolation_denominator * value) for value in row]
    for row in inverse_vandermonde
], dtype=np.int64)

sum_values = []
root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight * integer_symmetric[normalized_hamilton_triples[:, 0]]
        + third_weight * integer_symmetric[normalized_hamilton_triples[:, 1]]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight * integer_skew[normalized_hamilton_triples[:, 0]]
        + third_weight * integer_skew[normalized_hamilton_triples[:, 1]]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    sum_values.append(5 * node_a3 - 6 * node_a5)
    root_values.append(9 * node_a5 - 15 * node_a3 + 25 * node_a1)
stacked_sum_values = np.stack(sum_values)
stacked_root_values = np.stack(root_values)
interpolation_entry_bound = int(np.max(np.abs(interpolation_matrix)))
for stacked_values in (stacked_sum_values, stacked_root_values):
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    assert product_bound < np.iinfo(np.int64).max
sum_numerators = interpolation_matrix @ stacked_sum_values
root_numerators = interpolation_matrix @ stacked_root_values
assert np.all(sum_numerators % interpolation_denominator == 0)
assert np.all(root_numerators % interpolation_denominator == 0)
sum_coefficients = sum_numerators // interpolation_denominator
root_coefficients = root_numerators // interpolation_denominator

simplex_transforms = homogeneous_midpoint_transforms(homogeneous_exponents)

# The remaining Hamilton triples have an extremal pair on the boundary of the
# weight simplex.  At its equality point write the pair weights as 1+u, 1-u
# and the third weight as v.  The 354 pair incidences reduce to 18 polynomial
# types, up to u -> -u.  A rational blow-up covers a full neighborhood:
# use the sum barrier for |u| <= 3v and the root barrier for v <= |u|/3.
extremal_incidence_types = {}
boundary_local_pairs = {}
for triple_index in np.flatnonzero(~nonextremal_boundary):
    support_indices = (
        standard_hamilton_index,
        int(normalized_hamilton_triples[triple_index, 0]),
        int(normalized_hamilton_triples[triple_index, 1]),
    )
    local_pairs = []
    for first, second in ((0, 1), (0, 2), (1, 2)):
        support_pair = tuple(sorted((
            support_indices[first], support_indices[second]
        )))
        if support_pair not in extremal_hamilton_pairs:
            continue
        local_pairs.append((first, second))
        sum_poly = extremal_pair_polynomial(
            sum_coefficients[:, triple_index],
            homogeneous_exponents,
            first,
            second,
        )
        root_poly = extremal_pair_polynomial(
            root_coefficients[:, triple_index],
            homogeneous_exponents,
            first,
            second,
        )
        direct = (tuple(sorted(sum_poly.items())),
                  tuple(sorted(root_poly.items())))
        reversed_pair = (
            tuple(sorted(reverse_first_variable(sum_poly).items())),
            tuple(sorted(reverse_first_variable(root_poly).items())),
        )
        key = min(direct, reversed_pair)
        extremal_incidence_types[key] = extremal_incidence_types.get(key, 0) + 1
    boundary_local_pairs[int(triple_index)] = tuple(local_pairs)
assert len(boundary_local_pairs) == 351
assert sum(map(len, boundary_local_pairs.values())) == 354
assert len(extremal_incidence_types) == 18
assert sorted(extremal_incidence_types.values()) == (
    [6] * 2 + [18] * 13 + [36] * 3
)

blowup_slope = Fraction(3)
blowup_height = Fraction(1, 64)
blowup_width = blowup_slope * blowup_height
local_largest_control = None
for sum_items, root_items in extremal_incidence_types:
    sum_poly = dict(sum_items)
    root_poly = dict(root_items)

    # Central wedge: u=v(-3+6x), v=y/64.  The sum barrier has
    # a positive factor v, removed before checking the unit square.
    central_sum = multivariate_compose(sum_poly, (
        {(0, 1): -blowup_slope * blowup_height,
         (1, 1): 2 * blowup_slope * blowup_height},
        {(0, 1): blowup_height},
    ))
    assert all(exponent[1] >= 1 for exponent in central_sum)
    central_sum = {
        (exponent[0], exponent[1] - 1): coefficient
        for exponent, coefficient in central_sum.items()
    }
    local_controls = tensor_bernstein_controls(central_sum)

    # Outer wedges: u=+/-3x/64 and v=xy/64.  The root barrier has
    # a positive factor u^2, removed before the same exact check.
    for direction in (-1, 1):
        outer_root = multivariate_compose(root_poly, (
            {(1, 0): direction * blowup_width},
            {(1, 1): blowup_height},
        ))
        assert all(exponent[0] >= 2 for exponent in outer_root)
        outer_root = {
            (exponent[0] - 2, exponent[1]): coefficient
            for exponent, coefficient in outer_root.items()
        }
        local_controls.extend(tensor_bernstein_controls(outer_root))
    assert max(local_controls) < 0
    type_largest_control = max(local_controls)
    if local_largest_control is None:
        local_largest_control = type_largest_control
    else:
        local_largest_control = max(
            local_largest_control, type_largest_control
        )
assert local_largest_control == Fraction(-45, 16)

hamilton_largest_cover = 0
hamilton_deepest_cover = 0
for triple_index in np.flatnonzero(nonextremal_boundary):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        sum_coefficients[:, triple_index].astype(object),
        root_coefficients[:, triple_index].astype(object),
        simplex_transforms,
    )
    assert local_cells == 0
    hamilton_largest_cover = max(hamilton_largest_cover, cover_size)
    hamilton_deepest_cover = max(hamilton_deepest_cover, cover_depth)
assert hamilton_largest_cover == 76
assert hamilton_deepest_cover == 5

boundary_largest_cover = 0
boundary_deepest_cover = 0
boundary_local_cells = 0
for triple_index, local_pairs in boundary_local_pairs.items():
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        sum_coefficients[:, triple_index].astype(object),
        root_coefficients[:, triple_index].astype(object),
        simplex_transforms,
        local_pairs,
    )
    boundary_largest_cover = max(boundary_largest_cover, cover_size)
    boundary_deepest_cover = max(boundary_deepest_cover, cover_depth)
    boundary_local_cells += local_cells
assert boundary_largest_cover == 166
assert boundary_deepest_cover == 8
assert boundary_local_cells == 1122

# Extend the arbitrary-weight theorem from three Hamilton cycles to every
# support containing two Hamilton cycles and one five-cycle.  Fix one
# Hamilton cycle canonically; the second Hamilton cycle and the five-cycle
# then give 119 * 144 = 17,136 normalized supports.  No Hamilton--five-cycle
# pair is extremal.  The only equality-boundary supports are therefore the
# 288 choices in which the two Hamilton cycles form one of the two extremal
# pairs through the fixed cycle.
five_cycle_indices = np.array([
    index for index, cycle in enumerate(all_cycles) if len(cycle) == 5
], dtype=np.int64)
assert len(five_cycle_indices) == 144
hamilton_five_pairs = np.array([
    (hamilton_index, five_cycle_index)
    for hamilton_index in hamilton_indices
    for five_cycle_index in five_cycle_indices
], dtype=np.int64)
hamilton_five_symmetric = (
    integer_symmetric[hamilton_five_pairs[:, 0]]
    + integer_symmetric[hamilton_five_pairs[:, 1]]
)
hamilton_five_skew = (
    integer_skew[hamilton_five_pairs[:, 0]]
    + integer_skew[hamilton_five_pairs[:, 1]]
)
hf_a5, hf_a3, hf_a1 = batch_integer_pencil_coefficients(
    hamilton_five_symmetric, hamilton_five_skew
)
extremal_hamilton_five_mask = (
    (5 * hf_a3 == 6 * hf_a5)
    & (25 * hf_a1 == 9 * hf_a5)
)
assert not np.any(extremal_hamilton_five_mask)

normalized_hamilton_five_triples = np.array([
    (hamilton_index, five_cycle_index)
    for hamilton_index in other_hamilton_indices
    for five_cycle_index in five_cycle_indices
], dtype=np.int64)
mixed_nonextremal_boundary = np.array([
    tuple(sorted((standard_hamilton_index, int(hamilton_index))))
    not in extremal_hamilton_pairs
    for hamilton_index, _ in normalized_hamilton_five_triples
], dtype=bool)
assert len(normalized_hamilton_five_triples) == 17136
assert int(np.sum(mixed_nonextremal_boundary)) == 16848

mixed_sum_values = []
mixed_root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight * integer_symmetric[
            normalized_hamilton_five_triples[:, 0]
        ]
        + third_weight * integer_symmetric[
            normalized_hamilton_five_triples[:, 1]
        ]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight * integer_skew[
            normalized_hamilton_five_triples[:, 0]
        ]
        + third_weight * integer_skew[
            normalized_hamilton_five_triples[:, 1]
        ]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    mixed_sum_values.append(5 * node_a3 - 6 * node_a5)
    mixed_root_values.append(9 * node_a5 - 15 * node_a3 + 25 * node_a1)
mixed_stacked_sum_values = np.stack(mixed_sum_values)
mixed_stacked_root_values = np.stack(mixed_root_values)
mixed_product_bound = 0
for stacked_values in (
    mixed_stacked_sum_values, mixed_stacked_root_values
):
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    mixed_product_bound = max(mixed_product_bound, product_bound)
    assert product_bound < np.iinfo(np.int64).max
assert mixed_product_bound == 99603000000
mixed_sum_numerators = interpolation_matrix @ mixed_stacked_sum_values
mixed_root_numerators = interpolation_matrix @ mixed_stacked_root_values
assert np.all(mixed_sum_numerators % interpolation_denominator == 0)
assert np.all(mixed_root_numerators % interpolation_denominator == 0)
mixed_sum_coefficients = mixed_sum_numerators // interpolation_denominator
mixed_root_coefficients = mixed_root_numerators // interpolation_denominator

# At the 288 equality-boundary supports, the same rational blow-up box used
# above applies.  The five-cycle third component produces 12 polynomial types
# up to u -> -u, each occurring 24 times.  The exact tensor-product controls
# are again strictly negative on the central and outer wedges.
mixed_extremal_incidence_types = {}
for triple_index in np.flatnonzero(~mixed_nonextremal_boundary):
    sum_poly = extremal_pair_polynomial(
        mixed_sum_coefficients[:, triple_index],
        homogeneous_exponents,
        0,
        1,
    )
    root_poly = extremal_pair_polynomial(
        mixed_root_coefficients[:, triple_index],
        homogeneous_exponents,
        0,
        1,
    )
    direct = (tuple(sorted(sum_poly.items())),
              tuple(sorted(root_poly.items())))
    reversed_pair = (
        tuple(sorted(reverse_first_variable(sum_poly).items())),
        tuple(sorted(reverse_first_variable(root_poly).items())),
    )
    key = min(direct, reversed_pair)
    mixed_extremal_incidence_types[key] = (
        mixed_extremal_incidence_types.get(key, 0) + 1
    )
assert len(mixed_extremal_incidence_types) == 12
assert set(mixed_extremal_incidence_types.values()) == {24}

mixed_local_largest_control = None
for sum_items, root_items in mixed_extremal_incidence_types:
    sum_poly = dict(sum_items)
    root_poly = dict(root_items)
    central_sum = multivariate_compose(sum_poly, (
        {(0, 1): -blowup_slope * blowup_height,
         (1, 1): 2 * blowup_slope * blowup_height},
        {(0, 1): blowup_height},
    ))
    assert all(exponent[1] >= 1 for exponent in central_sum)
    central_sum = {
        (exponent[0], exponent[1] - 1): coefficient
        for exponent, coefficient in central_sum.items()
    }
    local_controls = tensor_bernstein_controls(central_sum)
    for direction in (-1, 1):
        outer_root = multivariate_compose(root_poly, (
            {(1, 0): direction * blowup_width},
            {(1, 1): blowup_height},
        ))
        assert all(exponent[0] >= 2 for exponent in outer_root)
        outer_root = {
            (exponent[0] - 2, exponent[1]): coefficient
            for exponent, coefficient in outer_root.items()
        }
        local_controls.extend(tensor_bernstein_controls(outer_root))
    assert max(local_controls) < 0
    type_largest_control = max(local_controls)
    if mixed_local_largest_control is None:
        mixed_local_largest_control = type_largest_control
    else:
        mixed_local_largest_control = max(
            mixed_local_largest_control, type_largest_control
        )
assert mixed_local_largest_control == Fraction(
    -148074017, 134217728
)

mixed_largest_cover = 0
mixed_deepest_cover = 0
for triple_index in np.flatnonzero(mixed_nonextremal_boundary):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        mixed_sum_coefficients[:, triple_index].astype(object),
        mixed_root_coefficients[:, triple_index].astype(object),
        simplex_transforms,
    )
    assert local_cells == 0
    mixed_largest_cover = max(mixed_largest_cover, cover_size)
    mixed_deepest_cover = max(mixed_deepest_cover, cover_depth)
assert mixed_largest_cover == 76
assert mixed_deepest_cover == 5

mixed_boundary_largest_cover = 0
mixed_boundary_deepest_cover = 0
mixed_boundary_local_cells = 0
for triple_index in np.flatnonzero(~mixed_nonextremal_boundary):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        mixed_sum_coefficients[:, triple_index].astype(object),
        mixed_root_coefficients[:, triple_index].astype(object),
        simplex_transforms,
        ((0, 1),),
    )
    mixed_boundary_largest_cover = max(
        mixed_boundary_largest_cover, cover_size
    )
    mixed_boundary_deepest_cover = max(
        mixed_boundary_deepest_cover, cover_depth
    )
    mixed_boundary_local_cells += local_cells
assert mixed_boundary_largest_cover == 103
assert mixed_boundary_deepest_cover == 8
assert mixed_boundary_local_cells == 816

# The next shorter architecture, two Hamilton cycles plus one four-cycle,
# also obeys the bound for every weight.  There are 90 labeled four-cycles,
# no extremal Hamilton--four-cycle pair and 10,710 normalized supports.  The
# 180 equality-boundary supports again arise only from the Hamilton pair.
four_cycle_indices = np.array([
    index for index, cycle in enumerate(all_cycles) if len(cycle) == 4
], dtype=np.int64)
assert len(four_cycle_indices) == 90
hamilton_four_pairs = np.array([
    (hamilton_index, four_cycle_index)
    for hamilton_index in hamilton_indices
    for four_cycle_index in four_cycle_indices
], dtype=np.int64)
hamilton_four_symmetric = (
    integer_symmetric[hamilton_four_pairs[:, 0]]
    + integer_symmetric[hamilton_four_pairs[:, 1]]
)
hamilton_four_skew = (
    integer_skew[hamilton_four_pairs[:, 0]]
    + integer_skew[hamilton_four_pairs[:, 1]]
)
h4_a5, h4_a3, h4_a1 = batch_integer_pencil_coefficients(
    hamilton_four_symmetric, hamilton_four_skew
)
extremal_hamilton_four_mask = (
    (5 * h4_a3 == 6 * h4_a5)
    & (25 * h4_a1 == 9 * h4_a5)
)
assert not np.any(extremal_hamilton_four_mask)

normalized_hamilton_four_triples = np.array([
    (hamilton_index, four_cycle_index)
    for hamilton_index in other_hamilton_indices
    for four_cycle_index in four_cycle_indices
], dtype=np.int64)
four_nonextremal_boundary = np.array([
    tuple(sorted((standard_hamilton_index, int(hamilton_index))))
    not in extremal_hamilton_pairs
    for hamilton_index, _ in normalized_hamilton_four_triples
], dtype=bool)
assert len(normalized_hamilton_four_triples) == 10710
assert int(np.sum(four_nonextremal_boundary)) == 10530

four_sum_values = []
four_root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight * integer_symmetric[
            normalized_hamilton_four_triples[:, 0]
        ]
        + third_weight * integer_symmetric[
            normalized_hamilton_four_triples[:, 1]
        ]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight * integer_skew[
            normalized_hamilton_four_triples[:, 0]
        ]
        + third_weight * integer_skew[
            normalized_hamilton_four_triples[:, 1]
        ]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    four_sum_values.append(5 * node_a3 - 6 * node_a5)
    four_root_values.append(9 * node_a5 - 15 * node_a3 + 25 * node_a1)
four_stacked_sum_values = np.stack(four_sum_values)
four_stacked_root_values = np.stack(four_root_values)
four_product_bound = 0
for stacked_values in (
    four_stacked_sum_values, four_stacked_root_values
):
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    four_product_bound = max(four_product_bound, product_bound)
    assert product_bound < np.iinfo(np.int64).max
assert four_product_bound == 99603000000
four_sum_numerators = interpolation_matrix @ four_stacked_sum_values
four_root_numerators = interpolation_matrix @ four_stacked_root_values
assert np.all(four_sum_numerators % interpolation_denominator == 0)
assert np.all(four_root_numerators % interpolation_denominator == 0)
four_sum_coefficients = four_sum_numerators // interpolation_denominator
four_root_coefficients = four_root_numerators // interpolation_denominator

four_extremal_incidence_types = {}
for triple_index in np.flatnonzero(~four_nonextremal_boundary):
    sum_poly = extremal_pair_polynomial(
        four_sum_coefficients[:, triple_index],
        homogeneous_exponents,
        0,
        1,
    )
    root_poly = extremal_pair_polynomial(
        four_root_coefficients[:, triple_index],
        homogeneous_exponents,
        0,
        1,
    )
    direct = (tuple(sorted(sum_poly.items())),
              tuple(sorted(root_poly.items())))
    reversed_pair = (
        tuple(sorted(reverse_first_variable(sum_poly).items())),
        tuple(sorted(reverse_first_variable(root_poly).items())),
    )
    key = min(direct, reversed_pair)
    four_extremal_incidence_types[key] = (
        four_extremal_incidence_types.get(key, 0) + 1
    )
assert len(four_extremal_incidence_types) == 12
assert sorted(four_extremal_incidence_types.values()) == (
    [6] * 2 + [12] * 6 + [24] * 4
)

# Eleven local types retain a negative term linear in the four-cycle weight.
# They use the same central sum-barrier and outer root-barrier certificates as
# above.  The exceptional type has positive quadratic sum barrier but
# negative-definite leading root barrier -3840*u^2 - 192*v^2.  Split the
# central ratio coordinate at u=0; after removing v^2, both exact Bernstein
# patches are strictly negative.  Its outer wedges still factor by u^2.
four_regular_largest_control = None
four_exceptional_largest_control = None
four_exceptional_types = 0
four_exceptional_incidences = 0
for type_key, incidence_count in four_extremal_incidence_types.items():
    sum_poly = dict(type_key[0])
    root_poly = dict(type_key[1])
    minimum_sum_degree = min(sum(exponent) for exponent in sum_poly)
    local_controls = []
    if minimum_sum_degree == 1:
        central_sum = multivariate_compose(sum_poly, (
            {(0, 1): -blowup_slope * blowup_height,
             (1, 1): 2 * blowup_slope * blowup_height},
            {(0, 1): blowup_height},
        ))
        assert all(exponent[1] >= 1 for exponent in central_sum)
        central_sum = {
            (exponent[0], exponent[1] - 1): coefficient
            for exponent, coefficient in central_sum.items()
        }
        local_controls.extend(tensor_bernstein_controls(central_sum))
    else:
        assert minimum_sum_degree == 2
        four_exceptional_types += 1
        four_exceptional_incidences += incidence_count
        leading_root = {
            exponent: coefficient
            for exponent, coefficient in root_poly.items()
            if sum(exponent) == 2
        }
        assert leading_root == {
            (0, 2): Fraction(-192),
            (2, 0): Fraction(-3840),
        }
        central_root = multivariate_compose(root_poly, (
            {(0, 1): -blowup_slope * blowup_height,
             (1, 1): 2 * blowup_slope * blowup_height},
            {(0, 1): blowup_height},
        ))
        assert all(exponent[1] >= 2 for exponent in central_root)
        central_root = {
            (exponent[0], exponent[1] - 2): coefficient
            for exponent, coefficient in central_root.items()
        }
        for side in (0, 1):
            central_half = multivariate_compose(central_root, (
                {(0, 0): Fraction(side, 2),
                 (1, 0): Fraction(1, 2)},
                {(0, 1): Fraction(1)},
            ))
            local_controls.extend(
                tensor_bernstein_controls(central_half)
            )

    for direction in (-1, 1):
        outer_root = multivariate_compose(root_poly, (
            {(1, 0): direction * blowup_width},
            {(1, 1): blowup_height},
        ))
        assert all(exponent[0] >= 2 for exponent in outer_root)
        outer_root = {
            (exponent[0] - 2, exponent[1]): coefficient
            for exponent, coefficient in outer_root.items()
        }
        local_controls.extend(tensor_bernstein_controls(outer_root))
    assert max(local_controls) < 0
    type_largest_control = max(local_controls)
    if minimum_sum_degree == 1:
        if four_regular_largest_control is None:
            four_regular_largest_control = type_largest_control
        else:
            four_regular_largest_control = max(
                four_regular_largest_control, type_largest_control
            )
    else:
        four_exceptional_largest_control = type_largest_control
assert four_exceptional_types == 1
assert four_exceptional_incidences == 6
assert four_regular_largest_control == Fraction(
    -180285485, 33554432
)
assert four_exceptional_largest_control == Fraction(-3, 64)

four_largest_cover = 0
four_deepest_cover = 0
for triple_index in np.flatnonzero(four_nonextremal_boundary):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        four_sum_coefficients[:, triple_index].astype(object),
        four_root_coefficients[:, triple_index].astype(object),
        simplex_transforms,
    )
    assert local_cells == 0
    four_largest_cover = max(four_largest_cover, cover_size)
    four_deepest_cover = max(four_deepest_cover, cover_depth)
assert four_largest_cover == 97
assert four_deepest_cover == 5

four_boundary_largest_cover = 0
four_boundary_deepest_cover = 0
four_boundary_local_cells = 0
for triple_index in np.flatnonzero(~four_nonextremal_boundary):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        four_sum_coefficients[:, triple_index].astype(object),
        four_root_coefficients[:, triple_index].astype(object),
        simplex_transforms,
        ((0, 1),),
    )
    four_boundary_largest_cover = max(
        four_boundary_largest_cover, cover_size
    )
    four_boundary_deepest_cover = max(
        four_boundary_deepest_cover, cover_depth
    )
    four_boundary_local_cells += local_cells
assert four_boundary_largest_cover == 103
assert four_boundary_deepest_cover == 9
assert four_boundary_local_cells == 414

# The remaining simple-cycle length is three.  A pure three-cycle has rank
# only two on the centered space, so both degree-five barriers vanish to
# cubic order at that weight-simplex corner.  A second rational blow-up is
# therefore needed in addition to the equality-edge blow-up.  Write the two
# Hamilton weights as s*x and s*(1-x), normalize the three-cycle weight to
# one, and split x into 16 rational intervals.  On each patch at least one
# cubic-factored barrier has strictly negative Bernstein controls.
three_cycle_indices = np.array([
    index for index, cycle in enumerate(all_cycles) if len(cycle) == 3
], dtype=np.int64)
assert len(three_cycle_indices) == 40

# Three distinct directed triangles have an unexpectedly rigid spectrum.  For
# all C(40,3) supports, recover the homogeneous degree-five pencil exactly.
# Every coefficient vanishes on every boundary edge, so it contains abc.  On
# each connected support the remaining quadratic is one of four positive
# cross-term sums, and the three pencil coefficients have the fixed ratio
# 18:12:2.  Thus both intrinsic frequencies equal 1/sqrt(3), independently of
# the three positive weights.
three_cycle_triples = np.array(
    list(combinations(three_cycle_indices, 3)), dtype=np.int64
)
assert len(three_cycle_triples) == 9880
triangle_coefficient_values = {"a5": [], "a3": [], "a1": []}
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[three_cycle_triples[:, 0]]
        + second_weight * integer_symmetric[three_cycle_triples[:, 1]]
        + third_weight * integer_symmetric[three_cycle_triples[:, 2]]
    )
    skew = (
        first_weight * integer_skew[three_cycle_triples[:, 0]]
        + second_weight * integer_skew[three_cycle_triples[:, 1]]
        + third_weight * integer_skew[three_cycle_triples[:, 2]]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    triangle_coefficient_values["a5"].append(node_a5)
    triangle_coefficient_values["a3"].append(node_a3)
    triangle_coefficient_values["a1"].append(node_a1)

triangle_coefficients = {}
for name, values in triangle_coefficient_values.items():
    stacked_values = np.stack(values)
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    assert product_bound < np.iinfo(np.int64).max
    numerators = interpolation_matrix @ stacked_values
    assert np.all(numerators % interpolation_denominator == 0)
    triangle_coefficients[name] = numerators // interpolation_denominator

triangle_factor_positions = np.array([
    all(power >= 1 for power in exponent)
    for exponent in homogeneous_exponents
])
assert np.all(
    triangle_coefficients["a5"][~triangle_factor_positions] == 0
)
assert np.all(
    triangle_coefficients["a3"][~triangle_factor_positions] == 0
)
assert np.all(
    triangle_coefficients["a1"][~triangle_factor_positions] == 0
)
triangle_quotient_exponents = [
    tuple(power - 1 for power in exponent)
    for exponent in np.array(homogeneous_exponents)[triangle_factor_positions]
]
assert triangle_quotient_exponents == [
    (0, 0, 2), (0, 1, 1), (0, 2, 0),
    (1, 0, 1), (1, 1, 0), (2, 0, 0),
]
triangle_a5_quotients = triangle_coefficients["a5"][
    triangle_factor_positions
]
triangle_a3_quotients = triangle_coefficients["a3"][
    triangle_factor_positions
]
triangle_a1_quotients = triangle_coefficients["a1"][
    triangle_factor_positions
]
connected_triangle_triples = np.any(triangle_a5_quotients != 0, axis=0)
assert int(np.sum(connected_triangle_triples)) == 3840
assert not np.any(
    triangle_a3_quotients[:, ~connected_triangle_triples]
)
assert not np.any(
    triangle_a1_quotients[:, ~connected_triangle_triples]
)
assert np.all(
    3 * triangle_a3_quotients[:, connected_triangle_triples]
    == 2 * triangle_a5_quotients[:, connected_triangle_triples]
)
assert np.all(
    9 * triangle_a1_quotients[:, connected_triangle_triples]
    == triangle_a5_quotients[:, connected_triangle_triples]
)
triangle_a5_types = Counter(
    tuple(int(value) for value in triangle_a5_quotients[:, support_index])
    for support_index in np.flatnonzero(connected_triangle_triples)
)
assert triangle_a5_types == Counter({
    (0, 0, 0, 18, 18, 0): 1170,
    (0, 18, 0, 0, 18, 0): 540,
    (0, 18, 0, 18, 0, 0): 1170,
    (0, 18, 0, 18, 18, 0): 960,
})
assert np.all(triangle_a5_quotients[:, connected_triangle_triples] >= 0)
assert np.all(
    np.sum(triangle_a5_quotients[:, connected_triangle_triples], axis=0)
    > 0
)

hamilton_three_pairs = np.array([
    (hamilton_index, three_cycle_index)
    for hamilton_index in hamilton_indices
    for three_cycle_index in three_cycle_indices
], dtype=np.int64)
hamilton_three_symmetric = (
    integer_symmetric[hamilton_three_pairs[:, 0]]
    + integer_symmetric[hamilton_three_pairs[:, 1]]
)
hamilton_three_skew = (
    integer_skew[hamilton_three_pairs[:, 0]]
    + integer_skew[hamilton_three_pairs[:, 1]]
)
h3_a5, h3_a3, h3_a1 = batch_integer_pencil_coefficients(
    hamilton_three_symmetric, hamilton_three_skew
)
extremal_hamilton_three_mask = (
    (5 * h3_a3 == 6 * h3_a5)
    & (25 * h3_a1 == 9 * h3_a5)
)
assert not np.any(extremal_hamilton_three_mask)

normalized_hamilton_three_triples = np.array([
    (hamilton_index, three_cycle_index)
    for hamilton_index in other_hamilton_indices
    for three_cycle_index in three_cycle_indices
], dtype=np.int64)
three_nonextremal_boundary = np.array([
    tuple(sorted((standard_hamilton_index, int(hamilton_index))))
    not in extremal_hamilton_pairs
    for hamilton_index, _ in normalized_hamilton_three_triples
], dtype=bool)
assert len(normalized_hamilton_three_triples) == 4760
assert int(np.sum(three_nonextremal_boundary)) == 4680

three_sum_values = []
three_root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight * integer_symmetric[
            normalized_hamilton_three_triples[:, 0]
        ]
        + third_weight * integer_symmetric[
            normalized_hamilton_three_triples[:, 1]
        ]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight * integer_skew[
            normalized_hamilton_three_triples[:, 0]
        ]
        + third_weight * integer_skew[
            normalized_hamilton_three_triples[:, 1]
        ]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    three_sum_values.append(5 * node_a3 - 6 * node_a5)
    three_root_values.append(
        9 * node_a5 - 15 * node_a3 + 25 * node_a1
    )
three_stacked_sum_values = np.stack(three_sum_values)
three_stacked_root_values = np.stack(three_root_values)
three_product_bound = max(
    len(homogeneous_exponents)
    * interpolation_entry_bound
    * int(np.max(np.abs(stacked_values)))
    for stacked_values in (
        three_stacked_sum_values, three_stacked_root_values
    )
)
assert three_product_bound == 99603000000
three_sum_numerators = interpolation_matrix @ three_stacked_sum_values
three_root_numerators = interpolation_matrix @ three_stacked_root_values
assert np.all(three_sum_numerators % interpolation_denominator == 0)
assert np.all(three_root_numerators % interpolation_denominator == 0)
three_sum_coefficients = three_sum_numerators // interpolation_denominator
three_root_coefficients = three_root_numerators // interpolation_denominator

three_edge_types = {}
for triple_index in np.flatnonzero(~three_nonextremal_boundary):
    sum_poly = extremal_pair_polynomial(
        three_sum_coefficients[:, triple_index],
        homogeneous_exponents,
        0,
        1,
    )
    root_poly = extremal_pair_polynomial(
        three_root_coefficients[:, triple_index],
        homogeneous_exponents,
        0,
        1,
    )
    direct = (tuple(sorted(sum_poly.items())),
              tuple(sorted(root_poly.items())))
    reversed_pair = (
        tuple(sorted(reverse_first_variable(sum_poly).items())),
        tuple(sorted(reverse_first_variable(root_poly).items())),
    )
    key = min(direct, reversed_pair)
    three_edge_types[key] = three_edge_types.get(key, 0) + 1
assert len(three_edge_types) == 4
assert sorted(three_edge_types.values()) == [8, 24, 24, 24]

three_edge_largest_control = None
for sum_items, root_items in three_edge_types:
    sum_poly = dict(sum_items)
    root_poly = dict(root_items)
    central_sum = multivariate_compose(sum_poly, (
        {(0, 1): -blowup_slope * blowup_height,
         (1, 1): 2 * blowup_slope * blowup_height},
        {(0, 1): blowup_height},
    ))
    assert all(exponent[1] >= 1 for exponent in central_sum)
    central_sum = {
        (exponent[0], exponent[1] - 1): coefficient
        for exponent, coefficient in central_sum.items()
    }
    controls = tensor_bernstein_controls(central_sum)
    for direction in (-1, 1):
        outer_root = multivariate_compose(root_poly, (
            {(1, 0): direction * blowup_width},
            {(1, 1): blowup_height},
        ))
        assert all(exponent[0] >= 2 for exponent in outer_root)
        outer_root = {
            (exponent[0] - 2, exponent[1]): coefficient
            for exponent, coefficient in outer_root.items()
        }
        controls.extend(tensor_bernstein_controls(outer_root))
    assert max(controls) < 0
    type_largest_control = max(controls)
    if three_edge_largest_control is None:
        three_edge_largest_control = type_largest_control
    else:
        three_edge_largest_control = max(
            three_edge_largest_control, type_largest_control
        )
assert three_edge_largest_control == Fraction(
    -79125745, 33554432
)


def short_cycle_corner_polynomial(coefficients):
    """Set the third weight to one in a homogeneous barrier."""
    out = {}
    for coefficient, exponent in zip(coefficients, homogeneous_exponents):
        key = exponent[:2]
        out[key] = out.get(key, Fraction(0)) + Fraction(int(coefficient))
    return {key: value for key, value in out.items() if value}


three_corner_types = {}
three_corner_leading_types = set()
for triple_index in range(len(normalized_hamilton_three_triples)):
    sum_poly = short_cycle_corner_polynomial(
        three_sum_coefficients[:, triple_index]
    )
    root_poly = short_cycle_corner_polynomial(
        three_root_coefficients[:, triple_index]
    )
    leading_key = (
        tuple(sorted(
            (exponent, coefficient)
            for exponent, coefficient in sum_poly.items()
            if sum(exponent) == 3
        )),
        tuple(sorted(
            (exponent, coefficient)
            for exponent, coefficient in root_poly.items()
            if sum(exponent) == 3
        )),
    )
    three_corner_leading_types.add(leading_key)
    direct = (tuple(sorted(sum_poly.items())),
              tuple(sorted(root_poly.items())))
    swapped = (
        tuple(sorted(
            ((second, first), coefficient)
            for (first, second), coefficient in sum_poly.items()
        )),
        tuple(sorted(
            ((second, first), coefficient)
            for (first, second), coefficient in root_poly.items()
        )),
    )
    key = min(direct, swapped)
    three_corner_types[key] = three_corner_types.get(key, 0) + 1
assert len(three_corner_types) == 230
assert len(three_corner_leading_types) == 19
assert sum(three_corner_types.values()) == 4760
assert min(three_corner_types.values()) == 4
assert max(three_corner_types.values()) == 48

three_corner_height = Fraction(1, 64)
three_corner_ratio_patches = 16
three_corner_barrier_uses = {"sum": 0, "root": 0}
three_corner_largest_control = None
for sum_items, root_items in three_corner_types:
    sum_poly = dict(sum_items)
    root_poly = dict(root_items)
    for patch in range(three_corner_ratio_patches):
        substitutions = (
            {(0, 1): three_corner_height
             * Fraction(patch, three_corner_ratio_patches),
             (1, 1): three_corner_height
             * Fraction(1, three_corner_ratio_patches)},
            {(0, 1): three_corner_height
             * Fraction(three_corner_ratio_patches - patch,
                        three_corner_ratio_patches),
             (1, 1): -three_corner_height
             * Fraction(1, three_corner_ratio_patches)},
        )
        candidate_controls = {}
        for name, poly in (("sum", sum_poly), ("root", root_poly)):
            transformed = multivariate_compose(poly, substitutions)
            assert all(exponent[1] >= 3 for exponent in transformed)
            transformed = {
                (exponent[0], exponent[1] - 3): coefficient
                for exponent, coefficient in transformed.items()
            }
            candidate_controls[name] = max(
                tensor_bernstein_controls(transformed)
            )
        barrier, control = min(
            candidate_controls.items(), key=lambda item: item[1]
        )
        assert control < 0
        three_corner_barrier_uses[barrier] += 1
        if three_corner_largest_control is None:
            three_corner_largest_control = control
        else:
            three_corner_largest_control = max(
                three_corner_largest_control, control
            )
assert three_corner_barrier_uses == {"sum": 3392, "root": 288}
assert three_corner_largest_control == Fraction(
    -82072901, 4398046511104
)

three_largest_cover = 0
three_deepest_cover = 0
three_corner_local_cells = 0
three_edge_local_cells = 0
for triple_index in range(len(normalized_hamilton_three_triples)):
    local_pairs = (
        () if three_nonextremal_boundary[triple_index]
        else ((0, 1),)
    )
    cover_size, cover_depth, corner_cells, edge_cells = (
        exact_simplex_cover_with_corner(
            three_sum_coefficients[:, triple_index].astype(object),
            three_root_coefficients[:, triple_index].astype(object),
            simplex_transforms,
            2,
            three_corner_height,
            local_pairs,
        )
    )
    three_largest_cover = max(three_largest_cover, cover_size)
    three_deepest_cover = max(three_deepest_cover, cover_depth)
    three_corner_local_cells += corner_cells
    three_edge_local_cells += edge_cells
assert three_largest_cover == 481
assert three_deepest_cover == 10
assert three_corner_local_cells == 10440
assert three_edge_local_cells == 112

# A complementary three-cycle architecture has one Hamilton cycle and two
# five-cycles.  When the five-cycles omit different states, ordinary exact
# simplex covers apply.  If they omit the same state, the six-state pencil
# vanishes when the Hamilton weight is zero, so both barriers have that weight
# as a factor.  Divide it out and certify the resulting degree-four barriers.
five_cycle_pairs = np.array(
    list(combinations(five_cycle_indices, 2)), dtype=np.int64
)
five_pair_symmetric = (
    integer_symmetric[five_cycle_pairs[:, 0]]
    + integer_symmetric[five_cycle_pairs[:, 1]]
)
five_pair_skew = (
    integer_skew[five_cycle_pairs[:, 0]]
    + integer_skew[five_cycle_pairs[:, 1]]
)
five_pair_a5, five_pair_a3, five_pair_a1 = (
    batch_integer_pencil_coefficients(five_pair_symmetric, five_pair_skew)
)
equal_barrier_five_pair = (
    (5 * five_pair_a3 == 6 * five_pair_a5)
    & (25 * five_pair_a1 == 9 * five_pair_a5)
)
missing_state = {
    int(index): next(iter(set(range(N)) - set(all_cycles[int(index)])))
    for index in five_cycle_indices
}
different_missing_state = np.array([
    missing_state[int(first)] != missing_state[int(second)]
    for first, second in five_cycle_pairs
], dtype=bool)
assert len(five_cycle_pairs) == 10296
assert int(np.sum(equal_barrier_five_pair)) == 1656
assert np.array_equal(equal_barrier_five_pair, ~different_missing_state)
assert np.array_equal(five_pair_a5 == 0, ~different_missing_state)
assert np.all(five_pair_a3[~different_missing_state] == 0)
assert np.all(five_pair_a1[~different_missing_state] == 0)

hamilton_two_five_supports = five_cycle_pairs[different_missing_state]
assert len(hamilton_two_five_supports) == 8640
five_sum_values = []
five_root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight
        * integer_symmetric[hamilton_two_five_supports[:, 0]]
        + third_weight
        * integer_symmetric[hamilton_two_five_supports[:, 1]]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight * integer_skew[hamilton_two_five_supports[:, 0]]
        + third_weight * integer_skew[hamilton_two_five_supports[:, 1]]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    five_sum_values.append(5 * node_a3 - 6 * node_a5)
    five_root_values.append(9 * node_a5 - 15 * node_a3 + 25 * node_a1)
stacked_five_sum_values = np.stack(five_sum_values)
stacked_five_root_values = np.stack(five_root_values)
for stacked_values in (
    stacked_five_sum_values, stacked_five_root_values
):
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    assert product_bound <= 80325000000
five_sum_numerators = interpolation_matrix @ stacked_five_sum_values
five_root_numerators = interpolation_matrix @ stacked_five_root_values
assert np.all(five_sum_numerators % interpolation_denominator == 0)
assert np.all(five_root_numerators % interpolation_denominator == 0)
five_sum_coefficients = five_sum_numerators // interpolation_denominator
five_root_coefficients = five_root_numerators // interpolation_denominator

five_largest_cover = 0
five_deepest_cover = 0
five_total_cells = 0
for support_index in range(len(hamilton_two_five_supports)):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        five_sum_coefficients[:, support_index].astype(object),
        five_root_coefficients[:, support_index].astype(object),
        simplex_transforms,
    )
    assert local_cells == 0
    five_largest_cover = max(five_largest_cover, cover_size)
    five_deepest_cover = max(five_deepest_cover, cover_depth)
    five_total_cells += cover_size
assert five_largest_cover == 64
assert five_deepest_cover == 5
assert five_total_cells == 228339

same_face_hamilton_two_five_supports = five_cycle_pairs[
    ~different_missing_state
]
assert len(same_face_hamilton_two_five_supports) == 1656
same_face_sum_values = []
same_face_root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight
        * integer_symmetric[same_face_hamilton_two_five_supports[:, 0]]
        + third_weight
        * integer_symmetric[same_face_hamilton_two_five_supports[:, 1]]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight
        * integer_skew[same_face_hamilton_two_five_supports[:, 0]]
        + third_weight
        * integer_skew[same_face_hamilton_two_five_supports[:, 1]]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    same_face_sum_values.append(5 * node_a3 - 6 * node_a5)
    same_face_root_values.append(
        9 * node_a5 - 15 * node_a3 + 25 * node_a1
    )
stacked_same_face_sum_values = np.stack(same_face_sum_values)
stacked_same_face_root_values = np.stack(same_face_root_values)
for stacked_values in (
    stacked_same_face_sum_values, stacked_same_face_root_values
):
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    assert product_bound < np.iinfo(np.int64).max
same_face_sum_numerators = (
    interpolation_matrix @ stacked_same_face_sum_values
)
same_face_root_numerators = (
    interpolation_matrix @ stacked_same_face_root_values
)
assert np.all(
    same_face_sum_numerators % interpolation_denominator == 0
)
assert np.all(
    same_face_root_numerators % interpolation_denominator == 0
)
same_face_sum_coefficients = (
    same_face_sum_numerators // interpolation_denominator
)
same_face_root_coefficients = (
    same_face_root_numerators // interpolation_denominator
)

degree_four_exponents = [
    exponent
    for exponent in product(range(5), repeat=3)
    if sum(exponent) == 4
]
degree_five_position = {
    exponent: index
    for index, exponent in enumerate(homogeneous_exponents)
}
zero_hamilton_positions = [
    index
    for index, (hamilton_power, _, _) in enumerate(homogeneous_exponents)
    if hamilton_power == 0
]
assert np.all(same_face_sum_coefficients[zero_hamilton_positions] == 0)
assert np.all(same_face_root_coefficients[zero_hamilton_positions] == 0)
quotient_positions = [
    degree_five_position[(hamilton_power + 1, second_power, third_power)]
    for hamilton_power, second_power, third_power in degree_four_exponents
]
same_face_sum_quotient = same_face_sum_coefficients[quotient_positions]
same_face_root_quotient = same_face_root_coefficients[quotient_positions]
degree_four_simplex_transforms = homogeneous_midpoint_transforms(
    degree_four_exponents
)

same_face_largest_cover = 0
same_face_deepest_cover = 0
same_face_total_cells = 0
for support_index in range(len(same_face_hamilton_two_five_supports)):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        same_face_sum_quotient[:, support_index].astype(object),
        same_face_root_quotient[:, support_index].astype(object),
        degree_four_simplex_transforms,
    )
    assert local_cells == 0
    same_face_largest_cover = max(
        same_face_largest_cover, cover_size
    )
    same_face_deepest_cover = max(
        same_face_deepest_cover, cover_depth
    )
    same_face_total_cells += cover_size
assert same_face_largest_cover == 55
assert same_face_deepest_cover == 5
assert same_face_total_cells == 41490

# The same factor-and-cover argument closes one Hamilton cycle plus two
# four-cycles.  A pure pair is nonsingular exactly when its union spans all
# six states.  Otherwise both barriers contain the Hamilton weight; their
# degree-four quotients admit smaller ordinary simplex covers.
four_cycle_pairs = np.array(
    list(combinations(four_cycle_indices, 2)), dtype=np.int64
)
four_pair_symmetric = (
    integer_symmetric[four_cycle_pairs[:, 0]]
    + integer_symmetric[four_cycle_pairs[:, 1]]
)
four_pair_skew = (
    integer_skew[four_cycle_pairs[:, 0]]
    + integer_skew[four_cycle_pairs[:, 1]]
)
four_pair_a5, four_pair_a3, four_pair_a1 = (
    batch_integer_pencil_coefficients(four_pair_symmetric, four_pair_skew)
)
nonsingular_four_pair = four_pair_a5 > 0
four_pair_union_size = np.array([
    len(
        set(all_cycles[int(first)])
        | set(all_cycles[int(second)])
    )
    for first, second in four_cycle_pairs
])
assert len(four_cycle_pairs) == 4005
assert int(np.sum(four_pair_union_size == 4)) == 225
assert int(np.sum(four_pair_union_size == 5)) == 2160
assert int(np.sum(four_pair_union_size == 6)) == 1620
assert np.array_equal(nonsingular_four_pair, four_pair_union_size == 6)
assert np.all(four_pair_a3[~nonsingular_four_pair] == 0)
assert np.all(four_pair_a1[~nonsingular_four_pair] == 0)

two_four_sum_values = []
two_four_root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight * integer_symmetric[four_cycle_pairs[:, 0]]
        + third_weight * integer_symmetric[four_cycle_pairs[:, 1]]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight * integer_skew[four_cycle_pairs[:, 0]]
        + third_weight * integer_skew[four_cycle_pairs[:, 1]]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    two_four_sum_values.append(5 * node_a3 - 6 * node_a5)
    two_four_root_values.append(
        9 * node_a5 - 15 * node_a3 + 25 * node_a1
    )
stacked_two_four_sum_values = np.stack(two_four_sum_values)
stacked_two_four_root_values = np.stack(two_four_root_values)
for stacked_values in (
    stacked_two_four_sum_values, stacked_two_four_root_values
):
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    assert product_bound < np.iinfo(np.int64).max
two_four_sum_numerators = (
    interpolation_matrix @ stacked_two_four_sum_values
)
two_four_root_numerators = (
    interpolation_matrix @ stacked_two_four_root_values
)
assert np.all(two_four_sum_numerators % interpolation_denominator == 0)
assert np.all(two_four_root_numerators % interpolation_denominator == 0)
two_four_sum_coefficients = (
    two_four_sum_numerators // interpolation_denominator
)
two_four_root_coefficients = (
    two_four_root_numerators // interpolation_denominator
)

two_four_largest_cover = 0
two_four_deepest_cover = 0
two_four_total_cells = 0
for support_index in np.flatnonzero(nonsingular_four_pair):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        two_four_sum_coefficients[:, support_index].astype(object),
        two_four_root_coefficients[:, support_index].astype(object),
        simplex_transforms,
    )
    assert local_cells == 0
    two_four_largest_cover = max(two_four_largest_cover, cover_size)
    two_four_deepest_cover = max(two_four_deepest_cover, cover_depth)
    two_four_total_cells += cover_size
assert two_four_largest_cover == 133
assert two_four_deepest_cover == 5
assert two_four_total_cells == 58455

assert np.all(
    two_four_sum_coefficients[zero_hamilton_positions][
        :, ~nonsingular_four_pair
    ] == 0
)
assert np.all(
    two_four_root_coefficients[zero_hamilton_positions][
        :, ~nonsingular_four_pair
    ] == 0
)
two_four_sum_quotient = two_four_sum_coefficients[quotient_positions]
two_four_root_quotient = two_four_root_coefficients[quotient_positions]
singular_four_largest_cover = 0
singular_four_deepest_cover = 0
singular_four_total_cells = 0
for support_index in np.flatnonzero(~nonsingular_four_pair):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        two_four_sum_quotient[:, support_index].astype(object),
        two_four_root_quotient[:, support_index].astype(object),
        degree_four_simplex_transforms,
    )
    assert local_cells == 0
    singular_four_largest_cover = max(
        singular_four_largest_cover, cover_size
    )
    singular_four_deepest_cover = max(
        singular_four_deepest_cover, cover_depth
    )
    singular_four_total_cells += cover_size
assert singular_four_largest_cover == 49
assert singular_four_deepest_cover == 4
assert singular_four_total_cells == 43425

# The remaining mixed architecture has one Hamilton cycle, one four-cycle and
# one five-cycle.  Their pure short-cycle edge is nonsingular exactly when the
# two short cycles together visit all six states.  On a common five-state face
# both barriers again contain the Hamilton weight, so the degree-four quotient
# certificate applies without a special corner argument.
four_five_pairs = np.array(
    list(product(four_cycle_indices, five_cycle_indices)), dtype=np.int64
)
four_five_pair_symmetric = (
    integer_symmetric[four_five_pairs[:, 0]]
    + integer_symmetric[four_five_pairs[:, 1]]
)
four_five_pair_skew = (
    integer_skew[four_five_pairs[:, 0]]
    + integer_skew[four_five_pairs[:, 1]]
)
four_five_pair_a5, four_five_pair_a3, four_five_pair_a1 = (
    batch_integer_pencil_coefficients(
        four_five_pair_symmetric, four_five_pair_skew
    )
)
nonsingular_four_five_pair = four_five_pair_a5 > 0
four_five_pair_union_size = np.array([
    len(
        set(all_cycles[int(four_cycle)])
        | set(all_cycles[int(five_cycle)])
    )
    for four_cycle, five_cycle in four_five_pairs
])
assert len(four_five_pairs) == 12960
assert int(np.sum(four_five_pair_union_size == 5)) == 4320
assert int(np.sum(four_five_pair_union_size == 6)) == 8640
assert np.array_equal(
    nonsingular_four_five_pair, four_five_pair_union_size == 6
)
assert np.all(four_five_pair_a3[~nonsingular_four_five_pair] == 0)
assert np.all(four_five_pair_a1[~nonsingular_four_five_pair] == 0)

four_five_sum_values = []
four_five_root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight * integer_symmetric[four_five_pairs[:, 0]]
        + third_weight * integer_symmetric[four_five_pairs[:, 1]]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight * integer_skew[four_five_pairs[:, 0]]
        + third_weight * integer_skew[four_five_pairs[:, 1]]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    four_five_sum_values.append(5 * node_a3 - 6 * node_a5)
    four_five_root_values.append(
        9 * node_a5 - 15 * node_a3 + 25 * node_a1
    )
stacked_four_five_sum_values = np.stack(four_five_sum_values)
stacked_four_five_root_values = np.stack(four_five_root_values)
for stacked_values in (
    stacked_four_five_sum_values, stacked_four_five_root_values
):
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    assert product_bound < np.iinfo(np.int64).max
four_five_sum_numerators = (
    interpolation_matrix @ stacked_four_five_sum_values
)
four_five_root_numerators = (
    interpolation_matrix @ stacked_four_five_root_values
)
assert np.all(
    four_five_sum_numerators % interpolation_denominator == 0
)
assert np.all(
    four_five_root_numerators % interpolation_denominator == 0
)
four_five_sum_coefficients = (
    four_five_sum_numerators // interpolation_denominator
)
four_five_root_coefficients = (
    four_five_root_numerators // interpolation_denominator
)

four_five_largest_cover = 0
four_five_deepest_cover = 0
four_five_total_cells = 0
for support_index in np.flatnonzero(nonsingular_four_five_pair):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        four_five_sum_coefficients[:, support_index].astype(object),
        four_five_root_coefficients[:, support_index].astype(object),
        simplex_transforms,
    )
    assert local_cells == 0
    four_five_largest_cover = max(
        four_five_largest_cover, cover_size
    )
    four_five_deepest_cover = max(
        four_five_deepest_cover, cover_depth
    )
    four_five_total_cells += cover_size
assert four_five_largest_cover == 85
assert four_five_deepest_cover == 5
assert four_five_total_cells == 262998

assert np.all(
    four_five_sum_coefficients[zero_hamilton_positions][
        :, ~nonsingular_four_five_pair
    ] == 0
)
assert np.all(
    four_five_root_coefficients[zero_hamilton_positions][
        :, ~nonsingular_four_five_pair
    ] == 0
)
four_five_sum_quotient = four_five_sum_coefficients[quotient_positions]
four_five_root_quotient = four_five_root_coefficients[quotient_positions]
singular_four_five_largest_cover = 0
singular_four_five_deepest_cover = 0
singular_four_five_total_cells = 0
for support_index in np.flatnonzero(~nonsingular_four_five_pair):
    cover_size, cover_depth, local_cells = exact_simplex_cover(
        four_five_sum_quotient[:, support_index].astype(object),
        four_five_root_quotient[:, support_index].astype(object),
        degree_four_simplex_transforms,
    )
    assert local_cells == 0
    singular_four_five_largest_cover = max(
        singular_four_five_largest_cover, cover_size
    )
    singular_four_five_deepest_cover = max(
        singular_four_five_deepest_cover, cover_depth
    )
    singular_four_five_total_cells += cover_size
assert singular_four_five_largest_cover == 58
assert singular_four_five_deepest_cover == 4
assert singular_four_five_total_cells == 94536

# One Hamilton cycle, one three-cycle and one five-cycle have the same exact
# support dichotomy.  A pure short-cycle pair is nonsingular when it visits all
# six states.  If both cycles lie on a common five-state face, factor the
# Hamilton weight from both barriers and certify the degree-four quotients.
three_five_pairs = np.array(
    list(product(three_cycle_indices, five_cycle_indices)), dtype=np.int64
)
three_five_pair_symmetric = (
    integer_symmetric[three_five_pairs[:, 0]]
    + integer_symmetric[three_five_pairs[:, 1]]
)
three_five_pair_skew = (
    integer_skew[three_five_pairs[:, 0]]
    + integer_skew[three_five_pairs[:, 1]]
)
three_five_pair_a5, three_five_pair_a3, three_five_pair_a1 = (
    batch_integer_pencil_coefficients(
        three_five_pair_symmetric, three_five_pair_skew
    )
)
nonsingular_three_five_pair = three_five_pair_a5 > 0
extremal_three_five_pair = (
    nonsingular_three_five_pair
    & (5 * three_five_pair_a3 == 6 * three_five_pair_a5)
    & (25 * three_five_pair_a1 == 9 * three_five_pair_a5)
)
assert not np.any(extremal_three_five_pair)
three_five_pair_union_size = np.array([
    len(
        set(all_cycles[int(three_cycle)])
        | set(all_cycles[int(five_cycle)])
    )
    for three_cycle, five_cycle in three_five_pairs
])
assert len(three_five_pairs) == 5760
assert int(np.sum(three_five_pair_union_size == 5)) == 2880
assert int(np.sum(three_five_pair_union_size == 6)) == 2880
assert np.array_equal(
    nonsingular_three_five_pair, three_five_pair_union_size == 6
)
assert np.all(three_five_pair_a3[~nonsingular_three_five_pair] == 0)
assert np.all(three_five_pair_a1[~nonsingular_three_five_pair] == 0)

three_five_sum_values = []
three_five_root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight * integer_symmetric[three_five_pairs[:, 0]]
        + third_weight * integer_symmetric[three_five_pairs[:, 1]]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight * integer_skew[three_five_pairs[:, 0]]
        + third_weight * integer_skew[three_five_pairs[:, 1]]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    three_five_sum_values.append(5 * node_a3 - 6 * node_a5)
    three_five_root_values.append(
        9 * node_a5 - 15 * node_a3 + 25 * node_a1
    )
stacked_three_five_sum_values = np.stack(three_five_sum_values)
stacked_three_five_root_values = np.stack(three_five_root_values)
for stacked_values in (
    stacked_three_five_sum_values, stacked_three_five_root_values
):
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    assert product_bound < np.iinfo(np.int64).max
three_five_sum_numerators = (
    interpolation_matrix @ stacked_three_five_sum_values
)
three_five_root_numerators = (
    interpolation_matrix @ stacked_three_five_root_values
)
assert np.all(
    three_five_sum_numerators % interpolation_denominator == 0
)
assert np.all(
    three_five_root_numerators % interpolation_denominator == 0
)
three_five_sum_coefficients = (
    three_five_sum_numerators // interpolation_denominator
)
three_five_root_coefficients = (
    three_five_root_numerators // interpolation_denominator
)

three_five_corner_types = {}
three_five_corner_leading_types = set()
for support_index in np.flatnonzero(nonsingular_three_five_pair):
    corner_polynomials = []
    for coefficients in (
        three_five_sum_coefficients[:, support_index],
        three_five_root_coefficients[:, support_index],
    ):
        poly = {}
        for coefficient, exponent in zip(
            coefficients, homogeneous_exponents
        ):
            key = (exponent[0], exponent[2])
            poly[key] = poly.get(key, Fraction(0)) + Fraction(
                int(coefficient)
            )
        corner_polynomials.append({
            key: value for key, value in poly.items() if value
        })
    sum_poly, root_poly = corner_polynomials
    leading_key = (
        tuple(sorted(
            (exponent, coefficient)
            for exponent, coefficient in sum_poly.items()
            if sum(exponent) == 3
        )),
        tuple(sorted(
            (exponent, coefficient)
            for exponent, coefficient in root_poly.items()
            if sum(exponent) == 3
        )),
    )
    three_five_corner_leading_types.add(leading_key)
    direct = (
        tuple(sorted(sum_poly.items())),
        tuple(sorted(root_poly.items())),
    )
    swapped = (
        tuple(sorted(
            ((second, first), coefficient)
            for (first, second), coefficient in sum_poly.items()
        )),
        tuple(sorted(
            ((second, first), coefficient)
            for (first, second), coefficient in root_poly.items()
        )),
    )
    key = min(direct, swapped)
    three_five_corner_types[key] = (
        three_five_corner_types.get(key, 0) + 1
    )
assert len(three_five_corner_types) == 256
assert len(three_five_corner_leading_types) == 16
assert sum(three_five_corner_types.values()) == 2880

three_five_corner_height = Fraction(1, 64)
three_five_corner_ratio_patches = 16
three_five_corner_barrier_uses = {"sum": 0, "root": 0}
three_five_corner_largest_control = None
for sum_items, root_items in three_five_corner_types:
    sum_poly = dict(sum_items)
    root_poly = dict(root_items)
    for patch in range(three_five_corner_ratio_patches):
        substitutions = (
            {(0, 1): three_five_corner_height
             * Fraction(patch, three_five_corner_ratio_patches),
             (1, 1): three_five_corner_height
             * Fraction(1, three_five_corner_ratio_patches)},
            {(0, 1): three_five_corner_height
             * Fraction(three_five_corner_ratio_patches - patch,
                        three_five_corner_ratio_patches),
             (1, 1): -three_five_corner_height
             * Fraction(1, three_five_corner_ratio_patches)},
        )
        candidate_controls = {}
        for name, poly in (("sum", sum_poly), ("root", root_poly)):
            transformed = multivariate_compose(poly, substitutions)
            assert all(exponent[1] >= 3 for exponent in transformed)
            transformed = {
                (exponent[0], exponent[1] - 3): coefficient
                for exponent, coefficient in transformed.items()
            }
            candidate_controls[name] = max(
                tensor_bernstein_controls(transformed)
            )
        barrier, control = min(
            candidate_controls.items(), key=lambda item: item[1]
        )
        assert control < 0
        three_five_corner_barrier_uses[barrier] += 1
        if three_five_corner_largest_control is None:
            three_five_corner_largest_control = control
        else:
            three_five_corner_largest_control = max(
                three_five_corner_largest_control, control
            )
assert three_five_corner_barrier_uses == {"sum": 3636, "root": 460}
assert three_five_corner_largest_control == Fraction(
    -173013241, 8796093022208
)

three_five_largest_cover = 0
three_five_deepest_cover = 0
three_five_total_cells = 0
three_five_corner_local_cells = 0
for support_index in np.flatnonzero(nonsingular_three_five_pair):
    cover_size, cover_depth, corner_cells, local_cells = (
        exact_simplex_cover_with_corner(
            three_five_sum_coefficients[:, support_index].astype(object),
            three_five_root_coefficients[:, support_index].astype(object),
            simplex_transforms,
            1,
            three_five_corner_height,
        )
    )
    assert local_cells == 0
    three_five_largest_cover = max(
        three_five_largest_cover, cover_size
    )
    three_five_deepest_cover = max(
        three_five_deepest_cover, cover_depth
    )
    three_five_total_cells += cover_size
    three_five_corner_local_cells += corner_cells
assert three_five_largest_cover == 475
assert three_five_deepest_cover == 10
assert three_five_total_cells == 409194
assert three_five_corner_local_cells == 7992

assert np.all(
    three_five_sum_coefficients[zero_hamilton_positions][
        :, ~nonsingular_three_five_pair
    ] == 0
)
assert np.all(
    three_five_root_coefficients[zero_hamilton_positions][
        :, ~nonsingular_three_five_pair
    ] == 0
)
three_five_sum_quotient = three_five_sum_coefficients[quotient_positions]
three_five_root_quotient = three_five_root_coefficients[quotient_positions]
three_five_quotient_corner_types = {}
for support_index in np.flatnonzero(~nonsingular_three_five_pair):
    corner_polynomials = []
    for coefficients in (
        three_five_sum_quotient[:, support_index],
        three_five_root_quotient[:, support_index],
    ):
        poly = {}
        for coefficient, exponent in zip(
            coefficients, degree_four_exponents
        ):
            key = (exponent[0], exponent[2])
            poly[key] = poly.get(key, Fraction(0)) + Fraction(
                int(coefficient)
            )
        corner_polynomials.append({
            key: value for key, value in poly.items() if value
        })
    sum_poly, root_poly = corner_polynomials
    direct = (
        tuple(sorted(sum_poly.items())),
        tuple(sorted(root_poly.items())),
    )
    swapped = (
        tuple(sorted(
            ((second, first), coefficient)
            for (first, second), coefficient in sum_poly.items()
        )),
        tuple(sorted(
            ((second, first), coefficient)
            for (first, second), coefficient in root_poly.items()
        )),
    )
    key = min(direct, swapped)
    three_five_quotient_corner_types[key] = (
        three_five_quotient_corner_types.get(key, 0) + 1
    )
assert len(three_five_quotient_corner_types) == 236
assert sum(three_five_quotient_corner_types.values()) == 2880

three_five_quotient_corner_height = Fraction(1, 64)
three_five_quotient_corner_barrier_uses = {"sum": 0, "root": 0}
three_five_quotient_corner_factor_orders = set()
three_five_quotient_corner_largest_control = None
for sum_items, root_items in three_five_quotient_corner_types:
    sum_poly = dict(sum_items)
    root_poly = dict(root_items)
    for patch in range(three_five_corner_ratio_patches):
        substitutions = (
            {(0, 1): three_five_quotient_corner_height
             * Fraction(patch, three_five_corner_ratio_patches),
             (1, 1): three_five_quotient_corner_height
             * Fraction(1, three_five_corner_ratio_patches)},
            {(0, 1): three_five_quotient_corner_height
             * Fraction(three_five_corner_ratio_patches - patch,
                        three_five_corner_ratio_patches),
             (1, 1): -three_five_quotient_corner_height
             * Fraction(1, three_five_corner_ratio_patches)},
        )
        candidate_controls = {}
        candidate_orders = {}
        for name, poly in (("sum", sum_poly), ("root", root_poly)):
            transformed = multivariate_compose(poly, substitutions)
            factor_order = min(exponent[1] for exponent in transformed)
            transformed = {
                (exponent[0], exponent[1] - factor_order): coefficient
                for exponent, coefficient in transformed.items()
            }
            candidate_controls[name] = max(
                tensor_bernstein_controls(transformed)
            )
            candidate_orders[name] = factor_order
        barrier, control = min(
            candidate_controls.items(), key=lambda item: item[1]
        )
        assert control < 0
        three_five_quotient_corner_barrier_uses[barrier] += 1
        three_five_quotient_corner_factor_orders.add(
            candidate_orders[barrier]
        )
        if three_five_quotient_corner_largest_control is None:
            three_five_quotient_corner_largest_control = control
        else:
            three_five_quotient_corner_largest_control = max(
                three_five_quotient_corner_largest_control, control
            )
assert three_five_quotient_corner_barrier_uses == {
    "sum": 3568,
    "root": 208,
}
assert three_five_quotient_corner_factor_orders == {2}
assert three_five_quotient_corner_largest_control == Fraction(-13, 8192)

singular_three_five_largest_cover = 0
singular_three_five_deepest_cover = 0
singular_three_five_total_cells = 0
singular_three_five_corner_local_cells = 0
for support_index in np.flatnonzero(~nonsingular_three_five_pair):
    cover_size, cover_depth, corner_cells, local_cells = (
        exact_simplex_cover_with_corner(
            three_five_sum_quotient[:, support_index].astype(object),
            three_five_root_quotient[:, support_index].astype(object),
            degree_four_simplex_transforms,
            1,
            three_five_quotient_corner_height,
        )
    )
    assert local_cells == 0
    singular_three_five_largest_cover = max(
        singular_three_five_largest_cover, cover_size
    )
    singular_three_five_deepest_cover = max(
        singular_three_five_deepest_cover, cover_depth
    )
    singular_three_five_total_cells += cover_size
    singular_three_five_corner_local_cells += corner_cells
assert singular_three_five_largest_cover == 202
assert singular_three_five_deepest_cover == 9
assert singular_three_five_total_cells == 159012
assert singular_three_five_corner_local_cells == 2316

# The last one-Hamilton length pair is one three-cycle and one four-cycle.
# The pure short-cycle edge has rank five exactly when its union visits all
# six states.  The other cases again factor by the Hamilton weight, but 60
# quotient corner patches close only nonstrictly; record them separately.
three_four_pairs = np.array(
    list(product(three_cycle_indices, four_cycle_indices)), dtype=np.int64
)
three_four_pair_symmetric = (
    integer_symmetric[three_four_pairs[:, 0]]
    + integer_symmetric[three_four_pairs[:, 1]]
)
three_four_pair_skew = (
    integer_skew[three_four_pairs[:, 0]]
    + integer_skew[three_four_pairs[:, 1]]
)
three_four_pair_a5, three_four_pair_a3, three_four_pair_a1 = (
    batch_integer_pencil_coefficients(
        three_four_pair_symmetric, three_four_pair_skew
    )
)
nonsingular_three_four_pair = three_four_pair_a5 > 0
extremal_three_four_pair = (
    nonsingular_three_four_pair
    & (5 * three_four_pair_a3 == 6 * three_four_pair_a5)
    & (25 * three_four_pair_a1 == 9 * three_four_pair_a5)
)
three_four_pair_union_size = np.array([
    len(set(all_cycles[int(three_cycle)])
        | set(all_cycles[int(four_cycle)]))
    for three_cycle, four_cycle in three_four_pairs
])
assert len(three_four_pairs) == 3600
assert int(np.sum(three_four_pair_union_size == 4)) == 720
assert int(np.sum(three_four_pair_union_size == 5)) == 2160
assert int(np.sum(three_four_pair_union_size == 6)) == 720
assert np.array_equal(
    nonsingular_three_four_pair, three_four_pair_union_size == 6
)
assert not np.any(extremal_three_four_pair)
assert np.all(three_four_pair_a3[~nonsingular_three_four_pair] == 0)
assert np.all(three_four_pair_a1[~nonsingular_three_four_pair] == 0)

three_four_sum_values = []
three_four_root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight * integer_symmetric[three_four_pairs[:, 0]]
        + third_weight * integer_symmetric[three_four_pairs[:, 1]]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight * integer_skew[three_four_pairs[:, 0]]
        + third_weight * integer_skew[three_four_pairs[:, 1]]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    three_four_sum_values.append(5 * node_a3 - 6 * node_a5)
    three_four_root_values.append(
        9 * node_a5 - 15 * node_a3 + 25 * node_a1
    )
stacked_three_four_sum_values = np.stack(three_four_sum_values)
stacked_three_four_root_values = np.stack(three_four_root_values)
for stacked_values in (
    stacked_three_four_sum_values, stacked_three_four_root_values
):
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    assert product_bound < np.iinfo(np.int64).max
three_four_sum_numerators = (
    interpolation_matrix @ stacked_three_four_sum_values
)
three_four_root_numerators = (
    interpolation_matrix @ stacked_three_four_root_values
)
assert np.all(
    three_four_sum_numerators % interpolation_denominator == 0
)
assert np.all(
    three_four_root_numerators % interpolation_denominator == 0
)
three_four_sum_coefficients = (
    three_four_sum_numerators // interpolation_denominator
)
three_four_root_coefficients = (
    three_four_root_numerators // interpolation_denominator
)


def short_corner_types(sum_coefficients, root_coefficients, exponents, mask):
    """Return corner-polynomial types, modulo exchanging small weights."""
    types = {}
    leading_types = set()
    for support_index in np.flatnonzero(mask):
        polynomials = []
        for coefficients in (
            sum_coefficients[:, support_index],
            root_coefficients[:, support_index],
        ):
            poly = {}
            for coefficient, exponent in zip(coefficients, exponents):
                key = (exponent[0], exponent[2])
                poly[key] = poly.get(key, Fraction(0)) + Fraction(
                    int(coefficient)
                )
            polynomials.append({
                key: value for key, value in poly.items() if value
            })
        sum_poly, root_poly = polynomials
        leading_types.add((
            tuple(sorted(
                (exponent, coefficient)
                for exponent, coefficient in sum_poly.items()
                if sum(exponent) == 3
            )),
            tuple(sorted(
                (exponent, coefficient)
                for exponent, coefficient in root_poly.items()
                if sum(exponent) == 3
            )),
        ))
        direct = (
            tuple(sorted(sum_poly.items())), tuple(sorted(root_poly.items()))
        )
        swapped = (
            tuple(sorted(
                ((second, first), coefficient)
                for (first, second), coefficient in sum_poly.items()
            )),
            tuple(sorted(
                ((second, first), coefficient)
                for (first, second), coefficient in root_poly.items()
            )),
        )
        key = min(direct, swapped)
        types[key] = types.get(key, 0) + 1
    return types, leading_types


def certify_corner_types(types, height, patches, strict=True):
    """Certify rational corner patches and report the selected controls."""
    uses = {"sum": 0, "root": 0}
    largest_strict_control = None
    zero_patches = 0
    factor_orders = set()
    for sum_items, root_items in types:
        for patch in range(patches):
            substitutions = (
                {(0, 1): height * Fraction(patch, patches),
                 (1, 1): height * Fraction(1, patches)},
                {(0, 1): height * Fraction(patches - patch, patches),
                 (1, 1): -height * Fraction(1, patches)},
            )
            candidates = {}
            orders = {}
            for name, poly in (
                ("sum", dict(sum_items)), ("root", dict(root_items))
            ):
                transformed = multivariate_compose(poly, substitutions)
                order = min(exponent[1] for exponent in transformed)
                transformed = {
                    (exponent[0], exponent[1] - order): coefficient
                    for exponent, coefficient in transformed.items()
                }
                candidates[name] = max(
                    tensor_bernstein_controls(transformed)
                )
                orders[name] = order
            barrier, control = min(
                candidates.items(), key=lambda item: item[1]
            )
            assert control < 0 if strict else control <= 0
            uses[barrier] += 1
            factor_orders.add(orders[barrier])
            if control == 0:
                zero_patches += 1
            elif largest_strict_control is None:
                largest_strict_control = control
            else:
                largest_strict_control = max(
                    largest_strict_control, control
                )
    return uses, largest_strict_control, zero_patches, factor_orders


three_four_corner_types, three_four_corner_leading_types = (
    short_corner_types(
        three_four_sum_coefficients,
        three_four_root_coefficients,
        homogeneous_exponents,
        nonsingular_three_four_pair,
    )
)
assert len(three_four_corner_types) == 62
assert len(three_four_corner_leading_types) == 8
assert sum(three_four_corner_types.values()) == 720
three_four_corner_height = Fraction(1, 64)
three_four_corner_patches = 16
(
    three_four_corner_barrier_uses,
    three_four_corner_largest_control,
    three_four_corner_zero_patches,
    three_four_corner_factor_orders,
) = certify_corner_types(
    three_four_corner_types,
    three_four_corner_height,
    three_four_corner_patches,
)
assert three_four_corner_barrier_uses == {"sum": 806, "root": 186}
assert three_four_corner_largest_control == Fraction(
    -2914375259, 140737488355328
)
assert three_four_corner_zero_patches == 0
assert three_four_corner_factor_orders == {3}

three_four_largest_cover = 0
three_four_deepest_cover = 0
three_four_total_cells = 0
three_four_corner_local_cells = 0
for support_index in np.flatnonzero(nonsingular_three_four_pair):
    cover_size, cover_depth, corner_cells, local_cells = (
        exact_simplex_cover_with_corner(
            three_four_sum_coefficients[:, support_index].astype(object),
            three_four_root_coefficients[:, support_index].astype(object),
            simplex_transforms,
            1,
            three_four_corner_height,
        )
    )
    assert local_cells == 0
    three_four_largest_cover = max(three_four_largest_cover, cover_size)
    three_four_deepest_cover = max(three_four_deepest_cover, cover_depth)
    three_four_total_cells += cover_size
    three_four_corner_local_cells += corner_cells
assert three_four_largest_cover == 520
assert three_four_deepest_cover == 10
assert three_four_total_cells == 175914
assert three_four_corner_local_cells == 3132

singular_three_four_pair = ~nonsingular_three_four_pair
assert np.all(
    three_four_sum_coefficients[zero_hamilton_positions][
        :, singular_three_four_pair
    ] == 0
)
assert np.all(
    three_four_root_coefficients[zero_hamilton_positions][
        :, singular_three_four_pair
    ] == 0
)
three_four_sum_quotient = three_four_sum_coefficients[quotient_positions]
three_four_root_quotient = three_four_root_coefficients[quotient_positions]
three_four_quotient_corner_types, _ = short_corner_types(
    three_four_sum_quotient,
    three_four_root_quotient,
    degree_four_exponents,
    singular_three_four_pair,
)
assert len(three_four_quotient_corner_types) == 232
assert sum(three_four_quotient_corner_types.values()) == 2880
(
    three_four_quotient_corner_barrier_uses,
    three_four_quotient_corner_largest_strict_control,
    three_four_quotient_corner_zero_patches,
    three_four_quotient_corner_factor_orders,
) = certify_corner_types(
    three_four_quotient_corner_types,
    three_four_corner_height,
    three_four_corner_patches,
    strict=False,
)
assert three_four_quotient_corner_barrier_uses == {
    "sum": 3476,
    "root": 236,
}
assert three_four_quotient_corner_largest_strict_control == Fraction(
    -89, 131072
)
assert three_four_quotient_corner_zero_patches == 60
assert three_four_quotient_corner_factor_orders == {2}

singular_three_four_largest_cover = 0
singular_three_four_deepest_cover = 0
singular_three_four_total_cells = 0
singular_three_four_corner_local_cells = 0
for support_index in np.flatnonzero(singular_three_four_pair):
    cover_size, cover_depth, corner_cells, local_cells = (
        exact_simplex_cover_with_corner(
            three_four_sum_quotient[:, support_index].astype(object),
            three_four_root_quotient[:, support_index].astype(object),
            degree_four_simplex_transforms,
            1,
            three_four_corner_height,
        )
    )
    assert local_cells == 0
    singular_three_four_largest_cover = max(
        singular_three_four_largest_cover, cover_size
    )
    singular_three_four_deepest_cover = max(
        singular_three_four_deepest_cover, cover_depth
    )
    singular_three_four_total_cells += cover_size
    singular_three_four_corner_local_cells += corner_cells
assert singular_three_four_largest_cover == 193
assert singular_three_four_deepest_cover == 9
assert singular_three_four_total_cells == 130356
assert singular_three_four_corner_local_cells == 2028

# One Hamilton cycle and two distinct three-cycles are the final
# one-Hamilton architecture.  Every pure three-cycle pair has an identically
# zero five-dimensional pencil: even a six-state union is two disconnected
# triangles.  Factor the Hamilton weight and certify both short-cycle corners.
three_cycle_pairs = np.array(
    list(combinations(three_cycle_indices, 2)), dtype=np.int64
)
three_pair_symmetric = (
    integer_symmetric[three_cycle_pairs[:, 0]]
    + integer_symmetric[three_cycle_pairs[:, 1]]
)
three_pair_skew = (
    integer_skew[three_cycle_pairs[:, 0]]
    + integer_skew[three_cycle_pairs[:, 1]]
)
three_pair_a5, three_pair_a3, three_pair_a1 = (
    batch_integer_pencil_coefficients(three_pair_symmetric, three_pair_skew)
)
three_pair_union_size = np.array([
    len(set(all_cycles[int(first)]) | set(all_cycles[int(second)]))
    for first, second in three_cycle_pairs
])
assert len(three_cycle_pairs) == 780
assert int(np.sum(three_pair_union_size == 3)) == 20
assert int(np.sum(three_pair_union_size == 4)) == 360
assert int(np.sum(three_pair_union_size == 5)) == 360
assert int(np.sum(three_pair_union_size == 6)) == 40
assert np.all(three_pair_a5 == 0)
assert np.all(three_pair_a3 == 0)
assert np.all(three_pair_a1 == 0)

hamilton_three_pair_sum_values = []
hamilton_three_pair_root_values = []
for first_weight, second_weight, third_weight in homogeneous_exponents:
    symmetric = (
        first_weight * integer_symmetric[standard_hamilton_index]
        + second_weight * integer_symmetric[three_cycle_pairs[:, 0]]
        + third_weight * integer_symmetric[three_cycle_pairs[:, 1]]
    )
    skew = (
        first_weight * integer_skew[standard_hamilton_index]
        + second_weight * integer_skew[three_cycle_pairs[:, 0]]
        + third_weight * integer_skew[three_cycle_pairs[:, 1]]
    )
    node_a5, node_a3, node_a1 = batch_integer_pencil_coefficients(
        symmetric, skew
    )
    hamilton_three_pair_sum_values.append(5 * node_a3 - 6 * node_a5)
    hamilton_three_pair_root_values.append(
        9 * node_a5 - 15 * node_a3 + 25 * node_a1
    )
stacked_hamilton_three_pair_sum_values = np.stack(
    hamilton_three_pair_sum_values
)
stacked_hamilton_three_pair_root_values = np.stack(
    hamilton_three_pair_root_values
)
for stacked_values in (
    stacked_hamilton_three_pair_sum_values,
    stacked_hamilton_three_pair_root_values,
):
    product_bound = (
        len(homogeneous_exponents)
        * interpolation_entry_bound
        * int(np.max(np.abs(stacked_values)))
    )
    assert product_bound < np.iinfo(np.int64).max
hamilton_three_pair_sum_numerators = (
    interpolation_matrix @ stacked_hamilton_three_pair_sum_values
)
hamilton_three_pair_root_numerators = (
    interpolation_matrix @ stacked_hamilton_three_pair_root_values
)
assert np.all(
    hamilton_three_pair_sum_numerators % interpolation_denominator == 0
)
assert np.all(
    hamilton_three_pair_root_numerators % interpolation_denominator == 0
)
hamilton_three_pair_sum_coefficients = (
    hamilton_three_pair_sum_numerators // interpolation_denominator
)
hamilton_three_pair_root_coefficients = (
    hamilton_three_pair_root_numerators // interpolation_denominator
)
assert np.all(
    hamilton_three_pair_sum_coefficients[zero_hamilton_positions] == 0
)
assert np.all(
    hamilton_three_pair_root_coefficients[zero_hamilton_positions] == 0
)
hamilton_three_pair_sum_quotient = (
    hamilton_three_pair_sum_coefficients[quotient_positions]
)
hamilton_three_pair_root_quotient = (
    hamilton_three_pair_root_coefficients[quotient_positions]
)

# Orient every unordered support both ways so the corner proof explicitly
# covers both pure-three-cycle vertices.
swap_degree_four_positions = np.array([
    degree_four_exponents.index((exponent[0], exponent[2], exponent[1]))
    for exponent in degree_four_exponents
])
oriented_three_pair_sum_quotient = np.concatenate((
    hamilton_three_pair_sum_quotient,
    hamilton_three_pair_sum_quotient[swap_degree_four_positions],
), axis=1)
oriented_three_pair_root_quotient = np.concatenate((
    hamilton_three_pair_root_quotient,
    hamilton_three_pair_root_quotient[swap_degree_four_positions],
), axis=1)
hamilton_three_pair_corner_types, _ = short_corner_types(
    oriented_three_pair_sum_quotient,
    oriented_three_pair_root_quotient,
    degree_four_exponents,
    np.ones(2 * len(three_cycle_pairs), dtype=bool),
)
assert len(hamilton_three_pair_corner_types) == 126
assert sum(hamilton_three_pair_corner_types.values()) == 1560
hamilton_three_pair_corner_height = Fraction(1, 64)
(
    hamilton_three_pair_corner_barrier_uses,
    hamilton_three_pair_corner_largest_strict_control,
    hamilton_three_pair_corner_zero_patches,
    hamilton_three_pair_corner_factor_orders,
) = certify_corner_types(
    hamilton_three_pair_corner_types,
    hamilton_three_pair_corner_height,
    16,
    strict=False,
)
assert hamilton_three_pair_corner_barrier_uses == {
    "sum": 1844,
    "root": 172,
}
assert hamilton_three_pair_corner_largest_strict_control == Fraction(
    -1, 32768
)
assert hamilton_three_pair_corner_zero_patches == 66
assert hamilton_three_pair_corner_factor_orders == {2}

hamilton_three_pair_largest_cover = 0
hamilton_three_pair_deepest_cover = 0
hamilton_three_pair_total_cells = 0
hamilton_three_pair_corner_local_cells = 0
for support_index in range(len(three_cycle_pairs)):
    cover_size, cover_depth, corner_cells = (
        exact_simplex_cover_with_two_corners(
            hamilton_three_pair_sum_quotient[
                :, support_index
            ].astype(object),
            hamilton_three_pair_root_quotient[
                :, support_index
            ].astype(object),
            degree_four_simplex_transforms,
            (1, 2),
            hamilton_three_pair_corner_height,
        )
    )
    hamilton_three_pair_largest_cover = max(
        hamilton_three_pair_largest_cover, cover_size
    )
    hamilton_three_pair_deepest_cover = max(
        hamilton_three_pair_deepest_cover, cover_depth
    )
    hamilton_three_pair_total_cells += cover_size
    hamilton_three_pair_corner_local_cells += corner_cells
assert hamilton_three_pair_largest_cover == 313
assert hamilton_three_pair_deepest_cover == 9
assert hamilton_three_pair_total_cells == 46788
assert hamilton_three_pair_corner_local_cells == 906

# A directed simple cycle supplies all of D.  Adding any reversible stationary
# flow adds a positive-semidefinite form to H and therefore contracts every
# singular value of H^{-1/2} D H^{-1/2}.  Check the cycle Fourier spectrum and
# independent random reversible attachments.  The contraction itself is the
# analytic factorization given in the accompanying note.
cycle_spectrum_error = 0.0
attachment_excess = 0.0
attachment_cases = 0
attachment_rng = np.random.default_rng(20261022)
for cycle_length in range(3, N + 1):
    directed_cycle = tuple(range(cycle_length))
    base_flow = np.zeros((N, N), dtype=float)
    for source, target_state in zip(
        directed_cycle,
        directed_cycle[1:] + directed_cycle[:1],
    ):
        base_flow[source, target_state] += 1.0
    base_symmetric, base_skew = grounded_forms_from_flow(base_flow)
    observed_base = intrinsic_frequencies(base_symmetric, base_skew)
    cycle_frequencies = [
        1 / math.tan(math.pi * mode / cycle_length)
        for mode in range(1, (cycle_length - 1) // 2 + 1)
    ]
    expected_base = tuple((cycle_frequencies + [0.0, 0.0])[:2])
    cycle_spectrum_error = max(
        cycle_spectrum_error,
        *(abs(observed - expected)
          for observed, expected in zip(observed_base, expected_base)),
    )

    for _ in range(60):
        stationary_flow = attachment_rng.lognormal(0.0, 1.0) * base_flow
        for source in range(N):
            for target_state in range(source + 1, N):
                reversible_weight = math.exp(
                    attachment_rng.uniform(-5.0, 2.0)
                )
                stationary_flow[source, target_state] += reversible_weight
                stationary_flow[target_state, source] += reversible_weight
        symmetric, skew = grounded_forms_from_flow(stationary_flow)
        observed = intrinsic_frequencies(symmetric, skew)
        attachment_excess = max(
            attachment_excess,
            *(value - ceiling
              for value, ceiling in zip(observed, expected_base)),
        )
        attachment_cases += 1

assert cycle_spectrum_error < 2e-12
assert attachment_excess < 2e-12

# The same contraction theorem combines with the exact two-cycle classification:
# a connected base formed from two arbitrarily weighted directed cycles stays
# below sqrt(3/5) after any number of reversible attachments.  Check this
# corollary independently of the one-cycle Fourier cases above.
two_cycle_attachment_cases = 0
two_cycle_attachment_excess = 0.0
two_cycle_bound_excess = 0.0
two_cycle_attachment_rng = np.random.default_rng(20261023)
for _ in range(240):
    first_cycle, second_cycle = cycle_pairs[
        int(two_cycle_attachment_rng.integers(len(cycle_pairs)))
    ]
    cycle_weights = np.exp(two_cycle_attachment_rng.uniform(-4.0, 4.0, 2))
    base_flow = np.zeros((N, N), dtype=float)
    for cycle_weight, cycle in zip(
        cycle_weights, (first_cycle, second_cycle)
    ):
        for source, target_state in zip(cycle, cycle[1:] + cycle[:1]):
            base_flow[source, target_state] += cycle_weight
    base_symmetric, base_skew = grounded_forms_from_flow(base_flow)
    base_frequencies = intrinsic_frequencies(base_symmetric, base_skew)

    stationary_flow = base_flow.copy()
    for source in range(N):
        for target_state in range(source + 1, N):
            if two_cycle_attachment_rng.random() < 0.65:
                reversible_weight = math.exp(
                    two_cycle_attachment_rng.uniform(-5.0, 2.0)
                )
                stationary_flow[source, target_state] += reversible_weight
                stationary_flow[target_state, source] += reversible_weight
    symmetric, skew = grounded_forms_from_flow(stationary_flow)
    observed = intrinsic_frequencies(symmetric, skew)
    two_cycle_attachment_excess = max(
        two_cycle_attachment_excess,
        *(value - ceiling
          for value, ceiling in zip(observed, base_frequencies)),
    )
    two_cycle_bound_excess = max(
        two_cycle_bound_excess,
        observed[1] - math.sqrt(Fraction(3, 5)),
    )
    two_cycle_attachment_cases += 1

assert two_cycle_attachment_excess < 2e-12
assert two_cycle_bound_excess < 2e-11

# Check the remaining base-support geometry explicitly.  If two cycle supports
# are disconnected on six states, their lengths can only be 3+3 or 2+4 (up to
# shorter unused supports).  The 3+3 case has second frequency 1/sqrt(3); the
# 2+4 case has only one positive frequency.  Complete reversible attachments
# connect the final flow and must contract these semidefinite-base spectra.
disconnected_attachment_cases = 0
disconnected_attachment_excess = 0.0
disconnected_attachment_rng = np.random.default_rng(20261024)
for case_index in range(120):
    if case_index % 2 == 0:
        base_cycles = ((0, 1, 2), (3, 4, 5))
    else:
        base_cycles = ((0, 1), (2, 3, 4, 5))
    base_flow = np.zeros((N, N), dtype=float)
    for cycle in base_cycles:
        if disconnected_attachment_rng.random() < 0.5:
            cycle = (cycle[0],) + tuple(reversed(cycle[1:]))
        cycle_weight = math.exp(
            disconnected_attachment_rng.uniform(-4.0, 4.0)
        )
        for source, target_state in zip(cycle, cycle[1:] + cycle[:1]):
            base_flow[source, target_state] += cycle_weight
    base_symmetric, base_skew = grounded_forms_from_flow(base_flow)
    base_frequencies = intrinsic_frequencies(base_symmetric, base_skew)

    stationary_flow = base_flow.copy()
    for source in range(N):
        for target_state in range(source + 1, N):
            reversible_weight = math.exp(
                disconnected_attachment_rng.uniform(-5.0, 2.0)
            )
            stationary_flow[source, target_state] += reversible_weight
            stationary_flow[target_state, source] += reversible_weight
    symmetric, skew = grounded_forms_from_flow(stationary_flow)
    observed = intrinsic_frequencies(symmetric, skew)
    disconnected_attachment_excess = max(
        disconnected_attachment_excess,
        *(value - ceiling
          for value, ceiling in zip(observed, base_frequencies)),
    )
    disconnected_attachment_cases += 1

assert disconnected_attachment_excess < 2e-12

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
print("unequal three-cycle symbolic identities: 3 exact coefficients")
print("three-cycle root barrier: -e2 (e1 e2 - 9 e3) <= 0")
print("coefficientwise Hamilton triple support orbits: 7")
print("first-cycle-normalized Hamilton triple supports: 123")
print(f"simple directed cycles enumerated: {len(all_cycles)}")
print(f"canonical connected cycle pairs classified exactly: {len(cycle_pairs)}")
print(f"distinct exact pencils: {len(pencil_types)}")
print(f"largest Bernstein cover: {largest_cover} intervals at depth {deepest_cover}")
print(f"five-state simple directed cycles: {len(five_state_cycles)}")
print(f"connected five-state cycle pairs: {len(five_state_pairs)}")
print(f"distinct five-state pencils: {len(five_state_pencil_types)}")
print("largest five-state Bernstein cover: "
      f"{five_state_largest_cover} intervals at depth "
      f"{five_state_deepest_cover}")
print(f"connected distinct equal-weight triples: {connected_triples}")
print(f"distinct normalized triple coefficient pairs: {len(triple_coefficient_pairs)}")
print(f"sharp equal-weight triple representatives: {len(equality_triples)}")
print("sharp equal-weight triple support orbits: 1")
print(f"one sharp equal-weight triple: {equality_triples[0]}")
print("equal-weight triple normalized pencil: t (t^2 + 3/5)^2")
print(f"unordered triples of directed three-cycles: {len(three_cycle_triples)}")
print("connected directed-three-cycle triples: "
      f"{int(np.sum(connected_triangle_triples))}")
print(f"directed-three-cycle quotient types: {len(triangle_a5_types)}")
print("directed-three-cycle normalized pencil: t (t^2 + 1/3)^2")
print("normalized Hamilton triples: 7021")
print("Hamilton triples without an extremal boundary pair: 6670")
print("largest Hamilton simplex cover: "
      f"{hamilton_largest_cover} triangles at depth "
      f"{hamilton_deepest_cover}")
print("Hamilton triples with an extremal boundary pair: 351")
print("extremal boundary polynomial types: 18")
print("largest extremal local Bernstein control: "
      f"{local_largest_control}")
print("largest extremal-boundary simplex cover: "
      f"{boundary_largest_cover} triangles at depth "
      f"{boundary_deepest_cover}")
print(f"extremal local cells used: {boundary_local_cells}")
print("normalized two-Hamilton/one-five-cycle supports: 17136")
print("mixed supports without an extremal boundary pair: 16848")
print("largest mixed simplex cover: "
      f"{mixed_largest_cover} triangles at depth "
      f"{mixed_deepest_cover}")
print("mixed supports with an extremal Hamilton boundary pair: 288")
print("mixed extremal boundary polynomial types: 12")
print("largest mixed extremal local Bernstein control: "
      f"{mixed_local_largest_control}")
print("largest mixed extremal-boundary simplex cover: "
      f"{mixed_boundary_largest_cover} triangles at depth "
      f"{mixed_boundary_deepest_cover}")
print(f"mixed extremal local cells used: {mixed_boundary_local_cells}")
print("normalized two-Hamilton/one-four-cycle supports: 10710")
print("four-cycle supports without an extremal boundary pair: 10530")
print("largest four-cycle simplex cover: "
      f"{four_largest_cover} triangles at depth "
      f"{four_deepest_cover}")
print("four-cycle supports with an extremal Hamilton boundary pair: 180")
print("four-cycle extremal boundary polynomial types: 12")
print("largest regular four-cycle local Bernstein control: "
      f"{four_regular_largest_control}")
print("largest exceptional four-cycle local Bernstein control: "
      f"{four_exceptional_largest_control}")
print("largest four-cycle extremal-boundary simplex cover: "
      f"{four_boundary_largest_cover} triangles at depth "
      f"{four_boundary_deepest_cover}")
print(f"four-cycle extremal local cells used: {four_boundary_local_cells}")
print("normalized two-Hamilton/one-three-cycle supports: 4760")
print("three-cycle supports without an extremal Hamilton pair: 4680")
print("three-cycle short-corner polynomial types: 230")
print("three-cycle short-corner leading cubic types: 19")
print("three-cycle short-corner barrier patches: "
      f"{three_corner_barrier_uses}")
print("largest three-cycle corner Bernstein control: "
      f"{three_corner_largest_control}")
print("three-cycle extremal Hamilton-edge polynomial types: 4")
print("largest three-cycle Hamilton-edge Bernstein control: "
      f"{three_edge_largest_control}")
print("largest three-cycle simplex cover: "
      f"{three_largest_cover} triangles at depth "
      f"{three_deepest_cover}")
print(f"three-cycle corner local cells used: {three_corner_local_cells}")
print(f"three-cycle edge local cells used: {three_edge_local_cells}")
print("five-cycle pairs classified exactly: 10296")
print("determinant-degenerate same-missing-state five-cycle pairs: 1656")
print("normalized one-Hamilton/two-five-cycle supports with different "
      "missing states: 8640")
print("largest one-Hamilton/two-five-cycle simplex cover: "
      f"{five_largest_cover} triangles at depth {five_deepest_cover}")
print(f"one-Hamilton/two-five-cycle terminal cells: {five_total_cells}")
print("same-face barrier factor: Hamilton weight")
print("largest same-face quotient simplex cover: "
      f"{same_face_largest_cover} triangles at depth "
      f"{same_face_deepest_cover}")
print(f"same-face quotient terminal cells: {same_face_total_cells}")
print("four-cycle pair union sizes: 225 on four states, 2160 on five, "
      "1620 on six")
print("largest one-Hamilton/two-four-cycle simplex cover: "
      f"{two_four_largest_cover} triangles at depth "
      f"{two_four_deepest_cover}")
print(f"one-Hamilton/two-four-cycle terminal cells: "
      f"{two_four_total_cells}")
print("largest singular four-cycle quotient simplex cover: "
      f"{singular_four_largest_cover} triangles at depth "
      f"{singular_four_deepest_cover}")
print(f"singular four-cycle quotient terminal cells: "
      f"{singular_four_total_cells}")
print("mixed four-/five-cycle pair union sizes: 4320 on five states, "
      "8640 on six")
print("largest mixed four-/five-cycle simplex cover: "
      f"{four_five_largest_cover} triangles at depth "
      f"{four_five_deepest_cover}")
print("mixed four-/five-cycle terminal cells: "
      f"{four_five_total_cells}")
print("largest singular mixed four-/five-cycle quotient simplex cover: "
      f"{singular_four_five_largest_cover} triangles at depth "
      f"{singular_four_five_deepest_cover}")
print("singular mixed four-/five-cycle quotient terminal cells: "
      f"{singular_four_five_total_cells}")
print("mixed three-/five-cycle pair union sizes: 2880 on five states, "
      "2880 on six")
print("extremal three-/five-cycle boundary pairs: "
      f"{int(np.sum(extremal_three_five_pair))}")
print("mixed three-/five-cycle corner polynomial types: "
      f"{len(three_five_corner_types)}")
print("mixed three-/five-cycle corner leading cubic types: "
      f"{len(three_five_corner_leading_types)}")
print("mixed three-/five-cycle corner barrier patches: "
      f"{three_five_corner_barrier_uses}")
print("largest mixed three-/five-cycle corner Bernstein control: "
      f"{three_five_corner_largest_control}")
print("largest mixed three-/five-cycle simplex cover: "
      f"{three_five_largest_cover} triangles at depth "
      f"{three_five_deepest_cover}")
print("mixed three-/five-cycle terminal cells: "
      f"{three_five_total_cells}")
print("mixed three-/five-cycle corner local cells used: "
      f"{three_five_corner_local_cells}")
print("largest singular mixed three-/five-cycle quotient simplex cover: "
      f"{singular_three_five_largest_cover} triangles at depth "
      f"{singular_three_five_deepest_cover}")
print("singular mixed three-/five-cycle quotient terminal cells: "
      f"{singular_three_five_total_cells}")
print("singular mixed three-/five-cycle quotient corner types: "
      f"{len(three_five_quotient_corner_types)}")
print("singular mixed three-/five-cycle quotient corner barrier patches: "
      f"{three_five_quotient_corner_barrier_uses}")
print("singular mixed three-/five-cycle quotient corner factor orders: "
      f"{sorted(three_five_quotient_corner_factor_orders)}")
print("largest singular mixed three-/five-cycle quotient corner control: "
      f"{three_five_quotient_corner_largest_control}")
print("singular mixed three-/five-cycle quotient corner local cells: "
      f"{singular_three_five_corner_local_cells}")
print("mixed three-/four-cycle pair union sizes: 720 on four states, "
      "2160 on five, 720 on six")
print("extremal three-/four-cycle boundary pairs: "
      f"{int(np.sum(extremal_three_four_pair))}")
print("mixed three-/four-cycle corner polynomial types: "
      f"{len(three_four_corner_types)}")
print("mixed three-/four-cycle corner leading cubic types: "
      f"{len(three_four_corner_leading_types)}")
print("mixed three-/four-cycle corner barrier patches: "
      f"{three_four_corner_barrier_uses}")
print("largest mixed three-/four-cycle corner Bernstein control: "
      f"{three_four_corner_largest_control}")
print("largest mixed three-/four-cycle simplex cover: "
      f"{three_four_largest_cover} triangles at depth "
      f"{three_four_deepest_cover}")
print("mixed three-/four-cycle terminal cells: "
      f"{three_four_total_cells}")
print("mixed three-/four-cycle corner local cells used: "
      f"{three_four_corner_local_cells}")
print("singular mixed three-/four-cycle quotient corner types: "
      f"{len(three_four_quotient_corner_types)}")
print("singular mixed three-/four-cycle quotient corner barrier patches: "
      f"{three_four_quotient_corner_barrier_uses}")
print("singular mixed three-/four-cycle quotient zero-control patches: "
      f"{three_four_quotient_corner_zero_patches}")
print("largest strict singular mixed three-/four-cycle quotient control: "
      f"{three_four_quotient_corner_largest_strict_control}")
print("singular mixed three-/four-cycle quotient factor orders: "
      f"{sorted(three_four_quotient_corner_factor_orders)}")
print("largest singular mixed three-/four-cycle quotient simplex cover: "
      f"{singular_three_four_largest_cover} triangles at depth "
      f"{singular_three_four_deepest_cover}")
print("singular mixed three-/four-cycle quotient terminal cells: "
      f"{singular_three_four_total_cells}")
print("singular mixed three-/four-cycle quotient corner local cells: "
      f"{singular_three_four_corner_local_cells}")
print("distinct three-cycle pair union sizes: 20 on three states, "
      "360 on four, 360 on five, 40 on six")
print("pure three-cycle pair pencils identically zero: "
      f"{len(three_cycle_pairs)}")
print("oriented one-Hamilton/two-three-cycle corner types: "
      f"{len(hamilton_three_pair_corner_types)}")
print("one-Hamilton/two-three-cycle corner barrier patches: "
      f"{hamilton_three_pair_corner_barrier_uses}")
print("one-Hamilton/two-three-cycle zero-control corner patches: "
      f"{hamilton_three_pair_corner_zero_patches}")
print("largest strict one-Hamilton/two-three-cycle corner control: "
      f"{hamilton_three_pair_corner_largest_strict_control}")
print("one-Hamilton/two-three-cycle corner factor orders: "
      f"{sorted(hamilton_three_pair_corner_factor_orders)}")
print("largest one-Hamilton/two-three-cycle quotient simplex cover: "
      f"{hamilton_three_pair_largest_cover} triangles at depth "
      f"{hamilton_three_pair_deepest_cover}")
print("one-Hamilton/two-three-cycle quotient terminal cells: "
      f"{hamilton_three_pair_total_cells}")
print("one-Hamilton/two-three-cycle corner local cells: "
      f"{hamilton_three_pair_corner_local_cells}")
print(f"directed-cycle Fourier cases: {N - 2}")
print(f"maximum directed-cycle spectrum discrepancy: {cycle_spectrum_error:.3e}")
print(f"random reversible-attachment cases: {attachment_cases}")
print(f"maximum attachment excess over cycle spectrum: {attachment_excess:.3e}")
print(f"random two-cycle attachment cases: {two_cycle_attachment_cases}")
print("maximum two-cycle attachment excess over base spectrum: "
      f"{two_cycle_attachment_excess:.3e}")
print("maximum attached two-cycle excess over sqrt(3/5): "
      f"{two_cycle_bound_excess:.3e}")
print("random disconnected-base attachment cases: "
      f"{disconnected_attachment_cases}")
print("maximum disconnected-base attachment excess: "
      f"{disconnected_attachment_excess:.3e}")
print("random stationary-law cases: 300")
print(f"maximum spectrum/formula discrepancy: {max_spectrum_error:.3e}")
print(f"maximum stationarity residual: {max_stationarity_error:.3e}")
print("portable assertion ceilings: spectrum 2e-11; stationarity 2e-10")
print(f"largest sampled second frequency: {largest_second:.12f}")
print(f"sharp sparse-cycle bound sqrt(3/5): {sharp_second:.12f}")
print(f"target cot(pi/5): {target:.12f}")
print("PASS")
