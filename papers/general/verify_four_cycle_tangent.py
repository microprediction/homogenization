"""Exact tangent certificate at the sharp three-cycle six-state flow."""

from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations


N = 6
GROUND = N - 1
BASE_CYCLES = (
    (0, 1, 2, 3),
    (0, 1, 4, 5),
    (2, 3, 4, 5),
)


def sign(perm):
    inversions = sum(
        perm[i] > perm[j]
        for i in range(len(perm))
        for j in range(i + 1, len(perm))
    )
    return -1 if inversions % 2 else 1


PERMUTATIONS = tuple(permutations(range(GROUND)))
SIGNS = tuple(sign(perm) for perm in PERMUTATIONS)


def determinant(matrix):
    return sum(
        permutation_sign
        * product(
            matrix[row][perm[row]]
            for row in range(GROUND)
        )
        for perm, permutation_sign in zip(PERMUTATIONS, SIGNS)
    )


def product(values):
    out = 1
    for value in values:
        out *= value
    return out


def integer_forms(cycle):
    """Return the integer matrices 2H and 2D for a unit cycle flow."""
    weights = [[0 for _ in range(N)] for _ in range(N)]
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        weights[source][target] += 1
    grounded = [[0 for _ in range(GROUND)] for _ in range(GROUND)]
    for row in range(GROUND):
        for column in range(GROUND):
            grounded[row][column] = (
                sum(weights[row]) if row == column else -weights[row][column]
            )
    symmetric = [[
        grounded[row][column] + grounded[column][row]
        for column in range(GROUND)
    ] for row in range(GROUND)]
    skew = [[
        grounded[row][column] - grounded[column][row]
        for column in range(GROUND)
    ] for row in range(GROUND)]
    return symmetric, skew


def matrix_add(left, right, multiplier=1):
    return [[
        left[row][column] + multiplier * right[row][column]
        for column in range(GROUND)
    ] for row in range(GROUND)]


def pencil_coefficients(symmetric, skew):
    """Recover a5, a3, a1 in det(t symmetric - skew)."""
    values = [
        determinant([[
            t * symmetric[row][column] - skew[row][column]
            for column in range(GROUND)
        ] for row in range(GROUND)])
        for t in (1, 2, 3)
    ]
    value_one, value_two, value_three = values
    a5 = (value_three - 4 * value_two + 5 * value_one) // 120
    a3 = (value_two - 2 * value_one) // 6 - 5 * a5
    a1 = value_one - a5 - a3
    assert 120 * a5 == value_three - 4 * value_two + 5 * value_one
    assert 6 * (a3 + 5 * a5) == value_two - 2 * value_one
    return a5, a3, a1


# A degree-five polynomial is determined by its values at 0,...,5.  These
# are the first three rows of the inverse Vandermonde matrix, giving its
# constant, linear and quadratic monomial coefficients.
INVERSE_VANDERMONDE_ROWS = (
    (Fraction(1), Fraction(0), Fraction(0), Fraction(0),
     Fraction(0), Fraction(0)),
    (Fraction(-137, 60), Fraction(5), Fraction(-5), Fraction(10, 3),
     Fraction(-5, 4), Fraction(1, 5)),
    (Fraction(15, 8), Fraction(-77, 12), Fraction(107, 12),
     Fraction(-13, 2), Fraction(61, 24), Fraction(-5, 12)),
)


def first_three_coefficients(values):
    return tuple(
        sum(weight * value for weight, value in zip(row, values))
        for row in INVERSE_VANDERMONDE_ROWS
    )


def ratio_expansion(numerator, denominator):
    """Constant through quadratic coefficients of numerator/denominator."""
    n0, n1, n2 = numerator
    d0, d1, d2 = denominator
    return (
        n0 / d0,
        n1 / d0 - n0 * d1 / d0**2,
        n2 / d0 - n1 * d1 / d0**2
        + n0 * (d1**2 / d0**3 - d2 / d0**2),
    )


all_cycles = []
for length in range(2, N + 1):
    for support in combinations(range(N), length):
        first = min(support)
        remaining = tuple(state for state in support if state != first)
        all_cycles.extend((first,) + perm for perm in permutations(remaining))

base_forms = tuple(integer_forms(cycle) for cycle in BASE_CYCLES)
base_symmetric = [[
    sum(forms[0][row][column] for forms in base_forms)
    for column in range(GROUND)
] for row in range(GROUND)]
base_skew = [[
    sum(forms[1][row][column] for forms in base_forms)
    for column in range(GROUND)
] for row in range(GROUND)]

base_coefficients = pencil_coefficients(base_symmetric, base_skew)
assert base_coefficients == (300, 360, 108)
assert (
    Fraction(base_coefficients[1], base_coefficients[0]),
    Fraction(base_coefficients[2], base_coefficients[0]),
) == (Fraction(6, 5), Fraction(9, 25))

bound = Fraction(-4, 25)
tangent_types = Counter()
equality_cycles = []
strict_cycles = 0
for cycle in all_cycles:
    cycle_symmetric, cycle_skew = integer_forms(cycle)
    sampled_coefficients = []
    for epsilon in range(6):
        symmetric = matrix_add(
            base_symmetric, cycle_symmetric, epsilon
        )
        skew = matrix_add(base_skew, cycle_skew, epsilon)
        sampled_coefficients.append(pencil_coefficients(symmetric, skew))

    a5 = first_three_coefficients(
        tuple(values[0] for values in sampled_coefficients)
    )
    a3 = first_three_coefficients(
        tuple(values[1] for values in sampled_coefficients)
    )
    a1 = first_three_coefficients(
        tuple(values[2] for values in sampled_coefficients)
    )
    sum_expansion = ratio_expansion(a3, a5)
    product_expansion = ratio_expansion(a1, a5)
    sum_zero, sum_linear, sum_quadratic = sum_expansion
    _, product_linear, product_quadratic = product_expansion

    # For squared roots z_+ and z_-, Delta=(z_+-z_-)^2.  The base has
    # the double root 3/5.  Exact cancellation of Delta's linear term
    # makes the lower-root right derivative
    # (sum_linear - sqrt(discriminant_quadratic))/2.
    discriminant_linear = 2 * sum_zero * sum_linear - 4 * product_linear
    discriminant_quadratic = (
        sum_linear**2 + 2 * sum_zero * sum_quadratic
        - 4 * product_quadratic
    )
    assert discriminant_linear == 0
    assert discriminant_quadratic >= 0
    tangent_types[(sum_linear, discriminant_quadratic)] += 1

    # Check (sum_linear - sqrt(discriminant_quadratic))/2 <= bound
    # without floating point or algebraic-number comparisons.
    required_root = sum_linear - 2 * bound
    if required_root < 0:
        strict_cycles += 1
    else:
        margin = discriminant_quadratic - required_root**2
        assert margin >= 0
        if margin == 0:
            equality_cycles.append(cycle)
        else:
            strict_cycles += 1

assert len(all_cycles) == 409
directed_cycles = sum(len(cycle) >= 3 for cycle in all_cycles)
reversible_cycles = sum(len(cycle) == 2 for cycle in all_cycles)
assert (directed_cycles, reversible_cycles) == (394, 15)
assert len(tangent_types) == 48
assert tuple(equality_cycles) == BASE_CYCLES
assert strict_cycles == 406

print(f"simple directed cycles tested: {len(all_cycles)}")
print(f"genuinely directed cycles tested: {directed_cycles}")
print(f"reversible two-cycles tested: {reversible_cycles}")
print(f"exact fourth-cycle tangent types: {len(tangent_types)}")
print("base normalized pencil: t (t^2 + 3/5)^2")
print("universal lower-root tangent ceiling: -4/25")
print(f"equality directions: {len(equality_cycles)} repeated base cycles")
print(f"strict directions: {strict_cycles} cycles")
print("genuinely new fourth-cycle directions: 391 strict")
print("PASS")
