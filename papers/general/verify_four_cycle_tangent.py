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


def trim(poly):
    poly = [Fraction(value) for value in poly]
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def poly_add(left, right):
    out = [Fraction(0)] * max(len(left), len(right))
    for index in range(len(out)):
        out[index] = (
            (left[index] if index < len(left) else 0)
            + (right[index] if index < len(right) else 0)
        )
    return trim(out)


def poly_mul(left, right):
    out = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            out[i + j] += a * b
    return trim(out)


def interpolate(values):
    """Interpolate values at 0,...,5 in the monomial basis."""
    out = [Fraction(0)]
    for node, value in enumerate(values):
        basis, denominator = [Fraction(1)], 1
        for other in range(len(values)):
            if other != node:
                basis = poly_mul(basis, [-other, 1])
                denominator *= node - other
        out = poly_add(
            out, [coefficient * Fraction(value, denominator)
                  for coefficient in basis]
        )
    return out


def poly_divmod(dividend, divisor):
    dividend, divisor = trim(dividend), trim(divisor)
    quotient = [Fraction(0)] * max(1, len(dividend) - len(divisor) + 1)
    while len(dividend) >= len(divisor) and dividend != [0]:
        degree = len(dividend) - len(divisor)
        scale = dividend[-1] / divisor[-1]
        quotient[degree] = scale
        for index, value in enumerate(divisor):
            dividend[index + degree] -= scale * value
        dividend = trim(dividend)
    return trim(quotient), dividend


def derivative(poly):
    return trim([index * poly[index] for index in range(1, len(poly))] or [0])


def square_free(poly):
    left, right = trim(poly), derivative(poly)
    while right != [0]:
        _, remainder = poly_divmod(left, right)
        left, right = right, remainder
    gcd = [value / left[-1] for value in left]
    quotient, remainder = poly_divmod(poly, gcd)
    assert remainder == [0]
    return quotient


def evaluate(poly, point):
    value = Fraction(0)
    for coefficient in reversed(poly):
        value = value * point + coefficient
    return value


def sturm_sequence(poly):
    sequence = [trim(poly), derivative(poly)]
    while sequence[-1] != [0]:
        _, remainder = poly_divmod(sequence[-2], sequence[-1])
        if remainder == [0]:
            break
        sequence.append([-value for value in remainder])
    return sequence


def variations(sequence, point):
    signs = [1 if value > 0 else -1
             for poly in sequence if (value := evaluate(poly, point))]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def positive_root_intervals(poly):
    while len(poly) > 1 and poly[0] == 0:
        poly = poly[1:]
    poly = square_free(poly)
    upper = Fraction(2) + max(
        (abs(value / poly[-1]) for value in poly[:-1]), default=0
    )
    sequence = sturm_sequence(poly)
    root_count = variations(sequence, 0) - variations(sequence, upper)
    pending = [(Fraction(0), upper, root_count)]
    intervals = []
    while pending:
        left, right, count = pending.pop()
        if count == 0:
            continue
        if count == 1 and left > 0:
            intervals.append((left, right))
            continue
        middle = (left + right) / 2
        while evaluate(poly, middle) == 0:
            middle = (left + middle) / 2
        left_count = variations(sequence, left) - variations(sequence, middle)
        pending.extend(((middle, right, count - left_count),
                        (left, middle, left_count)))
    return sorted(intervals), upper


def certify_barrier_pair(sum_barrier, root_barrier):
    """Prove A<=0 or B<=0 throughout the positive epsilon ray."""
    if trim(sum_barrier) == [0] or trim(root_barrier) == [0]:
        return 0, 1
    product_barrier = poly_mul(sum_barrier, root_barrier)
    intervals, upper = positive_root_intervals(product_barrier)
    samples = ([intervals[0][0] / 2]
               if intervals else [Fraction(1)])
    samples.extend(
        (left[1] + right[0]) / 2
        for left, right in zip(intervals, intervals[1:])
    )
    if intervals:
        samples.append(upper + 1)
    for epsilon in samples:
        assert evaluate(sum_barrier, epsilon) <= 0 \
            or evaluate(root_barrier, epsilon) <= 0
    return len(intervals), len(samples)


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
ray_types = Counter()
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

    a5_full = interpolate(
        tuple(values[0] for values in sampled_coefficients)
    )
    a3_full = interpolate(
        tuple(values[1] for values in sampled_coefficients)
    )
    a1_full = interpolate(
        tuple(values[2] for values in sampled_coefficients)
    )
    a5_full += [Fraction(0)] * (6 - len(a5_full))
    a3_full += [Fraction(0)] * (6 - len(a3_full))
    a1_full += [Fraction(0)] * (6 - len(a1_full))
    a5, a3, a1 = a5_full[:3], a3_full[:3], a1_full[:3]
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

    # If A=5a3-6a5 is nonpositive, the sum of the two squared
    # frequencies is at most 6/5.  If B=9a5-15a3+25a1 is nonpositive,
    # the squared-frequency polynomial is nonpositive at 3/5.  Either
    # condition puts its lower root at or below 3/5.
    assert a5_full[0] > 0 and all(value >= 0 for value in a5_full)
    sum_barrier = poly_add(
        [5 * value for value in a3_full],
        [-6 * value for value in a5_full],
    )
    root_barrier = poly_add(
        poly_add([9 * value for value in a5_full],
                 [-15 * value for value in a3_full]),
        [25 * value for value in a1_full],
    )
    ray_types[(tuple(sum_barrier), tuple(root_barrier))] += 1

assert len(all_cycles) == 409
directed_cycles = sum(len(cycle) >= 3 for cycle in all_cycles)
reversible_cycles = sum(len(cycle) == 2 for cycle in all_cycles)
assert (directed_cycles, reversible_cycles) == (394, 15)
assert len(tangent_types) == 48
assert tuple(equality_cycles) == BASE_CYCLES
assert strict_cycles == 406

maximum_positive_breakpoints = 0
maximum_sign_cells = 0
for sum_barrier, root_barrier in ray_types:
    breakpoints, cells = certify_barrier_pair(sum_barrier, root_barrier)
    maximum_positive_breakpoints = max(maximum_positive_breakpoints, breakpoints)
    maximum_sign_cells = max(maximum_sign_cells, cells)

assert len(ray_types) == 50
assert maximum_positive_breakpoints == 2
assert maximum_sign_cells == 3

print(f"simple directed cycles tested: {len(all_cycles)}")
print(f"genuinely directed cycles tested: {directed_cycles}")
print(f"reversible two-cycles tested: {reversible_cycles}")
print(f"exact fourth-cycle tangent types: {len(tangent_types)}")
print(f"exact fourth-cycle ray types: {len(ray_types)}")
print(f"maximum positive sign breakpoints: {maximum_positive_breakpoints}")
print(f"maximum open sign cells: {maximum_sign_cells}")
print("base normalized pencil: t (t^2 + 3/5)^2")
print("global ray ceiling: lower squared frequency <= 3/5")
print("universal lower-root tangent ceiling: -4/25")
print(f"equality directions: {len(equality_cycles)} repeated base cycles")
print(f"strict directions: {strict_cycles} cycles")
print("genuinely new fourth-cycle directions: 391 strict")
print("PASS")
