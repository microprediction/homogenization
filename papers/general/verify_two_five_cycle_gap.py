"""Exact certificate for the second frequency of two overlapping five-cycles."""

from fractions import Fraction
from itertools import combinations, permutations
from math import comb


def gaussian_poly_trim(poly):
    while len(poly) > 1 and poly[-1] == (0, 0):
        poly.pop()
    return poly


def gaussian_poly_add(left, right):
    out = [(0, 0)] * max(len(left), len(right))
    for index in range(len(out)):
        a = left[index] if index < len(left) else (0, 0)
        b = right[index] if index < len(right) else (0, 0)
        out[index] = (a[0] + b[0], a[1] + b[1])
    return gaussian_poly_trim(out)


def gaussian_poly_mul(left, right):
    out = [(0, 0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            old = out[i + j]
            out[i + j] = (
                old[0] + a[0] * b[0] - a[1] * b[1],
                old[1] + a[0] * b[1] + a[1] * b[0],
            )
    return gaussian_poly_trim(out)


def matrix_mul(left, right):
    size = len(left)
    out = [[[(0, 0)] for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for pivot in range(size):
            if left[row][pivot] != [(0, 0)]:
                for column in range(size):
                    product = gaussian_poly_mul(
                        left[row][pivot], right[pivot][column]
                    )
                    out[row][column] = gaussian_poly_add(
                        out[row][column], product
                    )
    return out


def permutation_sign(perm):
    inversions = sum(
        perm[left] > perm[right]
        for left in range(len(perm))
        for right in range(left + 1, len(perm))
    )
    return -1 if inversions % 2 else 1


def determinant_poly(matrix):
    """Exact determinant of a Gaussian-integer polynomial matrix."""
    out = [(0, 0)]
    for perm in permutations(range(len(matrix))):
        term = [(permutation_sign(perm), 0)]
        for row, column in enumerate(perm):
            term = gaussian_poly_mul(term, matrix[row][column])
        out = gaussian_poly_add(out, term)
    return out


def pencil_determinant(cycles, weights, state_count):
    """Return det(2(tH-iD)) in ascending powers of t."""
    size = state_count - 1
    matrix = [[[(0, 0)] for _ in range(size)] for _ in range(size)]
    for cycle, weight in zip(cycles, weights):
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            if source < size:
                matrix[source][source] = gaussian_poly_add(
                    matrix[source][source], [(0, 0), (2 * weight, 0)]
                )
            if source < size and target < size:
                matrix[source][target] = gaussian_poly_add(
                    matrix[source][target], [(0, weight), (-weight, 0)]
                )
                matrix[target][source] = gaussian_poly_add(
                    matrix[target][source], [(0, -weight), (-weight, 0)]
                )
    determinant = determinant_poly(matrix)
    assert all(imaginary == 0 for _, imaginary in determinant)
    return determinant + [(0, 0)] * (size + 1 - len(determinant))


def normalized_pencil(cycles, state_count):
    """Return det(tH-iD)/det(H), in ascending powers of t."""
    determinant = pencil_determinant(cycles, (1, 1), state_count)
    leading = determinant[-1][0]
    return tuple(Fraction(real, leading) for real, _ in determinant)


def threshold_characteristic(first_cycle, second_cycle, state_count):
    """Coefficients in x of det(xI-2(iD-H)), as polynomials in r."""
    size = state_count - 1
    matrix = [[[(0, 0)] for _ in range(size)] for _ in range(size)]
    for cycle, r_degree in ((first_cycle, 0), (second_cycle, 1)):
        monomial = [(0, 0)] * r_degree
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            if source < size:
                matrix[source][source] = gaussian_poly_add(
                    matrix[source][source], monomial + [(-2, 0)]
                )
            if source < size and target < size:
                matrix[source][target] = gaussian_poly_add(
                    matrix[source][target], monomial + [(1, -1)]
                )
                matrix[target][source] = gaussian_poly_add(
                    matrix[target][source], monomial + [(1, 1)]
                )

    auxiliary = [
        [[(1, 0)] if row == column else [(0, 0)]
         for column in range(size)]
        for row in range(size)
    ]
    coefficients = [(1,)]
    # Faddeev-LeVerrier is exact over the Gaussian-integer polynomial ring.
    for order in range(1, size + 1):
        auxiliary = matrix_mul(matrix, auxiliary)
        trace = [(0, 0)]
        for index in range(size):
            trace = gaussian_poly_add(trace, auxiliary[index][index])
        coefficient = []
        for real, imaginary in trace:
            assert real % order == 0 and imaginary % order == 0
            coefficient.append((-real // order, -imaginary // order))
        assert all(imaginary == 0 for _, imaginary in coefficient)
        coefficients.append(tuple(real for real, _ in coefficient))
        for index in range(size):
            auxiliary[index][index] = gaussian_poly_add(
                auxiliary[index][index], coefficient
            )
    return tuple(coefficients)


def trim(poly):
    poly = [Fraction(value) for value in poly]
    while len(poly) > 1 and poly[-1] == 0:
        poly.pop()
    return poly


def poly_mul(left, right):
    out = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            out[i + j] += a * b
    return trim(out)


def poly_power(poly, exponent):
    out = [Fraction(1)]
    for _ in range(exponent):
        out = poly_mul(out, poly)
    return out


def poly_add(left, right):
    out = [Fraction(0)] * max(len(left), len(right))
    for index in range(len(out)):
        out[index] = (
            (left[index] if index < len(left) else 0)
            + (right[index] if index < len(right) else 0)
        )
    return trim(out)


def interpolate(values):
    """Interpolate values at consecutive nonnegative integer arguments."""
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


def squared_frequency_coefficients(cycles, state_count):
    """Coefficients in y with roots equal to squared positive frequencies."""
    # A five-cycle form has rank four, so the determinant has ratio degree at most four.
    size = state_count - 1
    evaluations = []
    for ratio in range(5):
        determinant = pencil_determinant(cycles, (1, ratio), state_count)
        evaluations.append(
            [determinant[degree][0]
             for degree in range(size % 2, size + 1, 2)]
        )
    return [
        interpolate([values[index] for values in evaluations])
        for index in range(len(evaluations[0]))
    ]


def shift_coefficients(coefficients, threshold):
    """Return coefficients after substituting y=z+threshold."""
    out = []
    for degree in range(len(coefficients)):
        shifted = [Fraction(0)]
        for source in range(degree, len(coefficients)):
            scale = comb(source, degree) * threshold ** (source - degree)
            shifted = poly_add(
                shifted, [scale * value for value in coefficients[source]]
            )
        out.append(shifted)
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
    if len(poly) == 1:
        return [Fraction(0)]
    return trim([index * poly[index] for index in range(1, len(poly))])


def gcd(left, right):
    left, right = trim(left), trim(right)
    while right != [0]:
        _, remainder = poly_divmod(left, right)
        left, right = right, remainder
    return trim([value / left[-1] for value in left])


def square_free(poly):
    quotient, remainder = poly_divmod(poly, gcd(poly, derivative(poly)))
    assert remainder == [0]
    return quotient


def square_free_lcm(left, right):
    right = square_free(right)
    quotient, remainder = poly_divmod(right, gcd(left, right))
    assert remainder == [0]
    return poly_mul(left, quotient)


def evaluate(poly, point):
    value = Fraction(0)
    for coefficient in reversed(poly):
        value = value * point + coefficient
    return value


def sign_variations(values):
    signs = [1 if value > 0 else -1 for value in values if value]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def sturm_sequence(poly):
    sequence = [trim(poly), derivative(poly)]
    while sequence[-1] != [0]:
        _, remainder = poly_divmod(sequence[-2], sequence[-1])
        if remainder == [0]:
            break
        sequence.append([-value for value in remainder])
    return sequence


def sturm_count(sequence, left, right):
    return (
        sign_variations(evaluate(poly, left) for poly in sequence)
        - sign_variations(evaluate(poly, right) for poly in sequence)
    )


def positive_root_intervals(poly):
    while len(poly) > 1 and poly[0] == 0:
        poly = poly[1:]
    poly = square_free(poly)
    upper = Fraction(2) + max(
        (abs(value / poly[-1]) for value in poly[:-1]), default=0
    )
    sequence = sturm_sequence(poly)
    pending = [(Fraction(0), upper,
                sturm_count(sequence, Fraction(0), upper))]
    intervals = []
    while pending:
        left, right, count = pending.pop()
        if count == 0:
            continue
        if count == 1 and left > 0:
            intervals.append((left, right))
            continue
        # Select an exact rational split that is not itself a root.
        for denominator in range(2, 30):
            middle = ((denominator - 1) * left + right) / denominator
            if evaluate(poly, middle) != 0:
                break
        left_count = sturm_count(sequence, left, middle)
        pending.extend(((middle, right, count - left_count),
                        (left, middle, left_count)))
    return sorted(intervals), upper


def certify_one_positive_root(coefficients):
    """Certify at most one positive root for every positive weight ratio."""
    breakpoints = [Fraction(1)]
    for coefficient in coefficients:
        if any(coefficient):
            reduced = list(coefficient)
            while len(reduced) > 1 and reduced[0] == 0:
                reduced = reduced[1:]
            if len(reduced) > 1:
                breakpoints = square_free_lcm(breakpoints, reduced)
    intervals, upper = positive_root_intervals(breakpoints)
    if intervals:
        samples = [intervals[0][0] / 2]
        samples.extend(
            (left[1] + right[0]) / 2
            for left, right in zip(intervals, intervals[1:])
        )
        samples.append(upper + 1)
    else:
        samples = [Fraction(1)]
    for ratio in samples:
        values = [evaluate(coefficient, ratio)
                  for coefficient in coefficients]
        assert all(value != 0 for coefficient, value
                   in zip(coefficients, values) if any(coefficient))
        assert sign_variations(values) <= 1
    return len(intervals), len(samples)


first_cycle = tuple(range(5))
pair_counts = {}
patterns = {}
for overlap in range(2, 6):
    state_count = 10 - overlap
    pair_count = 0
    for shared in combinations(range(5), overlap):
        support = shared + tuple(range(5, state_count))
        start = min(support)
        remaining = tuple(vertex for vertex in support if vertex != start)
        for tail in permutations(remaining):
            characteristic = threshold_characteristic(
                first_cycle, (start,) + tail, state_count
            )
            patterns[characteristic] = patterns.get(characteristic, 0) + 1
            pair_count += 1
    pair_counts[overlap] = pair_count

assert pair_counts == {2: 240, 3: 240, 4: 120, 5: 24}
assert sum(pair_counts.values()) == 624
assert len(patterns) == 70

maximum_roots = 0
maximum_cells = 0
for coefficients in patterns:
    roots, cells = certify_one_positive_root(coefficients)
    maximum_roots = max(maximum_roots, roots)
    maximum_cells = max(maximum_cells, cells)

# Stronger sharp ceilings hold when three or four states are shared.
overlap_thresholds = {3: Fraction(1, 2), 4: Fraction(1, 3)}
overlap_type_counts = {}
overlap_maximum_roots = {}
for overlap, squared_threshold in overlap_thresholds.items():
    state_count = 10 - overlap
    shifted_types = {}
    for shared in combinations(range(5), overlap):
        support = shared + tuple(range(5, state_count))
        start = min(support)
        remaining = tuple(vertex for vertex in support if vertex != start)
        for tail in permutations(remaining):
            squared = squared_frequency_coefficients(
                (first_cycle, (start,) + tail), state_count
            )
            shifted = shift_coefficients(squared, squared_threshold)
            key = tuple(tuple(coefficient) for coefficient in shifted)
            shifted_types[key] = shifted_types.get(key, 0) + 1
    assert sum(shifted_types.values()) == pair_counts[overlap]
    overlap_type_counts[overlap] = len(shifted_types)
    overlap_maximum_roots[overlap] = 0
    for shifted in shifted_types:
        roots, _ = certify_one_positive_root(shifted)
        overlap_maximum_roots[overlap] = max(
            overlap_maximum_roots[overlap], roots
        )

assert overlap_type_counts == {3: 16, 4: 13}
assert overlap_maximum_roots == {3: 2, 4: 3}

# The common-five-state stratum has one algebraic sharp pair of inverse types.
five_types = {}
for tail in permutations(range(1, 5)):
    squared = squared_frequency_coefficients(
        (first_cycle, (0,) + tail), 5
    )
    key = tuple(tuple(coefficient) for coefficient in squared)
    five_types[key] = five_types.get(key, 0) + 1
assert len(five_types) == 8
assert sorted(five_types.values()) == [1, 1, 1, 1, 5, 5, 5, 5]

sharp_five = tuple(tuple(coefficient) for coefficient in
                   squared_frequency_coefficients(
                       (first_cycle, (0, 2, 4, 1, 3)), 5
                   ))
inverse_sharp_five = tuple(tuple(reversed(coefficient))
                           for coefficient in sharp_five)
assert five_types[sharp_five] == five_types[inverse_sharp_five] == 1

five_nonalgebraic_maximum_roots = 0
five_nonalgebraic_maximum_cells = 0
for squared in five_types:
    if squared not in (sharp_five, inverse_sharp_five):
        roots, cells = certify_one_positive_root(
            shift_coefficients(squared, Fraction(1, 7))
        )
        five_nonalgebraic_maximum_roots = max(
            five_nonalgebraic_maximum_roots, roots
        )
        five_nonalgebraic_maximum_cells = max(
            five_nonalgebraic_maximum_cells, cells
        )

# For a sharp type, the squared frequencies have midpoint A/D and
# separation 2|F|/(sqrt(5)D).  The unique positive zero rho of F is where
# they collide.  The derivative signs follow from the rationalized identity.
c0, c1, c2 = [list(coefficient) for coefficient in sharp_five]
a_poly = [1, 2, 1, 0, 1]
f_poly = [-1, -2, -2, -1, 1]
d_poly = [1, 3, 1]
denominator = poly_mul(d_poly, d_poly)
assert c1 == [-10 * value for value in a_poly]
assert c2 == [5 * value for value in denominator]
discriminant = poly_add(
    poly_mul(c1, c1), [-4 * value for value in poly_mul(c0, c2)]
)
assert discriminant == [80 * value for value in poly_mul(f_poly, f_poly)]

u_poly = poly_add(
    poly_mul(derivative(a_poly), denominator),
    [-value for value in poly_mul(a_poly, derivative(denominator))],
)
w_poly = poly_add(
    poly_mul(derivative(f_poly), denominator),
    [-value for value in poly_mul(f_poly, derivative(denominator))],
)
assert u_poly == [2 * value for value in poly_mul(
    d_poly, [-2, -4, -3, 1, 3]
)]
assert w_poly == poly_mul(d_poly, [4, 6, 3, 5, 7])
rationalized = poly_add(
    [5 * value for value in poly_mul(u_poly, u_poly)],
    [-4 * value for value in poly_mul(w_poly, w_poly)],
)
assert rationalized == [-16 * value for value in poly_mul(
    [-1, 1, 1], poly_power(d_poly, 5)
)]

rho_intervals, _ = positive_root_intervals(f_poly)
assert len(rho_intervals) == 1
assert evaluate(f_poly, Fraction(2)) < 0 < evaluate(f_poly, Fraction(3))
gamma_poly = [5, -20, 6, -100, 125]
gamma_relation = [Fraction(0)]
for coefficient, a_power, d_power in (
    (125, 4, 0), (-100, 3, 1), (6, 2, 2), (-20, 1, 3), (5, 0, 4)
):
    term = poly_mul(poly_power(a_poly, a_power),
                    poly_power(denominator, d_power))
    gamma_relation = poly_add(
        gamma_relation, [coefficient * value for value in term]
    )
_, remainder = poly_divmod(gamma_relation, f_poly)
assert remainder == [0]
gamma_intervals, _ = positive_root_intervals(gamma_poly)
assert len(gamma_intervals) == 2 and gamma_intervals[1][0] > Fraction(1, 4)
_, lower_remainder = poly_divmod(
    poly_add([7 * value for value in a_poly],
             [-value for value in denominator]), f_poly
)
_, upper_remainder = poly_divmod(
    poly_add(denominator, [-4 * value for value in a_poly]), f_poly
)
assert lower_remainder == [12, 20, 8]
assert upper_remainder == [-6, -8, 1, 3]

rho_left, rho_right = Fraction(2), Fraction(3)
for _ in range(80):
    rho_middle = (rho_left + rho_right) / 2
    if evaluate(f_poly, rho_middle) < 0:
        rho_left = rho_middle
    else:
        rho_right = rho_middle
rho = float((rho_left + rho_right) / 2)
gamma = ((rho ** 4 + rho ** 2 + 2 * rho + 1)
         / (rho ** 2 + 3 * rho + 1) ** 2)

# Equal weights on two cycles sharing the directed edge 4 -> 0 attain sigma_2=1.
sharp_pencil = normalized_pencil(
    ((0, 1, 2, 3, 4), (0, 5, 7, 6, 4)), 8
)
expected = poly_mul(
    [Fraction(0), Fraction(-1), Fraction(0), Fraction(1)],
    [Fraction(1, 5), Fraction(0), Fraction(-2), Fraction(0), Fraction(1)],
)
assert list(sharp_pencil) == expected

three_overlap_pencil = normalized_pencil(
    (first_cycle, (1, 3, 2, 5, 6)), 7
)
assert list(three_overlap_pencil) == poly_mul(
    poly_mul([0, 0, 1], [-1, 0, 1]),
    [Fraction(-1, 2), 0, 1],
)
four_overlap_pencil = normalized_pencil(
    (first_cycle, (0, 1, 4, 5, 2)), 6
)
assert list(four_overlap_pencil) == poly_mul(
    poly_mul([0, 1], [Fraction(-1, 2), 0, 1]),
    [Fraction(-1, 3), 0, 1],
)

print("five-cycle pairs by overlap 2 through 5: "
      + ", ".join(str(pair_counts[index]) for index in range(2, 6)))
print(f"total overlapping relative pairs: {sum(pair_counts.values())}")
print(f"distinct exact characteristic types: {len(patterns)}")
print(f"maximum positive coefficient breakpoints per type: {maximum_roots}")
print(f"maximum open weight-ratio cells per type: {maximum_cells}")
print("sharp normalized pencil: t(t^2-1)(t^4-2t^2+1/5)")
print("certified second-frequency ceiling: 1")
print("overlap-three types / breakpoint maximum: "
      f"{overlap_type_counts[3]} / {overlap_maximum_roots[3]}")
print("overlap-four types / breakpoint maximum: "
      f"{overlap_type_counts[4]} / {overlap_maximum_roots[4]}")
print("sharp overlap-three pencil: t^2(t^2-1)(t^2-1/2)")
print("sharp overlap-four pencil: t(t^2-1/2)(t^2-1/3)")
print("overlap-three / overlap-four ceilings: 1/sqrt(2), 1/sqrt(3)")
print("overlap-five placements / exact types: 24 / 8")
print("overlap-five rational types / squared ceiling: 6 / 1/7")
print("overlap-five maximum breakpoints / cells: "
      f"{five_nonalgebraic_maximum_roots} / "
      f"{five_nonalgebraic_maximum_cells}")
print("sharp ratio equation: rho^4-rho^3-2rho^2-2rho-1=0")
print("sharp squared-ceiling equation: 125g^4-100g^3+6g^2-20g+5=0")
print(f"sharp overlap-five ratio / ceiling: {rho:.6f} / {gamma ** 0.5:.6f}")
print("PASS")
