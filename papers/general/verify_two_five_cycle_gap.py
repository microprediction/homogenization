"""Exact certificate for the second frequency of two overlapping five-cycles."""

from fractions import Fraction
from itertools import combinations, permutations


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


def normalized_pencil(cycles, state_count):
    """Return det(tH-iD)/det(H), in ascending powers of t."""
    size = state_count - 1
    matrix = [[[(0, 0)] for _ in range(size)] for _ in range(size)]
    for cycle in cycles:
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            if source < size:
                matrix[source][source] = gaussian_poly_add(
                    matrix[source][source], [(0, 0), (2, 0)]
                )
            if source < size and target < size:
                matrix[source][target] = gaussian_poly_add(
                    matrix[source][target], [(0, 1), (-1, 0)]
                )
                matrix[target][source] = gaussian_poly_add(
                    matrix[target][source], [(0, -1), (-1, 0)]
                )
    determinant = determinant_poly(matrix)
    assert all(imaginary == 0 for _, imaginary in determinant)
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
    breakpoints = [Fraction(1)]
    for coefficient in coefficients:
        if any(coefficient):
            reduced = list(coefficient)
            while len(reduced) > 1 and reduced[0] == 0:
                reduced = reduced[1:]
            if len(reduced) > 1:
                breakpoints = square_free_lcm(breakpoints, reduced)
    intervals, upper = positive_root_intervals(breakpoints)
    maximum_roots = max(maximum_roots, len(intervals))
    if intervals:
        samples = [intervals[0][0] / 2]
        samples.extend(
            (left[1] + right[0]) / 2
            for left, right in zip(intervals, intervals[1:])
        )
        samples.append(upper + 1)
    else:
        samples = [Fraction(1)]
    maximum_cells = max(maximum_cells, len(samples))
    for ratio in samples:
        values = [evaluate(coefficient, ratio)
                  for coefficient in coefficients]
        assert all(value != 0 for coefficient, value
                   in zip(coefficients, values) if any(coefficient))
        assert sign_variations(values) <= 1

# Equal weights on two cycles sharing the directed edge 4 -> 0 attain sigma_2=1.
sharp_pencil = normalized_pencil(
    ((0, 1, 2, 3, 4), (0, 5, 7, 6, 4)), 8
)
expected = poly_mul(
    [Fraction(0), Fraction(-1), Fraction(0), Fraction(1)],
    [Fraction(1, 5), Fraction(0), Fraction(-2), Fraction(0), Fraction(1)],
)
assert list(sharp_pencil) == expected

print("five-cycle pairs by overlap 2 through 5: "
      + ", ".join(str(pair_counts[index]) for index in range(2, 6)))
print(f"total overlapping relative pairs: {sum(pair_counts.values())}")
print(f"distinct exact characteristic types: {len(patterns)}")
print(f"maximum positive coefficient breakpoints per type: {maximum_roots}")
print(f"maximum open weight-ratio cells per type: {maximum_cells}")
print("sharp normalized pencil: t(t^2-1)(t^4-2t^2+1/5)")
print("certified second-frequency ceiling: 1")
print("PASS")
