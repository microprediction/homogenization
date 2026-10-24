"""Exact local response of an eight-cycle to one added directed cycle.

For unit flow on the standard directed eight-cycle, let s(t) be the middle
squared intrinsic frequency after adding weight t on another simple directed
cycle.  The base value s(0)=1 is simple.  This certificate recovers the local
derivatives, classifies exact finite-level crossings, and globally optimizes
the unique runner-up support ray.
"""

from fractions import Fraction
from itertools import combinations, permutations
from math import comb, gcd


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
base_q_s = 3 * base_coefficients[0] - 2 * base_coefficients[1] + base_coefficients[2]
ray_cache = {}


def ray_coefficients(cycle):
    """Return a_j(t) in Q(t,s)=a0*s^3-a1*s^2+a2*s-a3."""

    if cycle in ray_cache:
        return ray_cache[cycle]
    symmetric, skew = cycle_forms(cycle)
    coefficients_by_u = []
    for u, inverse, base_determinant in base_data:
        perturbation = [
            [symmetric[i][j] + u * skew[i][j] for j in range(GROUND_SIZE)]
            for i in range(GROUND_SIZE)
        ]
        product = matrix_product(inverse, perturbation)
        power = [row[:] for row in product]
        traces = []
        for degree in range(1, GROUND_SIZE + 1):
            traces.append(sum((power[i][i] for i in range(GROUND_SIZE)), ZERO))
            if degree < GROUND_SIZE:
                power = matrix_product(power, product)

        # Newton identities give det(I+tA)=sum e_k(A)t^k.
        elementary = [ONE]
        for degree in range(1, GROUND_SIZE + 1):
            elementary.append(
                sum(
                    ((-1) ** (j - 1) * elementary[degree - j] * traces[j - 1]
                     for j in range(1, degree + 1)),
                    ZERO,
                ) / degree
            )
        coefficients_by_u.append(
            [base_determinant * entry for entry in elementary]
        )

    ray = tuple(
        tuple(
            interpolate_even(
                [coefficients_by_u[u][degree] for u in range(4)]
            )[coefficient]
            for degree in range(GROUND_SIZE + 1)
        )
        for coefficient in range(4)
    )
    ray_cache[cycle] = ray
    return ray


def frequency_derivatives(cycle):
    """Return exact s'(0), s''(0), s'''(0) for the middle squared frequency."""

    ray = ray_coefficients(cycle)
    first = [row[1] for row in ray]
    second = [row[2] for row in ray]
    third = [row[3] for row in ray]

    # Q(t,s)=a0(t)s^3-a1(t)s^2+a2(t)s-a3(t).  The determinant trace
    # identity gives coefficients through t^3; implicit differentiation at
    # (t,s)=(0,1) gives the three derivatives below.
    q_s = base_q_s
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


def truncated_product(left, right, degree):
    """Multiply two power series through the requested degree."""

    return [
        sum(
            (left[j] * right[power - j]
             for j in range(power + 1)
             if j < len(left) and power - j < len(right)),
            ZERO,
        )
        for power in range(degree + 1)
    ]


def ray_taylor_coefficients(cycle, degree=5):
    """Return the exact middle-root Taylor coefficients through ``degree``."""

    ray = ray_coefficients(cycle)

    root = [ONE]
    for order in range(1, degree + 1):
        trial = root + [ZERO]
        square = truncated_product(trial, trial, order)
        cube = truncated_product(square, trial, order)
        terms = [
            truncated_product(ray[0], cube, order),
            truncated_product(ray[1], square, order),
            truncated_product(ray[2], trial, order),
            ray[3],
        ]
        residual = terms[0][order] - terms[1][order] + terms[2][order]
        if order < len(terms[3]):
            residual -= terms[3][order]
        root.append(-residual / base_q_s)
    return tuple(root)


def polynomial_trim(poly):
    poly = list(poly)
    while poly and poly[-1] == 0:
        poly.pop()
    return poly


def polynomial_value(poly, point):
    return sum((coefficient * point ** degree
                for degree, coefficient in enumerate(poly)), ZERO)


def polynomial_add(left, right):
    result = [ZERO] * max(len(left), len(right))
    for degree in range(len(result)):
        result[degree] = (
            (left[degree] if degree < len(left) else ZERO)
            + (right[degree] if degree < len(right) else ZERO)
        )
    return polynomial_trim(result)


def polynomial_multiply(left, right):
    result = [ZERO] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return polynomial_trim(result)


def polynomial_scale(poly, scale):
    return polynomial_trim([scale * entry for entry in poly])


def polynomial_power(poly, power):
    result = [ONE]
    for _ in range(power):
        result = polynomial_multiply(result, poly)
    return result


def affine_substitute(poly, offset, scale):
    result = []
    affine = [offset, scale]
    for power, coefficient in enumerate(poly):
        result = polynomial_add(
            result,
            polynomial_scale(polynomial_power(affine, power), coefficient),
        )
    return result


def power_to_bernstein(poly):
    degree = len(poly) - 1
    return [
        sum(
            (poly[index] * Fraction(comb(k, index), comb(degree, index))
             for index in range(k + 1)),
            ZERO,
        )
        for k in range(degree + 1)
    ]


def bernstein_on_interval(poly, left, right):
    return power_to_bernstein(affine_substitute(poly, left, right - left))


def permutation_sign(permutation):
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def polynomial_determinant(matrix):
    result = []
    for permutation in permutations(range(len(matrix))):
        term = [Fraction(permutation_sign(permutation))]
        for row, column in enumerate(permutation):
            term = polynomial_multiply(term, matrix[row][column])
        result = polynomial_add(result, term)
    return result


def sylvester_resultant(left, right):
    """Resultant in s when entries are polynomials in t."""

    left_degree = len(left) - 1
    right_degree = len(right) - 1
    size = left_degree + right_degree
    matrix = [[[ZERO] for _ in range(size)] for _ in range(size)]
    for row in range(right_degree):
        matrix[row][row : row + left_degree + 1] = reversed(left)
    for row in range(left_degree):
        target = right_degree + row
        matrix[target][row : row + right_degree + 1] = reversed(right)
    return polynomial_determinant(matrix)


def polynomial_divmod(dividend, divisor):
    dividend = polynomial_trim(dividend)
    divisor = polynomial_trim(divisor)
    quotient = [ZERO] * max(1, len(dividend) - len(divisor) + 1)
    while dividend and len(dividend) >= len(divisor):
        degree = len(dividend) - len(divisor)
        coefficient = dividend[-1] / divisor[-1]
        quotient[degree] = coefficient
        for j, entry in enumerate(divisor):
            dividend[degree + j] -= coefficient * entry
        dividend = polynomial_trim(dividend)
    return polynomial_trim(quotient), dividend


def polynomial_gcd(left, right):
    left = polynomial_trim(left)
    right = polynomial_trim(right)
    while right:
        _, remainder = polynomial_divmod(left, right)
        left, right = right, remainder
    return [entry / left[-1] for entry in left]


def primitive_polynomial(poly):
    """Clear rational content by a positive scale, preserving all signs."""

    denominator_lcm = 1
    for entry in poly:
        denominator_lcm = (
            denominator_lcm * entry.denominator
            // gcd(denominator_lcm, entry.denominator)
        )
    integers = [entry.numerator * (denominator_lcm // entry.denominator)
                for entry in poly]
    content = 0
    for entry in integers:
        content = gcd(content, abs(entry))
    return polynomial_trim([Fraction(entry // content) for entry in integers])


def sturm_sequence(poly):
    derivative = [degree * poly[degree] for degree in range(1, len(poly))]
    sequence = [primitive_polynomial(polynomial_trim(poly)),
                primitive_polynomial(polynomial_trim(derivative))]
    while sequence[-1]:
        _, remainder = polynomial_divmod(sequence[-2], sequence[-1])
        if not remainder:
            break
        sequence.append(primitive_polynomial([-entry for entry in remainder]))
    return sequence


def sign_variations(values):
    signs = [1 if value > 0 else -1 for value in values if value]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def roots_between(sequence, left, right):
    assert polynomial_value(sequence[0], left)
    assert polynomial_value(sequence[0], right)
    return sign_variations(
        [polynomial_value(poly, left) for poly in sequence]
    ) - sign_variations([polynomial_value(poly, right) for poly in sequence])


def nonroot_split(poly, left, right):
    for denominator in range(2, 12):
        point = left + (right - left) / denominator
        if polynomial_value(poly, point):
            return point
    raise AssertionError("failed to find a rational non-root split")


def isolate_roots(sequence, left, right, count):
    intervals = []
    pending = [(left, right, count)]
    while pending:
        lower, upper, number = pending.pop()
        if number == 1:
            intervals.append((lower, upper))
            continue
        split = nonroot_split(sequence[0], lower, upper)
        left_count = roots_between(sequence, lower, split)
        if left_count:
            pending.append((lower, split, left_count))
        if number - left_count:
            pending.append((split, upper, number - left_count))
    return intervals


def signs_at_positive_roots(poly, classifier):
    """Return exact classifier signs at every distinct positive root of poly."""

    poly = polynomial_trim(poly)
    classifier = polynomial_trim(classifier)
    common = polynomial_gcd(poly, classifier)
    common_signs = []
    if len(common) > 1:
        common_sequence = sturm_sequence(common)
        common_bound = ONE + max(
            abs(entry / common[-1]) for entry in common[:-1]
        )
        while not polynomial_value(common, common_bound):
            common_bound += ONE
        common_count = roots_between(common_sequence, ZERO, common_bound)
        common_signs = [0] * common_count
        poly, remainder = polynomial_divmod(poly, common)
        assert not remainder
    if len(poly) == 1:
        return tuple(common_signs)
    sequence = sturm_sequence(poly)
    bound = ONE + max(abs(entry / poly[-1]) for entry in poly[:-1])
    while not polynomial_value(poly, bound):
        bound += ONE
    count = roots_between(sequence, ZERO, bound)
    classifier_sequence = sturm_sequence(classifier)
    signs = []
    for left, right in isolate_roots(sequence, ZERO, bound, count):
        for _ in range(100):
            left_value = polynomial_value(classifier, left)
            right_value = polynomial_value(classifier, right)
            if (left_value * right_value > 0
                    and roots_between(classifier_sequence, left, right) == 0):
                signs.append(1 if left_value > 0 else -1)
                break
            split = nonroot_split(poly, left, right)
            if roots_between(sequence, left, split) == 1:
                right = split
            else:
                left = split
        else:
            raise AssertionError("failed to separate a threshold crossing")
    return tuple(sorted(signs + common_signs))


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
maximal_representatives = []
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
            maximal_representatives.append((cycle, multiplicity))

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
fifth_order_classes = {}
representative_series = {}
assert len(maximal_representatives) == 3
assert all(multiplicity == 8 for _, multiplicity in maximal_representatives)
for cycle, multiplicity in maximal_representatives:
    series = ray_taylor_coefficients(cycle)
    assert series[1:4] == (
        frequency_derivatives(cycle)[0],
        frequency_derivatives(cycle)[1] / 2,
        frequency_derivatives(cycle)[2] / 6,
    )
    representative_series[cycle] = series
    key = (len(cycle), series[5])
    fifth_order_classes[key] = fifth_order_classes.get(key, 0) + multiplicity
assert fifth_order_classes == {
    (7, Fraction(-7)): 8,
    (7, Fraction(-8)): 8,
    (8, Fraction(-29, 2)): 8,
}
strong_cycle = (1, 2, 6, 7, 3, 4, 5)
strong_representative = min(rotation_orbit(strong_cycle))
assert representative_series[strong_representative] == (
    ONE, ZERO, ONE, Fraction(-2), Fraction(7, 2), Fraction(-7)
)

def classify_threshold(threshold):
    """Classify all positive level crossings by the exact sign of Q_s."""

    representative_classes = {}
    labeled_classes = {}
    negative_representatives = []
    for cycle in representatives:
        ray = ray_coefficients(cycle)
        threshold_poly = [
            ray[0][degree] * threshold ** 3
            - ray[1][degree] * threshold ** 2
            + ray[2][degree] * threshold
            - ray[3][degree]
            for degree in range(GROUND_SIZE + 1)
        ]
        derivative_poly = [
            3 * ray[0][degree] * threshold ** 2
            - 2 * ray[1][degree] * threshold
            + ray[2][degree]
            for degree in range(GROUND_SIZE + 1)
        ]
        crossing_class = signs_at_positive_roots(
            threshold_poly, derivative_poly
        )
        representative_classes[crossing_class] = (
            representative_classes.get(crossing_class, 0) + 1
        )
        multiplicity = len(rotation_orbit(cycle))
        labeled_classes[crossing_class] = (
            labeled_classes.get(crossing_class, 0) + multiplicity
        )
        if -1 in crossing_class:
            negative_representatives.append(cycle)
    return representative_classes, labeled_classes, negative_representatives


# At a simple root of Q(t,R), Q_s<0 holds exactly for the middle frequency;
# Q_s>0 identifies an extreme frequency, and Q_s=0 a collision.
(
    representative_crossing_classes,
    labeled_crossing_classes,
    negative_crossing_representatives,
) = classify_threshold(Fraction(21, 20))

assert representative_crossing_classes == {
    (): 828,
    (1,): 58,
    (1, 1): 1138,
    (-1, -1): 1,
}
assert labeled_crossing_classes == {
    (): 6593,
    (1,): 440,
    (1, 1): 9023,
    (-1, -1): 8,
}
assert negative_crossing_representatives == [strong_representative]

(
    runner_representative_crossing_classes,
    runner_labeled_crossing_classes,
    runner_negative_crossing_representatives,
) = classify_threshold(Fraction(26, 25))
runner_representative = (0, 1, 5, 2, 7, 3, 4)
assert runner_representative_crossing_classes == {
    (): 844,
    (1,): 58,
    (1, 1): 1121,
    (-1, -1): 2,
}
assert runner_labeled_crossing_classes == {
    (): 6721,
    (1,): 440,
    (1, 1): 8887,
    (-1, -1): 16,
}
assert sorted(runner_negative_crossing_representatives) == [
    strong_representative, runner_representative
]

# A rational runner-up witness: at t=3/10 all three squared frequencies
# are isolated, with the middle root strictly between 1.0445 and 1.0446.
runner_weight = Fraction(3, 10)
runner_ray = ray_coefficients(runner_representative)
runner_coefficients = [
    sum((row[degree] * runner_weight ** degree
         for degree in range(GROUND_SIZE + 1)), ZERO)
    for row in runner_ray
]


def runner_polynomial(squared_frequency):
    return (
        runner_coefficients[0] * squared_frequency ** 3
        - runner_coefficients[1] * squared_frequency ** 2
        + runner_coefficients[2] * squared_frequency
        - runner_coefficients[3]
    )


runner_brackets = (
    (Fraction(11, 100), Fraction(3, 25)),
    (Fraction(10445, 10000), Fraction(10446, 10000)),
    (Fraction(5, 4), Fraction(63, 50)),
)
assert [(
    runner_polynomial(left) < 0,
    runner_polynomial(right) > 0,
) for left, right in runner_brackets] == [
    (True, True), (False, False), (True, True)
]

# Optimize the runner-up ray itself.  Multiplication by 64 clears the common
# denominator without changing its roots.  Coefficients below are ascending
# in t, while the four rows are a0,a1,a2,a3 in
# Q(t,s)=a0(t)s^3-a1(t)s^2+a2(t)s-a3(t).
runner_integer_rows = [
    polynomial_trim([64 * entry for entry in row])
    for row in runner_ray
]
assert runner_integer_rows == [
    [4, 46, 178, 277, 194, 61, 7],
    [28, 122, 274, 413, 386, 185, 35],
    [28, 82, 118, 135, 118, 71, 21],
    [4, 6, 6, 7, 6, 3, 1],
]
runner_q_by_s = [
    polynomial_scale(runner_integer_rows[3], -ONE),
    runner_integer_rows[2],
    polynomial_scale(runner_integer_rows[1], -ONE),
    runner_integer_rows[0],
]
runner_qt_by_s = [
    [degree * coefficient for degree, coefficient in enumerate(row)][1:]
    for row in runner_q_by_s
]
runner_resultant = sylvester_resultant(runner_q_by_s, runner_qt_by_s)
runner_stationary_polynomial = [
    200, 2560, 11738, 13346, -93988, -504901, -1169222,
    -1081723, 2111574, 11288181, 27715398, 48696144,
    67093286, 74607693, 67472864, 49474353, 28937338,
    12756080, 3179004, -1071209, -2199774, -1897559,
    -1176030, -554559, -195908, -49138, -7744, -492, 48, 8,
]
assert runner_resultant == [ZERO] + [
    262144 * Fraction(entry) for entry in runner_stationary_polynomial
]

# The discriminant is positive for every t>=0.  On [0,1] this follows from
# four exact Bernstein patches, and after t=1+y all power coefficients are
# positive.  Hence the three positive branches never collide.
a, b, c, d = (
    runner_q_by_s[3], runner_q_by_s[2],
    runner_q_by_s[1], runner_q_by_s[0],
)
runner_discriminant = polynomial_add(
    polynomial_add(
        polynomial_multiply(polynomial_multiply(b, b), polynomial_multiply(c, c)),
        polynomial_scale(polynomial_multiply(a, polynomial_power(c, 3)), -4),
    ),
    polynomial_add(
        polynomial_scale(polynomial_multiply(polynomial_power(b, 3), d), -4),
        polynomial_add(
            polynomial_scale(polynomial_multiply(polynomial_multiply(a, a),
                                                  polynomial_multiply(d, d)), -27),
            polynomial_scale(polynomial_multiply(
                polynomial_multiply(a, b), polynomial_multiply(c, d)), 18),
        ),
    ),
)
assert runner_discriminant == [Fraction(entry) for entry in [
    131072, 1114112, 2867200, -3309568, -38432768, -102662144,
    -110059520, 133521408, 874393600, 2321432576, 4512198656,
    7132463104, 9483366400, 10772684800, 10557763584, 8979218432,
    6633365504, 4233895936, 2305417216, 1048928256, 386940928,
    110837760, 23068672, 3096576, 200704,
]]
for left, right in (
    (ZERO, Fraction(1, 4)),
    (Fraction(1, 4), Fraction(1, 2)),
    (Fraction(1, 2), ONE),
):
    assert min(bernstein_on_interval(runner_discriminant, left, right)) > 0
assert min(affine_substitute(runner_discriminant, ONE, ONE)) > 0

# Sturm isolation gives exactly four positive roots of R, one in each narrow
# rational interval.  The brackets are ordered by increasing t.
stationary_sequence = sturm_sequence(runner_stationary_polynomial)
stationary_bound = ONE + max(
    abs(Fraction(entry, runner_stationary_polynomial[-1]))
    for entry in runner_stationary_polynomial[:-1]
)
assert roots_between(stationary_sequence, ZERO, stationary_bound) == 4
stationary_t_brackets = (
    (Fraction(2983948320, 10**10), Fraction(2983948321, 10**10)),
    (Fraction(3885282266, 10**10), Fraction(3885282267, 10**10)),
    (Fraction(17327821484, 10**10), Fraction(17327821486, 10**10)),
    (Fraction(11735811033, 10**9), Fraction(11735811034, 10**9)),
)
assert [roots_between(stationary_sequence, left, right)
        for left, right in stationary_t_brackets] == [1, 1, 1, 1]

stationary_s_brackets = (
    (
        (Fraction(11684860, 10**8), Fraction(11684861, 10**8)),
        (Fraction(104456313, 10**8), Fraction(104456314, 10**8)),
        (Fraction(12621340, 10**7), Fraction(12621341, 10**7)),
    ),
    (
        (Fraction(10602873, 10**8), Fraction(10602874, 10**8)),
        (Fraction(94445507, 10**8), Fraction(94445508, 10**8)),
        (Fraction(11151855, 10**7), Fraction(11151856, 10**7)),
    ),
    (
        (Fraction(5841360, 10**8), Fraction(5841370, 10**8)),
        (Fraction(41069459, 10**8), Fraction(41069460, 10**8)),
        (Fraction(16574903, 10**7), Fraction(16574904, 10**7)),
    ),
    (
        (Fraction(5163038, 10**8), Fraction(5163040, 10**8)),
        (Fraction(55543185, 10**8), Fraction(55543186, 10**8)),
        (Fraction(32977536, 10**7), Fraction(32977538, 10**7)),
    ),
)


def runner_polynomial_in_t(squared_frequency):
    result = []
    for s_power, row in enumerate(runner_q_by_s):
        result = polynomial_add(
            result, polynomial_scale(row, squared_frequency ** s_power)
        )
    return result


# At every stationary t-bracket the three sign-changing s-brackets isolate
# all three roots.  Bernstein coefficients make each sign uniform throughout
# the whole t-bracket, not merely at a floating-point midpoint.
root_endpoint_signs = ((-1, 1), (1, -1), (-1, 1))
for t_bracket, root_brackets in zip(stationary_t_brackets,
                                    stationary_s_brackets):
    for (s_left, s_right), (left_sign, right_sign) in zip(
            root_brackets, root_endpoint_signs):
        left_values = bernstein_on_interval(
            runner_polynomial_in_t(s_left), *t_bracket
        )
        right_values = bernstein_on_interval(
            runner_polynomial_in_t(s_right), *t_bracket
        )
        assert min(left_sign * value for value in left_values) > 0
        assert min(right_sign * value for value in right_values) > 0


def positive_box_interval(rows, t_bracket, s_bracket):
    """Natural interval extension on a positive (t,s) box."""

    lower = ZERO
    upper = ZERO
    for s_power, row in enumerate(rows):
        for t_power, coefficient in enumerate(row):
            low_monomial = t_bracket[0] ** t_power * s_bracket[0] ** s_power
            high_monomial = t_bracket[1] ** t_power * s_bracket[1] ** s_power
            if coefficient >= 0:
                lower += coefficient * low_monomial
                upper += coefficient * high_monomial
            else:
                lower += coefficient * high_monomial
                upper += coefficient * low_monomial
    return lower, upper


# The resultant root must lie on the only branch whose Q_t interval can
# contain zero.  Thus roots 1 and 3 of R are middle-branch stationary points;
# roots 2 and 4 belong respectively to the high and low branches.
stationary_branch_indices = (1, 2, 1, 0)
for case, (t_bracket, root_brackets) in enumerate(zip(
        stationary_t_brackets, stationary_s_brackets)):
    relevant = stationary_branch_indices[case]
    for branch, s_bracket in enumerate(root_brackets):
        if branch == relevant:
            continue
        lower, upper = positive_box_interval(
            runner_qt_by_s, t_bracket, s_bracket
        )
        assert lower > 0 or upper < 0

# At t=0 the middle root is 1.  As t tends to infinity it approaches the
# middle root of 7s^3-35s^2+21s-1, which lies in (1/2,1).  The two isolated
# middle stationary values lie respectively above 1.04456313 and below
# 0.41069460.  Since there are no other middle stationary points, the first
# is the unique global maximum on t>=0.
runner_limit_polynomial = [-1, 21, -35, 7]
assert polynomial_value(runner_limit_polynomial, ZERO) < 0
assert polynomial_value(runner_limit_polynomial, Fraction(1, 10)) > 0
assert polynomial_value(runner_limit_polynomial, Fraction(1, 2)) > 0
assert polynomial_value(runner_limit_polynomial, ONE) < 0
assert polynomial_value(runner_limit_polynomial, 5) > 0
runner_max_t_bracket = stationary_t_brackets[0]
runner_max_s_bracket = stationary_s_brackets[0][1]
assert runner_max_s_bracket[0] > ONE
assert stationary_s_brackets[2][1][1] < ONE

# The next exact threshold gap identifies a tied third-place pair.  At 1.023
# precisely four orbits cross on the middle branch; at 1.024 only the winner
# and runner-up remain.  The two extra six-cycles have identical ray cubics,
# as also follows from reflection followed by reversing every directed edge.
(
    third_representative_crossing_classes,
    third_labeled_crossing_classes,
    third_negative_crossing_representatives,
) = classify_threshold(Fraction(1023, 1000))
assert third_representative_crossing_classes == {
    (): 874, (1,): 58, (1, 1): 1089, (-1, -1): 4,
}
assert third_labeled_crossing_classes == {
    (): 6961, (1,): 440, (1, 1): 8631, (-1, -1): 32,
}
third_representatives = [
    (0, 1, 2, 6, 7, 3),
    (0, 1, 2, 7, 3, 4),
]
assert sorted(third_negative_crossing_representatives) == sorted(
    [strong_representative, runner_representative] + third_representatives
)
assert ray_coefficients(third_representatives[0]) == ray_coefficients(
    third_representatives[1]
)
(
    fourth_representative_crossing_classes,
    fourth_labeled_crossing_classes,
    fourth_negative_crossing_representatives,
) = classify_threshold(Fraction(128, 125))
assert fourth_representative_crossing_classes == {
    (): 876, (1,): 58, (1, 1): 1089, (-1, -1): 2,
}
assert fourth_labeled_crossing_classes == {
    (): 6977, (1,): 440, (1, 1): 8631, (-1, -1): 16,
}
assert sorted(fourth_negative_crossing_representatives) == [
    strong_representative, runner_representative
]

# Relabel the previously optimized eight-/six-cycle family so its Hamilton
# cycle is standard; the added cycle lands in the third-place pair.
optimized_base = (0, 5, 6, 1, 2, 3, 7, 4)
optimized_six_cycle = (0, 5, 7, 6, 1, 4)
standard_labels = {vertex: index for index, vertex in enumerate(optimized_base)}
relabeled_six_cycle = tuple(standard_labels[vertex]
                            for vertex in optimized_six_cycle)
assert min(rotation_orbit(relabeled_six_cycle)) == third_representatives[1]

# The following pair is exchanged by swapping the two Hamilton cycles.  Since
# a common scale does not change normalized skew frequencies, their response
# curves obey s_2(t)=s_1(1/t) and have exactly the same maximum.
(
    fifth_lower_representative_classes,
    fifth_lower_labeled_classes,
    fifth_lower_negative_representatives,
) = classify_threshold(Fraction(101697, 100000))
assert fifth_lower_representative_classes == {
    (): 887, (1,): 58, (1, 1): 1074, (-1, -1): 6,
}
assert fifth_lower_labeled_classes == {
    (): 7065, (1,): 440, (1, 1): 8511, (-1, -1): 48,
}
fifth_representatives = [
    (0, 1, 3, 6, 7, 2, 4, 5),
    (0, 1, 5, 2, 6, 7, 3, 4),
]
top_four_representatives = [
    strong_representative, runner_representative, *third_representatives,
]
assert sorted(fifth_lower_negative_representatives) == sorted(
    top_four_representatives + fifth_representatives
)
(
    fifth_upper_representative_classes,
    fifth_upper_labeled_classes,
    fifth_upper_negative_representatives,
) = classify_threshold(Fraction(101699, 100000))
assert fifth_upper_representative_classes == {
    (): 889, (1,): 58, (1, 1): 1074, (-1, -1): 4,
}
assert fifth_upper_labeled_classes == {
    (): 7081, (1,): 440, (1, 1): 8511, (-1, -1): 32,
}
assert sorted(fifth_upper_negative_representatives) == sorted(
    top_four_representatives
)

standard_cycle = tuple(range(STATE_COUNT))
swapped_labels = {
    vertex: index for index, vertex in enumerate(fifth_representatives[1])
}
swapped_standard_cycle = tuple(swapped_labels[vertex]
                               for vertex in standard_cycle)
assert min(rotation_orbit(swapped_standard_cycle)) == fifth_representatives[0]
assert all(
    left == tuple(reversed(right))
    for left, right in zip(
        ray_coefficients(fifth_representatives[0]),
        ray_coefficients(fifth_representatives[1]),
    )
)

# One more threshold gap isolates the next orbit.  The lower threshold leaves
# the preceding six orbits plus one six-cycle; the upper threshold removes
# only that cycle.  Thus it is the unique seventh-place rotation orbit.
(
    seventh_lower_representative_classes,
    seventh_lower_labeled_classes,
    seventh_lower_negative_representatives,
) = classify_threshold(Fraction(101605, 100000))
assert seventh_lower_representative_classes == {
    (): 888, (1,): 58, (1, 1): 1072, (-1, -1): 7,
}
assert seventh_lower_labeled_classes == {
    (): 7073, (1,): 440, (1, 1): 8495, (-1, -1): 56,
}
seventh_representative = (0, 1, 5, 6, 2, 3)
top_six_representatives = top_four_representatives + fifth_representatives
assert sorted(seventh_lower_negative_representatives) == sorted(
    top_six_representatives + [seventh_representative]
)
(
    seventh_upper_representative_classes,
    seventh_upper_labeled_classes,
    seventh_upper_negative_representatives,
) = classify_threshold(Fraction(101607, 100000))
assert seventh_upper_representative_classes == {
    (): 889, (1,): 58, (1, 1): 1072, (-1, -1): 6,
}
assert seventh_upper_labeled_classes == {
    (): 7081, (1,): 440, (1, 1): 8495, (-1, -1): 48,
}
assert sorted(seventh_upper_negative_representatives) == sorted(
    top_six_representatives
)

# The same argument one layer lower identifies a unique eighth-place
# six-cycle.  No numerical maximizer is used: the exact threshold crossing
# sets differ by exactly this one rotation orbit.
(
    eighth_lower_representative_classes,
    eighth_lower_labeled_classes,
    eighth_lower_negative_representatives,
) = classify_threshold(Fraction(100514, 100000))
assert eighth_lower_representative_classes == {
    (): 893, (1,): 58, (1, 1): 1066, (-1, -1): 8,
}
assert eighth_lower_labeled_classes == {
    (): 7113, (1,): 440, (1, 1): 8447, (-1, -1): 64,
}
eighth_representative = (0, 1, 5, 2, 7, 4)
top_seven_representatives = top_six_representatives + [seventh_representative]
assert sorted(eighth_lower_negative_representatives) == sorted(
    top_seven_representatives + [eighth_representative]
)
(
    eighth_upper_representative_classes,
    eighth_upper_labeled_classes,
    eighth_upper_negative_representatives,
) = classify_threshold(Fraction(100516, 100000))
assert eighth_upper_representative_classes == {
    (): 894, (1,): 58, (1, 1): 1066, (-1, -1): 7,
}
assert eighth_upper_labeled_classes == {
    (): 7121, (1,): 440, (1, 1): 8447, (-1, -1): 56,
}
assert sorted(eighth_upper_negative_representatives) == sorted(
    top_seven_representatives
)

# A final nearby rational gap isolates the ninth-place orbit.  The number of
# middle-crossing representatives drops from nine to eight, with exactly the
# displayed six-cycle removed.
(
    ninth_lower_representative_classes,
    ninth_lower_labeled_classes,
    ninth_lower_negative_representatives,
) = classify_threshold(Fraction(100478, 100000))
assert ninth_lower_representative_classes == {
    (): 892, (1,): 58, (1, 1): 1066, (-1, -1): 9,
}
assert ninth_lower_labeled_classes == {
    (): 7105, (1,): 440, (1, 1): 8447, (-1, -1): 72,
}
ninth_representative = (0, 1, 5, 6, 2, 4)
top_eight_representatives = top_seven_representatives + [eighth_representative]
assert sorted(ninth_lower_negative_representatives) == sorted(
    top_eight_representatives + [ninth_representative]
)
(
    ninth_upper_representative_classes,
    ninth_upper_labeled_classes,
    ninth_upper_negative_representatives,
) = classify_threshold(Fraction(100480, 100000))
assert ninth_upper_representative_classes == {
    (): 893, (1,): 58, (1, 1): 1066, (-1, -1): 8,
}
assert ninth_upper_labeled_classes == {
    (): 7113, (1,): 440, (1, 1): 8447, (-1, -1): 64,
}
assert sorted(ninth_upper_negative_representatives) == sorted(
    top_eight_representatives
)
print("maximal-curvature third-derivative distribution:", maximal_curvature_thirds)
print("strong seven-cycle derivatives:", frequency_derivatives((1, 2, 6, 7, 3, 4, 5)))
print("maximal-curvature fifth-order classes:", fifth_order_classes)
print("strong seven-cycle Taylor coefficients:", representative_series[strong_representative])

print("simple directed cycles checked:", len(cycles))
print("rotation-orbit representatives:", len(representatives))
print("first-derivative distribution:", first_distribution)
print("first-order-neutral cycles:", first_distribution[ZERO])
print("neutral second-derivative distribution:", neutral_curvature_distribution)
print("curvature-two cycles by length:", neutral_curvature_lengths)
print("21/20 crossing classes (rotation representatives):",
      representative_crossing_classes)
print("21/20 crossing classes (labeled cycles):", labeled_crossing_classes)
print("supports with middle-frequency crossings:",
      negative_crossing_representatives)
print("26/25 crossing classes (rotation representatives):",
      runner_representative_crossing_classes)
print("26/25 crossing classes (labeled cycles):",
      runner_labeled_crossing_classes)
print("supports above 26/25:", sorted(runner_negative_crossing_representatives))
print("runner-up root brackets at t=3/10:", runner_brackets)
print("runner-up maximizing weight bracket:", runner_max_t_bracket)
print("runner-up sharp maximum bracket:", runner_max_s_bracket)
print("1023/1000 crossing classes:", third_representative_crossing_classes)
print("tied third-place representatives:", third_representatives)
print("128/125 crossing classes:", fourth_representative_crossing_classes)
print("1.01697 crossing classes:", fifth_lower_representative_classes)
print("tied fifth-place representatives:", fifth_representatives)
print("1.01699 crossing classes:", fifth_upper_representative_classes)
print("fifth-place maximum: 1.01697 < sigma_2^2 < 1.01699")
print("1.01605 crossing classes:", seventh_lower_representative_classes)
print("unique seventh-place representative:", seventh_representative)
print("1.01607 crossing classes:", seventh_upper_representative_classes)
print("seventh-place maximum: 1.01605 < sigma_2^2 < 1.01607")
print("1.00514 crossing classes:", eighth_lower_representative_classes)
print("unique eighth-place representative:", eighth_representative)
print("1.00516 crossing classes:", eighth_upper_representative_classes)
print("eighth-place maximum: 1.00514 < sigma_2^2 < 1.00516")
print("1.00478 crossing classes:", ninth_lower_representative_classes)
print("unique ninth-place representative:", ninth_representative)
print("1.00480 crossing classes:", ninth_upper_representative_classes)
print("ninth-place maximum: 1.00478 < sigma_2^2 < 1.00480")
print("PASS")
