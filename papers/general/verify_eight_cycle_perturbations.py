"""Exact local response of an eight-cycle to one added directed cycle.

For unit flow on the standard directed eight-cycle, let s(t) be the middle
squared intrinsic frequency after adding weight t on another simple directed
cycle.  The base value s(0)=1 is simple.  This certificate recovers the first
three derivatives for every cycle, breaks the remaining local ties through
fifth order, and classifies every crossing of the exact level s=21/20.
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


def sturm_sequence(poly):
    derivative = [degree * poly[degree] for degree in range(1, len(poly))]
    sequence = [polynomial_trim(poly), polynomial_trim(derivative)]
    while sequence[-1]:
        _, remainder = polynomial_divmod(sequence[-2], sequence[-1])
        if not remainder:
            break
        sequence.append([-entry for entry in remainder])
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
print("PASS")
