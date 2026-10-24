"""Exact certificate for the unrestricted eight-state phase obstruction.

The only external mathematical input is Gao--Yuan's weighted-circulation
coefficient inequality.  Everything after that theorem is checked here with
``fractions.Fraction`` arithmetic, including the Bernstein conversion and a
rational interval cover of the remaining three-variable implication.
"""

from fractions import Fraction
from heapq import heappop, heappush
from itertools import permutations
from math import comb


ZERO = Fraction(0)
ONE = Fraction(1)


def poly_add(left, right, scale=ONE):
    result = dict(left)
    for exponent, coefficient in right.items():
        result[exponent] = result.get(exponent, ZERO) + scale * coefficient
        if not result[exponent]:
            del result[exponent]
    return result


def poly_multiply(left, right):
    result = {}
    for left_exponent, left_coefficient in left.items():
        for right_exponent, right_coefficient in right.items():
            exponent = tuple(a + b for a, b in zip(left_exponent, right_exponent))
            result[exponent] = (
                result.get(exponent, ZERO)
                + left_coefficient * right_coefficient
            )
    return {exponent: coefficient for exponent, coefficient in result.items() if coefficient}


def poly_scale(poly, scale):
    return {exponent: scale * coefficient for exponent, coefficient in poly.items()}


def poly_derivative(poly, coordinate):
    result = {}
    for exponent, coefficient in poly.items():
        power = exponent[coordinate]
        if power:
            reduced = list(exponent)
            reduced[coordinate] -= 1
            result[tuple(reduced)] = coefficient * power
    return result


def poly_constant(value):
    value = Fraction(value)
    return {(0, 0, 0): value} if value else {}


X = {(1, 0, 0): ONE}
Y = {(0, 1, 0): ONE}
Z = {(0, 0, 1): ONE}
E1 = poly_add(poly_add(X, Y), Z)
E2 = poly_add(poly_add(poly_multiply(X, Y), poly_multiply(X, Z)), poly_multiply(Y, Z))
E3 = poly_multiply(poly_multiply(X, Y), Z)


def linear_symmetric(constant, e1, e2, e3):
    result = poly_constant(constant)
    result = poly_add(result, E1, Fraction(e1))
    result = poly_add(result, E2, Fraction(e2))
    return poly_add(result, E3, Fraction(e3))


# B_k / det(H), for k=0,1,2,3; palindromicity gives B_{7-k}=B_k.
B = [
    linear_symmetric(1, 1, 1, 1),
    linear_symmetric(7, 3, -1, -5),
    linear_symmetric(21, 1, -3, 9),
    linear_symmetric(35, -5, 3, -5),
]


def univariate_add(left, right):
    size = max(len(left), len(right))
    result = [ZERO] * size
    for index in range(size):
        result[index] = (
            (left[index] if index < len(left) else ZERO)
            + (right[index] if index < len(right) else ZERO)
        )
    while result and not result[-1]:
        result.pop()
    return result


def univariate_multiply(left, right):
    result = [ZERO] * (len(left) + len(right) - 1)
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            result[i + j] += a * b
    return result


def univariate_power(poly, power):
    result = [ONE]
    for _ in range(power):
        result = univariate_multiply(result, poly)
    return result


def univariate_scale(poly, scale):
    return [scale * coefficient for coefficient in poly]


def univariate_value(poly, value):
    result = ZERO
    for coefficient in reversed(poly):
        result = result * value + coefficient
    return result


def univariate_divmod(dividend, divisor):
    dividend = list(dividend)
    quotient = [ZERO] * max(1, len(dividend) - len(divisor) + 1)
    while dividend and len(dividend) >= len(divisor):
        degree = len(dividend) - len(divisor)
        factor = dividend[-1] / divisor[-1]
        quotient[degree] = factor
        for index, coefficient in enumerate(divisor):
            dividend[degree + index] -= factor * coefficient
        while dividend and not dividend[-1]:
            dividend.pop()
    while quotient and not quotient[-1]:
        quotient.pop()
    return quotient, dividend


def sturm_root_count(poly, left, right):
    derivative = [index * coefficient for index, coefficient in enumerate(poly)][1:]
    sequence = [poly, derivative]
    while sequence[-1]:
        _, remainder = univariate_divmod(sequence[-2], sequence[-1])
        if not remainder:
            break
        sequence.append(univariate_scale(remainder, -ONE))

    def variations(value):
        signs = []
        for member in sequence:
            evaluated = univariate_value(member, value)
            assert evaluated
            signs.append(1 if evaluated > 0 else -1)
        return sum(a != b for a, b in zip(signs, signs[1:]))

    return variations(left) - variations(right)


def affine_substitute(poly, offset, scale):
    result = []
    affine = [offset, scale]
    for power, coefficient in enumerate(poly):
        result = univariate_add(
            result,
            univariate_scale(univariate_power(affine, power), coefficient),
        )
    return result


def power_to_bernstein(poly):
    degree = len(poly) - 1
    return [
        sum(
            poly[index] * Fraction(comb(k, index), comb(degree, index))
            for index in range(k + 1)
        )
        for k in range(degree + 1)
    ]


# Verify the symmetric Bernstein-basis conversion coefficient by coefficient
# in 1,e1,e2,e3.  No symbolic package is used.
bernstein = [[ZERO] for _ in range(4)]
for k in range(8):
    basis = univariate_multiply(
        univariate_power([ZERO, ONE], k),
        univariate_power([ONE, -ONE], 7 - k),
    )
    row = [
        [Fraction(1), Fraction(1), Fraction(1), Fraction(1)],
        [Fraction(7), Fraction(3), Fraction(-1), Fraction(-5)],
        [Fraction(21), Fraction(1), Fraction(-3), Fraction(9)],
        [Fraction(35), Fraction(-5), Fraction(3), Fraction(-5)],
    ][min(k, 7 - k)]
    for coordinate, coefficient in enumerate(row):
        bernstein[coordinate] = univariate_add(
            bernstein[coordinate],
            [coefficient * value for value in basis],
        )

spectral = [
    [ONE],
    univariate_power([-ONE, Fraction(2)], 2),
    univariate_power([-ONE, Fraction(2)], 4),
    univariate_power([-ONE, Fraction(2)], 6),
]
assert bernstein == spectral


def circulation_inequality(index, beta=49):
    center = B[index]
    left = B[index - 1]
    right = B[index + 1]
    result = poly_scale(poly_multiply(center, center), Fraction(beta))
    result = poly_add(result, poly_multiply(center, poly_add(left, right)), Fraction(-1))
    return poly_add(
        result,
        poly_multiply(left, right),
        Fraction(-(beta - 2)),
    )


Q1 = circulation_inequality(1)
Q2 = circulation_inequality(2)
MONOTONE = poly_add(B[1], B[0], Fraction(-1))


def interval_polynomial(poly, box):
    """Natural interval extension, exact because all box endpoints are rational."""

    lower = ZERO
    upper = ZERO
    for powers, coefficient in poly.items():
        monomial_lower = ONE
        monomial_upper = ONE
        for (left, right), power in zip(box, powers):
            monomial_lower *= left**power
            monomial_upper *= right**power
        if coefficient >= 0:
            lower += coefficient * monomial_lower
            upper += coefficient * monomial_upper
        else:
            lower += coefficient * monomial_upper
            upper += coefficient * monomial_lower
    return lower, upper


# If y=sigma_2^2 >= 7/4, x>=y and B1>=B0 imply
#   y<=3, x<=19/3, z<=55/187<3/10.
# The first two bounds follow from xy<=3+x+y.  For the third,
# R=(3+x+y-xy)/(x+y+3xy-1) decreases in each of x,y because
# its derivative numerators are -4(y+1)^2 and -4(x+1)^2.
# Therefore z<=R(x,y)<=R(7/4,7/4)=55/187<3/10.
assert (Fraction(7, 4) + 3) / (Fraction(7, 4) - 1) == Fraction(19, 3)
X_PLUS_Y = poly_add(X, Y)
R_NUMERATOR = poly_add(
    poly_add(poly_constant(3), X_PLUS_Y), poly_multiply(X, Y), Fraction(-1)
)
R_DENOMINATOR = poly_add(
    poly_add(X_PLUS_Y, poly_multiply(X, Y), Fraction(3)), poly_constant(-1)
)
for coordinate, other in ((0, Y), (1, X)):
    derivative_numerator = poly_add(
        poly_multiply(poly_derivative(R_NUMERATOR, coordinate), R_DENOMINATOR),
        poly_multiply(R_NUMERATOR, poly_derivative(R_DENOMINATOR, coordinate)),
        Fraction(-1),
    )
    one_plus_other = poly_add(other, poly_constant(1))
    expected = poly_scale(poly_multiply(one_plus_other, one_plus_other), Fraction(-4))
    assert derivative_numerator == expected
assert (Fraction(3) + 2 * Fraction(7, 4) - Fraction(7, 4) ** 2) / (
    2 * Fraction(7, 4) + 3 * Fraction(7, 4) ** 2 - 1
) == Fraction(55, 187)
assert Fraction(55, 187) < Fraction(3, 10)
initial_box = (
    (Fraction(7, 4), Fraction(19, 3)),
    (Fraction(7, 4), Fraction(3)),
    (ZERO, Fraction(3, 10)),
)

queue = []
counter = 0
heappush(queue, (-1.0, counter, initial_box, 0))
leaves = {"domain": 0, "Q1": 0, "Q2": 0}
maximum_depth = 0

while queue:
    _, _, box, depth = heappop(queue)
    maximum_depth = max(maximum_depth, depth)
    (x_lower, x_upper), (y_lower, _), _ = box

    # This box contains no ordered point x>=y.
    if x_upper < y_lower:
        leaves["domain"] += 1
        continue

    # This box contains no point satisfying B1>=B0.
    if interval_polynomial(MONOTONE, box)[1] < 0:
        leaves["domain"] += 1
        continue

    # At every remaining point at least one Gao--Yuan inequality fails.
    if interval_polynomial(Q1, box)[1] < 0:
        leaves["Q1"] += 1
        continue
    if interval_polynomial(Q2, box)[1] < 0:
        leaves["Q2"] += 1
        continue

    assert depth < 64
    normalized_widths = [
        float(right - left) / float(initial_box[index][1] - initial_box[index][0])
        for index, (left, right) in enumerate(box)
    ]
    split_coordinate = max(range(3), key=normalized_widths.__getitem__)
    midpoint = sum(box[split_coordinate], ZERO) / 2
    for side in range(2):
        child = list(box)
        child[split_coordinate] = (
            (box[split_coordinate][0], midpoint)
            if side == 0
            else (midpoint, box[split_coordinate][1])
        )
        child = tuple(child)
        counter += 1
        priority = -max(
            float(right - left) / float(initial_box[index][1] - initial_box[index][0])
            for index, (left, right) in enumerate(child)
        )
        heappush(queue, (priority, counter, child, depth + 1))

# The coefficient relaxation is nearly sharp.  At this exact rational point,
# all four independent coefficients are positive and strictly increasing,
# while every internal beta=49 inequality is strict.  Palindromicity supplies
# the reflected half.  This does not assert that a graph realizes the point.
relaxation_point = (
    Fraction(1749, 1000),
    Fraction(1749, 1000),
    Fraction(33241, 250000),
)
point_box = tuple((value, value) for value in relaxation_point)
point_B = [interval_polynomial(poly, point_box)[0] for poly in B]
point_Q1 = interval_polynomial(Q1, point_box)[0]
point_Q2 = interval_polynomial(Q2, point_box)[0]
point_Q3 = Fraction(48) * point_B[3] * (point_B[3] - point_B[2])
assert ZERO < point_B[0] < point_B[1] < point_B[2] < point_B[3]
assert point_Q1 > 0 and point_Q2 > 0 and point_Q3 > 0
assert Fraction(7, 4) - relaxation_point[1] == Fraction(1, 1000)

# A graph-realizable lower witness.  Add weight 2/5 on the six-cycle
# (0,5,7,6,1,4) to weight one on the eight-cycle
# (0,5,6,1,2,3,7,4).  Both cycle flows are balanced.
state_count = 8
flow = [[ZERO for _ in range(state_count)] for _ in range(state_count)]
for cycle, weight in (
    ((0, 5, 6, 1, 2, 3, 7, 4), ONE),
    ((0, 5, 7, 6, 1, 4), Fraction(2, 5)),
):
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        flow[source][target] += weight
assert all(
    sum(flow[i], ZERO)
    == sum((flow[j][i] for j in range(state_count)), ZERO)
    for i in range(state_count)
)

laplacian = [
    [
        (sum(flow[i], ZERO) if i == j else ZERO) - flow[i][j]
        for j in range(state_count)
    ]
    for i in range(state_count)
]
half = Fraction(1, 2)
sym = [
    [half * (laplacian[i][j] + laplacian[j][i]) for j in range(state_count)]
    for i in range(state_count)
]
skew = [
    [half * (laplacian[i][j] - laplacian[j][i]) for j in range(state_count)]
    for i in range(state_count)
]


def permutation_sign(permutation):
    inversions = sum(
        permutation[i] > permutation[j]
        for i in range(len(permutation))
        for j in range(i + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def polynomial_determinant(matrix):
    """Leibniz determinant for a small matrix of univariate polynomials."""

    result = []
    for permutation in permutations(range(len(matrix))):
        term = [Fraction(permutation_sign(permutation))]
        for row, column in enumerate(permutation):
            term = univariate_multiply(term, matrix[row][column])
        result = univariate_add(result, term)
    return result


def dictionary_determinant(matrix):
    """Leibniz determinant for small matrices of sparse polynomials."""

    result = {}
    for permutation in permutations(range(len(matrix))):
        term = {(0, 0): Fraction(permutation_sign(permutation))}
        for row, column in enumerate(permutation):
            term = poly_multiply(term, matrix[row][column])
        result = poly_add(result, term)
    return result


def sylvester_resultant(left, right):
    """Resultant in the outer variable; coefficients are univariate lists."""

    left_degree = len(left) - 1
    right_degree = len(right) - 1
    size = left_degree + right_degree
    matrix = [[[ZERO] for _ in range(size)] for _ in range(size)]
    left_descending = list(reversed(left))
    right_descending = list(reversed(right))
    for row in range(right_degree):
        matrix[row][row : row + left_degree + 1] = left_descending
    for row in range(left_degree):
        target = right_degree + row
        matrix[target][row : row + right_degree + 1] = right_descending
    return polynomial_determinant(matrix)


# Ground state zero.  The quotient determinant is
# det(H+uD)/det(H)=product_j(1+s_j u^2), where s_j=sigma_j^2.
pencil = [
    [[sym[i][j], skew[i][j]] for j in range(1, state_count)]
    for i in range(1, state_count)
]
pencil_determinant = polynomial_determinant(pencil)
assert pencil_determinant == [
    Fraction(31787, 50000),
    ZERO,
    Fraction(98399, 50000),
    ZERO,
    Fraction(75537, 50000),
    ZERO,
    Fraction(8309, 50000),
]
det_h = pencil_determinant[0]
e1, e2, e3 = (pencil_determinant[index] / det_h for index in (2, 4, 6))
assert (e1, e2, e3) == (
    Fraction(14057, 4541),
    Fraction(10791, 4541),
    Fraction(1187, 4541),
)


def squared_frequency_polynomial(value):
    return 4541 * value**3 - 14057 * value**2 + 10791 * value - 1187


# The cubic has one root in each disjoint interval below.  Since its degree is
# three, these are all its roots, ordered as s_3<s_2<s_1.
frequency_intervals = (
    (Fraction(13, 100), Fraction(14, 100)),
    (Fraction(51, 50), Fraction(103, 100)),
    (Fraction(19, 10), Fraction(2)),
)
for left, right in frequency_intervals:
    assert squared_frequency_polynomial(left) * squared_frequency_polynomial(right) < 0
assert squared_frequency_polynomial(Fraction(51, 50)) > 0
assert squared_frequency_polynomial(Fraction(103, 100)) < 0

# Optimize the same graph-realizable family exactly.  Give the six-cycle
# weight t>0 and retain unit weight on the eight-cycle.  Sparse bivariate
# arithmetic below uses exponents (power of t, power of u).
family_flow = [[{} for _ in range(state_count)] for _ in range(state_count)]
for cycle, weight in (
    ((0, 5, 6, 1, 2, 3, 7, 4), {(0, 0): ONE}),
    ((0, 5, 7, 6, 1, 4), {(1, 0): ONE}),
):
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        family_flow[source][target] = poly_add(family_flow[source][target], weight)

family_laplacian = []
for i in range(state_count):
    row_sum = {}
    for entry in family_flow[i]:
        row_sum = poly_add(row_sum, entry)
    family_laplacian.append(
        [
            poly_add(row_sum if i == j else {}, family_flow[i][j], -ONE)
            for j in range(state_count)
        ]
    )
family_sym = [
    [
        poly_scale(poly_add(family_laplacian[i][j], family_laplacian[j][i]), half)
        for j in range(state_count)
    ]
    for i in range(state_count)
]
family_skew = [
    [
        poly_scale(poly_add(family_laplacian[i][j], family_laplacian[j][i], -ONE), half)
        for j in range(state_count)
    ]
    for i in range(state_count)
]
family_pencil = [
    [
        poly_add(
            family_sym[i][j],
            {(t_power, u_power + 1): coefficient for (t_power, u_power), coefficient in family_skew[i][j].items()},
        )
        for j in range(1, state_count)
    ]
    for i in range(1, state_count)
]

# Apart from the common factor (1+t)/64, these are respectively det(H)
# and the three even-u coefficients of det(H+uD).
family_coefficients = [
    [Fraction(value) for value in values]
    for values in (
        (4, 30, 63, 43, 9),
        (28, 90, 119, 95, 33),
        (28, 66, 69, 49, 19),
        (4, 6, 5, 5, 3),
    )
]
expected_family_determinant = {}
for u_power, coefficient in enumerate(family_coefficients):
    for t_power, value in enumerate(univariate_multiply([ONE, ONE], coefficient)):
        if value:
            expected_family_determinant[(t_power, 2 * u_power)] = value / 64
assert dictionary_determinant(family_pencil) == expected_family_determinant

# Q(t,s)=a0(t)s^3-a1(t)s^2+a2(t)s-a3(t) has the three squared
# frequencies as roots.  Store it as t coefficients whose entries are
# polynomials in s.
frequency_polynomial_by_t = [
    [
        -family_coefficients[3][t_power],
        family_coefficients[2][t_power],
        -family_coefficients[1][t_power],
        family_coefficients[0][t_power],
    ]
    for t_power in range(5)
]

# The cubic has three simple positive roots for every t>0.  Its discriminant
# factors as 4096(t+1)^2 D(t); Bernstein coefficients prove D>0 on [0,1],
# and translating t=1+y gives positive power coefficients for t>=1.
a, b, c, d = (
    family_coefficients[0],
    univariate_scale(family_coefficients[1], -ONE),
    family_coefficients[2],
    univariate_scale(family_coefficients[3], -ONE),
)
discriminant_terms = (
    (ONE, univariate_multiply(univariate_power(b, 2), univariate_power(c, 2))),
    (-4, univariate_multiply(a, univariate_power(c, 3))),
    (-4, univariate_multiply(univariate_power(b, 3), d)),
    (-27, univariate_multiply(univariate_power(a, 2), univariate_power(d, 2))),
    (18, univariate_multiply(univariate_multiply(a, b), univariate_multiply(c, d))),
)
cubic_discriminant = []
for scale, term in discriminant_terms:
    cubic_discriminant = univariate_add(cubic_discriminant, univariate_scale(term, scale))
positive_discriminant_factor = [
    Fraction(value)
    for value in (32, 176, 380, 372, 76, -88, 356, 1428, 2432, 2536, 1757, 818, 237, 32)
]
assert cubic_discriminant == univariate_scale(
    univariate_multiply([ONE, 2 * ONE, ONE], positive_discriminant_factor), 4096
)
assert all(value > 0 for value in power_to_bernstein(positive_discriminant_factor))
assert all(value > 0 for value in affine_substitute(positive_discriminant_factor, ONE, ONE))

def evaluate_in_s(poly_by_t, value):
    result = [univariate_value(coefficient, value) for coefficient in poly_by_t]
    while result and not result[-1]:
        result.pop()
    return result


assert all(value < 0 for value in evaluate_in_s(frequency_polynomial_by_t, ZERO))
assert evaluate_in_s(frequency_polynomial_by_t, Fraction(1, 3)) == univariate_scale(
    univariate_power([ONE, ONE], 3), Fraction(64, 27)
)
four_thirds_barrier = [Fraction(value) for value in (188, 186, -669, 179, 405)]
assert evaluate_in_s(frequency_polynomial_by_t, Fraction(4, 3)) == univariate_scale(
    four_thirds_barrier, Fraction(-1, 27)
)
assert all(value > 0 for value in power_to_bernstein(four_thirds_barrier))
assert all(value > 0 for value in affine_substitute(four_thirds_barrier, ONE, ONE))

# Eliminate t from Q=partial_t Q=0.  The only resultant root in the middle
# root's interval (1/3,4/3), other than the boundary artifact s=1 at t=0,
# is the unique root s_* of R below.
frequency_derivative_by_t = [
    univariate_scale(frequency_polynomial_by_t[index], index)
    for index in range(1, 5)
]
critical_value_polynomial = [
    Fraction(value)
    for value in (
        -17905, 419101, -4098198, 22555350, -76117347, 158042511,
        -183760580, 71460388, 70163097, -61548501, -8663238,
        10444326, 1249499, 35337,
    )
]
resultant_factorization = univariate_scale(
    univariate_multiply(
        univariate_multiply(
            univariate_multiply([-3, ONE], [-1, ONE]),
            univariate_power([ONE, ONE], 2),
        ),
        univariate_multiply(univariate_power([-ONE, 3 * ONE], 4), critical_value_polynomial),
    ),
    4,
)
assert sylvester_resultant(frequency_polynomial_by_t, frequency_derivative_by_t) == resultant_factorization
assert sturm_root_count(critical_value_polynomial, Fraction(1, 3), Fraction(4, 3)) == 1
critical_left = Fraction(102374, 100000)
critical_right = Fraction(102375, 100000)
assert sturm_root_count(critical_value_polynomial, critical_left, critical_right) == 1
# The s=1 resultant factor comes only from t=0.  For t>0 the two displayed
# factors would force t=2, which does not solve t^2+t-1=0.
assert evaluate_in_s(frequency_polynomial_by_t, ONE) == [ZERO, ZERO, 8 * ONE, -8 * ONE, -8 * ONE]
assert evaluate_in_s(frequency_derivative_by_t, ONE) == [ZERO, 16 * ONE, -24 * ONE, -32 * ONE]
assert Fraction(2) ** 2 + Fraction(2) - 1 != 0

# A(s)t+B(s) is the last linear subresultant.  Direct substitution proves
# that t_*=-B(s_*)/A(s_*) solves both equations; Bernstein enclosures on the
# isolating interval give 2/5<t_*<41/100.
subresultant_a = [
    Fraction(value)
    for value in (1577, -35461, 320639, -1504443, 3990298, -5841170,
                  3674142, 934090, -1943891, 148519, 253811, 12129)
]
subresultant_b = [
    Fraction(value)
    for value in (2349, -49451, 402423, -1716193, 4150290, -5483358,
                  2972670, 878926, -1380591, 71129, 144315, 7491)
]
minus_b = univariate_scale(subresultant_b, -ONE)
substituted_q = []
for power, coefficient in enumerate(frequency_polynomial_by_t):
    term = univariate_multiply(
        coefficient,
        univariate_multiply(
            univariate_power(minus_b, power),
            univariate_power(subresultant_a, 4 - power),
        ),
    )
    substituted_q = univariate_add(substituted_q, term)
substituted_dt_q = []
for power in range(1, 5):
    term = univariate_multiply(
        univariate_scale(frequency_polynomial_by_t[power], power),
        univariate_multiply(
            univariate_power(minus_b, power - 1),
            univariate_power(subresultant_a, 4 - power),
        ),
    )
    substituted_dt_q = univariate_add(substituted_dt_q, term)
assert not univariate_divmod(substituted_q, critical_value_polynomial)[1]
assert not univariate_divmod(substituted_dt_q, critical_value_polynomial)[1]

def bernstein_on_interval(poly, left, right):
    return power_to_bernstein(affine_substitute(poly, left, right - left))


assert all(
    value > 0
    for value in bernstein_on_interval(subresultant_a, critical_left, critical_right)
)
lower_t_test = univariate_add(minus_b, univariate_scale(subresultant_a, Fraction(-2, 5)))
upper_t_test = univariate_add(minus_b, univariate_scale(subresultant_a, Fraction(-41, 100)))
assert all(value > 0 for value in bernstein_on_interval(lower_t_test, critical_left, critical_right))
assert all(value < 0 for value in bernstein_on_interval(upper_t_test, critical_left, critical_right))

# At t=0 the middle root is one; as t tends to infinity the leading
# polynomial is (s-3)(3s-1)^2.  The simple middle root has exactly the one
# stationary point isolated above, hence that point is its global maximum.
assert univariate_value(frequency_polynomial_by_t[0], ONE) == 0
assert frequency_polynomial_by_t[4] == univariate_multiply(
    [-3, ONE], univariate_power([-ONE, 3 * ONE], 2)
)
assert squared_frequency_polynomial(Fraction(10237, 10000)) > 0
assert squared_frequency_polynomial(Fraction(10238, 10000)) < 0

# kappa^2=1+2/sqrt(5)>7/4 because 8>3*sqrt(5), checked by squaring.
assert 8**2 > 3**2 * 5

print("exact degree-seven Bernstein conversion: verified")
print("rational cover leaves:", leaves)
print("maximum subdivision depth:", maximum_depth)
print("proved sigma_2^2 < 7/4 < cot(pi/5)^2")
print("coefficient relaxation admits sigma_2^2 = 1749/1000")
print("exact eight-state two-cycle witness: 51/50 < sigma_2^2 < 103/100")
print("sharp two-cycle family maximum: 1.02374 < sigma_2^2 < 1.02375")
print("maximizing six-cycle weight: 2/5 < t < 41/100")
print("PASS")
