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

# kappa^2=1+2/sqrt(5)>7/4 because 8>3*sqrt(5), checked by squaring.
assert 8**2 > 3**2 * 5

print("exact degree-seven Bernstein conversion: verified")
print("rational cover leaves:", leaves)
print("maximum subdivision depth:", maximum_depth)
print("proved sigma_2^2 < 7/4 < cot(pi/5)^2")
print("coefficient relaxation admits sigma_2^2 = 1749/1000")
print("exact eight-state two-cycle witness: 51/50 < sigma_2^2 < 103/100")
print("PASS")
