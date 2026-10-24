"""Exact alternating-weight slice through a four-four-cycle equality orbit."""

from collections import defaultdict
from fractions import Fraction
from itertools import permutations


N = 6
GROUND = N - 1
CYCLES = (
    (0, 1, 2, 3),
    (0, 1, 2, 4),
    (0, 3, 5, 1),
    (0, 3, 5, 4),
)


def permutation_sign(permutation):
    inversions = sum(
        permutation[first] > permutation[second]
        for first in range(len(permutation))
        for second in range(first + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def integer_forms(cycle):
    """Return the integer matrices 2H and 2D for a unit cycle flow."""
    weights = [[0] * N for _ in range(N)]
    for source, target in zip(cycle, cycle[1:] + cycle[:1]):
        weights[source][target] += 1
    laplacian = [[0] * N for _ in range(N)]
    for row in range(N):
        laplacian[row][row] = sum(weights[row])
        for column in range(N):
            laplacian[row][column] -= weights[row][column]
    grounded = [row[:GROUND] for row in laplacian[:GROUND]]
    symmetric = [
        [grounded[row][column] + grounded[column][row]
         for column in range(GROUND)]
        for row in range(GROUND)
    ]
    skew = [
        [grounded[row][column] - grounded[column][row]
         for column in range(GROUND)]
        for row in range(GROUND)
    ]
    return symmetric, skew


def add_polynomials(*polynomials):
    result = defaultdict(int)
    for polynomial in polynomials:
        for monomial, coefficient in polynomial.items():
            result[monomial] += coefficient
    return {monomial: coefficient for monomial, coefficient in result.items()
            if coefficient}


def scale_polynomial(polynomial, scalar):
    return {monomial: scalar * coefficient
            for monomial, coefficient in polynomial.items()
            if scalar * coefficient}


def multiply_polynomials(first, second):
    result = defaultdict(int)
    for (first_x, first_y), first_coefficient in first.items():
        for (second_x, second_y), second_coefficient in second.items():
            result[first_x + second_x, first_y + second_y] += (
                first_coefficient * second_coefficient
            )
    return {monomial: coefficient for monomial, coefficient in result.items()
            if coefficient}


def power_polynomial(polynomial, exponent):
    result = {(0, 0): 1}
    for _ in range(exponent):
        result = multiply_polynomials(result, polynomial)
    return result


def determinant_polynomial():
    """Expand det(t(2H)-2D) exactly in x, y and t."""
    forms = [integer_forms(cycle) for cycle in CYCLES]
    symmetric_x = [
        [forms[0][0][row][column] + forms[2][0][row][column]
         for column in range(GROUND)]
        for row in range(GROUND)
    ]
    symmetric_y = [
        [forms[1][0][row][column] + forms[3][0][row][column]
         for column in range(GROUND)]
        for row in range(GROUND)
    ]
    skew_x = [
        [forms[0][1][row][column] + forms[2][1][row][column]
         for column in range(GROUND)]
        for row in range(GROUND)
    ]
    skew_y = [
        [forms[1][1][row][column] + forms[3][1][row][column]
         for column in range(GROUND)]
        for row in range(GROUND)
    ]

    determinant = defaultdict(int)
    for permutation in permutations(range(GROUND)):
        terms = {(0, 0, 0): permutation_sign(permutation)}
        for row, column in enumerate(permutation):
            entry = {
                (1, 0, 1): symmetric_x[row][column],
                (0, 1, 1): symmetric_y[row][column],
                (1, 0, 0): -skew_x[row][column],
                (0, 1, 0): -skew_y[row][column],
            }
            entry = {key: value for key, value in entry.items() if value}
            product = defaultdict(int)
            for left, left_coefficient in terms.items():
                for right, right_coefficient in entry.items():
                    monomial = tuple(a + b for a, b in zip(left, right))
                    product[monomial] += left_coefficient * right_coefficient
            terms = product
        for monomial, coefficient in terms.items():
            determinant[monomial] += coefficient
    return {monomial: coefficient for monomial, coefficient in determinant.items()
            if coefficient}


def coefficient_in_t(determinant, exponent):
    return {(x_power, y_power): coefficient
            for (x_power, y_power, t_power), coefficient in determinant.items()
            if t_power == exponent}


X = {(1, 0): 1}
Y = {(0, 1): 1}


def linear(x_coefficient, y_coefficient):
    return add_polynomials(
        scale_polynomial(X, x_coefficient),
        scale_polynomial(Y, y_coefficient),
    )


determinant = determinant_polynomial()
assert {monomial[2] for monomial in determinant} == {1, 3, 5}
a5 = coefficient_in_t(determinant, 5)
a3 = coefficient_in_t(determinant, 3)
a1 = coefficient_in_t(determinant, 1)

x_plus_y = linear(1, 1)
two_x_plus_y = linear(2, 1)
quadratic = add_polynomials(
    scale_polynomial(power_polynomial(X, 2), 8),
    scale_polynomial(multiply_polynomials(X, Y), 12),
    scale_polynomial(power_polynomial(Y, 2), 3),
)
cubic = add_polynomials(
    scale_polynomial(power_polynomial(X, 3), 4),
    scale_polynomial(multiply_polynomials(power_polynomial(X, 2), Y), 10),
    scale_polynomial(multiply_polynomials(X, power_polynomial(Y, 2)), 9),
    scale_polynomial(power_polynomial(Y, 3), 2),
)

expected_a5 = scale_polynomial(
    multiply_polynomials(
        multiply_polynomials(Y, x_plus_y),
        multiply_polynomials(two_x_plus_y, quadratic),
    ), 8)
expected_a3 = scale_polynomial(
    multiply_polynomials(multiply_polynomials(Y, x_plus_y), cubic), 16)
expected_a1 = scale_polynomial(
    multiply_polynomials(
        multiply_polynomials(power_polynomial(Y, 2), x_plus_y),
        power_polynomial(two_x_plus_y, 2),
    ), 8)
assert (a5, a3, a1) == (expected_a5, expected_a3, expected_a1)

# The squared frequencies are the roots of a5*z^2-a3*z+a1.  Verify
# Vieta's identities for the two displayed rational branches without division.
first_numerator, first_denominator = Y, two_x_plus_y
second_numerator = power_polynomial(two_x_plus_y, 2)
second_denominator = quadratic
common_denominator = multiply_polynomials(first_denominator, second_denominator)
sum_numerator = add_polynomials(
    multiply_polynomials(first_numerator, second_denominator),
    multiply_polynomials(second_numerator, first_denominator),
)
product_numerator = multiply_polynomials(first_numerator, second_numerator)
assert multiply_polynomials(a3, common_denominator) == multiply_polynomials(
    a5, sum_numerator)
assert multiply_polynomials(a1, common_denominator) == multiply_polynomials(
    a5, product_numerator)

# With r=y/x, z1=r/(r+2) increases, z2=(r+2)^2/(3r^2+12r+8)
# decreases, and their difference has the sign of r^2+2r-4.
# The unique positive crossing is r=sqrt(5)-1; its squared frequency is the
# smaller root of z^2-3z+1, namely (3-sqrt(5))/2.
assert Fraction(6, 5) ** 2 + 2 * Fraction(6, 5) - 4 < 0
rational_witness = (Fraction(3, 8), Fraction(64, 167))
assert rational_witness == (
    Fraction(6, 2 * 5 + 6),
    Fraction((2 * 5 + 6) ** 2, 8 * 5 ** 2 + 12 * 5 * 6 + 3 * 6 ** 2),
)
assert min(rational_witness) == Fraction(3, 8) > Fraction(1, 3)

print("orbit representative: {(0123),(0124),(0351),(0354)}")
print("alternating weights: (x,y,x,y), x>0, y>0")
print("squared-frequency branches: y/(2x+y) and "
      "(2x+y)^2/(8x^2+12xy+3y^2)")
print("equal weights: lower squared frequency = 1/3")
print("rational witness weights: (5,6,5,6)")
print(f"rational witness squared frequencies: {rational_witness}")
print("sharp ratio on the slice: y/x = sqrt(5)-1")
print("sharp lower squared frequency on the slice: (3-sqrt(5))/2")
print("PASS")
