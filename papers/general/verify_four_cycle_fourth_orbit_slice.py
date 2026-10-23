"""Exact equal-triple slice on the fourth four-cycle equality orbit."""

from collections import defaultdict
from fractions import Fraction
from itertools import permutations


N = 6
GROUND = N - 1
CYCLES = (
    (0, 1, 2, 3),
    (0, 1, 3, 4),
    (0, 1, 4, 2),
    (2, 3, 4, 5),
)


def permutation_sign(permutation):
    inversions = sum(
        permutation[first] > permutation[second]
        for first in range(len(permutation))
        for second in range(first + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def integer_forms(cycle):
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
    for first_monomial, first_coefficient in first.items():
        for second_monomial, second_coefficient in second.items():
            monomial = tuple(
                left + right
                for left, right in zip(first_monomial, second_monomial)
            )
            result[monomial] += first_coefficient * second_coefficient
    return {monomial: coefficient for monomial, coefficient in result.items()
            if coefficient}


def determinant_polynomial():
    """Expand det(t(2H)-2D) in four weights and t over the integers."""
    forms = [integer_forms(cycle) for cycle in CYCLES]
    determinant = defaultdict(int)
    for permutation in permutations(range(GROUND)):
        terms = {(0, 0, 0, 0, 0): permutation_sign(permutation)}
        for row, column in enumerate(permutation):
            entry = defaultdict(int)
            for index, (symmetric, skew) in enumerate(forms):
                weight_monomial = [0] * 5
                weight_monomial[index] = 1
                if symmetric[row][column]:
                    symmetric_monomial = weight_monomial.copy()
                    symmetric_monomial[4] = 1
                    entry[tuple(symmetric_monomial)] += symmetric[row][column]
                if skew[row][column]:
                    entry[tuple(weight_monomial)] -= skew[row][column]
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
    return {monomial[:4]: coefficient
            for monomial, coefficient in determinant.items()
            if monomial[4] == exponent}


def equal_triple_slice(polynomial):
    """Substitute (w0,w1,w2,w3)=(x,x,x,y)."""
    result = defaultdict(int)
    for monomial, coefficient in polynomial.items():
        result[(sum(monomial[:3]), monomial[3])] += coefficient
    return {monomial: coefficient for monomial, coefficient in result.items()
            if coefficient}


def linear_factor(x_coefficient, y_coefficient):
    return {(1, 0): x_coefficient, (0, 1): y_coefficient}


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


def evaluate(polynomial, values):
    return sum(
        coefficient * product(values[index] ** power
                              for index, power in enumerate(monomial))
        for monomial, coefficient in polynomial.items()
    )


determinant = determinant_polynomial()
assert {monomial[4] for monomial in determinant} == {1, 3, 5}
full_a5, full_a3, full_a1 = (
    coefficient_in_t(determinant, exponent) for exponent in (5, 3, 1)
)
a5, a3, a1 = (
    equal_triple_slice(polynomial)
    for polynomial in (full_a5, full_a3, full_a1)
)

x2y = {(2, 1): 1}
expected_a5 = scale_polynomial(
    multiply_polynomials(
        multiply_polynomials(x2y, linear_factor(5, 2)),
        linear_factor(5, 3),
    ), 18)
expected_a3 = scale_polynomial(
    multiply_polynomials(
        multiply_polynomials(x2y, linear_factor(1, 1)),
        linear_factor(17, 12),
    ), 12)
expected_a1 = scale_polynomial(
    multiply_polynomials(
        multiply_polynomials(x2y, linear_factor(1, 3)),
        linear_factor(3, 2),
    ), 6)
assert (a5, a3, a1) == (expected_a5, expected_a3, expected_a1)

# One squared frequency is identically 1/3.  The other is 3*a1/a5,
# and its displacement from 1/3 changes sign only at the displayed ratio.
root_barrier = add_polynomials(
    a5, scale_polynomial(a3, -3), scale_polynomial(a1, 9))
assert root_barrier == {}
threshold_polynomial = {(2, 0): 4, (1, 1): -2, (0, 2): -3}
sum_barrier = add_polynomials(scale_polynomial(a3, 3),
                              scale_polynomial(a5, -2))
assert sum_barrier == scale_polynomial(
    multiply_polynomials(x2y, threshold_polynomial), -72)

# With r=y/x, the derivative of the nonconstant root is
# 4*(21*r^2+66*r+50)/((2*r+5)^2*(3*r+5)^2), hence strictly positive.
derivative_numerator = (50, 66, 21)
assert all(coefficient > 0 for coefficient in derivative_numerator)

witness_below = (10, 1)
witness_above = (1, 1)
below_coefficients = tuple(evaluate(polynomial, witness_below)
                           for polynomial in (a5, a3, a1))
above_coefficients = tuple(evaluate(polynomial, witness_above)
                           for polynomial in (a5, a3, a1))
below_other_root = Fraction(3 * below_coefficients[2], below_coefficients[0])
above_other_root = Fraction(3 * above_coefficients[2], above_coefficients[0])
assert below_other_root == Fraction(8, 53)
assert above_other_root == Fraction(5, 14)

# The surrounding full cone is not pinned.  This exact off-slice witness has
# both roots above 1/3 because both barrier values are positive.
off_slice_witness = (3, 1, 2, 10)
off_slice_coefficients = tuple(
    evaluate(polynomial, off_slice_witness)
    for polynomial in (full_a5, full_a3, full_a1)
)
assert off_slice_coefficients == (851_780, 892_280, 202_820)
off_slice_sum_barrier = 3 * off_slice_coefficients[1] - 2 * off_slice_coefficients[0]
off_slice_root_barrier = (
    off_slice_coefficients[0] - 3 * off_slice_coefficients[1]
    + 9 * off_slice_coefficients[2]
)
assert off_slice_sum_barrier == 973_280
assert off_slice_root_barrier == 320

print("orbit representative: {(0123),(0134),(0142),(2345)}")
print("slice weights: (x,x,x,y), x>0, y>0")
print("squared frequencies: 1/3 and (x+3y)(3x+2y)/((5x+2y)(5x+3y))")
print("nonconstant root derivative numerator: 4*(21r^2+66r+50), r=y/x")
print("threshold ratio: y/x = (sqrt(13)-1)/3")
print("below threshold: lower squared frequency is the nonconstant root")
print("at or above threshold: lower squared frequency is 1/3")
print(f"weights (10,10,10,1): squared frequencies 1/3 and {below_other_root}")
print(f"weights (1,1,1,1): squared frequencies 1/3 and {above_other_root}")
print(f"off-slice weights: {off_slice_witness}")
print(f"off-slice pencil coefficients: {off_slice_coefficients}")
print(f"off-slice sum/root barriers: {off_slice_sum_barrier}, {off_slice_root_barrier}")
print("off-slice witness: both squared frequencies exceed 1/3")
print("PASS")
