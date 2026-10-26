"""Exact pinned-root theorem on the third four-cycle equality orbit."""

from collections import defaultdict
from fractions import Fraction
from itertools import permutations


N = 6
GROUND = N - 1
CYCLES = (
    (0, 1, 2, 3),
    (0, 1, 3, 2),
    (2, 3, 4, 5),
    (2, 4, 5, 3),
)


def permutation_sign(permutation):
    inversions = sum(
        permutation[first] > permutation[second]
        for first in range(len(permutation))
        for second in range(first + 1, len(permutation))
    )
    return -1 if inversions % 2 else 1


def integer_forms(cycle):
    """Return the integer matrices 2H and 2D for one unit cycle."""
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


def square_polynomial(polynomial):
    return multiply_polynomials(polynomial, polynomial)


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


W = tuple({tuple(1 if index == variable else 0 for index in range(4)): 1}
          for variable in range(4))
w0, w1, w2, w3 = W
s = add_polynomials(w0, w1)
t = add_polynomials(w2, w3)
delta_left = add_polynomials(w0, scale_polynomial(w1, -1))
delta_right = add_polynomials(w2, scale_polynomial(w3, -1))

determinant = determinant_polynomial()
assert {monomial[4] for monomial in determinant} == {1, 3, 5}
a5 = coefficient_in_t(determinant, 5)
a3 = coefficient_in_t(determinant, 3)
a1 = coefficient_in_t(determinant, 1)
assert a5 and all(coefficient > 0 for coefficient in a5.values())

sum_barrier = add_polynomials(
    scale_polynomial(a3, 3), scale_polynomial(a5, -2))
root_barrier = add_polynomials(
    a5, scale_polynomial(a3, -3), scale_polynomial(a1, 9))

# The value z=1/3 is a root at every weighting on this cone.
assert root_barrier == {}

# The second root cannot fall below 1/3: its displacement has a sum-of-squares
# numerator, vanishing precisely when both paired weights are balanced.
sum_of_squares = add_polynomials(
    multiply_polynomials(t, square_polynomial(delta_left)),
    multiply_polynomials(s, square_polynomial(delta_right)),
)
expected_sum_barrier = scale_polynomial(
    multiply_polynomials(multiply_polynomials(s, t), sum_of_squares), 24)
assert sum_barrier == expected_sum_barrier

# This also factors the discriminant and certifies the double-root locus.
discriminant = add_polynomials(
    square_polynomial(a3),
    scale_polynomial(multiply_polynomials(a5, a1), -4),
)
expected_discriminant = scale_polynomial(
    square_polynomial(
        multiply_polynomials(multiply_polynomials(s, t), sum_of_squares)
    ), 64)
assert discriminant == expected_discriminant

witness = (2, 1, 3, 1)
witness_coefficients = tuple(evaluate(polynomial, witness)
                             for polynomial in (a5, a3, a1))
witness_sum_of_squares = evaluate(sum_of_squares, witness)
witness_other_root = Fraction(3 * witness_coefficients[2],
                              witness_coefficients[0])
assert witness_coefficients == (13_320, 10_416, 1_992)
assert witness_sum_of_squares == 16
assert witness_other_root == Fraction(83, 185)

print("orbit representative: {(0123),(0132),(2345),(2453)}")
print("pair sums: s=w0+w1, t=w2+w3")
print("root barrier: a5-3*a3+9*a1 = 0")
print("sum barrier: 3*a3-2*a5 = 24*s*t*E")
print("E = t*(w0-w1)^2 + s*(w2-w3)^2")
print("other-root displacement: z_+-1/3 = 8*s*t*E/a5")
print("lower squared frequency: 1/3 for every positive weighting")
print("double root iff w0=w1 and w2=w3")
print(f"witness weights: {witness}")
print(f"witness pencil coefficients: {witness_coefficients}")
print(f"witness E: {witness_sum_of_squares}")
print(f"witness squared frequencies: 1/3 and {witness_other_root}")
print("PASS")
