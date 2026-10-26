"""Exact 1/3 crossing classification on the second four-cycle orbit."""

from collections import defaultdict
from fractions import Fraction
from itertools import permutations


N = 6
GROUND = N - 1
CYCLES = (
    (0, 1, 2, 3),
    (0, 1, 2, 4),
    (0, 3, 5, 4),
    (2, 3, 5, 4),
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


def power_polynomial(polynomial, exponent):
    result = {(0, 0, 0, 0): 1}
    for _ in range(exponent):
        result = multiply_polynomials(result, polynomial)
    return result


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


def evaluate(polynomial, values):
    return sum(
        coefficient * product(values[index] ** power
                              for index, power in enumerate(monomial))
        for monomial, coefficient in polynomial.items()
    )


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


W = tuple({tuple(1 if index == variable else 0 for index in range(4)): 1}
          for variable in range(4))
w0, w1, w2, w3 = W
s = add_polynomials(w0, w1)
t = add_polynomials(w2, w3)
p = multiply_polynomials(w0, w1)
q = multiply_polynomials(w2, w3)

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

# The sum-barrier cofactor expands with strictly positive coefficients.
positive_cofactor = add_polynomials(
    scale_polynomial(multiply_polynomials(p, s), 2),
    scale_polynomial(multiply_polynomials(q, t), 2),
    scale_polynomial(multiply_polynomials(power_polynomial(s, 2), t), 3),
    scale_polynomial(multiply_polynomials(s, power_polynomial(t, 2)), 3),
    scale_polynomial(multiply_polynomials(p, t), -5),
    scale_polynomial(multiply_polynomials(s, q), -5),
)
assert positive_cofactor
assert all(coefficient > 0 for coefficient in positive_cofactor.values())
expected_sum_barrier = scale_polynomial(
    multiply_polynomials(multiply_polynomials(s, t), positive_cofactor), 8)
assert sum_barrier == expected_sum_barrier

# The root barrier has a complete four-factor decomposition.
expected_root_barrier = scale_polynomial(
    multiply_polynomials(
        multiply_polynomials(s, t),
        multiply_polynomials(
            add_polynomials(p, scale_polynomial(q, -1)),
            add_polynomials(s, scale_polynomial(t, -1)),
        ),
    ), -32)
assert root_barrier == expected_root_barrier

# For positive weights, a5>0 and the sum barrier is strict.  Therefore the
# two squared roots cannot both lie below 1/3.  The sign of the root barrier,
# 9*a5*(1/3-z1)*(1/3-z2), gives the complete crossing classification.
witness = (9, 1, 4, 4)
witness_coefficients = tuple(evaluate(polynomial, witness)
                             for polynomial in (a5, a3, a1))
assert witness_coefficients == (1_809_920, 1_973_760, 460_800)
assert evaluate(sum_barrier, witness) == 2_301_440
assert evaluate(root_barrier, witness) == 35_840
discriminant = (witness_coefficients[1] ** 2
                - 4 * witness_coefficients[0] * witness_coefficients[2])
assert discriminant == 7_680 ** 2 * 9_489
assert Fraction(witness_coefficients[1], 2 * witness_coefficients[0]) == Fraction(771, 1414)
assert Fraction(7_680, 2 * witness_coefficients[0]) == Fraction(3, 1414)

print("orbit representative: {(0123),(0124),(0354),(2354)}")
print("pair sums: s=w0+w1, t=w2+w3")
print("pair products: p=w0*w1, q=w2*w3")
print("strict sum barrier: 3*a3-2*a5 > 0 for all positive weights")
print("root barrier: a5-3*a3+9*a1 = -32*s*t*(p-q)*(s-t)")
print("lower squared frequency > 1/3 iff (p-q)*(s-t) < 0")
print("lower squared frequency = 1/3 iff (p-q)*(s-t) = 0")
print("lower squared frequency < 1/3 iff (p-q)*(s-t) > 0")
print(f"witness weights: {witness}")
print(f"witness pencil coefficients: {witness_coefficients}")
print("witness squared frequencies: (771 +- 3*sqrt(9489))/1414")
print("PASS")
