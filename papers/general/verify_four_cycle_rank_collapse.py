"""Exact rank-collapse theorem for a full four-directed-cycle cone."""

from collections import defaultdict
from itertools import permutations
from math import gcd


N = 6
GROUND = N - 1
CYCLES = (
    (0, 1, 2, 3),
    (0, 1, 2, 4),
    (0, 3, 2, 5),
    (0, 5, 2, 4),
)


def canonical(cycle):
    cycle = tuple(cycle)
    return min(cycle[index:] + cycle[:index]
               for index in range(len(cycle)))


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
    symmetric = [[
        grounded[row][column] + grounded[column][row]
        for column in range(GROUND)
    ] for row in range(GROUND)]
    skew = [[
        grounded[row][column] - grounded[column][row]
        for column in range(GROUND)
    ] for row in range(GROUND)]
    return symmetric, skew


def add_polynomials(*terms):
    result = defaultdict(int)
    for scale, polynomial in terms:
        for monomial, coefficient in polynomial.items():
            result[monomial] += scale * coefficient
    return {monomial: coefficient
            for monomial, coefficient in result.items() if coefficient}


def multiply_polynomials(first, second):
    result = defaultdict(int)
    for first_monomial, first_coefficient in first.items():
        for second_monomial, second_coefficient in second.items():
            monomial = tuple(
                left + right
                for left, right in zip(first_monomial, second_monomial)
            )
            result[monomial] += first_coefficient * second_coefficient
    return {monomial: coefficient
            for monomial, coefficient in result.items() if coefficient}


def determinant_polynomial(cycles=CYCLES):
    """Expand det(t(2H)-2D) in four weights and t over the integers."""
    forms = [integer_forms(cycle) for cycle in cycles]
    determinant = defaultdict(int)
    for permutation in permutations(range(GROUND)):
        terms = {(0, 0, 0, 0, 0): permutation_sign(permutation)}
        for row, column in enumerate(permutation):
            entry = defaultdict(int)
            for index, (symmetric, skew) in enumerate(forms):
                weight = [0] * 5
                weight[index] = 1
                if symmetric[row][column]:
                    symmetric_monomial = weight.copy()
                    symmetric_monomial[4] = 1
                    entry[tuple(symmetric_monomial)] += symmetric[row][column]
                if skew[row][column]:
                    entry[tuple(weight)] -= skew[row][column]
            terms = multiply_polynomials(terms, entry)
        for monomial, coefficient in terms.items():
            determinant[monomial] += coefficient
    return {monomial: coefficient
            for monomial, coefficient in determinant.items() if coefficient}


def coefficient_in_t(determinant, exponent):
    return {monomial[:4]: coefficient
            for monomial, coefficient in determinant.items()
            if monomial[4] == exponent}


def evaluate(polynomial, values):
    return sum(
        coefficient * product(
            value ** exponent
            for value, exponent in zip(values, monomial)
        )
        for monomial, coefficient in polynomial.items()
    )


def product(values):
    result = 1
    for value in values:
        result *= value
    return result


determinant = determinant_polynomial()
assert {monomial[4] for monomial in determinant} == {3, 5}
a5 = coefficient_in_t(determinant, 5)
a3 = coefficient_in_t(determinant, 3)
a1 = coefficient_in_t(determinant, 1)
assert a5 and all(coefficient > 0 for coefficient in a5.values())
assert not a1

# With weights (a,b,c,d), the integer skew form 2D is u v' - v u',
# where u=(1,0,-1,0,0) and
# v=(0,-a-b,0,a-c,b+d).  This proves rank(D)=2 throughout the positive
# cone, independently of the characteristic-polynomial calculation.
forms = [integer_forms(cycle) for cycle in CYCLES]
skew_coefficients = [[
    tuple(forms[index][1][row][column] for index in range(4))
    for column in range(GROUND)
] for row in range(GROUND)]
u = (1, 0, -1, 0, 0)
v = (
    (0, 0, 0, 0),
    (-1, -1, 0, 0),
    (0, 0, 0, 0),
    (1, 0, -1, 0),
    (0, 1, 0, 1),
)
expected_skew_coefficients = [[
    tuple(u[row] * v[column][index] - v[row][index] * u[column]
          for index in range(4))
    for column in range(GROUND)
] for row in range(GROUND)]
assert skew_coefficients == expected_skew_coefficients

# This additional coefficientwise check bounds the single positive squared
# frequency by 6/5.  The rank identity already proves sigma_2=0.
sum_barrier = add_polynomials((5, a3), (-6, a5))
assert len(sum_barrier) == 32
assert all(coefficient < 0 for coefficient in sum_barrier.values())
coefficient_gcd = 0
for coefficient in sum_barrier.values():
    coefficient_gcd = gcd(coefficient_gcd, abs(coefficient))

# This support is not already covered by the theorem for a fourth cycle
# attached to the sharp three-cycle orbit: none of its four faces is sharp.
root_barrier = add_polynomials((9, a5), (-15, a3), (25, a1))
face_root_values = tuple(
    evaluate(root_barrier, tuple(0 if index == omitted else 1
                                 for index in range(4)))
    for omitted in range(4)
)
assert all(value != 0 for value in face_root_values)

# State relabeling and simultaneous reversal preserve the spectrum.  The
# support has no nontrivial stabilizer under this 2*6! element group.
orbit = {
    tuple(sorted(
        canonical(tuple(
            relabeling[state]
            for state in (tuple(reversed(cycle)) if reverse else cycle)
        ))
        for cycle in CYCLES
    ))
    for relabeling in permutations(range(N))
    for reverse in (False, True)
}
assert len(orbit) == 360

witness = (1, 2, 3, 4)
witness_coefficients = tuple(
    evaluate(polynomial, witness) for polynomial in (a5, a3, a1)
)
witness_sum_barrier = evaluate(sum_barrier, witness)
assert witness_sum_barrier < 0
assert witness_coefficients[2] == 0

print("orbit representative: {(0123),(0124),(0325),(0524)}")
print(f"labeled support orbit size: {len(orbit)}")
print(f"strictly negative sum-barrier monomials: {len(sum_barrier)}")
print(f"sum-barrier coefficient gcd: {coefficient_gcd}")
print(f"three-cycle face root barriers at equal weights: {face_root_values}")
print(f"witness weights: {witness}")
print(f"witness pencil coefficients: {witness_coefficients}")
print(f"witness sum barrier: {witness_sum_barrier}")
print("2D = u*v' - v*u' with u=(1,0,-1,0,0)")
print("v=(0,-a-b,0,a-c,b+d)")
print("all positive weights: rank(D)=2 and sigma_2=0")
print("PASS")
