"""Exact rank-collapse theorem for a full four-directed-cycle cone."""

from collections import Counter, defaultdict
from itertools import combinations, permutations
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

# Classify every four-component cone with the same rank-collapse property.
# A 5-by-5 skew matrix has rank at most two exactly when its five principal
# 4-by-4 Pfaffians vanish.  Those Pfaffians are quadratic in the cycle
# weights.  Their diagonal terms vanish for each individual cycle, so an
# entire cone has rank at most two exactly when every pair has zero polarized
# Pfaffian.  Rank-collapse supports are therefore cliques in one exact graph.
four_cycles = tuple(sorted({
    canonical(ordering)
    for support in combinations(range(N), 4)
    for ordering in permutations(support)
}))
assert len(four_cycles) == 90
cycle_index = {cycle: index for index, cycle in enumerate(four_cycles)}
skew_edges = tuple(combinations(range(GROUND), 2))
skew_index = {edge: index for index, edge in enumerate(skew_edges)}
skew_vectors = tuple(
    tuple(integer_forms(cycle)[1][row][column]
          for row, column in skew_edges)
    for cycle in four_cycles
)


def skew_entry(vector, row, column):
    if row < column:
        return vector[skew_index[(row, column)]]
    return -vector[skew_index[(column, row)]]


def polarized_pfaffian(left, right, quadruple):
    """Cross coefficient of a 4-by-4 Pfaffian."""
    a, b, c, d = quadruple
    return (
        skew_entry(left, a, b) * skew_entry(right, c, d)
        + skew_entry(right, a, b) * skew_entry(left, c, d)
        - skew_entry(left, a, c) * skew_entry(right, b, d)
        - skew_entry(right, a, c) * skew_entry(left, b, d)
        + skew_entry(left, a, d) * skew_entry(right, b, c)
        + skew_entry(right, a, d) * skew_entry(left, b, c)
    )


quadruples = tuple(combinations(range(GROUND), 4))
assert all(
    polarized_pfaffian(vector, vector, quadruple) == 0
    for vector in skew_vectors
    for quadruple in quadruples
)
adjacency = [0] * len(four_cycles)
for first, second in combinations(range(len(four_cycles)), 2):
    if all(polarized_pfaffian(
            skew_vectors[first], skew_vectors[second], quadruple) == 0
           for quadruple in quadruples):
        adjacency[first] |= 1 << second
        adjacency[second] |= 1 << first

degrees = tuple(mask.bit_count() for mask in adjacency)
compatible_pair_count = sum(degrees) // 2
assert compatible_pair_count == 1125
assert set(degrees) == {25}


def indices(mask):
    while mask:
        bit = mask & -mask
        yield bit.bit_length() - 1
        mask ^= bit


# Enumerate each 4-clique once in increasing vertex order.
rank_collapse_supports = []
all_vertices = (1 << len(four_cycles)) - 1
for first in range(len(four_cycles)):
    after_first = all_vertices ^ ((1 << (first + 1)) - 1)
    for second in indices(adjacency[first] & after_first):
        after_second = all_vertices ^ ((1 << (second + 1)) - 1)
        common = adjacency[first] & adjacency[second] & after_second
        for third in indices(common):
            after_third = all_vertices ^ ((1 << (third + 1)) - 1)
            for fourth in indices(common & adjacency[third] & after_third):
                rank_collapse_supports.append(
                    frozenset((first, second, third, fourth)))
assert len(rank_collapse_supports) == 7830


def connected(support):
    neighbors = [set() for _ in range(N)]
    for cycle_id in support:
        cycle = four_cycles[cycle_id]
        for source, target in zip(cycle, cycle[1:] + cycle[:1]):
            neighbors[source].add(target)
            neighbors[target].add(source)
    seen = {next(state for state in range(N) if neighbors[state])}
    frontier = list(seen)
    while frontier:
        for state in neighbors[frontier.pop()]:
            if state not in seen:
                seen.add(state)
                frontier.append(state)
    return len(seen) == N


connected_supports = tuple(filter(connected, rank_collapse_supports))
assert len(connected_supports) == 6705


# Exact Bron--Kerbosch enumeration of the maximal compatible families.
maximal_families = []


def maximal_cliques(clique, candidates, excluded):
    if not candidates and not excluded:
        maximal_families.append(frozenset(indices(clique)))
        return
    pivot_pool = candidates | excluded
    pivot = max(indices(pivot_pool),
                key=lambda vertex: (candidates & adjacency[vertex]).bit_count())
    extension = candidates & ~adjacency[pivot]
    for vertex in tuple(indices(extension)):
        vertex_bit = 1 << vertex
        maximal_cliques(
            clique | vertex_bit,
            candidates & adjacency[vertex],
            excluded & adjacency[vertex],
        )
        candidates &= ~vertex_bit
        excluded |= vertex_bit


maximal_cliques(0, all_vertices, 0)
maximal_family_set = set(maximal_families)
assert len(maximal_family_set) == 45
assert Counter(map(len, maximal_family_set)) == {6: 30, 12: 15}


def cycle_id(cycle):
    return cycle_index[canonical(cycle)]


face_families = {
    frozenset(cycle_index[cycle] for cycle in four_cycles
              if set(cycle) == set(support))
    for support in combinations(range(N), 4)
}
star_families = {
    frozenset(cycle_id((first, left, second, right))
              for left in range(N) if left not in (first, second)
              for right in range(N) if right not in (first, second, left))
    for first, second in combinations(range(N), 2)
}


def perfect_matchings(vertices):
    if not vertices:
        yield ()
        return
    first = vertices[0]
    for position in range(1, len(vertices)):
        second = vertices[position]
        remaining = vertices[1:position] + vertices[position + 1:]
        for matching in perfect_matchings(remaining):
            yield ((first, second),) + matching


matching_families = set()
for matching in perfect_matchings(tuple(range(N))):
    family = set()
    for (a, b), (c, d) in combinations(matching, 2):
        family.add(cycle_id((a, c, b, d)))
        family.add(cycle_id((a, d, b, c)))
    matching_families.add(frozenset(family))

assert tuple(map(len, (face_families, star_families,
                       matching_families))) == (15, 15, 15)
assert maximal_family_set == (
    face_families | star_families | matching_families)
assert all(any(support <= family for family in maximal_family_set)
           for support in rank_collapse_supports)


def transformed_support(support, relabeling, reverse):
    return frozenset(
        cycle_id(tuple(
            relabeling[state]
            for state in (tuple(reversed(four_cycles[index]))
                          if reverse else four_cycles[index])
        ))
        for index in support
    )


def support_orbit(support):
    return {
        transformed_support(support, relabeling, reverse)
        for relabeling in permutations(range(N))
        for reverse in (False, True)
    }


rank_collapse_support_set = set(rank_collapse_supports)
unclassified = set(rank_collapse_supports)
support_orbits = []
while unclassified:
    representative = next(iter(unclassified))
    support_orbits.append(support_orbit(representative))
    unclassified -= support_orbits[-1]
assert all(orbit_class <= rank_collapse_support_set
           for orbit_class in support_orbits)
assert len(support_orbits) == 21
assert sum(all(connected(support) for support in orbit_class)
           for orbit_class in support_orbits) == 16

for family_type in (face_families, star_families, matching_families):
    assert support_orbit(next(iter(family_type))) == family_type

witness_support = frozenset(cycle_index[canonical(cycle)]
                            for cycle in CYCLES)
assert witness_support in connected_supports
assert len(support_orbit(witness_support)) == 360

witness = (1, 2, 3, 4)
witness_coefficients = tuple(
    evaluate(polynomial, witness) for polynomial in (a5, a3, a1)
)
witness_sum_barrier = evaluate(sum_barrier, witness)
assert witness_sum_barrier < 0
assert witness_coefficients[2] == 0

print("orbit representative: {(0123),(0124),(0325),(0524)}")
print(f"labeled support orbit size: {len(orbit)}")
print(f"directed four-cycles: {len(four_cycles)}")
print(f"compatible pairs: {compatible_pair_count}")
print(f"compatibility degree: {degrees[0]}")
print(f"four-component rank-collapse supports: {len(rank_collapse_supports)}")
print(f"connected rank-collapse supports: {len(connected_supports)}")
print(f"support orbits: {len(support_orbits)} total, 16 connected")
print(f"maximal compatible families: {len(maximal_family_set)}")
print("maximal family types: 15 faces, 15 opposite-pair stars, "
      "15 perfect-matchings")
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
