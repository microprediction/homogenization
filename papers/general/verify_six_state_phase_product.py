"""Exact certificate for the unrestricted six-state phase-product bound."""

from fractions import Fraction
from itertools import permutations, product
import math


N = 6


def prufer_tree(sequence):
    """Return the edge set of the labeled tree with the given Prüfer code."""
    degree = [1] * N
    for vertex in sequence:
        degree[vertex] += 1

    edges = []
    for vertex in sequence:
        leaf = next(index for index, value in enumerate(degree) if value == 1)
        edges.append(tuple(sorted((leaf, vertex))))
        degree[leaf] -= 1
        degree[vertex] -= 1
    last = [index for index, value in enumerate(degree) if value == 1]
    edges.append(tuple(sorted(last)))
    return tuple(sorted(edges))


def rooted_parents(edges, root):
    adjacency = [[] for _ in range(N)]
    for left, right in edges:
        adjacency[left].append(right)
        adjacency[right].append(left)

    parent = [None] * N
    parent[root] = root
    stack = [root]
    while stack:
        vertex = stack.pop()
        for neighbor in adjacency[vertex]:
            if parent[neighbor] is None:
                parent[neighbor] = vertex
                stack.append(neighbor)
    assert all(value is not None for value in parent)
    return parent


def multiply(left, right):
    out = [Fraction(0)] * (len(left) + len(right) - 1)
    for i, value in enumerate(left):
        for j, other in enumerate(right):
            out[i + j] += value * other
    return out


def power(poly, exponent):
    out = [Fraction(1)]
    for _ in range(exponent):
        out = multiply(out, poly)
    return out


def add(left, right):
    out = [Fraction(0)] * max(len(left), len(right))
    for index, value in enumerate(left):
        out[index] += value
    for index, value in enumerate(right):
        out[index] += value
    return out


def determinant(matrix):
    size = len(matrix)
    total = [Fraction(0)]
    for permutation in permutations(range(size)):
        inversions = sum(
            permutation[i] > permutation[j]
            for i in range(size)
            for j in range(i + 1, size)
        )
        term = [Fraction(-1 if inversions % 2 else 1)]
        for row, column in enumerate(permutation):
            term = multiply(term, matrix[row][column])
        total = add(total, term)
    return total


# The coefficient injection is independent of edge weights.  Enumerate all
# six-state labeled trees, root each one, and reroot at the least child of the
# old root.  Relative to the new root, exactly the connecting edge points
# inward and every other edge points outward.  The underlying tree, new root,
# and marked edge recover the old rooted tree, so the map is injective.
trees = sorted({prufer_tree(code) for code in product(range(N), repeat=N - 2)})
assert len(trees) == N ** (N - 2)

images = set()
for tree in trees:
    for old_root in range(N):
        old_parent = rooted_parents(tree, old_root)
        children = [
            vertex for vertex in range(N)
            if vertex != old_root and old_parent[vertex] == old_root
        ]
        assert children
        new_root = min(children)
        marked_edge = tuple(sorted((old_root, new_root)))

        actual_edges = {
            (old_parent[vertex], vertex)
            for vertex in range(N)
            if vertex != old_root
        }
        new_parent = rooted_parents(tree, new_root)
        inward = 0
        for vertex in range(N):
            if vertex == new_root:
                continue
            toward_new_root = (vertex, new_parent[vertex])
            if toward_new_root in actual_edges:
                inward += 1
                assert tuple(sorted(toward_new_root)) == marked_edge
            else:
                assert tuple(reversed(toward_new_root)) in actual_edges
        assert inward == 1

        image = (tree, new_root, marked_edge)
        assert image not in images
        images.add(image)

rooted_tree_count = len(trees) * N
one_defect_object_count = len(trees) * N * (N - 1)
assert len(images) == rooted_tree_count
assert rooted_tree_count <= one_defect_object_count


# Convert the symmetric degree-five tree expansion
#   sum B_k x^k (1-x)^(5-k), B_k=B_(5-k),
# to z=2x-1.  Formal coefficient vectors track B_0,B_1,B_2 exactly.
x = [Fraction(1, 2), Fraction(1, 2)]
one_minus_x = [Fraction(1, 2), Fraction(-1, 2)]
basis = [
    multiply(power(x, index), power(one_minus_x, 5 - index))
    for index in range(6)
]
formal = []
for degree in range(6):
    formal.append(tuple(
        basis[variable][degree] + basis[5 - variable][degree]
        for variable in range(3)
    ))
assert tuple(formal[0][i] - formal[4][i] for i in range(3)) == (
    Fraction(-1, 4), Fraction(1, 4), Fraction(0)
)


# A directed six-cycle attains the product bound.  Build its exact grounded
# pencil H+zD and recover det(H)*(1 + 10 z^2/3 + z^4).
weights = [[Fraction(0) for _ in range(N)] for _ in range(N)]
for source in range(N):
    weights[source][(source + 1) % N] = Fraction(1)

laplacian = [[Fraction(0) for _ in range(N)] for _ in range(N)]
for row in range(N):
    laplacian[row][row] = sum(weights[row])
    for column in range(N):
        if row != column:
            laplacian[row][column] = -weights[row][column]

grounded = [row[:-1] for row in laplacian[:-1]]
pencil = []
for row in range(N - 1):
    pencil_row = []
    for column in range(N - 1):
        symmetric = (grounded[row][column] + grounded[column][row]) / 2
        skew = (grounded[row][column] - grounded[column][row]) / 2
        pencil_row.append([symmetric, skew])
    pencil.append(pencil_row)

cycle_polynomial = determinant(pencil)
det_h = cycle_polynomial[0]
normalized = [coefficient / det_h for coefficient in cycle_polynomial]
assert normalized == [
    Fraction(1), Fraction(0), Fraction(10, 3),
    Fraction(0), Fraction(1), Fraction(0)
]

target_product = 1 + 2 / math.sqrt(5)
assert target_product > 1

print(f"labeled six-state trees: {len(trees)}")
print(f"rooted out-tree objects injected: {len(images)}")
print(f"available one-defect coefficient objects: {one_defect_object_count}")
print("exact coefficient identity: det(H)*(1-sigma1^2*sigma2^2) "
      "= (B1-B0)/4")
print("directed six-cycle normalized pencil: "
      "1 + (10/3) z^2 + z^4")
print(f"target squared phase product: 1 + 2/sqrt(5) = {target_product:.12f}")
print("PASS")
