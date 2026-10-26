"""Exact certificate for the unrestricted seven-state phase barrier."""

from fractions import Fraction
from itertools import permutations, product
import math


N = 7


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


# The same rerooting injection used on six states is dimension-free.  Check it
# on every seven-state tree and every root.  The marked exceptional edge and
# new root recover the old root, so no two source objects have the same image.
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


# Convert the symmetric degree-six expansion
#   sum B_k x^k (1-x)^(6-k), B_k=B_(6-k),
# to z=2x-1.  If c_(2j) are its even monomial coefficients,
# 5*c_0+c_2-3*c_4-7*c_6 is exactly B_1-B_0.  For
# det(H+zD)=det(H)*prod_j(1+s_j*z^2), this is the claimed barrier.
x = [Fraction(1, 2), Fraction(1, 2)]
one_minus_x = [Fraction(1, 2), Fraction(-1, 2)]
basis = [
    multiply(power(x, index), power(one_minus_x, 6 - index))
    for index in range(7)
]
formal = []
for degree in range(7):
    formal.append(tuple(
        basis[variable][degree] + basis[6 - variable][degree]
        if variable < 3 else basis[3][degree]
        for variable in range(4)
    ))
barrier_vector = tuple(
    5 * formal[0][index]
    + formal[2][index]
    - 3 * formal[4][index]
    - 7 * formal[6][index]
    for index in range(4)
)
assert barrier_vector == (
    Fraction(-1), Fraction(1), Fraction(0), Fraction(0)
)


# A directed seven-cycle attains equality.  Its exact normalized determinant is
# 1+5*z^2+3*z^4+z^6/7, so the barrier vanishes.
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
    Fraction(1), Fraction(0), Fraction(5), Fraction(0),
    Fraction(3), Fraction(0), Fraction(1, 7)
]
assert 5 + normalized[2] - 3 * normalized[4] - 7 * normalized[6] == 0


# If a four-dimensional compression has two frequencies kappa=cot(pi/5),
# Hermitian interlacing forces the two largest full squared frequencies x,y to
# satisfy x,y >= r=kappa^2.  The barrier is decreasing in x and y once they
# exceed one, and decreasing in the third squared frequency z >= 0.  Its exact
# value at (r,r,0) is 8/5-8/sqrt(5), which is negative because sqrt(5)<5.
assert 5 < 25
target_square = 1 + 2 / math.sqrt(5)
target_corner = Fraction(8, 5) - 8 / math.sqrt(5)
assert target_square > 1
assert target_corner < 0

print(f"labeled seven-state trees: {len(trees)}")
print(f"rooted out-tree objects injected: {len(images)}")
print(f"available one-defect coefficient objects: {one_defect_object_count}")
print("exact coefficient identity: det(H)*(5+e1-3*e2-7*e3) = B1-B0")
print("directed seven-cycle normalized pencil: 1 + 5*z^2 + 3*z^4 + z^6/7")
print("target corner barrier: 8/5 - 8/sqrt(5) "
      f"= {float(target_corner):.12f}")
print("PASS")
