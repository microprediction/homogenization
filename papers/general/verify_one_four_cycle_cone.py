"""Exact full-cone certificate for one fourth cycle at the sharp triple."""

from fractions import Fraction
from math import comb

from verify_four_cycle_tangent import (
    GROUND,
    PERMUTATIONS,
    SIGNS,
    base_forms,
    integer_forms,
)


FOURTH_CYCLE = (0, 1, 2, 4)
DEGREE = 5
LOCAL_RADIUS = Fraction(1, 1024)


def poly_add(*terms):
    out = {}
    for scale, poly in terms:
        for exponent, coefficient in poly.items():
            out[exponent] = out.get(exponent, 0) + scale * coefficient
    return {key: Fraction(value) for key, value in out.items() if value}


def poly_mul(left, right):
    out = {}
    for left_exp, left_value in left.items():
        for right_exp, right_value in right.items():
            exponent = tuple(a + b for a, b in zip(left_exp, right_exp))
            out[exponent] = out.get(exponent, 0) + left_value * right_value
    return {key: value for key, value in out.items() if value}


def determinant_poly(forms, t):
    out = {}
    variables = len(forms)
    for perm, sign in zip(PERMUTATIONS, SIGNS):
        term = {(0,) * variables: sign}
        for row in range(GROUND):
            entry = {}
            for index, (symmetric, skew) in enumerate(forms):
                exponent = tuple(int(index == j) for j in range(variables))
                value = t * symmetric[row][perm[row]] - skew[row][perm[row]]
                if value:
                    entry[exponent] = value
            term = poly_mul(term, entry)
        out = poly_add((1, out), (1, term))
    return out


def barrier_pair():
    forms = base_forms + (integer_forms(FOURTH_CYCLE),)
    values = [determinant_poly(forms, t) for t in (1, 2, 3)]
    keys = set().union(*values)
    a5 = {
        key: Fraction(values[2].get(key, 0) - 4 * values[1].get(key, 0)
                      + 5 * values[0].get(key, 0), 120)
        for key in keys
    }
    a5 = {key: value for key, value in a5.items() if value}
    a3 = {
        key: Fraction(values[1].get(key, 0) - 2 * values[0].get(key, 0), 6)
        - 5 * a5.get(key, 0)
        for key in keys
    }
    a3 = {key: value for key, value in a3.items() if value}
    a1 = {
        key: values[0].get(key, 0) - a5.get(key, 0) - a3.get(key, 0)
        for key in keys
    }
    a1 = {key: value for key, value in a1.items() if value}
    return (
        poly_add((5, a3), (-6, a5)),
        poly_add((9, a5), (-15, a3), (25, a1)),
    )


def substitute(poly, substitutions):
    out = {}
    zero = (0,) * len(next(iter(substitutions[0])))
    for powers, coefficient in poly.items():
        term = {zero: coefficient}
        for index, power in enumerate(powers):
            for _ in range(power):
                term = poly_mul(term, substitutions[index])
        out = poly_add((1, out), (1, term))
    return out


def compactify(poly):
    """Set base sum to three and d=z/(1-z), clearing (1-z)^5."""
    x = {(1, 0, 0): 3}
    one_minus_x_y = {(0, 1, 0): 3, (1, 1, 0): -3}
    one_minus_x_one_minus_y = {
        (0, 0, 0): 3,
        (1, 0, 0): -3,
        (0, 1, 0): -3,
        (1, 1, 0): 3,
    }
    z = {(0, 0, 1): 1}
    out = substitute(poly, (x, one_minus_x_y, one_minus_x_one_minus_y, z))
    cleared = {}
    for (i, j, k), coefficient in out.items():
        for extra in range(DEGREE - k + 1):
            exponent = (i, j, k + extra)
            cleared[exponent] = cleared.get(exponent, 0) + (
                coefficient * comb(DEGREE - k, extra) * (-1) ** extra
            )
    return {key: value for key, value in cleared.items() if value}


def localize(poly):
    """Set weights to (1+p,1+q,1-p-q,v)."""
    return substitute(poly, (
        {(0, 0, 0): 1, (1, 0, 0): 1},
        {(0, 0, 0): 1, (0, 1, 0): 1},
        {(0, 0, 0): 1, (1, 0, 0): -1, (0, 1, 0): -1},
        {(0, 0, 1): 1},
    ))


def bernstein_controls(poly):
    controls = {}
    for i in range(DEGREE + 1):
        for j in range(DEGREE + 1):
            for k in range(DEGREE + 1):
                total = Fraction(0)
                for (r, s, t), coefficient in poly.items():
                    if r <= i and s <= j and t <= k:
                        total += coefficient * Fraction(
                            comb(i, r) * comb(j, s) * comb(k, t),
                            comb(DEGREE, r) * comb(DEGREE, s) * comb(DEGREE, t),
                        )
                controls[i, j, k] = total
    return controls


def split_curve(values):
    levels = [list(values)]
    while len(levels[-1]) > 1:
        old = levels[-1]
        levels.append([(old[i] + old[i + 1]) / 2 for i in range(len(old) - 1)])
    return (
        [levels[i][0] for i in range(DEGREE + 1)],
        [levels[DEGREE - i][i] for i in range(DEGREE + 1)],
    )


def split_axis(controls, axis):
    left, right = {}, {}
    other = [index for index in range(3) if index != axis]
    for first in range(DEGREE + 1):
        for second in range(DEGREE + 1):
            fixed = {other[0]: first, other[1]: second}
            line = []
            for moving in range(DEGREE + 1):
                index = tuple(moving if i == axis else fixed[i] for i in range(3))
                line.append(controls[index])
            halves = split_curve(line)
            for side, target in zip(halves, (left, right)):
                for moving, value in enumerate(side):
                    index = tuple(moving if i == axis else fixed[i] for i in range(3))
                    target[index] = value
    return left, right


def nonpositive(controls):
    return max(controls.values()) <= 0


def wholly_local(box):
    (x0, x1), (y0, y1), (z0, z1) = box
    p_values = (3 * x0 - 1, 3 * x1 - 1)
    q_values = (3 * (1 - x1) * y0 - 1, 3 * (1 - x0) * y1 - 1)
    if z1 == 1:
        return False
    v1 = z1 / (1 - z1)
    return (
        max(map(abs, p_values)) <= LOCAL_RADIUS
        and max(map(abs, q_values)) <= LOCAL_RADIUS
        and v1 <= LOCAL_RADIUS
    )


def cover(sum_controls, root_controls, maximum_depth=48):
    stack = [(sum_controls, root_controls, ((Fraction(0), Fraction(1)),) * 3, 0)]
    terminals = local_cells = deepest = 0
    while stack:
        first, second, box, depth = stack.pop()
        if nonpositive(first) or nonpositive(second):
            terminals += 1
            deepest = max(deepest, depth)
            continue
        if wholly_local(box):
            local_cells += 1
            deepest = max(deepest, depth)
            continue
        assert depth < maximum_depth, (box, max(first.values()), max(second.values()))
        axis = depth % 3
        first_halves = split_axis(first, axis)
        second_halves = split_axis(second, axis)
        lo, hi = box[axis]
        middle = (lo + hi) / 2
        for side in range(2):
            child_box = list(box)
            child_box[axis] = (lo, middle) if side == 0 else (middle, hi)
            stack.append((first_halves[side], second_halves[side], tuple(child_box), depth + 1))
    return terminals, local_cells, deepest


sum_barrier, root_barrier = barrier_pair()
local_root = localize(root_barrier)
assert all(sum(key) >= 2 for key in local_root)
quadratic = {key: value for key, value in local_root.items() if sum(key) == 2}
assert quadratic == {
    (2, 0, 0): -576, (1, 1, 0): -576, (0, 2, 0): -576,
    (1, 0, 1): -960, (0, 1, 1): -192, (0, 0, 2): -640,
}

# The matrix of -quadratic minus 64 I is positive definite by Sylvester.
positive_matrix = (
    (512, 288, 480),
    (288, 512, 96),
    (480, 96, 576),
)
principal_minors = (
    positive_matrix[0][0],
    positive_matrix[0][0] * positive_matrix[1][1]
    - positive_matrix[0][1] ** 2,
    positive_matrix[0][0]
    * (positive_matrix[1][1] * positive_matrix[2][2]
       - positive_matrix[1][2] * positive_matrix[2][1])
    - positive_matrix[0][1]
    * (positive_matrix[1][0] * positive_matrix[2][2]
       - positive_matrix[1][2] * positive_matrix[2][0])
    + positive_matrix[0][2]
    * (positive_matrix[1][0] * positive_matrix[2][1]
       - positive_matrix[1][1] * positive_matrix[2][0]),
)
assert principal_minors == (512, 179200, 7077888)
higher_absolute_sum = sum(
    abs(value) for key, value in local_root.items() if sum(key) >= 3
)
# On the local cube, every monomial of degree m >= 3 is at most
# radius^(m-2) (p^2+q^2+v^2) in absolute value.
assert higher_absolute_sum * LOCAL_RADIUS <= 64

terminal_cells, local_cells, maximum_depth = cover(
    bernstein_controls(compactify(sum_barrier)),
    bernstein_controls(compactify(root_barrier)),
)

print(f"fourth cycle: {FOURTH_CYCLE}")
print(f"local radius: {LOCAL_RADIUS}")
print(f"higher-order coefficient bound: {higher_absolute_sum}")
print(f"positive-definiteness minors: {principal_minors}")
print(f"global terminal boxes: {terminal_cells}")
print(f"local terminal boxes: {local_cells}")
print(f"maximum subdivision depth: {maximum_depth}")
print("full nonnegative four-weight cone: lower squared frequency <= 3/5")
print("PASS")
