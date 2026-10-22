"""Exact two-parameter slices through the sharp six-state four-cycle cones."""

from fractions import Fraction
from math import comb

from verify_four_cycle_tangent import (
    GROUND,
    PERMUTATIONS,
    SIGNS,
    all_cycles,
    base_forms,
    integer_forms,
)


def poly_add(*terms):
    out = {}
    for scale, poly in terms:
        for exponent, coefficient in poly.items():
            out[exponent] = out.get(exponent, 0) + scale * coefficient
    return {exponent: Fraction(value) for exponent, value in out.items() if value}


def poly_mul(left, right):
    out = {}
    for (i, j), left_value in left.items():
        for (k, ell), right_value in right.items():
            exponent = (i + k, j + ell)
            out[exponent] = out.get(exponent, 0) + left_value * right_value
    return {exponent: value for exponent, value in out.items() if value}


def determinant_poly(constant_h, linear_h, fourth_h,
                     constant_d, linear_d, fourth_d, t):
    matrix = []
    for row in range(GROUND):
        matrix_row = []
        for column in range(GROUND):
            entry = {
                (0, 0): t * constant_h[row][column] - constant_d[row][column],
                (1, 0): t * linear_h[row][column] - linear_d[row][column],
                (0, 1): t * fourth_h[row][column] - fourth_d[row][column],
            }
            matrix_row.append({key: value for key, value in entry.items() if value})
        matrix.append(matrix_row)

    out = {}
    for perm, sign in zip(PERMUTATIONS, SIGNS):
        term = {(0, 0): sign}
        for row in range(GROUND):
            term = poly_mul(term, matrix[row][perm[row]])
        out = poly_add((1, out), (1, term))
    return out


def barrier_pair(cycle, first, second):
    fourth_h, fourth_d = integer_forms(cycle)
    constant_h = [[
        sum(base_forms[index][0][row][column] for index in range(3))
        for column in range(GROUND)
    ] for row in range(GROUND)]
    constant_d = [[
        sum(base_forms[index][1][row][column] for index in range(3))
        for column in range(GROUND)
    ] for row in range(GROUND)]
    linear_h = [[
        base_forms[first][0][row][column] - base_forms[second][0][row][column]
        for column in range(GROUND)
    ] for row in range(GROUND)]
    linear_d = [[
        base_forms[first][1][row][column] - base_forms[second][1][row][column]
        for column in range(GROUND)
    ] for row in range(GROUND)]
    values = [
        determinant_poly(constant_h, linear_h, fourth_h,
                         constant_d, linear_d, fourth_d, t)
        for t in (1, 2, 3)
    ]
    a5 = poly_add(
        (Fraction(1, 120), values[2]),
        (Fraction(-1, 30), values[1]),
        (Fraction(1, 24), values[0]),
    )
    a3 = poly_add(
        (Fraction(1, 6), values[1]),
        (Fraction(-1, 3), values[0]),
        (-5, a5),
    )
    a1 = poly_add((1, values[0]), (-1, a5), (-1, a3))
    return (
        poly_add((5, a3), (-6, a5)),
        poly_add((9, a5), (-15, a3), (25, a1)),
    )


def compactify(poly):
    """Substitute u=2x-1 and v=y/(1-y), then multiply by (1-y)^5."""
    out = {}
    for (u_power, v_power), coefficient in poly.items():
        for x_power in range(u_power + 1):
            x_coefficient = (
                coefficient * comb(u_power, x_power) * 2 ** x_power
                * (-1) ** (u_power - x_power)
            )
            for extra_y in range(6 - v_power):
                exponent = (x_power, v_power + extra_y)
                out[exponent] = out.get(exponent, 0) + (
                    x_coefficient * comb(5 - v_power, extra_y)
                    * (-1) ** extra_y
                )
    return {exponent: Fraction(value) for exponent, value in out.items() if value}


def blow_up(poly, patch, sign=1):
    """Return a positive-scale quotient on one rational local patch."""
    vanishing_order = min(sum(exponent) for exponent in poly)
    out = {}
    for (u_power, v_power), coefficient in poly.items():
        if patch == "central":
            for x_power in range(u_power + 1):
                exponent = (x_power, u_power + v_power - vanishing_order)
                out[exponent] = out.get(exponent, 0) + (
                    coefficient * comb(u_power, x_power)
                    * 6 ** x_power * (-3) ** (u_power - x_power)
                    * Fraction(1, 64) ** (u_power + v_power)
                )
        else:
            exponent = (
                u_power + v_power - vanishing_order,
                v_power,
            )
            out[exponent] = out.get(exponent, 0) + (
                coefficient * (sign * 3) ** u_power
                * Fraction(1, 64) ** (u_power + v_power)
            )
    return {exponent: value for exponent, value in out.items() if value}


def bernstein_controls(poly, x_degree=None, y_degree=None):
    x_degree = x_degree if x_degree is not None else max(i for i, _ in poly)
    y_degree = y_degree if y_degree is not None else max(j for _, j in poly)
    return tuple(tuple(
        sum(
            coefficient
            * Fraction(comb(i, x_power), comb(x_degree, x_power))
            * Fraction(comb(j, y_power), comb(y_degree, y_power))
            for (x_power, y_power), coefficient in poly.items()
            if x_power <= i and y_power <= j
        )
        for j in range(y_degree + 1)
    ) for i in range(x_degree + 1))


def split_curve(controls):
    level = list(controls)
    left, right = [level[0]], [level[-1]]
    while len(level) > 1:
        level = [(a + b) / 2 for a, b in zip(level, level[1:])]
        left.append(level[0])
        right.append(level[-1])
    return tuple(left), tuple(reversed(right))


def split_x(controls):
    left = [[] for _ in controls]
    right = [[] for _ in controls]
    for column in range(len(controls[0])):
        column_left, column_right = split_curve(
            [controls[row][column] for row in range(len(controls))]
        )
        for row in range(len(controls)):
            left[row].append(column_left[row])
            right[row].append(column_right[row])
    return tuple(map(tuple, left)), tuple(map(tuple, right))


def split_y(controls):
    left, right = [], []
    for row in controls:
        row_left, row_right = split_curve(row)
        left.append(row_left)
        right.append(row_right)
    return tuple(left), tuple(right)


def children(controls):
    left, right = split_x(controls)
    lower_left, upper_left = split_y(left)
    lower_right, upper_right = split_y(right)
    return lower_left, upper_left, lower_right, upper_right


def nonpositive(controls):
    return max(value for row in controls for value in row) <= 0


def cover_pair(first, second, maximum_depth=12):
    pending = [(first, second, 0)]
    leaves = 0
    deepest = 0
    while pending:
        left, right, depth = pending.pop()
        if nonpositive(left) or nonpositive(right):
            leaves += 1
            deepest = max(deepest, depth)
            continue
        assert depth < maximum_depth
        pending.extend(
            (left_child, right_child, depth + 1)
            for left_child, right_child in zip(children(left), children(right))
        )
    return leaves, deepest


imbalance_pairs = ((0, 1), (0, 2), (1, 2))
types = {}
for first, second in imbalance_pairs:
    for cycle in all_cycles:
        sum_barrier, root_barrier = barrier_pair(cycle, first, second)
        key = (
            tuple(sorted(sum_barrier.items())),
            tuple(sorted(root_barrier.items())),
        )
        types[key] = types.get(key, 0) + 1

assert sum(types.values()) == 3 * 409

local_cells = 0
local_maximum = 0
global_cells = 0
global_maximum = 0
for key in types:
    sum_barrier, root_barrier = map(dict, key)
    for patch, sign in (("central", 1), ("outer", 1), ("outer", -1)):
        cells, depth = cover_pair(
            bernstein_controls(blow_up(sum_barrier, patch, sign)),
            bernstein_controls(blow_up(root_barrier, patch, sign)),
        )
        local_cells += cells
        local_maximum = max(local_maximum, cells)

    # Compactify the unbounded v-coordinate.  Ordinary dyadic subdivision
    # need not enter the local box: the three blow-ups already cover it.
    sum_controls = bernstein_controls(compactify(sum_barrier), 5, 5)
    root_controls = bernstein_controls(compactify(root_barrier), 5, 5)
    pending = [(
        sum_controls, root_controls,
        Fraction(0), Fraction(1), Fraction(0), Fraction(1), 0,
    )]
    leaves = 0
    while pending:
        left, right, x0, x1, y0, y1, depth = pending.pop()
        if nonpositive(left) or nonpositive(right):
            leaves += 1
            global_maximum = max(global_maximum, depth)
            continue
        if (x0 >= Fraction(61, 128) and x1 <= Fraction(67, 128)
                and y1 <= Fraction(1, 65)):
            leaves += 1
            global_maximum = max(global_maximum, depth)
            continue
        assert depth < 12
        x_midpoint, y_midpoint = (x0 + x1) / 2, (y0 + y1) / 2
        boxes = (
            (x0, x_midpoint, y0, y_midpoint),
            (x0, x_midpoint, y_midpoint, y1),
            (x_midpoint, x1, y0, y_midpoint),
            (x_midpoint, x1, y_midpoint, y1),
        )
        pending.extend(
            (left_child, right_child, *box, depth + 1)
            for left_child, right_child, box
            in zip(children(left), children(right), boxes)
        )
    global_cells += leaves

assert len(types) == 223
assert local_maximum <= 100
assert global_maximum <= 12

print(f"four-cycle imbalance incidences: {3 * len(all_cycles)}")
print(f"exact bivariate barrier types: {len(types)}")
print(f"local blow-up cells: {local_cells}")
print(f"maximum local cells per patch: {local_maximum}")
print(f"global compactified cells: {global_cells}")
print(f"maximum global subdivision depth: {global_maximum}")
print("slice theorem: lower squared frequency <= 3/5")
print("PASS")
