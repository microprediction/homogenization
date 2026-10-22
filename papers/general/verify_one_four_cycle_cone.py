"""Exact full-cone certificates for every fourth cycle at the sharp triple."""

from fractions import Fraction
from itertools import permutations
from math import comb

from verify_four_cycle_tangent import (
    BASE_CYCLES,
    GROUND,
    PERMUTATIONS,
    SIGNS,
    all_cycles,
    base_forms,
    integer_forms,
)


DEGREE = 5
LOCAL_RADIUS = Fraction(1, 1024)
QUADRATIC_MARGIN = 64


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


def barrier_pair(cycle):
    forms = base_forms + (integer_forms(cycle),)
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


def face_values(controls, fixed):
    """Return the Bernstein controls on a coordinate face of a box."""
    return [
        value for index, value in controls.items()
        if all(index[axis] == endpoint for axis, endpoint in fixed.items())
    ]


def on_global_boundary(box, fixed):
    return any(
        box[axis][0 if endpoint == 0 else 1] in (0, 1)
        for axis, endpoint in fixed.items()
    )


def strict_face_check(first, second, box, fixed=None):
    """Prove strictness on a box, deferring only outer-cube faces.

    A nonpositive Bernstein expansion with a negative control is negative on
    the relative interior.  Its coordinate boundary is checked recursively.
    """
    fixed = {} if fixed is None else fixed
    first_values = face_values(first, fixed)
    second_values = face_values(second, fixed)
    if max(first_values) < 0 or max(second_values) < 0:
        return 0, 0
    if not (
        (max(first_values) <= 0 and min(first_values) < 0)
        or (max(second_values) <= 0 and min(second_values) < 0)
    ):
        return (1, 0) if on_global_boundary(box, fixed) else (0, 1)
    deferred = unresolved = 0
    for axis in range(3):
        if axis in fixed:
            continue
        for endpoint in (0, DEGREE):
            child_fixed = dict(fixed)
            child_fixed[axis] = endpoint
            child_deferred, child_unresolved = strict_face_check(
                first, second, box, child_fixed
            )
            deferred += child_deferred
            unresolved += child_unresolved
    return deferred, unresolved


def cover(sum_controls, root_controls, maximum_depth=48):
    stack = [(sum_controls, root_controls, ((Fraction(0), Fraction(1)),) * 3, 0)]
    terminals = strict_cells = weak_cells = local_cells = deepest = 0
    deferred_faces = unresolved_faces = 0
    while stack:
        first, second, box, depth = stack.pop()
        if nonpositive(first) or nonpositive(second):
            terminals += 1
            if max(first.values()) < 0 or max(second.values()) < 0:
                strict_cells += 1
            else:
                weak_cells += 1
                deferred, unresolved = strict_face_check(first, second, box)
                deferred_faces += deferred
                unresolved_faces += unresolved
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
    return (
        terminals, strict_cells, weak_cells, local_cells, deepest,
        deferred_faces, unresolved_faces,
    )


def canonical(cycle):
    cycle = tuple(cycle)
    return min(cycle[index:] + cycle[:index] for index in range(len(cycle)))


def transform(cycle, relabeling, reverse):
    cycle = tuple(reversed(cycle)) if reverse else cycle
    return canonical(tuple(relabeling[state] for state in cycle))


def principal_minors(local_root):
    quadratic = {key: value for key, value in local_root.items() if sum(key) == 2}
    matrix = [[Fraction(0) for _ in range(3)] for _ in range(3)]
    for row in range(3):
        square = tuple(2 if index == row else 0 for index in range(3))
        matrix[row][row] = -quadratic.get(square, 0) - QUADRATIC_MARGIN
        for column in range(row):
            cross = tuple(
                int(index in (row, column)) for index in range(3)
            )
            matrix[row][column] = matrix[column][row] = (
                -quadratic.get(cross, Fraction(0)) / 2
            )
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2]
                        - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2]
                          - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1]
                          - matrix[1][1] * matrix[2][0])
    )
    return (
        matrix[0][0],
        matrix[0][0] * matrix[1][1] - matrix[0][1] ** 2,
        determinant,
    )


def parabolic_bounds(local_sum, local_root):
    """Certify a local cone by splitting at v=p^2+q^2."""
    radius = LOCAL_RADIUS
    linear_margin = -local_sum.get((0, 0, 1), 0)
    assert linear_margin > 0
    assert {
        exponent: coefficient for exponent, coefficient in local_sum.items()
        if sum(exponent) == 2 and exponent[2] == 0
    } == {(2, 0, 0): 144, (1, 1, 0): 144, (0, 2, 0): 144}
    root_quadratic = {
        exponent: coefficient for exponent, coefficient in local_root.items()
        if sum(exponent) == 2
    }
    assert root_quadratic.get((2, 0, 0)) == -576
    assert root_quadratic.get((1, 1, 0)) == -576
    assert root_quadratic.get((0, 2, 0)) == -576
    assert set(root_quadratic) <= {
        (2, 0, 0), (1, 1, 0), (0, 2, 0),
        (1, 0, 1), (0, 1, 1), (0, 0, 2),
    }
    central_remainder = {
        exponent: coefficient for exponent, coefficient in local_sum.items()
        if exponent != (0, 0, 1)
        and not (sum(exponent) == 2 and exponent[2] == 0)
    }
    outer_higher = {
        exponent: coefficient for exponent, coefficient in local_root.items()
        if sum(exponent) >= 3
    }
    assert all(
        (exponent[2] and sum(exponent) >= 2)
        or (not exponent[2] and sum(exponent) >= 3)
        for exponent in central_remainder
    )
    assert all(sum(exponent) >= 3 for exponent in outer_higher)
    central_coefficient_sum = sum(map(abs, central_remainder.values()))
    outer_coefficient_sum = sum(map(abs, outer_higher.values()))

    # Put R=p^2+q^2 and S=p^2+pq+q^2, so R/2 <= S <= 3R/2.
    # If v>=R, the sum barrier is at most
    #   (-linear_margin+216+central_coefficient_sum*radius)*v.
    central_margin = linear_margin - 216 - central_coefficient_sum * radius
    # If v<=R, its quadratic root barrier is at most
    #   (-288+(|cp|+|cq|)*radius+2|cv|*radius^2)*R;
    # every term of degree at least three costs at most radius*R.
    p_v = root_quadratic.get((1, 0, 1), 0)
    q_v = root_quadratic.get((0, 1, 1), 0)
    v_squared = root_quadratic.get((0, 0, 2), 0)
    outer_margin = (
        288 - (abs(p_v) + abs(q_v)) * radius
        - 2 * abs(v_squared) * radius ** 2
        - outer_coefficient_sum * radius
    )
    assert central_margin > 0 and outer_margin > 0
    return (
        linear_margin, central_coefficient_sum, outer_coefficient_sum,
        central_margin, outer_margin,
    )


base_set = frozenset(map(canonical, BASE_CYCLES))
symmetries = []
for relabeling in permutations(range(6)):
    for reverse in (False, True):
        image = frozenset(
            transform(cycle, relabeling, reverse) for cycle in BASE_CYCLES
        )
        if image == base_set:
            symmetries.append((relabeling, reverse))
assert len(symmetries) == 12

genuinely_new = {
    canonical(cycle) for cycle in all_cycles if len(cycle) >= 3
} - base_set
assert len(genuinely_new) == 391

remaining = set(genuinely_new)
representatives = []
orbits = []
while remaining:
    representative = min(remaining, key=lambda cycle: (len(cycle), cycle))
    orbit = frozenset(
        transform(representative, *symmetry) for symmetry in symmetries
    ) & genuinely_new
    representatives.append(representative)
    orbits.append(orbit)
    remaining -= orbit
REPRESENTATIVES = tuple(representatives)
covered_cycles = frozenset().union(*orbits)
assert len(REPRESENTATIVES) == 46
assert len(covered_cycles) == len(genuinely_new) == 391

records = []
quadratic_records = []
parabolic_records = []
for cycle, orbit in zip(REPRESENTATIVES, orbits):
    sum_barrier, root_barrier = barrier_pair(cycle)
    local_root = localize(root_barrier)
    assert all(sum(key) >= 2 for key in local_root)
    local_sum = localize(sum_barrier)
    if local_sum.get((0, 0, 1), 0) == 0:
        assert cycle == (0, 1, 2, 3, 4, 5)
        minors = principal_minors(local_root)
        assert all(value > 0 for value in minors)
        higher_sum = sum(
            abs(value) for key, value in local_root.items() if sum(key) >= 3
        )
        # Every degree-m monomial is at most
        # radius^(m-2) (p^2+q^2+v^2) in absolute value.
        assert higher_sum * LOCAL_RADIUS < QUADRATIC_MARGIN
        local_data = (higher_sum, minors)
        quadratic_records.append((higher_sum, minors))
    else:
        local_data = parabolic_bounds(local_sum, local_root)
        parabolic_records.append((cycle, len(orbit), local_data))
    cover_data = cover(
        bernstein_controls(compactify(sum_barrier)),
        bernstein_controls(compactify(root_barrier)),
    )
    records.append((
        cycle, len(orbit), local_data,
        *cover_data,
    ))

assert len(quadratic_records) == 1 and len(parabolic_records) == 45
assert sum(record[1] for record in parabolic_records) == 389
smallest_minors = quadratic_records[0][1]
assert smallest_minors == (512, 179200, 160563200)
smallest_central_margin = min(record[2][3] for record in parabolic_records)
smallest_outer_margin = min(record[2][4] for record in parabolic_records)
assert smallest_central_margin == Fraction(16521, 64)
assert smallest_outer_margin == Fraction(1973697, 8192)
global_cells = sum(record[3] for record in records)
strict_global_cells = sum(record[4] for record in records)
weak_global_cells = sum(record[5] for record in records)
local_cells = sum(record[6] for record in records)
deferred_faces = sum(record[8] for record in records)
unresolved_faces = sum(record[9] for record in records)
assert (global_cells, strict_global_cells, weak_global_cells) == (
    27586, 25564, 2022,
)
assert local_cells == 354
assert deferred_faces == 25946
assert unresolved_faces == 0
print(f"sharp-triple symmetry operations: {len(symmetries)}")
print(f"representative four-weight cones: {len(records)}")
print(f"covered fourth cycles: {len(covered_cycles)}")
print(f"orbit-size histogram: "
      f"{tuple((size, tuple(record[1] for record in records).count(size)) for size in (2, 3, 4, 6, 12))}")
print(f"uncovered genuinely new cycles: {len(genuinely_new - covered_cycles)}")
print(f"local radius: {LOCAL_RADIUS}")
print(f"parabolic representatives: {len(parabolic_records)}")
print(f"parabolic covered cycles: {sum(record[1] for record in parabolic_records)}")
print(f"smallest parabolic linear coefficient: "
      f"{min(record[2][0] for record in parabolic_records)}")
print(f"largest parabolic central coefficient sum: "
      f"{max(record[2][1] for record in parabolic_records)}")
print(f"largest parabolic outer coefficient sum: "
      f"{max(record[2][2] for record in parabolic_records)}")
print(f"smallest parabolic central margin: {smallest_central_margin}")
print(f"smallest parabolic outer margin: {smallest_outer_margin}")
print(f"Hamilton positive-definiteness minors: {smallest_minors}")
print(f"Hamilton higher-order coefficient bound: {quadratic_records[0][0]}")
print(f"global terminal boxes: {global_cells}")
print(f"strict global terminal boxes: {strict_global_cells}")
print(f"weak global terminal boxes: {weak_global_cells}")
print(f"local terminal boxes: {local_cells}")
print(f"maximum subdivision depth: {max(record[7] for record in records)}")
print(f"deferred outer-cube face incidences: {deferred_faces}")
print(f"unresolved interior faces: {unresolved_faces}")
print("positive four-weight cones: lower squared frequency < 3/5")
print("covered full four-weight cones: lower squared frequency <= 3/5")
print("PASS")
