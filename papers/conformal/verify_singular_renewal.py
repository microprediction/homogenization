"""Exact checks for unequal-scale singular-staircase renewal laws."""

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction as F
from math import gcd


def refine(segments, branches):
    result = []
    cumulative = F(0)
    for index, (left, width, weight) in enumerate(branches):
        for lo, hi, slope, intercept in segments:
            new_slope = weight * slope / width
            result.append((left + width * lo, left + width * hi, new_slope,
                           cumulative + weight * intercept - new_slope * left))
        cumulative += weight
        if index + 1 < len(branches):
            gap_left = left + width
            gap_right = branches[index + 1][0]
            if gap_left < gap_right:
                result.append((gap_left, gap_right, F(0), cumulative))
    return result


def affine_integral(slope, intercept, left, right):
    return slope * (right * right - left * left) / 2 + intercept * (right - left)


def absolute_affine_integral(slope, intercept, level, left, right):
    value_left = slope * left + intercept - level
    value_right = slope * right + intercept - level
    width = right - left
    if slope == 0 or value_left * value_right >= 0:
        return (abs(value_left) + abs(value_right)) * width / 2
    return (value_left * value_left + value_right * value_right) / (2 * slope)


def projection_error(segments, cells):
    total = F(0)
    segment_index = 0
    for cell in range(cells):
        left, right = F(cell, cells), F(cell + 1, cells)
        while segments[segment_index][1] <= left:
            segment_index += 1
        pieces = []
        index = segment_index
        while index < len(segments) and segments[index][0] < right:
            lo, hi, slope, intercept = segments[index]
            lo, hi = max(left, lo), min(right, hi)
            if lo < hi:
                pieces.append((lo, hi, slope, intercept))
            index += 1
        mean = cells * sum((affine_integral(slope, intercept, lo, hi)
                            for lo, hi, slope, intercept in pieces), F(0))
        total += sum((absolute_affine_integral(slope, intercept, mean, lo, hi)
                      for lo, hi, slope, intercept in pieces), F(0))
    return total


def stage_family(branches, maximum_level):
    assert branches[0][0] == 0
    assert branches[-1][0] + branches[-1][1] == 1
    assert sum((weight for _, _, weight in branches), F(0)) == 1
    levels = [[(F(0), F(1), F(1), F(0))]]
    for _ in range(maximum_level):
        levels.append(refine(levels[-1], branches))
    return levels


def scaled_error(segments, base, multiplier, power):
    cells = multiplier * base ** power
    return cells * projection_error(segments, cells)


def recurrence_data(base, branches):
    lengths = []
    weights = {}
    for _, width, weight in branches:
        power = 0
        test = width
        while test < 1:
            test *= base
            power += 1
        assert test == 1
        lengths.append(power)
        weights[power] = weights.get(power, F(0)) + weight
    return lengths, weights


def residue_limits(initial, weights):
    span = 0
    for power in weights:
        span = gcd(span, power)
    maximum = max(weights)
    reduced_maximum = maximum // span
    reduced = {power // span: weight for power, weight in weights.items()}
    mean = sum((power * weight for power, weight in reduced.items()), F(0))
    limits = []
    for residue in range(span):
        values = [initial[residue + span * q] for q in range(reduced_maximum)]
        forcing = F(0)
        for q, value in enumerate(values):
            forcing += value - sum((weight * values[q - power]
                                    for power, weight in reduced.items() if power <= q), F(0))
        limits.append(forcing / mean)
    return span, limits


def matrix_product(left, right):
    size = len(left)
    return [[sum((left[i][k] * right[k][j] for k in range(size)), F(0))
             for j in range(size)] for i in range(size)]


def dobrushin_contraction(matrix):
    size = len(matrix)
    overlap = min(
        sum((min(matrix[i][j], matrix[k][j]) for j in range(size)), F(0))
        for i in range(size) for k in range(size)
    )
    return overlap, 1 - overlap


def row_matrix_product(row, matrix):
    return [sum((row[k] * matrix[k][j] for k in range(len(row))), F(0))
            for j in range(len(row))]


def invariant_row(reduced):
    size = max(reduced)
    mean = sum((lag * weight for lag, weight in reduced.items()), F(0))
    row = [sum((weight for lag, weight in reduced.items() if lag > index), F(0)) / mean
           for index in range(size)]
    return row


def target_row_contraction(row, invariant):
    difference = [left - right for left, right in zip(row, invariant)]
    coefficient = sum((abs(value) for value in difference), F(0)) / 2
    witness = [F(int(value > 0)) for value in difference]
    attained = abs(sum((value * test for value, test in zip(difference, witness)), F(0)))
    assert attained == coefficient
    return coefficient


def companion_matrix(weights):
    span = 0
    for power in weights:
        span = gcd(span, power)
    reduced = {power // span: weight for power, weight in weights.items()}
    size = max(reduced)
    matrix = [[F(0) for _ in range(size)] for _ in range(size)]
    matrix[0] = [reduced.get(index + 1, F(0)) for index in range(size)]
    for index in range(1, size):
        matrix[index][index - 1] = 1
    return span, reduced, matrix


def companion_contraction(weights):
    span, _, matrix = companion_matrix(weights)
    size = len(matrix)
    power_matrix = [[F(int(i == j)) for j in range(size)] for i in range(size)]
    for block in range(1, size * size + 2):
        power_matrix = matrix_product(power_matrix, matrix)
        floor = min(min(row) for row in power_matrix)
        if floor > 0:
            overlap, contraction = dobrushin_contraction(power_matrix)
            coarse_contraction = 1 - size * floor
            assert F(0) <= contraction <= coarse_contraction < 1
            return (span, size, block, floor, overlap, contraction,
                    coarse_contraction)
    raise AssertionError("primitive companion power not found")


def recurrence_contraction_checks(initial, weights, limit_power=60):
    (span, size, block, floor, overlap, contraction,
     coarse_contraction) = companion_contraction(weights)
    matrix_span, reduced, matrix = companion_matrix(weights)
    assert matrix_span == span
    invariant = invariant_row(reduced)
    assert sum(invariant, F(0)) == 1
    assert row_matrix_product(invariant, matrix) == invariant
    limits = residue_limits(initial, weights)[1]
    bounds = []
    checks = 0
    for residue in range(span):
        values = [initial[residue + span * q] for q in range(size)]
        oscillation = max(values) - min(values)
        for q in range(size, limit_power // span + 1):
            values.append(sum((weight * values[q - lag]
                               for lag, weight in reduced.items()), F(0)))
        power_matrix = [[F(int(i == j)) for j in range(size)] for i in range(size)]
        exact_contraction = F(1)
        selected_contraction = F(1)
        for q in range(size - 1, len(values)):
            exponent = (q - size + 1) // block
            _, exact_contraction = dobrushin_contraction(power_matrix)
            selected_contraction = target_row_contraction(power_matrix[0], invariant)
            block_bound = contraction ** exponent
            assert selected_contraction <= exact_contraction <= block_bound
            bound = selected_contraction * oscillation
            assert abs(values[q] - limits[residue]) <= bound
            checks += 1
            power_matrix = matrix_product(power_matrix, matrix)
        bounds.append((oscillation, (limit_power // span - size + 1) // block,
                       exact_contraction, selected_contraction))
    return (span, size, block, floor, overlap, contraction,
            coarse_contraction, checks, bounds)


def coefficient_radius(base, branches, stage, multiplier=1):
    lengths, _ = recurrence_data(base, branches)
    largest_weight = max(weight for _, _, weight in branches)
    return 2 * multiplier * base ** (max(lengths) - 1) * largest_weight ** stage


def decimal_bound(value, rounding, places=12):
    with localcontext() as context:
        context.prec = 50
        number = Decimal(value.numerator) / Decimal(value.denominator)
        unit = Decimal(1).scaleb(-places)
        return format(number.quantize(unit, rounding=rounding), "f")


def interval_text(center, radius):
    return (f"[{decimal_bound(center - radius, ROUND_FLOOR)}, "
            f"{decimal_bound(center + radius, ROUND_CEILING)}]")


def check_family(base, branches):
    lengths, weights = recurrence_data(base, branches)
    maximum = max(lengths)
    levels = stage_family(branches, 10)
    checks = 0
    largest_cells = 0
    for stage in range(6, 9):
        for multiplier in range(1, 4):
            for power in range(maximum, maximum + 5):
                left = scaled_error(levels[stage], base, multiplier, power)
                right = sum((weight * scaled_error(levels[stage - 1], base, multiplier,
                                                    power - length)
                             for length, weight in weights.items()), F(0))
                assert left == right
                checks += 1
                largest_cells = max(largest_cells, multiplier * base ** power)

    initial = [scaled_error(levels[10], base, 1, power) for power in range(maximum)]
    span, limits = residue_limits(initial, weights)
    sequence = initial[:]
    for power in range(maximum, 61):
        sequence.append(sum((weight * sequence[power - length]
                             for length, weight in weights.items()), F(0)))
    residual = max(abs(sequence[power] - limits[power % span])
                   for power in range(60 - span + 1, 61))
    contraction = recurrence_contraction_checks(initial, weights)
    return checks, largest_cells, span, limits, residual, contraction


def true_recurrence_bound(contraction_data, coefficient_error, residue):
    oscillation, _, _, selected_contraction = contraction_data[8][residue]
    return selected_contraction * (oscillation + 2 * coefficient_error)


def main():
    aperiodic = ((F(0), F(1, 2), F(1, 2)),
                 (F(3, 4), F(1, 4), F(1, 2)))
    periodic = ((F(0), F(1, 4), F(1, 2)),
                (F(15, 16), F(1, 16), F(1, 2)))
    a = check_family(2, aperiodic)
    p = check_family(2, periodic)
    a_radius = coefficient_radius(2, aperiodic, 10)
    p_radius = coefficient_radius(2, periodic, 10)
    assert a[2] == 1 and p[2] == 2
    assert a[4] < F(1, 10 ** 10)
    assert p[4] < F(1, 10 ** 9)
    assert p[3][0] + p_radius < p[3][1] - p_radius
    expected = (2, 2, F(1, 4), F(3, 4), F(1, 4), F(1, 2))
    assert a[5][1:7] == expected
    assert p[5][1:7] == expected
    fixture_invariant = invariant_row({1: F(1, 2), 2: F(1, 2)})
    assert fixture_invariant == [F(2, 3), F(1, 3)]
    assert a[5][8][0][2] == F(1, 2) ** 59
    assert p[5][8][0][2] == F(1, 2) ** 29
    assert p[5][8][1][2] == F(1, 2) ** 29
    assert a[5][8][0][3] == F(1, 3) * F(1, 2) ** 59
    assert p[5][8][0][3] == F(1, 3) * F(1, 2) ** 29
    assert p[5][8][1][3] == F(1, 3) * F(1, 2) ** 29
    finite_checks = a[5][7] + p[5][7]
    a_finite_bound = true_recurrence_bound(a[5], a_radius, 0)
    p_even_bound = true_recurrence_bound(p[5], p_radius, 0)
    p_odd_bound = true_recurrence_bound(p[5], p_radius, 1)

    print(f"exact finite-stage renewal checks: {a[0] + p[0]}")
    print(f"largest aperiodic/periodic cell counts: {a[1]}/{p[1]}")
    print(f"horizontal lattice spans: {a[2]}/{p[2]}")
    print(f"aperiodic true-limit enclosure: {interval_text(a[3][0], a_radius)}")
    print("periodic even/odd true-limit enclosures: "
          f"{interval_text(p[3][0], p_radius)} / {interval_text(p[3][1], p_radius)}")
    print(f"largest power-60 recurrence residual: {float(max(a[4], p[4])):.3e}")
    print(f"exact finite-power contraction checks: {finite_checks}")
    print("companion block/minimum/overlap/exact/coarse contraction: "
          f"{a[5][2]}/{float(a[5][3]):.6f}/{float(a[5][4]):.6f}/"
          f"{float(a[5][5]):.6f}/{float(a[5][6]):.6f}")
    print("fixture invariant row: 2/3,1/3")
    print("exact full-power contractions at powers 60/60/61: "
          f"2^-59/2^-29/2^-29")
    print("selected-row contractions at powers 60/60/61: "
          f"2^-59/3/2^-29/3/2^-29/3")
    print("true recurrence-to-limit bounds at powers 60/60/61: "
          f"{float(a_finite_bound):.3e}/{float(p_even_bound):.3e}/{float(p_odd_bound):.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
