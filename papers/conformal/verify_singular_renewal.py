"""Exact checks for unequal-scale singular-staircase renewal laws."""

from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction as F
from math import gcd


APPROXIMATION_STAGE = 18


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


def atom_wasserstein(left_atoms, right_atoms):
    left = [[value, weight] for value, weight in sorted(left_atoms)]
    right = [[value, weight] for value, weight in sorted(right_atoms)]
    assert sum((weight for _, weight in left), F(0)) == 1
    assert sum((weight for _, weight in right), F(0)) == 1
    total = F(0)
    i = j = 0
    while i < len(left) and j < len(right):
        mass = min(left[i][1], right[j][1])
        total += mass * abs(left[i][0] - right[j][0])
        left[i][1] -= mass
        right[j][1] -= mass
        if left[i][1] == 0:
            i += 1
        if right[j][1] == 0:
            j += 1
    assert i == len(left) and j == len(right)
    return total


def signed_kr_norm(atoms):
    masses = {}
    for value, mass in atoms:
        masses[value] = masses.get(value, F(0)) + mass
    ordered = sorted((value, mass) for value, mass in masses.items() if mass)
    assert sum((mass for _, mass in ordered), F(0)) == 0
    total = F(0)
    cumulative = F(0)
    for (left, mass), (right, _) in zip(ordered, ordered[1:]):
        cumulative += mass
        total += abs(cumulative) * (right - left)
    return total


def extend_recurrence(initial, weights, horizon):
    values = initial[:horizon]
    for power in range(len(values), horizon):
        values.append(sum((weight * values[power - lag]
                           for lag, weight in weights.items()), F(0)))
    return values


def empirical_phase_check(initial, weights, limits, horizon):
    span, reduced, matrix = companion_matrix(weights)
    size = len(matrix)
    invariant = invariant_row(reduced)
    values = extend_recurrence(initial, weights, horizon)
    counts = [len(range(residue, horizon, span)) for residue in range(span)]
    maximum_n = max(counts) - size
    coefficients = []
    power_matrix = [[F(int(i == j)) for j in range(size)] for i in range(size)]
    for _ in range(max(0, maximum_n + 1)):
        coefficients.append(target_row_contraction(power_matrix[0], invariant))
        power_matrix = matrix_product(power_matrix, matrix)

    dynamic_bound = F(0)
    for residue, count in enumerate(counts):
        residue_values = [values[residue + span * q] for q in range(count)]
        boundary = [initial[residue + span * q] for q in range(size)]
        early_count = min(count, size - 1)
        dynamic_bound += sum((abs(residue_values[q] - limits[residue])
                              for q in range(early_count)), F(0))
        if count >= size:
            oscillation = max(boundary) - min(boundary)
            dynamic_bound += oscillation * sum(coefficients[:count - size + 1], F(0))

    remainder = horizon % span
    diameter = max(limits) - min(limits)
    count_bound = diameter * remainder * (span - remainder) / (span * horizon)
    bound = dynamic_bound / horizon + count_bound
    exact = atom_wasserstein(
        [(value, F(1, horizon)) for value in values],
        [(limit, F(1, span)) for limit in limits],
    )
    assert exact <= bound
    return exact, bound, count_bound


def empirical_second_order_check(initial, weights, limits, horizon):
    span, _, matrix = companion_matrix(weights)
    size = len(matrix)
    values = extend_recurrence(initial, weights, horizon)
    counts = [len(range(residue, horizon, span)) for residue in range(span)]
    assert min(counts) >= size

    exact = horizon * atom_wasserstein(
        [(value, F(1, horizon)) for value in values],
        [(limit, F(1, span)) for limit in limits],
    )
    atoms = []
    for residue, count in enumerate(counts):
        residue_values = [values[residue + span * q] for q in range(count)]
        for value in residue_values:
            atoms.extend(((value, F(1)), (limits[residue], F(-1))))
        atoms.append((limits[residue], F(count) - F(horizon, span)))
    assert signed_kr_norm(atoms) == exact

    tail_bound = F(0)
    for residue, count in enumerate(counts):
        boundary = [initial[residue + span * q] for q in range(size)]
        oscillation = max(boundary) - min(boundary)
        first_power = count - size + 1
        tail_bound += oscillation * selected_tail_bound(weights, first_power)
    return exact, tail_bound


def scaled_empirical_error(initial, weights, limits, horizon):
    span, _, _ = companion_matrix(weights)
    values = extend_recurrence(initial, weights, horizon)
    return horizon * atom_wasserstein(
        [(value, F(1, horizon)) for value in values],
        [(limit, F(1, span)) for limit in limits],
    )


def second_order_phase_distribution_check(initial, weights, limits, horizon,
                                          reference_start=120):
    span, _, matrix = companion_matrix(weights)
    size = len(matrix)
    assert horizon >= span * size
    phase_controls = []
    for residue in range(span):
        reference = reference_start + (residue - reference_start) % span
        phase_controls.append(
            empirical_second_order_check(initial, weights, limits, reference)
        )

    scaled = [scaled_empirical_error(initial, weights, limits, time)
              for time in range(1, horizon + 1)]
    centers = [control[0] for control in phase_controls]
    radii = [control[1] for control in phase_controls]
    exact_to_centers = atom_wasserstein(
        [(value, F(1, horizon)) for value in scaled],
        [(center, F(1, span)) for center in centers],
    )
    coupling = sum((abs(value - centers[time % span])
                    for time, value in enumerate(scaled, 1)), F(0)) / horizon
    remainder = horizon % span
    imbalance = F(remainder * (span - remainder), span * horizon)
    diameter = max(centers) - min(centers)
    center_bound = coupling + diameter * imbalance
    assert exact_to_centers <= center_bound

    true_phase_bound = (
        sum((abs(value - centers[time % span]) + radii[time % span]
             for time, value in enumerate(scaled, 1)), F(0)) / horizon
        + (diameter + 2 * max(radii)) * imbalance
    )
    return exact_to_centers, true_phase_bound, phase_controls


def rescaled_phase_second_order_identity(initial, weights, limits, horizon,
                                         reference_start=120,
                                         phase_centers=None):
    span, _, _ = companion_matrix(weights)
    if phase_centers is None:
        phase_centers = []
        for residue in range(span):
            reference = reference_start + (residue - reference_start) % span
            phase_centers.append(
                empirical_second_order_check(initial, weights, limits, reference)[0]
            )

    scaled = [scaled_empirical_error(initial, weights, limits, time)
              for time in range(1, horizon + 1)]
    exact = horizon * atom_wasserstein(
        [(value, F(1, horizon)) for value in scaled],
        [(center, F(1, span)) for center in phase_centers],
    )
    counts = [sum(time % span == residue for time in range(1, horizon + 1))
              for residue in range(span)]
    atoms = []
    for time, value in enumerate(scaled, 1):
        atoms.extend(((value, F(1)), (phase_centers[time % span], F(-1))))
    for residue, count in enumerate(counts):
        atoms.append((phase_centers[residue], F(count) - F(horizon, span)))
    assert signed_kr_norm(atoms) == exact
    return exact


def selected_tail_bound(weights, first_power):
    span, reduced, matrix = companion_matrix(weights)
    assert span >= 1 and first_power >= 0
    size = len(matrix)
    invariant = invariant_row(reduced)
    _, _, block, _, _, contraction, _ = companion_contraction(weights)
    power_matrix = [[F(int(i == j)) for j in range(size)] for i in range(size)]
    for _ in range(first_power):
        power_matrix = matrix_product(power_matrix, matrix)
    block_sum = F(0)
    for _ in range(block):
        block_sum += target_row_contraction(power_matrix[0], invariant)
        power_matrix = matrix_product(power_matrix, matrix)
    return block_sum / (1 - contraction)


def second_order_stability(initial, weights, coefficient_error, remainder):
    span, _, matrix = companion_matrix(weights)
    size = len(matrix)
    total = F(2 * remainder * (span - remainder), span) * coefficient_error
    choices = []
    for residue in range(span):
        boundary = [initial[residue + span * q] for q in range(size)]
        oscillation = max(boundary) - min(boundary)
        amplitude = 2 * oscillation + 2 * coefficient_error
        candidates = [
            2 * coefficient_error * cutoff
            + amplitude * selected_tail_bound(weights, cutoff)
            for cutoff in range(80)
        ]
        cutoff = min(range(len(candidates)), key=candidates.__getitem__)
        total += 2 * coefficient_error * (size - 1) + candidates[cutoff]
        choices.append(cutoff)
    return total, choices


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


def interval_text(center, radius, places=12):
    return (f"[{decimal_bound(center - radius, ROUND_FLOOR, places)}, "
            f"{decimal_bound(center + radius, ROUND_CEILING, places)}]")


def check_family(base, branches, approximation_stage=APPROXIMATION_STAGE):
    lengths, weights = recurrence_data(base, branches)
    maximum = max(lengths)
    levels = stage_family(branches, approximation_stage)
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

    initial = [scaled_error(levels[approximation_stage], base, 1, power)
               for power in range(maximum)]
    span, limits = residue_limits(initial, weights)
    sequence = extend_recurrence(initial, weights, 61)
    residual = max(abs(sequence[power] - limits[power % span])
                   for power in range(60 - span + 1, 61))
    contraction = recurrence_contraction_checks(initial, weights)
    return checks, largest_cells, span, limits, residual, contraction, initial, weights


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
    a_radius = coefficient_radius(2, aperiodic, APPROXIMATION_STAGE)
    p_radius = coefficient_radius(2, periodic, APPROXIMATION_STAGE)
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
    empirical_checks = 0
    second_order_identities = 0
    for horizon in range(1, 62):
        empirical_phase_check(a[6], a[7], a[3], horizon)
        empirical_phase_check(p[6], p[7], p[3], horizon)
        empirical_checks += 2
        if horizon >= 2:
            empirical_second_order_check(a[6], a[7], a[3], horizon)
            second_order_identities += 1
        if horizon >= 4:
            empirical_second_order_check(p[6], p[7], p[3], horizon)
            second_order_identities += 1
    a_empirical = empirical_phase_check(a[6], a[7], a[3], 61)
    p_empirical = empirical_phase_check(p[6], p[7], p[3], 61)
    a_true_empirical_bound = a_empirical[1] + 2 * a_radius
    p_true_empirical_bound = p_empirical[1] + 2 * p_radius
    a_second = empirical_second_order_check(a[6], a[7], a[3], 61)
    p_even_second = empirical_second_order_check(p[6], p[7], p[3], 120)
    p_odd_second = empirical_second_order_check(p[6], p[7], p[3], 121)
    a_stability = second_order_stability(a[6], a[7], a_radius, 0)
    p_even_stability = second_order_stability(p[6], p[7], p_radius, 0)
    p_odd_stability = second_order_stability(p[6], p[7], p_radius, 1)
    a_second_radius = a_second[1] + a_stability[0]
    p_even_second_radius = p_even_second[1] + p_even_stability[0]
    p_odd_second_radius = p_odd_second[1] + p_odd_stability[0]
    phase_gap_lower = (p_odd_second[0] - p_odd_second_radius
                       - p_even_second[0] - p_even_second_radius)
    assert phase_gap_lower > 0
    phase_distribution_checks = 0
    rescaled_second_order_identities = 0
    for horizon in range(2, 62):
        second_order_phase_distribution_check(a[6], a[7], a[3], horizon)
        phase_distribution_checks += 1
    for horizon in range(4, 62):
        second_order_phase_distribution_check(p[6], p[7], p[3], horizon)
        phase_distribution_checks += 1
    a_phase_centers = [a_second[0]]
    p_phase_centers = [p_even_second[0], p_odd_second[0]]
    for horizon in range(1, 62):
        rescaled_phase_second_order_identity(
            a[6], a[7], a[3], horizon, phase_centers=a_phase_centers
        )
        rescaled_phase_second_order_identity(
            p[6], p[7], p[3], horizon, phase_centers=p_phase_centers
        )
        rescaled_second_order_identities += 2
    a_phase_distribution = second_order_phase_distribution_check(
        a[6], a[7], a[3], 61
    )
    p_phase_distribution = second_order_phase_distribution_check(
        p[6], p[7], p[3], 61
    )
    a_true_phase_distribution_bound = (
        a_phase_distribution[1] + 62 * a_radius + a_stability[0]
    )
    p_true_phase_distribution_bound = (
        p_phase_distribution[1] + 62 * p_radius
        + (p_even_stability[0] + p_odd_stability[0]) / 2
    )

    print(f"exact finite-stage renewal checks: {a[0] + p[0]}")
    print(f"largest aperiodic/periodic cell counts: {a[1]}/{p[1]}")
    print(f"horizontal lattice spans: {a[2]}/{p[2]}")
    print("certified true cluster cardinalities: 1/2")
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
    print(f"exact finite empirical-phase inequalities: {empirical_checks}")
    print(f"stage-{APPROXIMATION_STAGE} empirical W1 at horizon 61 "
          "(aperiodic/periodic): "
          f"{float(a_empirical[0]):.12f}/{float(p_empirical[0]):.12f}")
    print("finite empirical phase bounds at horizon 61: "
          f"{float(a_empirical[1]):.12f}/{float(p_empirical[1]):.12f}")
    print("true-staircase empirical phase bounds at horizon 61: "
          f"{float(a_true_empirical_bound):.12f}/{float(p_true_empirical_bound):.12f}")
    print(f"exact scaled empirical KR identities: {second_order_identities}")
    print(f"stage-{APPROXIMATION_STAGE} staircase radii (aperiodic/periodic): "
          f"{float(a_radius):.14f}/{float(p_radius):.14f}")
    print("second-order stability cutoffs (aperiodic/even/odd): "
          f"{a_stability[1]}/{p_even_stability[1]}/{p_odd_stability[1]}")
    print("true-staircase second-order enclosures (aperiodic/even/odd): "
          f"{interval_text(a_second[0], a_second_radius)} / "
          f"{interval_text(p_even_second[0], p_even_second_radius)} / "
          f"{interval_text(p_odd_second[0], p_odd_second_radius)}")
    print("certified periodic second-order phase-gap lower bound: "
          f"{decimal_bound(phase_gap_lower, ROUND_FLOOR)}")
    print(f"exact rescaled phase-distribution inequalities: {phase_distribution_checks}")
    print("stage-18 rescaled phase-distribution W1 at horizon 61 "
          "(aperiodic/periodic): "
          f"{float(a_phase_distribution[0]):.12f}/"
          f"{float(p_phase_distribution[0]):.12f}")
    print("true-staircase rescaled phase-distribution bounds at horizon 61: "
          f"{float(a_true_phase_distribution_bound):.12f}/"
          f"{float(p_true_phase_distribution_bound):.12f}")
    print("exact second-layer rescaled signed-KR identities: "
          f"{rescaled_second_order_identities}")
    print("PASS")


if __name__ == "__main__":
    main()
