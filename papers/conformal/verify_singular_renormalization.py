"""Exact finite-stage checks for singular-staircase rate renormalization."""

from fractions import Fraction as F


def refine(segments, base, digits):
    count = len(digits)
    result = []
    for rank, digit in enumerate(digits):
        for left, right, slope, intercept in segments:
            result.append((F(digit, base) + left / base,
                           F(digit, base) + right / base,
                           slope * base / count,
                           F(rank, count) + (intercept - slope * digit) / count))
        if rank + 1 < count and digit + 1 < digits[rank + 1]:
            result.append((F(digit + 1, base), F(digits[rank + 1], base),
                           F(0), F(rank + 1, count)))
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


def stage_family(base, digits, maximum_level):
    levels = [[(F(0), F(1), F(1), F(0))]]
    for _ in range(maximum_level):
        levels.append(refine(levels[-1], base, digits))
    return levels


def check_family(base, digits):
    levels = stage_family(base, digits, 10)
    checks = 0
    largest_cells = 0
    for multiplier in range(1, 5):
        for power in range(1, 5):
            level = power + 5
            cells = multiplier * base ** power
            left = cells * projection_error(levels[level], cells)
            right = multiplier * projection_error(levels[level - power], multiplier)
            assert left == right
            checks += 1
            largest_cells = max(largest_cells, cells)
    return checks, largest_cells, levels


def main():
    checks3, cells3, ternary = check_family(3, (0, 2))
    checks5, cells5, quinary = check_family(5, (0, 4))

    # The finite stages converge uniformly at rate 2^-L. These exact checks
    # independently enclose the two known ternary base constants.
    level = 10
    c1 = projection_error(ternary[level], 1)
    c2 = 2 * projection_error(ternary[level], 2)
    assert abs(c1 - F(1, 6)) <= F(2, 2 ** level)
    assert abs(c2 - F(4, 15)) <= F(4, 2 ** level)

    q1 = projection_error(quinary[level], 1)
    q2 = 2 * projection_error(quinary[level], 2)
    assert abs(q1 - F(1, 10)) <= F(2, 2 ** level)

    print(f"exact finite-stage renormalization checks: {checks3 + checks5}")
    print(f"largest ternary/quinary cell counts: {cells3}/{cells5}")
    print(f"ternary base constants at stage {level}: {float(c1):.12f}/{float(c2):.12f}")
    print(f"quinary base constants at stage {level}: {float(q1):.12f}/{float(q2):.12f}")
    print("uniform scaled-error enclosure radii for m=1,2: "
          f"{2 / 2 ** level:.12f}/{4 / 2 ** level:.12f}")
    print("PASS")


if __name__ == "__main__":
    main()
