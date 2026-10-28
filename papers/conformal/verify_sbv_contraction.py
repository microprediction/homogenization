"""Exact checks for the SBV equal-mass contraction-rate decomposition."""

from fractions import Fraction as F


SLOPE = F(2)
JUMPS = ((F(1, 3), F(1, 2)),
         (F(2, 5), F(1, 3)),
         (F(5, 7), F(1, 4)))


def integral_affine(slope, intercept, left, right):
    return slope * (right * right - left * left) / 2 + intercept * (right - left)


def abs_affine_integral(slope, intercept, level, left, right):
    """Integral of |slope*u + intercept - level| over [left,right]."""
    yl = slope * left + intercept - level
    yr = slope * right + intercept - level
    width = right - left
    if slope == 0 or yl * yr >= 0:
        return (abs(yl) + abs(yr)) * width / 2
    return (yl * yl + yr * yr) / (2 * slope)


def cell_error(n, cell):
    left, right = F(cell, n), F(cell + 1, n)
    cuts = [left] + [s for s, _ in JUMPS if left < s < right] + [right]
    pieces = []
    total = F(0)
    for lo, hi in zip(cuts, cuts[1:]):
        midpoint = (lo + hi) / 2
        intercept = sum((jump for site, jump in JUMPS if site < midpoint), F(0))
        pieces.append((lo, hi, intercept))
        total += integral_affine(SLOPE, intercept, lo, hi)
    mean = n * total
    return sum((abs_affine_integral(SLOPE, intercept, mean, lo, hi)
                for lo, hi, intercept in pieces), F(0))


def scaled_error(n):
    return n * sum((cell_error(n, cell) for cell in range(n)), F(0))


def phase_term(n):
    total = SLOPE / 4
    for site, jump in JUMPS:
        phase = (n * site) % 1
        total += 2 * jump * phase * (1 - phase)
    return total


def main():
    global SLOPE
    checks = 0
    largest_residual = F(0)
    largest_ratio = F(0)
    for n in (24, 31, 47, 64, 83, 97, 105, 106):
        observed = scaled_error(n)
        predicted = phase_term(n)
        residual = abs(observed - predicted)
        bound = F(len(JUMPS), 2 * n) * SLOPE
        assert residual <= bound
        largest_residual = max(largest_residual, residual)
        largest_ratio = max(largest_ratio, residual / bound)
        checks += 1

    # With zero slope, separated jumps obey the phase formula exactly.
    old_slope = SLOPE
    SLOPE = F(0)
    for n in (24, 31, 47, 64, 83, 97, 105, 106):
        assert scaled_error(n) == phase_term(n)
        checks += 1
    SLOPE = old_slope

    aligned = scaled_error(105)
    adjacent = scaled_error(106)
    adjacent_phase = phase_term(106)
    adjacent_residual = abs(adjacent - adjacent_phase)

    print(f"exact SBV checks: {checks}")
    print(f"largest scaled residual: {float(largest_residual):.12f}")
    print(f"largest residual/bound: {float(largest_ratio):.12f}")
    print(f"N=105 scaled error: {float(aligned):.12f}")
    print(f"N=106 scaled error: {float(adjacent):.12f}")
    print(f"N=106 phase prediction: {float(adjacent_phase):.12f}")
    print(f"N=106 scaled residual: {float(adjacent_residual):.12f}")
    print("PASS")


if __name__ == "__main__":
    main()
