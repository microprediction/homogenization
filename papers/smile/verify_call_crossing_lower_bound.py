"""Exact certificate for the all-r transform-zero crossing bound.

The theorem in call-crossings-full.html counts transform zeros with their
full multiplicities.  This certificate checks that the crossing lower bound
cannot be improved from those identities alone.  The finite fixtures and
disjoint-interval continuous blurs use exact Fraction arithmetic.
"""

from fractions import Fraction as F
from itertools import product


def moments(points, weights):
    """Return mass, E[S], and the coefficients of log 2 and log scale."""
    mass = sum(weights, F(0))
    mean = sum((w * s for s, w in zip(points, weights)), F(0))
    # Points have the form 2**j / scale.  The log moment is
    # E[j] log 2 - log(scale), and the share-log moment has the analogous
    # coefficients E[S j] and -E[S].
    log2 = sum((w * j for j, w in enumerate(weights)), F(0))
    share_log2 = sum((w * s * j for j, (s, w) in enumerate(zip(points, weights))), F(0))
    return mass, mean, log2, share_log2


def call(points, weights, strike):
    return sum((w * max(s - strike, F(0)) for s, w in zip(points, weights)), F(0))


def matmul(a, b):
    return [[sum((a[i][k] * b[k][j] for k in range(len(b))), F(0))
             for j in range(len(b[0]))] for i in range(len(a))]


def matvec(a, x):
    return [sum((a[i][j] * x[j] for j in range(len(x))), F(0))
            for i in range(len(a))]


def rowdot(x, y):
    return sum((u * v for u, v in zip(x, y)), F(0))


def polymul(a, b):
    result = [F(0)] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            result[i + j] += x * y
    return result


def polynomial_from_roots(roots):
    result = [F(1)]
    for root in roots:
        result = polymul(result, [-root, F(1)])
    return result


def polyval(coefficients, point):
    """Evaluate an ascending-order polynomial exactly."""
    result = F(0)
    for coefficient in reversed(coefficients):
        result = result * point + coefficient
    return result


def derivative(coefficients):
    return [j * coefficients[j] for j in range(1, len(coefficients))]


def root_order(coefficients, point):
    order = 0
    current = coefficients
    while len(current) > 1 and polyval(current, point) == 0:
        order += 1
        current = derivative(current)
    return order


def sign_changes(values):
    """Count strict sign changes after deleting zeros."""
    signs = [(value > 0) - (value < 0) for value in values if value]
    return sum(left != right for left, right in zip(signs, signs[1:]))


def check_atom_variation_bound(max_atoms=8):
    """Exhaust small exact signed measures satisfying mass and mean balance."""
    checked = 0
    for atom_count in range(3, max_atoms + 1):
        points = list(range(1, atom_count + 1))
        for signed in product((-2, -1, 1, 2), repeat=atom_count):
            if sum(signed) or rowdot(signed, points):
                continue
            gaps = [sum(signed[j] * (points[j] - points[i])
                        for j in range(i + 1, atom_count))
                    for i in range(atom_count)]
            crossings = sign_changes(gaps[1:-1])
            variations = sign_changes(signed)
            assert crossings <= variations - 2
            positive_atoms = sum(weight > 0 for weight in signed)
            negative_atoms = atom_count - positive_atoms
            assert min(positive_atoms, negative_atoms) >= (crossings + 3) // 2
            assert max(positive_atoms, negative_atoms) >= (crossings + 4) // 2
            checked += 1
    assert checked == 288
    return checked


def check_reversal_moment_identity():
    """Check the leverage functional coefficient by coefficient, exactly."""
    q = [[F(-1), F(1), F(0)],
         [F(0), F(-2), F(2)],
         [F(3), F(0), F(-3)]]
    pi = [F(6, 11), F(3, 11), F(2, 11)]
    a = [F(7), F(11), F(13)]
    c = [F(2), F(3), F(5)]
    one = [F(1)] * 3
    assert [rowdot(pi, [q[i][j] for i in range(3)]) for j in range(3)] == [F(0)] * 3

    # Q^left_{ij}=pi_j Q_{ji}/pi_i and A=Q-D_c.
    qr = [[pi[j] * q[j][i] / pi[i] for j in range(3)] for i in range(3)]
    aa = [[q[i][j] - (c[i] if i == j else 0) for j in range(3)] for i in range(3)]
    ar = [[qr[i][j] - (c[i] if i == j else 0) for j in range(3)] for i in range(3)]
    power = [[F(int(i == j)) for j in range(3)] for i in range(3)]
    power_r = [[F(int(i == j)) for j in range(3)] for i in range(3)]
    pia = [pi[i] * a[i] for i in range(3)]

    for _ in range(7):
        forward_forcing = rowdot(pia, matvec(power, one))
        reverse_forcing = rowdot(pia, matvec(power_r, one))
        functional = forward_forcing - rowdot(pi, matvec(power, a))
        assert forward_forcing - reverse_forcing == functional
        assert rowdot(pi, matvec(power, one)) == rowdot(pi, matvec(power_r, one))
        power = matmul(power, aa)
        power_r = matmul(power_r, ar)


def check_two_crossings():
    # Signed coefficients of (z-1)^2(z-2)^2, normalized into two laws
    # and rescaled to unit mean.
    signed = [F(4), F(-12), F(13), F(-6), F(1)]
    assert signed == [F(4), F(-12), F(13), F(-6), F(1)]
    points = [F(1, 4), F(1, 2), F(1), F(2), F(4)]
    p = [max(w, F(0)) / 18 for w in signed]
    q = [max(-w, F(0)) / 18 for w in signed]

    mp, mq = moments(points, p), moments(points, q)
    assert mp[:4] == mq[:4]
    assert mp[0] == 1 and mp[1] == 1

    # Exact roots and one test point in each nonzero sign interval.
    roots = [F(5, 8), F(8, 5)]
    assert call(points, p, roots[0]) == call(points, q, roots[0])
    assert call(points, p, roots[1]) == call(points, q, roots[1])
    tests = [F(3, 8), F(3, 4), F(6, 5), F(9, 5), F(3)]
    signs = []
    for strike in tests:
        gap = call(points, p, strike) - call(points, q, strike)
        signs.append((gap > 0) - (gap < 0))
    assert signs == [1, -1, -1, 1, 1]
    return roots


def check_three_crossings():
    # Signed coefficients of (z-1)^2(z-2)(z-4)^2.  With stock points
    # 4**j / 32, the roots at 1 and 4 match the endpoint log moments,
    # while the root at 2 matches the square-root moment.
    signed = [F(-32), F(96), F(-106), F(53), F(-12), F(1)]
    points = [F(1, 32), F(1, 8), F(1, 2), F(2), F(8), F(32)]
    p = [max(w, F(0)) / 150 for w in signed]
    q = [max(-w, F(0)) / 150 for w in signed]

    assert sum(p, F(0)) == sum(q, F(0)) == 1
    assert sum((w * s for s, w in zip(points, p)), F(0)) == 1
    assert sum((w * s for s, w in zip(points, q)), F(0)) == 1

    # Since log_2(S_j)=2j-5, these are the ordinary and share-log
    # coefficients.  The square-root equality reduces to P(2)=0 after
    # cancelling the common factor 1/sqrt(32).
    exponents = [2 * j - 5 for j in range(6)]
    assert rowdot(p, exponents) == rowdot(q, exponents)
    assert rowdot([p[j] * points[j] for j in range(6)], exponents) == \
        rowdot([q[j] * points[j] for j in range(6)], exponents)
    sqrt_coefficients = [F(2 ** j) for j in range(6)]
    assert rowdot(p, sqrt_coefficients) == rowdot(q, sqrt_coefficients)

    roots = [F(11, 64), F(1), F(64, 11)]
    for root in roots:
        assert call(points, p, root) == call(points, q, root)
    tests = [F(1, 16), F(1, 4), F(3, 4), F(3, 2), F(4), F(6), F(16)]
    signs = []
    for strike in tests:
        gap = call(points, p, strike) - call(points, q, strike)
        signs.append((gap > 0) - (gap < 0))
    assert signs == [-1, 1, 1, -1, -1, 1, 1]
    return roots


def check_all_r_sharpness(max_r=10):
    """Check the geometric-grid construction, tail identity and roots exactly."""
    for r in range(2, max_r + 1):
        # Nodes s=l/(r-1) give a_l=2**l when b=2**(r-1).
        b = F(2 ** (r - 1))
        transform_roots = [F(2 ** ell) for ell in range(r)]
        s_poly = polynomial_from_roots(transform_roots)
        coefficients = polynomial_from_roots([F(1), b] + transform_roots)
        assert coefficients == polymul(polymul(s_poly, [-F(1), F(1)]), [-b, F(1)])
        assert all(coefficients[j] * coefficients[j + 1] < 0
                   for j in range(len(coefficients) - 1))

        normalizer = sum((max(c, F(0)) for c in coefficients), F(0))
        p = [max(c, F(0)) / normalizer for c in coefficients]
        q = [max(-c, F(0)) / normalizer for c in coefficients]
        raw_mean = sum((p[j] * b ** j for j in range(len(p))), F(0))
        points = [b ** j / raw_mean for j in range(len(p))]
        degree = len(coefficients) - 1
        reciprocal_scale = F(2 ** (((r - 1) * degree) // 2))
        assert raw_mean == reciprocal_scale
        for j, coefficient in enumerate(coefficients):
            assert coefficients[degree - j] * b ** (degree - j) == \
                F((-1) ** degree) * reciprocal_scale * coefficient
        assert len(points) == r + 3
        assert all(weight != 0 for weight in coefficients)
        positive_atoms = sum(c > 0 for c in coefficients)
        negative_atoms = sum(c < 0 for c in coefficients)
        assert sorted((positive_atoms, negative_atoms)) == \
            [len(points) // 2, (len(points) + 1) // 2]
        assert sum(p, F(0)) == sum(q, F(0)) == 1
        assert rowdot(p, points) == rowdot(q, points) == 1

        # Endpoint derivatives give the log and share-log identities.
        powers = [F(j) for j in range(len(p))]
        assert rowdot(p, powers) == rowdot(q, powers)
        assert rowdot([p[j] * points[j] for j in range(len(p))], powers) == \
            rowdot([q[j] * points[j] for j in range(len(p))], powers)

        # Every interior transform node is an exact fractional-moment root.
        for ell in range(1, r - 1):
            moment_weights = [F(2 ** (ell * j)) for j in range(len(p))]
            assert rowdot(p, moment_weights) == rowdot(q, moment_weights)

        gaps = [call(points, p, strike) - call(points, q, strike)
                for strike in points]
        assert gaps[0] == gaps[-1] == 0
        for i in range(1, r + 2):
            expected = b ** i * (b - 1) * s_poly[i - 1] / (normalizer * raw_mean)
            assert gaps[i] == expected != 0
        assert all(gaps[i] * gaps[i + 1] < 0 for i in range(1, r + 1))

        # The gap is affine between atoms, so each alternating pair has
        # exactly one root.  Check its rational location is strictly inside.
        roots = []
        for i in range(1, r + 1):
            tail_mass = sum((p[j] - q[j] for j in range(i + 1, len(p))), F(0))
            tail_mean = sum(((p[j] - q[j]) * points[j]
                             for j in range(i + 1, len(p))), F(0))
            root = tail_mean / tail_mass
            assert points[i] < root < points[i + 1]
            roots.append(root)
        assert len(roots) == r
        assert all(roots[i] * roots[-1 - i] == 1 for i in range(r))
        if r % 2:
            assert roots[r // 2] == 1

        # The symmetric node grid is stock-numeraire self-dual when r is
        # even and exchanges the two laws when r is odd.
        for j in range(len(points)):
            if r % 2:
                assert q[degree - j] == points[j] * p[j]
                assert p[degree - j] == points[j] * q[j]
            else:
                assert p[degree - j] == points[j] * p[j]
                assert q[degree - j] == points[j] * q[j]
            gap = call(points, p, points[j]) - call(points, q, points[j])
            dual_gap = call(points, p, 1 / points[j]) - call(points, q, 1 / points[j])
            assert gap == F((-1) ** r) * points[j] * dual_gap
    return max_r


def check_confluent_sharpness():
    """Check repeated transform nodes and their exact crossing counts."""
    # b=16 makes the nodes s=0, 1/2, 1 correspond to rational roots
    # 1, 4, 16.  The endpoint denominator removes one baseline root.
    b = F(16)
    cases = [(2, 1, 2), (1, 3, 1), (3, 2, 2)]
    totals = []
    for left, middle, right in cases:
        multiplicities = (left, middle, right)
        transform_roots = [F(1)] * left + [F(4)] * middle + [b] * right
        total = sum(multiplicities)
        s_poly = polynomial_from_roots(transform_roots)
        coefficients = polynomial_from_roots([F(1), b] + transform_roots)
        assert root_order(coefficients, F(1)) == left + 1
        assert root_order(coefficients, F(4)) == middle
        assert root_order(coefficients, b) == right + 1

        # Derivatives of Delta M at a node a are triangular combinations
        # of sum_j c_j a**j j**h.  These exact identities certify the
        # asserted endpoint and interior multiplicities.
        for root, order in ((F(1), left + 1), (F(4), middle), (b, right + 1)):
            for power in range(order):
                assert sum((c * root ** j * F(j) ** power
                            for j, c in enumerate(coefficients)), F(0)) == 0

        assert len(coefficients) == total + 3
        assert all(coefficients[j] * coefficients[j + 1] < 0
                   for j in range(len(coefficients) - 1))
        normalizer = sum((max(c, F(0)) for c in coefficients), F(0))
        p = [max(c, F(0)) / normalizer for c in coefficients]
        q = [max(-c, F(0)) / normalizer for c in coefficients]
        raw_mean = sum((p[j] * b ** j for j in range(len(p))), F(0))
        points = [b ** j / raw_mean for j in range(len(p))]
        assert sum(p, F(0)) == sum(q, F(0)) == 1
        assert rowdot(p, points) == rowdot(q, points) == 1

        gaps = [call(points, p, strike) - call(points, q, strike)
                for strike in points]
        assert gaps[0] == gaps[-1] == 0
        for i in range(1, total + 2):
            expected = b ** i * (b - 1) * s_poly[i - 1] / (normalizer * raw_mean)
            assert gaps[i] == expected != 0
        assert sign_changes(gaps[1:-1]) == total

        # Confluence is in transform space only.  The ordinary-strike
        # crossings remain distinct, simple roots that interlace the grid.
        roots = []
        for i in range(1, total + 1):
            tail_mass = sum((p[j] - q[j] for j in range(i + 1, len(p))), F(0))
            tail_mean = sum(((p[j] - q[j]) * points[j]
                             for j in range(i + 1, len(p))), F(0))
            root = tail_mean / tail_mass
            assert points[i] < root < points[i + 1]
            assert call(points, p, root) == call(points, q, root)
            roots.append(root)
        assert len(roots) == len(set(roots)) == total
        totals.append(total)
    return totals


def check_measure_blurs(max_r=10):
    """Check the exact disjoint continuous and mixed smoothing construction.

    Multiplication by W uniform on [3/4,5/4] replaces each atom x by a
    constant density on [3x/4,5x/4].  A half atom, half uniform W gives a
    genuinely mixed law.  The geometric grids below have adjacent ratio at
    least two, so the sign blocks are disjoint.  Mellin moments factor by the
    common positive moment E[W**s], preserving every node.
    """
    epsilon = F(1, 4)
    alpha = F(1, 2)
    assert (1 - alpha) + alpha == 1
    assert (1 - alpha) + alpha * ((1 - epsilon) + (1 + epsilon)) / 2 == 1
    checked = 0
    for r in range(2, max_r + 1):
        b = F(2 ** (r - 1))
        transform_roots = [F(2 ** ell) for ell in range(r)]
        coefficients = polynomial_from_roots([F(1), b] + transform_roots)
        normalizer = sum((max(c, F(0)) for c in coefficients), F(0))
        p = [max(c, F(0)) / normalizer for c in coefficients]
        raw_mean = sum((p[j] * b ** j for j in range(len(p))), F(0))
        points = [b ** j / raw_mean for j in range(len(p))]

        intervals = [(x * (1 - epsilon), x * (1 + epsilon)) for x in points]
        assert all(intervals[j][1] < intervals[j + 1][0]
                   for j in range(len(intervals) - 1))
        assert sign_changes(coefficients) == r + 2

        # Unit mass and mean survive because E[W]=1.  Each fractional
        # moment difference is multiplied by the same E[W**s].
        assert sum((max(c, F(0)) for c in coefficients), F(0)) == \
            sum((max(-c, F(0)) for c in coefficients), F(0))
        assert sum((c * b ** j for j, c in enumerate(coefficients)), F(0)) == 0
        for root in [F(1), b] + transform_roots:
            assert sum((c * root ** j for j, c in enumerate(coefficients)), F(0)) == 0
        checked += 1
    assert checked == max_r - 1
    return checked


def check_one_crossing():
    # Signed coefficients of (z-1)^2(z-2), again normalized and rescaled.
    signed = [F(-2), F(5), F(-4), F(1)]
    points = [F(1, 3), F(2, 3), F(4, 3), F(8, 3)]
    p = [max(w, F(0)) / 6 for w in signed]
    q = [max(-w, F(0)) / 6 for w in signed]

    mp, mq = moments(points, p), moments(points, q)
    assert mp[:3] == mq[:3]
    assert mp[0] == 1 and mp[1] == 1
    assert mp[3] != mq[3]

    root = F(8, 9)
    assert call(points, p, root) == call(points, q, root)
    tests = [F(1, 2), F(3, 4), F(1), F(2)]
    signs = []
    for strike in tests:
        gap = call(points, p, strike) - call(points, q, strike)
        signs.append((gap > 0) - (gap < 0))
    assert signs == [-1, -1, 1, 1]
    return root


def main():
    check_reversal_moment_identity()
    variation_cases = check_atom_variation_bound()
    max_r = check_all_r_sharpness()
    confluent_totals = check_confluent_sharpness()
    blur_cases = check_measure_blurs()
    roots3 = check_three_crossings()
    roots = check_two_crossings()
    root = check_one_crossing()
    print(f"all-r support- and atom-minimal sharpness family: r=2,...,{max_r}")
    print("confluent sharpness total multiplicities:", ", ".join(map(str, confluent_totals)))
    print(f"exact finite atom-variation cases: {variation_cases}")
    print(f"exact continuous/mixed-blur sharpness cases: {blur_cases}/{blur_cases}")
    print("three-moment sharp roots:", ", ".join(str(x) for x in roots3))
    print("two-moment sharp roots:", ", ".join(str(x) for x in roots))
    print("one-moment sharp root:", root)
    print("PASS")


if __name__ == "__main__":
    main()
