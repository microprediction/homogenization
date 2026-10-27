"""Exact certificate for the all-r transform-zero crossing bound.

The theorem in call-crossings-full.html counts transform zeros with their
full multiplicities.  This certificate checks that the crossing lower bound
cannot be improved from those identities alone, and that every larger fixed-
support feasibility cone is generated conformally by its minimal-support
circuits.  The finite fixtures and disjoint-interval continuous blurs use
exact Fraction arithmetic.
"""

from fractions import Fraction as F
from itertools import combinations, product
from math import comb


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


def matrix_rank(rows):
    """Return the exact rank of a Fraction matrix."""
    work = [row[:] for row in rows]
    rank = 0
    column_count = len(work[0]) if work else 0
    for column in range(column_count):
        pivot = next((i for i in range(rank, len(work))
                      if work[i][column]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        pivot_value = work[rank][column]
        work[rank] = [value / pivot_value for value in work[rank]]
        for i in range(len(work)):
            if i == rank or not work[i][column]:
                continue
            factor = work[i][column]
            work[i] = [left - factor * right
                       for left, right in zip(work[i], work[rank])]
        rank += 1
        if rank == len(work):
            break
    return rank


def determinant(rows):
    """Return an exact determinant by Fraction elimination."""
    assert all(len(row) == len(rows) for row in rows)
    work = [row[:] for row in rows]
    result = F(1)
    for column in range(len(work)):
        pivot = next((i for i in range(column, len(work))
                      if work[i][column]), None)
        if pivot is None:
            return F(0)
        if pivot != column:
            work[column], work[pivot] = work[pivot], work[column]
            result = -result
        pivot_value = work[column][column]
        result *= pivot_value
        for i in range(column + 1, len(work)):
            factor = work[i][column] / pivot_value
            for j in range(column + 1, len(work)):
                work[i][j] -= factor * work[column][j]
            work[i][column] = F(0)
    return result


def confluent_evaluation_rows(scaled_exponents, q_values):
    """Return the exact generalized exponential evaluation matrix."""
    multiplicities = {}
    for exponent in scaled_exponents:
        multiplicities[exponent] = multiplicities.get(exponent, 0) + 1
    return [
        [F(q) ** power * F(2 ** (exponent * q)) for q in q_values]
        for exponent in sorted(multiplicities)
        for power in range(multiplicities[exponent])
    ]


def circuit_vector(rows, columns):
    """Return the signed cofactor circuit on p+1 selected columns."""
    row_count = len(rows)
    assert len(columns) == row_count + 1
    selected = [[row[column] for column in columns] for row in rows]
    values = []
    for omitted in range(row_count + 1):
        minor = [row[:omitted] + row[omitted + 1:] for row in selected]
        values.append(F((-1) ** omitted) * determinant(minor))
    result = [F(0)] * len(rows[0])
    for column, value in zip(columns, values):
        result[column] = value
    assert all(rowdot(row, result) == 0 for row in rows)
    return result


def conformal_circuit(rows, vector):
    """Find an exact full-spark circuit in the vector's closed orthant."""
    support = [j for j, value in enumerate(vector) if value]
    for columns in combinations(support, len(rows) + 1):
        circuit = circuit_vector(rows, columns)
        products = [vector[j] * circuit[j] for j in columns]
        if all(value > 0 for value in products):
            return circuit
        if all(value < 0 for value in products):
            return [-value for value in circuit]
    raise AssertionError("the kernel orthant has no circuit")


def conformal_circuit_decomposition(rows, vector):
    """Decompose a kernel vector without cancelling any signed mass."""
    original = vector[:]
    residual = vector[:]
    components = []
    while any(residual):
        circuit = conformal_circuit(rows, residual)
        active = [j for j, value in enumerate(circuit) if value]
        scale = min(abs(residual[j] / circuit[j]) for j in active)
        component = [scale * value for value in circuit]
        updated = [left - right for left, right in zip(residual, component)]
        assert all(not left or not right or left * right > 0
                   for left, right in zip(residual, updated))
        assert sum(value != 0 for value in updated) < \
            sum(value != 0 for value in residual)
        assert all(rowdot(row, component) == 0 for row in rows)
        components.append(component)
        residual = updated
    assert [sum((component[j] for component in components), F(0))
            for j in range(len(original))] == original
    assert len(components) <= len(rows[0]) - len(rows)
    return components


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


def run_lengths(signs):
    """Return the lengths of consecutive constant-sign blocks."""
    lengths = [1]
    for left, right in zip(signs, signs[1:]):
        if left == right:
            lengths[-1] += 1
        else:
            lengths.append(1)
    return lengths


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


def check_minimal_support_rigidity(max_r=10):
    """Check uniqueness on irregular support grids by exact cofactors.

    Put the support at x_j=2**(denominator*q_j), where the integer q_j
    are triangular numbers rather than an arithmetic progression.  A
    scaled transform exponent k and derivative order h give the rational
    row q_j**h * 2**(k*q_j).  Exact rank and cofactor signs certify the
    one-dimensional alternating nullspace predicted by the ECT theorem.
    """
    cases = []
    for r in range(2, max_r + 1):
        denominator = r - 1
        scaled_exponents = [0, denominator] + list(range(r))
        cases.append((r, denominator, scaled_exponents))
    for left, middle, right in ((2, 1, 2), (1, 3, 1), (3, 2, 2)):
        total = left + middle + right
        scaled_exponents = [0, 2] + [0] * left + [1] * middle + [2] * right
        cases.append((total, 2, scaled_exponents))

    for crossing_count, denominator, scaled_exponents in cases:
        row_count = len(scaled_exponents)
        q_values = [j * (j + 1) // 2 for j in range(row_count + 1)]
        rows = confluent_evaluation_rows(scaled_exponents, q_values)
        signed = []
        for column in range(row_count + 1):
            minor = [row[:column] + row[column + 1:] for row in rows]
            signed.append(F((-1) ** column) * determinant(minor))

        assert len(rows) == row_count
        assert matrix_rank(rows) == row_count
        assert all(value != 0 for value in signed)
        assert all(signed[j] * signed[j + 1] < 0
                   for j in range(len(signed) - 1))
        assert all(rowdot(row, signed) == 0 for row in rows)

        normalizer = sum((max(value, F(0)) for value in signed), F(0))
        p = [max(value, F(0)) / normalizer for value in signed]
        q = [max(-value, F(0)) / normalizer for value in signed]
        raw_points = [F(2 ** (denominator * value)) for value in q_values]
        raw_mean = rowdot(p, raw_points)
        points = [point / raw_mean for point in raw_points]
        assert sum(p, F(0)) == sum(q, F(0)) == 1
        assert rowdot(p, points) == rowdot(q, points) == 1
        gaps = [call(points, p, strike) - call(points, q, strike)
                for strike in points]
        assert gaps[0] == gaps[-1] == 0
        assert sign_changes(gaps[1:-1]) == crossing_count
    return len(cases)


def check_larger_support_circuit_cones(max_r=7):
    """Check conformal extreme-ray decompositions on larger fixed grids.

    The kernel in any fixed coordinate orthant is a pointed polyhedral
    cone.  Full-spark evaluation makes every extreme ray a circuit on
    exactly row_count+1 columns.  The elimination routine above finds a
    sign-compatible circuit, subtracts the largest possible multiple, and
    therefore removes at least one support point at every step.
    """
    cases = []
    for r in range(2, max_r + 1):
        denominator = r - 1
        cases.append((r, denominator, [0, denominator] + list(range(r))))
    for left, middle, right in ((2, 1, 2), (1, 3, 1), (3, 2, 2)):
        total = left + middle + right
        cases.append((total, 2,
                      [0, 2] + [0] * left + [1] * middle + [2] * right))

    decomposition_count = 0
    circuit_count = 0
    for crossing_count, denominator, scaled_exponents in cases:
        row_count = len(scaled_exponents)
        for excess in (1, 2, 3):
            column_count = row_count + 1 + excess
            q_values = [j * (j + 1) // 2 for j in range(column_count)]
            rows = confluent_evaluation_rows(scaled_exponents, q_values)
            assert matrix_rank(rows) == row_count

            base = tuple(range(row_count))
            fundamental = []
            for column in range(row_count, column_count):
                circuit = circuit_vector(rows, base + (column,))
                if circuit[column] < 0:
                    circuit = [-value for value in circuit]
                assert all(circuit[j] != 0 for j in base + (column,))
                fundamental.append(circuit)

            # Choose a generic exact kernel vector.  Mixed coefficient
            # signs make this independent of the displayed fundamental
            # decomposition; exhaustive small choices avoid cancellations.
            vector = None
            coefficient_choices = tuple(range(-10, 0)) + tuple(range(1, 11))
            for coefficients in product(coefficient_choices, repeat=len(fundamental)):
                candidate = [
                    sum((coefficient * circuit[j]
                         for coefficient, circuit in zip(coefficients, fundamental)), F(0))
                    for j in range(column_count)
                ]
                if all(candidate):
                    vector = candidate
                    break
            assert vector is not None
            assert all(rowdot(row, vector) == 0 for row in rows)

            components = conformal_circuit_decomposition(rows, vector)
            assert len(components) <= excess + 1
            for component in components:
                support = [j for j, value in enumerate(component) if value]
                assert len(support) == row_count + 1
                assert all(component[j] * vector[j] > 0 for j in support)
                assert all(component[left] * component[right] < 0
                           for left, right in zip(support, support[1:]))

                positive_mass = sum((max(value, F(0)) for value in component), F(0))
                negative_mass = sum((max(-value, F(0)) for value in component), F(0))
                assert positive_mass == negative_mass > 0
                raw_points = [F(2 ** (denominator * value)) for value in q_values]
                positive_mean = sum((max(component[j], F(0)) * raw_points[j]
                                     for j in support), F(0))
                negative_mean = sum((max(-component[j], F(0)) * raw_points[j]
                                     for j in support), F(0))
                assert positive_mean == negative_mean > 0

                p = [max(value, F(0)) / positive_mass for value in component]
                q = [max(-value, F(0)) / negative_mass for value in component]
                points = [point * positive_mass / positive_mean for point in raw_points]
                gaps = [call(points, p, strike) - call(points, q, strike)
                        for strike in points]
                assert sign_changes(gaps[1:-1]) == crossing_count
                circuit_count += 1
            decomposition_count += 1
    return decomposition_count, circuit_count


def run_face_vector(signs, row_count):
    """Count faces by a run-length dynamic program.

    A state records the selected support size, the sign of the last selected
    run, and its restricted sign variation.  Choosing a nonempty subset of a
    run of length n has binomial weight C(n,j); only a change from the last
    selected sign increments the variation.
    """
    lengths = run_lengths(signs)
    first_sign = signs[0]
    states = {(0, None, 0): 1}
    for run_index, length in enumerate(lengths):
        sign = first_sign if run_index % 2 == 0 else -first_sign
        updated = dict(states)
        for (size, last_sign, variation), count in states.items():
            change = int(last_sign is not None and last_sign != sign)
            for chosen in range(1, length + 1):
                key = (size + chosen, sign, variation + change)
                updated[key] = updated.get(key, 0) + count * comb(length, chosen)
        states = updated

    dimension = len(signs) - row_count
    result = [1] + [0] * dimension
    for (size, last_sign, variation), count in states.items():
        if last_sign is not None and size >= row_count + 1 and variation >= row_count:
            result[size - row_count] += count
    return result


def closed_orthant_face_census(column_count, row_count):
    """Count feasible sign orbits and face incidences without enumeration."""
    dimension = column_count - row_count
    feasible_orbits = sum(comb(column_count - 1, changes)
                          for changes in range(row_count, column_count))
    incidences = [feasible_orbits]
    for face_dimension in range(1, dimension + 1):
        support_size = row_count + face_dimension
        restricted_words = sum(comb(support_size - 1, changes)
                               for changes in range(row_count, support_size))
        incidences.append(comb(column_count, support_size)
                          * 2 ** (column_count - support_size)
                          * restricted_words)
    return feasible_orbits, incidences


def check_orthant_circuit_combinatorics(max_row_count=9, max_excess=3):
    """Enumerate exact orthant feasibility, rays and face lattices."""
    orthant_count = 0
    feasible_count = 0
    ray_count = 0
    simplicial_count = 0
    face_count = 0
    face_vectors = set()
    run_enumerator_checks = 0
    closed_census_checks = 0
    for row_count in range(4, max_row_count + 1):
        for excess in range(max_excess + 1):
            column_count = row_count + 1 + excess
            dimension = column_count - row_count
            subsets = list(combinations(range(column_count), row_count + 1))
            case_feasible = 0
            case_face_incidences = [0] * (dimension + 1)

            # A rational Vandermonde representative independently checks
            # that every circuit has the predicted alternating orientation.
            points = [F(j + 1) for j in range(column_count)]
            rows = [[point ** degree for point in points]
                    for degree in range(row_count)]
            assert matrix_rank(rows) == row_count
            for columns in subsets:
                circuit = circuit_vector(rows, columns)
                values = [circuit[j] for j in columns]
                assert all(left * right < 0
                           for left, right in zip(values, values[1:]))

            # Global sign reversal gives the same orthant cone, so fix the
            # first sign positive and exhaust the remaining patterns.
            for tail in product((-1, 1), repeat=column_count - 1):
                signs = (1,) + tail
                compatible = [
                    columns for columns in subsets
                    if all(signs[left] * signs[right] < 0
                           for left, right in zip(columns, columns[1:]))
                ]
                variation = sign_changes(signs)
                assert bool(compatible) == (variation >= row_count)
                orthant_count += 1
                if not compatible:
                    continue

                feasible_count += 1
                case_feasible += 1
                ray_count += len(compatible)
                covered = {column for columns in compatible for column in columns}
                assert len(covered) == column_count
                assert len(compatible) >= column_count - row_count

                # Every face is obtained by forcing a set of coordinates
                # to zero.  A surviving support S is the exact relative-
                # interior support of a nonzero face precisely when its
                # restricted sign word has at least row_count changes.
                # Its dimension is |S|-row_count.  This exhausts the face
                # lattice, including the zero face in position zero.
                f_vector = [1] + [0] * dimension
                for size in range(row_count + 1, column_count + 1):
                    face_dimension = size - row_count
                    for support in combinations(range(column_count), size):
                        restricted = [signs[column] for column in support]
                        if sign_changes(restricted) >= row_count:
                            f_vector[face_dimension] += 1
                assert run_face_vector(signs, row_count) == f_vector
                run_enumerator_checks += 1
                assert f_vector[1] == len(compatible)
                assert f_vector[dimension] == 1
                assert sum((-1) ** k * count
                           for k, count in enumerate(f_vector)) == 0
                face_count += sum(f_vector)
                case_face_incidences = [left + right for left, right
                                        in zip(case_face_incidences, f_vector)]
                face_vectors.add(tuple(f_vector))

                if variation == row_count:
                    lengths = run_lengths(signs)
                    product_count = 1
                    for length in lengths:
                        product_count *= length
                    assert len(lengths) == row_count + 1
                    assert len(compatible) == product_count
                    is_simplicial = sum(length > 1 for length in lengths) <= 1
                    assert (len(compatible) == column_count - row_count) == is_simplicial
                    if is_simplicial:
                        assert f_vector == [comb(dimension, k)
                                            for k in range(dimension + 1)]
                    simplicial_count += is_simplicial
            closed_feasible, closed_incidences = \
                closed_orthant_face_census(column_count, row_count)
            assert case_feasible == closed_feasible
            assert case_face_incidences == closed_incidences
            closed_census_checks += 1
    return (orthant_count, feasible_count, ray_count, simplicial_count,
            face_count, len(face_vectors), run_enumerator_checks,
            closed_census_checks)


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
    rigidity_cases = check_minimal_support_rigidity()
    cone_cases, cone_circuits = check_larger_support_circuit_cones()
    (orthants, feasible_orthants, extreme_rays, simplicial,
     faces, face_vectors, run_enumerators, closed_censuses) = \
        check_orthant_circuit_combinatorics()
    blur_cases = check_measure_blurs()
    roots3 = check_three_crossings()
    roots = check_two_crossings()
    root = check_one_crossing()
    print(f"all-r support- and atom-minimal sharpness family: r=2,...,{max_r}")
    print("confluent sharpness total multiplicities:", ", ".join(map(str, confluent_totals)))
    print(f"irregular minimal-support rigidity cases: {rigidity_cases}")
    print(f"larger-support conformal circuit cones: {cone_cases} decompositions, "
          f"{cone_circuits} minimal circuits")
    print(f"orthant circuit enumeration: {orthants} patterns, "
          f"{feasible_orthants} feasible, {extreme_rays} extreme rays, "
          f"{simplicial} critical simplicial cones")
    print(f"complete orthant face lattices: {faces} faces, "
          f"{face_vectors} distinct f-vectors")
    print(f"run-length face enumerators: {run_enumerators} exact f-vector matches")
    print(f"closed orthant face censuses: {closed_censuses} exact incidence matches")
    print(f"exact finite atom-variation cases: {variation_cases}")
    print(f"exact continuous/mixed-blur sharpness cases: {blur_cases}/{blur_cases}")
    print("three-moment sharp roots:", ", ".join(str(x) for x in roots3))
    print("two-moment sharp roots:", ", ".join(str(x) for x in roots))
    print("one-moment sharp root:", root)
    print("PASS")


if __name__ == "__main__":
    main()
