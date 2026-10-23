"""Exact certificate for the one-, two-, and three-crossing moment bounds.

The theorem in call-crossings-full.html uses two positive-weight call
integrals.  This certificate checks that neither resulting crossing lower
bound can be improved from those moment identities alone.  Both fixtures are
finite martingale laws, so every calculation is exact Fraction arithmetic.
"""

from fractions import Fraction as F


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
    roots3 = check_three_crossings()
    roots = check_two_crossings()
    root = check_one_crossing()
    print("three-moment sharp roots:", ", ".join(str(x) for x in roots3))
    print("two-moment sharp roots:", ", ".join(str(x) for x in roots))
    print("one-moment sharp root:", root)
    print("PASS")


if __name__ == "__main__":
    main()
