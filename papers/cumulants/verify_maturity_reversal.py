"""Killed-chain covariance and a certified CIR reversal-gap maturity crossing."""
import numpy as np
import sympy as sp
from scipy.linalg import expm


def stationary(q, kappa, theta, a):
    n = len(kappa)
    pi = np.linalg.lstsq(np.vstack([q.T, np.ones(n)]), np.r_[np.zeros(n), 1.], rcond=None)[0]
    first = np.linalg.solve(np.diag(kappa) - q.T, pi * kappa * theta)
    second = np.linalg.solve(2 * np.diag(kappa) - q.T, first * (2 * kappa * theta + a))
    return pi, first, second


def integrated_moments(q, kappa, theta, a, time, rows):
    # Independent degree-two joint generator for z=integral(v-theta).
    n = len(kappa)
    pairs = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]
    where = {pair: j for j, pair in enumerate(pairs)}
    system, initial = np.zeros((6 * n, 6 * n)), np.zeros(6 * n)
    def block(pair):
        j = where[pair]
        return slice(j * n, (j + 1) * n)
    for zpower, vpower in pairs:
        target = block((zpower, vpower))
        system[target, target] = q.T - np.diag(vpower * kappa)
        if zpower:
            system[target, block((zpower - 1, vpower + 1))] += zpower * np.eye(n)
            system[target, block((zpower - 1, vpower))] -= zpower * theta * np.eye(n)
        if vpower:
            system[target, block((zpower, vpower - 1))] += np.diag(vpower * kappa * theta + vpower * (vpower - 1) * a / 2)
        if not zpower:
            initial[target] = rows[vpower]
    value = expm(time * system) @ initial
    mean, second = value[block((1, 0))].sum(), value[block((2, 0))].sum()
    return second - mean ** 2


def killed(q, kappa, theta, a, pi):
    b = np.diag(kappa) - q
    u = np.linalg.solve((2 * np.diag(kappa) - q).T, theta * pi * a)
    inverse = np.linalg.inv(b)
    return u, b, inverse


rng = np.random.default_rng(71474)
chains = []
for n in range(2, 6):
    for _ in range(2):
        q = rng.uniform(.1, 1., (n, n))
        np.fill_diagonal(q, 0.)
        chains.append(q - np.diag(q.sum(axis=1)))
for n in (3, 5):
    chains.append(np.roll(np.eye(n), 1, axis=1) - np.eye(n))
chains.append(np.array([[-1., 1, 0], [0, -1., 1], [4, 0, -4.]]))
checks = controls = 0
maximum = ratio = 0.
for q in chains:
    n = len(q)
    kappa, theta = rng.uniform(.5, 3., n), rng.uniform(.2, 1.2)
    a = 2 * theta * kappa * rng.uniform(.1, .8, n)
    pi = stationary(q, kappa, theta, a)[0]
    reverse = (q.T * pi[None, :]) / pi[:, None]
    for speed in (.1, 1., 8., 64.):
        for qq in (speed * q, speed * reverse):
            rows = stationary(qq, kappa, theta, a)
            u, b, inverse = killed(qq, kappa, theta, a, pi)
            assert np.max(abs(rows[1] - theta * pi)) <= 2e-10
            assert np.max(abs(rows[2] - theta ** 2 * pi - u)) <= 2e-10
            rate, boundary = 2 * u @ inverse @ np.ones(n), -2 * u @ inverse @ inverse @ np.ones(n)
            for time in (.02, .3, 2., 12.):
                tail = 2 * u @ inverse @ inverse @ expm(-time * b) @ np.ones(n)
                exact = rate * time + boundary + tail
                moment = integrated_moments(qq, kappa, theta, a, time, rows)
                error = abs(exact - moment) / max(1., abs(exact), abs(moment))
                assert error <= 2e-9
                maximum = max(maximum, error)
                envelope = 2 * u.sum() / min(kappa) ** 2 * np.exp(-min(kappa) * time)
                uniform = theta * max(a) / min(kappa) ** 3 * np.exp(-min(kappa) * time)
                assert -2e-10 <= tail <= envelope + 2e-10 and envelope <= uniform + 2e-10
                ratio = max(ratio, tail / envelope)
                checks += 1
        # With common kappa and common theta, the whole covariance is blind to Q.
        common = np.full(n, max(kappa))
        target = theta * (pi @ a) / (2 * common[0])
        for qq in (speed * q, speed * reverse):
            u, b, _ = killed(qq, common, theta, a, pi)
            assert abs(u @ expm(-.7 * b) @ np.ones(n) - target * np.exp(-.7 * common[0])) <= 2e-10
            controls += 1
print(f"Independent moment comparisons and uniform tail bounds: {checks}; maximum scaled error={maximum:.3e}")
print(f"Maximum tail/envelope={ratio:.9f}; common-kappa blindness controls: {controls}")


def rational_variance_interval(q, kappa, a, time, terms=100):
    """Exact rational uniformization and geometric bounds on BOTH exponential tails."""
    n, one = q.rows, sp.ones(q.rows, 1)
    b, c = sp.diag(*kappa) - q, 2 * sp.diag(*kappa) - q
    u = (sp.ones(1, n) / n) * sp.diag(*a) * c.inv()  # theta=1, uniform pi
    inverse = b.inv()
    assert min(u) >= 0 and min(inverse) >= 0
    lam = max(b[i, i] for i in range(n))
    p, x = sp.eye(n) - b / lam, lam * time
    assert min(p) >= 0 and max(p * one) <= 1
    assert terms + 2 > x
    vector, total, scalar, term = one, one.copy(), sp.Integer(1), sp.Integer(1)
    for j in range(1, terms + 1):
        vector = p * vector
        term *= x / j
        total += term * vector
        scalar += term
    remainder = term * x / (terms + 1) / (1 - x / (terms + 2))
    # exp(x) is between scalar and scalar+remainder; killed exp is entrywise positive.
    lower = total / (scalar + remainder)
    upper = total / scalar + remainder / scalar * one
    base = (2 * u * (time * inverse - inverse ** 2) * one)[0]
    return [base + (2 * u * inverse ** 2 * v)[0] for v in (lower, upper)]


q = sp.Matrix([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
kappa, a = sp.Matrix([1, 2, 4]), sp.Matrix([1, 1, sp.Rational(9, 10)])
values = []
for qq in (q, q.T):
    inverse = (sp.diag(*kappa) - qq).inv()
    u = sp.ones(1, 3) / 3 * sp.diag(*a) * (2 * sp.diag(*kappa) - qq).inv()
    values.append((sum(u), (2 * u * inverse * sp.ones(3, 1))[0]))
variance_gap, rate_gap = [values[0][j] - values[1][j] for j in (0, 1)]
assert min(2 * kappa - a) == 1 and min(a) > 0
assert variance_gap == -sp.Rational(1, 2010) and rate_gap == sp.Rational(9, 9715)
print('Three-state witness: directed unit cycle; theta=1, kappa=(1,2,4), a=(1,1,9/10); minimum Feller margin=1')
print('Exact forward-minus-reverse Var(v)=-1/2010; long-run variance-rate gap=9/9715')
for time, sign in ((sp.Integer(1), -1), (sp.Rational(11, 10), 1)):
    forward = rational_variance_interval(q, kappa, a, time)
    reverse = rational_variance_interval(q.T, kappa, a, time)
    low, high = forward[0] - reverse[1], forward[1] - reverse[0]
    grid = sp.Integer(10) ** 12
    low, high = sp.floor(grid * low) / grid, sp.ceiling(grid * high) / grid
    assert low <= high and (high < 0 if sign < 0 else low > 0)
    kk, aa = np.array(kappa, dtype=float).ravel(), np.array(a, dtype=float).ravel()
    moment_values = []
    for qq in (np.array(q, dtype=float), np.array(q.T, dtype=float)):
        moment_values.append(integrated_moments(qq, kk, 1., aa, float(time), stationary(qq, kk, 1., aa)))
    assert float(low) - 2e-10 <= moment_values[0] - moment_values[1] <= float(high) + 2e-10
    print(f"Rational certified gap at T={time}: [{float(low):.12f}, {float(high):.12f}]")
print('Independent witness moment checks: 2')
print("PASS")
