"""Exact all-maturity reversal profile for stationary switched CIR variance."""
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import expm


def invariant(q, theta, a, kappa):
    n = len(theta)
    pi = np.linalg.lstsq(np.vstack([q.T, np.ones(n)]), np.r_[np.zeros(n), 1.], rcond=None)[0]
    r = np.linalg.solve(kappa * np.eye(n) - q.T, kappa * pi * theta)
    s = np.linalg.solve(2 * kappa * np.eye(n) - q.T, r * (2 * kappa * theta + a))
    return pi, r, s


def moment_variance(q, theta, a, kappa, time, rows):
    # Integrate v-mu instead of v to avoid subtracting O(T**2) raw moments.
    n, mu = len(theta), rows[1].sum()
    pairs = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]
    where = {pair: j for j, pair in enumerate(pairs)}
    system, initial = np.zeros((6 * n, 6 * n)), np.zeros(6 * n)
    def block(pair):
        j = where[pair]
        return slice(j * n, (j + 1) * n)
    for zpower, vpower in pairs:
        target = block((zpower, vpower))
        system[target, target] = q.T - vpower * kappa * np.eye(n)
        if zpower:
            system[target, block((zpower - 1, vpower + 1))] += zpower * np.eye(n)
            system[target, block((zpower - 1, vpower))] -= zpower * mu * np.eye(n)
        if vpower:
            system[target, block((zpower, vpower - 1))] += np.diag(vpower * kappa * theta + vpower * (vpower - 1) * a / 2)
        if zpower == 0:
            initial[target] = rows[vpower]
    value = expm(time * system) @ initial
    mean, second = value[block((1, 0))].sum(), value[block((2, 0))].sum()
    return second - mean ** 2, max(1, abs(second), mean ** 2)


def profile(time, kappa):
    if kappa * time < .1:
        return 2 * quad(lambda u: (time - u) * np.exp(-kappa * u), 0, time)[0]
    return 2 * (kappa * time + np.expm1(-kappa * time)) / kappa ** 2


rng = np.random.default_rng(71472)
chains = []
for n in range(3, 7):
    for _ in range(2):
        q = rng.uniform(.05, 1, (n, n))
        np.fill_diagonal(q, 0)
        decay = n * q[~np.eye(n, dtype=bool)].min()
        q -= np.diag(q.sum(axis=1))
        # A common uniform reset at rate decay yields the total-variation envelope.
        chains.append((q, 2., decay))
for n in (3, 5, 7):
    chains.append((np.roll(np.eye(n), 1, axis=1) - np.eye(n), np.sqrt(n), 1 - np.cos(2 * np.pi / n)))
q = np.array([[-1., 1, 0], [0, -1., 1], [4, 0, -4.]])
center = np.eye(3) - np.ones((3, 1)) @ (np.array([4., 4., 1.]) / 9)[None, :]
jordan = (q + 3 * np.eye(3)) @ center
assert np.array_equal(q @ np.linalg.matrix_power(q + 3 * np.eye(3), 2), np.zeros((3, 3)))
chains.append((q, np.linalg.norm(center, np.inf) + np.linalg.norm(jordan, np.inf) / (1.5 * np.e), 1.5))
checks = rates = bounds = blind = 0
maximum = bound_ratio = 0.
for q, constant, decay in chains:
    n = len(q)
    theta = rng.uniform(.03, .1, n)
    for kappa in (.7, 2.):
        a = kappa * theta * rng.uniform(.1, .8, n)
        assert min(2 * kappa * theta - a) > 0
        pi, _, _ = invariant(q, theta, a, kappa)
        reverse = (q.T * pi[None, :]) / pi[:, None]
        f, g = theta - pi @ theta, a - pi @ a
        h = np.linalg.solve(np.ones((n, 1)) @ pi[None, :] - q, np.eye(n))
        leading = ((pi * f) @ h @ g - (pi * g) @ h @ f) / 2
        for speed in (.1, 1., 2., 4., 8., 16., 32.):
            resolvent = np.linalg.inv(kappa * np.eye(n) - speed * q)
            amplitude = ((pi * f) @ resolvent @ g - (pi * g) @ resolvent @ f) / 2
            rows = invariant(speed * q, theta, a, kappa)
            back = invariant(speed * reverse, theta, a, kappa)
            gap = rows[2].sum() - rows[1].sum() ** 2 - back[2].sum() + back[1].sum() ** 2
            assert abs(gap - amplitude) <= 2e-10
            bound = kappa * constant * max(abs(f)) * max(abs(g)) / (speed * decay) ** 2
            assert abs(amplitude - leading / speed) <= bound + 2e-10
            bound_ratio = max(bound_ratio, abs(amplitude - leading / speed) / bound)
            bounds += 1
            # Independent affine Poisson formula for the long-run rate.
            values = []
            for qq, rr in ((q, rows), (reverse, back)):
                pp, first, second = rr
                beta = np.linalg.solve(np.ones((n, 1)) @ pp[None, :] - speed * qq, f)
                values.append(2 * ((second - (pi @ theta) * first).sum() / kappa + (first - (pi @ theta) * pp) @ beta))
            assert abs(values[0] - values[1] - 2 * amplitude / kappa) <= 2e-10
            rates += 1
            for time in (0., .03, .3, 2., 20.):
                forward_value, scale = moment_variance(speed * q, theta, a, kappa, time, rows)
                reverse_value, other = moment_variance(speed * reverse, theta, a, kappa, time, back)
                error = abs(forward_value - reverse_value - profile(time, kappa) * amplitude) / max(scale, other)
                assert error <= 2e-9
                maximum = max(maximum, error)
                checks += 1
            affine = .1 * kappa * min(theta) + .3 * kappa * theta
            rf, rb = invariant(speed * q, theta, affine, kappa), invariant(speed * reverse, theta, affine, kappa)
            assert abs(rf[2].sum() - rb[2].sum()) <= 2e-10
            blind += 1
print(f"All-maturity profile mesh checks: {checks}; maximum scaled error={maximum:.3e}")
print(f"Independent long-run Poisson checks: {rates}; finite-rate bounds: {bounds}; maximum error/bound={bound_ratio:.9f}")
print(f"Affine-loading blindness controls: {blind}")

# Symbolic all-speed three-state witness, with strict positive Feller margins.
m = sp.symbols('m', positive=True)
q = sp.Matrix([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
theta = sp.Matrix([sp.Rational(3, 100), sp.Rational(4, 100), sp.Rational(5, 100)])
a = sp.Matrix([sp.Rational(2, 100), sp.Rational(4, 100), sp.Rational(1, 100)])
f, g = theta - sp.ones(3, 1) * sum(theta) / 3, a - sp.ones(3, 1) * sum(a) / 3
r = (sp.eye(3) - m * q).inv()
exact = sp.factor(((f.T * r * g)[0] - (g.T * r * f)[0]) / 6)
assert sp.simplify(exact + m / (12000 * (1 + 3 * m + 3 * m ** 2))) == 0
assert min(2 * theta - a) == sp.Rational(1, 25)
print('Three-cycle witness: kappa=1, theta=(.03,.04,.05), a=(.02,.04,.01); minimum Feller margin=1/25')
print('Exact amplitude: -m/[12000*(1+3*m+3*m**2)]; m*amplitude -> -1/36000')
gap = float(2 / sp.E * exact.subs(m, 1))
qq, tt, aa = np.array(q, dtype=float), np.array(theta, dtype=float).ravel(), np.array(a, dtype=float).ravel()
vf, sf = moment_variance(qq, tt, aa, 1., 1., invariant(qq, tt, aa, 1.))
vb, sb = moment_variance(qq.T, tt, aa, 1., 1., invariant(qq.T, tt, aa, 1.))
assert abs(vf - vb - gap) <= 2e-10 * max(sf, sb)
print(f"At m=T=1: variance gap={gap:.10e}; independent-return fourth-cumulant gap={3*gap:.10e}")
print("PASS")
