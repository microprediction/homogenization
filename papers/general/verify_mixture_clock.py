"""Global identification under known finite mixtures of exponential intervals."""
import numpy as np
import sympy as sp
from scipy.integrate import quad_vec
from scipy.linalg import expm, sqrtm


def kernel(q, rates, weights):
    eye = np.eye(len(q))
    return sum(w * rate * np.linalg.solve(rate * eye - q, eye)
               for rate, w in zip(rates, weights))


def invert_two(r, a, b, weight):
    eye = np.eye(len(r))
    inverse = np.linalg.solve(r, eye)
    c = weight * a + (1 - weight) * b
    d = c * inverse - (a + b) * eye
    delta = d @ d + 4 * a * b * (inverse - eye)
    return -(d + sqrtm(delta)) / 2


def integrated(q, rates, weights):
    scale = min(rates)
    def integrand(u):
        density = sum(w * rate / scale * np.exp(-rate * u / scale)
                      for rate, w in zip(rates, weights))
        if density == 0:
            return np.zeros_like(q)
        return density * expm(u * q / scale)
    return quad_vec(integrand, 0, np.inf, epsabs=2e-10, epsrel=2e-10)[0]


def norm(a):
    return np.linalg.norm(a, np.inf)


# Exact rational identities underlying the correct square-root branch.
z, a, b, c = sp.symbols("z a b c")
f = (a * b + c * z) / ((a + z) * (b + z))
d = c / f - a - b
s = 2 * z - d
k = a * b * (c - a) * (b - c) / c
assert sp.cancel(s - z - a * b / c - k / (a * b + c * z)) == 0
assert sp.cancel(d ** 2 + 4 * a * b * (1 / f - 1) - s ** 2) == 0
print("Exact rational square-root identities: PASS")

# Exact divided-difference signs for two different finite mixtures.
points = [sp.Integer(0), sp.Integer(2), sp.I, -sp.I, 1 + 2 * sp.I, 1 - 2 * sp.I, 3 + sp.I]
exact_signs = 0
for rates, weights in [([1, 4], [sp.Rational(1, 3), sp.Rational(2, 3)]),
                       ([1, 2, 5], [sp.Rational(1, 6), sp.Rational(1, 3), sp.Rational(1, 2)])]:
    for j, z in enumerate(points):
        fz = sum(w * rate / (rate + z) for rate, w in zip(rates, weights))
        for v in points[j + 1:]:
            fv = sum(w * rate / (rate + v) for rate, w in zip(rates, weights))
            divided = sum(w * rate / ((rate + z) * (rate + v)) for rate, w in zip(rates, weights))
            assert sp.cancel(fz - fv + (z - v) * divided) == 0
            assert sp.simplify(fz - fv) != 0
            if sp.im(z) > 0 and sp.im(v) > 0:
                assert sp.im(sp.expand_complex(divided)) < 0
            if sp.im(z) == 0 and sp.im(v) == 0:
                assert divided > 0
            exact_signs += 1
print(f"Exact finite-mixture divided-difference checks: {exact_signs}")

rng = np.random.default_rng(71371)
chains = []
for size in range(2, 8):
    for _ in range(2):
        q = rng.uniform(.01, 1, (size, size))
        np.fill_diagonal(q, 0)
        q -= np.diag(q.sum(axis=1))
        chains.append(q)
for size in (3, 4, 7):
    chains.append(np.roll(np.eye(size), 1, axis=1) - np.eye(size))
chains.append(np.array([[-1., 1, 0], [0, -1, 1], [4, 0, -4]]))
cases = quadratures = 0
max_recovery = max_quad = 0.
for index, q in enumerate(chains):
    for speed in (.1, 1., 10., 100.):
        for a, b, weight in ((.1, 2., .2), (.5, 5., .5), (1., 1.2, .8)):
            r = kernel(speed * q, [a, b], [weight, 1 - weight])
            rebuilt = invert_two(r, a, b, weight)
            error = norm(rebuilt - speed * q) / max(1, norm(speed * q))
            assert error <= 2e-8
            max_recovery = max(max_recovery, error)
            cases += 1
            if speed == 1 and index >= len(chains) - 4:
                difference = norm(r - integrated(q, [a, b], [weight, 1 - weight]))
                assert difference <= 2e-8
                max_quad = max(max_quad, difference)
                quadratures += 1
print(f"Two-rate matrix inversions: {cases}; maximum scaled error={max_recovery:.3e}")
print(f"Independent waiting-density quadratures: {quadratures}; maximum discrepancy={max_quad:.3e}")

cycle = sp.Matrix(np.roll(np.eye(4, dtype=int), 1, axis=1))
q_exact = cycle - sp.eye(4)
r_exact = (sp.eye(4) - q_exact).inv() / 3 + 8 * (4 * sp.eye(4) - q_exact).inv() / 3
gap = max(sum(abs(x) for x in row) for row in (r_exact - r_exact.T).tolist())
assert gap == sp.Rational(22, 65)
q = np.array(q_exact, dtype=float)
r = kernel(q, [1., 4.], [1 / 3, 2 / 3])
assert norm(r - np.array(r_exact, dtype=float)) <= 2e-10
assert norm(invert_two(r, 1., 4., 1 / 3) - q) <= 2e-9
print(f"Four-cycle witness: rates=(1,4), weight=1/3; generator gap={norm(q-q.T):.0f}; kernel gap=22/65={float(gap):.10f}")
print("PASS")
