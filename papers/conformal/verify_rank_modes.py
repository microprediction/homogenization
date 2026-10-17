"""An exact two-mode obstruction to emission-blind final-gap calibration."""
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import minimize_scalar


score = sp.symbols('s', real=True)
signals = [sp.Matrix([1, -1, 0]), sp.Matrix([1, 1, -2])]
pi = sp.ones(1, 3) / 3
p0 = sp.ones(3) / 3
p1, p2 = signals[0] * signals[0].T / 2, signals[1] * signals[1].T / 6
q = -p1 - 2 * p2
assert q * sp.ones(3, 1) == sp.zeros(3, 1) and pi * q == sp.zeros(1, 3)
assert min(q[i, j] for i in range(3) for j in range(3) if i != j) == sp.Rational(1, 6)
assert q * signals[0] == -signals[0] and q * signals[1] == -2 * signals[1]
assert min(1 - abs(h) / 4 for signal in signals for h in signal) == sp.Rational(1, 2)

# Expand the rank event test <= max(calibration_0, calibration_1).
a, b, c = sp.symbols('a b c')
conditional = sp.integrate(sp.expand((1 + c * (2 * score - 1)) * (
    1 - (score + a * (score ** 2 - score)) * (score + b * (score ** 2 - score)))), (score, 0, 1))
expected = sp.Rational(2, 3) + (a + b) / 12 - c / 6 - a * b / 30 + (a * c + b * c) / 60
assert sp.expand(conditional - expected) == 0
print('Exact rank polynomial: 2/3+(a+b)/12-c/6-a*b/30+(a*c+b*c)/60; cubic coefficient=0')

x, y, z = sp.symbols('x y z', real=True)
pd, pg = p0 + x * p1 + x ** 2 * p2, p0 + y * p1 + z * p2
coverages = []
for signal in signals:
    total = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                kernel = conditional.subs({a: signal[i] / 4, b: signal[j] / 4, c: signal[k] / 4})
                total += pd[i, j] * pg[j, k] * kernel / 3
    coverages.append(sp.factor(total))
assert sp.simplify(coverages[0] - sp.Rational(2, 3) - ((1 + x) * y - 2 * x) / 1440) == 0
assert sp.simplify(coverages[1] - sp.Rational(2, 3) - ((1 + x ** 2) * z - 2 * x ** 2) / 480) == 0
print('Exact symbolic state-path checks: 54 conditional kernels; modes -1,-2; minimum off-diagonal rate=1/6; minimum score density=1/2')
print('At d=log(2): coverage biases (y-2/3)/960 and (z-2/5)/384, y=E exp(-G), z=E exp(-2G)')

root = (sp.sqrt(159) - 3) / 15
optimum = (13 - sp.sqrt(159)) / 14400
assert sp.simplify(15 * root ** 2 + 6 * root - 10) == 0
assert 0 < root < sp.Rational(2, 3) and root ** 2 > sp.Rational(2, 5)
assert sp.simplify((sp.Rational(2, 3) - root) / 960 - optimum) == 0
assert sp.simplify((root ** 2 - sp.Rational(2, 5)) / 384 - optimum) == 0
assert sp.Rational(4, 9) > sp.Rational(2, 5)  # Jensen forbids simultaneous exact cancellation.
print(f"Sharp minimax error=(13-sqrt(159))/14400={float(optimum):.12e}")
print(f"Attaining deterministic gap=-log((sqrt(159)-3)/15)={float(-sp.log(root)):.12f}")

qq, pp0 = np.array(q, dtype=float), np.ones((3, 3)) / 3
stationary = np.ones(3) / 3
def numerical_coverage(signal, calibration, following, initial=stationary):
    def integrand(t):
        cdf = t + signal * (t * t - t) / 4
        density = 1 + signal * (2 * t - 1) / 4
        return 1 - (initial * cdf) @ calibration @ (cdf * (following @ density))
    return quad(integrand, 0., 1., epsabs=1e-11, epsrel=1e-11)[0]


laws = [(expm(gap * qq), np.exp(-gap), np.exp(-2 * gap)) for gap in (0., .2, .8, 2.)]
weights, gaps = np.array([.2, .3, .5]), np.array([.1, .5, 2.])
laws.append((sum(w * expm(g * qq) for w, g in zip(weights, gaps)), weights @ np.exp(-gaps), weights @ np.exp(-2 * gaps)))
# Independent exponential waiting time: use its exact resolvent, not time discretization.
rate = .8
laws.append((rate * np.linalg.inv(rate * np.eye(3) - qq), rate / (rate + 1), rate / (rate + 2)))
checks, maximum = 0, 0.
for gap in (.05, .3, np.log(2), 1.2, 3.):
    calibration, decay = expm(gap * qq), np.exp(-gap)
    for following, first, second in laws:
        for j, signal in enumerate(signals):
            hh = np.array(signal, dtype=float).ravel()
            direct = numerical_coverage(hh, calibration, following)
            predicted = float(coverages[j].subs({x: decay, y: first, z: second}))
            error = abs(direct - predicted)
            assert error <= 2e-10
            maximum = max(maximum, error)
            checks += 1
for signal in signals:
    assert abs(numerical_coverage(np.array(signal, dtype=float).ravel(), pp0, pp0) - 2 / 3) <= 2e-10
print(f"Independent matrix-exponential/resolvent and quadrature checks: {checks}; maximum error={maximum:.3e}; iid controls: 2")

rng = np.random.default_rng(71475)
general_checks = 0
for n in range(3, 7):
    generator = rng.uniform(.1, 1., (n, n))
    np.fill_diagonal(generator, 0.)
    generator -= np.diag(generator.sum(axis=1))
    initial = np.linalg.lstsq(np.vstack([generator.T, np.ones(n)]), np.r_[np.zeros(n), 1.], rcond=None)[0]
    signal = rng.normal(size=n)
    signal -= initial @ signal
    signal /= max(abs(signal))
    for d in (.1, .7, 2.):
        calibration = expm(d * generator)
        for g in (.03, .5, 3.):
            following = expm(g * generator)
            pair = lambda transition: (initial * signal) @ transition @ signal
            predicted = 2 / 3 + (pair(following) + pair(calibration @ following) - 2 * pair(calibration)) / 960
            assert abs(numerical_coverage(signal, calibration, following, initial) - predicted) <= 2e-10
            general_checks += 1
print(f"General nonreversible-chain rank identity checks: {general_checks}")
for _ in range(2000):
    gaps = rng.exponential(1., rng.integers(2, 9))
    weights = rng.dirichlet(np.ones(len(gaps)))
    first, second = weights @ np.exp(-gaps), weights @ np.exp(-2 * gaps)
    assert second >= first ** 2 - 2e-10
    error = max(abs(first - 2 / 3) / 960, abs(second - 2 / 5) / 384)
    assert error >= float(optimum) - 2e-10
objective = lambda gap: max(abs(np.exp(-gap) - 2 / 3) / 960, abs(np.exp(-2 * gap) - 2 / 5) / 384)
fit = minimize_scalar(objective, bounds=(0., 2.), method='bounded', options={'xatol': 1e-12})
assert abs(fit.fun - float(optimum)) <= 2e-10
assert abs(fit.x - float(-sp.log(root))) <= 2e-6
print('Randomized-gap lower-bound checks: 2000; independent scalar minimax optimization agrees')
print("PASS")
