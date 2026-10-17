"""Gamma renewal sampling: principal inversion and exact cycle aliases."""
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm, fractional_matrix_power
from scipy.special import gamma
from sympy import Matrix, eye, expand, sqrt


def kernel(q, shape, rate):
    return fractional_matrix_power(np.eye(len(q)) - q / rate, -shape)


def integrate_kernel(q, shape, rate):
    # u = z**(1/shape) removes the integrable Gamma density singularity.
    def integrand(z):
        u = z ** (1 / shape)
        if u > 700:  # Negligible Gamma tail on the tested shapes; avoid 0*expm(huge Q).
            return np.zeros_like(q)
        return np.exp(-u) * expm(u * q / rate) / (shape * gamma(shape))
    return quad_vec(integrand, 0, np.inf, epsabs=2e-10, epsrel=2e-10)[0]


def norm(a):
    return np.linalg.norm(a, np.inf)


rng = np.random.default_rng(71271)
chains = []
for size in range(3, 8):
    for _ in range(2):
        q = rng.uniform(.01, 1, (size, size))
        np.fill_diagonal(q, 0)
        q -= np.diag(q.sum(axis=1))
        chains.append(q)
for size in (3, 4, 7):
    chains.append(np.roll(np.eye(size), 1, axis=1) - np.eye(size))
chains.append(np.array([[-1., 1, 0], [0, -1, 1], [4, 0, -4]]))
counts = quadratures = 0
max_inverse = max_quad = 0.
for q in chains:
    for speed in (.1, 1., 10.):
        for rate in (.5, 2.):
            for shape in (.25, .5, 1., 1.5, 2.):
                r = kernel(speed * q, shape, rate)
                rebuilt = rate * (np.eye(len(q)) - fractional_matrix_power(r, -1 / shape))
                error = norm(rebuilt - speed * q) / max(1, norm(speed * q))
                assert error <= 2e-8
                assert norm(r.imag if np.iscomplexobj(r) else np.zeros_like(r)) <= 2e-10
                assert min(r.real.ravel()) >= -2e-10 and norm(r @ np.ones(len(q)) - 1) <= 2e-9
                if q is chains[-1]:
                    pi = np.array([4., 4., 1.]) / 9
                    center = np.eye(3) - np.ones((3, 1)) @ pi[None, :]
                    jordan = (q + 3 * np.eye(3)) @ center
                    base = 1 + 3 * speed / rate
                    explicit = np.eye(3) - center + base ** -shape * center
                    explicit += shape * speed / rate * base ** (-shape - 1) * jordan
                    assert norm(r - explicit) <= 2e-9
                max_inverse = max(max_inverse, error)
                counts += 1
                if speed == 1 and rate == 2 and any(q is c for c in chains[-4:]):
                    integrated = integrate_kernel(q, shape, rate)
                    difference = norm(r - integrated)
                    assert difference <= 2e-8
                    max_quad = max(max_quad, difference)
                    quadratures += 1
print(f"Principal inversions: {counts}; maximum scaled error={max_inverse:.3e}")
print(f"Independent Gamma-density quadratures: {quadratures}; maximum discrepancy={max_quad:.3e}")

# Exact radical arithmetic: at shape six and rate one, r=(1+sqrt(3))/2.
# Equality of A**6 and its transpose implies equality of their inverses.
cycle = Matrix(np.roll(np.eye(4, dtype=int), 1, axis=1))
a = eye(4) - (1 + sqrt(3)) / 2 * (cycle - eye(4))
power = eye(4)
for _ in range(6):
    power = (power * a).applyfunc(expand)
assert power == power.T
print("Exact shape-six four-cycle alias: radical identity PASS")

cycle = np.array(cycle, dtype=float)
print("shape  rate   cycle jump rate   generator gap   sampled-kernel gap")
aliases = 0
for shape in (4.1, 5., 6., 8., 12.):
    tangent = np.tan(np.pi / shape)
    for rate in (.5, 1., 2.):
        jump = rate * tangent / (1 - tangent)
        q = jump * (cycle - np.eye(4))
        # General powers are checked against the diagonalization-free integral.
        r = kernel(q, shape, rate)
        reverse = kernel(q.T, shape, rate)
        gap = norm(r - reverse)
        assert gap <= 2e-9
        assert norm(q - q.T) > .1
        if rate == 1 and shape in (5., 6.):
            integral = integrate_kernel(q, shape, rate)
            assert norm(r - integral) <= 2e-8
            assert norm(integral - integral.T) <= 2e-8
            print(f"{shape:4.0f} {rate:5.1f} {jump:17.10f} {norm(q-q.T):15.10f} {gap:.3e}")
        aliases += 1
print(f"Numerical alias cases: {aliases}; same label-path law for every fixed sample count")

# Global identification is not a uniform stability theorem.
pi = np.ones((4, 4)) / 4
print("shape-two fast-chain inverse conditioning (rate=1): m, kernel gap, generator gap")
for speed in (10., 100., 1000.):
    q1, q2 = speed * (pi - np.eye(4)), 2 * speed * (pi - np.eye(4))
    exact_gap = norm(np.eye(4) - pi) * ((1 + speed) ** -2 - (1 + 2 * speed) ** -2)
    assert abs(norm(kernel(q1, 2, 1) - kernel(q2, 2, 1)) - exact_gap) <= 2e-9
    print(f"{speed:5.0f} {exact_gap:.10e} {norm(q1-q2):.1f}")
print("PASS")
