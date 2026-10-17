"""Continuous exponential-mixture clocks: infinite mean, kernel and inverse checks."""
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm, logm


def norm(a):
    return np.linalg.norm(a, np.inf)


def transform(z):
    return 1 - z * np.log1p(1 / z)


def derivative(z):
    return -np.log1p(1 / z) + 1 / (1 + z)


def inverse(value):
    z = complex((1 - value) / (2 * value))
    z = complex(max(z.real, 1e-8), z.imag)
    for _ in range(100):
        residual = transform(z) - value
        if abs(residual) <= 2e-13:
            return z
        step, fraction = residual / derivative(z), 1.
        for _ in range(50):
            candidate = z - fraction * step
            if candidate.real > 0 and abs(transform(candidate) - value) < abs(residual):
                break
            fraction /= 2
        else:
            raise AssertionError('No damped Newton step')
        z = candidate
    raise AssertionError('No scalar inverse convergence')


def projection(q):
    n = len(q)
    pi = np.linalg.lstsq(np.vstack([q.T, np.ones(n)]), np.r_[np.zeros(n), 1.], rcond=None)[0]
    return np.ones((n, 1)) @ pi[None, :]


def kernel(q):
    pi = projection(q)
    b = -q + pi  # Zero mode shifted to one; only the centered part sees the logarithm.
    return np.eye(len(q)) - b @ logm(np.eye(len(q)) + np.linalg.inv(b)) + np.log(2) * pi


def density(t):
    if t < 1e-4:
        return .5 - t / 3 + t * t / 8 - t ** 3 / 30
    return (-np.expm1(-t) - t * np.exp(-t)) / t ** 2


rng = np.random.default_rng(29173)
chains = []
for n in range(2, 7):
    for _ in range(2):
        q = rng.uniform(.05, 1., (n, n))
        np.fill_diagonal(q, 0)
        q -= np.diag(q.sum(axis=1))
        chains.append(q)
cycle = np.roll(np.eye(4), 1, axis=1) - np.eye(4)
chains.append(cycle)
checks = time_checks = 0
rate_error = inverse_error = time_error = 0.
for base in chains:
    normalized = base / max(-np.diag(base))
    for speed in (.1, 1., 5.):
        q = normalized * speed
        r, pi = kernel(q), projection(q)
        center, b = np.eye(len(q)) - pi, -q + pi
        direct = pi + quad_vec(lambda rate: rate * np.linalg.solve(rate * np.eye(len(q)) + b, center),
                               0, 1, epsabs=1e-11, epsrel=1e-11)[0]
        rate_error = max(rate_error, norm(r - direct))
        values, vectors = np.linalg.eig(r)
        roots = [0 if abs(value - 1) < 1e-8 else inverse(value) for value in values]
        recovered = -(vectors @ np.diag(roots) @ np.linalg.inv(vectors))
        inverse_error = max(inverse_error, norm(recovered - q) / max(1, norm(q)))
        assert min(r.real.ravel()) >= -2e-10 and norm(r @ np.ones(len(q)) - 1) <= 2e-9
        checks += 1
        if len(q) == 4 and speed == 1:
            timed = pi + quad_vec(lambda t: density(t) * (expm(t * q) - pi),
                                 0, np.inf, epsabs=1e-10, epsrel=1e-10)[0]
            time_error = max(time_error, norm(r - timed))
            time_checks += 1
assert rate_error <= 2e-8 and inverse_error <= 2e-8 and time_error <= 2e-8
print(f"Uniform-rate mixture kernels and spectral inversions: {checks}; max rate-integral error={rate_error:.3e}; max scaled inverse error={inverse_error:.3e}")
print(f"Independent waiting-density integrals: {time_checks}; max error={time_error:.3e}")

# A repeated real eigenvalue with a nontrivial Jordan block, without diagonalizing it.
q = np.array([[-1., 1, 0], [0, -1, 1], [4, 0, -4]])
jordan_error = 0.
for speed in (.1, 1., 5.):
    r = kernel(speed * q)
    pi = projection(r - np.eye(3))  # Stationary projector recovered from the observed kernel.
    center = np.eye(3) - pi
    repeated = (np.trace(r) - 1) / 2
    root = inverse(repeated).real
    recovered = -root * center - (r - pi - repeated * center) / derivative(root)
    jordan_error = max(jordan_error, norm(recovered - speed * q) / max(1, norm(speed * q)))
assert jordan_error <= 2e-8
print(f"Defective Jordan inversions: 3; maximum scaled error={jordan_error:.3e}")
gap = np.log(2.5) - 2 * np.arctan(1 / 3)
assert abs(norm(kernel(cycle) - kernel(cycle.T)) - gap) <= 2e-9
print(f"Four-cycle direction gap: log(5/2)-2*atan(1/3)={gap:.12f}")
print("Uniform rate on (0,1): survival=(1-exp(-t))/t; E[tau]=infinity; F(0)=1, (F(x)-1)/x=-log(1+1/x)->-infinity")
print("PASS")
