"""All-prior, all-horizon covariance coercivity from finite jump paths."""
import math
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import eigh, expm, null_space
from scipy.sparse.csgraph import shortest_path

ROUND = 65536 * np.finfo(float).eps


def setup(q):
    n = len(q)
    pi = np.linalg.solve(q.T + np.ones((n, n)), np.ones(n))
    rate = float(np.max(-np.diag(q)))
    positive = q[q > 0]
    ratio = float(np.min(positive) / rate)
    diameter = int(np.max(shortest_path((q > 0).astype(float), directed=True, unweighted=True)))
    constant = math.exp(-2) / 32 * ((ratio / 12) ** diameter / math.factorial(diameter)) ** 2
    features = (np.eye(n) - np.ones((n, 1)) @ pi[None, :]) @ null_space(np.ones((1, n)))
    return pi, rate, diameter, constant, features


def moment_system(q, f):
    n, d = f.shape
    blocks = 1 + d + d * d
    system = np.kron(np.eye(blocks), q.T)
    for a in range(d):
        system[(1+a)*n:(2+a)*n, :n] = np.diag(f[:, a])
        for b in range(d):
            row = 1 + d + a*d + b
            system[row*n:(row+1)*n, (1+a)*n:(2+a)*n] += np.diag(f[:, b])
            system[row*n:(row+1)*n, (1+b)*n:(2+b)*n] += np.diag(f[:, a])
    return system


def covariance(value, n, d):
    means = value[n:(1+d)*n].reshape(d, n).sum(axis=1)
    second = value[(1+d)*n:].reshape(d, d, n).sum(axis=2)
    result = second - np.outer(means, means)
    return (result + result.T) / 2


def main():
    dense = np.array([[0, .8, .3, .2], [.4, 0, 1, .2], [.7, .5, 0, .3], [.2, .4, .6, 0]])
    cycle = np.roll(np.eye(4), 1, axis=1)
    path = np.diag(np.ones(3), 1) + np.diag(np.ones(3), -1)
    defective = np.array([[-8/15, 1/15, 7/15], [1/3, -2/3, 1/3], [1/5, 3/5, -4/5]])
    chains = [np.array([[-1., 1.], [1., -1.]]), defective]
    chains += [a - np.diag(a.sum(axis=1)) for a in (dense, cycle, path)]
    rng = np.random.default_rng(20261015)
    cases, ode_cases, scaling_cases = 0, 0, 0
    smallest_ratio, largest_identity = math.inf, 0.
    for q in chains:
        pi, rate, diameter, constant, f = setup(q)
        n, d = f.shape
        gram = f.T @ np.diag(pi) @ f
        normalized_system = moment_system(q / rate, f)
        priors = [*np.eye(n), pi, *rng.dirichlet(np.ones(n), size=2)]
        print(f"states {n}, diameter {diameter}, scalar constant {constant:.9e}")
        for s in [.03, .1, .3, 1., 3., 10., 30.]:
            propagator = expm(s * normalized_system)
            floor = 4 * constant * min(s ** (diameter + 2), s) / s ** 2
            for prior in priors:
                initial = np.r_[prior, np.zeros(normalized_system.shape[0] - n)]
                cov = covariance(propagator @ initial, n, d)
                eigenvalues = eigh(cov / s ** 2, gram, eigvals_only=True)
                assert eigenvalues[0] > 0
                assert eigenvalues[0] >= floor - ROUND * max(1, np.max(abs(eigenvalues)))
                smallest_ratio = min(smallest_ratio, eigenvalues[0] / floor)
                cases += 1
            # Independently integrate in normalized time with O(1) moments.
            if s in (.1, 1., 10.):
                initial = np.r_[priors[0], np.zeros(normalized_system.shape[0] - n)]
                scaled = moment_system(s*q/rate, f)
                solution = solve_ivp(lambda _, z: scaled @ z, (0., 1.), initial,
                                     method="DOP853", rtol=2e-11, atol=2e-12)
                assert solution.success
                reference = covariance(solution.y[:, -1], n, d)
                observed = covariance(propagator @ initial, n, d) / s ** 2
                error = np.linalg.norm(reference-observed, 2)
                assert error <= 2e-9 * max(1, np.linalg.norm(reference, 2))
                largest_identity = max(largest_identity, error)
                ode_cases += 1
            if s in (.3, 3.):
                initial = np.r_[pi, np.zeros(normalized_system.shape[0] - n)]
                for speed in (.25, 8.):
                    duration = s / (speed*rate)
                    physical = covariance(expm(duration*moment_system(speed*q, f)) @ initial, n, d)
                    expected = covariance(propagator @ initial, n, d) / (speed*rate) ** 2
                    assert np.linalg.norm(physical-expected, 2) <= ROUND * max(1, duration**2)
                    scaling_cases += 1
    print(f"all-prior matrix-bound cases {cases}; smallest observed/bound eigenvalue ratio {smallest_ratio:.6f}")
    print(f"independent moment-ODE cases {ode_cases}; maximum normalized discrepancy {largest_identity:.3e}")
    print(f"physical speed/time scaling cases {scaling_cases}")
    # Deterministic-start two-state variance: a positive covariance integral.
    for s in (.1, .03, .01, .003, .001):
        value = 2 * quad(lambda y: quad(lambda x: np.exp(-2*s*(y-x))
                         * (-np.expm1(-4*s*x))/s, 0, y, epsabs=1e-10)[0],
                         0, 1, epsabs=1e-10)[0]
        assert 4/3 - 2*s - ROUND <= value <= 4/3 + ROUND
        print(f"two-state s={s:g}: deterministic-start variance/s^3 {value:.9f}")
    print("PASS")


if __name__ == "__main__":
    main()
