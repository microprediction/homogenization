"""Short-time occupation covariance: shortest paths and explicit remainders."""
import math
import numpy as np
from scipy.linalg import expm
from scipy.sparse.csgraph import shortest_path
from verify_covariance_coercivity import moment_system, covariance

ROUND = 65536 * np.finfo(float).eps


def positive_moments(q, f, start, t):
    """Positive Taylor terms for exp(t*(B+aI)); physical occupation moments."""
    n, d = f.shape
    b = moment_system(q, f)
    a = max(-np.diag(q))
    positive = b+a*np.eye(len(b))
    assert np.min(positive) >= 0
    term = np.r_[np.eye(n)[start], np.zeros(len(b)-n)]
    total = term.copy()
    for order in range(1, 65):
        term = (t/order)*(positive @ term)
        total += term
    # Explicit norm tail for the positive exponential series.
    size = t*np.linalg.norm(positive, np.inf)
    tail = math.exp(size)*size**65/math.factorial(65)
    return total*math.exp(-a*t), tail


def main():
    cycle = np.roll(np.eye(5), 1, axis=1)
    path = np.diag([.7, 1.2, .9], 1)+np.diag([1.1, .8, .6], -1)
    diamond = np.array([[0, .6, .4, 0], [0, 0, .2, 1.], [.3, 0, 0, .8], [1., 0, 0, 0]])
    dense = np.array([[0, .8, .3], [.4, 0, 1.], [.7, .5, 0]])
    chains = [k-np.diag(k.sum(axis=1)) for k in (cycle, path, diamond, dense)]
    scalar_cases = matrix_cases = independent = 0
    max_utilization = max_identity = 0.
    rng = np.random.default_rng(2026101502)
    for q in chains:
        n = len(q)
        k = q-np.diag(np.diag(q))
        a = max(-np.diag(q))
        distances = shortest_path((k > 0).astype(float), directed=True, unweighted=True)
        for start in range(n):
            others = [j for j in range(n) if j != start]
            depths = distances[start, others].astype(int)
            f = np.eye(n)[:, others]
            coefficients = np.array([2*np.linalg.matrix_power(k, depth)[start, j]
                                     /math.factorial(depth+2) for j, depth in zip(others, depths)])
            observables = []
            for depth in sorted(set(depths)):
                g = rng.uniform(-1, 1, n)
                g[distances[start] < depth] = 0
                observables.append((depth, g, 2*(np.linalg.matrix_power(k, depth) @ (g*g))[start]
                                    /math.factorial(depth+2)))
            errors = []
            for t in (.0001, .0003, .001, .003, .01, .03, .1):
                value, tail = positive_moments(q, f, start, t)
                cov = covariance(value, n, n-1)
                scale = t**((depths+2)/2)
                scaled = cov/np.outer(scale, scale)
                assert 16*n*(1+t)**2*tail/min(scale)**2 <= ROUND
                errors.append(np.linalg.norm(scaled-np.diag(coefficients), 2))
                for j, depth in enumerate(depths):
                    remainder = (a**(depth+1)*(2/math.factorial(depth+2)+1/math.factorial(depth+1))*t
                                 + a**(2*depth)*t**depth/math.factorial(depth)**2)
                    discrepancy = abs(scaled[j, j]-coefficients[j])
                    assert discrepancy <= remainder+ROUND
                    max_utilization = max(max_utilization, discrepancy/remainder)
                    scalar_cases += 1
                    for h in range(j):
                        other = depths[h]
                        jumps = max(depth, other)+int(depth == other)
                        bound = t**2*((a*t)**jumps/math.factorial(jumps)
                                     +(a*t)**(depth+other)/(math.factorial(depth)*math.factorial(other)))
                        assert abs(scaled[j, h]) <= bound/(scale[j]*scale[h])+ROUND
                for depth, g, coefficient in observables:
                    measured = g[others] @ cov @ g[others]/t**(depth+2)
                    remainder = max(abs(g))**2*(a**(depth+1)
                                 *(2/math.factorial(depth+2)+1/math.factorial(depth+1))*t
                                 + a**(2*depth)*t**depth/math.factorial(depth)**2)
                    assert abs(measured-coefficient) <= remainder+ROUND*max(1, coefficient)
                    scalar_cases += 1
                matrix_cases += 1
                if t in (.03, .1):
                    b = moment_system(q, f)
                    initial = np.r_[np.eye(n)[start], np.zeros(len(b)-n)]
                    direct = covariance(expm(t*b) @ initial, n, n-1)/np.outer(scale, scale)
                    error = np.linalg.norm(direct-scaled, 2)
                    assert error <= 2e-8*max(1, np.linalg.norm(scaled, 2))
                    max_identity = max(max_identity, error)
                    independent += 1
            assert errors[0] <= .08*errors[-1]+ROUND
            assert np.linalg.eigvalsh(scaled)[0] > 0
    q = chains[0]
    t = .001
    value, _ = positive_moments(q, np.eye(5)[:, [4]], 0, t)
    ratio = covariance(value, 5, 1)[0, 0]/t**6
    print(f"scalar finite-remainder cases {scalar_cases}; scaled matrix cases {matrix_cases}")
    print(f"independent exponential cases {independent}; maximum scaled discrepancy {max_identity:.3e}")
    print(f"largest scalar remainder utilization {max_utilization:.9f}")
    print(f"five-cycle distance-four variance/t^6 at t=.001: {ratio:.12f}; limit {1/360:.12f}")
    print("PASS")


if __name__ == "__main__":
    main()
