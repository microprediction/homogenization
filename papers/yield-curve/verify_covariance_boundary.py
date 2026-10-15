"""Classical Poisson covariance boundary, with explicit all-prior error bounds."""
import math
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import eigh, expm, null_space

ROUND = 65536 * np.finfo(float).eps


def data(q):
    n = len(q)
    pi = np.linalg.solve(q.T + np.ones((n, n)), np.ones(n))
    projection = np.ones((n, 1)) @ pi[None, :]
    h = np.linalg.inv(projection-q)-projection
    f = (np.eye(n)-projection) @ null_space(np.ones((1, n)))
    u, v = h @ f, h @ h @ f
    gram = f.T @ np.diag(pi) @ u
    gram = (gram+gram.T)/2
    t0 = 1.
    while True:
        contraction = np.linalg.norm(expm(t0*q)-projection, np.inf)
        if 0 < contraction < 1:
            break
        t0 *= 2
        assert t0 < 10000
    return pi, h, f, u, v, gram, 2/contraction, -math.log(contraction)/t0


def boundary(prior, pi, h, f, u, v):
    result = f.T @ np.diag(prior @ h) @ u
    result = result+result.T-f.T @ np.diag(pi) @ v-v.T @ np.diag(pi) @ f
    return result-np.outer(prior @ u, prior @ u)


def system(q, f):
    n, d = f.shape
    result = np.kron(np.eye(1+d+d*d), q.T)
    for a in range(d):
        result[(1+a)*n:(2+a)*n, :n] = np.diag(f[:, a])
        for b in range(d):
            row = 1+d+a*d+b
            result[row*n:(row+1)*n, (1+a)*n:(2+a)*n] += np.diag(f[:, b])
            result[row*n:(row+1)*n, (1+b)*n:(2+b)*n] += np.diag(f[:, a])
    return result


def moments(value, n, d):
    mean = value[n:(1+d)*n].reshape(d, n).sum(axis=1)
    second = value[(1+d)*n:].reshape(d, d, n).sum(axis=2)
    covariance = second-np.outer(mean, mean)
    return (covariance+covariance.T)/2, mean


def main():
    defective = np.array([[-8/15, 1/15, 7/15], [1/3, -2/3, 1/3], [1/5, 3/5, -4/5]])
    dense = np.array([[-1.1, .8, .3], [.4, -1.4, 1.], [.7, .5, -1.2]])
    cycle = np.roll(np.eye(4), 1, axis=1)-np.eye(4)
    chains = [np.array([[-1., 1.], [1., -1.]]), defective, dense, cycle]
    rng = np.random.default_rng(20261015)
    cases, quadratures = 0, 0
    largest_ratio, largest_identity = 0., 0.
    for q in chains:
        pi, h, f, u, v, gram, constant, decay = data(q)
        n, d = f.shape
        feature_covariance = f.T @ np.diag(pi) @ f
        base_system = system(q, f)
        priors = [*np.eye(n), pi, *rng.dirichlet(np.ones(n), size=2)]
        for prior in priors:
            b = boundary(prior, pi, h, f, u, v)
            mixture = sum(prior[i]*boundary(np.eye(n)[i], pi, h, f, u, v) for i in range(n))
            spread = u.T @ np.diag(prior) @ u-np.outer(prior @ u, prior @ u)
            assert np.linalg.norm(b-mixture-spread, 2) <= ROUND*max(1, np.linalg.norm(b, 2))
            for s in (.03, .1, .3, 1., 2., 4., 8., 16.):
                initial = np.r_[prior, np.zeros(base_system.shape[0]-n)]
                covariance, mean = moments(expm(s*base_system) @ initial, n, d)
                leading_error = covariance-2*s*gram
                corrected_error = leading_error-b
                first_bound = 5*constant**2/(decay**2*np.min(pi))
                second_bound = constant**3*(6+2*decay*s)*np.exp(-decay*s)/(decay**2*np.min(pi))
                budget = ROUND*max(1, s*s, np.linalg.norm(b, 2))/np.min(pi)
                first_norm = max(abs(eigh(leading_error, feature_covariance, eigvals_only=True)))
                second_norm = max(abs(eigh(corrected_error, feature_covariance, eigvals_only=True)))
                assert first_norm <= first_bound+budget
                assert second_norm <= second_bound+budget
                largest_ratio = max(largest_ratio, first_norm/first_bound)
                expected_mean = prior @ (np.eye(n)-expm(s*q)) @ u
                assert np.max(abs(mean-expected_mean)) <= ROUND*max(1, s)
                if s in (.3, 2.) and np.array_equal(prior, pi):
                    expected = 2*s*gram-f.T @ np.diag(pi) @ (np.eye(n)-expm(s*q)) @ v
                    expected -= v.T @ (np.eye(n)-expm(s*q.T)) @ np.diag(pi) @ f
                    error = np.linalg.norm(covariance-expected, 2)
                    assert error <= ROUND*max(1, s*s)
                    largest_identity = max(largest_identity, error)
                cases += 1
        prior = priors[-1]
        s = 1.3
        raw, quad_error = quad_vec(lambda t: (
            f.T @ np.diag(prior @ expm(t*q)) @ (np.eye(n)-expm((s-t)*q)) @ u
            + u.T @ (np.eye(n)-expm((s-t)*q.T)) @ np.diag(prior @ expm(t*q)) @ f), 0, s)
        mean = prior @ (np.eye(n)-expm(s*q)) @ u
        observed, _ = moments(expm(s*base_system) @ np.r_[prior, np.zeros(base_system.shape[0]-n)], n, d)
        error = np.linalg.norm(observed-(raw-np.outer(mean, mean)), 2)
        assert error <= ROUND*max(1, s*s)+4*quad_error
        largest_identity = max(largest_identity, error)
        quadratures += 1
    print(f"all-prior/horizon boundary and bound cases {cases}; independent moment quadratures {quadratures}")
    print(f"largest first-bound utilization {largest_ratio:.9f}; maximum identity discrepancy {largest_identity:.3e}")
    pi, h, _, _, _, _, _, _ = data(chains[0])
    scalar = np.array([[1.], [-1.]])
    stationary_b = boundary(pi, pi, h, scalar, h@scalar, h@h@scalar)[0, 0]
    deterministic_b = boundary(np.array([1., 0.]), pi, h, scalar, h@scalar, h@h@scalar)[0, 0]
    assert abs(stationary_b+.5) <= ROUND and abs(deterministic_b+.75) <= ROUND
    print(f"unit-flip scalar boundary: stationary {stationary_b:.9f}; deterministic {deterministic_b:.9f}")
    print("PASS")


if __name__ == "__main__":
    main()
