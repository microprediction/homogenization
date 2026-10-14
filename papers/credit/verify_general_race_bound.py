"""All-speed competing-cause bounds, uniform over the initial prior."""

import numpy as np


ROUND_TOL = 4096*np.finfo(float).eps


def error(a, b):
    return np.linalg.norm(a-b)/max(1.0, np.linalg.norm(a), np.linalg.norm(b))


def check(q, pi, hazards, speed):
    n = len(pi)
    d = hazards.sum(axis=1)
    mean = pi @ d
    root = np.sqrt(pi)
    c = -root[:, None]*q/root[None, :]/np.sqrt(d[:, None]*d[None, :])
    zero = root*np.sqrt(d/mean)
    projection = np.outer(zero, zero)
    j = np.linalg.inv(c+projection)-projection
    # Columns encode every pure starting prior, in Euclidean coordinates for L2(pi).
    x = np.diag(1/np.sqrt(pi*d))-zero[:, None]/np.sqrt(mean)
    y = root[:, None]*hazards/np.sqrt(d[:, None])
    pooled = pi @ hazards/mean
    y -= zero[:, None]*np.sqrt(mean)*pooled
    contraction = np.linalg.inv(np.eye(n)+j/speed)
    coefficient = x.T @ j @ y
    remainder = -(j.T @ x).T @ contraction @ (j @ y)/speed**2
    constants = np.linalg.norm(j.T @ x, axis=0)[:, None]*np.linalg.norm(j @ y, axis=0)
    exact = np.linalg.solve(np.diag(d)-speed*q, hazards)
    residual = exact-pooled-coefficient/speed
    alpha = pi*d/mean
    h = np.linalg.inv(-q+np.outer(np.ones(n), pi))-np.outer(np.ones(n), pi)
    contrast = hazards-d[:, None]*pooled
    poisson_coefficient = (np.eye(n)-np.outer(np.ones(n), alpha)) @ h @ contrast
    max_error = max(error(residual, remainder), error(coefficient, poisson_coefficient),
                    error(alpha @ exact, pooled), np.max(abs(exact.sum(axis=1)-1)),
                    np.max(abs(coefficient.sum(axis=1))), np.max(abs(remainder.sum(axis=1))))
    assert np.linalg.norm(contraction, 2) <= 1+ROUND_TOL
    assert np.all(abs(residual) <= constants/speed**2+ROUND_TOL*np.maximum(1, abs(exact)))
    # A changing prior obeys the pure-prior envelope and the global max_i constant.
    prior = np.arange(1, n+1, dtype=float)**(1+speed/(1+speed))
    prior /= prior.sum()
    assert np.all(abs(prior @ residual) <= np.max(abs(residual), axis=0)+ROUND_TOL)
    assert np.all(abs(prior @ residual) <= constants.max(axis=0)/speed**2+ROUND_TOL)
    # Stationary contraction equals the oriented Green--Kubo formula already on the site.
    green_kubo = (pooled*((pi*d) @ h @ d)-(pi*d) @ h @ hazards)/mean
    max_error = max(max_error, error(pi @ coefficient, green_kubo))
    return max_error, exact[0], coefficient[0], constants.max(axis=0)


def main():
    rng = np.random.default_rng(2026101413)
    max_error = 0.0
    cases = 0
    for n in range(2, 9):
        for _ in range(6):
            q = rng.uniform(0.03, 2, (n, n))
            np.fill_diagonal(q, 0)
            np.fill_diagonal(q, -q.sum(axis=1))
            system = q.T.copy()
            system[-1] = 1
            target = np.zeros(n)
            target[-1] = 1
            pi = np.linalg.solve(system, target)
            hazards = rng.uniform(0.05, 3, (n, 3))
            for speed in np.geomspace(0.02, 100, 13):
                result = check(q, pi, hazards, speed)
                max_error = max(max_error, result[0])
                cases += 1
    # Irreducible defective generator: nonzero eigenvalue -3 has a Jordan block.
    q = np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]])
    assert np.linalg.matrix_rank(q+3*np.eye(3)) == 2
    pi = np.array([4., 4., 1.])/9
    hazards = np.array([[1., .5, .5], [.25, .5, .25], [2., 3., 5.]])
    for speed in [.02, 1., 100.]:
        max_error = max(max_error, check(q, pi, hazards, speed)[0])
    benchmark = check(q, pi, hazards, 1.)
    max_error = max(max_error, benchmark[0])
    # Constant total hazard and a zero cause: no division by a cause intensity is needed.
    boundary_hazards = np.array([[1., 1., 0.], [.5, 1.5, 0.], [1.75, .25, 0.]])
    for speed in [.02, 1., 100.]:
        max_error = max(max_error, check(q, pi, boundary_hazards, speed)[0])
    assert max_error <= ROUND_TOL
    print("general competing-cause finite-rate certificate")
    print("random chains 42; dimensions 2 through 8; causes 3; grid cases", cases)
    print("defective generator checked at speeds", [.02, 1., 100.])
    print("benchmark generator", q.tolist(), "stationary law", pi.tolist())
    print("benchmark hazards", hazards.tolist())
    print("constant-total-hazard and zero-cause boundary checks passed")
    print("benchmark pure-start probabilities", *(f"{x:.10f}" for x in benchmark[1]))
    print("benchmark pure-start first coefficients", *(f"{x:.10f}" for x in benchmark[2]))
    print("benchmark uniform-prior constants", *(f"{x:.10f}" for x in benchmark[3]))
    print("maximum normalized identity/simplex discrepancy", f"{max_error:.3e}")
    print("roundoff tolerance", f"{ROUND_TOL:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
