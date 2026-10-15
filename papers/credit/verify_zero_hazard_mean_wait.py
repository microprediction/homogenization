"""Zero-hazard calendar waits: centered bounds, speed order and reversible poles."""

import numpy as np
from scipy.linalg import null_space

from verify_nonreversible_mean_wait import calculate


ROUND_TOL = 65536 * np.finfo(float).eps


def check(q, pi, d, speed, reversible):
    n, mean = len(pi), pi @ d
    root = np.sqrt(pi)
    basis = null_space(root[None, :])
    l = basis.T @ (-root[:, None]*q/root[None, :]) @ basis
    f = basis.T @ (root*(d-mean))
    k = basis.T @ np.diag(d) @ basis - np.outer(f, f)/mean
    eigen, vectors = np.linalg.eigh(k)
    assert eigen[0] >= -ROUND_TOL*max(1., np.linalg.norm(k))
    square = (vectors*np.sqrt(np.maximum(0., eigen))) @ vectors.T
    h = np.linalg.inv(l)
    contraction = np.linalg.inv(np.eye(n-1)+square @ h @ square/speed)
    assert np.linalg.norm(contraction, 2) <= 1+ROUND_TOL
    coefficient = f @ h @ f/mean**2
    constant = np.linalg.norm(square @ h.T @ f)*np.linalg.norm(square @ h @ f)/mean**2
    centered = 1/mean+f @ np.linalg.solve(speed*l+k, f)/mean**2
    remainder = -(square @ h.T @ f) @ contraction @ (square @ h @ f)/(speed*mean)**2
    u = np.linalg.solve(np.diag(d)-speed*q, np.ones(n))
    wait = pi @ u
    residual = wait-1/mean-coefficient/speed
    scale = max(1., abs(wait), coefficient/speed, abs(remainder))
    error = max(abs(wait-centered), abs(residual-remainder))/scale
    assert abs(residual) <= constant/speed**2+ROUND_TOL*scale
    assert wait > 1/mean and coefficient > 0
    value, derivative, reversed_value, energy_error = calculate(q, pi, d, speed)
    assert derivative < 0 and value <= reversed_value+ROUND_TOL*max(1., abs(value))
    error = max(error, energy_error)
    palm = pi*d/mean
    assert abs(palm @ u-1/mean) <= ROUND_TOL*max(1., abs(wait))
    z, a = np.flatnonzero(d == 0), np.flatnonzero(d > 0)
    hz = np.linalg.inv(-q[np.ix_(z, z)])
    entrance = hz @ q[np.ix_(z, a)]
    trace = q[np.ix_(a, a)]+q[np.ix_(a, z)] @ entrance
    reward = np.ones(len(a))+q[np.ix_(a, z)] @ hz @ np.ones(len(z))
    ua = np.linalg.solve(np.diag(d[a])-speed*trace, reward)
    reconstructed = np.empty(n)
    reconstructed[a] = ua
    reconstructed[z] = hz @ np.ones(len(z))/speed+entrance @ ua
    error = max(error, np.linalg.norm(u-reconstructed)/max(1., np.linalg.norm(u)))
    pole = pi[z] @ hz @ np.ones(len(z))
    assert pole > 0 if len(z) else pole == 0
    assert np.max(abs(entrance.sum(axis=1)-1), initial=0.) <= ROUND_TOL
    # Exact next-term envelope, not a bare truncated-asymptotic comparison.
    gap = speed*wait-pole
    upper = speed*max(reward)/min(d[a])
    pole_budget = ROUND_TOL*max(1., abs(speed*wait), pole, upper)
    assert -pole_budget <= gap <= upper+pole_budget
    if reversible:
        rates, modes = np.linalg.eigh(l)
        inverse_root = (modes/np.sqrt(rates)) @ modes.T
        v = inverse_root @ k @ inverse_root
        g = inverse_root @ f/mean
        rho, modes = np.linalg.eigh(v)
        # rank(K)=|A|-1: its |Z| zero eigenvalues are known structurally.
        assert np.max(abs(rho[:len(z)]), initial=0.) <= ROUND_TOL*max(1., np.linalg.norm(v))
        rho[:len(z)] = 0.
        assert np.all(rho >= 0)
        weights = (modes.T @ g)**2
        spectral = 1/mean+np.sum(weights/(speed+rho))
        error = max(error, abs(wait-spectral)/scale,
                    abs(pole-sum(weights[:len(z)]))/max(1., pole))
        r = np.linalg.inv(np.diag(d)-speed*q)
        product = r @ np.ones(n)
        for order in range(1, 7):
            product = r @ q @ product
            direct = pi @ product  # derivative divided by order factorial
            expected = (-1)**order*np.sum(weights/(speed+rho)**(order+1))
            error = max(error, abs(direct-expected)/max(1., abs(direct), abs(expected)))
            assert (-1)**order*direct > 0
        partial = 1/mean
        for order in range(1, 5):
            moment = sum(weights*rho**(order-1))
            partial += (-1)**(order-1)*moment/speed**order
            signed = (-1)**order*(wait-partial)
            exact = sum(weights*(rho/speed)**order/(speed+rho))
            bound = sum(weights*rho**order)/speed**(order+1)
            budget = ROUND_TOL*max(1., abs(wait), abs(partial), abs(bound))
            assert -budget <= signed <= bound+budget and abs(signed-exact) <= budget
    return error, wait, coefficient, constant, pole


def main():
    rng = np.random.default_rng(20261015)
    worst, cases, zero_cases = 0., 0, 0
    for n in range(2, 9):
        for reversible in [False, True]:
            for active in [1, n-1, n]:
                for _ in range(3):
                    if reversible:
                        pi = rng.uniform(.3, 1., n)
                        pi /= sum(pi)
                        flow = rng.uniform(.05, .4, (n, n))
                        flow = (flow+flow.T)/2
                        q = flow/pi[:, None]
                    else:
                        q = rng.uniform(.05, 2., (n, n))
                    np.fill_diagonal(q, 0.)
                    np.fill_diagonal(q, -q.sum(axis=1))
                    if not reversible:
                        system = q.T.copy()
                        system[-1] = 1.
                        target = np.zeros(n)
                        target[-1] = 1.
                        pi = np.linalg.solve(system, target)
                    d = np.zeros(n)
                    d[rng.choice(n, active, replace=False)] = rng.uniform(.4, 4., active)
                    for speed in np.geomspace(.01, 100., 9):
                        error, *_ = check(q, pi, d, speed, reversible)
                        worst = max(worst, error)
                        cases += 1
                        zero_cases += active < n
    # Sparse directed cycles test inactive excursions beyond dense random designs.
    for n in [3, 4, 6]:
        q = np.roll(np.eye(n), 1, axis=1)-np.eye(n)
        d = np.zeros(n)
        d[0], d[-1] = 2., 3.
        for speed in [.01, 1., 100.]:
            error, *_ = check(q, np.ones(n)/n, d, speed, False)
            worst = max(worst, error)
            cases += 1
            zero_cases += 1
    q = np.array([[-1., 1.], [1., -1.]])
    error, wait, coefficient, constant, pole = check(q, np.ones(2)/2, np.array([0., 2.]), 1., True)
    assert abs(wait-1.5) <= ROUND_TOL and abs(coefficient-.5) <= ROUND_TOL
    assert abs(pole-.5) <= ROUND_TOL and constant <= ROUND_TOL
    assert max(worst, error) <= ROUND_TOL
    print("zero-hazard mean-wait certificate")
    print("chain-speed cases", cases, "dimensions 2 through 8; reversible orders 1 through 6")
    print("zero-hazard cases", zero_cases, "; remaining cases have strictly positive hazards")
    print(f"maximum normalized identity error {max(worst, error):.3e}; tolerance {ROUND_TOL:.3e}")
    print(f"two-state W(1)={wait:.10f}; fast coefficient={coefficient:.10f}; slow pole={pole:.10f}")
    print(f"two-state unsigned remainder constant={constant:.3e}; active-clock mean={1/2:.10f}")
    print("PASS")


if __name__ == "__main__":
    main()
