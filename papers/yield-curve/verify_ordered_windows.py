"""Recover a finite generator from two ordered window-covariance matrices."""
import numpy as np
from scipy.linalg import expm, expm_frechet, null_space, sqrtm


def coordinates(q):
    n = len(q)
    pi = np.linalg.solve(q.T+np.ones((n, n)), np.ones(n))
    u = null_space(np.sqrt(pi)[None, :])/np.sqrt(pi)[:, None]
    return pi, u, -u.T @ np.diag(pi) @ q @ u


def observed(q, pi, u, t):
    # Independent piecewise Feynman--Kac mixed derivatives, not the H^2 formula.
    derivatives = [expm_frechet(t*q, t*np.diag(f), compute_expm=False) for f in u.T]
    left = np.array([pi @ d for d in derivatives])
    right = np.array([d @ np.ones(len(q)) for d in derivatives]).T
    means = t*(pi @ u)
    return left @ right-np.outer(means, means), left @ expm(t*q) @ right-np.outer(means, means)


def recover(j0, j1):
    e = np.linalg.solve(j0, j1)
    difference = np.eye(len(j0))-e
    squared = np.linalg.solve(j0.T, (difference @ difference).T).T
    root = sqrtm(squared)
    assert np.max(abs(np.imag(root))) < 1e-8*max(1., np.linalg.norm(root, 2))
    return np.real(root), e


def main():
    rng = np.random.default_rng(20261023)
    chains = []
    for n in range(2, 8):
        for _ in range(4):
            q = rng.uniform(.05, 2., (n, n)); np.fill_diagonal(q, 0.)
            q -= np.diag(q.sum(1)); chains.append(q)
    for n in (3, 4, 7):
        chains.append(np.roll(np.eye(n), 1, axis=1)-np.eye(n))
    projection = np.ones((3, 3))/3
    chains.append(projection-np.eye(3)+.1*np.outer([1, -1, 0], [1, 1, -2]))
    # The two five-state stationary-occupation-law-equivalent block generators.
    a = np.array([[-5., 1.], [2., -7.]])
    outgoing, incoming = -a.sum(1)/3, -a.sum(0)/3
    b = np.array([[-11., 4., 4.], [2., -10., 5.], [6., 3., -12.]])
    blind = [np.block([[a, np.outer(outgoing, np.ones(3))], [np.outer(np.ones(3), incoming), b]]),
             np.block([[a.T, np.outer(incoming, np.ones(3))], [np.outer(np.ones(3), outgoing), b]])]
    chains += blind
    identity_error = recovery_error = semigroup_error = 0.; cases = 0
    for q in chains:
        pi, u, l = coordinates(q); h = np.linalg.inv(l)
        for exposure in (.05, .3, 1.5, 6.):
            t = exposure/max(-np.diag(q)); e = expm(-t*l)
            difference = np.eye(len(l))-e
            formula = h @ h @ difference @ difference
            j0, j1 = observed(q, pi, u, t)
            scale = max(np.linalg.norm(formula, 2), np.linalg.norm(formula @ e, 2))
            err = max(np.linalg.norm(j0-formula, 2), np.linalg.norm(j1-formula @ e, 2))
            identity_error = max(identity_error, err/max(scale, np.finfo(float).tiny))
            assert err <= 1e-9*scale+8192*np.finfo(float).eps*max(1., t*t)
            rebuilt_l, rebuilt_e = recover(j0, j1)
            reconstructed = -u @ rebuilt_l @ u.T @ np.diag(pi)
            error = np.linalg.norm(reconstructed-q, 2)/max(1., np.linalg.norm(q, 2))
            recovery_error = max(recovery_error, error)
            semigroup_error = max(semigroup_error, np.linalg.norm(rebuilt_e-e, 2))
            assert error < 1e-7 and np.linalg.norm(rebuilt_e-e, 2) < 1e-8
            cases += 1
    # Direct numerical double integration of the ordered stationary correlation.
    nodes, weights = np.polynomial.legendre.leggauss(20)
    quadrature_error = 0.
    for q in (chains[0], chains[-3], blind[0]):
        pi, u, l = coordinates(q); t = .4/max(-np.diag(q))
        points, weights_t = (nodes+1)*t/2, weights*t/2
        j0 = np.zeros_like(l); j1 = np.zeros_like(l)
        for v, wv in zip(points, weights_t):
            for w, ww in zip(points, weights_t):
                j0 += wv*ww*(u.T @ np.diag(pi) @ expm((v+w)*q) @ u)
                j1 += wv*ww*(u.T @ np.diag(pi) @ expm((v+w+t)*q) @ u)
        observed0, observed1 = observed(q, pi, u, t)
        error = max(np.linalg.norm(j0-observed0, 2), np.linalg.norm(j1-observed1, 2))
        quadrature_error = max(quadrature_error, error/max(1., t*t))
        assert error < 8192*np.finfo(float).eps*max(1., t*t)
    # Exact analytic snapshot alias for a directed three-cycle and its reversal.
    cycle = np.array([[0., 1., -1.], [-1., 0., 1.], [1., -1., 0.]])
    centered = np.eye(3)-projection; forward = -1.5*centered+.5*cycle
    t = 2*np.pi/np.sqrt(3); epsilon = np.exp(-np.pi*np.sqrt(3))
    snapshots = [expm(t*q) for q in (forward, forward.T)]
    snapshot_gap = np.linalg.norm(snapshots[0]-snapshots[1], 2)
    assert snapshot_gap < 8192*np.finfo(float).eps
    assert np.linalg.norm(snapshots[0]-(projection-epsilon*centered), 2) < 8192*np.finfo(float).eps
    pi, u, _ = coordinates(forward)
    j0, j1 = observed(forward, pi, u, t); k0, k1 = observed(forward.T, pi, u, t)
    raw_gap = u @ (j0-k0) @ u.T @ np.diag(pi)
    target = (1+epsilon)**2*cycle/3
    assert np.linalg.norm(raw_gap-target, 2) < 8192*np.finfo(float).eps
    for q, first, second in ((forward, j0, j1), (forward.T, k0, k1)):
        rebuilt, _ = recover(first, second)
        assert np.linalg.norm(-u @ rebuilt @ u.T @ np.diag(pi)-q, 2) < 1e-9
    print(f"snapshot alias: T={t:.10f}, epsilon={epsilon:.10f}, snapshot gap={snapshot_gap:.3e}")
    print(f"ordered-window operator (1,2) difference={raw_gap[0, 1]:.10f}; exact expression=(1+epsilon)^2/3")
    pi, u, _ = coordinates(blind[0])
    window_gap = np.linalg.norm(observed(blind[0], pi, u, .1)[0]-observed(blind[1], pi, u, .1)[0], 2)
    assert window_gap > 1e-4
    print(f"five-state occupation-blind pair: adjacent-window matrix gap at T=.1={window_gap:.10f}")
    print(f"{len(chains)} chains, {cases} fixed-window reconstructions; maximum relative generator error={recovery_error:.3e}")
    print(f"maximum relative FK/closed identity error={identity_error:.3e}; semigroup error={semigroup_error:.3e}")
    print(f"three independent double-quadrature comparisons; maximum scaled error={quadrature_error:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
