"""Population mesh bias and a negative-rate trap for ordered state windows."""
from fractions import Fraction as F

import numpy as np
from scipy.linalg import expm, expm_frechet, logm, sqrtm


def stationary(q):
    return np.linalg.solve(q.T+np.ones_like(q), np.ones(len(q)))


def continuous(q, t, pi):
    derivatives = [expm_frechet(t*q, t*np.diag(f), compute_expm=False) for f in np.eye(len(q))]
    left = np.array([pi @ d for d in derivatives])
    right = np.array([d @ np.ones(len(q)) for d in derivatives]).T
    return left @ right, left @ expm(t*q) @ right


def grid(q, h, m, pi):
    p = expm(h*q); powers = [np.eye(len(q))]
    for _ in range(3*m):
        powers.append(powers[-1] @ p)
    # Direct lag-count sum, independent of the compact P S_m^2 identity.
    result = [np.diag(pi) @ sum((m-abs(j))*powers[(i+1)*m+j]
                               for j in range(1-m, m))*h*h for i in (0, 1)]
    s = sum(powers[:m]); compact = np.diag(pi) @ p @ s @ s*h*h
    return result, compact, p


def naive(a0, a1, t, shift):
    pi = a0.sum(1)/(t*t); projector = np.outer(np.ones(len(pi)), pi)
    inverse = np.linalg.inv(a0); e = inverse @ a1; f = np.eye(len(pi))-e
    z = f @ f @ inverse @ np.diag(pi)+shift*shift*projector
    root = sqrtm(z)
    assert np.max(abs(np.imag(root))) <= 1e-8*max(1., np.linalg.norm(root, np.inf))
    return shift*projector-np.real(root), e


def main():
    rng = np.random.default_rng(20261027); chains = []
    for n in range(2, 7):
        for _ in range(3):
            q = rng.uniform(.1, 1., (n, n)); np.fill_diagonal(q, 0.)
            q -= np.diag(q.sum(1)); chains.append(q)
    for n in (3, 5):
        chains.append(np.roll(np.eye(n), 1, axis=1)-np.eye(n))
    chains.append(np.ones((3, 3))/3-np.eye(3)+.1*np.outer([1, -1, 0], [1, 1, -2]))
    chains.append(np.array([[-.5, .5], [.5, -.5]]))
    count = 0; worst_identity = worst_root = worst_mesh = worst_bias = worst_corrected = 0.
    for q in chains:
        pi = stationary(q); a = max(-np.diag(q))
        for exposure in (.2, 1., 2.):
            t = exposure/a; exact = continuous(q, t, pi)
            for m in (1, 3, 8):
                h = t/m; (g0, g1), compact, p = grid(q, h, m, pi)
                roundoff = 32768*np.finfo(float).eps*max(1., t*t)
                identity_error = np.linalg.norm(g0-compact, 'fro')
                worst_identity = max(worst_identity, identity_error/max(1., t*t))
                assert identity_error <= roundoff
                assert np.linalg.norm(g1-g0 @ expm(t*q), 'fro') <= roundoff
                assert np.linalg.norm(g0.sum(1)/(t*t)-pi) <= 32768*np.finfo(float).eps
                mesh_bound = t*t*h*h*a*a/3
                for observed, target in zip((g0, g1), exact):
                    error = np.linalg.norm(observed-target, 'fro')
                    assert error <= mesh_bound+roundoff
                    worst_mesh = max(worst_mesh, error/mesh_bound)
                recovered, e = naive(g0, g1, t, a)
                predicted = (expm(h*q/2)-expm(-h*q/2))/h
                error = np.linalg.norm(recovered-predicted, np.inf)/max(1., np.linalg.norm(q, np.inf))
                worst_root = max(worst_root, error); assert error < 1e-7
                x = h*recovered/2
                corrected = 2*logm(x+sqrtm(np.eye(len(q))+x @ x))/h
                corrected_error = np.linalg.norm(corrected-q, np.inf)/max(1., np.linalg.norm(q, np.inf))
                worst_corrected = max(worst_corrected, corrected_error); assert corrected_error < 1e-7
                assert np.linalg.norm(e-expm(t*q), np.inf) <= 1e-7
                actual = np.linalg.norm(recovered-q, np.inf)
                bias_bound = h*h*a**3*np.cosh(h*a)/3
                assert actual <= 2*(np.sinh(h*a)-h*a)/h+roundoff*max(1., a)
                assert actual <= bias_bound+roundoff*max(1., a)
                worst_bias = max(worst_bias, actual/bias_bound); count += 1
    print(f"{count} grid cases on {len(chains)} chains: raw moment, mean and semigroup identities checked")
    print(f"maximum scaled compact-moment discrepancy={worst_identity:.3e}; naive-root discrepancy={worst_root:.3e}")
    print(f"maximum moment error/mesh bound={worst_mesh:.6f}; generator bias/bound={worst_bias:.6f}")
    print(f"maximum corrected-generator discrepancy={worst_corrected:.3e}")
    # Exact Taylor remainder enclosure for a directed three-cycle.
    c = np.roll(np.eye(3, dtype=int), 1, axis=1); q = c-np.eye(3, dtype=int)
    assert np.array_equal(q @ q @ q, 3*c-3*c @ c)
    reverse = tuple(np.argwhere(c @ c == 1)[0]); h = F(1, 2)
    assert (q @ q @ q)[reverse] == -3 and np.linalg.norm(q, np.inf) == 2
    # cosh(1/2) <= 1/(1-(1/2)^2/2)=8/7, using (2j)! >= 2^j.
    remainder = h**4*F(8, 7)/60
    lower, upper = -h*h/8-remainder, -h*h/8+remainder
    assert remainder == F(1, 840)
    assert lower == -F(109, 3360) and upper == -F(101, 3360) < -F(3, 100)
    predicted = (expm(float(h)*q/2)-expm(-float(h)*q/2))/float(h)
    assert float(lower) <= predicted[reverse] <= float(upper)
    fixed_h = []
    for m in (1, 4, 16):
        (g0, g1), _, _ = grid(q, float(h), m, np.ones(3)/3)
        fixed_h.append(naive(g0, g1, float(h)*m, 1.)[0])
    horizon_gap = max(np.linalg.norm(x-fixed_h[0], np.inf) for x in fixed_h)
    assert horizon_gap < 1e-7
    print(f"three-cycle h=1/2: spurious reverse rate in [{lower},{upper}], strictly below -0.03")
    print(f"computed reverse rate={predicted[reverse]:.10f}; fixed-h horizon discrepancy={horizon_gap:.3e}")
    q = np.array([[-.5, .5], [.5, -.5]]); pi = np.ones(2)/2
    target = continuous(q, 1., pi)[0]; errors = []
    for m in (8, 16, 32, 64, 128):
        h = 1/m; (g0, g1), _, _ = grid(q, h, m, pi)
        error = np.linalg.norm(g0-target, 'fro'); errors.append(error)
        assert abs(error/h**2-(1-np.exp(-1))**2/24) <= .01*h*h+1e-10
        qh = (expm(h*q/2)-expm(-h*q/2))/h
        assert abs(np.linalg.norm(qh-q, np.inf)-2*(np.sinh(h/2)-h/2)/h) <= 1e-10
    order = np.log2(errors[-2]/errors[-1]); assert 1.99 < order < 2.01
    print(f"two-state T=1: final moment-bias order={order:.6f}; nonzero h^-2 limit={(1-np.exp(-1))**2/24:.9f}")
    print("PASS")


if __name__ == '__main__':
    main()
