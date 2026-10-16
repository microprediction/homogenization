"""Deterministic a posteriori error bounds for ordered-window identification."""
from fractions import Fraction as F

import numpy as np
from scipy.linalg import expm, expm_frechet, null_space, sqrtm


def bound(j0, j1, candidate, eta0, eta1):
    inverse = np.linalg.inv(j0); e = inverse @ j1; f = np.eye(len(j0))-e
    z = f @ f @ inverse; k = np.linalg.norm(inverse, 2)
    mu = np.linalg.eigvalsh((candidate+candidate.T)/2)[0]
    if k*eta0 >= 1 or mu <= 0:
        return None
    alpha = k/(1-k*eta0); beta = alpha*k*eta0
    d = alpha*(eta1+eta0*np.linalg.norm(e, 2)); phi = np.linalg.norm(f, 2)
    residual = np.linalg.norm(candidate @ candidate-z, 'fro')
    delta = alpha*(2*phi*d+d*d)+phi*phi*beta+residual
    if delta >= mu*mu:
        return None
    radius = delta/(mu+np.sqrt(mu*mu-delta))
    return radius, delta, mu


def data(q, t):
    n = len(q); pi = np.linalg.solve(q.T+np.ones((n, n)), np.ones(n))
    u = null_space(np.sqrt(pi)[None, :])/np.sqrt(pi)[:, None]
    l = -u.T @ np.diag(pi) @ q @ u
    derivatives = [expm_frechet(t*q, t*np.diag(f), compute_expm=False) for f in u.T]
    left = np.array([pi @ d for d in derivatives])
    right = np.array([d @ np.ones(n) for d in derivatives]).T
    return pi, u, l, left @ right, left @ expm(t*q) @ right


def main():
    # End-to-end scalar certificate, entirely rational except the displayed radius.
    j0, j1, noisy1 = F(1, 4), F(1, 8), F(31, 250)
    mu, eta1 = F(126, 125), F(1, 1000)
    assert j0 == (1-F(1, 2))**2 and j1 == j0*F(1, 2)
    assert abs(noisy1-j1) == eta1
    k = 1/j0; e = noisy1/j0; phi = 1-e; d = k*eta1
    delta = k*(2*phi*d+d*d)
    assert mu*mu == phi*phi*k and delta == F(253, 15625) < mu*mu
    error = mu-1; lower, upper = F(1008, 125000), F(1613, 200000)
    assert lower == F(8064, 1000000) and upper == F(8065, 1000000)
    assert 2*mu*lower-lower*lower < delta < 2*mu*upper-upper*upper
    assert error == F(1, 125) < lower
    print("two-state T=log(2): true J0=1/4, J1=1/8; observed J0=1/4, J1=31/250")
    print(f"eta0=0, eta1={eta1}; candidate L={mu}; propagated delta={delta}")
    print("exact guaranteed radius in (0.008064,0.008065); actual weighted generator error=1/125=0.008")
    # The root-stage radius itself is attained in one dimension, not the full data bound.
    assert F(1)**2-F(3, 4) == F(1, 2)**2 and F(1)-F(1, 2) == F(1, 2)
    print("sharp root-stage example: candidate=1, true root=1/2, delta=3/4, radius=1/2")
    rng = np.random.default_rng(20261024); chains = []
    for n in range(2, 7):
        for _ in range(3):
            q = rng.uniform(.1, 1., (n, n)); np.fill_diagonal(q, 0.)
            q -= np.diag(q.sum(1)); chains.append(q)
    for n in (3, 5):
        chains.append(np.roll(np.eye(n), 1, axis=1)-np.eye(n))
    chains.append(np.ones((3, 3))/3-np.eye(3)+.1*np.outer([1, -1, 0], [1, 1, -2]))
    chains.append(np.array([[-.5, .5], [.5, -.5]]))
    accepted = rejected = 0; worst_ratio = worst_lift = 0.; negative = 0
    for q in chains:
        for exposure in (.1, 1., 4.):
            t = exposure/max(-np.diag(q)); pi, u, l, j0, j1 = data(q, t)
            scale = np.linalg.svd(j0, compute_uv=False)[-1]
            for fraction in (1e-6, 1e-5, 1e-4):
                noises = [rng.normal(size=j0.shape) for _ in range(2)]
                noises = [noise*(fraction*scale/np.linalg.norm(noise, 'fro')) for noise in noises]
                observed0, observed1 = j0+noises[0], j1+noises[1]
                inverse = np.linalg.inv(observed0); e = inverse @ observed1
                f = np.eye(len(l))-e; z = f @ f @ inverse
                root = sqrtm(z)
                assert np.max(abs(np.imag(root))) < 1e-8*max(1., np.linalg.norm(root, 2))
                for residual_fraction in (0., .01):
                    candidate = np.real(root)+residual_fraction*fraction*np.eye(len(l))
                    eta0, eta1 = (np.linalg.norm(noise, 'fro') for noise in noises)
                    result = bound(observed0, observed1, candidate, eta0, eta1)
                    if result is None:
                        rejected += 1; continue
                    radius, delta, mu = result
                    actual = np.linalg.norm(candidate-l, 'fro')
                    roundoff = 8192*np.finfo(float).eps*max(1., np.linalg.norm(l, 'fro'))
                    assert actual <= radius+roundoff
                    assert np.linalg.norm(l @ l-candidate @ candidate, 'fro') <= delta+roundoff
                    rebuilt = -u @ candidate @ u.T @ np.diag(pi)
                    lifted = np.sqrt(pi)[:, None]*(rebuilt-q)/np.sqrt(pi)[None, :]
                    worst_lift = max(worst_lift, abs(np.linalg.norm(lifted, 'fro')-actual))
                    assert abs(np.linalg.norm(lifted, 'fro')-actual) <= roundoff
                    entry_budget = radius*np.sqrt(pi[None, :]/pi[:, None])
                    assert np.all(abs(rebuilt-q) <= entry_budget+roundoff)
                    negative += int(np.min(rebuilt[~np.eye(len(q), dtype=bool)]) < -roundoff)
                    worst_ratio = max(worst_ratio, actual/radius); accepted += 1
    # A failed sufficient test is deliberately reported as inconclusive.
    assert bound(np.eye(2), np.zeros((2, 2)), np.eye(2), 1., 0.) is None
    assert bound(np.eye(2), np.zeros((2, 2)), np.eye(2), 0., 1.) is None
    assert accepted > 250
    print(f"{len(chains)} chains: {accepted} bound-tested deterministic perturbations; {rejected} inconclusive")
    print(f"maximum actual error/radius={worst_ratio:.6f}; weighted-lift discrepancy={worst_lift:.3e}")
    print(f"{negative} bound-tested candidates have a negative off-diagonal rate: the bound is not a realizability test")
    print("two explicit sufficient-test rejections checked")
    print("PASS")


if __name__ == "__main__":
    main()
