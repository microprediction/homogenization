"""Recover a finite generator and its unknown stationary law from raw windows."""
from fractions import Fraction as F
from itertools import product

import numpy as np
from scipy.linalg import expm, expm_frechet, sqrtm


def moments(q, t):
    n = len(q); pi = np.linalg.solve(q.T+np.ones((n, n)), np.ones(n))
    derivatives = [expm_frechet(t*q, t*np.diag(f), compute_expm=False) for f in np.eye(n)]
    left = np.array([pi @ d for d in derivatives])
    right = np.array([d @ np.ones(n) for d in derivatives]).T
    return pi, left @ right, left @ expm(t*q) @ right


def setup(a0, a1, t, rate):
    pi = a0.sum(1)/(t*t); n = len(pi)
    if min(pi) <= 0 or abs(sum(pi)-1) > 1e-10:
        return None
    a = np.linalg.inv(a0); e = a @ a1; f = np.eye(n)-e
    z = f @ f @ a @ np.diag(pi)+rate*rate*np.outer(np.ones(n), pi)
    weight = np.sqrt(pi)[:, None]/np.sqrt(pi)[None, :]
    return pi, z, weight


def inverse(a0, a1, t, rate):
    state = setup(a0, a1, t, rate)
    if state is None:
        return None
    pi, z, weight = state
    root = sqrtm(weight*z)
    assert np.max(abs(np.imag(root))) <= 1e-8*max(1., np.linalg.norm(root, 2))
    return pi, np.real(root), z, weight


def certificate(a0, a1, t, rate, candidate, eta0, eta1):
    state = setup(a0, a1, t, rate)
    if state is None:
        return None
    pi, z, weight = state; n = len(pi)
    a_hat = np.linalg.inv(a0); e = a_hat @ a1; f = np.eye(n)-e
    k = np.linalg.norm(a_hat, 2); mu = np.linalg.eigvalsh((candidate+candidate.T)/2)[0]
    if k*eta0 >= 1 or mu <= 0:
        return None
    a = k/(1-k*eta0); b = a*k*eta0
    d = a*(eta1+eta0*np.linalg.norm(e, 2)); phi = np.linalg.norm(f, 2)
    x = np.sqrt(n)*eta0/(t*t); pmax = min(1., max(pi)+x)
    gamma = pmax*(a*(2*phi*d+d*d)+phi*phi*b)+(phi*phi*k+rate*rate*np.sqrt(n))*x
    residual = np.linalg.norm(candidate @ candidate-weight*z, 'fro')
    delta = np.sqrt(max(pi)/min(pi))*gamma+residual
    if delta >= mu*mu:
        return None
    radius = delta/(mu+np.sqrt(mu*mu-delta))
    bound = radius+rate*x/np.sqrt(min(pi))
    qbar = rate*np.outer(np.ones(n), pi)-candidate/weight
    return qbar, bound, x, delta, radius


def main():
    # Exact squared row-sum bound, attained by a total-mass-preserving perturbation.
    eps = F(1, 1000); perturbation = [[eps/2, eps/2], [-eps/2, -eps/2]]
    eta_squared = sum(v*v for row in perturbation for v in row)
    pi_error_squared = sum(sum(row)**2 for row in perturbation)
    assert pi_error_squared == 2*eta_squared == F(1, 500000)
    print("exact row-sum bound: T=1, perturbation rows=(1,1)/2000 and (-1,-1)/2000")
    print("stationary-law squared error=2*eta0^2=1/500000; the row-sum factor is attained")
    # Nonnegative block products have range one, not the signed range two.
    k = np.array([[F(1, 2), F(1, 3), F(1, 6)], [F(1, 6), F(1, 2), F(1, 3)],
                  [F(1, 3), F(1, 6), F(1, 2)]], dtype=object)
    p = k @ k @ k
    alpha = max(sum(abs(a-b) for a, b in zip(x, y))/2 for x in p for y in p)
    range_checks = 0
    for lag in (1, 2):
        paths = [[(end, int(start == 0 and (a if lag == 1 else b) == 1),
                   k[start, a]*k[a, b]*k[b, end])
                  for a, b, end in product(range(3), repeat=3)] for start in range(3)]
        g = np.array([sum(reward*mass for _, reward, mass in row) for row in paths])
        value = np.array([F(0)]*3); c = F(0)
        for _ in range(16):
            assert max(value)-min(value) <= c
            for row in paths:
                possible = [reward+value[end] for end, reward, _ in row]
                assert max(possible)-min(possible) <= 1+c
                range_checks += 1
            value = g+p @ value; c = 1+alpha*c
    print(f"{range_checks} exact nonnegative-reward conditional-range checks; alpha={alpha}")
    rng = np.random.default_rng(20261026); chains = []
    for n in range(2, 7):
        for _ in range(3):
            q = rng.uniform(.1, 1., (n, n)); np.fill_diagonal(q, 0.)
            q -= np.diag(q.sum(1)); chains.append(q)
    for n in (3, 5):
        chains.append(np.roll(np.eye(n), 1, axis=1)-np.eye(n))
    chains.append(np.ones((3, 3))/3-np.eye(3)+.1*np.outer([1, -1, 0], [1, 1, -2]))
    chains.append(np.array([[-.5, .5], [.5, -.5]]))
    exact_count = accepted = rejected = 0; worst_exact = worst_ratio = worst_pi = 0.
    for q in chains:
        n = len(q); rate = max(-np.diag(q))
        for exposure in (.1, 1., 4.):
            t = exposure/rate; pi, a0, a1 = moments(q, t)
            recovered, root, z, weight = inverse(a0, a1, t, rate)
            qbar = rate*np.outer(np.ones(n), recovered)-root/weight
            error = np.linalg.norm(qbar-q, 'fro')/max(1., np.linalg.norm(q, 'fro'))
            worst_exact = max(worst_exact, error)
            assert error < 1e-7
            assert np.linalg.norm(recovered-pi) <= 8192*np.finfo(float).eps
            target = q @ q+rate*rate*np.outer(np.ones(n), pi)
            assert np.linalg.norm(z-target, 'fro') <= 32768*np.finfo(float).eps*max(1., np.linalg.norm(target, 'fro'))
            exact_count += 1
            scale = np.linalg.svd(a0, compute_uv=False)[-1]
            for fraction in (1e-6, 1e-5, 1e-4):
                noise0 = rng.normal(size=a0.shape); noise0 -= noise0.mean()
                noise1 = rng.normal(size=a1.shape); noise1 -= noise1.mean(1)[:, None]
                noise1 += noise0.sum(1)[:, None]/n
                factor = fraction*scale/max(np.linalg.norm(noise0), np.linalg.norm(noise1))
                noise0 *= factor; noise1 *= factor
                observed0, observed1 = a0+noise0, a1+noise1
                phat, root, _, weight = inverse(observed0, observed1, t, rate)
                for residual_fraction in (0., .01):
                    candidate = root+residual_fraction*fraction*rate*np.eye(n)
                    result = certificate(observed0, observed1, t, rate, candidate,
                                         np.linalg.norm(noise0), np.linalg.norm(noise1))
                    if result is None:
                        rejected += 1; continue
                    qbar, bound, x, delta, radius = result
                    actual = np.linalg.norm(weight*(qbar-q), 'fro')
                    roundoff = 32768*np.finfo(float).eps*max(1., np.linalg.norm(q, 'fro'))
                    assert actual <= bound+roundoff
                    assert np.linalg.norm(phat-pi) <= x+roundoff
                    assert np.all(abs(qbar-q) <= bound/weight+roundoff)
                    true_root = weight*(-q+rate*np.outer(np.ones(n), pi))
                    assert np.linalg.norm(candidate-true_root, 'fro') <= radius+roundoff
                    assert np.linalg.norm(candidate @ candidate-true_root @ true_root, 'fro') <= delta+roundoff
                    worst_ratio = max(worst_ratio, actual/bound); worst_pi = max(worst_pi, np.linalg.norm(phat-pi)/x)
                    accepted += 1
    print(f"{exact_count} exact-input reconstructions on {len(chains)} chains; max relative error={worst_exact:.3e}")
    print(f"{accepted} bound-tested perturbations; {rejected} inconclusive; max actual generator error/bound={worst_ratio:.6f}")
    print(f"maximum stationary-law error/budget={worst_pi:.6f}; nonzero residual and entrywise bounds checked")
    assert certificate(np.eye(2), np.eye(2), 1., 1., np.eye(2), 0., 0.) is None
    print("non-simplex row-sum input correctly rejected")
    print("PASS")


if __name__ == '__main__':
    main()
