"""Finite-memory likelihood budgets near rank-one positive observation kernels.

Requires numpy and sympy; the diffusion regression uses the existing Fourier
kernel routine. Numerical quadrature is not an outward-rounded error proof.
"""
import itertools
import math
import numpy as np
import sympy as sp
from verify_filtered_likelihood import transition_kernels, averaged_densities


def update(nu, K):
    v = nu @ K
    return v/v.sum()


def logs(kernels, pi, initial, memory):
    nu = initial.copy()
    full, short = [], []
    for k, K in enumerate(kernels):
        start = max(0, k-memory)
        restarted = initial.copy() if start == 0 else pi.copy()
        for old in kernels[start:k]:
            restarted = update(restarted, old)
        full.append(math.log((nu @ K).sum()))
        short.append(math.log((restarted @ K).sum()))
        nu = update(nu, K)
    return np.array(full), np.array(short)


def sharpness():
    e = sp.symbols('e')
    pi, one = sp.ones(1, 2)/2, sp.ones(2, 1)
    P = sp.ones(2)/2
    A = sp.Matrix([[1, 1], [-1, 1]])/2
    M = P+e*A
    for r in range(5):
        S = [(pi*M**j*one)[0] for j in (r, r+1, r+2)]
        ratio = sp.factor(S[2]*S[0]/S[1]**2)
        assert sp.limit((ratio-1)/e**(r+2), e, 0) == -sp.Rational(1, 2**(r+2))
        if r == 1:
            assert sp.factor(ratio-(1-e**3/(8*(1+e)))) == 0
    print('Exact sharp orders r=0..4: leading log error = -(epsilon/2)^(r+2)')
    print('Two-transition witness: exact likelihood ratio = 1-epsilon^3/[8(1+epsilon)]')
    # All binary records: both the full and restarted products normalize.
    mats = [np.array(P+sign*sp.Rational(1, 8)*A, float)/2 for sign in (-1, 1)]
    p = np.array([.5, .5])
    worst = 0.
    for N in range(1, 8):
        for r in range(4):
            sums = np.zeros(2)
            for bits in itertools.product((0, 1), repeat=N):
                f, a = logs([mats[j] for j in bits], p, p, r)
                sums += np.exp([f.sum(), a.sum()])
            worst = max(worst, np.max(abs(sums-1)))
    assert worst < 2e-12
    print(f'28 exhaustive binary-panel normalizations: max error={worst:.3e}')


def budgets():
    rng = np.random.default_rng(20261018)
    count, worst = 0, 0.
    for size in (2, 3, 5):
        pi = np.ones(size)/size
        P = np.outer(np.ones(size), pi)
        for delta in (1/64, 1/32, 1/8):
            rho = 2*delta/(1-delta)
            E = rng.uniform(-1, 1, (128, size, size))
            E /= np.max(np.sum(abs(E), axis=2), axis=1)[:, None, None]
            kernels = (P+delta*E)/2
            assert kernels.min() > 0
            for N in (8, 32, 128):
                for initial in (pi, np.eye(size)[0]):
                    for r in range(4):
                        f, a = logs(kernels[:N], pi, initial, r)
                        bound = max(N-r-1, 0)*rho**(r+2)/2
                        error = abs(np.sum(f-a))
                        assert error <= bound+2e-12
                        assert np.max(abs(f[:r+1]-a[:r+1])) < 2e-12
                        worst = max(worst, error/bound)
                        count += 1
    print(f'{count} nonhomogeneous panel budgets: max error/bound={worst:.6f}')


def diffusion():
    path = np.array([.05, .0555, .047, .061, .052, .0445, .05])
    leading = averaged_densities(path)
    errors = []
    for rate in (40., 80., 160., 320.):
        kernels, pi = transition_kernels(path, rate)
        P = np.outer(np.ones(2), pi)
        delta = max(np.linalg.norm(K/p-P, np.inf) for K, p in zip(kernels, leading))
        assert delta < 1/3
        rho = 2*delta/(1-delta)
        row = []
        for r in (0, 1):
            f, a = logs(kernels*100, pi, pi, r)
            error = abs(np.sum(f-a))
            bound = (600-r-1)*rho**(r+2)/2
            assert error < bound+2e-10
            row.append(error)
        errors.append(row)
        print(f'rate={rate:.0f}, 600 transitions: delta={delta:.6f}, '
              f'log errors r=0,1: '+', '.join(f'{v:.3e}' for v in row))
    orders = np.log2(np.array(errors[-2])/np.array(errors[-1]))
    assert np.all(orders > np.array([1.8, 2.7]))
    print('Last-doubling diffusion orders: '+', '.join(f'{v:.6f}' for v in orders))
    refined, pi = transition_kernels(path, 320., nfreq=901, steps=1000)
    f, a = logs(refined*100, pi, pi, 1)
    change = abs(abs(np.sum(f-a))-errors[-1][1])
    assert change < 2e-10
    print(f'Refined fastest two-transition log-gap change={change:.3e}')


if __name__ == '__main__':
    sharpness()
    budgets()
    diffusion()
    print('PASS')
