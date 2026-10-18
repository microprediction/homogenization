"""Finite-rate endpoint interaction in a finite slow-fast Markov chain.

All norms are maximum absolute row sum. Requires numpy and scipy.
Floating matrix exponentials are regression checks, not interval enclosures.
"""
import math
import numpy as np
from scipy.linalg import expm


def norm(a):
    return np.linalg.norm(a, np.inf)


def bounds(t, speed, A, gap, ell):
    alpha = speed*gap
    z, u = math.exp(-alpha*t), A*ell/alpha
    h = A*z+u*(1-z)
    b = A*z+A*A*ell*t*z+u*u*(1-z-alpha*t*z)
    eta = A*z+A*ell*(2+ell*t)/alpha
    return h, u, b, eta


def model(rng, fast, slow, cycle=False, coupled=False):
    pi = np.ones(fast)/fast if cycle else rng.dirichlet(np.ones(fast))
    P = np.outer(np.ones(fast), pi)
    if cycle:
        Q = np.roll(np.eye(fast), 1, axis=1)-np.eye(fast)
        A, gap = math.sqrt(3), 1.5  # three-cycle, by its normal Fourier basis
    else:
        Q, A, gap = P-np.eye(fast), norm(np.eye(fast)-P), 1.
    blocks = rng.uniform(.02, .6, (fast, slow, slow))
    for B in blocks:
        np.fill_diagonal(B, 0)
        B -= np.diag(B.sum(axis=1))
    L = np.zeros((fast*slow, fast*slow))
    for x in range(slow):
        for y in range(slow):
            L[x*fast:(x+1)*fast, y*fast:(y+1)*fast] = np.diag(blocks[:, x, y])
    if coupled:
        J = rng.uniform(.01, .1, L.shape)
        np.fill_diagonal(J, 0)
        L += J-np.diag(J.sum(axis=1))
    bar = np.einsum('i,xiyj->xy', pi, L.reshape(slow, fast, slow, fast))
    return Q, pi, L, bar, A, gap


def uniformize(G, t):
    steps = max(1, math.ceil(-np.diag(G).min()*t))
    lam = -np.diag(G).min()
    W, term = np.eye(len(G))+G/lam, np.eye(len(G))
    U = term.copy()
    for k in range(1, 25):
        term = term @ W*(lam*t/steps)/k
        U += term
    return np.linalg.matrix_power(U*math.exp(-lam*t/steps), steps)


def semigroup_checks():
    rng = np.random.default_rng(20261018)
    count, worst = 0, 0.
    for fast, slow, cycle, coupled in ((2, 2, False, False), (3, 3, False, False),
                                     (3, 2, True, False), (2, 3, False, True)):
        Q, pi, L, bar, A, gap = model(rng, fast, slow, cycle, coupled)
        P = np.kron(np.eye(slow), np.outer(np.ones(fast), pi))
        C, ell = np.eye(len(P))-P, norm(L)
        for speed in (1., 4., 16., 64., 256., 1024.):
            for t in (.1, .4, 1.):
                G = speed*np.kron(np.eye(slow), Q)+L
                S, R = expm(t*G), np.kron(expm(t*bar), np.outer(np.ones(fast), pi))
                h, u, b, eta = bounds(t, speed, A, gap, ell)
                checks = ((norm(C@S), h), (norm(C@S@P), u),
                          (norm(P@S@C), u), (norm(C@S@C), b), (norm(S-R), eta))
                for actual, budget in checks:
                    assert actual <= budget+2e-11
                    worst = max(worst, actual/budget)
                for x in range(slow):
                    for y in range(slow):
                        K = S[x*fast:(x+1)*fast, y*fast:(y+1)*fast]
                        p = expm(t*bar)[x, y]
                        a, row = K.sum(axis=1), pi@K
                        D = K-np.outer(a, row)/row.sum()
                        assert norm(K/p-np.outer(np.ones(fast), pi)) <= eta/p+2e-11
                        assert norm(D/p) <= (b+u*u/row.sum())/p+2e-11
                        count += 1
                if speed == 64. and t == .4:
                    assert norm(S-uniformize(G, t)) < 2e-10
    print(f'{count} block interaction budgets; 360 semigroup inequalities: max ratio={worst:.6f}')
    print('Four independent uniformization checks: PASS')


def panel_check():
    Q, pi, L, bar, A, gap = model(np.random.default_rng(75), 3, 3)
    path = [0, 1, 2, 0, 2, 1, 0]*8
    errors, interactions = [], []
    for speed in (32., 64., 128., 256.):
        S = expm(.4*(speed*np.kron(np.eye(3), Q)+L))
        K = [S[3*x:3*x+3, 3*y:3*y+3] for x, y in zip(path[:-1], path[1:])]
        refs = [expm(.4*bar)[x, y] for x, y in zip(path[:-1], path[1:])]
        M = [k/p for k, p in zip(K, refs)]
        delta = max(norm(k-np.outer(np.ones(3), pi)) for k in M)
        gamma = max(norm(k-np.outer(k.sum(axis=1), pi@k)/(pi@k).sum()) for k in M)
        rho, xi = 2*delta/(1-delta), gamma/(1-delta)**2
        assert delta < 1/3
        for n, initial in enumerate((pi, np.eye(3)[0])):
            for r in range(3):
                nu, loggap = initial.copy(), 0.
                for j, k in enumerate(K):
                    start = max(0, j-r)
                    short = initial.copy() if start == 0 else pi.copy()
                    for old in K[start:j]:
                        short = short@old/(short@old).sum()
                    loggap += math.log((nu@k).sum()/(short@k).sum())
                    nu = nu@k/(nu@k).sum()
                budget = (len(K)-r-1)*rho*rho*min(rho, xi)**r/2
                assert abs(loggap) <= budget+2e-11
                if n == 0 and r == 1:
                    errors.append(abs(loggap))
        interactions.append(gamma)
        print(f'speed={speed:.0f}: interaction={gamma:.6e}, one-interval log gap={errors[-1]:.6e}')
    orders = np.log2(np.array([interactions[-2], errors[-2]])/[interactions[-1], errors[-1]])
    assert np.all(orders > [1.8, 3.6])
    print(f'Last-doubling interaction/log-gap orders: {orders[0]:.6f}, {orders[1]:.6f}')
    pstar = expm(.4*bar).min()
    delta = (A*math.exp(-1024*gap*.4)+A*norm(L)*(2+norm(L)*.4)/(1024*gap))/pstar
    assert delta < 1/3
    u = A*norm(L)/(1024*gap)
    Gamma = (A*(1+A*norm(L)*.4)*math.exp(-1024*gap*.4)+u*u*(1+1/(pstar*(1-delta))))/pstar
    S = expm(.4*(1024*np.kron(np.eye(3), Q)+L))
    for x in range(3):
        for y in range(3):
            M = S[3*x:3*x+3, 3*y:3*y+3]/expm(.4*bar)[x, y]
            D = M-np.outer(M.sum(axis=1), pi@M)/(pi@M).sum()
            assert norm(M-np.outer(np.ones(3), pi)) <= delta+2e-11
            assert norm(D) <= Gamma+2e-11
    print(f'Explicit sufficient relative-kernel budget at speed 1024: {delta:.6f}')
    print(f'Uniform interaction budget: {Gamma:.6e}; 24 panel budgets: PASS')


if __name__ == '__main__':
    semigroup_checks()
    panel_check()
    print('PASS')
