"""Single-path ordered-window budgets: exact block-reward and FK cross-checks."""
from fractions import Fraction as F
from itertools import product

import numpy as np
from scipy.linalg import expm, expm_frechet, null_space


def geometric(n, alpha):
    c = [0]
    for _ in range(n):
        c.append(1+alpha*c[-1])
    return c, c[-1]**2+sum((1+x)**2 for x in c[:-1])


def dobrushin(p):
    return max(sum(abs(x-y) for x, y in zip(a, b))/2 for a in p for b in p)


def main():
    # Exact finite-state block rewards, with dependence between successive blocks.
    k = np.array([[F(1, 2), F(1, 3), F(1, 6)],
                  [F(1, 6), F(1, 2), F(1, 3)],
                  [F(1, 3), F(1, 6), F(1, 2)]], dtype=object)
    p = k @ k @ k; alpha = dobrushin(p); feature = [-1, 0, 1]
    paths = [[] for _ in range(3)]
    for start in range(3):
        for a, b, end in product(range(3), repeat=3):
            probability = k[start, a]*k[a, b]*k[b, end]
            paths[start].append((end, feature[start]*feature[a],
                                 feature[start]*feature[b], probability))
    n = 16; c, h = geometric(n, alpha); checks = 0; max_mgf = 0.
    for channel in (1, 2):
        g = np.array([sum(path[channel]*path[3] for path in row) for row in paths])
        values = [np.array([F(0)]*3)]
        for _ in range(n):
            values.append(g+p @ values[-1])
        for m in range(n+1):
            assert max(values[m])-min(values[m]) <= 2*c[m]
            if m < n:
                for row in paths:
                    possible = [path[channel]+values[m][path[0]] for path in row]
                    assert max(possible)-min(possible) <= 2*(1+c[m])
                    checks += 1
        distribution = {(state, 0): F(1, 3) for state in range(3)}
        for _ in range(n):
            updated = {}
            for (state, total), mass in distribution.items():
                for path in paths[state]:
                    key = path[0], total+path[channel]
                    updated[key] = updated.get(key, F(0))+mass*path[3]
            distribution = updated
        mean = n*sum(g)/3
        assert sum(distribution.values()) == 1
        assert sum(total*mass for (_, total), mass in distribution.items()) == mean
        for threshold in range(1, n+1):
            tail = sum(mass for (_, total), mass in distribution.items()
                       if abs(total-mean) >= threshold)
            upper = 2*np.exp(-threshold**2/(2*float(h)))
            assert float(tail) <= upper+64*np.finfo(float).eps
        for tilt in (-2., -1., -.5, .5, 1., 2.):
            scaled = sum(float(mass)*np.exp(tilt*float(total-mean)-tilt**2*float(h)/2)
                         for (_, total), mass in distribution.items())
            max_mgf = max(max_mgf, scaled)
            assert scaled <= 1+64*np.finfo(float).eps
    # Closed form is only a cross-check; the recurrence avoids cancellation.
    for a in (F(0), F(1, 8), F(9, 10), F(999, 1000)):
        for size in (1, 2, 16, 100):
            cs, hs = geometric(size, a); v = 1/(1-a)
            closed = cs[-1]**2+size*(1+v)**2-2*v*(1+v)*cs[-1]
            closed += v*v*(1-a**(2*size))/(1-a*a)
            assert closed == hs <= v*v+size*(1+v)**2
    print(f"exact dependent three-state block test: alpha={alpha}, N={n}")
    print(f"{checks} exact conditional-range checks; 32 tail and 12 MGF cross-checks")
    print(f"largest normalized MGF={max_mgf:.6f}; 16 exact geometric-sum identities")
    rng = np.random.default_rng(20261016); count = 0; worst = 0.
    for states in range(2, 7):
        for _ in range(4):
            q = rng.uniform(.1, 1., (states, states)); np.fill_diagonal(q, 0)
            rho = np.min(q[~np.eye(states, dtype=bool)])
            q -= np.diag(q.sum(1))
            pi = np.linalg.solve(q.T+np.ones_like(q), np.ones(states))
            u = null_space(np.sqrt(pi)[None, :])/np.sqrt(pi)[:, None]
            l = -u.T @ np.diag(pi) @ q @ u
            for t in (.1, .5, 2.):
                derivatives = [expm_frechet(t*q, t*np.diag(f), compute_expm=False) for f in u.T]
                left = np.array([pi @ d for d in derivatives])
                right = np.array([d @ np.ones(states) for d in derivatives]).T
                e = expm(-t*l); j0 = np.linalg.solve(l @ l, (np.eye(len(l))-e) @ (np.eye(len(l))-e))
                for observed, exact in ((left @ right, j0), (left @ expm(t*q) @ right, j0 @ e)):
                    error = np.linalg.norm(observed-exact, 'fro')
                    worst = max(worst, error)
                    assert error <= 8192*np.finfo(float).eps*max(1., np.linalg.norm(exact, 'fro'))
                assert dobrushin(expm(3*t*q)) <= np.exp(-3*states*rho*t)+1e-14
                count += 1
    # Design illustration evaluated at population data, not a simulated confidence claim.
    size = 1000000; t = np.log(2.); confidence = .05
    _, hn = geometric(size, .125)
    eta = t*t*np.sqrt(2*hn*np.log(4/confidence))/size
    a = 4/(1-4*eta); b = 4*a*eta; d = 1.5*a*eta
    delta = a*(d+d*d)+b/4
    radius = delta/(1+np.sqrt(1-delta))
    assert 4*eta < 1 and delta < 1 and 0 < radius < 1
    print(f"{count} continuous-time FK moment/reset-bound cases; maximum moment discrepancy={worst:.3e}")
    print(f"two-state design: T=log(2), alpha=1/8, N={size}, delta={confidence}, H_N={hn:.6f}")
    print(f"simultaneous covariance budget eta={eta:.9f}; population-data generator radius={radius:.9f}")
    print("budget requires known pi/features and an independently justified mixing bound; PASS is not empirical coverage evidence")
    print("PASS")


if __name__ == '__main__':
    main()
