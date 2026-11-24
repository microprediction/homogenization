"""Alias-safe snapshot confidence: exact path laws and log error budgets."""
from collections import defaultdict
from fractions import Fraction as F
from functools import lru_cache
from itertools import permutations
from math import factorial, log, log1p, expm1
import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import brentq
from scipy.special import betaln


@lru_cache(None)
def mixture(v, c, u):
    return F(factorial(c)*factorial(v-c), factorial(v+1))/u**c/(1-u)**(v-c)


def interval(v, c, threshold):
    if not v:
        return 0., 1.
    k = log(threshold)
    if c == 0:
        return 0., -expm1(-(k+log(v+1))/v)
    if c == v:
        return np.exp(-(k+log(v+1))/v), 1.
    def objective(u):
        return betaln(c+1, v-c+1)-c*log(u)-(v-c)*log1p(-u)-k
    return (brentq(objective, np.nextafter(0., 1.), c/v),
            brentq(objective, c/v, np.nextafter(1., 0.)))


def norm(a):
    return np.linalg.norm(a, np.inf)


def log_certificate(phat, eta, h, m):
    n = len(phat)
    rho = norm(np.eye(n)-phat)
    d = np.linalg.det(phat)-n*eta
    if rho+eta >= 1 or d <= np.exp(-np.pi):
        return None
    x, power, series = np.eye(n)-phat, np.eye(n), np.zeros((n, n))
    for k in range(1, m+1):
        power = power@x
        series -= power/(h*k)
    tail = rho**(m+1)/(h*(m+1)*(1-rho))
    error = eta/(h*(1-rho-eta))+tail
    qbar = np.maximum(series, 0.)
    np.fill_diagonal(qbar, 0.)
    np.fill_diagonal(qbar, -qbar.sum(axis=1))
    return qbar, error+norm(qbar-series), d, rho, tail


# Exact full-filtration martingale means and simultaneous first-passage laws.
means, failures = 0, []
for pi in ((F(1, 2), F(9, 20), F(1, 20)), (F(1, 3),)*3):
    p = [[F(9, 10)*(i == j)+pi[j]/10 for j in range(3)] for i in range(3)]
    for start in range(3):
        law = {(start, (0,)*9): F(1)}
        alive = law.copy()
        for horizon in range(1, 8):
            def step(current, kill=False):
                out = defaultdict(F)
                for (i, counts), mass in current.items():
                    for j in range(3):
                        nxt = list(counts)
                        nxt[3*i+j] += 1
                        if kill and any(mixture(sum(nxt[3*r:3*r+3]), nxt[3*r+c], p[r][c]) > 45
                                        for r in range(3) for c in range(3)):
                            continue
                        out[(j, tuple(nxt))] += mass*p[i][j]
                return out
            law, alive = step(law), step(alive, True)
            assert sum(law.values()) == 1
            for i in range(3):
                for j in range(3):
                    assert sum(mass*mixture(sum(ct[3*i:3*i+3]), ct[3*i+j], p[i][j])
                               for (_, ct), mass in law.items()) == 1
                    means += 1
            failure = 1-sum(alive.values())
            assert failure <= F(1, 5)
            failures.append(float(failure))
print(f"Exact martingale means={means}; first-passage laws={len(failures)}, horizons 1-7")
print(f"Largest simultaneous crossing probability={max(failures):.9f}, delta=.2")


# Exact determinant perturbations: stochastic rows have Euclidean norm at most one.
def det(a):
    n = len(a)
    return sum((-1)**sum(t[i] > t[j] for i in range(n) for j in range(i+1, n))
               *np.prod([a[i][t[i]] for i in range(n)]) for t in permutations(range(n)))


rng = np.random.default_rng(63152)
checks = 0
for n in range(2, 6):
    for _ in range(12):
        def stochastic():
            rows = rng.integers(1, 20, (n, n))
            return [[F(int(x), int(sum(row))) for x in row] for row in rows]
        a, b = stochastic(), stochastic()
        eta = max(sum(abs(x-y) for x, y in zip(u, v)) for u, v in zip(a, b))
        assert abs(det(a)-det(b)) <= n*eta
        checks += 1
print(f"Exact rational determinant perturbation checks={checks}")


# Noncommuting perturbations; dense, sparse nonreversible, and defective Q.
cases, ratio, clip_cases, identity = 0, 0., 0, 0.
chains = [np.array([[-.25, .25, 0.], [0., -.25, .25], [1., 0., -1.]])]
jordan = chains[0]+.75*np.eye(3)
assert np.array_equal(chains[0]@jordan@jordan, np.zeros((3, 3))) and norm(chains[0]@jordan) > 0
for n in (3, 4, 5):
    cycle = np.roll(np.eye(n), 1, axis=1)-np.eye(n)
    dense = rng.uniform(.05, .4, (n, n))
    np.fill_diagonal(dense, 0.)
    np.fill_diagonal(dense, -dense.sum(axis=1))
    chains.extend((cycle, dense))
for q in chains:
    n = len(q)
    for h in (.03, .1, .2):
        p = expm(h*q)
        identity = max(identity, norm(logm(p).real/h-q)/max(1., norm(q)))
        for amplitude in (0., .0001, .001):
            alt = rng.uniform(0., 1., (n, n))
            alt /= alt.sum(axis=1)[:, None]
            phat = (1-amplitude)*p+amplitude*alt
            eta = norm(phat-p)
            result = log_certificate(phat, eta, h, 24)
            assert result is not None
            qb, budget, _, _, _ = result
            actual = norm(qb-q)
            slack = 1024*np.finfo(float).eps*max(1., norm(q), norm(qb))
            assert actual <= budget+slack
            assert np.max(abs(qb.sum(axis=1))) <= slack
            if budget > 1e-10:
                ratio = max(ratio, actual/budget)
            clip_cases += norm(qb-(logm(phat).real/h)) > 1e-8
            cases += 1
assert identity < 1e-10
print(f"Log/perturbation cases={cases} on {len(chains)} chains; max error/budget={ratio:.9f}")
print(f"Principal-log scaled identity residual={identity:.3e}; material adjustment cases={clip_cases}")


# Feasible balanced snapshot counts (Eulerian transition multigraph).
counts = np.array([[900000, 80000, 20000], [20000, 900000, 80000], [80000, 20000, 900000]])
assert np.array_equal(counts.sum(axis=0), counts.sum(axis=1)) and counts.min() > 0
phat, row_error, root_residual = counts/1000000, [], 0.
for i in range(3):
    bounds = [interval(1000000, int(counts[i, j]), 180) for j in range(3)]
    for j, endpoints in enumerate(bounds):
        c = int(counts[i, j])
        for u in endpoints:
            residual = betaln(c+1, 1000000-c+1)-c*log(u)-(1000000-c)*log1p(-u)-log(180)
            root_residual = max(root_residual, abs(residual))
    row_error.append(sum(max(abs(phat[i, j]-u) for u in bounds[j]) for j in range(3)))
assert root_residual < 2e-6*log(180)
eta = min(2., max(row_error))
qb, budget, d, rho, tail = log_certificate(phat, eta, .2, 24)
print(f"Counts per row=(900000,80000,20000) cyclically; V_i=1000000, h=.2, delta=.05, m=24")
print(f"eta={eta:.9f}, rho={rho:.9f}, determinant lower={d:.9f}, exp(-pi)={np.exp(-np.pi):.9f}")
print(f"Generator infinity-norm radius={budget:.9f}; series tail={tail:.3e}")
print(f"Endpoint maximum log-mixture residual={root_residual:.3e}")


# Existing snapshot aliasing survives outside the reporting gate.
c = np.roll(np.eye(3), 1, axis=1)
q1, q2 = 4*np.pi/np.sqrt(3)*(c-np.eye(3)), 4*np.pi/np.sqrt(3)*(c.T-np.eye(3))
p1, p2 = expm(q1), expm(q2)
assert norm(p1-p2) < 256*np.finfo(float).eps
assert log_certificate(p1, 0., 1., 24) is None
print(f"Aliased three-cycle snapshot gap={norm(p1-p2):.3e}; gate abstains")
print("PASS")
