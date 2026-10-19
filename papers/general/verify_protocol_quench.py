"""Fixed-dwell protocol matching; analytic bounds and independent matrix tests."""
from fractions import Fraction as F
import numpy as np
from numpy.linalg import norm, solve
from scipy.linalg import expm


def prepare(protocol):
    stages, elapsed, loga, b, previous = [], 0., 0., 0., None
    for Q, g, tau, mixing, gap in protocol:
        n = len(g)
        one = np.ones(n)
        pi = solve(Q.T + np.ones((n, n)), one)
        P = np.outer(one, pi)
        R = solve(Q + P, np.eye(n)) - P
        mean = pi @ g
        f = g - mean
        w = R @ f
        K = -pi @ (f * w)
        if previous is None:
            v = w.copy()
        else:
            b -= pi @ previous
            v = (np.eye(n) - P) @ (w - previous)
        z0 = R @ (-b * f + f * w)
        z1 = R @ (-(b + tau * K) * f + f * w)
        Z = max(norm(z0, np.inf), norm(z1, np.inf))
        core = 2 * Z + tau * (abs(K) * norm(w, np.inf) + norm(f, np.inf) * Z)
        core += norm(g, np.inf) * mixing * norm(v, np.inf) / gap
        stages.append(dict(Q=Q, g=g, tau=tau, pi=pi, mean=mean, w=w,
                           K=K, v=v, b=b, loga=loga, C=mixing, gap=gap,
                           core=core, Z=Z, start=elapsed))
        assert norm(pi @ R, np.inf) < 2e-11
        assert abs(pi @ v) < 2e-11
        previous = w
        b += tau * K
        elapsed += tau
        loga += tau * mean
    return stages


def composite(stage, x, eps):
    b = stage['b'] + x * stage['K']
    a = np.exp(stage['loga'] + stage['mean'] * x)
    return a * (1 + eps * (b - stage['w'])) + eps * np.exp(stage['loga']) * (
        expm(stage['Q'] * x / eps) @ stage['v'])


# Exact-rational bound for p_A=1/4,s_A=2,p_B=3/4,s_B=4,g=(1,0).
ZA, ZB = F(57, 512), F(75, 2048)
assert ZA == max(F(3, 4) * (b / 2 + F(1, 8)) for b in [F(0), F(3, 64)])
assert ZB == max(F(3, 4) * abs(b / 4 - F(1, 32)) for b in [F(19, 64), F(41, 128)])
bulkA = 2 * ZA + F(1, 2) * (F(3, 32) * F(3, 8) + F(3, 4) * ZA)
bulkB = 2 * ZB + F(1, 2) * (F(3, 64) * F(3, 16) + F(3, 4) * ZB)
constant = bulkA + bulkB + F(9, 32) + F(9, 128) + F(9, 16)
# Use 1/e<1 and e<3; the latter follows from the positive Taylor series.
assert sum((F(1, 1), F(1), F(1, 2), F(1, 6))) + F(1, 24) / (1 - F(1, 5)) < 3
assert 3 * constant < 4
assert F(1, 2) * (F(3, 32) + F(3, 64)) == F(9, 128)
assert F(9, 128) + F(1, 4) == F(41, 128)
assert F(9, 128) - F(1, 8) == -F(7, 128)
print('rational Z_A,Z_B =', ZA, ZB)
print('benchmark: (p_A,s_A)=(1/4,2), (p_B,s_B)=(3/4,4), g=(1,0), durations=(1/2,1/2)')
print('interfaces AB=1/4, BA=-1/8; mixing C=3/2, gaps=(2,4); uniform bound=4 eps^2')
print('rational constant majorant =', constant, 'and 3C =', 3 * constant, '< 4')
print('bulk coefficient = 9/128; AB = 41/128; BA = -7/128; difference = 3/8')


def reset(p, speed):
    pi = np.array([p, 1 - p])
    return speed * (np.outer(np.ones(2), pi) - np.eye(2))


A = (reset(.25, 2), np.array([1., 0.]), .5, 1.5, 2.)
B = (reset(.75, 4), np.array([1., 0.]), .5, 1.5, 4.)
protocols = [[A, B], [B, A], [A, B, A]]
N = np.outer([1., -1., 0.], [1., 1., -2.]) / 20
Qdef = np.ones((3, 3)) / 3 - np.eye(3) + N
assert norm(N @ N) < 1e-14
protocols.append([(Qdef, np.array([1., -.3, .2]), .4, 2., .7),
                  (Qdef.T, np.array([-.2, .4, .8]), .6, 2., .7)])
rng = np.random.default_rng(419)
for n in [2, 3, 4, 5]:
    for count in [2, 3, 4]:
        protocol = []
        for _ in range(count):
            rates = rng.uniform(.2, 1.8, (n, n))
            np.fill_diagonal(rates, 0)
            offmin = np.min(rates[~np.eye(n, dtype=bool)])
            Q = rates - np.diag(rates.sum(axis=1))
            # A common refresh to the uniform law at rate n*offmin couples
            # any two starts: ||exp(tQ)-P||_infinity <= 2 exp(-n*offmin*t).
            protocol.append((Q, rng.uniform(-1, 1, n), rng.uniform(.3, .8),
                             2., n * offmin))
        protocols.append(protocol)
max_ratio, max_continuity, tests = 0., 0., 0
for index, protocol in enumerate(protocols):
    current = prepare(protocol)
    T = sum(s['tau'] for s in current)
    H = max(norm(s['g'], np.inf) for s in current)
    C = sum(s['core'] for s in current)
    C += sum(s['C'] * norm(s['v'], np.inf) / (np.e * s['gap'] * s['tau'])
             for s in current[:-1])
    errors, omitted = [], []
    for eps in 2. ** -np.arange(1, 10):
        exact_start = np.ones(len(current[0]['g']))
        error = 0.
        for j, s in enumerate(current):
            xs = np.unique(np.r_[np.linspace(0, s['tau'], 21),
                                 np.minimum(s['tau'], eps * np.array([.1, .5, 1, 2, 4]))])
            if j:
                jump = composite(s, 0., eps) - composite(current[j-1], current[j-1]['tau'], eps)
                predicted = -eps * np.exp(current[j-1]['loga']) * (
                    expm(current[j-1]['Q'] * current[j-1]['tau'] / eps) @ current[j-1]['v'])
                max_continuity = max(max_continuity, norm(jump - predicted, np.inf))
            for x in xs:
                exact = expm((s['Q'] / eps + np.diag(s['g'])) * x) @ exact_start
                error = max(error, norm(exact - composite(s, x, eps), np.inf))
                tests += 1
            exact_start = expm((s['Q'] / eps + np.diag(s['g'])) * s['tau']) @ exact_start
        budget = eps**2 * np.exp(H * T) * C
        assert error <= budget + 2e-11 * np.exp(H * T)
        if index == 0:
            assert error <= 4 * eps**2 + 2e-11
        max_ratio = max(max_ratio, error / budget)
        errors.append(error)
        s = current[-1]
        bulk = sum(t['tau'] * t['K'] for t in current)
        outer = np.exp(s['loga'] + s['mean'] * s['tau']) * (1 + eps * (bulk - s['w']))
        omitted.append(abs(s['pi'] @ (exact_start - outer)))
    assert errors[-2] / errors[-1] > 3.8
    if index < 2:
        s, eps = current[-1], 2.**-9
        normalized = (s['pi'] @ exact_start / np.exp(.5) - 1) / eps
        expected = F(41, 128) if index == 0 else -F(7, 128)
        assert abs(normalized - float(expected)) < .002
        print(('AB' if index == 0 else 'BA'), 'scaled final coefficient =', normalized)
        print('uniform error =', errors[-1], 'last halving order =', np.log2(errors[-2] / errors[-1]))
        print('omitting interface: final readout error =', omitted[-1],
              'last halving order =', np.log2(omitted[-2] / omitted[-1]))
assert max_continuity < 2e-11
print('protocols =', len(protocols), 'sampled vector tests =', tests)
print('maximum error/budget =', max_ratio, 'matching identity residual =', max_continuity)
print('PASS')
