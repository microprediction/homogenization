"""Fast-switching curvature on the exact optimizer interval, for any prior."""
from fractions import Fraction as F
from math import factorial
import numpy as np
from numpy.linalg import norm, solve
from scipy.linalg import expm
from scipy.optimize import brentq


def constants(Q, b, s, gamma, T, C, gap):
    n = len(b)
    pi = solve(Q.T + np.ones((n, n)), np.ones(n))
    lo, hi = np.min(b / (gamma * s)), np.max(b / (gamma * s))
    fmax = max(norm((b - gamma*w*s) - pi @ (b - gamma*w*s), np.inf) for w in [lo, hi])
    candidates = [lo, hi]
    for i in range(n):
        for j in range(n):
            if s[i] != s[j]:
                w = (b[i] - b[j]) / (gamma * (s[i] - s[j]))
                if lo <= w <= hi:
                    candidates.append(w)
    spread = max(np.ptp(b*w - gamma*s*w*w/2) for w in candidates)
    B = 2*(1-gamma)*C*fmax**2*np.exp((1-gamma)*T*spread)/gap
    return pi, lo, hi, fmax, spread, B


def exact(Q, b, s, gamma, T, m, w, priors):
    n, p = len(b), 1-gamma
    q, h = b-gamma*w*s, b*w-gamma*w*w*s/2
    A = m*Q + p*np.diag(h)
    # Independent differentiated FK system and tilted path-moment system.
    D = np.kron(np.eye(3), A)
    D[n:2*n, :n] = p*np.diag(q)
    D[2*n:, :n] = -p*gamma*np.diag(s)
    D[2*n:, n:2*n] = 2*p*np.diag(q)
    U = (expm(T*D) @ np.r_[np.ones(n), np.zeros(2*n)]).reshape(3, n)
    M = np.kron(np.eye(4), A)
    M[n:2*n, :n] = np.diag(q)
    M[2*n:3*n, n:2*n] = 2*np.diag(q)
    M[3*n:, :n] = np.diag(s)
    V = (expm(T*M) @ np.r_[np.ones(n), np.zeros(3*n)]).reshape(4, n)
    z, dz, ddz = (priors @ U.T).T
    mass, er, er2, ev = (priors @ V.T).T
    variance = er2/mass - (er/mass)**2
    slope, curve = dz/z/(p*T), (ddz/z - (dz/z)**2)/(p*T)
    independent = -gamma*ev/mass/T + p*variance/T
    assert np.max(abs(curve-independent)/(1+abs(curve))) < 2e-10
    assert np.max(abs(slope-er/mass/T)/(1+abs(slope))) < 2e-10
    return np.log(z)/(p*T), slope, curve, variance


# Exact rational nonreversible, unequal-variance benchmark.
qrat = [[F(-3), F(2), F(1)], [F(1), F(-2), F(1)], [F(1,2), F(3,2), F(-2)]]
pirat = [F(5,24), F(11,24), F(1,3)]
brat, srat = [F(3,25), F(-2,25), F(1,20)], [F(1,25), F(9,100), F(1,16)]
assert all(sum(pirat[i]*qrat[i][j] for i in range(3)) == 0 for j in range(3))
L, U = -F(16,9), F(6)
def centered(w):
    v = [brat[i]-w*srat[i]/2 for i in range(3)]
    avg = sum(x*y for x, y in zip(pirat, v))
    return max(abs(x-avg) for x in v)
assert max(centered(L), centered(U)) == F(33,160)
candidates = [L, U]
for i in range(3):
    for j in range(i):
        w = 2*(brat[i]-brat[j])/(srat[i]-srat[j])
        if L <= w <= U:
            candidates.append(w)
def spread(w):
    values = [brat[i]*w-srat[i]*w*w/4 for i in range(3)]
    return max(values)-min(values)
assert max(map(spread, candidates)) == F(33,20)
x = F(33,40)
exp_upper = sum(x**k/F(factorial(k)) for k in range(13)) + x**13/F(factorial(13))/(1-x/14)
assert exp_upper < F(23,10)
assert F(363,128)*F(23,10) == F(8349,1280) < 8
kap = F(1,50) - F(363,6400)*F(23,10)/8
assert kap == F(1891,512000) > 0
print('benchmark Q =', qrat, 'pi =', pirat, 'b =', brat, 's =', srat)
print('gamma=1/2, T=1, C=2, gap=3/2; I=[-16/9,6], F*=33/160, D*=33/20')
print('exp(33/40)<23/10; threshold<8349/1280<8; curvature modulus at m>=8:', kap)

Q, b, s = np.array(qrat, float), np.array(brat, float), np.array(srat, float)
pi, lo, hi, fm, spreadmax, B = constants(Q, b, s, .5, 1., 2., 1.5)
priors = np.vstack([np.eye(3), pi])
w0 = F(24,169)
print('averaged fraction =', w0, 'numerical sufficient threshold =', B/.02)
for k, prior in enumerate(priors):
    evaluate = lambda w: exact(Q, b, s, .5, 1., 8., w, prior[None, :])
    root = brentq(lambda w: evaluate(w)[1][0], lo, hi, xtol=1e-12)
    value0, score, _, _ = evaluate(float(w0))
    distance, loss = abs(root-float(w0)), evaluate(root)[0][0]-value0[0]
    bound, lossbound = abs(score[0])/float(kap), score[0]**2/(2*float(kap))
    assert distance <= bound+2e-10 and loss <= lossbound+2e-10
    print('prior', k, 'optimizer =', root, 'distance / residual bound =', distance, bound,
          'CE loss / bound =', loss, lossbound)

rng, cases, utilization = np.random.default_rng(213), 0, 0.
models = [(Q, b, s, .5, 1., 2., 1.5)]
for n in [2, 3, 4, 5]:
    for _ in range(2):
        rates = rng.uniform(.2, 1.5, (n,n))
        np.fill_diagonal(rates, 0)
        gap = n*np.min(rates[~np.eye(n, dtype=bool)])
        generator = rates-np.diag(rates.sum(axis=1))
        models.append((generator, rng.uniform(-.1,.15,n), rng.uniform(.06,.12,n),
                       rng.uniform(.35,.8), rng.uniform(.25,1.3), 2., gap))
N = np.outer([1.,-1.,0.], [1.,1.,-2.])/20
assert norm(N@N) < 1e-14  # A nonzero nilpotent centered perturbation gives a defective generator.
models.append((np.ones((3,3))/3-np.eye(3)+N, b, s, .5, 1., 2., .7))
for Q, b, s, gamma, T, C, gap in models:
    pi, lo, hi, fm, dh, B = constants(Q,b,s,gamma,T,C,gap)
    priors = np.vstack([np.eye(len(b)), pi])
    threshold = B/(gamma*min(s))
    for m in [.25, 1.1, 4.]:
        m *= max(1., threshold)
        a = m*gap
        G = 1/a + np.expm1(-a*T)/(T*a*a)
        correction = 2*(1-gamma)*C*fm*fm*np.exp((1-gamma)*T*dh)*G
        for w in np.linspace(lo, hi, 9):
            _, _, curve, var = exact(Q,b,s,gamma,T,m,w,priors)
            bound = 2*C*T*fm*fm*np.exp((1-gamma)*T*dh)*G
            assert np.max(var) <= bound+2e-10*(1+bound)
            assert np.max(curve) <= -gamma*min(s)+correction+2e-10
            utilization = max(utilization, np.max(var)/bound)
            n = len(b)
            f = b-gamma*w*s
            f -= pi@f
            raw = np.kron(np.eye(3), m*Q)
            raw[n:2*n, :n], raw[2*n:, n:2*n] = np.diag(f), 2*np.diag(f)
            second = priors @ (expm(T*raw) @ np.r_[np.ones(n), np.zeros(2*n)])[2*n:]
            rawbound = 2*C*T*fm*fm*G
            assert max(second) <= rawbound+2e-10*(1+rawbound)
            if m > threshold:
                assert np.max(curve) < 0
            cases += len(priors)
    assert np.min(exact(Q,b,s,gamma,T,m,lo-1,priors)[1]) > 0
    assert np.max(exact(Q,b,s,gamma,T,m,hi+1,priors)[1]) < 0
common_s = np.array([.04, .09, .0625])
for m in [.1, 1., 8.]:
    for w in [1., 2., 3.]:
        _, score, _, _ = exact(models[0][0], common_s, common_s, .5, 1., m, w, np.eye(3))
        if w == 2:
            assert np.max(abs(score)) < 2e-10
        else:
            assert np.all(score*(2-w) > 0)
print('common regime fraction 2 is uniquely optimal at speeds .1,1,8 for every pure prior')
# Main-page audit: distinguish the averaged CE pair from its finite-speed values.
oldpi, oldb, olds = [F(3,4), F(1,4)], [F(3,25), -F(2,25)], [F(9,625), F(9,100)]
observed = F(3,100) + sum(v*x*x/y for v,x,y in zip(oldpi,oldb,olds))/6
blind = F(3,100) + sum(v*x for v,x in zip(oldpi,oldb))**2/(6*sum(v*y for v,y in zip(oldpi,olds)))
assert observed == F(853,5400) and F(545,10000) < blind < F(546,10000)
print('main averaged CE pair:', float(observed), float(blind),
      'rounded percentages:', f'{100*float(observed):.1f}%', f'{100*float(blind):.1f}%')
print('models =', len(models), 'prior/weight/speed cases =', cases,
      'largest tilted variance / bound =', utilization)
print('PASS')
