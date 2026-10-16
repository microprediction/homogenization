"""Two-state snapshot confidence: exact path laws and numerical inverse checks."""
from collections import defaultdict
from fractions import Fraction as F
from math import factorial, log, log1p, expm1
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import betaln


def mixture(v, c, u):
    return F(factorial(c)*factorial(v-c), factorial(v+1))/u**c/(1-u)**(v-c)


def interval(v, c, delta=.05):
    if not v:
        return 0., 1.
    k = log(2/delta)
    if not c:
        return 0., -expm1(-(k+log(v+1))/v)
    if c == v:
        return 1+expm1(-(k+log(v+1))/v), 1.
    def objective(u):
        return betaln(c+1, v-c+1)-c*log(u)-(v-c)*log1p(-u)-k
    mid = c/v
    return (brentq(objective, np.nextafter(0., 1.), mid),
            brentq(objective, mid, np.nextafter(1., 0.)))


def inverse(a, b, h=1.):
    s = a+b
    f = -log1p(-s)/(h*s)
    return np.array([[-a, a], [b, -b]])*f


# No independence assumption: enumerate full Markov path laws, aggregating counts.
means, crossings = 0, []
for a, b in [(F(1, 8), F(1, 4)), (F(1, 3), F(1, 3)), (F(2, 5), F(1, 2))]:
    for start in (0, 1):
        law = {(start, 0, 0, 0, 0): F(1)}
        alive = law.copy()
        for horizon in range(1, 15):
            def step(current, kill=False):
                out = defaultdict(F)
                for (i, v0, c0, v1, c1), mass in current.items():
                    for flip, prob in ((0, 1-(a, b)[i]), (1, (a, b)[i])):
                        v, c = [v0, v1], [c0, c1]
                        v[i] += 1
                        c[i] += flip
                        if kill and any(mixture(v[j], c[j], (a, b)[j]) > 10 for j in (0, 1)):
                            continue
                        out[(i ^ flip, v[0], c[0], v[1], c[1])] += mass*prob
                return out
            law, alive = step(law), step(alive, True)
            assert sum(law.values()) == 1
            for j in (0, 1):
                assert sum(mass*mixture(key[1+2*j], key[2+2*j], (a, b)[j])
                           for key, mass in law.items()) == 1
                means += 1
            failure = 1-sum(alive.values())
            assert failure <= F(1, 5)
            crossings.append(float(failure))
print(f"Exact full-path martingale means: {means}; first-crossing laws: {len(crossings)}; horizons 1-14")
print(f"Largest first-crossing probability, delta=.2: {max(crossings):.9f}")
assert F(1, 3)*((1-F(1, 4))+F(1, 4))/F(1, 3) == 1
print("At K=2, start=1: Pr(C1=1 | V1=1)=1 (not Binomial(1,a))")

# Independent matrix exponential reconstruction, with scales in physical rates.
err, cases = 0., 0
for q12 in (.001, .1, 1., 5.):
    for q21 in (.002, .3, 2.):
        q = np.array([[-q12, q12], [q21, -q21]])
        for h in (.01, .1, .5):
            p = expm(h*q)
            discrepancy = np.linalg.norm(inverse(p[0, 1], p[1, 0], h)-q)/max(1., np.linalg.norm(q))
            err = max(err, discrepancy)
            assert discrepancy < 256*np.finfo(float).eps*max(1., np.exp(h*(q12+q21)))
            cases += 1
print(f"Snapshot inverse: {cases} cases; max scaled error {err:.3e}")

# Numerical root residuals, including all-success, zero-success and unvisited rows.
residual = 0.
for v in (0, 1, 20, 1000, 1000000):
    for c in sorted({0, v//5, v//2, v}):
        lo, hi = interval(v, c)
        assert 0 <= lo <= hi <= 1 and (not v or lo <= c/v <= hi)
        for u in (lo, hi):
            if 0 < u < 1:
                value = betaln(c+1, v-c+1)-c*log(u)-(v-c)*log1p(-u)
                residual = max(residual, abs(value-log(40)))
assert residual < 2e-6*log(40)
print(f"Endpoint max log-mixture residual: {residual:.3e}")

la, ua = interval(600000, 120000)
lb, ub = interval(400000, 120000)
assert la > 0 and lb > 0 and ua+ub < 1
ql, qu, qc = inverse(la, lb), inverse(ua, ub), inverse(.2, .3)
pl, pu = lb/(ua+lb), ub/(la+ub)
corner_checks = 0
for a in np.linspace(la, ua, 21):
    for b in np.linspace(lb, ub, 21):
        q = inverse(a, b)
        assert ql[0, 1] <= q[0, 1] <= qu[0, 1]
        assert ql[1, 0] <= q[1, 0] <= qu[1, 0]
        assert pl <= b/(a+b) <= pu
        corner_checks += 1
radius = np.sqrt(2*sum(max(abs(qc[i, j]-ql[i, j]), abs(qc[i, j]-qu[i, j]))**2
                       for i, j in ((0, 1), (1, 0))))
print(f"Feasible record: h=1, V=(600000,400000), C=(120000,120000), delta=.05")
print(f"a=[{la:.9f},{ua:.9f}], b=[{lb:.9f},{ub:.9f}]")
print(f"q12=[{ql[0,1]:.9f},{qu[0,1]:.9f}], q21=[{ql[1,0]:.9f},{qu[1,0]:.9f}]")
print(f"pi1=[{pl:.9f},{pu:.9f}], generator Frobenius radius={radius:.9f}")
print(f"Monotone box checks: {corner_checks}; unvisited row interval={interval(0,0)}")
print("PASS")
