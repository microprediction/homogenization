"""Sharp two-moment modal sets and reversible Green--Kubo matrix budgets."""
import numpy as np
import sympy as sp
from scipy.linalg import null_space
from scipy.optimize import brentq


def norm(a):
    return np.linalg.norm(a, np.inf)


def two_point(mu, variance, lower):
    delta = mu-lower
    upper = mu+variance/delta
    weights = np.array([variance, delta*delta])/(variance+delta*delta)
    return np.array([lower, upper]), weights


def transform(rate, scales, weights):
    return np.sum(weights/(1+scales*rate))


def recover(rho, scales, weights):
    upper = 1.
    while transform(upper, scales, weights) > rho:
        upper *= 2
    return brentq(lambda x: transform(x, scales, weights)-rho, 0, upper,
                  xtol=1e-12, rtol=1e-12)


mu, second, rho = sp.Integer(1), sp.Rational(13, 12), sp.Rational(1, 2)
variance, threshold = second-mu*mu, 1-mu*mu/second
lower = (1-rho)/(mu*rho)
upper = mu*(1-rho)/(mu*mu-second*(1-rho))
assert (threshold, lower, upper, 2/upper, 2/lower) == (sp.Rational(1, 13), 1, sp.Rational(12, 11), sp.Rational(11, 6), 2)
print("Exact example: E[tau]=1, E[tau^2]=13/6; mixing variance=1/12; threshold=1/13")
print("rho=1/2: exact modal set (1,12/11); unit-variance feature GK set (11/6,2)")
assert (1-sp.Rational(1, 20))/sp.Rational(1, 20) == 19
print("rho=1/20: exact modal set (19,infinity); rho=1/13 is also unbounded above")

# All endpoint approximants retain strictly positive finite component means.
endpoint_rates = {}
for rho in (.5, 1/13, 1/20):
    rates = []
    for side in ('lower', 'upper'):
        sequence = []
        for k in range(1, 9):
            ell = 10.**(-k) if side == 'lower' else 1-10.**(-k)
            scales, weights = two_point(1., 1/12, ell)
            assert abs(weights @ scales-1) <= 2e-9
            assert abs(weights @ (scales*scales)-13/12) <= 2e-9
            rate = recover(rho, scales, weights)
            assert rate > (1-rho)/rho-2e-9
            sequence.append(rate)
        rates.append(sequence)
    endpoint_rates[rho] = rates
    assert abs(rates[1][-1]-(1-rho)/rho) <= 3e-7
assert abs(endpoint_rates[.5][0][-1]-12/11) <= 2e-8
assert endpoint_rates[1/13][0][-1] > 3e4 and endpoint_rates[1/20][0][-1] > 5e7
print(f"Positive two-point endpoint checks: 48; rho=1/2 upper-end approximant={endpoint_rates[.5][0][-1]:.12f}")
print(f"At lower component mean 1e-8: rho=1/13 rate={endpoint_rates[1/13][0][-1]:.6f}; rho=1/20 rate={endpoint_rates[1/20][0][-1]:.6f}")

rng, cases, modal_checks = np.random.default_rng(29185), 0, 0
max_violation = 0.
for n in range(2, 8):
    for _ in range(2):
        pi = rng.dirichlet(np.ones(n)*2)
        c = rng.uniform(.1, 1, (n, n))
        c = (c+c.T)/2
        np.fill_diagonal(c, 0)
        q = c/pi[:, None]
        q -= np.diag(q.sum(axis=1))
        q /= max(-np.diag(q))
        root = np.sqrt(pi)
        p = np.outer(root, root)
        center, identity = np.eye(n)-p, np.eye(n)
        for speed in (.2, 1., 5.):
            a = -speed*root[:, None]*q/root[None, :]
            rates, vectors = np.linalg.eigh(a)
            h = np.linalg.inv(a+p)-p
            for _ in range(3):
                scales = rng.uniform(.1, 3, 4)
                weights = rng.dirichlet(np.ones(4))
                mean = weights @ scales
                second = weights @ (scales*scales)
                v = second-mean*mean
                r = sum(w*np.linalg.inv(identity+t*a) for t, w in zip(scales, weights))
                k = mean*r @ (np.linalg.inv(identity-r+p)-p)
                shifted = (k+k.T)/2-v/mean*center
                values, basis = np.linalg.eigh(shifted)
                positive = (basis*np.maximum(values, 0)) @ basis.T
                features = rng.normal(size=(n, 3))
                features -= (pi @ features)[None, :]
                f = root[:, None]*features
                tests = [k-h, h-positive, v/mean*center-(k-h),
                         2*f.T @ (k-h) @ f, 2*f.T @ (v/mean*center-(k-h)) @ f]
                for matrix in tests:
                    scale = max(1, norm(k), norm(h), norm(f)**2*norm(h))
                    violation = -np.linalg.eigvalsh((matrix+matrix.T)/2).min()/scale
                    max_violation = max(max_violation, violation)
                    assert violation <= 2e-9
                for j in range(1, n):
                    observed = vectors[:, j] @ r @ vectors[:, j]
                    lo = (1-observed)/(mean*observed)
                    assert lo <= rates[j]+2e-8*max(1, rates[j])
                    denominator = mean*mean-second*(1-observed)
                    if denominator > 0:
                        hi = mean*(1-observed)/denominator
                        assert rates[j] <= hi+2e-8*max(1, rates[j])
                    modal_checks += 1
                cases += 1
print(f"Reversible operator/feature-Gram cases: {cases}; modal bounds checked: {modal_checks}; max scaled PSD violation={max_violation:.3e}")
# Directed and defective generators: positive-resolvent error, without eigenvectors of Q.
directed = []
for n in range(2, 8):
    q = rng.uniform(.05, 1, (n, n))
    np.fill_diagonal(q, 0)
    q -= np.diag(q.sum(axis=1))
    directed.extend([q/max(-np.diag(q)), np.roll(np.eye(n), 1, axis=1)-np.eye(n)])
directed.append(np.array([[-1., 1, 0], [0, -1, 1], [4, 0, -4]]))
clocks = [([.5, 1.5], [.5, .5]), ([.1, .7, 2, 5], [.2, .1, .3, .4]), ([.7], [1.])]
representation_error = directed_violation = 0.
for q in directed:
    n, identity = len(q), np.eye(len(q))
    pi = np.linalg.lstsq(np.vstack([q.T, np.ones(n)]), np.r_[np.zeros(n), 1.], rcond=None)[0]
    root = np.sqrt(pi)
    p, center = np.outer(root, root), identity-np.outer(root, root)
    for speed in (.2, 1., 5.):
        a = -speed*root[:, None]*q/root[None, :]
        h = np.linalg.inv(a+p)-p
        for scales, weights in clocks:
            t, w = np.array(scales), np.array(weights)
            mean, second = w @ t, w @ (t*t)
            budget = second/mean-mean
            r = sum(ww*np.linalg.inv(identity+tt*a) for tt, ww in zip(t, w))
            delta = mean*r @ (np.linalg.inv(identity-r+p)-p)-h
            basis = null_space(np.sqrt(w)[None, :])
            b = basis.T @ np.diag(1/t) @ np.sqrt(w)
            c = basis.T @ np.diag(1/t) @ basis
            poles, vectors = np.linalg.eigh(c)
            residues = mean*(vectors.T @ b)**2/poles
            represented = sum((z*np.linalg.solve(a+x*identity, center) for z, x in zip(residues, poles)), np.zeros_like(a))
            scale = max(1, norm(h), budget)
            representation_error = max(representation_error, norm(delta-represented)/scale,
                                       abs(np.sum(residues/poles)-budget)/scale)
            symmetric = (delta+delta.T)/2
            violation = max(-np.linalg.eigvalsh(symmetric).min(),
                            -np.linalg.eigvalsh(budget*center-symmetric).min(),
                            np.linalg.norm(delta, 2)-budget)/scale
            directed_violation = max(directed_violation, violation)
assert max(representation_error, directed_violation) <= 2e-9
print(f"Directed/defective resolvent checks: 117; max scaled representation error={representation_error:.3e}; bound violation={directed_violation:.3e}")
q = sp.Matrix([[-1, 1, 0], [0, -1, 1], [1, 0, -1]])
i, p = sp.eye(3), sp.ones(3)/3
r = ((i-q/2).inv()+(i-3*q/2).inv())/2
h, k = (-q+p).inv()-p, r*((i-r+p).inv()-p)
d = k-h
assert (d+d.T)/2 == sp.Rational(17, 158)*(i-p)
assert (d-d.T)/2 == sp.Rational(3, 158)*(q-q.T) and d.T*d == (i-p)/79
f = sp.Matrix([1, 0, -1])
assert ((f.T*(h+h.T)*f)[0]/3, (f.T*(k+k.T)*f)[0]/3) == (sp.Rational(2, 3), sp.Rational(64, 79))
print("Exact directed three-cycle: T=(1/2,3/2) equally; mu=1, v=1/4; Sym(delta)=(17/158)*(I-P); norm(delta)=1/sqrt(79)")
print("Skew(delta)=(3/158)*(Q-Q.T) is nonzero; feature h=(1,0,-1): true GK=2/3, surrogate GK=64/79")
print("Moment-model consistency: E[tau^2]>=2*E[tau]^2; equality forces a single exponential clock")
print("PASS")
