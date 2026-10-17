"""Exact mean-preserving clock/generator ambiguity, not a simulation claim."""
import numpy as np
import sympy as sp
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.stats import poisson


def norm(a):
    return np.linalg.norm(a, np.inf)


def alternative(q, a):
    identity = np.eye(len(q))
    return (1 - a*a) * q + a*a * (np.linalg.inv(identity - q) - identity)


def mixture(q, a):
    identity = np.eye(len(q))
    return (np.linalg.inv(identity - (1-a)*q) + np.linalg.inv(identity - (1+a)*q)) / 2


# Rational certificate: same physical mean and label kernel, different graphs and GK.
q = sp.Matrix([[-1, 1, 0], [1, -2, 1], [0, 1, -1]])
i, p, a = sp.eye(3), sp.ones(3)/3, sp.Rational(1, 2)
qbar = (1-a*a)*q + a*a*((i-q).inv()-i)
r = ((i-(1-a)*q).inv() + (i-(1+a)*q).inv())/2
assert r == (i-qbar).inv()
assert 32*qbar == sp.Matrix([[-27, 26, 1], [26, -52, 26], [1, 26, -27]])
assert 165*r == sp.Matrix([[107, 39, 19], [39, 87, 39], [19, 39, 107]])
assert qbar*sp.Matrix([1, 0, -1]) == -sp.Rational(7, 8)*sp.Matrix([1, 0, -1])
assert qbar*sp.Matrix([1, -2, 1]) == -sp.Rational(39, 16)*sp.Matrix([1, -2, 1])
h = sp.Matrix([1, 0, -1])
gk = [sp.Rational(2, 3)*(h.T*((-b+p).inv()-p)*h)[0] for b in (q, qbar)]
assert gk == [sp.Rational(4, 3), sp.Rational(32, 21)]
print("Exact a=1/2: clock rates=(2,2/3), weights=(1/2,1/2); both mean intervals=1")
print("32*Qbar=" + str(32*qbar) + "; 165*R=" + str(165*r))
print("Nonzero decay rates: Q=(1,3), Qbar=(7/8,39/16); not a scalar rescaling")
print("Fixed-feature Green-Kubo strengths: Q=4/3, Qbar=32/21; new edge Qbar[0,2]=1/32")

rng, chains = np.random.default_rng(29183), []
for n in range(2, 8):
    base = rng.uniform(.05, 1, (n, n))
    np.fill_diagonal(base, 0)
    base -= np.diag(base.sum(axis=1))
    chains.extend([base / max(-np.diag(base)), np.roll(np.eye(n), 1, axis=1)-np.eye(n)])
chains.extend([np.array([[-1., 1, 0], [0, -1, 1], [4, 0, -4]]),
               np.array([[-3., 1, 2], [0, 0, 0], [0, 0, 0]])])
checks = 0
kernel_error = generator_error = stationary_error = 0.
for base in chains:
    for speed in (.1, 1., 5.):
        qnum = speed*base
        identity = np.eye(len(base))
        stationary = np.linalg.lstsq(np.vstack([qnum.T, np.ones(len(base))]),
                                    np.r_[np.zeros(len(base)), 1.], rcond=None)[0]
        for alpha in (.1, .5, .9):
            qb = alternative(qnum, alpha)
            observed = mixture(qnum, alpha)
            kernel_error = max(kernel_error, norm(observed-np.linalg.inv(identity-qb)))
            generator_error = max(generator_error, norm(qb @ np.ones(len(base))),
                                  -np.min(qb-np.diag(np.diag(qb))))
            stationary_error = max(stationary_error, norm(stationary @ qb)/max(1, norm(qb)))
            assert norm(qb-qnum-alpha**2*(np.linalg.inv(identity-qnum)-identity-qnum)) <= 2e-9*max(1, norm(qnum))
            assert min(observed.ravel()) >= -2e-10
            checks += 1
assert max(kernel_error, generator_error, stationary_error) <= 2e-9
print(f"Mean-matched kernel pairs: {checks}; max kernel error={kernel_error:.3e}; generator residual={generator_error:.3e}; stationary residual={stationary_error:.3e}")

time_error = 0.
for base in (np.array([[-1., 1, 0], [1, -2, 1], [0, 1, -1]]), chains[5], chains[-2]):
    alpha, qb = .5, alternative(base, .5)
    density = lambda t: (np.exp(-t/(1-alpha))/(1-alpha) + np.exp(-t/(1+alpha))/(1+alpha))/2
    left = quad_vec(lambda t: density(t)*expm(t*base), 0, np.inf, epsabs=1e-10, epsrel=1e-10)[0]
    right = quad_vec(lambda t: np.exp(-t)*expm(t*qb), 0, np.inf, epsabs=1e-10, epsrel=1e-10)[0]
    time_error = max(time_error, norm(left-right), norm(left-mixture(base, alpha)))
assert time_error <= 2e-8
print(f"Independent semigroup integrals: 3 pairs; maximum error={time_error:.3e}")

# Compound-Poisson subordination with an explicit omitted-mass norm bound.
subchecks, max_ratio = 0, 0.
for base in chains:
    for alpha in (.5, .9):
        for t in (.2, 1., 3.):
            mu, identity = alpha**2*t, np.eye(len(base))
            cutoff = int(poisson.ppf(1-1e-7, mu))
            jump = np.linalg.inv(identity-base)
            power, series = identity, np.zeros_like(base)
            for k in range(cutoff+1):
                series += poisson.pmf(k, mu)*power
                power = power @ jump
            approximate = expm((1-alpha**2)*t*base) @ series
            error = norm(expm(t*alternative(base, alpha))-approximate)
            budget = poisson.sf(cutoff, mu)+2e-10
            assert error <= budget
            max_ratio = max(max_ratio, error/budget)
            subchecks += 1
print(f"Subordination truncation checks: {subchecks}; max error/(Poisson tail+roundoff)={max_ratio:.6f}")
print("Closeness: coupling tau_a=(1+a*sign)*Exp(1) gives W1<=a; Qbar-Q=a^2*((I-Q)^(-1)-I-Q)")
print("PASS")
