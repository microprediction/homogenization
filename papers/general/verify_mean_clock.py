"""Exact mean-preserving clock/generator ambiguity, not a simulation claim."""
import numpy as np
import sympy as sp
from scipy.integrate import quad_vec
from scipy.linalg import expm
from scipy.optimize import brentq
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

# Finite-moment extension: exact clock moments and positive divided differences.
def clock(k):
    m = k+1
    scales = [sp.Rational(1, 2)+sp.Rational(j, m) for j in range(m+1)]
    base = [sp.binomial(m, j)/2**m for j in range(m+1)]
    plus = [b*(1+sp.Rational((-1)**j, 2)) for j, b in enumerate(base)]
    minus = [b*(1-sp.Rational((-1)**j, 2)) for j, b in enumerate(base)]
    return m, scales, base, plus, minus


def exact_f(scales, weights, z):
    return sum(w/(1+t*z) for t, w in zip(scales, weights))


v1, v2 = np.array([1., 0, -1]), np.array([1., -2, 1])
p1, p2, p0 = np.outer(v1, v1)/2, np.outer(v2, v2)/6, np.ones((3, 3))/3
moment_checks = divided_checks = alias_checks = 0
moment_kernel_error = 0.
for k in range(1, 9):
    m, scales, base, plus, minus = clock(k)
    for ell in range(k+1):
        assert sum((u-v)*t**ell for t, u, v in zip(scales, plus, minus)) == 0
        moment_checks += 1  # Interval moments are ell! times these scale moments.
    assert sum(t*w for t, w in zip(scales, plus)) == 1
    for z in map(sp.Rational, ('1/10', '1/2', '1', '2', '5')):
        d = sp.factorial(m)*(z/m)**m/(2**m*sp.prod(1+t*z for t in scales))
        assert exact_f(scales, plus, z)-exact_f(scales, minus, z) == d > 0
        divided_checks += 1
    if k <= 6:
        tf, wp, wm = [np.array(x, dtype=float) for x in (scales, plus, minus)]
        f = lambda x, w: np.sum(w/(1+tf*x))
        roots = [brentq(lambda b: f(b, wm)-f(x, wp), 0, x, xtol=1e-12) for x in (1., 2.)]
        assert 0 < roots[1] < 2*roots[0] < 3*roots[0]
        qb = -roots[0]*p1-roots[1]*p2
        left, right = p0+f(1, wp)*p1+f(2, wp)*p2, p0+f(roots[0], wm)*p1+f(roots[1], wm)*p2
        moment_kernel_error = max(moment_kernel_error, norm(left-right))
        assert np.min(qb-np.diag(np.diag(qb))) >= -2e-10
        alias_checks += 1
assert moment_kernel_error <= 2e-9
print(f"Finite-moment clocks: K=1..8; {moment_checks} exact matching moments including masses; {divided_checks} positive rational-transform identities")
print(f"Three-state nonproportional aliases: {alias_checks}; maximum kernel discrepancy={moment_kernel_error:.3e}")
m, scales, base, plus, minus = clock(2)
assert plus == list(map(sp.Rational, ('3/16', '3/16', '9/16', '1/16')))
assert minus == list(map(sp.Rational, ('1/16', '9/16', '3/16', '3/16')))
assert 2*sum(w*t*t for w, t in zip(plus, scales)) == sp.Rational(13, 6)
bounds = [(sp.Rational(992415305905, 10**12), sp.Rational(992415305907, 10**12)),
          (sp.Rational(1972100882987, 10**12), sp.Rational(1972100882989, 10**12))]
for x, (lower, upper) in zip((1, 2), bounds):
    target = exact_f(scales, plus, x)
    assert exact_f(scales, minus, lower) > target > exact_f(scales, minus, upper)
assert bounds[1][1] < 2*bounds[0][0] < 3*bounds[0][0]
print("K=2 exact scales=(1/2,5/6,7/6,3/2); weights+=(3,3,9,1)/16; weights-=(1,9,3,3)/16; E[tau]=1, E[tau^2]=13/6")
print("Rational root enclosures: b1 in [0.992415305905,0.992415305907]; b2 in [1.972100882987,1.972100882989]; b2<2*b1<3*b1")
print("PASS")
