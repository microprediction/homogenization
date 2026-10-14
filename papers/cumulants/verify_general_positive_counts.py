"""Finite-chain gamma-Poisson count approximation, uniformly over initial priors."""

import numpy as np
from scipy.linalg import expm
from scipy.stats import nbinom, poisson

from polymoments import stationary
from verify_counts import count_distribution

ROUND_TOL = 32768*np.finfo(float).eps


def mixing_envelope(q, h=.5):
    p = expm(h*q)
    rho = .5*max(np.sum(abs(a-b)) for a in p for b in p)
    assert 0 < rho < 1
    return 2/rho, -np.log(rho)/h


def raw_moments(q, f, speed, horizon):
    n = len(f)
    system = np.kron(np.eye(5), speed*q)
    for j in range(1, 5):
        system[j*n:(j+1)*n, (j-1)*n:j*n] = j*np.diag(f)
    initial = np.zeros(5*n)
    initial[:n] = 1
    return (expm(horizon*system) @ initial).reshape(5, n)


def bounds(c, gamma, fmax, speed, horizon, minimum_mean):
    beta = speed*gamma
    v = 2*c*fmax**2*horizon/beta
    b3 = 18*c**2*fmax**3*horizon/beta**2+2*c**3*fmax**3/beta**3
    b4 = 24*c**2*fmax**4*horizon**2/beta**2+60*c**3*fmax**4*horizon/beta**3
    tv = 2*(b3+2*v*v/minimum_mean)/3+(b4+3*v*v+6*v**3/minimum_mean**2)/3
    return v, b3, b4, min(1., tv)


def main():
    designs = [
        ("unequal rates", np.array([[-1., 1.], [4., -4.]]), np.array([3., 0.])),
        ("directed cycle", np.array([[-1., 1., 0.], [0., -1., 1.], [1., 0., -1.]]), np.array([3., 1., 0.])),
        ("defective cycle", np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]]), np.array([2., 0., 1.])),
        ("four states", np.array([[-3., 2., 1., 0.], [0., -2., 1., 1.], [1., 0., -3., 2.], [2., 1., 0., -3.]]), np.array([0., .5, 2., 3.])),
    ]
    # (-3) has algebraic multiplicity two but eigenspace dimension one.
    assert np.linalg.matrix_rank(designs[2][1]+3*np.eye(3), tol=ROUND_TOL) == 2
    cases = count_cases = 0
    largest_tail = largest_scaled = 0.
    print("general positive-count certificate; zero intensities in every design")
    for name, q, intensity in designs:
        pi = stationary(q)
        average = pi @ intensity
        f = intensity-average
        fmax = max(abs(f))
        c, gamma = mixing_envelope(q)
        projector = np.outer(np.ones(len(pi)), pi)
        for t in np.r_[0., np.geomspace(.001, 20., 100)]:
            assert np.linalg.norm(expm(t*q)-projector, np.inf) <= c*np.exp(-gamma*t)+ROUND_TOL
        priors = list(np.eye(len(pi)))+[pi, np.ones(len(pi))/len(pi)]
        distances = []
        print(name, "C, gamma", f"{c:.6f}", f"{gamma:.6f}")
        for horizon in [.2, 1., 3.]:
            for speed in [1., 5., 20., 40., 80., 160., 320.]:
                raw = raw_moments(q, f, speed, horizon)
                minimum_mean = min(average*horizon+raw[1])
                assert minimum_mean > 0
                v_bound, b3, b4, tv_bound = bounds(c, gamma, fmax, speed, horizon, minimum_mean)
                for prior_index, prior in enumerate(priors):
                    a, second, third, fourth = (raw @ prior)[1:]
                    mu, v = average*horizon+a, second-a*a
                    m3 = third-3*a*second+2*a**3
                    m4 = fourth-4*a*third+6*a*a*second-3*a**4
                    scale = max(1., (fmax*horizon)**4)
                    tol = ROUND_TOL*scale
                    assert abs(a) <= c*fmax/(speed*gamma)+tol
                    assert 0 <= v <= v_bound+tol
                    assert abs(m3) <= b3+tol and -tol <= m4 <= b4+tol
                    moment_bound = min(1., 2*abs(m3-2*v*v/mu)/3+(m4+3*v*v+6*v**3/mu**2)/3)
                    assert moment_bound <= tv_bound+tol
                    cases += 1
                    if horizon != 1. or speed < 20.:
                        continue
                    shape, success = mu*mu/v, mu/(mu+v)
                    largest_scaled = max(largest_scaled, abs(nbinom.mean(shape, success)-mu)/max(1., mu),
                                         abs(nbinom.var(shape, success)-mu-v)/max(1., mu+v))
                    exact = count_distribution(speed*q, intensity, horizon, prior)
                    assert exact.min() >= -ROUND_TOL and abs(exact.sum()-1) <= ROUND_TOL
                    distance = .5*np.sum(abs(exact[:80]-nbinom.pmf(np.arange(80), shape, success)))
                    tail = .5*(nbinom.sf(79, shape, success)+poisson.sf(79, max(intensity)*horizon)
                               +poisson.sf(255, max(intensity)*horizon))
                    largest_tail = max(largest_tail, tail)
                    assert distance+tail <= moment_bound+ROUND_TOL
                    count_cases += 1
                    if prior_index == 0:
                        distances.append(distance)
                        print("  speed, pure-state-0 NB TV, all-prior bound", int(speed), f"{distance:.10f}", f"{tv_bound:.10f}")
                        if speed == 320.:
                            print("  scaled signed third moment", f"{speed**2*m3:.6f}")
        print("  last measured TV order", f"{np.log2(distances[-2]/distances[-1]):.6f}")
    assert largest_scaled <= ROUND_TOL
    equal = count_distribution(designs[2][1], np.full(3, 2.), 1., np.array([1., 0., 0.]))
    assert np.max(abs(equal[:80]-poisson.pmf(np.arange(80), 2.))) <= ROUND_TOL
    print("all-prior moment cases", cases, "; count-law comparisons", count_cases)
    print("maximum NB moment-matching scaled discrepancy", f"{largest_scaled:.3e}")
    print("maximum omitted-tail TV budget", f"{largest_tail:.3e}")
    print("roundoff tolerance", f"{ROUND_TOL:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
