"""NB terminal approximation versus gamma-clock path-TV obstruction."""

import numpy as np
from scipy.integrate import quad
from scipy.stats import nbinom, poisson

from polymoments import stationary
from verify_counts import count_distribution
from verify_uniform_positive_counts import mixing_envelope, raw_moments, symmetric_distribution


ROUND_TOL = 65536*np.finfo(float).eps


def batch_rate(mean_rate, theta):
    return mean_rate/theta*(np.log1p(theta)-theta/(1+theta))


def check_compound_representation():
    worst = 0.
    for theta in [.001, .01, .1, 1., 10.]:
        a, q = 2/theta, theta/(1+theta)
        j = np.arange(1, 2049)
        rates = a*q**j/j
        actual = batch_rate(2., theta)
        independent = a*quad(lambda s: s/(1+s)**2, 0., theta)[0]
        worst = max(worst, abs(actual-independent), abs(sum(rates[1:])-actual))
        assert abs(sum(j*rates)-2.) <= ROUND_TOL*2
        assert abs(sum(j*j*rates)-2*(1+theta)) <= ROUND_TOL*2*(1+theta)
        # Compound-Poisson coefficient recursion, independent of scipy's NB PMF.
        probabilities = np.zeros(120)
        probabilities[0] = np.exp(-a*np.log1p(theta))
        for k in range(1, len(probabilities)):
            probabilities[k] = sum(j[:k]*rates[:k]*probabilities[k-1::-1])/k
        nb = nbinom.pmf(np.arange(len(probabilities)), a, 1/(1+theta))
        worst = max(worst, np.max(abs(probabilities-nb)))
        assert 2*theta/(2*(1+theta)**2)-ROUND_TOL <= actual <= theta+ROUND_TOL
        assert theta-actual <= 4*theta**2/3+ROUND_TOL
    assert worst <= ROUND_TOL
    return worst


def main():
    worst = check_compound_representation()
    designs = [
        (np.array([[-1., 1.], [1., -1.]]), np.array([3., 1.]), (1., 2.)),
        (np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]]), np.array([2., 0., 1.]), None),
        (np.roll(np.eye(4), 1, axis=1)-np.eye(4), np.array([3., 0., 2., 1.]), None),
    ]
    cases, largest_tail, largest_error = 0, 0., 0.
    for index, (q, intensity, envelope) in enumerate(designs):
        pi = stationary(q)
        average = pi @ intensity
        f = intensity-average
        projection = np.outer(np.ones(len(pi)), pi)
        h = np.linalg.inv(-q+projection)-projection
        green = (pi*f) @ h @ f
        assert green > 0
        c, gamma = envelope if envelope else mixing_envelope(q)
        for horizon in [.1, 1., 3.]:
            for speed in [20., 40., 80., 160., 320.]:
                variance = (raw_moments(q, f, speed, horizon) @ pi)[2]
                mean = average*horizon
                theta = variance/mean
                rate = batch_rate(average, theta)
                lower = -np.expm1(-horizon*rate)
                variance_budget = 2*c*max(abs(f))**2/(speed*gamma)**2
                assert abs(variance-2*horizon*green/speed) <= variance_budget+ROUND_TOL
                budget = variance_budget/2+2*horizon*average*theta**2/3 \
                    +horizon**2*average**2*theta**2/8
                # This explicit next-order envelope is used, not a tight asymptotic fit.
                assert abs(lower-horizon*green/speed) <= budget+ROUND_TOL
                actual = count_distribution(speed*q, intensity, horizon, pi, M=256)
                if index == 0:
                    independent = symmetric_distribution(speed, intensity, horizon, pi, 256)
                    largest_error = max(largest_error, np.sum(abs(actual-independent)))
                surrogate = nbinom.pmf(np.arange(120), mean/theta, 1/(1+theta))
                terminal_tv = .5*np.sum(abs(actual[:120]-surrogate))
                tail = .5*(nbinom.sf(119, mean/theta, 1/(1+theta))
                           +poisson.sf(119, max(intensity)*horizon)
                           +poisson.sf(255, max(intensity)*horizon))
                largest_tail = max(largest_tail, tail)
                roundoff = ROUND_TOL*max(1., np.linalg.norm(speed*q, np.inf)*horizon)
                assert lower > terminal_tv+tail+roundoff
                if index == 0 and horizon == 1.:
                    print(f"speed {speed:.0f}: terminal TV {terminal_tv:.10f}; path lower bound {lower:.10f}; m*bound {speed*lower:.10f}")
                cases += 1
    assert largest_error <= ROUND_TOL*3
    print("count-path obstruction certificate; comparisons", cases)
    print(f"compound-Poisson identity discrepancy {worst:.3e}; tolerance {ROUND_TOL:.3e}")
    print(f"maximum independent two-state PGF discrepancy {largest_error:.3e}")
    print(f"maximum terminal omitted-tail TV budget {largest_tail:.3e}")
    print("symmetric Green-Kubo coefficient and m*path-bound limit", .5)
    print("PASS")


if __name__ == "__main__":
    main()
