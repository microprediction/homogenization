"""Even-moment boundary law and quartic population-coverage certificate.

Finite matrix moments are exact representations; floating-point evaluation
is not outward rounded. Gaussian Taylor bounds exclude numerical roundoff.
"""
from math import factorial, gamma, pi as PI

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import eval_hermitenorm, factorial2, roots_legendre
from scipy.stats import norm

from verify_path_crossover import asymmetric_occupation_density


def coefficients(q, values):
    n = len(q)
    stationary = np.linalg.solve(q.T + np.ones((n, n)), np.ones(n))
    f = values - stationary @ values
    group = np.linalg.inv(q + np.ones((n, 1)) * stationary) - stationary
    h = group @ f
    boundary = 2 * group @ (f * h)
    variance = -2 * stationary @ (f * h)
    skew = 3 * stationary @ (f * boundary)
    return stationary, f, h, boundary, variance, skew


def moments(q, f, c, order):
    """E_i[(I_c/sqrt(c))**j], j=0,...,order, by backward hierarchy."""
    n = len(q)
    block = np.kron(np.eye(order + 1), c * q)
    for j in range(1, order + 1):
        block[j*n:(j+1)*n, (j-1)*n:j*n] = j * np.sqrt(c) * np.diag(f)
    initial = np.r_[np.ones(n), np.zeros(n * order)]
    return (expm(block) @ initial).reshape(order + 1, n)


def derivative_bound(j):
    """Fourier bound on sup |phi**(j)|, including j=0."""
    return 2**((j - 1) / 2) * gamma((j + 1) / 2) / PI


def gaussian_budget(raw, stationary, sigma=0.5, target=0.9, degree=3):
    # raw[j,i] = E_i[(T/sigma)**j], T=mu*A**4 >= 0.
    pooled = raw @ stationary
    delta = raw - pooled[:, None]
    def cdf(q):
        z = q / sigma
        return norm.cdf(z) - norm.pdf(z) * sum(
            eval_hermitenorm(j-1, z) * pooled[j] / factorial(j)
            for j in range(1, degree + 1))
    remainder = derivative_bound(degree) * raw[degree+1] / factorial(degree+1)
    pooled_remainder = stationary @ remainder
    lower, upper = sigma * (norm.ppf(target) + np.array([-1., 1.]))
    assert cdf(lower) + pooled_remainder < target < cdf(upper) - pooled_remainder
    qhat = brentq(lambda q: cdf(q) - target, lower, upper, xtol=2e-13)
    # Markov's inequality at T <= sigma gives a true pooled-density floor.
    density_floor = (1 - pooled[1]) * min(
        norm.pdf((lower - sigma) / sigma), norm.pdf(upper / sigma)) / sigma
    assert density_floor > 0
    qerror = (abs(cdf(qhat) - target) + pooled_remainder) / density_floor
    lipschitz = sum(derivative_bound(j) * abs(delta[j]) / factorial(j)
                    for j in range(1, degree + 1)) / sigma
    lipschitz += derivative_bound(degree+1) * (
        raw[degree+1] + pooled[degree+1]) / (sigma * factorial(degree+1))
    z = qhat / sigma
    gap = -norm.pdf(z) * sum(eval_hermitenorm(j-1, z) * delta[j] / factorial(j)
                            for j in range(1, degree + 1))
    return qhat, gap, remainder + pooled_remainder + lipschitz * qerror


def occupation_expectation(c, mean, fun):
    values = []
    for i, rate in enumerate([1.7, 0.4]):
        atom = np.exp(-rate*c) * fun((1 if i == 0 else -1) - mean)
        integral = quad(lambda u: fun(2*u-1-mean) * asymmetric_occupation_density(
            u, c, 1.7, 0.4, i), 0, 1, epsabs=2e-12, epsrel=2e-11)[0]
        values.append(atom + integral)
    return np.array(values)


def main():
    fixtures = [
        (np.array([[-1.7, 1.7], [.4, -.4]]), np.array([1., -1.])),
        (np.array([[-1.8, 1.4, .4], [.3, -1.2, .9], [1.1, .2, -1.3]]),
         np.array([1., -.7, .2])),
    ]
    max_moment_error = max_coverage_error = 0.
    for number, (q, values) in enumerate(fixtures):
        stationary, f, h, boundary, variance, skew = coefficients(q, values)
        d = {k: k * factorial2(2*k-1) * (boundary * variance**(k-1)
             - 2*(k-1)/3 * h * skew * variance**(k-2)) for k in [2, 3, 4]}
        assert variance > 0
        assert max(abs(stationary @ v) for v in d.values()) < 2e-11
        errors = []
        for c in [32., 128., 512., 2048.]:
            m = moments(q, f, c, 8)
            error = max(np.max(abs(c*(m[2*k] - stationary @ m[2*k]) - d[k]))
                        / max(1., np.max(abs(d[k]))) for k in d)
            errors.append(error)
        assert errors[-1] < errors[0] / 30 and errors[-1] < .04
        print(f"model {number}: even moments k=2,3,4 relative-scaled error "
              f"{errors[0]:.9f} -> {errors[-1]:.9f}")
        coefficient = -norm.pdf(norm.ppf(.9)) / .5 * d[2]
        omitted_skew = -norm.pdf(norm.ppf(.9)) / .5 * (6*variance*boundary)
        print(f"model {number}: quartic limit {coefficient}; without skew {omitted_skew}")
        coverage_errors = []
        for c in [32., 64., 128., 256., 512.]:
            m = moments(q, f, c, 16)
            raw = np.array([m[4*j] / (.5*c*c)**j for j in range(5)])
            qhat, gap, budget = gaussian_budget(raw, stationary)
            coverage_errors.append(np.max(abs(c**3*gap - coefficient)))
            assert abs(stationary @ gap) < 2e-13
            print(f"model {number}, c={c:.0f}: c^3 gap={c**3*gap}, "
                  f"scaled Taylor/quantile budget={c**3*max(budget):.9g}")
            if number == 0 and c <= 64:
                nodes, weights = roots_legendre(32)
                mean = stationary @ values
                qexact = brentq(lambda threshold: stationary @ occupation_expectation(
                    c, mean, lambda a: norm.cdf((threshold-a**4)/.5)) - .9,
                    qhat-.1, qhat+.1, xtol=2e-13)
                z = qexact / .5
                def centered_kernel(a):
                    x = a**4 / .5
                    return -x * .5 * (weights @ norm.pdf(z - .5*(nodes+1)*x))
                independent = occupation_expectation(c, mean, centered_kernel)
                independent -= stationary @ independent
                assert np.all(abs(independent - gap) <= budget + 2e-10)
                max_coverage_error = max(max_coverage_error, np.max(abs(independent-gap)))
        assert coverage_errors[-1] < coverage_errors[0] / 10
        print(f"model {number}: quartic coefficient error "
              f"{coverage_errors[0]:.9f} -> {coverage_errors[-1]:.9f}")
        if number == 0:
            for c in [2., 8., 32.]:
                m = moments(q, f, c, 8)
                for j in [2, 4, 6, 8]:
                    independent = occupation_expectation(c, stationary @ values, lambda a: a**j)
                    max_moment_error = max(max_moment_error,
                        np.max(abs(independent - m[j]/c**(j/2))))
            for c in [128., 512., 2048.]:
                def noiseless_cdfs(radius):
                    low, high = (1 + stationary @ values + np.array([-radius, radius])) / 2
                    return np.array([quad(lambda u: asymmetric_occupation_density(
                        u, c, 1.7, .4, i), low, high, epsabs=2e-12)[0] for i in [0, 1]])
                # This bracket excludes both no-jump atoms; |A| <= r is
                # exactly both the quadratic and quartic quantile event.
                radius = brentq(lambda r: stationary @ noiseless_cdfs(r) - .9,
                                .001, .35, xtol=2e-13)
                gap = noiseless_cdfs(radius) - .9
                assert abs(stationary @ gap) < 2e-10 and abs(c*gap[0]) > .1
                print(f"c={c:.0f}: quadratic/quartic noiseless c-gap={c*gap}")
    assert max_moment_error < 2e-10
    print(f"12 independent Bessel moment comparisons: max error {max_moment_error:.3e}")
    print(f"2 independent pooled-quantile coverage comparisons: max error {max_coverage_error:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
