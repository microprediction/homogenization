"""Posterior coverage is a beta mean contraction, not generally beta."""
from fractions import Fraction as F
from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import betainc, beta


def check(condition):
    if not condition:
        raise AssertionError("latent-coverage certificate failed")


def nested(n, prior, r=F(1, 2)):
    weight = prior + (1-prior)*r**n
    slope = (prior + (1-prior)*r**(n+1))/weight

    def cdf(x):
        return weight*min(x/slope, 1)**n + (1-prior)*max(x**n-r**n, 0)

    def stoploss(t):
        low = (t**(n+1)/((n+1)*slope**n) if t <= slope
               else t-F(n, n+1)*slope)
        high = (0 if t <= r else
                (t**(n+1)-r**(n+1))/(n+1)-r**n*(t-r))
        return weight*low + (1-prior)*high

    moments = [F(n, n+j)*(weight*slope**j
               + (1-prior)*(1-r**(n+j))) for j in (1, 2)]
    return slope, cdf, stoploss, moments


def beta_stoploss(t, a, b):
    return t*betainc(a, b, t)-a/(a+b)*betainc(a+1, b, t)


def latent_posterior_variance(n, prior, r):
    weight = prior+(1-prior)*r**n
    first, second = ((prior+(1-prior)*r**(n+j))/weight for j in (1, 2))
    return F(n, n+2)*weight*(second-first*first)


def sharp_beta_contraction(a, b, y):
    """Return the sharp lower-tail envelope and its two-cell witness."""
    p = a/(a+b)
    root = brentq(
        lambda t: p*betainc(a+1, b, t)-y*betainc(a, b, t),
        y, 1,
    )
    lower_mass = betainc(a, b, root)
    lower_first = p*betainc(a+1, b, root)
    lower_mean = lower_first/lower_mass
    upper_mean = (p-lower_first)/(1-lower_mass)
    stoploss_ratio = (root*lower_mass-lower_first)/(root-y)
    return root, lower_mass, lower_mean, upper_mean, stoploss_ratio


def main():
    exact, integrated, boundary_checks, latent_decompositions = 0, 0, 0, 0
    max_error = 0.0
    cases = product((1, 2, 3, 5, 9, 19), (F(1, 10), F(1, 2), F(9, 10)),
                    (F(1, 2), F(18, 25), F(4, 5)))
    for n, prior, r in cases:
        p = F(n, n+1)
        slope, cdf, loss, moments = nested(n, prior, r)
        check(moments[0] == p)
        check(moments[1]-p*p <= p*(1-p)/(n+2))
        posterior_variance = latent_posterior_variance(n, prior, r)
        check(moments[1]-p*p+posterior_variance == p*(1-p)/(n+2))
        for future in (2, 5, 11):
            within = future*(p-moments[1])+future*(future-1)*posterior_variance
            between = future**2*(moments[1]-p*p)
            target = future*p*(1-p)*F(n+1+future, n+2)
            check(within+between == target)
            latent_decompositions += 1
        for j in range(1, 20):
            t = F(j, 20)
            check(loss(t) <= t**(n+1)/(n+1))
            if t < p:
                check(cdf(t) <= (t/p)**n)
                for delta in (F(1, 10), F(1, 4)):
                    x = (1-delta)*t
                    # B supported above/below all calibration scores gives R=0/1.
                    failure = cdf(max((x-delta)/(1-delta), 0))
                    check(failure <= cdf(t) <= (x/((1-delta)*p))**n)
                    boundary_checks += 1
            exact += 1
        for t in (0.3, 0.7, 0.95):
            direct = quad(lambda x: float(cdf(F(x))), 0, t,
                          points=[x for x in (float(r), float(slope)) if x < t],
                          epsabs=1e-11)[0]
            error = abs(direct-float(loss(F(t))))
            max_error = max(max_error, error)
            check(error < 2e-9)
            integrated += 1
    beta_checks, sharp_checks = 0, 0
    for n in (2, 5, 9, 19):
        for k in range(1, n+1):
            a, b, p = k, n+1-k, k/(n+1)
            for t in (0.2, 0.5, 0.8):
                direct = quad(lambda u: (t-u)*u**(a-1)*(1-u)**(b-1)/beta(a, b),
                              0, t, epsabs=1e-11)[0]
                error = abs(direct-beta_stoploss(t, a, b))
                max_error = max(max_error, error)
                check(error < 2e-9)
                beta_checks += 1
            x = 0.8*p
            result = minimize_scalar(lambda t: beta_stoploss(t, a, b)/(t-x),
                                     bounds=(x+1e-8, 1), method="bounded")
            root = brentq(lambda t: p*betainc(a+1, b, t)-x*betainc(a, b, t), x, 1)
            check(abs(result.fun-betainc(a, b, root)) < 2e-7)
            if k == n:
                check(abs(result.fun-(x/p)**n) < 2e-7)
            for fraction in (0.2, 0.5, 0.8):
                y = fraction*p
                cut, envelope, low, high, ratio = sharp_beta_contraction(a, b, y)
                check(y < cut < 1)
                check(abs(low-y) < 2e-12)
                check(high > y)
                check(abs(ratio-envelope) < 2e-12)
                if k == n:
                    check(abs(cut-y/p) < 2e-12)
                    check(abs(envelope-(y/p)**n) < 2e-12)
                sharp_checks += 1
    n, x, prior, r = 9, F(4, 5), F(1, 50), F(18, 25)
    slope, cdf, _, moments = nested(n, prior, r)
    check(r < slope < x < F(9, 10))
    check(moments[0] == F(9, 10))
    check(moments[1]-moments[0]**2 < F(9, 1100))
    check(cdf(x) == prior+(1-prior)*x**n > x**n)
    posterior_variance = latent_posterior_variance(n, prior, r)
    future = 20
    within = future*(moments[0]-moments[1])+future*(future-1)*posterior_variance
    between = future**2*(moments[1]-moments[0]**2)
    print(f"Nested-uniform exact stop-loss/tail checks: {exact}; mean/variance cases: 54")
    print(f"Independent nested-CDF integrals: {integrated}; beta-integral checks: {beta_checks}")
    print(f"Sharp beta-contraction envelopes and two-cell witnesses: {sharp_checks}")
    print(f"Finite-boundary comparisons: {boundary_checks}; independent beta minimizer checks: 35")
    print(f"Exact latent repeated-future variance decompositions: {latent_decompositions}")
    print(f"Maximum integration discrepancy: {max_error:.3e}; tolerance 2e-9")
    print(f"Witness: prior {prior}, wide support [0,{1/r}], low-branch slope {float(slope):.12f}")
    print(f"N=k=9, threshold .8: actual failure {cdf(x)} = {float(cdf(x)):.12f}")
    print(f"iid beta shortcut .8^9: {float(x**n):.12f}; valid bound (.8/.9)^9: {float((x/F(9,10))**n):.12f}")
    print(f"Mean {moments[0]}; variance {float(moments[1]-moments[0]**2):.12f}; iid {9/1100:.12f}")
    print(f"Expected posterior future-indicator covariance: {float(posterior_variance):.12f}")
    print(f"L=20 count variance: within-panel mean {float(within):.12f}; between-panel {float(between):.12f}; total {float(within+between):.12f}")
    delta = np.exp(-10)
    print(f"N=k=9, threshold .8: iid {.8**9:.12f}; latent {(.8/.9)**9:.12f}; finite-history {(.8/((1-delta)*.9))**9:.12f}")
    print("PASS")


if __name__ == "__main__":
    main()
