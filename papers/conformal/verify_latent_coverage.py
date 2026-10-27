"""Posterior coverage is a beta mean contraction, not generally beta."""
from fractions import Fraction as F
from itertools import product

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, minimize_scalar
from scipy.special import betainc, betaincc, beta


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


def sharp_beta_lower_contraction(a, b, y):
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


def sharp_beta_upper_contraction(a, b, y):
    """Return the sharp upper-tail envelope and its two-cell witness."""
    p = a/(a+b)
    root = brentq(
        lambda t: p*betaincc(a+1, b, t)-y*betaincc(a, b, t),
        0, y,
    )
    upper_mass = betaincc(a, b, root)
    upper_first = p*betaincc(a+1, b, root)
    upper_mean = upper_first/upper_mass
    lower_mean = (p-upper_first)/(1-upper_mass)
    stoploss_ratio = (upper_first-root*upper_mass)/(y-root)
    return root, upper_mass, lower_mean, upper_mean, stoploss_ratio


def sharp_atomic_envelope(support, probabilities, y, lower):
    """Return the exact stop-loss envelope and randomized cutoff witness."""
    candidates = [z for z in support if (z > y if lower else z < y)]
    if lower:
        objective = lambda t: sum(q*max(t-z, 0) for z, q in zip(support, probabilities))/(t-y)
    else:
        objective = lambda t: sum(q*max(z-t, 0) for z, q in zip(support, probabilities))/(y-t)
    cut = min(candidates, key=objective)
    envelope = objective(cut)
    strict = [(z, q) for z, q in zip(support, probabilities)
              if (z < cut if lower else z > cut)]
    mass = sum(q for _, q in strict)
    first = sum(z*q for z, q in strict)
    atom = sum(q for z, q in zip(support, probabilities) if z == cut)
    split = ((y*mass-first)/(cut-y) if lower
             else (first-y*mass)/(y-cut))
    check(0 <= split <= atom)
    check(first+cut*split == y*(mass+split))
    check(mass+split == envelope)
    return envelope, split, atom


def brute_atomic_cell_mass(support, probabilities, y):
    """Independent vertex enumeration for the largest cell with mean y."""
    best = F(0)
    for pivot in range(len(support)):
        others = [j for j in range(len(support)) if j != pivot]
        for mask in range(1 << len(others)):
            chosen = [probabilities[j] if mask & (1 << h) else F(0)
                      for h, j in enumerate(others)]
            imbalance = sum(q*(support[j]-y) for q, j in zip(chosen, others))
            pivot_mass = -imbalance/(support[pivot]-y)
            if 0 <= pivot_mass <= probabilities[pivot]:
                best = max(best, sum(chosen)+pivot_mass)
    return best


def atomic_quantile_segment(support, probabilities, left, right):
    """Exact mass and first moment on a randomized quantile segment."""
    cumulative = F(0)
    mass = first = F(0)
    for z, probability in zip(support, probabilities):
        overlap = max(F(0), min(right, cumulative+probability)-max(left, cumulative))
        mass += overlap
        first += z*overlap
        cumulative += probability
    check(mass == right-left)
    return mass, first


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
    beta_checks, lower_sharp_checks, upper_sharp_checks = 0, 0, 0
    two_sided_checks, disjoint_checks, overlap_checks = 0, 0, 0
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
                cut, envelope, low, high, ratio = sharp_beta_lower_contraction(a, b, y)
                check(y < cut < 1)
                check(abs(low-y) < 2e-12)
                check(high > y)
                check(abs(ratio-envelope) < 2e-12)
                if k == n:
                    check(abs(cut-y/p) < 2e-12)
                    check(abs(envelope-(y/p)**n) < 2e-12)
                lower_sharp_checks += 1
                y = p+fraction*(1-p)
                cut, envelope, low, high, ratio = sharp_beta_upper_contraction(a, b, y)
                check(0 < cut < y)
                check(low < y)
                check(abs(high-y) < 2e-12)
                check(abs(ratio-envelope) < 2e-12)
                if k == 1:
                    check(abs(cut-(y-p)/(1-p)) < 2e-12)
                    check(abs(envelope-((1-y)/(1-p))**n) < 2e-12)
                upper_sharp_checks += 1
            for lower_fraction in (0.2, 0.5, 0.8):
                lower = lower_fraction*p
                lower_cut, lower_mass, _, _, _ = sharp_beta_lower_contraction(a, b, lower)
                for upper_fraction in (0.2, 0.5, 0.8):
                    upper = p+upper_fraction*(1-p)
                    upper_cut, upper_mass, _, _, _ = sharp_beta_upper_contraction(a, b, upper)
                    envelope = min(1.0, lower_mass+upper_mass)
                    if lower_cut < upper_cut:
                        middle_mass = betainc(a, b, upper_cut)-betainc(a, b, lower_cut)
                        middle_first = p*(betainc(a+1, b, upper_cut)-betainc(a+1, b, lower_cut))
                        middle_mean = middle_first/middle_mass
                        check(lower < middle_mean < upper)
                        check(abs(envelope-(lower_mass+upper_mass)) < 2e-12)
                        disjoint_checks += 1
                    else:
                        witness_cut = (lower_cut+upper_cut)/2
                        witness_mass = betainc(a, b, witness_cut)
                        witness_first = p*betainc(a+1, b, witness_cut)
                        witness_lower_mean = witness_first/witness_mass
                        witness_upper_mean = (p-witness_first)/(1-witness_mass)
                        check(witness_lower_mean <= lower+2e-12)
                        check(witness_upper_mean >= upper-2e-12)
                        check(envelope == 1.0)
                        overlap_checks += 1
                    two_sided_checks += 1
    support = tuple(map(F, (0, F(1, 5), F(1, 2), F(4, 5), 1)))
    atomic_checks, fractional_splits = 0, 0
    bernoulli_lower = bernoulli_upper = 0
    atomic_two_sided = atomic_disjoint = atomic_overlap = 0
    weight_patterns = (
        (1, 1, 1, 1, 1), (1, 2, 3, 4, 5), (5, 4, 3, 2, 1),
        (2, 5, 1, 4, 3), (3, 1, 5, 2, 4), (4, 2, 1, 5, 3),
    )
    for weights in weight_patterns:
        total = sum(weights)
        probabilities = tuple(F(w, total) for w in weights)
        mean = sum(z*q for z, q in zip(support, probabilities))
        lower_envelopes, upper_envelopes = [], []
        for fraction in (F(2, 7), F(4, 7), F(6, 7)):
            for y, lower in ((fraction*mean, True),
                             (mean+fraction*(1-mean), False)):
                if y in support:
                    continue
                envelope, split, atom = sharp_atomic_envelope(
                    support, probabilities, y, lower,
                )
                check(envelope == brute_atomic_cell_mass(support, probabilities, y))
                (lower_envelopes if lower else upper_envelopes).append((y, envelope))
                fractional_splits += int(0 < split < atom)
                atomic_checks += 1
        for lower, lower_mass in lower_envelopes:
            for upper, upper_mass in upper_envelopes:
                if lower_mass+upper_mass <= 1:
                    lm, lf = atomic_quantile_segment(
                        support, probabilities, F(0), lower_mass,
                    )
                    um, uf = atomic_quantile_segment(
                        support, probabilities, 1-upper_mass, F(1),
                    )
                    check(lf == lower*lm and uf == upper*um)
                    witness = lower_mass+upper_mass
                    atomic_disjoint += 1
                else:
                    cut = 1-upper_mass
                    lm, lf = atomic_quantile_segment(
                        support, probabilities, F(0), cut,
                    )
                    um, uf = atomic_quantile_segment(
                        support, probabilities, cut, F(1),
                    )
                    check(lf <= lower*lm and uf == upper*um)
                    witness = F(1)
                    atomic_overlap += 1
                check(witness == min(F(1), lower_mass+upper_mass))
                atomic_two_sided += 1
    # On the canonical Bernoulli space, only the trivial and full sigma-fields
    # are available.  Enlarging the space splits an atom and strictly improves
    # each one-sided envelope.
    for mean in (F(1, 5), F(1, 3), F(1, 2), F(2, 3), F(4, 5)):
        for fraction in (F(1, 4), F(1, 2), F(3, 4)):
            lower = fraction*mean
            randomized = (1-mean)/(1-lower)
            deterministic = 1-mean
            split = lower*(1-mean)/(1-lower)
            check(0 < split < mean)
            check(randomized == deterministic+split > deterministic)
            check(randomized-deterministic == (1-mean)*lower/(1-lower))
            bernoulli_lower += 1

            upper = mean+fraction*(1-mean)
            randomized = mean/upper
            deterministic = mean
            split = mean*(1-upper)/upper
            check(0 < split < 1-mean)
            check(randomized == deterministic+split > deterministic)
            check(randomized-deterministic == mean*(1-upper)/upper)
            bernoulli_upper += 1
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
    print(f"Sharp lower/upper beta-contraction envelopes: {lower_sharp_checks}/{upper_sharp_checks}")
    print(f"Sharp two-sided beta partitions: {two_sided_checks} "
          f"({disjoint_checks} disjoint, {overlap_checks} overlapping)")
    print(f"Exact atomic envelope/LP checks: {atomic_checks}; "
          f"fractional cutoff splits: {fractional_splits}")
    print(f"Exact atomic two-sided partitions: {atomic_two_sided} "
          f"({atomic_disjoint} disjoint, {atomic_overlap} overlapping)")
    print(f"Exact Bernoulli no-split gaps: {bernoulli_lower} lower, "
          f"{bernoulli_upper} upper")
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
