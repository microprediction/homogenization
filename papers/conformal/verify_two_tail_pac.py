"""Integrated Dirichlet margins for dependent two-tail PAC confidence."""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, linprog, minimize_scalar
from scipy.special import betainc
from scipy.stats import beta, binom
from verify_path_crossover import iid_two_tail_confidence

ROUND = 65536*np.finfo(float).eps


def confidence(n, lower, upper, a, b):
    if min(a, b) <= 0:
        return 0.
    counts = np.arange(lower, upper)
    return float(np.sum(binom.pmf(counts, n, a)
                 * binom.sf(n-upper, n-counts, b/(1-a))))


def margin_density(n, lower, upper, a, b, w):
    x, y = a-w, b-w
    middle, right = upper-lower, n+1-upper
    return (beta.pdf(x, lower, n+1-lower)*betainc(right, middle, y/(1-x))
            + beta.pdf(y, right, upper)*betainc(lower, middle, x/(1-y)))


def transfer(n, lower, upper, a, b, eta_cal, eta_test):
    if not (1 <= lower < upper <= n and 0 < min(a, b) and a+b < 1
            and 0 <= eta_cal <= 1 and 0 <= eta_test <= 1):
        raise ValueError("invalid ranks, tail caps, or TV budgets")
    cost = lambda q: quad(lambda w: w*margin_density(n, lower, upper, a, b, w),
                          0, q, epsabs=2e-12, epsrel=2e-10)[0]
    if eta_test == 0:
        return max(0., confidence(n, lower, upper, a, b)-eta_cal), 0.
    if eta_test >= cost(min(a, b)):
        return 0., min(a, b)
    q = brentq(lambda w: cost(w)-eta_test, 0, min(a, b), xtol=2e-13)
    return max(0., confidence(n, lower, upper, a-q, b-q)-eta_cal), q


def binary_reference(n, lower, upper, a, b, rho):
    """Exact stationary Markov panels; S=(Y+U)/2 reveals Y in {0,1}."""
    success = distance = 0.
    groups = {}
    for mask in range(2**n):
        states = [(mask >> j) & 1 for j in range(n)]
        weight = .5*np.prod([(1+rho*(1 if x == y else -1))/2
                             for x, y in zip(states, states[1:])])
        distance += .5*abs(weight-2.**(-n))
        key = (sum(states), states[-1])
        groups[key] = groups.get(key, 0.)+weight
    for (ones, last), weight in groups.items():
        p0 = (1+rho*(1 if last == 0 else -1))/2
        quantile = lambda v: v/(2*p0) if v <= p0 else .5+(v-p0)/(2*(1-p0))
        x, y = quantile(a), quantile(1-b)
        dp = np.zeros((n+1, n+1)); dp[0, 0] = 1.
        for state in [0]*(n-ones)+[1]*ones:
            left, right = np.clip(2*x-state, 0, 1), 1-np.clip(2*y-state, 0, 1)
            updated = (1-left-right)*dp
            updated[1:, :] += left*dp[:-1, :]
            updated[:, 1:] += right*dp[:, :-1]
            dp = updated
        success += weight*np.sum(dp[lower:, n+1-upper:])
    return success, distance


def range_cost(n, a, b, q):
    """Sample-range survival and integrated margin, without quadrature."""
    x, y = a-q, b-q
    residuals = np.array([x, y, x+y])
    powers = np.exp(n*np.log1p(-residuals))
    baseline = np.exp((n+1)*np.log1p(-np.array([a, b, a+b])))
    survival = 1-powers[0]-powers[1]+powers[2]
    integral = np.dot([1., 1., -.5], (1-residuals)*powers-baseline)/(n+1)
    cost = q*(powers[0]+powers[1]-powers[2])-integral
    return max(0., survival), max(0., cost)


def range_certificate(n, a, b, eta_cal, eta_test):
    if eta_test == 0:
        return max(0., range_cost(n, a, b, 0.)[0]-eta_cal)
    if eta_test >= range_cost(n, a, b, min(a, b))[1]:
        return 0.
    q = brentq(lambda v: range_cost(n, a, b, v)[1]-eta_test,
               0., min(a, b), xtol=np.finfo(float).tiny, rtol=8*np.finfo(float).eps)
    return max(0., range_cost(n, a, b, q)[0]-eta_cal)


def minimum_range_size(a, b, eta_cal, eta_test, delta):
    """Finite-formula search; reject unresolved boundaries or very large sizes."""
    if not (0 < min(a, b) and a+b < 1 and 0 < delta < 1
            and 0 <= eta_cal <= 1 and 0 <= eta_test <= 1):
        raise ValueError("invalid caps, confidence, or TV budgets")
    capacity = min(a, b)*(delta-eta_cal)
    if delta <= eta_cal or eta_test >= capacity:
        return None
    if capacity-eta_test <= ROUND*max(capacity, eta_test):
        raise ValueError("feasibility margin needs higher precision")
    lower, upper = 1, 2
    feasible = lambda n: range_certificate(n, a, b, eta_cal, eta_test) >= 1-delta
    while not feasible(upper):
        lower, upper = upper, 2*upper
        if upper > 65536:
            raise ValueError("minimum-size search needs higher precision")
    while upper-lower > 1:
        middle = (lower+upper)//2
        if feasible(middle):
            upper = middle
        else:
            lower = middle
    return upper


def main():
    cases = comparisons = identities = 0
    setups = [(12, 1, 12, .2, .15), (30, 2, 28, .15, .2),
              (60, 2, 58, .1, .12), (100, 1, 100, .05, .05)]
    for n, lower, upper, a, b in setups:
        args = (n, lower, upper)
        for fraction in (0., .2, .6):
            x, y = a-fraction*min(a, b), b-fraction*min(a, b)
            observed = confidence(*args, x, y)
            reference = iid_two_tail_confidence(*args, x, y)
            direct = quad(lambda z: beta.pdf(z, lower, n+1-lower)
                          * betainc(n+1-upper, upper-lower, y/(1-z)), 0, x,
                          epsabs=2e-12, epsrel=2e-11)[0]
            assert abs(observed-reference) <= ROUND
            assert abs(observed-direct) <= ROUND
            identities += 1
        for eta_cal in (0., .0001, .002):
            previous = 1.
            for eta_test in (0., 1e-7, 1e-5, .0001, .002):
                result, q = transfer(*args, a, b, eta_cal, eta_test)
                assert 0 <= result <= previous+ROUND
                previous = result
                old = lambda slack: confidence(*args, a-slack, b-slack)-2*(eta_cal+eta_test)/slack
                optimized = minimize_scalar(lambda x: -old(x), bounds=(1e-9, min(a, b)-1e-9),
                                            method="bounded", options={"xatol": 1e-12})
                assert result >= max(0., -optimized.fun)-ROUND
                if q > 0:
                    density_cost = quad(lambda w: w*margin_density(*args, a, b, w), 0, q)[0]
                    survival_cost = quad(lambda w: confidence(*args, a-w, b-w), 0, q)[0]
                    survival_cost -= q*confidence(*args, a-q, b-q)
                    assert abs(density_cost-survival_cost) <= ROUND
                    comparisons += 1
                cases += 1
        assert transfer(*args, a, b, 0., 1.)[0] == 0
    # Finite probability-space rearrangement: fractional knapsack versus an independent LP.
    rng = np.random.default_rng(2026101503)
    for budget in (.0001, .001, .005, .02):
        allocations = rng.dirichlet([2, 27, 2], size=128)
        margins = np.maximum(0, np.minimum(.15-allocations[:, 0], .15-allocations[:, 2]))
        weights = np.full(len(margins), 1/len(margins))
        remaining, mass = budget, 0.
        for index in np.argsort(margins):
            take = weights[index] if margins[index] == 0 else min(weights[index], remaining/margins[index])
            mass += take
            remaining -= take*margins[index]
        solution = linprog(-weights, A_ub=[weights*margins], b_ub=[budget],
                           bounds=(0, 1), method="highs")
        assert solution.success and abs(mass+solution.fun) <= 1e-9
    n, lower, upper, a, b = setups[-1]
    result, q = transfer(n, lower, upper, a, b, 0., .0001)
    old = minimize_scalar(lambda s: -(confidence(n, lower, upper, a-s, b-s)-.0002/s),
                          bounds=(1e-9, .05-1e-9), method="bounded", options={"xatol": 1e-13})
    print(f"dependent-budget cases {cases}; cost-identity cases {comparisons}; independent Dirichlet cases {identities}")
    print("fractional-knapsack versus linear-program cases 4")
    print(f"N=100 sample range, caps=.05, eta_cal=0, eta_test=.0001: iid {confidence(n, lower, upper, a, b):.9f}")
    print(f"integrated-margin confidence {result:.9f}; boundary margin {q:.11f}; old optimized bound {-old.fun:.9f}")
    for rho in (.0, .02, .08, .2):
        exact, eta_cal = binary_reference(8, 1, 8, .2, .2, rho)
        bound, _ = transfer(8, 1, 8, .2, .2, eta_cal, rho/2)
        assert bound <= exact+ROUND
        if rho == 0:
            assert abs(exact-confidence(8, 1, 8, .2, .2)) <= ROUND
        print(f"Markov rho={rho:g}: exact confidence {exact:.9f}; certificate {bound:.9f}")
    designs = enclosures = closed_checks = 0
    for a, b in ((.05, .05), (.03, .08), (.2, .1)):
        c = min(a, b)
        for n in (12, 30, 100):
            for fraction in (.1, .5, .9):
                q = fraction*c
                survival, cost = range_cost(n, a, b, q)
                reference = quad(lambda v: v*margin_density(n, 1, n, a, b, v), 0, q)[0]
                assert abs(cost-reference) <= ROUND
                assert abs(survival-confidence(n, 1, n, a-q, b-q)) <= ROUND
                closed_checks += 1
        for delta in (.05, .1):
            for eta_cal in (0., .001):
                for fraction in (0., .1, .5, .9):
                    eta_test = fraction*c*(delta-eta_cal)
                    size = minimum_range_size(a, b, eta_cal, eta_test, delta)
                    assert size >= 2
                    assert range_certificate(size, a, b, eta_cal, eta_test) >= 1-delta-ROUND
                    assert size == 2 or range_certificate(size-1, a, b, eta_cal, eta_test) < 1-delta
                    previous = 0.
                    ceiling = max(0., 1-eta_cal-eta_test/c)
                    for n in (2, 8, 32, 128, 512, 2048):
                        value = range_certificate(n, a, b, eta_cal, eta_test)
                        assert previous-ROUND <= value <= ceiling+ROUND
                        assert value >= max(0., ceiling-3/(2*c*(n+1)))-ROUND
                        previous = value
                        enclosures += 1
                    assert minimum_range_size(a, b, eta_cal, c*(delta-eta_cal), delta) is None
                    designs += 1
    for n in (2, 8, 32, 128, 512):
        moment = quad(lambda x: 2*(1-x)**n-max(0., 1-2*x)**n, 0, 1, epsabs=2e-12)[0]
        assert abs(moment-3/(2*(n+1))) <= ROUND
    boundary = .05*.05
    try:
        minimum_range_size(.05, .05, 0., np.nextafter(boundary, 0.), .05)
    except ValueError:
        pass
    else:
        raise AssertionError("roundoff-indistinguishable feasibility was accepted")
    print(f"closed-form cost checks {closed_checks}; minimum-size designs {designs}; ceiling enclosures {enclosures}")
    print("independent outer-spacing moment checks 5; near-boundary precision guard PASS")
    print("two 5% caps, 95% confidence, eta_cal=0; eta_test : minimum N")
    for budget in (0., .0001, .001, .002, boundary):
        size = minimum_range_size(.05, .05, 0., budget, .05)
        if size is not None:
            for count in (size-1, size):
                reference, _ = transfer(count, 1, count, .05, .05, 0., budget)
                assert abs(reference-range_certificate(count, .05, .05, 0., budget)) <= 1e-9
        print(f"{budget:g} : {size}")
    print("independent quadrature at eight minimum-size boundary cases PASS")
    print("PASS")


if __name__ == "__main__":
    main()
