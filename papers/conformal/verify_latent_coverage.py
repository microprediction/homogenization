"""Posterior coverage is a beta mean contraction, not generally beta."""
from fractions import Fraction as F
from itertools import combinations, product
from math import comb, exp, factorial, inf, lcm, log, log1p, pi, sqrt

import mpmath as mp
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq, linprog, minimize_scalar
from scipy.special import betainc, betaincc, beta, expit, gammaln, ndtr, ndtri
from scipy.special import stdtr, stdtrit


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


def fractional_tail_rounding(support, probabilities, y, lower):
    """Greedy fractional optimum and its rounded-down canonical witness."""
    if lower:
        full = tuple(j for j, z in enumerate(support) if z <= y)
        candidates = sorted(
            (j for j, z in enumerate(support) if z > y),
            key=lambda j: support[j],
        )
        budget = sum(probabilities[j]*(y-support[j]) for j in full)
        unit_cost = lambda j: support[j]-y
    else:
        full = tuple(j for j, z in enumerate(support) if z >= y)
        candidates = sorted(
            (j for j, z in enumerate(support) if z < y),
            key=lambda j: support[j], reverse=True,
        )
        budget = sum(probabilities[j]*(support[j]-y) for j in full)
        unit_cost = lambda j: y-support[j]
    mass = rounded = sum(probabilities[j] for j in full)
    split = cutoff_mass = F(0)
    for j in candidates:
        cost = probabilities[j]*unit_cost(j)
        if cost <= budget:
            mass += probabilities[j]
            rounded += probabilities[j]
            budget -= cost
        elif budget > 0:
            split = budget/unit_cost(j)
            cutoff_mass = probabilities[j]
            mass += split
            budget = F(0)
            break
        else:
            break
    return mass, rounded, split, cutoff_mass


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


def brute_deterministic_tail_mass(support, probabilities, y, lower):
    """Largest tail event when the original atoms cannot be split."""
    best = F(0)
    for mask in range(1 << len(support)):
        chosen = [q for j, q in enumerate(probabilities) if mask & (1 << j)]
        imbalance = sum(probabilities[j]*(support[j]-y)
                        for j in range(len(support)) if mask & (1 << j))
        feasible = imbalance <= 0 if lower else imbalance >= 0
        if feasible:
            best = max(best, sum(chosen))
    return best


def dynamic_lower_tail_mass(support, probabilities, y):
    """Exact pseudo-polynomial signed-knapsack recursion."""
    gaps = tuple(q*(z-y) for z, q in zip(support, probabilities))
    scale = 1
    for value in tuple(probabilities)+gaps:
        scale = lcm(scale, value.denominator)
    costs = tuple(int(gap*scale) for gap in gaps)
    rewards = tuple(int(q*scale) for q in probabilities)
    states = {0: 0}
    for cost, reward in zip(costs, rewards):
        updated = states.copy()
        for gap, mass in states.items():
            updated[gap+cost] = max(updated.get(gap+cost, 0), mass+reward)
        states = updated
    return F(max(mass for gap, mass in states.items() if gap <= 0), scale)


def ordinary_knapsack_lower_tail_mass(support, probabilities, y):
    """Exact lower tail after including every atom at or below y."""
    low = tuple(j for j, z in enumerate(support) if z <= y)
    high = tuple(j for j, z in enumerate(support) if z > y)
    base = sum(probabilities[j] for j in low)
    budget = sum(probabilities[j]*(y-support[j]) for j in low)
    best = F(0)
    for mask in range(1 << len(high)):
        cost = sum(probabilities[j]*(support[j]-y)
                   for h, j in enumerate(high) if mask & (1 << h))
        profit = sum(probabilities[j]
                     for h, j in enumerate(high) if mask & (1 << h))
        if cost <= budget:
            best = max(best, profit)
    return base+best


def approximate_lower_tail_mass(support, probabilities, y, epsilon):
    """Profit-scaled knapsack FPTAS, retaining exact rational arithmetic."""
    check(F(0) < epsilon < F(1))
    low = tuple(j for j, z in enumerate(support) if z <= y)
    base = sum(probabilities[j] for j in low)
    budget = sum(probabilities[j]*(y-support[j]) for j in low)
    items = tuple((probabilities[j]*(support[j]-y), probabilities[j])
                  for j, z in enumerate(support)
                  if z > y and probabilities[j]*(z-y) <= budget)
    if not items:
        return base
    profit_max = max(profit for _, profit in items)
    unit = epsilon*profit_max/len(items)
    # rounded profit -> (least exact cost, exact profit of that witness)
    states = {0: (F(0), F(0))}
    for cost, profit in items:
        rounded = int(profit/unit)
        updated = states.copy()
        for score, (old_cost, old_profit) in states.items():
            new_cost, new_score = old_cost+cost, score+rounded
            if new_cost > budget:
                continue
            candidate = (new_cost, old_profit+profit)
            incumbent = updated.get(new_score)
            if (incumbent is None or candidate[0] < incumbent[0] or
                    (candidate[0] == incumbent[0] and candidate[1] > incumbent[1])):
                updated[new_score] = candidate
        states = updated
    return base+states[max(states)][1]


def certify_subset_sum_reduction(values, target):
    """Embed one SUBSET SUM instance in a fixed-space lower tail."""
    total = sum(values)
    check(0 < target < total)
    support = (F(1, 2)-F(target, 4*total),)+(F(3, 4),)*len(values)
    probabilities = (F(1, 2),)+tuple(F(value, 2*total) for value in values)
    mean = sum(z*q for z, q in zip(support, probabilities))
    check(mean == F(5, 8)-F(target, 8*total) > F(1, 2))
    deterministic = brute_deterministic_tail_mass(
        support, probabilities, F(1, 2), True,
    )
    reachable = {0}
    for value in values:
        reachable |= {subtotal+value for subtotal in tuple(reachable)}
    best = max(subtotal for subtotal in reachable if subtotal <= target)
    randomized = F(1, 2)+F(target, 2*total)
    check(deterministic == F(1, 2)+F(best, 2*total))
    check(deterministic == dynamic_lower_tail_mass(
        support, probabilities, F(1, 2),
    ))
    check((deterministic == randomized) == (target in reachable))
    return target in reachable


def brute_deterministic_band_mass(support, probabilities, lower, upper):
    """Three-way atom assignment for the largest two-sided tail event."""
    best = F(0)
    for assignment in product((0, 1, 2), repeat=len(support)):
        low_gap = sum(q*(z-lower) for z, q, cell in
                      zip(support, probabilities, assignment) if cell == 1)
        high_gap = sum(q*(z-upper) for z, q, cell in
                       zip(support, probabilities, assignment) if cell == 2)
        if low_gap <= 0 and high_gap >= 0:
            best = max(best, sum(q for q, cell in
                                 zip(probabilities, assignment) if cell))
    return best


def two_budget_band_mass(support, probabilities, lower, upper):
    """Exact two-capacity recursion after assigning the outer atoms."""
    low = tuple(j for j, z in enumerate(support) if z <= lower)
    high = tuple(j for j, z in enumerate(support) if z >= upper)
    middle = tuple(j for j, z in enumerate(support) if lower < z < upper)
    base = sum(probabilities[j] for j in low+high)
    low_capacity = sum(
        probabilities[j]*(lower-support[j]) for j in low
    )
    high_capacity = sum(
        probabilities[j]*(support[j]-upper) for j in high
    )
    low_costs = tuple(
        probabilities[j]*(support[j]-lower) for j in middle
    )
    high_costs = tuple(
        probabilities[j]*(upper-support[j]) for j in middle
    )
    scale = 1
    for value in (low_capacity, high_capacity)+low_costs+high_costs:
        scale = lcm(scale, value.denominator)
    low_limit = int(low_capacity*scale)
    high_limit = int(high_capacity*scale)
    low_costs = tuple(int(value*scale) for value in low_costs)
    high_costs = tuple(int(value*scale) for value in high_costs)
    states = {(0, 0): F(0)}
    for j, low_cost, high_cost in zip(middle, low_costs, high_costs):
        updated = states.copy()
        for (used_low, used_high), reward in states.items():
            if used_low+low_cost <= low_limit:
                state = (used_low+low_cost, used_high)
                updated[state] = max(
                    updated.get(state, F(0)), reward+probabilities[j],
                )
            if used_high+high_cost <= high_limit:
                state = (used_low, used_high+high_cost)
                updated[state] = max(
                    updated.get(state, F(0)), reward+probabilities[j],
                )
        states = updated
    return base+max(states.values())


def rounded_residual_band_lp(support, probabilities, middle, lower, upper,
                             low_capacity, high_capacity):
    """Round a vertex of the residual two-budget LP inward."""
    if not middle:
        return F(0), 0
    tolerance = 1e-8
    size = len(middle)
    low_costs = [float(probabilities[j]*(support[j]-lower)) for j in middle]
    high_costs = [float(probabilities[j]*(upper-support[j])) for j in middle]
    objective = [-float(probabilities[j]) for j in middle]*2
    constraints = [low_costs+[0.0]*size, [0.0]*size+high_costs]
    bounds = [float(low_capacity), float(high_capacity)]
    for h in range(size):
        row = [0.0]*(2*size)
        row[h] = row[size+h] = 1.0
        constraints.append(row)
        bounds.append(1.0)
    solution = linprog(
        objective, A_ub=constraints, b_ub=bounds,
        bounds=(0.0, 1.0), method="highs",
    )
    check(solution.success)
    selected_low, selected_high, fractional = [], [], 0
    for h, j in enumerate(middle):
        x, y = solution.x[h], solution.x[size+h]
        if x >= 1-tolerance and y <= tolerance:
            selected_low.append(j)
        elif y >= 1-tolerance and x <= tolerance:
            selected_high.append(j)
        elif x > tolerance or y > tolerance:
            fractional += 1
    check(fractional <= 2)
    exact_low = sum(
        probabilities[j]*(support[j]-lower) for j in selected_low
    )
    exact_high = sum(
        probabilities[j]*(upper-support[j]) for j in selected_high
    )
    check(exact_low <= low_capacity and exact_high <= high_capacity)
    reward = sum(probabilities[j] for j in selected_low+selected_high)
    return reward, fractional


def approximate_two_budget_band_mass(support, probabilities, lower, upper,
                                     epsilon):
    """Heavy-atom enumeration plus LP-vertex rounding for a band."""
    low = tuple(j for j, z in enumerate(support) if z <= lower)
    high = tuple(j for j, z in enumerate(support) if z >= upper)
    middle = tuple(j for j, z in enumerate(support) if lower < z < upper)
    base = sum(probabilities[j] for j in low+high)
    low_capacity = sum(
        probabilities[j]*(lower-support[j]) for j in low
    )
    high_capacity = sum(
        probabilities[j]*(support[j]-upper) for j in high
    )
    cutoff = epsilon/2
    heavy = tuple(j for j in middle if probabilities[j] > cutoff)
    light = tuple(j for j in middle if probabilities[j] <= cutoff)
    best, largest_fractional_count = base, 0
    for assignment in product((0, 1, 2), repeat=len(heavy)):
        low_used = sum(
            probabilities[j]*(support[j]-lower)
            for j, cell in zip(heavy, assignment) if cell == 1
        )
        high_used = sum(
            probabilities[j]*(upper-support[j])
            for j, cell in zip(heavy, assignment) if cell == 2
        )
        if low_used > low_capacity or high_used > high_capacity:
            continue
        heavy_reward = sum(
            probabilities[j] for j, cell in zip(heavy, assignment) if cell
        )
        light_reward, fractional = rounded_residual_band_lp(
            support, probabilities, light, lower, upper,
            low_capacity-low_used, high_capacity-high_used,
        )
        largest_fractional_count = max(largest_fractional_count, fractional)
        best = max(best, base+heavy_reward+light_reward)
    return best, len(heavy), largest_fractional_count


def relative_two_budget_band_mass(support, probabilities, lower, upper,
                                  epsilon):
    """Enumerate high-profit atoms and LP-round the residual problem."""
    low = tuple(j for j, z in enumerate(support) if z <= lower)
    high = tuple(j for j, z in enumerate(support) if z >= upper)
    middle = tuple(j for j, z in enumerate(support) if lower < z < upper)
    base = sum(probabilities[j] for j in low+high)
    low_capacity = sum(
        probabilities[j]*(lower-support[j]) for j in low
    )
    high_capacity = sum(
        probabilities[j]*(support[j]-upper) for j in high
    )
    heavy_count = (2*epsilon.denominator+epsilon.numerator-1)//epsilon.numerator
    best, largest_fractional_count = base, 0
    for count in range(min(heavy_count, len(middle))+1):
        for selected in combinations(middle, count):
            selected_set = set(selected)
            for assignment in product((1, 2), repeat=count):
                low_used = sum(
                    probabilities[j]*(support[j]-lower)
                    for j, cell in zip(selected, assignment) if cell == 1
                )
                high_used = sum(
                    probabilities[j]*(upper-support[j])
                    for j, cell in zip(selected, assignment) if cell == 2
                )
                if low_used > low_capacity or high_used > high_capacity:
                    continue
                reward = sum(probabilities[j] for j in selected)
                candidate = base+reward
                if count == heavy_count:
                    cutoff = min(probabilities[j] for j in selected)
                    residual = tuple(
                        j for j in middle
                        if j not in selected_set and probabilities[j] <= cutoff
                    )
                    residual_reward, fractional = rounded_residual_band_lp(
                        support, probabilities, residual, lower, upper,
                        low_capacity-low_used, high_capacity-high_used,
                    )
                    candidate += residual_reward
                    largest_fractional_count = max(
                        largest_fractional_count, fractional,
                    )
                best = max(best, candidate)
    return best, heavy_count, largest_fractional_count


def certify_two_sided_additive_scheme():
    """Numerical LP certificate for the additive two-budget scheme."""
    support = (F(0), F(1, 5), F(1, 2), F(4, 5), F(1))
    weight_patterns = (
        (1, 1, 1, 1, 1), (1, 2, 3, 4, 5), (5, 4, 3, 2, 1),
        (2, 5, 1, 4, 3), (3, 1, 5, 2, 4), (4, 2, 1, 5, 3),
    )
    checks = strict = largest_heavy = largest_fractional = 0
    for weights in weight_patterns:
        probabilities = tuple(F(w, sum(weights)) for w in weights)
        mean = sum(z*q for z, q in zip(support, probabilities))
        lower, upper = mean/2, (1+mean)/2
        exact = brute_deterministic_band_mass(
            support, probabilities, lower, upper,
        )
        for epsilon in (F(1, 2), F(1, 3)):
            approximate, heavy, fractional = approximate_two_budget_band_mass(
                support, probabilities, lower, upper, epsilon,
            )
            check(exact-epsilon <= approximate <= exact)
            strict += int(approximate < exact)
            largest_heavy = max(largest_heavy, heavy)
            largest_fractional = max(largest_fractional, fractional)
            checks += 1
    check(strict > 0)
    return checks, strict, largest_heavy, largest_fractional


def certify_two_sided_relative_scheme():
    """LP certificate for the relative two-budget approximation scheme."""
    support = tuple(F(j, 9) for j in range(10))
    weight_patterns = (
        (9, 8, 2, 9, 5, 9, 4, 4, 7, 2),
        (2, 7, 4, 4, 9, 5, 9, 2, 8, 9),
        (1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        (2, 5, 1, 8, 3, 7, 4, 9, 6, 2),
    )
    checks = strict = largest_fractional = 0
    epsilon = F(2, 3)
    for h, weights in enumerate(weight_patterns):
        probabilities = tuple(F(w, sum(weights)) for w in weights)
        mean = sum(z*q for z, q in zip(support, probabilities))
        if h == 0:
            lower, upper = F(115, 354), F(1823, 2124)
        else:
            lower, upper = mean/2, (1+mean)/2
        check(lower < mean < upper)
        exact = brute_deterministic_band_mass(
            support, probabilities, lower, upper,
        )
        approximate, enumerated, fractional = relative_two_budget_band_mass(
            support, probabilities, lower, upper, epsilon,
        )
        check(enumerated == 3)
        check((1-epsilon)*exact <= approximate <= exact)
        strict += int(approximate < exact)
        largest_fractional = max(largest_fractional, fractional)
        checks += 1
    check(strict == 2 and largest_fractional == 2)
    return checks, strict, largest_fractional


def band_saturates_by_one_budget(support, probabilities, lower, upper):
    """Decide exact full saturation by one low-budget knapsack."""
    low = tuple(j for j, z in enumerate(support) if z <= lower)
    high = tuple(j for j, z in enumerate(support) if z >= upper)
    middle = tuple(j for j, z in enumerate(support) if lower < z < upper)
    low_capacity = sum(
        probabilities[j]*(lower-support[j]) for j in low
    )
    high_capacity = sum(
        probabilities[j]*(support[j]-upper) for j in high
    )
    low_costs = tuple(
        probabilities[j]*(support[j]-lower) for j in middle
    )
    high_costs = tuple(
        probabilities[j]*(upper-support[j]) for j in middle
    )
    scale = 1
    for value in (low_capacity, high_capacity)+low_costs+high_costs:
        scale = lcm(scale, value.denominator)
    low_limit = int(low_capacity*scale)
    high_limit = int(high_capacity*scale)
    low_costs = tuple(int(value*scale) for value in low_costs)
    high_costs = tuple(int(value*scale) for value in high_costs)
    states = [-1]*(low_limit+1)
    states[0] = 0
    for low_cost, high_reward in zip(low_costs, high_costs):
        updated = states.copy()
        for used_low, old_reward in enumerate(states):
            if old_reward >= 0 and used_low+low_cost <= low_limit:
                updated[used_low+low_cost] = max(
                    updated[used_low+low_cost], old_reward+high_reward,
                )
        states = updated
    return max(states) >= sum(high_costs)-high_limit


def certify_band_saturation_reduction(values, target):
    """Reduce one SUBSET SUM instance to symmetric band saturation."""
    total = sum(values)
    check(0 < target < total)
    augmented = values+(2*total-target, total+target)
    augmented_total = sum(augmented)
    check(augmented_total == 4*total)
    support = (F(0),)+(F(1, 2),)*len(augmented)+(F(1),)
    probabilities = ((F(1, 4),)+
                     tuple(F(value, 2*augmented_total) for value in augmented)+
                     (F(1, 4),))
    lower, upper = F(1, 4), F(3, 4)
    check(sum(probabilities) == 1)
    check(sum(z*q for z, q in zip(support, probabilities)) == F(1, 2))
    reachable = {0}
    for value in values:
        reachable |= {subtotal+value for subtotal in tuple(reachable)}
    augmented_reachable = {0}
    for value in augmented:
        augmented_reachable |= {
            subtotal+value for subtotal in tuple(augmented_reachable)
        }
    attainable = target in reachable
    check((2*total in augmented_reachable) == attainable)
    check(band_saturates_by_one_budget(
        support, probabilities, lower, upper,
    ) == attainable)
    exhaustive = len(values) == 4
    if exhaustive:
        check((brute_deterministic_band_mass(
            support, probabilities, lower, upper,
        ) == 1) == attainable)
    return attainable, exhaustive


def brute_partition_band_mass(support, probabilities, lower, upper):
    """Independent exhaustive enumeration of every information partition."""
    best, size = F(0), len(support)
    for labels in product(range(size), repeat=size):
        if labels[0] or any(labels[j] > 1+max(labels[:j]) for j in range(1, size)):
            continue
        event = F(0)
        for label in range(1+max(labels)):
            members = [j for j in range(size) if labels[j] == label]
            mass = sum(probabilities[j] for j in members)
            first = sum(probabilities[j]*support[j] for j in members)
            if first <= lower*mass or first >= upper*mass:
                event += mass
        best = max(best, event)
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


def refine_atomic_space_at_quantile_cuts(support, probabilities, cuts):
    """Split canonical atoms at prescribed randomized-quantile cuts."""
    refined_support, refined_probabilities = [], []
    cumulative = F(0)
    for z, probability in zip(support, probabilities):
        endpoints = [cumulative]
        endpoints.extend(
            cut for cut in cuts if cumulative < cut < cumulative+probability
        )
        endpoints.append(cumulative+probability)
        endpoints.sort()
        for left, right in zip(endpoints, endpoints[1:]):
            refined_support.append(z)
            refined_probabilities.append(right-left)
        cumulative += probability
    check(cumulative == 1)
    check(sum(refined_probabilities) == 1)
    return tuple(refined_support), tuple(refined_probabilities)


def piecewise_linear_quantile_integral(knots, values, left, right):
    """Integrate a continuous piecewise-linear quantile exactly."""
    check(F(0) <= left <= right <= F(1))
    first = F(0)
    for start, end, low, high in zip(knots, knots[1:], values, values[1:]):
        segment_left, segment_right = max(left, start), min(right, end)
        if segment_left >= segment_right:
            continue
        slope = (high-low)/(end-start)
        first += low*(segment_right-segment_left)
        first += slope*((segment_right-start)**2-(segment_left-start)**2)/2
    return first


def piecewise_linear_quantile_value(knots, values, probability):
    """Evaluate the same exact quantile, including plateau interiors."""
    check(F(0) <= probability <= F(1))
    if probability == 1:
        return values[-1]
    for start, end, low, high in zip(knots, knots[1:], values, values[1:]):
        if start <= probability < end:
            return low+(high-low)*(probability-start)/(end-start)
    raise AssertionError("quantile coordinate not covered")


def certify_mixed_nonatomic_quantile_witnesses():
    """Exact tail witnesses for a nonatomic base with mixed score law."""
    knots = tuple(map(F, (0, F(1, 5), F(2, 5), F(3, 5), F(4, 5), 1)))
    values = tuple(map(F, (0, 0, F(1, 3), F(1, 3), 1, 1)))
    integral = lambda left, right: piecewise_linear_quantile_integral(
        knots, values, left, right,
    )
    value = lambda probability: piecewise_linear_quantile_value(
        knots, values, probability,
    )
    mean = integral(F(0), F(1))
    check(mean == F(13, 30))

    plateau_cuts = 0
    is_plateau_cut = lambda cut: any(
        start < cut < end and low == high
        for start, end, low, high in zip(knots, knots[1:], values, values[1:])
    )
    disjoint_pairs = (
        (F(1, 4), F(11, 20)), (F(1, 4), F(13, 20)),
        (F(1, 4), F(3, 4)), (F(1, 2), F(11, 20)),
        (F(1, 2), F(13, 20)), (F(1, 2), F(3, 4)),
        (F(13, 20), F(3, 4)),
    )
    for lower_cut, upper_cut in disjoint_pairs:
        lower = integral(F(0), lower_cut)/lower_cut
        upper = integral(upper_cut, F(1))/(1-upper_cut)
        check(lower < mean < upper)
        check(value(lower_cut) > lower)
        check(value(upper_cut) < upper)
        check(lower_cut <= upper_cut)
        middle = integral(lower_cut, upper_cut)/(upper_cut-lower_cut)
        check(lower < middle < upper)
        plateau_cuts += is_plateau_cut(lower_cut)+is_plateau_cut(upper_cut)

    shared_cuts = (F(9, 20), F(1, 2), F(11, 20))
    for cut in shared_cuts:
        lower = integral(F(0), cut)/cut
        upper = integral(cut, F(1))/(1-cut)
        check(lower < mean < upper)
        check(value(cut) > lower and value(cut) < upper)
        check(is_plateau_cut(cut))
        plateau_cuts += 1
    return len(disjoint_pairs), len(shared_cuts), plateau_cuts, mean


def certify_finite_action_purifications():
    """Purify three-action kernels for a continuous uniform score."""
    interval_mass = lambda left, right: right-left
    interval_first = lambda left, right: (right**2-left**2)/2
    thirds = (
        ((F(0), F(1, 3)),),
        ((F(1, 3), F(2, 3)),),
        ((F(2, 3), F(1)),),
    )
    base_first = tuple(
        sum(interval_first(left, right) for left, right in cells)
        for cells in thirds
    )
    check(base_first == (F(1, 18), F(1, 6), F(5, 18)))

    checks, largest_randomized_share = 0, F(0)
    for swap in (F(1, 18), F(1, 12), F(1, 9), F(1, 6)):
        randomized_share = 3*swap+9*swap**2
        check(0 < randomized_share < 1)
        randomized_masses = tuple(
            (1-randomized_share)*F(1, 3)+randomized_share*F(1, 3)
            for _ in thirds
        )
        randomized_first = tuple(
            (1-randomized_share)*first+randomized_share*F(1, 6)
            for first in base_first
        )

        pure_cells = (
            ((F(0), F(1, 3)-swap), (F(2, 3), F(2, 3)+swap)),
            ((F(1, 3), F(2, 3)),),
            ((F(1, 3)-swap, F(1, 3)), (F(2, 3)+swap, F(1))),
        )
        pure_masses = tuple(
            sum(interval_mass(left, right) for left, right in cells)
            for cells in pure_cells
        )
        pure_first = tuple(
            sum(interval_first(left, right) for left, right in cells)
            for cells in pure_cells
        )
        check(pure_masses == randomized_masses == (F(1, 3),)*3)
        check(pure_first == randomized_first)
        means = tuple(first/mass for first, mass in zip(pure_first, pure_masses))
        check(means[0] < means[1] < means[2])
        check(sum(mass*mean for mass, mean in zip(pure_masses, means)) == F(1, 2))
        breakpoints = (F(0),)+means+(F(1),)
        for left, right in zip(breakpoints, breakpoints[1:]):
            midpoint = (left+right)/2
            active_mass = sum(
                mass for mass, mean in zip(pure_masses, means)
                if mean > midpoint
            )
            candidates = [left, right]
            stationary = 1-active_mass
            if left <= stationary <= right:
                candidates.append(stationary)
            for strike in candidates:
                target_call = sum(
                    mass*max(mean-strike, 0)
                    for mass, mean in zip(pure_masses, means)
                )
                uniform_call = (1-strike)**2/2
                check(target_call <= uniform_call)
        checks += 1
        largest_randomized_share = max(largest_randomized_share, randomized_share)
    return checks, largest_randomized_share


def certify_continuous_backward_monge_maps():
    """Exact non-finite contractions of a uniform score."""
    maps, largest_multiplicity = 0, 0
    smallest_variance = F(1, 12)
    for blocks in range(2, 7):
        left = F(blocks-1, 2*blocks)
        right = F(blocks+1, 2*blocks)
        target_variance = F(1, 12*blocks**2)
        check((left+right)/2 == F(1, 2))
        check(target_variance < F(1, 12))

        # For m=u+left, the k preimages are u+j/k and have mean m.
        for sample in range(12):
            u = F(sample, 12*blocks)
            preimages = tuple(u+F(j, blocks) for j in range(blocks))
            image = u+left
            check(sum(preimages, F(0))/blocks == image)
            check(all(
                y-F(min(int(y*blocks), blocks-1), blocks)+left == image
                for y in preimages
            ))

        # Complete stop-loss check. Outside [left,right] it is immediate;
        # inside, source-call minus target-call is concave, so its minimum
        # is at an endpoint.
        source_call = lambda strike: (1-strike)**2/2
        target_call = lambda strike: blocks*(right-strike)**2/2
        check(1-blocks < 0)
        check(target_call(left) == F(1, 2)-left)
        check(source_call(left) >= target_call(left))
        check(source_call(right) >= target_call(right) == 0)

        maps += 1
        largest_multiplicity = blocks
        smallest_variance = min(smallest_variance, target_variance)
    return maps, largest_multiplicity, smallest_variance


def certify_countable_support_mixed_contraction():
    """A countably infinite target that mixes score atoms and continuum."""
    # The source is 1/2 delta_0 + 1/2 Uniform[0,1].  Cell n takes mass
    # 2^(-n-1) from each source component.  On the continuous component
    # it uses the dyadic interval (2^-n, 2^(-n+1)].
    checked_atoms = 12
    prefix_mass = prefix_first = F(0)
    for n in range(1, checked_atoms+1):
        continuous_left = F(1, 2**n)
        continuous_right = F(1, 2**(n-1))
        atomic_mass = F(1, 2**(n+1))
        continuous_mass = (continuous_right-continuous_left)/2
        continuous_first = (continuous_right**2-continuous_left**2)/4
        target_mass = F(1, 2**n)
        target_value = F(3, 2**(n+2))
        check(atomic_mass == continuous_mass)
        check(atomic_mass+continuous_mass == target_mass)
        check(continuous_first == target_mass*target_value)
        prefix_mass += target_mass
        prefix_first += continuous_first

    tail_mass = F(1, 2**checked_atoms)
    tail_first = F(1, 4**(checked_atoms+1))
    check(prefix_mass+tail_mass == 1)
    check(prefix_first+tail_first == F(1, 4))

    # Infinite geometric sums: E[M^2]=9/16 sum_{n>=1} 8^-n.
    target_second = F(9, 16)*F(1, 7)
    target_variance = target_second-F(1, 4)**2
    source_variance = F(1, 2)*F(1, 3)-F(1, 4)**2
    check(target_variance == F(1, 56))
    check(source_variance == F(5, 48))
    check(target_variance/source_variance == F(6, 35))
    return checked_atoms, tail_mass, target_variance, target_variance/source_variance


def certify_wasserstein_dense_mixed_contractions():
    """Finite contractions converging to a noncomponentwise mixed-law target."""
    # Source: 1/2 delta_0 + 1/2 Uniform[0,1].
    # Target: Uniform[0,1/2].  Its call is below the source call because
    # the gap on [0,1/2] is t(2-3t)/4 and on [1/2,1] is (1-t)^2/4.
    check(F(0)*(2-3*F(0))/4 == 0)
    check(F(1, 2)*(2-3*F(1, 2))/4 == F(1, 16) > 0)
    check(-F(3, 2) < 0)  # concavity of the first gap polynomial
    check((1-F(1, 2))**2/4 == F(1, 16))
    check((1-F(1))**2/4 == 0)

    cases = largest_support = 0
    smallest_error = F(1)
    target_variance = F(1, 2)**2/F(12)
    source_variance = F(1, 2)*F(1, 3)-F(1, 4)**2
    check(target_variance == F(1, 48))
    check(source_variance == F(5, 48))
    for support_size in (2, 4, 8, 16, 32, 64):
        target_points = tuple(
            F(2*k+1, 4*support_size) for k in range(support_size)
        )
        probability = F(1, support_size)
        mean = sum(target_points, F(0))*probability
        second = sum(x*x for x in target_points)*probability
        variance = second-mean**2
        wasserstein_error = F(1, 8*support_size)

        check(mean == F(1, 4))
        check(variance == F(support_size**2-1,
                            48*support_size**2))
        check(target_variance-variance == F(1, 48*support_size**2))

        # Each point is the barycenter of one equal-probability target cell.
        for k, point in enumerate(target_points):
            left = F(k, 2*support_size)
            right = F(k+1, 2*support_size)
            check(2*(right-left) == probability)
            check((right**2-left**2) == probability*point)

        cases += 1
        largest_support = support_size
        smallest_error = min(smallest_error, wasserstein_error)
    check(smallest_error == F(1, 512))
    return cases, largest_support, smallest_error, target_variance, source_variance


def equal_mass_quantization(points, masses, cells):
    """Exact W1 error of scalar equal-mass conditional-mean quantization."""
    check(tuple(points) == tuple(sorted(points)))
    check(sum(masses, F(0)) == 1)
    intervals = []
    cumulative = F(0)
    for point, mass in zip(points, masses):
        intervals.append((point, cumulative, cumulative+mass))
        cumulative += mass

    cell_means = []
    for cell in range(cells):
        left, right = F(cell, cells), F(cell+1, cells)
        first = F(0)
        for point, atom_left, atom_right in intervals:
            overlap = max(F(0), min(right, atom_right)-max(left, atom_left))
            first += overlap*point
        cell_means.append(cells*first)

    error = F(0)
    for cell, cell_mean in enumerate(cell_means):
        left, right = F(cell, cells), F(cell+1, cells)
        for point, atom_left, atom_right in intervals:
            overlap = max(F(0), min(right, atom_right)-max(left, atom_left))
            error += overlap*abs(point-cell_mean)
    return error


def certify_unbounded_quantization_rate():
    """The p=2 moment rate and failure of a uniform L1-only rate."""
    fixtures = (
        ((F(-3), F(-1), F(2), F(7)),
         (F(1, 8), F(3, 8), F(1, 4), F(1, 4))),
        ((F(0), F(1), F(10)), (F(3, 4), F(1, 8), F(1, 8))),
        ((F(-8), F(-2), F(0), F(1), F(9)),
         (F(1, 16), F(3, 16), F(1, 2), F(1, 8), F(1, 8))),
        ((F(-5), F(3)), (F(2, 3), F(1, 3))),
    )
    moment_cases = 0
    largest_squared_ratio = F(0)
    for points, masses in fixtures:
        mean = sum((mass*point for point, mass in zip(points, masses)), F(0))
        variance = sum((mass*(point-mean)**2
                        for point, mass in zip(points, masses)), F(0))
        check(variance > 0)
        for cells in (2, 4, 8, 16):
            error = equal_mass_quantization(points, masses, cells)
            squared_ratio = cells*error**2/(4*variance)
            check(squared_ratio <= 1)
            largest_squared_ratio = max(largest_squared_ratio, squared_ratio)
            moment_cases += 1

    l1_cases = 0
    largest_l1_error = F(0)
    for cells in (2, 4, 8, 16, 32):
        epsilon = F(1, cells**3)
        points = (F(0), F(cells**3))
        masses = (1-epsilon, epsilon)
        error = equal_mass_quantization(points, masses, cells)
        check(sum((mass*abs(point)
                   for point, mass in zip(points, masses)), F(0)) == 1)
        check(error == 2*(1-F(1, cells**2)))
        largest_l1_error = max(largest_l1_error, error)
        l1_cases += 1
    check(largest_l1_error == F(1023, 512))
    return moment_cases, largest_squared_ratio, l1_cases, largest_l1_error


def power_quantile_equal_mass_error(cells, power):
    """Exact W1 error for Q(u)=u**power and equal-mass barycenters."""
    check(cells >= 2 and power > 0)
    total = 0.0
    for cell in range(cells):
        left, right = cell/cells, (cell+1)/cells
        mean = cells*(right**(power+1)-left**(power+1))/(power+1)
        crossing = mean**(1/power)
        check(left < crossing < right)
        total += 2*((right**(power+1)-crossing**(power+1))/(power+1)
                    - mean*(right-crossing))
    return total


def direct_power_quantile_equal_mass_error(cells, power):
    """Independent quadrature of the bounded power-quantile example."""
    check(cells >= 2 and power > 0)
    total = 0.0
    for cell in range(cells):
        left, right = cell/cells, (cell+1)/cells
        mean = cells*quad(
            lambda u: u**power, left, right,
            epsabs=2e-13, epsrel=2e-13,
        )[0]
        crossing = mean**(1/power)
        total += quad(
            lambda u: abs(u**power-mean), left, right,
            points=[crossing], epsabs=2e-13, epsrel=2e-13,
        )[0]
    return total


def certify_bounded_quantization_rate():
    """Sharp one-quarter rate for bounded Sobolev quantiles."""
    formula_cases = 0
    largest_quadrature_error = 0.0
    for power in (0.5, 1.0, 2.0):
        for cells in (2, 4, 8, 16, 32, 64, 128):
            closed = power_quantile_equal_mass_error(cells, power)
            direct = direct_power_quantile_equal_mass_error(cells, power)
            discrepancy = abs(closed-direct)
            check(discrepancy < 2e-12)
            largest_quadrature_error = max(
                largest_quadrature_error, discrepancy,
            )
            formula_cases += 1

    singular_ratios = []
    for cells in (256, 1024, 4096, 16384, 65536):
        error = power_quantile_equal_mass_error(cells, 0.5)
        singular_ratios.append(cells*error)
    check(all(singular_ratios[j+1] > singular_ratios[j]
              for j in range(len(singular_ratios)-1)))
    check(0.2497 < singular_ratios[-1] < 0.25)
    check(abs(power_quantile_equal_mass_error(128, 1.0)-1/512) < 2e-14)
    return (formula_cases, largest_quadrature_error,
            power_quantile_equal_mass_error(128, 0.5),
            power_quantile_equal_mass_error(65536, 0.5),
            singular_ratios[-1])


def certify_bounded_atomic_rate_obstruction():
    """A Bernoulli jump prevents a universal bounded-support rate limit."""
    cases = 0
    for cells in range(2, 18):
        error = equal_mass_quantization(
            (F(0), F(1)), (F(1, 2), F(1, 2)), cells,
        )
        expected = F(0) if cells % 2 == 0 else F(1, 2*cells)
        check(error == expected)
        check(cells*error == (F(0) if cells % 2 == 0 else F(1, 2)))
        cases += 1
    return cases, F(0), F(1, 2)


def exponential_equal_mass_error(cells):
    """Closed-form W1 error for unit-exponential equal-mass barycenters."""
    check(cells >= 2)
    total = 1/mp.e
    for r in range(1, cells):
        rr = mp.mpf(r)
        total += ((rr+1)*mp.exp(rr*mp.log1p(1/rr)-1)
                  - rr*(rr+1)*mp.log1p(1/rr))
    return 2*total/cells


def direct_exponential_equal_mass_error(cells):
    """Independent quadrature of the same conditional-mean construction."""
    total = 0.0
    for cell in range(cells):
        left, right = cell/cells, (cell+1)/cells
        mean = cells*quad(
            lambda u: -log1p(-u), left, right,
            epsabs=1e-13, epsrel=1e-13, limit=100,
        )[0]
        crossing = 1-exp(-mean)
        total += quad(
            lambda u: abs(-log1p(-u)-mean), left, right,
            points=[crossing], epsabs=1e-13, epsrel=1e-13, limit=100,
        )[0]
    return total


def certify_exponential_quantization_rate():
    """Exact formula and logarithmic-over-linear exponential benchmark."""
    mp.mp.dps = 60
    formula_cases = 0
    largest_quadrature_error = 0.0
    error_at_128 = None
    for cells in (2, 4, 8, 16, 32, 64, 128):
        closed = float(exponential_equal_mass_error(cells))
        direct = direct_exponential_equal_mass_error(cells)
        discrepancy = abs(closed-direct)
        check(discrepancy < 2e-12)
        largest_quadrature_error = max(largest_quadrature_error, discrepancy)
        formula_cases += 1
        if cells == 128:
            error_at_128 = closed

    # The cell summand is 1/(8r)-1/(16r^2)+O(r^-3).  These finite
    # checks supplement, but do not replace, the analytic expansion.
    remainders = []
    for r in (32, 64, 128, 256, 512, 1024, 2048, 4096):
        rr = mp.mpf(r)
        term = ((rr+1)*mp.exp(rr*mp.log1p(1/rr)-1)
                - rr*(rr+1)*mp.log1p(1/rr))
        remainders.append(abs(rr**2*(term-1/(8*rr))+mp.mpf(1)/16))
    check(all(remainders[j+1] < remainders[j]
              for j in range(len(remainders)-1)))

    normalized_ratios = []
    for cells in (256, 1024, 4096, 16384):
        error = exponential_equal_mass_error(cells)
        normalized_ratios.append(cells*error/mp.log(cells))
    check(all(normalized_ratios[j+1] < normalized_ratios[j]
              for j in range(len(normalized_ratios)-1)))
    check(normalized_ratios[-1] > mp.mpf(1)/4)
    error_at_16384 = exponential_equal_mass_error(16384)
    ratio_at_16384 = normalized_ratios[-1]
    return (formula_cases, largest_quadrature_error, error_at_128,
            error_at_16384, ratio_at_16384)


def normal_pdf(x):
    """Standard normal density, including the value at infinity."""
    return 0.0 if x == inf else exp(-x*x/2)/sqrt(2*pi)


def gaussian_equal_mass_error(cells):
    """Exact W1 formula for even-cell standard-normal barycenters."""
    check(cells >= 2 and cells % 2 == 0)
    half_error = 0.0
    for r in range(cells//2):
        upper = inf if r == 0 else -ndtri(r/cells)
        lower = -ndtri((r+1)/cells)
        upper_density = normal_pdf(upper)
        mean = cells*(normal_pdf(lower)-upper_density)
        check(lower < mean < upper)
        half_error += (normal_pdf(mean)-upper_density
                       - mean*(ndtr(-mean)-r/cells))
    return 4*half_error


def direct_gaussian_equal_mass_error(cells):
    """Independent normal-density quadrature of the same construction."""
    check(cells >= 2 and cells % 2 == 0)
    half_error = 0.0
    for r in range(cells//2):
        upper = inf if r == 0 else -ndtri(r/cells)
        lower = -ndtri((r+1)/cells)
        first = quad(
            lambda x: x*normal_pdf(x), lower, upper,
            epsabs=2e-13, epsrel=2e-13, limit=100,
        )[0]
        mean = cells*first
        half_error += quad(
            lambda x: (mean-x)*normal_pdf(x), lower, mean,
            epsabs=2e-13, epsrel=2e-13, limit=100,
        )[0]
        half_error += quad(
            lambda x: (x-mean)*normal_pdf(x), mean, upper,
            epsabs=2e-13, epsrel=2e-13, limit=100,
        )[0]
    return 2*half_error


def certify_gaussian_quantization_rate():
    """Exact Gaussian formula and its square-root-logarithmic rate."""
    formula_cases = 0
    largest_quadrature_error = 0.0
    error_at_128 = None
    for cells in (2, 4, 8, 16, 32, 64, 128):
        closed = gaussian_equal_mass_error(cells)
        direct = direct_gaussian_equal_mass_error(cells)
        discrepancy = abs(closed-direct)
        check(discrepancy < 2e-12)
        largest_quadrature_error = max(
            largest_quadrature_error, discrepancy,
        )
        formula_cases += 1
        if cells == 128:
            error_at_128 = closed

    normalized_ratios = []
    quantile_ratios = []
    for cells in (256, 1024, 4096, 16384, 65536):
        error = gaussian_equal_mass_error(cells)
        normalized_ratios.append(cells*error/sqrt(2*log(cells)))
        quantile_ratios.append(2*cells*error/(-ndtri(1/cells)))
    check(all(normalized_ratios[j+1] < normalized_ratios[j]
              for j in range(len(normalized_ratios)-1)))
    check(all(quantile_ratios[j+1] < quantile_ratios[j]
              for j in range(len(quantile_ratios)-1)))
    check(normalized_ratios[-1] > 0.5)
    check(quantile_ratios[-1] > 1.0)
    return (formula_cases, largest_quadrature_error, error_at_128,
            gaussian_equal_mass_error(65536), normalized_ratios[-1],
            quantile_ratios[-1])


def logistic_quantile_antiderivative(u):
    """Antiderivative of log(u/(1-u)), continuously extended."""
    if u == 0.0 or u == 1.0:
        return 0.0
    return u*log(u)+(1-u)*log1p(-u)


def logistic_equal_mass_error(cells):
    """Exact W1 error for unit-scale logistic barycenters."""
    check(cells >= 2)
    total = 0.0
    for cell in range(cells):
        left, right = cell/cells, (cell+1)/cells
        mean = cells*(logistic_quantile_antiderivative(right)
                      - logistic_quantile_antiderivative(left))
        crossing = expit(mean)
        check(left < crossing < right)
        total += 2*(logistic_quantile_antiderivative(right)
                    - logistic_quantile_antiderivative(crossing)
                    - mean*(right-crossing))
    return total


def direct_logistic_equal_mass_error(cells):
    """Independent quadrature of the logistic quantile construction."""
    check(cells >= 2)
    total = 0.0
    quantile = lambda u: log(u)-log1p(-u)
    for cell in range(cells):
        left, right = cell/cells, (cell+1)/cells
        mean = cells*quad(
            quantile, left, right,
            epsabs=2e-13, epsrel=2e-13, limit=100,
        )[0]
        crossing = expit(mean)
        total += quad(
            lambda u: abs(quantile(u)-mean), left, right,
            points=[crossing], epsabs=2e-13, epsrel=2e-13, limit=100,
        )[0]
    return total


def certify_slow_tail_quantization_rate():
    """Logistic certificate for the slowly varying trimmed-range theorem."""
    formula_cases = 0
    largest_quadrature_error = 0.0
    error_at_128 = None
    for cells in (2, 4, 8, 16, 32, 64, 128):
        closed = logistic_equal_mass_error(cells)
        direct = direct_logistic_equal_mass_error(cells)
        discrepancy = abs(closed-direct)
        check(discrepancy < 2e-12)
        largest_quadrature_error = max(
            largest_quadrature_error, discrepancy,
        )
        formula_cases += 1
        if cells == 128:
            error_at_128 = closed

    normalized_ratios = []
    for cells in (256, 1024, 4096, 16384, 65536):
        error = logistic_equal_mass_error(cells)
        normalized_ratios.append(2*cells*error/log(cells-1))
    check(all(normalized_ratios[j+1] < normalized_ratios[j]
              for j in range(len(normalized_ratios)-1)))
    check(normalized_ratios[-1] > 1.0)
    return (formula_cases, largest_quadrature_error, error_at_128,
            logistic_equal_mass_error(65536), normalized_ratios[-1])


def pareto_cell_term(r, alpha):
    """Dimensionless cell error for a unit-scale Pareto law."""
    alpha = mp.mpf(alpha)
    check(alpha > 1)
    beta = 1-1/alpha
    rr = mp.mpf(r)
    increment = ((rr+1)**beta-rr**beta)/beta
    return ((1-beta)/beta*increment**(-(alpha-1))
            - rr**beta/beta+rr*increment)


def pareto_equal_mass_error(cells, alpha):
    """Closed-form W1 error for unit-scale Pareto barycenters."""
    alpha = mp.mpf(alpha)
    check(cells >= 2 and alpha > 1)
    beta = 1-1/alpha
    total = mp.fsum(pareto_cell_term(r, alpha) for r in range(cells))
    return 2*cells**(-beta)*total


def direct_pareto_equal_mass_error(cells, alpha):
    """Independent regularized quadrature of the Pareto construction."""
    beta = 1-1/alpha
    power = 1/beta
    exponent = 1/(alpha-1)
    total = 0.0
    for cell in range(cells):
        left = (cell/cells)**beta
        right = ((cell+1)/cells)**beta
        mean = cells*power*(right-left)
        crossing = mean**(-(alpha-1))
        check(left < crossing < right)
        total += quad(
            lambda y: power*abs(1-mean*y**exponent), left, right,
            points=[crossing], epsabs=1e-13, epsrel=1e-13,
        )[0]
    return total


def certify_pareto_quantization_rate():
    """Exact Pareto formula and its boundary-moment asymptotic rate."""
    mp.mp.dps = 60
    formula_cases = 0
    largest_quadrature_error = 0.0
    for alpha in (1.25, 1.5, 2.0, 3.0):
        for cells in (2, 4, 8, 16, 32, 64):
            closed = float(pareto_equal_mass_error(cells, alpha))
            direct = direct_pareto_equal_mass_error(cells, alpha)
            discrepancy = abs(closed-direct)
            check(discrepancy < 2e-12)
            largest_quadrature_error = max(
                largest_quadrature_error, discrepancy,
            )
            formula_cases += 1

    # At alpha=2 the twice-cell term collapses to one simple radical.
    for r in range(1024):
        rr = mp.mpf(r)
        radical_term = 1/(mp.sqrt(rr+1)+mp.sqrt(rr))**3
        check(abs(2*pareto_cell_term(r, 2)-radical_term) < 2e-14)

    cells = 16384
    partial_constant = mp.fsum(
        1/(mp.sqrt(r+1)+mp.sqrt(r))**3 for r in range(cells)
    )
    scaled_error = mp.sqrt(cells)*pareto_equal_mass_error(cells, 2)
    check(abs(partial_constant-scaled_error) < 2e-14)

    # The radical summand lies between 1/[8(r+1)^(3/2)] and
    # 1/[8r^(3/2)].  Integral comparison gives this rigorous tail bracket.
    constant_lower = partial_constant+1/(4*mp.sqrt(cells+1))
    constant_upper = (partial_constant+1/(4*mp.sqrt(cells))
                      + 1/(8*cells**mp.mpf("1.5")))
    check(constant_lower < constant_upper)
    check(constant_upper-constant_lower < mp.mpf("1.2e-7"))
    return (formula_cases, largest_quadrature_error, partial_constant,
            constant_lower, constant_upper)


def student_pdf(x, degrees):
    """Student density, including its value at either infinity."""
    if abs(x) == inf:
        return 0.0
    log_constant = (gammaln((degrees+1)/2)-gammaln(degrees/2)
                    - 0.5*log(degrees*pi))
    return exp(log_constant-(degrees+1)/2*log1p(x*x/degrees))


def student_first_antiderivative(x, degrees):
    """Antiderivative of x times the Student density for degrees > 1."""
    if abs(x) == inf:
        return 0.0
    return -(degrees+x*x)*student_pdf(x, degrees)/(degrees-1)


def student_equal_mass_error(cells, degrees=2.0):
    """Exact W1 formula for Student equal-mass barycenters."""
    check(cells >= 2 and degrees > 1)
    total = 0.0
    for cell in range(cells):
        left, right = cell/cells, (cell+1)/cells
        lower = -inf if cell == 0 else stdtrit(degrees, left)
        upper = inf if cell+1 == cells else stdtrit(degrees, right)
        mean = cells*(student_first_antiderivative(upper, degrees)
                      - student_first_antiderivative(lower, degrees))
        crossing = stdtr(degrees, mean)
        check(left < crossing < right)
        total += 2*(student_first_antiderivative(upper, degrees)
                    - student_first_antiderivative(mean, degrees)
                    - mean*(right-crossing))
    return total


def direct_student_equal_mass_error(cells, degrees=2.0):
    """Independent density quadrature of the Student construction."""
    check(cells >= 2 and degrees > 1)
    total = 0.0
    for cell in range(cells):
        left, right = cell/cells, (cell+1)/cells
        lower = -inf if cell == 0 else stdtrit(degrees, left)
        upper = inf if cell+1 == cells else stdtrit(degrees, right)
        mean = cells*quad(
            lambda x: x*student_pdf(x, degrees), lower, upper,
            epsabs=2e-13, epsrel=2e-13, limit=100,
        )[0]
        total += quad(
            lambda x: abs(x-mean)*student_pdf(x, degrees), lower, upper,
            points=None if abs(lower) == inf or abs(upper) == inf else [mean],
            epsabs=2e-13, epsrel=2e-13, limit=100,
        )[0]
    return total


def certify_regular_tail_quantization_rate():
    """Student-t certificate for the regular-variation profile theorem."""
    formula_cases = 0
    largest_quadrature_error = 0.0
    error_at_128 = None
    for cells in (2, 4, 8, 16, 32, 64, 128):
        closed = student_equal_mass_error(cells)
        direct = direct_student_equal_mass_error(cells)
        discrepancy = abs(closed-direct)
        check(discrepancy < 2e-12)
        largest_quadrature_error = max(
            largest_quadrature_error, discrepancy,
        )
        formula_cases += 1
        if cells == 128:
            error_at_128 = closed

    normalized_ratios = []
    for cells in (256, 1024, 4096, 16384, 65536):
        error = student_equal_mass_error(cells)
        tail_quantile = stdtrit(2.0, 1-1/cells)
        normalized_ratios.append(cells*error/(2*tail_quantile))
    check(all(normalized_ratios[j+1] < normalized_ratios[j]
              for j in range(len(normalized_ratios)-1)))
    check(1.2473 < normalized_ratios[-1] < 1.2474)
    return (formula_cases, largest_quadrature_error, error_at_128,
            student_equal_mass_error(65536), normalized_ratios[-1])


def certify_internal_randomizer_mixed_contraction():
    """An exact continuous contraction using a uniform independent of Z."""
    # Source: 1/2 delta_0 + 1/2 Uniform[0,1].  With U independent of Z,
    # set M=Z/2 off the atom and M=U/2 on the atom.  Both branches put
    # density one on [0,1/2], and only the positive branch contributes Z.
    cells = 16
    total_mass = total_first = F(0)
    for cell in range(cells):
        left = F(cell, 2*cells)
        right = F(cell+1, 2*cells)
        atomic_branch_mass = right-left
        positive_branch_mass = right-left
        target_mass = 2*(right-left)
        target_first = right**2-left**2

        check(atomic_branch_mass == F(1, 2*cells))
        check(positive_branch_mass == F(1, 2*cells))
        check(target_mass == F(1, cells))
        check(target_first == target_mass*(left+right)/2)
        total_mass += target_mass
        total_first += target_first

    target_variance = F(1, 2)**2/F(12)
    source_variance = F(1, 2)*F(1, 3)-F(1, 4)**2
    check(total_mass == 1)
    check(total_first == F(1, 4))
    check(target_variance == F(1, 48))
    check(source_variance == F(5, 48))
    check(target_variance/source_variance == F(1, 5))
    return cells, target_variance, source_variance, F(1, 5)


def symmetric_beta_coefficients(shape):
    """Polynomial coefficients of the Beta(shape, shape) density."""
    normalizer = F(factorial(2*shape-1), factorial(shape-1)**2)
    return {
        shape-1+j: normalizer*(-1)**j*comb(shape-1, j)
        for j in range(shape)
    }


def integrate_polynomial(coefficients, left, right, extra_power=0):
    return sum(
        coefficient*(right**(degree+extra_power+1)
                     - left**(degree+extra_power+1))
        / (degree+extra_power+1)
        for degree, coefficient in coefficients.items()
    )


def certify_atomic_fiber_continuous_contractions():
    """Continuous beta targets obtained by splitting Bernoulli score fibers."""

    targets = cell_identities = 0
    smallest_variance = F(1, 4)
    for shape in range(1, 6):
        coefficients = symmetric_beta_coefficients(shape)
        target_mass = integrate_polynomial(coefficients, F(0), F(1))
        target_first = integrate_polynomial(coefficients, F(0), F(1), 1)
        target_second = integrate_polynomial(coefficients, F(0), F(1), 2)
        target_variance = target_second-target_first**2
        check(target_mass == 1)
        check(target_first == F(1, 2))
        check(target_variance == F(1, 4*(2*shape+1)))

        low_total = high_total = F(0)
        for cell in range(12):
            left, right = F(cell, 12), F(cell+1, 12)
            mass = integrate_polynomial(coefficients, left, right)
            first = integrate_polynomial(coefficients, left, right, 1)
            high_joint = first
            low_joint = mass-first
            check(low_joint >= 0 and high_joint >= 0)
            check(low_joint+high_joint == mass)
            check(high_joint == first)
            low_total += low_joint
            high_total += high_joint
            cell_identities += 1
        check(low_total == high_total == F(1, 2))
        targets += 1
        smallest_variance = min(smallest_variance, target_variance)
    return targets, cell_identities, smallest_variance


def certify_mixed_componentwise_contractions():
    """Paste atomless and atomic-fiber martingales in a mixed score law."""
    fixtures = (
        (2, 2, F(1, 3)),
        (3, 3, F(1, 2)),
        (4, 4, F(2, 3)),
        (5, 5, F(3, 4)),
    )
    cases = interval_identities = 0
    smallest_variance = smallest_ratio = F(1)
    for blocks, shape, continuous_weight in fixtures:
        left = F(blocks-1, 2*blocks)
        right = F(blocks+1, 2*blocks)
        beta_coefficients = symmetric_beta_coefficients(shape)
        target_mass = target_first = target_second = F(0)
        for cell in range(12):
            cell_left, cell_right = F(cell, 12), F(cell+1, 12)
            overlap_left = max(cell_left, left)
            overlap_right = min(cell_right, right)
            uniform_mass = (blocks*(overlap_right-overlap_left)
                            if overlap_left < overlap_right else F(0))
            uniform_first = (blocks*(overlap_right**2-overlap_left**2)/2
                             if overlap_left < overlap_right else F(0))
            uniform_second = (blocks*(overlap_right**3-overlap_left**3)/3
                              if overlap_left < overlap_right else F(0))
            coordinate_left = max(cell_left-left, F(0))
            coordinate_right = min(cell_right-left, F(1, blocks))
            continuous_score_first = (
                sum(
                    (coordinate_right**2-coordinate_left**2)/2
                    + F(j, blocks)*(coordinate_right-coordinate_left)
                    for j in range(blocks)
                ) if coordinate_left < coordinate_right else F(0)
            )
            beta_mass = integrate_polynomial(
                beta_coefficients, cell_left, cell_right,
            )
            beta_first = integrate_polynomial(
                beta_coefficients, cell_left, cell_right, 1,
            )
            beta_second = integrate_polynomial(
                beta_coefficients, cell_left, cell_right, 2,
            )
            mass = (continuous_weight*uniform_mass
                    + (1-continuous_weight)*beta_mass)
            first = (continuous_weight*uniform_first
                     + (1-continuous_weight)*beta_first)
            second = (continuous_weight*uniform_second
                      + (1-continuous_weight)*beta_second)
            # Each component is a martingale coupling, so its score-first
            # measure equals its target first-moment measure on every cell.
            check(continuous_score_first == uniform_first)
            score_first = (continuous_weight*continuous_score_first
                           + (1-continuous_weight)*beta_first)
            check(score_first == first)
            check(mass >= 0)
            target_mass += mass
            target_first += first
            target_second += second
            interval_identities += 1
        target_variance = target_second-target_first**2
        source_variance = (continuous_weight*F(1, 12)
                           + (1-continuous_weight)*F(1, 4))
        expected_variance = (
            continuous_weight*F(1, 12*blocks**2)
            + (1-continuous_weight)*F(1, 4*(2*shape+1))
        )
        check(target_mass == 1 and target_first == F(1, 2))
        check(target_variance == expected_variance < source_variance)
        smallest_variance = min(smallest_variance, target_variance)
        smallest_ratio = min(smallest_ratio, target_variance/source_variance)
        cases += 1
    return cases, interval_identities, smallest_variance, smallest_ratio


def rounded_atomic_band_witness(support, probabilities,
                                lower_envelope, upper_envelope):
    """Round the randomized quantile-band witness to whole atoms."""
    check(tuple(support) == tuple(sorted(support)))
    randomized = min(F(1), lower_envelope+upper_envelope)
    lower_cut = (lower_envelope if lower_envelope+upper_envelope <= 1
                 else 1-upper_envelope)
    upper_cut = 1-upper_envelope
    cumulative = F(0)
    low_mass = low_first = high_mass = high_first = F(0)
    boundary_atoms = set()
    for j, (z, probability) in enumerate(zip(support, probabilities)):
        start, end = cumulative, cumulative+probability
        if end <= lower_cut:
            low_mass += probability
            low_first += z*probability
        if start >= upper_cut:
            high_mass += probability
            high_first += z*probability
        if start < lower_cut < end or start < upper_cut < end:
            boundary_atoms.add(j)
        cumulative = end
    check(cumulative == 1)
    rounded = low_mass+high_mass
    loss = randomized-rounded
    boundary_mass = sum(probabilities[j] for j in boundary_atoms)
    check(0 <= loss <= boundary_mass)
    check(len(boundary_atoms) <= 2)
    return (rounded, loss, boundary_mass,
            low_mass, low_first, high_mass, high_first)


def certify_two_sided_gap_sharpness():
    """Exact extremizers for the one- and two-boundary gap constants."""
    shared_checks = 0
    for central_mass in (F(1, 3), F(2, 5), F(1, 2), F(3, 5), F(4, 5)):
        outer_mass = (1-central_mass)/2
        support = (F(0), F(1, 2), F(1))
        probabilities = (outer_mass, central_mass, outer_mass)
        lower, upper = central_mass/2, 1-central_mass/2
        check(lower < F(1, 2) < upper)
        lower_mass = sharp_atomic_envelope(
            support, probabilities, lower, True,
        )[0]
        upper_mass = sharp_atomic_envelope(
            support, probabilities, upper, False,
        )[0]
        deterministic = brute_deterministic_band_mass(
            support, probabilities, lower, upper,
        )
        check(lower_mass == upper_mass == F(1, 2))
        check(deterministic == 1-central_mass)
        check(1-deterministic == central_mass == max(probabilities))
        rounded, loss, boundary_mass, *_ = rounded_atomic_band_witness(
            support, probabilities, lower_mass, upper_mass,
        )
        check(rounded == deterministic)
        check(loss == boundary_mass == central_mass)
        shared_checks += 1

    outer_mass, boundary_mass, boundary_score = F(1, 5), F(3, 10), F(1, 3)
    support = (F(0), boundary_score, 1-boundary_score, F(1))
    probabilities = (outer_mass, boundary_mass, boundary_mass, outer_mass)
    disjoint_checks, ratios = 0, []
    for eta in (F(1, 2), F(1, 4), F(1, 10), F(1, 20), F(1, 100)):
        split_mass = (1-eta)*boundary_mass
        lower = split_mass*boundary_score/(outer_mass+split_mass)
        upper = 1-lower
        lower_mass = sharp_atomic_envelope(
            support, probabilities, lower, True,
        )[0]
        upper_mass = sharp_atomic_envelope(
            support, probabilities, upper, False,
        )[0]
        randomized = lower_mass+upper_mass
        deterministic = brute_deterministic_band_mass(
            support, probabilities, lower, upper,
        )
        check(lower_mass == upper_mass == outer_mass+split_mass)
        check(randomized < 1)
        check(deterministic == 2*outer_mass)
        ratio = (randomized-deterministic)/(2*boundary_mass)
        check(ratio == 1-eta)
        rounded, loss, deleted_mass, *_ = rounded_atomic_band_witness(
            support, probabilities, lower_mass, upper_mass,
        )
        check(rounded == deterministic)
        check(loss == randomized-deterministic)
        check(deleted_mass == 2*boundary_mass)
        ratios.append(ratio)
        disjoint_checks += 1
    return shared_checks, disjoint_checks, max(ratios)


def convex_order_below(left_support, left_probabilities,
                       right_support, right_probabilities):
    """Certify finite-law convex order through exact call transforms."""
    check(sum(left_probabilities) == sum(right_probabilities) == 1)
    check(sum(z*q for z, q in zip(left_support, left_probabilities)) ==
          sum(z*q for z, q in zip(right_support, right_probabilities)))
    strikes = sorted(set(left_support+right_support))
    for strike in strikes:
        left_call = sum(q*max(z-strike, 0)
                        for z, q in zip(left_support, left_probabilities))
        right_call = sum(q*max(z-strike, 0)
                         for z, q in zip(right_support, right_probabilities))
        check(left_call <= right_call)


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
    atomic_checks = fractional_splits = gap_bound_checks = 0
    sharp_gap_equalities = 0
    convex_order_witnesses = deterministic_tail_checks = strict_relaxations = 0
    dynamic_tail_checks = knapsack_form_checks = fptas_checks = 0
    strict_fptas_cases = 0
    bernoulli_lower = bernoulli_upper = 0
    atomic_two_sided = atomic_disjoint = atomic_overlap = 0
    deterministic_band_checks = strict_band_relaxations = 0
    band_gap_bound_checks = band_shared_cut_checks = band_dp_checks = 0
    band_saturation_checks = nonatomic_refinement_checks = 0
    (mixed_disjoint_witnesses, mixed_shared_witnesses,
     mixed_plateau_cuts, mixed_quantile_mean) = (
        certify_mixed_nonatomic_quantile_witnesses()
    )
    finite_action_purifications, largest_randomized_share = (
        certify_finite_action_purifications()
    )
    (countable_support_atoms, countable_support_tail_mass,
     countable_support_variance, countable_support_ratio) = (
        certify_countable_support_mixed_contraction()
    )
    (dense_contraction_cases, dense_largest_support, dense_smallest_error,
     dense_target_variance, dense_source_variance) = (
        certify_wasserstein_dense_mixed_contractions()
    )
    (moment_rate_cases, largest_moment_squared_ratio,
     l1_rate_counterexamples, largest_l1_rate_error) = (
        certify_unbounded_quantization_rate()
    )
    (bounded_rate_cases, bounded_rate_quadrature_error,
     bounded_rate_error_128, bounded_rate_error_65536,
     bounded_rate_ratio_65536) = certify_bounded_quantization_rate()
    (bounded_atomic_cases, bounded_atomic_even_ratio,
     bounded_atomic_odd_ratio) = certify_bounded_atomic_rate_obstruction()
    (exponential_formula_cases, exponential_quadrature_error,
     exponential_error_128, exponential_error_16384,
     exponential_ratio_16384) = certify_exponential_quantization_rate()
    (gaussian_formula_cases, gaussian_quadrature_error,
     gaussian_error_128, gaussian_error_65536,
     gaussian_log_ratio_65536, gaussian_quantile_ratio_65536) = (
        certify_gaussian_quantization_rate()
    )
    (slow_tail_formula_cases, slow_tail_quadrature_error,
     slow_tail_error_128, slow_tail_error_65536,
     slow_tail_ratio_65536) = certify_slow_tail_quantization_rate()
    (pareto_formula_cases, pareto_quadrature_error,
     pareto_partial_constant, pareto_constant_lower,
     pareto_constant_upper) = certify_pareto_quantization_rate()
    (regular_tail_formula_cases, regular_tail_quadrature_error,
     regular_tail_error_128, regular_tail_error_65536,
     regular_tail_ratio_65536) = certify_regular_tail_quantization_rate()
    (internal_randomizer_cells, internal_randomizer_target_variance,
     internal_randomizer_source_variance, internal_randomizer_ratio) = (
        certify_internal_randomizer_mixed_contraction()
    )
    (continuous_monge_maps, largest_monge_multiplicity,
     smallest_monge_variance) = certify_continuous_backward_monge_maps()
    (atomic_fiber_targets, atomic_fiber_cell_identities,
     smallest_atomic_fiber_variance) = (
        certify_atomic_fiber_continuous_contractions()
    )
    (mixed_componentwise_targets, mixed_componentwise_identities,
     smallest_mixed_variance, smallest_mixed_ratio) = (
        certify_mixed_componentwise_contractions()
    )
    shared_gap_equalities, disjoint_gap_limits, largest_gap_ratio = (
        certify_two_sided_gap_sharpness()
    )
    (additive_scheme_checks, strict_additive_approximations,
     largest_heavy_set, largest_lp_fractional_set) = (
        certify_two_sided_additive_scheme()
    )
    (relative_scheme_checks, strict_relative_approximations,
     largest_relative_fractional_set) = certify_two_sided_relative_scheme()
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
                deterministic = brute_deterministic_tail_mass(
                    support, probabilities, y, lower,
                )
                fractional, rounded, cutoff_split, cutoff_mass = (
                    fractional_tail_rounding(
                        support, probabilities, y, lower,
                    )
                )
                check(fractional == envelope)
                check(rounded == fractional-cutoff_split <= deterministic)
                check(envelope-deterministic <= cutoff_split <= cutoff_mass)
                check(cutoff_mass <= max(probabilities))
                gap_bound_checks += 1
                if lower:
                    check(deterministic == dynamic_lower_tail_mass(
                        support, probabilities, y,
                    ))
                    check(deterministic == ordinary_knapsack_lower_tail_mass(
                        support, probabilities, y,
                    ))
                    dynamic_tail_checks += 1
                    knapsack_form_checks += 1
                    for epsilon in (F(1, 2), F(1, 3), F(1, 4), F(1, 5), F(1, 10)):
                        approximate = approximate_lower_tail_mass(
                            support, probabilities, y, epsilon,
                        )
                        check((1-epsilon)*deterministic <= approximate <= deterministic)
                        strict_fptas_cases += int(approximate < deterministic)
                        fptas_checks += 1
                check(deterministic <= envelope)
                deterministic_tail_checks += 1
                strict_relaxations += int(deterministic < envelope)
                other_mean = (mean-y*envelope)/(1-envelope)
                if lower:
                    contraction_support = (y, other_mean)
                    contraction_probabilities = (envelope, 1-envelope)
                else:
                    contraction_support = (other_mean, y)
                    contraction_probabilities = (1-envelope, envelope)
                convex_order_below(
                    contraction_support, contraction_probabilities,
                    support, probabilities,
                )
                convex_order_witnesses += 1
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
                cuts = ((lower_mass, 1-upper_mass)
                        if lower_mass+upper_mass <= 1 else (1-upper_mass,))
                refined_support, refined_probabilities = (
                    refine_atomic_space_at_quantile_cuts(
                        support, probabilities, tuple(set(cuts)),
                    )
                )
                check(brute_deterministic_band_mass(
                    refined_support, refined_probabilities, lower, upper,
                ) == witness)
                nonatomic_refinement_checks += 1
                deterministic = brute_deterministic_band_mass(
                    support, probabilities, lower, upper,
                )
                check(deterministic == two_budget_band_mass(
                    support, probabilities, lower, upper,
                ))
                band_dp_checks += 1
                check((deterministic == 1) == band_saturates_by_one_budget(
                    support, probabilities, lower, upper,
                ))
                band_saturation_checks += 1
                check(deterministic == brute_partition_band_mass(
                    support, probabilities, lower, upper,
                ))
                check(deterministic <= witness)
                (rounded, loss, boundary_mass,
                 rounded_low_mass, rounded_low_first,
                 rounded_high_mass, rounded_high_first) = (
                    rounded_atomic_band_witness(
                        support, probabilities, lower_mass, upper_mass,
                    )
                )
                check(rounded_low_first <= lower*rounded_low_mass)
                check(rounded_high_first >= upper*rounded_high_mass)
                check(rounded <= deterministic)
                check(witness-deterministic <= loss <= boundary_mass)
                two_largest = sum(sorted(probabilities, reverse=True)[:2])
                check(boundary_mass <= two_largest)
                if lower_mass+upper_mass > 1:
                    check(boundary_mass <= max(probabilities))
                    band_shared_cut_checks += 1
                band_gap_bound_checks += 1
                deterministic_band_checks += 1
                strict_band_relaxations += int(deterministic < witness)
                atomic_two_sided += 1
    tie_support = (F(0), F(1, 2), F(1, 2), F(1))
    for probabilities, y, lower in (
        ((F(1, 20), F(1, 10), F(1, 5), F(13, 20)), F(2, 5), True),
        ((F(13, 20), F(1, 5), F(1, 10), F(1, 20)), F(3, 5), False),
    ):
        deterministic = brute_deterministic_tail_mass(
            tie_support, probabilities, y, lower,
        )
        fractional, rounded, cutoff_split, cutoff_mass = fractional_tail_rounding(
            tie_support, probabilities, y, lower,
        )
        check(rounded <= deterministic <= fractional)
        check(fractional-deterministic <= cutoff_split <= cutoff_mass)
        check(cutoff_mass <= max(probabilities))
        gap_bound_checks += 1
    for probabilities, lower, upper in (
        ((F(1, 10), F(1, 5), F(1, 5), F(1, 2)), F(1, 10), F(9, 10)),
        ((F(1, 10), F(1, 5), F(1, 5), F(1, 2)), F(2, 5), F(4, 5)),
    ):
        lower_mass = sharp_atomic_envelope(
            tie_support, probabilities, lower, True,
        )[0]
        upper_mass = sharp_atomic_envelope(
            tie_support, probabilities, upper, False,
        )[0]
        randomized = min(F(1), lower_mass+upper_mass)
        deterministic = brute_deterministic_band_mass(
            tie_support, probabilities, lower, upper,
        )
        check(deterministic == two_budget_band_mass(
            tie_support, probabilities, lower, upper,
        ))
        band_dp_checks += 1
        check((deterministic == 1) == band_saturates_by_one_budget(
            tie_support, probabilities, lower, upper,
        ))
        band_saturation_checks += 1
        (rounded, loss, boundary_mass,
         rounded_low_mass, rounded_low_first,
         rounded_high_mass, rounded_high_first) = rounded_atomic_band_witness(
            tie_support, probabilities, lower_mass, upper_mass,
        )
        check(rounded_low_first <= lower*rounded_low_mass)
        check(rounded_high_first >= upper*rounded_high_mass)
        check(rounded <= deterministic <= randomized)
        check(randomized-deterministic <= loss <= boundary_mass)
        check(boundary_mass <= sum(sorted(probabilities, reverse=True)[:2]))
        if lower_mass+upper_mass > 1:
            check(boundary_mass <= max(probabilities))
            band_shared_cut_checks += 1
        band_gap_bound_checks += 1
    approximation_support = tuple(map(
        F, (0, F(1, 7), F(2, 7), F(4, 7), F(5, 7), 1),
    ))
    for weights in ((1, 1, 1, 1, 5, 4), (1, 1, 1, 1, 5, 5),
                    (1, 1, 1, 1, 5, 6), (1, 1, 1, 1, 6, 4),
                    (1, 1, 1, 1, 6, 5)):
        probabilities = tuple(F(w, sum(weights)) for w in weights)
        mean = sum(z*q for z, q in zip(approximation_support, probabilities))
        y = F(4, 5)*mean
        deterministic = brute_deterministic_tail_mass(
            approximation_support, probabilities, y, True,
        )
        check(deterministic == ordinary_knapsack_lower_tail_mass(
            approximation_support, probabilities, y,
        ))
        approximate = approximate_lower_tail_mass(
            approximation_support, probabilities, y, F(1, 2),
        )
        check(F(1, 2)*deterministic <= approximate < deterministic)
        knapsack_form_checks += 1
        fptas_checks += 1
        strict_fptas_cases += 1
    edge_support, edge_probabilities, edge_y = (F(0), F(1)), (F(1, 2), F(1, 2)), F(1, 4)
    edge_exact = brute_deterministic_tail_mass(
        edge_support, edge_probabilities, edge_y, True,
    )
    check(edge_exact == ordinary_knapsack_lower_tail_mass(
        edge_support, edge_probabilities, edge_y,
    ) == approximate_lower_tail_mass(
        edge_support, edge_probabilities, edge_y, F(1, 2),
    ) == F(1, 2))
    knapsack_form_checks += 1
    fptas_checks += 1
    subset_sum_checks = subset_sum_equalities = 0
    saturation_checks = saturation_equalities = saturation_exhaustive = 0
    for values in ((2, 3, 7, 8, 10), (1, 4, 6, 13),
                   (3, 5, 9, 12, 17), (2, 6, 11, 15, 18, 21)):
        for target in range(1, sum(values)):
            subset_sum_equalities += certify_subset_sum_reduction(values, target)
            subset_sum_checks += 1
            attainable, exhaustive = certify_band_saturation_reduction(
                values, target,
            )
            saturation_equalities += attainable
            saturation_exhaustive += exhaustive
            saturation_checks += 1
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
            fractional, rounded, cutoff_split, cutoff_mass = fractional_tail_rounding(
                (F(0), F(1)), (1-mean, mean), lower, True,
            )
            check(fractional == randomized)
            check(rounded == deterministic)
            check(randomized-deterministic == cutoff_split <= cutoff_mass)
            sharp_gap_equalities += 1
            convex_order_below(
                (lower, F(1)), (randomized, 1-randomized),
                (F(0), F(1)), (1-mean, mean),
            )
            convex_order_witnesses += 1
            bernoulli_lower += 1

            upper = mean+fraction*(1-mean)
            randomized = mean/upper
            deterministic = mean
            split = mean*(1-upper)/upper
            check(0 < split < 1-mean)
            check(randomized == deterministic+split > deterministic)
            check(randomized-deterministic == mean*(1-upper)/upper)
            fractional, rounded, cutoff_split, cutoff_mass = fractional_tail_rounding(
                (F(0), F(1)), (1-mean, mean), upper, False,
            )
            check(fractional == randomized)
            check(rounded == deterministic)
            check(randomized-deterministic == cutoff_split <= cutoff_mass)
            sharp_gap_equalities += 1
            convex_order_below(
                (F(0), upper), (1-randomized, randomized),
                (F(0), F(1)), (1-mean, mean),
            )
            convex_order_witnesses += 1
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
    print(f"Exact nonatomic-space refinements: {nonatomic_refinement_checks}")
    print(f"Exact mixed-law nonatomic quantile witnesses: "
          f"{mixed_disjoint_witnesses+mixed_shared_witnesses} "
          f"({mixed_disjoint_witnesses} disjoint, "
          f"{mixed_shared_witnesses} shared-cut); "
          f"plateau-cut splits: {mixed_plateau_cuts}; "
          f"mean: {mixed_quantile_mean}")
    print(f"Exact three-action nonatomic purifications: "
          f"{finite_action_purifications}; largest randomized share: "
          f"{largest_randomized_share}")
    print(f"Exact countable-support mixed contraction: "
          f"{countable_support_atoms} leading atom identities; "
          f"symbolic tail mass: {countable_support_tail_mass}; "
          f"target variance: {countable_support_variance}; "
          f"source-variance ratio: {countable_support_ratio}")
    print(f"Wasserstein-dense mixed contractions: {dense_contraction_cases}; "
          f"largest support: {dense_largest_support}; "
          f"smallest exact W1 error: {dense_smallest_error}; "
          f"target/source variances: {dense_target_variance}/"
          f"{dense_source_variance}")
    print(f"Unbounded quantization moment-rate cases: {moment_rate_cases}; "
          f"largest squared p=2 bound ratio: "
          f"{largest_moment_squared_ratio}; L1 rate counterexamples: "
          f"{l1_rate_counterexamples}; largest exact error: "
          f"{largest_l1_rate_error}")
    print(f"Bounded quantization formula checks: {bounded_rate_cases}; "
          f"largest quadrature discrepancy: "
          f"{bounded_rate_quadrature_error:.3e}; square-root-quantile W1 "
          f"at N=128: {bounded_rate_error_128:.15f}; W1 at N=65536: "
          f"{bounded_rate_error_65536:.15f}; sharp-rate ratio: "
          f"{bounded_rate_ratio_65536:.12f}")
    print(f"Bounded atomic rate obstructions: {bounded_atomic_cases}; "
          f"even/odd scaled errors: {bounded_atomic_even_ratio}/"
          f"{bounded_atomic_odd_ratio}")
    print(f"Exponential equal-mass formula checks: {exponential_formula_cases}; "
          f"largest quadrature discrepancy: "
          f"{exponential_quadrature_error:.3e}; W1 at N=128: "
          f"{exponential_error_128:.15f}; W1 at N=16384: "
          f"{float(exponential_error_16384):.15f}; normalized ratio: "
          f"{float(exponential_ratio_16384):.12f}")
    print(f"Gaussian equal-mass formula checks: {gaussian_formula_cases}; "
          f"largest quadrature discrepancy: {gaussian_quadrature_error:.3e}; "
          f"W1 at N=128: {gaussian_error_128:.15f}; "
          f"W1 at N=65536: {gaussian_error_65536:.15f}; "
          f"sqrt-log ratio: {gaussian_log_ratio_65536:.12f}; "
          f"quantile ratio: {gaussian_quantile_ratio_65536:.12f}")
    print(f"Slow-tail logistic formula checks: {slow_tail_formula_cases}; "
          f"largest quadrature discrepancy: {slow_tail_quadrature_error:.3e}; "
          f"W1 at N=128: {slow_tail_error_128:.15f}; "
          f"W1 at N=65536: {slow_tail_error_65536:.15f}; "
          f"trimmed-range ratio: {slow_tail_ratio_65536:.12f}")
    print(f"Pareto equal-mass formula checks: {pareto_formula_cases}; "
          f"largest quadrature discrepancy: {pareto_quadrature_error:.3e}; "
          f"alpha=2 scaled error at N=16384: "
          f"{float(pareto_partial_constant):.15f}; limiting-constant bracket: "
          f"[{float(pareto_constant_lower):.15f}, "
          f"{float(pareto_constant_upper):.15f}]")
    print(f"Regular-tail Student formula checks: "
          f"{regular_tail_formula_cases}; largest quadrature discrepancy: "
          f"{regular_tail_quadrature_error:.3e}; W1 at N=128: "
          f"{regular_tail_error_128:.15f}; W1 at N=65536: "
          f"{regular_tail_error_65536:.15f}; profile ratio: "
          f"{regular_tail_ratio_65536:.12f}")
    print(f"Exact internal-randomizer mixed contraction: "
          f"{internal_randomizer_cells} cell identities; "
          f"target/source variances: {internal_randomizer_target_variance}/"
          f"{internal_randomizer_source_variance}; "
          f"variance ratio: {internal_randomizer_ratio}")
    print(f"Exact continuous backward-Monge maps: {continuous_monge_maps}; "
          f"largest preimage multiplicity: {largest_monge_multiplicity}; "
          f"smallest target variance: {smallest_monge_variance}; "
          f"variance ratio: {12*smallest_monge_variance}")
    print(f"Exact atomic-fiber continuous contractions: {atomic_fiber_targets}; "
          f"interval barycenter identities: {atomic_fiber_cell_identities}; "
          f"smallest target variance: {smallest_atomic_fiber_variance}; "
          f"Bernoulli variance ratio: {4*smallest_atomic_fiber_variance}")
    print(f"Exact mixed componentwise contractions: {mixed_componentwise_targets}; "
          f"interval barycenter identities: {mixed_componentwise_identities}; "
          f"smallest target variance: {smallest_mixed_variance}; "
          f"smallest source-variance ratio: {smallest_mixed_ratio}")
    print(f"Exact Bernoulli no-split gaps: {bernoulli_lower} lower, "
          f"{bernoulli_upper} upper")
    print(f"Exact finite-law convex-order witnesses: {convex_order_witnesses}")
    print(f"Exact fixed-space 0-1 tail optima: {deterministic_tail_checks}; "
          f"strict fractional relaxations: {strict_relaxations}")
    print(f"Exact signed-knapsack dynamic programs: {dynamic_tail_checks}; "
          f"SUBSET SUM reductions: {subset_sum_checks} "
          f"({subset_sum_equalities} attainable targets)")
    print(f"Ordinary-knapsack reductions: {knapsack_form_checks}; "
          f"profit-scaled FPTAS checks: {fptas_checks} "
          f"({strict_fptas_cases} nonexact approximations)")
    print(f"Cutoff-atom integrality-gap bounds: {gap_bound_checks}; "
          f"sharp Bernoulli equalities: {sharp_gap_equalities}")
    print(f"Exact fixed-space two-sided optima: {deterministic_band_checks}; "
          f"strict randomized gaps: {strict_band_relaxations}")
    print(f"Two-sided boundary-atom gap bounds: {band_gap_bound_checks}; "
          f"shared-cut bounds: {band_shared_cut_checks}")
    print(f"Sharp two-sided gap constructions: {shared_gap_equalities} "
          f"shared-cut equalities; {disjoint_gap_limits} disjoint limits; "
          f"shared b in [1/3,1); disjoint masses 1/5,3/10,3/10,1/5 "
          f"at 0,1/3,2/3,1; largest boundary-mass ratio {largest_gap_ratio}")
    print(f"Exact two-budget band dynamic programs: {band_dp_checks}; "
          f"one-budget saturation decisions: {band_saturation_checks}")
    print(f"Two-sided additive approximation checks: {additive_scheme_checks}; "
          f"strict approximations: {strict_additive_approximations}; "
          f"largest heavy set: {largest_heavy_set}; "
          f"largest fractional LP set: {largest_lp_fractional_set}")
    print(f"Two-sided relative PTAS checks: {relative_scheme_checks}; "
          f"strict approximations: {strict_relative_approximations}; "
          f"enumerated atoms: 3; largest fractional LP set: "
          f"{largest_relative_fractional_set}")
    print(f"Two-sided saturation reductions: {saturation_checks} "
          f"({saturation_equalities} attainable; "
          f"{saturation_exhaustive} exhaustive assignment checks)")
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
