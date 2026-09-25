"""Certificate for the exact Cox-count and factorial-cumulant identities.

Two independent finite-dimensional calculations are compared:

1. Polynomial moments of the integrated intensity Lambda_T are propagated by
   the backward Feynman--Kac moment hierarchy.
2. The joint regime/count master equation is truncated at a count so remote
   that its remaining mass is below rounding error, and ordinary cumulants of
   N_T are computed from that distribution.

For a Cox count, factorial cumulants of N_T must equal ordinary cumulants of
Lambda_T.  The certificate checks this through order four for one stream and
through bidegree (2,2) for two conditionally independent streams driven by a
nonreversible three-state chain.  A third calculation adds a regime-dependent
common-shock stream and verifies the all-order correction through bidegree
(3,3), first for unit jumps and then for arbitrary integer-valued bivariate
marks.  It then verifies the closed two-state finite-horizon formula used on
the counts page and its O(lambda^-2) first-order residual.
"""

import itertools
import math
import numpy as np
from scipy.linalg import expm
from scipy.sparse import diags, eye, kron
from scipy.sparse.linalg import expm_multiply
from scipy.stats import poisson


def cumulants4(raw):
    """First four cumulants from the first four raw moments."""
    m1, m2, m3, m4 = raw
    return np.array([
        m1,
        m2 - m1 * m1,
        m3 - 3 * m2 * m1 + 2 * m1 ** 3,
        m4 - 4 * m3 * m1 - 3 * m2 * m2 + 12 * m2 * m1 * m1 - 6 * m1 ** 4,
    ])


def factorial_cumulants4(ordinary):
    """Invert kappa_r = sum_k S(r,k) factorial_k through order four."""
    k1, k2, k3, k4 = ordinary
    return np.array([
        k1,
        k2 - k1,
        k3 - 3 * k2 + 2 * k1,
        k4 - 6 * k3 + 11 * k2 - 6 * k1,
    ])


def mixed_cumulants22(raw):
    """Mixed cumulants kappa_11, kappa_21, kappa_12 and kappa_22.

    ``raw[a, b]`` is E[X^a Y^b], with 0 <= a,b <= 2.  The last
    expression is the joint-cumulant partition formula for (X,X,Y,Y).
    """
    mx, my = raw[1, 0], raw[0, 1]
    m20, m02, m11 = raw[2, 0], raw[0, 2], raw[1, 1]
    m21, m12, m22 = raw[2, 1], raw[1, 2], raw[2, 2]
    k11 = m11 - mx * my
    k21 = m21 - m20 * my - 2.0 * m11 * mx + 2.0 * mx ** 2 * my
    k12 = m12 - m02 * mx - 2.0 * m11 * my + 2.0 * my ** 2 * mx
    k22 = (m22 - m20 * m02 - 2.0 * m11 ** 2
           - 2.0 * m21 * my - 2.0 * m12 * mx
           + 2.0 * m20 * my ** 2 + 2.0 * m02 * mx ** 2
           + 8.0 * m11 * mx * my - 6.0 * mx ** 2 * my ** 2)
    return np.array([k11, k21, k12, k22])


def mixed_factorial_cumulants22(ordinary):
    """Convert mixed ordinary cumulants by signed Stirling transforms."""
    k11, k21, k12, k22 = ordinary
    return np.array([k11, k21 - k11, k12 - k11,
                     k22 - k21 - k12 + k11])


def _set_partitions(labels):
    """Yield every set partition of a short labelled tuple."""
    if not labels:
        yield []
        return
    first, rest = labels[0], labels[1:]
    for partition in _set_partitions(rest):
        yield [(first,)] + partition
        for j in range(len(partition)):
            yield (partition[:j]
                   + [(first,) + partition[j]]
                   + partition[j + 1:])


def joint_cumulant(repeats, moment):
    """Joint cumulant from a callable returning multivariate raw moments.

    ``repeats[j]`` is the number of copies of coordinate ``j``.  The usual
    partition formula is practical here because the certificate stops at six
    arguments (Bell number 203).
    """
    labels = tuple(axis for axis, count in enumerate(repeats)
                   for _ in range(count))
    total = 0.0
    for partition in _set_partitions(labels):
        coefficient = ((-1.0) ** (len(partition) - 1)
                       * math.factorial(len(partition) - 1))
        term = coefficient
        for block in partition:
            degrees = tuple(block.count(axis) for axis in range(len(repeats)))
            term *= moment(degrees)
        total += term
    return total


def bivariate_factorial_cumulant_grid(p, degree=3):
    """Mixed factorial cumulants from a bivariate count distribution."""
    grid = np.arange(p.shape[0], dtype=float)
    falling = np.ones((degree + 1, len(grid)))
    for r in range(1, degree + 1):
        falling[r] = falling[r - 1] * (grid - r + 1.0)

    def factorial_moment(degrees):
        a, b = degrees
        return np.einsum("i,j,ij->", falling[a], falling[b], p)

    answer = np.empty((degree, degree))
    for r in range(1, degree + 1):
        for s in range(1, degree + 1):
            answer[r - 1, s - 1] = joint_cumulant(
                (r, s), factorial_moment)
    return answer


def integrated_intensity_cumulants(Q, rates, T, prior, degree=4):
    """Cumulants of integral_0^T rates[Y_s] ds from a backward moment ODE."""
    Q = np.asarray(Q, float)
    rates = np.asarray(rates, float)
    prior = np.asarray(prior, float)
    n = len(rates)
    A = np.zeros(((degree + 1) * n, (degree + 1) * n))
    D = np.diag(rates)
    for k in range(degree + 1):
        sl = slice(k * n, (k + 1) * n)
        A[sl, sl] = Q
        if k:
            A[sl, slice((k - 1) * n, k * n)] = k * D
    y0 = np.zeros((degree + 1) * n)
    y0[:n] = 1.0
    y = expm(A * T) @ y0
    raw = np.array([prior @ y[k * n:(k + 1) * n] for k in range(1, degree + 1)])
    return cumulants4(raw)


def count_cumulants(Q, rates, T, prior, max_count=80):
    """Cumulants from the forward regime/count master equation."""
    Q = np.asarray(Q, float)
    rates = np.asarray(rates, float)
    prior = np.asarray(prior, float)
    n = len(rates)
    D = np.diag(rates)
    A = np.zeros(((max_count + 1) * n, (max_count + 1) * n))
    for k in range(max_count + 1):
        sl = slice(k * n, (k + 1) * n)
        A[sl, sl] = Q.T - D
        if k:
            A[sl, slice((k - 1) * n, k * n)] = D
    y0 = np.zeros((max_count + 1) * n)
    y0[:n] = prior
    y = expm(A * T) @ y0
    p = np.array([y[k * n:(k + 1) * n].sum() for k in range(max_count + 1)])
    k = np.arange(max_count + 1, dtype=float)
    raw = np.array([np.dot(k ** r, p) for r in range(1, 5)])
    return cumulants4(raw), p


def integrated_intensity_mixed_cumulants(Q, rates1, rates2, T, prior):
    """Mixed cumulants of two integrated rates, through bidegree (2,2)."""
    Q = np.asarray(Q, float)
    rates1, rates2 = np.asarray(rates1, float), np.asarray(rates2, float)
    prior = np.asarray(prior, float)
    n = len(prior)
    pairs = [(a, b) for a in range(3) for b in range(3)]
    pos = {pair: j for j, pair in enumerate(pairs)}
    A = np.zeros((len(pairs) * n, len(pairs) * n))
    for j, (a, b) in enumerate(pairs):
        sl = slice(j * n, (j + 1) * n)
        A[sl, sl] = Q
        if a:
            lo = pos[a - 1, b]
            A[sl, slice(lo * n, (lo + 1) * n)] = a * np.diag(rates1)
        if b:
            lo = pos[a, b - 1]
            A[sl, slice(lo * n, (lo + 1) * n)] = b * np.diag(rates2)
    y0 = np.zeros(len(pairs) * n)
    y0[:n] = 1.0
    y = expm(A * T) @ y0
    raw = np.empty((3, 3))
    for j, (a, b) in enumerate(pairs):
        raw[a, b] = prior @ y[j * n:(j + 1) * n]
    return mixed_cumulants22(raw)


def integrated_intensity_joint_cumulant111(Q, rates1, rates2, rates3,
                                             T, prior):
    """Third joint cumulant of three integrated rates."""
    Q = np.asarray(Q, float)
    rates = [np.asarray(item, float) for item in (rates1, rates2, rates3)]
    prior = np.asarray(prior, float)
    n = len(prior)
    triples = [(a, b, c) for a in range(2) for b in range(2)
               for c in range(2)]
    pos = {triple: j for j, triple in enumerate(triples)}
    A = np.zeros((len(triples) * n, len(triples) * n))
    for j, triple in enumerate(triples):
        sl = slice(j * n, (j + 1) * n)
        A[sl, sl] = Q
        for axis, degree in enumerate(triple):
            if degree:
                lower = list(triple)
                lower[axis] -= 1
                lo = pos[tuple(lower)]
                A[sl, slice(lo * n, (lo + 1) * n)] = np.diag(rates[axis])
    y0 = np.zeros(len(triples) * n)
    y0[:n] = 1.0
    y = expm(A * T) @ y0
    raw = {}
    for j, triple in enumerate(triples):
        raw[triple] = prior @ y[j * n:(j + 1) * n]
    ma, mb, mc = raw[1, 0, 0], raw[0, 1, 0], raw[0, 0, 1]
    return (raw[1, 1, 1] - ma * raw[0, 1, 1]
            - mb * raw[1, 0, 1] - mc * raw[1, 1, 0]
            + 2.0 * ma * mb * mc)


def integrated_intensity_joint_cumulant(Q, rates, repeats, T, prior):
    """Joint cumulant of repeated integrated-rate coordinates."""
    Q = np.asarray(Q, float)
    rates = [np.asarray(item, float) for item in rates]
    repeats = tuple(repeats)
    prior = np.asarray(prior, float)
    n = len(prior)
    degrees = list(itertools.product(*[range(r + 1) for r in repeats]))
    pos = {degree: j for j, degree in enumerate(degrees)}
    generator = np.zeros((len(degrees) * n, len(degrees) * n))
    for j, degree in enumerate(degrees):
        sl = slice(j * n, (j + 1) * n)
        generator[sl, sl] = Q
        for axis, power in enumerate(degree):
            if power:
                lower = list(degree)
                lower[axis] -= 1
                lo = pos[tuple(lower)]
                generator[sl, slice(lo * n, (lo + 1) * n)] = (
                    power * np.diag(rates[axis]))
    y0 = np.zeros(len(degrees) * n)
    y0[:n] = 1.0
    y = expm(generator * T) @ y0
    raw = {
        degree: prior @ y[j * n:(j + 1) * n]
        for j, degree in enumerate(degrees)
    }
    return joint_cumulant(repeats, raw.__getitem__)


def common_shock_factorial_cumulant(Q, rates1, rates2, common_rates,
                                      r, s, T, prior):
    """All-order common-shock formula for mixed factorial cumulants."""
    rates_a = np.asarray(rates1) + np.asarray(common_rates)
    rates_b = np.asarray(rates2) + np.asarray(common_rates)
    answer = 0.0
    for k in range(min(r, s) + 1):
        coefficient = (math.comb(r, k) * math.comb(s, k)
                       * math.factorial(k))
        answer += coefficient * integrated_intensity_joint_cumulant(
            Q, (rates_a, rates_b, common_rates), (r - k, s - k, k),
            T, prior)
    return answer


def mark_factorial_moments(marks, probabilities, degree):
    """Return E[(J1)_a (J2)_b] for a bivariate integer mark."""
    marks = np.asarray(marks, int)
    probabilities = np.asarray(probabilities, float)
    if marks.ndim != 2 or marks.shape[1] != 2:
        raise ValueError("marks must have shape (number of marks, 2)")
    if np.any(marks < 0) or np.any(probabilities < 0):
        raise ValueError("marks and probabilities must be nonnegative")
    if len(probabilities) != len(marks) or not np.isclose(
            probabilities.sum(), 1.0):
        raise ValueError("mark probabilities must match and sum to one")
    falling = np.ones((2, degree + 1, len(marks)))
    for axis in range(2):
        for order in range(1, degree + 1):
            falling[axis, order] = (
                falling[axis, order - 1] * (marks[:, axis] - order + 1))
    return np.einsum("ai,bi,i->ab", falling[0], falling[1], probabilities)


def marked_common_shock_factorial_cumulant(
        Q, rates1, rates2, common_rates, mark_moments, r, s, T, prior):
    """Partition formula for an arbitrarily marked common Poisson stream.

    A block containing ``a`` derivatives in the first coordinate and ``b``
    in the second contributes the integrated-rate variable W_ab.  Singleton
    variables are A=Lambda1+E[J1]C and B=Lambda2+E[J2]C; every larger block is
    E[(J1)_a (J2)_b] C.  Summing their joint cumulants over set partitions is
    the multivariate logarithmic Faa di Bruno formula.
    """
    rates1, rates2 = np.asarray(rates1), np.asarray(rates2)
    common_rates = np.asarray(common_rates)
    variables = {
        (1, 0): rates1 + mark_moments[1, 0] * common_rates,
        (0, 1): rates2 + mark_moments[0, 1] * common_rates,
    }
    labels = tuple([0] * r + [1] * s)
    answer = 0.0
    for partition in _set_partitions(labels):
        multiplicities = {}
        block_rates = {}
        vanishes = False
        for block in partition:
            block_type = (block.count(0), block.count(1))
            if block_type not in variables:
                coefficient = mark_moments[block_type]
                if coefficient == 0.0:
                    vanishes = True
                    break
                variables[block_type] = coefficient * common_rates
            multiplicities[block_type] = multiplicities.get(block_type, 0) + 1
            block_rates[block_type] = variables[block_type]
        if vanishes:
            continue
        types = tuple(multiplicities)
        answer += integrated_intensity_joint_cumulant(
            Q,
            tuple(block_rates[item] for item in types),
            tuple(multiplicities[item] for item in types),
            T,
            prior,
        )
    return answer


def common_shock_factorial_cumulants22(Q, rates1, rates2, common_rates,
                                        T, prior):
    """Predicted mixed factorial cumulants with a shared Poisson component.

    Conditional on the regime path, N1=I1+C and N2=I2+C, where the three
    Poisson components have cumulative intensities Lambda1, Lambda2 and Cbar.
    Put A=Lambda1+Cbar and B=Lambda2+Cbar.  Substitution of v=t1*t2 in the
    joint cumulant generating function gives the four returned terms.
    """
    rates_a = np.asarray(rates1) + np.asarray(common_rates)
    rates_b = np.asarray(rates2) + np.asarray(common_rates)
    base = integrated_intensity_mixed_cumulants(
        Q, rates_a, rates_b, T, prior)
    cov_ac = integrated_intensity_mixed_cumulants(
        Q, rates_a, common_rates, T, prior)[0]
    cov_bc = integrated_intensity_mixed_cumulants(
        Q, rates_b, common_rates, T, prior)[0]
    common_kappa = integrated_intensity_cumulants(
        Q, common_rates, T, prior)
    joint_abc = integrated_intensity_joint_cumulant111(
        Q, rates_a, rates_b, common_rates, T, prior)
    return base + np.array([
        common_kappa[0],
        2.0 * cov_ac,
        2.0 * cov_bc,
        4.0 * joint_abc + 2.0 * common_kappa[1],
    ])


def bivariate_count_mixed_cumulants(
        Q, rates1, rates2, T, prior, max_count=75, common_rates=None,
        common_marks=None, common_mark_probabilities=None):
    """Mixed count cumulants from a sparse two-count forward master equation."""
    Q = np.asarray(Q, float)
    rates1, rates2 = np.asarray(rates1, float), np.asarray(rates2, float)
    prior = np.asarray(prior, float)
    common_rates = (np.zeros_like(rates1) if common_rates is None
                    else np.asarray(common_rates, float))
    if common_marks is None:
        common_marks = np.array([[1, 1]])
        common_mark_probabilities = np.array([1.0])
    else:
        if common_mark_probabilities is None:
            raise ValueError("common mark probabilities are required")
        common_marks = np.asarray(common_marks, int)
        common_mark_probabilities = np.asarray(
            common_mark_probabilities, float)
    if (common_marks.ndim != 2 or common_marks.shape != (len(common_marks), 2)
            or len(common_marks) == 0
            or len(common_mark_probabilities) != len(common_marks)
            or np.any(common_marks < 0)
            or np.any(common_mark_probabilities < 0)
            or not np.isclose(common_mark_probabilities.sum(), 1.0)):
        raise ValueError("invalid common mark distribution")
    n, m = len(prior), max_count + 1
    shift = diags(np.ones(m - 1), -1, shape=(m, m), format="csr")
    count_eye = eye(m, format="csr")
    regime_eye = eye(m * m, format="csr")
    within = Q.T - np.diag(rates1 + rates2 + common_rates)
    A = (kron(regime_eye, within, format="csr")
         + kron(kron(shift, count_eye), np.diag(rates1), format="csr")
         + kron(kron(count_eye, shift), np.diag(rates2), format="csr"))
    shift_powers = {
        0: count_eye,
        **{
            jump: diags(np.ones(m - jump), -jump, shape=(m, m), format="csr")
            for jump in range(1, common_marks.max() + 1)
        },
    }
    for mark, probability in zip(common_marks, common_mark_probabilities):
        A += kron(
            kron(shift_powers[mark[0]], shift_powers[mark[1]]),
            probability * np.diag(common_rates),
            format="csr",
        )
    y0 = np.zeros(m * m * n)
    y0[:n] = prior
    p = np.asarray(expm_multiply(A * T, y0)).reshape(m, m, n).sum(axis=2)
    grid = np.arange(m, dtype=float)
    raw = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            raw[a, b] = np.einsum("i,j,ij->", grid ** a, grid ** b, p)
    return mixed_cumulants22(raw), p


def compound_poisson_tail_bound(
        threshold, horizon, idiosyncratic_rate, common_rate,
        marks, probabilities, coordinate):
    """Chernoff bound for a coordinate of the dominating marked count."""
    marks = np.asarray(marks, float)[:, coordinate]
    probabilities = np.asarray(probabilities, float)
    candidates = np.linspace(0.02, 4.0, 2000)
    exponents = (
        horizon * idiosyncratic_rate * np.expm1(candidates)
        + horizon * common_rate
        * (np.exp(np.outer(candidates, marks)) @ probabilities - 1.0)
        - threshold * candidates
    )
    return float(np.exp(exponents.min()))


def two_state_exact(rates, T, lam, sign=1.0):
    """Exact known-start mean and overdispersion for the symmetric chain."""
    mean_rate = 0.5 * (rates[0] + rates[1])
    half_difference = 0.5 * (rates[0] - rates[1])
    eps = 1.0 / lam
    L = 1.0 - math.exp(-2.0 * lam * T)
    mean = mean_rate * T + sign * 0.5 * eps * half_difference * L
    overdispersion = (eps * half_difference ** 2 * T
                      - eps ** 2 * half_difference ** 2 * (0.5 * L + 0.25 * L ** 2))
    return mean, overdispersion


def main():
    # A genuinely nonreversible chain: all three stationary edge currents are nonzero.
    Q = np.array([[-3.0, 2.0, 1.0],
                  [1.0, -4.0, 3.0],
                  [2.0, 1.0, -3.0]])
    rates = np.array([0.5, 3.0, 6.0])
    prior = np.array([0.2, 0.5, 0.3])
    intensity_kappa = integrated_intensity_cumulants(Q, rates, 1.3, prior)
    count_kappa, p = count_cumulants(Q, rates, 1.3, prior)
    factorial_kappa = factorial_cumulants4(count_kappa)
    identity_error = np.max(np.abs(factorial_kappa - intensity_kappa))
    # Conditional on every path, Lambda_T <= max(rates) T, so this Poisson
    # survival probability bounds all mass omitted by the count truncation.
    tail_bound = poisson.sf(len(p) - 1, rates.max() * 1.3)

    # Two conditionally independent count streams share the same hidden chain.
    # Mixed factorial cumulants should remove each stream's own shot noise and
    # recover the joint cumulants of the two cumulative intensities.
    rates_b = np.array([4.0, 0.75, 2.5])
    mixed_intensity = integrated_intensity_mixed_cumulants(
        Q, rates, rates_b, 1.3, prior)
    mixed_count, p_bi = bivariate_count_mixed_cumulants(
        Q, rates, rates_b, 1.3, prior)
    mixed_factorial = mixed_factorial_cumulants22(mixed_count)
    mixed_error = np.max(np.abs(mixed_factorial - mixed_intensity))
    bivariate_tail_bound = (poisson.sf(p_bi.shape[0] - 1, rates.max() * 1.3)
                            + poisson.sf(p_bi.shape[1] - 1, rates_b.max() * 1.3))

    # Shared events are a different observation model.  Conditional on the
    # path, a third Poisson stream increments both observed counts at once.
    # Its shot noise survives the mixed factorial transform.  The predicted
    # terms follow by substituting v=t1*t2 in K(A*t1+B*t2+Cbar*v).
    common_rates = np.array([0.4, 1.2, 0.7])
    common_count, p_common = bivariate_count_mixed_cumulants(
        Q, rates, rates_b, 1.3, prior, max_count=80,
        common_rates=common_rates)
    common_factorial = mixed_factorial_cumulants22(common_count)
    common_predicted = common_shock_factorial_cumulants22(
        Q, rates, rates_b, common_rates, 1.3, prior)
    common_error = np.max(np.abs(common_factorial - common_predicted))
    common_factorial_grid = bivariate_factorial_cumulant_grid(
        p_common, degree=3)
    common_predicted_grid = np.array([
        [common_shock_factorial_cumulant(
            Q, rates, rates_b, common_rates, r, s, 1.3, prior)
         for s in range(1, 4)]
        for r in range(1, 4)
    ])
    common_all_order_error = np.max(
        np.abs(common_factorial_grid - common_predicted_grid))
    common_tail_bound = (
        poisson.sf(p_common.shape[0] - 1,
                   (rates + common_rates).max() * 1.3)
        + poisson.sf(p_common.shape[1] - 1,
                     (rates_b + common_rates).max() * 1.3)
    )

    # Marked common events replace the unit increment (1,1) by an iid
    # integer-valued vector J.  The independent forward equation shifts by
    # each possible mark.  The partition formula uses only the falling-
    # factorial moments E[(J1)_a (J2)_b] and integrated-rate cumulants.
    common_marks = np.array([
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 2],
        [3, 1],
        [1, 3],
    ])
    common_mark_probabilities = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
    mark_moments = mark_factorial_moments(
        common_marks, common_mark_probabilities, degree=3)
    marked_count, p_marked = bivariate_count_mixed_cumulants(
        Q,
        rates,
        rates_b,
        1.3,
        prior,
        max_count=110,
        common_rates=common_rates,
        common_marks=common_marks,
        common_mark_probabilities=common_mark_probabilities,
    )
    marked_factorial_grid = bivariate_factorial_cumulant_grid(
        p_marked, degree=3)
    marked_predicted_grid = np.array([
        [marked_common_shock_factorial_cumulant(
            Q,
            rates,
            rates_b,
            common_rates,
            mark_moments,
            r,
            s,
            1.3,
            prior,
        ) for s in range(1, 4)]
        for r in range(1, 4)
    ])
    marked_error = np.max(
        np.abs(marked_factorial_grid - marked_predicted_grid))
    marked_tail_bound = sum(
        compound_poisson_tail_bound(
            p_marked.shape[axis] - 1,
            1.3,
            (rates, rates_b)[axis].max(),
            common_rates.max(),
            common_marks,
            common_mark_probabilities,
            axis,
        )
        for axis in range(2)
    )

    # The marked partition theorem must reduce exactly to the previous
    # matching formula when every common mark is (1,1).
    unit_mark_moments = mark_factorial_moments(
        np.array([[1, 1]]), np.array([1.0]), degree=3)
    unit_partition_grid = np.array([
        [marked_common_shock_factorial_cumulant(
            Q,
            rates,
            rates_b,
            common_rates,
            unit_mark_moments,
            r,
            s,
            1.3,
            prior,
        ) for s in range(1, 4)]
        for r in range(1, 4)
    ])
    unit_reduction_error = np.max(
        np.abs(unit_partition_grid - common_predicted_grid))

    # With only a deterministic common component, marginal cumulative rates
    # have no stochastic covariance, yet kappa_11^(F) equals its mean.  This
    # is the sharp counterexample to identification without conditional
    # independence.
    common_only_rate = 0.8
    zero = np.zeros(3)
    deterministic_common = np.full(3, common_only_rate)
    common_only_count, _ = bivariate_count_mixed_cumulants(
        Q, zero, zero, 1.3, prior, max_count=45,
        common_rates=deterministic_common)
    common_only_factorial = mixed_factorial_cumulants22(common_only_count)
    common_only_expected = np.array([common_only_rate * 1.3, 0.0, 0.0, 0.0])
    common_only_error = np.max(
        np.abs(common_only_factorial - common_only_expected))

    # The page's symmetric two-state example, checked for both known starts.
    rates2 = np.array([8.0, 1.0])
    closed_errors = []
    for lam in (5.0, 10.0, 20.0, 40.0):
        Q2 = lam * np.array([[-1.0, 1.0], [1.0, -1.0]])
        for start, sign in ((0, 1.0), (1, -1.0)):
            prior2 = np.eye(2)[start]
            kap = integrated_intensity_cumulants(Q2, rates2, 1.0, prior2)
            mean, overdispersion = two_state_exact(rates2, 1.0, lam, sign)
            closed_errors.extend((abs(kap[0] - mean), abs(kap[1] - overdispersion)))

    # The omitted boundary term is second order at fixed positive maturity.
    lams = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    residuals = []
    half_difference = 3.5
    for lam in lams:
        _, exact_gap = two_state_exact(rates2, 1.0, lam)
        residuals.append(abs(exact_gap - half_difference ** 2 / lam))
    residual_order = -np.polyfit(np.log(lams), np.log(residuals), 1)[0]

    # Numbers printed on the page, obtained here by independent constructions.
    Q10 = 10.0 * np.array([[-1.0, 1.0], [1.0, -1.0]])
    intensity10 = integrated_intensity_cumulants(Q10, rates2, 1.0, [1.0, 0.0])
    count10, _ = count_cumulants(Q10, rates2, 1.0, [1.0, 0.0])

    print("nonreversible factorial identity max error", f"{identity_error:.3e}")
    print("count-truncation tail bound", f"{tail_bound:.3e}")
    print("mixed factorial identity max error", f"{mixed_error:.3e}")
    print("bivariate count-truncation tail bound", f"{bivariate_tail_bound:.3e}")
    print("mixed factorial/intensity cumulants (11, 21, 12, 22)",
          " ".join(f"{x:.8f}" for x in mixed_intensity))
    print("common-shock formula max error", f"{common_error:.3e}")
    print("common-shock all-order formula max error through (3,3)",
          f"{common_all_order_error:.3e}")
    print("common-shock count-truncation tail bound", f"{common_tail_bound:.3e}")
    print("common-shock mixed factorial cumulants (11, 21, 12, 22)",
          " ".join(f"{x:.8f}" for x in common_factorial))
    print("common-shock (3,3) factorial cumulant",
          f"{common_factorial_grid[2, 2]:.8f}")
    print("marked common-shock formula max error through (3,3)",
          f"{marked_error:.3e}")
    print("marked common-shock count-truncation tail bound",
          f"{marked_tail_bound:.3e}")
    print("marked common-shock (3,3) factorial cumulant",
          f"{marked_factorial_grid[2, 2]:.8f}")
    print("unit-mark reduction max error", f"{unit_reduction_error:.3e}")
    print("common-only mixed factorial cumulants",
          " ".join(f"{x:.8f}" for x in common_only_factorial))
    print("two-state closed-form max error", f"{max(closed_errors):.3e}")
    print("first-order overdispersion residual order", f"{residual_order:.6f}")
    print("lambda=10 ordinary count cumulants", " ".join(f"{x:.8f}" for x in count10))
    print("lambda=10 factorial/intensity cumulants", " ".join(f"{x:.8f}" for x in intensity10))

    assert identity_error < 2e-9
    assert tail_bound < 1e-45
    assert mixed_error < 2e-9
    assert bivariate_tail_bound < 1e-45
    assert common_error < 2e-9
    assert common_all_order_error < 2e-8
    assert common_tail_bound < 1e-45
    assert marked_error < 2e-7
    assert marked_tail_bound < 1e-35
    assert unit_reduction_error < 2e-10
    assert common_only_error < 2e-9
    assert max(closed_errors) < 2e-12
    assert abs(residual_order - 2.0) < 0.01
    assert abs(count10[1] - 5.808125) < 2e-10
    print("PASS")


if __name__ == "__main__":
    main()
