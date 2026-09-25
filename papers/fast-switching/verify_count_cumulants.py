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
nonreversible three-state chain.  It then verifies the closed two-state
finite-horizon formula used on the counts page and its O(lambda^-2)
first-order residual.
"""

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


def bivariate_count_mixed_cumulants(Q, rates1, rates2, T, prior,
                                     max_count=75):
    """Mixed count cumulants from a sparse two-count forward master equation."""
    Q = np.asarray(Q, float)
    rates1, rates2 = np.asarray(rates1, float), np.asarray(rates2, float)
    prior = np.asarray(prior, float)
    n, m = len(prior), max_count + 1
    shift = diags(np.ones(m - 1), -1, shape=(m, m), format="csr")
    count_eye = eye(m, format="csr")
    regime_eye = eye(m * m, format="csr")
    within = Q.T - np.diag(rates1 + rates2)
    A = (kron(regime_eye, within, format="csr")
         + kron(kron(shift, count_eye), np.diag(rates1), format="csr")
         + kron(kron(count_eye, shift), np.diag(rates2), format="csr"))
    y0 = np.zeros(m * m * n)
    y0[:n] = prior
    p = np.asarray(expm_multiply(A * T, y0)).reshape(m, m, n).sum(axis=2)
    grid = np.arange(m, dtype=float)
    raw = np.empty((3, 3))
    for a in range(3):
        for b in range(3):
            raw[a, b] = np.einsum("i,j,ij->", grid ** a, grid ** b, p)
    return mixed_cumulants22(raw), p


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
    print("two-state closed-form max error", f"{max(closed_errors):.3e}")
    print("first-order overdispersion residual order", f"{residual_order:.6f}")
    print("lambda=10 ordinary count cumulants", " ".join(f"{x:.8f}" for x in count10))
    print("lambda=10 factorial/intensity cumulants", " ".join(f"{x:.8f}" for x in intensity10))

    assert identity_error < 2e-9
    assert tail_bound < 1e-45
    assert mixed_error < 2e-9
    assert bivariate_tail_bound < 1e-45
    assert max(closed_errors) < 2e-12
    assert abs(residual_order - 2.0) < 0.01
    assert abs(count10[1] - 5.808125) < 2e-10
    print("PASS")


if __name__ == "__main__":
    main()
