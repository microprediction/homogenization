"""Checks finite-chain conditional-coverage crossover and calibration variance.

Scores conditional on terminal regime are exponential with scales 1 and 3.
The Monte Carlo uses CTMC endpoint transitions, not the analytic coverage
or covariance formula used by the note.  Nonreversible three-state examples
check the exact semigroup identity and the additive-reversibilization bounds;
a reversible three-state example checks the corresponding specialization.
An exact hidden-state enumeration checks the burn-in transfer from an
arbitrary initial regime law to a stationary calibration panel.  It also
checks exact transient mean and variance identities for the empirical CDF.
"""
import itertools
import math

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

TARGET = 0.9
EPS = 0.2
SCALES = np.array([1.0, 3.0])


def pooled_cdf(x):
    return 1 - np.exp(-x / SCALES).mean()


def stationary(Q):
    A = Q.T.copy()
    A[-1] = 1
    b = np.zeros(len(Q))
    b[-1] = 1
    return np.linalg.solve(A, b)


def cdf_vector(x, scales):
    return 1 - np.exp(-x / scales)


def pooled_quantile(scales, pi):
    return brentq(lambda x: pi @ cdf_vector(x, scales) - TARGET,
                  0.0, 50.0)


def endpoint_simulation(rng, P, scales, q, draws=120000):
    observed = []
    for start in range(len(scales)):
        final = rng.choice(len(scales), size=draws, p=P[start])
        scores = rng.exponential(scale=scales[final])
        observed.append(np.mean(scores <= q))
    return np.asarray(observed)


def additive_gap(Q, pi):
    """Gap of -(Q+Q*)/2 on centered L2(pi)."""
    D = np.diag(pi)
    Q_star = np.diag(1 / pi) @ Q.T @ D
    Q_symmetric = (Q + Q_star) / 2
    similarity = (np.diag(np.sqrt(pi)) @ Q_symmetric
                  @ np.diag(1 / np.sqrt(pi)))
    assert np.allclose(similarity, similarity.T, atol=2e-15)
    eigenvalues = np.linalg.eigvalsh(-similarity)
    assert abs(eigenvalues[0]) < 2e-14 and eigenvalues[1] > 0
    return eigenvalues[1]


def additive_gap_witness(Q, pi):
    """Return the gap and a centered unit eigenfunction attaining its slope."""
    D_sqrt = np.diag(np.sqrt(pi))
    D_inv_sqrt = np.diag(1 / np.sqrt(pi))
    Q_star = np.diag(1 / pi) @ Q.T @ np.diag(pi)
    similarity = D_sqrt @ ((Q + Q_star) / 2) @ D_inv_sqrt
    eigenvalues, eigenvectors = np.linalg.eigh(-similarity)
    witness = D_inv_sqrt @ eigenvectors[:, 1]
    assert abs(pi @ witness) < 2e-14
    assert abs(pi @ witness ** 2 - 1) < 2e-14
    return eigenvalues[1], witness


def finite_variance_factor(a, n):
    """Exact geometric factor for n observations with correlation envelope a^lag."""
    direct = 1 + 2 / n * sum((n - lag) * a ** lag
                             for lag in range(1, n))
    closed = ((1 + a) / (1 - a)
              - 2 * a * (1 - a ** n) / (n * (1 - a) ** 2))
    assert abs(direct - closed) < 2e-14
    return closed


def binary_panel_law(initial, P, success, n):
    """Exact law of n binary emissions from a finite-state hidden chain."""
    law = []
    for pattern in itertools.product((0, 1), repeat=n):
        forward = initial.copy()
        for t, bit in enumerate(pattern):
            emission = success if bit else 1 - success
            forward = forward * emission
            if t + 1 < n:
                forward = forward @ P
        law.append(forward.sum())
    law = np.asarray(law)
    assert abs(law.sum() - 1) < 2e-14
    return law


def panel_moments_from_law(law, n):
    """Mean and variance of the empirical success rate from a panel law."""
    values = np.asarray([sum(pattern) / n
                         for pattern in itertools.product((0, 1), repeat=n)])
    mean = law @ values
    variance = law @ values ** 2 - mean ** 2
    return mean, variance


def transient_panel_moments(initial, P, success, n):
    """Exact empirical-CDF moments from transient Markov reward sums."""
    marginals = []
    second_moment = 0.0
    distribution = initial.copy()
    powers = [np.linalg.matrix_power(P, lag) for lag in range(n)]
    for left in range(n):
        marginal = distribution @ success
        marginals.append(marginal)
        second_moment += marginal
        for right in range(left + 1, n):
            joint = distribution @ (success * (powers[right - left] @ success))
            second_moment += 2 * joint
        distribution = distribution @ P
    mean = sum(marginals) / n
    variance = second_moment / n ** 2 - mean ** 2
    return mean, variance


def transient_panel_checks(Q, pi, gamma_s, success):
    """Check transient panel moments and the within-panel relaxation bound."""
    initial = np.array([1.0, 0.0, 0.0])
    density_norm = math.sqrt(1 / pi[0] - 1)
    centered = success - pi @ success
    score_norm = math.sqrt(pi @ centered ** 2)
    spacing = 0.3
    n = 4
    P = expm(spacing * Q)
    a = math.exp(-gamma_s * spacing)
    averaging_factor = (1 - a ** n) / (n * (1 - a))
    rows = []
    for scaled_burnin in (0.0, 0.25, 0.5, 1.0):
        burned = initial @ expm(scaled_burnin * Q)
        law = binary_panel_law(burned, P, success, n)
        enumerated_mean, enumerated_variance = panel_moments_from_law(law, n)
        exact_mean, exact_variance = transient_panel_moments(
            burned, P, success, n)
        assert abs(enumerated_mean - exact_mean) < 2e-15
        assert abs(enumerated_variance - exact_variance) < 2e-15
        bias = exact_mean - pi @ success
        score_bound = (density_norm * score_norm
                       * math.exp(-gamma_s * scaled_burnin)
                       * averaging_factor)
        assert abs(bias) <= score_bound + 2e-14
        rows.append((scaled_burnin, bias, exact_variance, score_bound))

    # A symmetric two-state chain and success vector (0,1) attain the
    # universal 1/2-prefactor bound for every n, spacing and burn-in.
    sharp_Q = np.array([[-1.0, 1.0], [1.0, -1.0]])
    sharp_pi = np.array([0.5, 0.5])
    sharp_initial = np.array([1.0, 0.0])
    sharp_success = np.array([0.0, 1.0])
    sharp_n, sharp_spacing, sharp_burnin = 7, 0.35, 0.2
    sharp_P = expm(sharp_spacing * sharp_Q)
    sharp_burned = sharp_initial @ expm(sharp_burnin * sharp_Q)
    sharp_mean, sharp_variance = transient_panel_moments(
        sharp_burned, sharp_P, sharp_success, sharp_n)
    sharp_a = math.exp(-2 * sharp_spacing)
    sharp_factor = (1 - sharp_a ** sharp_n) / (sharp_n * (1 - sharp_a))
    sharp_bound = 0.5 * math.exp(-2 * sharp_burnin) * sharp_factor
    assert abs(abs(sharp_mean - sharp_pi @ sharp_success) - sharp_bound) < 2e-15
    sharp_law = binary_panel_law(sharp_burned, sharp_P, sharp_success, sharp_n)
    law_mean, law_variance = panel_moments_from_law(sharp_law, sharp_n)
    assert abs(law_mean - sharp_mean) < 2e-15
    assert abs(law_variance - sharp_variance) < 2e-15

    print("transient empirical-CDF moments:")
    for scaled_burnin, bias, variance, score_bound in rows:
        print(f"  mb={scaled_burnin:4.2f} bias {bias:+.8f},"
              f" variance {variance:.8f}, bound {score_bound:.8f}")
    print("  sharp two-state bias:",
          f"exact {sharp_mean - 0.5:+.8f}, bound {sharp_bound:.8f}")


def burnin_transfer_checks(Q, pi, gamma_s, success):
    """Check TV transfer from a nonstationary start to a score panel."""
    initial = np.array([1.0, 0.0, 0.0])
    density_norm = math.sqrt(1 / pi[0] - 1)
    panel_transition = expm(0.3 * Q)
    stationary_panel = binary_panel_law(pi, panel_transition, success, 4)
    rows = []
    for scaled_burnin in (0.0, 0.25, 0.5, 1.0):
        burned = initial @ expm(scaled_burnin * Q)
        state_tv = 0.5 * np.abs(burned - pi).sum()
        panel = binary_panel_law(burned, panel_transition, success, 4)
        panel_tv = 0.5 * np.abs(panel - stationary_panel).sum()
        l2_bound = 0.5 * density_norm * math.exp(-gamma_s * scaled_burnin)
        assert panel_tv <= state_tv + 2e-14
        assert state_tv <= l2_bound + 2e-14
        rows.append((scaled_burnin, panel_tv, state_tv, l2_bound))

    tolerance = 0.01
    required_scaled_burnin = max(
        0.0, math.log(density_norm / (2 * tolerance)) / gamma_s)
    burned = initial @ expm(required_scaled_burnin * Q)
    required_state_tv = 0.5 * np.abs(burned - pi).sum()
    assert required_state_tv <= tolerance + 2e-14
    print("nonstationary panel burn-in transfer:")
    for scaled_burnin, panel_tv, state_tv, l2_bound in rows:
        print(f"  mb={scaled_burnin:4.2f} panel TV {panel_tv:.8f},"
              f" state TV {state_tv:.8f}, L2 bound {l2_bound:.8f}")
    print("  1% L2 burn-in certificate:",
          f"mb >= {required_scaled_burnin:.8f},"
          f" exact state TV {required_state_tv:.8f}")
    transient_panel_checks(Q, pi, gamma_s, success)


def nonreversible_contraction_checks():
    # This chain has a nonuniform invariant law and violates detailed balance.
    # Its additive reversibilization nevertheless gives an L2(pi) contraction
    # rate for the original, nonnormal semigroup.
    Q = np.array([[-3.0, 2.7, 0.3],
                  [0.2, -2.2, 2.0],
                  [2.4, 0.4, -2.8]])
    pi = stationary(Q)
    assert not np.allclose(np.diag(pi) @ Q, Q.T @ np.diag(pi))
    gamma_s, witness = additive_gap_witness(Q, pi)
    scales = np.array([0.7, 1.5, 4.0])
    q = pooled_quantile(scales, pi)
    f = cdf_vector(q, scales)
    centered = f - TARGET
    norm = math.sqrt(pi @ centered ** 2)

    largest_ratio = 0.0
    for c in np.linspace(0.05, 2.0, 40):
        evolved = expm(c * Q) @ centered
        ratio = math.sqrt(pi @ evolved ** 2) / (norm * math.exp(-gamma_s * c))
        largest_ratio = max(largest_ratio, ratio)
        assert ratio <= 1 + 2e-14
        statewise_bound = np.sqrt(1 / pi - 1) * norm * math.exp(-gamma_s * c)
        assert np.all(abs(evolved) <= statewise_bound + 2e-14)

    # The additive gap is not merely sufficient.  Its eigenfunction attains
    # the logarithmic norm derivative at zero, so every larger prefactor-one
    # exponential rate fails immediately.
    sharp_time = 1e-3
    witness_norm = math.sqrt(pi @ witness ** 2)
    witness_evolved = expm(sharp_time * Q) @ witness
    sharp_ratio = (math.sqrt(pi @ witness_evolved ** 2)
                   / (witness_norm * math.exp(-gamma_s * sharp_time)))
    faster_ratio = (math.sqrt(pi @ witness_evolved ** 2)
                    / (witness_norm * math.exp(-1.01 * gamma_s * sharp_time)))
    assert sharp_ratio <= 1 + 2e-14
    assert faster_ratio > 1

    m, h, n = 5.0, 0.1, 100
    P = expm(m * h * Q)
    exact_var = TARGET * (1 - TARGET) / n
    Pk = np.eye(3)
    for lag in range(1, n):
        Pk = Pk @ P
        covariance = pi @ (centered * (Pk @ centered))
        exact_var += 2 * (n - lag) * covariance / n ** 2
    a = math.exp(-m * gamma_s * h)
    finite_factor = finite_variance_factor(a, n)
    contraction_bound = TARGET * (1 - TARGET) / n * finite_factor
    assert exact_var <= contraction_bound + 2e-15
    print("nonreversible additive-reversibilization bound:",
          f"gap {gamma_s:.8f}, largest score ratio {largest_ratio:.8f},")
    print("sharp exponent witness:",
          f"envelope ratio {sharp_ratio:.12f},",
          f"1%-faster ratio {faster_ratio:.12f}")
    print("nonreversible calibration variance:",
          f"exact {exact_var:.8f}, finite-n bound {contraction_bound:.8f},",
          f"factor {finite_factor:.8f}")

    # Sharpness of the finite-n variance factor.  A symmetric two-state chain
    # and conditional CDF values (0, 1) attain every covariance envelope
    # exactly; continuous score laws with disjoint supports realize this case.
    saturation_n = 17
    saturation_a = math.exp(-0.7)
    saturation_factor = finite_variance_factor(saturation_a, saturation_n)
    saturation_bound = 0.25 / saturation_n * saturation_factor
    saturation_exact = (0.25 / saturation_n
                        + 0.5 / saturation_n ** 2
                        * sum((saturation_n - lag) * saturation_a ** lag
                              for lag in range(1, saturation_n)))
    assert abs(saturation_exact - saturation_bound) < 2e-15
    print("finite-n factor saturation:",
          f"n {saturation_n}, a {saturation_a:.8f},",
          f"variance {saturation_exact:.8f}")
    burnin_transfer_checks(Q, pi, gamma_s, f)


def finite_chain_checks(rng):
    # A directed cycle is irreducible and nonreversible.  Its conditional
    # coverage is a matrix-exponential mixture, not one scalar exponential.
    Q_cycle = np.array([[-1.7, 1.5, 0.2],
                        [0.2, -1.7, 1.5],
                        [1.5, 0.2, -1.7]])
    scales = np.array([0.7, 1.5, 4.0])
    pi = stationary(Q_cycle)
    q = pooled_quantile(scales, pi)
    f = cdf_vector(q, scales)
    print("\nnonreversible finite-chain endpoint theorem")
    for c in (0.0, 0.5, 1.0, 2.0):
        P = expm(c * Q_cycle)
        theory = P @ f
        observed = endpoint_simulation(rng, P, scales, q)
        assert max(abs(observed - theory)) < 0.004
        print(f"c={c:3.1f} theory {theory.round(5)} simulation {observed.round(5)}")
    # Delta=c/m leaves the same nonzero crossover at every switching rate.
    assert np.allclose(expm(5 * Q_cycle * (1 / 5)),
                       expm(20 * Q_cycle * (1 / 20)), atol=2e-15)

    nonreversible_contraction_checks()

    # Symmetric Q is reversible with uniform pi.  Check the L2 spectral-gap
    # bound and the general finite-chain covariance identity.
    Q_rev = np.array([[-1.0, 1.0, 0.0],
                      [1.0, -3.0, 2.0],
                      [0.0, 2.0, -2.0]])
    pi = stationary(Q_rev)
    assert np.allclose(pi, np.ones(3) / 3)
    gap = additive_gap(Q_rev, pi)
    q = pooled_quantile(scales, pi)
    f = cdf_vector(q, scales)
    centered = f - TARGET
    f_norm = math.sqrt(pi @ centered ** 2)
    for c in (0.25, 0.5, 1.0, 2.0):
        errors = abs(expm(c * Q_rev) @ f - TARGET)
        bounds = np.sqrt(1 / pi - 1) * f_norm * math.exp(-gap * c)
        assert np.all(errors <= bounds + 1e-14)

    m, h, n, reps = 5.0, 0.1, 100, 6000
    P = expm(m * h * Q_rev)
    exact_var = TARGET * (1 - TARGET) / n
    Pk = np.eye(3)
    for lag in range(1, n):
        Pk = Pk @ P
        covariance = pi @ (centered * (Pk @ centered))
        exact_var += 2 * (n - lag) * covariance / n ** 2
    a = math.exp(-m * gap * h)
    spectral_bound = finite_variance_factor(a, n) / (4 * n)
    assert exact_var <= spectral_bound

    states = rng.choice(3, size=reps, p=pi)
    samples = np.empty((reps, n), dtype=np.int8)
    for t in range(n):
        u = rng.random(reps)
        cumulative = np.cumsum(P[states], axis=1)
        states = (u[:, None] > cumulative).sum(axis=1)
        samples[:, t] = (rng.exponential(scale=scales[states]) <= q)
    empirical_var = samples.mean(axis=1).var(ddof=1)
    assert abs(empirical_var - exact_var) / exact_var < 0.08
    print("reversible calibration variance:",
          f"empirical {empirical_var:.8f}, exact {exact_var:.8f},",
          f"spectral bound {spectral_bound:.8f}, gap {gap:.6f}")


def main():
    rng = np.random.default_rng(20260923)
    q = brentq(lambda x: pooled_cdf(x) - TARGET, 0.0, 30.0)
    conditional = 1 - np.exp(-q / SCALES)
    print(f"pooled 90% threshold: {q:.8f}")
    print("Delta/epsilon  theory in states 1,2  independent simulation in states 1,2")
    for ratio in (0.0, 0.5, 1.0, 2.0, 4.0):
        decay = math.exp(-2 * ratio)
        theory = TARGET + decay * (conditional - TARGET)
        observed = []
        for start in (0, 1):
            # Symmetric two-state CTMC endpoint transition, then fresh emission.
            final = np.where(rng.random(150000) < (1 + decay) / 2, start, 1 - start)
            score = rng.exponential(scale=SCALES[final])
            observed.append(np.mean(score <= q))
        assert max(abs(np.array(observed) - theory)) < 0.004
        print(f"{ratio:5.1f}          {theory.round(5)}            {np.round(observed, 5)}")

    n, h, reps = 100, 0.1, 6000
    decay = math.exp(-2 * h / EPS)
    states = rng.integers(0, 2, size=reps)
    samples = np.empty((reps, n), dtype=np.int8)
    for t in range(n):
        states ^= (rng.random(reps) < (1 - decay) / 2)
        samples[:, t] = (rng.exponential(scale=SCALES[states]) <= q)
    empirical_var = samples.mean(axis=1).var(ddof=1)
    d = (conditional[0] - conditional[1]) / 2
    predicted_var = (TARGET * (1 - TARGET) / n
                     + 2 * d ** 2 / n ** 2
                     * sum((n - lag) * decay ** lag for lag in range(1, n)))
    print(f"empirical calibration-CDF variance: {empirical_var:.8f}")
    print(f"exact calibration-CDF variance:     {predicted_var:.8f}")
    assert abs(empirical_var - predicted_var) / predicted_var < 0.07
    finite_chain_checks(rng)
    print("PASS: finite-chain crossover, additive/reversible spectral bounds,"
          " calibration variance, nonstationary burn-in transfer, and"
          " transient panel moments")


if __name__ == "__main__":
    main()
