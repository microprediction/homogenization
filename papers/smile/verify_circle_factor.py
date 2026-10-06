"""Certificate for time reversal of Green--Kubo forms.

The transpose identity is first checked for a nonreversible finite-state
Markov chain with a nonuniform invariant law.  The same certificate records
the important converse distinction: the full centered Green--Kubo form
detects nonreversibility, whereas a selected feature block need not do so.
It then checks that two independent Gaussian feature probes detect every
finite-state irreversible chain almost surely and that the mean-square
antisymmetric signal is the squared Hilbert--Schmidt norm of the skew
resolvent.  Fourth-moment calculations for both Gaussian and bounded
Rademacher probes turn this population result into a finite random-probe
certificate: 47 independent probe pairs give a universal one-percent miss
bound at half the root-mean-square signal.  Unlike Gaussian probes, a single
Rademacher pair can miss a nonzero skew form with positive probability.
Median-of-means aggregation upgrades the finite-variance energy identity to
an explicit relative-error confidence bound logarithmic in the failure level.
A finite group-inverse perturbation calculation then propagates generator and
stationary-law errors into a deterministic interval for the population skew
energy, without claiming a trajectory-level concentration theorem.

For dY=c dt+sqrt(2D)dW modulo 2 pi and the Fourier pair (cos(nY),
sin(nY)), the Green--Kubo matrix is then checked in closed form, by direct
quadrature, and by a periodic chain approximation.  A final variable-
coefficient circle example verifies the scalar-diffusion specialization.
"""
import math
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm, null_space

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "general"))
from effective_generator import (
    effective_generator,
    full_generator,
    gk,
    group_inverse,
    stationary,
)


D = 1.0
C = 2.0
SCALES = (4, 8, 16, 32)


def exact_block(mode, drift=C, diffusion=D):
    """Green--Kubo block for (cos(mode*y), sin(mode*y))."""
    decay = diffusion * mode**2
    frequency = drift * mode
    return np.array([[decay, frequency], [-frequency, decay]]) / (
        2 * (decay**2 + frequency**2)
    )


def quadrature_block(mode, drift=C, diffusion=D):
    """Directly integrate the four stationary correlation functions."""
    decay = diffusion * mode**2
    frequency = drift * mode
    cc = lambda t: 0.5 * math.exp(-decay * t) * math.cos(frequency * t)
    cs = lambda t: 0.5 * math.exp(-decay * t) * math.sin(frequency * t)
    return np.array(
        [
            [quad(cc, 0, np.inf, epsabs=2e-13)[0], quad(cs, 0, np.inf, epsabs=2e-13)[0]],
            [-quad(cs, 0, np.inf, epsabs=2e-13)[0], quad(cc, 0, np.inf, epsabs=2e-13)[0]],
        ]
    )


def circle_chain(size, drift=C, diffusion=D):
    """Second-order periodic CTMC approximation to c*d_y+D*d_yy."""
    h = 2 * math.pi / size
    forward = diffusion / h**2 + drift / (2 * h)
    backward = diffusion / h**2 - drift / (2 * h)
    if min(forward, backward) <= 0:
        raise ValueError("grid too coarse for positive CTMC rates")
    q = np.zeros((size, size))
    for i in range(size):
        q[i, (i + 1) % size] = forward
        q[i, (i - 1) % size] = backward
        q[i, i] = -forward - backward
    y = h * np.arange(size)
    return q, [np.cos(y), np.sin(y)]


def variable_circle_chain(size, current=0.08):
    """Flux-form CTMC for a periodic diffusion with prescribed density/current.

    The continuum generator is
        Lf = p^{-1}(a p f')' + current * p^{-1} f'.
    The first term is self-adjoint in L2(p), and the second is skew-adjoint.
    Centered differences preserve the same adjoint decomposition exactly on
    the grid while converging at second order.
    """
    h = 2 * math.pi / size
    y = h * np.arange(size)
    y_half = y + 0.5 * h
    density = (1.0 + 0.35 * np.cos(y)) / (2 * math.pi)
    half_density = (1.0 + 0.35 * np.cos(y_half)) / (2 * math.pi)
    half_diffusivity = 0.8 + 0.2 * np.sin(y_half)
    conductance = half_density * half_diffusivity
    q = np.zeros((size, size))
    for i in range(size):
        forward = conductance[i] / (density[i] * h**2) + current / (
            2 * density[i] * h
        )
        backward = conductance[(i - 1) % size] / (
            density[i] * h**2
        ) - current / (2 * density[i] * h)
        if min(forward, backward) <= 0:
            raise ValueError("grid too coarse for positive CTMC rates")
        q[i, (i + 1) % size] = forward
        q[i, (i - 1) % size] = backward
        q[i, i] = -forward - backward
    features = [np.cos(y), np.sin(y) + 0.25 * np.cos(2 * y)]
    return q, features


def reverse_generator(q):
    """Stationary time reversal of a row-convention CTMC generator."""
    pi = stationary(q)
    return np.diag(1.0 / pi) @ q.T @ np.diag(pi)


def general_reversal_check():
    """Test transpose, full-space detection, and feature-level blindness."""
    q = np.array(
        [
            [-2.5, 2.0, 0.4, 0.1],
            [0.2, -2.1, 1.6, 0.3],
            [0.7, 0.1, -2.6, 1.8],
            [1.1, 0.5, 0.2, -1.8],
        ]
    )
    pi = stationary(q)
    q_reverse = reverse_generator(q)
    features = np.array(
        [
            [1.0, -0.4, 0.2, 0.7],
            [-0.3, 0.8, 1.1, -0.5],
            [0.6, 0.1, -0.9, 0.4],
        ]
    )
    k = gk(q, features)
    k_reverse = gk(q_reverse, features)
    transpose_error = np.max(abs(k_reverse - k.T))

    semigroup_error = 0.0
    f, h = features[:2]
    f, h = f - pi @ f, h - pi @ h
    for time in (0.1, 0.7, 2.0):
        lhs = pi @ (f * (expm(time * q_reverse) @ h))
        rhs = pi @ (h * (expm(time * q) @ f))
        semigroup_error = max(semigroup_error, abs(lhs - rhs))

    # gk centers the coordinate indicators, which then span the full
    # mean-zero subspace.
    full_features = np.eye(len(pi))
    k_full = gk(q, full_features)
    full_asymmetry = np.max(abs(k_full - k_full.T))
    flux_defect = np.max(
        abs(np.diag(pi) @ q - q.T @ np.diag(pi))
    )

    # A one-feature block is necessarily symmetric, even for this chain.
    scalar_asymmetry = np.max(abs(gk(q, features[:1]) - gk(q, features[:1]).T))

    # A reversible chain with the same nonuniform invariant law is generated
    # from symmetric edge conductances c_ij = pi_i q_ij.
    conductance = np.array(
        [
            [0.0, 0.11, 0.07, 0.05],
            [0.11, 0.0, 0.13, 0.09],
            [0.07, 0.13, 0.0, 0.17],
            [0.05, 0.09, 0.17, 0.0],
        ]
    )
    q_reversible = conductance / pi[:, None]
    np.fill_diagonal(q_reversible, 0.0)
    np.fill_diagonal(q_reversible, -q_reversible.sum(axis=1))
    k_reversible = gk(q_reversible, full_features)
    reversible_asymmetry = np.max(abs(k_reversible - k_reversible.T))

    assert np.max(abs(pi @ q)) < 2e-14
    assert np.max(abs(pi @ q_reverse)) < 2e-14
    assert transpose_error < 2e-14
    assert semigroup_error < 2e-14
    assert flux_defect > 1e-2
    assert full_asymmetry > 1e-2
    assert scalar_asymmetry == 0.0
    assert reversible_asymmetry < 2e-14
    return {
        "pi": pi,
        "transpose_error": transpose_error,
        "semigroup_error": semigroup_error,
        "flux_defect": flux_defect,
        "full_asymmetry": full_asymmetry,
        "scalar_asymmetry": scalar_asymmetry,
        "reversible_asymmetry": reversible_asymmetry,
    }


def canonical_skew_resolvent(q):
    """Skew Green--Kubo resolvent in canonical Euclidean coordinates."""
    pi = stationary(q)
    root = np.sqrt(pi)
    q_group = group_inverse(q)
    resolvent = -(root[:, None] * q_group) / root[None, :]
    return pi, q_group, 0.5 * (resolvent - resolvent.T)


def generator_perturbation_check():
    """Check the deterministic plug-in bound for an estimated generator."""
    q = np.array(
        [
            [-2.5, 2.0, 0.4, 0.1],
            [0.2, -2.1, 1.6, 0.3],
            [0.7, 0.1, -2.6, 1.8],
            [1.1, 0.5, 0.2, -1.8],
        ]
    )
    direction = np.array(
        [
            [0.0, 0.35, -0.20, -0.15],
            [-0.08, 0.0, 0.31, -0.23],
            [0.26, -0.12, 0.0, -0.14],
            [-0.19, 0.27, -0.08, 0.0],
        ]
    )
    np.fill_diagonal(direction, -direction.sum(axis=1))
    direction /= np.linalg.norm(direction, 2)

    pi, q_group, skew = canonical_skew_resolvent(q)
    projector = np.outer(np.ones(len(pi)), pi)
    inverse_shift = np.linalg.inv(q - projector)
    root = np.sqrt(pi)
    energy = np.sum(skew**2)
    errors = []
    bounds = []
    stationary_identity_errors = []
    energy_band_violations = []
    amplitudes = 2.0 ** -np.arange(4, 10)
    for amplitude in amplitudes:
        q_hat = q + amplitude * direction
        assert np.min(q_hat - np.diag(np.diag(q_hat))) >= 0.0
        pi_hat, group_hat, skew_hat = canonical_skew_resolvent(q_hat)
        projector_hat = np.outer(np.ones(len(pi_hat)), pi_hat)

        generator_error = q_hat - q
        stationary_identity = pi_hat - pi + pi_hat @ generator_error @ q_group
        stationary_identity_errors.append(np.linalg.norm(stationary_identity))

        shifted_error = (q_hat - projector_hat) - (q - projector)
        inverse_norm = np.linalg.norm(inverse_shift, 2)
        shifted_size = np.linalg.norm(shifted_error, 2)
        assert inverse_norm * shifted_size < 1.0
        projector_error = np.linalg.norm(projector_hat - projector, 2)
        group_bound = (
            inverse_norm**2
            * shifted_size
            / (1.0 - inverse_norm * shifted_size)
            + projector_error
        )

        root_hat = np.sqrt(pi_hat)
        inverse_root = 1.0 / root
        inverse_root_hat = 1.0 / root_hat
        resolvent_bound = (
            np.max(root_hat) * np.max(inverse_root_hat) * group_bound
            + np.max(abs(root_hat - root))
            * np.linalg.norm(q_group, 2)
            * np.max(inverse_root_hat)
            + np.max(root)
            * np.linalg.norm(q_group, 2)
            * np.max(abs(inverse_root_hat - inverse_root))
        )
        frobenius_bound = math.sqrt(len(pi)) * resolvent_bound
        error = np.linalg.norm(skew_hat - skew)
        assert error <= frobenius_bound * (1.0 + 2e-13)

        energy_hat = np.sum(skew_hat**2)
        lower = max(math.sqrt(energy_hat) - frobenius_bound, 0.0) ** 2
        upper = (math.sqrt(energy_hat) + frobenius_bound) ** 2
        energy_band_violations.append(max(lower - energy, energy - upper, 0.0))
        errors.append(error)
        bounds.append(frobenius_bound)

    error_rate = np.polyfit(np.log2(amplitudes), np.log2(errors), 1)[0]
    bound_rate = np.polyfit(np.log2(amplitudes), np.log2(bounds), 1)[0]
    assert max(stationary_identity_errors) < 2e-15
    assert max(energy_band_violations) < 1e-15
    assert 0.98 < error_rate < 1.02
    assert 0.97 < bound_rate < 1.03
    return {
        "energy": energy,
        "error_rate": error_rate,
        "bound_rate": bound_rate,
        "largest_error": errors[0],
        "largest_bound": bounds[0],
        "maximum_stationary_identity_error": max(stationary_identity_errors),
        "maximum_energy_band_violation": max(energy_band_violations),
    }


def random_probe_detection_check(sample_count=400_000):
    """Check the exact Gaussian two-feature identification law.

    In an L2(pi)-orthonormal basis of the centered space, let R=(-Q)^(-1)
    and J=(R-R.T)/2.  For independent standard Gaussian coordinates x,y,
    the antisymmetric Green--Kubo entry is X=x.T J y.  Consequently

        E X^2 = ||J||_F^2,
        E X^4 = 3 ||J||_F^4 + 6 ||J.T J||_F^2,
        E exp(i t X) = det(I + t^2 J.T J)^(-1/2).

    The nonzero singular values of the real skew matrix J occur in equal
    pairs, so E X^4 <= 6 (E X^2)^2.  Paley--Zygmund then gives

        P(|X| >= theta ||J||_F) >= (1-theta^2)^2 / 6.

    If the chain is irreversible, J is nonzero and X=0 has probability zero.
    """
    q = np.array(
        [
            [-2.5, 2.0, 0.4, 0.1],
            [0.2, -2.1, 1.6, 0.3],
            [0.7, 0.1, -2.6, 1.8],
            [1.1, 0.5, 0.2, -1.8],
        ]
    )
    pi = stationary(q)
    weight = np.diag(pi)
    euclidean_basis = null_space(np.sqrt(pi)[None, :])
    basis = np.diag(1.0 / np.sqrt(pi)) @ euclidean_basis
    generator = basis.T @ weight @ q @ basis
    resolvent = np.linalg.inv(-generator)
    skew_resolvent = 0.5 * (resolvent - resolvent.T)
    exact_second_moment = np.sum(skew_resolvent**2)
    assert exact_second_moment > 1e-4

    # Verify that the coordinate bilinear form is exactly the antisymmetric
    # entry obtained from the original-state Green--Kubo calculation.
    probe_x = np.array([0.4, -1.1, 0.7])
    probe_y = np.array([-0.3, 0.8, 1.2])
    features = np.array([basis @ probe_x, basis @ probe_y])
    block = gk(q, features)
    block_signal = 0.5 * (block[0, 1] - block[1, 0])
    coordinate_signal = probe_x @ skew_resolvent @ probe_y
    coordinate_error = abs(block_signal - coordinate_signal)

    rng = np.random.default_rng(20261002)
    x = rng.normal(size=(sample_count, len(probe_x)))
    y = rng.normal(size=(sample_count, len(probe_x)))
    signals = np.einsum("bi,ij,bj->b", x, skew_resolvent, y)
    empirical_second_moment = np.mean(signals**2)
    relative_second_moment_error = abs(
        empirical_second_moment / exact_second_moment - 1.0
    )

    singular_squares = np.linalg.svd(skew_resolvent, compute_uv=False) ** 2
    fourth_spectral_sum = np.sum(singular_squares**2)
    spectral_concentration = fourth_spectral_sum / exact_second_moment**2
    exact_fourth_moment = (
        3.0 * exact_second_moment**2 + 6.0 * fourth_spectral_sum
    )
    empirical_fourth_moment = np.mean(signals**4)
    relative_fourth_moment_error = abs(
        empirical_fourth_moment / exact_fourth_moment - 1.0
    )
    relative_energy_variance = (
        exact_fourth_moment / exact_second_moment**2 - 1.0
    )

    # The finite relative-variance bound becomes an exponential-confidence
    # energy estimate by median amplification.  A block mean of b squared
    # signals is outside relative error eps with probability at most
    # v/(b eps^2).  Taking b >= 4v/eps^2 makes that probability at most 1/4;
    # the median of k independent block means then fails with probability at
    # most exp(-k/8).  Use the worst-case Gaussian constant v=5 here.
    mom_epsilon = 0.5
    mom_delta = 0.05
    mom_blocks = math.ceil(8.0 * math.log(1.0 / mom_delta))
    if mom_blocks % 2 == 0:
        mom_blocks += 1
    gaussian_mom_block_size = math.ceil(
        4.0 * 5.0 / mom_epsilon**2
    )
    gaussian_mom_sample_size = (
        mom_blocks * gaussian_mom_block_size
    )
    mom_failure_bound = math.exp(-mom_blocks / 8.0)
    gaussian_mom_trials = sample_count // gaussian_mom_sample_size
    gaussian_mom_values = signals[
        :gaussian_mom_trials * gaussian_mom_sample_size
    ] ** 2
    gaussian_mom_estimates = np.median(
        gaussian_mom_values.reshape(
            gaussian_mom_trials,
            mom_blocks,
            gaussian_mom_block_size,
        ).mean(axis=2),
        axis=1,
    )
    gaussian_mom_relative_errors = np.abs(
        gaussian_mom_estimates / exact_second_moment - 1.0
    )
    gaussian_mom_failure_frequency = np.mean(
        gaussian_mom_relative_errors > mom_epsilon
    )

    threshold_fraction = 0.5
    threshold = threshold_fraction * math.sqrt(exact_second_moment)
    empirical_detection_probability = np.mean(abs(signals) >= threshold)
    paley_zygmund_bound = (1.0 - threshold_fraction**2) ** 2 / (
        3.0 + 6.0 * spectral_concentration
    )
    universal_detection_bound = (1.0 - threshold_fraction**2) ** 2 / 6.0
    probes_for_one_percent = math.ceil(
        math.log(0.01) / math.log1p(-universal_detection_bound)
    )
    one_percent_miss_bound = (
        1.0 - universal_detection_bound
    ) ** probes_for_one_percent

    frequency = 0.75 / np.linalg.norm(skew_resolvent, 2)
    empirical_characteristic = np.mean(np.exp(1j * frequency * signals))
    exact_characteristic = np.linalg.det(
        np.eye(len(probe_x))
        + frequency**2 * skew_resolvent.T @ skew_resolvent
    ) ** (-0.5)
    characteristic_error = abs(
        empirical_characteristic - exact_characteristic
    )

    # Bounded sign probes have the same second moment and no larger fourth
    # moment.  If q_i=sum_j J_ij^2, direct Rademacher expansion gives
    #
    # E (x'Jy)^4 = 3 S2^2 + 6 S4 - 12 sum_i q_i^2
    #                + 4 sum_ij J_ij^4.
    #
    # The last two terms have nonpositive sum because sum_ij J_ij^4 is at
    # most sum_i q_i^2.  More sharply, sum_i q_i^2 >= S2^2/d, so the fourth
    # moment is at most (6-8/d)S2^2.  The dimension-free constant six is
    # asymptotically sharp for the dense rank-two harmonic family below.
    dimension = len(probe_x)
    assert dimension >= 2
    integers = np.arange(2**dimension, dtype=np.uint64)
    bits = ((integers[:, None] >> np.arange(dimension, dtype=np.uint64))
            & 1)
    sign_vectors = 2.0 * bits.astype(float) - 1.0
    rademacher_signals = np.einsum(
        "ai,ij,bj->ab", sign_vectors, skew_resolvent, sign_vectors
    ).ravel()
    row_energies = np.sum(skew_resolvent**2, axis=1)
    exact_rademacher_fourth_moment = (
        3.0 * exact_second_moment**2
        + 6.0 * fourth_spectral_sum
        - 12.0 * np.sum(row_energies**2)
        + 4.0 * np.sum(skew_resolvent**4)
    )
    enumerated_rademacher_second_moment = np.mean(rademacher_signals**2)
    enumerated_rademacher_fourth_moment = np.mean(rademacher_signals**4)
    rademacher_relative_energy_variance = (
        exact_rademacher_fourth_moment / exact_second_moment**2 - 1.0
    )
    dimension_moment_constant = 6.0 - 8.0 / dimension
    rademacher_variance_constant = 5.0 - 8.0 / dimension
    rademacher_mom_block_size = math.ceil(
        4.0 * rademacher_variance_constant / mom_epsilon**2
    )
    rademacher_mom_sample_size = (
        mom_blocks * rademacher_mom_block_size
    )
    mom_rng = np.random.default_rng(20261005)
    rademacher_mom_trials = 400
    rademacher_mom_values = mom_rng.choice(
        rademacher_signals,
        size=rademacher_mom_trials * rademacher_mom_sample_size,
    ).reshape(
        rademacher_mom_trials,
        mom_blocks,
        rademacher_mom_block_size,
    ) ** 2
    rademacher_mom_estimates = np.median(
        rademacher_mom_values.mean(axis=2), axis=1
    )
    rademacher_mom_relative_errors = np.abs(
        rademacher_mom_estimates / exact_second_moment - 1.0
    )
    rademacher_mom_failure_frequency = np.mean(
        rademacher_mom_relative_errors > mom_epsilon
    )
    dimension_detection_bound = (
        (1.0 - threshold_fraction**2) ** 2 / dimension_moment_constant
    )
    dimension_probes_for_one_percent = math.ceil(
        math.log(0.01) / math.log1p(-dimension_detection_bound)
    )
    dimension_one_percent_miss_bound = (
        1.0 - dimension_detection_bound
    ) ** dimension_probes_for_one_percent
    rademacher_detection_probability = np.mean(
        abs(rademacher_signals) >= threshold
    )
    rademacher_zero_probability = np.mean(
        abs(rademacher_signals) < 1e-14
    )

    canonical_skew = np.array([[0.0, 1.0], [-1.0, 0.0]])
    canonical_signs = np.array([
        [-1.0, -1.0], [-1.0, 1.0],
        [1.0, -1.0], [1.0, 1.0],
    ])
    canonical_signals = np.einsum(
        "ai,ij,bj->ab", canonical_signs, canonical_skew, canonical_signs
    ).ravel()
    canonical_zero_probability = np.mean(canonical_signals == 0.0)

    # Dense rank-two harmonic skew matrices prove that the dimension-free
    # constant six cannot be lowered.  For d >= 5, let a_j and b_j be the
    # normalized cosine and sine vectors on the d-cycle and put
    # J=(ab'-ba')/sqrt(2).  Then S2=1, S4=1/2, q_i=1/d, and
    # sum_ij J_ij^4=3/(2d^2), so the exact ratio is
    # 6-12/d+6/d^2 -> 6.
    harmonic_dimensions = (5, 8, 16, 32, 64, 128)
    maximum_harmonic_formula_error = 0.0
    harmonic_moment_ratios = []
    for harmonic_dimension in harmonic_dimensions:
        angles = (
            2.0 * math.pi * np.arange(harmonic_dimension)
            / harmonic_dimension
        )
        harmonic_a = math.sqrt(2.0 / harmonic_dimension) * np.cos(angles)
        harmonic_b = math.sqrt(2.0 / harmonic_dimension) * np.sin(angles)
        harmonic_skew = (
            np.outer(harmonic_a, harmonic_b)
            - np.outer(harmonic_b, harmonic_a)
        ) / math.sqrt(2.0)
        harmonic_second = np.sum(harmonic_skew**2)
        harmonic_singular_squares = (
            np.linalg.svd(harmonic_skew, compute_uv=False) ** 2
        )
        harmonic_spectral_fourth = np.sum(harmonic_singular_squares**2)
        harmonic_rows = np.sum(harmonic_skew**2, axis=1)
        harmonic_fourth = (
            3.0 * harmonic_second**2
            + 6.0 * harmonic_spectral_fourth
            - 12.0 * np.sum(harmonic_rows**2)
            + 4.0 * np.sum(harmonic_skew**4)
        )
        harmonic_ratio = harmonic_fourth / harmonic_second**2
        predicted_harmonic_ratio = (
            6.0 - 12.0 / harmonic_dimension
            + 6.0 / harmonic_dimension**2
        )
        maximum_harmonic_formula_error = max(
            maximum_harmonic_formula_error,
            abs(harmonic_ratio - predicted_harmonic_ratio),
        )
        harmonic_moment_ratios.append(harmonic_ratio)

    # Independently enumerate sign pairs for random skew matrices in
    # dimensions two through six.  Normalize errors by S2^2 so the test is
    # insensitive to the random matrix scale.
    moment_rng = np.random.default_rng(1989)
    randomized_moment_cases = 0
    maximum_rademacher_moment_error = 0.0
    maximum_rademacher_moment_ratio = 0.0
    for random_dimension in range(2, 7):
        random_integers = np.arange(2**random_dimension, dtype=np.uint64)
        random_bits = (
            (random_integers[:, None]
             >> np.arange(random_dimension, dtype=np.uint64)) & 1
        )
        random_signs = 2.0 * random_bits.astype(float) - 1.0
        for _ in range(20):
            raw = moment_rng.normal(
                size=(random_dimension, random_dimension))
            random_skew = 0.5 * (raw - raw.T)
            random_second = np.sum(random_skew**2)
            random_spectral_fourth = np.sum(
                (random_skew @ random_skew.T)**2)
            random_rows = np.sum(random_skew**2, axis=1)
            random_fourth_formula = (
                3.0 * random_second**2
                + 6.0 * random_spectral_fourth
                - 12.0 * np.sum(random_rows**2)
                + 4.0 * np.sum(random_skew**4)
            )
            random_signals = np.einsum(
                "ai,ij,bj->ab",
                random_signs, random_skew, random_signs
            ).ravel()
            random_fourth_enumerated = np.mean(random_signals**4)
            maximum_rademacher_moment_error = max(
                maximum_rademacher_moment_error,
                abs(random_fourth_enumerated - random_fourth_formula)
                / random_second**2,
            )
            maximum_rademacher_moment_ratio = max(
                maximum_rademacher_moment_ratio,
                random_fourth_formula / random_second**2,
            )
            assert random_fourth_formula <= (
                6.0 * random_second**2 * (1.0 + 2e-14)
            )
            randomized_moment_cases += 1

    assert coordinate_error < 2e-14
    assert relative_second_moment_error < 8e-3
    assert spectral_concentration <= 0.5 + 2e-14
    assert exact_fourth_moment <= 6.0 * exact_second_moment**2 * (1.0 + 2e-14)
    assert relative_energy_variance <= 5.0 + 2e-14
    assert mom_blocks == 25
    assert gaussian_mom_block_size == 80
    assert gaussian_mom_sample_size == 2000
    assert mom_failure_bound < mom_delta
    assert gaussian_mom_trials == 200
    assert gaussian_mom_failure_frequency < 0.02
    assert relative_fourth_moment_error < 3e-2
    assert empirical_detection_probability >= paley_zygmund_bound
    assert probes_for_one_percent == 47
    assert one_percent_miss_bound < 0.01
    assert characteristic_error < 3e-3
    assert np.count_nonzero(signals == 0.0) == 0
    assert abs(
        enumerated_rademacher_second_moment / exact_second_moment - 1.0
    ) < 2e-14
    assert abs(
        enumerated_rademacher_fourth_moment
        / exact_rademacher_fourth_moment - 1.0
    ) < 2e-14
    assert exact_rademacher_fourth_moment <= exact_fourth_moment + 2e-14
    assert exact_rademacher_fourth_moment <= (
        6.0 * exact_second_moment**2 * (1.0 + 2e-14)
    )
    assert exact_rademacher_fourth_moment <= (
        dimension_moment_constant
        * exact_second_moment**2 * (1.0 + 2e-14)
    )
    assert rademacher_relative_energy_variance <= 5.0 + 2e-14
    assert rademacher_detection_probability >= universal_detection_bound
    assert rademacher_detection_probability >= dimension_detection_bound
    assert dimension_probes_for_one_percent == 25
    assert dimension_one_percent_miss_bound < 0.01
    assert rademacher_variance_constant == 7.0 / 3.0
    assert rademacher_mom_block_size == 38
    assert rademacher_mom_sample_size == 950
    assert rademacher_mom_failure_frequency < 0.02
    assert rademacher_zero_probability > 0.0
    assert canonical_zero_probability == 0.5
    assert maximum_harmonic_formula_error < 2e-13
    assert harmonic_moment_ratios[-1] > 5.9
    assert randomized_moment_cases == 100
    assert maximum_rademacher_moment_error < 2e-14
    return {
        "skew_hilbert_schmidt": math.sqrt(exact_second_moment),
        "exact_second_moment": exact_second_moment,
        "empirical_second_moment": empirical_second_moment,
        "relative_second_moment_error": relative_second_moment_error,
        "spectral_concentration": spectral_concentration,
        "exact_fourth_moment": exact_fourth_moment,
        "empirical_fourth_moment": empirical_fourth_moment,
        "relative_fourth_moment_error": relative_fourth_moment_error,
        "relative_energy_variance": relative_energy_variance,
        "mom_epsilon": mom_epsilon,
        "mom_delta": mom_delta,
        "mom_blocks": mom_blocks,
        "mom_failure_bound": mom_failure_bound,
        "gaussian_mom_block_size": gaussian_mom_block_size,
        "gaussian_mom_sample_size": gaussian_mom_sample_size,
        "gaussian_mom_trials": gaussian_mom_trials,
        "gaussian_mom_failure_frequency": (
            gaussian_mom_failure_frequency
        ),
        "gaussian_mom_maximum_relative_error": np.max(
            gaussian_mom_relative_errors
        ),
        "threshold_fraction": threshold_fraction,
        "empirical_detection_probability": empirical_detection_probability,
        "paley_zygmund_bound": paley_zygmund_bound,
        "universal_detection_bound": universal_detection_bound,
        "probes_for_one_percent": probes_for_one_percent,
        "one_percent_miss_bound": one_percent_miss_bound,
        "frequency": frequency,
        "exact_characteristic": exact_characteristic,
        "empirical_characteristic": empirical_characteristic,
        "characteristic_error": characteristic_error,
        "coordinate_error": coordinate_error,
        "exact_rademacher_fourth_moment": exact_rademacher_fourth_moment,
        "enumerated_rademacher_fourth_moment": (
            enumerated_rademacher_fourth_moment
        ),
        "rademacher_relative_energy_variance": (
            rademacher_relative_energy_variance
        ),
        "dimension_moment_constant": dimension_moment_constant,
        "rademacher_variance_constant": rademacher_variance_constant,
        "rademacher_mom_block_size": rademacher_mom_block_size,
        "rademacher_mom_sample_size": rademacher_mom_sample_size,
        "rademacher_mom_trials": rademacher_mom_trials,
        "rademacher_mom_failure_frequency": (
            rademacher_mom_failure_frequency
        ),
        "rademacher_mom_maximum_relative_error": np.max(
            rademacher_mom_relative_errors
        ),
        "dimension_detection_bound": dimension_detection_bound,
        "dimension_probes_for_one_percent": (
            dimension_probes_for_one_percent
        ),
        "dimension_one_percent_miss_bound": (
            dimension_one_percent_miss_bound
        ),
        "rademacher_detection_probability": (
            rademacher_detection_probability
        ),
        "rademacher_zero_probability": rademacher_zero_probability,
        "canonical_zero_probability": canonical_zero_probability,
        "maximum_harmonic_formula_error": maximum_harmonic_formula_error,
        "largest_harmonic_dimension": harmonic_dimensions[-1],
        "largest_harmonic_moment_ratio": harmonic_moment_ratios[-1],
        "randomized_moment_cases": randomized_moment_cases,
        "maximum_rademacher_moment_error": (
            maximum_rademacher_moment_error
        ),
        "maximum_rademacher_moment_ratio": (
            maximum_rademacher_moment_ratio
        ),
    }


def normalized_resolvent_check():
    """Check the sharp energy-normalized resolvent factorization.

    On the centered L2(pi) space write L=S+A, B=-S, and
    C=B^(-1/2) A B^(-1/2).  Then

        B^(1/2)(-L)^(-1)B^(1/2)=(I-C)^(-1),

    whose symmetric and skew parts are (I-C^2)^(-1) and
    C(I-C^2)^(-1), respectively.
    """
    q = np.array(
        [
            [-2.5, 2.0, 0.4, 0.1],
            [0.2, -2.1, 1.6, 0.3],
            [0.7, 0.1, -2.6, 1.8],
            [1.1, 0.5, 0.2, -1.8],
        ]
    )
    pi = stationary(q)
    weight = np.diag(pi)

    # If y=sqrt(pi) f, centering is orthogonality to sqrt(pi).
    euclidean_basis = null_space(np.sqrt(pi)[None, :])
    basis = np.diag(1.0 / np.sqrt(pi)) @ euclidean_basis
    assert np.max(abs(basis.T @ weight @ basis - np.eye(len(pi) - 1))) < 2e-14
    assert np.max(abs(pi @ basis)) < 2e-14

    generator = basis.T @ weight @ q @ basis
    symmetric_generator = 0.5 * (generator + generator.T)
    skew_generator = 0.5 * (generator - generator.T)
    energy = -symmetric_generator
    eigenvalues, eigenvectors = np.linalg.eigh(energy)
    assert np.min(eigenvalues) > 0.0
    energy_half = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    energy_inverse_half = (
        eigenvectors * (1.0 / np.sqrt(eigenvalues))
    ) @ eigenvectors.T

    normalized_skew = energy_inverse_half @ skew_generator @ energy_inverse_half
    resolvent = np.linalg.inv(-generator)
    normalized_resolvent = energy_half @ resolvent @ energy_half
    identity = np.eye(len(eigenvalues))
    factorized = np.linalg.inv(identity - normalized_skew)
    symmetric_factor = np.linalg.inv(
        identity - normalized_skew @ normalized_skew
    )
    skew_factor = normalized_skew @ symmetric_factor

    factorization_error = np.max(abs(normalized_resolvent - factorized))
    symmetric_error = np.max(abs(
        0.5 * (normalized_resolvent + normalized_resolvent.T)
        - symmetric_factor
    ))
    skew_error = np.max(abs(
        0.5 * (normalized_resolvent - normalized_resolvent.T)
        - skew_factor
    ))

    eta = np.linalg.norm(normalized_skew, 2)
    symmetric_eigenvalues = np.linalg.eigvalsh(symmetric_factor)
    lower_bound = 1.0 / (1.0 + eta ** 2)
    skew_bound = eta / (1.0 + eta ** 2) if eta <= 1.0 else 0.5
    skew_norm = np.linalg.norm(skew_factor, 2)

    # Normalize instead by the actual symmetric Green--Kubo form.  Because
    # symmetric_factor is a function of normalized_skew, all three matrices
    # commute and the intrinsic skew operator is exactly C.  The full
    # resolvent becomes I+C in this metric.
    sym_values, sym_vectors = np.linalg.eigh(symmetric_factor)
    symmetric_inverse_half = (
        sym_vectors * (1.0 / np.sqrt(sym_values))
    ) @ sym_vectors.T
    intrinsic_skew = (
        symmetric_inverse_half @ skew_factor @ symmetric_inverse_half
    )
    intrinsic_full = (
        symmetric_inverse_half @ factorized @ symmetric_inverse_half
    )
    intrinsic_skew_error = np.max(abs(intrinsic_skew - normalized_skew))
    intrinsic_full_error = np.max(
        abs(intrinsic_full - (identity + normalized_skew))
    )
    intrinsic_skew_norm = np.linalg.norm(intrinsic_skew, 2)
    intrinsic_full_norm = np.linalg.norm(intrinsic_full, 2)
    intrinsic_full_bound = math.sqrt(1.0 + eta ** 2)

    assert factorization_error < 3e-14
    assert symmetric_error < 3e-14
    assert skew_error < 3e-14
    assert np.min(symmetric_eigenvalues) >= lower_bound - 3e-14
    assert np.max(symmetric_eigenvalues) <= 1.0 + 3e-14
    assert skew_norm <= skew_bound + 3e-14
    assert intrinsic_skew_error < 3e-14
    assert intrinsic_full_error < 3e-14
    assert abs(intrinsic_skew_norm - eta) < 3e-14
    assert abs(intrinsic_full_norm - intrinsic_full_bound) < 3e-14
    return {
        "factorization_error": factorization_error,
        "symmetric_error": symmetric_error,
        "skew_error": skew_error,
        "eta": eta,
        "lower_bound": lower_bound,
        "symmetric_min": np.min(symmetric_eigenvalues),
        "symmetric_max": np.max(symmetric_eigenvalues),
        "skew_norm": skew_norm,
        "skew_bound": skew_bound,
        "intrinsic_skew_error": intrinsic_skew_error,
        "intrinsic_full_error": intrinsic_full_error,
        "intrinsic_skew_norm": intrinsic_skew_norm,
        "intrinsic_full_norm": intrinsic_full_norm,
        "intrinsic_full_bound": intrinsic_full_bound,
    }


def first_order(lbar, correction, maturity, payoff):
    """Duhamel correction exp(TL)f+int exp((T-s)L)D exp(sL)f ds."""
    n = len(payoff)
    block = np.zeros((2 * n, 2 * n))
    block[:n, :n] = lbar
    block[n:, n:] = lbar
    block[:n, n:] = correction
    semigroup = expm(maturity * block)
    return semigroup[:n, :n] @ payoff + semigroup[:n, n:] @ payoff


def effective_check(drift):
    """Finite-rate errors using the discrete circle as an independent driver."""
    q0, phis = circle_chain(48, drift=drift)
    pi = stationary(q0)
    k0 = gk(q0, phis)
    a_cos = np.array([[-0.20, 0.35], [-0.10, 0.05]])
    a_sin = np.array([[0.15, -0.20], [0.40, -0.25]])
    operators = [a_cos, a_sin]
    lbar = np.array([[-0.7, 0.2], [0.1, -0.5]])
    payoff = np.array([1.0, -0.3])
    maturity = 0.8
    errors = {"averaged": [], "symmetric": [], "full": []}
    values = []
    for speed in SCALES:
        q = speed * q0
        generator = full_generator(q, lbar, operators, phis)
        truth = np.kron(pi, np.eye(2)) @ (
            expm(maturity * generator) @ np.kron(np.ones(len(pi)), payoff)
        )
        k = k0 / speed
        full = effective_generator(np.zeros_like(lbar), operators, k)
        sym = effective_generator(np.zeros_like(lbar), operators, k, "sym")
        approx_full = first_order(lbar, full, maturity, payoff)
        approx_sym = first_order(lbar, sym, maturity, payoff)
        averaged = expm(maturity * lbar) @ payoff
        errors["averaged"].append(np.linalg.norm(truth - averaged))
        errors["symmetric"].append(np.linalg.norm(truth - approx_sym))
        errors["full"].append(np.linalg.norm(truth - approx_full))
        values.append(truth)
    return k0, errors, values


def rate(errors):
    return math.log(errors[-2] / errors[-1], 2)


def main():
    print("1. the stationary Markov reversal transposes the Green--Kubo form")
    general = general_reversal_check()
    print("   invariant law: " + " ".join(f"{x:.6f}" for x in general["pi"]))
    print(
        f"   semigroup adjoint error {general['semigroup_error']:.2e}; "
        f"Green--Kubo transpose error {general['transpose_error']:.2e}"
    )
    print(
        f"   irreversible full-form asymmetry {general['full_asymmetry']:.3e}; "
        f"one-feature asymmetry {general['scalar_asymmetry']:.1e}; "
        f"reversible full-form asymmetry {general['reversible_asymmetry']:.2e}"
    )

    print("2. estimated generators obey a deterministic skew-resolvent bound")
    perturbation = generator_perturbation_check()
    print(
        f"   skew error/bound at largest perturbation "
        f"{perturbation['largest_error']:.9f}/"
        f"{perturbation['largest_bound']:.9f}; convergence rates "
        f"{perturbation['error_rate']:.6f}/"
        f"{perturbation['bound_rate']:.6f}"
    )
    print(
        f"   stationary identity/energy-band violations "
        f"{perturbation['maximum_stationary_identity_error']:.2e}/"
        f"{perturbation['maximum_energy_band_violation']:.2e}"
    )

    print("3. two Gaussian features detect finite-state irreversibility almost surely")
    probes = random_probe_detection_check()
    print(
        f"   skew Hilbert--Schmidt norm {probes['skew_hilbert_schmidt']:.9f}; "
        f"second moment exact/simulated {probes['exact_second_moment']:.9f}/"
        f"{probes['empirical_second_moment']:.9f}"
    )
    print(
        f"   characteristic function exact/simulated "
        f"{probes['exact_characteristic']:.9f}/"
        f"{probes['empirical_characteristic'].real:.9f}; errors "
        f"{probes['relative_second_moment_error']:.2e}/"
        f"{probes['characteristic_error']:.2e}"
    )
    print(
        f"   fourth moment exact/simulated "
        f"{probes['exact_fourth_moment']:.9f}/"
        f"{probes['empirical_fourth_moment']:.9f}; relative energy-variance "
        f"{probes['relative_energy_variance']:.9f}"
    )
    print(
        f"   half-RMS detection probability simulated/bounded "
        f"{probes['empirical_detection_probability']:.9f}/"
        f"{probes['universal_detection_bound']:.9f}; "
        f"{probes['probes_for_one_percent']} probes give miss bound "
        f"{probes['one_percent_miss_bound']:.6f}"
    )
    print(
        f"   Gaussian median-of-means blocks/size/failure bound "
        f"{probes['mom_blocks']}/{probes['gaussian_mom_block_size']}/"
        f"{probes['mom_failure_bound']:.6f}; empirical failures/max error "
        f"{probes['gaussian_mom_failure_frequency']:.6f}/"
        f"{probes['gaussian_mom_maximum_relative_error']:.6f}"
    )
    print(
        f"   Rademacher fourth moment exact/enumerated "
        f"{probes['exact_rademacher_fourth_moment']:.9f}/"
        f"{probes['enumerated_rademacher_fourth_moment']:.9f}; "
        f"relative energy-variance "
        f"{probes['rademacher_relative_energy_variance']:.9f}"
    )
    print(
        f"   Rademacher half-RMS detection/zero probabilities "
        f"{probes['rademacher_detection_probability']:.9f}/"
        f"{probes['rademacher_zero_probability']:.9f}; canonical zero "
        f"probability {probes['canonical_zero_probability']:.9f}"
    )
    print(
        f"   dimension-aware fourth-moment/detection bounds "
        f"{probes['dimension_moment_constant']:.9f}/"
        f"{probes['dimension_detection_bound']:.9f}; "
        f"{probes['dimension_probes_for_one_percent']} probes give miss "
        f"bound {probes['dimension_one_percent_miss_bound']:.6f}"
    )
    print(
        f"   Rademacher median-of-means blocks/size/failure bound "
        f"{probes['mom_blocks']}/{probes['rademacher_mom_block_size']}/"
        f"{probes['mom_failure_bound']:.6f}; empirical failures/max error "
        f"{probes['rademacher_mom_failure_frequency']:.6f}/"
        f"{probes['rademacher_mom_maximum_relative_error']:.6f}"
    )
    print(
        f"   harmonic sharpness dimension/ratio/error "
        f"{probes['largest_harmonic_dimension']}/"
        f"{probes['largest_harmonic_moment_ratio']:.9f}/"
        f"{probes['maximum_harmonic_formula_error']:.2e}"
    )
    print(
        f"   random skew Rademacher moment cases/error/max ratio "
        f"{probes['randomized_moment_cases']}/"
        f"{probes['maximum_rademacher_moment_error']:.2e}/"
        f"{probes['maximum_rademacher_moment_ratio']:.9f}"
    )

    print("4. the energy-normalized resolvent factorization gives sharp bounds")
    normalized = normalized_resolvent_check()
    print(
        f"   factorization/symmetric/skew errors "
        f"{normalized['factorization_error']:.2e}, "
        f"{normalized['symmetric_error']:.2e}, "
        f"{normalized['skew_error']:.2e}"
    )
    print(
        f"   eta {normalized['eta']:.9f}; symmetric spectrum "
        f"[{normalized['symmetric_min']:.9f}, "
        f"{normalized['symmetric_max']:.9f}] versus lower bound "
        f"{normalized['lower_bound']:.9f}; skew norm/bound "
        f"{normalized['skew_norm']:.9f}/{normalized['skew_bound']:.9f}"
    )
    print(
        f"   intrinsic skew norm/eta "
        f"{normalized['intrinsic_skew_norm']:.9f}/{normalized['eta']:.9f}; "
        f"full norm/bound {normalized['intrinsic_full_norm']:.9f}/"
        f"{normalized['intrinsic_full_bound']:.9f}; identity errors "
        f"{normalized['intrinsic_skew_error']:.2e}/"
        f"{normalized['intrinsic_full_error']:.2e}"
    )

    print("5. exact Fourier blocks against direct correlation quadrature")
    for mode in range(1, 6):
        exact = exact_block(mode)
        numerical = quadrature_block(mode)
        error = np.max(np.abs(exact - numerical))
        print(f"   mode {mode}: max error {error:.2e}, anti entry {exact[0, 1]:+.8f}")
        assert error < 2e-11

    print("6. periodic CTMC discretization converges to the diffusion block")
    grid_errors = []
    for size in (32, 64, 128):
        q, phis = circle_chain(size)
        k = gk(q, phis)
        error = np.max(np.abs(k - exact_block(1)))
        grid_errors.append(error)
        print(f"   N={size:3d}: max error {error:.3e}; K12={k[0, 1]:+.8f}")
    spatial_rate = rate(grid_errors)
    assert 1.9 < spatial_rate < 2.1

    print("7. reversing current preserves the symmetric block and flips the antisymmetric block")
    forward = exact_block(1, C)
    reverse = exact_block(1, -C)
    assert np.max(abs(0.5 * (forward + forward.T) - 0.5 * (reverse + reverse.T))) < 1e-15
    assert np.max(abs(0.5 * (forward - forward.T) + 0.5 * (reverse - reverse.T))) < 1e-15
    print(f"   c=+{C:g}: K12={forward[0, 1]:+.6f}; c=-{C:g}: K12={reverse[0, 1]:+.6f}")

    print("8. the full commutator rule has a second-order finite-rate residual")
    k_forward, errors_forward, values_forward = effective_check(C)
    k_reverse, errors_reverse, values_reverse = effective_check(-C)
    for name in errors_forward:
        r = rate(errors_forward[name])
        print(f"   {name:9s}: " + " ".join(f"{x:.3e}" for x in errors_forward[name]) + f"  rate {r:.3f}")
    assert 1.8 < rate(errors_forward["full"]) < 2.2
    assert 0.8 < rate(errors_forward["averaged"]) < 1.2
    assert 0.8 < rate(errors_forward["symmetric"]) < 1.2
    assert np.max(abs(0.5 * (k_forward + k_forward.T) - 0.5 * (k_reverse + k_reverse.T))) < 2e-12
    assert np.max(abs(0.5 * (k_forward - k_forward.T) + 0.5 * (k_reverse - k_reverse.T))) < 2e-12

    direction_errors = []
    for speed, value_f, value_r in zip(SCALES, values_forward, values_reverse):
        q0, phis = circle_chain(48, drift=C)
        k = gk(q0, phis) / speed
        a_cos = np.array([[-0.20, 0.35], [-0.10, 0.05]])
        a_sin = np.array([[0.15, -0.20], [0.40, -0.25]])
        anti = 0.5 * (k - k.T)
        directional = effective_generator(np.zeros((2, 2)), [a_cos, a_sin], 2 * anti)
        lbar = np.array([[-0.7, 0.2], [0.1, -0.5]])
        payoff = np.array([1.0, -0.3])
        predicted = first_order(lbar, directional, 0.8, payoff) - expm(0.8 * lbar) @ payoff
        direction_errors.append(np.linalg.norm((value_f - value_r) - predicted))
    direction_rate = rate(direction_errors)
    print("   forward-minus-reverse residual: " + " ".join(f"{x:.3e}" for x in direction_errors) + f"  rate {direction_rate:.3f}")
    assert 1.8 < direction_rate < 2.2

    print("9. a nonconstant periodic diffusion obeys the exact current-reversal theorem")
    blocks = []
    for size in (32, 64, 128, 256):
        q_forward, features = variable_circle_chain(size)
        q_reverse, _ = variable_circle_chain(size, current=-0.08)
        pi = stationary(q_forward)
        expected_density = 1.0 + 0.35 * np.cos(2 * math.pi * np.arange(size) / size)
        expected_density /= expected_density.sum()
        assert np.max(abs(pi - expected_density)) < 2e-12
        k_forward = gk(q_forward, features)
        k_reverse = gk(q_reverse, features)
        transpose_error = np.max(abs(k_reverse - k_forward.T))
        assert transpose_error < 2e-11
        blocks.append(k_forward)
        print(
            f"   N={size:3d}: K12-K21={k_forward[0, 1] - k_forward[1, 0]:+.8f}; "
            f"reversal error {transpose_error:.2e}"
        )
    differences = [np.max(abs(blocks[i] - blocks[i + 1])) for i in range(3)]
    variable_rate = rate(differences)
    assert 1.9 < variable_rate < 2.1
    assert abs(blocks[-1][0, 1] - blocks[-1][1, 0]) > 1e-3

    print(
        f"PASS: circle factor, spatial rate {spatial_rate:.3f}, "
        f"variable-coefficient rate {variable_rate:.3f}, "
        f"full-rule rate {rate(errors_forward['full']):.3f}"
    )


if __name__ == "__main__":
    main()
