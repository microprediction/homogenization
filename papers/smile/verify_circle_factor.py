"""Certificate for time reversal of Green--Kubo forms.

The transpose identity is first checked for a nonreversible finite-state
Markov chain with a nonuniform invariant law.  The same certificate records
the important converse distinction: the full centered Green--Kubo form
detects nonreversibility, whereas a selected feature block need not do so.

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
from effective_generator import effective_generator, full_generator, gk, stationary


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

    assert factorization_error < 3e-14
    assert symmetric_error < 3e-14
    assert skew_error < 3e-14
    assert np.min(symmetric_eigenvalues) >= lower_bound - 3e-14
    assert np.max(symmetric_eigenvalues) <= 1.0 + 3e-14
    assert skew_norm <= skew_bound + 3e-14
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

    print("2. the energy-normalized resolvent factorization gives sharp bounds")
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

    print("3. exact Fourier blocks against direct correlation quadrature")
    for mode in range(1, 6):
        exact = exact_block(mode)
        numerical = quadrature_block(mode)
        error = np.max(np.abs(exact - numerical))
        print(f"   mode {mode}: max error {error:.2e}, anti entry {exact[0, 1]:+.8f}")
        assert error < 2e-11

    print("4. periodic CTMC discretization converges to the diffusion block")
    grid_errors = []
    for size in (32, 64, 128):
        q, phis = circle_chain(size)
        k = gk(q, phis)
        error = np.max(np.abs(k - exact_block(1)))
        grid_errors.append(error)
        print(f"   N={size:3d}: max error {error:.3e}; K12={k[0, 1]:+.8f}")
    spatial_rate = rate(grid_errors)
    assert 1.9 < spatial_rate < 2.1

    print("5. reversing current preserves the symmetric block and flips the antisymmetric block")
    forward = exact_block(1, C)
    reverse = exact_block(1, -C)
    assert np.max(abs(0.5 * (forward + forward.T) - 0.5 * (reverse + reverse.T))) < 1e-15
    assert np.max(abs(0.5 * (forward - forward.T) + 0.5 * (reverse - reverse.T))) < 1e-15
    print(f"   c=+{C:g}: K12={forward[0, 1]:+.6f}; c=-{C:g}: K12={reverse[0, 1]:+.6f}")

    print("6. the full commutator rule has a second-order finite-rate residual")
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

    print("7. a nonconstant periodic diffusion obeys the exact current-reversal theorem")
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
