"""Certificate for an irreversible one-dimensional diffusion on a circle.

For dY=c dt+sqrt(2D)dW modulo 2 pi and the Fourier pair (cos(nY),
sin(nY)), the Green--Kubo matrix is available in closed form.  This file
checks it three ways and verifies the resulting commutator correction against
a finite-rate coupled evolution.  It also checks the general current
decomposition of a scalar periodic diffusion with nonconstant invariant
density and diffusivity.
"""
import math
import os
import sys

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm

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
    print("1. exact Fourier blocks against direct correlation quadrature")
    for mode in range(1, 6):
        exact = exact_block(mode)
        numerical = quadrature_block(mode)
        error = np.max(np.abs(exact - numerical))
        print(f"   mode {mode}: max error {error:.2e}, anti entry {exact[0, 1]:+.8f}")
        assert error < 2e-11

    print("2. periodic CTMC discretization converges to the diffusion block")
    grid_errors = []
    for size in (32, 64, 128):
        q, phis = circle_chain(size)
        k = gk(q, phis)
        error = np.max(np.abs(k - exact_block(1)))
        grid_errors.append(error)
        print(f"   N={size:3d}: max error {error:.3e}; K12={k[0, 1]:+.8f}")
    spatial_rate = rate(grid_errors)
    assert 1.9 < spatial_rate < 2.1

    print("3. reversing current preserves the symmetric block and flips the antisymmetric block")
    forward = exact_block(1, C)
    reverse = exact_block(1, -C)
    assert np.max(abs(0.5 * (forward + forward.T) - 0.5 * (reverse + reverse.T))) < 1e-15
    assert np.max(abs(0.5 * (forward - forward.T) + 0.5 * (reverse - reverse.T))) < 1e-15
    print(f"   c=+{C:g}: K12={forward[0, 1]:+.6f}; c=-{C:g}: K12={reverse[0, 1]:+.6f}")

    print("4. the full commutator rule has a second-order finite-rate residual")
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

    print("5. a nonconstant periodic diffusion obeys the exact current-reversal theorem")
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
