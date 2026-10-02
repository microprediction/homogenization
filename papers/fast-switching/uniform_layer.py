"""Uniform composite approximations for two-state fast switches.

Let ``eps = 1 / lambda`` and

    omega' = q(t) (1 - omega**2) - 2 omega / eps,  omega(0) = 0,
    (log m)' = b(t) + q(t) omega(t),                 a_+/- = m (1 +/- omega).

The original first- and second-order functions treat the symmetric chain.
The asymmetric functions through second order use arbitrary positive
off-diagonal rates.  The accompanying certificate also implements a direct
slow/fast-coordinate Volterra recursion for arbitrary initial vectors,
including zero stationary mean.  All inputs may be scalars or NumPy arrays.
"""
import numpy as np


def first_order(t, eps, int_b, int_q2, q, q0, int_q_layer, sign=+1):
    """Uniform O(eps**2) approximation for q in C^1.

    ``int_q_layer`` is integral_0^t q(s) exp(-2 s / eps) ds.  Retaining it
    makes the mean correction explicit; omitting it would still change the
    answer only at order eps**2.
    """
    t = np.asarray(t)
    layer = np.exp(-2 * t / eps)
    omega = eps / 2 * (np.asarray(q) - q0 * layer)
    log_m = np.asarray(int_b) + eps / 2 * np.asarray(int_q2) - eps * q0 / 2 * np.asarray(int_q_layer)
    return np.exp(log_m) * (1 + sign * omega)


def first_order_asymmetric(
    t, eps, int_g_bar, int_delta2, delta, delta0, int_delta_layer,
    rate12, rate21, state=0,
):
    """Uniform O(eps**2) approximation for unequal two-state rates.

    The fast generator is ``[[-rate12, rate12], [rate21, -rate21]] / eps``.
    Write ``pi`` for its stationary law, ``rho = rate12 + rate21``,
    ``g_bar = pi1*g1 + pi2*g2``, and ``delta = g1 - g2``.
    ``int_delta_layer`` is integral_0^t delta(s) exp(-rho*s/eps) ds.

    ``state`` is zero or one and selects the corresponding component of the
    backward solution started from the all-ones vector.
    """
    if rate12 <= 0 or rate21 <= 0:
        raise ValueError("both switching rates must be positive")
    if state not in (0, 1):
        raise ValueError("state must be zero or one")
    rho = rate12 + rate21
    pi1, pi2 = rate21 / rho, rate12 / rho
    variance_weight = pi1 * pi2
    t = np.asarray(t)
    layer = np.exp(-rho * t / eps)
    ratio = eps * (np.asarray(delta) - delta0 * layer) / rho
    log_m = np.asarray(int_g_bar) \
        + eps * variance_weight * np.asarray(int_delta2) / rho \
        - eps * variance_weight * delta0 * np.asarray(int_delta_layer) / rho
    component_weight = pi2 if state == 0 else -pi1
    return np.exp(log_m) * (1 + component_weight * ratio)


def first_order_asymmetric_initial(
    t, eps, int_g_bar, int_delta, int_delta2, delta, delta0,
    int_delta_layer, rate12, rate21, initial, state=0,
):
    """Uniform O(eps**2) composite for regime-dependent initial data.

    ``initial`` is the two-component value at ``t=0``.  Its weighted mean
    ``m0`` and contrast ``r0`` generate an order-zero layer ``r0*E``.  The
    remaining layer terms are the first Duhamel correction to the nonlinear
    ratio equation.  The returned component matches ``initial[state]``
    exactly at zero.  The uniform error statement assumes real bounded
    forcing and nonnegative initial data not both zero, or more generally an
    exact weighted mean bounded away from zero on the interval of interest.
    """
    if rate12 <= 0 or rate21 <= 0:
        raise ValueError("both switching rates must be positive")
    if state not in (0, 1):
        raise ValueError("state must be zero or one")
    initial = np.asarray(initial)
    if initial.shape != (2,):
        raise ValueError("initial must have exactly two components")
    rho = rate12 + rate21
    pi1, pi2 = rate21 / rho, rate12 / rho
    skew_weight = pi2 - pi1
    variance_weight = pi1 * pi2
    m0 = pi1 * initial[0] + pi2 * initial[1]
    if m0 == 0:
        raise ValueError("the stationary weighted initial mean must be nonzero")
    r0 = (initial[0] - initial[1]) / m0
    t = np.asarray(t)
    layer = np.exp(-rho * t / eps)
    int_delta = np.asarray(int_delta)
    int_delta_layer = np.asarray(int_delta_layer)
    ratio = r0 * layer \
        + eps * (np.asarray(delta) - delta0 * layer) / rho \
        + skew_weight * r0 * int_delta * layer \
        - variance_weight * r0 ** 2 * int_delta_layer * layer
    log_mean_ratio = np.asarray(int_g_bar) \
        + variance_weight * r0 * int_delta_layer \
        + eps * variance_weight * np.asarray(int_delta2) / rho
    mean = m0 * np.exp(log_mean_ratio)
    component_weight = pi2 if state == 0 else -pi1
    return mean * (1 + component_weight * ratio)


def second_order_asymmetric(
    t, eps, int_g_bar, int_delta, int_delta2, int_delta3,
    delta, delta_prime, delta0, delta_prime0, int_delta_layer,
    rate12, rate21, state=0,
):
    """Uniform O(eps**3) approximation for unequal two-state rates.

    The notation is that of ``first_order_asymmetric``.  ``int_delta`` and
    ``int_delta3`` are the integrals of delta and delta**3.  The resonant
    term ``int_delta * exp(-rho*t/eps)`` is absent for a symmetric chain.
    """
    if rate12 <= 0 or rate21 <= 0:
        raise ValueError("both switching rates must be positive")
    if state not in (0, 1):
        raise ValueError("state must be zero or one")
    rho = rate12 + rate21
    pi1, pi2 = rate21 / rho, rate12 / rho
    skew_weight = pi2 - pi1
    variance_weight = pi1 * pi2
    t = np.asarray(t)
    delta = np.asarray(delta)
    layer = np.exp(-rho * t / eps)
    outer2 = (skew_weight * delta ** 2 - np.asarray(delta_prime)) / rho ** 2
    outer20 = (skew_weight * delta0 ** 2 - delta_prime0) / rho ** 2
    ratio = eps * (delta - delta0 * layer) / rho \
        + eps ** 2 * (outer2 - outer20 * layer) \
        - eps * skew_weight * delta0 * np.asarray(int_delta) * layer / rho
    log_m = np.asarray(int_g_bar) \
        + eps * variance_weight * np.asarray(int_delta2) / rho \
        - eps * variance_weight * delta0 * np.asarray(int_delta_layer) / rho \
        + eps ** 2 * variance_weight / rho ** 2 * (
            skew_weight * np.asarray(int_delta3)
            - (delta ** 2 - delta0 ** 2) / 2
        )
    component_weight = pi2 if state == 0 else -pi1
    return np.exp(log_m) * (1 + component_weight * ratio)


def second_order_outer_asymmetric(
    eps, int_g_bar, int_delta2, int_delta3,
    delta, delta_prime, delta0, rate12, rate21, state=0,
):
    """Matched unequal-rate outer formula with O(eps**3) error.

    For ``delta0 != 0`` the error is uniform after
    ``t >= 2*eps*log(1/eps)/rho``.  A fixed positive maturity also suffices.
    The constant ``-eps**2*pi1*pi2*delta0**2/rho**2`` in ``log_m`` is the
    permanent trace of the initial layer; merely deleting the exponential
    terms from ``second_order_asymmetric`` would leave O(eps**2) error.
    """
    if rate12 <= 0 or rate21 <= 0:
        raise ValueError("both switching rates must be positive")
    if state not in (0, 1):
        raise ValueError("state must be zero or one")
    rho = rate12 + rate21
    pi1, pi2 = rate21 / rho, rate12 / rho
    skew_weight = pi2 - pi1
    variance_weight = pi1 * pi2
    delta = np.asarray(delta)
    ratio = eps * delta / rho + eps ** 2 * (
        skew_weight * delta ** 2 - np.asarray(delta_prime)
    ) / rho ** 2
    log_m = np.asarray(int_g_bar) \
        + eps * variance_weight * np.asarray(int_delta2) / rho \
        + eps ** 2 * variance_weight / rho ** 2 * (
            skew_weight * np.asarray(int_delta3)
            - (delta ** 2 + delta0 ** 2) / 2
        )
    component_weight = pi2 if state == 0 else -pi1
    return np.exp(log_m) * (1 + component_weight * ratio)


def second_order_zero_start(t, eps, int_b, int_q2, q, q_prime, q_prime_0, sign=+1):
    """Backward-compatible specialization of ``second_order`` to q(0)=0."""
    zeros = np.zeros_like(np.asarray(t), dtype=np.result_type(q, float))
    return second_order(
        t, eps, int_b, int_q2, q, q_prime, 0.0, q_prime_0, zeros, sign
    )


def second_order(
    t, eps, int_b, int_q2, q, q_prime, q0, q_prime_0, int_q_layer,
    sign=+1,
):
    """Uniform O(eps**3) approximation for arbitrary q(0) and q in C^2.

    ``int_q_layer`` is integral_0^t q(s) exp(-2 s / eps) ds.  The terms
    involving it retain the permanent contribution accumulated while the
    initial mismatch decays.
    """
    t = np.asarray(t)
    q = np.asarray(q)
    layer = np.exp(-2 * t / eps)
    omega = eps * (q - q0 * layer) / 2 \
        - eps ** 2 * (np.asarray(q_prime) - q_prime_0 * layer) / 4
    log_m = np.asarray(int_b) + eps / 2 * np.asarray(int_q2) \
        - eps * q0 * np.asarray(int_q_layer) / 2 \
        - eps ** 2 * (q ** 2 - q0 ** 2) / 8 \
        + eps ** 2 * q_prime_0 * np.asarray(int_q_layer) / 4
    return np.exp(log_m) * (1 + sign * omega)
