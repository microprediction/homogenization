"""Uniform composite approximations for the symmetric two-state fast switch.

Let ``eps = 1 / lambda`` and

    omega' = q(t) (1 - omega**2) - 2 omega / eps,  omega(0) = 0,
    (log m)' = b(t) + q(t) omega(t),                 a_+/- = m (1 +/- omega).

The functions below implement the composite formulas proved on the
``layers.html`` page.  Their inputs may be scalars or NumPy arrays.
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
