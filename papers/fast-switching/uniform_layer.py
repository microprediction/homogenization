"""Uniform composite approximations for the symmetric two-state fast switch.

Let ``eps = 1 / lambda`` and

    omega' = q(t) (1 - omega**2) - 2 omega / eps,  omega(0) = 0,
    (log m)' = b(t) + q(t) omega(t),                 a_+/- = m (1 +/- omega).

The functions below implement the two composite formulas proved on the
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
    """Uniform O(eps**3) approximation when q(0) = 0 and q is C^2."""
    t = np.asarray(t)
    q = np.asarray(q)
    omega = eps * q / 2 - eps ** 2 * np.asarray(q_prime) / 4 \
        + eps ** 2 * q_prime_0 * np.exp(-2 * t / eps) / 4
    log_m = np.asarray(int_b) + eps / 2 * np.asarray(int_q2) - eps ** 2 * q ** 2 / 8
    return np.exp(log_m) * (1 + sign * omega)
