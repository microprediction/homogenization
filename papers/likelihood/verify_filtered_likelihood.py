"""Certificate for stationary-start composite versus exact filtered likelihood.

The exact matrix transition density retains the ending regime.  It is used to
propagate the hidden-state filter.  For fixed observation spacing and a fixed
panel, the difference from multiplying stationary-start transition densities
is second order in the inverse switching rate once the incoming filter is
pi + O(1/lambda).  A specified initial regime contributes a first-order term
only on the first observation interval.  The certificate also extracts the
matrix-kernel coefficient and compares it with the independently extrapolated
exact likelihood gaps.
"""
import json
import math
import os

import numpy as np


def matrix_density(xp, x, Q, kappa, theta, variance, delta, nfreq=601, steps=700):
    """Matrix density K_ij(x,xp) by vectorized Fourier inversion and RK4."""
    Q = np.asarray(Q, float)
    theta, variance = np.asarray(theta, float), np.asarray(variance, float)
    # The averaged Gaussian scale sets a conservative Fourier cutoff.
    pi = stationary(Q)
    v = (pi @ variance) * (1.0 - math.exp(-2.0 * kappa * delta)) / (2.0 * kappa)
    cutoff = 13.0 / math.sqrt(v)
    u = np.linspace(-cutoff, cutoff, nfreq)
    ident = np.eye(len(pi), dtype=complex)
    fundamental = np.broadcast_to(ident, (nfreq, len(pi), len(pi))).copy()
    dt = delta / steps

    def rhs(t, value):
        decay = math.exp(-kappa * t)
        g = (1j * u[:, None] * kappa * theta[None, :] * decay
             - 0.5 * u[:, None] ** 2 * variance[None, :] * decay ** 2)
        A = Q[None, :, :] + np.eye(len(pi))[None, :, :] * g[:, :, None]
        return np.einsum("uij,ujk->uik", A, value)

    for step in range(steps):
        t = step * dt
        k1 = rhs(t, fundamental)
        k2 = rhs(t + 0.5 * dt, fundamental + 0.5 * dt * k1)
        k3 = rhs(t + 0.5 * dt, fundamental + 0.5 * dt * k2)
        k4 = rhs(t + dt, fundamental + dt * k3)
        fundamental += dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    phase = np.exp(1j * u * math.exp(-kappa * delta) * x
                   - 1j * u * xp)
    density = np.trapezoid(fundamental * phase[:, None, None], u, axis=0).real / (2.0 * math.pi)
    # Fourier noise is many orders below the positive entries for this certificate.
    return density


def stationary(Q):
    w, v = np.linalg.eig(np.asarray(Q, float).T)
    pi = np.real(v[:, np.argmin(np.abs(w))])
    return pi / pi.sum()


def transition_kernels(path, rate, nfreq=601, steps=700):
    kappa, delta = 0.5, 0.25
    theta = np.array([0.08, 0.02])
    variance = np.array([0.0009, 0.0001])
    Q = rate * np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = stationary(Q)
    kernels = [matrix_density(xp, x, Q, kappa, theta, variance, delta,
                              nfreq=nfreq, steps=steps)
               for x, xp in zip(path[:-1], path[1:])]
    return kernels, pi


def likelihoods(kernels, pi, initial):
    filt = np.asarray(initial, float).copy()
    full = composite = 0.0
    predictive_gaps, filter_gaps = [], []
    for K in kernels:
        pred = float(filt @ K @ np.ones(2))
        pred_pi = float(pi @ K @ np.ones(2))
        if min(pred, pred_pi) <= 0.0:
            raise RuntimeError("Fourier inversion produced a nonpositive predictive density")
        full += math.log(pred)
        composite += math.log(pred_pi)
        predictive_gaps.append(math.log(pred) - math.log(pred_pi))
        filt = filt @ K / pred
        filter_gaps.append(float(np.max(np.abs(filt - pi))))
    return full, composite, predictive_gaps, filter_gaps


def averaged_densities(path):
    """Leading Gaussian transition densities along the fixed path."""
    kappa, delta = 0.5, 0.25
    theta_bar, variance_bar = 0.05, 0.0005
    decay = math.exp(-kappa * delta)
    v = variance_bar * (1.0 - decay ** 2) / (2.0 * kappa)
    ans = []
    for x, xp in zip(path[:-1], path[1:]):
        mean = decay * x + (1.0 - decay) * theta_bar
        ans.append(math.exp(-0.5 * (xp - mean) ** 2 / v) / math.sqrt(2.0 * math.pi * v))
    return np.asarray(ans)


def kernel_first_order(kernels_slow, rate_slow, kernels_fast, rate_fast,
                       leading, pi):
    """Richardson estimate of K_1 in K_m=p0 1 pi'+K_1/m+O(m^-2)."""
    if rate_fast != 2.0 * rate_slow:
        raise ValueError("certificate expects one rate doubling")
    outer = np.outer(np.ones(len(pi)), pi)
    estimates = []
    for p0, Ks, Kf in zip(leading, kernels_slow, kernels_fast):
        As = rate_slow * (Ks - p0 * outer)
        Af = rate_fast * (Kf - p0 * outer)
        estimates.append(2.0 * Af - As)
    return estimates


def likelihood_coefficients(K1s, leading, pi, initial):
    """First initial-prior coefficient and subsequent second-order coefficient."""
    one = np.ones(len(pi))
    initial = np.asarray(initial, float)
    h0 = K1s[0] @ one
    first = float((initial - pi) @ h0 / leading[0])
    if np.max(np.abs(initial - pi)) < 1.0e-14:
        r = np.zeros_like(pi)
        start = 0
    else:
        r = (initial @ K1s[0] - float(initial @ h0) * pi) / leading[0]
        start = 1
    second = 0.0
    for K1, p0 in zip(K1s[start:], leading[start:]):
        h = K1 @ one
        second += float(r @ h / p0)
        r = (pi @ K1 - float(pi @ h) * pi) / p0
    return first, second


def observed_order(errors):
    errors = np.asarray(errors, float)
    return float(np.log(errors[-2] / errors[-1]) / np.log(2.0))


def main():
    # A fixed, ordinary path under the averaged AR(1) scale.  Keeping it fixed
    # isolates the asymptotic likelihood calculation from Monte Carlo noise.
    path = np.array([0.0500, 0.0555, 0.0470, 0.0610, 0.0520, 0.0445, 0.0580])
    rates = np.array([40.0, 80.0, 160.0, 320.0])
    stationary_errors, specified_first, specified_tail, filter_errors = [], [], [], []
    rows, kernels_by_rate = [], {}
    for rate in rates:
        kernels, pi = transition_kernels(path, rate)
        kernels_by_rate[rate] = kernels
        full, comp, gaps, fgaps = likelihoods(kernels, pi, [0.5, 0.5])
        full_s, comp_s, gaps_s, _ = likelihoods(kernels, pi, [1.0, 0.0])
        stationary_errors.append(abs(full - comp))
        specified_first.append(abs(gaps_s[0]))
        specified_tail.append(abs(sum(gaps_s[1:])))
        filter_errors.append(max(fgaps))
        rows.append([rate, full - comp, gaps_s[0], sum(gaps_s[1:]), max(fgaps)])
        print(f"rate={rate:5.0f}: stationary log gap={full-comp:+.3e}, "
              f"specified first={gaps_s[0]:+.3e}, specified tail={sum(gaps_s[1:]):+.3e}, "
              f"max filter gap={max(fgaps):.3e}")

    orders = {
        "stationary_log_gap": observed_order(stationary_errors),
        "specified_first_gap": observed_order(specified_first),
        "specified_tail_gap": observed_order(specified_tail),
        "filter_gap": observed_order(filter_errors),
    }
    # An independently refined inversion guards against mistaking quadrature or
    # ODE error for the second-order likelihood gap at the fastest rate.
    kernels_ref, pi = transition_kernels(path, rates[-1], nfreq=901, steps=1000)
    full_ref, comp_ref, _, _ = likelihoods(kernels_ref, pi, [0.5, 0.5])
    resolution_error = abs((full_ref - comp_ref) - rows[-1][1])

    leading = averaged_densities(path)
    K1s = kernel_first_order(kernels_by_rate[rates[-2]], rates[-2],
                             kernels_by_rate[rates[-1]], rates[-1],
                             leading, pi)
    _, stationary_coefficient = likelihood_coefficients(K1s, leading, pi, pi)
    specified_first_coefficient, specified_tail_coefficient = likelihood_coefficients(
        K1s, leading, pi, [1.0, 0.0])
    scaled_fastest = {
        "stationary": rates[-1] ** 2 * rows[-1][1],
        "specified_first": rates[-1] * rows[-1][2],
        "specified_tail": rates[-1] ** 2 * rows[-1][3],
    }
    scaled_previous = {
        "stationary": rates[-2] ** 2 * rows[-2][1],
        "specified_first": rates[-2] * rows[-2][2],
        "specified_tail": rates[-2] ** 2 * rows[-2][3],
    }
    extrapolated = {key: 2.0 * scaled_fastest[key] - scaled_previous[key]
                    for key in scaled_fastest}
    coefficients = {
        "stationary": stationary_coefficient,
        "specified_first": specified_first_coefficient,
        "specified_tail": specified_tail_coefficient,
    }
    coefficient_errors = {key: abs(extrapolated[key] - coefficients[key])
                          for key in extrapolated}
    print("orders:", ", ".join(f"{k}={v:.6f}" for k, v in orders.items()))
    print(f"refined fastest-rate gap change={resolution_error:.3e}")
    print("coefficient check:", ", ".join(
        f"{key} extrapolated={extrapolated[key]:+.6f}, predicted={coefficients[key]:+.6f}"
        for key in extrapolated))
    out = {"path": path.tolist(), "rows": rows, "orders": orders,
           "refined_fastest_gap": full_ref - comp_ref,
           "resolution_error": resolution_error,
           "scaled_fastest_gaps": scaled_fastest,
           "extrapolated_coefficients": extrapolated,
           "predicted_coefficients": coefficients,
           "coefficient_errors": coefficient_errors}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "filtered_likelihood_results.json"), "w") as f:
        json.dump(out, f, indent=2)

    ok = (orders["stationary_log_gap"] > 1.8
          and 0.85 < orders["specified_first_gap"] < 1.15
          and orders["specified_tail_gap"] > 1.8
          and 0.85 < orders["filter_gap"] < 1.15
          and resolution_error < 1.0e-8
          and max(coefficient_errors.values()) < 0.01)
    print("PASS" if ok else "FAIL")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
