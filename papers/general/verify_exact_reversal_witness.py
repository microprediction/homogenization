"""Certificate for the exact first-arrival time-reversal witness."""

import numpy as np

from effective_generator import stationary, time_reversal


ROUNDOFF_TOL = 512 * np.finfo(float).eps


def probabilities(generator, hazards):
    """Stationary first-arrival probabilities of all causes."""
    pi = stationary(generator)
    total = hazards.sum(axis=0)
    values = np.linalg.solve(np.diag(total) - generator, hazards.T)
    return pi @ values


def witness(generator, total):
    """Return pi and w=(R^*-R)1 for stationary time reversal."""
    pi = stationary(generator)
    reverse = time_reversal(generator)
    one = np.ones(len(total))
    forward_lifetime = np.linalg.solve(np.diag(total) - generator, one)
    reverse_lifetime = np.linalg.solve(np.diag(total) - reverse, one)
    return pi, reverse_lifetime - forward_lifetime


def normalized_error(left, right):
    """Roundoff-scaled absolute discrepancy."""
    return np.max(np.abs(left - right)) / max(
        1.0, np.max(np.abs(left)), np.max(np.abs(right))
    )


def identity_errors(generator, hazards):
    """Errors in the exact gap and the two witness orthogonalities."""
    total = hazards.sum(axis=0)
    pi, w = witness(generator, total)
    direct = probabilities(generator, hazards) - probabilities(
        time_reversal(generator), hazards
    )
    predicted = hazards @ (pi * w)
    gap_error = normalized_error(direct, predicted)
    one_error = abs(pi @ w) / max(1.0, np.linalg.norm(w))
    total_error = abs((pi * w) @ total) / max(
        1.0, np.linalg.norm(w) * np.linalg.norm(total)
    )
    simplex_error = abs(direct.sum())
    return gap_error, one_error, total_error, simplex_error, direct, w


def nonaffine_blind_pair(generator, total):
    """Construct a positive blind pair with h1 outside span{1,total}."""
    pi, w = witness(generator, total)
    constraints = np.vstack((pi * w, np.ones(len(total)), total))
    _, _, right_vectors = np.linalg.svd(constraints)
    direction = right_vectors[-1]
    scale = 0.2 * min(
        np.min(0.4 * total / np.maximum(np.abs(direction), 1e-100)),
        np.min(0.6 * total / np.maximum(np.abs(direction), 1e-100)),
    )
    first = 0.4 * total + scale * direction
    hazards = np.array((first, total - first))
    affine_fit = np.linalg.lstsq(
        np.column_stack((np.ones(len(total)), total)), first, rcond=None
    )[0]
    affine_residual = np.linalg.norm(
        first - np.column_stack((np.ones(len(total)), total)) @ affine_fit
    )
    return hazards, affine_residual


def main():
    cycle = np.array([
        [-2.1, 2.0, 0.1],
        [0.1, -2.1, 2.0],
        [2.0, 0.1, -2.1],
    ])
    visible = np.array([
        [0.07, 0.01, 0.01],
        [0.01, 0.07, 0.01],
    ])
    errors = list(identity_errors(cycle, visible)[:4])
    visible_gap = identity_errors(cycle, visible)[4]

    # If h2=alpha+beta*h1, then h1 is affine in the total hazard.
    first = np.array([0.07, 0.01, 0.04])
    affine = np.array((first, 0.02 + 0.5 * first))
    affine_gap = np.max(np.abs(
        probabilities(cycle, affine)
        - probabilities(time_reversal(cycle), affine)
    ))

    constant_first = np.array([0.08, 0.02, 0.05])
    constant = np.array((constant_first, 0.10 - constant_first))
    constant_total_gap = np.max(np.abs(
        probabilities(cycle, constant)
        - probabilities(time_reversal(cycle), constant)
    ))

    four_state = np.array([
        [-3.4, 2.0, 1.0, 0.4],
        [0.2, -2.4, 1.8, 0.4],
        [0.5, 0.3, -2.8, 2.0],
        [1.7, 0.2, 0.1, -2.0],
    ])
    total = np.array([0.08, 0.05, 0.11, 0.07])
    blind, nonaffine_residual = nonaffine_blind_pair(four_state, total)
    blind_gap = np.max(np.abs(
        probabilities(four_state, blind)
        - probabilities(time_reversal(four_state), blind)
    ))
    errors.extend(identity_errors(four_state, blind)[:4])

    rng = np.random.default_rng(20261012)
    for _ in range(32):
        rates = rng.uniform(0.1, 2.0, (4, 4))
        np.fill_diagonal(rates, 0.0)
        rates[np.diag_indices(4)] = -rates.sum(axis=1)
        hazards = rng.uniform(0.005, 0.12, (3, 4))
        errors.extend(identity_errors(rates, hazards)[:4])

    maximum_identity_error = max(errors)
    assert maximum_identity_error <= ROUNDOFF_TOL
    assert affine_gap <= ROUNDOFF_TOL
    assert constant_total_gap <= ROUNDOFF_TOL
    assert blind_gap <= ROUNDOFF_TOL
    assert nonaffine_residual > 1e-4
    assert np.max(np.abs(visible_gap)) > 1e-3
    assert np.allclose(visible_gap.sum(), 0.0, atol=ROUNDOFF_TOL)

    print("exact first-arrival reversal-witness certificate")
    print("  visible two-cause gaps "
          + " ".join(f"{value:+.10e}" for value in visible_gap))
    print(f"  maximum affine-pair gap {affine_gap:.3e}")
    print(f"  maximum constant-total-hazard gap {constant_total_gap:.3e}")
    print(f"  nonaffine blind-pair residual {nonaffine_residual:.10e}")
    print(f"  maximum nonaffine blind-pair gap {blind_gap:.3e}")
    print(f"  maximum normalized identity error {maximum_identity_error:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
