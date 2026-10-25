"""Certificate for exact direction visibility in a two-channel arrival race.

For total hazard d and stationary time reversal Q*, put

    R = (diag(d) - Q)^-1,  R* = (diag(d) - Q*)^-1,
    w_d = (R* - R) 1.

Then p_r(Q) - p_r(Q*) = <w_d, h_r>_pi exactly.  The witness is
orthogonal to both 1 and d.  Consequently every hazard affine in d is
blind, while four or more regimes also admit non-affine blind hazards.
"""
import numpy as np

from effective_generator import (
    first_arrival_probability,
    stationary,
    time_reversal,
)


ROUND_TOL = 64 * np.finfo(float).eps


def inner(pi, left, right):
    """L2(pi) inner product."""
    return float(pi @ (np.asarray(left) * np.asarray(right)))


def witness(Q, total_hazard):
    """Return pi, Q*, R, R* and the exact reversal witness w_d."""
    pi = stationary(Q)
    reversed_Q = time_reversal(Q)
    killing = np.diag(total_hazard)
    resolvent = np.linalg.inv(killing - Q)
    reversed_resolvent = np.linalg.inv(killing - reversed_Q)
    w_d = (reversed_resolvent - resolvent) @ np.ones(len(pi))
    return pi, reversed_Q, resolvent, reversed_resolvent, w_d


def main():
    rates = np.array([
        [0.0, 2.0, 0.4, 0.1],
        [0.2, 0.0, 1.7, 0.5],
        [1.1, 0.3, 0.0, 2.2],
        [0.6, 1.4, 0.2, 0.0],
    ])
    Q = rates.copy()
    Q[np.diag_indices(4)] = -Q.sum(axis=1)
    total = np.array([0.05, 0.08, 0.12, 0.19])
    pi, reversed_Q, resolvent, reversed_resolvent, w_d = witness(Q, total)

    ones = np.ones(4)
    adjoint_error = np.max(abs(
        np.diag(pi) @ reversed_resolvent
        - resolvent.T @ np.diag(pi)
    ))
    resolvent_error = max(
        np.max(abs(resolvent @ total - ones)),
        np.max(abs(reversed_resolvent @ total - ones)),
    )
    orthogonality_error = max(
        abs(inner(pi, w_d, ones)),
        abs(inner(pi, w_d, total)),
    )
    assert adjoint_error < ROUND_TOL
    assert resolvent_error < ROUND_TOL
    assert orthogonality_error < ROUND_TOL
    assert np.max(abs(Q - reversed_Q)) > 1.0
    assert np.max(abs(w_d)) > 0.05

    rng = np.random.default_rng(20261025)
    identity_error = 0.0
    for _ in range(64):
        first = total * rng.uniform(0.1, 0.9, len(total))
        hazards = np.array([first, total - first])
        direct_gap = (
            first_arrival_probability(Q, hazards)
            - first_arrival_probability(reversed_Q, hazards)
        )
        witness_gap = inner(pi, w_d, first)
        identity_error = max(identity_error, abs(direct_gap - witness_gap))
    assert identity_error < ROUND_TOL

    # A visible positive allocation: move from d/2 in the witness direction.
    visible_first = total / 2 + 0.02 * w_d / np.max(abs(w_d))
    visible_hazards = np.array([visible_first, total - visible_first])
    visible_gap = (
        first_arrival_probability(Q, visible_hazards)
        - first_arrival_probability(reversed_Q, visible_hazards)
    )
    assert np.min(visible_hazards) > 0
    assert visible_gap > 7e-4
    assert abs(visible_gap - inner(pi, w_d, visible_first)) < ROUND_TOL

    # Every affine allocation is blind because w_d is orthogonal to 1 and d.
    affine_first = 0.01 * ones + 0.3 * total
    affine_hazards = np.array([affine_first, total - affine_first])
    affine_gap = (
        first_arrival_probability(Q, affine_hazards)
        - first_arrival_probability(reversed_Q, affine_hazards)
    )
    assert np.min(affine_hazards) > 0
    assert abs(affine_gap) < ROUND_TOL

    # In four states w_d^perp is three-dimensional, larger than span{1,d}.
    # Project a seed onto w_d^perp, then perturb d/2 in that non-affine direction.
    seed = np.array([1.0, -1.0, 0.5, -0.3])
    blind_direction = seed - w_d * inner(pi, w_d, seed) / inner(pi, w_d, w_d)
    blind_direction /= np.max(abs(blind_direction))
    blind_first = total / 2 + 0.02 * blind_direction
    blind_hazards = np.array([blind_first, total - blind_first])
    blind_gap = (
        first_arrival_probability(Q, blind_hazards)
        - first_arrival_probability(reversed_Q, blind_hazards)
    )
    affine_design = np.column_stack([ones, total])
    affine_fit = affine_design @ np.linalg.lstsq(
        affine_design, blind_first, rcond=None
    )[0]
    nonaffine_residual = np.max(abs(blind_first - affine_fit))
    assert np.min(blind_hazards) > 0
    assert abs(inner(pi, w_d, blind_first)) < ROUND_TOL
    assert abs(blind_gap) < ROUND_TOL
    assert nonaffine_residual > 0.02

    print(
        "resolvent identity:",
        f"64 allocations, max error {identity_error:.3e};",
        f"adjoint error {adjoint_error:.3e}",
    )
    print(
        "witness orthogonality:",
        f"max error {orthogonality_error:.3e};",
        f"visible reversal gap {visible_gap:.9f}",
    )
    print(
        "blind allocations:",
        f"affine gap {affine_gap:.3e};",
        f"non-affine gap {blind_gap:.3e};",
        f"non-affine residual {nonaffine_residual:.6f}",
    )
    print("PASS")


if __name__ == "__main__":
    main()
