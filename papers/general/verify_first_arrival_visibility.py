"""Certificate for exact direction visibility in a two-channel arrival race.

For total hazard d and stationary time reversal Q*, put

    R = (diag(d) - Q)^-1,  R* = (diag(d) - Q*)^-1,
    w_d = (R* - R) 1.

Then p_r(Q) - p_r(Q*) = <w_d, h_r>_pi exactly for every number of
competing channels.  The witness is orthogonal to both 1 and d.
Consequently every hazard affine in d is blind, while four or more
regimes also admit non-affine blind hazards.  If d is constant then the
witness vanishes and every allocation is blind.
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

    # The same one-witness identity holds for any number of channels, and
    # the reversal gaps sum to zero because the channel hazards sum to d.
    raw = rng.uniform(0.1, 1.0, (5, len(total)))
    many_hazards = total * raw / raw.sum(axis=0)
    many_gaps = np.array([
        first_arrival_probability(Q, many_hazards, winner=index)
        - first_arrival_probability(reversed_Q, many_hazards, winner=index)
        for index in range(len(many_hazards))
    ])
    many_witness_gaps = np.array([
        inner(pi, w_d, hazard) for hazard in many_hazards
    ])
    many_error = np.max(abs(many_gaps - many_witness_gaps))
    assert many_error < ROUND_TOL
    assert abs(many_gaps.sum()) < ROUND_TOL

    # For fixed Q and d, all attainable gap vectors form the zero-sum
    # l1 ball of radius 2A, where A is the positive mass of pi*d*w_d.
    signed_budget = pi * total * w_d
    budget = np.maximum(signed_budget, 0.0).sum()
    assert abs(signed_budget.sum()) < ROUND_TOL
    assert np.sum(abs(many_gaps)) / 2 <= budget + ROUND_TOL

    # Construct strictly positive hazard splits for arbitrary interior
    # points of the ball and recover their prescribed gaps directly.
    target_error = 0.0
    minimum_hazard = np.inf
    cause_count = len(many_hazards)
    for _ in range(25):
        target = rng.normal(size=cause_count)
        target -= target.mean()
        target *= (
            rng.uniform(0.05, 0.95) * 2 * budget / np.sum(abs(target))
        )
        positive_mass = np.maximum(target, 0.0).sum()
        common = (1.0 - positive_mass / budget) / cause_count
        positive_split = np.maximum(target, 0.0) / budget + common
        negative_split = np.maximum(-target, 0.0) / budget + common
        shares = np.empty((cause_count, len(total)))
        for state, coefficient in enumerate(signed_budget):
            if coefficient > 0:
                shares[:, state] = positive_split
            elif coefficient < 0:
                shares[:, state] = negative_split
            else:
                shares[:, state] = 1.0 / cause_count
        target_hazards = shares * total
        target_gaps = np.array([
            first_arrival_probability(Q, target_hazards, winner=index)
            - first_arrival_probability(
                reversed_Q, target_hazards, winner=index
            )
            for index in range(cause_count)
        ])
        target_error = max(target_error, np.max(abs(target_gaps - target)))
        minimum_hazard = min(minimum_hazard, np.min(target_hazards))
    assert target_error < ROUND_TOL
    assert minimum_hazard > 0

    # The boundary is sharp: send every positive witness state to cause 0
    # and every negative witness state to cause 1.
    sharp_shares = np.zeros((cause_count, len(total)))
    for state, coefficient in enumerate(signed_budget):
        if coefficient > 0:
            sharp_shares[0, state] = 1.0
        elif coefficient < 0:
            sharp_shares[1, state] = 1.0
        else:
            sharp_shares[2, state] = 1.0
    sharp_hazards = sharp_shares * total
    sharp_gaps = np.array([
        first_arrival_probability(Q, sharp_hazards, winner=index)
        - first_arrival_probability(
            reversed_Q, sharp_hazards, winner=index
        )
        for index in range(cause_count)
    ])
    sharp_target = np.zeros(cause_count)
    sharp_target[0], sharp_target[1] = budget, -budget
    sharp_error = np.max(abs(sharp_gaps - sharp_target))
    assert sharp_error < ROUND_TOL
    assert abs(np.sum(abs(sharp_gaps)) / 2 - budget) < ROUND_TOL

    # The vertices are A(e_r-e_s).  They give the exact support function
    # A(max c-min c), so no extra cause improves a scalar contrast.
    vertex_gaps = {}
    vertex_error = 0.0
    for upper in range(cause_count):
        for lower in range(cause_count):
            if upper == lower:
                continue
            shares = np.zeros((cause_count, len(total)))
            for state, coefficient in enumerate(signed_budget):
                shares[upper if coefficient >= 0 else lower, state] = 1.0
            hazards = shares * total
            gaps = np.array([
                first_arrival_probability(Q, hazards, winner=index)
                - first_arrival_probability(
                    reversed_Q, hazards, winner=index
                )
                for index in range(cause_count)
            ])
            target = np.zeros(cause_count)
            target[upper], target[lower] = budget, -budget
            vertex_error = max(vertex_error, np.max(abs(gaps - target)))
            vertex_gaps[upper, lower] = gaps
    assert vertex_error < ROUND_TOL

    support_error = 0.0
    for _ in range(25):
        contrast = rng.normal(size=cause_count)
        upper, lower = np.argmax(contrast), np.argmin(contrast)
        attained = contrast @ vertex_gaps[upper, lower]
        predicted = budget * (contrast[upper] - contrast[lower])
        support_error = max(support_error, abs(attained - predicted))
    assert support_error < ROUND_TOL

    # Constant total killing makes R 1 = R* 1 = 1/c, so every split and
    # every number of channels is exactly blind even for a nonreversible Q.
    constant_total = np.full(len(total), 0.11)
    _, constant_reversed_Q, _, _, constant_witness = witness(
        Q, constant_total
    )
    constant_raw = rng.uniform(0.1, 1.0, (5, len(total)))
    constant_hazards = (
        constant_total * constant_raw / constant_raw.sum(axis=0)
    )
    constant_gaps = np.array([
        first_arrival_probability(Q, constant_hazards, winner=index)
        - first_arrival_probability(
            constant_reversed_Q, constant_hazards, winner=index
        )
        for index in range(len(constant_hazards))
    ])
    constant_error = max(
        np.max(abs(constant_witness)) / max(1.0, 1.0 / constant_total[0]),
        np.max(abs(constant_gaps)),
    )
    assert constant_error < ROUND_TOL

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
        f"five-channel error {many_error:.3e};",
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
    print(
        "constant total hazard:",
        f"five-channel max scaled error {constant_error:.3e}",
    )
    print(
        "attainable gap polytope:",
        f"TV budget {budget:.9f};",
        f"25 interior targets max error {target_error:.3e};",
        f"sharp-boundary error {sharp_error:.3e}",
    )
    print(
        "support function:",
        f"{cause_count * (cause_count - 1)} vertices max error "
        f"{vertex_error:.3e};",
        f"25 contrasts max error {support_error:.3e}",
    )
    print("PASS")


if __name__ == "__main__":
    main()

