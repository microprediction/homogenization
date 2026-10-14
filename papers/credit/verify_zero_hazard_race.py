"""Exact trace reduction and all-speed bounds with zero-hazard regimes."""

from fractions import Fraction

import numpy as np

from verify_general_race_bound import ROUND_TOL, check, error


def verify(q, pi, hazards, speeds):
    total = hazards.sum(axis=1)
    active, inactive = np.flatnonzero(total > 0), np.flatnonzero(total == 0)
    hitting = -np.linalg.solve(q[np.ix_(inactive, inactive)], q[np.ix_(inactive, active)])
    entrance = np.zeros((len(pi), len(active)))
    entrance[active] = np.eye(len(active))
    entrance[inactive] = hitting
    trace = q[np.ix_(active, active)] + q[np.ix_(active, inactive)] @ hitting
    trace_pi = pi[active]/pi[active].sum()
    assert hitting.min() >= -ROUND_TOL
    discrepancy = max(error(hitting.sum(axis=1), np.ones(len(inactive))),
                      error(trace.sum(axis=1), np.zeros(len(active))),
                      error(trace_pi @ trace, np.zeros(len(active))))
    # The single-active-state trace generator is exactly zero; remove Schur roundoff.
    if len(active) == 1:
        trace = np.zeros((1, 1))
    h = np.linalg.inv(-q+np.outer(np.ones(len(pi)), pi))-np.outer(np.ones(len(pi)), pi)
    pooled = pi @ hazards/(pi @ total)
    alpha = pi*total/(pi @ total)
    coefficient = (np.eye(len(pi))-np.outer(np.ones(len(pi)), alpha)) @ h @ (
        hazards-total[:, None]*pooled)
    ht = np.linalg.inv(-trace+np.outer(np.ones(len(active)), trace_pi)) - np.outer(
        np.ones(len(active)), trace_pi)
    trace_total = total[active]
    trace_alpha = trace_pi*trace_total/(trace_pi @ trace_total)
    trace_coefficient = (np.eye(len(active))-np.outer(np.ones(len(active)), trace_alpha)) @ ht @ (
        hazards[active]-trace_total[:, None]*pooled)
    discrepancy = max(discrepancy, error(coefficient, entrance @ trace_coefficient))
    for speed in speeds:
        result = check(trace, trace_pi, hazards[active], speed)
        truth = np.linalg.solve(np.diag(total)-speed*q, hazards)
        reduced = entrance @ np.linalg.solve(np.diag(trace_total)-speed*trace, hazards[active])
        residual = truth-pooled-coefficient/speed
        discrepancy = max(discrepancy, result[0], error(truth, reduced),
                          error(alpha @ truth, pooled))
        assert np.all(abs(residual) <= result[3]/speed**2+ROUND_TOL*np.maximum(1, abs(truth)))
        if len(active) == 1:
            assert error(truth, np.tile(pooled, (len(pi), 1))) <= ROUND_TOL
    return discrepancy, entrance, trace, truth, coefficient


def main():
    rng = np.random.default_rng(2026101414)
    maximum = 0.0
    cases = 0
    for n in range(3, 9):
        for repeat in range(6):
            q = rng.uniform(.03, 2., (n, n))
            np.fill_diagonal(q, 0)
            np.fill_diagonal(q, -q.sum(axis=1))
            system = q.T.copy()
            system[-1] = 1
            target = np.zeros(n)
            target[-1] = 1
            pi = np.linalg.solve(system, target)
            hazards = rng.uniform(.05, 3., (n, 3))
            hazards[1+repeat % (n-1):] = 0
            maximum = max(maximum, verify(q, pi, hazards, np.geomspace(.02, 100, 13))[0])
            cases += 1
    q = np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]])
    pi = np.array([4., 4., 1.])/9
    hazards = np.array([[1., 1.], [0., 0.], [.5, 3.5]])
    benchmark = verify(q, pi, hazards, [1.])
    maximum = max(maximum, benchmark[0])
    assert error(pi @ benchmark[3], [float(Fraction(41, 120)), float(Fraction(79, 120))]) <= ROUND_TOL
    assert error(pi @ benchmark[4], [-1/18, 1/18]) <= ROUND_TOL
    assert maximum <= ROUND_TOL
    print("zero-hazard trace certificate; random chains", cases, "speeds per chain 13; grid cases", 13*cases)
    print("maximum normalized reduction/coefficient/bound discrepancy", f"{maximum:.3e}")
    print("benchmark trace generator", benchmark[2].tolist())
    print("benchmark zero-state hazards", hazards.tolist())
    print("benchmark original-stationary entrance prior", (pi @ benchmark[1]).tolist(), "trace stationary prior", [4/5, 1/5])
    print("benchmark stationary-original cause probabilities", ["41/120", "79/120"])
    print("benchmark first coefficients", ["-1/18", "1/18"])
    print("single-active-state exact cancellation checked")
    print("roundoff tolerance", f"{ROUND_TOL:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
