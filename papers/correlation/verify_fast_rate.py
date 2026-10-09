"""Focused certificate for the finite-state fast-rate cumulant theorem.

On a nonreversible three-state chain, this compares exact mixed occupation
cumulants with the derivatives of the principal tilted-generator eigenvalue.
It also checks the induced singleton/pair rate hierarchy for bivariate returns.
Ends with PASS or FAIL.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from correlation import (cumulants_exact, occupation_joint_cumulants_cauchy,
                         occupation_joint_cumulant_bulk_cauchy)


T = 1.0
SPEEDS = [1, 2, 4, 8, 16]
Q = np.array([[-3.0, 2.0, 1.0],
              [1.0, -4.0, 3.0],
              [2.0, 1.0, -3.0]])
MU1 = np.array([0.10, -0.15, 0.03])
MU2 = np.array([0.05, 0.08, -0.20])
VOL1 = np.array([0.15, 0.28, 0.18])
VOL2 = np.array([0.12, 0.22, 0.30])
RHO = np.array([0.10, 0.75, -0.35])


def rate(values):
    return math.log(values[-2] / values[-1], 2)


def main():
    ok = True
    bulk = occupation_joint_cumulant_bulk_cauchy(
        Q, max_order=5, r=0.2, N=32
    )
    stationary = np.linalg.solve(
        np.vstack([Q.T[:-1], np.ones(3)]), np.r_[np.zeros(2), 1.0]
    )
    first_error = max(abs(bulk[(a,)] - stationary[a + 1])
                      for a in range(2))
    print(f"principal-eigenvalue first-derivative error {first_error:.2e}")
    ok &= first_error < 2e-13

    exact = {
        speed: occupation_joint_cumulants_cauchy(
            T, speed * Q, max_order=5, r=0.32, N=32
        )
        for speed in SPEEDS
    }
    for order in range(2, 6):
        keys = [key for key in bulk if len(key) == order]
        magnitudes = [max(abs(exact[speed][key]) for key in keys)
                      for speed in SPEEDS]
        residuals = [max(abs(
            exact[speed][key] - T * speed ** (1 - order) * bulk[key]
        ) for key in keys) for speed in SPEEDS]
        magnitude_rate = rate(magnitudes)
        residual_rate = rate(residuals)
        print(f"occupation order {order}: rate {magnitude_rate:.3f}; "
              f"bulk-residual rate {residual_rate:.3f}")
        ok &= magnitude_rate > order - 1.1
        ok &= residual_rate > order - 0.1

    tensors = [cumulants_exact(
        T, speed * Q, MU1, MU2, VOL1, VOL2, RHO,
        r=0.32, N=48, max_order=5
    ) for speed in SPEEDS]
    for order in range(3, 6):
        keys = ['k' + '1' * (order - j) + '2' * j
                for j in range(order + 1)]
        magnitudes = [max(abs(tensor[key]) for key in keys)
                      for tensor in tensors]
        return_rate = rate(magnitudes)
        target = math.ceil(order / 2) - 1
        print(f"return order {order}: rate {return_rate:.3f} "
              f"(target {target})")
        ok &= return_rate > target - 0.1

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
