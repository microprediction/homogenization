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
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from correlation import (cov_entries, cumulants_exact,
                         critical_return_cumulant_coefficients,
                         gaussian_vector_occupation_cumulant,
                         initial_covariance_boundary,
                         initial_layer_coefficient,
                         occupation_covariance_rate,
                         occupation_third_cumulant_rate,
                         occupation_joint_cumulants_cauchy,
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
        Q, max_order=6, r=0.2, N=32
    )
    stationary = np.linalg.solve(
        np.vstack([Q.T[:-1], np.ones(3)]), np.r_[np.zeros(2), 1.0]
    )
    first_error = max(abs(bulk[(a,)] - stationary[a + 1])
                      for a in range(2))
    print(f"principal-eigenvalue first-derivative error {first_error:.2e}")
    ok &= first_error < 2e-13

    covariance_rate = occupation_covariance_rate(Q)
    second_error = max(abs(
        bulk[tuple(sorted((a, b)))] - covariance_rate[a, b]
    ) for a in range(2) for b in range(2))
    print(f"group-inverse second-derivative error {second_error:.2e}")
    ok &= second_error < 2e-13

    third_rate = occupation_third_cumulant_rate(Q)
    third_error = max(abs(
        bulk[tuple(sorted((a, b, c)))] - third_rate[a, b, c]
    ) for a in range(2) for b in range(2) for c in range(2))
    print(f"group-inverse third-derivative error {third_error:.2e}")
    ok &= third_error < 2e-13

    exact = {
        speed: occupation_joint_cumulants_cauchy(
            T, speed * Q, max_order=6, r=0.32, N=32
        )
        for speed in SPEEDS
    }
    for order in range(2, 7):
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
        # At order six the Cauchy extraction is already close to its
        # cancellation floor at the last speed; allow 0.15 in the fitted
        # exponent while still separating the predicted powers 6 and 5.
        ok &= residual_rate > order - 0.15

    tensors = [cumulants_exact(
        T, speed * Q, MU1, MU2, VOL1, VOL2, RHO,
        r=0.32, N=48, max_order=6
    ) for speed in SPEEDS]
    for order in range(3, 7):
        keys = ['k' + '1' * (order - j) + '2' * j
                for j in range(order + 1)]
        magnitudes = [max(abs(tensor[key]) for key in keys)
                      for tensor in tensors]
        return_rate = rate(magnitudes)
        target = math.ceil(order / 2) - 1
        print(f"return order {order}: rate {return_rate:.3f} "
              f"(target {target})")
        ok &= return_rate > target - 0.1

    c11, c22, c12 = cov_entries(VOL1, VOL2, RHO)
    means = np.column_stack([MU1, MU2])
    covariances = np.array([
        [[c11[z], c12[z]], [c12[z], c22[z]]] for z in range(3)
    ])

    print("critical boundary layer s*T=tau")
    tau = 0.75
    critical_initial = np.array([0.2, 0.3, 0.5])
    critical_occupation = occupation_joint_cumulants_cauchy(
        tau, Q, max_order=6, r=0.32, N=32, initial=critical_initial
    )
    critical_speeds = [1, 4, 16, 64, 256]
    coefficient_table = {}
    maximum_identity_error = 0.0
    critical_exact = {}
    for speed in critical_speeds:
        critical_exact[speed] = cumulants_exact(
            tau, Q, MU1 / math.sqrt(speed), MU2 / math.sqrt(speed),
            VOL1, VOL2, RHO, r=0.32, N=48, max_order=6,
            initial=critical_initial
        )
    time_change_speed = 16
    original_short_time = cumulants_exact(
        tau / time_change_speed, time_change_speed * Q,
        MU1, MU2, VOL1, VOL2, RHO, r=0.08, N=64, max_order=4,
        initial=critical_initial
    )
    time_change_error = 0.0
    for order in range(2, 5):
        for twos in range(order + 1):
            key = 'k' + '1' * (order - twos) + '2' * twos
            time_change_error = max(time_change_error, abs(
                time_change_speed ** (order / 2) * original_short_time[key]
                - critical_exact[time_change_speed][key]
            ))
    for order in range(2, 7):
        for twos in range(order + 1):
            indices = (0,) * (order - twos) + (1,) * twos
            key = 'k' + '1' * (order - twos) + '2' * twos
            coefficients = critical_return_cumulant_coefficients(
                indices, tau, means[0], covariances[0],
                means[1:] - means[0], covariances[1:] - covariances[0],
                critical_occupation
            )
            coefficient_table[key] = coefficients
            for speed in critical_speeds:
                predicted = sum(value * speed ** (-singletons / 2)
                                for singletons, value in coefficients.items())
                maximum_identity_error = max(
                    maximum_identity_error,
                    abs(critical_exact[speed][key] - predicted)
                )

    variance_limit = coefficient_table['k11'][0]
    fourth_limit = coefficient_table['k1111'][0]
    sixth_limit = coefficient_table['k111111'][0]
    excess_kurtosis_limit = fourth_limit / variance_limit ** 2
    print(f"  exact coefficient identity error {maximum_identity_error:.2e}")
    print(f"  direct short-time rescaling error through order four "
          f"{time_change_error:.2e}")
    print(f"  asset-1 limit: variance {variance_limit:.8f}; "
          f"fourth cumulant {fourth_limit:.8e}; "
          f"sixth cumulant {sixth_limit:.8e}; "
          f"excess kurtosis {excess_kurtosis_limit:.6f}")
    for order in (3, 5):
        key = 'k' + '1' * order
        leading = coefficient_table[key][1]
        errors = [abs(math.sqrt(speed) * critical_exact[speed][key]
                      - leading) for speed in critical_speeds]
        ratio = errors[-2] / errors[-1]
        print(f"  order {order}: sqrt(s)-scaled limit {leading:+.8e}; "
              f"last quadrupling error ratio {ratio:.3f} (target 4)")
        ok &= ratio > 3.8
    for order in (4, 6):
        key = 'k' + '1' * order
        leading = coefficient_table[key][0]
        errors = [abs(critical_exact[speed][key] - leading)
                  for speed in critical_speeds]
        ratio = errors[-2] / errors[-1]
        print(f"  order {order}: persistent limit {leading:+.8e}; "
              f"last quadrupling error ratio {ratio:.3f} (target 4)")
        ok &= ratio > 3.8
    ok &= maximum_identity_error < 1e-11
    ok &= time_change_error < 1e-9
    ok &= abs(excess_kurtosis_limit) > 1e-4

    leading_occupation = {
        tuple(sorted((a, b))): covariance_rate[a, b]
        for a in range(2) for b in range(2)
    }
    for order in range(3, 5):
        for colors in np.ndindex(*(2,) * order):
            leading_occupation[tuple(sorted(colors))] = 0.0
    green_kubo_leading = {}
    for order in (3, 4):
        keys, leading = [], {}
        for j in range(order + 1):
            indices = (0,) * (order - j) + (1,) * j
            key = 'k' + '1' * (order - j) + '2' * j
            keys.append(key)
            leading[key] = gaussian_vector_occupation_cumulant(
                indices, T, means[0], covariances[0],
                means[1:] - means[0], covariances[1:] - covariances[0],
                leading_occupation
            )
        green_kubo_leading[order] = leading
        errors = [max(abs(tensor[key] - leading[key] / speed)
                      for key in keys)
                  for speed, tensor in zip(SPEEDS, tensors)]
        leading_error_rate = rate(errors)
        print(f"return order {order}: Green--Kubo leading-error rate "
              f"{leading_error_rate:.3f} (target 2)")
        ok &= leading_error_rate > 1.9

    leading_occupation = {}
    for occupation_order in range(1, 7):
        for colors in np.ndindex(*(2,) * occupation_order):
            key = tuple(sorted(colors))
            leading_occupation[key] = (
                T * third_rate[key] if occupation_order == 3 else 0.0
            )
    for order in (5, 6):
        keys, leading = [], {}
        for j in range(order + 1):
            indices = (0,) * (order - j) + (1,) * j
            key = 'k' + '1' * (order - j) + '2' * j
            keys.append(key)
            leading[key] = gaussian_vector_occupation_cumulant(
                indices, T, means[0], covariances[0],
                means[1:] - means[0], covariances[1:] - covariances[0],
                leading_occupation
            )
        errors = [max(abs(tensor[key] - leading[key] / speed ** 2)
                      for key in keys)
                  for speed, tensor in zip(SPEEDS, tensors)]
        leading_error_rate = rate(errors)
        print(f"return order {order}: third-spectral leading-error rate "
              f"{leading_error_rate:.3f} (target 3)")
        ok &= leading_error_rate > 2.85

    projection = np.outer(np.ones(3), stationary)
    group_inverse = np.linalg.inv(Q - projection) + projection
    mean_contrasts = means[1:] - means[0]
    mean_covariance_rate = (
        mean_contrasts.T @ covariance_rate @ mean_contrasts
    )
    stationary_covariance = np.tensordot(
        stationary, covariances, axes=(0, 0)
    )
    for label, initial in (
        ("state_zero", np.array([1.0, 0.0, 0.0])),
        ("mixed", np.array([0.2, 0.3, 0.5])),
    ):
        occupation_boundary = initial_layer_coefficient(
            Q, np.eye(3), initial
        )
        semigroup_errors = []
        covariance_errors = []
        occupation_covariance_boundary = initial_covariance_boundary(
            Q, np.eye(3)[:, 1:], initial
        )
        occupation_boundary_errors = []
        covariance_boundary = initial_layer_coefficient(
            Q, covariances, initial
        )
        initial_tensors = []
        for speed in SPEEDS:
            occupation = occupation_joint_cumulants_cauchy(
                T, speed * Q, max_order=2, r=0.32, N=32,
                initial=initial
            )
            exact_occupation_mean = np.array([
                T - occupation[(0,)] - occupation[(1,)],
                occupation[(0,)], occupation[(1,)]
            ])
            semigroup_mean = (
                T * stationary + occupation_boundary / speed
                + initial @ expm(speed * T * Q) @ group_inverse / speed
            )
            semigroup_errors.append(float(np.max(np.abs(
                exact_occupation_mean - semigroup_mean
            ))))
            exact_occupation_covariance = np.array([
                [occupation[(0, 0)], occupation[(0, 1)]],
                [occupation[(0, 1)], occupation[(1, 1)]]
            ])
            occupation_boundary_errors.append(float(np.max(np.abs(
                speed ** 2 * (
                    exact_occupation_covariance
                    - T * covariance_rate / speed
                ) - occupation_covariance_boundary
            ))))

            tensor = cumulants_exact(
                T, speed * Q, MU1, MU2, VOL1, VOL2, RHO,
                r=0.32, N=48, max_order=4, initial=initial
            )
            initial_tensors.append(tensor)
            exact_covariance = np.array([
                [tensor['k11'], tensor['k12']],
                [tensor['k12'], tensor['k22']]
            ])
            first_order_covariance = (
                T * stationary_covariance
                + (covariance_boundary + T * mean_covariance_rate) / speed
            )
            covariance_errors.append(float(np.max(np.abs(
                exact_covariance - first_order_covariance
            ))))
        covariance_remainder_rate = rate(covariance_errors)
        print(f"initial layer {label}: occupation semigroup error "
              f"{max(semigroup_errors):.2e}; covariance remainder rate "
              f"{covariance_remainder_rate:.3f} (target 2)")
        print(f"  second boundary at speed 4: occupation error "
              f"{occupation_boundary_errors[2]:.2e}")
        second_occupation = {}
        for occupation_order in range(1, 5):
            for colors in np.ndindex(*(2,) * occupation_order):
                key = tuple(sorted(colors))
                if occupation_order == 2:
                    second_occupation[key] = (
                        occupation_covariance_boundary[key]
                    )
                elif occupation_order == 3:
                    second_occupation[key] = T * third_rate[key]
                else:
                    second_occupation[key] = 0.0
        for order in (3, 4):
            second = {}
            for j in range(order + 1):
                indices = (0,) * (order - j) + (1,) * j
                key = 'k' + '1' * (order - j) + '2' * j
                second[key] = gaussian_vector_occupation_cumulant(
                    indices, T, means[0], covariances[0],
                    means[1:] - means[0], covariances[1:] - covariances[0],
                    second_occupation
                )
            errors = [max(abs(
                tensor[key] - green_kubo_leading[order][key] / speed
                - second[key] / speed ** 2
            ) for key in second)
                for speed, tensor in zip(SPEEDS, initial_tensors)]
            second_order_rate = rate(errors)
            print(f"  return order {order}: complete second-order error "
                  f"rate {second_order_rate:.3f} (target 3)")
            ok &= second_order_rate > 2.9
        if label == "state_zero":
            total = covariance_boundary + T * mean_covariance_rate
            print(f"  asset-2 coefficient: boundary "
                  f"{covariance_boundary[1, 1]:+.8f}; Green--Kubo "
                  f"{T * mean_covariance_rate[1, 1]:+.8f}; "
                  f"total {total[1, 1]:+.8f}")
        ok &= max(semigroup_errors) < 3e-13
        ok &= occupation_boundary_errors[2] < 1e-8
        ok &= covariance_remainder_rate > 1.95

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == '__main__':
    raise SystemExit(main())
