"""Certificate for all stationary moments of a switched CIR variance.

For the row-vector convention used in the paper, the invariant component
moments r[p, i] = E[v**p 1_{Y=i}] obey a triangular sequence of killed-chain
linear systems.  This script checks that recursion against one large
polynomial-semigroup exponential, verifies the constant-coefficient Gamma
special case, tests the explicit inverse-speed expansion through cubic order,
and checks convergence of the full Taylor series inside the computable pole
disk on a nonreversible three-state chain.  It also verifies a noncancellation
criterion under which that disk is the exact Taylor disk.
"""
import math
import os
import sys

import mpmath as mp
import numpy as np
from scipy.linalg import expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "general"))
from effective_generator import group_inverse, stationary


DEGREE = 6
SPEEDS = (8, 16, 32, 64, 128)
ASYMPTOTIC_ORDER = 3
CONVERGENCE_ORDER = 30
SHARPNESS_ORDER = 100


def stationary_moments(speed, q0, pi, c, kappa, variance, degree=DEGREE):
    """Return rows r_p,i = E[v^p 1_{Y=i}], p=0,...,degree."""
    size = len(pi)
    rows = [np.asarray(pi, float)]
    for order in range(1, degree + 1):
        forcing = order * c + order * (order - 1) * variance / 2
        killed = order * np.diag(kappa) - speed * q0
        rows.append(np.linalg.solve(killed.T, rows[-1] * forcing))
        assert np.min(rows[-1]) >= -2e-14
        assert np.linalg.norm(rows[-1] @ killed - rows[-2] * forcing) < 3e-13
    return np.asarray(rows)


def moment_semigroup(speed, q0, initial_regime, v0, c, kappa, variance,
                     degree=DEGREE, maturity=80.0):
    """Evolve every regime-resolved monomial moment in one block exponential."""
    states = len(initial_regime)
    generator = np.zeros(((degree + 1) * states, (degree + 1) * states))
    for order in range(degree + 1):
        block = speed * q0 - order * np.diag(kappa)
        lo, hi = order * states, (order + 1) * states
        generator[lo:hi, lo:hi] = block
        if order < degree:
            forcing = (order + 1) * c + order * (order + 1) * variance / 2
            next_lo, next_hi = hi, hi + states
            generator[lo:hi, next_lo:next_hi] = np.diag(forcing)
    initial = np.concatenate([initial_regime * v0**order for order in range(degree + 1)])
    return (initial @ expm(maturity * generator)).reshape(degree + 1, states)


def fast_coefficients(q0, pi, c, kappa, variance, degree=DEGREE,
                      asymptotic_order=ASYMPTOTIC_ORDER):
    """Return r[p,n] in r_p(m)=sum_n m**(-n) r[p,n]+remainder."""
    inverse = group_inverse(q0)
    bar_kappa = pi @ kappa
    coefficients = np.zeros((degree + 1, asymptotic_order + 1, len(pi)))
    centered = np.zeros_like(coefficients)
    scalar = np.zeros((degree + 1, asymptotic_order + 1))
    coefficients[0, 0] = pi
    scalar[0, 0] = 1.0
    for inverse_order in range(asymptotic_order + 1):
        for moment_order in range(1, degree + 1):
            forcing = (
                moment_order * c
                + moment_order * (moment_order - 1) * variance / 2
            )
            scalar[moment_order, inverse_order] = (
                coefficients[moment_order - 1, inverse_order] @ forcing
                - moment_order
                * (centered[moment_order, inverse_order] @ kappa)
            ) / (moment_order * bar_kappa)
            coefficients[moment_order, inverse_order] = (
                centered[moment_order, inverse_order]
                + scalar[moment_order, inverse_order] * pi
            )
            if inverse_order < asymptotic_order:
                source = (
                    moment_order
                    * coefficients[moment_order, inverse_order]
                    * kappa
                    - coefficients[moment_order - 1, inverse_order] * forcing
                )
                assert abs(source.sum()) < 8e-13
                centered[moment_order, inverse_order + 1] = source @ inverse
                assert abs(centered[moment_order, inverse_order + 1].sum()) < 8e-13
    return coefficients, scalar, centered


def gamma_moments(c, kappa, variance, degree=DEGREE):
    values = [1.0]
    for order in range(1, degree + 1):
        values.append(values[-1] * (c + (order - 1) * variance / 2) / kappa)
    return np.asarray(values)


def convergence_certificate(q0, c, kappa, variance, degree=DEGREE,
                            expansion_order=CONVERGENCE_ORDER, speed=8,
                            sharpness_order=SHARPNESS_ORDER):
    """High-precision Taylor convergence inside the explicit pole disk.

    For epsilon=1/speed, the pth exact moment row is a product of factors

        epsilon * (p * epsilon * D_kappa - Q_0)^(-1).

    The apparent singularity at epsilon=0 is removable.  Every other pole of
    the pth factor is lambda/p, where lambda is a nonzero eigenvalue of
    D_kappa^(-1) Q_0.  Thus the moments through ``degree`` are analytic in the
    guaranteed disk with radius min(abs(lambda))/degree.
    """
    eigenvalues = np.linalg.eigvals(np.linalg.solve(np.diag(kappa), q0))
    nonzero = eigenvalues[np.abs(eigenvalues) > 1e-11]
    radius = np.min(np.abs(nonzero)) / degree
    assert 1 / speed < radius

    old_dps = mp.mp.dps
    mp.mp.dps = 160
    try:
        size = len(kappa)
        q_mp = mp.matrix([[mp.mpf(str(value)) for value in row] for row in q0])
        c_mp = [mp.mpf(str(value)) for value in c]
        kappa_mp = [mp.mpf(str(value)) for value in kappa]
        variance_mp = [mp.mpf(str(value)) for value in variance]

        stationary_system = q_mp.T.copy()
        stationary_rhs = mp.matrix(size, 1)
        for column in range(size):
            stationary_system[size - 1, column] = 1
        stationary_rhs[size - 1] = 1
        pi = mp.lu_solve(stationary_system, stationary_rhs).T
        ones = mp.matrix(size, 1)
        for index in range(size):
            ones[index] = 1
        projection = ones * pi
        inverse = (q_mp + projection) ** -1 - projection
        bar_kappa = sum(pi[index] * kappa_mp[index]
                        for index in range(size))

        def zero_row():
            return mp.matrix(1, size)

        coefficient_order = max(expansion_order, sharpness_order)
        coefficients = [
            [zero_row() for _ in range(coefficient_order + 1)]
            for _ in range(degree + 1)
        ]
        centered = [
            [zero_row() for _ in range(coefficient_order + 1)]
            for _ in range(degree + 1)
        ]
        scalar = [
            [mp.mpf("0") for _ in range(coefficient_order + 1)]
            for _ in range(degree + 1)
        ]
        coefficients[0][0] = pi
        scalar[0][0] = 1
        for inverse_order in range(coefficient_order + 1):
            for moment_order in range(1, degree + 1):
                forcing = [
                    moment_order * c_mp[index]
                    + mp.mpf(moment_order * (moment_order - 1))
                    * variance_mp[index] / 2
                    for index in range(size)
                ]
                previous = sum(
                    coefficients[moment_order - 1][inverse_order][index]
                    * forcing[index] for index in range(size)
                )
                centered_kappa = sum(
                    centered[moment_order][inverse_order][index]
                    * kappa_mp[index] for index in range(size)
                )
                scalar[moment_order][inverse_order] = (
                    previous - moment_order * centered_kappa
                ) / (moment_order * bar_kappa)
                coefficients[moment_order][inverse_order] = (
                    centered[moment_order][inverse_order]
                    + scalar[moment_order][inverse_order] * pi
                )
                if inverse_order < coefficient_order:
                    source = mp.matrix([[
                        moment_order
                        * coefficients[moment_order][inverse_order][index]
                        * kappa_mp[index]
                        - coefficients[moment_order - 1][inverse_order][index]
                        * forcing[index]
                        for index in range(size)
                    ]])
                    assert abs(sum(source)) < mp.mpf("1e-65")
                    centered[moment_order][inverse_order + 1] = source * inverse

        speed_mp = mp.mpf(speed)
        exact_rows = [pi]
        for moment_order in range(1, degree + 1):
            forcing = mp.matrix([
                exact_rows[-1][index]
                * (moment_order * c_mp[index]
                   + mp.mpf(moment_order * (moment_order - 1))
                   * variance_mp[index] / 2)
                for index in range(size)
            ])
            killed = (mp.diag([moment_order * value for value in kappa_mp])
                      - speed_mp * q_mp)
            exact_rows.append(mp.lu_solve(killed.T, forcing).T)
        exact_scalar = [sum(row) for row in exact_rows]

        truncations = tuple(range(3, expansion_order + 1, 3))
        errors = []
        for truncation in truncations:
            errors.append(max(
                abs(exact_scalar[moment_order] - sum(
                    scalar[moment_order][inverse_order]
                    / speed_mp ** inverse_order
                    for inverse_order in range(truncation + 1)
                ))
                for moment_order in range(1, degree + 1)
            ))
        assert all(later < earlier for earlier, later in zip(errors, errors[1:]))
        assert errors[-1] < mp.mpf("1.6e-20")

        # The guaranteed radius is exact if its nearest candidate pole is
        # simple, isolated, and has nonzero residue after multiplication by
        # the preceding moment factors.  Here the nearest pole belongs only
        # to the degree-th factor.  Evaluate its residue by a high-precision
        # punctured limit and independently recover it from the 100th Taylor
        # coefficient: if R(epsilon) has residue A at epsilon_*, then
        # r_n ~ -A epsilon_*^(-n-1).
        d_kappa = mp.diag(kappa_mp)
        eigenvalues, right_eigenvectors = mp.eig(d_kappa ** -1 * q_mp)
        nonzero_eigenvalues = [
            value for value in eigenvalues if abs(value) > mp.mpf("1e-60")
        ]
        candidate_poles = sorted(
            [
                value / moment_order
                for moment_order in range(1, degree + 1)
                for value in nonzero_eigenvalues
            ],
            key=abs,
        )
        pole = candidate_poles[0]
        assert abs(abs(pole) - mp.mpf(str(radius))) < mp.mpf("2e-15")
        assert abs(candidate_poles[1]) > abs(pole) * mp.mpf("1.05")

        def exact_row(epsilon):
            row = pi.copy()
            for moment_order in range(1, degree + 1):
                forcing = [
                    moment_order * c_mp[index]
                    + mp.mpf(moment_order * (moment_order - 1))
                    * variance_mp[index] / 2
                    for index in range(size)
                ]
                row = (
                    row
                    * mp.diag(forcing)
                    * (
                        epsilon
                        * (moment_order * epsilon * d_kappa - q_mp) ** -1
                    )
                )
            return row

        residue_step = mp.mpf("1e-35")
        residue = residue_step * exact_row(pole + residue_step)
        residue_check_step = mp.mpf("1e-30")
        residue_check = residue_check_step * exact_row(
            pole + residue_check_step
        )
        residue_norm = mp.norm(residue)
        scalar_residue = sum(residue)
        assert residue_norm > mp.mpf("1e-10") ** 2
        assert abs(scalar_residue) > mp.mpf("1e-10") ** 2
        assert mp.norm(residue_check - residue) / residue_norm < mp.mpf("1e-4")

        # Independently evaluate the rank-one matrix-pencil residue from its
        # generalized left and right eigenvectors.
        pole_eigenvalue = degree * pole
        right_index = min(
            range(len(eigenvalues)),
            key=lambda index: abs(eigenvalues[index] - pole_eigenvalue),
        )
        left_eigenvalues, left_eigenvectors = mp.eig(
            d_kappa ** -1 * q_mp.T
        )
        left_index = min(
            range(len(left_eigenvalues)),
            key=lambda index: abs(
                left_eigenvalues[index] - pole_eigenvalue
            ),
        )
        right_vector = right_eigenvectors[:, right_index]
        left_vector = left_eigenvectors[:, left_index]
        prefix = pi.copy()
        for moment_order in range(1, degree):
            forcing = [
                moment_order * c_mp[index]
                + mp.mpf(moment_order * (moment_order - 1))
                * variance_mp[index] / 2
                for index in range(size)
            ]
            prefix = (
                prefix
                * mp.diag(forcing)
                * (
                    pole
                    * (moment_order * pole * d_kappa - q_mp) ** -1
                )
            )
        last_forcing = [
            degree * c_mp[index]
            + mp.mpf(degree * (degree - 1)) * variance_mp[index] / 2
            for index in range(size)
        ]
        eigenvector_residue = (
            (prefix * mp.diag(last_forcing) * right_vector)[0]
            * pole
            / (degree * (left_vector.T * d_kappa * right_vector)[0])
            * left_vector.T
        )
        residue_formula_error = (
            mp.norm(eigenvector_residue - residue) / residue_norm
        )
        assert residue_formula_error < mp.mpf("1e-25")
        coefficient_residue = (
            -pole ** (sharpness_order + 1)
            * coefficients[degree][sharpness_order]
        )
        residue_relative_error = (
            mp.norm(coefficient_residue - residue) / residue_norm
        )
        assert residue_relative_error < mp.mpf("2.2e-4")
        return (
            radius,
            truncations,
            errors,
            pole,
            residue_norm,
            scalar_residue,
            residue_formula_error,
            residue_relative_error,
        )
    finally:
        mp.mp.dps = old_dps


def observed_order(errors):
    return math.log(errors[-2] / errors[-1], 2)


def main():
    q0 = np.array(
        [
            [-1.4, 1.0, 0.4],
            [0.2, -1.1, 0.9],
            [0.7, 0.3, -1.0],
        ]
    )
    pi = stationary(q0)
    c = np.array([0.060, 0.120, 0.090])
    kappa = np.array([0.70, 1.40, 0.95])
    variance = np.array([0.035, 0.080, 0.050])
    assert np.min(2 * c - variance) > 0

    exact = stationary_moments(16, q0, pi, c, kappa, variance)
    evolved = moment_semigroup(16, q0, np.array([0.8, 0.1, 0.1]), 0.17,
                               c, kappa, variance)
    semigroup_error = np.max(np.abs(exact - evolved))
    assert semigroup_error < 2e-12

    common_c, common_kappa, common_variance = 0.08, 1.2, 0.04
    common = stationary_moments(
        7, q0, pi,
        np.full(3, common_c),
        np.full(3, common_kappa),
        np.full(3, common_variance),
    ).sum(axis=1)
    gamma_error = np.max(
        np.abs(common - gamma_moments(common_c, common_kappa, common_variance))
    )
    assert gamma_error < 2e-14

    coefficients, scalar, centered = fast_coefficients(
        q0, pi, c, kappa, variance
    )
    (
        radius,
        truncations,
        convergence_errors,
        nearest_pole,
        residue_norm,
        scalar_residue,
        residue_formula_error,
        residue_relative_error,
    ) = convergence_certificate(q0, c, kappa, variance)
    residuals = {order: [] for order in range(ASYMPTOTIC_ORDER + 1)}
    component_residuals = {order: [] for order in range(ASYMPTOTIC_ORDER + 1)}
    for speed in SPEEDS:
        moments = stationary_moments(speed, q0, pi, c, kappa, variance)
        for inverse_order in range(ASYMPTOTIC_ORDER + 1):
            powers = speed ** -np.arange(inverse_order + 1, dtype=float)
            approximation = np.einsum(
                "pn,n->p", scalar[:, :inverse_order + 1], powers
            )
            component_approximation = np.einsum(
                "pni,n->pi", coefficients[:, :inverse_order + 1], powers
            )
            residuals[inverse_order].append(
                np.max(np.abs(moments.sum(axis=1)[1:] - approximation[1:]))
            )
            component_residuals[inverse_order].append(
                np.max(np.abs(moments[1:] - component_approximation[1:]))
            )
    scalar_rates = np.asarray(
        [observed_order(residuals[order]) for order in range(ASYMPTOTIC_ORDER + 1)]
    )
    component_rates = np.asarray(
        [observed_order(component_residuals[order])
         for order in range(ASYMPTOTIC_ORDER + 1)]
    )
    assert np.min(scalar_rates - np.arange(1, ASYMPTOTIC_ORDER + 2)) > -0.06
    assert np.min(component_rates - np.arange(1, ASYMPTOTIC_ORDER + 2)) > -0.06

    print("all-order stationary switched-CIR moment certificate")
    print("stationary law:", np.array2string(pi, precision=10))
    print("semigroup discrepancy:", f"{semigroup_error:.3e}")
    print("Gamma reduction discrepancy:", f"{gamma_error:.3e}")
    print("leading moments p=1..6:", np.array2string(scalar[1:, 0], precision=10))
    print("first coefficients p=1..6:", np.array2string(scalar[1:, 1], precision=10))
    print("scalar residual orders n=0..3:", np.array2string(scalar_rates, precision=6))
    print("component residual orders n=0..3:", np.array2string(component_rates, precision=6))
    print("largest cubic scalar residual at m=128:", f"{residuals[3][-1]:.3e}")
    print("guaranteed Taylor radius through p=6:", f"{radius:.10f}")
    print("epsilon/radius at speed 8:", f"{1 / (8 * radius):.10f}")
    print("Taylor truncation orders:", " ".join(map(str, truncations)))
    print("maximum scalar Taylor errors:",
          " ".join(mp.nstr(value, 7) for value in convergence_errors))
    print("nearest nonremovable pole:", mp.nstr(nearest_pole, 12))
    print("nearest-pole row residue norm:", mp.nstr(residue_norm, 12))
    print("nearest-pole scalar residue:", mp.nstr(scalar_residue, 12))
    print(
        "eigenvector-residue relative error:",
        mp.nstr(residue_formula_error, 5),
    )
    print(
        "100th-coefficient residue relative error:",
        mp.nstr(residue_relative_error, 12),
    )


if __name__ == "__main__":
    main()
