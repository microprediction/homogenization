"""Certificate for all stationary moments of a switched CIR variance.

For the row-vector convention used in the paper, the invariant component
moments r[p, i] = E[v**p 1_{Y=i}] obey a triangular sequence of killed-chain
linear systems.  This script checks that recursion against one large
polynomial-semigroup exponential, verifies the constant-coefficient Gamma
special case, tests the explicit inverse-speed expansion through cubic order,
and checks convergence of the full Taylor series inside the computable pole
disk on a nonreversible three-state chain.  It also verifies a noncancellation
criterion under which that disk is the exact Taylor disk, and checks a
switching-speed-uniform factorial moment bound through order eighty.  The
latter proves exponential integrability, moment determinacy, and a uniform
zero-free disk on which the cumulant series has explicit Cauchy coefficient
and truncation bounds, without claiming that the inverse-speed expansion is
uniform in moment order.
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
MOMENT_BOUND_DEGREE = 80
BOUND_SPEEDS = (0.25, 1, 8, 128)
CUMULANT_TRUNCATION_ORDER = 30


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


def factorial_moment_bound(c, kappa, variance, degree=MOMENT_BOUND_DEGREE):
    """Uniform Gamma-majorant bound implied by the exact moment recursion."""
    c_star = float(np.max(c))
    kappa_star = float(np.min(kappa))
    variance_star = float(np.max(variance))
    bound = np.ones(degree + 1)
    for order in range(1, degree + 1):
        bound[order] = (
            bound[order - 1]
            * (c_star + (order - 1) * variance_star / 2)
            / kappa_star
        )
    return bound


def exponential_integrability_certificate(q0, pi, c, kappa, variance):
    """Check the speed-uniform factorial bound, including beyond Feller.

    Multiplying the exact pth stationary moment equation by the all-ones
    vector gives

        r_p kappa = r_(p-1) (c + (p-1) variance/2).

    Positivity therefore majorizes every scalar moment by Gamma moments with
    the worst statewise coefficients.  The check uses eighty moments at four
    switching speeds and repeats the test after violating every statewise
    Feller inequality.
    """
    bound = factorial_moment_bound(c, kappa, variance)
    ratios = []
    mgf_partials = []
    variance_star = float(np.max(variance))
    kappa_star = float(np.min(kappa))
    c_star = float(np.max(c))
    theta = variance_star / (2 * kappa_star)
    beta = 2 * c_star / variance_star
    eta = 0.8 / theta
    mgf_bound = (1 - eta * theta) ** (-beta)
    for speed in BOUND_SPEEDS:
        moments = stationary_moments(
            speed, q0, pi, c, kappa, variance, degree=MOMENT_BOUND_DEGREE
        ).sum(axis=1)
        ratios.append(float(np.max(moments[1:] / bound[1:])))
        mgf_partials.append(sum(
            eta ** order * moments[order] / math.factorial(order)
            for order in range(MOMENT_BOUND_DEGREE + 1)
        ))
    assert max(ratios) < 1
    assert max(mgf_partials) < mgf_bound

    # Moment finiteness is distinct from boundary inaccessibility.  This
    # positive example violates 2*c_i >= variance_i in every regime.
    nonfeller_c = np.array([0.010, 0.030, 0.020])
    assert np.max(2 * nonfeller_c - variance) < 0
    nonfeller_bound = factorial_moment_bound(nonfeller_c, kappa, variance)
    nonfeller_ratios = []
    for speed in BOUND_SPEEDS:
        moments = stationary_moments(
            speed, q0, pi, nonfeller_c, kappa, variance,
            degree=MOMENT_BOUND_DEGREE,
        ).sum(axis=1)
        nonfeller_ratios.append(
            float(np.max(moments[1:] / nonfeller_bound[1:]))
        )
    assert max(nonfeller_ratios) < 1
    return (
        theta,
        beta,
        eta,
        mgf_bound,
        max(ratios),
        max(mgf_partials),
        max(nonfeller_ratios),
    )


def uniform_cumulant_disk_certificate(q0, c, kappa, variance):
    """Certify the zero-free MGF disk and its cumulant-series bounds.

    The Gamma moment majorant gives, for complex z with theta*abs(z) < 1,

        abs(M_m(z) - 1) <= (1 - theta*abs(z))**(-beta) - 1.

    The right side is below one on the disk of radius
    theta**(-1) * (1 - 2**(-1/beta)), uniformly in the switching speed.
    Thus the analytic branch K_m=log(M_m), K_m(0)=0, exists there.  On a
    smaller circle of radius r its modulus is at most

        L_r = -log(2 - (1 - theta*r)**(-beta)).

    Cauchy's estimate then bounds the nth cumulant by n!*L_r/r**n and the
    order-N Taylor tail at eta by
    L_r*(eta/r)**(N+1)/(1-eta/r).  The high-precision check below verifies
    these inequalities at four switching speeds.  It also brackets the exact
    positive-real MGF by an 80-term moment sum plus the remaining Gamma-
    majorant tail, so the cumulant truncation is tested without exponentiating
    a truncated cumulant polynomial.
    """
    old_dps = mp.mp.dps
    mp.mp.dps = 140
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
        pi_mp = mp.lu_solve(stationary_system, stationary_rhs).T

        theta = max(variance_mp) / (2 * min(kappa_mp))
        beta = 2 * max(c_mp) / max(variance_mp)
        zero_free_radius = (1 - mp.power(2, -1 / beta)) / theta
        circle_radius = mp.mpf("0.8") * zero_free_radius
        circle_deviation = (1 - theta * circle_radius) ** (-beta) - 1
        log_bound = -mp.log(1 - circle_deviation)
        evaluation_point = circle_radius / 2
        truncation_order = CUMULANT_TRUNCATION_ORDER
        tail_bound = (
            log_bound
            * (evaluation_point / circle_radius) ** (truncation_order + 1)
            / (1 - evaluation_point / circle_radius)
        )

        largest_coefficient_ratio = mp.mpf("0")
        largest_actual_tail = mp.mpf("0")
        largest_mgf_bracket = mp.mpf("0")
        for speed in BOUND_SPEEDS:
            rows = [pi_mp]
            speed_mp = mp.mpf(str(speed))
            for order in range(1, MOMENT_BOUND_DEGREE + 1):
                forcing = mp.matrix([
                    rows[-1][index]
                    * (order * c_mp[index]
                       + mp.mpf(order * (order - 1))
                       * variance_mp[index] / 2)
                    for index in range(size)
                ])
                killed = (
                    mp.diag([order * value for value in kappa_mp])
                    - speed_mp * q_mp
                )
                rows.append(mp.lu_solve(killed.T, forcing).T)
            moments = [sum(row) for row in rows]

            cumulants = [mp.mpf("0")] * (MOMENT_BOUND_DEGREE + 1)
            for order in range(1, MOMENT_BOUND_DEGREE + 1):
                cumulants[order] = moments[order] - sum(
                    mp.binomial(order - 1, index - 1)
                    * cumulants[index]
                    * moments[order - index]
                    for index in range(1, order)
                )
                coefficient_bound = (
                    mp.factorial(order) * log_bound / circle_radius ** order
                )
                ratio = abs(cumulants[order]) / coefficient_bound
                largest_coefficient_ratio = max(
                    largest_coefficient_ratio, ratio
                )
                assert ratio < 1

            cumulant_partial = sum(
                cumulants[order] * evaluation_point ** order
                / mp.factorial(order)
                for order in range(1, truncation_order + 1)
            )
            moment_partial = sum(
                moments[order] * evaluation_point ** order
                / mp.factorial(order)
                for order in range(MOMENT_BOUND_DEGREE + 1)
            )
            majorant_partial = sum(
                mp.rf(beta, order) * (theta * evaluation_point) ** order
                / mp.factorial(order)
                for order in range(MOMENT_BOUND_DEGREE + 1)
            )
            majorant_full = (1 - theta * evaluation_point) ** (-beta)
            lower_log_mgf = mp.log(moment_partial)
            upper_log_mgf = mp.log(
                moment_partial + majorant_full - majorant_partial
            )
            assert lower_log_mgf >= cumulant_partial - tail_bound
            assert upper_log_mgf <= cumulant_partial + tail_bound
            largest_actual_tail = max(
                largest_actual_tail,
                abs(cumulant_partial - lower_log_mgf),
                abs(cumulant_partial - upper_log_mgf),
            )
            largest_mgf_bracket = max(
                largest_mgf_bracket, upper_log_mgf - lower_log_mgf
            )

        return (
            zero_free_radius,
            circle_radius,
            circle_deviation,
            log_bound,
            evaluation_point,
            tail_bound,
            largest_coefficient_ratio,
            largest_actual_tail,
            largest_mgf_bracket,
        )
    finally:
        mp.mp.dps = old_dps


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

    (
        majorant_scale,
        majorant_shape,
        test_eta,
        exponential_bound,
        largest_moment_ratio,
        largest_mgf_partial,
        largest_nonfeller_ratio,
    ) = exponential_integrability_certificate(q0, pi, c, kappa, variance)
    (
        cumulant_radius,
        cumulant_circle,
        cumulant_deviation,
        cumulant_log_bound,
        cumulant_evaluation_point,
        cumulant_tail_bound,
        largest_cumulant_bound_ratio,
        largest_actual_cumulant_tail,
        largest_mgf_log_bracket,
    ) = uniform_cumulant_disk_certificate(q0, c, kappa, variance)

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
    print(
        "Gamma majorant scale/shape:",
        f"{majorant_scale:.12f}/{majorant_shape:.12f}",
    )
    print(
        "largest exact-moment / majorant ratio through p=80:",
        f"{largest_moment_ratio:.12f}",
    )
    print(
        "largest non-Feller moment / majorant ratio through p=80:",
        f"{largest_nonfeller_ratio:.12f}",
    )
    print(
        "80-term exponential partial / rigorous bound at eta="
        f"{test_eta:.6f}: {largest_mgf_partial:.12f}/{exponential_bound:.12f}",
    )
    print("uniform zero-free cumulant radius:", mp.nstr(cumulant_radius, 13))
    print(
        "Cauchy circle radius/deviation/log bound:",
        "/".join(mp.nstr(value, 13) for value in (
            cumulant_circle, cumulant_deviation, cumulant_log_bound
        )),
    )
    print(
        "largest cumulant / Cauchy coefficient bound through p=80:",
        mp.nstr(largest_cumulant_bound_ratio, 13),
    )
    print(
        f"order-{CUMULANT_TRUNCATION_ORDER} tail bound at eta="
        f"{mp.nstr(cumulant_evaluation_point, 13)}:",
        mp.nstr(cumulant_tail_bound, 13),
    )
    print(
        "largest certified actual tail / MGF-log bracket width:",
        f"{mp.nstr(largest_actual_cumulant_tail, 7)}/"
        f"{mp.nstr(largest_mgf_log_bracket, 7)}",
    )
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
