"""Certificate for simultaneous switching of CIR drift and vol-of-vol.

The correct regime features are c_y=kappa_y theta_y, kappa_y, and
a_y=xi_y^2.  Under regime-dependent leverage, eta_y=rho_y*xi_y is the
additional feature because it is the coefficient of the cross derivative.
Polynomial closure gives an exact finite-dimensional check of the general
Green--Kubo rule.
"""
import math
import os
import sys

import numpy as np
from scipy.linalg import eig, expm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "general"))
from effective_generator import (
    effective_generator,
    full_generator,
    gk,
    group_inverse,
    stationary,
)


SPEEDS = (8, 16, 32, 64)


def polynomial_operators(degree=5):
    """Matrices for d_v, -v d_v, and (v/2)d_vv in (1,v,...)."""
    derivative = np.zeros((degree + 1, degree + 1))
    multiply_v = np.zeros_like(derivative)
    for power in range(1, degree + 1):
        derivative[power - 1, power] = power
        multiply_v[power, power - 1] = 1.0
    a_c = derivative
    a_kappa = -multiply_v @ derivative
    a_variance = 0.5 * multiply_v @ derivative @ derivative
    return a_c, a_kappa, a_variance, multiply_v


def integrated_variance_operators(degree=4):
    """Polynomial operators on monomials v^i z^j, i+j <= degree, with dz=v dt."""
    basis = [(i, total - i) for total in range(degree + 1) for i in range(total + 1)]
    index = {powers: position for position, powers in enumerate(basis)}
    size = len(basis)

    def operator(action):
        matrix = np.zeros((size, size))
        for column, powers in enumerate(basis):
            for coefficient, target in action(*powers):
                if coefficient and target in index:
                    matrix[index[target], column] += coefficient
        return matrix

    d_v = operator(lambda i, j: [(i, (i - 1, j))] if i else [])
    minus_v_d_v = operator(lambda i, j: [(-i, (i, j))] if i else [])
    v_d_vv = operator(lambda i, j: [(i * (i - 1), (i - 1, j))] if i >= 2 else [])
    v_d_z = operator(lambda i, j: [(j, (i + 1, j - 1))] if j else [])
    return basis, d_v, minus_v_d_v, v_d_vv, v_d_z


def leveraged_operators(degree=4):
    """Polynomial operators on monomials M^i v^j z^k for Heston leverage."""
    basis = [
        (i, j, total - i - j)
        for total in range(degree + 1)
        for i in range(total + 1)
        for j in range(total - i + 1)
    ]
    index = {powers: position for position, powers in enumerate(basis)}
    size = len(basis)

    def operator(action):
        matrix = np.zeros((size, size))
        for column, powers in enumerate(basis):
            for coefficient, target in action(*powers):
                if coefficient and target in index:
                    matrix[index[target], column] += coefficient
        return matrix

    d_v = operator(lambda i, j, k: [(j, (i, j - 1, k))] if j else [])
    minus_v_d_v = operator(lambda i, j, k: [(-j, (i, j, k))] if j else [])
    v_d_vv = operator(
        lambda i, j, k: [(j * (j - 1), (i, j - 1, k))] if j >= 2 else []
    )
    v_d_mm = operator(
        lambda i, j, k: [(i * (i - 1), (i - 2, j + 1, k))] if i >= 2 else []
    )
    v_d_mdv = operator(
        lambda i, j, k: [(i * j, (i - 1, j, k))] if i and j else []
    )
    v_d_z = operator(
        lambda i, j, k: [(k, (i, j + 1, k - 1))] if k else []
    )
    return basis, d_v, minus_v_d_v, v_d_vv, v_d_mm, v_d_mdv, v_d_z


def first_order(lbar, correction, maturity, payoff):
    """Duhamel term for a first-order generator perturbation."""
    size = len(payoff)
    block = np.zeros((2 * size, 2 * size))
    block[:size, :size] = lbar
    block[size:, size:] = lbar
    block[:size, size:] = correction
    semigroup = expm(maturity * block)
    return semigroup[:size, :size] @ payoff + semigroup[:size, size:] @ payoff


def reversed_generator(q, pi):
    return np.diag(1.0 / pi) @ q.T @ np.diag(pi)


def rate(errors):
    return math.log(errors[-2] / errors[-1], 2)


def cumulants(raw):
    """First four cumulants from raw moments E[Z],...,E[Z^4]."""
    m1, m2, m3, m4 = raw
    return np.array(
        [
            m1,
            m2 - m1**2,
            m3 - 3 * m2 * m1 + 2 * m1**3,
            m4 - 4 * m3 * m1 - 3 * m2**2 + 12 * m2 * m1**2 - 6 * m1**4,
        ]
    )


def cumulants_any_order(raw):
    """Cumulants from raw moments E[Z],...,E[Z^K]."""
    raw = np.asarray(raw, dtype=float)
    values = []
    for order, moment in enumerate(raw, start=1):
        value = moment
        for index in range(1, order):
            value -= (
                math.comb(order - 1, index - 1)
                * values[index - 1]
                * raw[order - index - 1]
            )
        values.append(value)
    return np.asarray(values)


def independent_logreturn_cumulants(variance_cumulants):
    """Map cumulants of V to those of X=-V/2+sqrt(V)Z.

    The identity is a finite-jet statement and therefore does not require an
    MGF.  If ``variance_cumulants[j-1]`` is kappa_j(V), the kth return
    cumulant is

      sum_{j=ceil(k/2)}^k (-1)^k k!/(2^j j!) binom(j,k-j) kappa_j(V).
    """
    variance_cumulants = np.asarray(variance_cumulants, dtype=float)
    result = np.zeros_like(variance_cumulants)
    for order in range(1, len(result) + 1):
        for index in range((order + 1) // 2, order + 1):
            coefficient = (
                (-1) ** order
                * math.factorial(order)
                * math.comb(index, order - index)
                / (2**index * math.factorial(index))
            )
            result[order - 1] += (
                coefficient * variance_cumulants[index - 1]
            )
    return result


def cumulant_correction(raw, correction):
    """Directional derivative of the first four cumulants in the moment correction."""
    m1, m2, m3, m4 = raw
    d1, d2, d3, d4 = correction
    return np.array(
        [
            d1,
            d2 - 2 * m1 * d1,
            d3 - 3 * (d2 * m1 + m2 * d1) + 6 * m1**2 * d1,
            d4
            - 4 * (d3 * m1 + m3 * d1)
            - 6 * m2 * d2
            + 12 * (d2 * m1**2 + 2 * m2 * m1 * d1)
            - 24 * m1**3 * d1,
        ]
    )


def exact_integrated_mean(speed, maturity, v0, q0, pi, c, kappa):
    """Exact E[int_0^T v_s ds] from the closed regime-resolved first moment."""
    drift = speed * q0 - np.diag(kappa)
    stationary_components = np.linalg.solve(drift.T, -(pi * c))
    initial_components = pi * v0
    transient_integral = (initial_components - stationary_components) @ np.linalg.solve(
        drift, expm(maturity * drift) - np.eye(len(pi))
    )
    return (
        maturity * stationary_components.sum() + transient_integral.sum(),
        stationary_components.sum(),
    )


def stationary_cir_moments(speed, q0, pi, c, kappa, variance):
    """Regime-resolved stationary first and second moments of switched CIR."""
    first_drift = speed * q0 - np.diag(kappa)
    first = np.linalg.solve(first_drift.T, -(pi * c))
    second_drift = speed * q0 - 2 * np.diag(kappa)
    second = np.linalg.solve(
        second_drift.T, -((2 * c + variance) * first)
    )
    return first, second


def stationary_cir_third_moments(speed, q0, pi, c, kappa, variance):
    """Regime-resolved invariant moments through order three."""
    first, second = stationary_cir_moments(
        speed, q0, pi, c, kappa, variance
    )
    third_drift = speed * q0 - 3 * np.diag(kappa)
    third = np.linalg.solve(
        third_drift.T, -(3 * (c + variance) * second)
    )
    return first, second, third


def stationary_cir_fourth_moments(speed, q0, pi, c, kappa, variance):
    """Regime-resolved invariant moments through order four."""
    first, second, third = stationary_cir_third_moments(
        speed, q0, pi, c, kappa, variance
    )
    fourth_drift = speed * q0 - 4 * np.diag(kappa)
    fourth = np.linalg.solve(
        fourth_drift.T, -((4 * c + 6 * variance) * third)
    )
    return first, second, third, fourth


def integrated_variance_rate(speed, q0, pi, c, kappa, variance):
    """Exact long-run variance rate of int_0^T v_s ds under stationarity.

    The centered Poisson equation -L phi=v-mu has the affine solution
    phi_i(v)=alpha_i*v+beta_i.  This routine returns
    2 E[(v-mu) phi_Y(v)] together with the stationary moments and affine
    coefficients used in that identity.
    """
    first, second = stationary_cir_moments(
        speed, q0, pi, c, kappa, variance
    )
    mean = first.sum()
    drift = speed * q0 - np.diag(kappa)
    alpha = np.linalg.solve(drift, -np.ones(len(pi)))
    forcing = mean * np.ones(len(pi)) - c * alpha
    assert abs(pi @ forcing) < 2e-13
    beta = group_inverse(q0) @ forcing / speed
    assert np.linalg.norm(speed * q0 @ beta - forcing) < 2e-12
    assert abs(pi @ beta) < 2e-13
    asymptotic_variance = 2 * (
        alpha @ (second - mean * first)
        + beta @ (first - mean * pi)
    )
    return asymptotic_variance, mean, first, second, alpha, beta


def integrated_variance_third_rate(speed, q0, pi, c, kappa, variance):
    """Exact long-run third-cumulant rate of the integrated variance.

    With g=v-mu, first solve -L phi=g and center phi under the full
    invariant law.  The second Poisson equation

        -L psi = g phi - E[g phi]

    then gives lim_T kappa_3(int_0^T v_s ds)/T = 6 E[g psi].  Both
    solutions are polynomial, so the formula reduces to finite linear
    systems even when all CIR coefficients switch.
    """
    first, second, third = stationary_cir_third_moments(
        speed, q0, pi, c, kappa, variance
    )
    mean = first.sum()
    first_drift = speed * q0 - np.diag(kappa)
    second_drift = speed * q0 - 2 * np.diag(kappa)

    alpha = np.linalg.solve(first_drift, -np.ones(len(pi)))
    forcing = mean * np.ones(len(pi)) - c * alpha
    beta = group_inverse(q0) @ forcing / speed
    assert np.linalg.norm(speed * q0 @ beta - forcing) < 2e-12

    # pi@beta=0 is a convenient chain normalization, but the Poisson
    # solution must be centered under the joint invariant law, not under pi.
    phi_mean = first @ alpha + pi @ beta
    centered_beta = beta - phi_mean * np.ones(len(pi))
    half_variance_rate = (
        alpha @ (second - mean * first)
        + centered_beta @ (first - mean * pi)
    )

    quadratic = np.linalg.solve(second_drift, -alpha)
    linear = np.linalg.solve(
        first_drift,
        -(centered_beta - mean * alpha) - (2 * c + variance) * quadratic,
    )
    constant_forcing = (
        mean * centered_beta
        + half_variance_rate * np.ones(len(pi))
        - c * linear
    )
    assert abs(pi @ constant_forcing) < 3e-13
    constant = group_inverse(q0) @ constant_forcing / speed
    assert np.linalg.norm(
        speed * q0 @ constant - constant_forcing
    ) < 3e-12

    third_rate = 6 * (
        quadratic @ (third - mean * second)
        + linear @ (second - mean * first)
        + constant @ (first - mean * pi)
    )
    return (
        third_rate,
        mean,
        first,
        second,
        third,
        alpha,
        phi_mean,
        centered_beta,
        quadratic,
        linear,
        constant,
    )


def _pointwise_polynomial_product(left, right, states, degree):
    """Multiply regimewise coefficient vectors, truncated at ``degree``."""
    width = degree + 1
    left_blocks = left.reshape(states, width)
    right_blocks = right.reshape(states, width)
    product = np.zeros_like(left_blocks)
    for state in range(states):
        product[state] = np.convolve(
            left_blocks[state], right_blocks[state]
        )[:width]
    return product.ravel()


def _centered_polynomial_poisson(generator, invariant, constant, forcing):
    """Solve -L u=forcing with invariant mean zero in coefficient space."""
    size = len(forcing)
    bordered = np.zeros((size + 1, size + 1))
    bordered[:size, :size] = -generator
    bordered[:size, size] = constant
    bordered[size, :size] = invariant
    solution = np.linalg.solve(bordered, np.r_[forcing, 0.0])
    assert abs(solution[size]) < 2e-12
    assert np.linalg.norm(-generator @ solution[:size] - forcing) < 2e-11
    assert abs(invariant @ solution[:size]) < 2e-12
    return solution[:size]


def _stationary_polynomial_context(speed, q0, pi, c, kappa, variance, degree):
    """Coefficient generator, invariant functional, constant, and centered v."""
    width = degree + 1
    states = len(pi)
    a_c, a_kappa, a_variance, _ = polynomial_operators(degree)
    averaged = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
    )
    generator = full_generator(
        speed * q0,
        averaged,
        [a_c, a_kappa, a_variance],
        [c, kappa, variance],
    )
    moments = [pi]
    for power in range(1, degree + 1):
        drift = speed * q0 - power * np.diag(kappa)
        lower = (
            power * (c + 0.5 * (power - 1) * variance) * moments[-1]
        )
        moments.append(np.linalg.solve(drift.T, -lower))
    invariant = np.zeros(states * width)
    for power, values in enumerate(moments):
        invariant[power::width] = values
    constant = np.zeros(states * width)
    constant[::width] = 1.0
    mean = moments[1].sum()
    g_block = np.zeros(width)
    g_block[:2] = (-mean, 1.0)
    g = np.tile(g_block, states)
    return generator, invariant, constant, g


def integrated_variance_third_intercept(speed, q0, pi, c, kappa, variance):
    """Exact bounded intercept in the stationary third cumulant.

    Write R for the centered Poisson inverse and g=v-E[v].  With

        phi=R g,  psi=R(g phi-E[g phi]),
        eta=R phi, zeta=R psi,
        omega=R(g eta-E[g eta]),

    the third cumulant equals tau*T+chi+an exponentially decaying remainder,
    where tau=6 E[g psi] and chi=-6 E[g(zeta+omega)].  Degree-three
    polynomial closure evaluates every expectation and Poisson solve exactly.
    """
    degree = 3
    states = len(pi)
    generator, invariant, constant, g = _stationary_polynomial_context(
        speed, q0, pi, c, kappa, variance, degree
    )

    def product(left, right):
        return _pointwise_polynomial_product(
            left, right, states, degree
        )

    def center(function):
        return function - (invariant @ function) * constant

    def poisson(forcing):
        return _centered_polynomial_poisson(
            generator, invariant, constant, forcing
        )

    phi = poisson(g)
    psi = poisson(center(product(g, phi)))
    eta = poisson(phi)
    zeta = poisson(psi)
    omega = poisson(center(product(g, eta)))
    third_rate = 6 * (invariant @ product(g, psi))
    intercept = -6 * (invariant @ product(g, zeta + omega))
    return intercept, third_rate


def integrated_variance_fourth_rate(speed, q0, pi, c, kappa, variance):
    """Exact long-run fourth-cumulant rate of the integrated variance.

    If R is the centered Poisson inverse and g=v-E[v], define

        phi=R g,
        c2=E[g phi],
        psi=R(g phi-c2),
        c3=E[g psi],
        xi=R(g psi-c2 phi-c3).

    Differentiating the normalized Feynman--Kac eigenpair through fourth
    order gives lim_T kappa_4(int_0^T v_s ds)/T=24 E[g xi].  The subtraction
    c2*phi is the lower-cumulant partition term and cannot be dropped.
    """
    degree = 4
    states = len(pi)
    generator, invariant, constant, g = _stationary_polynomial_context(
        speed, q0, pi, c, kappa, variance, degree
    )

    def product(left, right):
        return _pointwise_polynomial_product(
            left, right, states, degree
        )

    def center(function):
        return function - (invariant @ function) * constant

    def poisson(forcing):
        return _centered_polynomial_poisson(
            generator, invariant, constant, forcing
        )

    phi = poisson(g)
    c2 = invariant @ product(g, phi)
    psi = poisson(center(product(g, phi)))
    c3 = invariant @ product(g, psi)
    forcing = product(g, psi) - c2 * phi - c3 * constant
    assert abs(invariant @ forcing) < 2e-12
    xi = poisson(forcing)
    fourth_rate = 24 * (invariant @ product(g, xi))

    # This intentionally incorrect variant is returned for the certificate:
    # centering g*psi alone omits the lower-cumulant partition c2*phi.
    naive_xi = poisson(center(product(g, psi)))
    naive_rate = 24 * (invariant @ product(g, naive_xi))
    assert abs(invariant @ phi) < 2e-12
    assert abs(invariant @ psi) < 2e-12
    assert abs(invariant @ xi) < 2e-12
    return fourth_rate, naive_rate


def integrated_variance_cumulant_rates(
    max_order, speed, q0, pi, c, kappa, variance
):
    """Exact long-run cumulant rates through any fixed polynomial order.

    For centered g=v-E[v], differentiate the normalized eigenpair

        (L + theta*g) h(theta) = lambda(theta) h(theta), E[h(theta)]=1.

    If h_k and gamma_k are the kth derivatives at zero, then

        gamma_k = k E[g h_{k-1}],
        -L h_k = k g h_{k-1}
                   - sum_{j=1}^k binom(k,j) gamma_j h_{k-j}.

    Polynomial closure makes this an exact finite sequence of linear solves
    for each fixed ``max_order``.  No uniformity in the order is asserted.
    """
    if max_order < 1:
        raise ValueError("max_order must be positive")
    degree = max_order
    states = len(pi)
    generator, invariant, constant, g = _stationary_polynomial_context(
        speed, q0, pi, c, kappa, variance, degree
    )

    def product(left, right):
        return _pointwise_polynomial_product(left, right, states, degree)

    def poisson(forcing):
        return _centered_polynomial_poisson(
            generator, invariant, constant, forcing
        )

    derivatives = [constant]
    rates = np.zeros(max_order + 1)
    for order in range(1, max_order + 1):
        rates[order] = order * (invariant @ product(g, derivatives[order - 1]))
        forcing = order * product(g, derivatives[order - 1])
        for index in range(1, order + 1):
            forcing -= (
                math.comb(order, index)
                * rates[index]
                * derivatives[order - index]
            )
        assert abs(invariant @ forcing) < 2e-11
        derivatives.append(poisson(forcing))
    return rates[1:], derivatives


def cumulant_rate_first_corrections(
    max_order, q0, pi, c, kappa, variance, radius=0.25, samples=256
):
    """First inverse-speed coefficients of the long-run cumulant rates.

    On polynomials through degree max_order, eliminate the mean-zero regime
    block of the tilted generator.  Its slow Schur complement is

        Lbar + theta*M_v + epsilon*D + O(epsilon**2),

    where epsilon=1/m and D=sum K_rs A_r A_s is the usual Green--Kubo
    effective-generator correction.  If Lambda(theta, epsilon) is the
    eigenvalue continuing zero, the returned values are

        eta_k = d_theta^k d_epsilon Lambda(0, 0),  1 <= k <= max_order.

    A Cauchy integral extracts the theta derivatives of the simple averaged
    eigenvalue's first-order perturbation l_theta D h_theta.  The radius must
    remain inside a neighborhood where that eigenvalue is simple.
    """
    if max_order < 1:
        raise ValueError("max_order must be positive")
    if samples <= 2 * max_order:
        raise ValueError("samples must exceed twice max_order")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if not np.allclose(stationary(q0), pi, atol=2e-13):
        raise ValueError("pi must be stationary for q0")

    a_c, a_kappa, a_variance, multiply_v = polynomial_operators(max_order)
    operators = [a_c, a_kappa, a_variance]
    features = [c, kappa, variance]
    averaged = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
    )
    correction = effective_generator(
        np.zeros_like(averaged), operators, gk(q0, features)
    )

    angles = 2 * np.pi * np.arange(samples) / samples
    boundary_values = np.zeros(samples, dtype=complex)
    for index, angle in enumerate(angles):
        theta = radius * np.exp(1j * angle)
        tilted = averaged.astype(complex) + theta * multiply_v
        eigenvalues, right_vectors = eig(tilted)
        branch = np.argmin(np.abs(eigenvalues))
        eigenvalue = eigenvalues[branch]
        right = right_vectors[:, branch]
        left_values, left_vectors = eig(tilted.T)
        left_branch = np.argmin(np.abs(left_values - eigenvalue))
        left = left_vectors[:, left_branch]
        pairing = left @ right
        assert abs(pairing) > 1e-8
        boundary_values[index] = left @ correction @ right / pairing

    derivatives = np.zeros(max_order + 1)
    imaginary_errors = []
    for order in range(max_order + 1):
        coefficient = np.mean(
            boundary_values * np.exp(-1j * order * angles)
        ) / radius**order
        derivative = math.factorial(order) * coefficient
        derivatives[order] = derivative.real
        imaginary_errors.append(abs(derivative.imag))
    assert max(imaginary_errors) < 2e-10
    return derivatives[1:], correction


def cumulant_rate_second_corrections(
    max_order, q0, pi, c, kappa, variance, radius=0.25, samples=256
):
    """Second inverse-speed coefficients of the long-run cumulant rates.

    Put epsilon=1/m and split the regime coordinate into its stationary
    subspace P=1*pi and the centered subspace N=I-P.  If B_theta is the
    order-one tilted polynomial generator and S=Q_0^# on N, Schur
    elimination gives the nonlinear slow pencil

        A_theta + epsilon*D_theta + epsilon**2*E_theta(lambda)
        + O(epsilon**3),

    where

        D_theta = -E B_theta S B_theta J,
        E_theta(lambda) = E B_theta S(B_theta-lambda)S B_theta J.

    Here J injects a regime-free polynomial and E averages it with pi.
    For the simple eigenpair A_theta h=lambda_bar h, l h=1, let

        eta=l D_theta h,
        (A_theta-lambda_bar) h_1=-(D_theta-eta)h,  l h_1=0.

    The second eigenvalue coefficient is then

        zeta(theta)=l E_theta(lambda_bar)h+l D_theta h_1.

    Cauchy extraction returns zeta_k=d_theta^k zeta(0), so that, for
    every fixed k,

        gamma_k(m)=gamma_bar_k+eta_k/m+zeta_k/m**2+O_k(m**-3).

    The eigenvector-response term l D_theta h_1 is essential.  No
    uniformity in cumulant order is asserted.
    """
    if max_order < 1:
        raise ValueError("max_order must be positive")
    if samples <= 2 * max_order:
        raise ValueError("samples must exceed twice max_order")
    if radius <= 0:
        raise ValueError("radius must be positive")
    if not np.allclose(stationary(q0), pi, atol=2e-13):
        raise ValueError("pi must be stationary for q0")

    a_c, a_kappa, a_variance, multiply_v = polynomial_operators(max_order)
    operators = [a_c, a_kappa, a_variance]
    features = [c, kappa, variance]
    averaged = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
    )
    states = len(pi)
    width = max_order + 1
    inject = np.kron(np.ones((states, 1)), np.eye(width))
    average = np.kron(pi.reshape(1, states), np.eye(width))
    fast_inverse = np.kron(group_inverse(q0), np.eye(width))
    identity = np.eye(states * width)
    green_kubo = effective_generator(
        np.zeros_like(averaged), operators, gk(q0, features)
    )

    angles = 2 * np.pi * np.arange(samples) / samples
    first_boundary = np.zeros(samples, dtype=complex)
    second_boundary = np.zeros(samples, dtype=complex)
    missing_response_boundary = np.zeros(samples, dtype=complex)
    for index, angle in enumerate(angles):
        theta = radius * np.exp(1j * angle)
        slow = averaged.astype(complex) + theta * multiply_v
        full_order_one = np.kron(np.eye(states), slow)
        for feature, operator in zip(features, operators):
            centered = feature - pi @ feature
            full_order_one += np.kron(np.diag(centered), operator)

        schur_first_operator = -(
            average @ full_order_one @ fast_inverse
            @ full_order_one @ inject
        )
        assert np.max(np.abs(schur_first_operator - green_kubo)) < 3e-13
        # Use the algebraically identical Green--Kubo assembly below.  At
        # high theta-derivative orders, Cauchy extraction would otherwise
        # magnify roundoff from repeatedly forming the large Schur product.
        first_operator = green_kubo.astype(complex)
        eigenvalues, right_vectors = eig(slow)
        branch = np.argmin(np.abs(eigenvalues))
        eigenvalue = eigenvalues[branch]
        right = right_vectors[:, branch]
        left_values, left_vectors = eig(slow.T)
        left_branch = np.argmin(np.abs(left_values - eigenvalue))
        left = left_vectors[:, left_branch]
        pairing = left @ right
        assert abs(pairing) > 1e-8
        left /= pairing

        first_value = left @ first_operator @ right
        second_operator = (
            average @ full_order_one @ fast_inverse
            @ (full_order_one - eigenvalue * identity)
            @ fast_inverse @ full_order_one @ inject
        )
        bordered = np.zeros((width + 1, width + 1), dtype=complex)
        bordered[:width, :width] = slow - eigenvalue * np.eye(width)
        bordered[:width, width] = right
        bordered[width, :width] = left
        rhs = np.r_[-(first_operator - first_value * np.eye(width)) @ right, 0.0]
        solution = np.linalg.solve(bordered, rhs)
        right_first = solution[:width]
        assert abs(left @ right_first) < 2e-10
        assert abs(solution[width]) < 2e-9
        first_boundary[index] = first_value
        missing_response_boundary[index] = left @ second_operator @ right
        second_boundary[index] = (
            missing_response_boundary[index]
            + left @ first_operator @ right_first
        )

    first_derivatives = np.zeros(max_order + 1)
    second_derivatives = np.zeros(max_order + 1)
    missing_response_derivatives = np.zeros(max_order + 1)
    imaginary_errors = []
    for order in range(max_order + 1):
        phase = np.exp(-1j * order * angles) / radius**order
        factor = math.factorial(order)
        first_derivative = factor * np.mean(first_boundary * phase)
        second_derivative = factor * np.mean(second_boundary * phase)
        missing_derivative = factor * np.mean(
            missing_response_boundary * phase
        )
        first_derivatives[order] = first_derivative.real
        second_derivatives[order] = second_derivative.real
        missing_response_derivatives[order] = missing_derivative.real
        imaginary_errors.extend(
            [
                abs(first_derivative.imag),
                abs(second_derivative.imag),
                abs(missing_derivative.imag),
            ]
        )
    assert max(imaginary_errors) < 2e-9
    # Reuse the dedicated first-order extractor so the high-order Cauchy
    # coefficients are bit-for-bit consistent with the preceding theorem.
    # The Schur identity itself was checked at every boundary point above.
    independent_first, _ = cumulant_rate_first_corrections(
        max_order, q0, pi, c, kappa, variance, radius, samples
    )
    first_derivatives[1:] = independent_first
    return (
        first_derivatives[1:],
        second_derivatives[1:],
        missing_response_derivatives[1:],
    )


def integrated_variance_boundary_constants(
    max_order, speed, q0, pi, c, kappa, variance,
    initial_regime=None, initial_variance=None,
    initial_moment_components=None,
):
    """Boundary cumulants from right/left derivative recursions.

    ``initial_moment_components[i, j]`` may supply
    E[1_{Y_0=i} v_0^j] directly.  This permits initial laws with the required
    finite moments but no positive moment-generating function.
    """
    states = len(pi)
    degree = max_order
    generator, invariant, constant, g = _stationary_polynomial_context(
        speed, q0, pi, c, kappa, variance, degree
    )
    rates, right = integrated_variance_cumulant_rates(
        max_order, speed, q0, pi, c, kappa, variance
    )
    size = len(constant)
    multiply_g = np.zeros((size, size))
    for column in range(size):
        basis_vector = np.zeros(size)
        basis_vector[column] = 1.0
        multiply_g[:, column] = _pointwise_polynomial_product(
            g, basis_vector, states, degree
        )

    left = [invariant]
    for order in range(1, max_order + 1):
        forcing = -order * (left[order - 1] @ multiply_g)
        for index in range(1, order + 1):
            forcing += (
                math.comb(order, index)
                * rates[index - 1]
                * left[order - index]
            )
        normalization = -sum(
            math.comb(order, index) * (left[index] @ right[order - index])
            for index in range(order)
        )
        bordered = np.zeros((size + 1, size + 1))
        bordered[:size, :size] = generator.T
        bordered[:size, size] = invariant
        bordered[size, :size] = constant
        solution = np.linalg.solve(
            bordered, np.r_[forcing, normalization]
        )
        assert abs(solution[size]) < 3e-10
        assert np.linalg.norm(
            solution[:size] @ generator - forcing
        ) < 3e-9
        left.append(solution[:size])

    if initial_moment_components is not None:
        if initial_regime is not None or initial_variance is not None:
            raise ValueError(
                "moment components and point-start arguments are exclusive"
            )
        components = np.asarray(initial_moment_components, dtype=float)
        if components.shape != (states, degree + 1):
            raise ValueError(
                "initial_moment_components must have shape "
                f"({states}, {degree + 1})"
            )
        initial = components.ravel()
        if abs(np.sum(components[:, 0]) - 1.0) > 2e-12:
            raise ValueError("zeroth initial moment must have total mass one")
    elif initial_regime is None:
        initial = invariant
    else:
        if initial_variance is None:
            raise ValueError("initial_variance is required for a point start")
        initial = np.zeros(size)
        width = degree + 1
        block = initial_regime * width
        initial[block:block + width] = initial_variance ** np.arange(width)

    amplitude_derivatives = np.ones(max_order + 1)
    amplitude_derivatives[0] = 1.0
    for order in range(1, max_order + 1):
        amplitude_derivatives[order] = sum(
            math.comb(order, index)
            * (initial @ right[index])
            * (left[order - index] @ constant)
            for index in range(order + 1)
        )
    boundary = np.zeros(max_order + 1)
    for order in range(1, max_order + 1):
        boundary[order] = amplitude_derivatives[order]
        for index in range(1, order):
            boundary[order] -= (
                math.comb(order - 1, index - 1)
                * boundary[index]
                * amplitude_derivatives[order - index]
            )
    return boundary[1:], amplitude_derivatives[1:], right, left


def centered_integrated_cumulants(
    max_order, maturity, speed, q0, pi, c, kappa, variance,
    initial_regime=None, initial_variance=None,
    initial_moment_components=None,
):
    """Exact cumulants of int_0^T (v_s-E[v]) ds by polynomial closure.

    With no initial state specified, the joint process starts in stationarity.
    Otherwise ``initial_regime`` and ``initial_variance`` give a point start,
    or ``initial_moment_components[i, j]`` supplies
    E[1_{Y_0=i} v_0^j].
    """
    basis, a_c, a_kappa, v_d_vv, v_d_z = integrated_variance_operators(
        max_order
    )
    index = {powers: position for position, powers in enumerate(basis)}
    d_z = np.zeros((len(basis), len(basis)))
    for column, (v_power, z_power) in enumerate(basis):
        if z_power:
            d_z[index[(v_power, z_power - 1)], column] = z_power

    states = len(pi)
    _, invariant, _, _ = _stationary_polynomial_context(
        speed, q0, pi, c, kappa, variance, max_order
    )
    width = max_order + 1
    mean = sum(
        invariant[state * width + 1] for state in range(states)
    )
    a_variance = 0.5 * v_d_vv
    averaged = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
        + v_d_z
        - mean * d_z
    )
    generator = full_generator(
        speed * q0,
        averaged,
        [a_c, a_kappa, a_variance],
        [c, kappa, variance],
    )

    initial = np.zeros(states * len(basis))
    if initial_moment_components is not None:
        if initial_regime is not None or initial_variance is not None:
            raise ValueError(
                "moment components and point-start arguments are exclusive"
            )
        components = np.asarray(initial_moment_components, dtype=float)
        if components.shape != (states, max_order + 1):
            raise ValueError(
                "initial_moment_components must have shape "
                f"({states}, {max_order + 1})"
            )
        if abs(np.sum(components[:, 0]) - 1.0) > 2e-12:
            raise ValueError("zeroth initial moment must have total mass one")
        for state in range(states):
            block = state * len(basis)
            for position, (v_power, z_power) in enumerate(basis):
                if z_power == 0:
                    initial[block + position] = components[state, v_power]
    elif initial_regime is None:
        for state in range(states):
            block = state * len(basis)
            for position, (v_power, z_power) in enumerate(basis):
                if z_power == 0:
                    initial[block + position] = invariant[
                        state * width + v_power
                    ]
    else:
        if initial_variance is None:
            raise ValueError("initial_variance is required for a point start")
        block = initial_regime * len(basis)
        for position, (v_power, z_power) in enumerate(basis):
            if z_power == 0:
                initial[block + position] = initial_variance**v_power

    semigroup = expm(maturity * generator)
    raw = []
    for order in range(1, max_order + 1):
        payoff = np.zeros(len(basis))
        payoff[index[(0, order)]] = 1.0
        raw.append(
            initial @ semigroup @ np.kron(np.ones(states), payoff)
        )
    cumulant_values = []
    for order, moment in enumerate(raw, start=1):
        value = moment
        for index_value in range(1, order):
            value -= (
                math.comb(order - 1, index_value - 1)
                * cumulant_values[index_value - 1]
                * raw[order - index_value - 1]
            )
        cumulant_values.append(value)
    return np.array(cumulant_values)


def integrated_variance_intercept(speed, q0, pi, c, kappa, variance):
    """Exact bounded intercept in the stationary long-time variance.

    If -L phi=v-mu and -L psi=phi-E[phi], both Poisson solutions are
    affine.  The resolvent identity gives

        Var(int_0^T v_s ds) = sigma^2*T - 2 E[(v-mu) psi] + o(1).

    The normalization pi@q=0 fixes the otherwise arbitrary constant in
    psi; the intercept is independent of that normalization.
    """
    rate_value, mean, first, second, alpha, beta = integrated_variance_rate(
        speed, q0, pi, c, kappa, variance
    )
    drift = speed * q0 - np.diag(kappa)
    p = np.linalg.solve(drift, -alpha)
    phi_mean = first @ alpha + pi @ beta
    forcing = -beta + phi_mean * np.ones(len(pi)) - c * p
    assert abs(pi @ forcing) < 3e-13
    q = group_inverse(q0) @ forcing / speed
    assert np.linalg.norm(speed * q0 @ q - forcing) < 3e-12
    assert abs(pi @ q) < 3e-13
    intercept = -2 * (
        p @ (second - mean * first)
        + q @ (first - mean * pi)
    )
    return intercept, rate_value, mean, first, second, alpha, beta, p, q


def integrated_variance_remainder(
    speed, maturity, q0, pi, c, kappa, variance
):
    """Exact decaying remainder after the stationary slope and intercept."""
    (
        _, _, mean, first, second, _, _, p, q
    ) = integrated_variance_intercept(speed, q0, pi, c, kappa, variance)
    a_c, a_kappa, a_variance, _ = polynomial_operators(1)
    averaged = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
    )
    generator = full_generator(
        speed * q0, averaged, [a_c, a_kappa, a_variance],
        [c, kappa, variance]
    )
    payoff = np.array([
        coefficient
        for state in range(len(pi))
        for coefficient in (q[state], p[state])
    ])
    conditional = expm(maturity * generator) @ payoff
    covariance = 0.0
    for state in range(len(pi)):
        constant, linear = conditional[2 * state:2 * state + 2]
        covariance += constant * (first[state] - mean * pi[state])
        covariance += linear * (second[state] - mean * first[state])
    return 2 * covariance


def verify_uniform_variance_remainder():
    """Certify one switching-rate-uniform exponential remainder envelope.

    The theorem applies to every speed m >= m0.  A finite grid cannot prove
    that statement, so this certificate has a narrower role: it checks the
    exact affine-Poisson remainder against the independent degree-two moment
    semigroup, and illustrates one common exponential envelope on reversible
    and nonreversible examples over two decades of switching rates.
    """
    q2 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    models = [
        (
            "two-state",
            q2,
            np.array([0.5, 0.5]),
            np.array([0.04, 0.16]),
            np.array([0.8, 2.0]),
            np.array([0.04, 0.04]),
        )
    ]
    q3 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi3 = stationary(q3)
    kappa3 = np.array([1.1, 2.3, 3.0])
    models.append(
        (
            "three-state nonreversible",
            q3,
            pi3,
            kappa3 * np.array([0.035, 0.080, 0.050]),
            kappa3,
            np.array([0.20, 0.25, 0.22]) ** 2,
        )
    )

    speeds = 2.0 ** np.arange(8)
    maturities = np.linspace(0.0, 12.0, 49)
    gamma = 0.4
    results = {}
    for label, q0, pi, c, kappa, variance in models:
        envelopes = []
        terminal_remainders = []
        composite_errors = []
        for speed in speeds:
            intercept, slope, _, _, _, _, _, _, _ = (
                integrated_variance_intercept(
                    speed, q0, pi, c, kappa, variance
                )
            )
            weighted_remainders = []
            for maturity in maturities:
                remainder = integrated_variance_remainder(
                    speed, maturity, q0, pi, c, kappa, variance
                )
                independent = stationary_integrated_variance(
                    speed, maturity, q0, pi, c, kappa, variance
                )
                composite_errors.append(
                    abs(independent - slope * maturity - intercept - remainder)
                )
                weighted_remainders.append(
                    math.exp(gamma * maturity) * abs(remainder)
                )
            envelopes.append(max(weighted_remainders))
            terminal_remainders.append(
                abs(
                    integrated_variance_remainder(
                        speed, maturities[-1], q0, pi, c, kappa, variance
                    )
                )
            )

        # At long maturities the independent check subtracts the O(T) slope
        # and O(1) intercept, so its floating-point cancellation is larger
        # than the short-maturity checks above.
        assert max(composite_errors) < 4e-13
        if label == "two-state":
            assert max(envelopes) < 1.60e-3
            assert max(terminal_remainders) < 6.2e-10
        else:
            assert max(envelopes) < 4.11e-4
            assert max(terminal_remainders) < 6.6e-15
        results[label] = {
            "envelopes": envelopes,
            "terminal_remainders": terminal_remainders,
            "composite_error": max(composite_errors),
        }

    print("5b. switching-rate-uniform exponential variance remainder")
    print(
        f"   tested m=1,...,128 and T in [0,12] with gamma={gamma:.1f}; "
        f"largest envelopes {max(results['two-state']['envelopes']):.10f}/"
        f"{max(results['three-state nonreversible']['envelopes']):.10f}"
    )
    print(
        "   T=12 remainder maxima "
        f"{max(results['two-state']['terminal_remainders']):.2e}/"
        f"{max(results['three-state nonreversible']['terminal_remainders']):.2e}; "
        "composite discrepancies "
        f"{results['two-state']['composite_error']:.2e}/"
        f"{results['three-state nonreversible']['composite_error']:.2e}"
    )
    return gamma, results


def stationary_integrated_variance(
    speed, maturity, q0, pi, c, kappa, variance
):
    """Exact Var(int_0^T v_s ds) from an independent degree-two semigroup."""
    first, second = stationary_cir_moments(
        speed, q0, pi, c, kappa, variance
    )
    basis, a_c, a_kappa, v_d_vv, v_d_z = integrated_variance_operators(2)
    a_variance = 0.5 * v_d_vv
    averaged = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
        + v_d_z
    )
    generator = full_generator(
        speed * q0, averaged, [a_c, a_kappa, a_variance],
        [c, kappa, variance]
    )
    initial = np.zeros(len(pi) * len(basis))
    for state in range(len(pi)):
        block = slice(state * len(basis), (state + 1) * len(basis))
        for position, (v_power, z_power) in enumerate(basis):
            if z_power == 0:
                initial[block][position] = (pi[state], first[state], second[state])[
                    v_power
                ]
    payoff_mean = np.zeros(len(basis))
    payoff_second = np.zeros(len(basis))
    payoff_mean[basis.index((0, 1))] = 1.0
    payoff_second[basis.index((0, 2))] = 1.0
    semigroup = expm(maturity * generator)
    mean = initial @ semigroup @ np.kron(np.ones(len(pi)), payoff_mean)
    raw_second = initial @ semigroup @ np.kron(
        np.ones(len(pi)), payoff_second
    )
    return raw_second - mean**2


def stationary_integrated_third_cumulant(
    speed, maturity, q0, pi, c, kappa, variance
):
    """Exact third cumulant from an independent degree-three semigroup."""
    first, second, third = stationary_cir_third_moments(
        speed, q0, pi, c, kappa, variance
    )
    basis, a_c, a_kappa, v_d_vv, v_d_z = integrated_variance_operators(3)
    a_variance = 0.5 * v_d_vv
    averaged = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
        + v_d_z
    )
    generator = full_generator(
        speed * q0,
        averaged,
        [a_c, a_kappa, a_variance],
        [c, kappa, variance],
    )
    initial = np.zeros(len(pi) * len(basis))
    invariant_moments = (pi, first, second, third)
    for state in range(len(pi)):
        block = slice(state * len(basis), (state + 1) * len(basis))
        for position, (v_power, z_power) in enumerate(basis):
            if z_power == 0:
                initial[block][position] = invariant_moments[v_power][state]

    semigroup = expm(maturity * generator)
    raw = []
    for order in (1, 2, 3):
        payoff = np.zeros(len(basis))
        payoff[basis.index((0, order))] = 1.0
        raw.append(
            initial @ semigroup @ np.kron(np.ones(len(pi)), payoff)
        )
    first_raw, second_raw, third_raw = raw
    return (
        third_raw
        - 3 * second_raw * first_raw
        + 2 * first_raw**3
    )


def stationary_integrated_fourth_cumulant(
    speed, maturity, q0, pi, c, kappa, variance
):
    """Exact fourth cumulant from an independent degree-four semigroup."""
    first, second, third, fourth = stationary_cir_fourth_moments(
        speed, q0, pi, c, kappa, variance
    )
    basis, a_c, a_kappa, v_d_vv, v_d_z = integrated_variance_operators(4)
    a_variance = 0.5 * v_d_vv
    averaged = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
        + v_d_z
    )
    generator = full_generator(
        speed * q0,
        averaged,
        [a_c, a_kappa, a_variance],
        [c, kappa, variance],
    )
    initial = np.zeros(len(pi) * len(basis))
    invariant_moments = (pi, first, second, third, fourth)
    for state in range(len(pi)):
        block = slice(state * len(basis), (state + 1) * len(basis))
        for position, (v_power, z_power) in enumerate(basis):
            if z_power == 0:
                initial[block][position] = invariant_moments[v_power][state]

    semigroup = expm(maturity * generator)
    raw = []
    for order in (1, 2, 3, 4):
        payoff = np.zeros(len(basis))
        payoff[basis.index((0, order))] = 1.0
        raw.append(
            initial @ semigroup @ np.kron(np.ones(len(pi)), payoff)
        )
    return cumulants(raw)[3]


def spectral_mean_composite(speed, maturity, initial_components, q0, pi, c, kappa):
    """Exact mean and its exact-slope/principal-mode uniform composite."""
    drift = speed * q0 - np.diag(kappa)
    stationary_components = np.linalg.solve(drift.T, -(pi * c))
    difference = initial_components - stationary_components
    values, left, right = eig(drift, left=True, right=True)
    principal = np.argmax(values.real)
    eigenvalue = values[principal].real
    right_vector = right[:, principal].real
    left_vector = left[:, principal].real
    projection = np.outer(right_vector, left_vector) / (left_vector @ right_vector)
    exact = maturity * stationary_components.sum() + (
        difference @ np.linalg.solve(
            drift, expm(maturity * drift) - np.eye(len(pi))
        )
    ).sum()
    slow_integral = np.expm1(eigenvalue * maturity) / eigenvalue
    composite = (
        maturity * stationary_components.sum()
        + (difference @ projection @ np.ones(len(pi))) * slow_integral
    )
    return exact, composite, eigenvalue


def green_kubo_mean_composite(
    speed, maturity, initial_components, q0, pi, c, kappa
):
    """Uniform mean composite using the exact slope but explicit slow-mode data."""
    drift = speed * q0 - np.diag(kappa)
    stationary_components = np.linalg.solve(drift.T, -(pi * c))
    difference = initial_components - stationary_components
    centered_kappa = kappa - pi @ kappa
    corrector = group_inverse(q0) @ centered_kappa
    kappa_gk = -pi @ (centered_kappa * corrector)
    approximate_eigenvalue = -(pi @ kappa) + kappa_gk / speed
    approximate_right_vector = np.ones(len(pi)) + corrector / speed
    exact = maturity * stationary_components.sum() + (
        difference @ np.linalg.solve(
            drift, expm(maturity * drift) - np.eye(len(pi))
        )
    ).sum()
    slow_integral = np.expm1(approximate_eigenvalue * maturity) / (
        approximate_eigenvalue
    )
    composite = (
        maturity * stationary_components.sum()
        + (difference @ approximate_right_vector) * slow_integral
    )
    return exact, composite, approximate_eigenvalue, kappa_gk


def stationary_mean_coefficients(order, q0, pi, c, kappa):
    """Poisson recursion for the stationary first-moment expansion."""
    group = group_inverse(q0)
    kappa_matrix = np.diag(kappa)
    average_kappa = pi @ kappa
    rows = [(pi @ c / average_kappa) * pi]

    # At first order the affine forcing enters once.  At later orders the
    # preceding coefficient is simply propagated through the killing rate.
    centered = (rows[0] @ kappa_matrix - pi * c) @ group
    rows.append(centered - (centered @ kappa / average_kappa) * pi)
    for index in range(1, order):
        centered = rows[index] @ kappa_matrix @ group
        rows.append(centered - (centered @ kappa / average_kappa) * pi)

    assert abs(rows[0] @ kappa - pi @ c) < 2e-14
    assert all(abs(row @ kappa) < 2e-13 for row in rows[1:])
    coefficients = np.array([row.sum() for row in rows])
    return rows, coefficients


def truncated_stationary_mean_composite(
    speed, maturity, initial_components, q0, pi, c, kappa, order
):
    """Explicit slow-mode composite with a finite stationary expansion."""
    drift = speed * q0 - np.diag(kappa)
    exact_stationary = np.linalg.solve(drift.T, -(pi * c))
    exact = maturity * exact_stationary.sum() + (
        (initial_components - exact_stationary)
        @ np.linalg.solve(drift, expm(maturity * drift) - np.eye(len(pi)))
    ).sum()

    rows, coefficients = stationary_mean_coefficients(order, q0, pi, c, kappa)
    powers = speed ** np.arange(order + 1)
    stationary_approximation = sum(
        (row / powers[index] for index, row in enumerate(rows)),
        start=np.zeros(len(pi)),
    )
    slope_approximation = np.sum(coefficients / powers)
    centered_kappa = kappa - pi @ kappa
    corrector = group_inverse(q0) @ centered_kappa
    kappa_gk = -pi @ (centered_kappa * corrector)
    approximate_eigenvalue = -(pi @ kappa) + kappa_gk / speed
    slow_integral = np.expm1(approximate_eigenvalue * maturity) / (
        approximate_eigenvalue
    )
    composite = (
        maturity * slope_approximation
        + (initial_components - stationary_approximation)
        @ (np.ones(len(pi)) + corrector / speed)
        * slow_integral
    )
    return exact, composite, exact_stationary.sum(), slope_approximation


def verify_stationary_mean_expansion():
    """Check the Poisson recursion and its sharp growing-horizon scope."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    speeds = np.array([8, 16, 32, 64, 128], dtype=float)
    rows, coefficients = stationary_mean_coefficients(3, q0, pi, c, kappa)
    expected = np.array([1 / 14, -9 / 2450, 18 / 8575])
    assert np.max(np.abs(coefficients[:3] - expected)) < 2e-15

    slope_errors = {order: [] for order in (1, 2, 3)}
    window_errors = {order: [] for order in (1, 2, 3)}
    initial_components = pi * np.array([0.02, 0.08])
    for speed in speeds:
        drift = speed * q0 - np.diag(kappa)
        exact_stationary = np.linalg.solve(drift.T, -(pi * c))
        for order in (1, 2, 3):
            approximation = sum(
                coefficients[index] / speed**index
                for index in range(order + 1)
            )
            slope_errors[order].append(abs(exact_stationary.sum() - approximation))
            horizon = speed ** (order - 1)
            maturities = np.r_[0.0, np.geomspace(1e-8, horizon, 900)]
            errors = []
            for maturity in maturities:
                exact, composite, _, _ = truncated_stationary_mean_composite(
                    speed,
                    maturity,
                    initial_components,
                    q0,
                    pi,
                    c,
                    kappa,
                    order,
                )
                errors.append(abs(exact - composite))
            window_errors[order].append(max(errors))

    slope_rates = {order: rate(values) for order, values in slope_errors.items()}
    window_rates = {order: rate(values) for order, values in window_errors.items()}
    for order in (1, 2, 3):
        assert order + 0.75 < slope_rates[order] < order + 1.25
        assert 1.75 < window_rates[order] < 2.25

    # Repeat the recursion itself on a nonreversible chain, independently
    # checking the stationary solve and the predicted p+1 slope orders.
    q3 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi3 = stationary(q3)
    kappa3 = np.array([1.1, 2.3, 3.0])
    c3 = kappa3 * np.array([0.035, 0.080, 0.050])
    _, coefficients3 = stationary_mean_coefficients(3, q3, pi3, c3, kappa3)
    three_state_rates = []
    for order in (1, 2, 3):
        errors = []
        for speed in speeds:
            drift = speed * q3 - np.diag(kappa3)
            exact_slope = np.linalg.solve(drift.T, -(pi3 * c3)).sum()
            approximation = sum(
                coefficients3[index] / speed**index
                for index in range(order + 1)
            )
            errors.append(abs(exact_slope - approximation))
        three_state_rates.append(rate(errors))
    assert all(
        order + 0.7 < value < order + 1.3
        for order, value in zip((1, 2, 3), three_state_rates)
    )

    print("3. Poisson-recursive stationary mean on growing horizons")
    print(
        "   two-state coefficients mu_0,mu_1,mu_2,mu_3: "
        + " ".join(f"{value:.10f}" for value in coefficients)
    )
    print(
        "   slope remainder rates p=1,2,3: "
        + "/".join(f"{slope_rates[order]:.3f}" for order in (1, 2, 3))
    )
    print(
        "   composite rates on T<=m^(p-1): "
        + "/".join(f"{window_rates[order]:.3f}" for order in (1, 2, 3))
    )
    print(
        "   nonreversible slope remainder rates: "
        + "/".join(f"{value:.3f}" for value in three_state_rates)
    )
    return slope_rates, window_rates, three_state_rates


def verify_uniform_spectral_mean():
    """Check uniform O(m^-2), and O(m^-3) without initial regime dependence."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    speeds = (8, 16, 32, 64, 128)
    independent_errors = []
    conditional_errors = []
    explicit_independent_errors = []
    explicit_conditional_errors = []
    eigenvalue_errors = []
    eigenvalues = []
    for speed in speeds:
        maturities = np.r_[0.0, np.geomspace(1e-8, float(speed**2), 1600)]
        errors = {"independent": [], "conditional": []}
        explicit_errors = {"independent": [], "conditional": []}
        initial = {
            "independent": pi * 0.05,
            "conditional": pi * np.array([0.02, 0.08]),
        }
        for maturity in maturities:
            for label, initial_components in initial.items():
                exact, composite, eigenvalue = spectral_mean_composite(
                    speed,
                    maturity,
                    initial_components,
                    q0,
                    pi,
                    c,
                    kappa,
                )
                errors[label].append(abs(exact - composite))
                exact_explicit, explicit, approximate_eigenvalue, kappa_gk = (
                    green_kubo_mean_composite(
                        speed,
                        maturity,
                        initial_components,
                        q0,
                        pi,
                        c,
                        kappa,
                    )
                )
                assert abs(exact - exact_explicit) < 2e-14
                explicit_errors[label].append(abs(exact - explicit))
        independent_errors.append(max(errors["independent"]))
        conditional_errors.append(max(errors["conditional"]))
        explicit_independent_errors.append(max(explicit_errors["independent"]))
        explicit_conditional_errors.append(max(explicit_errors["conditional"]))
        eigenvalues.append(eigenvalue)
        eigenvalue_errors.append(abs(eigenvalue - approximate_eigenvalue))

    independent_rate = rate(independent_errors)
    conditional_rate = rate(conditional_errors)
    assert 2.8 < independent_rate < 3.2
    assert 1.8 < conditional_rate < 2.2
    explicit_independent_rate = rate(explicit_independent_errors)
    explicit_conditional_rate = rate(explicit_conditional_errors)
    assert 1.8 < explicit_independent_rate < 2.2
    assert 1.8 < explicit_conditional_rate < 2.2
    # This symmetric example has an additional cancellation: the eigenvalue
    # remainder is third rather than merely second order.
    assert 2.8 < rate(eigenvalue_errors) < 3.2
    print("2. slope-and-principal-mode uniform mean composite")
    print(
        "   independent initial variance: "
        + " ".join(f"{value:.3e}" for value in independent_errors)
        + f"  rate {independent_rate:.3f}"
    )
    print(
        "   regime-dependent initial mean: "
        + " ".join(f"{value:.3e}" for value in conditional_errors)
        + f"  rate {conditional_rate:.3f}"
    )
    print(f"   slow eigenvalue at m=128: {eigenvalues[-1]:.10f}")
    print("   exact slope plus explicit Green--Kubo slow mode")
    print(
        "      independent initial variance: "
        + " ".join(f"{value:.3e}" for value in explicit_independent_errors)
        + f"  rate {explicit_independent_rate:.3f}"
    )
    print(
        "      regime-dependent initial mean: "
        + " ".join(f"{value:.3e}" for value in explicit_conditional_errors)
        + f"  rate {explicit_conditional_rate:.3f}"
    )
    print(
        f"      lambda_hat=-bar(kappa)+K_kappa,kappa/m; "
        f"K_kappa,kappa={kappa_gk:.10f}; eigenvalue rate "
        f"{rate(eigenvalue_errors):.3f}"
    )

    # Recheck the explicit formula on a nonreversible three-state chain.
    q3 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi3 = stationary(q3)
    kappa3 = np.array([1.1, 2.3, 3.0])
    c3 = kappa3 * np.array([0.035, 0.080, 0.050])
    three_state_errors = {"independent": [], "conditional": []}
    initial3 = {
        "independent": pi3 * 0.05,
        "conditional": pi3 * np.array([0.02, 0.08, 0.05]),
    }
    for speed in speeds:
        maturities = np.r_[0.0, np.geomspace(1e-8, float(speed**2), 1600)]
        for label, initial_components in initial3.items():
            errors3 = []
            for maturity in maturities:
                exact, composite, _, _ = green_kubo_mean_composite(
                    speed,
                    maturity,
                    initial_components,
                    q3,
                    pi3,
                    c3,
                    kappa3,
                )
                errors3.append(abs(exact - composite))
            three_state_errors[label].append(max(errors3))
    three_state_rates = {
        label: rate(values) for label, values in three_state_errors.items()
    }
    assert 1.8 < three_state_rates["independent"] < 2.2
    assert 1.8 < three_state_rates["conditional"] < 2.2
    print(
        "      nonreversible three-state rates independent/conditional: "
        f"{three_state_rates['independent']:.3f}/"
        f"{three_state_rates['conditional']:.3f}"
    )
    return (
        independent_rate,
        conditional_rate,
        explicit_independent_rate,
        explicit_conditional_rate,
    )


def verify_long_maturity_nonuniformity():
    """Certify the sharp failure of an absolute maturity-uniform O(m^-2) bound."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    # xi=(0.2,0.2) makes both statewise Feller inequalities strict.  Vol-of-vol
    # does not enter this first-moment obstruction.
    assert np.all(2 * c > np.array([0.2, 0.2]) ** 2)

    mu0 = 1.0 / 14.0
    mu1 = -9.0 / 2450.0
    joint_scale_limit = 18.0 / 8575.0
    speeds = (8, 16, 32, 64, 128)
    residuals = []
    slope_errors = []
    for speed in speeds:
        maturity = float(speed**2)
        integrated_mean, stationary_mean = exact_integrated_mean(
            speed, maturity, mu0, q0, pi, c, kappa
        )
        closed_mean = (25.0 * speed + 13.0) / (50.0 * (7.0 * speed + 4.0))
        slope_remainder = 18.0 / (1225.0 * speed * (7.0 * speed + 4.0))
        assert abs(stationary_mean - closed_mean) < 1e-15
        assert abs(stationary_mean - mu0 - mu1 / speed - slope_remainder) < 1e-15
        residuals.append(integrated_mean - mu0 * maturity - mu1 * maturity / speed)
        slope_errors.append(abs(speed**2 * slope_remainder - joint_scale_limit))

    assert residuals[-1] > joint_scale_limit
    assert abs(residuals[-1] - joint_scale_limit) < 1.2e-5
    assert 0.9 < rate(slope_errors) < 1.1
    print("1. long-maturity obstruction under simultaneous switching")
    print("   mu_m=(25m+13)/(50(7m+4))")
    print("   mu_m-mu_0-mu_1/m=18/(1225m(7m+4)) > 0")
    print(
        "   T=m^2 corrected residual: "
        + " ".join(f"{value:.10f}" for value in residuals)
        + f"  limit {joint_scale_limit:.10f}"
    )
    return residuals


def verify_long_run_variance_rate():
    """Check the two Poisson formulas against a separate moment semigroup."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    variance = np.array([0.04, 0.04])
    speeds = np.array([8, 16, 32, 64, 128], dtype=float)

    # For this rational two-state example, exact elimination gives the
    # following finite-rate expression. Coefficients are in ascending powers
    # of m, so this check is independent of the Poisson linear algebra above.
    numerator = np.array([288, 4276, 19103, 35325, 28825, 8750], dtype=float)
    denominator = np.array(
        [0, 1280000, 7840000, 17640000, 17150000, 6002500],
        dtype=float,
    )
    intercept_numerator = np.array(
        [-1152, -9072, -53800, -216369, -474170, -562900, -349500, -87500],
        dtype=float,
    )
    intercept_denominator = np.array(
        [0, 0, 10240000, 80640000, 250880000, 384160000, 288120000, 84035000],
        dtype=float,
    )
    sigma0 = 1.0 / 686.0
    sigma1 = 153.0 / 240100.0
    sigma2 = -369.0 / 1680700.0
    chi0 = -5.0 / 4802.0
    chi1 = -99.0 / 168070.0
    chi2 = 477.0 / 5882450.0
    exact_rates = []
    exact_intercepts = []
    averaged_errors = []
    corrected_errors = []
    intercept_averaged_errors = []
    intercept_corrected_errors = []
    scaled_second_remainders = []
    intercept_scaled_second_remainders = []
    semigroup_errors = []
    intercept_semigroup_errors = []
    finite_composite_errors = []
    for speed in speeds:
        intercept, exact, _, _, _, _, _, _, _ = integrated_variance_intercept(
            speed, q0, pi, c, kappa, variance
        )
        closed = sum(numerator[j] * speed**j for j in range(6)) / sum(
            denominator[j] * speed**j for j in range(6)
        )
        closed_intercept = sum(
            intercept_numerator[j] * speed**j for j in range(8)
        ) / sum(intercept_denominator[j] * speed**j for j in range(8))
        assert abs(exact - closed) < 3e-16
        assert abs(intercept - closed_intercept) < 3e-16
        exact_rates.append(exact)
        exact_intercepts.append(intercept)
        averaged_errors.append(abs(exact - sigma0))
        corrected_errors.append(abs(exact - sigma0 - sigma1 / speed))
        intercept_averaged_errors.append(abs(intercept - chi0))
        intercept_corrected_errors.append(
            abs(intercept - chi0 - chi1 / speed)
        )
        scaled_second_remainders.append(
            speed**2 * (exact - sigma0 - sigma1 / speed)
        )
        intercept_scaled_second_remainders.append(
            speed**2 * (intercept - chi0 - chi1 / speed)
        )

        # The degree-two semigroup starts the joint process in its invariant
        # law. Differencing two long maturities removes the bounded intercept.
        variance_20 = stationary_integrated_variance(
            speed, 20.0, q0, pi, c, kappa, variance
        )
        variance_40 = stationary_integrated_variance(
            speed, 40.0, q0, pi, c, kappa, variance
        )
        independent_slope = (variance_40 - variance_20) / 20.0
        semigroup_errors.append(abs(independent_slope - exact))
        intercept_semigroup_errors.append(
            abs(variance_20 - 20.0 * exact - intercept)
        )
        remainder = integrated_variance_remainder(
            speed, 0.75, q0, pi, c, kappa, variance
        )
        variance_short = stationary_integrated_variance(
            speed, 0.75, q0, pi, c, kappa, variance
        )
        finite_composite_errors.append(
            abs(variance_short - 0.75 * exact - intercept - remainder)
        )

    averaged_rate = rate(averaged_errors)
    corrected_rate = rate(corrected_errors)
    intercept_averaged_rate = rate(intercept_averaged_errors)
    intercept_corrected_rate = rate(intercept_corrected_errors)
    assert 0.95 < averaged_rate < 1.05
    assert 1.95 < corrected_rate < 2.05
    assert 0.95 < intercept_averaged_rate < 1.05
    assert 1.90 < intercept_corrected_rate < 2.05
    assert abs(scaled_second_remainders[-1] - sigma2) < 2e-6
    assert abs(intercept_scaled_second_remainders[-1] - chi2) < 1e-6
    assert max(semigroup_errors) < 3e-12
    assert max(intercept_semigroup_errors) < 3e-12
    assert max(finite_composite_errors) < 3e-14

    # Repeat the exact Poisson-versus-semigroup check on a nonreversible
    # three-state model to ensure the formula does not use reversibility.
    q3 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi3 = stationary(q3)
    kappa3 = np.array([1.1, 2.3, 3.0])
    c3 = kappa3 * np.array([0.035, 0.080, 0.050])
    variance3 = np.array([0.20, 0.25, 0.22]) ** 2
    (
        nonreversible_intercept,
        nonreversible_rate,
        _, _, _, _, _, _, _,
    ) = integrated_variance_intercept(
        32.0, q3, pi3, c3, kappa3, variance3
    )
    variance_20 = stationary_integrated_variance(
        32.0, 20.0, q3, pi3, c3, kappa3, variance3
    )
    variance_40 = stationary_integrated_variance(
        32.0, 40.0, q3, pi3, c3, kappa3, variance3
    )
    nonreversible_error = abs(
        (variance_40 - variance_20) / 20.0 - nonreversible_rate
    )
    nonreversible_intercept_error = abs(
        variance_20 - 20.0 * nonreversible_rate - nonreversible_intercept
    )
    nonreversible_remainder = integrated_variance_remainder(
        32.0, 0.75, q3, pi3, c3, kappa3, variance3
    )
    nonreversible_short = stationary_integrated_variance(
        32.0, 0.75, q3, pi3, c3, kappa3, variance3
    )
    nonreversible_composite_error = abs(
        nonreversible_short
        - 0.75 * nonreversible_rate
        - nonreversible_intercept
        - nonreversible_remainder
    )
    assert nonreversible_error < 3e-12
    assert nonreversible_intercept_error < 3e-12
    assert nonreversible_composite_error < 3e-14

    print("4. exact long-run integrated-variance variance rate")
    print(
        "   sigma_m^2: " + " ".join(f"{value:.10f}" for value in exact_rates)
    )
    print(
        f"   averaged/corrected rates {averaged_rate:.3f}/{corrected_rate:.3f}; "
        f"m^2 remainder {scaled_second_remainders[-1]:+.10f} -> {sigma2:+.10f}"
    )
    print(
        f"   polynomial-semigroup discrepancy {max(semigroup_errors):.2e}; "
        f"nonreversible discrepancy {nonreversible_error:.2e}"
    )
    print("5. exact bounded intercept in the stationary variance")
    print(
        "   chi_m: "
        + " ".join(f"{value:.10f}" for value in exact_intercepts)
    )
    print(
        f"   averaged/corrected rates {intercept_averaged_rate:.3f}/"
        f"{intercept_corrected_rate:.3f}; m^2 remainder "
        f"{intercept_scaled_second_remainders[-1]:+.10f} -> {chi2:+.10f}"
    )
    print(
        f"   polynomial-semigroup discrepancy "
        f"{max(intercept_semigroup_errors):.2e}; nonreversible discrepancy "
        f"{nonreversible_intercept_error:.2e}"
    )
    print(
        f"   finite-T composite discrepancy {max(finite_composite_errors):.2e}; "
        f"nonreversible discrepancy {nonreversible_composite_error:.2e}"
    )
    return (
        averaged_rate,
        corrected_rate,
        max(semigroup_errors),
        intercept_averaged_rate,
        intercept_corrected_rate,
        max(intercept_semigroup_errors),
    )


def verify_long_run_third_cumulant_rate():
    """Check the third-cumulant slope and intercept Poisson formulas."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    variance = np.array([0.04, 0.04])
    speeds = np.array([8, 16, 32, 64, 128], dtype=float)

    averaged_rate = integrated_variance_third_rate(
        1.0,
        np.zeros((1, 1)),
        np.ones(1),
        np.array([pi @ c]),
        np.array([pi @ kappa]),
        np.array([pi @ variance]),
    )[0]
    averaged_intercept = integrated_variance_third_intercept(
        1.0,
        np.zeros((1, 1)),
        np.ones(1),
        np.array([pi @ c]),
        np.array([pi @ kappa]),
        np.array([pi @ variance]),
    )[0]
    exact_rates = []
    exact_intercepts = []
    semigroup_errors = []
    intercept_semigroup_errors = []
    uncentered_errors = []
    for speed in speeds:
        result = integrated_variance_third_rate(
            speed, q0, pi, c, kappa, variance
        )
        (
            exact,
            mean,
            first,
            second,
            _,
            alpha,
            phi_mean,
            centered_beta,
            _,
            _,
            _,
        ) = result
        exact_rates.append(exact)
        intercept, intercept_rate = integrated_variance_third_intercept(
            speed, q0, pi, c, kappa, variance
        )
        exact_intercepts.append(intercept)
        assert abs(intercept_rate - exact) < 3e-14
        half_variance_rate = (
            alpha @ (second - mean * first)
            + centered_beta @ (first - mean * pi)
        )

        # If phi is normalized only by pi@beta=0 rather than E_Pi[phi]=0,
        # the third derivative gains this spurious normalization term.
        uncentered = exact + 6 * phi_mean * half_variance_rate
        uncentered_errors.append(abs(uncentered - exact))

        cumulant_20 = stationary_integrated_third_cumulant(
            speed, 20.0, q0, pi, c, kappa, variance
        )
        cumulant_40 = stationary_integrated_third_cumulant(
            speed, 40.0, q0, pi, c, kappa, variance
        )
        independent_slope = (cumulant_40 - cumulant_20) / 20.0
        semigroup_errors.append(abs(independent_slope - exact))
        intercept_semigroup_errors.append(
            abs(cumulant_20 - exact * 20.0 - intercept)
        )

    averaged_errors = np.abs(np.array(exact_rates) - averaged_rate)
    convergence_rate = rate(averaged_errors)
    intercept_errors = np.abs(
        np.array(exact_intercepts) - averaged_intercept
    )
    intercept_convergence_rate = rate(intercept_errors)
    assert 0.9 < convergence_rate < 1.1
    assert 0.9 < intercept_convergence_rate < 1.1
    assert max(semigroup_errors) < 5e-13
    assert max(intercept_semigroup_errors) < 5e-13
    assert min(uncentered_errors) > 2e-4

    # Repeat the exact formula on a nonreversible chain.  This also checks
    # the ordering in the nested resolvents, which would be invisible in a
    # reversible example.
    q3 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi3 = stationary(q3)
    kappa3 = np.array([1.1, 2.3, 3.0])
    c3 = kappa3 * np.array([0.035, 0.080, 0.050])
    variance3 = np.array([0.20, 0.25, 0.22]) ** 2
    nonreversible_rate = integrated_variance_third_rate(
        32.0, q3, pi3, c3, kappa3, variance3
    )[0]
    nonreversible_intercept, nonreversible_intercept_rate = (
        integrated_variance_third_intercept(
            32.0, q3, pi3, c3, kappa3, variance3
        )
    )
    assert abs(nonreversible_intercept_rate - nonreversible_rate) < 3e-14
    nonreversible_20 = stationary_integrated_third_cumulant(
        32.0, 20.0, q3, pi3, c3, kappa3, variance3
    )
    nonreversible_40 = stationary_integrated_third_cumulant(
        32.0, 40.0, q3, pi3, c3, kappa3, variance3
    )
    nonreversible_error = abs(
        (nonreversible_40 - nonreversible_20) / 20.0
        - nonreversible_rate
    )
    nonreversible_10 = stationary_integrated_third_cumulant(
        32.0, 10.0, q3, pi3, c3, kappa3, variance3
    )
    nonreversible_intercept_error = abs(
        nonreversible_10
        - 10.0 * nonreversible_rate
        - nonreversible_intercept
    )
    assert nonreversible_error < 2e-13
    assert nonreversible_intercept_error < 2e-13

    print("5c. exact long-run integrated-variance third-cumulant rate")
    print(
        "   tau_m: " + " ".join(f"{value:.10f}" for value in exact_rates)
    )
    print(
        f"   averaged limit {averaged_rate:.10f}; convergence rate "
        f"{convergence_rate:.3f}"
    )
    print(
        f"   polynomial-semigroup discrepancy {max(semigroup_errors):.2e}; "
        f"nonreversible discrepancy {nonreversible_error:.2e}"
    )
    print(
        f"   omitting joint-invariant centering changes the m=8 rate by "
        f"{uncentered_errors[0]:.10f}"
    )
    print("5d. exact bounded third-cumulant intercept")
    print(
        "   chi_m: "
        + " ".join(f"{value:.10f}" for value in exact_intercepts)
    )
    print(
        f"   averaged limit {averaged_intercept:.10f}; convergence rate "
        f"{intercept_convergence_rate:.3f}"
    )
    print(
        f"   T=20 residual {max(intercept_semigroup_errors):.2e}; "
        f"nonreversible T=10 residual {nonreversible_intercept_error:.2e}"
    )
    return (
        convergence_rate,
        max(semigroup_errors),
        nonreversible_error,
        intercept_convergence_rate,
        max(intercept_semigroup_errors),
        nonreversible_intercept_error,
    )


def verify_long_run_fourth_cumulant_rate():
    """Check the fourth eigenvalue derivative against a degree-four semigroup."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    variance = np.array([0.04, 0.04])
    speeds = np.array([8, 16, 32, 64, 128], dtype=float)

    averaged_rate = integrated_variance_fourth_rate(
        1.0,
        np.zeros((1, 1)),
        np.ones(1),
        np.array([pi @ c]),
        np.array([pi @ kappa]),
        np.array([pi @ variance]),
    )[0]
    exact_rates = []
    semigroup_errors = []
    naive_errors = []
    for speed in speeds:
        exact, naive = integrated_variance_fourth_rate(
            speed, q0, pi, c, kappa, variance
        )
        exact_rates.append(exact)
        naive_errors.append(abs(naive - exact))
        cumulant_20 = stationary_integrated_fourth_cumulant(
            speed, 20.0, q0, pi, c, kappa, variance
        )
        cumulant_40 = stationary_integrated_fourth_cumulant(
            speed, 40.0, q0, pi, c, kappa, variance
        )
        independent_slope = (cumulant_40 - cumulant_20) / 20.0
        semigroup_errors.append(abs(independent_slope - exact))

    averaged_errors = np.abs(np.array(exact_rates) - averaged_rate)
    convergence_rate = rate(averaged_errors)
    assert 0.9 < convergence_rate < 1.1
    assert max(semigroup_errors) < 1e-12
    assert min(naive_errors) > 7e-6

    q3 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi3 = stationary(q3)
    kappa3 = np.array([1.1, 2.3, 3.0])
    c3 = kappa3 * np.array([0.035, 0.080, 0.050])
    variance3 = np.array([0.20, 0.25, 0.22]) ** 2
    nonreversible_rate = integrated_variance_fourth_rate(
        32.0, q3, pi3, c3, kappa3, variance3
    )[0]
    nonreversible_10 = stationary_integrated_fourth_cumulant(
        32.0, 10.0, q3, pi3, c3, kappa3, variance3
    )
    nonreversible_20 = stationary_integrated_fourth_cumulant(
        32.0, 20.0, q3, pi3, c3, kappa3, variance3
    )
    nonreversible_error = abs(
        (nonreversible_20 - nonreversible_10) / 10.0
        - nonreversible_rate
    )
    assert nonreversible_error < 2e-13

    print("5e. exact long-run integrated-variance fourth-cumulant rate")
    print(
        "   upsilon_m: "
        + " ".join(f"{value:.10f}" for value in exact_rates)
    )
    print(
        f"   averaged limit {averaged_rate:.10f}; convergence rate "
        f"{convergence_rate:.3f}"
    )
    print(
        f"   degree-four semigroup discrepancy {max(semigroup_errors):.2e}; "
        f"nonreversible discrepancy {nonreversible_error:.2e}"
    )
    print(
        f"   omitting c2*phi changes the m=8 rate by "
        f"{naive_errors[0]:.10f}"
    )
    return convergence_rate, max(semigroup_errors), nonreversible_error


def verify_all_fixed_order_cumulant_rates(max_order=8):
    """Check the all-fixed-order recursion against closed CIR coefficients."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    variance = np.array([0.04, 0.04])
    speed = 8.0

    switched_rates, _ = integrated_variance_cumulant_rates(
        max_order, speed, q0, pi, c, kappa, variance
    )
    variance_rate = integrated_variance_rate(
        speed, q0, pi, c, kappa, variance
    )[0]
    third_rate = integrated_variance_third_rate(
        speed, q0, pi, c, kappa, variance
    )[0]
    fourth_rate = integrated_variance_fourth_rate(
        speed, q0, pi, c, kappa, variance
    )[0]
    low_order_error = np.max(
        np.abs(switched_rates[1:4] - [variance_rate, third_rate, fourth_rate])
    )
    assert low_order_error < 2e-13

    averaged_c = np.array([pi @ c])
    averaged_kappa = np.array([pi @ kappa])
    averaged_variance = np.array([pi @ variance])
    averaged_rates, _ = integrated_variance_cumulant_rates(
        max_order,
        1.0,
        np.zeros((1, 1)),
        np.ones(1),
        averaged_c,
        averaged_kappa,
        averaged_variance,
    )
    closed_rates = np.zeros(max_order)
    for order in range(2, max_order + 1):
        odd_double_factorial = math.prod(range(1, 2 * order - 2, 2))
        closed_rates[order - 1] = (
            averaged_c[0]
            * odd_double_factorial
            * averaged_variance[0] ** (order - 1)
            / averaged_kappa[0] ** (2 * order - 1)
        )
    positive = closed_rates > 0
    averaged_relative_error = np.max(
        np.abs(averaged_rates[positive] / closed_rates[positive] - 1.0)
    )
    assert averaged_relative_error < 3e-11

    print("5f. all-fixed-order long-run cumulant recursion")
    print(
        "   switched m=8 orders 2--8: "
        + " ".join(f"{value:.10e}" for value in switched_rates[1:])
    )
    print(
        f"   low-order identity discrepancy {low_order_error:.2e}; "
        f"closed CIR orders 2--{max_order} relative discrepancy "
        f"{averaged_relative_error:.2e}"
    )
    return low_order_error, averaged_relative_error


def verify_all_fixed_order_rate_corrections(max_order=8):
    """Check the Green--Kubo first coefficient through fixed order eight."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    variance = np.array([0.04, 0.04])
    corrections, _ = cumulant_rate_first_corrections(
        max_order, q0, pi, c, kappa, variance
    )

    # These two coefficients were obtained independently above: the exact
    # stationary mean is rational in m, and so is the exact variance rate.
    mean_coefficient_error = abs(corrections[0] + 9.0 / 2450.0)
    variance_coefficient_error = abs(
        corrections[1] - 153.0 / 240100.0
    )
    assert mean_coefficient_error < 2e-14
    assert variance_coefficient_error < 2e-14

    averaged_rates, _ = integrated_variance_cumulant_rates(
        max_order,
        1.0,
        np.zeros((1, 1)),
        np.ones(1),
        np.array([pi @ c]),
        np.array([pi @ kappa]),
        np.array([pi @ variance]),
    )
    speeds = np.array([16, 32, 64, 128, 256], dtype=float)
    uncorrected_errors = []
    corrected_errors = []
    scaled_second_remainders = []
    for speed in speeds:
        exact_rates, _ = integrated_variance_cumulant_rates(
            max_order, speed, q0, pi, c, kappa, variance
        )
        uncorrected_errors.append(
            np.abs(exact_rates[1:] - averaged_rates[1:])
        )
        remainder = (
            exact_rates[1:]
            - averaged_rates[1:]
            - corrections[1:] / speed
        )
        corrected_errors.append(np.abs(remainder))
        scaled_second_remainders.append(speed**2 * remainder)

    uncorrected_errors = np.array(uncorrected_errors)
    corrected_errors = np.array(corrected_errors)
    uncorrected_rates = np.log2(
        uncorrected_errors[-2] / uncorrected_errors[-1]
    )
    corrected_rates = np.log2(
        corrected_errors[-2] / corrected_errors[-1]
    )
    assert np.all((uncorrected_rates > 0.9) & (uncorrected_rates < 1.1))
    assert np.all((corrected_rates > 1.85) & (corrected_rates < 2.15))

    print("5g. first inverse-speed coefficient of every fixed-order rate")
    print(
        f"   eta_2--eta_{max_order}: "
        + " ".join(f"{value:.10e}" for value in corrections[1:])
    )
    print(
        "   uncorrected residual orders: "
        + " ".join(f"{value:.6f}" for value in uncorrected_rates)
    )
    print(
        "   corrected residual orders:   "
        + " ".join(f"{value:.6f}" for value in corrected_rates)
    )
    print(
        f"   independent eta_1/eta_2 errors "
        f"{mean_coefficient_error:.2e}/{variance_coefficient_error:.2e}"
    )
    return (
        corrections,
        uncorrected_rates,
        corrected_rates,
        np.array(scaled_second_remainders),
    )


def verify_all_fixed_order_second_rate_corrections(max_order=8):
    """Check the explicit second inverse-speed coefficient through order eight."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    variance = np.array([0.04, 0.04])
    first, second, missing_response = cumulant_rate_second_corrections(
        max_order, q0, pi, c, kappa, variance
    )
    independent_first, _ = cumulant_rate_first_corrections(
        max_order, q0, pi, c, kappa, variance
    )
    first_error = np.max(np.abs(first - independent_first))
    assert first_error < 2e-11

    # The exact stationary mean and variance-rate rational functions derived
    # independently above supply the first two second-order coefficients.
    mean_second_error = abs(second[0] - 18.0 / 8575.0)
    variance_second_error = abs(second[1] + 369.0 / 1680700.0)
    assert mean_second_error < 2e-13
    assert variance_second_error < 2e-13

    averaged_rates, _ = integrated_variance_cumulant_rates(
        max_order,
        1.0,
        np.zeros((1, 1)),
        np.ones(1),
        np.array([pi @ c]),
        np.array([pi @ kappa]),
        np.array([pi @ variance]),
    )
    speeds = np.array([6.0, 12.0, 24.0])
    corrected_errors = []
    missing_response_errors = []
    for speed in speeds:
        exact_rates, _ = integrated_variance_cumulant_rates(
            max_order, speed, q0, pi, c, kappa, variance
        )
        corrected_errors.append(
            np.abs(
                exact_rates[1:]
                - averaged_rates[1:]
                - first[1:] / speed
                - second[1:] / speed**2
            )
        )
        missing_response_errors.append(
            np.abs(
                exact_rates[1:]
                - averaged_rates[1:]
                - first[1:] / speed
                - missing_response[1:] / speed**2
            )
        )
    corrected_errors = np.array(corrected_errors)
    missing_response_errors = np.array(missing_response_errors)
    corrected_rates = np.log2(
        corrected_errors[-2] / corrected_errors[-1]
    )
    missing_response_rates = np.log2(
        missing_response_errors[-2] / missing_response_errors[-1]
    )
    assert np.all((corrected_rates > 2.75) & (corrected_rates < 3.35))
    assert np.all(
        (missing_response_rates > 1.8) & (missing_response_rates < 2.2)
    )

    # A nonreversible three-state chain checks that neither detailed balance
    # nor the symmetric two-state coordinate was used in the Schur formula.
    q_three = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi_three = stationary(q_three)
    kappa_three = np.array([1.1, 2.3, 3.0])
    c_three = kappa_three * np.array([0.035, 0.080, 0.050])
    variance_three = np.array([0.20, 0.25, 0.22]) ** 2
    nonreversible_order = min(max_order, 6)
    first_three, second_three, _ = cumulant_rate_second_corrections(
        nonreversible_order,
        q_three,
        pi_three,
        c_three,
        kappa_three,
        variance_three,
        radius=0.18,
        samples=384,
    )
    averaged_three, _ = integrated_variance_cumulant_rates(
        nonreversible_order,
        1.0,
        np.zeros((1, 1)),
        np.ones(1),
        np.array([pi_three @ c_three]),
        np.array([pi_three @ kappa_three]),
        np.array([pi_three @ variance_three]),
    )
    nonreversible_errors = []
    for speed in speeds:
        exact_three, _ = integrated_variance_cumulant_rates(
            nonreversible_order,
            speed,
            q_three,
            pi_three,
            c_three,
            kappa_three,
            variance_three,
        )
        nonreversible_errors.append(
            np.abs(
                exact_three[1:]
                - averaged_three[1:]
                - first_three[1:] / speed
                - second_three[1:] / speed**2
            )
        )
    nonreversible_errors = np.array(nonreversible_errors)
    nonreversible_rates = np.log2(
        nonreversible_errors[-2] / nonreversible_errors[-1]
    )
    assert np.all(
        (nonreversible_rates > 2.75) & (nonreversible_rates < 3.25)
    )

    print("5h. second inverse-speed coefficient of every fixed-order rate")
    print(
        f"   zeta_2--zeta_{max_order}: "
        + " ".join(f"{value:.10e}" for value in second[1:])
    )
    print(
        "   twice-corrected residual orders: "
        + " ".join(f"{value:.6f}" for value in corrected_rates)
    )
    print(
        "   without eigenvector response:    "
        + " ".join(f"{value:.6f}" for value in missing_response_rates)
    )
    print(
        "   nonreversible orders 2--6:        "
        + " ".join(f"{value:.6f}" for value in nonreversible_rates)
    )
    print(
        f"   independent eta / zeta_1 / zeta_2 errors "
        f"{first_error:.2e}/{mean_second_error:.2e}/"
        f"{variance_second_error:.2e}"
    )
    return (
        second,
        corrected_rates,
        missing_response_rates,
        nonreversible_rates,
        first_error,
        mean_second_error,
        variance_second_error,
    )


def verify_all_fixed_order_cumulant_intercepts(max_order=6):
    """Check fixed-order boundary constants for stationary and point starts."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    variance = np.array([0.04, 0.04])
    speed = 8.0
    rates, _ = integrated_variance_cumulant_rates(
        max_order, speed, q0, pi, c, kappa, variance
    )

    intercepts = {}
    discrepancies = {}
    recursion_errors = {}
    starts = {
        "stationary": {},
        "point": {"initial_regime": 0, "initial_variance": 0.04},
    }
    for name, start in starts.items():
        cumulants_20 = centered_integrated_cumulants(
            max_order, 20.0, speed, q0, pi, c, kappa, variance, **start
        )
        cumulants_40 = centered_integrated_cumulants(
            max_order, 40.0, speed, q0, pi, c, kappa, variance, **start
        )
        intercept_20 = cumulants_20 - 20.0 * rates
        intercept_40 = cumulants_40 - 40.0 * rates
        intercepts[name] = intercept_40
        discrepancies[name] = np.max(np.abs(intercept_40 - intercept_20))
        recursive, _, _, _ = integrated_variance_boundary_constants(
            max_order, speed, q0, pi, c, kappa, variance, **start
        )
        recursion_errors[name] = np.max(np.abs(recursive - intercept_40))

    second_intercept = integrated_variance_intercept(
        speed, q0, pi, c, kappa, variance
    )[0]
    third_intercept = integrated_variance_third_intercept(
        speed, q0, pi, c, kappa, variance
    )[0]
    known_error = np.max(
        np.abs(
            intercepts["stationary"][1:3]
            - [second_intercept, third_intercept]
        )
    )
    assert known_error < 2e-10
    assert discrepancies["stationary"] < 2e-9
    assert discrepancies["point"] < 2e-9
    assert recursion_errors["stationary"] < 2e-9
    assert recursion_errors["point"] < 2e-9

    print("5i. all-fixed-order cumulant intercepts")
    for name in starts:
        print(
            f"   {name} orders 1--{max_order}: "
            + " ".join(f"{value:+.10e}" for value in intercepts[name])
        )
    print(
        f"   T=20/40 discrepancy stationary/point "
        f"{discrepancies['stationary']:.2e}/{discrepancies['point']:.2e}; "
        f"Perron-recursion discrepancy "
        f"{recursion_errors['stationary']:.2e}/{recursion_errors['point']:.2e}; "
        f"known order-2/3 discrepancy {known_error:.2e}"
    )
    return intercepts, discrepancies, recursion_errors, known_error


def verify_uniform_fixed_order_cumulant_remainders(max_order=6):
    """Illustrate one switching-rate-uniform envelope at every tested order.

    The theorem is analytic and applies at each fixed order.  This finite
    grid checks its ingredients independently: exact polynomial-semigroup
    cumulants are compared with the Perron rate and boundary constant, for
    reversible and nonreversible chains and for stationary and point starts.
    """
    q2 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    q3 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi3 = stationary(q3)
    kappa3 = np.array([1.1, 2.3, 3.0])
    pareto_shape = 6.5
    pareto_mean = 0.04
    pareto_lower = pareto_mean * (pareto_shape - 1.0) / pareto_shape
    pareto_moments = np.array(
        [pareto_shape * pareto_lower**order / (pareto_shape - order)
         for order in range(max_order + 1)]
    )
    pareto_components = np.array([0.5, 0.5])[:, None] * (
        pareto_moments[None, :]
    )
    cases = [
        (
            "two-state stationary",
            q2,
            np.array([0.5, 0.5]),
            np.array([0.04, 0.16]),
            np.array([0.8, 2.0]),
            np.array([0.04, 0.04]),
            {},
        ),
        (
            "two-state point",
            q2,
            np.array([0.5, 0.5]),
            np.array([0.04, 0.16]),
            np.array([0.8, 2.0]),
            np.array([0.04, 0.04]),
            {"initial_regime": 0, "initial_variance": 0.04},
        ),
        (
            "two-state Pareto moments",
            q2,
            np.array([0.5, 0.5]),
            np.array([0.04, 0.16]),
            np.array([0.8, 2.0]),
            np.array([0.04, 0.04]),
            {"initial_moment_components": pareto_components},
        ),
        (
            "three-state nonreversible",
            q3,
            pi3,
            kappa3 * np.array([0.035, 0.080, 0.050]),
            kappa3,
            np.array([0.20, 0.25, 0.22]) ** 2,
            {},
        ),
    ]
    speeds = 2.0 ** np.arange(7)
    maturities = np.linspace(0.0, 10.0, 21)
    gamma = 0.35
    results = {}
    for label, q0, pi, c, kappa, variance, start in cases:
        order_envelopes = np.zeros(max_order)
        terminal_remainders = np.zeros(max_order)
        for speed in speeds:
            rates, _ = integrated_variance_cumulant_rates(
                max_order, speed, q0, pi, c, kappa, variance
            )
            boundary, _, _, _ = integrated_variance_boundary_constants(
                max_order, speed, q0, pi, c, kappa, variance, **start
            )
            for maturity in maturities:
                exact = centered_integrated_cumulants(
                    max_order, maturity, speed, q0, pi, c, kappa,
                    variance, **start
                )
                remainder = exact - maturity * rates - boundary
                order_envelopes = np.maximum(
                    order_envelopes,
                    np.exp(gamma * maturity) * np.abs(remainder),
                )
                if maturity == maturities[-1]:
                    terminal_remainders = np.maximum(
                        terminal_remainders, np.abs(remainder)
                    )
        assert np.all(np.isfinite(order_envelopes))
        results[label] = (order_envelopes, terminal_remainders)

    # These are regression ceilings, not constants in the theorem.
    assert np.max(results["two-state stationary"][0]) < 1.60e-3
    assert np.max(results["two-state point"][0]) < 3.27e-2
    assert np.max(results["two-state Pareto moments"][0]) < 2.25e-2
    assert np.max(results["three-state nonreversible"][0]) < 4.11e-4

    print("5j. switching-rate-uniform fixed-order cumulant remainders")
    print(
        f"   tested orders 1--{max_order}, m=1,...,64, T in [0,10], "
        f"gamma={gamma:.2f}"
    )
    for label in results:
        envelope, terminal = results[label]
        print(
            f"   {label}: envelope "
            + " ".join(f"{value:.3e}" for value in envelope)
        )
        print(
            f"      T=10 remainder "
            + " ".join(f"{value:.3e}" for value in terminal)
        )
    return gamma, results


def verify_finite_moment_initial_jet(max_order=6):
    """Certify fixed-order cumulants for an initial law with no MGF.

    A Pareto initial variance with shape 13/2 has moments through degree six,
    but its seventh moment and every positive exponential moment are infinite.
    The finite polynomial jet nevertheless determines cumulants through order
    six and their exact long-maturity slopes and boundary constants.
    """
    assert max_order <= 6
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    variance = np.array([0.04, 0.04])
    speed = 8.0

    # For a Pareto variable with lower endpoint x_min and shape alpha,
    # E[X^j] = alpha*x_min^j/(alpha-j) for j < alpha.  This choice has
    # mean 0.04 while retaining moments only through degree six.
    alpha = 6.5
    mean = 0.04
    lower = mean * (alpha - 1.0) / alpha
    moments = np.array(
        [alpha * lower**order / (alpha - order)
         for order in range(max_order + 1)]
    )
    moment_components = pi[:, None] * moments[None, :]
    assert max_order < alpha < max_order + 1
    assert abs(moments[1] - mean) < 2e-16

    rates, _ = integrated_variance_cumulant_rates(
        max_order, speed, q0, pi, c, kappa, variance
    )
    boundary, _, _, _ = integrated_variance_boundary_constants(
        max_order,
        speed,
        q0,
        pi,
        c,
        kappa,
        variance,
        initial_moment_components=moment_components,
    )
    maturities = np.arange(4.0, 13.0, 2.0)
    remainders = []
    terminal = None
    for maturity in maturities:
        exact = centered_integrated_cumulants(
            max_order,
            maturity,
            speed,
            q0,
            pi,
            c,
            kappa,
            variance,
            initial_moment_components=moment_components,
        )
        terminal = exact - maturity * rates - boundary
        remainders.append(np.max(np.abs(terminal)))
    remainders = np.asarray(remainders)
    fitted_decay = -np.polyfit(maturities, np.log(remainders), 1)[0]

    assert np.all(np.diff(remainders) < 0.0)
    assert fitted_decay > 1.35
    assert remainders[3] < 2.4e-8

    print("5k. finite-moment initial jet without an MGF")
    print(
        f"   Pareto shape {alpha:.1f}, mean {mean:.4f}; moments 1--{max_order} "
        "finite, seventh moment and every positive MGF infinite"
    )
    print(
        "   max remainders at T=4,6,8,10,12: "
        + " ".join(f"{value:.3e}" for value in remainders)
    )
    print(
        f"   fitted decay {fitted_decay:.3f}; T=12 orderwise remainder "
        + " ".join(f"{value:.3e}" for value in np.abs(terminal))
    )
    return fitted_decay, remainders, terminal


def verify_independent_return_finite_jet(max_order=6):
    """Transport the variance jet to an independent log return.

    The initial variance is Pareto with moments only through order six, so
    neither this check nor the theorem can rely on a two-sided MGF.  A direct
    polynomial semigroup for (v, V, M) is compared with the triangular
    cumulant map obtained by substituting (s^2-s)/2 in the formal variance
    cumulant jet.
    """
    assert max_order <= 6
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    variance = np.array([0.04, 0.04])
    speed = 8.0

    alpha = 6.5
    initial_mean = 0.04
    lower = initial_mean * (alpha - 1.0) / alpha
    initial_moments = np.array(
        [alpha * lower**order / (alpha - order)
         for order in range(max_order + 1)]
    )
    moment_components = pi[:, None] * initial_moments[None, :]

    rates, _ = integrated_variance_cumulant_rates(
        max_order, speed, q0, pi, c, kappa, variance
    )
    _, invariant, _, _ = _stationary_polynomial_context(
        speed, q0, pi, c, kappa, variance, max_order
    )
    width = max_order + 1
    stationary_mean = sum(
        invariant[state * width + 1] for state in range(len(pi))
    )
    rates[0] = stationary_mean
    boundary, _, _, _ = integrated_variance_boundary_constants(
        max_order,
        speed,
        q0,
        pi,
        c,
        kappa,
        variance,
        initial_moment_components=moment_components,
    )
    return_rates = independent_logreturn_cumulants(rates)
    return_boundary = independent_logreturn_cumulants(boundary)

    (
        basis,
        a_c,
        a_kappa,
        v_d_vv,
        v_d_mm,
        v_d_mdv,
        v_d_z,
    ) = leveraged_operators(max_order)
    a_variance = 0.5 * v_d_vv
    averaged = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
        + 0.5 * v_d_mm
        + v_d_z
    )
    generator = full_generator(
        speed * q0,
        averaged,
        [a_c, a_kappa, a_variance, v_d_mdv],
        [c, kappa, variance, np.zeros(len(pi))],
    )
    initial = np.zeros(len(pi) * len(basis))
    for state in range(len(pi)):
        block = state * len(basis)
        for position, (m_power, v_power, z_power) in enumerate(basis):
            if m_power == 0 and z_power == 0:
                initial[block + position] = moment_components[
                    state, v_power
                ]

    payoffs = []
    for order in range(1, max_order + 1):
        payoff = np.zeros(len(basis))
        for z_power in range(order + 1):
            m_power = order - z_power
            payoff[basis.index((m_power, 0, z_power))] = (
                math.comb(order, z_power) * (-0.5) ** z_power
            )
        payoffs.append(payoff)

    direct_errors = []
    transported_remainder_errors = []
    return_remainders = []
    maturities = np.array([4.0, 8.0, 12.0])
    for maturity in maturities:
        semigroup = expm(maturity * generator)
        raw_return = np.array(
            [
                initial
                @ semigroup
                @ np.kron(np.ones(len(pi)), payoff)
                for payoff in payoffs
            ]
        )
        direct = cumulants_any_order(raw_return)
        variance_cumulants = centered_integrated_cumulants(
            max_order,
            maturity,
            speed,
            q0,
            pi,
            c,
            kappa,
            variance,
            initial_moment_components=moment_components,
        )
        variance_cumulants[0] += maturity * stationary_mean
        transported = independent_logreturn_cumulants(variance_cumulants)
        direct_errors.append(np.max(np.abs(direct - transported)))

        variance_remainder = (
            variance_cumulants - maturity * rates - boundary
        )
        return_remainder = (
            direct - maturity * return_rates - return_boundary
        )
        transported_remainder_errors.append(
            np.max(
                np.abs(
                    return_remainder
                    - independent_logreturn_cumulants(variance_remainder)
                )
            )
        )
        return_remainders.append(np.max(np.abs(return_remainder)))

    direct_errors = np.asarray(direct_errors)
    transported_remainder_errors = np.asarray(
        transported_remainder_errors
    )
    return_remainders = np.asarray(return_remainders)
    assert np.max(direct_errors) < 2e-11
    assert np.max(transported_remainder_errors) < 2e-11
    assert np.all(np.diff(return_remainders) < 0.0)

    # Recover the previously stated low-order formulas exactly.
    trial = np.arange(1.0, max_order + 1.0)
    mapped = independent_logreturn_cumulants(trial)
    assert abs(mapped[1] - (trial[0] + trial[1] / 4.0)) < 2e-15
    assert abs(mapped[2] - (-1.5 * trial[1] - trial[2] / 8.0)) < 2e-15
    assert abs(
        mapped[3]
        - (3.0 * trial[1] + 1.5 * trial[2] + trial[3] / 16.0)
    ) < 2e-15

    print("5l. independent-return finite jet without an MGF")
    print(
        f"   Pareto shape {alpha:.1f}; orders 1--{max_order}; "
        "formal substitution q(s)=(s^2-s)/2"
    )
    print(
        "   direct-semigroup identity errors T=4,8,12: "
        + " ".join(f"{value:.3e}" for value in direct_errors)
    )
    print(
        "   transported-remainder errors: "
        + " ".join(
            f"{value:.3e}" for value in transported_remainder_errors
        )
    )
    print(
        "   max return remainders: "
        + " ".join(f"{value:.3e}" for value in return_remainders)
    )
    return direct_errors, transported_remainder_errors, return_remainders


def leveraged_return_jet_context(
    max_order, speed, q0, pi, c, kappa, variance, eta
):
    """Polynomial generator and first two return-tilt derivatives.

    For X=M-int(v)/2 and eta_i=rho_i*xi_i, exponential conjugation gives

      H(theta)=L+theta B1+theta**2 B2/2,
      B1=eta_i*v*d_v-v/2,  B2=v.

    The matrices act on regime-resolved polynomials in v through degree
    ``max_order``.  This identity is used only as a finite formal jet, so no
    moment-generating function is required.
    """
    generator, invariant, constant, _ = _stationary_polynomial_context(
        speed, q0, pi, c, kappa, variance, max_order
    )
    _, a_kappa, _, multiply_v = polynomial_operators(max_order)
    states = len(pi)
    identity = np.eye(states)
    v_d_v = -a_kappa
    first_tilt = (
        np.kron(np.diag(eta), v_d_v)
        - 0.5 * np.kron(identity, multiply_v)
    )
    second_tilt = np.kron(identity, multiply_v)
    return generator, invariant, constant, first_tilt, second_tilt


def leveraged_return_cumulant_rates(
    max_order, speed, q0, pi, c, kappa, variance, eta
):
    """Exact long-run leveraged-return cumulant rates at fixed order."""
    (
        generator,
        invariant,
        constant,
        first_tilt,
        second_tilt,
    ) = leveraged_return_jet_context(
        max_order, speed, q0, pi, c, kappa, variance, eta
    )

    derivatives = [constant]
    rates = np.zeros(max_order + 1)
    for order in range(1, max_order + 1):
        source = order * first_tilt @ derivatives[order - 1]
        if order >= 2:
            source += (
                math.comb(order, 2)
                * second_tilt @ derivatives[order - 2]
            )
        rates[order] = invariant @ source
        forcing = source.copy()
        for index in range(1, order + 1):
            forcing -= (
                math.comb(order, index)
                * rates[index]
                * derivatives[order - index]
            )
        assert abs(invariant @ forcing) < 3e-10
        derivatives.append(
            _centered_polynomial_poisson(
                generator, invariant, constant, forcing
            )
        )
    return rates[1:], derivatives


def leveraged_return_boundary_constants(
    max_order,
    speed,
    q0,
    pi,
    c,
    kappa,
    variance,
    eta,
    initial_moment_components,
):
    """Exact arbitrary-start boundary cumulants of the leveraged return."""
    (
        generator,
        invariant,
        constant,
        first_tilt,
        second_tilt,
    ) = leveraged_return_jet_context(
        max_order, speed, q0, pi, c, kappa, variance, eta
    )
    rates, right = leveraged_return_cumulant_rates(
        max_order, speed, q0, pi, c, kappa, variance, eta
    )
    rates = np.r_[0.0, rates]
    size = len(constant)
    left = [invariant]
    for order in range(1, max_order + 1):
        forcing = -order * (left[order - 1] @ first_tilt)
        if order >= 2:
            forcing -= (
                math.comb(order, 2)
                * (left[order - 2] @ second_tilt)
            )
        for index in range(1, order + 1):
            forcing += (
                math.comb(order, index)
                * rates[index]
                * left[order - index]
            )
        normalization = -sum(
            math.comb(order, index)
            * (left[index] @ right[order - index])
            for index in range(order)
        )
        bordered = np.zeros((size + 1, size + 1))
        bordered[:size, :size] = generator.T
        bordered[:size, size] = invariant
        bordered[size, :size] = constant
        solution = np.linalg.solve(
            bordered, np.r_[forcing, normalization]
        )
        assert abs(solution[size]) < 2e-8
        assert np.linalg.norm(
            solution[:size] @ generator - forcing
        ) < 2e-8
        left.append(solution[:size])

    components = np.asarray(initial_moment_components, dtype=float)
    states = len(pi)
    if components.shape != (states, max_order + 1):
        raise ValueError("invalid initial moment-component array")
    initial = components.ravel()
    amplitude_derivatives = np.ones(max_order + 1)
    for order in range(1, max_order + 1):
        amplitude_derivatives[order] = sum(
            math.comb(order, index)
            * (initial @ right[index])
            * (left[order - index] @ constant)
            for index in range(order + 1)
        )
    boundary = np.zeros(max_order + 1)
    for order in range(1, max_order + 1):
        boundary[order] = amplitude_derivatives[order]
        for index in range(1, order):
            boundary[order] -= (
                math.comb(order - 1, index - 1)
                * boundary[index]
                * amplitude_derivatives[order - index]
            )
    return boundary[1:], right, left


def leveraged_return_cumulants(
    max_order,
    maturity,
    speed,
    q0,
    pi,
    c,
    kappa,
    variance,
    eta,
    initial_moment_components,
):
    """Exact finite return cumulants from the formal tilted polynomial jet."""
    (
        generator,
        _,
        constant,
        first_tilt,
        second_tilt,
    ) = leveraged_return_jet_context(
        max_order, speed, q0, pi, c, kappa, variance, eta
    )
    width = len(constant)
    jet = np.zeros(
        ((max_order + 1) * width, (max_order + 1) * width)
    )
    for order in range(max_order + 1):
        block = slice(order * width, (order + 1) * width)
        jet[block, block] = generator
        if order:
            previous = slice(
                (order - 1) * width, order * width
            )
            jet[block, previous] = order * first_tilt
        if order >= 2:
            previous_two = slice(
                (order - 2) * width, (order - 1) * width
            )
            jet[block, previous_two] = (
                math.comb(order, 2) * second_tilt
            )
    initial_function = np.zeros((max_order + 1) * width)
    initial_function[:width] = constant
    solution = expm(maturity * jet) @ initial_function
    initial = np.asarray(initial_moment_components, dtype=float).ravel()
    raw = np.array([
        initial @ solution[order * width:(order + 1) * width]
        for order in range(1, max_order + 1)
    ])
    return cumulants_any_order(raw)


def verify_leveraged_return_finite_jet(max_order=6):
    """Certify the all-order leveraged-return slope/boundary theorem."""
    q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
    pi = np.array([0.5, 0.5])
    c = np.array([0.04, 0.16])
    kappa = np.array([0.8, 2.0])
    xi = np.array([0.20, 0.20])
    variance = xi**2
    rho = np.array([-0.8, 0.35])
    eta = rho * xi

    alpha = 6.5
    initial_mean = 0.04
    lower = initial_mean * (alpha - 1.0) / alpha
    initial_moments = np.array([
        alpha * lower**order / (alpha - order)
        for order in range(max_order + 1)
    ])
    initial_components = pi[:, None] * initial_moments[None, :]

    # A separate joint (M,v,z) polynomial system checks the tilted-generator
    # jet at one speed and three maturities.
    (
        basis,
        a_c,
        a_kappa,
        v_d_vv,
        v_d_mm,
        v_d_mdv,
        v_d_z,
    ) = leveraged_operators(max_order)
    joint_averaged = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + 0.5 * (pi @ variance) * v_d_vv
        + (pi @ eta) * v_d_mdv
        + 0.5 * v_d_mm
        + v_d_z
    )
    joint_generator = full_generator(
        8.0 * q0,
        joint_averaged,
        [a_c, a_kappa, 0.5 * v_d_vv, v_d_mdv],
        [c, kappa, variance, eta],
    )
    joint_initial = np.zeros(len(pi) * len(basis))
    for state in range(len(pi)):
        offset = state * len(basis)
        for position, (m_power, v_power, z_power) in enumerate(basis):
            if m_power == 0 and z_power == 0:
                joint_initial[offset + position] = initial_components[
                    state, v_power
                ]
    payoffs = []
    for order in range(1, max_order + 1):
        payoff = np.zeros(len(basis))
        for z_power in range(order + 1):
            m_power = order - z_power
            payoff[basis.index((m_power, 0, z_power))] = (
                math.comb(order, z_power) * (-0.5) ** z_power
            )
        payoffs.append(np.kron(np.ones(len(pi)), payoff))

    independent_errors = []
    for maturity in (4.0, 8.0, 12.0):
        semigroup = expm(maturity * joint_generator)
        raw = np.array([
            joint_initial @ semigroup @ payoff for payoff in payoffs
        ])
        joint_cumulants = cumulants_any_order(raw)
        tilted_cumulants = leveraged_return_cumulants(
            max_order,
            maturity,
            8.0,
            q0,
            pi,
            c,
            kappa,
            variance,
            eta,
            initial_components,
        )
        independent_errors.append(
            np.max(np.abs(joint_cumulants - tilted_cumulants))
        )
    independent_errors = np.asarray(independent_errors)
    assert np.max(independent_errors) < 2e-13

    speeds = 2.0 ** np.arange(7)
    maturities = np.linspace(0.0, 10.0, 21)
    gamma = 0.35
    envelope = np.zeros(max_order)
    terminal = np.zeros(max_order)
    for speed in speeds:
        rates, _ = leveraged_return_cumulant_rates(
            max_order, speed, q0, pi, c, kappa, variance, eta
        )
        boundary, _, _ = leveraged_return_boundary_constants(
            max_order,
            speed,
            q0,
            pi,
            c,
            kappa,
            variance,
            eta,
            initial_components,
        )
        for maturity in maturities:
            exact = leveraged_return_cumulants(
                max_order,
                maturity,
                speed,
                q0,
                pi,
                c,
                kappa,
                variance,
                eta,
                initial_components,
            )
            remainder = exact - maturity * rates - boundary
            envelope = np.maximum(
                envelope,
                np.exp(gamma * maturity) * np.abs(remainder),
            )
            if maturity == maturities[-1]:
                terminal = np.maximum(terminal, np.abs(remainder))
    assert np.all(np.isfinite(envelope))
    assert np.max(envelope) < 0.068

    print("5m. leveraged-return finite jet without an MGF")
    print(
        f"   switching rho={tuple(float(value) for value in rho)}; "
        f"Pareto shape {alpha:.1f}; "
        f"orders 1--{max_order}, m=1,...,64, T in [0,10]"
    )
    print(
        "   joint/tilted identity errors T=4,8,12: "
        + " ".join(f"{value:.3e}" for value in independent_errors)
    )
    print(
        "   uniform rescaled envelope: "
        + " ".join(f"{value:.3e}" for value in envelope)
    )
    print(
        "   T=10 remainder: "
        + " ".join(f"{value:.3e}" for value in terminal)
    )
    return independent_errors, envelope, terminal


def verify_averaged_admissibility(pi, c, xi, rho):
    """Sharp Feller-margin and effective-correlation checks."""
    variance = xi**2
    eta = rho * xi
    effective_xi = math.sqrt(pi @ variance)
    effective_rho = (pi @ eta) / effective_xi
    correlation_envelope = (pi @ xi) / effective_xi
    coefficient_of_variation = math.sqrt(
        pi @ (xi - pi @ xi) ** 2
    ) / (pi @ xi)

    # The averaged boundary margin is exactly the stationary average of the
    # statewise margins. Statewise Feller admissibility therefore implies
    # averaged admissibility, although the converse need not hold.
    averaged_margin = 2 * (pi @ c) - pi @ variance
    statewise_margin_average = pi @ (2 * c - variance)
    assert abs(averaged_margin - statewise_margin_average) < 2e-16
    assert averaged_margin > 0.0

    # For fixed positive xi, the full attainable interval is sharper than
    # |rho_eff| <= 1. Its endpoints occur only at rho_i = +/-1 for all i.
    assert abs(effective_rho) <= correlation_envelope + 2e-16
    assert abs(
        correlation_envelope
        - 1 / math.sqrt(1 + coefficient_of_variation**2)
    ) < 2e-15
    endpoint_plus = (pi @ xi) / effective_xi
    endpoint_minus = -(pi @ xi) / effective_xi
    assert abs(endpoint_plus - correlation_envelope) < 2e-16
    assert abs(endpoint_minus + correlation_envelope) < 2e-16
    target = 0.37 * correlation_envelope
    assert abs((pi @ (0.37 * xi)) / effective_xi - target) < 2e-16

    # Independent random stress test of the sharp envelope and covariance
    # determinant. The latter is a_bar-eta_bar^2 in units of v^2.
    rng = np.random.default_rng(20260927)
    weights = rng.dirichlet(np.ones(5), size=10000)
    xis = rng.uniform(0.03, 1.25, size=(10000, 5))
    rhos = rng.uniform(-1.0, 1.0, size=(10000, 5))
    mean_xi = np.sum(weights * xis, axis=1)
    mean_xi2 = np.sum(weights * xis**2, axis=1)
    mean_eta = np.sum(weights * xis * rhos, axis=1)
    random_effective_rho = mean_eta / np.sqrt(mean_xi2)
    random_envelope = mean_xi / np.sqrt(mean_xi2)
    envelope_slack = random_envelope - np.abs(random_effective_rho)
    envelope_excess = max(0.0, -np.min(envelope_slack))
    determinant_floor = np.min(mean_xi2 - mean_eta**2)
    assert envelope_excess < 2e-15
    assert determinant_floor > 0.0
    return {
        "effective_rho": effective_rho,
        "envelope": correlation_envelope,
        "cv": coefficient_of_variation,
        "margin": averaged_margin,
        "envelope_excess": envelope_excess,
        "envelope_slack": np.min(envelope_slack),
        "determinant_floor": determinant_floor,
    }


def main():
    long_maturity_residuals = verify_long_maturity_nonuniformity()
    uniform_mean_rates = verify_uniform_spectral_mean()
    expansion_rates = verify_stationary_mean_expansion()
    variance_rate_results = verify_long_run_variance_rate()
    uniform_remainder_results = verify_uniform_variance_remainder()
    third_rate_results = verify_long_run_third_cumulant_rate()
    fourth_rate_results = verify_long_run_fourth_cumulant_rate()
    all_order_rate_results = verify_all_fixed_order_cumulant_rates()
    all_order_rate_correction_results = (
        verify_all_fixed_order_rate_corrections()
    )
    all_order_second_rate_results = (
        verify_all_fixed_order_second_rate_corrections()
    )
    all_order_intercept_results = verify_all_fixed_order_cumulant_intercepts()
    uniform_all_order_results = (
        verify_uniform_fixed_order_cumulant_remainders()
    )
    finite_moment_jet_results = verify_finite_moment_initial_jet()
    independent_return_jet_results = verify_independent_return_finite_jet()
    leveraged_return_jet_results = verify_leveraged_return_finite_jet()
    q0 = np.array(
        [
            [-3.0, 2.7, 0.3],
            [0.2, -2.2, 2.0],
            [2.4, 0.4, -2.8],
        ]
    )
    pi = stationary(q0)
    kappa = np.array([1.1, 2.3, 3.0])
    theta = np.array([0.035, 0.080, 0.050])
    c = kappa * theta
    xi = np.array([0.20, 0.25, 0.22])
    variance = xi**2
    assert np.all(2 * c > variance)

    a_c, a_kappa, a_variance, multiply_v = polynomial_operators()
    operators = [a_c, a_kappa, a_variance]
    lbar = (
        (pi @ c) * a_c
        + (pi @ kappa) * a_kappa
        + (pi @ variance) * a_variance
    )
    payoff = np.array([0.7, -0.2, 0.3, -0.1, 0.04, -0.005])
    maturity = 0.7

    k0 = gk(q0, [c, kappa, variance])
    identity = np.eye(len(payoff))
    derivative = a_c
    derivative2 = derivative @ derivative
    derivative3 = derivative2 @ derivative
    derivative4 = derivative3 @ derivative
    explicit = (
        (k0[1, 1] * multiply_v - k0[0, 1] * identity) @ derivative
        + (
            (k0[0, 0] + 0.5 * k0[0, 2]) * identity
            - (k0[0, 1] + k0[1, 0]) * multiply_v
            - (0.5 * k0[1, 2] + k0[2, 1]) * multiply_v
            + k0[1, 1] * (multiply_v @ multiply_v)
        )
        @ derivative2
        + (
            0.5 * (k0[0, 2] + k0[2, 0] + k0[2, 2]) * multiply_v
            - 0.5 * (k0[1, 2] + k0[2, 1]) * (multiply_v @ multiply_v)
        )
        @ derivative3
        + 0.25 * k0[2, 2] * (multiply_v @ multiply_v) @ derivative4
    )
    abstract = effective_generator(np.zeros_like(lbar), operators, k0)
    formula_error = np.max(abs(explicit - abstract))
    assert formula_error < 2e-15

    print("6. switched coordinates and the explicit CIR corrector")
    print(f"   stationary law {pi}")
    print(f"   kappa*theta {c}; average theta is {(pi @ c) / (pi @ kappa):.8f}")
    print(f"   xi^2 {variance}; effective xi is {math.sqrt(pi @ variance):.8f}")
    print(f"   operator formula error {formula_error:.2e}")
    print(f"   K(c,kappa,xi^2)=\n{k0}")

    errors = {"averaged": [], "symmetric": [], "full": []}
    forward_values = []
    for speed in SPEEDS:
        generator = full_generator(speed * q0, lbar, operators, [c, kappa, variance])
        truth = np.kron(pi, np.eye(len(payoff))) @ (
            expm(maturity * generator) @ np.kron(np.ones(len(pi)), payoff)
        )
        full_correction = abstract / speed
        symmetric_correction = effective_generator(
            np.zeros_like(lbar), operators, k0 / speed, "sym"
        )
        averaged = expm(maturity * lbar) @ payoff
        full = first_order(lbar, full_correction, maturity, payoff)
        symmetric = first_order(lbar, symmetric_correction, maturity, payoff)
        errors["averaged"].append(np.linalg.norm(truth - averaged))
        errors["symmetric"].append(np.linalg.norm(truth - symmetric))
        errors["full"].append(np.linalg.norm(truth - full))
        forward_values.append(truth)

    print("7. polynomial semigroup check")
    for name, values in errors.items():
        print(
            f"   {name:9s}: "
            + " ".join(f"{value:.3e}" for value in values)
            + f"  rate {rate(values):.3f}"
        )
    assert 0.8 < rate(errors["averaged"]) < 1.2
    assert 0.8 < rate(errors["symmetric"]) < 1.2
    assert 1.8 < rate(errors["full"]) < 2.2

    q_reverse = reversed_generator(q0, pi)
    k_reverse = gk(q_reverse, [c, kappa, variance])
    assert np.max(abs(k_reverse - k0.T)) < 2e-14
    reverse_values = []
    direction_errors = []
    for speed, forward_value in zip(SPEEDS, forward_values):
        generator = full_generator(
            speed * q_reverse, lbar, operators, [c, kappa, variance]
        )
        reverse_value = np.kron(pi, np.eye(len(payoff))) @ (
            expm(maturity * generator) @ np.kron(np.ones(len(pi)), payoff)
        )
        reverse_values.append(reverse_value)
        direction_correction = effective_generator(
            np.zeros_like(lbar), operators, (k0 - k0.T) / speed
        )
        predicted = (
            first_order(lbar, direction_correction, maturity, payoff)
            - expm(maturity * lbar) @ payoff
        )
        direction_errors.append(
            np.linalg.norm((forward_value - reverse_value) - predicted)
        )
    direction_rate = rate(direction_errors)
    assert 1.8 < direction_rate < 2.2
    anti = 0.5 * (k0 - k0.T)
    print("8. cycle reversal")
    print(
        "   antisymmetric coefficients "
        f"c/kappa={anti[0, 1]:+.10f}, c/xi^2={anti[0, 2]:+.10f}, "
        f"kappa/xi^2={anti[1, 2]:+.10f}"
    )
    print(
        "   corrected forward-minus-reverse residual: "
        + " ".join(f"{value:.3e}" for value in direction_errors)
        + f"  rate {direction_rate:.3f}"
    )
    basis, a_c_z, a_kappa_z, v_d_vv, v_d_z = integrated_variance_operators()
    a_variance_z = 0.5 * v_d_vv
    lbar_z = (
        (pi @ c) * a_c_z
        + (pi @ kappa) * a_kappa_z
        + (pi @ variance) * a_variance_z
        + v_d_z
    )
    correction_z = effective_generator(
        np.zeros_like(lbar_z), [a_c_z, a_kappa_z, a_variance_z], k0
    )
    initial = np.array([0.06**i * 0.0**j for i, j in basis])
    moment_payoffs = []
    for power in range(1, 5):
        payoff_z = np.zeros(len(basis))
        payoff_z[basis.index((0, power))] = 1.0
        moment_payoffs.append(payoff_z)

    averaged_moments = np.array(
        [initial @ (expm(maturity * lbar_z) @ payoff_z) for payoff_z in moment_payoffs]
    )
    corrected_unit = np.array(
        [
            initial @ first_order(lbar_z, correction_z, maturity, payoff_z)
            for payoff_z in moment_payoffs
        ]
    )
    moment_correction = corrected_unit - averaged_moments
    averaged_cumulants = cumulants(averaged_moments)
    cumulant_first = cumulant_correction(averaged_moments, moment_correction)
    martingale_average = 3 * averaged_cumulants[1]
    martingale_first = 3 * cumulant_first[1]
    logreturn_average = (
        3 * averaged_cumulants[1]
        + 1.5 * averaged_cumulants[2]
        + averaged_cumulants[3] / 16
    )
    logreturn_first = (
        3 * cumulant_first[1] + 1.5 * cumulant_first[2] + cumulant_first[3] / 16
    )

    cumulant_errors = {"martingale": [], "log return": []}
    averaged_errors = {"martingale": [], "log return": []}
    for speed in SPEEDS:
        generator = full_generator(
            speed * q0,
            lbar_z,
            [a_c_z, a_kappa_z, a_variance_z],
            [c, kappa, variance],
        )
        exact_moments = []
        for payoff_z in moment_payoffs:
            coefficients = np.kron(pi, np.eye(len(basis))) @ (
                expm(maturity * generator) @ np.kron(np.ones(len(pi)), payoff_z)
            )
            exact_moments.append(initial @ coefficients)
        exact_cumulants = cumulants(np.array(exact_moments))
        exact_martingale = 3 * exact_cumulants[1]
        exact_logreturn = (
            3 * exact_cumulants[1]
            + 1.5 * exact_cumulants[2]
            + exact_cumulants[3] / 16
        )
        averaged_errors["martingale"].append(abs(exact_martingale - martingale_average))
        averaged_errors["log return"].append(abs(exact_logreturn - logreturn_average))
        cumulant_errors["martingale"].append(
            abs(exact_martingale - martingale_average - martingale_first / speed)
        )
        cumulant_errors["log return"].append(
            abs(exact_logreturn - logreturn_average - logreturn_first / speed)
        )

    print("9. integrated-variance cumulants")
    print(
        f"   kappa4(M): averaged {martingale_average:.10f}, coefficient {martingale_first:+.10f}"
    )
    print(
        f"   kappa4(X): averaged {logreturn_average:.10f}, coefficient {logreturn_first:+.10f}"
    )
    for name in cumulant_errors:
        print(
            f"   {name:10s}: averaged rate {rate(averaged_errors[name]):.3f}; "
            f"corrected residual "
            + " ".join(f"{value:.3e}" for value in cumulant_errors[name])
            + f"  rate {rate(cumulant_errors[name]):.3f}"
        )
        assert 0.8 < rate(averaged_errors[name]) < 1.2
        assert 1.8 < rate(cumulant_errors[name]) < 2.2

    (
        basis_x,
        a_c_x,
        a_kappa_x,
        v_d_vv_x,
        v_d_mm,
        v_d_mdv,
        v_d_z_x,
    ) = leveraged_operators()
    a_variance_x = 0.5 * v_d_vv_x
    operators_x = [a_c_x, a_kappa_x, a_variance_x, v_d_mdv]
    initial_x = np.array([0.0**i * 0.06**j * 0.0**k for i, j, k in basis_x])
    payoff_x = []
    for power in range(1, 5):
        payoff = np.zeros(len(basis_x))
        for z_power in range(power + 1):
            m_power = power - z_power
            payoff[basis_x.index((m_power, 0, z_power))] = (
                math.comb(power, z_power) * (-0.5) ** z_power
            )
        payoff_x.append(payoff)

    def leverage_case(eta):
        """Fourth-cumulant expansion for cross coefficient eta=rho*xi."""
        features_x = [c, kappa, variance, eta]
        k0_x = gk(q0, features_x)
        lbar_x = (
            (pi @ c) * a_c_x
            + (pi @ kappa) * a_kappa_x
            + (pi @ variance) * a_variance_x
            + 0.5 * v_d_mm
            + (pi @ eta) * v_d_mdv
            + v_d_z_x
        )
        correction_x = effective_generator(
            np.zeros_like(lbar_x), operators_x, k0_x
        )
        averaged_raw_x = np.array(
            [initial_x @ (expm(maturity * lbar_x) @ payoff) for payoff in payoff_x]
        )
        corrected_unit_x = np.array(
            [
                initial_x @ first_order(lbar_x, correction_x, maturity, payoff)
                for payoff in payoff_x
            ]
        )
        correction_raw_x = corrected_unit_x - averaged_raw_x
        averaged_kappa_x = cumulants(averaged_raw_x)[3]
        correction_kappa_x = cumulant_correction(
            averaged_raw_x, correction_raw_x
        )[3]

        averaged_x_errors = []
        corrected_x_errors = []
        for speed in SPEEDS:
            generator_x = full_generator(
                speed * q0, lbar_x, operators_x, features_x
            )
            exact_raw_x = []
            for payoff in payoff_x:
                coefficients = np.kron(pi, np.eye(len(basis_x))) @ (
                    expm(maturity * generator_x)
                    @ np.kron(np.ones(len(pi)), payoff)
                )
                exact_raw_x.append(initial_x @ coefficients)
            exact_kappa_x = cumulants(np.array(exact_raw_x))[3]
            averaged_x_errors.append(abs(exact_kappa_x - averaged_kappa_x))
            corrected_x_errors.append(
                abs(exact_kappa_x - averaged_kappa_x - correction_kappa_x / speed)
            )
        result = (
            averaged_kappa_x,
            correction_kappa_x,
            averaged_x_errors,
            corrected_x_errors,
        )
        assert 0.8 < rate(averaged_x_errors) < 1.2
        assert 1.8 < rate(corrected_x_errors) < 2.2
        return result, k0_x

    leverage_results = {}
    for rho in (0.0, -0.7):
        leverage_results[f"rho={rho:+.1f}"] = leverage_case(rho * xi)[0]

    # For fixed rho, the minimal eta feature is exactly equivalent to using
    # xi as the feature and absorbing rho into the operator.
    rho_fixed = -0.7
    correction_eta_fixed = effective_generator(
        np.zeros_like(a_c_x),
        operators_x,
        gk(q0, [c, kappa, variance, rho_fixed * xi]),
    )
    correction_xi_fixed = effective_generator(
        np.zeros_like(a_c_x),
        [a_c_x, a_kappa_x, a_variance_x, rho_fixed * v_d_mdv],
        gk(q0, [c, kappa, variance, xi]),
    )
    fixed_coordinate_error = np.max(
        abs(correction_eta_fixed - correction_xi_fixed)
    )
    assert fixed_coordinate_error < 2e-15

    rho_switch = np.array([-0.80, -0.25, 0.35])
    eta_switch = rho_switch * xi
    admissibility = verify_averaged_admissibility(pi, c, xi, rho_switch)
    switched_result, k0_eta = leverage_case(eta_switch)
    leverage_results["switched rho"] = switched_result
    effective_xi = math.sqrt(pi @ variance)
    effective_rho = (pi @ eta_switch) / effective_xi
    assert np.all(np.abs(rho_switch) <= 1.0)
    assert abs(effective_rho) <= 1.0
    assert abs(effective_rho - admissibility["effective_rho"]) < 2e-16

    # Reversal transposes the four-feature Green--Kubo matrix.  The new
    # directional terms are K^anti_{c,eta}[A_c,A_eta] and
    # K^anti_{a,eta}[A_a,A_eta]; A_kappa commutes with A_eta.
    k0_eta_reverse = gk(q_reverse, [c, kappa, variance, eta_switch])
    eta_reversal_error = np.max(abs(k0_eta_reverse - k0_eta.T))
    assert eta_reversal_error < 2e-14
    commutator_c_eta = a_c_x @ v_d_mdv - v_d_mdv @ a_c_x
    commutator_kappa_eta = a_kappa_x @ v_d_mdv - v_d_mdv @ a_kappa_x
    commutator_a_eta = a_variance_x @ v_d_mdv - v_d_mdv @ a_variance_x
    index_x = {powers: position for position, powers in enumerate(basis_x)}
    expected_c_eta = np.zeros_like(commutator_c_eta)
    expected_a_eta = np.zeros_like(commutator_a_eta)
    for column, (m_power, v_power, z_power) in enumerate(basis_x):
        if m_power and v_power:
            target = (m_power - 1, v_power - 1, z_power)
            expected_c_eta[index_x[target], column] = m_power * v_power
        if m_power and v_power >= 2:
            target = (m_power - 1, v_power - 1, z_power)
            expected_a_eta[index_x[target], column] = (
                0.5 * m_power * v_power * (v_power - 1)
            )
    assert np.max(abs(commutator_c_eta - expected_c_eta)) < 2e-14
    assert np.linalg.norm(commutator_kappa_eta) < 2e-14
    assert np.max(abs(commutator_a_eta - expected_a_eta)) < 2e-14
    commutator_error = max(
        np.max(abs(commutator_c_eta - expected_c_eta)),
        np.linalg.norm(commutator_kappa_eta),
        np.max(abs(commutator_a_eta - expected_a_eta)),
    )

    independent = leverage_results["rho=+0.0"]
    assert abs(independent[0] - logreturn_average) < 2e-15
    assert abs(independent[1] - logreturn_first) < 2e-15
    print("10. simultaneous switching with regime-dependent leverage")
    print(
        f"   rho states {rho_switch}; eta=rho*xi {eta_switch}; "
        f"effective rho {effective_rho:+.8f}"
    )
    print(
        f"   sharp correlation envelope +/-{admissibility['envelope']:.8f}; "
        f"CV(xi) {admissibility['cv']:.8f}; averaged Feller margin "
        f"{admissibility['margin']:.8f}"
    )
    print(
        f"   10000-case envelope excess {admissibility['envelope_excess']:.2e}; "
        f"minimum slack {admissibility['envelope_slack']:.3e}; "
        f"minimum covariance determinant {admissibility['determinant_floor']:.3e}"
    )
    print(
        f"   fixed-rho coordinate error {fixed_coordinate_error:.2e}; "
        f"commutator error {commutator_error:.2e}; "
        f"reversal error {eta_reversal_error:.2e}"
    )
    for label, result in leverage_results.items():
        average, coefficient, average_errors, corrected_errors = result
        print(
            f"   {label:12s}: averaged {average:.10f}, coefficient {coefficient:+.10f}; "
            f"rates {rate(average_errors):.3f}/{rate(corrected_errors):.3f}"
        )
        print("      corrected residual " + " ".join(f"{x:.3e}" for x in corrected_errors))

    print(
        f"PASS: full rate {rate(errors['full']):.3f}, direction rate {direction_rate:.3f}, "
        f"cumulant rates {rate(cumulant_errors['martingale']):.3f}/"
        f"{rate(cumulant_errors['log return']):.3f}, leverage rate "
        f"{rate(leverage_results['switched rho'][3]):.3f}, long-maturity residual "
        f"{long_maturity_residuals[-1]:.10f}, uniform mean rates "
        f"{uniform_mean_rates[0]:.3f}/{uniform_mean_rates[1]:.3f}, "
        f"growing-window rate {expansion_rates[1][3]:.3f}, "
        f"variance-rate/intercept orders {variance_rate_results[1]:.3f}/"
        f"{variance_rate_results[4]:.3f}, uniform remainder gamma "
        f"{uniform_remainder_results[0]:.1f}, fixed-order uniform gamma "
        f"{uniform_all_order_results[0]:.2f}, third rate/intercept convergence "
        f"{third_rate_results[0]:.3f}/{third_rate_results[3]:.3f}, "
        f"fourth-rate convergence {fourth_rate_results[0]:.3f}, "
        f"all-order corrected-rate floor "
        f"{min(all_order_rate_correction_results[2]):.3f}, "
        f"twice-corrected floor "
        f"{min(all_order_second_rate_results[1]):.3f}, finite-moment jet decay "
        f"{finite_moment_jet_results[0]:.3f}, independent-return identity "
        f"{np.max(independent_return_jet_results[0]):.1e}, "
        f"leveraged-return identity "
        f"{np.max(leveraged_return_jet_results[0]):.1e}"
    )


if __name__ == "__main__":
    main()
