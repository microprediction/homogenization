"""Certificate for the exact Poisson--Beta CEV occupation-time mixture.

For a symmetric two-state chain with switching rate ``lam``, the jump count on
``[0,T]`` is Poisson(lam*T).  Conditional on that count, the normalized holding
times are uniform spacings, hence Dirichlet.  Starting in state 1, its occupation
fraction is therefore

    1                              if N = 0,
    Beta(k + 1, k)                 if N = 2k >= 2,
    Beta(k + 1, k + 1)             if N = 2k + 1.

Starting in state 2 reflects the fraction about 1/2.  When r=q, the CEV random
clock is affine in this occupation fraction.  The script compares the resulting
Poisson--Beta price with the independent moment-ODE/Chebyshev reconstruction
used by the model page and verifies the certified Poisson-tail truncation bound.
For unequal rates it also checks the exact first two occupation moments and the
resulting finite-rate Taylor bounds for the CEV price.  Exponentially weighted
occupation moments extend those bounds to nonzero carry without asserting an
occupation-time representation for the full clock law.  For nonzero carry, an
exact Kummer-function Laplace transform of the weighted occupation clock is
also checked against the independent time-inhomogeneous Feynman--Kac system.
Finally, the exact Bessel density certifies the sharp ``kappa**(-1/2)``
Wasserstein rate for merely Lipschitz clock payoffs.  A complex Feynman--Kac
calculation verifies the corresponding weighted-clock central limit theorem
at nonzero carry, the first characteristic correction for both stationary and
specified initial regimes, and the stationary second characteristic
correction obtained from the fourth cumulant and the variance endpoint term.
The same calculation is continued through second order for fixed initial
regimes, both around the stationary clock and around the exact conditional
mean.
It also checks the joint first correction for two different deterministic
clock loadings.  This multivariate check makes the limiting covariance a Gram
matrix and separates its rank from any single fixed-loading calculation.
These sharp ``kappa**(-1/2)`` scales contrast with the ``kappa**(-1)``
centered Taylor bound for twice differentiable payoffs.
"""

import math

import mpmath as mp
import numpy as np
from numpy.polynomial import chebyshev as ch
from scipy.integrate import quad, solve_ivp, tplquad
from scipy.special import iv, roots_jacobi
from scipy.stats import ncx2, poisson


S0 = 100.0
R = Q = 0.02
BETA = 0.6
K = 100.0
T = 1.0
SIGMA = (2.5, 1.2)
LAM = 10.0


def cev_call(total_variance, rate=R, dividend=Q):
    """Undiscounted driftless-forward CEV call for a given total variance."""
    forward = S0 * math.exp((rate - dividend) * T)
    den = (1.0 - BETA) ** 2 * total_variance
    x = K ** (2.0 * (1.0 - BETA)) / den
    y = forward ** (2.0 * (1.0 - BETA)) / den
    degrees = 1.0 / (1.0 - BETA)
    return forward * (1.0 - ncx2.cdf(x, degrees + 2.0, y)) - K * ncx2.cdf(y, degrees, x)


def beta_expectation(function, a, b, nodes=80):
    """Gauss--Jacobi evaluation of E[f(B)] for B ~ Beta(a,b)."""
    z, weights = roots_jacobi(nodes, b - 1.0, a - 1.0)
    values = np.array([function(0.5 * (point + 1.0)) for point in z])
    return float(weights @ values / weights.sum())


def conditional_clock_price(jump_count, start):
    """E[C(V)|N=n,Y_0=start], with states numbered 0 and 1."""
    s1, s2 = SIGMA[0] ** 2, SIGMA[1] ** 2

    def price_from_fraction(fraction):
        return cev_call(T * (s2 + (s1 - s2) * fraction))

    if jump_count == 0:
        return price_from_fraction(1.0 if start == 0 else 0.0)
    if jump_count % 2:
        k = (jump_count - 1) // 2
        return beta_expectation(price_from_fraction, k + 1, k + 1)
    k = jump_count // 2
    a, b = (k + 1, k) if start == 0 else (k, k + 1)
    return beta_expectation(price_from_fraction, a, b)


def poisson_beta_price(start, max_jumps):
    """Discounted partial Poisson--Beta mixture through ``max_jumps``."""
    mu = LAM * T
    total = 0.0
    probability = math.exp(-mu)
    for n in range(max_jumps + 1):
        if n:
            probability *= mu / n
        total += probability * conditional_clock_price(n, start)
    return math.exp(-R * T) * total


def moment_ode_price(start, degree=32):
    """Independent Chebyshev/moment-ODE reconstruction of the clock mixture."""
    s1, s2 = SIGMA[0] ** 2, SIGMA[1] ** 2
    vbar = 0.5 * (s1 + s2) * T
    half_width = 0.5 * (s1 - s2) * T
    nodes = np.cos(np.pi * (np.arange(72) + 0.5) / 72)
    coefficients = ch.chebfit(nodes, [cev_call(vbar + half_width * x) for x in nodes], degree)
    polynomial = ch.cheb2poly(coefficients)
    generator = np.array([[-LAM, LAM], [LAM, -LAM]])
    signs = np.array([1.0, -1.0])

    def rhs(_, flat):
        moments = flat.reshape(degree + 1, 2)
        out = np.zeros_like(moments)
        for order in range(degree + 1):
            out[order] = generator @ moments[order]
            if order:
                out[order] += order * signs * moments[order - 1] / T
        return out.ravel()

    initial = np.zeros((degree + 1, 2))
    initial[0] = 1.0
    moments = solve_ivp(rhs, (0.0, T), initial.ravel(), method="DOP853", rtol=2e-13, atol=2e-15).y[:, -1]
    moments = moments.reshape(degree + 1, 2)
    return math.exp(-R * T) * float(polynomial @ moments[:, start])


def unequal_occupation_density(u, start, rate_12, rate_21):
    """Interior density of state-1 occupation time for arbitrary two-state rates."""
    z = rate_12 * rate_21 * u * (T - u)
    root = math.sqrt(max(z, 0.0))
    i0 = iv(0, 2.0 * root)
    i1_over_root = iv(1, 2.0 * root) / root if root > 1e-10 else 1.0 + 0.5 * z
    exponential = math.exp(-rate_12 * u - rate_21 * (T - u))
    if start == 0:
        bracket = rate_12 * i0 + rate_12 * rate_21 * u * i1_over_root
    else:
        bracket = rate_21 * i0 + rate_12 * rate_21 * (T - u) * i1_over_root
    return exponential * bracket


def bessel_price(start, rate_12, rate_21):
    """Exact zero-carry price from the unequal-rate Bessel occupation density."""
    s1, s2 = SIGMA[0] ** 2, SIGMA[1] ** 2

    def clock_price(u):
        return cev_call(s2 * T + (s1 - s2) * u)

    atom = math.exp(-(rate_12 if start == 0 else rate_21) * T)
    atom_price = clock_price(T if start == 0 else 0.0)
    interior = quad(
        lambda u: unequal_occupation_density(u, start, rate_12, rate_21) * clock_price(u),
        0.0,
        T,
        epsabs=2e-12,
        epsrel=2e-12,
        limit=250,
    )[0]
    return math.exp(-R * T) * (atom * atom_price + interior)


def unequal_moment_ode_price(start, rate_12, rate_21, degree=28):
    """Independent polynomial-moment reconstruction for arbitrary transition rates."""
    s1, s2 = SIGMA[0] ** 2, SIGMA[1] ** 2
    nodes = np.cos(np.pi * (np.arange(72) + 0.5) / 72)
    coefficients = ch.chebfit(
        nodes,
        [cev_call(T * (s2 + 0.5 * (s1 - s2) * (x + 1.0))) for x in nodes],
        degree,
    )
    polynomial = ch.cheb2poly(coefficients)
    generator = np.array([[-rate_12, rate_12], [rate_21, -rate_21]])
    state_one = np.array([1.0, 0.0])

    def rhs(_, flat):
        moments = flat.reshape(degree + 1, 2)
        out = np.zeros_like(moments)
        for order in range(degree + 1):
            out[order] = generator @ moments[order]
            if order:
                out[order] += order * (2.0 / T) * state_one * moments[order - 1]
        return out.ravel()

    initial = np.empty((degree + 1, 2))
    for order in range(degree + 1):
        initial[order] = (-1.0) ** order
    moments = solve_ivp(rhs, (0.0, T), initial.ravel(), method="DOP853", rtol=2e-13, atol=2e-15).y[:, -1]
    moments = moments.reshape(degree + 1, 2)
    return math.exp(-R * T) * float(polynomial @ moments[:, start])


def unequal_occupation_mean(start, rate_12, rate_21):
    total_rate = rate_12 + rate_21
    stationary_one = rate_21 / total_rate
    transient = (1.0 - math.exp(-total_rate * T)) / total_rate
    if start == 0:
        return stationary_one * T + (1.0 - stationary_one) * transient
    return stationary_one * T - stationary_one * transient


def unequal_occupation_moments(start, rate_12, rate_21):
    """Exact mean and variance of U_T for arbitrary two-state rates.

    Write kappa=a+b, p=b/kappa, q=a/kappa, and d=q or -p according
    as the chain starts in state 1 or 2.  Integrating

        P_i(Y_s=1,Y_t=1)=P_i(Y_s=1)P_1(Y_{t-s}=1),  s<t,

    gives the second moment below.
    """
    total_rate = rate_12 + rate_21
    p = rate_21 / total_rate
    q = rate_12 / total_rate
    d = q if start == 0 else -p
    exponential = math.exp(-total_rate * T)
    transient = (1.0 - exponential) / total_rate
    j = T / total_rate - (1.0 - exponential) / total_rate ** 2
    h = (1.0 - exponential * (1.0 + total_rate * T)) / total_rate ** 2
    mean = p * T + d * transient
    second = p * p * T * T + 2.0 * p * (q + d) * j + 2.0 * d * q * h
    return mean, second - mean * mean


def occupation_wasserstein_distance(start, rate_12, rate_21):
    """W_1 distance from U_T to its exact-mean point mass.

    For a point mass at m=E[U_T], the transport cost and the Kantorovich dual
    both equal E|U_T-m|.  The endpoint atom is retained explicitly and the
    interior expectation is evaluated from the unequal-rate Bessel density.
    """
    mean, variance = unequal_occupation_moments(start, rate_12, rate_21)
    exit_rate = rate_12 if start == 0 else rate_21
    endpoint = T if start == 0 else 0.0
    atom_cost = math.exp(-exit_rate * T) * abs(endpoint - mean)
    interior_cost = quad(
        lambda u: abs(u - mean)
        * unequal_occupation_density(u, start, rate_12, rate_21),
        0.0,
        T,
        points=[mean],
        epsabs=2e-13,
        epsrel=2e-13,
        limit=250,
    )[0]
    return atom_cost + interior_cost, variance


def _exp_integral(decay):
    """Integral of exp(-decay*t) on [0,T], with its continuous limit."""
    if abs(decay) < 1e-10:
        return T
    return -math.expm1(-decay * T) / decay


def weighted_occupation_moments(start, rate_12, rate_21, clock_growth):
    """Exact moments of W=integral exp(h(T-t)) 1_{Y_t=1} dt.

    The displayed closed form has removable singularities at h=0 and h=kappa.
    The zero-growth limit is evaluated by the occupation formula above; the
    certificate stays away from the second isolated representation singularity.
    """
    if abs(clock_growth) < 1e-8:
        return unequal_occupation_moments(start, rate_12, rate_21)
    total_rate = rate_12 + rate_21
    p = rate_21 / total_rate
    q = rate_12 / total_rate
    d = q if start == 0 else -p
    h = clock_growth
    exp_h = math.exp(h * T)
    r0 = exp_h * _exp_integral(h)
    rk = exp_h * _exp_integral(h + total_rate)
    if abs(total_rate - h) < 1e-8:
        a_term = quad(
            lambda s: math.exp(h * (T - s))
            * quad(
                lambda t: math.exp(h * (T - t)) * math.exp(-total_rate * (t - s)),
                s,
                T,
                epsabs=2e-13,
                epsrel=2e-13,
            )[0],
            0.0,
            T,
            epsabs=2e-13,
            epsrel=2e-13,
        )[0]
    else:
        a_term = math.exp(2.0 * h * T) * (
            _exp_integral(2.0 * h) - _exp_integral(h + total_rate)
        ) / (total_rate - h)
    b_term = (
        math.exp(2.0 * h * T) * _exp_integral(2.0 * h + total_rate)
        - exp_h * _exp_integral(h + total_rate)
    ) / h
    c_term = math.exp(2.0 * h * T) * (
        _exp_integral(h + total_rate) - _exp_integral(2.0 * h + total_rate)
    ) / h
    mean = p * r0 + d * rk
    second = p * p * r0 * r0 + 2.0 * p * q * a_term + 2.0 * d * p * b_term + 2.0 * d * q * c_term
    return mean, second - mean * mean


def weighted_laplace_kummer(start, rate_12, rate_21, clock_growth, transform_argument):
    """Exact Laplace transform of the exponentially weighted occupation clock.

    For h != 0 and z > 0, eliminate the state-1 component from the two-state
    Feynman--Kac system.  With w=-(z/h)exp(h*tau), the state-2 component solves
    Kummer's equation with parameters b/h and 1+(a+b)/h.  On the negative real
    axis the individual Tricomi terms are complex on their principal branches,
    but the Wronskian combination below is real.  Exceptional Kummer parameters
    are interpreted by continuation; the certificate avoids those isolated
    representations.
    """
    if transform_argument == 0.0:
        return mp.mpf(1.0)
    with mp.workdps(80):
        h = mp.mpf(clock_growth)
        if h == 0.0:
            raise ValueError("use the constant-coefficient matrix exponential when h=0")
        a = mp.mpf(rate_12)
        b = mp.mpf(rate_21)
        z = mp.mpf(transform_argument)
        alpha = b / h
        gamma = 1.0 + (a + b) / h
        w0 = -z / h
        wT = w0 * mp.exp(h * T)

        def kummer_m(w):
            return mp.hyp1f1(alpha, gamma, w)

        def kummer_m_prime(w):
            return alpha * mp.hyp1f1(alpha + 1.0, gamma + 1.0, w) / gamma

        def kummer_u(w):
            return mp.hyperu(alpha, gamma, w)

        def kummer_u_prime(w):
            return -alpha * mp.hyperu(alpha + 1.0, gamma + 1.0, w)

        m0 = kummer_m(w0)
        mp0 = kummer_m_prime(w0)
        u0 = kummer_u(w0)
        up0 = kummer_u_prime(w0)
        denominator = m0 * up0 - mp0 * u0
        state_two = (up0 * kummer_m(wT) - mp0 * kummer_u(wT)) / denominator
        state_two_w = (
            up0 * kummer_m_prime(wT) - mp0 * kummer_u_prime(wT)
        ) / denominator
        result = state_two + h * wT * state_two_w / b if start == 0 else state_two
    return result


def weighted_laplace_ode(start, rate_12, rate_21, clock_growth, transform_argument):
    """Independent Feynman--Kac evaluation of the weighted-clock transform."""

    def rhs(tau, values):
        penalty = transform_argument * math.exp(clock_growth * tau)
        return [
            -(rate_12 + penalty) * values[0] + rate_12 * values[1],
            rate_21 * values[0] - rate_21 * values[1],
        ]

    solution = solve_ivp(
        rhs,
        (0.0, T),
        [1.0, 1.0],
        method="DOP853",
        rtol=2e-13,
        atol=2e-15,
    )
    return float(solution.y[start, -1])


def weighted_centered_characteristic(start, rate_12, rate_21, clock_growth, argument):
    """Characteristic function of sqrt(kappa) times the centered weighted clock."""
    total_rate = rate_12 + rate_21
    mean, _ = weighted_occupation_moments(start, rate_12, rate_21, clock_growth)

    def rhs(tau, values):
        frequency = argument * math.sqrt(total_rate) * math.exp(clock_growth * tau)
        return np.array(
            [
                (-rate_12 + 1j * frequency) * values[0] + rate_12 * values[1],
                rate_21 * values[0] - rate_21 * values[1],
            ],
            dtype=complex,
        )

    solution = solve_ivp(
        rhs,
        (0.0, T),
        np.ones(2, dtype=complex),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    return np.exp(-1j * argument * math.sqrt(total_rate) * mean) * solution.y[start, -1]


def weighted_stationary_centered_characteristic(start, rate_12, rate_21, clock_growth, argument):
    """Characteristic function for a fixed start, centered at the stationary clock."""
    total_rate = rate_12 + rate_21
    p = rate_21 / total_rate
    r0 = math.exp(clock_growth * T) * _exp_integral(clock_growth)

    def rhs(tau, values):
        frequency = argument * math.sqrt(total_rate) * math.exp(clock_growth * tau)
        return np.array(
            [
                (-rate_12 + 1j * frequency) * values[0] + rate_12 * values[1],
                rate_21 * values[0] - rate_21 * values[1],
            ],
            dtype=complex,
        )

    solution = solve_ivp(
        rhs,
        (0.0, T),
        np.ones(2, dtype=complex),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    return np.exp(-1j * argument * math.sqrt(total_rate) * p * r0) * solution.y[start, -1]


def weighted_stationary_characteristic(rate_12, rate_21, clock_growth, argument):
    """Characteristic function under the stationary initial regime law."""
    total_rate = rate_12 + rate_21
    p = rate_21 / total_rate
    q = rate_12 / total_rate
    r0 = math.exp(clock_growth * T) * _exp_integral(clock_growth)

    def rhs(tau, values):
        frequency = argument * math.sqrt(total_rate) * math.exp(clock_growth * tau)
        return np.array(
            [
                (-rate_12 + 1j * frequency) * values[0] + rate_12 * values[1],
                rate_21 * values[0] - rate_21 * values[1],
            ],
            dtype=complex,
        )

    solution = solve_ivp(
        rhs,
        (0.0, T),
        np.ones(2, dtype=complex),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    raw = p * solution.y[0, -1] + q * solution.y[1, -1]
    return np.exp(-1j * argument * math.sqrt(total_rate) * p * r0) * raw


def joint_weighted_stationary_centered_characteristic(
    start, rate_12, rate_21, clock_growths, arguments
):
    """Joint clock characteristic function, centered at stationary means.

    The vector is tested in the supplied Fourier direction ``arguments``.
    Each component has loading exp(h_j*(T-t)); the backward Feynman--Kac
    clock therefore uses exp(h_j*tau).
    """
    total_rate = rate_12 + rate_21
    p = rate_21 / total_rate
    stationary_mean = p * sum(
        argument * math.exp(growth * T) * _exp_integral(growth)
        for growth, argument in zip(clock_growths, arguments)
    )

    def rhs(tau, values):
        loading = sum(
            argument * math.exp(growth * tau)
            for growth, argument in zip(clock_growths, arguments)
        )
        frequency = math.sqrt(total_rate) * loading
        return np.array(
            [
                (-rate_12 + 1j * frequency) * values[0] + rate_12 * values[1],
                rate_21 * values[0] - rate_21 * values[1],
            ],
            dtype=complex,
        )

    solution = solve_ivp(
        rhs,
        (0.0, T),
        np.ones(2, dtype=complex),
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    )
    return (
        np.exp(-1j * math.sqrt(total_rate) * stationary_mean)
        * solution.y[start, -1]
    )


def joint_weighted_raw_moments_ode(start, rate_12, rate_21, clock_growths):
    """First and second moments of two clocks from a bivariate FK hierarchy."""
    if len(clock_growths) != 2:
        raise ValueError("the certificate uses exactly two clock loadings")
    generator = np.array([[-rate_12, rate_12], [rate_21, -rate_21]])

    def rhs(tau, flat):
        moments = flat.reshape(3, 3, 2)
        out = np.zeros_like(moments)
        rewards = [
            np.array([math.exp(growth * tau), 0.0])
            for growth in clock_growths
        ]
        for first_order in range(3):
            for second_order in range(3 - first_order):
                out[first_order, second_order] = (
                    generator @ moments[first_order, second_order]
                )
                if first_order:
                    out[first_order, second_order] += (
                        first_order
                        * rewards[0]
                        * moments[first_order - 1, second_order]
                    )
                if second_order:
                    out[first_order, second_order] += (
                        second_order
                        * rewards[1]
                        * moments[first_order, second_order - 1]
                    )
        return out.ravel()

    initial = np.zeros((3, 3, 2))
    initial[0, 0] = 1.0
    solution = solve_ivp(
        rhs,
        (0.0, T),
        initial.ravel(),
        method="DOP853",
        rtol=2e-13,
        atol=2e-15,
    )
    moments = solution.y[:, -1].reshape(3, 3, 2)
    means = np.array([moments[1, 0, start], moments[0, 1, start]])
    second = np.array(
        [
            [moments[2, 0, start], moments[1, 1, start]],
            [moments[1, 1, start], moments[0, 2, start]],
        ]
    )
    return means, second - np.outer(means, means)


def weighted_clock_raw_moments_ode(start, rate_12, rate_21, clock_growth, degree=3):
    """Independent raw moments from the polynomial Feynman--Kac hierarchy."""
    generator = np.array([[-rate_12, rate_12], [rate_21, -rate_21]])

    def rhs(tau, flat):
        moments = flat.reshape(degree + 1, 2)
        out = np.zeros_like(moments)
        reward = np.array([math.exp(clock_growth * tau), 0.0])
        for order in range(degree + 1):
            out[order] = generator @ moments[order]
            if order:
                out[order] += order * reward * moments[order - 1]
        return out.ravel()

    initial = np.zeros((degree + 1, 2))
    initial[0] = 1.0
    solution = solve_ivp(
        rhs,
        (0.0, T),
        initial.ravel(),
        method="DOP853",
        rtol=2e-13,
        atol=2e-15,
    )
    return solution.y[:, -1].reshape(degree + 1, 2)[:, start]


def weighted_third_cumulant_integral(start, rate_12, rate_21, clock_growth):
    """Exact ordered-simplex integral of the fixed-start joint cumulant."""
    total_rate = rate_12 + rate_21
    p = rate_21 / total_rate
    q = rate_12 / total_rate
    c = q - p
    d = q if start == 0 else -p

    def integrand(t3, t2, t1):
        bulk = p * q * c * math.exp(-total_rate * (t3 - t1))
        boundary = (
            c ** 2 * d * math.exp(-total_rate * t3)
            - c * d ** 2 * math.exp(-total_rate * (t1 + t3))
            - 2.0 * p * q * d * math.exp(-total_rate * (t2 + t3 - t1))
            - 2.0 * c * d ** 2 * math.exp(-total_rate * (t2 + t3))
            + 2.0 * d ** 3 * math.exp(-total_rate * (t1 + t2 + t3))
        )
        weight = math.exp(clock_growth * (3.0 * T - t1 - t2 - t3))
        return 6.0 * weight * (bulk + boundary)

    return tplquad(
        integrand,
        0.0,
        T,
        lambda t1: t1,
        lambda t1: T,
        lambda t1, t2: t2,
        lambda t1, t2: T,
        epsabs=2e-11,
        epsrel=2e-11,
    )[0]


def weighted_stationary_fourth_cumulant_integral(rate_12, rate_21, clock_growth):
    """Exact ordered-simplex integral of the stationary fourth cumulant.

    The base time is integrated analytically, leaving a three-dimensional
    simplex integral in the consecutive gaps.
    """
    total_rate = rate_12 + rate_21
    p = rate_21 / total_rate
    q = rate_12 / total_rate
    r = p * q
    c = q - p

    def integrand(gap_3, gap_2, gap_1):
        span = gap_1 + gap_2 + gap_3
        if abs(clock_growth) < 1e-12:
            base_integral = T - span
        else:
            base_integral = (
                math.exp(clock_growth * (4.0 * T - 3.0 * gap_1 - 2.0 * gap_2 - gap_3))
                * -math.expm1(-4.0 * clock_growth * (T - span))
                / (4.0 * clock_growth)
            )
        joint_cumulant = (
            r * c ** 2 * math.exp(-total_rate * span)
            - 2.0
            * r ** 2
            * math.exp(-total_rate * (gap_1 + 2.0 * gap_2 + gap_3))
        )
        return 24.0 * base_integral * joint_cumulant

    return tplquad(
        integrand,
        0.0,
        T,
        lambda gap_1: 0.0,
        lambda gap_1: T - gap_1,
        lambda gap_1, gap_2: 0.0,
        lambda gap_1, gap_2: T - gap_1 - gap_2,
        epsabs=2e-11,
        epsrel=2e-11,
    )[0]


def cev_derivative_estimates(rate=R, dividend=Q, degree=56):
    """Numerical derivative-supremum estimates for the finite-rate table.

    The theorem uses the true suprema M_1 and M_2.  Here two stable Chebyshev
    reconstructions on the compact attainable-clock interval supply estimates
    that are rounded upward for the illustrative table; this is not an interval
    proof of the derivative suprema.
    """
    clock_growth = 2.0 * (1.0 - BETA) * (rate - dividend)
    clock_mass = math.expm1(clock_growth * T) / clock_growth if abs(clock_growth) > 1e-10 else T
    lower = min(SIGMA) ** 2 * clock_mass
    upper = max(SIGMA) ** 2 * clock_mass
    center = 0.5 * (lower + upper)
    half_width = 0.5 * (upper - lower)
    nodes = np.cos(np.pi * (np.arange(3 * degree) + 0.5) / (3 * degree))
    coefficients = ch.chebfit(
        nodes,
        [cev_call(center + half_width * x, rate, dividend) for x in nodes],
        degree,
    )
    grid = np.linspace(-1.0, 1.0, 20001)
    first = ch.chebval(grid, ch.chebder(coefficients, 1)) / half_width
    second = ch.chebval(grid, ch.chebder(coefficients, 2)) / half_width ** 2
    return float(np.max(np.abs(first))), float(np.max(np.abs(second)))


def finite_rate_bounds(start, rate_12, rate_21, m1, m2):
    """Exact prices, two approximations, and their fixed-maturity bounds."""
    mean, variance = unequal_occupation_moments(start, rate_12, rate_21)
    total_rate = rate_12 + rate_21
    stationary_one = rate_21 / total_rate
    d = rate_12 / total_rate if start == 0 else -stationary_one
    transient = (1.0 - math.exp(-total_rate * T)) / total_rate
    delta = SIGMA[0] ** 2 - SIGMA[1] ** 2
    mean_clock = SIGMA[1] ** 2 * T + delta * mean
    stationary_clock = SIGMA[1] ** 2 * T + delta * stationary_one * T
    discount = math.exp(-R * T)
    exact = bessel_price(start, rate_12, rate_21)
    centered = discount * cev_call(mean_clock)
    stationary = discount * cev_call(stationary_clock)
    centered_bound = 0.5 * discount * m2 * delta ** 2 * variance
    stationary_bound = discount * (
        m1 * abs(delta * d) * transient + 0.5 * m2 * delta ** 2 * variance
    )
    return exact, centered, centered_bound, stationary, stationary_bound


def weighted_moment_ode_price(start, rate_12, rate_21, rate, dividend, degree=32):
    """Independent clock-moment reconstruction at nonzero carry."""
    clock_growth = 2.0 * (1.0 - BETA) * (rate - dividend)
    clock_mass = math.expm1(clock_growth * T) / clock_growth if abs(clock_growth) > 1e-10 else T
    s1, s2 = SIGMA[0] ** 2, SIGMA[1] ** 2
    sbar = 0.5 * (s1 + s2)
    half_difference = 0.5 * (s1 - s2)
    center = sbar * clock_mass
    half_width = half_difference * clock_mass
    nodes = np.cos(np.pi * (np.arange(72) + 0.5) / 72)
    coefficients = ch.chebfit(
        nodes,
        [cev_call(center + half_width * x, rate, dividend) for x in nodes],
        degree,
    )
    polynomial = ch.cheb2poly(coefficients)
    generator = np.array([[-rate_12, rate_12], [rate_21, -rate_21]])
    signs = np.array([1.0, -1.0])

    def rhs(tau, flat):
        moments = flat.reshape(degree + 1, 2)
        out = np.zeros_like(moments)
        rho = signs * math.exp(clock_growth * tau) / clock_mass
        for order in range(degree + 1):
            out[order] = generator @ moments[order]
            if order:
                out[order] += order * rho * moments[order - 1]
        return out.ravel()

    initial = np.zeros((degree + 1, 2))
    initial[0] = 1.0
    moments = solve_ivp(rhs, (0.0, T), initial.ravel(), method="DOP853", rtol=2e-13, atol=2e-15).y[:, -1]
    moments = moments.reshape(degree + 1, 2)
    return math.exp(-rate * T) * float(polynomial @ moments[:, start])


def weighted_finite_rate_bounds(start, rate_12, rate_21, rate, dividend, m1, m2):
    """Price approximations and bounds for the exponentially weighted clock."""
    clock_growth = 2.0 * (1.0 - BETA) * (rate - dividend)
    clock_mass = math.expm1(clock_growth * T) / clock_growth if abs(clock_growth) > 1e-10 else T
    mean, variance = weighted_occupation_moments(start, rate_12, rate_21, clock_growth)
    total_rate = rate_12 + rate_21
    stationary_one = rate_21 / total_rate
    d = rate_12 / total_rate if start == 0 else -stationary_one
    weighted_memory = math.exp(clock_growth * T) * _exp_integral(clock_growth + total_rate)
    delta = SIGMA[0] ** 2 - SIGMA[1] ** 2
    mean_clock = SIGMA[1] ** 2 * clock_mass + delta * mean
    stationary_clock = (SIGMA[1] ** 2 + delta * stationary_one) * clock_mass
    discount = math.exp(-rate * T)
    exact = weighted_moment_ode_price(start, rate_12, rate_21, rate, dividend)
    centered = discount * cev_call(mean_clock, rate, dividend)
    stationary = discount * cev_call(stationary_clock, rate, dividend)
    centered_bound = 0.5 * discount * m2 * delta ** 2 * variance
    stationary_bound = discount * (
        m1 * abs(delta * d) * weighted_memory + 0.5 * m2 * delta ** 2 * variance
    )
    return exact, centered, centered_bound, stationary, stationary_bound


def occupation_mean(start):
    """Closed-form E[U_T] for the symmetric chain."""
    memory = (1.0 - math.exp(-2.0 * LAM * T)) / (4.0 * LAM)
    return T / 2.0 + (memory if start == 0 else -memory)


def mixture_occupation_mean(start, max_jumps=70):
    mu = LAM * T
    total = 0.0
    probability = math.exp(-mu)
    for n in range(max_jumps + 1):
        if n:
            probability *= mu / n
        if n == 0:
            mean_fraction = 1.0 if start == 0 else 0.0
        elif n % 2:
            mean_fraction = 0.5
        else:
            k = n // 2
            mean_fraction = (k + 1.0) / (2.0 * k + 1.0) if start == 0 else k / (2.0 * k + 1.0)
        total += probability * T * mean_fraction
    return total


def main():
    certified_cutoff = 30
    reference_cutoff = 70
    tail_bound = S0 * math.exp(-Q * T) * poisson.sf(certified_cutoff, LAM * T)
    print("symmetric two-state CEV occupation-time certificate")
    print(f"lambda*T={LAM*T:g}; certified cutoff N={certified_cutoff}; price tail bound={tail_bound:.3e}")
    for start in (0, 1):
        partial = poisson_beta_price(start, certified_cutoff)
        reference = poisson_beta_price(start, reference_cutoff)
        moment = moment_ode_price(start)
        actual_tail = abs(reference - partial)
        mean_error = abs(mixture_occupation_mean(start) - occupation_mean(start))
        print(
            f"start {start + 1}: mixture={reference:.10f}; moment ODE={moment:.10f}; "
            f"difference={abs(reference-moment):.3e}; actual tail={actual_tail:.3e}; "
            f"mean error={mean_error:.3e}"
        )
        assert abs(reference - moment) < 2e-9
        assert actual_tail <= tail_bound
        assert mean_error < 2e-14

    equal_rate_reduction = max(
        abs(bessel_price(start, LAM, LAM) - poisson_beta_price(start, reference_cutoff))
        for start in (0, 1)
    )
    print(f"equal-rate Bessel/Poisson-Beta difference={equal_rate_reduction:.3e}")
    assert equal_rate_reduction < 2e-13

    rate_12, rate_21 = 7.0, 13.0
    print(f"unequal rates: q12={rate_12:g}, q21={rate_21:g}")
    for start in (0, 1):
        atom = math.exp(-(rate_12 if start == 0 else rate_21) * T)
        mass = atom + quad(
            lambda u: unequal_occupation_density(u, start, rate_12, rate_21),
            0.0,
            T,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=250,
        )[0]
        mean = quad(
            lambda u: u * unequal_occupation_density(u, start, rate_12, rate_21),
            0.0,
            T,
            epsabs=2e-13,
            epsrel=2e-13,
            limit=250,
        )[0] + atom * (T if start == 0 else 0.0)
        bessel = bessel_price(start, rate_12, rate_21)
        moment = unequal_moment_ode_price(start, rate_12, rate_21)
        print(
            f"start {start + 1}: mass error={abs(mass-1):.3e}; "
            f"mean error={abs(mean-unequal_occupation_mean(start, rate_12, rate_21)):.3e}; "
            f"Bessel={bessel:.10f}; moment ODE={moment:.10f}; difference={abs(bessel-moment):.3e}"
        )
        assert abs(mass - 1.0) < 3e-13
        assert abs(mean - unequal_occupation_mean(start, rate_12, rate_21)) < 3e-13
        assert abs(bessel - moment) < 5e-12

    m1_raw, m2_raw = cev_derivative_estimates()
    m1_check, m2_check = cev_derivative_estimates(degree=48)
    assert abs(m1_raw - m1_check) < 1e-7
    assert abs(m2_raw - m2_check) < 1e-7
    # Rounded upward from the stable dense-grid Chebyshev calculations above.
    m1, m2 = 2.625, 0.919
    assert m1_raw < m1 and m2_raw < m2
    print(f"finite-rate derivative estimates (rounded up): M1={m1:g}, M2={m2:g}")
    for multiplier in (1, 2, 4, 8):
        a, b = multiplier * rate_12, multiplier * rate_21
        for start in (0, 1):
            atom = math.exp(-(a if start == 0 else b) * T)
            second = quad(
                lambda u: u * u * unequal_occupation_density(u, start, a, b),
                0.0,
                T,
                epsabs=2e-13,
                epsrel=2e-13,
                limit=250,
            )[0] + atom * (T * T if start == 0 else 0.0)
            mean, variance = unequal_occupation_moments(start, a, b)
            assert abs(second - (variance + mean * mean)) < 4e-13
            exact, centered, centered_bound, stationary, stationary_bound = finite_rate_bounds(
                start, a, b, m1, m2
            )
            centered_error = abs(exact - centered)
            stationary_error = abs(exact - stationary)
            print(
                f"kappa={a+b:g}, start {start+1}: Var(U)={variance:.10f}; "
                f"mean-clock error/bound={centered_error:.3e}/{centered_bound:.3e}; "
                f"stationary-clock error/bound={stationary_error:.3e}/{stationary_bound:.3e}"
            )
            assert centered_error <= centered_bound
            assert stationary_error <= stationary_bound

    # Sharp Lipschitz regularity gap.  Under a=kappa*q, b=kappa*p,
    # sqrt(kappa)(U_T-EU_T) converges to N(0,2*p*q*T), so
    # E|U_T-EU_T| ~ sqrt(4*p*q*T/(pi*kappa)).
    p, q = 0.65, 0.35
    kappas = (20.0, 40.0, 80.0, 160.0, 320.0, 640.0, 1280.0)
    wasserstein_ratios = {0: [], 1: []}
    variance_ratios = {0: [], 1: []}
    for kappa in kappas:
        a, b = kappa * q, kappa * p
        normal_absolute_mean = math.sqrt(4.0 * p * q * T / (math.pi * kappa))
        for start in (0, 1):
            distance, variance = occupation_wasserstein_distance(start, a, b)
            assert distance <= math.sqrt(variance) * (1.0 + 2e-12)
            wasserstein_ratios[start].append(distance / normal_absolute_mean)
            variance_ratios[start].append(distance / math.sqrt(variance))
    for start in (0, 1):
        assert all(
            later > earlier
            for earlier, later in zip(wasserstein_ratios[start], wasserstein_ratios[start][1:])
        )
        assert abs(wasserstein_ratios[start][-1] - 1.0) < 6e-4
        assert abs(variance_ratios[start][-1] - math.sqrt(2.0 / math.pi)) < 2e-4
        print(
            f"Wasserstein start {start+1}: ratios to sharp normal asymptotic="
            + ", ".join(f"{value:.7f}" for value in wasserstein_ratios[start])
            + f"; terminal W1/sd={variance_ratios[start][-1]:.7f}"
        )

    carry_rate, carry_dividend = 0.05, 0.01
    clock_growth = 2.0 * (1.0 - BETA) * (carry_rate - carry_dividend)
    m1_raw, m2_raw = cev_derivative_estimates(carry_rate, carry_dividend)
    m1_check, m2_check = cev_derivative_estimates(carry_rate, carry_dividend, degree=48)
    assert abs(m1_raw - m1_check) < 1e-7
    assert abs(m2_raw - m2_check) < 1e-7
    m1, m2 = 2.58, 0.87
    assert m1_raw < m1 and m2_raw < m2
    print(
        f"nonzero carry r-q={carry_rate-carry_dividend:g}, h={clock_growth:g}; "
        f"derivative estimates (rounded up): M1={m1:g}, M2={m2:g}"
    )
    transform_error = 0.0
    transform_imaginary = 0.0
    transform_cases = (
        (rate_12, rate_21, clock_growth, (0.01, 0.1, 1.0, 5.0)),
        (0.8, 1.1, -0.7, (0.1, 1.0, 3.0)),
    )
    for a, b, h, arguments in transform_cases:
        for argument in arguments:
            for start in (0, 1):
                kummer = weighted_laplace_kummer(start, a, b, h, argument)
                ode = weighted_laplace_ode(start, a, b, h, argument)
                transform_error = max(transform_error, float(abs(mp.re(kummer) - ode)))
                transform_imaginary = max(transform_imaginary, float(abs(mp.im(kummer))))
                assert 0.0 < mp.re(kummer) <= 1.0
    print(
        f"weighted-clock Kummer/Feynman--Kac max error={transform_error:.3e}; "
        f"max cancelled imaginary part={transform_imaginary:.3e}"
    )
    assert transform_error < 4e-12
    assert transform_imaginary < 1e-20

    # Weighted additive-functional CLT.  With a=kappa*q and b=kappa*p,
    # sqrt(kappa)(W-EW) converges to N(0, 2*p*q*int_0^T w(t)^2 dt).
    # The complex Feynman--Kac ODE is independent of the moment calculation.
    p, q = 0.65, 0.35
    weighted_energy = math.expm1(2.0 * clock_growth * T) / (2.0 * clock_growth)
    limiting_variance = 2.0 * p * q * weighted_energy
    characteristic_errors = []
    variance_errors = []
    arguments = np.linspace(-2.0, 2.0, 17)
    for kappa in (20.0, 40.0, 80.0, 160.0, 320.0, 640.0):
        a, b = kappa * q, kappa * p
        max_error = 0.0
        for start in (0, 1):
            _, variance = weighted_occupation_moments(start, a, b, clock_growth)
            variance_errors.append(abs(kappa * variance - limiting_variance))
            for argument in arguments:
                exact_cf = weighted_centered_characteristic(start, a, b, clock_growth, argument)
                normal_cf = math.exp(-0.5 * limiting_variance * argument * argument)
                max_error = max(max_error, abs(exact_cf - normal_cf))
        characteristic_errors.append(max_error)
    characteristic_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(characteristic_errors, characteristic_errors[1:])
    ]
    print(
        "weighted-clock CLT max characteristic errors="
        + ", ".join(f"{value:.3e}" for value in characteristic_errors)
        + "; halving orders="
        + ", ".join(f"{value:.3f}" for value in characteristic_orders)
        + f"; terminal scaled-variance error={max(variance_errors[-2:]):.3e}"
    )
    assert characteristic_errors[-1] < 1.5e-2
    assert all(later < earlier for earlier, later in zip(characteristic_errors, characteristic_errors[1:]))
    assert max(variance_errors[-2:]) < 2e-3

    # Under a stationary start the leading non-Gaussian term is explicit:
    # kappa^2 Cum_3(W) -> 6*p*q*(q-p)*int w^3.  Hence sqrt(kappa)
    # times the characteristic-function error has the limit below.
    weighted_third_energy = math.expm1(3.0 * clock_growth * T) / (3.0 * clock_growth)
    edgeworth_errors = []
    edgeworth_arguments = (0.5, 1.0, 1.5)
    for kappa in (40.0, 80.0, 160.0, 320.0, 640.0):
        a, b = kappa * q, kappa * p
        max_error = 0.0
        for argument in edgeworth_arguments:
            exact_cf = weighted_stationary_characteristic(a, b, clock_growth, argument)
            normal_cf = math.exp(-0.5 * limiting_variance * argument * argument)
            predicted = (
                -1j
                * argument ** 3
                * p
                * q
                * (q - p)
                * weighted_third_energy
                * normal_cf
            )
            scaled_error = math.sqrt(kappa) * (exact_cf - normal_cf)
            max_error = max(max_error, abs(scaled_error - predicted))
        edgeworth_errors.append(max_error)
    edgeworth_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(edgeworth_errors, edgeworth_errors[1:])
    ]
    print(
        "weighted stationary first characteristic correction residuals="
        + ", ".join(f"{value:.3e}" for value in edgeworth_errors)
        + "; halving orders="
        + ", ".join(f"{value:.3f}" for value in edgeworth_orders)
    )
    assert edgeworth_errors[-1] < 1e-2
    assert all(later < earlier for earlier, later in zip(edgeworth_errors, edgeworth_errors[1:]))

    # The stationary expansion continues one order further.  The exact
    # variance has
    #   kappa*Var(W) = 2*p*q*A_2 - p*q*(1+exp(2*h*T))/kappa + O(kappa^-2),
    # while the ordered four-time joint cumulant gives
    #   kappa^3*Cum_4(W) -> 24*p*q*(1-5*p*q)*A_4.
    # Exponentiating the cumulant series adds one half the square of the
    # first (skew) correction.
    weighted_fourth_energy = math.expm1(4.0 * clock_growth * T) / (4.0 * clock_growth)
    variance_second_coefficient = -p * q * (1.0 + math.exp(2.0 * clock_growth * T))
    fourth_cumulant_limit = (
        24.0 * p * q * (1.0 - 5.0 * p * q) * weighted_fourth_energy
    )
    stationary_raw = []
    for start in (0, 1):
        stationary_raw.append(
            weighted_clock_raw_moments_ode(start, 7.0, 13.0, clock_growth, degree=4)
        )
    stationary_raw = p * stationary_raw[0] + q * stationary_raw[1]
    stationary_mean = stationary_raw[1]
    stationary_fourth_cumulant = (
        stationary_raw[4]
        - 4.0 * stationary_raw[3] * stationary_mean
        - 3.0 * stationary_raw[2] ** 2
        + 12.0 * stationary_raw[2] * stationary_mean ** 2
        - 6.0 * stationary_mean ** 4
    )
    fourth_integral = weighted_stationary_fourth_cumulant_integral(
        7.0, 13.0, clock_growth
    )
    fourth_integral_error = abs(stationary_fourth_cumulant - fourth_integral)
    print(
        "weighted stationary exact fourth-cumulant formula error="
        f"{fourth_integral_error:.3e}"
    )
    assert fourth_integral_error < 2e-12

    second_edgeworth_errors = []
    variance_second_errors = []
    fourth_cumulant_errors = []
    for kappa in (40.0, 80.0, 160.0, 320.0, 640.0):
        a, b = kappa * q, kappa * p
        raw_by_start = [
            weighted_clock_raw_moments_ode(start, a, b, clock_growth, degree=4)
            for start in (0, 1)
        ]
        raw = p * raw_by_start[0] + q * raw_by_start[1]
        mean = raw[1]
        variance = raw[2] - mean ** 2
        cumulant_four = (
            raw[4]
            - 4.0 * raw[3] * mean
            - 3.0 * raw[2] ** 2
            + 12.0 * raw[2] * mean ** 2
            - 6.0 * mean ** 4
        )
        variance_second_errors.append(
            abs(kappa * (kappa * variance - limiting_variance) - variance_second_coefficient)
        )
        fourth_cumulant_errors.append(
            abs(kappa ** 3 * cumulant_four - fourth_cumulant_limit)
        )
        max_error = 0.0
        for argument in edgeworth_arguments:
            exact_cf = weighted_stationary_characteristic(a, b, clock_growth, argument)
            normal_cf = math.exp(-0.5 * limiting_variance * argument * argument)
            first_polynomial = (
                -1j * argument ** 3 * p * q * (q - p) * weighted_third_energy
            )
            second_polynomial = (
                0.5
                * p
                * q
                * (1.0 + math.exp(2.0 * clock_growth * T))
                * argument ** 2
                + p
                * q
                * (1.0 - 5.0 * p * q)
                * weighted_fourth_energy
                * argument ** 4
                - 0.5
                * (p * q * (q - p) * weighted_third_energy) ** 2
                * argument ** 6
            )
            scaled_error = kappa * (
                exact_cf - normal_cf - first_polynomial * normal_cf / math.sqrt(kappa)
            )
            max_error = max(
                max_error, abs(scaled_error - second_polynomial * normal_cf)
            )
        second_edgeworth_errors.append(max_error)
    second_edgeworth_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(second_edgeworth_errors, second_edgeworth_errors[1:])
    ]
    fourth_cumulant_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(fourth_cumulant_errors, fourth_cumulant_errors[1:])
    ]
    print(
        "weighted stationary second characteristic correction residuals="
        + ", ".join(f"{value:.3e}" for value in second_edgeworth_errors)
        + "; halving orders="
        + ", ".join(f"{value:.3f}" for value in second_edgeworth_orders)
    )
    print(
        "weighted stationary scaled fourth-cumulant errors="
        + ", ".join(f"{value:.3e}" for value in fourth_cumulant_errors)
        + "; halving orders="
        + ", ".join(f"{value:.3f}" for value in fourth_cumulant_orders)
        + f"; terminal variance-coefficient error={variance_second_errors[-1]:.3e}"
    )
    assert second_edgeworth_errors[-1] < 4e-3
    assert all(
        later < earlier
        for earlier, later in zip(second_edgeworth_errors, second_edgeworth_errors[1:])
    )
    assert fourth_cumulant_errors[-1] < 4e-3
    assert variance_second_errors[-1] < 2e-6

    # A fixed initial regime changes the first correction only through the
    # initial-layer mean.  Indeed kappa*R(kappa) -> exp(h*T), while the
    # leading third cumulant is the same bulk term as under stationarity.
    # The polynomial moment hierarchy and the complex transform below are
    # independent numerical checks of those two assertions.
    fixed_start_errors = []
    third_cumulant_errors = []
    third_cumulant_limit = 6.0 * p * q * (q - p) * weighted_third_energy
    initial_weight = math.exp(clock_growth * T)
    exact_cumulant_error = 0.0
    for start in (0, 1):
        raw = weighted_clock_raw_moments_ode(start, 7.0, 13.0, clock_growth)
        cumulant_three = raw[3] - 3.0 * raw[2] * raw[1] + 2.0 * raw[1] ** 3
        integral = weighted_third_cumulant_integral(start, 7.0, 13.0, clock_growth)
        exact_cumulant_error = max(exact_cumulant_error, abs(cumulant_three - integral))
    print(f"weighted fixed-start exact third-cumulant formula error={exact_cumulant_error:.3e}")
    assert exact_cumulant_error < 2e-12

    for kappa in (40.0, 80.0, 160.0, 320.0, 640.0):
        a, b = kappa * q, kappa * p
        max_error = 0.0
        max_cumulant_error = 0.0
        for start in (0, 1):
            d = q if start == 0 else -p
            raw = weighted_clock_raw_moments_ode(start, a, b, clock_growth)
            cumulant_three = raw[3] - 3.0 * raw[2] * raw[1] + 2.0 * raw[1] ** 3
            max_cumulant_error = max(
                max_cumulant_error,
                abs(kappa ** 2 * cumulant_three - third_cumulant_limit),
            )
            for argument in edgeworth_arguments:
                exact_cf = weighted_stationary_centered_characteristic(
                    start, a, b, clock_growth, argument
                )
                normal_cf = math.exp(-0.5 * limiting_variance * argument * argument)
                predicted = (
                    1j * argument * d * initial_weight
                    - 1j
                    * argument ** 3
                    * p
                    * q
                    * (q - p)
                    * weighted_third_energy
                ) * normal_cf
                scaled_error = math.sqrt(kappa) * (exact_cf - normal_cf)
                max_error = max(max_error, abs(scaled_error - predicted))
        fixed_start_errors.append(max_error)
        third_cumulant_errors.append(max_cumulant_error)
    fixed_start_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(fixed_start_errors, fixed_start_errors[1:])
    ]
    third_cumulant_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(third_cumulant_errors, third_cumulant_errors[1:])
    ]
    print(
        "weighted fixed-start first characteristic correction residuals="
        + ", ".join(f"{value:.3e}" for value in fixed_start_errors)
        + "; halving orders="
        + ", ".join(f"{value:.3f}" for value in fixed_start_orders)
    )
    print(
        "weighted fixed-start scaled third-cumulant errors="
        + ", ".join(f"{value:.3e}" for value in third_cumulant_errors)
        + "; halving orders="
        + ", ".join(f"{value:.3f}" for value in third_cumulant_orders)
    )
    assert fixed_start_errors[-1] < 2e-2
    assert all(later < earlier for earlier, later in zip(fixed_start_errors, fixed_start_errors[1:]))
    assert third_cumulant_errors[-1] < 1e-2
    assert all(
        later < earlier for earlier, later in zip(third_cumulant_errors, third_cumulant_errors[1:])
    )

    # At second characteristic order a fixed start changes the variance
    # coefficient as well as the first-order mean.  For d_i=q or -p,
    #   Cov_i(g_s,g_t)=p*q*e^-k(t-s)+(q-p)d_i*e^-kt-d_i^2*e^-k(s+t), s<=t.
    # The last two terms are pinned to the initial corner and give the
    # displayed start-dependent kappa^-2 variance coefficient.  The leading
    # fourth cumulant is still the stationary bulk coefficient because its
    # initial-corner remainder integrates to O(kappa^-4).
    fixed_second_errors = []
    exact_mean_second_errors = []
    fixed_variance_second_errors = []
    fixed_fourth_cumulant_errors = []
    for kappa in (40.0, 80.0, 160.0, 320.0, 640.0):
        a, b = kappa * q, kappa * p
        max_fixed_error = 0.0
        max_exact_mean_error = 0.0
        max_variance_error = 0.0
        max_fourth_error = 0.0
        for start in (0, 1):
            d = q if start == 0 else -p
            start_variance_coefficient = (
                variance_second_coefficient
                + math.exp(2.0 * clock_growth * T)
                * (2.0 * (q - p) * d - d ** 2)
            )
            raw = weighted_clock_raw_moments_ode(
                start, a, b, clock_growth, degree=4
            )
            mean = raw[1]
            variance = raw[2] - mean ** 2
            cumulant_four = (
                raw[4]
                - 4.0 * raw[3] * mean
                - 3.0 * raw[2] ** 2
                + 12.0 * raw[2] * mean ** 2
                - 6.0 * mean ** 4
            )
            max_variance_error = max(
                max_variance_error,
                abs(
                    kappa * (kappa * variance - limiting_variance)
                    - start_variance_coefficient
                ),
            )
            max_fourth_error = max(
                max_fourth_error,
                abs(kappa ** 3 * cumulant_four - fourth_cumulant_limit),
            )
            for argument in edgeworth_arguments:
                normal_cf = math.exp(-0.5 * limiting_variance * argument ** 2)
                skew_polynomial = (
                    -1j
                    * argument ** 3
                    * p
                    * q
                    * (q - p)
                    * weighted_third_energy
                )
                fixed_first_polynomial = (
                    1j * argument * d * initial_weight + skew_polynomial
                )
                fixed_second_polynomial = (
                    -0.5 * argument ** 2 * start_variance_coefficient
                    + p
                    * q
                    * (1.0 - 5.0 * p * q)
                    * weighted_fourth_energy
                    * argument ** 4
                    + 0.5 * fixed_first_polynomial ** 2
                )
                exact_fixed = weighted_stationary_centered_characteristic(
                    start, a, b, clock_growth, argument
                )
                fixed_scaled_error = kappa * (
                    exact_fixed
                    - normal_cf
                    - fixed_first_polynomial * normal_cf / math.sqrt(kappa)
                )
                max_fixed_error = max(
                    max_fixed_error,
                    abs(fixed_scaled_error - fixed_second_polynomial * normal_cf),
                )

                exact_mean_second_polynomial = (
                    -0.5 * argument ** 2 * start_variance_coefficient
                    + p
                    * q
                    * (1.0 - 5.0 * p * q)
                    * weighted_fourth_energy
                    * argument ** 4
                    + 0.5 * skew_polynomial ** 2
                )
                exact_centered = weighted_centered_characteristic(
                    start, a, b, clock_growth, argument
                )
                exact_mean_scaled_error = kappa * (
                    exact_centered
                    - normal_cf
                    - skew_polynomial * normal_cf / math.sqrt(kappa)
                )
                max_exact_mean_error = max(
                    max_exact_mean_error,
                    abs(
                        exact_mean_scaled_error
                        - exact_mean_second_polynomial * normal_cf
                    ),
                )
        fixed_second_errors.append(max_fixed_error)
        exact_mean_second_errors.append(max_exact_mean_error)
        fixed_variance_second_errors.append(max_variance_error)
        fixed_fourth_cumulant_errors.append(max_fourth_error)
    fixed_second_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(fixed_second_errors, fixed_second_errors[1:])
    ]
    exact_mean_second_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(exact_mean_second_errors, exact_mean_second_errors[1:])
    ]
    print(
        "weighted fixed-start second characteristic correction residuals="
        + ", ".join(f"{value:.3e}" for value in fixed_second_errors)
        + "; halving orders="
        + ", ".join(f"{value:.3f}" for value in fixed_second_orders)
    )
    print(
        "weighted exact-mean second characteristic correction residuals="
        + ", ".join(f"{value:.3e}" for value in exact_mean_second_errors)
        + "; halving orders="
        + ", ".join(f"{value:.3f}" for value in exact_mean_second_orders)
    )
    print(
        "weighted fixed-start terminal variance/fourth-cumulant coefficient errors="
        f"{fixed_variance_second_errors[-1]:.3e}/"
        f"{fixed_fourth_cumulant_errors[-1]:.3e}"
    )
    assert fixed_second_errors[-1] < 2e-2
    assert exact_mean_second_errors[-1] < 1.3e-2
    assert all(
        later < earlier
        for earlier, later in zip(fixed_second_errors, fixed_second_errors[1:])
    )
    assert all(
        later < earlier
        for earlier, later in zip(exact_mean_second_errors, exact_mean_second_errors[1:])
    )
    assert fixed_variance_second_errors[-1] < 5e-5
    assert fixed_fourth_cumulant_errors[-1] < 6e-3

    # Several loadings share one fast chain.  Cramer--Wold reduces the joint
    # expansion to the scalar result with loading sum_j theta_j w_j.  An
    # independent bivariate polynomial Feynman--Kac hierarchy checks the
    # covariance Gram matrix, while the complex hierarchy checks the joint
    # first characteristic correction.
    joint_growths = (0.032, -0.018)
    joint_directions = ((0.7, -0.4), (0.4, 0.9), (-0.6, 0.75))
    limiting_covariance = np.array(
        [
            [
                2.0
                * p
                * q
                * math.exp((left + right) * T)
                * _exp_integral(left + right)
                for right in joint_growths
            ]
            for left in joint_growths
        ]
    )
    joint_first_errors = []
    joint_covariance_errors = []
    for kappa in (40.0, 80.0, 160.0, 320.0, 640.0):
        a, b = kappa * q, kappa * p
        max_first_error = 0.0
        max_covariance_error = 0.0
        for start in (0, 1):
            d = q if start == 0 else -p
            _, covariance = joint_weighted_raw_moments_ode(
                start, a, b, joint_growths
            )
            max_covariance_error = max(
                max_covariance_error,
                float(np.max(np.abs(kappa * covariance - limiting_covariance))),
            )
            for direction in joint_directions:
                exact_cf = joint_weighted_stationary_centered_characteristic(
                    start, a, b, joint_growths, direction
                )
                loading = lambda tau: sum(
                    coefficient * math.exp(growth * tau)
                    for coefficient, growth in zip(direction, joint_growths)
                )
                second_energy = quad(
                    lambda tau: loading(tau) ** 2,
                    0.0,
                    T,
                    epsabs=2e-13,
                    epsrel=2e-13,
                )[0]
                third_energy = quad(
                    lambda tau: loading(tau) ** 3,
                    0.0,
                    T,
                    epsabs=2e-13,
                    epsrel=2e-13,
                )[0]
                initial_loading = loading(T)
                normal_cf = math.exp(-p * q * second_energy)
                first_polynomial = (
                    1j * d * initial_loading
                    - 1j * p * q * (q - p) * third_energy
                )
                scaled_error = math.sqrt(kappa) * (exact_cf - normal_cf)
                max_first_error = max(
                    max_first_error,
                    abs(scaled_error - first_polynomial * normal_cf),
                )
        joint_first_errors.append(max_first_error)
        joint_covariance_errors.append(max_covariance_error)
    joint_first_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(joint_first_errors, joint_first_errors[1:])
    ]
    joint_covariance_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(joint_covariance_errors, joint_covariance_errors[1:])
    ]
    print(
        "joint weighted-clock first characteristic correction residuals="
        + ", ".join(f"{value:.3e}" for value in joint_first_errors)
        + "; halving orders="
        + ", ".join(f"{value:.3f}" for value in joint_first_orders)
    )
    print(
        "joint weighted-clock scaled covariance errors="
        + ", ".join(f"{value:.3e}" for value in joint_covariance_errors)
        + "; halving orders="
        + ", ".join(f"{value:.3f}" for value in joint_covariance_orders)
    )
    assert joint_first_errors[-1] < 1.5e-2
    assert joint_covariance_errors[-1] < 2e-3
    assert all(
        later < earlier
        for earlier, later in zip(joint_first_errors, joint_first_errors[1:])
    )
    assert all(
        later < earlier
        for earlier, later in zip(joint_covariance_errors, joint_covariance_errors[1:])
    )

    for multiplier in (1, 2, 4, 8):
        a, b = multiplier * rate_12, multiplier * rate_21
        total_rate = a + b
        p = b / total_rate
        q = a / total_rate
        for start in (0, 1):
            d = q if start == 0 else -p
            probability = lambda t: p + d * math.exp(-total_rate * t)
            transition = lambda lag: p + q * math.exp(-total_rate * lag)
            weight = lambda t: math.exp(clock_growth * (T - t))
            mean_numeric = quad(
                lambda t: weight(t) * probability(t), 0.0, T, epsabs=2e-13, epsrel=2e-13
            )[0]
            second_numeric = 2.0 * quad(
                lambda s: weight(s)
                * probability(s)
                * quad(
                    lambda t: weight(t) * transition(t - s),
                    s,
                    T,
                    epsabs=2e-13,
                    epsrel=2e-13,
                )[0],
                0.0,
                T,
                epsabs=2e-13,
                epsrel=2e-13,
            )[0]
            mean, variance = weighted_occupation_moments(start, a, b, clock_growth)
            assert abs(mean - mean_numeric) < 2e-13
            assert abs(variance + mean * mean - second_numeric) < 3e-13
            exact, centered, centered_bound, stationary, stationary_bound = weighted_finite_rate_bounds(
                start, a, b, carry_rate, carry_dividend, m1, m2
            )
            centered_error = abs(exact - centered)
            stationary_error = abs(exact - stationary)
            print(
                f"weighted kappa={total_rate:g}, start {start+1}: Var(W)={variance:.10f}; "
                f"mean-clock error/bound={centered_error:.3e}/{centered_bound:.3e}; "
                f"stationary-clock error/bound={stationary_error:.3e}/{stationary_bound:.3e}"
            )
            assert centered_error <= centered_bound
            assert stationary_error <= stationary_bound


if __name__ == "__main__":
    main()
