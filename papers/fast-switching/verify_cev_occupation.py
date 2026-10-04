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
"""

import math

import mpmath as mp
import numpy as np
from numpy.polynomial import chebyshev as ch
from scipy.integrate import quad, solve_ivp
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
