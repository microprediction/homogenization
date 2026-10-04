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
"""

import math

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


def cev_call(total_variance):
    """Undiscounted driftless-forward CEV call at the zero-carry benchmark."""
    forward = S0 * math.exp((R - Q) * T)
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


if __name__ == "__main__":
    main()
