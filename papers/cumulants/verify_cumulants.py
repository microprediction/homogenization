"""Certificate for the integrated-variance cumulant theorem.

The variance follows a CIR process whose long-run level is selected by a
stationary two-state chain.  The certificate compares two independent
calculations:

1. a closed covariance decomposition of the integrated variance; and
2. a matrix exponential for every mixed polynomial moment E[V^a v^b 1_{Y=i}]
   of total degree at most four.

It also checks the exact normal-mixture identities for the Brownian return and
the risk-neutral log return, the uniform finite-rate error bound, and an exact
joint-cumulant decomposition when the price and variance Brownian motions are
correlated.  The leveraged calculation uses a separate closed polynomial
moment system for (M,V,v,Y).  Finally, exact switched affine transforms verify
the first-order European option correction both without and with leverage.
"""
import math

import numpy as np
from scipy.integrate import quad
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.special import roots_legendre
from scipy.stats import norm


KAPPA = 2.0
XI = 0.2
THETA = np.array([0.08, 0.02])
V0 = 0.04
T = 1.5
SPEEDS = [1, 2, 4, 8, 16, 32]


def stationary(q):
    """Stationary row vector of an irreducible row-generator."""
    n = len(q)
    return np.linalg.lstsq(
        np.vstack([q.T, np.ones(n)]),
        np.r_[np.zeros(n), 1.0],
        rcond=None,
    )[0]


def cumulants(raw):
    """First four cumulants from raw moments [1, m1, ..., m4]."""
    m1, m2, m3, m4 = raw[1:5]
    return np.array(
        [
            m1,
            m2 - m1**2,
            m3 - 3 * m2 * m1 + 2 * m1**3,
            m4 - 4 * m3 * m1 - 3 * m2**2 + 12 * m2 * m1**2 - 6 * m1**4,
        ]
    )


def integrated_cir_moments(q, theta, kappa, xi, v0, maturity, order=4):
    """Raw moments of V=int_0^T v_s ds by a polynomial moment ODE.

    For z_{a,b,i}=E[V^a v^b 1_{Y=i}], Ito's formula gives

      z'_{a,b} = (Q^T-b*kappa I)z_{a,b}
                 + a z_{a-1,b+1}
                 + diag(b*kappa*theta + b(b-1)xi^2/2) z_{a,b-1}.

    The system closes at each total polynomial degree.
    """
    q = np.asarray(q, float)
    theta = np.asarray(theta, float)
    n = len(theta)
    pairs = [(a, degree - a) for degree in range(order + 1) for a in range(degree + 1)]
    where = {pair: j for j, pair in enumerate(pairs)}
    size = n * len(pairs)
    generator = np.zeros((size, size))
    initial = np.zeros(size)
    pi = stationary(q)

    def block(pair):
        j = where[pair]
        return slice(j * n, (j + 1) * n)

    for a, b in pairs:
        target = block((a, b))
        generator[target, target] += q.T - b * kappa * np.eye(n)
        if a:
            generator[target, block((a - 1, b + 1))] += a * np.eye(n)
        if b:
            coefficient = b * kappa * theta + 0.5 * b * (b - 1) * xi**2
            generator[target, block((a, b - 1))] += np.diag(coefficient)
        if a == 0:
            initial[target] = v0**b * pi

    solution = expm(maturity * generator) @ initial
    return np.array([1.0] + [solution[block((a, 0))].sum() for a in range(1, order + 1)])


def bfun(t, kappa=KAPPA):
    return -math.expm1(-kappa * t) / kappa


def i2(maturity=T, kappa=KAPPA):
    """Integral_0^T B_kappa(t)^2 dt."""
    return (
        maturity
        - 2 * (1 - math.exp(-kappa * maturity)) / kappa
        + (1 - math.exp(-2 * kappa * maturity)) / (2 * kappa)
    ) / kappa**2


def overlap(u, maturity=T, kappa=KAPPA):
    """H_T(u)=Integral_0^(T-u) B(x)B(x+u) dx, in closed form."""
    length = maturity - u
    if length <= 0:
        return 0.0
    e = math.exp(-kappa * u)
    return (
        length
        - (1 + e) * (1 - math.exp(-kappa * length)) / kappa
        + e * (1 - math.exp(-2 * kappa * length)) / (2 * kappa)
    ) / kappa**2


def averaged_cir_variance(maturity=T):
    theta_bar = float(THETA.mean())
    return XI**2 * quad(
        lambda s: bfun(maturity - s) ** 2
        * (theta_bar + (V0 - theta_bar) * math.exp(-KAPPA * s)),
        0,
        maturity,
        epsabs=2e-14,
    )[0]


def exact_regime_variance(speed, maturity=T):
    delta = float((THETA[0] - THETA[1]) / 2)
    return 2 * KAPPA**2 * quad(
        lambda u: delta**2 * math.exp(-2 * speed * u) * overlap(u, maturity),
        0,
        maturity,
        epsabs=2e-14,
    )[0]


def raw_log_return_moments(raw_v):
    """Raw moments of X=-V/2+sqrt(V)Z, Z standard normal."""
    e1, e2, e3, e4 = raw_v[1:5]
    return np.array(
        [
            1.0,
            -0.5 * e1,
            e1 + 0.25 * e2,
            -1.5 * e2 - 0.125 * e3,
            3 * e2 + 1.5 * e3 + 0.0625 * e4,
        ]
    )


def leveraged_joint_moments(q, theta, kappa, xi, v0, maturity, rho, order=4):
    """Raw E[M^a V^b] for leveraged CIR, for every a+b <= order.

    Here dM=sqrt(v)dW, dV=v dt, and d<W,B>=rho dt where B drives v.  For
    z_{a,b,c,i}=E[M^a V^b v^c 1_{Y=i}], Ito's formula gives

      z'_{a,b,c} = (Q^T-c*kappa I) z_{a,b,c}
        + b z_{a,b-1,c+1}
        + a(a-1)/2 z_{a-2,b,c+1}
        + diag(c*kappa*theta+c(c-1)xi^2/2) z_{a,b,c-1}
        + rho*xi*a*c z_{a-1,b,c}.

    Total polynomial degree never increases, so all moments through degree
    four form one finite linear system.
    """
    q = np.asarray(q, float)
    theta = np.asarray(theta, float)
    n = len(theta)
    triples = [
        (a, b, degree - a - b)
        for degree in range(order + 1)
        for a in range(degree + 1)
        for b in range(degree - a + 1)
    ]
    where = {triple: j for j, triple in enumerate(triples)}
    size = n * len(triples)
    generator = np.zeros((size, size))
    initial = np.zeros(size)
    pi = stationary(q)

    def block(triple):
        j = where[triple]
        return slice(j * n, (j + 1) * n)

    for a, b, c in triples:
        target = block((a, b, c))
        generator[target, target] += q.T - c * kappa * np.eye(n)
        if b:
            generator[target, block((a, b - 1, c + 1))] += b * np.eye(n)
        if a >= 2:
            generator[target, block((a - 2, b, c + 1))] += 0.5 * a * (a - 1) * np.eye(n)
        if c:
            coefficient = c * kappa * theta + 0.5 * c * (c - 1) * xi**2
            generator[target, block((a, b, c - 1))] += np.diag(coefficient)
        if a and c:
            generator[target, block((a - 1, b, c))] += rho * xi * a * c * np.eye(n)
        if a == 0 and b == 0:
            initial[target] = v0**c * pi

    solution = expm(maturity * generator) @ initial
    return {
        (a, b): solution[block((a, b, 0))].sum()
        for a in range(order + 1)
        for b in range(order + 1 - a)
    }


def set_partitions(items):
    """Yield each set partition once; only used for at most four labels."""
    if not items:
        yield []
        return
    first, rest = items[0], items[1:]
    for partition in set_partitions(rest):
        yield [[first]] + [block[:] for block in partition]
        for j in range(len(partition)):
            yield [
                (block + [first]) if k == j else block[:]
                for k, block in enumerate(partition)
            ]


def joint_cumulant(raw, m_count, v_count):
    """Joint cumulant with m_count copies of M and v_count copies of V."""
    labels = ["M"] * m_count + ["V"] * v_count
    answer = 0.0
    for partition in set_partitions(labels):
        blocks = len(partition)
        product = 1.0
        for block in partition:
            product *= raw[(block.count("M"), block.count("V"))]
        answer += (-1) ** (blocks - 1) * math.factorial(blocks - 1) * product
    return answer


def log_return_raw_moments(joint_raw):
    """Raw moments of X=M-V/2 from joint moments of (M,V)."""
    return np.array(
        [
            sum(
                math.comb(order, v_power)
                * (-0.5) ** v_power
                * joint_raw[(order - v_power, v_power)]
                for v_power in range(order + 1)
            )
            for order in range(5)
        ]
    )


def affine_log_mgf(q, theta, kappa, xi, v0, maturity, rho, argument):
    """Log E exp(argument*(M-V/2)) from the affine Riccati system."""
    q = np.asarray(q, float)
    theta = np.asarray(theta, float)
    n = len(theta)
    u = complex(argument)
    w = -0.5 * u

    def rhs(_, state):
        b = state[0]
        amplitude = state[1:]
        b_prime = (
            -kappa * b
            + 0.5 * xi**2 * b**2
            + rho * xi * u * b
            + 0.5 * u**2
            + w
        )
        amplitude_prime = q @ amplitude + kappa * theta * b * amplitude
        return np.r_[b_prime, amplitude_prime]

    solution = solve_ivp(
        rhs,
        (0.0, maturity),
        np.r_[0.0j, np.ones(n, dtype=complex)],
        method="DOP853",
        rtol=2e-12,
        atol=2e-14,
    ).y[:, -1]
    transform = np.exp(solution[0] * v0) * np.dot(stationary(q), solution[1:])
    return np.log(transform)


def affine_cumulant_four(q, theta, kappa, xi, v0, maturity, rho):
    """Fourth cumulant by Cauchy coefficient extraction of the affine log MGF."""
    radius = 0.18
    order = 32
    angles = 2 * np.pi * np.arange(order) / order
    values = np.array(
        [
            affine_log_mgf(
                q, theta, kappa, xi, v0, maturity, rho, radius * np.exp(1j * angle)
            )
            for angle in angles
        ]
    )
    coefficient = np.mean(values * np.exp(-4j * angles)) / radius**4
    return float((math.factorial(4) * coefficient).real)


def averaged_transform_and_correction(arguments, maturity=T, rho=0.0):
    """Averaged-Heston MGF and coefficient of its 1/m switching correction."""
    arguments = np.asarray(arguments, complex)
    count = len(arguments)
    theta_bar = float(THETA.mean())
    delta = float((THETA[0] - THETA[1]) / 2)

    def rhs(_, state):
        b = state[:count]
        db = (
            -KAPPA * b
            + 0.5 * XI**2 * b**2
            + rho * XI * arguments * b
            + 0.5 * (arguments**2 - arguments)
        )
        return np.r_[db, b, b**2]

    solution = solve_ivp(
        rhs,
        (0.0, maturity),
        np.zeros(3 * count, dtype=complex),
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    ).y[:, -1]
    b = solution[:count]
    integral_b = solution[count : 2 * count]
    integral_b2 = solution[2 * count :]
    averaged = np.exp(b * V0 + KAPPA * theta_bar * integral_b)
    correction = 0.5 * KAPPA**2 * delta**2 * integral_b2
    return averaged, correction


def exact_switched_transform(speed, arguments, maturity=T, rho=0.0):
    """Exact MGF of the stationary-start, symmetric two-state switched CIR."""
    arguments = np.asarray(arguments, complex)
    count = len(arguments)

    def rhs(_, state):
        b = state[:count]
        plus = state[count : 2 * count]
        minus = state[2 * count :]
        db = (
            -KAPPA * b
            + 0.5 * XI**2 * b**2
            + rho * XI * arguments * b
            + 0.5 * (arguments**2 - arguments)
        )
        return np.r_[
            db,
            speed * (minus - plus) + KAPPA * THETA[0] * b * plus,
            speed * (plus - minus) + KAPPA * THETA[1] * b * minus,
        ]

    solution = solve_ivp(
        rhs,
        (0.0, maturity),
        np.r_[np.zeros(count, dtype=complex), np.ones(2 * count, dtype=complex)],
        method="DOP853",
        rtol=2e-11,
        atol=2e-13,
    ).y[:, -1]
    b = solution[:count]
    return np.exp(b * V0) * 0.5 * (
        solution[count : 2 * count] + solution[2 * count :]
    )


def call_price_terms(
    speed=None, strike=1.0, alpha=1.0, cutoff=80.0, order=320, rho=0.0
):
    """Carr--Madan call price and explicit first-order coefficient.

    If speed is None, return the averaged price and the coefficient c1 in
    C_m=C_av+c1/m+O(m^-2).  Otherwise return the exact switched price.
    """
    nodes, weights = roots_legendre(order)
    frequencies = 0.5 * cutoff * (nodes + 1)
    weights = 0.5 * cutoff * weights
    arguments = alpha + 1 + 1j * frequencies
    denominator = (
        alpha**2
        + alpha
        - frequencies**2
        + 1j * (2 * alpha + 1) * frequencies
    )
    oscillation = np.exp(-1j * frequencies * math.log(strike))
    scale = strike ** (-alpha) / math.pi

    if speed is None:
        transform, log_correction = averaged_transform_and_correction(
            arguments, rho=rho
        )
        price = scale * np.dot(weights, np.real(oscillation * transform / denominator))
        correction = scale * np.dot(
            weights, np.real(oscillation * transform * log_correction / denominator)
        )
        return float(price), float(correction)

    transform = exact_switched_transform(speed, arguments, rho=rho)
    price = scale * np.dot(weights, np.real(oscillation * transform / denominator))
    return float(price)


def black_scholes_call(strike, volatility, maturity=T):
    """Normalized zero-rate Black--Scholes call with spot one."""
    scale = volatility * math.sqrt(maturity)
    if scale <= 0:
        return max(1.0 - strike, 0.0)
    d1 = (-math.log(strike) + 0.5 * scale**2) / scale
    d2 = d1 - scale
    return float(norm.cdf(d1) - strike * norm.cdf(d2))


def black_scholes_vega(strike, volatility, maturity=T):
    """Derivative of the normalized call with respect to volatility."""
    scale = volatility * math.sqrt(maturity)
    d1 = (-math.log(strike) + 0.5 * scale**2) / scale
    return float(norm.pdf(d1) * math.sqrt(maturity))


def implied_volatility(strike, price, maturity=T):
    """Black--Scholes implied volatility on the positive-volatility branch."""
    intrinsic = max(1.0 - strike, 0.0)
    if not intrinsic < price < 1.0:
        raise ValueError("call price must lie strictly inside arbitrage bounds")
    return float(
        brentq(
            lambda volatility: black_scholes_call(
                strike, volatility, maturity
            ) - price,
            1e-10,
            5.0,
            xtol=2e-14,
        )
    )


def black_scholes_fourier_check(total_variance=0.06, alpha=1.0, cutoff=80.0, order=640):
    """Check the damped inversion against the closed ATM Black--Scholes call."""
    nodes, weights = roots_legendre(order)
    frequencies = 0.5 * cutoff * (nodes + 1)
    weights = 0.5 * cutoff * weights
    arguments = alpha + 1 + 1j * frequencies
    transform = np.exp(0.5 * total_variance * (arguments**2 - arguments))
    denominator = (
        alpha**2
        + alpha
        - frequencies**2
        + 1j * (2 * alpha + 1) * frequencies
    )
    fourier = np.dot(weights, np.real(transform / denominator)) / math.pi
    closed = math.erf(math.sqrt(total_variance) / (2 * math.sqrt(2)))
    return float(fourier), closed


def main():
    assert 2 * KAPPA * THETA.min() > XI**2  # strict positivity for every regime
    baseline = averaged_cir_variance()
    delta = float((THETA[0] - THETA[1]) / 2)
    residuals = []

    print("m  kappa4(M), exact   first order      residual       m^2 residual    kappa4(X)")
    for speed in SPEEDS:
        q = speed * np.array([[-1.0, 1.0], [1.0, -1.0]])
        raw_v = integrated_cir_moments(q, THETA, KAPPA, XI, V0, T)
        k_v = cumulants(raw_v)

        decomposition = baseline + exact_regime_variance(speed)
        assert abs(k_v[1] - decomposition) < 2e-13

        # Conditional normality: M=sqrt(V)Z and X=-V/2+sqrt(V)Z.
        kappa4_m = 3 * k_v[1]
        kappa4_x_formula = 3 * k_v[1] + 1.5 * k_v[2] + k_v[3] / 16
        kappa4_x_direct = cumulants(raw_log_return_moments(raw_v))[3]
        assert abs(kappa4_x_formula - kappa4_x_direct) < 2e-13

        green_kubo = delta**2 / (2 * speed)
        first_order = 3 * (baseline + 2 * KAPPA**2 * green_kubo * i2())
        residual = kappa4_m - first_order
        residuals.append(abs(residual))

        # The theorem gives |R_4| <= 12 int_0^infty u |C_m(u)| du.
        bound = 3 * delta**2 / speed**2
        assert abs(residual) <= bound + 2e-14
        print(
            f"{speed:2d}  {kappa4_m:.10f}      {first_order:.10f}"
            f"  {residual:+.3e}    {speed**2 * residual:+.3e}    {kappa4_x_formula:.10f}"
        )

    rate = math.log(residuals[-2] / residuals[-1], 2)
    print(f"last residual rate {rate:.3f} (want 2)")
    assert 1.8 < rate < 2.2

    # Under leverage, X=M-V/2 still has an exact fourth-cumulant
    # decomposition, but the four normal-mixture reductions no longer hold.
    speed = 8.0
    q = speed * np.array([[-1.0, 1.0], [1.0, -1.0]])
    print("\nrho   kappa4(X)   independent shortcut   error      mixed corrections")
    for rho in (0.0, -0.7, 0.7):
        raw = leveraged_joint_moments(q, THETA, KAPPA, XI, V0, T, rho)
        k40 = joint_cumulant(raw, 4, 0)
        k31 = joint_cumulant(raw, 3, 1)
        k22 = joint_cumulant(raw, 2, 2)
        k13 = joint_cumulant(raw, 1, 3)
        k04 = joint_cumulant(raw, 0, 4)
        exact_identity = k40 - 2 * k31 + 1.5 * k22 - 0.5 * k13 + k04 / 16
        direct = cumulants(log_return_raw_moments(raw))[3]
        assert abs(exact_identity - direct) < 3e-13
        affine = affine_cumulant_four(q, THETA, KAPPA, XI, V0, T, rho)
        assert abs(affine - direct) < 3e-9

        independent_shortcut = (
            3 * joint_cumulant(raw, 0, 2)
            + 1.5 * joint_cumulant(raw, 0, 3)
            + k04 / 16
        )
        if rho == 0:
            assert abs(k40 - 3 * joint_cumulant(raw, 0, 2)) < 3e-13
            assert abs(k31) < 3e-13
            assert abs(k22 - joint_cumulant(raw, 0, 3)) < 3e-13
            assert abs(k13) < 3e-13
            assert abs(direct - independent_shortcut) < 3e-13
        else:
            assert abs(direct - independent_shortcut) > 1e-5
        print(
            f"{rho:+.1f}  {direct:+.9f}   {independent_shortcut:+.9f}"
            f"   {direct-independent_shortcut:+.3e}"
            f"   ({k40:+.3e},{-2*k31:+.3e},{1.5*k22:+.3e},{-0.5*k13:+.3e},{k04/16:+.3e})"
        )

    # The same two-state amplitude gives an explicit European call correction.
    # This is a fixed-maturity, damped-Fourier statement; it is not the
    # maturity-uniform cumulant bound proved above.
    fourier_bs, closed_bs = black_scholes_fourier_check()
    assert abs(fourier_bs - closed_bs) < 2e-10
    for option_rho in (0.0, -0.7):
        averaged_call, call_coefficient = call_price_terms(rho=option_rho)
        averaged_call_fine, call_coefficient_fine = call_price_terms(
            order=640, rho=option_rho
        )
        assert abs(averaged_call - averaged_call_fine) < 2e-10
        assert abs(call_coefficient - call_coefficient_fine) < 2e-10
        coarse_exact = call_price_terms(speed=8, rho=option_rho)
        fine_exact = call_price_terms(speed=8, order=640, rho=option_rho)
        assert abs(coarse_exact - fine_exact) < 2e-10
        option_residuals = []
        print(
            f"\nrho={option_rho:+.1f}: m  exact call    averaged + c1/m"
            "    residual      m^2 residual"
        )
        for speed in (4, 8, 16, 32):
            exact_call = call_price_terms(speed=speed, rho=option_rho)
            approximation = averaged_call + call_coefficient / speed
            residual = exact_call - approximation
            option_residuals.append(abs(residual))
            print(
                f"{speed:2d}  {exact_call:.10f}   {approximation:.10f}"
                f"   {residual:+.3e}   {speed**2*residual:+.3e}"
            )
        option_rate = math.log(option_residuals[-2] / option_residuals[-1], 2)
        assert 1.8 < option_rate < 2.2
        print(
            f"averaged call {averaged_call:.10f}; c1 {call_coefficient:+.10f}; "
            f"last residual rate {option_rate:.3f} (want 2)"
        )

    # On any compact strike set whose averaged implied volatilities stay away
    # from zero, Black--Scholes vega is bounded below.  The implicit-function
    # theorem therefore transports the price expansion to implied volatility,
    # with coefficient c1 / vega.  Check the leveraged smile at three strikes.
    print("\nLeveraged implied-volatility correction (rho=-0.7)")
    print(
        " strike   averaged IV     IV coefficient    last residual rate"
    )
    for strike in (0.85, 1.0, 1.15):
        averaged_call, call_coefficient = call_price_terms(
            strike=strike, rho=-0.7, order=640
        )
        coarse_call, coarse_coefficient = call_price_terms(
            strike=strike, rho=-0.7, order=320
        )
        assert abs(averaged_call - coarse_call) < 5e-10
        assert abs(call_coefficient - coarse_coefficient) < 5e-10
        averaged_iv = implied_volatility(strike, averaged_call)
        vega = black_scholes_vega(strike, averaged_iv)
        assert vega > 0.35
        iv_coefficient = call_coefficient / vega
        iv_residuals = []
        for speed in (4, 8, 16, 32):
            exact_call = call_price_terms(
                speed=speed, strike=strike, rho=-0.7, order=640
            )
            exact_iv = implied_volatility(strike, exact_call)
            iv_residuals.append(
                abs(exact_iv - averaged_iv - iv_coefficient / speed)
            )
        iv_rate = math.log(iv_residuals[-2] / iv_residuals[-1], 2)
        assert 1.9 < iv_rate < 2.2
        print(
            f" {strike:5.2f}     {averaged_iv:.10f}"
            f"     {iv_coefficient:+.10f}          {iv_rate:.3f}"
        )

    print(
        "PASS: cumulants, leverage, and the explicit option-price and "
        "implied-volatility corrections"
    )


if __name__ == "__main__":
    main()
