"""All-prior, all-maturity CIR integrated-variance boundary correction."""
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm


def variance(generator, theta, kappa, xi, v0, maturity, prior, method="expm", moments=None):
    """Center V by its stationary drift to avoid subtracting O(T^2) moments."""
    size = len(theta)
    stationary = np.linalg.lstsq(np.vstack([generator.T, np.ones(size)]), np.r_[np.zeros(size), 1.], rcond=None)[0]
    level = stationary @ theta
    pairs = [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)]
    where = {pair: j for j, pair in enumerate(pairs)}
    system = np.zeros((6 * size, 6 * size))
    initial = np.zeros(6 * size)
    def block(pair):
        j = where[pair]
        return slice(j * size, (j + 1) * size)
    for a, b in pairs:
        target = block((a, b))
        system[target, target] = generator.T - b * kappa * np.eye(size)
        if a:
            system[target, block((a - 1, b + 1))] += a * np.eye(size)
            system[target, block((a - 1, b))] -= a * level * np.eye(size)
        if b:
            system[target, block((a, b - 1))] += np.diag(b * kappa * theta + b * (b - 1) * xi ** 2 / 2)
        if a == 0:
            initial[target] = v0 ** b * prior if moments is None else moments[b]
    if method == "expm":
        result = expm(maturity * system) @ initial
    else:
        solution = solve_ivp(lambda t, y: system @ y, (0, maturity), initial,
                             method="Radau", jac=system, rtol=2e-11, atol=2e-13)
        assert solution.success
        result = solution.y[:, -1]
    mean, second = result[block((1, 0))].sum(), result[block((2, 0))].sum()
    return second - mean ** 2, max(1, abs(second), mean ** 2)


def loading(time, kappa):
    return -np.expm1(-kappa * time) / kappa


def profile(time, kappa):
    x = kappa * time
    if x < .1:
        return quad(lambda u: loading(time - u, kappa) ** 2 * np.exp(-kappa * u), 0, time)[0]
    return (-np.expm1(-2 * x) - 2 * x * np.exp(-x)) / kappa ** 3


def integrated_loading(time, kappa):
    if kappa * time < .1:
        return quad(lambda u: loading(u, kappa) ** 2, 0, time)[0]
    return (time + 2 * np.expm1(-kappa * time) / kappa - np.expm1(-2 * kappa * time) / (2 * kappa)) / kappa ** 2


chains = [(np.array([[-1., 1.], [1., -1.]]), np.array([.5, .5]), 1., 2., np.array([.08, .02]))]
# Cycles are normal with uniform pi: ||P_t-Pi||_inf <= sqrt(n)*exp(-gamma*t).
for size in (3, 5):
    generator = np.roll(np.eye(size), 1, axis=1) - np.eye(size)
    chains.append((generator, np.ones(size) / size, np.sqrt(size), 1 - np.cos(2 * np.pi / size), np.linspace(.025, .1, size)))
defective = np.array([[-1., 1., 0], [0, -1., 1.], [4., 0, -4.]])
pi = np.array([4., 4., 1.]) / 9
centered = np.eye(3) - np.ones((3, 1)) @ pi[None, :]
jordan = (defective + 3 * np.eye(3)) @ centered
# P_t-Pi = exp(-3t)*(centered+t*jordan); weaken the decay to gamma=1.5.
assert np.array_equal(defective @ np.linalg.matrix_power(defective + 3 * np.eye(3), 2), np.zeros((3, 3)))
envelope = np.linalg.norm(centered, np.inf) + np.linalg.norm(jordan, np.inf) / (1.5 * np.e)
chains.append((defective, pi, envelope, 1.5, np.array([.03, .07, .12])))
kappa, xi, v0 = 2., .2, .04
checked = independent = long_checks = random_checks = 0
random_long_checks = 0
maximum_ratio = maximum_identity_error = 0.
for generator, pi, constant, decay, theta in chains:
    size = len(theta)
    assert 2 * kappa * min(theta) >= xi ** 2
    centered_theta = theta - pi @ theta
    h = np.linalg.solve(np.ones((size, 1)) @ pi[None, :] - generator, centered_theta)
    green_kubo = pi @ (centered_theta * h)
    amplitude = max(abs(centered_theta))
    priors = list(np.eye(size)) + [pi, np.ones(size) / size]
    for speed in (1., 2., 4., 8., 16., 32.):
        bound = ((4 * constant + 3 * constant ** 2) * amplitude ** 2 + 2 * xi ** 2 * constant * amplitude / kappa) / (speed * decay) ** 2
        for time in (0., .03, .3, 1.5, 20., speed ** 2):
            j = profile(time, kappa)
            i2 = integrated_loading(time, kappa)
            averaged = xi ** 2 * ((pi @ theta) * i2 + (v0 - pi @ theta) * j)
            stationary_first = averaged + 2 * kappa ** 2 * green_kubo * i2 / speed
            for prior in priors:
                exact, scale = variance(speed * generator, theta, kappa, xi, v0, time, prior)
                approximation = stationary_first + xi ** 2 * kappa * j * (prior @ h) / speed
                slack = 1e-9 * scale
                assert abs(exact - approximation) <= bound + slack
                maximum_ratio = max(maximum_ratio, max(0, abs(exact - approximation) - slack) / bound)
                checked += 1
                # Conditional point masses and shape-three Pareto laws have the
                # same chosen conditional means; the latter have no positive MGF.
                means = .025 + .03 * np.arange(size) / size
                for second_factor in (1., 4 / 3):
                    moments = np.array([prior, prior * means, second_factor * prior * means ** 2])
                    mu = sum(moments[1])
                    r = moments[1] - mu * prior
                    random_av = loading(time, kappa) ** 2 * (sum(moments[2]) - mu ** 2)
                    random_av += xi ** 2 * ((pi @ theta) * i2 + (mu - pi @ theta) * j)
                    prediction = random_av + 2 * kappa ** 2 * green_kubo * i2 / speed
                    prediction += (xi ** 2 * kappa * j * (prior @ h) + 2 * kappa * loading(time, kappa) ** 2 * (r @ h)) / speed
                    random_bound = bound + 2 * constant * amplitude * np.linalg.norm(r, 1) / (speed * decay) ** 2
                    observed, random_scale = variance(speed * generator, theta, kappa, xi, v0, time, prior, moments=moments)
                    assert abs(observed - prediction) <= random_bound + 1e-9 * random_scale
                    random_checks += 1
                    if second_factor > 1 and speed == 32 and time == speed ** 2:
                        referee, referee_scale = variance(speed * generator, theta, kappa, xi, v0, time, prior, "Radau", moments)
                        assert abs(observed - referee) <= 1e-9 * max(random_scale, referee_scale)
                        random_long_checks += 1
                if speed in (8., 32.) and time == speed ** 2:
                    referee, referee_scale = variance(speed * generator, theta, kappa, xi, v0, time, prior, "Radau")
                    assert abs(exact - referee) < 1e-9 * max(scale, referee_scale)
                    long_checks += 1
            if speed in (1., 8.) and time in (.3, 1.5):
                prior = priors[0]
                exact, scale = variance(speed * generator, theta, kappa, xi, v0, time, prior)
                path_only, path_scale = variance(speed * generator, theta, kappa, 0., v0, time, prior)
                upper = min(speed * decay * time, 80.)
                noise_memory = xi ** 2 * kappa / (speed * decay) * quad(
                    lambda u: prior @ expm(u * generator / decay) @ centered_theta * profile(time - u / (speed * decay), kappa),
                    0, upper, epsabs=1e-12)[0]
                discrepancy = abs(exact - path_only - averaged - noise_memory)
                assert discrepancy < 1e-9 * max(scale, path_scale)
                maximum_identity_error = max(maximum_identity_error, discrepancy / max(scale, path_scale))
                independent += 1
print(f"All-prior/maturity bound cases: {checked}; maximum roundoff-adjusted utilization={maximum_ratio:.9f}")
print(f"Independent noise-decomposition checks: {independent}; maximum scaled discrepancy={maximum_identity_error:.3e}")
print(f"Independent stiff-ODE long-horizon checks: {long_checks}")
print(f"Correlated finite-second-moment initial-law bound cases: {random_checks}")
print(f"Independent random-start long-horizon stiff-ODE checks: {random_long_checks}")
generator, pi, _, _, theta = chains[0]
time = 1.5
print(f"Witness parameters: theta={theta}, kappa={kappa}, xi={xi}, deterministic v0={v0}, T={time}")
limit = xi ** 2 * kappa * .03 * profile(time, kappa)
print(f"At T=1.5, J={profile(time, kappa):.10f}; predicted limit m*(Var_plus-Var_minus)={limit:.10f}")
print("m   m*(Var_plus-Var_minus)   corrected plus error")
for speed in (4., 8., 16., 32., 64., 128.):
    plus, _ = variance(speed * generator, theta, kappa, xi, v0, time, np.array([1., 0.]))
    minus, _ = variance(speed * generator, theta, kappa, xi, v0, time, np.array([0., 1.]))
    exact_difference = 2 * xi ** 2 * kappa * .03 * quad(lambda u: np.exp(-2 * u) * profile(time - u / speed, kappa), 0, speed * time)[0] / speed
    assert abs(plus - minus - exact_difference) < 1e-10
    averaged = xi ** 2 * (.05 * integrated_loading(time, kappa) - .01 * profile(time, kappa))
    approximation = averaged + kappa ** 2 * .03 ** 2 * integrated_loading(time, kappa) / speed + xi ** 2 * kappa * profile(time, kappa) * .015 / speed
    assert abs(plus - approximation) <= (7 * .03 ** 2 + 2 * xi ** 2 * .03 / kappa) / (2 * speed) ** 2 + 1e-10
    print(f"{speed:3.0f} {speed * (plus - minus):.10f} {plus - approximation:+.10e}")
assert abs(128 * (plus - minus) / limit - 1) < .01
mu, amplitude = .05, .02
means = np.array([mu + amplitude, mu - amplitude])
correlated = np.array([pi, pi * means, pi * means ** 2])
independent_moments = np.array([pi, pi * mu, pi * (mu ** 2 + amplitude ** 2)])
limit = kappa * amplitude * .03 * loading(time, kappa) ** 2
print(f"Stationary-regime same-marginals witness: mu={mu}, a={amplitude}; predicted m*(Var_corr-Var_indep)={limit:.10f}")
for speed in (4., 16., 64., 128.):
    corr, scale = variance(speed * generator, theta, kappa, xi, mu, time, pi, moments=correlated)
    indep, other_scale = variance(speed * generator, theta, kappa, xi, mu, time, pi, moments=independent_moments)
    exact_gap = 2 * kappa * loading(time, kappa) * amplitude * .03 / speed * quad(
        lambda u: np.exp(-2 * u) * loading(time - u / speed, kappa), 0, speed * time)[0]
    assert abs(corr - indep - exact_gap) <= 1e-10 * max(scale, other_scale)
    print(f"{speed:3.0f} scaled correlation gap={speed * (corr - indep):.10f}")
assert abs(128 * (corr - indep) / limit - 1) < .01
print("PASS")
