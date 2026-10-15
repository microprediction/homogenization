"""Exact speed classification for stationary two-state, common-kappa CIR.

Compare scalar formulas against generic stationary moment/Poisson solves,
and covariance quadrature against the independent degree-two semigroup.
These floating-point checks support the proof; they are not interval bounds.
"""
import math

import numpy as np
from scipy.integrate import quad

from verify_switching_kappa import (
    integrated_variance_rate,
    stationary_integrated_variance,
)


def formulas(alpha, beta, kappa, theta, a, speed):
    pi = np.array([beta, alpha]) / (alpha + beta)
    mean, abar = pi @ theta, pi @ a
    vt = pi @ (theta - mean)**2
    cov = pi @ ((theta - mean) * (a - abar))
    r = speed * (alpha + beta)
    amplitude = kappa * vt + cov / 2
    variance = abar * mean / (2 * kappa) + amplitude / (kappa + r)
    rate = abar * mean / kappa**2 + 2 * vt / r + cov / (kappa * (kappa + r))
    derivative = -2 * vt / r**2 - cov / (kappa * (kappa + r)**2)
    return mean, variance, rate, derivative, vt, cov, amplitude


def covariance(t, kappa, r, variance, vt):
    # Divided difference with a continuous branch at r=kappa.
    gap = abs(kappa - r)
    h = t * math.exp(-kappa * t) if gap == 0 else (
        math.exp(-min(kappa, r) * t) * (-math.expm1(-gap * t)) / gap
    )
    return math.exp(-kappa * t) * variance + kappa**2 * vt / (kappa + r) * h


def close(actual, expected, scale=1.0):
    tolerance = max(1e-14, 2048 * np.finfo(float).eps * max(scale, abs(expected)))
    assert abs(actual - expected) <= tolerance, (actual, expected, tolerance)


def check_model(alpha, beta, kappa, theta, a, speed, maturity=None):
    assert np.all(2 * kappa * theta > a) and np.all(a > 0)
    pi = np.array([beta, alpha]) / (alpha + beta)
    q0 = np.array([[-alpha, alpha], [beta, -beta]])
    kappas = np.full(2, kappa)
    mean, variance, rate, _, vt, _, _ = formulas(
        alpha, beta, kappa, theta, a, speed
    )
    exact_rate, exact_mean, _, second, _, _ = integrated_variance_rate(
        speed, q0, pi, kappa * theta, kappas, a
    )
    close(exact_mean, mean)
    close(second.sum() - exact_mean**2, variance, second.sum())
    close(exact_rate, rate)
    if maturity is not None:
        integrated = 2 * quad(
            lambda t: (maturity - t) * covariance(
                t, kappa, speed * (alpha + beta), variance, vt
            ), 0, maturity, epsabs=1e-14, epsrel=1e-11
        )[0]
        semigroup = stationary_integrated_variance(
            speed, maturity, q0, pi, kappa * theta, kappas, a
        )
        close(semigroup, integrated, (mean * maturity)**2 + integrated)


def main():
    rng = np.random.default_rng(20261015)
    for index in range(160):
        alpha, beta, kappa = rng.uniform(0.3, 2.0, 3)
        theta = rng.uniform(0.02, 0.2, 2)
        a = 2 * kappa * theta * rng.uniform(0.02, 0.95, 2)
        speed = math.exp(rng.uniform(-2, 3))
        check_model(alpha, beta, kappa, theta, a, speed,
                    rng.uniform(0.2, 4) if index < 32 else None)

    # All three nonconstant-mean regimes, plus constant theta/switching a.
    theta = np.array([0.03, 0.04])
    cases = [
        (theta, np.array([0.01, 0.01]), 1),
        (theta, np.array([0.03, 0.01]), 0),
        (theta, np.array([0.05, 0.001]), -1),
        (np.array([0.04, 0.04]), np.array([0.05, 0.001]), 0),
    ]
    for levels, a, sign in cases:
        base = formulas(1, 1, 1, levels, a, 1)
        vt, cov, amplitude = base[4:]
        if sign == 0:
            close(amplitude, 0)
        if sign:
            assert sign * amplitude > 0
        for speed in (0.1, 0.5, 1, 4, 16):
            check_model(1, 1, 1, levels, a, speed, 2.0)
            derivative = formulas(1, 1, 1, levels, a, speed)[3]
            if vt == 0:
                close(derivative, 0)
            elif sign >= 0:
                assert derivative < 0
        if sign < 0:
            q = math.sqrt(2 * vt / -cov)
            optimum = q / (2 * (1 - q))
            close(formulas(1, 1, 1, levels, a, optimum)[3], 0)
            assert formulas(1, 1, 1, levels, a, optimum / 2)[3] < 0
            assert formulas(1, 1, 1, levels, a, optimum * 2)[3] > 0
            print("Benchmark: alpha=beta=kappa=1; theta=(0.03,0.04); a=(0.05,0.001)")
            print(f"U-shaped benchmark: m*={optimum:.10f}")
            print("Feller margins:", 2 * levels - a)
            print(f"Var(theta)={vt:.10f}; Cov(theta,a)={cov:.10f}; A={amplitude:.10f}")
            print(f"averaged variance={np.mean(a)*np.mean(levels)/2:.10f}; "
                  f"averaged rate={np.mean(a)*np.mean(levels):.10f}")
            print("Var(v)=0.00044625-0.00003625/(1+2m)")
            print("rate=0.0008925+0.00005/(2m)-0.0001225/(1+2m)")
            for speed in (0.25, optimum, 1, 4, 16):
                result = formulas(1, 1, 1, levels, a, speed)
                print(f"m={speed:.10f}: Var(v)={result[1]:.10f}, rate={result[2]:.10f}")
    print("160 random moment/Poisson checks; 52 finite-horizon semigroup checks")
    print("Four classification cases; resonance r=kappa included")
    print("PASS")


if __name__ == "__main__":
    main()
