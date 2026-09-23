"""Exact simulation of a discretely observed OU process whose level and variance switch with a two-state chain:
exponential holding times for the regime, and exact Gaussian OU transitions within each holding period."""
import math
import numpy as np


def simulate(n, D, kappa, thetas, s, lam, x0, rng):
    x, y, t_next = x0, rng.integers(0, 2), rng.exponential(1 / lam)
    out = np.empty(n + 1)
    out[0] = x
    t = 0.0
    for k in range(1, n + 1):
        T = k * D
        while True:
            end = min(t_next, T)
            h = end - t
            e = math.exp(-kappa * h)
            x = x * e + thetas[y] * (1 - e) + math.sqrt(s[y] * (1 - e * e) / (2 * kappa)) * rng.standard_normal()
            t = end
            if end == T:
                break
            y = 1 - y
            t_next = t + rng.exponential(1 / lam)
        out[k] = x
    return out
