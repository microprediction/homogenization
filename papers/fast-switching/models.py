"""Models that reduce to a' = (Q + diag g(t)) a, and so fall to the fast-switching engine.

Each builder returns (g, gfuncs, prefactor): g for the engine (ExpSum or Cheb), gfuncs as plain callables for an
independent numerical solution, and the x-dependent factor that multiplies a_i(t).
"""
import math
import cmath
import numpy as np
from fastswitch import ExpSum, Cheb


# ---------------------------------------------------------------- Gaussian factors (sums of exponentials)
def gaussian_factors(kappas, thetas, sigmas, rhos, weights):
    """Factors dx_j = kappa_j (theta_j[y] - x_j) dt + sigma_j[y] dW_j, corr(dW_j, dW_l) = rhos[y][j][l].
    Quantity E[exp(-int sum_j c_j x_j)] with c = weights: prefactor exp(-sum_j c_j B_j x_j),
    B_j = c_j (1 - exp(-kappa_j t)) / kappa_j, and
    g_i = -sum_j kappa_j theta_j[i] B_j + (1/2) sum_{j,l} rho_i[j][l] sigma_j[i] sigma_l[i] B_j B_l."""
    J, n = len(kappas), len(thetas[0])
    Bs = [ExpSum({0: weights[j] / kappas[j], kappas[j]: -weights[j] / kappas[j]}) for j in range(J)]
    g = []
    for i in range(n):
        gi = ExpSum()
        for j in range(J):
            gi = gi + Bs[j].scale(-kappas[j] * thetas[j][i])
            for l in range(J):
                gi = gi + (Bs[j] * Bs[l]).scale(0.5 * rhos[i][j][l] * sigmas[j][i] * sigmas[l][i])
        g.append(gi)
    gfuncs = [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in g]

    def prefactor(t, xs):
        return math.exp(-sum(Bs[j].value(t) * xs[j] for j in range(J)))
    return g, gfuncs, prefactor


# ---------------------------------------------------------------- CIR with a switching mean level (Chebyshev)
def cir_switching_mean(kappa, thetas, sigma, T):
    """dx = kappa (theta[y] - x) dt + sigma sqrt(x) dW. B solves B' = 1 - kappa B - sigma^2 B^2 / 2, B(0) = 0,
    independent of the regime, so u_i = exp(-B x) a_i with g_i = -kappa theta_i B."""
    h = math.sqrt(kappa ** 2 + 2 * sigma ** 2)

    def B(t):
        e = math.exp(h * t) - 1
        return 2 * e / ((h + kappa) * e + 2 * h)
    gfuncs = [(lambda th: (lambda t: -kappa * th * B(t)))(th) for th in thetas]
    Bc = Cheb.fit(B, T, 80)
    g = [Bc.scale(-kappa * th) for th in thetas]
    return g, gfuncs, (lambda t, x: math.exp(-B(t) * x)), B


# ---------------------------------------------------------------- Vasicek with jumps at a switching intensity
def vasicek_jumps(kappa, thetas, sigmas, intensities, jump_mean, T):
    """dx = kappa (theta[y] - x) dt + sigma[y] dW + dJ, J compound Poisson at rate intensities[y] with
    exponential jumps of mean m. u_i = exp(-B x) a_i, g_i = -kappa theta_i B + sigma_i^2 B^2 / 2
    + l_i (1 / (1 + m B) - 1), B = (1 - exp(-kappa t)) / kappa."""
    def B(t):
        return (1 - math.exp(-kappa * t)) / kappa

    def gf(th, s, l):
        return lambda t: -kappa * th * B(t) + 0.5 * s * s * B(t) ** 2 + l * (1 / (1 + jump_mean * B(t)) - 1)
    gfuncs = [gf(thetas[i], sigmas[i], intensities[i]) for i in range(len(thetas))]
    g = [Cheb.fit(f, T, 80) for f in gfuncs]
    return g, gfuncs, (lambda t, x: math.exp(-B(t) * x))


# ---------------------------------------------------------------- Markov-modulated Poisson counts
def mmpp(rates, z):
    """N counts at rate rates[y]. a_i(t) = E[z^N_t | y_0 = i] solves a' = (Q + (z - 1) diag rates) a."""
    g = [ExpSum({0: (z - 1) * r}) for r in rates]
    gfuncs = [(lambda c: (lambda t: c))((z - 1) * r) for r in rates]
    return g, gfuncs


# ---------------------------------------------------------------- Heston with a switching long-run variance
def heston_switching_theta(u, kappa, thetas, xi, rho, T):
    """Log-price X with dX = -v/2 dt + sqrt(v) dW, dv = kappa (theta[y] - v) dt + xi sqrt(v) dZ, corr rho.
    E[exp(i u X_T)] = exp(i u X_0 + D(T) v_0) a_i(T), with D the Heston Riccati solution (regime free) and
    g_i = kappa theta_i D."""
    d = cmath.sqrt((rho * xi * 1j * u - kappa) ** 2 + xi ** 2 * (1j * u + u * u))
    gm = (kappa - rho * xi * 1j * u - d) / (kappa - rho * xi * 1j * u + d)

    def D(t):
        e = cmath.exp(-d * t)
        return (kappa - rho * xi * 1j * u - d) / xi ** 2 * (1 - e) / (1 - gm * e)
    gfuncs = [(lambda th: (lambda t: kappa * th * D(t)))(th) for th in thetas]
    Dc = Cheb.fit(D, T, 80)
    g = [Dc.scale(kappa * th) for th in thetas]
    return g, gfuncs, D
