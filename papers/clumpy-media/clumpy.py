"""Transmission of a straight ray through a purely absorbing Markovian mixture of materials.

Along the ray the material y_s is a Markov chain in path length s with generator Q and stationary law pi (the
volume fractions). Material i absorbs at rate Sigma_i per unit length. The mean transmission of a slab of
thickness L is

    T(L) = E[exp(-int_0^L Sigma(y_s) ds)] = pi . expm((Q - diag Sigma) L) . 1,

the regime-switching survival problem with path length in place of time and g_i = -Sigma_i constant.

Closed forms (any number of materials), with Sigma~ = Sigma - <Sigma>, Q# the group inverse of Q:
    <Sigma> = pi . Sigma                                   atomic mix
    K   = -pi . (Sigma~ Q# Sigma~)   = int_0^inf C(s) ds    Green-Kubo, C the opacity autocovariance
    M3  =  pi . (Sigma~ Q# (Sigma~ Q# Sigma~))              = int int E[Sigma~(0) Sigma~(s) Sigma~(s+u)] ds du
    C1  =  pi . (Sigma~ Q# Q# Sigma~) = int_0^inf s C(s) ds  initial-layer constant
    log T(L) = -(<Sigma> - K + M3) L - C1 + O(third order).
Two materials, mean chord lengths lamA, lamB: K = pA pB d^2 lam_c, M3 = pA pB (pB - pA) d^3 lam_c^2,
C1 = pA pB d^2 lam_c^2, d = SigA - SigB, lam_c = lamA lamB / (lamA + lamB).
"""
import math
import os
import sys
import numpy as np
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'fast-switching'))
sys.path.insert(0, os.path.join(HERE, '..', 'general'))
from fastswitch import FastSwitch, ExpSum  # noqa: E402
from effective_generator import stationary, group_inverse  # noqa: E402


# ------------------------------------------------------------------ the medium
def binary_Q(lamA, lamB):
    """Two materials with exponential chords of mean lamA, lamB."""
    return np.array([[-1 / lamA, 1 / lamA], [1 / lamB, -1 / lamB]])


def tessellation_Q(p, lam_c):
    """Cells of a Poisson tessellation coloured independently with probabilities p: along any line the
    interfaces are Poisson with rate 1/lam_c and each new cell draws a fresh colour."""
    p = np.asarray(p, float)
    n = len(p)
    return (np.outer(np.ones(n), p) - np.eye(n)) / lam_c


def binary_lams(pA, lam_c):
    """Mean chords (lamA, lamB) with volume fraction pA and correlation length lam_c."""
    return lam_c / (1 - pA), lam_c / pA


# ------------------------------------------------------------------ references
def exact_T(Q, Sig, L):
    """Matrix exponential."""
    Q, Sig = np.asarray(Q, float), np.asarray(Sig, float)
    pi = stationary(Q)
    return float(pi @ expm((Q - np.diag(Sig)) * L) @ np.ones(len(Sig)))


def two_exponential_T(SA, SB, lamA, lamB, L):
    """The classical exact law for two materials (Levermore, Pomraning, Sanzo and Wong 1986)."""
    pA, pB = lamA / (lamA + lamB), lamB / (lamA + lamB)
    m = pA * SA + pB * SB
    St = pB * SA + pA * SB + 1 / lamA + 1 / lamB
    beta = pA * pB * (SA - SB) ** 2
    s = math.sqrt((m - St) ** 2 + 4 * beta)
    rp, rm = (m + St + s) / 2, (m + St - s) / 2
    return ((St - rm) * math.exp(-rm * L) + (rp - St) * math.exp(-rp * L)) / (rp - rm)


def two_exponential_parts(SA, SB, lamA, lamB):
    """Slow rate r_-, fast rate r_+, weight on the slow exponential."""
    pA, pB = lamA / (lamA + lamB), lamB / (lamA + lamB)
    m = pA * SA + pB * SB
    St = pB * SA + pA * SB + 1 / lamA + 1 / lamB
    beta = pA * pB * (SA - SB) ** 2
    s = math.sqrt((m - St) ** 2 + 4 * beta)
    rp, rm = (m + St + s) / 2, (m + St - s) / 2
    return rm, rp, (St - rm) / (rp - rm)


# ------------------------------------------------------------------ closed forms
def coefficients(Q, Sig):
    """<Sigma>, K, M3, C1 for any chain."""
    Q, Sig = np.asarray(Q, float), np.asarray(Sig, float)
    pi, Qs = stationary(Q), group_inverse(Q)
    St = Sig - pi @ Sig
    K = -pi @ (St * (Qs @ St))
    M3 = pi @ (St * (Qs @ (St * (Qs @ St))))
    C1 = pi @ (St * (Qs @ (Qs @ St)))
    return float(pi @ Sig), float(K), float(M3), float(C1)


def binary_coefficients(SA, SB, lamA, lamB):
    pA, pB = lamA / (lamA + lamB), lamB / (lamA + lamB)
    lc, d = lamA * lamB / (lamA + lamB), SA - SB
    b = pA * pB * d * d
    return pA * SA + pB * SB, b * lc, b * (pB - pA) * d * lc ** 2, b * lc ** 2


def log_T_closed(coef, L, order):
    """log T through order 0 (atomic mix), 1 (Green-Kubo) or 2 (second-order rate and layer constant)."""
    m, K, M3, C1 = coef
    out = -m * L
    if order >= 1:
        out += K * L
    if order >= 2:
        out += -M3 * L - C1
    return out


def engine_log_T(Q, Sig, L, orders):
    """log T through each requested order from the fast-switching engine (constant g_i = -Sigma_i)."""
    fs = FastSwitch(np.asarray(Q, float), [ExpSum.const(-float(s)) for s in Sig], order=max(orders))
    return {n: float(math.log(fs.pi @ fs.a(L, order=n))) for n in orders}


# ------------------------------------------------------------------ Monte Carlo
def mc_chain(Q, Sig, L, n, rng):
    """Sample the material sequence along the ray directly (chord-length sampling)."""
    Q, Sig = np.asarray(Q, float), np.asarray(Sig, float)
    pi = stationary(Q)
    k = len(Sig)
    rates = -np.diag(Q)
    jump = (Q - np.diag(np.diag(Q))) / rates[:, None]
    cum = np.cumsum(jump, axis=1)
    state = rng.choice(k, size=n, p=pi)
    pos, tau = np.zeros(n), np.zeros(n)
    alive = np.ones(n, bool)
    while alive.any():
        idx = np.nonzero(alive)[0]
        st = state[idx]
        step = np.minimum(rng.exponential(1 / rates[st]), L - pos[idx])
        tau[idx] += step * Sig[st]
        pos[idx] += step
        u = rng.random(len(idx))
        state[idx] = (u[:, None] > cum[st]).sum(axis=1)
        alive[idx] = pos[idx] < L - 1e-12
    x = np.exp(-tau)
    return float(x.mean()), float(x.std() / math.sqrt(n))


def mc_tessellation(p, Sig, lam_c, L, n_media, rays_per_medium, rng, r0=1.0):
    """Ray tracing through realizations of a coloured isotropic Poisson plane tessellation in three dimensions.

    Planes n.x = q with n uniform on the sphere and q uniform on [-R, R]; a line meets them at rate
    (plane density per unit q) * E|n.u| = density / 2, so the density is 2 / lam_c. Each cell (a sign pattern
    over the planes) gets an independent colour with probabilities p. Rays are segments of length L in
    uniformly random directions, centred uniformly in a ball of radius r0, all inside the ball of radius R."""
    p, Sig = np.asarray(p, float), np.asarray(Sig, float)
    R = L / 2 + r0
    vals = []
    for _ in range(n_media):
        m = rng.poisson(2 * R * 2 / lam_c)
        nrm = rng.normal(size=(m, 3))
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        q = rng.uniform(-R, R, m)
        colour = {}
        for _ in range(rays_per_medium):
            u = rng.normal(size=3); u /= np.linalg.norm(u)
            c = rng.normal(size=3); c *= r0 * rng.random() ** (1 / 3) / np.linalg.norm(c)
            x0 = c - u * L / 2
            nu = nrm @ u
            with np.errstate(divide='ignore', invalid='ignore'):
                t = (q - nrm @ x0) / nu
            t = np.sort(t[(t > 0) & (t < L)])
            edges = np.concatenate([[0.0], t, [L]])
            mids = (edges[:-1] + edges[1:]) / 2
            pts = x0[None, :] + mids[:, None] * u[None, :]
            signs = (pts @ nrm.T - q[None, :]) > 0
            packed = np.packbits(signs, axis=1)
            tau = 0.0
            for row, length in zip(packed, np.diff(edges)):
                key = row.tobytes()
                col = colour.get(key)
                if col is None:
                    col = colour[key] = rng.choice(len(p), p=p)
                tau += Sig[col] * length
            vals.append(math.exp(-tau))
    vals = np.array(vals)
    # rays in one medium are correlated: the standard error is taken over media
    per = vals.reshape(n_media, rays_per_medium).mean(axis=1)
    return float(vals.mean()), float(per.std(ddof=1) / math.sqrt(n_media))


def tau_cumulants(Q, Sig, L):
    """First three cumulants of the optical depth tau = int_0^L Sigma(y_s) ds, from moments computed exactly by the
    block matrix exponential of Van Loan: E tau^k = k! pi . [expm(M L)]_(0,k) . 1."""
    Q, D = np.asarray(Q, float), np.diag(np.asarray(Sig, float))
    n = len(Sig)
    M = np.zeros((4 * n, 4 * n))
    for k in range(4):
        M[k * n:(k + 1) * n, k * n:(k + 1) * n] = Q
        if k < 3:
            M[k * n:(k + 1) * n, (k + 1) * n:(k + 2) * n] = D
    E = expm(M * L)
    pi, one = stationary(Q), np.ones(n)
    m = [math.factorial(k) * pi @ E[0:n, k * n:(k + 1) * n] @ one for k in (1, 2, 3)]
    return m[0], m[1] - m[0] ** 2, m[2] - 3 * m[1] * m[0] + 2 * m[0] ** 3
