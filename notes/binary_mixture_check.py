"""Transmission through a purely absorbing binary Markovian mixture.

T(L) = E[exp(-int_0^L Sigma(y_s) ds)], y a two-state Markov chain in path length,
mean chord lengths lamA, lamB, started from its stationary law (volume fractions).
Compares the exact answer (matrix exponential; equals the Levermore-Pomraning /
Avaste-Vainikko two-exponential law) with atomic mix and with the Green-Kubo
first correction  Sigma_eff = <Sigma> - pA pB (SigA - SigB)^2 lam_c,
lam_c = lamA lamB / (lamA + lamB).  Also a Monte Carlo check of the exact law.
"""
import numpy as np
from scipy.linalg import expm

def exact(SA, SB, lA, lB, L):
    Q = np.array([[-1/lA, 1/lA], [1/lB, -1/lB]])
    p = np.array([lA, lB]) / (lA + lB)
    return p @ expm((Q - np.diag([SA, SB])) * L) @ np.ones(2)

def lp_law(SA, SB, lA, lB, L):
    pA, pB = lA/(lA+lB), lB/(lA+lB)
    m = pA*SA + pB*SB
    St = pB*SA + pA*SB + 1/lA + 1/lB
    beta = (SA-SB)**2*pA*pB
    s = np.sqrt((m-St)**2 + 4*beta)
    rp, rm = (m+St+s)/2, (m+St-s)/2
    return ((St-rm)*np.exp(-rm*L) + (rp-St)*np.exp(-rp*L))/(rp-rm)

def approx(SA, SB, lA, lB, L, order):
    pA, pB = lA/(lA+lB), lB/(lA+lB)
    lc = lA*lB/(lA+lB); d = SA-SB; m = pA*SA+pB*SB; beta = pA*pB*d*d
    logT = -m*L
    if order >= 1: logT += beta*lc*L                                   # Green-Kubo
    if order >= 2: logT += beta*(pA-pB)*d*lc**2*L - beta*lc**2          # rate + initial-layer constant
    return np.exp(logT)

def monte_carlo(SA, SB, lA, lB, L, n=200000, rng=np.random.default_rng(1)):
    pA = lA/(lA+lB); tau = np.zeros(n)
    state = rng.random(n) < pA            # True = material A
    pos = np.zeros(n); alive = np.ones(n, bool)
    while alive.any():
        lam = np.where(state, lA, lB)
        step = np.minimum(rng.exponential(lam), L-pos)
        tau += alive*step*np.where(state, SA, SB)
        pos += alive*step
        alive &= pos < L - 1e-12
        state = ~state
    return np.exp(-tau).mean(), np.exp(-tau).std()/np.sqrt(n)

if __name__ == "__main__":
    SA, SB, L = 2.0, 0.2, 3.0          # dense and thin material, slab of 3 units
    ratio = 0.5                         # lamA = ratio * lamB (unequal fractions: pA = 1/3)
    print("lam_c      exact        |err| atomic   |err| 1st    |err| 2nd   (errors in log T)")
    for lc in [0.4, 0.2, 0.1, 0.05, 0.025, 0.0125]:
        lB = lc*(1+ratio)/ratio; lA = ratio*lB
        T = exact(SA, SB, lA, lB, L)
        assert abs(T - lp_law(SA, SB, lA, lB, L)) < 1e-12
        e = [abs(np.log(approx(SA, SB, lA, lB, L, k)) - np.log(T)) for k in (0, 1, 2)]
        print(f"{lc:<9.4f} {T:.8f}   {e[0]:.3e}      {e[1]:.3e}    {e[2]:.3e}")
    lc = 0.2; lB = lc*(1+ratio)/ratio; lA = ratio*lB
    mc, se = monte_carlo(SA, SB, lA, lB, L)
    print(f"Monte Carlo at lam_c=0.2: {mc:.5f} +- {se:.5f}; exact {exact(SA,SB,lA,lB,L):.5f}; "
          f"atomic mix {np.exp(-(SA/3+2*SB/3)*L):.5f}")
