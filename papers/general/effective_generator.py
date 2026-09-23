"""First-order effective generator for any pricing equation whose parameters are switched by a fast chain.

Model: the price u_y(t, x) in regime y solves  d_t u = L_y u + Q u,  with  L_y = Lbar + sum_k phi_k(y) A_k,
where the phi_k are the switched parameters (mean zero under pi after moving averages into Lbar) and the A_k are
the operators they multiply. With the regime drawn from pi and a regime-free payoff, the price is to first order
    d_t ubar = L_eff ubar,   L_eff = Lbar + sum_{j,k} K_jk A_j A_k,   K_jk = K(phi_j, phi_k) = -pi.(phi_j Q# phi_k),
where the correction is applied as a first-order (Duhamel) perturbation of exp(t Lbar), not exponentiated.
Splitting K into symmetric and antisymmetric parts,
    sum K_jk A_j A_k = sum_{j,k} Ksym_jk A_j A_k + sum_{j<k} Kanti_jk [A_j, A_k],   Kanti = (K - K^T) / 2,
so an irreversible chain matters only through commutators.
"""
import numpy as np


def stationary(Q):
    w, vl = np.linalg.eig(np.asarray(Q, float).T)
    p = np.real(vl[:, np.argmin(abs(w))])
    return p / p.sum()


def group_inverse(Q):
    Q = np.asarray(Q, float)
    pi = stationary(Q)
    P = np.outer(np.ones(len(pi)), pi)
    return np.linalg.inv(Q - P) + P


def gk(Q, phis):
    """Green-Kubo matrix K_jk = -pi.(phi_j~ Q# phi_k~) of the switched parameters (rows of phis)."""
    pi, Qs = stationary(Q), group_inverse(Q)
    F = np.array([np.asarray(p, float) - pi @ np.asarray(p, float) for p in phis])
    return np.array([[-pi @ (F[j] * (Qs @ F[k])) for k in range(len(F))] for j in range(len(F))])


def averages(Q, phis):
    pi = stationary(Q)
    return np.array([pi @ np.asarray(p, float) for p in phis])


def effective_generator(Lbar, As, K, part='full'):
    """Lbar + sum K_jk A_j A_k; part='sym' keeps only the symmetric part of K."""
    if part == 'sym':
        K = 0.5 * (K + K.T)
    L = np.array(Lbar, dtype=complex if np.iscomplexobj(Lbar) else float)
    for j, Aj in enumerate(As):
        for k, Ak in enumerate(As):
            L = L + K[j, k] * (Aj @ Ak)
    return L


def full_generator(Q, Lbar, As, phis):
    """The coupled system on (regime x slow state): Q kron I + I kron Lbar + sum diag(phi_k~) kron A_k."""
    Q = np.asarray(Q, float)
    n, m = Q.shape[0], Lbar.shape[0]
    pi = stationary(Q)
    G = np.kron(Q, np.eye(m)) + np.kron(np.eye(n), Lbar)
    for p, A in zip(phis, As):
        p = np.asarray(p, float)
        G = G + np.kron(np.diag(p - pi @ p), A)
    return G
