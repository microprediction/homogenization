"""Early growth of an epidemic whose rates are switched by a fast Markov chain.

Linearized model: x' = A(y_t) x, x = infected compartments, y_t a Markov chain with generator Q (rows sum to
zero, Q_ij = rate i -> j) and stationary law pi. Write A_y = Abar + sum_k phi_k(y) B_k with pi . phi_k = 0.
Let r = lambda(Abar) be the principal eigenvalue and xbar, ybar its right and left eigenvectors, ybar . xbar = 1.
The Green-Kubo matrix is K_lk = int_0^inf Cov(phi_l(y_0), phi_k(y_t)) dt  (l earlier, k later).

Two growth rates:
  mean growth    Lambda_mean = principal eigenvalue of  Q^T (x) I + blockdiag(A_i)    (E[x_t] ~ e^{Lambda_mean t})
                 ~ r + sum_{k,l} K_lk ybar^T B_k B_l xbar
  almost-sure    Lambda_as   = lim (1/t) log |x_t|                                  (typical path, extinction)
                 ~ r + sum_{k,l} K_lk ybar^T B_k (I - xbar ybar^T) B_l xbar         (Monmarche-Schreiber-Strickler)
  gap            Lambda_mean - Lambda_as ~ K(rho, rho) >= 0,  rho(y) = ybar^T A_y xbar  (projected rate)

SIR (x = I):          A_y = beta_y - gamma.
SEIR (x = (E, I)):    A_y = [[-sigma, beta_y], [sigma, -gamma_y]].
"""
import os, sys
import numpy as np
from scipy.special import roots_jacobi
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'general'))
from effective_generator import stationary, gk  # noqa: E402

E_I = np.array([[0.0, 1.0], [0.0, 0.0]])    # operator multiplied by beta in SEIR: e_E e_I^T (nilpotent)
I_I = np.array([[0.0, 0.0], [0.0, -1.0]])   # operator multiplied by gamma in SEIR: -e_I e_I^T


# ------------------------------------------------------------------ closed forms
def two_state_K(dbeta, lam):
    """K for a symmetric two-state chain switching at rate lam each way, parameter swinging by +-dbeta."""
    return dbeta ** 2 / (2 * lam)


def sir_mean_two_state(beta1, beta2, gamma, lam):
    """Exact mean growth of linearized SIR, two-state symmetric switching of beta at rate lam."""
    bb, bt = (beta1 + beta2) / 2, (beta1 - beta2) / 2
    return bb - gamma + np.sqrt(lam ** 2 + bt ** 2) - lam


def seir_rate(beta, sigma, gamma):
    """Principal eigenvalue of [[-sigma, beta], [sigma, -gamma]]."""
    D = (sigma - gamma) ** 2 + 4 * sigma * beta
    return (-(sigma + gamma) + np.sqrt(D)) / 2


def seir_vectors(beta, sigma, gamma):
    """r, xbar, ybar, sqrt(D) for the averaged SEIR matrix; xbar = (r + gamma, sigma), ybar . xbar = 1."""
    D = (sigma - gamma) ** 2 + 4 * sigma * beta
    r = (-(sigma + gamma) + np.sqrt(D)) / 2
    x = np.array([r + gamma, sigma])
    y = np.array([sigma, r + sigma]) / (sigma * np.sqrt(D))
    return r, x, y, np.sqrt(D)


def seir_beta_mean_two_state(beta1, beta2, sigma, gamma, lam, tol=1e-15):
    """Exact mean growth for SEIR with beta on a symmetric two-state chain: mu = r(beta_eff(mu)),
    beta_eff = bbar + bt^2 sigma / ((mu + 2 lam + sigma)(mu + 2 lam + gamma) - sigma bbar)."""
    bb, bt = (beta1 + beta2) / 2, (beta1 - beta2) / 2
    mu = seir_rate(bb, sigma, gamma)
    for _ in range(200):
        z = mu + 2 * lam
        new = seir_rate(bb + bt ** 2 * sigma / ((z + sigma) * (z + gamma) - sigma * bb), sigma, gamma)
        if abs(new - mu) < tol:
            return new
        mu = new
    return mu


def seir_beta_formulas(beta1, beta2, sigma, gamma, lam):
    """First-order a.s. growth and second-order mean growth, SEIR, beta on symmetric two-state chain."""
    bb, bt = (beta1 + beta2) / 2, (beta1 - beta2) / 2
    r, x, y, sD = seir_vectors(bb, sigma, gamma)
    K = two_state_K(bt, lam)
    return {'r': r, 'K': K, 'yE_xI': sigma / sD, 'as1': r - K * sigma ** 2 / sD ** 2,
            'mean2': r + bt ** 2 * sigma ** 2 / (4 * lam ** 2 * sD)}


def seir_beta_lyapunov_two_state(beta1, beta2, sigma, gamma, rate12, rate21=None, nodes=128):
    """Exact top Lyapunov exponent for two-state SEIR transmission switching.

    Put z = E/I.  In regime i,

        z' = F_i(z) = beta_i + (gamma-sigma) z - sigma z^2.

    If beta_hi > beta_lo, the projective process is confined between the
    positive roots a_lo < z < a_hi.  Its stationary densities solve

        (F_hi p_hi)' = -rate_hi,lo p_hi + rate_lo,hi p_lo,
        F_hi p_hi + F_lo p_lo = 0.

    Factoring F_i = -sigma (z-a_i)(z-b_i), b_i < 0 < a_i,
    gives an explicit beta-weight density.  Gauss-Jacobi quadrature evaluates
    its normalization and first moment without endpoint singularities.  Since
    (log I)' = sigma z-gamma and z stays bounded, the top exponent is
    sigma E[z]-gamma.
    """
    if rate21 is None:
        rate21 = rate12
    if min(beta1, beta2, sigma, gamma, rate12, rate21) <= 0:
        raise ValueError("rates and transmission parameters must be positive")
    if abs(beta1 - beta2) <= 1e-15 * max(beta1, beta2):
        pi1 = rate21 / (rate12 + rate21)
        return {'lyapunov': seir_rate(beta1, sigma, gamma), 'mean_z': None,
                'state_mass': [pi1, 1 - pi1], 'roots': None, 'powers': None}

    # Re-label by transmission level.  a is high -> low and b is low -> high.
    if beta1 > beta2:
        beta_hi, beta_lo, a, b = beta1, beta2, rate12, rate21
        high_first = True
    else:
        beta_hi, beta_lo, a, b = beta2, beta1, rate21, rate12
        high_first = False

    def projective_roots(beta):
        disc = np.sqrt((gamma - sigma) ** 2 + 4 * sigma * beta)
        return ((gamma - sigma + disc) / (2 * sigma),
                (gamma - sigma - disc) / (2 * sigma), disc)

    ahi, bhi, dhi = projective_roots(beta_hi)
    alo, blo, dlo = projective_roots(beta_lo)
    khi, klo = a / dhi, b / dlo
    width = ahi - alo

    # After z = alo + width*t, the common irrelevant factor is suppressed.
    # State hi has weight t^klo (1-t)^(khi-1); state lo has
    # t^(klo-1) (1-t)^khi.
    entries = []
    for state, (alpha, beta_power) in enumerate(((khi - 1, klo), (khi, klo - 1))):
        x, weights = roots_jacobi(nodes, alpha, beta_power)
        t = (x + 1) / 2
        z = alo + width * t
        if state == 0:
            log_smooth = -(khi + 1) * np.log(z - bhi) - klo * np.log(z - blo)
        else:
            log_smooth = -khi * np.log(z - bhi) - (klo + 1) * np.log(z - blo)
        entries.append((weights, z, log_smooth))
    scale = max(np.max(entry[2]) for entry in entries)
    masses, moments = [], []
    for weights, z, log_smooth in entries:
        w = weights * np.exp(log_smooth - scale)
        masses.append(float(w.sum()))
        moments.append(float(w @ z))
    total = sum(masses)
    mean_z = sum(moments) / total
    normalized = [m / total for m in masses]
    state_mass = normalized if high_first else normalized[::-1]
    return {'lyapunov': float(sigma * mean_z - gamma), 'mean_z': float(mean_z),
            'state_mass': state_mass, 'roots': [float(alo), float(ahi), float(blo), float(bhi)],
            'powers': [float(klo), float(khi)]}


def seir_beta_slow_switching(beta1, beta2, sigma, gamma, rate12_scale=1.0, rate21_scale=None):
    """Slow-switching limit and first coefficient for two-state SEIR.

    The actual switching rates are c*rate12_scale and c*rate21_scale,
    with c tending to zero.  The coefficient is the stationary switch flux
    times the two deterministic projective relaxation areas.
    """
    if rate21_scale is None:
        rate21_scale = rate12_scale
    if min(beta1, beta2, sigma, gamma, rate12_scale, rate21_scale) <= 0:
        raise ValueError("rates and transmission parameters must be positive")

    def projective_roots(beta):
        disc = np.sqrt((gamma - sigma) ** 2 + 4 * sigma * beta)
        return ((gamma - sigma + disc) / (2 * sigma),
                (gamma - sigma - disc) / (2 * sigma))

    if beta1 >= beta2:
        beta_hi, beta_lo = beta1, beta2
        rate_hi_lo, rate_lo_hi = rate12_scale, rate21_scale
    else:
        beta_hi, beta_lo = beta2, beta1
        rate_hi_lo, rate_lo_hi = rate21_scale, rate12_scale
    ahi, bhi = projective_roots(beta_hi)
    alo, blo = projective_roots(beta_lo)
    pi_hi = rate_lo_hi / (rate_hi_lo + rate_lo_hi)
    slow_limit = pi_hi * seir_rate(beta_hi, sigma, gamma) + (1.0 - pi_hi) * seir_rate(beta_lo, sigma, gamma)
    if beta_hi == beta_lo:
        return {'limit': float(slow_limit), 'coefficient': 0.0, 'cross_ratio': 1.0,
                'pi_high': float(pi_hi)}

    cross_ratio = ((ahi - blo) * (alo - bhi)) / ((alo - blo) * (ahi - bhi))
    switch_flux_scale = rate_hi_lo * rate_lo_hi / (rate_hi_lo + rate_lo_hi)
    coefficient = switch_flux_scale * np.log(cross_ratio)
    return {'limit': float(slow_limit), 'coefficient': float(coefficient),
            'cross_ratio': float(cross_ratio), 'pi_high': float(pi_hi)}


def seir_beta_gamma_formulas(Q, beta, gamma, sigma):
    """SEIR with beta and gamma switched: first-order mean and a.s. growth in closed form."""
    pi = stationary(Q)
    bb, gb = pi @ beta, pi @ gamma
    r, x, y, sD = seir_vectors(bb, sigma, gb)
    K = gk(Q, [beta, gamma])                       # K[0,1] = K_beta,gamma (beta earlier, gamma later)
    Kbb, Kbg, Kgb, Kgg = K[0, 0], K[0, 1], K[1, 0], K[1, 1]
    sb, sg = sigma / sD, -(r + sigma) / sD         # d r / d beta, d r / d gamma (projected sensitivities)
    mean = r + ((r + sigma) * Kgg - sigma * Kgb) / sD
    gap = sb ** 2 * Kbb + sg ** 2 * Kgg + sb * sg * (Kbg + Kgb)
    return {'r': r, 'bbar': bb, 'gbar': gb, 'Kbb': Kbb, 'Kbg': Kbg, 'Kgb': Kgb, 'Kgg': Kgg,
            'sb': sb, 'sg': sg, 'mean1': mean, 'as1': mean - gap, 'gap': gap,
            'Kanti_bg': 0.5 * (Kbg - Kgb), 'cycle_pred': (Kbg - Kgb) * sigma / sD}


# ------------------------------------------------------------------ general first-order formulas
def principal(A):
    w, V = np.linalg.eig(A)
    i = np.argmax(w.real)
    wl, U = np.linalg.eig(A.T)
    j = np.argmax(wl.real)
    x, y = np.real(V[:, i]), np.real(U[:, j])
    x = x / x.sum()
    y = y / (y @ x)
    return w[i].real, x, y


def first_order(Q, Abar, Bs, phis):
    """(mean, almost-sure, gap) growth rates to first order for A_y = Abar + sum phi_k(y) B_k."""
    r, x, y = principal(Abar)
    K = gk(Q, phis)
    P = np.eye(len(x)) - np.outer(x, y)
    n = len(Bs)
    mean = r + sum(K[l, k] * y @ Bs[k] @ Bs[l] @ x for k in range(n) for l in range(n))
    asr = r + sum(K[l, k] * y @ Bs[k] @ P @ Bs[l] @ x for k in range(n) for l in range(n))
    s = np.array([y @ B @ x for B in Bs])
    return mean, asr, s @ K @ s


def regime_matrices(Abar, Bs, phis, pi):
    phis = [np.asarray(p, float) - pi @ np.asarray(p, float) for p in phis]
    return [Abar + sum(p[i] * B for p, B in zip(phis, Bs)) for i in range(len(pi))]


# ------------------------------------------------------------------ references
def mean_growth(Q, As):
    """Principal eigenvalue of the combined system Q^T (x) I + blockdiag(A_i)."""
    n, m = len(As), As[0].shape[0]
    G = np.kron(np.asarray(Q, float).T, np.eye(m))
    for i, A in enumerate(As):
        G[i * m:(i + 1) * m, i * m:(i + 1) * m] += A
    return np.max(np.linalg.eigvals(G).real), G


def lyapunov_mc(Q, As, jumps=20000, chains=2000, seed=1, burn=200):
    """Almost-sure growth rate by Monte Carlo of the switched ODE, no time discretization.

    Each holding interval is propagated exactly with exp(tau A_i). The control variate int r(y_s) ds, whose time
    average is lambda(Abar) exactly, is subtracted from log(ybar . x); returns (estimate, standard error)."""
    Q = np.asarray(Q, float)
    pi = stationary(Q)
    Abar = sum(p * A for p, A in zip(pi, As))
    r, xbar, ybar = principal(Abar)
    rho = np.array([ybar @ A @ xbar for A in As])
    rates = -np.diag(Q)
    J = Q / rates[:, None]
    np.fill_diagonal(J, 0.0)
    cumJ = np.cumsum(J, axis=1)
    mus, Vs, Vi = [], [], []
    for A in As:
        w, V = np.linalg.eig(A)
        mus.append(w); Vs.append(V); Vi.append(np.linalg.inv(V))
    mus, Vs, Vi = np.array(mus), np.array(Vs), np.array(Vi)
    rng = np.random.default_rng(seed)
    s = rng.choice(len(pi), size=chains, p=pi)
    x = np.tile(xbar, (chains, 1))
    L, T = np.zeros(chains), np.zeros(chains)
    for step in range(jumps + burn):
        tau = rng.exponential(1.0, chains) / rates[s]
        c = np.einsum('nij,nj->ni', Vi[s], x) * np.exp(mus[s] * tau[:, None])
        xn = np.einsum('nij,nj->ni', Vs[s], c).real
        z = xn @ ybar
        if step >= burn:
            L += np.log(z) - rho[s] * tau
            T += tau
        x = xn / z[:, None]
        u = rng.random(chains)
        s = (u[:, None] > cumJ[s]).sum(axis=1)
    est = r + L / T
    return est.mean(), est.std(ddof=1) / np.sqrt(chains)


# ------------------------------------------------------------------ three-state cycle in closed form
def cycle_generator(a, b):
    """Three-state cycle 1 -> 2 -> 3 -> 1 at rate a, reverse steps at rate b."""
    return np.array([[-(a + b), a, b], [b, -(a + b), a], [a, b, -(a + b)]], float)


def cycle_K(f, g, a, b):
    """Green-Kubo K(f, g) = int Cov(f(y_0), g(y_t)) dt for the cycle: pi is uniform and
    K = [3 (a + b) f~.g~ + (a - b) W(f, g)] / (18 (a^2 + a b + b^2)),  W = sum_j (f_j g_{j+1} - f_{j+1} g_j)."""
    f, g = np.asarray(f, float), np.asarray(g, float)
    ft, gt = f - f.mean(), g - g.mean()
    W = sum(f[j] * g[(j + 1) % 3] - f[(j + 1) % 3] * g[j] for j in range(3))
    return (3 * (a + b) * ft @ gt + (a - b) * W) / (18 * (a * a + a * b + b * b)), W
