"""SIR with individual social activity switching between two levels: closed forms and a check.

Activity a in {a1, a2}, stationary weights p, mean 1, variance s2, third central moment c * s2 (two-point law:
delta^2 = c delta + s2). Activity relaxes at rate kappa (Markov, both directions). Force of infection on an
individual of activity a: beta * a * Phi, Phi = sum_k a_k I_k (proportionate mixing, infectivity = activity).
Susceptibles and infecteds both keep switching.

Closed forms (derived in FINDINGS.md):
  growth rate r:        1 = beta * (1/(r+gamma) + s2/(r+gamma+kappa))
  basic R0:             R0 = beta * (1/gamma + s2/(gamma+kappa))
  immunity factor, growth phase (R_e ~ R0 (1 - lam (1-S))), rho = r/(r+kappa):
      lam = ((1 + s2 rho)/gamma + s2 (1 + (1+c) rho)/(gamma+kappa)) / (1/gamma + s2/(gamma+kappa))
  kappa -> 0: lam = <a^3>/<a^2> (persistent heterogeneity);  kappa -> inf: lam = 1.
  Long-run herd-immunity threshold (susceptibles remixed): 1 - 1/R0.
"""
import numpy as np, json, os
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

HERE = os.path.dirname(os.path.abspath(__file__))


def two_point(s2, c):
    d = np.roots([1, -c, -s2])          # delta^2 - c delta - s2 = 0
    d1, d2 = max(d), min(d)
    p1 = -d2 / (d1 - d2)
    return np.array([1 + d1, 1 + d2]), np.array([p1, 1 - p1])


def growth(beta, gamma, kappa, s2):
    return brentq(lambda r: beta * (1 / (r + gamma) + s2 / (r + gamma + kappa)) - 1, -gamma + 1e-9, 50)


def R0(beta, gamma, kappa, s2):
    return beta * (1 / gamma + s2 / (gamma + kappa))


def lam(r, gamma, kappa, s2, c):
    rho = r / (r + kappa) if kappa > 0 else 1.0
    num = (1 + s2 * rho) / gamma + s2 * (1 + (1 + c) * rho) / (gamma + kappa)
    return num / (1 / gamma + s2 / (gamma + kappa))


def generator(p, kappa):
    # two-state chain with stationary law p and relaxation rate kappa: rates k*p2 (1->2), k*p1 (2->1)
    return kappa * np.array([[-p[1], p[1]], [p[0], -p[0]]])


def simulate(beta, gamma, kappa, s2, c, i0=1e-7, T=400):
    a, p = two_point(s2, c); G = generator(p, kappa)
    def rhs(t, y):
        S, I = y[:2], y[2:]
        Phi = a @ I
        inf = beta * a * S * Phi
        return np.r_[-inf + G.T @ S, inf - gamma * I + G.T @ I]
    y0 = np.r_[p * (1 - i0), p * i0 * a / (a @ p)]
    sol = solve_ivp(rhs, (0, T), y0, rtol=1e-10, atol=1e-14, dense_output=True, max_step=0.5)
    return sol, a, p, G


def Re_along(sol, a, p, G, beta, gamma, ts):
    """Instantaneous next-generation number: beta * a^T (gamma - G^T)^{-1} (a * S(t))."""
    M = np.linalg.inv(gamma * np.eye(2) - G.T)
    return np.array([beta * a @ M @ (a * sol.sol(t)[:2]) for t in ts])


if __name__ == "__main__":
    gamma = 0.2
    print("check against the ODE (numerical solution)")
    for s2, c, kappa in [(1.0, 2.0, 0.0), (1.0, 2.0, 0.05), (1.0, 2.0, 0.3), (2.0, 2.0, 0.1)]:
        beta = 0.45
        r = growth(beta, gamma, kappa, s2)
        sol, a, p, G = simulate(beta, gamma, kappa, s2, c)
        ts = np.linspace(0, 400, 40001); y = sol.sol(ts)
        att = 1 - y[:2].sum(0); I = y[2:].sum(0)
        w = (att > 1e-6) & (att < 1e-3)
        r_ode = np.polyfit(ts[w], np.log(I[w]), 1)[0]
        w2 = (att > 1e-5) & (att < 0.003)
        Re = Re_along(sol, a, p, G, beta, gamma, ts[w2])
        slope = np.polyfit(att[w2], Re, 1)[0]
        print(f"s2={s2} c={c} kappa={kappa}: r formula {r:.5f} ode {r_ode:.5f} | lam formula "
              f"{lam(r, gamma, kappa, s2, c):.4f} ode {-slope / R0(beta, gamma, kappa, s2):.4f}")
