"""How much of the immunity factor can age structure explain? Prem et al. (2021) contact matrices, no fitting.

x_i = infected fraction of age group i; dx_i = beta s_i sigma_i sum_j M_ij x_j - gamma x_i with M the contact
matrix (contacts of an i person with j people per day) and sigma_i relative susceptibility.
Next-generation matrix in fractions: (beta/gamma) diag(sigma s) M. Early infections have the age profile v (right
eigenvector of diag(sigma) M) and u is the left eigenvector. Depleting s along v gives
    lam_age = [sum_i u_i v_i^2 / sum_i u_i v_i] / sum_i n_i v_i       (n_i population shares, v in fractions)
Check: simulate the age SIR to the peak of prevalence and compare the overall attack with 1 - R^(-1/lam_age)."""
import numpy as np, pandas as pd, os
from scipy.integrate import solve_ivp
HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "..", "data", "contacts")
wb = pd.read_csv(os.path.join(C, "worldbank_pop_age_2020.csv"), dtype={"group": str})
G = ["0004","0509","1014","1519","2024","2529","3034","3539","4044","4549","5054","5559","6064","6569","7074"]


def pop16(c):
    w = wb[wb.country == c].groupby("group")["pop"].sum()
    return np.array([w[g] for g in G] + [w["7579"] + w["80UP"]], float)


def lam_age(M, n, sigma):
    A = np.diag(sigma) @ M
    w, V = np.linalg.eig(A); k = np.argmax(w.real); v = np.abs(V[:, k].real)
    wl, U = np.linalg.eig(A.T); u = np.abs(U[:, np.argmax(wl.real)].real)
    return (u @ (v ** 2) / (u @ v)) / (n @ v), w[k].real, v


def peak_attack(M, n, sigma, R, gamma=0.2):
    A = np.diag(sigma) @ M; rho = np.max(np.linalg.eigvals(A).real); beta = R * gamma / rho
    def f(t, y):
        s, x = y[:16], y[16:]
        inf = beta * s * sigma * (M @ x)
        return np.r_[-inf, inf - gamma * x]
    y0 = np.r_[np.ones(16) - 1e-7, np.full(16, 1e-7)]
    sol = solve_ivp(f, (0, 2000), y0, rtol=1e-9, atol=1e-13, dense_output=True, max_step=1)
    t = np.linspace(0, 2000, 40001); Y = sol.sol(t)
    prev = n @ Y[16:]; k = int(np.argmax(prev))
    return float(n @ (1 - Y[:16, k])), 1 - Y[:16, k]


if __name__ == "__main__":
    for c in ["ESP", "SWE", "USA", "GBR", "CHE", "BRA"]:
        M = pd.read_csv(os.path.join(C, f"prem2021_all_{c}.csv")).values.astype(float)
        N = pop16(c); n = N / N.sum()
        for label, sigma in (("equal susceptibility", np.ones(16)), ("under-20s at half", np.r_[np.full(4, 0.5), np.ones(12)])):
            lam, rho, v = lam_age(M, n, sigma)
            line = f"{c} {label:22s} lam_age {lam:.2f}"
            for R in (1.3, 2.5):
                a, _ = peak_attack(M, n, sigma, R)
                line += f" | R {R}: attack at peak {a:.3f}, rule {1 - R ** (-1 / lam):.3f}, textbook {1 - 1 / R:.3f}"
            print(line, flush=True)
