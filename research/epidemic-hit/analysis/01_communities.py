"""Many communities, each an SIR epidemic whose transmission rate switches between two levels on its own Markov clock.

Question: fit a homogeneous SIR to the early growth of the national total. Does the fitted R overstate the R implied
by the final attack rate, and by how much?

Per community: dS = -beta_X S I, dI = (beta_X S - gamma) I, regime X in {hi, lo} switching at rate lam each way,
beta = beta_bar +/- delta. Communities are independent, seeded equally. Euler-exact regime path on a fine grid.
Reported:
- r_typ = beta_bar - gamma (typical growth of one community, exact for SIR);
- r_mean = beta_bar - gamma - lam + sqrt(lam^2 + delta^2) (top eigenvalue of Q + diag(beta_i - gamma), exact);
- r_obs: growth of the national total over the early window (national attack < 1%);
- R_growth = 1 + r_obs / gamma; z(R) the homogeneous final size; z_obs the national final attack rate.
"""
import numpy as np, json, os, sys

rng = np.random.default_rng(3)


def z_final(R):
    z = 0.9
    for _ in range(200):
        z = 1 - np.exp(-R * z)
    return z if R > 1 else 0.0


def run(beta_bar, delta, gamma, lam, M=4000, i0=1e-6, dt=0.05, T=600):
    n = int(T / dt)
    X = rng.integers(0, 2, M)            # stationary start
    S = np.ones(M) - i0; I = np.full(M, i0)
    tot, att = [], []
    for k in range(n):
        flip = rng.random(M) < lam * dt
        X = np.where(flip, 1 - X, X)
        b = beta_bar + np.where(X == 1, delta, -delta)
        inf = b * S * I * dt
        S = S - inf; I = I + inf - gamma * I * dt
        if k % 20 == 0:
            tot.append(I.mean()); att.append(1 - S.mean())
    t = np.arange(len(tot)) * 20 * dt
    tot, att = np.array(tot), np.array(att)
    w = (att > 20 * i0) & (att < 0.01)
    r_obs = np.polyfit(t[w], np.log(tot[w]), 1)[0] if w.sum() > 5 else np.nan
    kp = int(np.argmax(tot))
    return r_obs, att[-1], att[kp], t, tot, att


if __name__ == "__main__":
    gamma = 0.2
    rows = []
    for beta_bar, delta, lam in [(0.3, 0.0, 0.1), (0.3, 0.1, 0.1), (0.3, 0.2, 0.1), (0.3, 0.2, 0.3),
                                 (0.3, 0.2, 0.03), (0.3, 0.25, 0.05), (0.26, 0.2, 0.05)]:
        r_typ = beta_bar - gamma
        r_mean = beta_bar - gamma - lam + np.sqrt(lam ** 2 + delta ** 2)
        r_obs, z_obs, a_peak, *_ = run(beta_bar, delta, gamma, lam)
        Rg = 1 + r_obs / gamma
        row = dict(beta_bar=beta_bar, delta=delta, lam=lam, R_bar=beta_bar / gamma, r_typ=r_typ, r_mean=r_mean,
                   r_obs=r_obs, R_growth=Rg, z_of_R_growth=z_final(Rg), z_of_R_bar=z_final(beta_bar / gamma), z_obs=z_obs,
                   hit_growth=1 - 1 / Rg, attack_at_peak=a_peak, K_first=delta ** 2 / (2 * lam))
        rows.append(row)
        print(" ".join(f"{k}={v:.3f}" for k, v in row.items()), flush=True)
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "out", "01_communities.json"), "w"), indent=1)
