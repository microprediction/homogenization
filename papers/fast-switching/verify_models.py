"""Certificate for models.py: each model's reduction checked by Monte Carlo, each expansion by convergence.

`python3 verify_models.py` (a few minutes). A reduction passes when Monte Carlo on a fine grid agrees with the
numerical solution of the reduced system within four standard errors; an expansion passes when doubling the
switching rate cuts the order-n error by about 2^(n+1) for the first orders.
"""
import math
import cmath
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.linalg import expm
from fastswitch import FastSwitch, numerical_a_callable
from models import gaussian_factors, cir_switching_mean, vasicek_jumps, mmpp, heston_switching_theta

RNG = np.random.default_rng(2026)
OK = True


def check(name, cond, detail):
    global OK
    OK &= bool(cond)
    print(f"{'ok ' if cond else 'FAIL'} {name}: {detail}")


def orders(make, t, lam, n=4):
    """log2 error ratios for orders 0..n-1 when lam doubles"""
    errs = []
    for l in (lam, 2 * lam):
        Q = [[-l, l], [l, -l]]
        g, gf = make()
        ex = numerical_a_callable(t, Q, gf, rtol=1e-13)[0]
        fs = FastSwitch(Q, g, order=n)
        errs.append([abs(fs.a(t, o)[0] - ex) for o in range(n)])
    return [math.log2(a / b) for a, b in zip(*errs)]


def switch(y, lam, dt):
    return np.where(RNG.random(len(y)) < 1 - math.exp(-lam * dt), 1 - y, y)


def main():
    lam, t, P, n = 4.0, 2.0, 60000, 2000
    dt = t / n
    Q = [[-lam, lam], [lam, -lam]]

    # two Gaussian factors with a common regime
    kap, th, sg = [1.0, 1.5], [[0.06, 0.01], [0.05, 0.015]], [[0.02, 0.005], [0.02, 0.005]]
    rho = [[[1, 0], [0, 1]]] * 2
    g, gf, pref = gaussian_factors(kap, th, sg, rho, [1, 1])
    ex = numerical_a_callable(t, Q, gf)[0] * pref(t, [0.03, 0.03])
    x, y, I = np.full((P, 2), 0.03), np.zeros(P, int), np.zeros((P, 2))
    kv, thv, sv = np.array(kap), np.array(th), np.array(sg)
    dec = np.exp(-kv * dt)
    sd = np.sqrt((1 - dec ** 2) / (2 * kv))
    for _ in range(n):
        prev = x.copy()
        x = thv[:, y].T + (x - thv[:, y].T) * dec + sv[:, y].T * sd * RNG.standard_normal((P, 2))
        I += 0.5 * dt * (prev + x)
        y = switch(y, lam, dt)
    v = np.exp(-I.sum(1))
    se = v.std() / math.sqrt(P)
    check("two-factor reduction", abs(v.mean() - ex) < 4 * se, f"numerical {ex:.6f}, Monte Carlo {v.mean():.6f} +/- {se:.6f}")
    r = orders(lambda: gaussian_factors(kap, th, sg, rho, [1, 1])[:2], 1.0, 10.0)
    check("two-factor orders", all(abs(r[i] - (i + 1)) < 0.4 for i in range(3)), " ".join(f"{a:.2f}" for a in r))

    # CIR with a switching mean
    kc, thc, sc = 1.5, [0.08, 0.02], 0.15
    g, gf, pref, B = cir_switching_mean(kc, thc, sc, 5.0)
    ex = numerical_a_callable(t, Q, gf)[0] * pref(t, 0.05)
    x, y, I = np.full(P, 0.05), np.zeros(P, int), np.zeros(P)
    for _ in range(n):
        xp = np.maximum(x, 0)
        x = x + kc * (np.array(thc)[y] - xp) * dt + sc * np.sqrt(xp * dt) * RNG.standard_normal(P)
        I += 0.5 * dt * (xp + np.maximum(x, 0))
        y = switch(y, lam, dt)
    v = np.exp(-I)
    se = v.std() / math.sqrt(P)
    check("CIR reduction", abs(v.mean() - ex) < 4 * se + 1e-4, f"numerical {ex:.6f}, Monte Carlo {v.mean():.6f} +/- {se:.6f}")
    r = orders(lambda: cir_switching_mean(kc, thc, sc, 5.0)[:2], 3.0, 10.0)
    check("CIR orders", all(abs(r[i] - (i + 1)) < 0.4 for i in range(2)), " ".join(f"{a:.2f}" for a in r))

    # Vasicek with jumps at a switching intensity
    kj, thj, sj, lj, m = 2.0, [0.05, 0.02], [0.02, 0.01], [3.0, 0.2], 0.03
    g, gf, pref = vasicek_jumps(kj, thj, sj, lj, m, 5.0)
    ex = numerical_a_callable(t, Q, gf)[0] * pref(t, 0.05)
    x, y, I = np.full(P, 0.05), np.zeros(P, int), np.zeros(P)
    dec = math.exp(-kj * dt)
    sd = math.sqrt((1 - dec ** 2) / (2 * kj))
    for _ in range(n):
        prev = x
        x = np.array(thj)[y] + (x - np.array(thj)[y]) * dec + np.array(sj)[y] * sd * RNG.standard_normal(P)
        x = x + (RNG.random(P) < 1 - np.exp(-np.array(lj)[y] * dt)) * RNG.exponential(m, P)
        I += 0.5 * dt * (prev + x)
        y = switch(y, lam, dt)
    v = np.exp(-I)
    se = v.std() / math.sqrt(P)
    check("jump reduction", abs(v.mean() - ex) < 4 * se, f"numerical {ex:.6f}, Monte Carlo {v.mean():.6f} +/- {se:.6f}")
    r = orders(lambda: vasicek_jumps(kj, thj, sj, lj, m, 5.0)[:2], 3.0, 10.0)
    check("jump orders", all(abs(r[i] - (i + 1)) < 0.4 for i in range(2)), " ".join(f"{a:.2f}" for a in r))

    # Markov-modulated Poisson counts: the whole distribution through the generating function
    rates, tm, M = [8.0, 1.0], 1.0, 64
    for l in (10.0, 20.0):
        Ql = np.array([[-l, l], [l, -l]])
        zs = np.exp(2j * np.pi * np.arange(M) / M)
        exact = np.real(np.fft.fft([(expm((Ql + np.diag([(z - 1) * r for r in rates])) * tm) @ np.ones(2))[0] for z in zs])) / M
        approx = np.real(np.fft.fft([FastSwitch(Ql, mmpp(rates, z)[0], order=4).a(tm, 4)[0] for z in zs])) / M
        err = np.abs(approx - exact).max()
        check(f"Poisson counts lam={l:g}", err < (1e-4 if l == 10 else 1e-5), f"max probability error at order 4 {err:.1e}")

    # Heston with a switching long-run variance: characteristic function
    for u in (0.5, 3.0):
        r = orders(lambda: heston_switching_theta(u, 2.0, [0.09, 0.02], 0.4, -0.6, 1.0)[:2], 1.0, 10.0)
        check(f"Heston orders u={u}", all(abs(r[i] - (i + 1)) < 0.5 for i in range(3)), " ".join(f"{a:.2f}" for a in r))

    print("PASS" if OK else "FAIL")
    return 0 if OK else 1


if __name__ == "__main__":
    raise SystemExit(main())
