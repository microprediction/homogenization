"""Certificate for the first-order effective generator.

1. Random non-commuting operators and a strongly irreversible chain: the rule's error falls like |Q|^-2, while
   the averaged generator and the symmetric-only rule do not reach second order.
2. Vasicek with switching reversion speed and volatility (non-commuting): same, on the discretized pricing equation.
3. Black-Scholes with switching variance: call prices from the rule against the numerical solution.
4. Many-name credit: joint survival of every subset from one Green-Kubo matrix of hazards.
"""
import math, cmath, os, sys
import numpy as np
from scipy.linalg import expm
from effective_generator import stationary, gk, averages, effective_generator, full_generator
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fast-switching'))

Q3 = np.array([[-3, 2, 1], [1, -2, 1], [0.5, 1.5, -2]], float)          # irreversible
QC = np.array([[-2.1, 2, 0.1], [0.1, -2.1, 2], [2, 0.1, -2.1]], float)   # strongly one-way cycle 1 -> 2 -> 3 -> 1


def first_order(L, D, T, f):
    """exp(T L) f plus the Duhamel term int_0^T exp((T-s) L) D exp(s L) f ds, via one block exponential."""
    n = L.shape[0]
    M = np.zeros((2 * n, 2 * n), dtype=np.result_type(L, D))
    M[:n, :n], M[n:, n:], M[:n, n:] = L, L, D
    E = expm(T * M)
    return E[:n, :n] @ f + E[:n, n:] @ f
SCALES = [10, 20, 40, 80]


def _N(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def bs_price(S, K, T, r, V):
    """Black-Scholes call with total variance V."""
    d1 = (math.log(S / K) + r * T + V / 2) / math.sqrt(V)
    return S * _N(d1) - K * math.exp(-r * T) * _N(d1 - math.sqrt(V))


def bs_volga(S, K, T, r, V):
    """Second derivative of the call in total variance V."""
    d1 = (math.log(S / K) + r * T + V / 2) / math.sqrt(V)
    d2 = d1 - math.sqrt(V)
    return S * math.exp(-d1 * d1 / 2) / math.sqrt(2 * math.pi) * (d1 * d2 - 1) / (4 * V ** 1.5)


def slope(errs):
    return math.log(errs[-1] / errs[-2]) / math.log(SCALES[-1] / SCALES[-2])


def compare(Q0, Lbar_of, As, phis, f, T, readout):
    """Errors of averaged / symmetric-only / full rule against the coupled system, over the scales."""
    rows = {'averaged': [], 'symmetric part only': [], 'rule': []}
    for m in SCALES:
        Q = m * Q0
        pi = stationary(Q)
        Lbar = Lbar_of(averages(Q, phis))
        n, mm = Q.shape[0], Lbar.shape[0]
        truth = readout(np.kron(pi, np.eye(mm)) @ (expm(T * full_generator(Q, Lbar, As, phis)) @ np.kron(np.ones(n), f)))
        K = gk(Q, phis)
        rows['averaged'].append(abs(readout(expm(T * Lbar) @ f) - truth))
        rows['symmetric part only'].append(abs(readout(first_order(Lbar, effective_generator(0 * Lbar, As, K, 'sym'), T, f)) - truth))
        rows['rule'].append(abs(readout(first_order(Lbar, effective_generator(0 * Lbar, As, K), T, f)) - truth))
    return rows


def report(rows, want):
    ok = True
    for name, errs in rows.items():
        s = slope(errs)
        ok &= (s > -1.5) if want[name] == -1 else abs(s - want[name]) < 0.15   # first order: clearly not second
        print(f"   {name:22s} " + "  ".join(f"{e:.3e}" for e in errs) + f"   slope {s:+.2f}")
    return ok


def main():
    ok = True
    rng = np.random.default_rng(3)

    print("1. random operators, strongly irreversible chain")
    mdim = 6
    As = [rng.normal(size=(mdim, mdim)) * 0.3 for _ in range(3)]
    L0 = rng.normal(size=(mdim, mdim)) * 0.3 - np.eye(mdim)
    phis = [rng.normal(size=3) for _ in range(3)]
    f = rng.normal(size=mdim)
    Lbar_of = lambda avg: L0 + sum(a * A for a, A in zip(avg, As))
    ok &= report(compare(QC, Lbar_of, As, phis, f, 1.0, lambda v: v[0]), {'averaged': -1, 'symmetric part only': -1, 'rule': -2})

    print("2. Vasicek with switching speed and volatility, discretized, bond price at x = 4%, T = 2")
    x = np.linspace(-0.35, 0.45, 161)
    h = x[1] - x[0]
    D1 = (np.diag(np.ones(160), 1) - np.diag(np.ones(160), -1)) / (2 * h)
    D2 = (np.diag(np.ones(160), 1) - 2 * np.eye(161) + np.diag(np.ones(160), -1)) / h ** 2
    theta = 0.04
    Ak = np.diag(theta - x) @ D1          # multiplies the speed kappa
    As_ = 0.5 * D2                        # multiplies the variance sigma^2
    kap, s2 = [0.3, 1.5, 3.0], [0.0004, 0.0036, 0.0016]
    i0 = int(np.argmin(abs(x - 0.04)))
    Lbar_of = lambda avg: avg[0] * Ak + avg[1] * As_ - np.diag(x)
    smooth = np.exp(-1.5 * x + 3 * x ** 2)
    inner = slice(10, -10)
    cerr = np.abs(((Ak @ As_ - As_ @ Ak) @ smooth - D2 @ smooth)[inner]).max() / np.abs((D2 @ smooth)[inner]).max()
    print(f"   commutator [A_kappa, A_sigma2] acts as d_xx on a smooth function: relative error {cerr:.1e}")
    ok &= cerr < 1e-2
    ok &= report(compare(QC, Lbar_of, [Ak, As_], [kap, s2], np.ones(161), 2.0, lambda v: v[i0]),
                 {'averaged': -1, 'symmetric part only': -1, 'rule': -2})

    print("3. Black-Scholes with switching variance: calls, S = 100, T = 1, r = 2%, strikes 70 to 130")
    from options import bs_call
    sig, r, T, S = [0.1, 0.3, 0.2], 0.02, 1.0, 100.0
    errs = []
    for m in [2, 5, 20]:
        Q = m * Q3
        pi = stationary(Q)
        V = pi @ np.square(sig) * T
        Kvv = gk(Q, [np.square(sig)])[0, 0]
        worst, worst_avg = 0, 0
        for Kx in [70, 80, 90, 100, 110, 120, 130]:
            num = sum(pi[i] * bs_call(S, Kx, T, r, sig, Q, i) for i in range(3))
            rule = bs_price(S, Kx, T, r, V) + T * Kvv * bs_volga(S, Kx, T, r, V)
            worst, worst_avg = max(worst, abs(rule - num)), max(worst_avg, abs(bs_price(S, Kx, T, r, V) - num))
        errs.append(worst)
        print(f"   generator x{m:3d}: largest error, averaged variance {worst_avg:.2e}; rule {worst:.2e}")
    ok &= errs[2] < errs[1] / 10

    print("4. four names, three-state chain: log joint survival of every subset, T = 5")
    H = np.array([[0.01, 0.03, 0.08], [0.02, 0.02, 0.06], [0.05, 0.01, 0.01], [0.03, 0.04, 0.02]])
    import itertools
    for m in [10, 40]:
        Q = m * Q3
        pi = stationary(Q)
        K = gk(Q, list(H))
        Ks = 0.5 * (K + K.T)
        worst = 0
        for r_ in range(1, 5):
            for S in itertools.combinations(range(4), r_):
                g = -H[list(S)].sum(0)
                truth = math.log(pi @ expm(5.0 * (Q + np.diag(g))) @ np.ones(3))
                one = np.zeros(4); one[list(S)] = 1
                rule = 5.0 * (-(one @ H @ pi) + one @ Ks @ one)
                worst = max(worst, abs(rule - truth))
        print(f"   generator x{m}: largest error over 15 subsets {worst:.2e}")
        if m == 40:
            ok &= worst < 1e-4
    w = np.linalg.eigvalsh(Ks)
    print(f"   eigenvalues of the dependence matrix {np.array2string(w, precision=2)}: rank at most 2 for three states")
    ok &= abs(w).min() < 1e-12 * abs(w).max() and w.min() > -1e-12 * abs(w).max()
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
