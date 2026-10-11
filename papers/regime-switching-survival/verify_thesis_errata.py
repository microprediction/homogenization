"""Certificate for the errata page of the 2001 thesis (page: thesis-errata.html).

1. Switching rate. The generator printed on p. 85 has off-diagonal rate lambda/2, so the regime contrast decays like
   exp(-lambda s); with rate lambda out of each state, as on p. 125 and on this site, it decays like exp(-2 lambda s).
   The exact antisymmetric corrections of the two examples agree with the thesis simulations only for the second.
2. Dynamic mean Vasicek, Eq. (3.7). The conditional mean of the level at calendar time s is weighted by b(T - s),
   the remaining maturity. Eq. (3.7) as printed pairs exp(-lambda s) with b(s) and reproduces the thesis's
   dynamic-mean entries; the corrected kernel is close to the exact two-state solution and to the simulations.
3. Appendix B.4, deterministic shift, checked with a deterministic rate: the caplet and floorlet need the discount
   A(t, T) from today to the fixing day, the second leg of the swaplet needs A(t, tau + dtau), and the futures
   recursion under the risk-neutral measure needs exp(-int r).
Writes thesis_errata.json for the page. Ends with PASS or FAIL.
`python3 verify_thesis_errata.py`
"""
import json, math, os
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm

HERE = os.path.dirname(os.path.abspath(__file__))
TH, SIG = (0.35, 0.15), 0.0001
EXAMPLES = [dict(pages="125-126", lam=5.0, kappa=2.0, T=15.0, thesis_dmv=0.0057143, sim=0.010020, se=0.0003),
            dict(pages="127-128", lam=1.0, kappa=25.0, T=5.0, thesis_dmv=0.0956, sim=0.0516, se=0.0004)]


def antisymmetric(rate, kappa, T):
    """|P_1 - P_2| / (2 P_mean) from the two-state linear system, switching at `rate` out of each state."""
    b = lambda r: (1 - math.exp(-kappa * r)) / kappa
    th, Q = np.array(TH), rate * np.array([[-1.0, 1.0], [1.0, -1.0]])
    f = lambda r, a: (-kappa * th * b(r) + 0.5 * SIG ** 2 * b(r) ** 2) * a + Q @ a
    a = solve_ivp(f, (0, T), [1.0, 1.0], method='DOP853', rtol=1e-13, atol=1e-16).y[:, -1]
    mean = math.exp(quad(lambda r: -kappa * np.mean(th) * b(r) + 0.5 * SIG ** 2 * b(r) ** 2, 0, T, epsabs=1e-14)[0])
    return abs(a[0] - a[1]) / (2 * mean)


def dynamic_mean(decay, kappa, T, remaining):
    """sinh(kappa theta~ int_0^T exp(-decay s) b ds), with b at the remaining maturity T - s or, as printed, at s."""
    b = lambda r: (1 - math.exp(-kappa * r)) / kappa
    I = quad(lambda s: math.exp(-decay * s) * b(T - s if remaining else s), 0, T, epsabs=1e-15, epsrel=1e-13)[0]
    return math.sinh(kappa * 0.5 * (TH[0] - TH[1]) * I)


def main():
    ok, out = True, {}
    print("1. decay of the regime contrast E[theta(y_s) | y_0] - mean, at s = 0.3, lambda = 5")
    lam, s = 5.0, 0.3
    for name, Q, want in [("p. 85 generator, off-diagonal lambda/2", -lam * np.array([[0.5, -0.5], [-0.5, 0.5]]), math.exp(-lam * s)),
                          ("rate lambda out of each state", lam * np.array([[-1.0, 1.0], [1.0, -1.0]]), math.exp(-2 * lam * s))]:
        c = (expm(s * Q) @ np.array([1.0, -1.0]))[0]
        print(f"   {name}: {c:.12f} against {want:.12f}")
        ok &= abs(c - want) < 1e-13

    print("2. antisymmetric correction: Eq. (3.7) as printed, corrected kernel, exact, thesis simulation")
    rows = []
    for e in EXAMPLES:
        printed = dynamic_mean(e['lam'], e['kappa'], e['T'], remaining=False)
        fixed = dynamic_mean(2 * e['lam'], e['kappa'], e['T'], remaining=True)
        exact, half = antisymmetric(e['lam'], e['kappa'], e['T']), antisymmetric(e['lam'] / 2, e['kappa'], e['T'])
        rows.append(dict(pages=e['pages'], lam=e['lam'], kappa=e['kappa'], T=e['T'], printed=printed, corrected=fixed,
                         exact=exact, exact_half_rate=half, simulation=e['sim'], se=e['se']))
        print(f"   pp. {e['pages']}: printed {printed:.7f} (thesis table {e['thesis_dmv']})  corrected {fixed:.7f}"
              f"  exact {exact:.7f}  exact at rate lambda/2 {half:.7f}  simulation {e['sim']} +- {e['se']}")
        ok &= abs(printed - e['thesis_dmv']) < 0.6 * 10 ** -len(str(e['thesis_dmv']).split('.')[1])   # as printed
        ok &= abs(exact - e['sim']) < 2 * e['se'] and abs(half - e['sim']) > 10 * e['se']
        ok &= abs(fixed - exact) < abs(printed - exact) / 20
    out['dynamic_mean'] = rows

    print("3. Appendix B.4 with a deterministic rate 5%: t = 0, T = tau = 1, dtau = 1")
    A = lambda t, u: math.exp(-0.05 * (u - t))
    for K, name in [(0.0, "caplet"), (0.10, "floorlet")]:
        Kt = 1 / (1 + K) / A(1, 2)                                   # shifted strike; the unshifted bond is 1
        option = max(Kt - 1, 0) if name == "caplet" else max(1 - Kt, 0)
        printed, fixed = A(1, 1) / Kt * option, A(0, 1) * A(1, 1) / Kt * option
        direct = abs(A(0, 1) - (1 + K) * A(0, 2))                   # the payoff is deterministic and in the money
        print(f"   {name}, K = {K}: printed {printed:.10f}  with A(t, T) {fixed:.10f}  direct {direct:.10f}")
        ok &= abs(fixed - direct) < 1e-15 and abs(printed / fixed - 1 / A(0, 1)) < 1e-14
        out[name] = dict(printed=printed, corrected=fixed)
    swap_printed, swap_fixed = A(0, 1) - A(0, 1), A(0, 1) - A(0, 2)
    print(f"   swaplet, K = 0: printed {swap_printed:.10f}  with A(t, tau + dtau) on the second leg {swap_fixed:.10f}")
    ok &= swap_printed == 0 and abs(swap_fixed - out['caplet']['corrected']) < 1e-15
    out['swaplet'] = dict(printed=swap_printed, corrected=swap_fixed)
    dt = 1 / 252
    fut_printed, fut_fixed = math.exp(0.05 * dt) / A(0, dt), math.exp(-0.05 * dt) / A(0, dt)
    print(f"   futures recursion applied to f_i = 1 over one day: printed {fut_printed:.10f}  with exp(-int r) {fut_fixed:.10f}")
    ok &= abs(fut_fixed - 1) < 1e-15 and abs(fut_printed - math.exp(0.1 * dt)) < 1e-15
    json.dump(out, open(os.path.join(HERE, 'thesis_errata.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
