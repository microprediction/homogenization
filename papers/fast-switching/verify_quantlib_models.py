"""Certificate for the QuantLib models added to the engine: Merton76, variance gamma and Bates with switched parameters.

1. Expansion order for the supplied g: the expansion's error against the numerical solution of the same reduced system
   falls one order per order. Orders whose error has reached rounding (below 1e-13) are not rated. This checks the
   recursion, not that g is the right model.
2. The exponents themselves, independently of the recursion: each regime's exponent vanishes at u = 0 and at u = -i
   (the stock is a martingale at zero rates; for Bates the Heston coefficient D(t, -i) vanishes too), and
   g(-u) = conj(g(u)) for real u.
3. With the regimes frozen (both regimes equal) the characteristic function is exp(int g), and its Lewis call price
   matches an independent price: Merton's series of Black-Scholes prices, and QuantLib's VarianceGammaEngine and
   BatesEngine (skipped when QuantLib is not installed: pip install QuantLib).
`python3 verify_quantlib_models.py`
"""
import math
import cmath
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm
from fastswitch import FastSwitch, numerical_a_callable
from quantlib_models import merton76, variance_gamma, bates

Q0 = np.array([[-1.0, 1.0], [1.0, -1.0]])
MJ, SJ = -0.1, 0.15                                  # log-jump mean and standard deviation
S0, T = 100.0, 1.0


def rates(make, t, u_list, lams=(20.0, 40.0), orders=range(0, 5), floor=1e-13):
    out = []
    for u in u_list:
        errs = {}
        for lam in lams:
            g, gf = make(u)[:2]
            ref = numerical_a_callable(t, lam * Q0, gf, rtol=1e-13)[0]
            fs = FastSwitch(lam * Q0, g, order=max(orders))
            errs[lam] = [abs(fs.a(t, o)[0] - ref) for o in orders]
        out.append([math.log2(a / b) for a, b in zip(errs[lams[0]], errs[lams[1]]) if b > floor])
    return out


def lewis_call(phi, K):
    """Call price at zero rates from the characteristic function phi of log(S_T / S_0) (Lewis 2001)."""
    k = math.log(S0 / K)
    f = lambda u: (cmath.exp(1j * u * k) * phi(u - 0.5j)).real / (u * u + 0.25)
    return S0 - math.sqrt(S0 * K) / math.pi * quad(f, 0, math.inf, epsabs=1e-13, epsrel=1e-13, limit=1000)[0]


def merton_series_call(K, sigma, lam):
    """Merton (1976): a Poisson mixture of Black-Scholes prices, zero rates."""
    kb = math.exp(MJ + 0.5 * SJ ** 2) - 1
    lp = lam * (1 + kb)

    def bs(r, v):
        d1 = (math.log(S0 / K) + (r + 0.5 * v) * T) / math.sqrt(v * T)
        return S0 * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d1 - math.sqrt(v * T))
    return sum(math.exp(-lp * T) * (lp * T) ** n / math.factorial(n) *
               bs(-lam * kb + n * math.log(1 + kb) / T, sigma ** 2 + n * SJ ** 2 / T) for n in range(80))


def quantlib_calls(kind, Ks, **p):
    import QuantLib as ql
    today = ql.Date(2, 1, 2026)
    ql.Settings.instance().evaluationDate = today
    dc = ql.Actual365Fixed()
    flat = ql.YieldTermStructureHandle(ql.FlatForward(today, 0.0, dc))
    spot = ql.QuoteHandle(ql.SimpleQuote(S0))
    if kind == 'vg':
        engine = ql.VarianceGammaEngine(ql.VarianceGammaProcess(spot, flat, flat, p['sigma'], p['nu'], p['theta']))
    else:
        proc = ql.BatesProcess(flat, flat, spot, p['v0'], p['kappa'], p['theta'], p['xi'], p['rho'], p['lam'], MJ, SJ)
        engine = ql.BatesEngine(ql.BatesModel(proc), 192)
    out = []
    for K in Ks:
        opt = ql.VanillaOption(ql.PlainVanillaPayoff(ql.Option.Call, K), ql.EuropeanExercise(today + 365))
        opt.setPricingEngine(engine)
        out.append(opt.NPV())
    return out


def main():
    ok = True
    models = [
        ("Merton76: volatility and jump intensity", lambda u: merton76(u, [0.1, 0.3], [0.5, 3.0], MJ, SJ)),
        ("variance gamma: sigma, nu, theta", lambda u: variance_gamma(u, [0.12, 0.3], [0.2, 0.5], [-0.05, -0.2])),
        ("Bates: variance level and jump intensity", lambda u: bates(u, 2.0, [0.09, 0.02], 0.4, -0.6, [0.5, 3.0], MJ, SJ, 1.0)),
    ]
    print("1. expansion order for the supplied g")
    for name, make in models:
        rr = rates(make, 1.0, [0.5, 2.0], lams=(4.0, 8.0))
        for u, r in zip([0.5, 2.0], rr):
            good = all(abs(x - (k + 1)) < 0.35 for k, x in enumerate(r))
            ok &= good
            print(f"{'ok ' if good else 'BAD'} {name}, u={u}: convergence orders " + " ".join(f"{x:.2f}" for x in r) + " (expected 1..5)")

    print("2. the exponents at u = 0 and u = -i, and conjugate symmetry")
    ts = np.linspace(0.0, 1.0, 11)
    for name, make in models:
        worst = 0.0
        for u in (0.0, -1j):
            worst = max(worst, max(abs(f(t)) for f in make(u)[1] for t in ts))
        if name.startswith("Bates"):
            worst = max(worst, max(abs(make(-1j)[2](t)) for t in ts[1:]))
        sym = max(abs(f(t) - np.conj(h(t))) for u in (0.5, 2.0)
                  for f, h in zip(make(-u)[1], make(u)[1]) for t in ts)
        good = worst < 1e-12 and sym < 1e-12
        ok &= good
        print(f"{'ok ' if good else 'BAD'} {name}: largest |g| at u = 0, -i {worst:.1e}; conjugate symmetry {sym:.1e}")

    print("3. frozen regimes: Lewis price from exp(int g) against an independent price")
    Ks = (80.0, 100.0, 120.0)
    ours = [lewis_call(lambda u: cmath.exp(merton76(u, [0.2, 0.2], [3.0, 3.0], MJ, SJ)[0][0].integral(T)), K) for K in Ks]
    ref = [merton_series_call(K, 0.2, 3.0) for K in Ks]
    d = max(abs(a - b) for a, b in zip(ours, ref))
    ok &= d < 1e-10
    print(f"{'ok ' if d < 1e-10 else 'BAD'} Merton76 against Merton's series: largest difference {d:.1e}")
    try:
        import QuantLib  # noqa: F401
        have_ql = True
    except ImportError:
        have_ql = False
        print("    QuantLib not installed: variance gamma and Bates comparisons skipped")
    if have_ql:
        vg = dict(sigma=0.2, nu=0.3, theta=-0.1)
        ours = [lewis_call(lambda u: cmath.exp(variance_gamma(u, [vg['sigma']] * 2, [vg['nu']] * 2, [vg['theta']] * 2)[0][0].integral(T)), K)
                for K in Ks]
        d = max(abs(a - b) for a, b in zip(ours, quantlib_calls('vg', Ks, **vg)))
        ok &= d < 1e-6
        print(f"{'ok ' if d < 1e-6 else 'BAD'} variance gamma against QuantLib VarianceGammaEngine: largest difference {d:.1e}")
        bp = dict(v0=0.05, kappa=2.0, theta=0.04, xi=0.4, rho=-0.6, lam=1.0)

        def phi_bates(u):
            g, _, D = bates(u, bp['kappa'], [bp['theta']] * 2, bp['xi'], bp['rho'], [bp['lam']] * 2, MJ, SJ, T)
            return cmath.exp(D(T) * bp['v0'] + g[0].integral(T))
        ours = [lewis_call(phi_bates, K) for K in Ks]
        d = max(abs(a - b) for a, b in zip(ours, quantlib_calls('bates', Ks, **bp)))
        ok &= d < 1e-8
        print(f"{'ok ' if d < 1e-8 else 'BAD'} Bates against QuantLib BatesEngine: largest difference {d:.1e}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
