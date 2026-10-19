"""Sharp bounded-noise entropy recovery; exact rational indistinguishability."""
from fractions import Fraction as F
import math
from decimal import Decimal, getcontext
import numpy as np


def call(law, k):
    return sum((p*max(s-k, F(0)) for s, p in law), F(0))


def adversaries(m, delta):
    lam, c = (m-1)/(4*m), (m-1)/8
    assert delta <= F(1, 4) and delta <= lam/2 and delta*delta <= c*lam/2
    q, u, eps = 2*delta*delta/c, c/delta, delta/lam
    d = (1-2*delta)/(1-lam-q)
    r = [(F(0), lam), (u, q), (d, 1-lam-q)]
    p = [(eps, lam), (u/2, q), (d, 1-lam-q)]
    return p, r, (lam, c, q, u, eps)


def dec(x):
    return Decimal(x.numerator)/x.denominator


def entropy(law):
    return sum((dec(p)*dec(s)*dec(s).ln() for s, p in law if s), Decimal(0))


def exact_aliases():
    count = 0
    getcontext().prec = 90
    for m in (F(101, 100), F(5, 4), F(2), F(23, 5), F(100)):
        lam, c = (m-1)/(4*m), (m-1)/8
        # A rational cutoff not requiring a rounded square root.
        cutoff = min(F(1, 4), lam/2, c*lam/2, F(1))
        for scale in (1, 2, 10, 100, 10000):
            delta = cutoff/scale
            p, r, (lam, c, q, u, eps) = adversaries(m, delta)
            for law in (p, r):
                assert min(x for x, _ in law) >= 0 and min(w for _, w in law) >= 0
                assert sum(w for _, w in law) == 1
                assert sum(w*x for x, w in law) == 1
                assert sum(w*x*x for x, w in law) <= 1+3*(m-1)/4 < m
            assert sum(w*x*x for x, w in p)-sum(w*x*x for x, w in r) == delta*delta/lam-3*c/2 <= -c
            nodes = sorted(set([F(0), eps, u/2, u] + [x for x, _ in p+r]))
            probes = nodes + [(a+b)/2 for a, b in zip(nodes[:-1], nodes[1:])] + [2*nodes[-1]]
            for k in probes:
                trapezoid = lam*k if k <= eps else delta if k <= u/2 else q*(u-k) if k <= u else F(0)
                gap = call(r, k)-call(p, k)
                assert gap == trapezoid and 0 <= gap <= delta
                mid = (call(r, k)+call(p, k))/2
                assert abs(mid-call(p, k)) <= delta/2 and abs(mid-call(r, k)) <= delta/2
            assert call(r, eps)-call(p, eps) == delta
            exact_gap = dec(delta)*dec(2*lam*c/(delta*delta)).ln()
            evaluated = entropy(r)-entropy(p)
            # Portable absolute floor; exact rational checks above use no tolerance.
            assert abs(evaluated-exact_gap) <= Decimal('1e-12')*(1+abs(entropy(r))+abs(entropy(p)))
            count += 1
    print(f"Rational adversarial pairs: {count}; all masses/means/moments and trapezoid knots exact")


def upper_strips():
    rng = np.random.default_rng(210)
    maximum_ratio, count = 0., 0
    for eta in (.01, .001, .0001):
        lo, hi = eta, 1/eta
        n = math.ceil(math.log(hi/lo)/math.log1p(math.sqrt(8*eta)))
        h = math.log(hi/lo)/n
        k = np.exp(np.linspace(math.log(lo), math.log(hi), n+1))
        dh = h/(-math.expm1(-h))-1
        w = np.r_[dh, np.full(n-1, h), h-dh]
        assert np.min(w) > 0 and abs(w.sum()-math.log(hi/lo)) < 2e-12
        assert math.expm1(h)**2/8 <= eta+1e-12
        for _ in range(20):
            s = np.r_[0., np.exp(rng.uniform(-12, 10, 12))]
            prob = rng.dirichlet(np.ones(len(s)))
            s /= prob @ s
            m = prob @ (s*s)
            true_h = float(sum(p*x*math.log(x) for x, p in zip(s, prob) if x))
            quotes = np.maximum(s[:, None]-k, 0).T @ prob
            fk = k*np.log(k)-k+1
            g = np.interp(s, k, fk)
            left, right = s < lo, s > hi
            g[left] = 1-lo+math.log(lo)*s[left]
            g[right] = 1-hi+math.log(hi)*s[right]
            replicated = 1-lo+math.log(lo) + w @ quotes
            assert abs(prob @ g-replicated) < 2e-10*(1+abs(replicated))
            bias = prob @ g-true_h
            assert -eta*(1+m/2)-2e-10 <= bias <= eta+2e-10
            center = 1-lo+math.log(lo) + w @ quotes + m*eta/4
            budget = 2*eta*math.log(1/eta)+(1+m/4)*eta
            for sign in (-1, 1):
                estimate = 1-lo+math.log(lo) + w @ (quotes+sign*eta) + m*eta/4
                assert abs(estimate-center-sign*eta*w.sum()) < 2e-10*(1+abs(center))
                assert abs(estimate-true_h) <= budget+2e-10*(1+abs(true_h))
                maximum_ratio = max(maximum_ratio, abs(estimate-true_h)/budget)
            count += 1
    print(f"Noisy finite-strip laws: {count}; largest error/budget = {maximum_ratio:.6f}")


if __name__ == '__main__':
    exact_aliases()
    upper_strips()
    m = F(2)
    print("M=2: eta, lower bound, upper bound, lower/(eta log(1/eta)), upper/(eta log(1/eta))")
    for power in (3, 6, 12, 30):
        eta = F(1, 10**power)
        _, _, (lam, c, _, _, _) = adversaries(m, 2*eta)
        e = dec(eta)
        lower = 2*e*(1/e).ln()+e*dec(lam*c/2).ln()
        upper = 2*e*(1/e).ln()+(1+dec(m)/4)*e
        scale = e*(1/e).ln()
        print(f"1e-{power}: {float(lower):.12g}, {float(upper):.12g}, {float(lower/scale):.9f}, {float(upper/scale):.9f}")
    print("PASS")
