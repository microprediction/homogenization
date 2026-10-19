"""Finite entropy quotes: graded-grid upper bound and exact fixed-design aliases."""
from fractions import Fraction as F
import math
import numpy as np


def f(s):
    s = np.asarray(s, dtype=float)
    out = 1 - s
    positive = s > 0
    out[positive] += s[positive] * np.log(s[positive])
    return out


def design(n):
    knots = np.r_[np.arange(n + 1, dtype=float)**2 / n**2,
                  n**2 / np.arange(n - 1, 0, -1, dtype=float)**2]
    slopes = np.diff(f(knots)) / np.diff(knots)
    weights = np.diff(np.r_[slopes, math.log(knots[-1])])
    assert len(weights) == 2*n - 1 and np.all(weights > 0)
    assert abs(weights.sum() - (1 + 4*math.log(n))) < 2e-10
    return knots, slopes, weights


def chord(s, knots):
    out = np.interp(s, knots, f(knots))
    tail = s > knots[-1]
    out[tail] = f(knots[-1:])[0] + math.log(knots[-1])*(s[tail]-knots[-1])
    return out


def law(rng):
    s = np.r_[0., np.exp(rng.uniform(-8, 7, 14))]
    p = rng.dirichlet(np.ones(len(s)))
    s /= p @ s
    return s, p


def calls(s, p, strikes):
    return np.maximum(s[:, None] - strikes, 0).T @ p


def test_interpolation():
    rng = np.random.default_rng(208)
    utilization = 0.
    count = 0
    for n in (2, 3, 8, 32, 100):
        knots, slopes, w = design(n)
        s = np.r_[0., np.geomspace(knots[1]/100, knots[-1]*100, 1000)]
        g = chord(s, knots)
        replicated = 1 + slopes[0]*s + np.maximum(s[:, None]-knots[1:], 0) @ w
        scale = 1 + np.abs(g)
        assert np.max(np.abs(replicated-g)/scale) < 2e-10
        err = g-f(s)
        assert np.all(err >= -s*s/(2*n*n)-2e-9*(1+s*s))
        assert np.all(err <= (9/8 + 4.5*s*s)/(n*n)+2e-9*(1+s*s))
        for _ in range(20):
            sp, pp = law(rng)
            sr, pr = law(rng)
            m = max(pp @ sp**2, pr @ sr**2)
            hp, hr = pp @ f(sp), pr @ f(sr)
            cp, cr = calls(sp, pp, knots[1:]), calls(sr, pr, knots[1:])
            a = w @ (cp-cr)
            budget = (9/8 + 5*m)/n**2
            utilization = max(utilization, abs(a-(hp-hr))/budget)
            assert abs(a-(hp-hr)) <= budget + 2e-10*(1+abs(a)+hp+hr)
            for stocks, prob, c, h in ((sp, pp, cp, hp), (sr, pr, cr, hr)):
                mean_err = prob @ (chord(stocks, knots)-f(stocks))
                assert -m/(2*n*n)-2e-10 <= mean_err <= (9/8+4.5*m)/n**2+2e-10
                estimate = 1 + slopes[0] + w @ c - (9/16+2*m)/n**2
                assert abs(estimate-h) <= budget/2 + 2e-10*(1+h)
            eta = 1e-6
            noisy = w @ ((cp+eta)-(cr-eta))
            assert abs((noisy-a)-2*eta*w.sum()) < 2e-10*(1+abs(a))
            count += 1
    print(f"Atomic/default law pairs: {count}; maximum error/budget = {utilization:.6f}")


def rational_calls(stocks, probs, k):
    return sum((p*max(s-k, F(0)) for s, p in zip(stocks, probs)), F(0))


def test_aliases():
    rng = np.random.default_rng(209)
    count, minimum_ratio = 0, math.inf
    for m in (F(101, 100), F(5, 4), F(23, 5)):
        lam = min(F(1, 4), m-1)
        for n in (0, 1, 2, 10, 50):
            for _ in range(5):
                strikes = [F(int(k), 10000) for k in rng.integers(0, 25000, n)]
                boundaries = sorted(set([F(1, 2), F(3, 2)] +
                                        [k for k in strikes if F(1, 2) < k < F(3, 2)]))
                a, b = max(zip(boundaries[:-1], boundaries[1:]), key=lambda ab: ab[1]-ab[0])
                h, c = b-a, (a+b)/2
                d = (1-lam*c)/(1-lam)
                sp, pp = [a, b, d], [lam/2, lam/2, 1-lam]
                sr, pr = [c, d], [lam, 1-lam]
                # Exact rational inequalities/identities, not floating tolerances.
                assert h >= F(1, n+1) and min(sp+sr) >= 0
                for stocks, prob in ((sp, pp), (sr, pr)):
                    assert sum(prob) == 1 and sum(p*s for p, s in zip(prob, stocks)) == 1
                    assert sum(p*s*s for p, s in zip(prob, stocks)) <= 1+7*lam/12 <= m
                for k in strikes + [a, b]:
                    assert rational_calls(sp, pp, k) == rational_calls(sr, pr, k)
                gap = lam*(F(1, 2)*float(f(np.array([float(a)]))[0]) +
                           F(1, 2)*float(f(np.array([float(b)]))[0]) -
                           float(f(np.array([float(c)]))[0]))
                lower = float(lam*h*h/12)
                assert gap >= lower - 2e-12
                minimum_ratio = min(minimum_ratio, gap/lower)
                count += 1
    # The scalar grid inequality used for every upper cell, checked algebraically.
    # 6(l-1)^2-l(2l-1) = (l-2)(4l-3) >= 0 for l >= 2.
    for ell in range(2, 1000):
        assert 6*(ell-1)**2-ell*(2*ell-1) == (ell-2)*(4*ell-3) >= 0
    print(f"Exact rational alias designs: {count}; minimum entropy-gap/lower-bound = {minimum_ratio:.6f}")
    print("At M=5/4: lambda=1/4; minimax lower bound = 1/[96(N+1)^2]")


if __name__ == "__main__":
    test_interpolation()
    test_aliases()
    m, epsilon = 4.6, 46e-6
    n = max(2, math.ceil(math.sqrt((9/8+5*m)/epsilon)))
    knots, _, w = design(n)
    print(f"Design M={m}, epsilon={epsilon:g}: n={n}, quotes={len(w)}")
    print(f"E_n={(9/8+5*m)/n**2:.12g}; W_n={w.sum():.9f}; strikes=[{knots[1]:.9g}, {knots[-1]:.0f}]")
    print("PASS")
