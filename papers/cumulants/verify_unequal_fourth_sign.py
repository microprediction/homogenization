"""Exact stationary two-state fourth-cumulant sign diagram."""

import mpmath as mp
import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.stats import poisson
from verify_counts import count_distribution

mp.mp.dps = 60
ROUND_TOL = 8192*np.finfo(float).eps
COUNT_TOL = 2e-8  # Fourier factorial moments amplify inversion noise by k**4.


def shapes(x):
    y = mp.mpf(str(x))
    b = y-3+(y*y/2+2*y+3)*mp.exp(-y)
    c = y-mp.mpf("2.5")+2*(y+1)*mp.exp(-y)+mp.exp(-2*y)/2
    return float(b), float(c)


def threshold(x):
    b, c = shapes(x)
    return b/(4*b+c)


def threshold_mp(x):
    b = x-3+(x*x/2+2*x+3)*mp.exp(-x)
    c = x-mp.mpf("2.5")+2*(x+1)*mp.exp(-x)+mp.exp(-2*x)/2
    return b/(4*b+c)


def crossing_mp(w, guess):
    lo = mp.mpf("1e-5")
    hi = max(mp.mpf(1), 2*guess)
    while threshold_mp(hi) < w:
        hi *= 2
    for _ in range(240):
        mid = (lo+hi)/2
        if threshold_mp(mid) < w:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def fourth(p, speed, horizon):
    w = p*(1-p)
    b, c = shapes(speed*horizon)
    return 24*w*(b-w*(4*b+c))/speed**4


def moment_fourth(p, speed, horizon):
    q = speed*np.array([[-(1-p), 1-p], [p, -p]])
    f = np.array([1-p, -p])
    system = np.kron(np.eye(5), q)
    for degree in range(1, 5):
        system[2*degree:2*degree+2, 2*degree-2:2*degree] = degree*np.diag(f)
    initial = np.zeros(10)
    initial[:2] = 1
    raw = (expm(horizon*system) @ initial).reshape(5, 2) @ np.array([p, 1-p])
    a, v2, v3, v4 = raw[1:]
    k4 = v4-4*a*v3-3*v2*v2+12*v2*a*a-6*a**4
    return k4, max(1., abs(v4), 3*v2*v2)


def main():
    xs = np.geomspace(1e-4, 1e4, 300)
    qs = np.array([threshold(x) for x in xs])
    assert np.all(np.diff(qs) > 0) and np.all((qs > 1/6) & (qs < 1/5))
    matrix_error = 0.
    cases = 0
    for p in [.05, .15, .22, .25, .3, .5, .75, .85, .95]:
        for speed in [.5, 1., 3.]:
            for x in [.001, .01, .1, .5, 1., 3., 10., 50., 200.]:
                horizon = x/speed
                exact = fourth(p, speed, horizon)
                independent, scale = moment_fourth(p, speed, horizon)
                matrix_error = max(matrix_error, abs(exact-independent)/scale)
                assert abs(exact-independent) <= ROUND_TOL*scale
                cases += 1
    p = .25
    w = p*(1-p)
    crossing = brentq(lambda x: threshold(x)-w, .01, 100., xtol=1e-12)
    assert abs(threshold(crossing)-w) <= ROUND_TOL
    assert fourth(p, 1., crossing/2) < 0 < fourth(p, 1., crossing*2)
    small_relative_error = 0.
    for delta_text in [".001", ".0005", ".0002", ".0001", ".00005"]:
        delta = mp.mpf(delta_text)
        root = crossing_mp(mp.mpf(1)/6+delta, 90*delta)
        cubic = 90*delta+1350*delta**2+mp.mpf(210600)/7*delta**3
        fourth_coefficient = (root-cubic)/delta**4
        small_relative_error = max(
            small_relative_error, abs(float(fourth_coefficient/729000-1)))
        assert abs(float(threshold_mp(root)-(mp.mpf(1)/6+delta))) <= 1e-12
    assert small_relative_error <= .03
    upper_scaled_error = 0.
    for eta_text in [".01", ".005", ".002", ".001", ".0005"]:
        eta = mp.mpf(eta_text)
        rational_center = 1/(50*eta)+mp.mpf(29)/10
        root = crossing_mp(mp.mpf(1)/5-eta, rational_center)
        scale = eta**-3*mp.exp(-1/(50*eta))
        upper_scaled_error = max(
            upper_scaled_error, abs(float((root-rational_center)/scale)))
        assert abs(float(threshold_mp(root)-(mp.mpf(1)/5-eta))) <= 1e-12
    assert upper_scaled_error <= 1e-4
    print("stationary two-state phase diagram: 0<w<=1/6 positive; 1/6<w<1/5 one crossing; 1/5<=w<=1/4 negative")
    print("threshold strictly increases from 1/6 to 1/5; numerical grid points", len(xs))
    print("p=0.25, covariance decay rate 1, unique crossing horizon", f"{crossing:.10f}")
    count_error = tail_budget = 0.
    for horizon in [1., 5.]:
        exact = fourth(p, 1., horizon)
        q = np.array([[-.75, .75], [.25, -.25]])
        law = count_distribution(q, np.array([1., 0.]), horizon, np.array([p, 1-p]))
        k = np.arange(80)
        fm = [np.sum(law[:80]*np.prod([k-j for j in range(degree)], axis=0))
              for degree in range(1, 5)]
        a, b, c, d = fm
        from_count = d-4*c*a-3*b*b+12*b*a*a-6*a**4
        scale = max(1., abs(d), 4*abs(c*a), 3*b*b, 12*abs(b*a*a), 6*a**4)
        count_error = max(count_error, abs(from_count-exact)/scale)
        # Domination by Pois(T): bound factorial-moment tails and modulo-256 aliasing.
        tail = [horizon**j*(poisson.sf(79-j, horizon)+poisson.sf(255-j, horizon))
                for j in range(1, 5)]
        budget = 52*horizon**3*tail[0]+18*horizon**2*tail[1]+4*horizon*tail[2]+tail[3]
        tail_budget = max(tail_budget, budget)
        assert abs(from_count-exact) <= COUNT_TOL*scale+budget
        assert np.sign(from_count) == np.sign(exact)
        print(" p=0.25, T, integrated-rate kappa4, count fourth factorial cumulant",
              horizon, f"{exact:.10f}", f"{from_count:.10f}")
    # Recover the original symmetric formula with x=lambda*T, not x=(alpha+beta)*T.
    for x in [.1, .5, 1., 3., 10.]:
        y = mp.mpf(str(x))
        original = -(3*y-mp.mpf("3.75")+(6*y+3)*mp.exp(-2*y)+3*mp.exp(-4*y)/4)
        assert abs(fourth(.5, 2., x)-float(original)/16) <= ROUND_TOL
    print("independent matrix-moment cases", cases)
    print("maximum scaled matrix discrepancy", f"{matrix_error:.3e}")
    print("maximum scaled count factorial-cumulant discrepancy", f"{count_error:.3e}")
    print("maximum factorial-cumulant truncation/alias budget", f"{tail_budget:.3e}")
    print("maximum relative lower-end fourth-coefficient discrepancy", f"{small_relative_error:.3e}")
    print("maximum scaled upper-end crossing remainder", f"{upper_scaled_error:.3e}")
    print("matrix and count tolerances", f"{ROUND_TOL:.3e}", f"{COUNT_TOL:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()

