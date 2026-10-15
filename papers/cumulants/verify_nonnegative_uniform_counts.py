"""Maturity-uniform count bounds with zero-intensity regimes."""

import numpy as np
from scipy.linalg import expm
from scipy.stats import nbinom, poisson

from polymoments import stationary
from verify_counts import count_distribution
from verify_uniform_positive_counts import constants, mixing_envelope, raw_moments, smooth_bound

ROUND_TOL = 65536*np.finfo(float).eps


def nonnegative_constants(q, intensity, c, mixing, theta=.1, block=5.):
    average = stationary(q) @ intensity
    fmax, high = max(abs(intensity-average)), max(intensity)
    killed = q-theta*np.diag(intensity)
    survival = max(expm(block*killed) @ np.ones(len(intensity)))
    assert 0 < survival < 1
    k = -np.log(survival)/block
    effective_floor, decay = k/(2*theta), k/(2*mixing)
    layer = 2*c*fmax/average
    coeff = constants(c, fmax, effective_floor, high)
    coeff[1] += 2*high**3*layer**3
    coeff[2] += 8*high**4/(survival*decay**4)+10*high**4*layer**4/3
    return coeff, effective_floor, survival, k, layer, killed


def main():
    designs = [
        ("zero symmetric", np.array([[-1., 1.], [1., -1.]]), np.array([3., 0.]), (1., 2.)),
        ("zero defective", np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]]), np.array([2., 0., 1.]), None),
    ]
    cases = layers = kernel_cases = 0
    tail_budget = 0.
    print("nonnegative uniform-count certificate; each design has a zero intensity")
    print("theta, killed block length, split horizon", .1, 5., 1.)
    for name, q, intensity, envelope in designs:
        pi = stationary(q)
        average, high = pi @ intensity, max(intensity)
        f = intensity-average
        c, mixing = envelope if envelope else mixing_envelope(q)
        coeff, a, survival, k, layer, killed = nonnegative_constants(q, intensity, c, mixing)
        print(name, "effective a, survival, k, layer", *(f"{x:.6f}" for x in [a, survival, k, layer]))
        for s in [.01, .2, 1., 4.9, 5., 5.1, 10., 20., 100.]:
            laplace = expm(s*killed) @ np.ones(len(pi))
            assert max(laplace) <= np.exp(-k*s)/survival+ROUND_TOL
            kernel_cases += 1
        for speed in [20., 80., 320., 1280.]:
            beta = speed*mixing
            uniform = min(1., coeff @ beta**(-np.arange(2, 5, dtype=float)))
            print(" speed, all-horizon bound", int(speed), f"{uniform:.10f}")
            for horizon in [.002, .02, .2, 1., 5., 20.]:
                raw = raw_moments(q, f, speed, horizon)
                limit = int(max(80., poisson.ppf(1-1e-12, high*horizon)+50))
                fft_size = 2**int(np.ceil(np.log2(2*limit)))
                for index, prior in enumerate(list(np.eye(len(pi)))+[pi]):
                    mean, second, third, fourth = (raw @ prior)[1:]
                    mu, v = average*horizon+mean, second-mean**2
                    m3 = third-3*mean*second+2*mean**3
                    m4 = fourth-4*mean*third+6*mean**2*second-3*mean**4
                    shape, success = mu*mu/v, mu/(mu+v)
                    assert mu > 0 and v > 0
                    assert abs(nbinom.mean(shape, success)-mu) <= ROUND_TOL*max(1., mu)
                    assert abs(nbinom.var(shape, success)-mu-v) <= ROUND_TOL*max(1., mu+v)
                    exact = count_distribution(speed*q, intensity, horizon, prior, M=fft_size)
                    tv = .5*np.sum(abs(exact[:limit]-nbinom.pmf(np.arange(limit), shape, success)))
                    tail = .5*(nbinom.sf(limit-1, shape, success)+poisson.sf(limit-1, high*horizon)
                               +poisson.sf(fft_size-1, high*horizon))
                    tail_budget = max(tail_budget, tail)
                    rounding = ROUND_TOL*max(1., np.linalg.norm(speed*q, np.inf)*horizon)
                    assert exact.min() >= -rounding and abs(exact.sum()-1) <= rounding
                    if horizon <= layer/beta:
                        local = 2*high**3*horizon**3+10*high**4*horizon**4/3
                        layers += 1
                    elif horizon <= 1.:
                        assert mu >= a*horizon-rounding
                        local = 2*abs(m3-2*v*v/mu)/3+(m4+3*v*v+6*v**3/mu**2)/3
                    else:
                        assert mu >= a*horizon-rounding
                        local = smooth_bound(mu, v, m3, m4, a, high, horizon) \
                            +high**4*horizon**4*np.exp(-k*speed*horizon/2)/(3*survival)
                    local = min(1., local)
                    assert tv+tail <= local+rounding and local <= uniform+rounding
                    if speed == 320. and horizon == 1. and index == 1:
                        print(" zero-regime start T=1 TV", f"{tv:.10f}")
                    cases += 1
    all_zero = count_distribution(designs[1][1], np.zeros(3), 1., np.array([1., 0., 0.]))
    assert abs(all_zero[0]-1) <= ROUND_TOL and np.max(abs(all_zero[1:])) <= ROUND_TOL
    print("count cases, short-layer cases, killed-kernel cases", cases, layers, kernel_cases)
    print("maximum omitted-tail TV budget", f"{tail_budget:.3e}")
    print("all-zero intensity gives the exact zero count")
    print("PASS")


if __name__ == "__main__":
    main()
