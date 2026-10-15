"""Positive-intensity count approximation, uniformly over horizon and initial prior."""

import math
import numpy as np
from scipy.stats import gamma as gamma_law, nbinom, poisson
from scipy.linalg import expm

from polymoments import stationary
from verify_counts import count_distribution

ROUND_TOL = 65536*np.finfo(float).eps
KAPPA = np.log(2.)-.5


def mixing_envelope(q, h=.5):
    p = expm(h*q)
    rho = .5*max(np.sum(abs(a-b)) for a in p for b in p)
    assert 0 < rho < 1
    return 2/rho, -np.log(rho)/h


def raw_moments(q, f, speed, horizon):
    n = len(f)
    system = np.kron(np.eye(5), speed*q)
    for j in range(1, 5):
        system[j*n:(j+1)*n, (j-1)*n:j*n] = j*np.diag(f)
    initial = np.zeros(5*n)
    initial[:n] = 1
    return (expm(horizon*system) @ initial).reshape(5, n)


def constants(c, fmax, low, high, split=1.):
    alpha = 2*c*fmax**2
    t3, u3 = 18*c*c*fmax**3, 2*c**3*fmax**3
    t4, u4 = 24*c*c*fmax**4, 60*c**3*fmax**4
    small2 = 2*split*(t3+2*alpha**2/low)/3+split**2*(t4/3+alpha**2)
    small3 = 2*u3/3+split*(u4/3+2*alpha**3/low**2)
    large2 = np.sqrt(6)*(t3+2*alpha**2/low)/(12*low**1.5*np.sqrt(split)) \
        +np.sqrt(24)*(t4+12*alpha**2)/(48*low**2)
    large3 = np.sqrt(6)*u3/(12*(low*split)**1.5) \
        +np.sqrt(24)*(u4+24*alpha**3/low**2)/(48*low**2*split)
    decay = KAPPA*low**2/alpha
    return np.array([max(small2, large2), max(small3, large3), 8*high**4/decay**4])


def smooth_bound(mu, v, m3, m4, low, high, horizon):
    gamma4 = 3*v*v+6*v**3/mu**2
    return np.sqrt(6)*abs(m3-2*v*v/mu)/(12*mu**1.5) \
        +np.sqrt(24)*(m4+4*gamma4)/(48*(low*horizon)**2) \
        +high**4*horizon**4*np.exp(-KAPPA*mu*mu/v)/3


def symmetric_distribution(speed, intensity, horizon, prior, size):
    z = np.exp(2j*np.pi*np.arange(size)/size)-1
    average, contrast = intensity.mean(), (intensity[0]-intensity[1])/2
    d = np.sqrt(speed**2+(contrast*z)**2)
    weight = (speed+(prior[0]-prior[1])*contrast*z)/d
    pgf = .5*((1+weight)*np.exp(horizon*(average*z+(contrast*z)**2/(d+speed)))
              +(1-weight)*np.exp(horizon*(average*z-speed-d)))
    return np.fft.fft(pgf).real/size


def main():
    # Independently check the Poisson derivative norm and Charlier L2 identity.
    for x in [.2, 1., 5., 30., 100.]:
        k = np.arange(int(poisson.ppf(1-1e-12, x))+50)
        p = poisson.pmf(k, x)
        for degree in [3, 4]:
            deriv = sum((-1)**(degree-j)*math.comb(degree, j)*poisson.pmf(k-j, x)
                        for j in range(degree+1))
            assert np.sum(abs(deriv)) <= np.sqrt(math.factorial(degree))*x**(-degree/2)+ROUND_TOL
            assert abs(np.sum(deriv**2/p)*x**degree/math.factorial(degree)-1) <= ROUND_TOL
    for shape in [.2, 1., 3., 10., 100.]:
        assert gamma_law.cdf(shape/2, shape) <= np.exp(-KAPPA*shape)+ROUND_TOL
    designs = [
        ("symmetric", np.array([[-1., 1.], [1., -1.]]), np.array([3., 1.]), (1., 2.)),
        ("defective", np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]]), np.array([2., 1., 1.5]), None),
    ]
    count_cases = 0
    largest_tail = largest_spectral = 0.
    print("uniform positive-count certificate; horizons", [.02, .2, 1., 5., 20., 100.])
    for name, q, intensity, envelope in designs:
        pi = stationary(q)
        average = pi @ intensity
        f = intensity-average
        low, high = min(intensity), max(intensity)
        c, mixing = envelope if envelope else mixing_envelope(q)
        coeff = constants(c, max(abs(f)), low, high)
        print(name, "A2, A3, A4", *(f"{x:.6g}" for x in coeff))
        priors = list(np.eye(len(pi)))+[pi]
        for speed in [20., 40., 80., 160.]:
            beta = speed*mixing
            uniform = min(1., coeff @ beta**(-np.arange(2, 5, dtype=float)))
            print(" speed, all-horizon bound", int(speed), f"{uniform:.10f}")
            for horizon in [.02, .2, 1., 5., 20., 100.]:
                raw = raw_moments(q, f, speed, horizon)
                limit = int(max(80., poisson.ppf(1-1e-12, high*horizon)+50))
                fft_size = 2**int(np.ceil(np.log2(2*limit)))
                for index, prior in enumerate(priors):
                    a, second, third, fourth = (raw @ prior)[1:]
                    mu, v = average*horizon+a, second-a*a
                    m3, m4 = third-3*a*second+2*a**3, fourth-4*a*third+6*a*a*second-3*a**4
                    shape, success = mu*mu/v, mu/(mu+v)
                    # Mean/variance errors inherit the scale of their exact count moments.
                    assert abs(nbinom.mean(shape, success)-mu) <= ROUND_TOL*max(1., mu)
                    assert abs(nbinom.var(shape, success)-mu-v) <= ROUND_TOL*max(1., mu+v)
                    exact = count_distribution(speed*q, intensity, horizon, prior, M=fft_size)
                    if name == "symmetric":
                        spectral = symmetric_distribution(speed, intensity, horizon, prior, fft_size)
                        gap = np.sum(abs(spectral-exact))
                        largest_spectral = max(largest_spectral, gap)
                        assert gap <= ROUND_TOL*max(1., high*horizon)
                    roundoff = ROUND_TOL*max(1., np.linalg.norm(speed*q, np.inf)*horizon)
                    assert exact.min() >= -roundoff and abs(exact.sum()-1) <= roundoff
                    tv = .5*np.sum(abs(exact[:limit]-nbinom.pmf(np.arange(limit), shape, success)))
                    tail = .5*(nbinom.sf(limit-1, shape, success)+poisson.sf(limit-1, high*horizon)
                               +poisson.sf(fft_size-1, high*horizon))
                    largest_tail = max(largest_tail, tail)
                    assert tv+tail <= smooth_bound(mu, v, m3, m4, low, high, horizon)+roundoff
                    local_bound = (2*abs(m3-2*v*v/mu)/3+(m4+3*v*v+6*v**3/mu**2)/3
                                   if horizon <= 1. else smooth_bound(mu, v, m3, m4, low, high, horizon))
                    assert local_bound <= uniform+roundoff
                    assert tv+tail <= uniform+roundoff
                    if index == 0 and speed == 80. and horizon in [1., 20., 100.]:
                        print(" T, pure-first-state TV", horizon, f"{tv:.10f}")
                    count_cases += 1
    equal = count_distribution(designs[1][1], np.full(3, 2.), 1., np.array([1., 0., 0.]))
    assert np.max(abs(equal[:80]-poisson.pmf(np.arange(80), 2.))) <= ROUND_TOL
    print("count-law comparisons", count_cases, "including every pure initial prior; constant-intensity check passed")
    print("maximum omitted-tail TV budget", f"{largest_tail:.3e}")
    print("maximum symmetric closed-form PGF L1 discrepancy", f"{largest_spectral:.3e}")
    print("base roundoff tolerance", f"{ROUND_TOL:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
