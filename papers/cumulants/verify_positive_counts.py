"""A probability-preserving Cox-count corrector and explicit TV bounds."""

import numpy as np
from scipy.linalg import expm
from scipy.stats import nbinom, poisson

from verify_counts import count_distribution


ROUND_TOL = 8192*np.finfo(float).eps


def central_moments(rate, contrast, maturity):
    q = rate*np.array([[-1., 1.], [1., -1.]])
    system = np.kron(np.eye(5), q)
    for degree in range(1, 5):
        system[2*degree:2*degree+2, 2*degree-2:2*degree] = degree*np.diag([contrast, -contrast])
    initial = np.zeros(10)
    initial[:2] = 1
    moments = (expm(maturity*system) @ initial).reshape(5, 2).mean(axis=1)
    return moments


def main():
    intensities = np.array([8., 1.])
    mean, contrast, maturity = 4.5, 3.5, 1.
    largest_error = tail_budget = 0.
    distances = []
    print("positive Cox-count certificate; intensities (8, 1), stationary start, T=1")
    print("speed; Poisson TV; negative-binomial TV; moment bound; explicit speed bound")
    for rate in [5., 10., 20., 40., 80., 160.]:
        variance = contrast**2*(maturity/rate + np.expm1(-2*rate*maturity)/(2*rate**2))
        moments = central_moments(rate, contrast, maturity)
        fourth_bound = 3*contrast**4*maturity**2/rate**2
        largest_error = max(largest_error, abs(moments[2]-variance)/max(1., variance), abs(moments[3]))
        assert moments[4] <= fourth_bound+ROUND_TOL*max(1., fourth_bound)
        shape, scale = mean**2/variance, variance/mean
        success = 1/(1+scale)
        assert abs(nbinom.mean(shape, success)-mean) <= ROUND_TOL*max(1., mean)
        assert abs(nbinom.var(shape, success)-(mean+variance)) <= ROUND_TOL*max(1., mean+variance)
        q = rate*np.array([[-1., 1.], [1., -1.]])
        exact_mod = count_distribution(q, intensities, maturity, np.array([.5, .5]), M=256)
        assert exact_mod.min() >= -ROUND_TOL and abs(exact_mod.sum()-1) <= ROUND_TOL
        indices = np.arange(80)
        negative_binomial = nbinom.pmf(indices, shape, success)
        averaged = poisson.pmf(indices, mean)
        # The exact count tail is dominated by Pois(8); modulo aliasing is smaller still.
        nb_tail = nbinom.sf(79, shape, success)
        true_tail = poisson.sf(79, 8.)
        alias_tail = poisson.sf(255, 8.)
        tails = .5*(nb_tail+true_tail+alias_tail)
        tail_budget = max(tail_budget, tails)
        nb_distance = .5*np.sum(abs(exact_mod[:80]-negative_binomial))
        poisson_distance = .5*np.sum(abs(exact_mod[:80]-averaged))
        gamma_third = 2*variance**2/mean
        gamma_fourth = 3*variance**2+6*variance**3/mean**2
        moment_bound = min(1., 2*abs(moments[3]-gamma_third)/3+(moments[4]+gamma_fourth)/3)
        explicit = min(1., contrast**4*(2*maturity**2+4*maturity/(3*4.5))/rate**2
                       +2*contrast**6*maturity/(4.5**2*rate**3))
        assert nb_distance+tails <= moment_bound+ROUND_TOL
        assert moment_bound <= explicit+ROUND_TOL
        assert nb_distance < poisson_distance
        distances.append(nb_distance)
        print(f"{rate:.0f}; {poisson_distance:.10f}; {nb_distance:.10f}; {moment_bound:.10f}; {explicit:.10f}")
    order = np.log2(distances[-2]/distances[-1])
    assert 1.9 < order < 2.1
    assert largest_error <= ROUND_TOL
    # Equal regime intensities have deterministic integrated rate and exactly Poisson counts.
    equal = count_distribution(np.array([[-1., 1.], [1., -1.]]), np.array([2., 2.]), 1., [.5, .5])
    assert np.max(abs(equal[:80]-poisson.pmf(np.arange(80), 2.))) <= ROUND_TOL
    # The general moment theorem also covers asymmetric, non-telegraph mixing laws.
    for probability in [.2, .5, .8]:
        for amplitude in [.1, .3, .9]:
            mu = 2.
            var = amplitude**2*probability*(1-probability)
            third = amplitude**3*probability*(1-probability)*(1-2*probability)
            fourth = amplitude**4*probability*(1-probability)*(
                probability**3+(1-probability)**3)
            shape, success = mu**2/var, mu/(mu+var)
            mixture = ((1-probability)*poisson.pmf(indices, mu-amplitude*probability)
                       +probability*poisson.pmf(indices, mu+amplitude*(1-probability)))
            gamma_count = nbinom.pmf(indices, shape, success)
            bound = 2*abs(third-2*var**2/mu)/3+(fourth+3*var**2+6*var**3/mu**2)/3
            tails = .5*(nbinom.sf(79, shape, success)
                        +(1-probability)*poisson.sf(79, mu-amplitude*probability)
                        +probability*poisson.sf(79, mu+amplitude*(1-probability)))
            assert .5*np.sum(abs(mixture-gamma_count))+tails <= bound+ROUND_TOL
    print("asymmetric two-point mixing checks", 9)
    print("last measured negative-binomial error order", f"{order:.6f}")
    print("maximum central-moment identity discrepancy", f"{largest_error:.3e}")
    print("maximum omitted-tail TV budget", f"{tail_budget:.3e}")
    print("roundoff tolerance", f"{ROUND_TOL:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
