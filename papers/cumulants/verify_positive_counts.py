"""A probability-preserving Cox-count corrector and explicit TV bounds."""

import numpy as np
from scipy.linalg import expm
from scipy.stats import nbinom, poisson

from verify_counts import count_distribution


ROUND_TOL = 8192*np.finfo(float).eps


def central_moments(rate, contrast, maturity, prior=(.5, .5)):
    q = rate*np.array([[-1., 1.], [1., -1.]])
    system = np.kron(np.eye(5), q)
    for degree in range(1, 5):
        system[2*degree:2*degree+2, 2*degree-2:2*degree] = degree*np.diag([contrast, -contrast])
    initial = np.zeros(10)
    initial[:2] = 1
    moments = (expm(maturity*system) @ initial).reshape(5, 2) @ np.asarray(prior)
    return moments


def arbitrary_prior_checks():
    largest_error, cases = 0., 0
    endpoint_distances = {-1.: [], 1.: []}
    for intensities in [np.array([8., 1.]), np.array([2., 0.])]:
        average, contrast = intensities.mean(), (intensities[0]-intensities[1])/2
        designs = [(r, 1.) for r in [1., 5., 10., 20., 40., 80., 160.]]+[(2., .2), (2., 3.)]
        for rate, maturity in designs:
            e = np.exp(-2*rate*maturity)
            a = -np.expm1(-2*rate*maturity)/(2*rate)
            b = maturity/rate + np.expm1(-2*rate*maturity)/(2*rate**2)
            minimum_mean = average*maturity-abs(contrast)*a
            uniform_bound = min(1., contrast**4*(2*maturity**2+4*maturity**2/(3*minimum_mean))/rate**2
                                +(2*abs(contrast)**3/3+contrast**4*maturity/2
                                  +2*contrast**6*maturity**3/minimum_mean**2)/rate**3)
            for eta in [-1., -.5, 0., .5, 1.]:
                prior = np.array([(1+eta)/2, (1-eta)/2])
                raw = central_moments(rate, contrast, maturity, prior)
                mean = average*maturity+contrast*eta*a
                variance = contrast**2*(b-eta**2*a*a)
                third = raw[3]-3*raw[1]*raw[2]+2*raw[1]**3
                fourth = raw[4]-4*raw[1]*raw[3]+6*raw[1]**2*raw[2]-3*raw[1]**4
                exact_third = contrast**3*(3*eta*(4*rate*maturity*e+e*e-1)+eta**3*(1-e)**3)/(4*rate**3)
                largest_error = max(largest_error, abs(third-exact_third)/max(1., abs(raw[3])),
                                    abs(raw[2]-raw[1]**2-variance)/max(1., raw[2]),
                                    abs(average*maturity+raw[1]-mean)/max(1., mean))
                assert abs(third) <= abs(contrast)**3/rate**3+ROUND_TOL*max(1., abs(raw[3]))
                assert fourth <= 3*contrast**4*maturity**2/rate**2+1.5*contrast**4*maturity/rate**3+ROUND_TOL*max(1., raw[4])
                shape, success = mean**2/variance, mean/(mean+variance)
                q = rate*np.array([[-1., 1.], [1., -1.]])
                exact = count_distribution(q, intensities, maturity, prior)
                distance = .5*np.sum(abs(exact[:80]-nbinom.pmf(np.arange(80), shape, success)))
                tail = .5*(nbinom.sf(79, shape, success)+poisson.sf(79, intensities.max()*maturity)
                           +poisson.sf(255, intensities.max()*maturity))
                moment_bound = min(1., 2*abs(third-2*variance**2/mean)/3
                                   +(fourth+3*variance**2+6*variance**3/mean**2)/3)
                assert distance+tail <= moment_bound+ROUND_TOL
                assert moment_bound <= uniform_bound+ROUND_TOL
                if intensities[0] == 8. and eta in endpoint_distances and maturity == 1.:
                    endpoint_distances[eta].append(distance)
                    if rate == 40.:
                        print("pure-start eta, NB TV, uniform-prior bound", eta, f"{distance:.10f}", f"{uniform_bound:.10f}")
                cases += 1
    assert largest_error <= ROUND_TOL
    print("arbitrary-prior grid cases", cases, "including zero lower intensity; horizons", [.2, 1., 3.])
    print("maximum arbitrary-prior moment discrepancy", f"{largest_error:.3e}")
    print("pure-start last measured orders", *(f"{np.log2(x[-2]/x[-1]):.6f}" for x in endpoint_distances.values()))


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
    arbitrary_prior_checks()
    print("roundoff tolerance", f"{ROUND_TOL:.3e}")
    print("PASS")


if __name__ == "__main__":
    main()
