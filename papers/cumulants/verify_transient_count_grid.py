"""All-prior Cox count-grid decoupling without conditioning on rare states."""

import numpy as np
from scipy.linalg import expm
from scipy.integrate import quad_vec
from scipy.stats import nbinom, poisson

from polymoments import stationary
from verify_stationary_count_grid import SIZE, ROUND_TOL, conditional_start, end_joint, envelope, kernels, product


def replacement(q, intensity, speed, duration, prior, pi):
    average = pi @ intensity
    if np.ptp(intensity) == 0:
        return poisson.pmf(np.arange(SIZE), average*duration), poisson.sf(SIZE-1, average*duration)
    f, n = intensity-average, len(pi)
    generator = np.kron(np.eye(3), speed*q)
    for degree in [1, 2]:
        generator[degree*n:(degree+1)*n, (degree-1)*n:degree*n] = degree*np.diag(f)
    initial = np.zeros(3*n)
    initial[:n] = 1.
    raw = (expm(duration*generator) @ initial).reshape(3, n) @ prior
    mean, variance = average*duration+raw[1], raw[2]-raw[1]**2
    assert mean > 0 and variance > 0
    shape, success = mean*mean/variance, mean/(mean+variance)
    assert abs(nbinom.mean(shape, success)-mean) <= ROUND_TOL*max(1., mean)
    assert abs(nbinom.var(shape, success)-mean-variance) <= ROUND_TOL*max(1., mean+variance)
    return nbinom.pmf(np.arange(SIZE), shape, success), nbinom.sf(SIZE-1, shape, success)


def duhamel_identity(q, intensity):
    pi, prior = stationary(q), np.array([0., 1., 0.])
    average = pi @ intensity
    f, projection = intensity-average, np.outer(np.ones(len(pi)), pi)
    worst = 0.
    for speed in [.5, 20.]:
        for duration in [.02, 1.]:
            for z in [.3, np.exp(.7j)]:
                generator = speed*q+(z-1)*np.diag(intensity)
                def integrand(t):
                    joint = prior @ expm(t*generator)
                    sigma = prior @ expm(speed*t*q)
                    centered = joint*f-(joint @ f)*sigma
                    return (z-1)*np.exp(average*(z-1)*(duration-t))*centered @ (
                        expm(speed*(duration-t)*q)-projection)
                integral, quadrature_error = quad_vec(integrand, 0., duration, epsabs=2e-12, epsrel=2e-10)
                joint = prior @ expm(duration*generator)
                residual = joint-joint.sum()*(prior @ expm(speed*duration*q))
                discrepancy = np.linalg.norm(residual-integral)
                assert discrepancy <= ROUND_TOL+2*quadrature_error
                worst = max(worst, discrepancy)
    return worst


def check(q, intensity, speed, lengths, prior, blocks, known=None):
    pi = stationary(q)
    c, gamma = known if known else envelope(q)
    fmax = max(abs(intensity-pi @ intensity))
    d = lambda t: min(2., 2*c*fmax*(-np.expm1(-speed*gamma*t))/(speed*gamma))
    e = lambda t: min(2., 4*c*fmax*(-np.expm1(-speed*gamma*t))/(speed*gamma))
    joint = conditional_start(blocks) @ prior
    allowance = ROUND_TOL*max(1., speed*np.linalg.norm(q, np.inf)*sum(lengths))
    assert joint.min() >= -allowance and abs(joint.sum()-1) <= allowance
    largest_identity_error = 0.
    for cut in range(1, len(lengths)+1):
        elapsed = sum(lengths[:cut])
        sigma = prior @ expm(speed*elapsed*q)
        past = end_joint(blocks[:cut], prior)
        past_law = past.sum(axis=-1)
        residual = past-past_law[..., None]*sigma
        assert np.sum(abs(residual)) <= e(elapsed)+allowance
        if cut < len(lengths):
            future = conditional_start(blocks[cut:])
            future_law = future @ sigma
            centered = np.tensordot(residual, future-(future @ pi)[..., None], axes=([-1], [-1]))
            direct = joint-np.multiply.outer(past_law, future_law)
            largest_identity_error = max(largest_identity_error, np.sum(abs(direct-centered)))
            assert .5*np.sum(abs(direct)) <= .5*e(elapsed)*d(sum(lengths[cut:]))+allowance
    marginals, approximations, marginal_errors, tails = [], [], [], []
    elapsed = 0.
    alias = sum(poisson.sf(SIZE-1, max(intensity)*t) for t in lengths)
    for t, block in zip(lengths, blocks):
        sigma = prior @ expm(speed*elapsed*q)
        marginal = np.einsum('i,aij->a', sigma, block)
        surrogate, tail = replacement(q, intensity, speed, t, sigma, pi)
        marginals.append(marginal)
        approximations.append(surrogate)
        marginal_errors.append(.5*np.sum(abs(marginal-surrogate))
                               +poisson.sf(SIZE-1, max(intensity)*t)+tail/2)
        tails.append(tail)
        elapsed += t
    measured = .5*np.sum(abs(joint-product(marginals)))
    bound = sum(.5*e(sum(lengths[:j]))*d(lengths[j]) for j in range(1, len(lengths)))
    assert measured+2*alias <= bound+allowance
    nb_error = .5*np.sum(abs(joint-product(approximations)))
    budget = alias+sum(tails)/2
    assert nb_error+budget <= bound+sum(marginal_errors)+allowance
    return measured, bound, nb_error, bound+sum(marginal_errors), largest_identity_error, alias, budget


def main():
    designs = [
        (np.array([[-1., 1.], [1., -1.]]), np.array([3., 1.]), (1., 2.)),
        (np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]]), np.array([2., 0., 1.]), None),
        (np.roll(np.eye(4), 1, axis=1)-np.eye(4), np.array([3., 0., 2., 1.]), None),
    ]
    cases = 0
    max_error = max_alias = max_nb_budget = 0.
    for index, (q, intensity, known) in enumerate(designs):
        pi = stationary(q)
        priors = list(np.eye(len(pi)))+[np.ones(len(pi))/len(pi), pi]
        for speed in [20., 80., 320.]:
            grids = [[.01, .1], [.1, 1.], [1., 3.], [.0001, 1.], [.2, .3, .5],
                     [1/speed, 1.], [1/speed**2, 1.], [1., 1.]]
            for lengths in grids:
                blocks = [kernels(q, intensity, speed, t) for t in lengths]
                for j, prior in enumerate(priors):
                    result = check(q, intensity, speed, lengths, prior, blocks, known)
                    max_error, max_alias, max_nb_budget = (max(x, y) for x, y in zip(
                        [max_error, max_alias, max_nb_budget], result[4:]))
                    if index == 1 and j == 1 and speed == 80. and lengths == [1/speed, 1.]:
                        print("defective chain, inactive-state start, speed 80, bins (1/80,1)")
                        print(f"TV to independent transient marginals {result[0]:.10f}; bound {result[1]:.10f}")
                        print(f"TV to independent prior-specific NB {result[2]:.10f}; triangle bound {result[3]:.10f}")
                    cases += 1
    for constant in [0., 2.]:
        q, _, known = designs[0]
        intensity, lengths = np.full(2, constant), [.1, 1.]
        blocks = [kernels(q, intensity, 20., t) for t in lengths]
        result = check(q, intensity, 20., lengths, np.array([1., 0.]), blocks, known)
        assert result[0] <= ROUND_TOL and result[2] <= ROUND_TOL
    assert max_error <= 10*ROUND_TOL
    integral_error = duhamel_identity(designs[1][0], designs[1][1])
    print("all-prior count-grid certificate; prior/chain/grid/speed cases", cases)
    print(f"maximum cut identity L1 discrepancy {max_error:.3e}; base tolerance {ROUND_TOL:.3e}")
    print(f"maximum Duhamel quadrature identity discrepancy {integral_error:.3e}")
    print(f"maximum exact-law modulo alias probability budget {max_alias:.3e}")
    print(f"maximum NB comparison truncation/alias TV budget {max_nb_budget:.3e}")
    print("constant and all-zero intensities are exact for every prior")
    print("PASS")


if __name__ == "__main__":
    main()
