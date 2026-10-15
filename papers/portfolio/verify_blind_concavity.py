"""Exact constant-fraction curvature, optimizer certificates and nonconcavity."""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq

from portfolio import blind_weights, stationary


ROUND = 65536*np.finfo(float).eps


def evaluate(q, prior, r, mu, sigma, gamma, duration, weight):
    """Tilted path moments and an independent differentiated FK system."""
    n = len(prior)
    variance = np.asarray(sigma)**2
    h = r+weight*(mu-r)-gamma*weight**2*variance/2
    reward = mu-r-gamma*weight*variance
    potential = (1-gamma)*h
    shift = max(potential)
    a = q+np.diag(potential-shift)
    moment = np.kron(np.eye(4), a)
    moment[n:2*n, :n] = np.diag(reward)
    moment[2*n:3*n, n:2*n] = 2*np.diag(reward)
    moment[3*n:, :n] = np.diag(variance)
    initial = np.zeros(4*n)
    initial[:n] = 1.
    values = (expm(duration*moment) @ initial).reshape(4, n) @ prior
    z, first, second, integrated_var = values
    tilted_variance = second/z-(first/z)**2
    gradient = first/(z*duration)
    curvature = (-gamma*integrated_var/z+(1-gamma)*tilted_variance)/duration
    if gamma == 1:
        unweighted = np.kron(np.eye(2), q)
        unweighted[n:, :n] = np.diag(h)
        ce = prior @ (expm(duration*unweighted) @ initial[:2*n])[n:]/duration
        return ce, gradient, curvature, 0.
    ce = (np.log(z)+duration*shift)/((1-gamma)*duration)
    differentiated = np.kron(np.eye(3), a)
    differentiated[n:2*n, :n] = (1-gamma)*np.diag(reward)
    differentiated[2*n:, n:2*n] = 2*(1-gamma)*np.diag(reward)
    differentiated[2*n:, :n] = -gamma*(1-gamma)*np.diag(variance)
    d0, d1, d2 = (expm(duration*differentiated) @ initial[:3*n]).reshape(3, n) @ prior
    direct_gradient = d1/(d0*(1-gamma)*duration)
    direct_curvature = (d2/d0-(d1/d0)**2)/((1-gamma)*duration)
    discrepancy = max(abs(gradient-direct_gradient), abs(curvature-direct_curvature))
    return ce, gradient, curvature, discrepancy


def main():
    rng = np.random.default_rng(617)
    dense = rng.uniform(.1, 1., (5, 5))
    np.fill_diagonal(dense, 0.)
    dense -= np.diag(dense.sum(axis=1))
    designs = [np.array([[-3., 3.], [9., -9.]]),
               np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]]),
               np.roll(np.eye(4), 1, axis=1)-np.eye(4), dense]
    cases = 0
    largest = largest_fd = 0.
    for q in designs:
        n = len(q)
        mu = np.linspace(-.15, .25, n)
        sigma = np.linspace(.12, .3, n)
        priors = [*np.eye(n), np.ones(n)/n, stationary(q)]
        for prior in priors:
            for speed in [.1, 1., 10.]:
                for gamma in [1., 1.1, 3.]:
                    for duration in [.2, 2.]:
                        for weight in [-.5, .7, 2.]:
                            fun = lambda w: evaluate(speed*q, prior, .03, mu, sigma, gamma, duration, w)
                            ce, derivative, curvature, error = fun(weight)
                            scale = max(1., speed*np.linalg.norm(q, np.inf)*duration,
                                        abs(ce), abs(derivative), abs(curvature))
                            assert error <= ROUND*scale
                            assert curvature <= -gamma*min(sigma**2)+ROUND*scale
                            # Independent central difference with a truncation-sized allowance.
                            step = 2e-4
                            fd = (fun(weight+step)[0]-fun(weight-step)[0])/(2*step)
                            fd_error = abs(fd-derivative)
                            assert fd_error <= 2e-6*scale
                            largest = max(largest, error/scale)
                            largest_fd = max(largest_fd, fd_error/scale)
                            cases += 1
    print("exact strong-concavity cases", cases)
    print(f"maximum normalized FK derivative discrepancy {largest:.3e}; tolerance {ROUND:.3e}")
    print(f"maximum normalized finite-difference discrepancy {largest_fd:.3e}; allowance 2.000e-06")
    for speed in [.1, 1., 10.]:
        for gamma in [.5, 1., 3.]:
            values = evaluate(speed*designs[1], np.array([0., 1., 0.]), .03,
                              np.full(3, .1), np.full(3, .2), gamma, 2., .7)
            expected = [.03+.7*.07-gamma*.7**2*.04/2, .07-gamma*.7*.04, -gamma*.04]
            assert max(abs(np.array(values[:3])-expected)) <= ROUND*max(1., speed)
    print("constant-coefficient exactness: 9 speed/risk-aversion cases")
    q = designs[0]
    prior = stationary(q)
    mu, sigma = np.array([.15, -.05]), np.array([.12, .3])
    for speed in [.25, 1., 4.]:
        fun = lambda w: evaluate(speed*q, prior, .03, mu, sigma, 3., 5., w)
        w0, correction = blind_weights(speed*q, .03, mu, sigma, 3.)
        candidate = w0+correction
        optimum = brentq(lambda w: fun(w)[1], -3., 6., xtol=1e-12)
        ce, derivative, _, _ = fun(candidate)
        kappa = 3*min(sigma**2)
        distance_bound, gap_bound = abs(derivative)/kappa, derivative**2/(2*kappa)
        assert abs(candidate-optimum) <= distance_bound+ROUND
        assert 0 <= fun(optimum)[0]-ce+ROUND <= gap_bound+2*ROUND
        print(f"speed {speed:g}: first-corrected fraction {candidate:.10f}; optimum {optimum:.10f}")
        print(f"fraction error {abs(candidate-optimum):.10f}; residual bound {distance_bound:.10f}; CE gap bound {gap_bound:.10f}")
    # Strictly concave CRRA utility need not give a concave constant-fraction map.
    q = np.array([[-1., 1.], [1., -1.]])
    fun = lambda w: evaluate(q, np.array([.5, .5]), 0., np.array([2., -2.]), np.ones(2), .5, 1., w)
    zero, derivative, curvature, _ = fun(0.)
    predicted = -.5+2*(1-(1-np.exp(-2))/2)
    assert abs(curvature-predicted) <= ROUND and abs(derivative) <= ROUND
    assert curvature > .6 and fun(1.)[0] > zero+.25
    assert abs(fun(1.)[0]-fun(-1.)[0]) <= ROUND
    positive = brentq(lambda w: fun(w)[1], 1., 6., xtol=1e-12)
    assert fun(positive)[2] < 0
    print(f"gamma 1/2 example: CE(0) {zero:.10f}; CE(+/-1) {fun(1.)[0]:.10f}; CE''(0) {curvature:.10f}")
    print(f"positive local maximizer {positive:.10f}; CE {fun(positive)[0]:.10f}")
    print("PASS")


if __name__ == "__main__":
    main()
