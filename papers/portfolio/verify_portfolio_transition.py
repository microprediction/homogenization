"""Exact symmetric portfolio transition via a Robin--Dirichlet resolvent."""

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from scipy.integrate import quad


ROUND = 65536*np.finfo(float).eps


def closed(speed, duration, tilt):
    x, y, t = speed*duration, tilt*duration, duration
    d = np.hypot(x, y)
    log_moment = d-x+np.log(((1+x/d)+(1-x/d)*np.exp(-2*d))/2)
    if tilt == 0:
        response = t*t*(1/x+np.expm1(-2*x)/(2*x*x))
    else:
        response = t*t*(np.tanh(d)/d+x/d**2-x*np.tanh(d)/d**3)/(1+x*np.tanh(d)/d)
    return float(log_moment), float(response)


def matrix(speed, duration, tilt):
    q = speed*np.array([[-1., 1.], [1., -1.]])
    shift = np.hypot(speed, tilt)-speed
    a = q+np.diag([tilt-shift, -tilt-shift])
    system = np.kron(np.eye(3), a)
    system[2:4, :2] = np.diag([1., -1.])
    system[4:, 2:4] = 2*np.diag([1., -1.])
    z, first, second = (expm(duration*system) @ np.array([1., 1., 0., 0., 0., 0.])).reshape(3, 2).mean(axis=1)
    return np.log(z)+duration*shift, first/z, second/z-(first/z)**2


def main():
    size, cases, largest = 512, 0, 0.
    largest_tail = 0.
    for x in [.02, .2, 1., 5., 20.]:
        roots = np.array([brentq(lambda u: np.cos(u)+x*np.sin(u)/u,
                                 (j+.5)*np.pi, (j+1)*np.pi, xtol=2e-12)
                          for j in range(size)])
        eigenvalues = x*x+roots**2
        for duration in [.25, 1., 3.]:
            speed = x/duration
            for y in [0., .2, 1., 3., 10.]:
                tilt = y/duration
                log_moment, response = closed(speed, duration, tilt)
                log_matrix, first, second = matrix(speed, duration, tilt)
                observed = second if tilt == 0 else first/tilt
                scale = max(1., duration**2, abs(log_moment), response)
                error = max(abs(log_matrix-log_moment), abs(observed-response))/scale
                assert error <= ROUND
                assert second <= closed(speed, duration, 0.)[1]+ROUND*scale
                d = np.hypot(x, y)
                green_trace, quadrature_error = quad(
                    lambda u: 2*duration**2*(np.cosh(d*u)+x*np.sinh(d*u)/d)
                    *np.sinh(d*(1-u))/(d*(np.cosh(d)+x*np.sinh(d)/d)), 0., 1., epsabs=1e-11)
                assert abs(green_trace-response) <= ROUND*scale+4*quadrature_error
                partial = 2*duration**2*np.sum(1/(eigenvalues+y*y))
                tail = 2*duration**2/(np.pi**2*(size-.5))
                assert partial <= response+ROUND*scale
                assert response <= partial+tail+ROUND*scale
                largest = max(largest, error)
                largest_tail = max(largest_tail, tail)
                cases += 1
    print("transform/response/spectral-tail cases", cases)
    print(f"maximum normalized closed-versus-matrix discrepancy {largest:.3e}; tolerance {ROUND:.3e}")
    print(f"spectral terms {size}; largest analytic omitted-tail bound {largest_tail:.10f}")
    classified = split = 0
    for gamma in [.25, .5, .8]:
        for amplitude in [1., 2.]:
            for duration in [.25, 1., 3.]:
                last_optimum = np.inf
                for speed in [.1, 1., 5., 20.]:
                    variance = closed(speed, duration, 0.)[1]
                    criterion = -gamma+(1-gamma)*amplitude**2*variance/duration
                    def derivative_over_weight(w):
                        tilt = (1-gamma)*amplitude*w
                        return -gamma+(1-gamma)*amplitude**2*closed(speed, duration, tilt)[1]/duration
                    grid = np.linspace(0., 2*amplitude/gamma, 101)
                    values = np.array([derivative_over_weight(w) for w in grid])
                    assert np.all(np.diff(values) <= ROUND*max(1., abs(criterion)))
                    if criterion > 0:
                        optimum = brentq(derivative_over_weight, 0., grid[-1], xtol=1e-12)
                        assert derivative_over_weight(optimum*.9) > 0
                        assert derivative_over_weight(optimum*1.1) < 0
                        split += 1
                    else:
                        assert np.max(values) <= ROUND
                        optimum = 0.
                    assert optimum <= last_optimum+ROUND*max(1., optimum)
                    last_optimum = optimum
                    classified += 1
    print("portfolio parameter classifications", classified, "; two-optimum cases", split)
    critical_float = brentq(lambda m: 2*(1/m+np.expm1(-2*m)/(2*m*m))-.5, 3., 4., xtol=1e-12)
    assert abs(2*closed(critical_float, 1., 0.)[1]-.5) <= ROUND
    print(f"gamma 1/2, amplitude 2, variance 1, horizon 1: critical speed {critical_float:.10f}")
    for speed in [1., 3., critical_float, 4.]:
        criterion = 2*closed(speed, 1., 0.)[1]-.5
        optimum = brentq(lambda w: 2*closed(speed, 1., w)[1]-.5, 0., 8., xtol=1e-12) if criterion > ROUND else 0.
        value = -optimum**2/4+2*closed(speed, 1., optimum)[0]
        print(f"speed {speed:.10f}: curvature at zero {criterion:.10f}; nonnegative optimum {optimum:.10f}; CE {value:.10f}")
    print("PASS")


if __name__ == "__main__":
    main()
