"""Finite capped Poisson tags: local clock/speed aliases with fixed moments.

Run with mpmath installed. Closed-form truncated exponential moments build
the perturbation; independent quadrature checks the entire observed law.
"""
import mpmath as mp


def check(condition):
    if not condition:
        raise AssertionError("Finite Poisson-tag certificate failed")


def basis(cap, moments, speed, tick):
    return ([(j, mp.mpf(0)) for j in range(moments+1)]
            + [(j, tick) for j in range(cap)] + [(0, speed)]
            + [(j, tick+speed) for j in range(cap)])


def truncated(power, rate, length):
    z = rate*length
    return mp.factorial(power)/rate**(power+1)*(1-mp.exp(-z)*sum(
        z**j/mp.factorial(j) for j in range(power+1)))


def bump(t, length):
    return 16*t*t*(length-t)**2/length**4


def fit(cap, moments, shift, length=8):
    # Baseline Gamma(2,1), c0=2, known auxiliary tick rate s=1.
    length = mp.mpf(length)
    c0, tick = mp.mpf(2), mp.mpf(1)
    fixed = basis(cap, moments, c0, tick)
    moving = basis(cap, moments, c0+shift, tick)
    target = mp.matrix([mp.factorial(j+1)/(1+a)**(j+2) for j, a in fixed])
    original = mp.matrix([mp.factorial(j+1)/(1+a)**(j+2) for j, a in moving])
    def entry(left, right):
        j, a = left
        k, b = right
        return 16/length**4*sum(coef*truncated(j+k+power+1, 1+a+b, length)
                              for power, coef in ((2, length**2), (3, -2*length), (4, 1)))
    matrix = mp.matrix([[entry(left, right) for right in fixed] for left in moving])
    weights = mp.lu_solve(matrix, target-original)
    # 0<=bump<=1; t^j exp(-a t)<=length^j on its support.
    bound = sum(abs(weights[i])*length**j for i, (j, _) in enumerate(fixed))
    check(bound < mp.mpf('.25'))
    def density(t):
        tilt = sum(weights[i]*t**j*mp.exp(-a*t) for i, (j, a) in enumerate(fixed))
        return t*mp.exp(-t)*(1+(bump(t, length)*tilt if 0 <= t <= length else 0))
    def integrate(fun):
        return mp.quad(lambda t: fun(t)*density(t), [0, length/4, length/2, length, mp.inf])
    moment_error = max(abs(integrate(lambda t, j=j: t**j)-mp.factorial(j+1))
                       for j in range(moments+1))
    law_error = mp.mpf(0)
    # Direct quadrature of all (Y0,YT,J), uniform Y0; include the capped tail.
    for tag in range(cap+1):
        def tag_weight(t):
            if tag < cap:
                return mp.exp(-t)*t**tag/mp.factorial(tag)
            return 1-mp.exp(-t)*sum(t**j/mp.factorial(j) for j in range(cap))
        p = integrate(tag_weight)
        v = integrate(lambda t: tag_weight(t)*mp.exp(-(c0+shift)*t))
        if tag < cap:
            p0 = mp.mpf(tag+1)/2**(tag+2)
            v0 = mp.mpf(tag+1)/4**(tag+2)
        else:
            p0 = 1-sum(mp.mpf(j+1)/2**(j+2) for j in range(cap))
            v0 = mp.mpf(1)/9-sum(mp.mpf(j+1)/4**(j+2) for j in range(cap))
        for start in range(2):
            for end in range(2):
                sign = 1 if start == end else -1
                law_error = max(law_error, abs((p+sign*v-p0-sign*v0)/4))
    check(max(moment_error, law_error) < mp.mpf('1e-12'))
    return bound, moment_error, law_error


def uncapped():
    cases = [mp.matrix([[-2, 2, 0], [0, -2, 2], [2, 0, -2]]),
             mp.matrix([[-1, 1, 0], [0, -1, 1], [0, 0, 0]]),
             mp.matrix([[-3, 1, 2], [0, 0, 0], [0, 0, 0]])]
    times, weights = (mp.mpf(1), mp.sqrt(2)), (mp.mpf(1)/3, mp.mpf(2)/3)
    error = mp.mpf(0)
    for q in cases:
        kernels = [mp.expm(t*q) for t in times]
        for tick in (mp.mpf('.7'), mp.mpf(2)):
            rk = [sum((w*mp.exp(-tick*t)*(tick*t)**k/mp.factorial(k)*a
                       for t, w, a in zip(times, weights, kernels)), mp.zeros(3))
                  for k in range(71)]
            tail = sum(w*mp.gammainc(71, 0, tick*t)/mp.factorial(70)
                       for t, w in zip(times, weights))
            for z in (mp.mpf('.1'), mp.mpf('.4'), mp.mpf('.8')):
                observed = sum((z**k*a for k, a in enumerate(rk)), mp.zeros(3))
                direct = sum((w*mp.exp(-tick*(1-z)*t)*a
                              for t, w, a in zip(times, weights, kernels)), mp.zeros(3))
                delta = mp.mnorm(observed-direct, mp.inf)
                check(delta <= tail+mp.mpf('1e-12'))
                error = max(error, delta)
    print(f"18 uncapped matrix PGF checks (directed, Jordan, absorbing): error {float(error):.3e}")
    # Infinite-mean clock f(t)=1/t^2, t>=1: no raw-moment assumption needed.
    for z in (mp.mpf('.2'), mp.mpf('.5'), mp.mpf('.8')):
        counts = [mp.gammainc(k-1, 1, mp.inf)/mp.factorial(k) for k in range(71)]
        observed = sum(z**k*p for k, p in enumerate(counts))
        direct = mp.exp(z-1)-(1-z)*mp.e1(1-z)
        check(abs(observed-direct) <= z**71+mp.mpf('1e-12'))
    print("3 infinite-mean Pareto clock PGFs: explicit omitted-count bound passed")
    p, eye = mp.ones(3)/3, mp.eye(3)
    e = eye-p
    j = mp.matrix([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])/mp.sqrt(3)
    q0, q1 = -4*mp.pi*e, -4*mp.pi*e+2*mp.pi*j
    check(min(q1[a, b] for a in range(3) for b in range(3) if a != b) > 0)
    gap = max(mp.mnorm(mp.expm(t*q0)-mp.expm(t*q1), mp.inf) for t in (1, 2, 3))
    check(gap < mp.mpf('1e-12'))
    off_grid = mp.mnorm(mp.expm(mp.sqrt(2)*q0)-mp.expm(mp.sqrt(2)*q1), mp.inf)
    check(off_grid > mp.mpf('1e-10'))
    print(f"Grid alias kernel error {float(gap):.3e}; sqrt(2)-time gap {float(off_grid):.6e}")


def main():
    mp.mp.dps = 85
    maximum = [mp.mpf(0)]*3
    cases = 0
    for cap, moments, delta in ((1, 0, '1e-5'), (2, 2, '1e-9'), (3, 3, '1e-13')):
        for sign in (-1, 1):
            # Small shifts keep the analytically bounded density ratio positive.
            result = fit(cap, moments, sign*mp.mpf(delta))
            maximum = [max(a, b) for a, b in zip(maximum, result)]
            cases += 1
    print(f"{cases} aliases, caps 1/2/3, preserved raw moments through 0/2/3")
    print("Speed shifts: +/-1e-5, +/-1e-9, +/-1e-13 about c0=2; tick rate s=1")
    print(f"Global relative-density perturbation bound: {float(maximum[0]):.6e}")
    print(f"Independent moment quadrature error: {float(maximum[1]):.3e}")
    print(f"Independent full endpoint/tag-law error: {float(maximum[2]):.3e}")
    # Collision c=s: strict decrease of L(c) forces c=s when L(c)=p(J=0).
    check(mp.mpf(1)/4 == 1/(1+mp.mpf(1))**2)
    print("At the exceptional speed c=s=1, L(c)=p(J=0)=1/4")
    uncapped()
    print("PASS")


if __name__ == "__main__":
    main()
