"""A rational negative hard cutoff with simple entirely real generator spectrum.

No floating eigenvalue or matrix-exponential output establishes either claim.
The proof and error bounds are in tools/pages/real-spectrum-cutoff.html.
"""
from fractions import Fraction as R
from math import factorial

import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import expm


def main():
    m, degree = 22, 160
    T, eps, h = R(33, 2), R(1, 10**100), R(1, 3000)
    rates = [R(1001+i, 1000) for i in range(m)]
    g = [29, 29, 25, 19, 11, 2, -7, -15, -21, -26, -28,
         -28, -26, -21, -15, -7, 2, 11, 19, 25, 29, 29]
    S = sum(R(g[i])/rates[i] for i in range(m))
    A = sum(abs(R(g[i])/rates[i]) for i in range(m))
    M = max(map(abs, g))
    c = 1/(1/eps+sum(1/q for q in rates))
    f = [-eps*S]+list(map(R, g))
    pi = [c/eps]+[c/q for q in rates]
    Q = [[R(0) for _ in range(m+1)] for _ in range(m+1)]
    for i, q in enumerate([eps]+rates):
        Q[i][i], Q[i][(i+1) % (m+1)] = -q, q
    assert sum(pi) == 1 and sum(pi[i]*f[i] for i in range(m+1)) == 0
    assert all(sum(pi[i]*Q[i][j] for i in range(m+1)) == 0 for j in range(m+1))
    # Characteristic polynomial, exact sign changes in 22 disjoint intervals.
    def characteristic(z):
        value = z+eps
        product = eps
        for q in rates:
            value *= z+q
            product *= q
        return value-product
    assert characteristic(0) == 0
    ratio = eps*2**m/(h**m/2)
    assert ratio < 1
    for i, q in enumerate(rates):
        left, right = -q-h, -q+h
        assert right < 0 and abs(left+eps) > R(1, 2)
        assert abs(right+eps) > R(1, 2)
        assert characteristic(left)*characteristic(right) < 0
        if i:
            assert right < -rates[i-1]-h
    # Baseline killed path: integer powers of 1000 J, not rational full Q powers.
    vector = g.copy()
    polynomial = R(0)
    for k in range(degree+1):
        coefficient = sum(R(1000*g[i]*vector[i], 1001+i) for i in range(m))
        polynomial += T**(k+1)*coefficient/R(factorial(k+1)*1000**k)
        vector = [-(1001+i)*vector[i]+((1001+i)*vector[i+1] if i<m-1 else 0)
                  for i in range(m)]
    x = 2*max(rates)*T
    tail = A*M*T*x**(degree+1)/factorial(degree+2)/(1-x/(degree+3))
    # Markov infinity-norm contraction and Duhamel control the rare return.
    perturbation = eps*(abs(S)+A)*(T*abs(S)+T*T*M)
    assert -104 < polynomial-tail-perturbation
    assert polynomial+tail+perturbation < -103
    print(f'n={m+1}; m={m}; T={T}; epsilon=1e-100; q_i=1+i/1000 (i=1,...,22)')
    print('g=', g)
    print(f'root interval radius={h}; real nonzero simple roots={m}; root budget ratio={float(ratio):.12g}')
    print(f'degree={degree}; baseline={float(polynomial):.12g}; Taylor tail={float(tail):.12g}')
    print(f'S={float(S):.12g}; A={float(A):.12g}; M={M}; perturbation={float(perturbation):.12g}')
    print('certified full-chain scaled cutoff: -104 < R_T,f/c < -103, c=(1/epsilon+sum(1/q_i))^-1')
    # Independent floating value checks; no eigenvalue test on this ill-conditioned chain.
    q = np.array(rates, dtype=float)
    J = np.diag(-q)
    J[np.arange(m-1), np.arange(1, m)] = q[:-1]
    gf = np.array(g, dtype=float)
    weights = gf/q
    Z = np.zeros_like(J)
    integral = expm(float(T)*np.block([[J, np.eye(m)], [Z, Z]]))[:m, m:]
    baseline = weights@integral@gf
    assert abs(baseline-float(polynomial)) < 3e-8*(1+A*M*float(T))
    quadrature, _ = quad_vec(lambda t: weights@expm(t*J)@gf, 0, float(T), epsabs=1e-7)
    assert abs(quadrature-float(polynomial)) < 3e-8*(1+float(A)*M*float(T))
    qf = np.array(Q, dtype=float)
    ff = np.array(f, dtype=float)
    scaled_weights = np.array([float(-S)]+[float(R(g[i])/rates[i]) for i in range(m)])
    block = np.zeros((2*(m+1), 2*(m+1)))
    block[:m+1, :m+1], block[:m+1, m+1:] = qf, np.eye(m+1)
    full = scaled_weights@expm(float(T)*block)[:m+1, m+1:]@ff
    assert abs(full-float(polynomial)) < 3e-8*(1+float(A)*M*float(T))
    print(f'independent block/quadrature/full scaled values: {baseline:.12g}, {quadrature:.12g}, {full:.12g}')
    print('PASS')


if __name__ == '__main__':
    main()
