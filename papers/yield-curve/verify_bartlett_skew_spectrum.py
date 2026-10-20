"""Exact leading coefficients and high-precision controls for the skew hierarchy.

The all-r proof and finite-T exterior-invariant bound are in the site note.
Numerical convergence checks supplement that proof, not interval enclosures.
"""
from fractions import Fraction as R
from math import comb, factorial

import mpmath as mp
import numpy as np
from scipy.linalg import expm


def determinant(A):
    B = [[R(x) for x in row] for row in A]
    value = R(1)
    for j in range(len(B)):
        p = next(i for i in range(j, len(B)) if B[i][j])
        if p != j:
            B[j], B[p] = B[p], B[j]
            value = -value
        pivot = B[j][j]
        value *= pivot
        for i in range(j+1, len(B)):
            factor = B[i][j]/pivot
            B[i] = [x-factor*y for x, y in zip(B[i], B[j])]
    return value


def family(r):
    n = 2*r+1
    Q = np.zeros((n, n), dtype=object)
    for i, j in ((0, r), (r, 2*r), (2*r, 0)):
        Q[i, j] = 1
    for root in (0, r):
        for i in range(root, root+r-1):
            Q[i, i+1] = Q[i+1, i] = 1
    for i in range(n):
        Q[i, i] = -sum(Q[i])
    return Q


def modes(A):
    values = mp.eigsy(-A*A, eigvals_only=True)
    return [mp.sqrt(values[len(values)-1-2*j]) for j in range((len(values)-1)//2)]


def evaluate(Q, T, degree=100):
    n = len(Q)
    q = mp.matrix(Q.tolist())
    power, M, coefficient = mp.eye(n), T*mp.eye(n)/2, T/2
    for k in range(1, degree+1):
        power = power*q
        coefficient *= T/(k+2)
        M += coefficient*power
    A = (M-M.T)/2
    # The extra stationary projector makes the full symmetric matrix invertible.
    S = (M+M.T)/2+mp.ones(n)/n
    d, V = mp.eigsy(S)
    W = V*mp.diag([1/mp.sqrt(x) for x in d])*V.T
    return modes(A), modes(W*A*W), M


def main():
    mp.mp.dps = 120
    print('120 decimal digits; degree=100; T=1/16,1/32,1/64; entry bounds 54 T and 55')
    for r in range(1, 7):
        Q = family(r)
        assert not any(Q.sum(axis=0)) and not any(Q.sum(axis=1))
        power = np.eye(len(Q), dtype=object)
        for k in range(2*r):
            assert np.array_equal(power[:r, :r], power[:r, :r].T)
            assert np.array_equal(power[r:2*r, r:2*r], power[r:2*r, r:2*r].T)
            delta = power[:r, r:2*r]-power[r:2*r, :r].T
            for i in range(r):
                for j in range(r):
                    if k <= i+j+1:
                        assert delta[i, j] == int(k == i+j+1)
            power = power@Q
        constants, deltas, errors = [], [R(1)], []
        for j in range(1, r+1):
            C = [[R(1, 2*factorial(i+k+3)) for k in range(j)] for i in range(j)]
            delta = determinant(C)
            formula = R((-1)**(j*(j-1)//2), 2**j)
            for k in range(j):
                formula *= R(factorial(k), factorial(j+k+2))
            beta = R(factorial(j-1)*factorial(j+1), 2*factorial(2*j)*factorial(2*j+1))
            assert delta == formula and abs(delta/deltas[-1]) == beta
            assert determinant([[1+int(i == k) for k in range(2*j)] for i in range(2*j)]) == 2*j+1
            a = mp.sqrt(mp.mpf(2*j+1)/(2*j-1))*mp.mpf(beta.numerator)/beta.denominator
            constants.append(a)
            deltas.append(delta)
            E = factorial(j)*(54*j*55**(j-1)+(comb(r, j)**2-1)*55**j)
            errors.append(E)
        results = [evaluate(Q, mp.mpf(1)/d) for d in (16, 32, 64)]
        raw_ratios, sector_ratios, orders = [], [], []
        for j, a in enumerate(constants):
            raw = [v[0][j]/(a*(mp.mpf(1)/d)**(2*j+2)) for v, d in zip(results, (16,32,64))]
            sector = results[-1][1][j]/(2*a*(mp.mpf(1)/64)**(2*j+1))
            order = mp.log(abs(1-raw[-2])/abs(1-raw[-1]), 2)
            assert abs(raw[-1]-1) < mp.mpf('.04') and mp.mpf('.9') < order < mp.mpf('1.1')
            assert abs(sector-1) < mp.mpf('.05')
            raw_ratios.append(float(raw[-1])); sector_ratios.append(float(sector)); orders.append(float(order))
            # Exterior-invariant identity and the explicit finite-T error bound.
            T = mp.mpf(1)/64
            squared = [x*x for x in results[-1][0]]
            elementary = [mp.mpf(1)]+[mp.mpf(0)]*r
            for x in squared:
                for k in range(r, 0, -1):
                    elementary[k] += x*elementary[k-1]
            delta = deltas[j+1]
            lead = mp.mpf(abs(delta.numerator))/delta.denominator
            b = mp.sqrt(elementary[j+1]/(2*j+3))/T**((j+1)*(j+2))
            assert abs(b-lead) <= errors[j]*T
        print(f'r={r} n={2*r+1}; a_j={[float(a) for a in constants]}')
        print('raw ratios at T=1/64:', raw_ratios, '; first-order remainder slopes:', orders)
        print('normalized-sector ratios at T=1/64:', sector_ratios)
        if r == 6:
            print(f'13-state T=1/64: largest={float(results[-1][0][0]):.12g}; weakest={float(results[-1][0][-1]):.12g}')
            print(f'weakest normalized={float(results[-1][1][-1]):.12g}; exact leading a_6={float(constants[-1]):.12g}')
            M = results[-1][2]
            A = (M-M.T)/2
            _, U = mp.eigsy(-A*A)
            P = U[:, 1:3]*U[:, 1:3].T
            low_rank = (mp.eye(len(Q))-P)*A*(mp.eye(len(Q))-P)
            error = A-low_rank
            distance = modes(error)[0]
            assert abs(distance/results[-1][0][-1]-1) < mp.mpf('1e-20')
            print('canonical weakest-pair removal: spectral distance / weakest mode =', float(distance/results[-1][0][-1]))
            with mp.workdps(170):
                repeated = evaluate(Q, mp.mpf(1)/64)
                assert abs(repeated[0][-1]/results[-1][0][-1]-1) < mp.mpf('1e-20')
        # Independent double-precision value check at a moderate horizon.
        n = len(Q); block = np.zeros((3*n, 3*n))
        block[:n, :n] = np.array(Q, float)
        block[:n, n:2*n] = block[n:2*n, 2*n:] = np.eye(n)
        M = expm(block)[:n, 2*n:]
        _, _, exact = evaluate(Q, mp.mpf(1))
        assert np.linalg.norm(M-np.array(exact.tolist(), float), np.inf) < 3e-8*(1+np.linalg.norm(M, np.inf))
    print('PASS')


if __name__ == '__main__':
    main()
