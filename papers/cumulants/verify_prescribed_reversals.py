"""Prescribed maturity reversals: exact rank minors, CIR margins and Taylor signs."""
from fractions import Fraction as F
from math import ceil, factorial, prod

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from polymoments import Poly, cir_generator, switched_expectation


def eye(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def multiply(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def add(a, b):
    return [[x+y for x, y in zip(u, v)] for u, v in zip(a, b)]


def determinant(a):
    a, answer = [row[:] for row in a], F(1)
    for j in range(len(a)):
        pivot = next((i for i in range(j, len(a)) if a[i][j]), None)
        if pivot is None:
            return F(0)
        if pivot != j:
            a[j], a[pivot], answer = a[pivot], a[j], -answer
        value = a[j][j]
        answer *= value
        for i in range(j+1, len(a)):
            factor = a[i][j]/value
            a[i] = [x-factor*y for x, y in zip(a[i], a[j])]
    return answer


def coefficients(matrix):
    n = len(matrix)
    return [(sum(matrix[i])-sum(row[i] for row in matrix))/n for i in range(n)]


def rank_minors(n):
    cycle = [[F(j == (i+1) % n)-F(i == j) for j in range(n)] for i in range(n)]
    diagonal = [[-F(i+3)*(i == j) for j in range(n)] for i in range(n)]
    killed = add(diagonal, [[x/10 for x in row] for row in cycle])
    power, base, derivative = eye(n), eye(n), [[F(0)]*n for i in range(n)]
    actual, leading = [], []
    for j in range(1, n):
        power = multiply(power, killed)
        derivative = add(multiply(derivative, diagonal), multiply(base, cycle))
        base = multiply(base, diagonal)
        if j >= 2:
            actual.append(coefficients(power)[1:n-1])
            leading.append(coefficients(derivative)[1:n-1])
            weights = coefficients(power)
            assert sum(weights) == sum(w*(i+3) for i, w in enumerate(weights)) == 0
    expected = F((-1)**((n-2)*(n-1)//2)*factorial(n-2)*factorial(n-1)
                 *prod(factorial(j) for j in range(1, n-2)), n**(n-2))
    assert determinant(leading) == expected and determinant(actual) != 0
    return cycle, killed, determinant(actual), expected


def rational_mean(generator, a, k, t, degree=160):
    """Use centered a and v0=0: both removed common contributions cancel exactly."""
    n = len(a)
    matrix = [[F(0)]*(n+2) for i in range(n+2)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = generator[j][i]-(k[i] if i == j else 0)
        matrix[i][n], matrix[-1][i] = a[i]/n, F(1)
    matrix = [[t*x for x in row] for row in matrix]
    bound = ceil(max(sum(map(abs, row)) for row in matrix))
    term = [F(0)]*n+[F(1), F(0)]
    total = term[:]
    for j in range(1, degree+1):
        term = [sum(x*y for x, y in zip(row, term))/j for row in matrix]
        total = [x+y for x, y in zip(total, term)]
    return total[-1]/2, F(3**bound*bound**(degree+1), 2*factorial(degree+1))


def independent_values(q, a, k, t):
    n, pi = len(a), np.ones(len(a))/len(a)
    block = np.zeros((3*n, 3*n))
    block[:n, :n] = q-np.diag(k)
    block[n:2*n, :n], block[2*n:, n:2*n] = np.eye(n), np.eye(n)
    integral = expm(t*block)[2*n:, :n]
    direct = ((pi*a)@integral@np.ones(n)-pi@integral@a)/2
    p = Poly(1)
    physical_a = 60+a
    operators = [cir_generator(p, ki, ai/ki, xi, 0)
                 for ai, ki, xi in zip(physical_a, k, 2*np.arange(1, n+1))]
    mean = lambda generator: switched_expectation(generator, operators, t,
                               p.monomial(b=1), p, .04, pi)/2
    return direct, mean(q)-mean(q.T)


def main():
    for n in range(3, 11):
        cycle, killed, minor, leading = rank_minors(n)
        characteristic = lambda z: prod(z+i+3+F(1, 10) for i in range(n))-F(1, 10)**n
        assert all(characteristic(-F(i+3)-F(1, 5))*characteristic(-F(i+3)) < 0 for i in range(n))
        delta = F(1, 4*n*n)
        upper = F(-8*n*n+2*n+1, 32*n**4)
        assert upper < 0
        for s in (-delta, 1+delta):
            assert max(F(2*(i+1))**2*delta**2/2
                       +(-F(i+1)*s-2)*delta+(s*s-s)/2 for i in range(n)) <= upper
        print(f'n={n}: exact rank {n-2}; leading minor coefficient {leading}; CIR upper bound {upper}')
        if n == 5:
            print(f'Five-state minor at epsilon=1/10: {minor}')
    n = 5
    cycle, killed, _, _ = rank_minors(n)
    q = [[x/10 for x in row] for row in cycle]
    reverse = list(map(list, zip(*q)))
    k = list(map(F, range(3, 8)))
    a = list(map(F, ('0', '-.335', '1', '-.759', '0')))
    xi = list(map(F, (2, 4, 6, 8, 10)))
    margin = min(2*(60+ai)-x*x for ai, x in zip(a, xi))
    assert margin == 20
    assert all(k[i] == 2+xi[i]/2 for i in range(n))
    print(f'Fixture a={list(map(float, [60+x for x in a]))}; strict Feller margin {margin}')
    print('k=(3,4,5,6,7), xi=(2,4,6,8,10), kappa=2, rho=-1/2, v0=.04')
    print('Five disjoint rational spectral intervals: centers -3.1,-4.1,-5.1,-6.1,-7.1; radius .1')
    print('Degree-160 rational actions; each directional tail <= 3^L L^161/(2*161!)')
    qf, af, kf = np.array(q, float), np.array(a, float), np.array(k, float)
    maximum = 0.
    for t in (F(1, 2), F(1), F(2)):
        v, e = rational_mean(q, a, k, t)
        w, f = rational_mean(reverse, a, k, t)
        value, error = v-w, e+f
        scaled = value*10**12
        lower = F(scaled.numerator//scaled.denominator, 10**12)
        upper = lower+F(1, 10**12)
        assert lower < value-error < value+error < upper and lower*upper > 0
        assert (value < 0) == (t == 1)
        observed = independent_values(qf, af, kf, float(t))
        maximum = max(maximum, *(abs(x-float(value)) for x in observed))
        print(f'T={t}: D in [{float(lower):.12f},{float(upper):.12f}], rational tail < {float(error):.3g}')
    assert maximum < 3e-9
    print(f'Independent semigroup and main polynomial moments: max difference {maximum:.3g}; tolerance 3e-9')
    for left, right in ((.5, 1.), (1., 2.)):
        root = brentq(lambda t: independent_values(qf, af, kf, t)[0], left, right)
        print(f'Unique root in ({left:g},{right:g}): numerical estimate {root:.8f}')
    print('PASS')


if __name__ == '__main__':
    main()
