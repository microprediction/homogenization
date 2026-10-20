"""Maturity-zero theorem: rational spectral/sign certificates and independent moments."""
from fractions import Fraction as F
from math import ceil, factorial, prod

import numpy as np
from scipy.linalg import expm
from scipy.optimize import brentq
from polymoments import Poly, cir_generator, switched_expectation


def identity(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def matmul(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def action(a, x):
    return [sum(u*v for u, v in zip(row, x)) for row in a]


def functional(matrix, pi, a):
    return sum(pi[i]*matrix[i][j]*(a[i]-a[j]) for i in range(len(a)) for j in range(len(a)))


def forward(q, pi, a, k, t, degree=160):
    """Rational first-moment action; remainder bounds the final J/2 coordinate."""
    n = len(q)
    matrix = [[F(0) for j in range(n+2)] for i in range(n+2)]
    for i in range(n):
        for j in range(n):
            matrix[i][j] = q[j][i]-(k[i] if i == j else 0)
        matrix[i][n] = a[i]*pi[i]
        matrix[n+1][i] = 1
    scaled = [[t*x for x in row] for row in matrix]
    bound = ceil(max(sum(map(abs, row)) for row in scaled))
    initial = [p/25 for p in pi]+[F(1), F(0)]
    term, total = initial[:], initial[:]
    for j in range(1, degree+1):
        term = [x/j for x in action(scaled, term)]
        total = [x+y for x, y in zip(total, term)]
    error = F(3**bound*bound**(degree+1), 2*factorial(degree+1))
    return total[-1]/2, error


def compare(q, a, k, t):
    n = len(q)
    pi = np.linalg.solve(q.T+np.ones((n, n)), np.ones(n))
    reverse = q.T*pi[None, :]/pi[:, None]
    killed = q-np.diag(k)
    block = np.zeros((3*n, 3*n))
    block[:n, :n], block[n:2*n, :n], block[2*n:, n:2*n] = killed, np.eye(n), np.eye(n)
    integral = expm(t*block)[2*n:, :n]
    predicted = ((pi*a)@integral@np.ones(n)-pi@integral@a)/2
    def mean(generator):
        matrix = np.zeros((n+2, n+2))
        matrix[:n, :n] = generator.T-np.diag(k)
        matrix[:n, n], matrix[-1, :n] = a*pi, 1
        return (expm(t*matrix)@np.r_[pi/25, 1, 0])[-1]/2
    p = Poly(1)
    operators = [cir_generator(p, ki, ai/ki, .2, 0) for ai, ki in zip(a, k)]
    polynomial = lambda generator: switched_expectation(generator, operators, t,
                                    p.monomial(b=1), p, .04, pi)/2
    return predicted, mean(q)-mean(reverse), polynomial(q)-polynomial(reverse)


def main():
    rates = [F(1, 100), F(1), F(2), F(3)]
    pi = [F(x, 611) for x in (600, 6, 3, 2)]
    q = [[rates[i]*(F(j == (i+1) % 4)-F(i == j)) for j in range(4)] for i in range(4)]
    a = list(map(F, (5, 2, 8, 2)))
    xi, k = [F(1, 5), F(2, 5), F(1), F(3, 2)], [F(53, 25), F(56, 25), F(13, 5), F(29, 10)]
    reverse = [[pi[j]*q[j][i]/pi[i] for j in range(4)] for i in range(4)]
    killed = [[q[i][j]-(k[i] if i == j else 0) for j in range(4)] for i in range(4)]
    powers = [identity(4)]
    for j in range(3):
        powers.append(matmul(powers[-1], killed))
    assert functional(powers[0], pi, a) == functional(powers[1], pi, a) == 0
    alpha = functional(powers[2], pi, a)
    beta = functional(powers[3], pi, a)-sum(killed[i][i] for i in range(4))*alpha
    assert alpha < 0 < beta
    diagonal = [rates[i]+k[i] for i in range(4)]
    characteristic = lambda z: prod(z+r for r in diagonal)-prod(rates)
    assert min(diagonal) > F(1, 10)
    for r in diagonal:
        assert characteristic(-r-F(1, 10))*characteristic(-r+F(1, 10)) < 0
    assert all(diagonal[i+1]-diagonal[i] > F(1, 5) for i in range(3))
    product = identity(4)
    for r in diagonal:
        product = matmul(product, [[killed[i][j]+r*(i == j) for j in range(4)] for i in range(4)])
    assert product == [[prod(rates)*F(i == j) for j in range(4)] for i in range(4)]
    margin = min(2*ai-x*x for ai, x in zip(a, xi))
    assert margin == F(7, 4) and all(ki == 2+F(3, 5)*x for ki, x in zip(k, xi))
    for moment, loading in ((F(2), F(1, 2)), (F(-1, 10), F(1, 10))):
        coefficient = max(x*x*loading*loading/2+(-F(3, 5)*x*moment-2)*loading
                          +(moment*moment-moment)/2 for x in xi)
        assert coefficient < 0
        print(f'Moment supersolution r={moment}, b={loading}: max v coefficient {coefficient}')
    print(f'Exact numerator alpha*z+beta: alpha={alpha}, beta={beta}; Feller margin={margin}')
    print('Four disjoint rational spectral intervals: centers -', ', -'.join(map(str, diagonal)), '; radius 1/10')
    for t in (F(1), F(2)):
        left, e1 = forward(q, pi, a, k, t)
        right, e2 = forward(reverse, pi, a, k, t)
        value, error = left-right, e1+e2
        assert error < F(1, 10**60)
        lower = F((value*10**12).numerator//(value*10**12).denominator, 10**12)
        upper = lower+F(1, 10**12)
        assert lower < value-error < value+error < upper and lower*upper > 0
        assert (value < 0) == (t == 1)
        print(f'T={t}: D in [{float(lower):.12f}, {float(upper):.12f}], rational tail < {float(error):.3g}')
        observed = compare(np.array(q, float), np.array(a, float), np.array(k, float), float(t))
        assert max(abs(x-float(value)) for x in observed) < 3e-9
    rng, maximum = np.random.default_rng(20261020), 0.
    for n in (3, 4, 5, 7):
        rates = np.r_[.01, np.arange(1, n)]
        base = np.diag(-rates)+np.diag(rates[:-1], 1)
        base[-1, 0] = rates[-1]
        for speed in (.5, 1., 2.):
            for t in (.2, 1., 2.):
                values = compare(speed*base, rng.uniform(2, 5, n), rng.uniform(2, 3, n), t)
                maximum = max(maximum, np.ptp(values))
    assert maximum < 3e-9
    print(f'36 independent semigroup/forward/polynomial comparisons: max difference {maximum:.3g}')
    q, a, k = np.array(q, float), np.array(a, float), np.array(k, float)
    root = brentq(lambda t: compare(q, a, k, t)[0], 1., 2.)
    print(f'Unique positive maturity zero: numerical estimate {root:.10f}; exact bracket (1,2)')
    unit = [[F(j == (i+1) % 4)-F(i == j) for j in range(4)] for i in range(4)]
    skew = [[(unit[i][j]-unit[j][i])/2 for j in range(4)] for i in range(4)]
    projection = matmul(list(map(list, zip(*skew))), skew)
    assert projection != [[F(0)]*4 for i in range(4)] and matmul(projection, projection) == projection
    pi4, a4, k4 = [F(1, 4)]*4, list(map(F, (2, 1, 5, 5))), list(map(F, ('2.12', '2.24', '2.6', '2.9')))
    for t in (F(3, 5), F(4, 5)):
        v, e = forward(unit, pi4, a4, k4, t)
        w, f = forward(list(map(list, zip(*unit))), pi4, a4, k4, t)
        assert abs(v-w) > e+f and ((v-w < 0) == (t == F(3, 5)))
    print('Unit four-cycle: exact skew norm 1 and sign brackets (.6,.8); one zero on (0,pi)')
    jordan = np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]])
    assert np.max(abs((jordan+3*np.eye(3))@(jordan+3*np.eye(3))@jordan)) < 3e-9
    assert max(abs(x) for x in compare(jordan, np.array([2., 3., 4.]), np.ones(3)*2, 1.)) < 3e-9
    print('Repeated real eigenvalue / defective base-chain cancellation: PASS')
    print('PASS')


if __name__ == '__main__':
    main()
