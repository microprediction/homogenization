"""Six-state infinite maturity reversals: exact cancellations and rational signs."""
from fractions import Fraction as F
from math import ceil, factorial, prod

import numpy as np
from scipy.linalg import expm
from polymoments import Poly, cir_generator, switched_expectation


def identity(n):
    return [[F(i == j) for j in range(n)] for i in range(n)]


def multiply(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def functional(matrix, a):
    n = len(a)
    return sum(a[i]*(sum(matrix[i])-sum(row[i] for row in matrix)) for i in range(n))/n


def shifted_action(matrix, t, degree=300):
    """Enclose (e_1-r_0 e_0)^T exp(t*(A+5I)) e_5 without exponential underflow."""
    scaled = [[t*x for x in row] for row in matrix]
    bound = ceil(max(sum(map(abs, row)) for row in scaled))
    term, total = [F(i == 5) for i in range(6)], [F(i == 5) for i in range(6)]
    for j in range(1, degree+1):
        term = [sum(x*y for x, y in zip(row, term))/j for row in scaled]
        total = [x+y for x, y in zip(total, term)]
    error = F(2*3**bound*bound**(degree+1), factorial(degree+1))
    return total[1]-F(99, 100)*total[0], error


def main():
    n = 6
    r = [F(99, 100), F(100, 99)]+[F(1)]*4
    k, xi = [2+x for x in r], [2*x for x in r]
    centered = [F(0), F(338101, 494010), F(3383, 4990), -F(1577, 4990), F(1), F(0)]
    a = [10+x for x in centered]
    q = [[F(j == (i+1) % n)-F(i == j) for j in range(n)] for i in range(n)]
    killed = [[q[i][j]-(k[i] if i == j else 0) for j in range(n)] for i in range(n)]
    shifted = [[killed[i][j]+5*(i == j) for j in range(n)] for i in range(n)]
    characteristic = [F(1)]
    product = identity(n)
    for ri in r:
        d = 3+ri
        characteristic = [characteristic[0]]+[characteristic[j]+d*characteristic[j-1]
                          for j in range(1, len(characteristic))]+[d*characteristic[-1]]
        factor = [[killed[i][j]+d*(i == j) for j in range(n)] for i in range(n)]
        product = multiply(product, factor)
    characteristic[-1] -= 1
    assert product == identity(n) and prod(r) == 1
    quotient = [characteristic[0]]
    for value in characteristic[1:]:
        quotient.append(value-3*quotient[-1])
    assert quotient.pop() == 0  # p(z)=(z+3)*q5(z), exact Perron cancellation.
    power, moments, derivatives = identity(n), [], []
    for j in range(6):
        moments.append(functional(power, a))
        derivatives.append(power[1][5]-r[0]*power[0][5])
        power = multiply(power, killed)
    assert moments[:2] == [0, 0] and derivatives[:5] == [0, 0, 0, 0, 1]
    numerator = [sum(characteristic[j]*moments[ell+2-j] for j in range(ell+1)) for ell in range(4)]
    alpha = F(199, 296406000)
    assert numerator == [alpha, 3*alpha, 0, 0]
    margin = min(2*ai-x*x for ai, x in zip(a, xi))
    assert margin > 0
    print('Q is the unit one-way six-cycle; pi is uniform; v0=.04; kappa=2; rho=-1/2')
    print(f'r={r}; xi=2r; k=2+r; a=10+{centered}')
    print(f'Exact numerator alpha*z^2*(z+3): alpha={alpha}; Feller margin={margin}')
    print(f'q5 coefficients (descending): {quotient}')
    for s, b in ((F(2), F(1, 2)), (-F(1, 10), F(1, 10))):
        maximum = max(x*x*b*b/2+(-x*s/2-2)*b+(s*s-s)/2 for x in xi)
        assert maximum < 0
        print(f'Moment supersolution s={s}, b={b}: max v coefficient {maximum}')
    eta, radius = max(abs(x-1) for x in r), F(1, 50)
    separation, diameter = 1-2*radius, 2+2*radius
    assert eta == F(1, 99) < radius and separation == F(24, 25) and diameter == F(51, 25)
    ratio = F(3, 2)*(diameter/separation)**4
    exp4_lower = sum(F(4**j, factorial(j)) for j in range(5))
    assert ratio == F(250563, 8192) < exp4_lower == F(103, 3)
    print(f'Spectral perturbation {eta} < disk radius {radius}; separation {separation}; diameter {diameter}')
    print(f'Extremum error ratio <= {ratio}*exp(-24*T/25); exp(4)>{exp4_lower}; T>=5 suffices')
    expected = (126, -1267, 786335, -397276026, 176052884169)
    largest_error, floating_error = F(0), 0.
    for t, lower in zip((4, 6, 10, 14, 18), expected):
        value, error = shifted_action(shifted, F(t))
        assert lower < value-error < value+error < lower+1
        assert lower*(lower+1) > 0
        largest_error = max(largest_error, error)
        vector = expm(t*np.array(shifted, float))[:, 5]
        observed = vector[1]-.99*vector[0]
        # Scale to the two positive components being subtracted, not their tiny difference.
        floating_error = max(floating_error, abs(observed-float(value))/max(1., abs(vector[1]), abs(vector[0])))
        print(f'T={t}: H=2*exp(5T)*D/alpha in [{lower},{lower+1}]')
    assert largest_error < F(105, 10**129) and floating_error < 3e-9
    print(f'Degree-300 rational actions: uniform tail < 1.05e-127; scaled expm discrepancy {floating_error:.3g}')
    af, kf, qf = np.array(a, float), np.array(k, float), np.array(q, float)
    p = Poly(1)
    operators = [cir_generator(p, ki, ai/ki, x, 0) for ai, ki, x in zip(af, kf, np.array(xi, float))]
    discrepancy = 0.
    for t in (.25, .5, 1., 2.):
        means = [switched_expectation(g, operators, t, p.monomial(b=1), p, .04, np.ones(6)/6)/2
                 for g in (qf, qf.T)]
        e = expm(t*np.array(killed, float))
        predicted = float(alpha)*(e[1, 5]-.99*e[0, 5])/2
        discrepancy = max(discrepancy, abs(means[0]-means[1]-predicted)/max(1., *map(abs, means)))
    assert discrepancy < 3e-9
    print(f'Four independent physical polynomial-moment comparisons: scaled error {discrepancy:.3g}; tolerance 3e-9')
    print('PASS')


if __name__ == '__main__':
    main()
