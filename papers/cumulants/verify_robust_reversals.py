"""Robust but finite CIR maturity reversals; rational signs and perturbation bounds."""
from fractions import Fraction as F
from math import ceil, factorial

import numpy as np
from scipy.linalg import expm
from polymoments import Poly, cir_generator, switched_expectation


def multiply(a, b):
    return [[sum(x*y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def solve(a, b):
    c = [row[:]+[v] for row, v in zip(a, b)]
    for j in range(len(b)):
        pivot = next(i for i in range(j, len(b)) if c[i][j])
        c[j], c[pivot] = c[pivot], c[j]
        c[j] = [v/c[j][j] for v in c[j]]
        for i in range(len(b)):
            if i != j:
                v = c[i][j]
                c[i] = [x-v*y for x, y in zip(c[i], c[j])]
    return [row[-1] for row in c]


def functional(matrix, a):
    return sum(a[i]*(sum(matrix[i])-sum(row[i] for row in matrix)) for i in range(6))/6


def action(matrix, t):
    scaled = [[t*x for x in row] for row in matrix]
    bound = ceil(max(sum(map(abs, row)) for row in scaled))
    term = total = [F(i == 5) for i in range(6)]
    for j in range(1, 301):
        term = [sum(x*y for x, y in zip(row, term))/j for row in scaled]
        total = [x+y for x, y in zip(total, term)]
    return total[1]-F(99, 100)*total[0], F(2*3**bound*bound**301, factorial(301))


def gap(q, k, a, t):
    a_matrix = q-np.diag(k)
    augmented = np.zeros((18, 18))
    augmented[:6, :6] = a_matrix
    augmented[:6, 6:12] = augmented[6:12, 12:] = np.eye(6)
    m = expm(t*augmented)[:6, 12:]
    return (a @ m @ np.ones(6)-np.ones(6) @ m @ a)/12


def main():
    n, epsilon = 6, F(1, 10**50)
    r = [F(99, 100), F(100, 99)]+[F(1)]*4
    k, xi = [2+x for x in r], [2*x for x in r]
    centered = [F(0), F(338101, 494010), F(3383, 4990), -F(1577, 4990), F(1), F(0)]
    a = [10+x for x in centered]
    q = [[F(j == (i+1) % n)-F(i == j) for j in range(n)] for i in range(n)]
    killed = [[q[i][j]-(k[i] if i == j else 0) for j in range(n)] for i in range(n)]
    shifted = [[killed[i][j]+5*(i == j) for j in range(n)] for i in range(n)]
    identity = [[F(i == j) for j in range(n)] for i in range(n)]
    polynomial, product = [F(1)], identity
    for ri in r:
        d = 3+ri
        polynomial = [polynomial[0]]+[polynomial[j]+d*polynomial[j-1]
                      for j in range(1, len(polynomial))]+[d*polynomial[-1]]
        product = multiply(product, [[killed[i][j]+d*(i == j) for j in range(n)] for i in range(n)])
    polynomial[-1] -= 1
    assert product == identity
    power, moments, derivatives = identity, [], []
    for j in range(6):
        moments.append(functional(power, a))
        derivatives.append(power[1][5]-r[0]*power[0][5])
        power = multiply(power, killed)
    alpha = F(199, 296406000)
    numerator = [sum(polynomial[j]*moments[ell+2-j] for j in range(ell+1)) for ell in range(4)]
    assert numerator == [alpha, 3*alpha, 0, 0] and derivatives[:5] == [0, 0, 0, 0, 1]
    assert derivatives[5] == 3-polynomial[1]
    negative = [[-x for x in row] for row in killed]
    transpose = list(map(list, zip(*negative)))
    u, v = solve(negative, [F(1)]*n), solve(transpose, [F(1)]*n)
    b, c = (u[0]-v[0])/12, -(solve(negative, u)[0]-solve(transpose, v)[0])/12
    assert b == -F(2125, 121623804) < 0
    assert max(map(abs, centered)) == 1 and min(k) > 2
    print(f'epsilon={epsilon}; Q_epsilon=P-I+epsilon^2*(Pi-I); a_epsilon=a+epsilon*e0')
    print(f'r={r}; a0={a}; k=2+r; xi=2r; kappa=2; rho=-1/2; v0=.04; pi=1/6')
    print(f'First-order affine coefficients: b={b} ({float(b):.12g}), c={c} ({float(c):.12g})')
    print('Bounds: |D_e-D_0|<=epsilon*T; |beta_e-b*epsilon|<=epsilon^2; |gamma_e-c*epsilon|<=epsilon^2')
    assert epsilon < abs(b)/2
    qe = [[q[i][j]+epsilon**2*(F(1, 6)-F(i == j)) for j in range(n)] for i in range(n)]
    assert all(sum(row) == 0 for row in qe) and all(sum(col) == 0 for col in zip(*qe))
    assert min(qe[i][j] for i in range(n) for j in range(n) if i != j) == epsilon**2/6
    me = [[-qe[i][j]+k[i]*(i == j) for j in range(n)] for i in range(n)]
    mt = list(map(list, zip(*me)))
    ue, ve = solve(me, [F(1)]*n), solve(mt, [F(1)]*n)
    ae = [ai+epsilon*(i == 0) for i, ai in enumerate(a)]
    beta = sum(ai*(x-y) for ai, x, y in zip(ae, ue, ve))/12
    gamma = -sum(ai*(x-y) for ai, x, y in zip(ae, solve(me, ue), solve(mt, ve)))/12
    assert beta < 0 and abs(beta-b*epsilon) < epsilon**2 and abs(gamma-c*epsilon) < epsilon**2
    radius = F(1, 50)
    assert F(1, 99) < radius and F(43, 25)**2 < 3  # omega > .86-.02 = 21/25.
    residue_bound = 5*alpha*F(25, 24)**4/2
    ratio = F(3, 2)*F(51, 24)**4
    print(f'alpha={alpha}; radius=1/50; sigma in (-88/25,-87/25); other real parts <-112/25')
    print(f'Separation >24/25; diameter <51/25; ratio constant={ratio}; C0={residue_bound}')
    assert ratio < F(9, 10)*sum(F(4**j, factorial(j)) for j in range(5))
    assert sum(F(3**j, factorial(j)) for j in range(15)) > 20
    assert sum(F(2**j, factorial(j)) for j in range(10)) > 7
    largest = F(0)
    for t, lower in zip((4, 6, 10, 14, 18), (126, -1267, 786335, -397276026, 176052884169)):
        value, error = action(shifted, F(t))
        assert lower < value-error < value+error < lower+1
        distance = min(abs(lower), abs(lower+1))
        assert alpha*distance/(2*3**(5*t)) > epsilon*t
        largest = max(largest, error)
        print(f'T={t}: baseline H in [{lower},{lower+1}]; perturbation cannot change sign')
    assert largest < F(105, 10**129)
    print('Degree-300 action error <1.05e-127; signs +,-,+,-,+')
    t = 40
    error = epsilon*(t+1)+residue_bound/(20**t*epsilon)+(F(1, 4)+epsilon*(t+1)/2)/7**t
    value = b*t+c
    assert -F(685, 10**6) < value-error < value+error < -F(684, 10**6)
    assert b+epsilon < 0
    print('D_epsilon(40)/epsilon in [-.000685,-.000684]; the analytic upper bound decreases for T>=40')
    print('At least five positive maturity sign reversals; no zeros at or beyond 40; finitely many overall')
    margin = min(2*ai-x*x for ai, x in zip(a, xi))
    assert margin == F(38343, 2495)
    print(f'Feller margin >= {margin}; minimum off-diagonal rate epsilon^2/6={epsilon**2/6}')
    for s, vexp in ((F(2), F(1, 2)), (-F(1, 10), F(1, 10))):
        maximum = max(x*x*vexp*vexp/2+(-x*s/2-2)*vexp+(s*s-s)/2 for x in xi)
        assert maximum < 0
        print(f'Moment supersolution s={s}, b={vexp}: max v coefficient={maximum}')
    print(f'Log-count lower coefficient omega/(a*pi), a=88/25, omega>21/25; m0={alpha*F(25,51)**4/10}')
    qf, kf = np.array(q, float), np.array(k, float)
    discrepancy = 0.
    p = Poly(1)
    for perturb in (0., .01, .1):
        qp = qf+perturb**2*(np.ones((n, n))/n-np.eye(n))
        ap = np.array(a, float)+perturb*np.eye(n)[0]
        operators = [cir_generator(p, ki, ai/ki, x, 0) for ai, ki, x in zip(ap, kf, np.array(xi, float))]
        for t in (.5, 1., 2.):
            means = [switched_expectation(g, operators, t, p.monomial(b=1), p, .04, np.ones(n)/n)/2
                     for g in (qp, qp.T)]
            discrepancy = max(discrepancy, abs(means[0]-means[1]-gap(qp, kf, ap, t)))
            assert abs(gap(qp, kf, ap, t)-gap(qf, kf, np.array(a, float), t)) < perturb*t+3e-9
    assert discrepancy < 3e-9
    print(f'Nine independent polynomial-moment checks: max error {discrepancy:.3g}; tolerance 3e-9')
    print('PASS')


if __name__ == '__main__':
    main()
