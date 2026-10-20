"""Finite-window reversibility and a rational skew-rank inflation certificate."""
from fractions import Fraction as R
from math import factorial
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import eigh, expm, null_space
from effective_generator import stationary


def close(a, b):
    assert np.linalg.norm(np.asarray(a)-b) <= 3e-8*(1+np.linalg.norm(b))


def bartlett(Q, T):
    n = len(Q); I = np.eye(n); Z = np.zeros((n, n))
    return expm(T*np.block([[Q, I, Z], [Z, Z, I], [Z, Z, Z]]))[:n, 2*n:]/T


def skew(M):
    return (M-M.T)/2


def pf(A):
    return A[0, 1]*A[2, 3]-A[0, 2]*A[1, 3]+A[0, 3]*A[1, 2]


Q = np.array([[-2, 1, 0, 1, 0], [0, -2, 1, 0, 1], [1, 0, -1, 0, 0], [1, 0, 0, -1, 0], [0, 1, 0, 0, -1]])
F = np.vstack((np.eye(4, dtype=int), -np.ones((1, 4), dtype=int)))
Z = np.array([[3, 1, 0, 2], [-2, 1, 0, -3], [3, 1, 5, 2], [3, 1, 0, 7], [-7, -4, -5, -8]], dtype=object)/R(5)
assert np.array_equal(-Q@Z, F) and all(x == 0 for x in Z.sum(axis=0))
assert all(x == 0 for x in Q.sum(axis=0)) and all(x == 0 for x in Q.sum(axis=1))
K = F.T@Z/R(5); Ak = skew(K)
assert np.array_equal(K*5, np.array([[2, 1, 1, 2], [1, 1, 1, 1], [2, 1, 2, 2], [2, 1, 1, 3]]))
assert pf(Ak) == 0 and Ak[0, 2] == -R(1, 10)  # Nonzero skew 4x4 with zero Pfaffian: rank two.
N = 24; term = np.eye(5, dtype=object)/R(2); total = term.copy()
for k in range(1, N+1):
    term = term@Q/R(k+2); total += term
Bq = F.T@total@F/R(5); Aq = skew(Bq); pq = pf(Aq)
assert max(abs(x) for x in Q.flat) <= 2 and max(abs(Q).sum(axis=1)) == 4
error = R(4)**(N+1)/R(factorial(N+3))/(1-R(4, N+4))
delta = R(2, 5)*error; M = max(abs(x) for x in Aq.flat)
budget = 6*M*delta+3*delta**2
assert -R(1, 500000) < pq-budget < pq+budget < -R(1, 600000)
B = F.T@bartlett(Q, 1.)@F/5; A = skew(B)
close(B, np.asarray(Bq, float)); close(pf(A), float(pq))
assert min(eigh((B+B.T)/2, eigvals_only=True)) > 0
assert min(eigh(np.asarray((K+K.T)/2, float), eigvals_only=True)) > 0
print('five states: uniform pi=1/5; F=(e1-e5,e2-e5,e3-e5,e4-e5); one directed triangle and two reversible pendants')
print('exact K=[[2,1,1,2],[1,1,1,1],[2,1,2,2],[2,1,1,3]]/5; skew rank=2; both symmetric ranks=4')
print('T=1: exact rational Pfaffian enclosure (-1/500000,-1/600000); finite-window skew rank=4')
print(f'degree={N}; ||Q||_infinity=4; feature weighted absolute sum=2/5; entry error<={float(delta):.12g}; Pfaffian error<={float(budget):.12g}')
print(f'Pfaffian={pf(A):.12g}; determinant={np.linalg.det(A):.12g}; smallest skew singular value={np.linalg.svd(A, compute_uv=False)[-1]:.12g}')
value, error = quad_vec(lambda t: (1-t)*F.T@expm(t*Q)@F/5, 0, 1)
close(value, B)
print(f'independent matrix quadrature discrepancy={np.linalg.norm(value-B):.3g}')

# Exact Jordan obstruction: a nontrivial block survives because b_T'(lambda)<0.
L = np.array([[2., 1, 0], [0, 2, 1], [0, 0, 2]])
lambda0 = 2.; T = 1.; nilpotent = L-lambda0*np.eye(3)
derivatives = [quad_vec(lambda t: (-t)**k*(1-t)*np.exp(-lambda0*t), 0, 1)[0] for k in range(3)]
expected = derivatives[0]*np.eye(3)+derivatives[1]*nilpotent+derivatives[2]/2*nilpotent@nilpotent
close(bartlett(-L, T), expected); assert derivatives[1] < 0
assert np.linalg.norm(skew(expected)) > .01
print(f'defective 3x3 block: lambda=2, T=1, b_prime={derivatives[1]:.12g}; its Bartlett image is not symmetric')
L = np.array([[1., 5], [0, 2]])  # Positive-stable but not accretive: the lemma needs only the spectrum.
assert min(eigh((L+L.T)/2, eigvals_only=True)) < 0
image = bartlett(-L, 1.); b1, b2 = np.exp(-1), (1+np.exp(-2))/4
close(image, np.array([[b1, 5*(b2-b1)], [0, b2]]))
print('positive-stable non-accretive triangular control: eigenvalues=(1,2); Bartlett image is not symmetric')

rng = np.random.default_rng(20261020); scalar_checks = 0
for _ in range(40):
    a, b, T = np.exp(rng.uniform(-2, 2, 3)); z = a+1j*b
    direct, error = quad_vec(lambda t: (1-t/T)*np.exp(-z*t), 0, T)
    sine, error = quad_vec(lambda u: quad_vec(lambda t: np.exp(-a*t)*np.sin(b*t), 0, u)[0], 0, T)
    close(direct.imag, -sine/T); assert direct.imag < 0; scalar_checks += 1
print(f'strict scalar sine-sign/Fubini controls={scalar_checks}')

chain_checks = 0; clocks = 0
for n in (3, 4, 5, 7):
    for reversible in (False, True):
        for _ in range(4):
            pi = rng.uniform(.2, 2, n); pi /= pi.sum()
            X = rng.uniform(.05, 1, (n, n)); flow = (X+X.T)/2; np.fill_diagonal(flow, 0)
            if not reversible:
                for j in range(n): flow[j, (j+1) % n] += .5
            Qr = flow/pi[:, None]; Qr -= np.diag(Qr.sum(axis=1)); close(stationary(Qr), pi)
            U = null_space(np.sqrt(pi)[None, :])/np.sqrt(pi)[:, None]
            L = -U.T@(pi[:, None]*Qr)@U
            for scaled_T in (.1, 1., 10.):
                T = scaled_T/np.linalg.norm(L, 2); Bt = bartlett(-L, T)
                if reversible: close(Bt, Bt.T)
                else: assert np.linalg.norm(skew(Bt)) > 1e-10*T
                chain_checks += 1
            if not reversible:
                T = 1/np.linalg.norm(L, 2)
                for c in (.5, .2, .05):
                    Bt = bartlett(-c*L, T)
                    remainder = np.linalg.norm(Bt-T/2*np.eye(n-1)+c*T*T/6*L, 2)
                    bound = c*c*T**3*np.linalg.norm(L, 2)**2*np.exp(c*T*np.linalg.norm(L, 2))/24
                    assert remainder <= bound+3e-8*T; clocks += 1
print(f'full-feature reversibility controls={chain_checks}; fixed-T clock remainder checks={clocks}; no statistical margin claimed')
print('PASS')
