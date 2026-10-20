"""All-horizon three-state cutoff positivity: exact controls and floating checks."""
from fractions import Fraction as R
from math import factorial
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import eigh, expm, null_space
from effective_generator import stationary


def close(a, b):
    assert np.linalg.norm(np.asarray(a)-b) <= 3e-8*(1+np.linalg.norm(b))


def cutoff(L, T):
    n = len(L); Z = np.zeros((n, n)); I = np.eye(n)
    return expm(T*np.block([[-L, I], [Z, Z]]))[:n, n:]


def centered(Q, pi):
    U = null_space(np.sqrt(pi)[None, :])/np.sqrt(pi)[:, None]
    return -U.T@(pi[:, None]*Q)@U, U


def sector(L):
    d, V = eigh((L+L.T)/2); assert d[0] > 0
    W = (V/np.sqrt(d))@V.T
    return np.linalg.norm(W@(L-L.T)/2@W, 2)


# Rational identities establish the entire defective semigroup formula.
Q = np.array([[-1, 1, 0], [0, -1, 1], [4, 0, -4]])
pi = np.array([R(4, 9), R(4, 9), R(1, 9)])
f = np.array([1, 1, -8]); I = np.eye(3, dtype=int); N = Q+3*I
assert pi@Q@f == pi@f == 0
assert np.array_equal(N@N, 9*np.outer(np.ones(3, dtype=int), pi))
assert not np.array_equal(N, 3*np.outer(np.ones(3, dtype=int), pi))
assert (pi*f)@f == 8 and (pi*f)@N@f == -12
print('defective: rates=(1,1,4), pi=(4/9,4/9,1/9), f=(1,1,-8); centered eigenvalue=-3')
print('exact: (Q+3I)^2=9*1*pi^T; C(t)=exp(-3t)*(8-12t); K=4/3')
print('exact: R(T)=4/3*(1+(3T-1)*exp(-3T)); B(T)=4/3*(1-exp(-3T))')
print('C negative for t>2/3; R positive for all T>0 and above K for T>1/3')
for T in (.01, .1, 1/3, 2/3, 1., 3., 10.):
    pif = np.asarray(pi, float)*f
    Ct = pif@expm(T*Q)@f; Rt = pif@cutoff(-Q, T)@f
    close(Ct, np.exp(-3*T)*(8-12*T))
    close(Rt, 4/3*(1+(3*T-1)*np.exp(-3*T)))
    Bt, error = quad_vec(lambda t: (1-t/T)*pif@expm(t*Q)@f, 0, T)
    close(Bt, 4/3*(-np.expm1(-3*T)))
print(f'T=1: C={-4*np.exp(-3):.12g}; R={4/3*(1+2*np.exp(-3)):.12g}')

rng = np.random.default_rng(20261020); chains = []; real_count = complex_count = 0
for family in range(3):
    for _ in range(32):
        if family == 0:
            X = np.exp(rng.uniform(-2, 2, (3, 3))); np.fill_diagonal(X, 0)
            Q = X-np.diag(X.sum(axis=1)); pi = stationary(Q)
        else:
            pi = rng.uniform(.2, 2, 3); pi /= pi.sum()
            u = rng.uniform(.1, 2, 3); j = rng.uniform(-.2, .2) if family == 1 else 1.
            if family == 2: u = np.full(3, .5)  # Sparse one-way cycles.
            X = np.array([[0, u[0]+j/2, u[2]-j/2], [u[0]-j/2, 0, u[1]+j/2], [u[2]+j/2, u[1]-j/2, 0]])
            Q = X/pi[:, None]; Q -= np.diag(Q.sum(axis=1))
        flow = pi[:, None]*Q; u12, u13, u23 = ((flow+flow.T)/2)[np.triu_indices(3, 1)]
        current = flow[0, 1]-flow[1, 0]
        close([flow[1, 2]-flow[2, 1], flow[2, 0]-flow[0, 2]], np.full(2, current))
        expected = abs(current)/(2*np.sqrt(u12*u13+u12*u23+u13*u23))
        L, U = centered(Q, pi); close(sector(L), expected)
        assert expected <= 1/np.sqrt(3)+3e-8
        discriminant = np.trace(L)**2-4*np.linalg.det(L)
        real_count += discriminant >= 0; complex_count += discriminant < 0
        chains.append(L)
Ld, _ = centered(np.array([[-1, 1, 0], [0, -1, 1], [4, 0, -4]]), np.array([4, 4, 1])/9)
chains.append(Ld)

# Matrix lemma beyond chains: real spectrum needs no small-sector assumption.
matrices = []
for _ in range(32):
    d = rng.uniform(.1, .999); c = rng.uniform(0, d)
    matrices.append(np.array([[1-d, c], [-c, 1+d]]))
    d = rng.uniform(0, .49); c = rng.uniform(d, np.sqrt((1-d*d)/3))
    L = np.array([[1-d, c], [-c, 1+d]]); assert c+d <= 1+3e-8
    matrices.append(L)
matrices.append(np.array([[.01, .99], [-.99, 1.99]]))  # High-sector defective real case.
checks = 0
for L in chains+matrices:
    for scale in (.001, .01, .1, 1., 3., 10., 100.):
        T = scale/np.linalg.norm(L, 2); M = cutoff(L, T)
        assert min(eigh((M+M.T)/2, eigvals_only=True)) > 0
        V = np.array([[1., 0, 1], [0, 1, 1]]); kernel = np.array([-1, -1, 1])
        G = V.T@(M+M.T)/2@V; close(G@kernel, np.zeros(3)); checks += 1
print(f'3-state sector bound=1/sqrt(3); random chains=96, real-spectrum={real_count}, complex-spectrum={complex_count}; plus one exact defective chain')
print(f'general matrix controls=65; cutoff/congruence checks={checks}; n<=3 positive at every horizon by proof')
for L in (chains[0], chains[-1], matrices[-1]):
    for T in (.1, 1., 5.):
        value, error = quad_vec(lambda t: expm(-t*L), 0, T)
        close(value, cutoff(L, T))
print('independent time quadratures=9; real-spectrum matrix lemma includes large-sector defective matrices')

# Recheck the known 12-state upper witness with strict rational sign bounds.
x = R(13, 10); lower = sum(x**k/R(factorial(k)) for k in range(7))
upper = lower+x**7/R(factorial(7))/(1-x/8)
assert R(433, 250)**2 < 3 and 3*R(22, 7)*R(67, 500) < x
assert upper < R(37, 10) and R(67, 250)*R(37, 10) == R(2479, 2500) < 1
a = 1-np.sqrt(3)/2; T = 3*np.pi
r12 = (a-.5*np.exp(-a*T))/(2*(a*a+.25)); assert r12 < -.013
print(f'12-cycle upper witness: T=3pi; hard cutoff={r12:.12g}; unrestricted minimum in [4,12], not determined')
print('exact 12-cycle sign: sqrt(3)>433/250; a<67/500; 3pi*a<13/10; exp(13/10)<37/10; 2a*exp(3pi*a)<2479/2500<1')
print('PASS')
