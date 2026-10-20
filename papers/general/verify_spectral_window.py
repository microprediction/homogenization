"""Positive spectral windows preserve the full strong sector, not observed blocks."""
from fractions import Fraction as R
import numpy as np
from scipy.integrate import quad_vec
from scipy.linalg import eigh, expm, null_space
from effective_generator import gk, stationary


def close(a, b):
    assert np.linalg.norm(np.asarray(a)-b) <= 3e-8*(1+np.linalg.norm(b))


def sector(M):
    S, A = (M+M.T)/2, (M-M.T)/2
    d, U = eigh(S)
    assert d[0] > 0
    W = (U/np.sqrt(d))@U.T
    return np.linalg.norm(W@A@W, 2)


def bartlett(L, T):
    """Second integrated semigroup via a block exponential, without cancellation."""
    n = len(L); I = np.eye(n); Z = np.zeros((n, n))
    E = expm(T*np.block([[-L, I, Z], [Z, Z, I], [Z, Z, Z]]))
    return E[:n, 2*n:]/T


def pair(L, omega):
    return np.linalg.solve(L+omega**2*np.linalg.inv(L), np.eye(len(L)))


def centered(Q, pi):
    V = null_space(np.sqrt(pi)[None, :]); U = V/np.sqrt(pi)[:, None]
    return -U.T@(pi[:, None]*Q)@U, U


# Rational Poisson certificate: no symbolic package or numerical inverse required.
n = 5; P = np.roll(np.eye(n, dtype=int), 1, axis=1); Q = P-np.eye(n, dtype=int)
Pi = np.full((n, n), R(1, n), dtype=object)
H = np.array([[R([2, 1, 0, -1, -2][(j-i) % n], 5) for j in range(n)] for i in range(n)])
F = np.array([[1, -1], [-1, 1], [0, -1], [0, 1], [0, 0]])
assert np.array_equal(P.T@P, np.eye(n, dtype=int)) and (F*F).sum() == 6
assert np.array_equal(-Q@H, np.eye(n, dtype=int)-Pi)
assert all(x == 0 for x in H.sum(axis=0)) and all(x == 0 for x in H.sum(axis=1))
K = F.T@H@F/5
assert np.array_equal(K, np.array([[R(1, 5), R(-1, 5)], [R(-1, 5), R(2, 5)]]))
D = F.T@(Q-Q.T)@F
assert R(int(D[0, 1]), 10) == R(1, 10)
T = R(1, 100); leading = T*T/60
tail = R(6, 5)*4*T**3*R(50, 49)/24
assert leading == R(1, 600000) and tail == R(1, 4900000)
assert leading-tail == R(43, 29400000) > 0
L, U = centered(Q, np.ones(n)/n); V = U.T@F/n
B = V.T@bartlett(L, float(T))@V
redundant = np.column_stack((np.eye(2), [1, 1]))
Br = redundant.T@B@redundant; kernel = np.array([-1, -1, 1])
close(Br@kernel, np.zeros(3)); close((Br+Br.T)@kernel, np.zeros(3))
close(gk(Q, F.T), np.asarray(K, float))
assert ((B-B.T)/2)[0, 1] > float(leading-tail)
print('five-cycle: f=(1,-1,0,0,0), g=(-1,1,-1,1,0); K=[[1,-1],[-1,2]]/5; s(K)=0')
print('exact: skew(F^T D_pi Q F)[1,2]=1/10; T=1/100; leading=1/600000; tail<=1/4900000')
print('analytic estimates: ||F||_pi^2<=6/5; ||Q||_2<=2; exp(1/50)<=50/49')
print(f'exact positive lower bound=43/29400000; Bartlett skew[1,2]={((B-B.T)/2)[0,1]:.12g}; s(B)={sector(B):.12g}')

# An irreducible defective generator: centered minimal polynomial (z+3)^2.
Qd = np.array([[-1, 1, 0], [0, -1, 1], [4, 0, -4]])
pid = np.array([R(4, 9), R(4, 9), R(1, 9)])
N = Qd+3*np.eye(3, dtype=int)
assert np.array_equal(N@N, 9*np.outer(np.ones(3, dtype=int), pid))
assert not np.array_equal(N, 3*np.outer(np.ones(3, dtype=int), pid))
Ld, Ud = centered(Qd, np.asarray(pid, float)); Bd = bartlett(Ld, 1.)
b = (2+np.exp(-3))/9; derivative = (-1-5*np.exp(-3))/27
close(Bd, b*np.eye(2)+derivative*(Ld-3*np.eye(2)))
close(sector(Ld), 1/np.sqrt(3))
assert sector(Bd) <= sector(Ld)+3e-8
print(f'defective 3-cycle: pi=(4/9,4/9,1/9); eigenvalues 0,-3,-3; kappa=1/sqrt(3); T=1; s(B)={sector(Bd):.12g}')

rng = np.random.default_rng(20261020); matrices = [Ld]; chain_count = 0
for n in (3, 4, 5, 8):
    for _ in range(8):
        # Balanced stationary flows, with independent holding-time changes.
        flow = np.zeros((n, n))
        for j in range(n):
            cyc = rng.permutation(n)[:rng.integers(2, n+1)]
            rate = rng.uniform(.1, 2)
            for a, b in zip(cyc, np.roll(cyc, -1)):
                flow[a, b] += rate
        flow += .05*(np.ones((n, n))-np.eye(n))
        pi = rng.uniform(.2, 2, n); pi /= pi.sum()
        Q = flow/pi[:, None]; Q -= np.diag(Q.sum(axis=1))
        close(stationary(Q), pi)
        L, U = centered(Q, pi); close(gk(Q, U.T), np.linalg.inv(L))
        matrices.append(L); chain_count += 1
for n in (2, 3, 5, 8):
    for _ in range(8):
        X = rng.normal(size=(n, n)); A = (X-X.T)/2
        root = np.diag(np.exp(rng.uniform(-1, 1, n)))
        matrices.append(root@(np.eye(n)+A)@root)

count = 0; maximum = 0.
for L in matrices:
    kappa = sector(L); close(sector(np.linalg.inv(L)), kappa)
    for omega in (0., .3, 3.):
        Rw = pair(L, omega); I = np.eye(len(L))
        close(Rw, (np.linalg.inv(L-1j*omega*I)+np.linalg.inv(L+1j*omega*I))/2)
        assert sector(Rw) <= kappa+3e-8*(1+kappa)
    for scaled_T in (.02, .2, 2., 20.):
        T = scaled_T/np.linalg.norm(L, 2); B = bartlett(L, T)
        assert sector(B) <= kappa+3e-8*(1+kappa)
        V = rng.normal(size=(len(L), min(2, len(L))))
        assert sector(V.T@B@V) <= kappa+3e-8*(1+kappa)
        maximum = max(maximum, sector(B)/kappa); count += 1
    # Signed time window with a positive atomic spectral measure.
    weights, frequencies = np.array([.2, .3, .5]), np.array([0., .7, 2.1])
    W = sum(a*pair(L, w) for a, w in zip(weights, frequencies))
    assert sector(W) <= kappa+3e-8*(1+kappa)
print(f'full-space Bartlett checks={count}; random nonuniform chains={chain_count}; general accretive matrices=32; max sector ratio={maximum:.12g}')

# Independent time and frequency integration on the defective, hence nonnormal, case.
L = Ld; T = 1.; target = bartlett(L, T)
time_value, time_error = quad_vec(lambda t: (1-t/T)*expm(-t*L), 0, T)
close(time_value, target)
gamma = eigh((L+L.T)/2, eigvals_only=True)[0]; end = 25/gamma
atomic_time, _ = quad_vec(lambda t: sum(a*np.cos(w*t) for a, w in zip(weights, frequencies))*expm(-t*L), 0, end)
atomic_pair = sum(a*pair(L, w) for a, w in zip(weights, frequencies))
close(atomic_time, atomic_pair)  # Time tail <= exp(-25)/gamma, inside tolerance.
Omega = 80*np.linalg.norm(L, 2)
def spectral_density(w):
    return T/(2*np.pi)*np.sinc(T*w/(2*np.pi))**2
frequency_value, quadrature_error = quad_vec(lambda w: 2*spectral_density(w)*pair(L, w), 0, Omega, epsabs=1e-10)
tail_bound = 8*np.linalg.norm(L, 2)/(3*np.pi*T*Omega**3)
discrepancy = np.linalg.norm(frequency_value-target, 2)
assert discrepancy <= tail_bound+quadrature_error+3e-8*(1+np.linalg.norm(target))
print(f'defective time discrepancy={np.linalg.norm(time_value-target):.3g}; frequency discrepancy={discrepancy:.3g}; analytic frequency tail bound={tail_bound:.3g}')
print('PASS')
