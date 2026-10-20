"""Population rectangular/Bartlett windows: normal-chain threshold and sector."""
from fractions import Fraction as R
from math import factorial
import numpy as np
from scipy.linalg import expm, eigh, null_space
from scipy.integrate import quad_vec
from effective_generator import gk, stationary


def close(a, b):
    assert np.linalg.norm(np.asarray(a)-b) <= 3e-8*(1+np.linalg.norm(b))


def windows(Q, pi, F, T):
    """Augmented exponential avoids cancellation of two nearly equal inverses."""
    n = len(Q); Z = np.zeros((n, n)); I = np.eye(n)
    E = expm(T*np.block([[Q, I, Z], [Z, Z, I], [Z, Z, Z]]))
    return F.T@(pi[:, None]*E[:n, n:2*n])@F, F.T@(pi[:, None]*E[:n, 2*n:]/T)@F


# Exact rational proof of the integer state threshold; 31/10<pi<22/7 is classical.
x = R(13, 10)
lower = sum(x**j/R(factorial(j)) for j in range(7))
upper = lower+x**7/R(factorial(7))/(1-x/8)
assert lower > R(7, 2) and upper < R(37, 10)
t = R(31, 110)
assert (t+t**3/3)*R(7, 2) > 1
assert R(91, 20) < R(93, 20) and R(481, 100) > R(33, 7)
assert R(17, 10)**2 < 3
print("exact threshold: cot(pi/11)<7/2<exp(13/10)<37/10<2+sqrt(3)")
print("exact comparisons: 91/20<93/20<3pi/2<33/7<481/100; normal-chain threshold=12")

n = 12; P = np.roll(np.eye(n), 1, axis=1); Q = P-np.eye(n); pi = np.ones(n)/n
F = np.column_stack((np.cos(2*np.pi*np.arange(n)/n), np.sin(2*np.pi*np.arange(n)/n)))
T = 3*np.pi; a, b = 1-np.sqrt(3)/2, .5; z = a-1j*b
rect, bart = windows(Q, pi, F, T)
expected_rect = (a-.5*np.exp(-a*T))/(2*(a*a+.25))
expected_bart = np.real(1/z-(1-np.exp(-z*T))/(T*z*z))/2
close(rect[0, 0], expected_rect); close(bart[0, 0], expected_bart)
assert rect[0, 0] < -.013 and bart[0, 0] > .44
close(gk(Q, F.T)[0, 0], .25)
print(f"12-cycle: a={a:.12g}, b=1/2, T=3pi, K_f=1/4; R_T,f={rect[0,0]:.12g}; B_T,f={bart[0,0]:.12g}")
ir, er = quad_vec(lambda t: F.T@(pi[:, None]*(expm(t*Q)@F)), 0, T)
ib, eb = quad_vec(lambda t: (1-t/T)*F.T@(pi[:, None]*(expm(t*Q)@F)), 0, T)
close(ir, rect); close(ib, bart)
print(f"independent quadrature discrepancies: rectangular={np.linalg.norm(ir-rect):.3g}, Bartlett={np.linalg.norm(ib-bart):.3g}")

# Strictly positive off-diagonal circulant perturbation; same exact Fourier mode.
Qp = P-np.eye(n)+.001*(P.T-np.eye(n))+.001*(np.ones((n, n))/n-np.eye(n))
assert min(Qp[i, j] for i in range(n) for j in range(n) if i != j) > 0
ap, bp = 1.001*a+.001, .999*b; Tp = 3*np.pi/(2*bp)
# Rational lower bound on sqrt(3) certifies the perturbed sign too.
assert R(173205, 100000)**2 < 3 and upper < R(369, 100)
ap_upper = R(1001, 1000)*(1-R(173205, 200000))+R(1, 1000)
assert R(999, 2000) > R(369, 100)*ap_upper
assert R(369, 100)*R(13, 10) > R(33, 7)
rp, _ = windows(Qp, pi, F, Tp)
formula = (ap-bp*np.exp(-ap*Tp))/(2*(ap*ap+bp*bp))
close(rp[0, 0], formula); assert formula < -.007
print(f"positive-rate perturbation: forward extra=1, reverse extra=.001, uniform refresh=.001; R_T,f={formula:.12g}")
print("exact perturbed sign: b/a>369/100>exp(13/10); (369/100)*(13/10)>33/7>3pi/2")

rng = np.random.default_rng(20261020)
normal_cases = 0
for n in range(3, 15):
    for repeat in range(3):
        P = np.roll(np.eye(n), 1, axis=1); Q = np.zeros((n, n))
        for j in range(1, n):
            rate = (1. if j == 1 else 0.) if repeat == 0 else rng.uniform(.05, 1)
            Q += rate*(np.linalg.matrix_power(P, j)-np.eye(n))
        pi = np.ones(n)/n; F = rng.normal(size=(n, min(3, n-1))); F -= pi@F
        W = np.sqrt(pi)[:, None]*Q/np.sqrt(pi)[None, :]
        close(W@W.T, W.T@W)
        eigen = np.linalg.eigvals(Q); modes = eigen[eigen.real < -1e-7]
        kappa = max(abs(modes.imag)/(-modes.real))
        assert kappa <= 1/np.tan(np.pi/n)+3e-8
        for T in (.03, .3, 3., 30.):
            rect, bart = windows(Q, pi, F, T)
            G, A = (bart+bart.T)/2, (bart-bart.T)/2
            assert min(eigh(G, eigvals_only=True)) > 0
            for sign in [-1, 1]:
                assert min(eigh(kappa*G+sign*1j*A, eigvals_only=True)) >= -3e-8*(1+np.linalg.norm(G))
            if n <= 11:
                assert min(eigh((rect+rect.T)/2, eigvals_only=True)) >= -3e-8*(1+np.linalg.norm(rect))
            normal_cases += 1
print(f"circulant normal window/sector checks: {normal_cases}; states 3..14, four horizons")

# Weighted-normal, genuinely nonuniform stationary laws via product chains.
for repeat in range(12):
    prob = rng.uniform(.15, .85); pi2 = np.array([prob, 1-prob])
    Q2 = np.outer(np.ones(2), pi2)-np.eye(2)
    Q3 = np.roll(np.eye(3), 1, axis=1)-np.eye(3)
    Q = np.kron(Q2, np.eye(3))+np.kron(np.eye(2), Q3)
    pi = np.kron(pi2, np.ones(3)/3); close(stationary(Q), pi)
    W = np.sqrt(pi)[:, None]*Q/np.sqrt(pi)[None, :]; close(W@W.T, W.T@W)
    F = rng.normal(size=(6, 3)); F -= pi@F
    for T in (.1, 1., 10.):
        rect, bart = windows(Q, pi, F, T); G, A = (bart+bart.T)/2, (bart-bart.T)/2
        for sign in [-1, 1]:
            assert min(eigh(G/np.sqrt(3)+sign*1j*A, eigvals_only=True)) >= -3e-8*(1+np.linalg.norm(G))
        assert min(eigh((rect+rect.T)/2, eigvals_only=True)) >= -3e-8*(1+np.linalg.norm(rect))
print("nonuniform-stationary weighted-normal checks: 36; six-state product chains")

# Exploratory control for the separately open NONNORMAL extension, not a proof.
largest = 0.; checks = 0
for n in (3, 4, 5, 8):
    for repeat in range(12):
        flow = np.zeros((n, n))
        for _ in range(2*n):
            cycle = rng.choice(n, size=rng.integers(2, n+1), replace=False)
            flow[cycle, np.roll(cycle, -1)] += rng.uniform(.1, 2)
        flow[np.arange(n), np.roll(np.arange(n), -1)] += .1
        pi = rng.uniform(.5, 2, n); pi /= pi.sum()
        Q = flow/pi[:, None]; Q -= np.diag(Q.sum(axis=1))
        F = null_space(np.ones((1, n))); F -= pi@F
        K = gk(Q, F.T); G, A = (K+K.T)/2, (K-K.T)/2
        limit_sector = max(eigh(1j*A, G, eigvals_only=True))
        if limit_sector < 1e-7: continue
        for T in (.05, .5, 5., 50.):
            _, bart = windows(Q, pi, F, T/max(-np.diag(Q)))
            G, A = (bart+bart.T)/2, (bart-bart.T)/2
            finite_sector = max(eigh(1j*A, G, eigvals_only=True))
            assert finite_sector <= limit_sector+3e-8*(1+limit_sector)
            largest = max(largest, finite_sector/limit_sector); checks += 1
print(f"exploratory nonnormal checks={checks}; max Bartlett/long-run sector={largest:.9g}; no general theorem asserted")
print("PASS")
