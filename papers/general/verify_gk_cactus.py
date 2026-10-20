"""Cactus block spectra, higher-mode capacities, and exact repeated-mode counts."""
from fractions import Fraction as R
import numpy as np
from scipy.linalg import block_diag, eigh, expm, null_space
from scipy.integrate import quad_vec
from effective_generator import gk


def close(a, b):
    assert np.linalg.norm(np.asarray(a, float)-b) <= 3e-8*(1+np.linalg.norm(b))


def spectrum(M):
    return eigh(1j*(M-M.T)/2, (M+M.T)/2, eigvals_only=True)[::-1]


def cactus(specs, exact=False):
    """Attach (root,length,current,reverse-edge-flows); length two is a bridge."""
    dtype = object if exact else float
    flow, U, blocks = np.zeros((1, 1), dtype=dtype), np.zeros((1, 0), dtype=dtype), []
    for root, length, current, reverse in specs:
        n, m = U.shape; vertices = [root, *range(n, n+length-1)]
        flow = np.pad(flow, ((0, length-1), (0, length-1)))
        U = np.vstack((U, np.tile(U[root], (length-1, 1))))
        U = np.column_stack((U, np.zeros((len(U), length-1), dtype=dtype)))
        U[n:, m:] = np.eye(length-1, dtype=dtype)
        local = np.zeros((length, length), dtype=dtype)
        edges = [(0, 1)] if length == 2 else [(k, (k+1) % length) for k in range(length)]
        for k, (i, j) in enumerate(edges):
            a, b = reverse[k]+max(current, 0), reverse[k]+max(-current, 0)
            local[i, j] += a; local[j, i] += b
            flow[vertices[i], vertices[j]] += a; flow[vertices[j], vertices[i]] += b
        C = np.diag(local.sum(axis=1))-local
        blocks.append(C[1:, 1:].T)
    return flow, U, blocks


def generator(flow, pi):
    Q = flow/pi[:, None]
    return Q-np.diag(Q.sum(axis=1))


rng = np.random.default_rng(20261020)
for trial in range(72):
    specs, n, lengths = [], 1, []
    for _ in range(1+trial % 4):
        length = int(rng.integers(2, 10))
        current = 0 if length == 2 else rng.choice([-1, 1])*rng.uniform(.3, 2)
        reverse = rng.uniform(.1, 2, size=1 if length == 2 else length)
        specs.append((int(rng.integers(n)), length, current, reverse)); lengths.append(length)
        n += length-1
    flow, U, blocks = cactus(specs)
    C = np.diag(flow.sum(axis=1))-flow
    close(U.T@C.T@U, block_diag(*blocks))
    lam = spectrum(U.T@C.T@U)
    close(lam, np.sort(np.concatenate([spectrum(M) for M in blocks]))[::-1])
    for length, M in zip(lengths, blocks):
        if length == 2:
            close(spectrum(M), np.zeros(1)); continue
        pos = spectrum(M)[:(length-1)//2]
        assert np.all(pos <= 1/np.tan(np.pi*np.arange(1, len(pos)+1)/length)+3e-8)
    for q in range(3, 9):
        kappa = 1/np.tan(np.pi/q)
        assert sum(lam >= kappa-3e-8) <= sum(length//q for length in lengths if length >= 3)
    pi = rng.uniform(.5, 2, n); pi /= pi.sum(); Q = generator(flow, pi)
    F = -Q@U; close(gk(Q, F.T), block_diag(*blocks))
    T = null_space(np.ones((1, n)))
    close(spectrum(T.T@C.T@T), lam)
print("arbitrary cactus attachments: 72; variable reverse traffic, both orientations, lengths 2..9")

# Exact non-bouquet example: triangles at different articulations, then a bridge.
specs = [(0, 3, R(2), [R(1), R(2), R(3)]),
         (2, 3, R(-1), [R(2), R(1), R(4)]), (4, 2, R(0), [R(3)])]
flow, U, blocks = cactus(specs, exact=True)
C = np.diag(flow.sum(axis=1))-flow
assert np.array_equal(U.T@C.T@U, block_diag(*blocks))
pi = np.array([R(i, 21) for i in range(1, 7)], dtype=object)
Q = generator(flow, pi); F = -Q@U
assert all(sum(pi[i]*Q[i, j] for i in range(6)) == 0 for j in range(6))
assert np.array_equal(F.T@(pi[:, None]*U), block_diag(*blocks))
print("exact rational: six-state triangle-chain-bridge; grounded K=", block_diag(*blocks).tolist())

J = np.array([[0., 1.], [-1., 0.]])
for q in range(3, 9):
    for r in range(1, 5):
        for endpoint in (False, True):
            cap = 1/np.tan(np.pi/q)
            lower = 0 if q == 3 else 1/np.tan(np.pi/(q-1))
            kappa = cap if endpoint else .6*cap+.4*lower
            total = 2/(q*(1-np.cos(2*np.pi/q)))
            reverse = max(0., (total-2*kappa/(q*np.sin(2*np.pi/q)))/2)
            specs = [(0, q, 2*kappa/(q*np.sin(2*np.pi/q)), np.full(q, reverse)) for _ in range(r)]
            flow, _, _ = cactus(specs); n = len(flow)
            assert n == 1+r*(q-1)
            U = np.zeros((n, 2*r))
            for j in range(r):
                theta = 2*np.pi*np.arange(1, q)/q
                U[1+j*(q-1):1+(j+1)*(q-1), 2*j:2*j+2] = np.column_stack((np.cos(theta)-1, np.sin(theta)))
            pi = rng.uniform(.5, 2, n); pi /= pi.sum(); Q = generator(flow, pi)
            O, _ = np.linalg.qr(rng.normal(size=(2*r, 2*r)))
            X = rng.normal(size=(2*r, 2*r)); G = X.T@X+np.eye(2*r)
            values, V = eigh(G); root = O.T@(V*np.sqrt(values))@V.T
            F = -Q@U@root; target = root.T@block_diag(*[np.eye(2)+kappa*J]*r)@root
            close(gk(Q, F.T), target)
            P = np.outer(np.ones(n), pi)
            close(F.T@(pi[:, None]*np.linalg.solve(-Q+P, F)), target)
print("matching repeated-mode realizations: 48; r=1..4, q=3..8, interior and one-way endpoints")

# Integer dynamic program over all multisets of cycle lengths; bridges only use budget.
for q in range(3, 9):
    budget = 4*(q-1)
    best = [0]*(budget+1)
    for b in range(1, budget+1):
        best[b] = max([best[b-1]]+[best[b-(length-1)]+length//q for length in range(3, b+2)])
    for r in range(1, 5):
        assert best[r*(q-1)-1] < r and best[r*(q-1)] == r
    if q == 5:
        print("q=5 capacity by N-1 budget 0..8:", best[:9])
print("K*=I4+cot(pi/5)*diag(J,J): r=2, q=5, exact cactus minimum=9; unrestricted 6..9 remains open")

# Independent correlation quadrature on the exact rational non-bouquet fixture.
flow, U, blocks = cactus([(a, b, float(c), list(map(float, d))) for a, b, c, d in
                         [(0, 3, 2, [1, 2, 3]), (2, 3, -1, [2, 1, 4]), (4, 2, 0, [3])]])
pi = np.arange(1., 7)/21; Q = generator(flow, pi); F = -Q@U
weighted = np.sqrt(pi)[:, None]*Q/np.sqrt(pi)[None, :]
gap = eigh(-(weighted+weighted.T)/2, eigvals_only=True)[1]; end = 30/gap
value, error = quad_vec(lambda t: F.T@(pi[:, None]*(expm(t*Q)@F)), 0, end)
tail = np.sqrt(5)*np.linalg.norm(np.sqrt(pi)[:, None]*F, 2)**2*np.exp(-30)/gap
assert np.linalg.norm(value-block_diag(*blocks)) <= tail+error+3e-8*(1+np.linalg.norm(value))
print(f"correlation quadrature: discrepancy={np.linalg.norm(value-block_diag(*blocks)):.3g}, tail<={tail:.3g}")
print("PASS")
