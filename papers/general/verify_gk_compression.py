"""Real-feature Green--Kubo feasibility: interlacing and one-way cycle counts."""
import numpy as np
from fractions import Fraction
from scipy.linalg import null_space, block_diag, expm
from scipy.integrate import quad_vec
from numpy.linalg import eigh, norm, inv, matrix_rank
from effective_generator import gk

J = np.array([[0.,1.],[-1.,0.]])


def canonical(B):
    B = (B-B.T)/2
    values,V = eigh(1j*B)
    positive = np.flatnonzero(values > 2e-9*(1+norm(B)))
    columns = []
    for j in positive[::-1]: columns.extend([np.sqrt(2)*V[:,j].real,np.sqrt(2)*V[:,j].imag])
    columns.extend(null_space(B,rcond=2e-9*(1+norm(B))/max(norm(B),1e-300)).T)
    P = np.array(columns).T
    assert norm(P.T@P-np.eye(len(B))) < 3e-7
    return values[positive][::-1],P


def real_compression(B,target):
    """Strict/simple-spectrum residue construction; limits cover degeneracies in the proof."""
    m,d = len(B),len(target); W = np.eye(m); current = (B-B.T)/2
    target = (target-target.T)/2
    mu = eigh(1j*target)[0][::-1]
    while len(current) > d:
        size = len(current); lam = eigh(1j*current)[0][::-1]
        gap = size-1-d
        lower,upper = lam[1:].copy(),lam[:-1].copy()
        lower[:d] = np.maximum(lower[:d],mu)
        upper[gap:gap+d] = np.minimum(upper[gap:gap+d],mu)
        assert min(upper-lower) > -3e-8*(1+norm(current))
        row = (lower+upper)/2; row = (row-row[::-1])/2
        sigma,P = canonical(current); tau = row[:(size-1)//2]
        weights = []
        for j,s in enumerate(sigma):
            denominator = np.prod(np.delete(sigma**2-s*s,j))
            if size%2: denominator *= -s*s
            weights.append(np.prod(tau*tau-s*s)/denominator)
        if size%2: weights.append(np.prod(tau*tau)/np.prod(sigma*sigma))
        assert min(weights) > -3e-7 and abs(sum(weights)-1) < 3e-7
        weights = np.maximum(weights,0); weights /= sum(weights)
        normal = np.zeros(size)
        for j,w in enumerate(weights[:len(sigma)]): normal += np.sqrt(w)*P[:,2*j]
        if size%2: normal += np.sqrt(weights[-1])*P[:,-1]
        V = null_space(normal[None,:]); W = W@V; current = V.T@current@V
        current = (current-current.T)/2
        assert norm(eigh(1j*current)[0][::-1]-row) < 3e-7*(1+norm(B))
    _,P = canonical(current); _,R = canonical(target)
    W = W@P@R.T
    assert norm(W.T@W-np.eye(d)) < 3e-7 and norm(W.T@B@W-target) < 3e-7*(1+norm(B))
    return W


def whiten(C):
    T = null_space(np.ones((1,len(C))))
    M = T.T@C.T@T; H,D = (M+M.T)/2,(M-M.T)/2
    values,V = eigh(H); inverse = (V/np.sqrt(values))@V.T
    return T@inverse,inverse@D@inverse


def direct(Q,pi,F):
    P = np.outer(np.ones(len(pi)),pi)
    return F.T@(pi[:,None]*((-inv(Q+P)+P)@F))


def interlaces(lam,mu,tol=3e-8):
    d = len(mu)
    return len(lam) >= d and min(lam[:d]-mu) >= -tol and min(mu-lam[-d:]) >= -tol


# Exact even/odd residue fixtures; clearing denominators leaves polynomials in t^2 of degree <=2.
even = [Fraction(5,8),Fraction(3,8)]
odd = [Fraction(175,288),Fraction(9,32),Fraction(1,9)]
assert sum(even) == sum(odd) == 1 and min(even+odd) > 0
for t in map(Fraction,[1,2,3,4]):
    assert t*(t*t+4)/((t*t+9)*(t*t+1)) == even[0]*t/(t*t+9)+even[1]*t/(t*t+1)
    assert (t*t+4)*(t*t+Fraction(1,4))/(t*(t*t+9)*(t*t+1)) == odd[0]*t/(t*t+9)+odd[1]*t/(t*t+1)+odd[2]/t
print('exact partial-fraction weights, even:',list(map(str,even)),'odd:',list(map(str,odd)))

rng = np.random.default_rng(20261020)
cases,max_error = 0,0.
for N in range(2,13):
    flow = np.zeros((N,N))
    for _ in range(3*N):
        cycle = rng.choice(N,size=rng.integers(2,N+1),replace=False)
        flow[cycle,np.roll(cycle,-1)] += rng.uniform(.2,2.)
    flow[np.arange(N),np.roll(np.arange(N),-1)] += .5
    C = np.diag(flow.sum(axis=1))-flow
    E,B = whiten(C); lam = eigh(1j*B)[0][::-1]
    for d in range(1,N):
        V,_ = np.linalg.qr(rng.normal(size=(N-1,d)))
        targetB = V.T@B@V
        mu = eigh(1j*targetB)[0][::-1]
        assert interlaces(lam,mu)
        W = real_compression(B,targetB)
        X = rng.normal(size=(d,d)); G = X.T@X+np.eye(d)
        values,R = eigh(G); root = (R*np.sqrt(values))@R.T
        U = E@W@root; wanted = G+root@targetB@root
        for repeat in range(2):
            pi = rng.uniform(.5,2.,size=N); pi /= sum(pi)
            Q = -C/pi[:,None]; F = -Q@U
            actual = direct(Q,pi,F)
            error = norm(actual-wanted)/(1+norm(wanted)); max_error = max(max_error,error)
            assert error < 3e-7 and norm(gk(Q,F.T)-wanted) < 3e-7*(1+norm(wanted))
            assert norm(pi@Q) < 3e-8*(1+norm(Q)) and norm(pi@F) < 3e-8*(1+norm(F))
            assert matrix_rank(F) == d
            assert interlaces(-lam[::-1],mu)  # Reversal leaves the embedding class unchanged.
            cases += 1
print('real-feature constructions:',cases,'dimensions 1..11; state counts 2..12; two prescribed laws per target')
print('maximum normalized Green--Kubo discrepancy:',max_error)

# Every signed mode matters: counts are exact in the theorem, floating here only for interior checks.
cycle_tests = 0
for N in range(2,16):
    lam = 1/np.tan(np.pi*np.arange(1,N)/N)
    for d in range(1,N):
        values = rng.uniform(.05,1.,size=d//2)
        B = block_diag(*[k*J for k in values],np.zeros((d%2,d%2)))
        mu = eigh(1j*B)[0][::-1]
        threshold = max(d+1,max(np.arange(1,d+1)*np.pi/np.arctan2(np.ones(d),mu)))
        predicted = int(np.ceil(threshold))
        assert interlaces(1/np.tan(np.pi*np.arange(1,predicted)/predicted),mu)
        if predicted > d+1:
            assert not interlaces(1/np.tan(np.pi*np.arange(1,predicted-1)/(predicted-1)),mu,1e-10)
        assert interlaces(lam,mu) == (N >= predicted)
        cycle_tests += 1
print('one-way cycle signed-spectrum/count comparisons:',cycle_tests)

# Explicit ten-state witness for the two-strong-mode target from issue #222.
N = 10; k = np.arange(N); high = np.sqrt(1+2/np.sqrt(5))
potentials = []
for mode in [1,2]:
    theta = 2*np.pi*mode/N; scale = np.sqrt(N*(1-np.cos(theta))/2)
    potentials.extend([np.cos(k*theta)/scale,np.sin(k*theta)/scale])
U = np.array(potentials).T; ratio = high/(1/np.tan(np.pi/10))
U[:,0] = ratio*U[:,0]+np.sqrt(1-ratio*ratio)*(-1.)**k/np.sqrt(2*N)
wanted = block_diag(np.eye(2)+high*J,np.eye(2)+high*J)
assert Fraction(2,9) > Fraction(1,5)  # For N<=9, lambda_2=cot(2*pi/N)<high.
for repeat in range(12):
    pi = rng.uniform(.5,2.,size=N); pi /= sum(pi)
    P = np.roll(np.eye(N),1,axis=1); C = np.eye(N)-P; Q = -C/pi[:,None]; F = -Q@U
    assert norm(direct(Q,pi,F)-wanted) < 3e-7
    assert norm(gk(Q,F.T)-wanted) < 3e-7
E,B = whiten(C)
assert norm(eigh(1j*B)[0][::-1]-1/np.tan(np.pi*np.arange(1,N)/N)) < 3e-7
gap = eigh(-(np.sqrt(pi)[:,None]*Q/np.sqrt(pi)[None,:]+np.sqrt(pi)[None,:]*Q.T/np.sqrt(pi)[:,None])/2)[0][1]
endpoint = 30/gap
integral,_ = quad_vec(lambda t: F.T@(pi[:,None]*(expm(t*Q)@F)),0,endpoint,epsabs=1e-9)
tail = 2*norm(np.sqrt(pi)[:,None]*F,2)**2*np.exp(-gap*endpoint)/gap
assert norm(integral-wanted) < tail+3e-7*(1+norm(wanted))
print('K*=I4+cot(pi/5)*diag(J,J): one-way-cycle minimum=10; twelve holding-time witnesses; quadrature error=',norm(integral-wanted))
print('N=9 second mode cot(2*pi/9)=',1/np.tan(2*np.pi/9),'target=',high,'; exact angle comparison 2/9>1/5')

for N in range(2,13):
    P = np.roll(np.eye(N),1,axis=1); E,B = whiten(np.eye(N)-P)
    sigma,R = canonical(B); rank_half = len(sigma); d = N-1-rank_half
    W = np.column_stack([R[:,2*j] for j in range(rank_half)]+list(R[:,2*rank_half:].T))
    assert W.shape[1] == d and norm(W.T@B@W) < 3e-7
    if d < N-1: assert not interlaces(eigh(1j*B)[0][::-1],np.zeros(d+1),1e-10)
print('maximal reversible-looking feature dimensions on one-way cycles N=2..12:',[N//2 for N in range(2,13)])
print('PASS')
