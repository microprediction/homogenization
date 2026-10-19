"""Sparse finite-chain, all-prior/all-horizon constant-fraction curvature checks."""
from fractions import Fraction as F
from math import factorial, ceil
import numpy as np
from numpy.linalg import eigvals, solve, norm
from scipy.linalg import expm


def envelope(Q, theta=1.):
    # Nonnegative uniformization truncation plus a geometric Poisson-tail bound.
    # For general floating inputs this is a consistency calculation, not enclosure.
    n, rate = len(Q), max(-np.diag(Q))
    x, R = rate*theta, np.eye(n)+Q/rate
    degree = max(n-1, ceil(x)+2)
    power, S = np.eye(n), np.eye(n)
    for k in range(1, degree+1):
        power = power@R
        S += x**k/factorial(k)*power
    tail = x**(degree+1)/factorial(degree+1)/(1-x/(degree+2))
    a0 = min(S.min(axis=0)/(S.max(axis=0)+tail))
    P = expm(theta*Q)
    assert min(S.ravel()) > 0
    assert np.min(P-np.exp(-x)*S) >= -2e-12
    assert np.max(P-np.exp(-x)*(S+tail)) <= 2e-12
    assert a0 <= min(P.min(axis=0)/P.max(axis=0))+2e-12
    return a0


def constants(b, s, gamma):
    lo, hi = min(b/(gamma*s)), max(b/(gamma*s))
    B = max(np.ptp(b-gamma*w*s) for w in [lo, hi])
    points = [lo, hi]
    for i in range(len(b)):
        for j in range(i):
            if s[i] != s[j]:
                w = (b[i]-b[j])/(gamma*(s[i]-s[j]))
                if lo <= w <= hi: points.append(w)
    delta = (1-gamma)*max(np.ptp(b*w-gamma*s*w*w/2) for w in points)
    return lo, hi, B, delta


def moments(A, q, s, gamma, T, priors):
    n, p = len(q), 1-gamma
    M = np.kron(np.eye(4), A)
    M[n:2*n,:n], M[2*n:3*n,n:2*n], M[3*n:,:n] = np.diag(q), 2*np.diag(q), np.diag(s)
    z, er, er2, ev = (priors@(expm(T*M)@np.r_[np.ones(n),np.zeros(3*n)]).reshape(4,n).T).T
    var = er2/z-(er/z)**2
    curve = -gamma*ev/z/T+p*var/T
    D = np.kron(np.eye(3), A)
    D[n:2*n,:n], D[2*n:,:n], D[2*n:,n:2*n] = p*np.diag(q), -p*gamma*np.diag(s), 2*p*np.diag(q)
    mass, dz, ddz = (priors@(expm(T*D)@np.r_[np.ones(n),np.zeros(2*n)]).reshape(3,n).T).T
    tol = 3e-9*(1+max(abs(er2/z))/T)
    assert max(abs(curve-(ddz/mass-(dz/mass)**2)/(p*T))) <= tol
    return er/z/T, curve, var, tol


Q0 = np.roll(np.eye(3),1,axis=1)-np.eye(3)
b0, s0, gamma = np.array([.12,-.08,.05]), np.array([.04,.09,.0625]), .5
# Exact uniformization witness: I+F+F^2/2 has entries 1, 1, 1/2.
tail = F(1,6)/(1-F(1,4))
a0 = F(1,2)/(1+tail)
assert tail == F(2,9) and a0 == F(9,22)
br, sr = [F(3,25),-F(2,25),F(1,20)], [F(1,25),F(9,100),F(1,16)]
ends = [min(2*x/y for x,y in zip(br,sr)),max(2*x/y for x,y in zip(br,sr))]
assert ends == [-F(16,9),F(6)]
B = max(max(br[i]-w*sr[i]/2 for i in range(3))-min(br[i]-w*sr[i]/2 for i in range(3)) for w in ends)
points = ends.copy()
for i in range(3):
    for j in range(i):
        w = 2*(br[i]-br[j])/(sr[i]-sr[j])
        if ends[0] <= w <= ends[1]: points.append(w)
D = max(max(br[i]*w-sr[i]*w*w/4 for i in range(3))-min(br[i]*w-sr[i]*w*w/4 for i in range(3)) for w in points)
d, C = D/2, B*B/(4*a0*F(1,50))
assert (B,D,d,C,d+C) == (F(7,20),F(33,20),F(33,40),F(539,144),F(3289,720))
assert d+C < 5 and F(1,50)*(1-C/(5-d)) == F(311,150300) > 0
assert d+2*C == F(374,45) < 9
print('directed-cycle Q=',Q0,'b=',b0,'s=',s0,'gamma=1/2, theta=1')
print('exact: tail=2/9, a0=9/22, I=[-16/9,6], B=7/20, D=33/20')
print('d=33/40, C=539/144, d+C=3289/720<5; m>=5 modulus>=311/150300')
print('d+2C=374/45<9; m>=9 modulus>=1/100')

rng = np.random.default_rng(216)
models = [(Q0,b0,s0,gamma,float(a0))]
for n in [4,5,6]:
    Q = np.roll(np.diag(rng.uniform(.5,1.5,n)),1,axis=1)
    Q -= np.diag(Q.sum(axis=1))
    models.append((Q,rng.uniform(-.1,.15,n),rng.uniform(.04,.1,n),.5,envelope(Q)))
Q = np.diag([1.,2.,3.],1)+np.diag([2.,1.,1.],-1)
Q -= np.diag(Q.sum(axis=1))
models.append((Q,np.array([.12,-.08,.05,.02]),np.array([.04,.09,.0625,.07]),.6,envelope(Q)))
N = np.outer([1.,-1.,0.],[1.,1.,-2.])/20
assert norm(N@N) < 2e-14
Q = np.ones((3,3))/3-np.eye(3)+N
models.append((Q,b0,s0,.5,envelope(Q)))
cases, utilization, contraction = 0, 0., 0.
for model, (Q,b,s,gamma,a0) in enumerate(models):
    n, p = len(b), 1-gamma
    priors = np.vstack([np.eye(n),solve(Q.T+np.ones((n,n)),np.ones(n))])
    lo,hi,B,d = constants(b,s,gamma)
    C = p*B*B/(2*a0*gamma*min(s))
    speeds = [max(.1,.5*(d+C)),max(5.,1.2*(d+C))]+([5.] if model == 0 else [])
    for m in speeds:
        tau, alpha = 1/m, a0*np.exp(-d/m)
        modulus = gamma*min(s)-p*B*B*tau/(2*alpha)
        if m > d:
            assert modulus >= gamma*min(s)*(1-C/(m-d))-2e-12
        if m > d+C: assert modulus > 0
        for w in np.linspace(lo,hi,5):
            q = b-gamma*w*s
            A = m*Q+np.diag(p*(b*w-gamma*s*w*w/2))
            A -= max(eigvals(A).real)*np.eye(n)  # Hold shift fixed in derivative blocks.
            # Arbitrary positive terminal weights: no bounded future ratio is assumed.
            H = expm(tau*A)
            for g in [np.ones(n),np.exp(np.linspace(-12,12,n))]:
                K = H*g[None,:]/(H@g)[:,None]
                v = expm(Q).max(axis=0)
                mu = v*g/sum(v*g)
                assert np.min(K-alpha*mu[None,:]) >= -3e-10
            for T in [1e-4,.01,1.,10.,100.,1e4]:
                score,curve,var,tol = moments(A,q,s,gamma,T,priors)
                budget = min(B*B*T/4,B*B*tau/(2*alpha))
                assert max(var/T) <= budget+tol and max(curve) <= -modulus+tol
                if w == lo: assert min(score) >= -tol
                if w == hi: assert max(score) <= tol
                utilization = max(utilization,max(var/T)/budget)
                lag = min(T/3,2.25*tau)
                g = expm(A*(T/3))@np.ones(n)
                H = expm(lag*A)
                K = H*g[None,:]/(H@g)[:,None]
                beta = max(sum(abs(K[i]-K[j]))/2 for i in range(n) for j in range(n))
                bound = (1-alpha)**int(np.floor(lag/tau))
                assert max(abs(K.sum(axis=1)-1)) < 3e-10
                assert beta <= bound+3e-10
                if lag >= tau: contraction = max(contraction,beta/bound)
                cases += len(priors)
print('models=',len(models),'prior/weight/speed/horizon cases=',cases,'T=1e-4..1e4')
print('max variance/budget=',utilization,'max block contraction/bound=',contraction)
print('PASS')
