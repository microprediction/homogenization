"""Horizon-uniform curvature from the finite-horizon Doob chain, not exp(T osc h)."""
from fractions import Fraction as F
import numpy as np
from numpy.linalg import eigvals, eig, solve, norm
from scipy.linalg import expm
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss


def data(Q, b, s, gamma):
    n, p = len(b), 1-gamma
    lo, hi = np.min(b/(gamma*s)), np.max(b/(gamma*s))
    B = max(np.ptp(b-gamma*w*s) for w in [lo, hi])
    points = [lo, hi]
    for i in range(n):
        for j in range(i):
            if s[i] != s[j]:
                w = (b[i]-b[j])/(gamma*(s[i]-s[j]))
                if lo <= w <= hi:
                    points.append(w)
    delta = p*max(np.ptp(b*w-gamma*s*w*w/2) for w in points)
    qmin = min(Q[~np.eye(n, dtype=bool)])
    c = p*B*B/(2*n*gamma*min(s))
    return lo, hi, B, delta, qmin, np.sqrt(c*(c+delta))/qmin


def matrices(Q, b, s, gamma, m, w):
    p = 1-gamma
    q, V = b-gamma*w*s, p*(b*w-gamma*s*w*w/2)
    A = m*Q+np.diag(V)
    # A common scalar shift prevents overflow; derivative blocks hold it fixed.
    shift = max(eigvals(A).real)
    return A-shift*np.eye(len(b)), q


def moments(A, q, s, gamma, T, priors):
    n, p = len(q), 1-gamma
    M = np.kron(np.eye(4), A)
    M[n:2*n,:n], M[2*n:3*n,n:2*n] = np.diag(q), 2*np.diag(q)
    M[3*n:,:n] = np.diag(s)
    mass, er, er2, ev = (priors @ (expm(T*M) @ np.r_[np.ones(n),np.zeros(3*n)]).reshape(4,n).T).T
    var = er2/mass-(er/mass)**2
    curve = -gamma*ev/mass/T+p*var/T
    D = np.kron(np.eye(3), A)
    D[n:2*n,:n], D[2*n:,:n], D[2*n:,n:2*n] = p*np.diag(q), -p*gamma*np.diag(s), 2*p*np.diag(q)
    z, dz, ddz = (priors @ (expm(T*D) @ np.r_[np.ones(n),np.zeros(2*n)]).reshape(3,n).T).T
    other = (ddz/z-(dz/z)**2)/(p*T)
    # Long-horizon raw moments can cancel; allow their scale, not a sub-ULP test.
    tol = 2e-9*(1+np.max(abs(er2/mass))/T)
    assert max(abs(curve-other)) <= tol
    return er/mass/T, curve, var


Q = np.array([[-3,2,1],[1,-2,1],[.5,1.5,-2.]])
b, s, gamma = np.array([.12,-.08,.05]), np.array([.04,.09,.0625]), .5
lo, hi, B, delta, qmin, critical = data(Q,b,s,gamma)
# Rational benchmark: a=m*qmin>=1, delta=33/40, B=7/20.
c = F(49,96)
assert c*(c+F(33,40))/F(1,2)**2 == F(31409,11520) < 4
assert F(3,2)**2-1-F(33,40)*F(3,2) == F(1,80) > 0
kap = F(1,50)-F(1,2)*F(7,20)**2/4
assert kap == F(3,640) > 0
brat, srat = [F(3,25),-F(2,25),F(1,20)], [F(1,25),F(9,100),F(1,16)]
ends = [-F(16,9), F(6)]
assert max(max(brat[i]-w*srat[i]/2 for i in range(3))-min(brat[i]-w*srat[i]/2 for i in range(3)) for w in ends) == F(7,20)
points = ends.copy()
for i in range(3):
    for j in range(i):
        w = 2*(brat[i]-brat[j])/(srat[i]-srat[j])
        if ends[0] <= w <= ends[1]: points.append(w)
assert max(max(brat[i]*w-srat[i]*w*w/4 for i in range(3))-min(brat[i]*w-srat[i]*w*w/4 for i in range(3)) for w in points) == F(33,20)
assert abs(delta-float(F(33,40))) < 2e-12 and abs(B-float(F(7,20))) < 2e-12
print('benchmark Q=', Q, 'b=', b, 's=', s, 'gamma=1/2; I=[-16/9,6]')
print('qmin=1/2, B=7/20, D=33/20, delta=33/40, c=49/96')
print('threshold squared=31409/11520; threshold=', critical, '<2')
print('polynomial at R=3/2: 1/80; at every m>=2: R<3/2, reset rate k>2, curvature modulus>=', kap)

rng = np.random.default_rng(215)
models = [(Q,b,s,gamma)]
for n in [2,3,4,5]:
    rates = rng.uniform(.2,1.5,(n,n)); np.fill_diagonal(rates,0)
    models.append((rates-np.diag(rates.sum(axis=1)),rng.uniform(-.1,.15,n),
                   rng.uniform(.06,.12,n),rng.uniform(.35,.8)))
N = np.outer([1.,-1.,0.],[1.,1.,-2.])/20
assert norm(N@N) < 1e-14
models.append((np.ones((3,3))/3-np.eye(3)+N,b,s,gamma))
cases, ratio, hratio, value_ratio = 0, 0., 0., 0.
for Q,b,s,gamma in models:
    n, p = len(b), 1-gamma
    pi = solve(Q.T+np.ones((n,n)),np.ones(n))
    priors = np.vstack([np.eye(n),pi])
    lo,hi,B,delta,qmin,critical = data(Q,b,s,gamma)
    for m in [max(.1,.5*critical),max(2.,1.2*critical)]:
        a = m*qmin
        R = (delta+np.hypot(delta,2*a))/(2*a)
        rate = n*a/R
        modulus = gamma*min(s)-p*B*B/(2*rate)
        for T in [1e-4,.01,1.,10.,1e2,1e4]:
            for w in np.linspace(lo,hi,7):
                A,q = matrices(Q,b,s,gamma,m,w)
                score,curve,var = moments(A,q,s,gamma,T,priors)
                # G<=T/2 avoids cancellation at tiny rate*T, conservatively.
                G = (T/2 if rate*T < 1e-5 else 1/rate+np.expm1(-rate*T)/(T*rate*rate))
                budget = B*B*T*G/2
                tol = 2e-9*(1+T*np.max(abs(q))**2)
                assert max(var/T) <= budget/T+tol
                assert max(curve) <= -modulus+tol
                if w == lo: assert min(score) >= -tol
                if w == hi: assert max(score) <= tol
                if m > critical:
                    assert max(curve) < 0
                ratio = max(ratio,max(var)/budget)
                u = expm(A*T) @ np.ones(n)
                logmass = np.log(priors@u)
                assert max(abs(logmass)) <= np.log(R)+2e-8
                value_ratio = max(value_ratio,max(abs(logmass))/np.log(R))
                hratio = max(hratio,max(u)/min(u)/R)
                assert max(u)/min(u) <= R+2e-8
                # Exact canonical transition, tested at nontrivial reset-clock lags.
                t0, lag = T/3,min(T/3,1/rate)
                u0, u1 = expm(A*(T-t0))@np.ones(n), expm(A*(T-t0-lag))@np.ones(n)
                K = expm(A*lag)*u1[None,:]/u0[:,None]
                rates = m*Q*u1[None,:]/u1[:,None]
                assert min(rates[~np.eye(n,dtype=bool)]) >= a/R-2e-8*(1+m*norm(Q,np.inf)*R)
                assert max(abs(K.sum(axis=1)-1)) < 2e-8
                beta = max(np.sum(abs(K[i]-K[j]))/2 for i in range(n) for j in range(n))
                assert beta <= np.exp(-rate*lag)+2e-8
                cases += len(priors)

# Independent ordered-time covariance quadrature for a pure starting state.
Q,b,s,gamma = models[0]
A,q = matrices(Q,b,s,gamma,2.,.8)
T, prior = .7,np.array([[1.,0.,0.]])
Z = (prior @ expm(A*T) @ np.ones(3)).item()
nodes, weights = leggauss(24)
answer = 0.
def mean(t):
    return (prior @ expm(A*t) @ np.diag(q) @ expm(A*(T-t)) @ np.ones(3)).item()/Z
for x,wx in zip(nodes,weights):
    t = T*(x+1)/2
    for y,wy in zip(nodes,weights):
        r = t*(y+1)/2
        joint = (prior @ expm(A*r) @ np.diag(q) @ expm(A*(t-r)) @ np.diag(q) @ expm(A*(T-t)) @ np.ones(3)).item()/Z
        answer += wx*wy*T*t/2*(joint-mean(r)*mean(t))
var = moments(A,q,s,gamma,T,prior)[2][0]
assert abs(answer-var) < 2e-9
print('models=',len(models),'prior/weight/speed/horizon cases=',cases,'T range=1e-4..1e4')
print('largest variance/budget=',ratio,'largest Harnack ratio/bound=',hratio)
print('largest centered log-value error / log R=',value_ratio)
print('independent tilted covariance quadrature error=',abs(answer-var))
lo,hi,B,delta,qmin,_ = data(Q,b,s,gamma)
def limiting_score(w):
    A,q = matrices(Q,b,s,gamma,2.,w)
    ev,v = eig(A); el,l = eig(A.T)
    weights = v[:,np.argmax(ev.real)].real*l[:,np.argmax(el.real)].real
    return weights@q/sum(weights)
winf = brentq(limiting_score,lo,hi)
R = (delta+np.hypot(delta,2.))/2
for T in [1.,100.,1e4]:
    for prior in np.eye(3):
        root = brentq(lambda w: moments(*matrices(Q,b,s,gamma,2.,w),s,gamma,T,prior[None,:])[0][0],lo,hi)
        assert abs(root-winf) <= 2*np.sqrt(np.log(R)/((1-gamma)*float(kap)*T))+2e-8
print('Perron optimizer=',winf,'and 9 optimizer-distance checks pass')
print('PASS')
