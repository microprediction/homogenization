"""Two-scale optimizer bounds and the T=c/m initial-prior profile."""
from fractions import Fraction as F
from math import factorial
import numpy as np
from numpy.linalg import solve, eigvals, inv, norm
from scipy.linalg import expm, expm_frechet
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss


def data(Q, b, s, gamma, mixing):
    n = len(b)
    pi = solve(Q.T+np.ones((n,n)),np.ones(n))
    R = inv(Q+np.outer(np.ones(n),pi))-np.outer(np.ones(n),pi)
    lo,hi = min(b/(gamma*s)),max(b/(gamma*s))
    B = max(np.ptp(b-gamma*w*s) for w in [lo,hi])
    points = [lo,hi]
    for i in range(n):
        for j in range(i):
            if s[i] != s[j]:
                w = (b[i]-b[j])/(gamma*(s[i]-s[j]))
                if lo <= w <= hi: points.append(w)
    delta = abs(1-gamma)*max(np.ptp(b*w-gamma*s*w*w/2) for w in points)
    span, flux = 0., 0.
    for w in [lo,hi]:
        q,psi = b-gamma*w*s,R@(b-gamma*w*s)
        assert max(abs(Q@psi-q+pi@q)) < 2e-12 and abs(pi@psi) < 2e-12
        span = max(span,np.ptp(psi))
        flux = max(flux,max((Q-np.diag(np.diag(Q)))*abs(psi[None,:]-psi[:,None])@np.ones(n)))
    return pi,R,lo,hi,B,delta,span,flux,delta*mixing


def score(Q,b,s,gamma,m,w,T,priors,independent=False):
    n,p = len(b),1-gamma
    factor = p if p != 0 else 1.  # At log utility use the additive score moment.
    A = m*Q+np.diag(p*(b*w-gamma*s*w*w/2))
    A -= max(eigvals(A).real)*np.eye(n)  # Scalar shift held fixed in derivatives.
    q = b-gamma*w*s
    M = np.kron(np.eye(2),A); M[n:,:n] = factor*np.diag(q)
    u,du = (expm(T*M)@np.r_[np.ones(n),np.zeros(n)]).reshape(2,n)
    out = (priors@du)/(factor*T*(priors@u))
    if independent:
        F1 = expm_frechet(T*A,factor*T*np.diag(q),compute_expm=False)@np.ones(n)
        other = (priors@F1)/(factor*T*(priors@u))
        assert max(abs(out-other)) < 3e-8*(1+max(abs(q)))
    return out,A,u


Q2,b2,s2 = np.array([[-1.,1.],[1.,-1.]]),np.array([1.,-1.]),np.ones(2)
base = (Q2,b2,s2,.5,.5)
ends,psi = [F(-2),F(2)],[F(-1,2),F(1,2)]
assert max(psi)-min(psi) == max(abs(psi[1]-psi[0]),abs(psi[0]-psi[1])) == 1
assert max(abs(2*w) for w in ends) == 4
for w in ends:
    q = [1-w/2,-1-w/2]
    assert max(q)-min(q) == 2 and (q[0]+q[1])/2 == -w/2
    assert psi[1]-psi[0] == q[0]-(q[0]+q[1])/2
assert F(1,2)*4*F(1,2) == 1  # Omega=delta*mixing=1.
assert sum(F(2)**k/F(factorial(k)) for k in range(3)) > 4
assert F(3,4)-F(2,8) == F(1,2)
assert 2+F(1,2)/(1-F(1,3)) < 3  # Geometric factorial-tail bound for e<3.
print('benchmark Q=[[-1,1],[1,-1]], b=(1,-1), s=(1,1), gamma=1/2, prior=(1,0)')
print('I=[-2,2], B=2, D=4, mixing area=1/2, Psi=Gamma=Omega=1, psi=(-1/2,1/2), w0=0')
print('bounds: 2/(m*T)+2/(m-1), m>1; critical profile (1-exp(-2*c))/c, error<=2*c/m')
print('exact Taylor witness e^2>4; w1>3/4; all m>=8,T=1/m have w*>1/2')
print('benchmark global curvature <= -1/2+1/(2*m); all m>1 give uniqueness at every horizon')
for m in [2.,8.,512.]:
    for T in [.001,1.,100.]:
        for w in np.linspace(-2,2,5):
            _,A,_ = score(Q2,b2,s2,.5,m,w,T,np.array([1.,0.]))
            M = np.kron(np.eye(3),A); q = b2-w/2
            M[2:4,:2],M[4:,:2],M[4:,2:4] = np.diag(q)/2,-np.eye(2)/4,np.diag(q)
            z,dz,ddz = (expm(T*M)@np.r_[np.ones(2),np.zeros(4)]).reshape(3,2)
            for prior in [np.array([1.,0.]),np.array([0.,1.]),np.array([.5,.5])]:
                curve = ((prior@ddz)/(prior@z)-((prior@dz)/(prior@z))**2)/(T/2)
                assert curve <= -.5+.5/m+3e-8*(1+T)
Q3 = np.roll(np.eye(3),1,axis=1)-np.eye(3)
b3,s3 = np.array([.12,-.08,.05]),np.array([.04,.09,.0625])
Q4 = np.roll(np.eye(4),1,axis=1)-np.eye(4)
# Cycles: shortest paths give a>=n/(3*(n-1)!), using e<3.
# Birth-death: rate-2 uniformization gives a>=1/6, using e^2<9.
models = [base,(Q3,b3,s3,.5,2.),
          (Q4,np.r_[b3,.03],np.r_[s3,.07],.5,4.5),
          (np.array([[-1.,1.,0.],[1.,-2.,1.],[0.,1.,-1.]]),b3,s3,2.,6.),
          (np.array([[-3.,2.,1.],[1.,-2.,1.],[.5,1.5,-2.]]),b3,s3,.5,2/3)]
N = np.outer([1.,-1.,0.],[1.,1.,-2.])/20
assert norm(N@N) < 2e-14
# Defective model: every off-diagonal >=2/9, so common refresh gives area<=3/2.
models.append((np.ones((3,3))/3-np.eye(3)+N,b3,s3,1.,1.5))
assert F(1,3)-F(1,10) > F(2,9)
cases,roots,usage,profile_usage,log_error = 0,0,0.,0.,0.
for Q,b,s,gamma,mixing in models:
    pi,R,lo,hi,B,delta,span,flux,omega = data(Q,b,s,gamma,mixing)
    n,p = len(b),1-gamma
    priors = np.vstack([np.eye(n),pi])
    for m in [2*omega+2,8*omega+8]:
        for T in [1e-4,.01,1.,10.,100.,1e4]:
            c = m*T
            occupation = np.outer(np.ones(n),pi)+(expm(c*Q)-np.eye(n))@R/c
            rho = priors@occupation
            assert min(rho.ravel()) >= -2e-10 and max(abs(rho.sum(axis=1)-1)) < 2e-10
            budget = span/(m*T)+omega*flux/(m-omega)
            short_budget = B*delta*T/4
            for w in np.linspace(lo,hi,5):
                value,A,u = score(Q,b,s,gamma,m,w,T,priors,True)
                q = b-gamma*w*s
                assert max(abs(value-pi@q)) <= budget+3e-8
                assert max(abs(value-rho@q)) <= short_budget+3e-8
                assert max(u)/min(u) <= m/(m-omega)+3e-8
                usage = max(usage,max(abs(value-pi@q))/budget)
                if short_budget > 0: profile_usage = max(profile_usage,max(abs(value-rho@q))/short_budget)
                cases += len(priors)
            for prior,rhoc in zip(priors,rho):
                root = brentq(lambda w: score(Q,b,s,gamma,m,w,T,prior)[0],lo,hi,xtol=2e-11)
                w0,wc = pi@b/(gamma*(pi@s)),rhoc@b/(gamma*(rhoc@s))
                assert abs(root-w0) <= budget/(gamma*(pi@s))+3e-8
                assert abs(root-wc) <= short_budget/(gamma*(rhoc@s))+3e-8
                if gamma == 1.: log_error = max(log_error,abs(root-wc))
                roots += 1
print('models=',len(models),'score cases=',cases,'critical-point roots=',roots,'T=1e-4..1e4')
print('risk aversions=',sorted(set(model[3] for model in models)),'max exact log-profile error=',log_error)
print('largest score/homogenization budget=',usage,'largest score/short-profile budget=',profile_usage)

# Independent quadrature of the tilted Dynkin identity, including initial reweighting.
quad_error = 0.
nodes,weights = leggauss(64)
for model in [models[0],models[1],models[3],models[-1]]:
    Q,b,s,gamma,mixing = model
    pi,R,lo,hi,B,delta,span,flux,omega = data(*model)
    m,T,w = 2*omega+4,.7,.37*lo+.63*hi
    lhs,A,u = score(Q,b,s,gamma,m,w,T,pi)
    q,psi = b-gamma*w*s,R@(b-gamma*w*s)
    Z = pi@u
    boundary = (pi@expm(T*A)@psi-pi@(psi*u))/Z
    integral = 0.
    for x,weight in zip(nodes,weights):
        t = T*(x+1)/2
        future = expm((T-t)*A)@np.ones(len(b))
        marginal = (pi@expm(t*A))*future/Z
        correction = m*np.sum(Q*(future[None,:]/future[:,None]-1)*(psi[None,:]-psi[:,None]),axis=1)
        integral += weight*T/2*(marginal@correction)
    err = abs(lhs-pi@q-(boundary-integral)/(m*T))
    assert err < 3e-9
    quad_error = max(quad_error,err)
print('independent tilted Dynkin quadrature error=',quad_error)
for c in [.05,.25,1.,4.,16.]:
    wc = -np.expm1(-2*c)/c
    for multiplier in [1,2,4,8,16,64]:
        m = max(8.,8*c)*multiplier
        T = c/m  # T<=1/8 gives strict concavity directly from the path range bound.
        root = brentq(lambda w: score(Q2,b2,s2,.5,m,w,T,np.array([1.,0.]))[0],-2,2)
        assert abs(root-wc) <= 2*c/m+3e-8
        if c == 1.:
            assert root > .5
            if m == 512.: print('c=1,m=512 root=',root,'limit=',wc,'error bound=',2/m)
print('30 critical-profile roots pass; limit 1-exp(-2)=',-np.expm1(-2))
print('PASS')
