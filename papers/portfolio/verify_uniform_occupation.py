"""Exact rational block bounds and all-horizon occupation-fraction consistency tests."""
from fractions import Fraction as F
from math import factorial, ceil
import numpy as np
from numpy.linalg import solve, inv, eigvals
from scipy.linalg import expm, expm_frechet
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss


def rational_matrix(rows):
    return [[F(str(x)) for x in row] for row in rows]


def multiply(A,B):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*B)] for row in A]


def block_ratio(Q):
    n = len(Q); rate = ceil(max(-Q[i][i] for i in range(n)))
    assert all(sum(row) == 0 for row in Q)
    assert all(Q[i][j] >= 0 for i in range(n) for j in range(n) if i != j)
    degree = max(n-1,rate+2)
    power = [[F(i==j) for j in range(n)] for i in range(n)]
    R = [[F(i==j)+Q[i][j]/rate for j in range(n)] for i in range(n)]
    S = [row.copy() for row in power]
    for k in range(1,degree+1):
        power = multiply(power,R)
        S = [[S[i][j]+F(rate**k,factorial(k))*power[i][j] for j in range(n)] for i in range(n)]
    tail = F(rate**(degree+1),factorial(degree+1))/(1-F(rate,degree+2))
    assert min(x for row in S for x in row) > 0
    return min(min(col)/(max(col)+tail) for col in zip(*S))


def constants(b,s,gamma):
    ends = [min(x/(gamma*y) for x,y in zip(b,s)),max(x/(gamma*y) for x,y in zip(b,s))]
    points = ends.copy()
    for i in range(len(b)):
        for j in range(i):
            if s[i] != s[j]:
                w = (b[i]-b[j])/(gamma*(s[i]-s[j]))
                if ends[0] <= w <= ends[1]: points.append(w)
    B = max(max(b[i]-gamma*w*s[i] for i in range(len(b)))-min(b[i]-gamma*w*s[i] for i in range(len(b))) for w in ends)
    D = max(max(b[i]*w-gamma*s[i]*w*w/2 for i in range(len(b)))-min(b[i]*w-gamma*s[i]*w*w/2 for i in range(len(b))) for w in points)
    return ends,B,abs(1-gamma)*D


def evaluate(Q,b,s,gamma,m,w,T,priors,rho,independent=False):
    n,p = len(b),1-gamma
    h,q = b*w-gamma*s*w*w/2,b-gamma*w*s
    A = m*Q+np.diag(p*h); shift = max(eigvals(A).real)
    A -= shift*np.eye(n)  # Hold the scalar shift fixed in score derivatives.
    factor = p if p != 0 else 1.
    M = np.kron(np.eye(2),A); M[n:,:n] = factor*np.diag(q)
    u,du = (expm(T*M)@np.r_[np.ones(n),np.zeros(n)]).reshape(2,n)
    score = (priors@du)/(factor*T*(priors@u))
    value = (np.log(priors@u)/T+shift)/p if p != 0 else rho@h
    if independent:
        d = expm_frechet(T*A,factor*T*np.diag(q),compute_expm=False)@np.ones(n)
        assert np.max(abs(score-(priors@d)/(factor*T*(priors@u)))) < 3e-8*(1+max(abs(q)))
    return value,score


b3,s3 = [.12,-.08,.05],[.04,.09,.0625]
cycle = lambda n: [[int(j==(i+1)%n)-int(i==j) for j in range(n)] for i in range(n)]
models = [(rational_matrix(cycle(2)),[1,-1],[1,1],F(1,2)),
          (rational_matrix(cycle(3)),b3,s3,F(1,2)),
          (rational_matrix(cycle(4)),b3+[.03],s3+[.07],F(1,2)),
          (rational_matrix([[-1,1,0],[1,-2,1],[0,1,-1]]),b3,s3,F(2)),
          (rational_matrix([[-3,2,1],[1,-2,1],[.5,1.5,-2]]),b3,s3,F(1,2))]
N = [[F(x*y,20) for y in [1,1,-2]] for x in [1,-1,0]]
assert all(x == 0 for row in multiply(N,N) for x in row)
models.append(([[F(1,3)-int(i==j)+N[i][j] for j in range(3)] for i in range(3)],b3,s3,F(1)))
assert sum(F(2)**k/F(factorial(k)) for k in range(5)) == 7
assert block_ratio(models[0][0]) > F(3,4)
assert constants([F(1),F(-1)],[F(1),F(1)],F(1,2)) == ([F(-2),F(2)],F(2),F(2))
print('benchmark Q=[[-1,1],[1,-1]], b=(1,-1), s=(1,1), gamma=1/2, theta=1')
print('exact: a0=3/4, B=2, D=4, delta=2; positive Taylor remainder proves e^2>7')
print('pure-start wc=(1-exp(-2*m*T))/(m*T); E=min(T,(4/3)*expm1(2/m))')
print('distance<=2E, CE-rate loss<=E^2; for m>2: distance<=16/[3(m-2)], loss<=64/[9(m-2)^2]')
assert (F(2)/(2*F(3,4))*2/F(1,2)) == F(16,3)
assert (F(2)/(2*F(3,4))*2)**2/(2*F(1,2)) == F(64,9)
cases,roots,score_usage,loss_usage,log_error = 0,0,0.,0.,0.
prepared = []
for index,(Qr,br,sr,gr) in enumerate(models):
    br,sr = [F(str(x)) for x in br],[F(str(x)) for x in sr]
    ends,Br,dr = constants(br,sr,gr)
    a0r = F(3,4) if index == 0 else block_ratio(Qr)
    print('model',index,'exact a0=',a0r,'B=',Br,'delta=',dr)
    # All maxima and minorization masses used in the analytic budgets are rational.
    Q,b,s,gamma = np.array(Qr,float),np.array(br,float),np.array(sr,float),float(gr)
    lo,hi,B,delta,a0 = float(ends[0]),float(ends[1]),float(Br),float(dr),float(a0r)
    n,p = len(b),1-gamma
    pi = solve(Q.T+np.ones((n,n)),np.ones(n))
    P = np.outer(np.ones(n),pi); R = inv(Q+P)-P
    priors = np.vstack([np.eye(n),pi])
    assert a0 <= min(expm(Q).min(axis=0)/expm(Q).max(axis=0))+2e-12
    C = max(1-gr,F(0))*Br*Br/(2*a0r*gr*min(sr))
    prepared.append((Q,b,s,gamma,lo,hi,B,delta,a0,pi,R))
    for multiplier in [2,4,8]:
        m = float(multiplier*(dr+C+1))
        # The same covariance estimate with q in both slots certifies strict
        # concavity on I at these test speeds, making each bracketed root global.
        modulus = gamma*min(s)-max(p,0)*B*B*np.exp(delta/m)/(2*m*a0)
        assert modulus > 0
        for T in [1e-4,.01,1.,10.,100.,1e4]:
            rho = priors@(P+(expm(m*T*Q)-np.eye(n))@R/(m*T))
            E = min(B*delta*T/4,B*np.expm1(delta/m)/(2*a0))
            for w in np.linspace(lo,hi,5):
                _,score = evaluate(Q,b,s,gamma,m,w,T,priors,rho,True)
                difference = max(abs(score-rho@(b-gamma*w*s)))
                assert difference <= E+3e-8
                if E > 0: score_usage = max(score_usage,difference/E)
                cases += len(priors)
            for prior,occupation in zip(priors,rho):
                wc = occupation@b/(gamma*(occupation@s))
                root = brentq(lambda w: evaluate(Q,b,s,gamma,m,w,T,prior,occupation)[1],lo,hi,xtol=2e-11)
                optimum,_ = evaluate(Q,b,s,gamma,m,root,T,prior,occupation)
                approx,_ = evaluate(Q,b,s,gamma,m,wc,T,prior,occupation)
                budget = E*E/(2*gamma*(occupation@s))
                assert abs(root-wc) <= E/(gamma*(occupation@s))+3e-8
                assert -3e-8 <= optimum-approx <= budget+3e-8
                if budget > 0: loss_usage = max(loss_usage,max(0,optimum-approx)/budget)
                if p == 0: log_error = max(log_error,abs(root-wc))
                roots += 1
print('models=',len(models),'score cases=',cases,'global root/value checks=',roots,'T=1e-4..1e4')
print('risk aversions=',sorted(set(float(model[3]) for model in models)))
print('max score/budget=',score_usage,'max CE-rate loss/budget=',loss_usage,'max log-profile error=',log_error)

# Independently integrate the covariance along the tilt, on a sparse pure start.
Q,b,s,gamma,lo,hi,B,delta,a0,pi,R = prepared[1]
m,T,w,prior = 12.,.7,.37*lo+.63*hi,np.array([1.,0.,0.])
h,q,p = b*w-gamma*s*w*w/2,b-gamma*w*s,1-gamma
rho = prior@(np.outer(np.ones(3),pi)+(expm(m*T*Q)-np.eye(3))@R/(m*T))
nodes,weights = leggauss(32); integral = 0.
for x,weight in zip(nodes,weights):
    lam = (x+1)/2
    A = m*Q+lam*p*np.diag(h); A -= max(eigvals(A).real)*np.eye(3)
    M = np.kron(np.eye(4),A)
    M[3:6,:3],M[6:9,:3],M[9:,3:6],M[9:,6:9] = np.diag(q),p*np.diag(h),p*np.diag(h),np.diag(q)
    z,er,eh,erh = (prior@(expm(T*M)@np.r_[np.ones(3),np.zeros(9)]).reshape(4,3).T)
    covariance = (erh/z-er*eh/(z*z))/T
    bound = B*delta*np.exp(lam*delta/m)/(2*m*a0)
    assert abs(covariance) <= bound+3e-9
    integral += weight*covariance/2
observed = evaluate(Q,b,s,gamma,m,w,T,prior,rho)[1]-rho@q
assert abs(integral-observed) < 3e-9
print('independent tilt-covariance quadrature error=',abs(integral-observed))
print('PASS')
