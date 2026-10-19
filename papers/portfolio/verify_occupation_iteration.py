"""Rational block constants and finite-horizon occupation-ratio solver tests."""
from fractions import Fraction as F
from math import ceil, factorial
import numpy as np
from numpy.linalg import solve, inv, eigvals
from scipy.linalg import expm, expm_frechet
from scipy.optimize import brentq


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
    osc = lambda x: max(x)-min(x)
    B = max(osc([x-gamma*w*y for x,y in zip(b,s)]) for w in ends)
    D = max(osc([x*w-gamma*y*w*w/2 for x,y in zip(b,s)]) for w in points)
    return ends,B,abs(1-gamma)*D


def evaluate(Q,b,s,gamma,m,T,prior,w,derivative=False):
    n,p = len(b),1-gamma
    h,q = b*w-gamma*s*w*w/2,b-gamma*w*s
    A = m*Q+p*np.diag(h); shift = max(eigvals(A).real)
    A -= shift*np.eye(n)  # Scalar shift held fixed in every derivative.
    M = np.kron(np.eye(3),A)
    M[n:2*n,:n],M[2*n:,:n] = np.diag(b),np.diag(s)
    initial = np.r_[np.ones(n),np.zeros(2*n)]
    moments = expm(T*M)@initial
    z,eb,es = moments.reshape(3,n)@prior
    ratio,score = eb/(gamma*es),(eb-gamma*w*es)/(T*z)
    value = (np.log(z)/T+shift)/p if p else prior@expm_frechet(T*m*Q,T*np.diag(h),compute_expm=False)@np.ones(n)/T
    if not derivative: return ratio,score,value
    dz,db,ds = (expm_frechet(T*M,T*np.kron(np.eye(3),p*np.diag(q)),compute_expm=False)@initial).reshape(3,n)@prior
    fp = (db/es-eb*ds/es**2)/gamma
    # Independent n-dimensional Frechet integrals, rather than the 3-block system.
    eb2 = prior@expm_frechet(T*A,T*np.diag(b),compute_expm=False)@np.ones(n)
    es2 = prior@expm_frechet(T*A,T*np.diag(s),compute_expm=False)@np.ones(n)
    assert abs(ratio-eb2/(gamma*es2)) <= 3e-8*(1+abs(ratio))
    curvature = ((db-gamma*es-gamma*w*ds)/z-(eb-gamma*w*es)*dz/z**2)/T
    rawscale = 1+(abs(db)+gamma*abs(w*ds))/max(abs(es),1e-100)/gamma
    return ratio,score,value,fp,curvature,rawscale


cycle = lambda n: [[int(j==(i+1)%n)-int(i==j) for j in range(n)] for i in range(n)]
b3,s3 = [.12,-.08,.05],[.04,.09,.0625]
models = [(cycle(2),[1,-1],[1,1],F(1,2)),(cycle(3),b3,s3,F(1,2)),
          (cycle(4),b3+[.03],s3+[.07],F(1,2)),
          ([[-1,1,0],[1,-2,1],[0,1,-1]],b3,s3,F(2)),
          ([[-3,2,1],[1,-2,1],[.5,1.5,-2]],b3,s3,F(1,2))]
N = [[F(x*y,20) for y in [1,1,-2]] for x in [1,-1,0]]
assert all(x == 0 for row in multiply(N,N) for x in row)
models.append(([[F(1,3)-int(i==j)+N[i][j] for j in range(3)] for i in range(3)],b3,s3,F(1)))
assert sum(F(2)**k/F(factorial(k)) for k in range(5)) == 7
print('benchmark: Q=[[-1,1],[1,-1]], b=(1,-1), s=(1,1), gamma=1/2, theta=1')
print('exact a0=3/4, B=2, D=4, d=2, C=8/3; m>14/3 sufficient at every T and prior')
assert F(2)+F(8,3) == F(14,3)
assert F(8,3)**2/F(1,2) == F(128,9)
print('L,E<=8/[3(m-2)]; one-step policy error<=128/[9(m-2)^2]')
print('one-step CE-rate loss<= (1+8/[3(m-2)])/4 * (128/[9(m-2)^2])^2')
cases,iterations,oracle_tests = 0,0,0
max_derivative,max_policy,max_loss = 0.,0.,0.
for index,(Qr,br,sr,gr) in enumerate(models):
    Qr = [[F(str(x)) for x in row] for row in Qr]
    br,sr = [F(str(x)) for x in br],[F(str(x)) for x in sr]
    ends,Br,dr = constants(br,sr,gr)
    ar = F(3,4) if index == 0 else block_ratio(Qr)
    assert block_ratio(Qr) >= ar if index == 0 else ar > 0
    Cr = abs(1-gr)*Br**2/(2*ar*gr*min(sr))
    if index == 0:
        assert (ends,Br,dr,Cr) == ([F(-2),F(2)],F(2),F(2),F(8,3))
    print('model',index,'gamma=',gr,'a0=',ar,'B=',Br,'d=',dr,'C=',Cr)
    Q,b,s,gamma = np.array(Qr,float),np.array(br,float),np.array(sr,float),float(gr)
    lo,hi,B,d,a0,C = *map(float,ends),float(Br),float(dr),float(ar),float(Cr)
    n,p = len(b),1-gamma
    pi = solve(Q.T+np.ones((n,n)),np.ones(n)); P = np.outer(np.ones(n),pi); R = inv(Q+P)-P
    for multiplier in [2,4,8]:
        m = float(multiplier*(dr+Cr+1))
        for T in [1e-4,.01,1.,10.,100.,1e3]:
            L = abs(p)*B*B/(gamma*min(s))*min(T/4,np.exp(d/m)/(2*m*a0))
            H = gamma*max(s)+gamma*min(s)*L
            E = min(B*d*T/4,B*np.expm1(d/m)/(2*a0))
            assert L < 1 and L <= C/(m-d)+3e-12
            for prior in [*np.eye(n),pi]:
                rho = prior@(P+(expm(m*T*Q)-np.eye(n))@R/(m*T))
                wc = rho@b/(gamma*(rho@s)); a = gamma*(rho@s)
                root = brentq(lambda w: evaluate(Q,b,s,gamma,m,T,prior,w)[1],lo,hi,xtol=2e-11)
                optimum = evaluate(Q,b,s,gamma,m,T,prior,root)[2]
                for w in np.linspace(lo,hi,3):
                    f,score,value,fp,curvature,scale = evaluate(Q,b,s,gamma,m,T,prior,w,True)
                    assert lo-3e-8 <= f <= hi+3e-8
                    # Derivatives subtract raw moments; budget their uncancelled scale.
                    assert abs(fp) <= L+3e-10*scale
                    assert abs(curvature) <= H+3e-8*scale
                    assert abs(w-root) <= abs(f-w)/(1-L)+3e-8
                    assert optimum-value <= H*(abs(f-w)/(1-L))**2/2+3e-8
                    for sign in [-1,1]:
                        eps_check = 1e-4*(hi-lo)
                        assert abs(w-root) <= (abs(f+sign*eps_check-w)+eps_check)/(1-L)+3e-8
                    if L: max_derivative = max(max_derivative,abs(fp)/L)
                    cases += 1
                w = wc
                for k in range(4):
                    f,score,value = evaluate(Q,b,s,gamma,m,T,prior,w)
                    budget = L**k*E/a
                    assert abs(w-root) <= budget+3e-8
                    assert -3e-8 <= optimum-value <= H*budget**2/2+3e-8
                    if budget > 1e-7: max_policy = max(max_policy,abs(w-root)/budget)
                    if H*budget**2/2 > 1e-7: max_loss = max(max_loss,max(0,optimum-value)/(H*budget**2/2))
                    w = f; iterations += 1
                eps = 1e-4*(hi-lo)
                for sign in [-1,1]:
                    w = wc
                    for k in range(1,5):
                        f = evaluate(Q,b,s,gamma,m,T,prior,w)[0]
                        w = np.clip(f+sign*eps,lo,hi)
                        budget = L**k*E/a+eps*(1-L**k)/(1-L)
                        assert abs(w-root) <= budget+3e-8
                        oracle_tests += 1
print('models=',len(models),'ratio/derivative cases=',cases,'exact iteration checks=',iterations,'perturbed checks=',oracle_tests)
print('T=1e-4..1e3; max derivative/L=',max_derivative,'policy/budget=',max_policy,'loss/budget=',max_loss)
# Independent five-point derivative check on a sparse chain at moderate horizon.
Q = np.array(cycle(3),float); b,s = np.array(b3),np.array(s3)
args = Q,b,s,.5,12.,.7,np.array([1.,0.,0.])
w,step = .37,1e-3
fd = sum(c*evaluate(*args,w+j*step)[0] for j,c in [(-2,1),(-1,-8),(1,8),(2,-1)])/(12*step)
analytic = evaluate(*args,w,True)[3]
assert abs(fd-analytic) < 3e-8*(1+abs(analytic))
print('independent five-point ratio derivative error=',abs(fd-analytic))
Q,b,s,gamma,m,T,prior = np.array(cycle(2),float),np.array([1.,-1.]),np.ones(2),2.,.1,100.,np.ones(2)/2
observed = evaluate(Q,b,s,gamma,m,T,prior,0.,True)[3]
expected = -19/4-np.exp(-20)/4
assert abs(observed-expected) < 3e-8 and observed < -1
print('slow-speed control: gamma=2, m=1/10, T=100, stationary prior, unique optimum w*=0')
print('locally repelling Fprime(0)=',observed,'exact=-19/4-exp(-20)/4 < -1')
print('PASS')
