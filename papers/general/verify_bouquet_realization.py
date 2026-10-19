"""Shared-hub directed Green--Kubo realizations and an exact rational fixture."""
from fractions import Fraction as Fr
import numpy as np
from numpy.linalg import inv, norm, eigh, matrix_rank
from scipy.linalg import block_diag, expm
from scipy.integrate import quad_vec
from effective_generator import gk

J = np.array([[0.,1.],[-1.,0.]])


def bouquet(kappas,zeros,masses=None):
    lengths = []
    for kappa in kappas:
        n = 3
        while kappa > 1/np.tan(np.pi/n)+2e-12: n += 1
        lengths.append(n)
    N,d = 1+zeros+sum(n-1 for n in lengths),2*len(kappas)+zeros
    pi = np.ones(N)/N if masses is None else np.asarray(masses,float)/sum(masses)
    assert len(pi) == N and min(pi) > 0
    flow,U = np.zeros((N,N)),np.zeros((N,d)); vertex = 1
    for j,(n,kappa) in enumerate(zip(lengths,kappas)):
        theta = 2*np.pi/n
        total,difference = 2/(n*(1-np.cos(theta))),2*kappa/(n*np.sin(theta))
        a,b = (total+difference)/2,(total-difference)/2
        assert min(a,b) >= -3e-12*(1+total)
        a,b = max(a,0.),max(b,0.)  # Roundoff only at exact one-way endpoints.
        cycle = [0,*range(vertex,vertex+n-1)]; vertex += n-1
        for k,i in enumerate(cycle):
            nxt = cycle[(k+1)%n]; flow[i,nxt] += a; flow[nxt,i] += b
            U[i,2*j:2*j+2] = [np.cos(k*theta)-1,np.sin(k*theta)]
    for j in range(zeros):
        flow[0,vertex] = flow[vertex,0] = 1
        U[vertex,2*len(kappas)+j] = 1; vertex += 1
    Q = flow/pi[:,None]; Q -= np.diag(Q.sum(axis=1))
    F = -Q@U
    return Q,pi,U,F,lengths


def direct_gk(Q,pi,F):
    P = np.outer(np.ones(len(pi)),pi)
    R = -inv(Q+P)+P
    return F.T@(pi[:,None]*(R@F))


def intrinsic(K):
    G,A = (K+K.T)/2,(K-K.T)/2
    values,V = eigh(G); W = V/np.sqrt(values)
    return np.sort(np.linalg.svd(W.T@A@W,compute_uv=False))


def multiply(A,B):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*B)] for row in A]


# Rational triangle flows and leaf: no trigonometric or matrix-inverse roundoff.
flow = [[Fr(0) for _ in range(6)] for _ in range(6)]
for cycle in [[0,1,2],[0,3,4]]:
    for k,i in enumerate(cycle):
        j = cycle[(k+1)%3]; flow[i][j] += 3; flow[j][i] += 1
flow[0][5] = flow[5][0] = Fr(2)
pi = [Fr(i,21) for i in range(1,7)]
Q = [[flow[i][j]/pi[i] for j in range(6)] for i in range(6)]
for i in range(6): Q[i][i] = -sum(Q[i])
assert all(sum(pi[i]*Q[i][j] for i in range(6)) == 0 for j in range(6))
U = [[Fr(i==j+1) for j in range(5)] for i in range(6)]
F = [[-x for x in row] for row in multiply(Q,U)]
assert all(sum(pi[i]*F[i][j] for i in range(6)) == 0 for j in range(5))
K = multiply([list(col) for col in zip(*F)],[[pi[i]*x for x in row] for i,row in enumerate(U)])
target = [[Fr(0) for _ in range(5)] for _ in range(5)]
for start in [0,2]:
    for i,row in enumerate([[4,-1],[-3,4]]): target[start+i][start:start+2] = map(Fr,row)
target[4][4] = Fr(2)
assert K == target
assert norm(gk(np.array(Q,float),np.array(F,float).T)-np.array(target,float)) < 3e-8
assert matrix_rank(np.array(F,float)) == 5
assert abs(max(intrinsic(np.array(K,float)))-1/np.sqrt(12)) < 3e-9
print('exact rational: two triangles, forward/reverse flows=3,1; leaf flows=2; pi=(1,2,3,4,5,6)/21')
print('K=diag([[4,-1],[-3,4]],[[4,-1],[-3,4]],2); rank(G)=5, minimum states=6, skew norm=1/sqrt(12)')

rng = np.random.default_rng(20261019)
cases,quadratures,max_error,max_tail_usage = 0,0,0.,0.
fixtures = [(d,repeat%((d//2)+1),False) for d in range(1,9) for repeat in range(6)]
fixtures += [(d,d//2,True) for d in [2,3,4,6,8]]
for d,r,strong in fixtures:
    kappas = rng.uniform(.05,.55,size=r) if not strong else rng.uniform(.8,2.2,size=r)
    zeros = d-2*r
    Q,pi,U,F,lengths = bouquet(kappas,zeros)
    weights = rng.uniform(.5,2.,size=len(pi))
    Q,pi,U,F,lengths = bouquet(kappas,zeros,weights)
    assert len(pi) == 1+zeros+sum(n-1 for n in lengths)
    if not strong: assert len(pi) == d+1
    O,_ = np.linalg.qr(rng.normal(size=(d,d)))
    X = rng.normal(size=(d,d)); G = X.T@X+np.eye(d)
    values,V = eigh(G); sqrtG = (V*np.sqrt(values))@V.T
    canonical = block_diag(*[np.eye(2)+k*J for k in kappas],np.eye(zeros))
    C = O.T@sqrtG; features = F@C
    wanted = C.T@canonical@C
    actual = direct_gk(Q,pi,features)
    error = norm(actual-wanted)/(1+norm(wanted)); max_error = max(max_error,error)
    assert error < 3e-8 and norm(gk(Q,features.T)-wanted) < 3e-8*(1+norm(wanted))
    assert norm(pi@Q) < 3e-9*(1+norm(Q)) and norm(pi@features) < 3e-9*(1+norm(features))
    assert matrix_rank(features) == d
    # Poisson inverse and the algebraic potential representation are independent.
    centered = U-np.outer(np.ones(len(pi)),pi@U)
    P = np.outer(np.ones(len(pi)),pi)
    assert norm((-inv(Q+P)+P)@features-centered@C) < 3e-8*(1+norm(centered@C))
    reverse = Q.T*pi[None,:]/pi[:,None]
    assert norm(direct_gk(reverse,pi,features)-actual.T) < 3e-8*(1+norm(actual))
    if d <= 3 and strong:
        weightedQ = np.sqrt(pi)[:,None]*Q/np.sqrt(pi)[None,:]
        gap = eigh(-(weightedQ+weightedQ.T)/2)[0][1]
        endpoint = 30/gap
        integral,quad_error = quad_vec(lambda t: features.T@(pi[:,None]*(expm(t*Q)@features)),0,endpoint,epsabs=1e-9)
        tail = np.sqrt(d)*norm(np.sqrt(pi)[:,None]*features,2)**2*np.exp(-gap*endpoint)/gap
        assert norm(integral-wanted) <= tail+3e-8*(1+norm(wanted))
        max_tail_usage = max(max_tail_usage,norm(integral-wanted)/(tail+3e-8*(1+norm(wanted))))
        quadratures += 1
    cases += 1
print('realization cases=',cases,'rank-minimal weak-sector cases=48, dimensions=1..8, prescribed nonuniform priors')
print('rank-optimal sufficient skew norm <=1/sqrt(3)=',1/np.sqrt(3),'; N=d+1')
print('max normalized realization error=',max_error,'independent correlation quadratures=',quadratures,'max error/budget=',max_tail_usage)

# Boundary obstruction: one five-cycle has unequal intrinsic skew pairs,
# independently of its holding times or a full feature congruence.
high,low = np.sqrt(1+2/np.sqrt(5)),np.sqrt(1-2/np.sqrt(5))
for _ in range(12):
    pi = rng.uniform(.5,2.,size=5); pi /= sum(pi)
    cycle = np.roll(np.eye(5),1,axis=1)
    Q = (cycle-np.eye(5))/pi[:,None]
    features = np.eye(5)[:,1:]; features -= np.ones((5,1))@(pi@features)[None,:]
    assert norm(intrinsic(direct_gk(Q,pi,features))-[low,low,high,high]) < 3e-8
Q,pi,U,F,lengths = bouquet([high,high],0)
wanted = block_diag(np.eye(2)+high*J,np.eye(2)+high*J)
assert lengths == [5,5] and len(pi) == 9
assert norm(direct_gk(Q,pi,F)-wanted) < 3e-8
print('four-feature target K*=I4+cot(pi/5)*diag(J,J); high=',high,'low=',low)
print('five-state saturation rigidity excludes 5 states; shared-hub witness uses 9, so 6<=minimum<=9')
print('holding-time invariant five-cycle skew singular values=',[low,low,high,high])
print('PASS')
