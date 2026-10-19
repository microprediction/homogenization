"""Singular Green--Kubo admissibility and a sharp state-budget distance barrier."""
from fractions import Fraction as Fr
import numpy as np
from numpy.linalg import norm,eigh,inv,matrix_rank
from scipy.linalg import block_diag,expm
from scipy.integrate import quad_vec
from scipy.optimize import minimize
from effective_generator import gk

J = np.array([[0.,1.],[-1.,0.]])


def bouquet(kappas,zeros):
    lengths = [next(n for n in range(3,300) if k <= 1/np.tan(np.pi/n)+2e-12) for k in kappas]
    N,d = 1+zeros+sum(n-1 for n in lengths),2*len(kappas)+zeros
    flow,U = np.zeros((N,N)),np.zeros((N,d)); vertex = 1
    for j,(n,kappa) in enumerate(zip(lengths,kappas)):
        theta = 2*np.pi/n; total = 2/(n*(1-np.cos(theta))); diff = 2*kappa/(n*np.sin(theta))
        forward,backward = (total+diff)/2,(total-diff)/2
        assert backward >= -3e-12*(1+total)
        cycle = [0,*range(vertex,vertex+n-1)]; vertex += n-1
        for k,i in enumerate(cycle):
            nxt = cycle[(k+1)%n]; flow[i,nxt] += forward; flow[nxt,i] += max(backward,0.)
            U[i,2*j:2*j+2] = [np.cos(k*theta)-1,np.sin(k*theta)]
    for j in range(zeros):
        flow[0,vertex] = flow[vertex,0] = 1; U[vertex,2*len(kappas)+j] = 1; vertex += 1
    pi = np.arange(1,N+1,dtype=float); pi /= sum(pi)
    Q = flow/pi[:,None]; Q -= np.diag(Q.sum(axis=1))
    return Q,pi,U


def direct(Q,pi,F):
    P = np.outer(np.ones(len(pi)),pi)
    return F.T@(pi[:,None]*((-inv(Q+P)+P)@F))


def bound(N,a):
    c = 1/np.tan(np.pi/N)
    upper = 2*a*a/(c*(np.sqrt(c*c+4*a*a)+c))  # Rationalized to avoid cancellation.
    return c,c/(c+1)*upper,upper


rng = np.random.default_rng(20261019)
cases,max_error = 0,0.
for rank in range(1,7):
    for repeat in range(6):
        r = repeat%((rank//2)+1); kappas = rng.uniform(.1,1.5,size=r)
        Q,pi,U = bouquet(kappas,rank-2*r)
        K0 = block_diag(*[np.eye(2)+k*J for k in kappas],np.eye(rank-2*r))
        d = rank+1+repeat%3; O,_ = np.linalg.qr(rng.normal(size=(d,d)))
        E = O[:,:rank]; amplitudes = rng.uniform(.6,1.8,size=rank)
        C = np.diag(amplitudes)@E.T; F = -Q@U@C; wanted = C.T@K0@C
        actual = direct(Q,pi,F); error = norm(actual-wanted)/(1+norm(wanted)); max_error = max(max_error,error)
        assert error < 3e-8 and norm(gk(Q,F.T)-wanted) < 3e-8*(1+norm(wanted))
        kernel = O[:,rank:]; G,A = (actual+actual.T)/2,(actual-actual.T)/2
        assert norm(G@kernel)+norm(A@kernel)+norm(F@kernel) < 3e-8*(1+norm(F)+norm(actual))
        assert matrix_rank(F) == rank and matrix_rank((wanted+wanted.T)/2) == rank
        cases += 1
print('singular realizations=',cases,'positive ranks 1..6, ambient dimensions 2..9; arbitrary rotated kernels')
print('maximum normalized realization discrepancy=',max_error)

# Random finite-chain blocks, including singular features, obey the uniform N-state sector LMI.
sector_cases = 0
for N in range(3,10):
    c = 1/np.tan(np.pi/N)
    for repeat in range(16):
        rates = rng.uniform(.05,1.,size=(N,N)); np.fill_diagonal(rates,0)
        Q = rates-np.diag(rates.sum(axis=1)); F = rng.normal(size=(N,2))
        if repeat%4 == 0: F[:,1] = 2*F[:,0]
        K = gk(Q,F.T); G,A = (K+K.T)/2,(K-K.T)/2
        assert min(eigh(c*G+1j*A)[0]) >= -3e-8*(1+norm(K))
        c,lower,upper = bound(N,1.)
        target = np.diag([1.,0.])+J
        assert norm(K-target,2) >= lower-3e-8*(1+norm(K))
        sector_cases += 1
print('random state-budget sector/distance checks=',sector_cases)

# Uniform directed-cycle witnesses attain the explicit upper bound exactly in the analytic formula.
ratios = []
for N in [3,4,5,8,16,32,64,128]:
    c,lower,upper = bound(N,1.); theta = 2*np.pi/N; k = np.arange(N)
    U = np.column_stack([np.cos(k*theta),np.sin(k*theta)])/np.sqrt(N*(1-np.cos(theta))/2)
    U = U@np.diag(np.sqrt([1+upper,upper]))
    pi = np.ones(N)/N; Q = N*(np.roll(np.eye(N),1,axis=1)-np.eye(N)); F = -Q@U
    wanted = np.diag([1+upper,upper])+J
    assert norm(direct(Q,pi,F)-wanted) < 3e-7*(1+norm(wanted))
    assert norm(gk(Q,F.T)-wanted) < 3e-7*(1+norm(wanted))
    flipped = F@np.diag([1.,-1.])
    assert norm(direct(Q,pi,flipped)-(wanted-2*J)) < 3e-7*(1+norm(wanted))
    assert abs(upper*(1+upper)*c*c-1) < 3e-8
    ratios.append((N,N*N*lower/np.pi**2,N*N*upper/np.pi**2))
print('a=1; N and normalized lower/upper N^2*rho/pi^2:',ratios)
print('sharp asymptotic constant pi^2*a^2 at a=1:',np.pi**2)

# Exact four-state primal/dual witness, with rational generator, potentials and block.
P = [[Fr(int(j==(i+1)%4)) for j in range(4)] for i in range(4)]
Q = [[Fr(2,3)*(P[i][j]-int(i==j)) for j in range(4)] for i in range(4)]
assert all(sum(row) == 0 for row in Q) and all(sum(Q[i][j] for i in range(4)) == 0 for j in range(4))
U = [[Fr(x),Fr(y)] for x,y in [(2,0),(0,1),(-2,0),(0,-1)]]
F = [[-sum(Q[i][k]*U[k][j] for k in range(4)) for j in range(2)] for i in range(4)]
assert all(sum(F[i][j] for i in range(4)) == 0 for j in range(2))
K = [[sum(F[k][i]*U[k][j]/4 for k in range(4)) for j in range(2)] for i in range(2)]
assert K == [[Fr(4,3),Fr(2,3)],[-Fr(2,3),Fr(1,3)]]
a = Fr(14,15); target = [[Fr(1),a],[-a,Fr(0)]]
dual = [[Fr(1,5),-Fr(2,5)],[Fr(2,5),Fr(4,5)]]
assert dual[0][0]*dual[1][1] == dual[0][1]**2  # The corresponding Hermitian dual is PSD, rank one.
pair = lambda X: sum(dual[i][j]*X[i][j] for i in range(2) for j in range(2))
assert pair(K) == 0 and pair(target) == -Fr(41,75)
assert sum(x*x for row in dual for x in row)+2*(dual[0][0]*dual[1][1]-dual[0][1]*dual[1][0]) == Fr(41,25)
delta = [[K[i][j]-target[i][j] for j in range(2)] for i in range(2)]
assert all(sum(delta[k][i]*delta[k][j] for k in range(2)) == (Fr(41,225) if i==j else 0) for i in range(2) for j in range(2))
Q,U,F,K,target = map(lambda x:np.array(x,float),[Q,U,F,K,target])
assert norm(gk(Q,F.T)-K) < 3e-8
distance = np.sqrt(41)/15
assert abs(norm(K-target,2)-distance) < 3e-8
print('exact four-state benchmark: a=14/15, K=[[4/3,2/3],[-2/3,1/3]], rho_4=sqrt(41)/15=',distance)
print('Q=(2/3)*(P-I), pi=(1,1,1,1)/4; potentials=(2,0,-2,0),(0,1,0,-1); F=-Q*U')
print('dual M=[[1,-2],[2,4]]/5; z=(1,2i)/sqrt(5)')
print('dual value=-41/75; dual nuclear norm=sqrt(41)/5; squared primal distance=41/225')

# Smooth restricted-family projections independently find the globally optimal dual-certified witness.
# Here x=1+t,y=t; no numerical optimization is used to prove optimality outside this family.
def objective(x): return x[0]**2+(x[1]-float(a))**2
for start in [[1.,.3],[1.,.8],[.5,.5]]:
    result = minimize(objective,start,method='SLSQP',bounds=[(0,None),(None,None)],
        constraints={'type':'ineq','fun':lambda x:x[0]*(1+x[0])-x[1]**2},options={'ftol':1e-11,'maxiter':800})
    assert result.success and abs(result.fun-distance**2) < 3e-7
integral,_ = quad_vec(lambda t:F.T@(expm(t*Q)@F)/4,0,80,epsabs=1e-10)
gap = 2/3; tail = 2*norm(F/2,2)**2*np.exp(-gap*80)/gap
assert norm(integral-K) < tail+3e-8*(1+norm(K))
print('independent restricted-family projections=3; correlation quadrature discrepancy=',norm(integral-K))
print('PASS')
