"""Sharp Green--Kubo sector counts: rational flows and floating consistency checks."""
from fractions import Fraction as F
import numpy as np
from numpy.linalg import solve, inv, eigh, norm, cholesky, eigvals
from scipy.linalg import block_diag, expm
from scipy.integrate import quad_vec


def extract_cycles(flow):
    n = len(flow); residual = [row.copy() for row in flow]; cycles = []
    assert all(sum(row) == sum(flow[j][i] for j in range(n)) for i,row in enumerate(flow))
    while any(x > 0 for row in residual for x in row):
        start = next(i for i,row in enumerate(residual) if any(x > 0 for x in row))
        path,seen,i = [],{},start
        while i not in seen:
            seen[i] = len(path); path.append(i)
            i = next(j for j,x in enumerate(residual[i]) if x > 0)
        cycle = path[seen[i]:]; weight = min(residual[j][cycle[(k+1)%len(cycle)]] for k,j in enumerate(cycle))
        for k,j in enumerate(cycle): residual[j][cycle[(k+1)%len(cycle)]] -= weight
        cycles.append((cycle,weight))
    rebuilt = [[F(0) for _ in range(n)] for _ in range(n)]
    for cycle,weight in cycles:
        for k,i in enumerate(cycle): rebuilt[i][cycle[(k+1)%len(cycle)]] += weight
    assert rebuilt == flow
    return cycles


def forms(Q,pi,features):
    P = np.outer(np.ones(len(pi)),pi)
    features = features-P@features
    R = -inv(Q+P)+P
    K = features.T@(pi[:,None]*(R@features))
    return K,(K+K.T)/2,(K-K.T)/2


def sector(G,A):
    values,U = eigh(G)
    keep = values > 1e-10*max(values)
    W = U[:,keep]/np.sqrt(values[keep])
    assert norm(A@U[:,~keep]) <= 3e-9*(1+norm(A))
    return norm(W.T@A@W,2)


def cycle_model(n,kappa):
    angle = 2*np.pi/n
    total,difference = 1/(1-np.cos(angle)),kappa/np.sin(angle)
    a,b = (total+difference)/2,(total-difference)/2
    assert min(a,b) >= -3e-12*(1+total)
    # Clamp only roundoff at the analytically valid equality endpoint.
    a,b = max(0.,a),max(0.,b)
    C = np.roll(np.eye(n),1,axis=1)
    Q = a*C+b*C.T-(a+b)*np.eye(n)
    e = np.sqrt(2)*np.column_stack([np.cos(angle*np.arange(n)),np.sin(angle*np.arange(n))])
    return Q,np.ones(n)/n,e


rng = np.random.default_rng(20261019)
random_cases,flow_cycles,worst = 0,0,0.
for n in range(2,13):
    for repeat in range(12):
        flow = [[F(0) for _ in range(n)] for _ in range(n)]
        for trial in range(13):
            cyc = list(range(n)) if trial == 0 else list(rng.choice(n,size=int(rng.integers(2,n+1)),replace=False))
            weight = F(int(rng.integers(1,10)),int(rng.integers(1,10)))
            for k,i in enumerate(cyc): flow[i][cyc[(k+1)%len(cyc)]] += weight
        cycles = extract_cycles(flow); flow_cycles += len(cycles)
        masses = [F(int(rng.integers(1,8))) for _ in range(n)]; mass = sum(masses)
        pir = [x/mass for x in masses]
        Qr = [[flow[i][j]/pir[i] for j in range(n)] for i in range(n)]
        for i in range(n): Qr[i][i] = -sum(Qr[i])
        assert all(sum(pir[i]*Qr[i][j] for i in range(n)) == 0 for j in range(n))
        Q,pi = np.array(Qr,float),np.array(pir,float)
        K,G,A = forms(Q,pi,np.eye(n))
        chi,bound = sector(G,A),1/np.tan(np.pi/n)
        assert chi <= bound+3e-8*(1+bound)
        if n > 2: worst = max(worst,chi/bound)
        Qrev = (Q.T*pi[None,:])/pi[:,None]
        assert norm(forms(Qrev,pi,np.eye(n))[0]-K.T) <= 3e-9*(1+norm(K))
        pair = rng.normal(size=(n,2)); kp,gp,ap = forms(Q,pi,pair)
        determinant = np.linalg.det(gp)
        assert abs(ap[0,1]) <= bound*np.sqrt(max(0.,determinant))+3e-9*(1+norm(kp))
        for speed in [.1,10.]:
            km,gm,am = forms(speed*Q,pi,pair)
            assert norm(speed*km-kp) < 3e-8*(1+norm(kp))
        random_cases += 1
print('random nonnormal/sparse cases=',random_cases,'exact rational decomposed cycles=',flow_cycles,'largest sector/bound=',worst)

sharp_error,realizations = 0.,0
for n in [3,4,5,7,12,24,48]:
    bound = 1/np.tan(np.pi/n)
    Q,pi,e = cycle_model(n,bound)
    K,G,A = forms(Q,pi,e)
    sharp_error = max(sharp_error,abs(sector(G,A)-bound)/(1+bound))
    assert abs(sector(G,A)-bound) < 3e-8*(1+bound)
    for fraction in [-1.,-.6,0.,.3,1.]:
        kappa = fraction*bound
        Q,pi,e = cycle_model(n,kappa)
        X = rng.normal(size=(2,2)); targetG = X.T@X+np.eye(2)/2
        C = cholesky(targetG).T
        features = np.sqrt(1+kappa*kappa)*e@C
        target = targetG+kappa*np.sqrt(np.linalg.det(targetG))*np.array([[0,1],[-1,0]])
        assert norm(forms(Q,pi,features)[0]-target) < 3e-8*(1+norm(target))
        realizations += 1
print('sharp cycle checks=7; max normalized error=',sharp_error,'full two-feature realizations=',realizations)

# Exact algebra certifies 2 > cot(pi/6), 2 < cot(pi/7), not sampled trigonometry.
polynomial = lambda x: x**3-21*x*x+35*x-7
assert F(2)**2 > 3 and polynomial(F(1,4)) == F(29,64)
assert 3*F(1,3)**2-42*F(1,3)+35 == F(64,3)
print('exact: chi=2 > sqrt(3)=cot(pi/6); p(1/4)=29/64, min pprime on [0,1/3]=64/3')
Q,pi,e = cycle_model(7,2.)
target = np.array([[.1,.2],[-.2,.1]])
K,G,A = forms(Q,pi,e/np.sqrt(2))
assert norm(K-target) < 3e-9
print('seven-state witness: D=1,c=2 circle block K=[[1/10,1/5],[-1/5,1/10]], rank(G)=2, minimum states=7')
print('cyclic rates a,b=',Q[0,1],Q[0,-1],'features=(cos(2*pi*j/7),sin(2*pi*j/7))')
integral,error = quad_vec(lambda t: (e/np.sqrt(2)).T@(pi[:,None]*(expm(t*Q)@(e/np.sqrt(2)))),0,40,epsabs=1e-10)
assert norm(integral-target) <= 3e-8
print('independent correlation integral to 40: discrepancy=',norm(integral-target),'quadrature estimate=',error,'analytic tail <=',np.exp(-40))

epsilon = F(1,1000); lower = (F(1,5)-epsilon)/(F(1,10)+epsilon)
assert lower == F(199,101) and lower**2 > 3
print('deterministic error epsilon=1/1000: chi_low=199/101=',float(lower),'>sqrt(3), still at least seven states')
for _ in range(100):
    error_matrix = rng.normal(size=(2,2)); error_matrix *= float(epsilon)/norm(error_matrix,2)
    estimate = target+error_matrix; ghat,ahat = (estimate+estimate.T)/2,(estimate-estimate.T)/2
    observed_lower = max(0,abs(ahat[0,1])-float(epsilon))/np.sqrt(np.linalg.det(ghat+float(epsilon)*np.eye(2)))
    assert observed_lower <= 2+3e-9

# A defective generator has real spectrum yet a nonzero directional response.
x,y = np.array([1.,-1.,0.]),np.array([1.,1.,-2.])
assert np.dot(x,y) == 0 and sum(x) == 0 and sum(y) == 0
N = np.outer(x,y)/20; assert norm(N@N) < 3e-14
Q = np.ones((3,3))/3-np.eye(3)+N
K,G,A = forms(Q,np.ones(3)/3,np.eye(3))
assert max(abs(eigvals(Q).imag)) < 3e-7*(1+norm(Q)) and sector(G,A) > .05
print('defective control: analytic spectrum=(0,-1,-1); numerical eigenvalues=',eigvals(Q),'intrinsic sector=',sector(G,A))

# Positive common loading congruences retain sector, although rank may grow.
Q,pi,e = cycle_model(3,1/np.sqrt(3))
K,G,A = forms(Q,pi,np.sqrt(1+1/3)*e)
aggregate = block_diag(K,K)
ag,aa = (aggregate+aggregate.T)/2,(aggregate-aggregate.T)/2
assert np.linalg.matrix_rank(ag) == 4 and abs(sector(ag,aa)-1/np.sqrt(3)) < 3e-9
print('three-state common-loading aggregate: symmetric rank=4, sector=',sector(ag,aa),'<=cot(pi/3)')
for _ in range(100):
    loadings = [rng.normal(size=(5,2)) for _ in range(3)]
    aggregate = sum(w*M@K@M.T for w,M in zip([.2,.3,.5],loadings))
    assert sector((aggregate+aggregate.T)/2,(aggregate-aggregate.T)/2) <= 1/np.sqrt(3)+3e-8
print('additional positive common-loading aggregation checks=100')
print('PASS')
