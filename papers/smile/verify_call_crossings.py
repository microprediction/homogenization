"""Exact tilted moments and four call crossings; floating roots are not enclosures."""
import math
import sys
from pathlib import Path

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from scipy.linalg import expm
from scipy.optimize import brentq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'cumulants'))
from verify_cumulants import leveraged_joint_moments


def law(q):
    n = len(q)
    return np.linalg.solve(q.T+np.ones((n,n)),np.ones(n))


def jets(q, a, speed, center, rho=-.6, prior=None):
    n, kappa, xi = len(a),2.,.3
    def rhs(_,y):
        b, f = y[:3],y[3:].reshape(3,n)
        bp = np.array([(rho*xi*center-kappa)*b[0]+center-.5,
            (rho*xi*center-kappa)*b[1]+rho*xi*b[0]+xi*xi*b[0]**2/2+.5,
            (rho*xi*center-kappa)*b[2]+rho*xi*b[1]+xi*xi*b[0]*b[1]])
        fp = speed*(q@f.T).T
        for j in range(3):
            fp[j] += a*(b[j]+sum(b[l]*f[j-l-1] for l in range(j)))
        return np.r_[bp,fp.ravel()]
    sol = solve_ivp(rhs,(0.,1.),np.zeros(3+3*n),method='DOP853',rtol=3e-13,atol=3e-15)
    assert sol.success
    b,f = sol.y[:3,-1],sol.y[3:,-1].reshape(3,n)
    ee = [1.]
    for j in range(1,4): ee.append(sum(l*.04*b[l-1]*ee[j-l] for l in range(1,j+1))/j)
    means = np.r_[1.,f@(law(q) if prior is None else prior)]
    return np.array([math.factorial(j)*sum(ee[l]*means[j-l] for l in range(j+1)) for j in range(4)])


def rational_witness():
    """Exact matrix Taylor action plus a rational induced-norm tail bound."""
    pairs = [(p,d-p) for d in range(4) for p in range(d+1)]; where = {v:i for i,v in enumerate(pairs)}
    n = 3*len(pairs); initial = sp.zeros(n,1)
    for (p,v),j in where.items():
        if p == 0:
            for i in range(3): initial[3*j+i] = sp.Rational(1,3)*sp.Rational(1,25)**v
    def matrix(reverse):
        q = sp.Matrix([[-1,1,0],[0,-1,1],[1,0,-1]])
        if reverse: q = q.T
        aa = sp.zeros(n)
        for (p,v),j in where.items():
            for i in range(3):
                row = 3*j+i
                for h in range(3): aa[row,3*j+h] += q.T[i,h]
                aa[row,row] -= 2*v
                if p: aa[row,3*where[(p-1,v+1)]+i] -= sp.Rational(p,2)
                if p >= 2: aa[row,3*where[(p-2,v+1)]+i] += sp.Rational(p*(p-1),2)
                if v: aa[row,3*where[(p,v-1)]+i] += v*(i+1)+sp.Rational(9*v*(v-1),200)
                if p and v: aa[row,3*where[(p-1,v)]+i] -= sp.Rational(9*p*v,50)
        return aa
    values,budgets,norms = [],[],[]
    for reverse in (False,True):
        aa = matrix(reverse); term = initial.copy(); total = term.copy(); order = 160
        for j in range(1,order+1): term = aa*term/j; total += term
        norm = math.ceil(max(sum(abs(v) for v in aa.row(i)) for i in range(n)))
        norms.append(norm)
        budgets.append(sp.Rational(3**norm*norm**(order+1),math.factorial(order+1)))
        values.append(sum(total[3*where[(3,0)]+i] for i in range(3)))
    difference,tail = values[0]-values[1],sum(budgets)
    assert abs(difference)-tail > sp.Rational(3,100000) and tail < sp.Rational(13,10**77)
    print(f'Rational witness: dimension={n}, Taylor order={order}, induced-norm ceilings={norms}; absolute gap > 3e-5; tail < 1.3e-76')
    print(f'Exact rational m=1 third-moment witness: gap={float(difference):.12g}; rigorous absolute tail < {float(tail):.3e}')
    return float(difference)


def spectral_data(speed, order):
    q = np.array([[-1.,1.,0.],[0.,-1.,1.],[1.,0.,-1.]])
    a = np.array([1.,2.,3.]); nodes,weights = leggauss(order); u,ww = [],[]
    for left,right in zip((0.,2.,5.,10.,20.),(2.,5.,10.,20.,40.)):
        u.extend((left+right)/2+(right-left)*nodes/2); ww.extend((right-left)*weights/2)
    u,ww = np.array(u),np.array(ww); s = 2+1j*u; n = len(u)
    def rhs(_,y):
        b = y[:n]; f,r = y[n:].reshape(2,3,n)
        return np.r_[.045*b*b+(-.18*s-2)*b+s*(s-1)/2,
                     (speed*q@f+a[:,None]*b*f).ravel(),(speed*q.T@r+a[:,None]*b*r).ravel()]
    sol = solve_ivp(rhs,(0.,1.),np.r_[np.zeros(n),np.ones(6*n)].astype(complex),method='DOP853',rtol=5e-14,atol=5e-16)
    assert sol.success
    b = sol.y[:n,-1]; f,r = sol.y[n:,-1].reshape(2,3,n)
    delta = np.exp(.04*b)*(f-r).mean(axis=0)
    def price(strike):
        return float(np.exp(-strike)*np.real(np.sum(ww*np.exp(-1j*u*strike)*delta/((1+1j*u)*(2+1j*u))))/math.pi)
    return price


def main():
    print('Fixture: unit forward three-cycle, a=(1,2,3), kappa=2, xi=.3, rho=-.6, T=1, v0=.04')
    rng = np.random.default_rng(691); maximum = 0.
    for n in (3,4):
        for _ in range(3):
            q = rng.uniform(.2,1.5,(n,n)); np.fill_diagonal(q,0.); q -= np.diag(q.sum(axis=1))
            pi = law(q); reverse = q.T*pi[None,:]/pi[:,None]; a = rng.uniform(.5,3.,n)
            for speed in (1.,8.):
                for rho in (-.6,0.,1.):
                    for center in (0.,1.):
                        forward,backward = [jets(g,a,speed,center,rho) for g in (q,reverse)]
                        maximum = max(maximum,max(abs(forward[:3]-backward[:3])))
                        kk,rr = (2.,rho) if center == 0 else (2-rho*.3,-rho)
                        raw = leveraged_joint_moments(speed*q,a/kk,kk,.3,.04,1.,rr,order=2)
                        independent = np.array([1.,raw[(1,0)]-raw[(0,1)]/2,raw[(2,0)]-raw[(1,1)]+raw[(0,2)]/4])
                        if center: independent[1] *= -1
                        assert max(abs(forward[:3]-independent)) < 3e-10
    assert maximum < 3e-10
    print(f'72 normal/share-tilted jet pairs, including rho=1; independent polynomial moments: max reversal gap {maximum:.3e}')
    witness = rational_witness()
    example = np.array([[-1.,1.,0.],[0.,-1.,1.],[1.,0.,-1.]])
    assert abs(witness-jets(example,np.array([1.,2.,3.]),1.,0.)[3]+jets(example.T,np.array([1.,2.,3.]),1.,0.)[3]) < 3e-11
    for speed in (1.,8.,32.):
        rootsets=[]
        for order in (24,48):
            price = spectral_data(speed,order)
            points = [-3.,-1.,0.,1.,3.]; signs = np.array([price(k) for k in points])
            print(f'm={speed:.0f}, order={order}: gaps at {points} = {signs}')
            assert np.all(signs[:-1]*signs[1:] < 0)
            rootsets.append(np.array([brentq(price,x,y,xtol=2e-8) for x,y in zip(points,points[1:])]))
        assert max(abs(rootsets[0]-rootsets[1])) < 2e-5
        print(f'm={speed:.0f}: four floating crossings {rootsets[-1]}; refinement {max(abs(rootsets[0]-rootsets[1])):.3e}')
    ar = 2+.18*4; dr = math.sqrt(ar*ar-.09*4*3); p = 2.
    ce = math.exp((3+.04)*(ar-dr)/(.09*p)+(2*math.log(2)+2)/(.09*p))
    decay = .8/(.3*math.sqrt(2))
    tail = 2*math.exp(3)*ce*math.exp(-decay*40)/(math.pi*decay*40**2)
    assert tail < 1e-14
    print(f'Combined exact-call Fourier tail at U=40, |k|<=3: {tail:.3e}; C={ce:.9g}, beta={decay:.9g}')
    q = np.array([[-1.,1.,0.],[0.,-1.,1.],[1.,0.,-1.]])
    a = np.array([1.,2.,3.]); prior = np.array([1.,0.,0.])
    failure = max(abs(jets(q,a,8.,0.,prior=prior)[:3]-jets(q.T,a,8.,0.,prior=prior)[:3]))
    assert failure > 1e-4
    print(f'Nonstationary-start countercheck: first-two-moment gap {failure:.9g}; four-crossing theorem not asserted')
    print('PASS')


if __name__ == '__main__':
    main()
