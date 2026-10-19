"""Global cubic direction signal: floating diagnostics, not interval arithmetic."""
import math

import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, solve_ivp


def chain(q, a):
    n = len(a)
    pi = np.linalg.solve(q.T+np.ones((n,n)),np.ones(n))
    s = np.linalg.inv(q+np.ones((n,1))*pi)-np.ones((n,1))*pi
    c, d = a-pi@a, np.diag(a-pi@a)
    x = [s@c,s@s@c,s@d@s@c,s@s@s@c,2*s@s@d@s@c+s@d@s@s@c,
         s@s@c*(pi@(c*(s@c)))-s@d@s@d@s@c]
    delta = pi@(c*(s@s@d@s@c-s@d@s@s@c))
    weighted = np.sqrt(pi)[:,None]*q/np.sqrt(pi)[None,:]
    return pi,c,x,delta,1/math.sqrt(min(pi)),np.linalg.eigvalsh(-(weighted+weighted.T)/2)[1]


def bounds(q, a, kappa, xi, rho, gamma, p, t, v0):
    ar, qr = kappa-rho*xi*p*gamma, p/(p-1)
    dr = math.sqrt(ar*ar-xi*xi*p*gamma*(p*gamma-1))
    ag, eta = kappa-rho*xi*gamma, 1-rho*rho
    d0, cc = math.sqrt(ag*ag-xi*xi*gamma*(gamma-1)), xi*math.sqrt(eta)
    ell = -2*rho*xi*ag-xi*xi*(2*gamma-1)
    h0, r0 = 1+abs(ell)/(2*d0*cc), ag/d0+abs(rho)/math.sqrt(eta)
    z0 = 2/(1+r0); nu = min(z0/2,-math.expm1(-z0/(2*math.sqrt(h0))))
    g0 = 2*max(ag+math.sqrt(h0)*d0,abs(rho)*xi+math.sqrt(h0)*cc)/(xi*xi*nu)
    aa = max(ag,abs(rho)*xi)
    g1 = xi*xi*g0*g0/2+aa*g0+gamma*gamma/2
    g2 = (xi*xi*g0+aa)*g1; g3 = xi*xi*g1*g1+(xi*xi*g0+aa)*g2
    ce = math.exp((max(a)*t+v0)*(ar-dr)/(p*xi*xi)+(2*min(a)*math.log(2)+min(a)*kappa*t)/(qr*xi*xi))
    decay = min(a)*t*math.sqrt(eta)/(xi*math.sqrt(qr))
    pi,c,x,delta,_,_ = chain(q,a)
    budgets = []
    for generator in (q,q.T*pi[None,:]/pi[:,None]):
        _,_,xx,_,mix,gap = chain(generator,a)
        norms = np.array([max(abs(v)) for v in xx]); n, contrast = mix/gap,max(abs(c))
        z1,z2,z3 = g0*norms[0],g1*norms[1]+g0*g0*norms[2],g2*norms[3]+g0*g1*norms[4]+g0**3*norms[5]
        dz3 = g3*norms[3]+(g1*g1+g0*g2)*norms[4]+3*g0*g0*g1*norms[5]
        layer = g1*norms[1]+g2*norms[3]
        ln = np.linalg.norm((np.eye(len(a))-np.ones((len(a),1))*pi)@np.diag(c),np.inf)
        residual = dz3+g0*(ln*z3+contrast*(z1+z2+z3)**2)
        ff = contrast*(4*mix*layer*g1/gap**2+2*n*residual*t*g0)
        budgets.append((max(1.,6*n*contrast*g0,3*(z1+z2+z3)),ff,2*n*contrast**2*t*g0*g0,residual))
    threshold, ff, hh = max(v[0] for v in budgets),sum(v[1] for v in budgets),max(v[2] for v in budgets)
    dd, eps = abs(delta)*g0**3/3,1/math.sqrt(1+hh)
    m0 = math.ceil(max(1.,4/eps**2,(threshold*eps)**2))
    factor = ce/(math.pi*(gamma-1)*gamma)
    j5 = sum(math.comb(5,j)*math.factorial(j)/decay**(j+1) for j in range(6))
    inner = math.e*factor*(ff+dd*hh)*j5
    ee = 2/decay+dd*(eps**3/decay+3*eps*eps/decay**2+6*eps/decay**3+6/decay**4)
    outer = factor*math.exp(decay)*ee*(8/(math.e*decay*eps))**8
    return np.array([g0,g1,g2,g3]),ce,decay,eps,m0,inner+outer,dd,budgets


def riccati(s, t, kappa, xi, rho):
    b = kappa-rho*xi*s; d = np.sqrt(b*b-xi*xi*s*(s-1))
    g = (b-d)/(b+d)
    return (b-d)*(-np.expm1(-d*t))/(xi*xi*(1-g*np.exp(-d*t)))


def transforms(u, m, q, a, params):
    kappa,xi,rho,gamma,t,v0 = params
    pi,_,_,delta,_,_ = chain(q,a); rev = q.T*pi[None,:]/pi[:,None]
    s, n = gamma+1j*np.atleast_1d(u),len(np.atleast_1d(u))
    def rhs(time,y):
        b, integral = y[:2*n].reshape(2,n)
        f, r = y[2*n:].reshape(2,len(a),n)
        return np.r_[xi*xi*b*b/2+(rho*xi*s-kappa)*b+s*(s-1)/2,b,
                     (m*q@f+a[:,None]*b*f).ravel(),(m*rev@r+a[:,None]*b*r).ravel()]
    sol = solve_ivp(rhs,(0.,t),np.r_[np.zeros(2*n),np.ones(2*len(a)*n)].astype(complex),
                    method='DOP853',rtol=5e-14,atol=5e-16,dense_output=True)
    assert sol.success
    b, integral = sol.y[:2*n,-1].reshape(2,n)
    assert np.max(abs(b-riccati(s,t,kappa,xi,rho))/(1+abs(b))) < 3e-11
    f,r = sol.y[2*n:,-1].reshape(2,len(a),n)
    avg = np.exp(v0*b+(pi@a)*integral)
    return np.exp(v0*b)*(pi@(f-r)),avg*delta*b**3/3,sol


def prices(m, order, q, a, params):
    nodes, weights = leggauss(order)
    u, ww = [],[]
    for left,right in zip((0.,2.,5.,10.,20.),(2.,5.,10.,20.,40.)):
        u.extend((left+right)/2+(right-left)*nodes/2); ww.extend((right-left)*weights/2)
    u,ww = np.array(u),np.array(ww)
    actual,coefficient,_ = transforms(u,m,q,a,params)
    strikes = np.array([-.2,0.,.2]); alpha,gamma = params[3]-1,params[3]
    kernel = np.exp(-1j*np.outer(strikes,u))/((alpha+1j*u)*(gamma+1j*u))
    return np.exp(-alpha*strikes)[None,:]*np.real(np.array([kernel@(ww*actual),kernel@(ww*coefficient)]))/math.pi


def main():
    q = np.array([[-1.,1.,0.],[0.,-1.,1.],[1.,0.,-1.]])
    a, params = np.array([1.,2.,3.]),(2.,.3,-.6,2.,1.,.04)
    kappa,xi,rho,gamma,t,v0 = params
    pi,c,x,delta,mix,gap = chain(q,a)
    assert abs(delta+2/27) < 3e-12 and np.all(2*a > xi*xi)
    sc,sd = sp.Matrix([-1,0,1]),sp.diag(-1,0,1)
    ss = (sp.Matrix(q.astype(int))+sp.ones(3)/3).inv()-sp.ones(3)/3
    exact = (sc.T*(ss**2*sd*ss*sc-ss*sd*ss**2*sc))[0]/3
    assert exact == -sp.Rational(2,27)
    print(f'Exact rational cycle Delta={exact}; strict Feller check: PASS')
    gg,ce,decay,eps,m0,total,dd,budgets = bounds(q,a,kappa,xi,rho,gamma,2.,t,v0)
    print(f'Q={q.tolist()}, a={a}, kappa={kappa}, xi={xi}, rho={rho}, gamma=p=2, T={t}, v0={v0}; Delta={delta:.12g}')
    print(f'g0..g3={gg}; C={ce:.9g}; beta={decay:.9g}; epsilon={eps:.9g}; m0={m0}; A_K for |k|<=.2={math.exp(.2)*total:.9g}')
    checks = 0
    for frequency in np.r_[0.,np.geomspace(1e-3,1e5,30)]:
        s,w = gamma+1j*frequency,1+frequency
        for time in np.r_[0.,np.geomspace(1e-5,2.,20)/w]:
            b = riccati(s,time,kappa,xi,rho)
            bp = xi*xi*b*b/2+(rho*xi*s-kappa)*b+s*(s-1)/2
            bpp = (xi*xi*b+rho*xi*s-kappa)*bp
            bppp = xi*xi*bp*bp+(xi*xi*b+rho*xi*s-kappa)*bpp
            assert np.all(abs(np.array([b,bp,bpp,bppp])) <= gg*w**np.arange(1,5)+3e-9)
            for direction,generator in enumerate((q,q.T*pi[None,:]/pi[:,None])):
                _,_,xx,_,_,_ = chain(generator,a)
                z = [-b*xx[0],-bp*xx[1]+b*b*xx[2],-bpp*xx[3]+b*bp*xx[4]+b**3*xx[5]]
                dz = [-bp*xx[0],-bpp*xx[1]+2*b*bp*xx[2],-bppp*xx[3]+(bp*bp+b*bpp)*xx[4]+3*b*b*bp*xx[5]]
                speed = 2*w
                outer = sum(v/speed**(j+1) for j,v in enumerate(z))
                deriv = sum(v/speed**(j+1) for j,v in enumerate(dz))
                ll = (np.eye(3)-np.ones((3,1))*pi)@np.diag(c)
                rr = deriv-speed*generator@outer-b*(c+ll@outer-outer*(pi@(c*outer)))
                assert max(abs(rr)) <= budgets[direction][3]*w**4/speed**3+3e-9
            checks += 1
    print(f'{checks} frequency/time checks through the third Riccati derivative, plus both weighted outer residuals: PASS')
    for speed in (m0,2*m0,4*m0):
        w = eps*math.sqrt(speed)
        assert max(v[0] for v in budgets)*w <= speed*(1+3e-12)
        assert max(v[2] for v in budgets)*w*w/speed <= 1+3e-12
        for frequency in (0.,1.,10.):
            s,ww = gamma+1j*frequency,1+frequency
            b0p = s*(s-1)/2; b0pp = (rho*xi*s-kappa)*b0p
            layer = -b0p*x[1]/speed**2-b0pp*x[3]/speed**3
            assert np.max(abs(layer)) <= (gg[1]*max(abs(x[1]))+gg[2]*max(abs(x[3])))*ww**2/speed**2+3e-12
    independent = []
    for strike in (-.2,0.,.2):
        def integrand(u):
            s = gamma+1j*u; bb = kappa-rho*xi*s; d = np.sqrt(bb*bb-xi*xi*s*(s-1)); g = (bb-d)/(bb+d)
            ib = ((bb-d)*t-2*(np.log1p(-g*np.exp(-d*t))-np.log1p(-g)))/(xi*xi)
            b = riccati(s,t,kappa,xi,rho)
            return float(np.real(np.exp(-1j*u*strike+v0*b+(pi@a)*ib)*delta*b**3/(3*(1+1j*u)*(2+1j*u))))
        independent.append(math.exp(-strike)*quad(integrand,0.,40.,epsabs=1e-12,epsrel=1e-11)[0]/math.pi)
    print(f'Independent closed-Riccati/adaptive-quadrature d3={independent}; growing-cutoff and initial-layer checks: PASS')
    errors,gaps,refinement = [],[],0.
    for speed in (8.,16.,32.,64.,128.):
        lo,hi = [prices(speed,order,q,a,params) for order in (24,48)]
        refinement = max(refinement,np.max(abs(lo-hi)))
        assert np.max(abs(hi[1]-independent)) < 3e-11
        gaps.append(max(abs(hi[0]))); errors.append(max(abs(hi[0]-hi[1]/speed**3)))
        print(f'm={speed:.0f}: call gaps={hi[0]}; d3={hi[1]}; max corrected error={errors[-1]:.9g}; m^4 error={speed**4*errors[-1]:.9g}')
    assert 6 < gaps[-2]/gaps[-1] < 10 and 10 < errors[-2]/errors[-1] < 24
    assert refinement < 3e-12 and refinement < errors[-1]/10
    upper,min_speed = 40.,8.
    # |D|>=u^2 and u>=U: w^3/u^2<=u+3+3/U+1/U^2.
    tail = math.exp(.2)*ce*math.exp(-decay*upper)/math.pi
    tail *= 2/(decay*upper**2)+dd/min_speed**3*((upper+3+3/upper+1/upper**2)/decay+1/decay**2)
    assert tail < errors[-1]/10
    print(f'Order-doubling discrepancy={refinement:.3e}; exact-envelope combined U=40 tail bound at m>=8={tail:.3e}')
    print('PASS')


if __name__ == '__main__':
    main()
