"""Global switched-Heston expansion: floating regression, not interval arithmetic."""
import math

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad, solve_ivp
from scipy.linalg import expm


def chain_data(generator, immigration):
    n = len(immigration)
    pi = np.linalg.solve(generator.T+np.ones((n,n)),np.ones(n))
    centered = immigration-pi@immigration
    chi = np.linalg.solve(generator+np.ones((n,1))*pi,centered)
    weighted = np.sqrt(pi)[:,None]*generator/np.sqrt(pi)[None,:]
    gap = np.linalg.eigvalsh(-(weighted+weighted.T)/2)[1]
    mixing = 1/math.sqrt(min(pi))
    assert min(pi) > 0 and gap > 0 and abs(pi@chi) < 3e-12
    return pi, chi, -pi@(centered*chi), mixing, gap


def constants(kappa, xi, rho, gamma, p, immigration, t, v0, generator):
    eta, r, q = 1-rho*rho, p*gamma, p/(p-1)
    ar = kappa-rho*xi*r
    dr = math.sqrt(ar*ar-xi*xi*r*(r-1))
    assert ar > 0 and dr > 0 and eta > 0
    a = kappa-rho*xi*gamma
    d0, c = math.sqrt(a*a-xi*xi*gamma*(gamma-1)), xi*math.sqrt(eta)
    ell = -2*rho*xi*a-xi*xi*(2*gamma-1)
    h0, r0 = 1+abs(ell)/(2*d0*c), a/d0+abs(rho)/math.sqrt(eta)
    z = 2/(1+r0)
    nu = min(z/2, -math.expm1(-z/(2*math.sqrt(h0))))
    cb = 2*max(a+math.sqrt(h0)*d0, abs(rho)*xi+math.sqrt(h0)*c)/(xi*xi*nu)
    cd = xi*xi*cb*cb/2+max(a,abs(rho)*xi)*cb+gamma*gamma/2
    pi, chi, green, mixing, gap = chain_data(generator,immigration)
    contrast, n = max(abs(immigration-pi@immigration)), mixing/gap
    h = t*abs(green)*cb*cb
    r3 = t*cb*contrast*n*((14/3)*n*contrast**2*cb**2+cd*max(abs(chi)))
    eps = 1/math.sqrt(1+h)
    m0 = math.ceil(max(4/eps**2, (6*n*contrast*cb*eps)**2, r3*r3*eps**6))
    moment = math.exp((max(immigration)*t+v0)*(ar-dr)/(xi*xi))
    c0 = 2**(2*min(immigration)/xi**2)*math.exp(min(immigration)*kappa*t/xi**2)
    ce, decay = moment**(1/p)*c0**(1/q), min(immigration)*t*math.sqrt(eta)/(xi*math.sqrt(q))
    j = lambda n: sum(math.comb(n,i)*math.factorial(i)/decay**(i+1) for i in range(n+1))
    factor = ce/(math.pi*(gamma-1)*gamma)
    inner = factor*(math.exp(2)*r3*j(3)+math.e*h*h*j(4)/2)
    e = (2+h*eps*eps)/decay+2*h*eps/decay**2+2*h/decay**3
    outer = factor*math.exp(decay)*e*(4/(math.e*decay*eps))**4
    return cb, cd, nu, h, r3, eps, m0, ce, decay, inner+outer


def closed_b(s, t, kappa, xi, rho):
    b = kappa-rho*xi*s
    d = np.sqrt(b*b-xi*xi*s*(s-1))
    g = (b-d)/(b+d)
    denominator = 1-g*np.exp(-d*t)
    return (b-d)*(-np.expm1(-d*t))/(xi*xi*denominator), denominator


def transforms(u, speed, t, immigration, kappa, xi, rho, gamma, v0, generator):
    s, n = gamma+1j*np.atleast_1d(u), len(np.atleast_1d(u))
    pi, _, green, _, _ = chain_data(generator,immigration)
    def rhs(_, y):
        b, ib, ib2 = y[:3*n].reshape(3,n)
        amp = y[3*n:].reshape(len(pi),n)
        return np.r_[xi*xi*b*b/2+(rho*xi*s-kappa)*b+s*(s-1)/2,
            b, b*b, (speed*generator@amp+immigration[:,None]*b*amp).ravel()]
    sol = solve_ivp(rhs,(0.,t),np.r_[np.zeros(3*n),np.ones(len(pi)*n)].astype(complex),
                    rtol=2e-11,atol=2e-13,method='BDF' if speed>1000 else 'DOP853',dense_output=True)
    assert sol.success
    b, ib, ib2 = sol.y[:3*n,-1].reshape(3,n)
    averaged = np.exp(b*v0+(pi@immigration)*ib)
    exact = np.exp(b*v0)*(pi@sol.y[3*n:,-1].reshape(len(pi),n))
    h = green*ib2
    return exact, averaged, h, sol


def call_values(speed, order, immigration, generator):
    nodes, weights = leggauss(order)
    u, weights = (nodes+1)*120, weights*120/math.pi
    exact, avg, h, _ = transforms(u,speed,1.,immigration,1.4,.8,-.6,2.,.04,generator)
    strikes = np.array([-.2,0.,.2])
    phase = np.exp(-1j*np.outer(strikes,u))/((1+1j*u)*(2+1j*u))
    return np.exp(-strikes)[None,:]*np.real(
        np.array([phase@(weights*exact),phase@(weights*avg),phase@(weights*avg*h)]))


def main():
    gamma, p, t, v0 = 2., 2., 1., .04
    immigration = np.array([.2,.5])
    generator = np.array([[-1.,1.],[1.,-1.]])
    ode_error, bounds, ratios = 0., 0, 0
    for kappa, xi, rho in ((1.4,.8,-.6),(5.,.8,.6),(2.,.4,0.)):
        cb, cd, nu, *_ = constants(kappa,xi,rho,gamma,p,immigration,t,v0,generator)
        u = np.array([0.,1.,5.,20.,80.])
        _, _, _, sol = transforms(u,4.,t,immigration,kappa,xi,rho,gamma,v0,generator)
        for time in np.r_[0.,np.geomspace(1e-6,1.,15)]:
            actual = sol.sol(time).reshape(5,len(u))[0]
            closed, _ = closed_b(gamma+1j*u,time,kappa,xi,rho)
            ode_error = max(ode_error,np.max(abs(actual-closed)/(1+abs(closed))))
        for frequency in np.r_[0.,np.geomspace(1e-4,1e6,40)]:
            s = gamma+1j*frequency
            for time in np.r_[0.,np.geomspace(1e-5,10.,20)/(1+frequency)]:
                b, den = closed_b(s,time,kappa,xi,rho)
                bp = xi*xi*b*b/2+(rho*xi*s-kappa)*b+s*(s-1)/2
                assert abs(den) >= nu-2e-12
                assert abs(b) <= cb*(1+frequency)+2e-9
                assert abs(bp) <= cd*(1+frequency)**2+2e-9
                bounds += 1
    assert ode_error < 3e-9
    print(f'{bounds} denominator/B/B-prime envelopes; closed/ODE relative error {ode_error:.3e}')
    kappa, xi, rho = 1.4, .8, -.6
    cb, cd, nu, hcap, r3, eps, m0, ce, decay, total = constants(
        kappa,xi,rho,gamma,p,immigration,t,v0,generator)
    print(f'Fixture: immigration={immigration}, kappa={kappa}, xi={xi}, rho={rho}, '
          f'T={t}, v0={v0}, gamma={gamma}, p={p}')
    print(f'Cb={cb:.9g}; Cd={cd:.9g}; epsilon={eps:.9g}; m0={m0}; '
          f'A_K for |k|<=.2: {math.exp(.2)*total:.9g}')
    for speed in (m0,2*m0,4*m0):
        upper = eps*math.sqrt(speed)-1
        u = np.linspace(0.,upper,11)
        exact, avg, h, _ = transforms(u,speed,t,immigration,kappa,xi,rho,gamma,v0,generator)
        remainder = np.log(exact/avg)-h/speed
        bound = r3*(1+u)**3/speed**2
        assert np.all(abs(remainder) <= bound+3e-9)
        assert np.max(abs(h))/speed <= 1+3e-9 and np.max(bound) <= 1+3e-9
        ratios += len(u)
    print(f'{ratios} inner logarithmic-remainder bounds: PASS')
    strikes, residuals, refinement = np.array([-.2,0.,.2]), [], 0.
    for speed in (16.,32.,64.,128.):
        prices = []
        for order in (192,384):
            prices.append(call_values(speed,order,immigration,generator))
        refinement = max(refinement,np.max(abs(prices[0]-prices[1])))
        value = prices[-1]
        residual = value[0]-value[1]-value[2]/speed
        residuals.append(max(abs(residual)))
        if speed >= m0:
            assert max(abs(residual)) <= math.exp(.2)*total/speed**2+3e-9
        print(f'm={speed:.0f}: max call residual={max(abs(residual)):.9g}; '
              f'm^2 residual={speed**2*max(abs(residual)):.9g}')
    assert refinement < 3e-9
    assert 3 < residuals[-2]/residuals[-1] < 5
    # Exact tails and correction tail, independent of quadrature refinement.
    upper, speed = 240., 16.
    # |D(u)|>=u^2; both u^-2 and (1+1/u)^2 decrease for u>=upper.
    tail = math.exp(.2)*ce*math.exp(-decay*upper)/(math.pi*decay)
    tail *= 2/upper**2+hcap*(1+1/upper)**2/speed
    print(f'Quadrature refinement {refinement:.3e}; combined U=240 tail budget at m>=16: {tail:.3e}')
    for name, generator, immigration in (
        ('unequal',np.array([[-1.7,1.7],[.4,-.4]]),np.array([.2,.5])),
        ('cycle',np.array([[-3.,3.,0.],[0.,-2.,2.],[1.,0.,-1.]]),np.array([.2,.35,.5]))):
        pi, chi, green, mixing, gap = chain_data(generator,immigration)
        centered = immigration-pi@immigration
        integral = quad(lambda x: pi@(centered*(expm(x*generator)@centered)),0.,40.,epsabs=1e-10)[0]
        reverse = generator.T*pi[None,:]/pi[:,None]
        assert abs(integral-green) < 3e-9 and abs(chain_data(reverse,immigration)[2]-green) < 3e-12
        for time in (0.,.01,.1,1.,4.,10.):
            assert np.linalg.norm(expm(time*generator)-np.ones((len(pi),1))*pi,np.inf) <= mixing*math.exp(-gap*time)+3e-12
        cb, cd, nu, cap, r3, eps, m0, ce, decay, total = constants(
            kappa,xi,rho,gamma,p,immigration,t,v0,generator)
        for speed in (m0,2*m0):
            u = np.linspace(0.,eps*math.sqrt(speed)-1,7)
            actual, avg, h, _ = transforms(u,speed,t,immigration,kappa,xi,rho,gamma,v0,generator)
            assert np.max(abs(np.log(actual/avg)-h/speed)/(r3*(1+u)**3/speed**2+3e-9)) <= 1
        errors = []
        for speed in (16.,32.,64.,128.):
            values = [call_values(speed,order,immigration,generator) for order in (192,384)]
            refinement = max(refinement,np.max(abs(values[0]-values[1])))
            errors.append(max(abs(values[-1][0]-values[-1][1]-values[-1][2]/speed)))
        assert 3 < errors[-2]/errors[-1] < 5 and refinement < 3e-9
        print(f'{name}: Q={generator.tolist()}, immigration={immigration}, K={green:.10g}, m0={m0}, A_K={math.exp(.2)*total:.9g}; '
              f'call residuals at 16/128={errors[0]:.9g}/{errors[-1]:.9g}; refinement={refinement:.3e}')
    print('PASS')


if __name__ == '__main__':
    main()
