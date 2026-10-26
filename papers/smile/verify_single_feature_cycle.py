"""Cubic one-feature reversal signal; floating checks are not interval proofs."""
import math

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.linalg import expm


def data(q, c):
    n = len(c)
    pi = np.linalg.solve(q.T+np.ones((n,n)),np.ones(n))
    projector = np.ones((n,1))*pi
    s = np.linalg.inv(q+projector)-projector
    d = np.diag(c)
    dot = lambda x: pi@(c*x)
    scalars = np.array([dot(s@c),dot(s@s@c),dot(s@d@s@c),dot(s@s@s@c),
                       dot(s@s@d@s@c),dot(s@d@s@s@c),dot(s@d@s@d@s@c)])
    return pi, s, scalars


def linear_log(q, c, speed, b, t=1.):
    pi, _, _ = data(q,c)
    sol = solve_ivp(lambda time,f: speed*q@f+b(time)*c*f,(0.,t),np.ones(len(c),dtype=complex),
                    method='DOP853',rtol=3e-13,atol=3e-15)
    assert sol.success
    return np.log(pi@sol.y[:,-1])


def ramp_budget(q, c):
    """Analytic coefficient-sum upper bounds for b(t)=t on [0,1]."""
    pi, s, a = data(q,c)
    d, norm = np.diag(c), lambda x: np.linalg.norm(x,np.inf)
    weighted = np.sqrt(pi)[:,None]*q/np.sqrt(pi)[None,:]
    gap, mixing = np.linalg.eigvalsh(-(weighted+weighted.T)/2)[1], 1/math.sqrt(min(pi))
    z1 = norm(s@c)
    z2 = norm(s@s@c)+norm(s@d@s@c)
    linear = 2*s@s@d@s@c+s@d@s@s@c
    cubic = s@s@c*a[0]-s@d@s@d@s@c
    z3 = norm(linear)+norm(cubic)
    dz3 = norm(linear)+3*norm(cubic)
    z, cc = z1+z2+z3, norm(c)
    lip = 8*cc/3
    layer = norm(s@s@c)
    residual = dz3+norm((np.eye(len(c))-np.ones((len(c),1))*pi)@d)*z3+cc*z*z
    threshold = max(1.,6*mixing*cc/gap,3*z,2*mixing*lip/gap)
    constant = cc*(4*mixing*layer/gap**2+2*mixing*residual/gap)
    return threshold, constant


def three_cycle_factorization():
    """Check the exact orientation/Vandermonde formula, including both blind cases."""
    cases = 0
    for p, q in ((1, 2), (2, 5), (5, 1), (2, 2)):
        generator = sp.Matrix([[-p-q, p, q], [q, -p-q, p], [p, q, -p-q]])
        inverse = (generator+sp.ones(3)/3).inv()-sp.ones(3)/3
        for values in ((-1, 0, 1), (-2, 1, 1), (3, -4, 1)):
            c = sp.Matrix(values); diagonal = sp.diag(*values)
            delta = ((c.T*inverse**2*diagonal*inverse*c)[0]
                     -(c.T*inverse*diagonal*inverse**2*c)[0])/3
            vandermonde = (c[0]-c[1])*(c[1]-c[2])*(c[2]-c[0])
            target = -sp.Rational(p-q, 27)*vandermonde/(p*p+p*q+q*q)**2
            assert sp.factor(delta-target) == 0
            cases += 1
    return cases


def main():
    q = np.array([[-1.,1.,0.],[0.,-1.,1.],[1.,0.,-1.]])
    c = np.array([-1.,0.,1.])
    pi, s, a = data(q,c)
    reverse = q.T*pi[None,:]/pi[:,None]
    _, _, ar = data(reverse,c)
    assert np.max(abs(a[[0,1,2,3,6]]-ar[[0,1,2,3,6]])) < 3e-12
    assert abs(a[4]-ar[5])+abs(a[5]-ar[4]) < 3e-12
    # Exact rational certificate, independent of the floating group inverse.
    sq, sc, spi = sp.Matrix(q.astype(int)), sp.Matrix([-1,0,1]), sp.ones(1,3)/3
    proj = sp.ones(3,1)*spi
    ss = (sq+proj).inv()-proj
    sd = sp.diag(-1,0,1)
    u, v = (spi*sd*ss*ss*sd*ss*sc)[0], (spi*sd*ss*sd*ss*ss*sc)[0]
    assert (u,v,u-v,(u-v)/3) == (-sp.Rational(1,27),sp.Rational(1,27),-sp.Rational(2,27),-sp.Rational(2,81))
    factor_cases = three_cycle_factorization()
    time, eps = sp.symbols('t eps')
    bt = time+time**2/5+sp.I*time**3/10
    z1 = -bt*ss*sc
    z2 = -sp.diff(bt,time)*ss**2*sc+bt**2*ss*sd*ss*sc
    z3 = -sp.diff(bt,time,2)*ss**3*sc+bt*sp.diff(bt,time)*(2*ss**2*sd*ss*sc+ss*sd*ss**2*sc)+bt**3*(ss**2*sc*(spi*sd*ss*sc)[0]-ss*sd*ss*sd*ss*sc)
    outer = eps*z1+eps**2*z2+eps**3*z3
    residual = outer.diff(time)-sq*outer/eps-bt*(sc+(sp.eye(3)-proj)*sd*outer-outer*(spi*sd*outer)[0])
    assert all(sp.expand(x).coeff(eps,j) == 0 for x in residual for j in range(3))
    rng = np.random.default_rng(721)
    for nstates in (2,3,4,5):
        for _ in range(8):
            generator = rng.uniform(.1,1.5,(nstates,nstates)); np.fill_diagonal(generator,0.)
            generator -= np.diag(generator.sum(axis=1))
            law, _, _ = data(generator,np.zeros(nstates))
            feature = rng.normal(size=nstates); feature -= law@feature
            _, _, forward = data(generator,feature)
            _, _, backward = data(generator.T*law[None,:]/law[:,None],feature)
            assert np.max(abs(forward[[0,1,2,3,6]]-backward[[0,1,2,3,6]])) < 3e-10
            assert np.max(abs(forward[[4,5]]-backward[[5,4]])) < 3e-10
    print('9 exact outer-recursion identities and 32 random-chain adjoint checks: PASS')
    print(f'{factor_cases} exact three-cycle orientation/Vandermonde factorizations: PASS')
    print(f'Exact cycle: Q={q.tolist()}, pi=(1,1,1)/3, c=(-1,0,1), U={u}, V={v}, Delta={u-v}, coefficient={(u-v)/3}')
    thresholds, budgets = zip(ramp_budget(q,c),ramp_budget(reverse,c))
    m0, budget, coefficient = max(thresholds), sum(budgets), float((u-v)/3)
    norm_exact = lambda x: max(sum(abs(x[i,j]) for j in range(x.cols)) for i in range(x.rows))
    for inverse in (ss,ss.T):
        lin = 2*inverse**2*sd*inverse*sc+inverse*sd*inverse**2*sc
        cub = inverse**2*sc*(spi*sd*inverse*sc)[0]-inverse*sd*inverse*sd*inverse*sc
        zz = [norm_exact(inverse*sc),norm_exact(inverse**2*sc)+norm_exact(inverse*sd*inverse*sc),norm_exact(lin)+norm_exact(cub)]
        rr = norm_exact(lin)+3*norm_exact(cub)+norm_exact((sp.eye(3)-proj)*sd)*zz[2]+sum(zz)**2
        assert zz == [sp.Rational(2,3),sp.Rational(5,9),sp.Rational(11,27)]
        assert norm_exact(inverse**2*sc) == sp.Rational(1,3) and rr == sp.Rational(2638,729)
    assert abs(m0-float(4*sp.sqrt(3))) < 3e-12
    assert abs(budget-float(23696*sp.sqrt(3)/2187)) < 3e-11
    print('Exact budget inputs, both directions: M=sqrt(3), lambda=3/2, Z=(2/3,5/9,11/27), I=1/3, R=2638/729')
    print('Exact threshold=4*sqrt(3); directional remainder constant=23696*sqrt(3)/2187')
    print(f'Ramp b(t)=t, T=1: threshold {m0:.9g}; directional remainder <= {budget:.9g}/m^4')
    print(f'Conservative rounded bound: m >= {math.ceil(m0)}, directional error <= {math.ceil(100*budget)/100:.2f}/m^4')
    errors, gaps = [], []
    for speed in (8.,16.,32.,64.,128.,256.):
        logs = [linear_log(generator,c,speed,lambda t:t) for generator in (q,reverse)]
        gap = (logs[0]-logs[1]).real
        error = abs(gap-coefficient/speed**3)
        gaps.append(abs(gap)); errors.append(error)
        if speed >= m0:
            assert error <= budget/speed**4+3e-11
        # Verify every individual log coefficient, not only the cancelling difference.
        for actual, scalars in zip(logs,(a,ar)):
            c1,c2,j,c3,uu,vv,w = scalars
            approximation = -c1/(3*speed)+(-c2/2+j/4)/speed**2+((2*uu+vv)/3+(c2*c1-w)/5)/speed**3
            assert abs(actual-approximation) <= max(budgets)/speed**4+3e-11
        print(f'm={speed:.0f}: gap={gap:.9g}; m^3 gap={speed**3*gap:.9f}; m^4 residual={speed**4*error:.9g}')
    assert 7 < gaps[-2]/gaps[-1] < 9 and 10 < errors[-2]/errors[-1] < 22
    assert abs(256**3*(-gaps[-1])-coefficient) < .001
    integrate = lambda x: complex(sp.integrate(x,(time,0,1)))
    end = complex(bt.subs(time,1))
    for generator, scalars in zip((q,reverse),(a,ar)):
        c1,c2,j,c3,uu,vv,w = scalars
        coeffs = [-c1*integrate(bt**2),-c2*end**2/2+j*integrate(bt**3),
                  -c3*integrate(bt*sp.diff(bt,time,2))+(2*uu+vv)*end**3/3+(c2*c1-w)*integrate(bt**4)]
        errs = [abs(linear_log(generator,c,m,lambda t:t+t*t/5+1j*t**3/10)-sum(v/m**(i+1) for i,v in enumerate(coeffs))) for m in (16.,32.,64.)]
        assert 8 < errs[-2]/errs[-1] < 24 and errs[-1] < 1e-5
    print('Complex cubic schedule, including nonzero b-double-prime: fourth-order individual-log residuals PASS')
    # Constant and symmetric schedules are exactly blind; reverse schedules agree.
    for speed in (1.,4.,16.):
        potential = np.diag(.7*c)
        assert abs(pi@expm(speed*q+potential)@np.ones(3)-pi@expm(speed*reverse+potential)@np.ones(3)) < 3e-10
        sym = lambda t:t*(1-t)
        assert abs(linear_log(q,c,speed,sym)-linear_log(reverse,c,speed,sym)) < 3e-10
        assert abs(linear_log(q,c,speed,lambda t:t)-linear_log(reverse,c,speed,lambda t:1-t)) < 3e-10
        # Independent forward-row Feynman--Kac solve uses the reversed duration schedule.
        row = solve_ivp(lambda t,p: speed*q.T@p+(1-t)*c*p,(0.,1.),pi.astype(complex),
                        method='DOP853',rtol=3e-13,atol=3e-15)
        assert row.success and abs(np.log(sum(row.y[:,-1]))-linear_log(q,c,speed,lambda t:t)) < 3e-10
    print('9 occupation/symmetric/reversed-schedule identities: PASS')
    # The Heston loading is checked independently, without assuming a price inversion.
    levels = np.array([1.,2.,3.])
    assert np.all(2*levels > .3**2) and np.max(abs(levels-pi@levels-c)) < 3e-12
    print('Positive switched CIR: a=(1,2,3), kappa=2, xi=.3, rho=-.6; strict Feller check: PASS')
    for argument in (2.+0j,2.+3j):
        def rhs(_,y):
            b = y[0]
            return np.array([.3**2*b*b/2+(-.6*.3*argument-2)*b+argument*(argument-1)/2])
        ric = solve_ivp(rhs,(0.,1.),[0j],method='DOP853',rtol=3e-13,atol=3e-15,dense_output=True)
        assert ric.success
        b = lambda t: ric.sol(t)[0]
        predicted = (a[4]-a[5])*b(1.)**3/3
        deviations = []
        for speed in (16.,32.,64.,128.):
            gap = linear_log(q,c,speed,b)-linear_log(reverse,c,speed,b)
            deviations.append(abs(speed**3*gap-predicted))
        assert deviations[-1] < .035*max(1.,abs(predicted))
        assert deviations[-1] < .65*deviations[-2]
        print(f'Heston kappa=2, xi=.3, rho=-.6, T=1, s={argument}: cubic target={predicted}; last error={deviations[-1]:.6g}')
    print('PASS')


if __name__ == '__main__':
    main()
