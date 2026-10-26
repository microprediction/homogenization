"""A short-maturity three-quote alias near equal Erlang-stage rates.

Symbolic identities prove the rescaled implicit-function limit. Numerical
roots illustrate that exact analytic branch; they are not interval enclosures.
"""
import mpmath as mp
import sympy as sp


def check(condition):
    if not condition:
        raise AssertionError("Erlang quote-crossover certificate failed")


def symbolic():
    kap, d, x, y, eps, eta = sp.symbols('kap d x y eps eta', nonzero=True)
    u = [[sp.Integer(1)]*4]
    for n in range(1, 6):
        u.append([sp.expand(((x if j < 2 else y)*(u[n-1][(j+1) % 4]-u[n-1][j])
                  +sum((d if j < 2 else -d)*(-kap)**h/sp.factorial(h)*u[n-1-h][j]
                       for h in range(1, n)))/n) for j in range(4)])
    c4 = sp.factor(sp.Matrix([u[n] for n in (0, 2, 3, 4)]).det())
    c5 = sp.factor(sp.Matrix([u[n] for n in (0, 2, 3, 5)]).det())
    check(c4 == d**3*kap**3*x*y*(x-y)/18)
    check(sp.factor(c5.subs(y, x)-2*d**4*kap**4*x**2/45) == 0)
    t = sp.symbols('t1:4')
    v4 = sp.Matrix([[z**p for p in (2, 3, 4)] for z in t]).det()
    v5 = sp.Matrix([[z**p for p in (2, 3, 5)] for z in t]).det()
    check(sp.factor(v5-v4*sum(t)) == 0)
    scale = d**3*kap**3*x**2*v4/18
    limit = sp.limit((-c4.subs(y, x+eps*eta)*v4/eps-c5.subs(y, x+eps*eta)*v5)/scale, eps, 0)
    check(sp.factor(limit-eta+sp.Rational(4, 5)*d*kap*sum(t)) == 0)
    minor = sp.Matrix([[u[n][j]-u[n][3] for j in (0, 1)] for n in (2, 3)]).det()
    check(sp.factor(minor+d**2*kap**2*x/3) == 0)
    print("Exact symbolic limit F(0,eta)=eta-(4/5)*d*kappa*(t1+t2+t3); derivative 1")
    print("Exact leading rank-two minor: -d^2*kappa^2*x/3 times the two-node determinant")
    # Exact continuum bound for kappa=d=1, x=2, every y in [1,3].
    check(sp.factor(c5.subs({kap: 1, d: 1, x: 2})-y*(y+2)*(2*y-3)/45) == 0)
    s = sp.Matrix([[u[n][j]-u[n][3] for j in range(3)] for n in range(2, 6)])
    s = s.subs({kap: 1, d: 1, x: 2})
    frob_bound = sum(sum(abs(c)*3**p[0] for p, c in sp.Poly(a, y).terms())**2 for a in s)
    frob_bound3 = sum(sum(abs(c)*3**p[0] for p, c in sp.Poly(a, y).terms())**2
                      for a in s[:3, :])
    # |C4|>=1/36 outside |y-2|<1/4; C5>=7/96>1/36 inside.
    check(sp.Rational(7, 96) > sp.Rational(1, 36))
    return 1/(18*frob_bound), 1/(18*frob_bound3)


def response(eps, eta, order=34, nodes=(1, 2, 3)):
    x, y = mp.mpf(2), 2+eps*eta
    u = [[mp.mpf(1)]*4]
    for n in range(1, order+1):
        u.append([((x if j < 2 else y)*(u[n-1][(j+1) % 4]-u[n-1][j])
                   +sum((1 if j < 2 else -1)*(-1)**h/mp.factorial(h)*u[n-1-h][j]
                        for h in range(1, n)))/n for j in range(4)])
    values = [[mp.polyval([row[j] for row in reversed(u)], eps*t) for j in range(4)]
              for t in nodes]
    return mp.matrix([[a[j]-a[3] for j in range(3)] for a in values]), values


def four_quotes(gamma, branch_eta):
    nodes = (1, 2, 3, 4)
    w = sp.Matrix([[t**n for n in range(2, 6)] for t in nodes])
    check(w.det() == 6912)
    # sigma_min(W)>=det(W)/||W||_F^3>=det(W)/(sum|W_ij|)^3.
    alpha = gamma*w.det()/sum(abs(a) for a in w)**3
    # Radius one: M<=exp(5+e)<3^8 and 8*sqrt(3)<16.
    beta = sp.Integer(16)*3**8*4**6
    to_mp = lambda a: mp.mpf(int(a.p))/int(a.q)
    alpha, beta = to_mp(alpha), to_mp(beta)
    recovery, minimum, residual = mp.mpf(0), mp.inf, mp.mpf(0)
    prior = [mp.mpf(j)/10 for j in (1, 2, 3, 4)]
    for denominator in (128, 256, 512, 1024):
        eps = mp.mpf(1)/denominator
        for eta in (-1/eps, -mp.mpf('.1')/eps, 0, mp.mpf('.1')/eps, 1/eps, branch_eta):
            h, values = response(eps, eta, nodes=nodes)
            short, _ = response(eps, eta, order=5, nodes=nodes)
            residual = max(residual, mp.sqrt(sum(abs(a)**2 for a in h-short))/(beta*eps**6))
            minimum = min(minimum, mp.svd(h, compute_uv=False)[2]/eps**5)
            observed = mp.matrix([sum(p*v for p, v in zip(prior, a))-a[3] for a in values])
            fitted = list(mp.qr_solve(h, observed)[0]); fitted.append(1-sum(fitted))
            recovery = max(recovery, max(abs(a-b) for a, b in zip(prior, fitted)))
    check(residual <= 1+mp.mpf('1e-12') and recovery < mp.mpf('1e-12'))
    print(f"24 four-quote recoveries, including near-alias designs: prior error {float(recovery):.3e}; min sigma_min/epsilon^5 {float(minimum):.6e}")
    print(f"Exact continuum-bound constants: alpha={float(alpha):.6e}, beta={float(beta):.0f}")
    with mp.workdps(150):
        eps = alpha/(4*beta)
        check(eps < mp.mpf(1)/8)
        for y in (1, 2, 3):
            h, _ = response(eps, (y-2)/eps, nodes=nodes)
            check(mp.svd(h, compute_uv=False)[2]/eps**5 >= alpha/2)
        print(f"Three checks inside the proved uniform range: epsilon={float(eps):.6e}; remainder/bound max {float(residual):.3e}")


def three_quotes_separated(gamma):
    nodes = (1, 2, 3)
    w = sp.Matrix([[t**n for n in range(2, 5)] for t in nodes])
    check(w.det() == 72)
    # sigma_min(W)>=det(W)/||W||_F^2>=det(W)/(sum|W_ij|)^2.
    alpha = gamma*w.det()/sum(abs(a) for a in w)**2
    # Radius one: M<=exp(5+e)<3^8 and the 3-by-3 tail factor is 12.
    beta = sp.Integer(12)*3**8*3**5
    to_mp = lambda a: mp.mpf(int(a.p))/int(a.q)
    alpha, beta = to_mp(alpha), to_mp(beta)
    recovery, minimum, residual = mp.mpf(0), mp.inf, mp.mpf(0)
    prior = [mp.mpf(j)/10 for j in (1, 2, 3, 4)]
    rates = (1, mp.mpf('1.5'), mp.mpf('1.75'), mp.mpf('2.25'), mp.mpf('2.5'), 3)
    for denominator in (128, 256, 512, 1024):
        eps = mp.mpf(1)/denominator
        for y in rates:
            h, values = response(eps, (y-2)/eps, nodes=nodes)
            short, _ = response(eps, (y-2)/eps, order=4, nodes=nodes)
            residual = max(residual, mp.sqrt(sum(abs(a)**2 for a in h-short))/(beta*eps**5))
            minimum = min(minimum, mp.svd(h, compute_uv=False)[2]/eps**4)
            observed = mp.matrix([sum(p*v for p, v in zip(prior, a))-a[3] for a in values])
            fitted = list(mp.lu_solve(h, observed)); fitted.append(1-sum(fitted))
            recovery = max(recovery, max(abs(a-b) for a, b in zip(prior, fitted)))
    check(residual <= 1+mp.mpf('1e-12') and recovery < mp.mpf('1e-12'))
    print(f"24 separated-rate three-quote recoveries: prior error {float(recovery):.3e}; min sigma_min/epsilon^4 {float(minimum):.6e}")
    print(f"Separated-rate exact constants: alpha={float(alpha):.6e}, beta={float(beta):.0f}")
    with mp.workdps(150):
        eps = alpha/(4*beta)
        check(eps < mp.mpf(1)/8)
        for y in (1, mp.mpf('1.75'), mp.mpf('2.25'), 3):
            h, _ = response(eps, (y-2)/eps, nodes=nodes)
            check(mp.svd(h, compute_uv=False)[2]/eps**4 >= alpha/2)
        print(f"Four separated-rate checks inside the proved uniform range: epsilon={float(eps):.6e}; remainder/bound max {float(residual):.3e}")


def main():
    gamma, gamma3 = symbolic()
    mp.mp.dps = 90
    errors, etas = [], []
    for denominator in (128, 256, 512, 1024):
        eps = mp.mpf(1)/denominator
        # d=kappa=1, x=2, nodes (1,2,3): scale=16, predicted eta*=24/5.
        def equation(eta):
            return mp.det(response(eps, eta)[0])/(16*eps**10)
        check(equation(4) < 0 < equation(6))
        eta = mp.findroot(equation, (4, 6), tol=mp.mpf('1e-60'))
        h, values = response(eps, eta)
        h_long, _ = response(eps, eta, order=42)
        check(mp.mnorm(h-h_long, mp.inf) < mp.mpf('1e-12'))
        # Cross product of the first two rows is the exact-kernel construction.
        v = [h[0, (j+1) % 3]*h[1, (j+2) % 3]-h[0, (j+2) % 3]*h[1, (j+1) % 3]
             for j in range(3)]
        check(abs(v[2]/(-mp.mpf(8)/3*eps**5)-1) < mp.mpf('.2'))
        w = v+[-sum(v)]
        w = [a/max(abs(b) for b in w) for a in w]
        priors = [[mp.mpf(1)/4+sign*a/8 for a in w] for sign in (-1, 1)]
        check(min(a for p in priors for a in p) >= mp.mpf(1)/8)
        alias = max(abs(sum((priors[1][j]-priors[0][j])*a[j] for j in range(4))) for a in values)
        errors.append(abs(eta-mp.mpf(24)/5)); etas.append(eta)
        check(alias < mp.mpf('1e-12') and abs(equation(eta)) < mp.mpf('1e-12'))
        print(f"epsilon=1/{denominator}: eta={float(eta):.12f}, y={float(2+eps*eta):.12f}; price-alias residual {float(alias):.3e}")
    # Independent direct ODE at the last branch root, not the Taylor recursion.
    eps, eta = mp.mpf(1)/1024, etas[-1]
    def rhs(t, a):
        return [(2 if j < 2 else 2+eps*eta)*(a[(j+1) % 4]-a[j])
                +(1 if j < 2 else -1)*mp.expm1(-t)*a[j] for j in range(4)]
    solution = mp.odefun(rhs, 0, [mp.mpf(1)]*4, tol=mp.mpf('1e-55'), degree=35)
    ode_error = max(abs(solution(eps*t)[j]-values[t-1][j]) for t in (1, 2, 3) for j in range(4))
    check(ode_error < mp.mpf('1e-12'))
    print(f"Independent ODE amplitude discrepancy {float(ode_error):.3e}; eta convergence order {float(mp.log(errors[-2]/errors[-1])/mp.log(2)):.6f}")
    check(errors[-1] < errors[0] and errors[-1] < mp.mpf('.1'))
    three_quotes_separated(gamma3)
    four_quotes(gamma, etas[-1])
    print("PASS")


if __name__ == "__main__":
    main()
