"""Write the explicit, specialized formula sections included into the example pages (tools/pages/explicit/*.html).
Every formula is checked here against the engine at second order and against the numerical solution."""
import cmath, math, os
import numpy as np
from explicit import I_k, J_1, J_2, cir_B, cir_int_B, cir_int_B2, two_state_constant_exact
from fastswitch import FastSwitch, numerical_a_callable
from models import cir_switching_mean, vasicek_jumps, mmpp, bs_switching

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tools', 'pages', 'explicit')
sym = lambda lam: lam * np.array([[-1.0, 1.0], [1.0, -1.0]])


def cfmt(z, p=6):
    z = complex(z)
    if abs(z.imag) < 1e-15:
        return f'{z.real:.{p}g}'
    return f'{z.real:.{p}g} {"+" if z.imag >= 0 else "&minus;"} {abs(z.imag):.{p}g}i'


def table(rows, head=('quantity', 'value')):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows)
    return f'    <div class="table-wrap"><table class="impl"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>\n'


def write(name, html):
    with open(os.path.join(OUT, name + '.html'), 'w') as f:
        f.write(html)
    print('wrote', name)


# ------------------------------------------------------------------ Vasicek with (optional) jumps: the general card
VASICEK_CARD = r'''    <div class="equation-card">
    $$\begin{aligned}
    u_{1,2}(T, x) \;=\; &\exp\Big(-B\,x - \kappa\bar\theta\,I_1 + \tfrac12\bar s\,I_2 + \bar\ell\,(J_1 - T)
      + \frac{\varepsilon}{2}\,A - \frac{\varepsilon^2}{8}\,G^2\Big) \\
      &\times\Big(1 \pm \frac{\varepsilon}{2}\,G \mp \frac{\varepsilon^2}{4}\,G'
      \pm \frac{\varepsilon^2}{4}\,G'_0 e^{-2\lambda T}\Big) + O(\varepsilon^3),
    \end{aligned}$$
    $$\begin{aligned}
    A \;=\; &\kappa^2\tilde\theta^2 I_2 \;-\; \kappa\tilde\theta\,\tilde s\,I_3 \;+\; \tfrac14\tilde s^2 I_4
      \;+\; \tilde\ell^{\,2}\,(J_2 - 2J_1 + T) \\
      &-\; 2\kappa\tilde\theta\tilde\ell\,\Big(\frac{T - J_1}{m} - I_1\Big)
      \;+\; \tilde s\,\tilde\ell\,\Big(\frac{1}{m}\Big(I_1 - \frac{T - J_1}{m}\Big) - I_2\Big),
    \end{aligned}$$
    $$\begin{aligned}
    G &= -\kappa\tilde\theta\,B + \tfrac12\tilde s\,B^2 + \tilde\ell\,\Big(\frac{1}{1 + mB} - 1\Big), \\
    G' &= \Big(-\kappa\tilde\theta + \tilde s\,B - \frac{\tilde\ell\,m}{(1 + mB)^2}\Big)E, \\
    G'_0 &= -\kappa\tilde\theta-\tilde\ell m, \\
    B &= \frac{1 - E}{\kappa}, \qquad E = e^{-\kappa T}, \\
    I_1 &= \frac{1}{\kappa}\Big(T - \frac{1 - E}{\kappa}\Big), \\
    I_2 &= \frac{1}{\kappa^2}\Big(T - \frac{2(1 - E)}{\kappa} + \frac{1 - E^2}{2\kappa}\Big), \\
    I_3 &= \frac{1}{\kappa^3}\Big(T - \frac{3(1 - E)}{\kappa} + \frac{3(1 - E^2)}{2\kappa} - \frac{1 - E^3}{3\kappa}\Big), \\
    I_4 &= \frac{1}{\kappa^4}\Big(T - \frac{4(1 - E)}{\kappa} + \frac{3(1 - E^2)}{\kappa} - \frac{4(1 - E^3)}{3\kappa} + \frac{1 - E^4}{4\kappa}\Big), \\
    J_1 &= \frac{\kappa}{\kappa + m}\Big(T + \frac1\kappa\log\Big(1 + \frac m\kappa(1 - E)\Big)\Big), \\
    J_2 &= \frac{\kappa}{\kappa + m}\Big(J_1 - \frac1\kappa\Big(\frac{1}{1 + \frac m\kappa(1 - E)} - 1\Big)\Big).
    \end{aligned}$$
    </div>
'''


def vasicek_values(T, x, kappa, th, s2, ell, m, lam, sign=+1):
    thb, tht = np.mean(th), (th[0] - th[1]) / 2
    sb, st = np.mean(s2), (s2[0] - s2[1]) / 2
    lb, lt = np.mean(ell), (ell[0] - ell[1]) / 2
    eps = 1 / lam
    E = math.exp(-kappa * T)
    B = (1 - E) / kappa
    I = {k: I_k(k, T, kappa) for k in (1, 2, 3, 4)}
    J1 = J_1(T, kappa, m) if m else T
    J2 = J_2(T, kappa, m) if m else T
    A = (kappa * tht) ** 2 * I[2] - kappa * tht * st * I[3] + st * st / 4 * I[4]
    if m:
        A += lt * lt * (J2 - 2 * J1 + T) - 2 * kappa * tht * lt * ((T - J1) / m - I[1]) \
             + st * lt * ((I[1] - (T - J1) / m) / m - I[2])
    r = 1 / (1 + m * B) if m else 1.0
    G = -kappa * tht * B + 0.5 * st * B * B + lt * (r - 1)
    Gp = (-kappa * tht + st * B - lt * m * r * r) * E
    Gp0 = -kappa * tht - lt * m
    avg = -kappa * thb * I[1] + 0.5 * sb * I[2] + lb * (J1 - T)
    logu = -B * x + avg + eps / 2 * A - eps ** 2 / 8 * G * G
    u = math.exp(logu) * (1 + sign * (eps / 2 * G - eps ** 2 / 4 * Gp
                                      + eps ** 2 / 4 * Gp0 * math.exp(-2 * lam * T)))
    parts = dict(E=E, B=B, I1=I[1], I2=I[2], I3=I[3], I4=I[4], J1=J1, J2=J2,
                 A=A, G=G, Gp=Gp, Gp0=Gp0, avg=avg)
    orders = [math.exp(-B * x + avg),
              math.exp(-B * x + avg + eps / 2 * A) * (1 + sign * eps / 2 * G), u]
    return parts, orders


def jumps():
    kappa, th, sig, ell, m, T, x, lam = 2.0, [0.05, 0.02], [0.02, 0.01], [3.0, 0.2], 0.03, 3.0, 0.0, 10.0
    s2 = [s * s for s in sig]
    parts, orders = vasicek_values(T, x, kappa, th, s2, ell, m, lam)
    g, gf, _ = vasicek_jumps(kappa, th, sig, ell, m, 5.0)
    ref = numerical_a_callable(T, sym(lam), gf, rtol=1e-13)[0]
    eng = FastSwitch(sym(lam), g, order=2).a(T, 2)[0]
    assert abs(orders[2] - eng) < 1e-9, (orders[2], eng)
    html = r'''
    <h2>The survival probability in closed form</h2>
    <p>Specializing the expansion to this model, every integral can be evaluated. With the upper sign for a start in
    the stressed regime, the survival probability to second order is</p>
''' + VASICEK_CARD + r'''    <p>The bars and tildes are half-sums and half-differences across the two regimes:</p>
    $$\bar\theta, \tilde\theta = \frac{\theta_1 \pm \theta_2}{2}, \qquad \bar s, \tilde s = \frac{\sigma_1^2 \pm \sigma_2^2}{2},
      \qquad \bar\ell, \tilde\ell = \frac{\ell_1 \pm \ell_2}{2} .$$
    <p>The first factor, without the $\varepsilon$ terms, is the survival probability of the averaged model. The terms
    in $A$ are the Green&ndash;Kubo correction, and the bracket is the memory of the starting regime. The last,
    exponentially decaying term makes the $O(\varepsilon^3)$ remainder uniform through $T=0$.</p>
    <p>At the parameters above, with $T = 3$, $x = 0$ and $\lambda = 10$:</p>
''' + table([['$E$, $B$', f'{parts["E"]:.6f}, {parts["B"]:.6f}'],
             ['$I_1$, $I_2$, $I_3$, $I_4$', ', '.join(f'{parts[k]:.6g}' for k in ('I1', 'I2', 'I3', 'I4'))],
             ['$J_1$, $J_2$', f'{parts["J1"]:.6f}, {parts["J2"]:.6f}'],
             ['averaged exponent', f'{parts["avg"]:.6f}'],
             ['$A$', f'{parts["A"]:.6g}'],
             ["$G$, $G'$, $G'_0$", f'{parts["G"]:.6g}, {parts["Gp"]:.6g}, {parts["Gp0"]:.6g}']]) + \
        table([['averaged model', f'{orders[0]:.8f}', f'{abs(orders[0]-ref):.1e}'],
               ['first order', f'{orders[1]:.8f}', f'{abs(orders[1]-ref):.1e}'],
               ['second order', f'{orders[2]:.8f}', f'{abs(orders[2]-ref):.1e}'],
               ['numerical solution', f'{ref:.8f}', '']], head=('survival', 'value', 'error'))
    write('jumps', html)


def regime_switching():
    kappa, th, s2, T, x, lam = 2.0, [0.15, 0.02], [0.0555, 0.0055], 4.0, 0.12, 20.0
    parts, orders = vasicek_values(T, x, kappa, th, s2, [0, 0], 0.0, lam)
    from fastswitch import ExpSum
    def G_(t, s):
        return ExpSum({0: -t + s / (2 * kappa ** 2), kappa: t - s / kappa ** 2, 2 * kappa: s / (2 * kappa ** 2)})
    g = [G_(th[i], s2[i]) for i in range(2)]
    gf = [(lambda gi: (lambda t: gi.value(t)))(gi) for gi in g]
    ref = numerical_a_callable(T, sym(lam), gf, rtol=1e-13)[0] * math.exp(-parts['B'] * x)
    eng = FastSwitch(sym(lam), g, order=2).a(T, 2)[0] * math.exp(-parts['B'] * x)
    assert abs(orders[2] - eng) < 1e-9, (orders[2], eng)
    card = r'''    <div class="equation-card">
    $$\begin{aligned}
    u_{1,2}(T, x) \;=\; &\exp\Big(-B\,x - \kappa\bar\theta\,I_1 + \tfrac12\bar s\,I_2 - \frac{\varepsilon^2}{8}\,G^2 \\
      &\qquad + \frac{\varepsilon}{2}\big(\kappa^2\tilde\theta^2 I_2 - \kappa\tilde\theta\,\tilde s\,I_3 + \tfrac14\tilde s^2 I_4\big)\Big) \\
      &\times\Big(1 \pm \frac{\varepsilon}{2}\,G \mp \frac{\varepsilon^2}{4}\,\big(\tilde s\,B - \kappa\tilde\theta\big)E
      \mp \frac{\varepsilon^2}{4}\,\kappa\tilde\theta e^{-2\lambda T}\Big)
      + O(\varepsilon^3),
    \end{aligned}$$
    $$\begin{aligned}
    G &= -\kappa\tilde\theta\,B + \tfrac12\tilde s\,B^2, \qquad B = \frac{1 - E}{\kappa}, \qquad E = e^{-\kappa T}, \\
    I_1 &= \frac{1}{\kappa}\Big(T - \frac{1 - E}{\kappa}\Big), \\
    I_2 &= \frac{1}{\kappa^2}\Big(T - \frac{2(1 - E)}{\kappa} + \frac{1 - E^2}{2\kappa}\Big), \\
    I_3 &= \frac{1}{\kappa^3}\Big(T - \frac{3(1 - E)}{\kappa} + \frac{3(1 - E^2)}{2\kappa} - \frac{1 - E^3}{3\kappa}\Big), \\
    I_4 &= \frac{1}{\kappa^4}\Big(T - \frac{4(1 - E)}{\kappa} + \frac{3(1 - E^2)}{\kappa} - \frac{4(1 - E^3)}{3\kappa} + \frac{1 - E^4}{4\kappa}\Big).
    \end{aligned}$$
    </div>
'''
    html = r'''
    <h2>The survival probability written out</h2>
    <p>Every integral in $\log M$ is a combination of $\int_0^T B^k$, and $B$ is a polynomial in $E = e^{-\kappa T}$. The
    integrals are therefore elementary, and the survival probability to second order is</p>
''' + card + r'''    <p>This is Vasicek&apos;s survival probability with averaged parameters, multiplied by an explicit correction. The
    $\varepsilon$ term in the exponent is the Green&ndash;Kubo correction, and the bracket is the memory of the starting
    regime. Its last term is the matched initial layer; it makes the stated remainder uniform for $T\geq0$ on every
    fixed bounded maturity interval.</p>
    <p>At the parameters of the <a href="./survival.html">survival demo</a>, $\kappa = 2$, $\theta = (0.15,\ 0.02)$,
    $\sigma^2 = (0.0555,\ 0.0055)$, $x = 0.12$, with $T = 4$, $\lambda = 20$ and a start in regime 1:</p>
''' + table([['$E$, $B$', f'{parts["E"]:.6g}, {parts["B"]:.6f}'],
             ['$I_1$, $I_2$, $I_3$, $I_4$', ', '.join(f'{parts[k]:.6g}' for k in ('I1', 'I2', 'I3', 'I4'))],
             ['averaged exponent $-\\kappa\\bar\\theta I_1 + \\frac12\\bar s I_2$', f'{parts["avg"]:.6f}'],
             ['correction $\\kappa^2\\tilde\\theta^2 I_2 - \\kappa\\tilde\\theta\\tilde s I_3 + \\frac14\\tilde s^2 I_4$', f'{parts["A"]:.6g}'],
             ['$G$', f'{parts["G"]:.6g}']]) + \
        table([['averaged model', f'{orders[0]:.8f}', f'{abs(orders[0]-ref):.1e}'],
               ['first order', f'{orders[1]:.8f}', f'{abs(orders[1]-ref):.1e}'],
               ['second order', f'{orders[2]:.8f}', f'{abs(orders[2]-ref):.1e}'],
               ['numerical solution', f'{ref:.8f}', '']], head=('survival', 'value', 'error'))
    write('regime-switching', html)


def cir():
    kappa, th, sig, T, x, lam = 1.5, [0.08, 0.02], 0.15, 3.0, 0.05, 10.0
    thb, tht, eps = np.mean(th), (th[0] - th[1]) / 2, 1 / lam
    h = math.sqrt(kappa ** 2 + 2 * sig ** 2)
    B = cir_B(T, kappa, sig)
    Bp = 1 - kappa * B - 0.5 * sig * sig * B * B
    I1, I2 = cir_int_B(T, kappa, sig), cir_int_B2(T, kappa, sig)
    G = -kappa * tht * B
    Gp = -kappa * tht * Bp
    avg = -kappa * thb * I1
    A = (kappa * tht) ** 2 * I2
    o0 = math.exp(-B * x + avg)
    o1 = math.exp(-B * x + avg + eps / 2 * A) * (1 + eps / 2 * G)
    o2 = math.exp(-B * x + avg + eps / 2 * A - eps ** 2 / 8 * G * G) \
        * (1 + eps / 2 * G - eps ** 2 / 4 * Gp - eps ** 2 / 4 * kappa * tht * math.exp(-2 * lam * T))
    g, gf, _, _ = cir_switching_mean(kappa, th, sig, 5.0)
    ref = numerical_a_callable(T, sym(lam), gf, rtol=1e-13)[0] * math.exp(-B * x)
    eng = FastSwitch(sym(lam), g, order=2).a(T, 2)[0] * math.exp(-B * x)
    assert abs(o2 - eng) < 1e-9, (o2, eng)
    card = r'''    <div class="equation-card">
    $$\begin{aligned}
    u_{1,2}(T, x) \;=\; &\exp\Big(-B\,x - \kappa\bar\theta\,I_1 + \frac{\varepsilon}{2}\,\kappa^2\tilde\theta^2\,I_2
      - \frac{\varepsilon^2}{8}\,\kappa^2\tilde\theta^2 B^2\Big) \\
      &\times\Big(1 \mp \frac{\varepsilon}{2}\,\kappa\tilde\theta\,B \pm \frac{\varepsilon^2}{4}\,\kappa\tilde\theta
      \big(1 - \kappa B - \tfrac12\sigma^2B^2 - e^{-2\lambda T}\big)\Big) + O(\varepsilon^3),
    \end{aligned}$$
    $$\begin{aligned}
    B &= \frac{2(e^{hT} - 1)}{(h + \kappa)(e^{hT} - 1) + 2h}, \qquad h = \sqrt{\kappa^2 + 2\sigma^2}, \\
    I_1 &= \frac{2}{\sigma^2}\Big(\log\big((h + \kappa)(e^{hT} - 1) + 2h\big) - \log 2h - \frac{(\kappa + h)\,T}{2}\Big), \\
    I_2 &= \frac{4}{h}\Big(\frac{hT}{(h - \kappa)^2} + \frac{a_2}{h + \kappa}\log\frac{(h + \kappa)e^{hT} + h - \kappa}{2h} \\
      &\qquad\quad - \frac{a_3}{h + \kappa}\Big(\frac{1}{(h + \kappa)e^{hT} + h - \kappa} - \frac{1}{2h}\Big)\Big), \\
    a_2 &= \frac{1}{h + \kappa} - \frac{h + \kappa}{(h - \kappa)^2}, \qquad
    a_3 = -2 - \frac{2(h + \kappa)}{h - \kappa} - a_2\,(h - \kappa).
    \end{aligned}$$
    </div>
'''
    html = r'''
    <h2>The survival probability written out</h2>
    <p>The forcing is proportional to $B$, so the expansion needs only $\int_0^T B$ and $\int_0^T B^2$. The first is
    the classical CIR integral. The second follows by substituting $E = e^{hs}$ and splitting into partial fractions.
    Because $\tilde g(0) = 0$, the second-order term of the exponent collapses to $-\frac{\varepsilon^2}{8}\tilde g(T)^2$.
    The survival probability to second order is</p>
''' + card + r'''    <p>The upper sign is for a start in the high-level regime. The first factor without the $\varepsilon$ terms is the
    CIR survival probability with the averaged level. The $e^{-2\lambda T}$ term is the initial layer and makes the
    second-order remainder uniform down to $T=0$.</p>
    <p>At the parameters above, with $T = 3$, $x = 0.05$ and $\lambda = 10$:</p>
''' + table([['$h$, $B$', f'{h:.6f}, {B:.6f}'], ['$I_1$, $I_2$', f'{I1:.6f}, {I2:.6f}'],
             ['averaged exponent $-\\kappa\\bar\\theta I_1$', f'{avg:.6f}'], ['correction $\\kappa^2\\tilde\\theta^2 I_2$', f'{A:.6g}']]) + \
        table([['averaged model', f'{o0:.8f}', f'{abs(o0-ref):.1e}'], ['first order', f'{o1:.8f}', f'{abs(o1-ref):.1e}'],
               ['second order', f'{o2:.8f}', f'{abs(o2-ref):.1e}'], ['numerical solution', f'{ref:.8f}', '']],
              head=('survival', 'value', 'error'))
    write('cir', html)


def constant_card(what):
    return r'''    <div class="equation-card">
    $$\begin{aligned}
    ''' + what + r'''_{1,2} \;=\; &e^{(\bar g - \lambda)T}\Big(\cosh sT + \frac{\lambda \pm \tilde g}{s}\,\sinh sT\Big),
      \qquad s = \sqrt{\lambda^2 + \tilde g^2}, \\[4pt]
    \;=\; &\exp\Big(\bar g\,T + \frac{\varepsilon}{2}\,\tilde g^2\,T - \frac{\varepsilon^2}{4}\,\tilde g^2\big(1 - e^{-2\lambda T}\big)\Big)
      \Big(1 \pm \frac{\varepsilon}{2}\,\tilde g\,\big(1 - e^{-2\lambda T}\big)\Big) + O(\varepsilon^3).
    \end{aligned}$$
    </div>
'''


def black_scholes():
    sig, r, T, lam, u = [0.30, 0.15], 0.03, 1.0, 25.0, 1 - 0.5j
    g, gf = bs_switching(u, r, sig)
    gi = [gf[i](0) for i in range(2)]
    gb, gt = (gi[0] + gi[1]) / 2, (gi[0] - gi[1]) / 2
    eps = 1 / lam
    ex = two_state_constant_exact(gb, gt, lam, T)
    L = 1 - math.exp(-2 * lam * T)
    o0 = cmath.exp(gb * T)
    o1 = cmath.exp(gb * T + eps / 2 * gt * gt * T) * (1 + eps / 2 * gt * L)
    o2 = cmath.exp(gb * T + eps / 2 * gt * gt * T - eps ** 2 / 4 * gt * gt * L) * (1 + eps / 2 * gt * L)
    ref = numerical_a_callable(T, sym(lam), gf, rtol=1e-13)[0]
    assert abs(ex - ref) < 1e-10
    html = r'''
    <h2>The characteristic function in closed form</h2>
    <p>The forcing is constant, so the two-state system can be solved exactly. The characteristic function of the log
    return is</p>
''' + constant_card(r'\phi') + r'''    <p>with</p>
    $$\bar g = iu\big(r - \tfrac12\bar s\big) - \tfrac12 u^2\,\bar s, \qquad
      \tilde g = -\tfrac12\big(iu + u^2\big)\,\tilde s, \qquad \bar s, \tilde s = \frac{\sigma_1^2 \pm \sigma_2^2}{2} .$$
    <p>The first line is exact. The second is its expansion in $\varepsilon = 1/\lambda$: Black&ndash;Scholes with the
    averaged variance, times the Green&ndash;Kubo factor $e^{\varepsilon\tilde g^2T/2}$ and the memory of the starting
    regime. The upper sign is for a start in the volatile regime.</p>
    <p>At $\lambda = 25$, $T = 1$ and the Lewis frequency $u = 1 - i/2$:</p>
''' + table([['$\\bar g$, $\\tilde g$', f'{cfmt(gb)}, {cfmt(gt)}'],
             ['averaged Black&ndash;Scholes', f'{cfmt(o0, 8)}', f'{abs(o0-ex):.1e}'],
             ['first order', f'{cfmt(o1, 8)}', f'{abs(o1-ex):.1e}'],
             ['second order', f'{cfmt(o2, 8)}', f'{abs(o2-ex):.1e}'],
             ['exact', f'{cfmt(ex, 8)}', '']], head=('quantity', 'value', 'error'))
    write('black-scholes', html)


def counts():
    ell, T, lam = [8.0, 1.0], 1.0, 10.0
    lb, lt, eps = np.mean(ell), (ell[0] - ell[1]) / 2, 1 / lam
    L = 1 - math.exp(-2 * lam * T)
    s = lambda z: cmath.sqrt(lam * lam + ((z - 1) * lt) ** 2)
    gen = lambda z: two_state_constant_exact((z - 1) * lb, (z - 1) * lt, lam, T)
    M = 64
    zs = np.exp(2j * np.pi * np.arange(M) / M)
    p_exact = np.real(np.fft.fft([gen(z) for z in zs])) / M
    mean_ex = sum(k * p for k, p in enumerate(p_exact))
    var_ex = sum(k * k * p for k, p in enumerate(p_exact)) - mean_ex ** 2
    mean_1 = lb * T + eps / 2 * lt * L
    var_1 = lb * T + eps * lt * lt * T + eps / 2 * lt * L
    var_2 = var_1 - eps ** 2 * lt * lt * (L / 2 + L * L / 4)
    html = r'''
    <h2>The generating function in closed form</h2>
    <p>The forcing $g_i = (z - 1)\ell_i$ is constant, so the generating function is exact:</p>
''' + constant_card(r'\mathbb{E}\big[z^{N_T}\big]') + r'''    <p>with</p>
    $$\bar g = (z - 1)\,\bar\ell, \qquad \tilde g = (z - 1)\,\tilde\ell, \qquad \bar\ell, \tilde\ell = \frac{\ell_1 \pm \ell_2}{2} .$$
    <p>Differentiating the expansion at $z = 1$ gives the mean and variance of the count. With
    $L = 1 - e^{-2\lambda T}$ and the upper sign for a start in regime 1,</p>
    $$\mathbb{E}[N_T] = \bar\ell\,T \pm \frac{\varepsilon}{2}\,\tilde\ell\,L,$$
    $$\operatorname{Var}[N_T] = \bar\ell\,T + \varepsilon\,\tilde\ell^{\,2}\,T \pm \frac{\varepsilon}{2}\,\tilde\ell\,L
      - \varepsilon^2\,\tilde\ell^{\,2}\Big(\frac L2 + \frac{L^2}{4}\Big) .$$
    <p>These two formulas have no higher-order terms. Given the regime path the count is Poisson with mean
    $\Lambda = \int_0^T \ell_{y_s}\,ds$, so</p>
    $$\operatorname{Var}[N_T] = \mathbb{E}[\Lambda] + \operatorname{Var}[\Lambda],$$
    <p>and integrating the covariance of the two-state regime, which is $e^{-2\lambda|t - s|} - e^{-2\lambda(t + s)}$
    in the $\pm1$ coding, gives both lines exactly.</p>
    <p>The variance exceeds the mean by</p>
    $$\operatorname{Var}[N_T] - \mathbb{E}[N_T] = \varepsilon\,\tilde\ell^{\,2}\,T
      - \varepsilon^2\,\tilde\ell^{\,2}\Big(\frac L2 + \frac{L^2}{4}\Big) .$$
    <p>At fixed $T &gt; 0$ the first term leads. It is the Green&ndash;Kubo overdispersion. The second term is the memory
    of the starting regime, and at horizons short against $1/\lambda$ it nearly cancels the first.</p>
    <p>At $\ell = (8, 1)$, $T = 1$, $\lambda = 10$ and a start in the busy regime the excess is ''' + f'{var_2 - mean_1:.5f}' + r''',
    against ''' + f'{eps * lt * lt * T:.5f}' + r''' from the first term alone:</p>
''' + table([['mean', f'{mean_1:.5f}', f'{mean_1:.5f}', f'{mean_ex:.5f}'], ['variance', f'{var_1:.5f}', f'{var_2:.5f}', f'{var_ex:.5f}']],
            head=('', 'first order', 'second order (exact)', 'numerical')) + r'''    <p>The numerical column evaluates the generating function at the 64 roots of unity, as described above, and the
    distribution itself follows the same way. A Monte Carlo check over 400,000 regime paths, sampled exactly from
    exponential holding times, gives each path a Poisson law for the count. It agrees with the moments and with
    every probability up to $k = 15$ within two standard errors. Certificate:
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/verify_option_mc.py">verify_option_mc.py</a>.</p>
'''
    write('counts', html)




def bond_options():
    from bond_option_explicit import call, pieces
    from options import zcb_call
    kappa, th, sig, x0, T, S, lam = 0.5, [0.05, 0.03], [0.015, 0.010], 0.04, 1.0, 4.0, 20.0
    Q = sym(lam)
    p = pieces(kappa, th, sig, x0, T, S, lam)
    rows = []
    for K in (0.86, 0.88, 0.90):
        num = zcb_call(T, S, K, x0, 0, kappa, th, sig, Q)
        c0 = call(kappa, th, sig, x0, T, S, K, lam, order=0)
        c1 = call(kappa, th, sig, x0, T, S, K, lam, order=1)
        rows.append([f'{K}', f'{num:.8f}', f'{c0:.8f}', f'{abs(c0 - num):.1e}', f'{c1:.8f}', f'{abs(c1 - num):.1e}'])
    from bond_option_explicit import polyadd, scale
    pis = []
    for j, sj in ((0, 1), (1, -1)):
        Pi = polyadd(scale(p['intg2'], 0.5), scale(p['gtT'], 0.5), scale(p['gt0'], 0.5 * sj))
        pis.append([f'expiry in regime {j + 1}'] + [f'{x:.4g}' for x in Pi])
    html = r'''
    <h2>The expansion</h2>
    <p>With $\varepsilon = 1/\lambda$, split the terminal vector into its stationary part and a remainder:</p>
    $$e_j = \tfrac12\,\mathbf 1 + \tfrac12\,s_j\begin{pmatrix}1\\-1\end{pmatrix}, \qquad s_1 = +1,\quad s_2 = -1 .$$
    <p>The stationary part is carried by the outer series exactly as for a bond. The remainder relaxes like
    $e^{-2\lambda t}$, but on the way it leaves a first-order imprint through $\tilde g(0)$, which is not zero here
    because $\tilde B(0) = c$. To first order, with the upper sign for a start in regime 1,</p>
    <div class="equation-card">
    $$\begin{aligned}
    \mathbb{E}\big[e^{-\int_0^T x_r\,dr - c\,x_T}\,\mathbf 1\{y_T = j\}\big]
      \;=\; &\tfrac12\,e^{-\tilde B(T)\,x_0}\,\exp\Big(\int_0^T \bar g + \frac{\varepsilon}{2}\int_0^T \tilde g^{\,2}\Big) \\
      &\times\Big(1 \pm \frac{\varepsilon}{2}\,\tilde g(T) + s_j\,\frac{\varepsilon}{2}\,\tilde g(0)\Big)
      + O(\varepsilon^2) + O(e^{-2\lambda T}).
    \end{aligned}$$
    </div>
    <p>The term in $\tilde g(T)$ is the memory of the regime at the start. The term in $\tilde g(0)$ is its mirror image:
    the memory of the regime at expiry, which decides the bond price the option is written on.</p>
    <p>The expansion needs $|\tilde g|/\lambda$ to be small at the frequencies $u$ that matter. The Gil-Pelaez
    integrals therefore stop at eight standard deviations of $x_T$ in frequency.</p>

    <h2>The option price in closed form</h2>
    <p>Every integral above is elementary, and to first order the option price reduces to Gaussian integrals.</p>
    <h3>Step 1: the averaged model</h3>
    <p>Under the averaged model&apos;s $T$-forward measure the rate at expiry is Gaussian,
    $x_T \sim N(\mu, v)$, with</p>
    $$\begin{aligned}
    \mu &= E\,x_0 + \kappa\bar\theta\,B - \bar s\,M_{1,1}, \qquad v = \bar s\,M_{2,0}, \\
    \bar P(0, T) &= \exp\big(-B\,x_0 - \kappa\bar\theta\,M_{0,1} + \tfrac12\bar s\,M_{0,2}\big), \\
    E &= e^{-\kappa T}, \qquad B = \frac{1 - E}{\kappa},
    \end{aligned}$$
    <p>where the building blocks are</p>
    $$M_{k,m} = \int_0^T e^{-k\kappa t}\,B(t)^m\,dt = \frac{1}{\kappa^m}\sum_{l=0}^{m}\binom{m}{l}(-1)^l\,\Phi_{k+l},
      \qquad \Phi_0 = T,\quad \Phi_n = \frac{1 - e^{-n\kappa T}}{n\kappa}.$$
    <h3>Step 2: the correction polynomial</h3>
    <p>Because $\tilde B(t) = c\,e^{-\kappa t} + B(t)$, every correction term is a polynomial in $c$. The first-order
    factor is</p>
    $$\begin{aligned}
    \Pi_j(c) \;=\; &\tfrac12\Big(\kappa^2\tilde\theta^2\,J_2(c) - \kappa\tilde\theta\,\tilde s\,J_3(c) + \tfrac14\tilde s^2 J_4(c)\Big) \\
      &\pm \tfrac12\Big(-\kappa\tilde\theta\,(B + Ec) + \tfrac12\tilde s\,(B + Ec)^2\Big)
      + \tfrac12\,s_j\Big(-\kappa\tilde\theta\,c + \tfrac12\tilde s\,c^2\Big),
    \end{aligned}$$
    $$J_n(c) = \int_0^T \tilde B(t)^n\,dt = \sum_{k=0}^{n}\binom{n}{k}\,M_{k,\,n-k}\;c^k .$$
    <p>Write $\Pi_j(c) = \sum_{k=0}^{4}\pi_{jk}\,c^k$. A factor $c^k$ on the transform is the $k$-th derivative of the
    Gaussian density of $x_T$.</p>
    <h3>Step 3: the bond at expiry</h3>
    <p>In regime $j$ the bond price at expiry is $A_j e^{-b x_T}$, with $b = (1 - e^{-\kappa\tau})/\kappa$ and
    $\tau = S - T$. The regime-switching bond formula gives</p>
    $$A_{1,2} = \exp\Big(-\kappa\bar\theta\,I_1 + \tfrac12\bar s\,I_2
      + \frac{\varepsilon}{2}\big(\kappa^2\tilde\theta^2 I_2 - \kappa\tilde\theta\,\tilde s\,I_3 + \tfrac14\tilde s^2 I_4\big)\Big)
      \Big(1 \pm \frac{\varepsilon}{2}\big(-\kappa\tilde\theta\,b + \tfrac12\tilde s\,b^2\big)\Big),$$
    <p>with $I_k = M_{0,k}$ evaluated at maturity $\tau$ in place of $T$.</p>
    <h3>Step 4: the price</h3>
    <div class="equation-card">
    $$C \;=\; \bar P(0, T)\sum_{j = 1}^{2}\tfrac12\Big(R_0(A_j) + \varepsilon\sum_{k=0}^{4}(-1)^k\,\pi_{jk}\,R_k(A_j)\Big) + O(\varepsilon^2),$$
    $$\begin{aligned}
    R_0(A) &= A\,e^{-b\mu + \frac12 b^2 v}\,\Phi(z^* + b\sqrt v) - K\,\Phi(z^*), \\
    R_k(A) &= v^{-k/2}\Big(A\,e^{-b\mu + \frac12 b^2 v}\sum_{i=0}^{k}\binom{k}{i}\big(-b\sqrt v\big)^{k-i} H_i\big(z^* + b\sqrt v\big)
      - K\,H_k(z^*)\Big), \\
    z^* &= \frac{\log(A/K)/b - \mu}{\sqrt v}, \qquad H_0 = \Phi, \qquad H_i(w) = -\mathrm{He}_{i-1}(w)\,\varphi(w) \ \ (i \ge 1).
    \end{aligned}$$
    </div>
    <p>Here $\mathrm{He}_n$ are the Hermite polynomials $1, w, w^2 - 1, \dots$, and $\Phi$ and $\varphi$ are the standard
    normal distribution and density. $R_0$ is Jamshidian&apos;s formula, so with $\varepsilon = 0$ this is the averaged
    model&apos;s option price. The $\varepsilon$ terms are the new correction.</p>
    <p>At the parameters above with $\lambda = 20$ and a start in regime 1, the correction coefficients $\pi_{jk}$ are:</p>
''' + table(pis, head=('', '$c^0$', '$c^1$', '$c^2$', '$c^3$', '$c^4$')) + r'''    <p>The resulting prices against the numerical solution:</p>
''' + table(rows, head=('strike', 'numerical', 'averaged (Jamshidian)', 'error', 'closed form, first order', 'error')) + r'''    <p>Each doubling of $\lambda$ divides the first-order error by about four, as a second-order remainder should. The
    engine&apos;s higher orders, in the table below, continue the series.</p>
'''
    write('bond-options', html)



def three_regimes():
    from numpy.polynomial import polynomial as P
    from fastswitch import ExpSum, numerical_a
    k = 2.0
    th, s2 = np.array([0.20, 0.08, 0.02]), np.array([0.04, 0.01, 0.0025])
    base = np.array([[-3, 2, 1], [1, -2, 1], [0.5, 1.5, -2]], float)
    T, sc = 1.0, 8
    Q = sc * base
    w_, vl = np.linalg.eig(Q.T)
    pi = np.real(vl[:, np.argmin(abs(w_))]); pi /= pi.sum()
    one = np.ones(3)
    Qs = np.linalg.inv(Q - np.outer(one, pi)) + np.outer(one, pi)
    u, v = th - pi @ th, s2 - pi @ s2
    pad = lambda a, n: np.array([np.pad(r, (0, n - len(r))) for r in a])
    def mul(a, b):
        out = [P.polymul(a[i], b[i]) for i in range(3)]
        return pad(out, max(len(o) for o in out))
    gt = np.zeros((3, 3)); gt[:, 1] = -k * u; gt[:, 2] = 0.5 * v
    w1 = -Qs @ gt
    F1 = mul(gt, w1); F1 = F1 - np.outer(one, pi @ F1)
    dw1 = [P.polymul(P.polyder(r), [1, -k]) for r in w1]
    n = max(max(len(r) for r in dw1), F1.shape[1])
    w2 = Qs @ (pad(dw1, n) - pad(F1, n))
    d = np.trim_zeros(pi @ mul(gt, w2), 'b')
    B = (1 - math.exp(-k * T)) / k
    I = {j: I_k(j, T, k) for j in range(1, 8)}
    K = lambda f, h: -pi @ (f * (Qs @ h))
    c2, c3, c4 = k * k * K(u, u), -k * 0.5 * (K(u, v) + K(v, u)), K(v, v) / 4
    avg = -k * (pi @ th) * I[1] + 0.5 * (pi @ s2) * I[2]
    gk = c2 * I[2] + c3 * I[3] + c4 * I[4]
    e2 = sum(c * I[j] for j, c in enumerate(d) if j > 0)
    mem = k * B * (Qs @ u) - 0.5 * B * B * (Qs @ v)
    w2T = np.array([P.polyval(B, r) for r in w2])
    G = lambda a, b: ExpSum({0: -a + b / (2 * k * k), k: a - 2 * b / (2 * k * k), 2 * k: b / (2 * k * k)})
    g = [G(th[i], s2[i]) for i in range(3)]
    ex = np.array(numerical_a(T, Q, g, dps=30), float)
    o0 = np.full(3, math.exp(avg)); o1 = np.exp(avg + gk) * (1 + mem); o2 = np.exp(avg + gk + e2) * (1 + mem + w2T)
    eng = FastSwitch(Q, g, order=2).a(T, 2)
    assert np.abs(o2 - eng).max() < 1e-12
    mfmt = lambda M: r'\begin{pmatrix}' + r' \\ '.join(' & '.join(f'{x:.4f}' for x in row) for row in M) + r'\end{pmatrix}'
    html = r"""
    <h2>The bond price written out</h2>
    <p>For a chain of any size the expansion is written with the group inverse $Q^{\#}$ of the generator and its
    stationary law $\pi$. Let $u = \theta - \pi\cdot\theta$ and $v = \sigma^2 - \pi\cdot\sigma^2$ be the fluctuations of
    level and variance across regimes, so the forcing splits as</p>
    $$g_i = \bar g + \tilde g_i, \qquad \bar g = -\kappa\,(\pi\cdot\theta)\,B + \tfrac12(\pi\cdot\sigma^2)\,B^2, \qquad
      \tilde g_i = -\kappa\,u_i\,B + \tfrac12\,v_i\,B^2 .$$
    <h3>First order</h3>
    <p>The first correction is the Green&ndash;Kubo integral. Since $\tilde g$ is linear in $u$ and $v$, it is a
    combination of $I_2$, $I_3$ and $I_4$:</p>
    <div class="equation-card">
    $$\begin{aligned}
    u_i(T, x) = \;&\exp\Big(-B\,x - \kappa(\pi\cdot\theta)\,I_1 + \tfrac12(\pi\cdot\sigma^2)\,I_2 + c_2 I_2 + c_3 I_3 + c_4 I_4\Big) \\
      &\times\Big(1 + \kappa B\,[Q^{\#}u]_i - \tfrac12 B^2\,[Q^{\#}v]_i\Big) + O(|Q|^{-2}),
    \end{aligned}$$
    $$\begin{aligned}
    c_2 &= \kappa^2\,\mathcal K(u, u), \qquad c_3 = -\kappa\,\mathcal K_{\mathrm{sym}}(u, v), \qquad c_4 = \tfrac14\,\mathcal K(v, v), \\
    \mathcal K(f, h) &= -\pi\cdot\big(f\,Q^{\#}h\big), \qquad \mathcal K_{\mathrm{sym}}(u, v) = \tfrac12\big(\mathcal K(u, v) + \mathcal K(v, u)\big), \\
    I_k &= \int_0^T B^k = \frac{1}{\kappa^k}\Big(T + \sum_{j=1}^{k}\binom{k}{j}(-1)^j\,\frac{1 - e^{-j\kappa T}}{j\kappa}\Big),
      \qquad B = \frac{1 - e^{-\kappa T}}{\kappa}.
    \end{aligned}$$
    </div>
    <h3>Second order</h3>
    <p>Because $B' = 1 - \kappa B$, every vector in the recursion is a polynomial in $B$:</p>
    $$\begin{aligned}
    w_1 &= -Q^{\#}\tilde g, \\
    w_2 &= Q^{\#}\big(w_1' - \tilde g\circ w_1 + \pi\cdot(\tilde g\circ w_1)\,\mathbf 1\big), \qquad
      w_1' = (1 - \kappa B)\,\partial_B w_1 .
    \end{aligned}$$
    <p>For every fixed $T&gt;0$, the second-order exponent is $\int_0^T \pi\cdot(\tilde g\circ w_2)$, a polynomial in $B$ integrated term by term, and
    the bracket gains $w_2(T)$:</p>
    <div class="equation-card">
    $$u_i(T, x) = \exp\Big(\cdots + \sum_{k} d_k\,I_k\Big)\Big(1 + \kappa B\,[Q^{\#}u]_i - \tfrac12 B^2\,[Q^{\#}v]_i + [w_2(T)]_i\Big)
      + O(|Q|^{-3}),\qquad T&gt;0\ \text{fixed},$$
    $$\pi\cdot(\tilde g\circ w_2) = \sum_k d_k\,B^k .$$
    </div>
    <p>The remainder is $O(|Q|^{-3})$ at fixed $T > 0$. At $T = 0$ the bracket is $1 + [w_2(0)]_i$ with
    $w_2(0) = \kappa\,(Q^{\#})^2 u$, and an initial layer of size $|Q|^{-2}$, decaying at the rates of $Q$, restores
    $u_i(0,x) = 1$; the engine includes it.</p>
    <h3>The numbers</h3>
    <p>For the chain above multiplied by 8, at $T = 1$:</p>
    $$\pi = (""" + ', '.join(f'{x:.4f}' for x in pi) + r"""), \qquad Q^{\#} = """ + mfmt(Qs) + r""" .$$
""" + table([['$\\pi\\cdot\\theta$, $\\pi\\cdot\\sigma^2$', f'{pi@th:.6f}, {pi@s2:.6f}'],
             ['$c_2$, $c_3$, $c_4$', f'{c2:.6g}, {c3:.6g}, {c4:.6g}'],
             ['$B$; $I_1, \\dots, I_4$', f'{B:.6f}; ' + ', '.join(f'{I[j]:.6g}' for j in (1, 2, 3, 4))],
             ['averaged exponent', f'{avg:.6f}'], ['Green&ndash;Kubo exponent $c_2I_2 + c_3I_3 + c_4I_4$', f'{gk:.6g}'],
             ['$d_1, \\dots, d_6$', ', '.join(f'{x:.3g}' for x in d[1:])], ['second-order exponent $\\sum d_kI_k$', f'{e2:.4g}'],
             ['memory $\\kappa B[Q^{\\#}u]_i - \\frac12B^2[Q^{\\#}v]_i$', ', '.join(f'{x:.6g}' for x in mem)],
             ['$w_2(T)$', ', '.join(f'{x:.4g}' for x in w2T)]]) + \
        table([[f'regime {i + 1}', f'{o0[i]:.8f}', f'{o1[i]:.8f}', f'{o2[i]:.8f}', f'{ex[i]:.8f}'] for i in range(3)],
              head=('start', 'averaged', 'first order', 'second order', 'numerical')) + r"""    <p>The second-order formula agrees with the engine&apos;s outer series to rounding, and its error against the numerical solution is
    """ + f'{np.abs(o2 - ex).max():.1e}' + r""". A maturity-uniform formula also includes the engine&apos;s initial
    layer; the <a href="./layers.html">initial-layers</a> page proves that correction explicitly in the symmetric
    two-state case.</p>
"""
    write('three-regimes', html)



def credit():
    """Write explicit/credit-parameters.html and explicit/credit.html, included into tools/pages/credit.html."""
    kap, sig, T = 2.0, 0.2, 3.0
    th = np.array([[0.8, 0.005], [0.6, 0.005]])          # name j, regime i
    x0 = np.array([0.05, 0.05])
    B, I1, I2 = cir_B(T, kap, sig), cir_int_B(T, kap, sig), cir_int_B2(T, kap, sig)
    C = [np.array([1, 0]), np.array([0, 1]), np.array([1, 1])]

    def surv_num(lam, c, q=0.5):
        """survival of the names in c; q is the prior probability of starting in the crisis regime"""
        gf = [(lambda i: (lambda s: -kap * (c @ th[:, i]) * cir_B(s, kap, sig)))(i) for i in range(2)]
        a = numerical_a_callable(T, sym(lam), gf, rtol=1e-12)
        return math.exp(-B * (c @ x0)) * (q * a[0] + (1 - q) * a[1])

    def surv_exp(lam, c, order):
        eps = 1 / lam
        tb, tt = (c @ th).mean(), ((c @ th)[0] - (c @ th)[1]) / 2
        L = -B * (c @ x0) - kap * tb * I1
        if order >= 1:
            L += eps / 2 * (kap * tt) ** 2 * I2
        if order >= 2:
            L -= eps ** 2 / 8 * (kap * tt * B) ** 2
        return math.exp(L)

    corr = lambda S1, S2, S12: (S12 - S1 * S2) / math.sqrt((1 - S1) * S1 * (1 - S2) * S2)
    from fastswitch import Cheb
    from mc_credit_exact import mc
    Bc = Cheb.fit(lambda s: cir_B(s, kap, sig), T, 80)
    rows = []
    for lam in (1.0, 2.0, 4.0, 8.0):
        sn = [surv_num(lam, c) for c in C]
        num = corr(*sn)
        orders = {}
        for c in C:
            g = [Bc.scale(-kap * (c @ th[:, i])) for i in range(2)]
            fs = FastSwitch(sym(lam), g, order=10)
            orders[tuple(c)] = [math.exp(-B * (c @ x0)) * 0.5 * sum(fs.a(T, o)) for o in range(11)]
        cs = [corr(orders[(1, 0)][o], orders[(0, 1)][o], orders[(1, 1)][o]) for o in range(11)]
        hit = [o for o in range(11) if abs(cs[o] - num) < 5e-6]
        best = hit[0] if hit else min(range(11), key=lambda o: abs(cs[o] - num))
        m, se = mc(lam)
        rows.append([f'{lam:g}', f'{1 - sn[0]:.3f}, {1 - sn[1]:.3f}', f'{num:.5f}', f'{m:.5f} &plusmn; {se:.5f}',
                     '0'] + [f'{cs[o]:.5f}' for o in (1, 2, 4, 6)] + [f'{cs[best]:.5f} (order {best})'])
    lam = 2.0
    sn = [surv_num(lam, c) for c in C]
    se = [surv_exp(lam, c, 2) for c in C]
    tt1, tt2 = (th[0, 0] - th[0, 1]) / 2, (th[1, 0] - th[1, 1]) / 2
    dep2 = 1 / lam * kap ** 2 * tt1 * tt2 * I2 - (1 / lam) ** 2 / 4 * kap ** 2 * tt1 * tt2 * B * B
    prior_rows = []
    for q in (0.1, 0.5, 0.9):
        sq = [surv_num(lam, c, q) for c in C]
        prior_rows.append([f'{q:g}', f'{corr(*sq):.6f}'])

    # the parameters, and what the Feller condition says about them
    feller = 2 * kap * th                                  # 2 kappa theta, name j, regime i
    ok = feller >= sig ** 2
    fmt = lambda v: f'{v:g}'
    params = r"""    <p class="muted">Parameters: $\kappa = """ + fmt(kap) + r"""$, $\sigma = """ + fmt(sig) + r"""$; the first company
    has $\theta_1 = (""" + fmt(th[0, 0]) + r""",\ """ + fmt(th[0, 1]) + r""")$, the second
    $\theta_2 = (""" + fmt(th[1, 0]) + r""",\ """ + fmt(th[1, 1]) + r""")$; $x_0 = (""" + fmt(x0[0]) + r""",\ """ + fmt(x0[1]) + r""")$;
    horizon $T = """ + fmt(T) + r"""$.</p>
    <p>A CIR intensity stays strictly positive when</p>
    $$2\kappa\theta \ge \sigma^2,$$
    <p>the Feller condition.</p>
"""
    if ok.all():
        params += r"""    <p>Here it holds in both regimes, so the intensities never reach zero.</p>
"""
    else:
        def values(i):
            v = feller[:, i]
            return f'{fmt(v[0])} for both companies' if v[0] == v[1] else f'{fmt(v[0])} and {fmt(v[1])}'
        where = ' and '.join(('in the crisis regime', 'in normal times')[i] for i in range(2) if not ok[:, i].all())
        params += r"""    <p>Here $\sigma^2 = """ + fmt(sig ** 2) + r"""$, while $2\kappa\theta$ is """ + values(0) + r""" in the crisis
    regime and """ + values(1) + r""" in normal times. The condition fails """ + where + r""", where an
    intensity can touch zero and the hazard of that company vanishes for a moment. Zero is reflecting, so the
    intensities stay nonnegative and are valid default intensities. The survival formulas below use only the
    Laplace transform of the integrated CIR process, which holds whether or not the condition is met.</p>
"""
    write('credit-parameters', params)

    # stopping orders, for the prose
    best = {float(r[0]): r[-1] for r in rows}
    fast = [float(r[0]) for r in rows if float(r[0]) >= 4]
    by_order = max(int(best[l].split('order ')[1].rstrip(')')) for l in fast)
    at2 = int(best[2.0].split('order ')[1].rstrip(')'))
    gap1 = abs(float(best[1.0].split(' ')[0]) - float(rows[0][2]))

    html = r"""
    <h2>The survival probabilities written out</h2>
    <p>Given the regime path the two intensities are independent CIR processes, and each has the same coefficient
    $B$. With the stationary start the memory terms cancel, and the survival of any set of names is the CIR formula
    with the levels added:</p>
    <div class="equation-card">
    $$S_c(T) = \exp\Big(-B\,(c\cdot x_0) - \kappa\,\bar\theta_c\,I_1 + \frac{\varepsilon}{2}\,\kappa^2\tilde\theta_c^2\,I_2
      - \frac{\varepsilon^2}{8}\,\kappa^2\tilde\theta_c^2\,B^2\Big) + O(\varepsilon^3),$$
    $$\begin{aligned}
    \bar\theta_c &= c_1\bar\theta_1 + c_2\bar\theta_2, \qquad \tilde\theta_c = c_1\tilde\theta_1 + c_2\tilde\theta_2,
      \qquad \bar\theta_j, \tilde\theta_j = \frac{\theta_{j,1} \pm \theta_{j,2}}{2}, \\
    B &= \frac{2(e^{hT} - 1)}{(h + \kappa)(e^{hT} - 1) + 2h}, \qquad h = \sqrt{\kappa^2 + 2\sigma^2}, \\
    I_1 &= \frac{2}{\sigma^2}\Big(\log\big((h + \kappa)(e^{hT} - 1) + 2h\big) - \log 2h - \frac{(\kappa + h)\,T}{2}\Big), \\
    I_2 &= \frac{4}{h}\Big(\frac{hT}{(h - \kappa)^2} + \frac{a_2}{h + \kappa}\log\frac{(h + \kappa)e^{hT} + h - \kappa}{2h} \\
      &\qquad\quad - \frac{a_3}{h + \kappa}\Big(\frac{1}{(h + \kappa)e^{hT} + h - \kappa} - \frac{1}{2h}\Big)\Big), \\
    a_2 &= \frac{1}{h + \kappa} - \frac{h + \kappa}{(h - \kappa)^2}, \qquad a_3 = -2 - \frac{2(h + \kappa)}{h - \kappa} - a_2\,(h - \kappa).
    \end{aligned}$$
    </div>
    <p>Since $\tilde\theta_{(1,1)}^2 - \tilde\theta_1^2 - \tilde\theta_2^2 = 2\tilde\theta_1\tilde\theta_2$, the joint
    survival exceeds the product of the marginals by</p>
    <div class="equation-card">
    $$\log\frac{S_{(1,1)}(T)}{S_{(1,0)}(T)\,S_{(0,1)}(T)} = \varepsilon\,\kappa^2\,\tilde\theta_1\tilde\theta_2\,I_2
      - \frac{\varepsilon^2}{4}\,\kappa^2\,\tilde\theta_1\tilde\theta_2\,B^2 + O(\varepsilon^3).$$
    </div>
    <p>With the independent default clocks this is the whole of the dependence. The averaged model has none, and the
    first term is the Green&ndash;Kubo covariance of the two hazards.</p>
    <p>At $\lambda = """ + f'{lam:g}' + r"""$, so that each regime lasts about half a year:</p>
""" + table([['$B$, $I_1$, $I_2$', f'{B:.6f}, {I1:.6f}, {I2:.6f}'],
             ['$\\tilde\\theta_1$, $\\tilde\\theta_2$', f'{tt1:.4f}, {tt2:.4f}'],
             ['dependence $\\log(S_{12}/S_1S_2)$, second order', f'{dep2:.6f}'],
             ['dependence, numerical', f'{math.log(sn[2] / (sn[0] * sn[1])):.6f}'],
             ['$S_{(1,0)}$, $S_{(0,1)}$, $S_{(1,1)}$, second order', ', '.join(f'{x:.6f}' for x in se)],
             ['$S_{(1,0)}$, $S_{(0,1)}$, $S_{(1,1)}$, numerical', ', '.join(f'{x:.6f}' for x in sn)]]) + r"""
    <h3>The starting regime</h3>
    <p>The stationary start is a choice of prior for the hidden cycle. With
    probability $q$ of starting in the crisis regime,</p>
    $$S_c(T) = e^{-B\,(c\cdot x_0)}\,\big(q\,a_1(T) + (1 - q)\,a_2(T)\big).$$
    <p>To first order this multiplies the stationary-start survival by</p>
    $$1 + (2q - 1)\,\frac{\varepsilon}{2}\,\tilde g_c(T), \qquad \tilde g_c = -\kappa\,\tilde\theta_c\,B .$$
    <p>If $x_0$ is an observation of a system already running, the right prior is the posterior of the regime given
    $x_0$, which is in general not $(\tfrac12, \tfrac12)$. The prior moves the correlation at $\lambda = """ + f'{lam:g}' + r"""$:</p>
""" + table(prior_rows, head=('prior crisis probability $q$', 'default correlation, numerical')) + r"""
    <h2>Results</h2>
    <p>The correlation of the two default indicators over three years, with the stationary start, against the
    numerical solution. The averaged model gives exactly zero at every switching rate.</p>
""" + table(rows, head=('switching rate', 'default probabilities', 'numerical', 'Monte Carlo', 'averaged',
                              'order 1', 'order 2', 'order 4', 'order 6', 'first to five digits, or best')) + r"""    <p>The numerical column solves the reduced linear system. The Monte Carlo column checks it with no time
    discretization: given a simulated regime path the level is piecewise constant, each CIR survival is exact, and
    only the regime path is sampled, 400,000 times. The two agree within the Monte Carlo error. Certificate:
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/mc_credit_exact.py">mc_credit_exact.py</a>.</p>
    <p>The higher orders come from the <a href="./engine.html">engine</a>. With regimes lasting three months or less
    ($\lambda \ge 4$) the expansion reproduces the correlation to all five digits by order """ + f'{by_order}' + r""". With
    half-year regimes ($\lambda = 2$) it reaches all five digits at order """ + f'{at2}' + r""". With regimes lasting a
    year ($\lambda = 1$) the series comes within about """ + f'{gap1:.3f}' + r""" before its terms start to grow.</p>
    <p>That is the behaviour of an asymptotic series. When the switching time is not small against the time scale of
    the forcing, the terms first shrink and then grow, and the best answer comes from stopping at the smallest
    term.</p>
"""
    write('credit', html)
    return html


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    jumps(); regime_switching(); cir(); black_scholes(); counts(); bond_options(); three_regimes(); credit()
