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
      &\times\Big(1 \pm \frac{\varepsilon}{2}\,G \mp \frac{\varepsilon^2}{4}\,G'\Big) + O(\varepsilon^3),
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
    avg = -kappa * thb * I[1] + 0.5 * sb * I[2] + lb * (J1 - T)
    logu = -B * x + avg + eps / 2 * A - eps ** 2 / 8 * G * G
    u = math.exp(logu) * (1 + sign * (eps / 2 * G - eps ** 2 / 4 * Gp))
    parts = dict(E=E, B=B, I1=I[1], I2=I[2], I3=I[3], I4=I[4], J1=J1, J2=J2, A=A, G=G, Gp=Gp, avg=avg)
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
    in $A$ are the Green&ndash;Kubo correction, and the bracket is the memory of the starting regime.</p>
    <p>At the parameters above, with $T = 3$, $x = 0$ and $\lambda = 10$:</p>
''' + table([['$E$, $B$', f'{parts["E"]:.6f}, {parts["B"]:.6f}'],
             ['$I_1$, $I_2$, $I_3$, $I_4$', ', '.join(f'{parts[k]:.6g}' for k in ('I1', 'I2', 'I3', 'I4'))],
             ['$J_1$, $J_2$', f'{parts["J1"]:.6f}, {parts["J2"]:.6f}'],
             ['averaged exponent', f'{parts["avg"]:.6f}'],
             ['$A$', f'{parts["A"]:.6g}'],
             ["$G$, $G'$", f'{parts["G"]:.6g}, {parts["Gp"]:.6g}']]) + \
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
      &\times\Big(1 \pm \frac{\varepsilon}{2}\,G \mp \frac{\varepsilon^2}{4}\,\big(\tilde s\,B - \kappa\tilde\theta\big)E\Big)
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
    regime.</p>
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
    o2 = math.exp(-B * x + avg + eps / 2 * A - eps ** 2 / 8 * G * G) * (1 + eps / 2 * G - eps ** 2 / 4 * Gp)
    g, gf, _, _ = cir_switching_mean(kappa, th, sig, 5.0)
    ref = numerical_a_callable(T, sym(lam), gf, rtol=1e-13)[0] * math.exp(-B * x)
    eng = FastSwitch(sym(lam), g, order=2).a(T, 2)[0] * math.exp(-B * x)
    assert abs(o2 - eng) < 1e-9, (o2, eng)
    card = r'''    <div class="equation-card">
    $$\begin{aligned}
    u_{1,2}(T, x) \;=\; &\exp\Big(-B\,x - \kappa\bar\theta\,I_1 + \frac{\varepsilon}{2}\,\kappa^2\tilde\theta^2\,I_2
      - \frac{\varepsilon^2}{8}\,\kappa^2\tilde\theta^2 B^2\Big) \\
      &\times\Big(1 \mp \frac{\varepsilon}{2}\,\kappa\tilde\theta\,B \pm \frac{\varepsilon^2}{4}\,\kappa\tilde\theta
      \big(1 - \kappa B - \tfrac12\sigma^2B^2\big)\Big) + O(\varepsilon^3),
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
    CIR survival probability with the averaged level.</p>
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
    <p>Differentiating the expansion at $z = 1$ gives the mean and variance of the count to second order. With
    $L = 1 - e^{-2\lambda T}$,</p>
    $$\mathbb{E}[N_T] = \bar\ell\,T \pm \frac{\varepsilon}{2}\,\tilde\ell\,L,$$
    $$\operatorname{Var}[N_T] = \bar\ell\,T + \varepsilon\,\tilde\ell^{\,2}\,T \pm \frac{\varepsilon}{2}\,\tilde\ell\,L
      - \varepsilon^2\,\tilde\ell^{\,2}\Big(\frac L2 + \frac{L^2}{4}\Big) .$$
    <p>The variance exceeds the mean by $\varepsilon\tilde\ell^2T$, the Green&ndash;Kubo overdispersion. At
    $\ell = (8, 1)$, $T = 1$, $\lambda = 10$ and a start in the busy regime:</p>
''' + table([['mean', f'{mean_1:.5f}', f'{mean_1:.5f}', f'{mean_ex:.5f}'], ['variance', f'{var_1:.5f}', f'{var_2:.5f}', f'{var_ex:.5f}']],
            head=('', 'first order', 'second order', 'exact')) + r'''    <p>The distribution itself follows by evaluating the exact generating function at the 64 roots of unity, as
    described above.</p>
'''
    write('counts', html)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    jumps(); regime_switching(); cir(); black_scholes(); counts()
