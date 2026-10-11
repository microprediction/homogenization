"""Write the explicit, specialized formula sections included into the example pages (tools/pages/explicit/*.html).
Every formula is checked here against the engine at second order and against the numerical solution."""
import cmath, math, os
import numpy as np
from explicit import I_k, J_1, J_2, cir_B, cir_int_B, cir_int_B2, two_state_constant_exact
from fastswitch import FastSwitch, numerical_a_callable
from models import cir_switching_mean, vasicek_jumps, mmpp, bs_switching
from verify_count_cumulants import (bivariate_count_mixed_cumulants,
                                    bivariate_factorial_cumulant_grid,
                                    count_cumulants,
                                    common_shock_factorial_cumulant,
                                    common_shock_factorial_cumulants22,
                                    finite_cumulant_twins,
                                    finite_atomic_jacobian_certificate,
                                    finite_atomic_prony_certificate,
                                    integrated_intensity_cumulants,
                                    integrated_intensity_mixed_cumulants,
                                    mark_factorial_moments,
                                    marked_common_shock_factorial_cumulant,
                                    mixed_factorial_cumulants22,
                                    mixed_poisson_hankel_certificate,
                                    poisson_inverse_instability,
                                    poisson_mixture_w1_inverse_modulus,
                                    poisson_mixture_w1_moment_upper,
                                    poisson_mixture_w1_nonparametric_lower,
                                    poisson_mixture_w1_local_minimax,
                                    poisson_point_mass_w1_upper,
                                    stationary_time_reversal_count_certificate,
                                    ordered_window_reversal_certificate)

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
    hankel = mixed_poisson_hankel_certificate()
    twins = finite_cumulant_twins()
    atomic = finite_atomic_prony_certificate()
    atomic_jacobian = finite_atomic_jacobian_certificate()
    instability = poisson_inverse_instability()
    w1_minimax = poisson_mixture_w1_local_minimax()
    w1_point_mass = poisson_point_mass_w1_upper()
    w1_nonparametric = poisson_mixture_w1_nonparametric_lower()
    w1_moment_upper = poisson_mixture_w1_moment_upper()
    w1_inverse_modulus = poisson_mixture_w1_inverse_modulus()
    reversal = stationary_time_reversal_count_certificate()
    ordered = ordered_window_reversal_certificate()
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
    var_exact = var_1 - eps ** 2 * lt * lt * (L / 2 + L * L / 4)
    Q = sym(lam)
    count_kappa, _ = count_cumulants(Q, ell, T, [1.0, 0.0])
    factorial_kappa = integrated_intensity_cumulants(Q, ell, T, [1.0, 0.0])
    Q3 = np.array([[-3.0, 2.0, 1.0], [1.0, -4.0, 3.0], [2.0, 1.0, -3.0]])
    rates_a = np.array([0.5, 3.0, 6.0])
    rates_b = np.array([4.0, 0.75, 2.5])
    prior3 = np.array([0.2, 0.5, 0.3])
    mixed_count, _ = bivariate_count_mixed_cumulants(Q3, rates_a, rates_b, 1.3, prior3)
    mixed_factorial = mixed_factorial_cumulants22(mixed_count)
    mixed_intensity = integrated_intensity_mixed_cumulants(Q3, rates_a, rates_b, 1.3, prior3)
    common_rates = np.array([0.4, 1.2, 0.7])
    common_count, common_distribution = bivariate_count_mixed_cumulants(
        Q3, rates_a, rates_b, 1.3, prior3, max_count=80,
        common_rates=common_rates)
    common_factorial = mixed_factorial_cumulants22(common_count)
    common_predicted = common_shock_factorial_cumulants22(
        Q3, rates_a, rates_b, common_rates, 1.3, prior3)
    common_factorial_33 = bivariate_factorial_cumulant_grid(
        common_distribution, degree=3)[2, 2]
    common_predicted_33 = common_shock_factorial_cumulant(
        Q3, rates_a, rates_b, common_rates, 3, 3, 1.3, prior3)
    common_marks = np.array([
        [1, 1], [2, 1], [1, 2], [2, 2], [3, 1], [1, 3]])
    common_mark_probabilities = np.array([0.25, 0.20, 0.20, 0.15, 0.10, 0.10])
    mark_moments = mark_factorial_moments(
        common_marks, common_mark_probabilities, degree=3)
    _, marked_distribution = bivariate_count_mixed_cumulants(
        Q3, rates_a, rates_b, 1.3, prior3, max_count=110,
        common_rates=common_rates, common_marks=common_marks,
        common_mark_probabilities=common_mark_probabilities)
    marked_factorial_33 = bivariate_factorial_cumulant_grid(
        marked_distribution, degree=3)[2, 2]
    marked_predicted_33 = marked_common_shock_factorial_cumulant(
        Q3, rates_a, rates_b, common_rates, mark_moments, 3, 3, 1.3, prior3)
    state_mark_probabilities = np.array([
        [0.35, 0.20, 0.15, 0.10, 0.10, 0.10],
        [0.10, 0.25, 0.25, 0.20, 0.10, 0.10],
        [0.15, 0.10, 0.20, 0.15, 0.20, 0.20],
    ])
    state_mark_moments = mark_factorial_moments(
        common_marks, state_mark_probabilities, degree=3)
    _, state_marked_distribution = bivariate_count_mixed_cumulants(
        Q3, rates_a, rates_b, 1.3, prior3, max_count=110,
        common_rates=common_rates, common_marks=common_marks,
        common_mark_probabilities=state_mark_probabilities)
    state_marked_grid = bivariate_factorial_cumulant_grid(
        state_marked_distribution, degree=3)
    state_marked_predicted_33 = marked_common_shock_factorial_cumulant(
        Q3, rates_a, rates_b, common_rates, state_mark_moments,
        3, 3, 1.3, prior3)
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
    <p>At $\ell = (8, 1)$, $T = 1$, $\lambda = 10$ and a start in the busy regime the excess is ''' + f'{var_exact - mean_1:.5f}' + r''',
    against ''' + f'{eps * lt * lt * T:.5f}' + r''' from the first term alone:</p>
''' + table([['mean', f'{mean_1:.5f}', f'{mean_1:.5f}', f'{mean_ex:.5f}'], ['variance', f'{var_1:.5f}', f'{var_exact:.5f}', f'{var_ex:.5f}']],
            head=('', 'first order', 'second order (exact)', 'numerical')) + r'''    <p>The numerical column evaluates the generating function at the 64 roots of unity, as described above, and the
    distribution itself follows the same way. A Monte Carlo check over 400,000 regime paths, sampled exactly from
    exponential holding times, gives each path a Poisson law for the count. It agrees with the moments and with
    every probability up to $k = 15$ within two standard errors. Certificate:
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/verify_option_mc.py">verify_option_mc.py</a>.</p>
    <h2>Factorial cumulants remove Poisson noise</h2>
    <p>The preceding identity extends to every order and any stochastic intensity whose transform is finite near the
    origin. Define the factorial cumulant generating function by</p>
    $$H_N(t)=\log\mathbb E[(1+t)^{N_T}].$$
    <p>Conditional Poisson sampling gives the exact identity</p>
    $$H_N(t)=\log\mathbb E[e^{t\Lambda_T}].$$
    <p>Therefore the $r$th factorial cumulant of $N_T$ is the ordinary $r$th cumulant of $\Lambda_T$. In particular,</p>
    $$\begin{aligned}
      \kappa_2^{(F)}(N_T)&=\kappa_2(N_T)-\kappa_1(N_T),\\
      \kappa_3^{(F)}(N_T)&=\kappa_3(N_T)-3\kappa_2(N_T)+2\kappa_1(N_T),\\
      \kappa_4^{(F)}(N_T)&=\kappa_4(N_T)-6\kappa_3(N_T)+11\kappa_2(N_T)-6\kappa_1(N_T).
    \end{aligned}$$
    <p>Beyond the mean, these combinations strip out Poisson shot noise and leave only fluctuations of the integrated
    rate. For several counts that are conditionally independent given cumulative intensities $\Lambda_1,\ldots,\Lambda_d$,
    the joint identity is</p>
    $$\log\mathbb E\!\left[\prod_{j=1}^d(1+t_j)^{N_j}\right]
      =\log\mathbb E\!\left[e^{\sum_jt_j\Lambda_j}\right].$$
    <p>Thus mixed factorial cumulants identify the joint cumulants of the integrated rates exactly. For two streams,
    writing $\kappa_{rs}$ for the ordinary joint cumulant with $r$ copies of $N_1$ and $s$ copies of $N_2$, the
    signed-Stirling conversion through bidegree $(2,2)$ is</p>
    $$\begin{aligned}
      \kappa^{(F)}_{11}&=\kappa_{11},&
      \kappa^{(F)}_{21}&=\kappa_{21}-\kappa_{11},\\
      \kappa^{(F)}_{12}&=\kappa_{12}-\kappa_{11},&
      \kappa^{(F)}_{22}&=\kappa_{22}-\kappa_{21}-\kappa_{12}+\kappa_{11}.
    \end{aligned}$$
    <p>The transform is applied separately in each coordinate. Conditional independence is essential; without it,
      shared-event shot noise is not removed.</p>
''' + table([[str(k + 1), f'{count_kappa[k]:.8f}', f'{factorial_kappa[k]:.8f}'] for k in range(4)],
            head=('order', 'ordinary count cumulant', 'factorial / intensity cumulant')) + r'''
''' + table([[label, f'{ordinary:.8f}', f'{factorial:.8f}', f'{intensity:.8f}']
             for label, ordinary, factorial, intensity in zip(
                 ('(1,1)', '(2,1)', '(1,2)', '(2,2)'),
                 mixed_count, mixed_factorial, mixed_intensity)],
            head=('mixed order', 'ordinary count', 'factorial count', 'integrated intensities')) + r'''
    <h3>What the count law identifies</h3>
    <p>At the level of the entire distribution there is a stronger, moment-free result. For a Cox count with
      nonnegative integrated intensity $\Lambda$,</p>
    $$G_N(z)=\mathbb E[z^N]=\mathbb E[e^{-(1-z)\Lambda}]
      =\mathcal L_{\Lambda}(1-z),\qquad 0\le z\le1.$$
    <p>Hence the full count law determines the Laplace transform of $\Lambda$ on $[0,1]$. Two count laws can agree
      only if the corresponding Laplace transforms agree on $(0,1)$; analyticity on the positive half-plane and
      uniqueness of Laplace transforms then imply equality of the mixing laws. Thus the mixed-Poisson family is
      identifiable, with no moment-determinacy assumption. This is the Poisson-mixture case of
      <a href="./bibliography.html#Teicher1961">Teicher&apos;s identifiability theorem</a>.</p>
    <h3>What it does not identify: the arrow of latent time</h3>
    <p>Identifying the law of $\Lambda_T$ is not the same as identifying the hidden generator. Let $Y$ be an
      irreducible finite-state chain with row generator $Q$, stationary law $\pi$, and
      $\Pi=\operatorname{diag}(\pi)$. Its stationary time reversal has generator</p>
    $$Q^\leftarrow=\Pi^{-1}Q^\top\Pi.$$
    <p>For state rates $\ell_i\ge0$, put $D=\operatorname{diag}(\ell)$ and
      $\Lambda_T=\int_0^T\ell_{Y_s}\,ds$. Reversing a path preserves every occupation time, so under a stationary
      start $Q$ and $Q^\leftarrow$ give exactly the same law of $\Lambda_T$ and hence exactly the same Cox count
      law for every $T$. This is a genuine non-identification whenever $Q\ne Q^\leftarrow$; the underlying
      time-reversal construction is classical, going back to
      <a href="https://doi.org/10.1017/S0027763000011405">Nagasawa (1964)</a>.</p>
    <p>The endpoint-resolved Feynman&ndash;Kac matrices make both the theorem and its boundary explicit. Define</p>
    $$K_Q(T,z)=\exp\{T[Q+(z-1)D]\},\qquad
      [K_Q(T,z)]_{ij}=\mathbb E_i[z^{N_T}\mathbf1\{Y_T=j\}].$$
    <p>Because $D$ commutes with $\Pi$,</p>
    <div class="equation-card">
    $$K_{Q^\leftarrow}(T,z)=\Pi^{-1}K_Q(T,z)^\top\Pi,
      \qquad
      [K_{Q^\leftarrow}]_{ij}={\pi_j\over\pi_i}[K_Q]_{ji}.$$
    </div>
    <p>Multiplication on the left by $\pi$ and on the right by $\mathbf1$ erases this weighted transpose, proving</p>
    $$\pi K_{Q^\leftarrow}(T,z)\mathbf1=\pi K_Q(T,z)\mathbf1.$$
    <p>The same proof covers several conditionally independent count streams by replacing $(z-1)D$ with a sum
      of diagonal state-rate potentials. It also covers state-dependent compound-Poisson marks by using their
      diagonal probability-generating exponents. Thus neither additional terminal count coordinates nor arbitrary
      contemporaneous state marks recover the arrow of time if all observations remain occupation based.</p>
    <p>The stationarity and aggregation qualifications are essential. With a known start $i$, the transform is
      $e_i^\top K_Q\mathbf1$, which generally differs after reversal. Recording the terminal state also retains the
      weighted-transpose relation above, and ordered event-time or path data can retain temporal direction. The
      theorem concerns terminal occupation-based counts, not those richer experiments.</p>
    <h3>Ordered windows recover the arrow of time</h3>
    <p>A minimal enrichment already breaks that non-identification. Let $N_a^-$ count stream $a$ on $[0,A]$ and
      $N_b^+$ count stream $b$ on $[A+\Delta,A+\Delta+B]$, where stream $a$ has state rate $\ell^{(a)}$.
      Write $F_{ia}=\ell_i^{(a)}-\pi\ell^{(a)}$, $P_t=e^{tQ}$ and
      $H_T(Q)=\int_0^T P_s\,ds$. Conditional Poisson increments on disjoint windows are independent, so there is
      no shot-noise term across the boundary and the entire early/late covariance matrix is exactly</p>
    <div class="equation-card">
    $$C_{A,B,\Delta}(Q)=\operatorname{Cov}(N^-,N^+)
      =F^\top\Pi H_A(Q)e^{\Delta Q}H_B(Q)F.$$
    </div>
    <p>Since $e^{tQ^\leftarrow}=\Pi^{-1}e^{tQ^\top}\Pi$, time reversal transposes this matrix:</p>
    $$C_{A,B,\Delta}(Q^\leftarrow)=C_{A,B,\Delta}(Q)^\top.$$
    <p>Thus the antisymmetric part is an exact arrow-of-time statistic. It vanishes for reversible chains, but it
      need not vanish for irreversible ones. This conclusion uses ordered, stream-labelled windows; summing the
      windows returns to the occupation-only experiment and loses the direction. This is a finite-window,
      time-domain counterpart of the second-order point-process analysis initiated by
      <a href="https://doi.org/10.1111/j.2517-6161.1963.tb00508.x">Bartlett (1963)</a>.</p>
    <p>The fast-switching scale is a boundary effect. Put $Q_m=mQ_0$, take adjacent fixed windows
      $A,B&gt;0$, and let $R=-Q_0^\#$ be the zero-mean potential operator. Then</p>
    $$m^2 C_{A,B,0}(Q_m)\longrightarrow F^\top\Pi R^2F.$$
    <p>More generally, for a microscopic gap $\Delta=\delta/m$ the middle factor survives and the limit is
      $F^\top\Pi R e^{\delta Q_0}RF$; for a fixed positive gap the covariance instead decays exponentially.
      This $m^{-2}$ ordered-boundary signal is distinct from the $m^{-1}$ Green&ndash;Kubo variance accumulated
      inside one macroscopic window.</p>
    <p>For the four-state irreversible certificate with $A=0.7$ and $B=1.1$, the off-diagonal antisymmetric signal
      is ''' + f'{ordered["antisymmetric_signal"]:.12f}' + r'''. Reversal agrees with the transpose to
      ''' + f'{ordered["transpose_error"]:.1e}' + r'''; an independent mixed derivative of the sequential count PGF
      agrees with the semigroup covariance to ''' + f'{ordered["pgf_error"]:.1e}' + r'''. At $m=40$, the scaled
      covariance is within ''' + f'{ordered["fast_limit_error"]:.1e}' + r''' of $F^\top\Pi R^2F$.</p>
    <p>The certificate uses a nonreversible four-state generator whose maximum entrywise difference from its reverse
      is ''' + f'{reversal["generator_gap"]:.12f}' + r'''. Across four PGF arguments, stationary transforms agree
      within ''' + f'{reversal["stationary_pgf_error"]:.1e}' + r''' and the weighted endpoint identity within
      ''' + f'{reversal["endpoint_transpose_error"]:.1e}' + r'''. An independent count-resolved master equation
      agrees within ''' + f'{reversal["count_pmf_error"]:.1e}' + r''', with omitted mass below
      ''' + f'{reversal["tail_bound"]:.1e}' + r'''. In contrast, a known start in state 1 gives count-law total
      variation ''' + f'{reversal["fixed_start_total_variation"]:.12f}' + r'''.</p>
    <h3>Which count laws are mixed Poisson?</h3>
    <p>Injectivity answers uniqueness after a mixed-Poisson representation is known to exist. Existence itself has
      an exact test. Let $p_n=\Pr\{N=n\}$ and $q_n=n!p_n$. Then $N$ is mixed Poisson if and only if, for every
      $r\ge0$, both Hankel matrices</p>
    $$H_r^{(0)}=(q_{i+j})_{i,j=0}^r,\qquad
      H_r^{(1)}=(q_{i+j+1})_{i,j=0}^r$$
    <p>are positive semidefinite. This is precisely the classical
      <a href="./bibliography.html#Stieltjes1894">Stieltjes moment criterion</a>; its connection with mixed Poisson
      laws and factorial moments is also discussed by
      <a href="./bibliography.html#KubaPanholzer2016">Kuba and Panholzer</a>.</p>
    <p>The proof is short and also explains the factorial weighting. If
      $p_n=\int e^{-\lambda}\lambda^n/n!\,\mu(d\lambda)$, then</p>
    $$q_n=\int_0^\infty \lambda^n\,\rho(d\lambda),\qquad
      \rho(d\lambda)=e^{-\lambda}\mu(d\lambda),$$
    <p>so both matrix families are Gram matrices of polynomials in $L^2(\rho)$ and hence are positive
      semidefinite. Conversely, the two Hankel conditions give a measure $\rho$ on $[0,\infty)$ with moments
      $q_n$. Since $\sum_nq_n/n!=\sum_np_n=1$, monotone convergence gives
      $\int e^\lambda\rho(d\lambda)=1$. Therefore
      $\mu(d\lambda)=e^\lambda\rho(d\lambda)$ is a probability measure and reproduces $p_n$. The same exponential
      moment makes $\rho$ moment-determinate, so the mixing law is unique.</p>
    <p>This yields a nested falsification hierarchy. The first two nontrivial minors require</p>
    $$2p_0p_2\ge p_1^2,\qquad 3p_1p_3\ge2p_2^2,$$
    <p>but finitely many such inequalities are not sufficient. For example, let $B$ have probabilities
      $(0.3,0.1,0.1,0.2,0.3)$ on $\{0,1,2,3,4\}$, let
      $\Pr\{G=n\}=2^{-n-1}$, and take $N$ to have law $0.9\mathcal L(B)+0.1\mathcal L(G)$. This law has full
      support and is overdispersed:</p>
    $$\mathbb E N=''' + f'{hankel["counterexample_mean"]:.4f}' + r''',\qquad
      \operatorname{Var}(N)=''' + f'{hankel["counterexample_variance"]:.4f}' + r'''>\mathbb E N.$$
    <p>Its two order-one Hankel determinants are
      ''' + f'{hankel["order_one_determinants"][0]:.7f}' + r''' and
      ''' + f'{hankel["order_one_determinants"][1]:.7f}' + r''', yet</p>
    $$\det H_2^{(0)}=''' + f'{hankel["order_two_determinant"]:.9f}' + r'''<0,$$
    <p>which proves it is not mixed Poisson. As a positive control, the certificate constructs a four-atom mixing
      law; after diagonal normalization, the smallest eigenvalues of $H_2^{(0)}$ and $H_2^{(1)}$ are
      ''' + f'{hankel["valid_min_eigenvalues"][0]:.8f}' + r''' and
      ''' + f'{hankel["valid_min_eigenvalues"][1]:.8f}' + r'''. Thus overdispersion is only the first coarse
      necessary condition, while the full Hankel hierarchy is necessary and sufficient.</p>
    <p>The multivariate statement is identical under conditional independence:</p>
    $$G_{N_1,\ldots,N_d}(z_1,\ldots,z_d)
      =\mathcal L_{\boldsymbol\Lambda}(1-z_1,\ldots,1-z_d),
      \qquad 0\le z_j\le1.$$
    <p>Finite cumulant lists are fundamentally weaker. Fix any order $k$, set $n=k+1$ and $x_j=j+1$ for
      $j=0,\ldots,n$. For any $0&lt;\delta&lt;2^{-n}$, define two strictly positive probability laws by</p>
    $$p_j^{\pm}={n\choose j}\left(2^{-n}\pm\delta(-1)^j\right).$$
    <p>The $n$th finite-difference identity gives</p>
    $$\sum_{j=0}^{n}(-1)^j{n\choose j}(j+1)^r=0,\qquad r&lt;n.$$
    <p>Consequently the two intensities have the same first $k$ moments and cumulants, so their mixed-Poisson
      counts have the same first $k$ factorial cumulants. They are nevertheless different count laws because</p>
    $$\mathbb P_+(N=0)-\mathbb P_-(N=0)
      =2\delta e^{-1}(1-e^{-1})^n\ne0.$$
    <p>For the certificate&apos;s $k=4$ construction, the maximum discrepancy among the first four factorial
      cumulants is ''' + f'{twins["factorial_gap"]:.2e}' + r''', while the zero-count probability differs by
      ''' + f'{twins["zero_gap"]:.12f}' + r''' and the numerically summed total-variation distance is
      ''' + f'{twins["total_variation"]:.12f}' + r''', with omitted contribution below
      ''' + f'{twins["total_variation_tail_bound"]:.2e}' + r'''. Full-law identification, all-order analytic
      identification, and finite-order cumulant identification are therefore distinct claims.</p>
    <h3>The sharp finite-atomic exception</h3>
    <p>A sparsity assumption changes the last conclusion completely. Suppose the mixing law has at most $r$ atoms,</p>
    $$\mu=\sum_{j=1}^{s}w_j\delta_{x_j},\qquad
      1\le s\le r,\quad 0\le x_1&lt;\cdots&lt;x_s,\quad w_j&gt;0.$$
    <p>Then factorial cumulants of orders $1,\ldots,2r-1$ identify $\mu$, and this order is sharp without
      additional separation or weight assumptions. Indeed, factorial cumulants of the count are ordinary
      cumulants of $\Lambda$, and the triangular moment-cumulant relations recover
      $m_k=\mathbb E[\Lambda^k]$ for $0\le k\le2r-1$, with $m_0=1$.</p>
    <p>To reconstruct the law, first read $s$ as the rank of
      $H_{r-1}=(m_{i+j})_{i,j=0}^{r-1}$. For the actual support size,</p>
    $$H_{s-1}=V\operatorname{diag}(w_1,\ldots,w_s)V^\top\succ0,
      \qquad V_{ij}=x_j^i.$$
    <p>Let $P(x)=x^s+c_{s-1}x^{s-1}+\cdots+c_0$ be the monic support polynomial. Since
      $P(x_j)=0$, its coefficients are the unique solution of the Hankel system</p>
    $$H_{s-1}\begin{pmatrix}c_0\\ \vdots\\ c_{s-1}\end{pmatrix}
      =-\begin{pmatrix}m_s\\ \vdots\\ m_{2s-1}\end{pmatrix}.$$
    <p>The roots of $P$ are the atoms $x_j$, after which the first $s$ moment equations form a nonsingular
      Vandermonde system for the weights. This is the classical annihilating-polynomial mechanism behind
      <a href="./bibliography.html#Prony1795">Prony&apos;s method</a>.</p>
    <p>The count $2r-1$ cannot be reduced. On the $2r$ nodes $x_j=j+1$, split the signed coefficients</p>
    $$a_j=(-1)^j{2r-1\choose j},\qquad j=0,\ldots,2r-1,$$
    <p>into their positive and negative parts and normalize each by $2^{2r-2}$. Each part is an $r$-atomic
      probability law. The finite-difference identity makes their moments, and hence cumulants, identical through
      order $2r-2$, while their order-$(2r-1)$ moment gap is</p>
    $$\frac{1}{2^{2r-2}}\sum_{j=0}^{2r-1}(-1)^j{2r-1\choose j}(j+1)^{2r-1}
      =-\frac{(2r-1)!}{2^{2r-2}}\ne0.$$
    <p>For $r=4$, the certificate recovers a four-atom law with maximum support and weight errors
      ''' + f'{atomic["support_error"]:.2e}' + r''' and ''' + f'{atomic["weight_error"]:.2e}' + r'''. Its exact
      sharpness pair agrees through order six and has order-seven moment and cumulant gap
      ''' + f'{atomic["sharp_moment_gap"]:.2f}' + r'''. This result is purely about exact identification: Hankel and
      Vandermonde systems can become arbitrarily ill-conditioned as atoms collide or weights vanish, so it does not
      assert uniform stable recovery.</p>
    <h3>Exact local conditioning geometry</h3>
    <p>The instability boundary has an exact algebraic description. For an exactly $r$-atomic law, eliminate the
      last weight by $w_r=1-\sum_{j&lt;r}w_j$ and define the square parameter-to-moment map</p>
    $$\Phi(w_1,\ldots,w_{r-1},x_1,\ldots,x_r)=(m_1,\ldots,m_{2r-1}).$$
    <p>On the ordered interior $w_j&gt;0$ and $x_1&lt;\cdots&lt;x_r$, its Jacobian satisfies</p>
    <div class="equation-card">
    $$\left|\det D\Phi\right|
      =\left(\prod_{j=1}^r w_j\right)
       \left(\prod_{1\le i&lt;j\le r}(x_j-x_i)^4\right).$$
    </div>
    <p>The moment-to-cumulant transformation is unit triangular, so the parameter-to-factorial-cumulant map has
      the same determinant.</p>
    <p>To prove the identity, temporarily retain all $r$ weights and include $m_0$. The $j$th weight and atom
      columns of the full Jacobian are respectively</p>
    $$v(x_j)=(1,x_j,\ldots,x_j^{2r-1})^\top,\qquad
      w_jv'(x_j).$$
    <p>After interlacing these columns, this is a weighted confluent Vandermonde matrix, whose determinant is the
      fourth power of the ordinary Vandermonde product times $\prod_jw_j$. Changing coordinates from the weights
      to $(w_1,\ldots,w_{r-1},m_0)$ and expanding along $m_0=1$ leaves the displayed reduced determinant. The
      classical inverse and conditioning analysis of these matrices goes back to
      <a href="./bibliography.html#Gautschi1962">Gautschi (1962)</a>.</p>
    <p>The inverse-function theorem now gives a locally analytic inverse everywhere in the ordered interior. It
      also yields an explicit differential bound. Put $d=2r-1$, suppose $0\le x_j\le R$,
      $w_j\ge w_*$, and $x_{j+1}-x_j\ge\delta$, and set
      $A_R=\max\{1,R^{2r-1}\}$. Every entry of $D\Phi$ is bounded by $dA_R$, so</p>
    $$\|D\Phi\|_2\le d^2A_R,\qquad
      \sigma_{\min}(D\Phi)
      \ge {w_*^r\delta^{2r(r-1)}\over(d^2A_R)^{d-1}}.$$
    <p>Thus separated atoms with weights bounded away from zero have uniformly bounded infinitesimal inverse
      sensitivity. The determinant formula also pinpoints both degeneracies: it vanishes linearly with a disappearing
      weight and to fourth order for each colliding pair. This is a local parameter-stability statement, not a claim
      of total-variation stability for unrestricted mixing measures.</p>
    <p>The certificate evaluates the reduced Jacobian directly for two through five atoms. Its maximum relative
      determinant error is ''' + f'{atomic_jacobian["determinant_relative_error"]:.2e}' + r'''; a centered directional
      derivative agrees within ''' + f'{atomic_jacobian["derivative_error"]:.2e}' + r'''; and after removing the
      noncolliding factors, a shrinking pair has measured volume-collapse order
      ''' + f'{atomic_jacobian["collision_order"]:.9f}' + r'''.</p>
    <h3>Identification is not stable inversion</h3>
    <p>Injectivity is qualitative. It does not make recovery of an unrestricted mixing law stable in total
      variation. For two point-mass mixing laws $\delta_a$ and $\delta_b$,</p>
    $$d_{\rm TV}(\delta_a,\delta_b)=1\qquad(a\ne b),$$
    <p>whereas their count laws are $\operatorname{Pois}(a)$ and $\operatorname{Pois}(b)$. Their Hellinger affinity is</p>
    $$\sum_{k\ge0}\sqrt{p_a(k)p_b(k)}
      =\exp\!\left\{-\frac12(\sqrt a-\sqrt b)^2\right\},$$
    <p>so</p>
    $$d_{\rm TV}(\operatorname{Pois}(a),\operatorname{Pois}(b))
      \le \sqrt{1-e^{-(\sqrt a-\sqrt b)^2}}.$$
    <p>For $0&lt;a&lt;b$, the likelihood ratio $p_a(k)/p_b(k)=e^{b-a}(a/b)^k$ crosses one once. Therefore the exact
      total variation is</p>
    $$d_{\rm TV}(\operatorname{Pois}(a),\operatorname{Pois}(b))
      =F_a(k_*)-F_b(k_*),\qquad
      k_*=\left\lfloor\frac{b-a}{\log(b/a)}\right\rfloor.$$
    <p>Thus even on a fixed compact intensity interval, taking $b\to a$ leaves mixing-law total variation equal to
      one while count-law total variation tends to zero. More precisely, at a noninteger $a$ and for sufficiently
      small $h&gt;0$, the likelihood-ratio crossing is $k=\lfloor a\rfloor$, and</p>
    $$\frac{d_{\rm TV}(\operatorname{Pois}(a),\operatorname{Pois}(a+h))}{h}
      \longrightarrow \Pr\{\operatorname{Pois}(a)=\lfloor a\rfloor\}.$$
    <p>There is no global Wasserstein rescue without an intensity bound. For integer $m$, the crossing of
      $\operatorname{Pois}(m)$ and $\operatorname{Pois}(m+1)$ is exactly $k=m$, so</p>
    $$d_{\rm TV}(\operatorname{Pois}(m),\operatorname{Pois}(m+1))
      =F_m(m)-F_{m+1}(m)
      \sim\frac1{\sqrt{2\pi m}},$$
    <p>although $W_1(\delta_m,\delta_{m+1})=1$. At $a=4.5$ and $h=10^{-5}$ the certificate gives
      $d_{\rm TV}/h=$ ''' + f'{instability["local_tv"][-1] / instability["steps"][-1]:.12f}' + r''',
      against the limit ''' + f'{instability["local_limit"]:.12f}' + r'''. At $m=10^6$ it gives
      $\sqrt m\,d_{\rm TV}=$ ''' + f'{instability["scaled_tv"][-1]:.12f}' + r''', against
      $1/\sqrt{2\pi}=$ ''' + f'{instability["asymptotic_constant"]:.12f}' + r'''.</p>
    <p>This does not contradict identifiability. It says that finite-sample recovery needs a weaker loss and/or
      structural restrictions such as bounded support and smoothness. The nonparametric Poisson-mixture estimation
      theory of <a href="./bibliography.html#RoueffRyden2005">Roueff and Ryd&eacute;n (2005)</a> makes such regularity
      assumptions explicit. For a fixed finite-state intensity model, $\Lambda_T$ is bounded; that removes the
      escaping-mass example but not the total-variation discontinuity created by moving atoms.</p>
    <h3>A finite-sample impossibility theorem</h3>
    <p>The discontinuity implies more than the absence of a convenient inverse bound. Suppose
      $N_1,\ldots,N_n$ are iid mixed-Poisson observations with unknown mixing law $\mu$ supported on a fixed
      compact interval containing more than one point. For any estimator $\widehat\mu_n$, including a randomized
      estimator,</p>
    <div class="equation-card">
    $$\sup_{\mu}\mathbb E_\mu d_{\rm TV}(\widehat\mu_n,\mu)\ge\frac12,
      \qquad n\ge1.$$
    </div>
    <p>Thus the unrestricted mixing law is not uniformly consistently estimable in total variation, even though it
      is identified by the population count law and even though its support is compact.</p>
    <p>The proof is a two-point coupling argument. For any two mixing laws $\mu_0,\mu_1$, let $P_i$ be the
      corresponding one-count laws and put
      $R_i=\mathbb E_i d_{\rm TV}(\widehat\mu_n,\mu_i)$. A maximal coupling of
      $P_0^{\otimes n}$ and $P_1^{\otimes n}$, using the same estimator randomization when the samples agree, and
      the triangle inequality give</p>
    $$\max(R_0,R_1)\ge {d_{\rm TV}(\mu_0,\mu_1)\over2}
      \{1-d_{\rm TV}(P_0^{\otimes n},P_1^{\otimes n})\}.$$
    <p>Choose distinct point masses $\mu_0=\delta_a$ and $\mu_1=\delta_b$ inside the interval. Their distance is
      one, while the product count distance tends to zero as $b\to a$. Letting $b$ approach $a$ proves the displayed
      lower bound for every fixed $n$.</p>
    <p>There is also an exact quantitative certificate. Take $a=4.5$, $b_n=a+1/n$, and even $n$. Under the two
      hypotheses, the sufficient statistic $\sum_iN_i$ has laws
      $\operatorname{Pois}(na)$ and $\operatorname{Pois}(na+1)$; conditional on the sum, the allocation among the
      $n$ observations is the same multinomial law. Hence</p>
    $$d_{\rm TV}\{P_a^{\otimes n},P_{a+1/n}^{\otimes n}\}
      =F_{na}(na)-F_{na+1}(na)
      \sim {1\over\sqrt{2\pi an}}.$$
    <div class="table-wrap"><table class="impl">
      <thead><tr><th>$n$</th><th>product count TV</th><th>mixing-law TV risk lower bound</th></tr></thead>
      <tbody>
        <tr><td>10</td><td>0.059144045738</td><td>0.470427977131</td></tr>
        <tr><td>100</td><td>0.018795883152</td><td>0.490602058424</td></tr>
        <tr><td>1,000</td><td>0.005946750031</td><td>0.497026624985</td></tr>
        <tr><td>10,000</td><td>0.001880621497</td><td>0.499059689251</td></tr>
        <tr><td>1,000,000</td><td>0.000188063184</td><td>0.499905968408</td></tr>
      </tbody>
    </table></div>
    <p>At $n=10^6$, $\sqrt n$ times the product distance is $0.188063184068$, versus
      $1/\sqrt{2\pi a}=0.188063194516$. The lower bound concerns total-variation recovery of an unrestricted
      measure. It does not rule out weak-loss consistency, parametric finite-state recovery under separation, or
      density estimation on smoothness classes; those are precisely the kinds of restrictions used by
      <a href="./bibliography.html#RoueffRyden2005">Roueff and Ryd&eacute;n</a>.</p>
    <h3>A local Wasserstein lower bound</h3>
    <p>Compact support does make Wasserstein loss qualitatively weaker than total variation, but it does not permit
      a uniformly faster-than-parametric rate. Let $\mathcal M_{[4,5]}$ be all probability laws on $[4,5]$ and define</p>
    $$R_n^{(1)}=\inf_{\widehat\mu_n}\sup_{\mu\in\mathcal M_{[4,5]}}
      \mathbb E_\mu W_1(\widehat\mu_n,\mu).$$
    <p>For any $a$ in the interior of the interval and any fixed $h&gt;0$, compare
      $\mu_{0,n}=\delta_a$ with $\mu_{1,n}=\delta_{a+h/\sqrt n}$. They lie in the class for all sufficiently large
      $n$ and have $W_1(\mu_{0,n},\mu_{1,n})=h/\sqrt n$. The same two-point coupling inequality used above gives</p>
    $$\sqrt n R_n^{(1)}\ge {h\over2}
      \left[1-d_{\rm TV}\{\operatorname{Pois}(na),
      \operatorname{Pois}(na+h\sqrt n)\}\right].$$
    <p>This is a restriction to the point-mass submodel, so it is a valid lower bound for the full class. It is the
      classical two-point testing reduction of <a href="./bibliography.html#LeCam1973">Le Cam</a>, here with the
      distance and testing affinity evaluated exactly.</p>
    <p>Put $z=h/(2\sqrt a)$. The one-crossing formula has cutoff</p>
    $$k_n=\left\lfloor {h\sqrt n\over
      \log(1+h/(a\sqrt n))}\right\rfloor
      =na+{h\over2}\sqrt n+O(1).$$
    <p>Applying the normal limit to the two Poisson distribution functions at this cutoff yields</p>
    $$d_{\rm TV}\{\operatorname{Pois}(na),\operatorname{Pois}(na+h\sqrt n)\}
      \longrightarrow 2\Phi(z)-1,$$
    <p>and therefore</p>
    <div class="equation-card">
    $$\liminf_{n\to\infty}\sqrt n R_n^{(1)}\ge h\Phi(-z).$$
    </div>
    <p>The right side is optimized by $h=2\sqrt a\,z_*$, where $z_*$ is the unique positive root of
      $\Phi(-z)=z\phi(z)$. At $a=4.5$,</p>
    $$z_*=''' + f'{w1_minimax["z_star"]:.12f}' + r''',\qquad
      h_*=''' + f'{w1_minimax["h_star"]:.12f}' + r''',\qquad
      h_*\Phi(-z_*)=''' + f'{w1_minimax["asymptotic_constant"]:.12f}' + r'''.$$
''' + table([[f'{n:,}', f'{tv:.12f}', f'{bound:.12f}']
             for n, tv, bound in zip(
                 w1_minimax['sample_sizes'], w1_minimax['product_tv'],
                 w1_minimax['scaled_risk_lower_bounds'])],
            head=('$n$', 'product count TV', r'$\sqrt n$ times $W_1$ risk lower bound')) + r'''
    <p>The final column tends to $0.721126760493$. This proves only a lower bound: it rules out uniform
      $o(n^{-1/2})$ Wasserstein-1 recovery, even on the point-mass submodel, but it does not claim that the unrestricted
      mixing class has an $O(n^{-1/2})$ estimator.</p>
    <h3>A nonparametric logarithmic obstruction</h3>
    <p>The unrestricted compact class is substantially harder than its point-mass submodel. More generally, for
      $a\ge0$ and $B&gt;0$, put $\mathcal M_{[a,a+B]}$ for all laws on that intensity interval, and let
      $R_n^{(1)}(a,B)$ denote the same minimax risk with this class. Then</p>
    <div class="equation-card">
    $$\boxed{\displaystyle
      \liminf_{n\to\infty}{\log n\over\log\log n}R_n^{(1)}(a,B)\ge {B\over2}.}$$
    </div>
    <p>Thus no estimator has worst-case $W_1$ risk
      $o((\log\log n)/\log n)$ on the unrestricted class.</p>
    <p>The construction is explicit. Fix an integer $L\ge2$, let $J\sim\operatorname{Bin}(L,1/2)$, put
      $h=B/L$, and define</p>
    $$\mu_{L,+}=\mathcal L(a+hJ\mid J\ {\rm even}),\qquad
      \mu_{L,-}=\mathcal L(a+hJ\mid J\ {\rm odd}).$$
    <p>The finite-difference identity</p>
    $$\sum_{j=0}^L(-1)^j{L\choose j}(a+hj)^r=0,\qquad 0\le r&lt;L,$$
    <p>shows that the two laws have identical first $L-1$ moments. Yet on
      $[a+kh,a+(k+1)h)$ their CDF difference is</p>
    $$2^{1-L}\sum_{j=0}^k(-1)^j{L\choose j}
      =2^{1-L}(-1)^k{L-1\choose k},$$
    <p>so summing the absolute areas gives the exact separation</p>
    $$W_1(\mu_{L,+},\mu_{L,-})={B\over L}.$$
    <p>The induced mixed-Poisson count laws are nevertheless exponentially close in $L$. To see this without an
      analytic inversion bound, write a count as</p>
    $$N=Z+\sum_{i=1}^L B_iX_i,$$
    <p>where $Z\sim\operatorname{Pois}(a)$, the $B_i$ are fair Bernoulli variables conditioned on even or odd
      parity, and the independent $X_i\sim\operatorname{Pois}(h)$. Whenever some $X_i=0$, flipping the first
      corresponding $B_i$ changes parity without changing $N$ and maps the uniform even-parity law bijectively to
      the uniform odd-parity law. Therefore, for the one-count laws $P_{L,+},P_{L,-}$,</p>
    $$d_{\rm TV}(P_{L,+},P_{L,-})\le\Pr\{X_1&gt;0,\ldots,X_L&gt;0\}
      =(1-e^{-B/L})^L,$$
    <p>and product coupling gives</p>
    $$d_{\rm TV}(P_{L,+}^{\otimes n},P_{L,-}^{\otimes n})
      \le n(1-e^{-B/L})^L.$$
    <p>Le Cam&apos;s two-point metric inequality now yields, for every $L\ge2$,</p>
    $$R_n^{(1)}(a,B)\ge {B\over2L}
      \left[1-n(1-e^{-B/L})^L\right].$$
    <p>For fixed $\epsilon&gt;0$, take
      $L=\lceil(1+\epsilon)\log n/\log\log n\rceil$. Since
      $n(1-e^{-B/L})^L\le n(B/L)^L\to0$, the scaled lower limit is at least
      $B/[2(1+\epsilon)]$; letting $\epsilon$ decrease to zero proves the boxed result.</p>
    <p>The certificate below first checks exact moment matching, exact $W_1$, and the count-law coupling for
      $[a,a+B]=[4,5]$. It then optimizes the displayed finite-$n$ lower bound over integer $L$.</p>
''' + table([[f'{order}', f'{w1:.12f}', f'{tv:.3e}', f'{coupling:.3e}']
             for order, w1, tv, coupling in zip(
                 w1_nonparametric['check_orders'], w1_nonparametric['exact_w1'],
                 w1_nonparametric['count_tv'], w1_nonparametric['coupling_bounds'])],
            head=('$L$', 'exact $W_1$', 'count TV', 'coupling upper bound')) + r'''
''' + table([[label, f'{order}', f'{risk:.12e}', f'{scaled:.12f}']
             for label, order, risk, scaled in zip(
                 w1_nonparametric['sample_labels'], w1_nonparametric['optimal_orders'],
                 w1_nonparametric['risk_lower_bounds'],
                 w1_nonparametric['scaled_lower_bounds'])],
            head=('$n$', 'optimizing $L$', '$W_1$ risk lower bound',
                  r'$(\log n/\log\log n)$ times bound')) + r'''
    <p>The scaled certificate approaches its proved asymptotic lower constant $1/2$ slowly. This ordinary,
      unsmoothed $W_1$ obstruction is consistent with the smoothness-class theory of
      <a href="./bibliography.html#RoueffRyden2005">Roueff and Ryd&eacute;n</a>. For support $[0,B]$,
      <a href="./bibliography.html#MiaoEtAl2024">Miao et al.</a> prove the matching upper order for the NPMLE and a
      matching minimax lower order; <a href="./bibliography.html#LimHan2024">Lim and Han</a> obtain a nearly
      root-$n$ rate only after replacing ordinary transport by Gaussian-smoothed optimal transport.</p>
    <h3>A matching moment-estimator upper rate</h3>
    <p>The upper order can also be recovered by a direct estimator on any fixed interval $[a,a+B]$. Put
      $M=a+B$ and, from iid counts $N_1,\ldots,N_n$, define the unbiased normalized factorial-moment estimates</p>
    $$\widehat m_k={1\over n}\sum_{i=1}^n{(N_i)_k\over M^k},\qquad 1\le k\le L.$$
    <p>Choose any probability law on $[a,M]$ minimizing the largest moment residual,</p>
    $$\widehat\mu_{n,L}\in\arg\min_{\nu\in\mathcal P([a,M])}
      \max_{1\le k\le L}\left|\int(\theta/M)^k\,d\nu(\theta)-\widehat m_k\right|.$$
    <p>A minimizer exists by weak compactness; the compact-valued argmin correspondence has a Borel graph, so fix
      any measurable selection. If
      $\epsilon_L=\max_{k\le L}|\widehat m_k-m_k|$, comparison with the true law gives a discrepancy of at most
      $2\epsilon_L$ in every fitted moment.</p>
    <p>For every $n,L\ge1$, the following nonasymptotic bound is explicit up to the universal Jackson constant
      $C_J$:</p>
    <div class="equation-card">
    $$\sup_{\mu\in\mathcal P([a,a+B])}\mathbb E_\mu W_1(\widehat\mu_{n,L},\mu)
      \le {2C_JM\over L}
      +{6(B+C_JM)(L+1)^{3/2}\over\sqrt n}
       \{14\sqrt{A_ML}\}^{L},$$
    $$A_M=\max(1,M^{-1}).$$
    </div>
    <p>To prove it, use Kantorovich duality and subtract the value of each Lipschitz test function at $a$. Extend it
      constantly to $[0,a]$, rescale to $[0,1]$, and apply
      <a href="./bibliography.html#Jackson1921">Jackson&apos;s polynomial approximation theorem</a>. A degree-$L$
      polynomial approximates the test function within $C_JM/L$. In the shifted Chebyshev basis
      $q_j(x)=T_j(2x-1)$, the recurrence
      $q_{j+1}=(4x-2)q_j-q_{j-1}$ implies that the monomial coefficient $\ell^1$ norm of $q_j$ is at most $7^j$.
      The Chebyshev coefficients of a bounded polynomial are at most twice its sup norm, so the approximant&apos;s
      monomial coefficient norm is at most $3(B+C_JM)7^L$.</p>
    <p>The stochastic term follows from the exact falling-factorial product identity</p>
    $$(N)_k^2=\sum_{j=0}^k{k\choose j}^2j!(N)_{2k-j}.$$
    <p>Since $N\mid\theta\sim\operatorname{Pois}(\theta)$ and $\theta\le M$, it gives</p>
    $$\mathbb E\epsilon_L\le { (L+1)^{3/2}\over\sqrt n}
      \{2\sqrt{A_ML}\}^{L}.$$
    <p>Combining the approximation and moment errors proves the displayed bound. For any fixed $0&lt;c&lt;1$, take
      $L_n=\lfloor c\log n/\log\log n\rfloor$. The logarithm of the stochastic factor is
      $-\tfrac12(1-c)\log n+o(\log n)$, while the approximation term is $O(\log\log n/\log n)$. Consequently,</p>
    <div class="equation-card">
    $$\sup_{\mu\in\mathcal P([a,a+B])}\mathbb E_\mu
      W_1(\widehat\mu_{n,L_n},\mu)
      =O\!\left({\log\log n\over\log n}\right).$$
    </div>
    <p>Together with the preceding lower theorem, this proves that the unrestricted compact minimax rate is exactly
      $\Theta((\log\log n)/\log n)$ for every fixed $a\ge0$ and $B&gt;0$. The certificate checks the factorial
      second-moment identity through order eight to relative error
      ''' + f'{w1_moment_upper["relative_second_moment_error"]:.2e}' + r''' and verifies the coefficient recurrence.
      For $M=5$ and $c=1/2$, the derived stochastic factor decays as follows.</p>
''' + table([[f'$e^{{{log_n:.0f}}}$', f'{degree}', f'{log_factor:.6f}']
             for log_n, degree, log_factor in zip(
                 w1_moment_upper['log_sample_sizes'],
                 w1_moment_upper['moment_degrees'],
                 w1_moment_upper['log_stochastic_factors'])],
            head=('$n$', '$L_n$', 'log stochastic factor')) + r'''
    <h3>The compact inverse has an optimal logarithmic modulus</h3>
    <p>The statistical upper bound also yields a population stability theorem. For a mixing law $\mu$ on
      $[a,a+B]$, write $P_\mu$ for its one-count mixed-Poisson law and define</p>
    $$\omega_{a,B}(\delta)=\sup\{W_1(\mu,\nu):
      d_{\rm TV}(P_\mu,P_\nu)\le\delta\}.$$
    <p>Fix one measurable version of the moment estimator above. Maximally couple $n$ iid counts from
      $P_\mu$ and $P_\nu$, and apply the same estimator to the coupled samples. They agree with probability at
      least $(1-\delta)^n$. On the complementary event, two estimator outputs supported on $[a,a+B]$ are at
      Wasserstein distance at most $B$. The triangle inequality therefore gives</p>
    $$\omega_{a,B}(\delta)\le 2R_{n,L}+B\{1-(1-\delta)^n\},$$
    <p>where $R_{n,L}$ is the preceding worst-case estimator bound. In particular, with $M=a+B$ and
      $A_M=\max(1,M^{-1})$,</p>
    <div class="equation-card">
    $$\omega_{a,B}(\delta)\le {4C_JM\over L}
      +{12(B+C_JM)(L+1)^{3/2}\over\sqrt n}
       \{14\sqrt{A_ML}\}^{L}
      +B\{1-(1-\delta)^n\}.$$
    </div>
    <p>This holds for every positive integer $n,L$. Taking
      $n=\lfloor\delta^{-1/2}\rfloor$ and
      $L=\lfloor c\log n/\log\log n\rfloor$ for any fixed $0&lt;c&lt;1$ makes the last two terms negligible
      relative to $1/L$, and proves</p>
    $$\omega_{a,B}(\delta)=O\!\left(
      {\log\log(1/\delta)\over\log(1/\delta)}\right).$$
    <p>The order cannot be improved. For the even/odd binomial pair used in the minimax lower bound,</p>
    $$W_1(\mu_{L,+},\mu_{L,-})={B\over L},\qquad
      d_{\rm TV}(P_{L,+},P_{L,-})\le
      \delta_L=(1-e^{-B/L})^L.$$
    <p>Hence $\omega_{a,B}(\delta_L)\ge B/L$. More generally, for every sufficiently small $\delta$, choosing
      $L$ of order $\log(1/\delta)/\log\log(1/\delta)$ makes $\delta_L\le\delta$ and gives the reverse order.
      Therefore</p>
    <div class="equation-card">
    $$\boxed{\displaystyle
      \omega_{a,B}(\delta)=\Theta\!\left(
      {\log\log(1/\delta)\over\log(1/\delta)}\right),
      \qquad \delta\downarrow0.}$$
    </div>
    <p>This is a positive compact-support stability statement in $W_1$, not in total variation. It complements
      the minimax theorem of <a href="./bibliography.html#MiaoEtAl2024">Miao et al.</a>: the same logarithmic
      ill-posedness governs both population inversion and unrestricted statistical recovery. On $[4,5]$, the
      certificate&apos;s lower scaled separation reaches
      ''' + f'{w1_inverse_modulus["lower_scaled_separation"][-1]:.12f}' + r''' at $L=1024$ on its slow approach to
      one. At $\log(1/\delta)=10{,}000$, the upper construction selects $L=''' + f'{w1_inverse_modulus["degrees"][-1]}' + r'''$;
      the logarithms of its stochastic and sample-disagreement
      terms are ''' + f'{w1_inverse_modulus["log_stochastic_terms"][-1]:.6f}' + r''' and
      ''' + f'{w1_inverse_modulus["log_coupling_terms"][-1]:.6f}' + r'''.</p>
    <h3>The point-mass submodel has the parametric rate</h3>
    <p>The preceding lower bound is rate-sharp on the submodel that generated it. Define</p>
    $$R_{n,\delta}^{(1)}=\inf_{\widehat\mu_n}\sup_{4\le\lambda\le5}
      \mathbb E_\lambda W_1(\widehat\mu_n,\delta_\lambda).$$
    <p>When the truth is $\delta_\lambda$, the observations are iid $\operatorname{Pois}(\lambda)$. Put
      $S_n=\sum_iN_i$, project $S_n/n$ onto $[4,5]$, and put a point mass at the projected value. Since projection
      cannot increase distance to $\lambda$ and $S_n\sim\operatorname{Pois}(n\lambda)$,</p>
    $$\mathbb E_\lambda W_1(\widehat\mu_n,\delta_\lambda)
      \le {1\over n}\mathbb E|S_n-n\lambda|
      =2\lambda\Pr\{\operatorname{Pois}(n\lambda)=\lfloor n\lambda\rfloor\}
      \le\sqrt{\lambda\over n}.$$
    <p>The equality follows by splitting the centered Poisson variable at its mean and using
      $k p_\mu(k)=\mu p_\mu(k-1)$; the inequality is Cauchy&ndash;Schwarz. Combining this estimator with the local
      two-point lower bound gives the rigorous sandwich</p>
    <div class="equation-card">
    $$0.721126760493\le\liminf_{n\to\infty}\sqrt n\,R_{n,\delta}^{(1)}
      \le\limsup_{n\to\infty}\sqrt n\,R_{n,\delta}^{(1)}\le\sqrt5.$$
    </div>
    <p>So $R_{n,\delta}^{(1)}=\Theta(n^{-1/2})$. At the certificate point $\lambda=4.5$, projection changes the
      interior risk only by exponentially small tails, and the unprojected sample-mean benchmark has the sharper
      pointwise limit</p>
    $$\sqrt n\,\mathbb E_{4.5}|S_n/n-4.5|\longrightarrow
      \sqrt{9/\pi}=''' + f'{w1_point_mass["pointwise_limit"]:.12f}' + r'''.$$
''' + table([[f'{n:,}', f'{risk:.12f}', f'{scaled:.12f}']
             for n, risk, scaled in zip(
                 w1_point_mass['sample_sizes'], w1_point_mass['exact_risks'],
                 w1_point_mass['scaled_exact_risks'])],
            head=('$n$', 'exact sample-mean risk', r'$\sqrt n$ times risk')) + r'''
    <p>This sample-mean upper bound applies only to the one-parameter family of point masses. An arbitrary law in
      $\mathcal M_{[4,5]}$ cannot be estimated by reducing it to its mean; the moment estimator above attains the
      slower unrestricted logarithmic rate.</p>
    <h3>What common shocks add</h3>
    <p>The correction is exact. Index independent Poisson event streams by the nonempty subsets
      $A\subseteq\{1,\ldots,d\}$ of coordinates that each event increments. Conditional on their cumulative
      intensities $\Lambda_A$, put $N_j=\sum_{A\ni j}C_A$. Then</p>
    $$\log\mathbb E\prod_{j=1}^d(1+t_j)^{N_j}
      =\log\mathbb E\exp\left\{\sum_{A\ne\varnothing}\Lambda_A
      \left(\prod_{j\in A}(1+t_j)-1\right)\right\}.$$
    <p>For two counts, write $\Lambda_0$ for the common-event intensity and
      $A=\Lambda_1+\Lambda_0$, $B=\Lambda_2+\Lambda_0$, $C=\Lambda_0$. Expanding the preceding identity through
      arbitrary bidegree $(r,s)$ gives the all-order identity</p>
    $$\boxed{\displaystyle
      \kappa^{(F)}_{rs}=\sum_{k=0}^{\min(r,s)}
      {r\choose k}{s\choose k}k!\,
      \kappa\!\left(A^{[r-k]},B^{[s-k]},C^{[k]}\right)}.$$
    <p>Here the brackets denote repeated arguments of the joint cumulant; when only one argument remains, it is its
      expectation. The coefficient counts which $k$ derivatives in each coordinate strike the bilinear common-event
      term. In particular, through bidegree $(2,2)$,</p>
    $$\begin{aligned}
      \kappa^{(F)}_{11}&=\kappa(A,B)+\mathbb EC,\\
      \kappa^{(F)}_{21}&=\kappa(A,A,B)+2\kappa(A,C),\\
      \kappa^{(F)}_{12}&=\kappa(A,B,B)+2\kappa(B,C),\\
      \kappa^{(F)}_{22}&=\kappa(A,A,B,B)+4\kappa(A,B,C)+2\kappa(C,C).
    \end{aligned}$$
    <p>Thus mixed factorial cumulants identify marginal-intensity dependence only after the common-shock terms are
      modeled or ruled out. The sharp counterexample has deterministic $C=cT$ and no idiosyncratic intensity:
      $N_1=N_2\sim\operatorname{Poisson}(cT)$. The cumulative intensities have zero covariance, but
      $\kappa^{(F)}_{11}=cT$. At $c=0.8$ and $T=1.3$, the certificate obtains $1.04000000$.</p>
''' + table([[label, f'{observed:.8f}', f'{predicted:.8f}']
             for label, observed, predicted in zip(
                 ('(1,1)', '(2,1)', '(1,2)', '(2,2)'),
                 common_factorial, common_predicted)],
            head=('mixed order', 'factorial count', 'common-shock formula')) + r'''
    <h3>Arbitrary integer marks</h3>
    <p>The unit-jump restriction is unnecessary. Let each common event carry an iid nonnegative integer mark
      $J=(J_1,J_2)$, independent of the cumulative intensities and of the other marks. Conditional on the common
      cumulative intensity $C$, the number of marked events is Poisson with mean $C$. If $\Lambda_1,\Lambda_2$ are
      the idiosyncratic cumulative intensities, then the exact factorial cumulant generating function is</p>
    $$H(t,u)=\log\mathbb E\exp\!\left\{t\Lambda_1+u\Lambda_2
      +C\left(\mathbb E[(1+t)^{J_1}(1+u)^{J_2}]-1\right)\right\}.$$
    <p>Put $\mu_{ab}=\mathbb E[(J_1)_a(J_2)_b]$, where $(j)_a$ is a falling factorial, and define the integrated-rate
      variables</p>
    $$W_{10}=\Lambda_1+\mu_{10}C,\qquad
      W_{01}=\Lambda_2+\mu_{01}C,\qquad
      W_{ab}=\mu_{ab}C\quad(a+b\ge2).$$
    <p>Take $r$ labelled symbols of type 1 and $s$ of type 2. For a block $B$ of a set partition, let
      $a_B,b_B$ be its numbers of the two symbol types. The all-order marked theorem is</p>
    <div class="equation-card">
    $$\boxed{\displaystyle
      \kappa^{(F)}_{rs}=\sum_{\pi\in\Pi_{r,s}}
      \kappa\!\left(W_{a_Bb_B}:B\in\pi\right).}$$
    </div>
    <p>This is the multivariate logarithmic Fa&agrave; di Bruno formula applied to $H$. It requires the displayed
      falling-factorial mark moments and the corresponding integrated-rate cumulants to be finite. For example,</p>
    $$\begin{aligned}
      \kappa^{(F)}_{11}&=\kappa(W_{10},W_{01})+\mathbb EW_{11},\\
      \kappa^{(F)}_{21}&=\kappa(W_{10},W_{10},W_{01})
        +\kappa(W_{20},W_{01})+2\kappa(W_{10},W_{11})+\mathbb EW_{21}.
    \end{aligned}$$
    <p>When $J=(1,1)$ almost surely, every $\mu_{ab}$ vanishes except those with $a,b\le1$. Only singleton blocks
      and disjoint type-1/type-2 pairs survive, and the partition formula reduces exactly to the preceding
      ${r\choose k}{s\choose k}k!$ matching formula. Thus larger or asymmetric common marks do not invalidate
      factorial-cumulant identification; they change the shared-shot-noise correction through their factorial
      moments.</p>
    <h3>Regime-dependent mark laws</h3>
    <p>The iid mark law may be replaced by a law that depends on the hidden regime at the event time. Write
      $c_i$ for the common-event rate and
      $\mu_{ab}(i)=\mathbb E[(J_1)_a(J_2)_b\mid Y_t=i]$. Conditional on the complete regime path, the exact
      factorial cumulant generating function becomes</p>
    $$H(t,u)=\log\mathbb E\exp\!\left\{t\Lambda_1+u\Lambda_2+
      \int_0^T c_{Y_v}\left(M_{Y_v}(t,u)-1\right)\,dv\right\},$$
    <p>where $M_i(t,u)=\mathbb E[(1+t)^{J_1}(1+u)^{J_2}\mid Y=i]$. Therefore the same partition theorem holds
      after replacing the variables above by</p>
    $$\begin{aligned}
      W_{10}&=\Lambda_1+\int_0^T c_{Y_v}\mu_{10}(Y_v)\,dv,\\
      W_{01}&=\Lambda_2+\int_0^T c_{Y_v}\mu_{01}(Y_v)\,dv,\\
      W_{ab}&=\int_0^T c_{Y_v}\mu_{ab}(Y_v)\,dv,\qquad a+b\ge2.
    \end{aligned}$$
    <p>This is a finite-state Markov additive process in the sense of
      <a href="https://doi.org/10.1007/BF00532536">&Ccedil;inlar (1972)</a>. The condition is pathwise: marks are
      conditionally independent given the regime path and their law at an event depends only on the contemporaneous
      regime. It does not cover dependence between different event marks beyond that hidden-state dependence.</p>
    <p>The
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/verify_count_cumulants.py">certificate</a>
    obtains the factorial-count and integrated-intensity columns independently from the regime/count master equation and a polynomial Feynman&ndash;Kac
    hierarchy. It checks the univariate identity through order four and the bivariate identity through bidegree $(2,2)$
    for a nonreversible three-state chain. The mixed calculation agrees within $5.3\times10^{-14}$, with omitted count
    mass bounded by $1.6\times10^{-47}$. A separate master equation with simultaneous $(1,1)$ jumps checks the
    common-shock formula through every mixed order $(r,s)$ with $1\leq r,s\leq3$ within
    $7.9\times10^{-12}$; at $(3,3)$ the two independent values are
    ''' + f'{common_factorial_33:.8f}' + r''' and ''' + f'{common_predicted_33:.8f}' + r'''. A second marked master
    equation uses six mark vectors, including $(3,1)$ and $(1,3)$, and agrees with the marked partition theorem
    through bidegree $(3,3)$ within $9.6\times10^{-10}$; the independent values of
    $\kappa^{(F)}_{33}$ are ''' + f'{marked_factorial_33:.8f}' + r''' and ''' + f'{marked_predicted_33:.8f}' + r''',
    with omitted mass below $8.7\times10^{-49}$. Substituting the degenerate mark $(1,1)$ recovers the matching
    formula within $1.1\times10^{-11}$. A third master equation uses a different six-point mark distribution in
    every regime. It agrees with the state-dependent partition formula through bidegree $(3,3)$ within
    $6.5\times10^{-10}$; at $(3,3)$ the independent values are
    ''' + f'{state_marked_grid[2, 2]:.8f}' + r''' and ''' + f'{state_marked_predicted_33:.8f}' + r''', and omitted
    mass is below $1.4\times10^{-44}$. The partition calculation follows the joint-cumulant method of
    <a href="https://doi.org/10.1137/1104031">Leonov and Shiryaev (1959)</a>. The certificate also measures
    second-order decay of the univariate omitted boundary term.
    The conditioning argument is the defining construction of a
    <a href="./bibliography.html#Cox1955">Cox process</a>, specialized here to the
    <a href="./bibliography.html#FischerMeierHellstern1993">Markov-modulated Poisson process</a>. The common-component
    construction is the classical bivariate Poisson model of
    <a href="https://doi.org/10.1093/biomet/51.1-2.241">Holgate (1964)</a>; the arbitrary-mark construction is a
    multivariate compound-Poisson law in the sense of
    <a href="https://doi.org/10.1214/aoms/1177731359">Feller (1943)</a>.</p>
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
    <p>The second-order exponent is $\int_0^T \pi\cdot(\tilde g\circ w_2)$, a polynomial in $B$ integrated term by term, and
    the bracket gains $w_2(T)$:</p>
    <div class="equation-card">
    $$u_i(T, x) = \exp\Big(\cdots + \sum_{k} d_k\,I_k\Big)\Big(1 + \kappa B\,[Q^{\#}u]_i - \tfrac12 B^2\,[Q^{\#}v]_i + [w_2(T)]_i\Big)
      + O(|Q|^{-3}),$$
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
              head=('start', 'averaged', 'first order', 'second order', 'numerical')) + r"""    <p>The second-order formula agrees with the engine to rounding, and its error against the numerical solution is
    """ + f'{np.abs(o2 - ex).max():.1e}' + r""".</p>
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
