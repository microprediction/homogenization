"""Second half of the model pages (run through model_pages.py): Hull-White, G2++, equity with stochastic rates, CEV
and the instruments. As there, every closed form is checked against the numerical solution before it is printed."""
import cmath
import math
from math import comb
import numpy as np
from fastswitch import ExpSum, numerical_a_callable
from pages_examples import table, e, write, sym, two_state_expansion, error_rows
from model_pages import cz, cg, half, lewis, SECOND_ORDER, U0

SRC = 'https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/model_pages2.py'


def Is(k, T):
    """E = e^{-kT}, B = (1 - E)/k and I_n = int_0^T B^n for n = 1..4."""
    E = math.exp(-k * T)
    I1 = (T - (1 - E) / k) / k
    I2 = (T - 2 * (1 - E) / k + (1 - E ** 2) / (2 * k)) / k ** 2
    I3 = (T - 3 * (1 - E) / k + 3 * (1 - E ** 2) / (2 * k) - (1 - E ** 3) / (3 * k)) / k ** 3
    I4 = (T - 4 * (1 - E) / k + 3 * (1 - E ** 2) / k - 4 * (1 - E ** 3) / (3 * k) + (1 - E ** 4) / (4 * k)) / k ** 4
    return E, (1 - E) / k, I1, I2, I3, I4


def g10(x, p=6):
    return f'{x:.{p}g}'.replace('e-0', 'e-').replace('-', '&minus;')


I_LINES = r'''    B &= \frac{1 - E}{a}, \qquad E = e^{-aT}, \\
    I_1 &= \frac{1}{a}\Big(T - \frac{1 - E}{a}\Big), \\
    I_2 &= \frac{1}{a^2}\Big(T - \frac{2(1 - E)}{a} + \frac{1 - E^2}{2a}\Big), \\
    I_3 &= \frac{1}{a^3}\Big(T - \frac{3(1 - E)}{a} + \frac{3(1 - E^2)}{2a} - \frac{1 - E^3}{3a}\Big), \\
    I_4 &= \frac{1}{a^4}\Big(T - \frac{4(1 - E)}{a} + \frac{3(1 - E^2)}{a} - \frac{4(1 - E^3)}{3a} + \frac{1 - E^4}{4a}\Big)'''


# ====================================================================================== Hull-White
def hull_white_page():
    a, sig, r, T = 0.3, [0.02, 0.008], 0.03, 5.0
    sb, st = half([s * s for s in sig])

    def gfuncs():
        return [lambda t, s=s: 0.5 * s * s * ((1 - math.exp(-a * t)) / a) ** 2 for s in sig]

    def ratio_num(lam, T, reg):
        E, B, I1, I2, I3, I4 = Is(a, T)
        return numerical_a_callable(T, sym(lam), gfuncs(), rtol=1e-13)[reg].real * math.exp(-0.5 * sb * I2)

    def ratio(lam, T, sign, order):
        eps = 1 / lam
        E, B, I1, I2, I3, I4 = Is(a, T)
        if order == 0:
            return 1.0
        if order == 1:
            return math.exp(eps / 8 * st * st * I4) * (1 + sign * eps / 4 * st * B * B)
        return math.exp(eps / 8 * st * st * I4 - eps * eps / 32 * st * st * B ** 4) * (1 + sign * eps / 4 * st * B * B - sign * eps * eps / 4 * st * B * E)
    lam = 5.0; eps = 1 / lam
    E, B, I1, I2, I3, I4 = Is(a, T)
    ref = ratio_num(lam, T, 0)
    assert abs(ratio(lam, T, 1, 2) - ref) < 0.05 * abs(ratio(lam, T, 1, 1) - ref) < 0.002 * abs(1 - ref)
    q_rows = [['$E$, $B$', f'{E:.6f}, &nbsp; {B:.6f}'], ['$I_2$, $I_4$', f'{I2:.5f}, &nbsp; {I4:.5f}'],
              [r'$\bar s$, $\tilde s$', f'{g10(sb)}, &nbsp; {g10(st)}'],
              [r'Green&ndash;Kubo exponent $\frac\varepsilon8\tilde s^2I_4$', g10(eps / 8 * st * st * I4, 4)],
              [r'second-order exponent $-\frac{\varepsilon^2}{32}\tilde s^2B^4$', g10(-eps * eps / 32 * st * st * B ** 4, 4)],
              [r'memory, first order $\frac\varepsilon4\tilde sB^2$', g10(eps / 4 * st * B * B, 4)],
              [r'memory, second order $-\frac{\varepsilon^2}4\tilde sBE$', g10(-eps * eps / 4 * st * B * E, 4)]]
    o_rows = [[lab, f'{ratio(lam, T, 1, o):.10f}', e(abs(ratio(lam, T, 1, o) - ref))] for o, lab in ((0, 'averaged model'), (1, 'first order'), (2, 'second order'))]
    o_rows.append(['numerical solution', f'{ref:.10f}', ''])
    l_rows = []
    for lam_ in (2.0, 5.0, 10.0, 20.0):
        rf = ratio_num(lam_, T, 0)
        l_rows.append([f'{lam_:g}', f'{rf:.10f}'] + [e(abs(ratio(lam_, T, 1, o) - rf)) for o in (0, 1, 2)])
    y_rows = []
    for TT in (1.0, 2.0, 5.0, 10.0, 30.0):
        row = [f'{TT:g}']
        for reg, sign in ((0, 1), (1, -1)):
            row += [f'{-math.log(ratio_num(lam, TT, reg)) / TT * 1e4:.4f}'.replace('-', '&minus;'),
                    f'{-math.log(ratio(lam, TT, sign, 2)) / TT * 1e4:.4f}'.replace('-', '&minus;')]
        y_rows.append(row)

    def mk(lam_):
        Bx = ExpSum({0: 1 / a, a: -1 / a})
        return sym(lam_), [(Bx * Bx).scale(0.5 * s * s) for s in sig], gfuncs()
    e_rows = error_rows(mk, T, (2.0, 5.0, 10.0, 20.0))

    body = r'''    <h1>Hull&ndash;White with a switching volatility</h1>
    <p class="subtitle">A short rate fitted to today&apos;s curve whose volatility follows a hidden regime, and how far the regime moves the curve.</p>

    <h2>The model</h2>
    <p>In the <a href="./bibliography.html#HullWhite1990">Hull&ndash;White model</a> the short rate is a Gaussian
    factor plus a deterministic shift chosen to reproduce today&apos;s discount curve. Here the factor&apos;s volatility
    is set by a hidden regime $y_t \in \{1, 2\}$:</p>
    $$r_t = x_t + \varphi(t), \qquad dx_t = -a\,x_t\,dt + \sigma_{y_t}\,dW_t, \qquad x_0 = 0 .$$
    <p>The regime switches at rate $\lambda$ in each direction. The factor reverts over $1/a$ and the regime changes
    every $1/\lambda$. The market supplies the discount curve $P^M(0,T)$ and its instantaneous forward rate
    $f^M(0,t)$.</p>
    <p class="muted">Parameters: $a = 0.3$, $\sigma = (0.02, 0.008)$, a flat curve at 3%, and bonds of maturity
    $T = 5$ unless stated.</p>

    <h2>The quantity</h2>
    <p>The zero-coupon bond seen from each starting regime,</p>
    $$P_i(0,T) = \mathbb{E}\big[e^{-\int_0^T r_s\,ds} \mid y_0 = i\big]
      = e^{-\int_0^T\varphi}\;\mathbb{E}\big[e^{-\int_0^T x_s\,ds} \mid x_0 = 0,\ y_0 = i\big],$$
    <p>and its ratio to the market bond $P^M(0,T)$.</p>

    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is Hull&ndash;White with the average variance
    $\bar\sigma^2 = \tfrac12(\sigma_1^2 + \sigma_2^2)$. That model reproduces the curve when the shift is</p>
    $$\varphi(t) = f^M(0,t) + \tfrac12\bar\sigma^2\,B(t)^2, \qquad B(t) = \frac{1 - e^{-at}}{a},$$
    <p>and this is the shift used throughout: the averaged model fits the curve exactly, and the switching then moves
    each regime&apos;s bond away from it by an amount the expansion computes.</p>

    <h2>Reduction to a linear system</h2>
    <p>By the Feynman&ndash;Kac formula the functions
    $u_i(t,x) = \mathbb{E}[e^{-\int_0^t x_s\,ds} \mid x_0 = x,\ y_0 = i]$ solve the coupled equations</p>
    $$\partial_t u_i = -a\,x\,\partial_x u_i + \tfrac12\sigma_i^2\,\partial_{xx}u_i - x\,u_i + \sum_j Q_{ij}\,u_j, \qquad u_i(0,x) = 1.$$
    <p>Try the Vasicek form $u_i = e^{-B(t)x}a_i(t)$. Then $\partial_x u_i = -B\,u_i$ and $\partial_{xx}u_i = B^2u_i$, and
    after dividing by $e^{-Bx}$ the terms proportional to $x$ are</p>
    $$-B'\,x = a B\,x - x .$$
    <p>They cancel when $B' = 1 - aB$, that is $B(t) = (1 - e^{-at})/a$, the same $B$ as in the shift. The reversion
    speed does not switch, so one $B$ works for both regimes. What remains involves $t$ only:
    $a' = (Q + \operatorname{diag} g)a$, $a(0) = \mathbf 1$ and</p>
    $$g_i(t) = \tfrac12\sigma_i^2\,B(t)^2 .$$
    <p>The factor starts at $x_0 = 0$, so the prefactor $e^{-Bx_0}$ is one. Integrating the shift,
    $\int_0^T\varphi = -\log P^M(0,T) + \tfrac12\bar\sigma^2\int_0^TB^2$, and the bond is</p>
    $$P_i(0,T) = P^M(0,T)\;e^{-\frac12\bar\sigma^2\int_0^TB^2}\;a_i(T) .$$
    <p>The middle factor is $e^{-\int\bar g}$: it cancels the averaged part of $a_i$, which is how the averaged model
    returns the curve. Everything left over is the effect of the switching.</p>
'''
    body += two_state_expansion(r'$\tilde g = \tfrac12\,\tilde s\,B^2, \qquad \tilde s = \tfrac12(\sigma_1^2 - \sigma_2^2)$')
    body += r'''
    <h2>The bond in closed form</h2>
''' + SECOND_ORDER + r'''
    <h3>Step 1: the forcing</h3>
    <p>With $\bar s, \tilde s = \tfrac12(\sigma_1^2 \pm \sigma_2^2)$,</p>
    $$\bar g = \tfrac12\bar s\,B^2, \qquad \tilde g = \tfrac12\tilde s\,B^2, \qquad \tilde g(0) = 0, \qquad
      \tilde g' = \tilde s\,B\,B' = \tilde s\,B\,e^{-at} .$$

    <h3>Step 2: the integrals</h3>
    <p>Powers of $B$ are sums of exponentials, so with $I_n = \int_0^TB^n$,</p>
    $$\int_0^T\bar g = \tfrac12\bar s\,I_2, \qquad \int_0^T\tilde g^{\,2} = \tfrac14\tilde s^2\,I_4 .$$
    <p>The first is cancelled by the shift. The second is the Green&ndash;Kubo term.</p>

    <h3>Step 3: the result</h3>
    <div class="equation-card">
    $$\frac{P_{1,2}(0,T)}{P^M(0,T)} = \exp\Big(\frac{\varepsilon}{8}\,\tilde s^2\,I_4 - \frac{\varepsilon^2}{32}\,\tilde s^2\,B^4\Big)
      \Big(1 \pm \frac{\varepsilon}{4}\,\tilde s\,B^2 \mp \frac{\varepsilon^2}{4}\,\tilde s\,B\,E\Big) + O(\varepsilon^3),$$
    $$\begin{aligned}
''' + I_LINES + r''' .
    \end{aligned}$$
    </div>
    <p>The upper sign is for a start in regime 1, the volatile one. Three things can be read from the card.</p>
    <ul>
      <li>At order zero the ratio is one: every regime sees the market curve.</li>
      <li>At first order the two regimes move in opposite directions by $\tfrac14\varepsilon\tilde sB^2$. A start in the
        volatile regime means more variance in $\int r$ over the next $1/(2\lambda)$, more convexity, and a higher bond
        price.</li>
      <li>The Green&ndash;Kubo term raises both bonds, by $\tfrac18\varepsilon\tilde s^2I_4$. It is quadratic in the small
        number $\tilde s$, and at these parameters it is a thousand times smaller than the memory term.</li>
    </ul>
    <p>At $\lambda = 5$ and $T = 5$:</p>
''' + table(['quantity', 'value'], q_rows) + table(['bond ratio, regime 1', 'value', 'error'], o_rows) + r'''
    <h2>Fitting the curve from the starting regime</h2>
    <p>The shift above fits the averaged model. If the starting regime $i$ is known, the fit can be made exact for
    that regime instead, because $a_i$ is known: choose</p>
    $$\varphi_i(t) = f^M(0,t) + \frac{d}{dt}\log a_i(t), \qquad\text{so that}\qquad e^{-\int_0^T\varphi_i}\,a_i(T) = P^M(0,T) \text{ for every } T.$$
    <p>To first order this is the averaged shift plus two corrections,</p>
    $$\varphi_{1,2}(t) = f^M(0,t) + \tfrac12\bar s\,B^2 + \frac{\varepsilon}{8}\,\tilde s^2B^4 \pm \frac{\varepsilon}{2}\,\tilde s\,B\,e^{-at} + O(\varepsilon^2),$$
    <p>the derivative of the exponent and of the bracket in the card. <a href="https://github.com/microprediction/regimelib">regimelib</a>&apos;s
    <code>SwitchingHullWhite</code> uses the averaged shift, so that the model is QuantLib&apos;s when the regimes coincide.</p>

    <h2>Results</h2>
    <p>The ratio of the regime-1 bond to the market bond at $T = 5$, and the error of each order of the card:</p>
''' + table(['switching rate', 'numerical', 'order 0', 'order 1', 'order 2'], l_rows) + r'''    <p>The error in $a_1(5)$ after each order of the <a href="./engine.html">engine</a>, against a numerical solution:</p>
''' + table(['switching rate'] + [f'order {o}' for o in range(7)], e_rows) + r'''    <p>In yield terms the effect is small. The shift of the zero rate away from the market curve at $\lambda = 5$, in
    basis points, from the numerical solution and from the card:</p>
''' + table(['maturity', 'regime 1, numerical', 'regime 1, card', 'regime 2, numerical', 'regime 2, card'], y_rows) + r'''    <p>The shift peaks near five years and then falls, because the memory of the starting regime is a fixed amount
    of extra variance spread over a longer bond. Options are a different matter: a bond option depends on the
    volatility over its life, and there the regimes differ at leading order, as on the
    <a href="./bond-options.html">bond options</a> page.</p>
    <p>Certificate: <a href="''' + SRC + r'''">model_pages2.py</a>.</p>
'''
    write('hull-white.html', 'Hull-White with a switching volatility', body)


# ====================================================================================== G2++
def g2_page():
    a, b, sig, eta, rho, T = 0.5, 0.1, [0.015, 0.006], [0.012, 0.008], [-0.4, -0.7], 5.0
    s = [x * x for x in sig]; h = [x * x for x in eta]; c = [rho[i] * sig[i] * eta[i] for i in range(2)]
    (sb, st), (hb, ht), (cb, ct) = half(s), half(h), half(c)

    def M(p, q, T):
        Phi = lambda k: T if k == 0 else (1 - math.exp(-k * T)) / k
        return a ** -p * b ** -q * sum((-1) ** (j + k) * comb(p, j) * comb(q, k) * Phi(j * a + k * b) for j in range(p + 1) for k in range(q + 1))

    def gfuncs():
        Ba = lambda t: (1 - math.exp(-a * t)) / a
        Bb = lambda t: (1 - math.exp(-b * t)) / b
        return [lambda t, i=i: 0.5 * s[i] * Ba(t) ** 2 + 0.5 * h[i] * Bb(t) ** 2 + c[i] * Ba(t) * Bb(t) for i in range(2)]

    def pieces(T):
        Ea, Eb = math.exp(-a * T), math.exp(-b * T)
        Ba, Bb = (1 - Ea) / a, (1 - Eb) / b
        A = 0.25 * st * st * M(4, 0, T) + 0.25 * ht * ht * M(0, 4, T) + (ct * ct + 0.5 * st * ht) * M(2, 2, T) + st * ct * M(3, 1, T) + ht * ct * M(1, 3, T)
        G = 0.5 * st * Ba * Ba + 0.5 * ht * Bb * Bb + ct * Ba * Bb
        Gp = st * Ba * Ea + ht * Bb * Eb + ct * (Ea * Bb + Ba * Eb)
        avg = 0.5 * sb * M(2, 0, T) + 0.5 * hb * M(0, 2, T) + cb * M(1, 1, T)
        return Ea, Eb, Ba, Bb, A, G, Gp, avg

    def ratio_num(lam, reg):
        return numerical_a_callable(T, sym(lam), gfuncs(), rtol=1e-13)[reg].real * math.exp(-pieces(T)[7])

    def ratio(lam, sign, order):
        eps = 1 / lam
        Ea, Eb, Ba, Bb, A, G, Gp, avg = pieces(T)
        if order == 0:
            return 1.0
        if order == 1:
            return math.exp(eps / 2 * A) * (1 + sign * eps / 2 * G)
        return math.exp(eps / 2 * A - eps * eps / 8 * G * G) * (1 + sign * eps / 2 * G - sign * eps * eps / 4 * Gp)
    lam = 5.0; eps = 1 / lam
    Ea, Eb, Ba, Bb, A, G, Gp, avg = pieces(T)
    ref = ratio_num(lam, 0)
    assert abs(ratio(lam, 1, 2) - ref) < 0.05 * abs(ratio(lam, 1, 1) - ref) < 0.002 * abs(1 - ref)
    q_rows = [['$E_a$, $E_b$', f'{Ea:.6f}, &nbsp; {Eb:.6f}'], ['$B_a$, $B_b$', f'{Ba:.6f}, &nbsp; {Bb:.6f}'],
              ['$M_{40}$, $M_{31}$, $M_{22}$, $M_{13}$, $M_{04}$', ', &nbsp; '.join(f'{M(p, q, T):.4f}' for p, q in ((4, 0), (3, 1), (2, 2), (1, 3), (0, 4)))],
              [r'$\tilde s$, $\tilde h$, $\tilde c$', ', &nbsp; '.join(g10(x, 4) for x in (st, ht, ct))],
              ['$A$', g10(A, 5)], ["$G$, $G'$", f'{g10(G, 5)}, &nbsp; {g10(Gp, 5)}'],
              [r'Green&ndash;Kubo exponent $\frac\varepsilon2A$', g10(eps / 2 * A, 4)],
              [r'second-order exponent $-\frac{\varepsilon^2}8G^2$', g10(-eps * eps / 8 * G * G, 4)],
              [r'memory, first order $\frac\varepsilon2G$', g10(eps / 2 * G, 4)],
              [r"memory, second order $-\frac{\varepsilon^2}4G'$", g10(-eps * eps / 4 * Gp, 4)]]
    o_rows = [[lab, f'{ratio(lam, 1, o):.10f}', e(abs(ratio(lam, 1, o) - ref))] for o, lab in ((0, 'averaged model'), (1, 'first order'), (2, 'second order'))]
    o_rows.append(['numerical solution', f'{ref:.10f}', ''])
    l_rows = []
    for lam_ in (2.0, 5.0, 10.0, 20.0):
        rf = ratio_num(lam_, 0)
        l_rows.append([f'{lam_:g}', f'{rf:.10f}'] + [e(abs(ratio(lam_, 1, o) - rf)) for o in (0, 1, 2)])

    def mk(lam_):
        Bx, By = ExpSum({0: 1 / a, a: -1 / a}), ExpSum({0: 1 / b, b: -1 / b})
        g = [(Bx * Bx).scale(0.5 * s[i]) + (By * By).scale(0.5 * h[i]) + (Bx * By).scale(c[i]) for i in range(2)]
        return sym(lam_), g, gfuncs()
    e_rows = error_rows(mk, T, (2.0, 5.0, 10.0, 20.0))

    body = r'''    <h1>G2++ with switching volatilities and correlation</h1>
    <p class="subtitle">Two Gaussian factors fitted to today&apos;s curve, with both volatilities and their correlation set by a hidden regime.</p>

    <h2>The model</h2>
    <p>In the two-factor Gaussian model <a href="./bibliography.html#BrigoMercurio2006">G2++</a> the short rate is the
    sum of two correlated mean-reverting factors and a deterministic shift. Here the two volatilities and the
    correlation are set by a hidden regime $y_t \in \{1, 2\}$:</p>
    $$r_t = x_t + z_t + \varphi(t), \qquad dx_t = -a\,x_t\,dt + \sigma_{y_t}\,dW^1_t, \qquad dz_t = -b\,z_t\,dt + \eta_{y_t}\,dW^2_t,
      \qquad d\langle W^1, W^2\rangle = \rho_{y_t}\,dt,$$
    <p>with $x_0 = z_0 = 0$. The regime switches at rate $\lambda$ in each direction. The factors revert over $1/a$
    and $1/b$; the regime changes every $1/\lambda$.</p>
    <p class="muted">Parameters: $a = 0.5$, $b = 0.1$, $\sigma = (0.015, 0.006)$, $\eta = (0.012, 0.008)$,
    $\rho = (-0.4, -0.7)$, a flat curve at 3%, and bonds of maturity $T = 5$.</p>

    <h2>The quantity</h2>
    <p>The zero-coupon bond seen from each starting regime,</p>
    $$P_i(0,T) = e^{-\int_0^T\varphi}\;\mathbb{E}\big[e^{-\int_0^T (x_s + z_s)\,ds} \mid x_0 = z_0 = 0,\ y_0 = i\big],$$
    <p>and its ratio to the market bond $P^M(0,T)$.</p>

    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is G2++ with the averaged covariance matrix: variances
    $\bar s = \tfrac12(\sigma_1^2 + \sigma_2^2)$ and $\bar h = \tfrac12(\eta_1^2 + \eta_2^2)$, and covariance
    $\bar c = \tfrac12(\rho_1\sigma_1\eta_1 + \rho_2\sigma_2\eta_2)$. The average is taken of the covariance
    $c_i = \rho_i\sigma_i\eta_i$, not of the correlation. That model reproduces the curve when</p>
    $$\varphi(t) = f^M(0,t) + \tfrac12\bar s\,B_a(t)^2 + \tfrac12\bar h\,B_b(t)^2 + \bar c\,B_a(t)B_b(t),
      \qquad B_a(t) = \frac{1 - e^{-at}}{a}, \quad B_b(t) = \frac{1 - e^{-bt}}{b} .$$

    <h2>Reduction to a linear system</h2>
    <p>The functions $u_i(t,x,z) = \mathbb{E}[e^{-\int_0^t (x_s + z_s)ds} \mid x_0 = x, z_0 = z, y_0 = i]$ solve</p>
    $$\begin{aligned}\partial_t u_i = \;&-a\,x\,\partial_x u_i - b\,z\,\partial_z u_i + \tfrac12\sigma_i^2\,\partial_{xx}u_i
      + \tfrac12\eta_i^2\,\partial_{zz}u_i + \rho_i\sigma_i\eta_i\,\partial_{xz}u_i \\
      &- (x + z)\,u_i + \sum_j Q_{ij}\,u_j, \qquad u_i(0,x,z) = 1.\end{aligned}$$
    <p>Try $u_i = e^{-B_a(t)x - B_b(t)z}a_i(t)$. The terms proportional to $x$ and to $z$ cancel separately when
    $B_a' = 1 - aB_a$ and $B_b' = 1 - bB_b$, which give the two functions above. Neither reversion speed switches, so
    they serve both regimes. What remains involves $t$ only: $a' = (Q + \operatorname{diag} g)a$, $a(0) = \mathbf 1$
    and</p>
    $$g_i(t) = \tfrac12\sigma_i^2\,B_a^2 + \tfrac12\eta_i^2\,B_b^2 + \rho_i\sigma_i\eta_i\,B_aB_b .$$
    <p>This is half the variance rate of $\int r$ in regime $i$. As for
    <a href="./hull-white.html">Hull&ndash;White</a>, the shift cancels the averaged part:</p>
    $$P_i(0,T) = P^M(0,T)\;e^{-\int_0^T\bar g}\;a_i(T) .$$
'''
    body += two_state_expansion(r'$\tilde g = \tfrac12\tilde s\,B_a^2 + \tfrac12\tilde h\,B_b^2 + \tilde c\,B_aB_b$')
    body += r'''
    <h2>The bond in closed form</h2>
''' + SECOND_ORDER + r'''
    <h3>Step 1: the forcing</h3>
    <p>The three switched quantities are the two variances and the covariance. Write their half-differences</p>
    $$\tilde s = \tfrac12(\sigma_1^2 - \sigma_2^2), \qquad \tilde h = \tfrac12(\eta_1^2 - \eta_2^2), \qquad
      \tilde c = \tfrac12(\rho_1\sigma_1\eta_1 - \rho_2\sigma_2\eta_2),$$
    <p>so that $\tilde g = \tfrac12\tilde s\,B_a^2 + \tfrac12\tilde h\,B_b^2 + \tilde c\,B_aB_b$, with $\tilde g(0) = 0$ and,
    since $B_a' = e^{-at}$ and $B_b' = e^{-bt}$,</p>
    $$\tilde g' = \tilde s\,B_a\,e^{-at} + \tilde h\,B_b\,e^{-bt} + \tilde c\,\big(e^{-at}B_b + B_a\,e^{-bt}\big).$$

    <h3>Step 2: the integrals</h3>
    <p>The square of $\tilde g$ is a quartic in $B_a$ and $B_b$, so it needs the integrals</p>
    $$M_{pq} = \int_0^T B_a^p\,B_b^q\,dt .$$
    <p>Expanding both powers by the binomial theorem gives a sum of exponentials, each integrated at once:</p>
    $$M_{pq} = \frac{1}{a^p\,b^q}\sum_{j=0}^{p}\sum_{k=0}^{q}(-1)^{j+k}\binom pj\binom qk\,\Phi(ja + kb), \qquad
      \Phi(0) = T, \quad \Phi(\kappa) = \frac{1 - e^{-\kappa T}}{\kappa} .$$
    <p>Collecting the six products in $\tilde g^{\,2}$,</p>
    $$\int_0^T\tilde g^{\,2} = \tfrac14\tilde s^2M_{40} + \tfrac14\tilde h^2M_{04} + \big(\tilde c^2 + \tfrac12\tilde s\tilde h\big)M_{22}
      + \tilde s\,\tilde c\,M_{31} + \tilde h\,\tilde c\,M_{13} .$$

    <h3>Step 3: the result</h3>
    <div class="equation-card">
    $$\frac{P_{1,2}(0,T)}{P^M(0,T)} = \exp\Big(\frac{\varepsilon}{2}\,A - \frac{\varepsilon^2}{8}\,G^2\Big)
      \Big(1 \pm \frac{\varepsilon}{2}\,G \mp \frac{\varepsilon^2}{4}\,G'\Big) + O(\varepsilon^3),$$
    $$\begin{aligned}
    A &= \tfrac14\tilde s^2M_{40} + \tfrac14\tilde h^2M_{04} + \big(\tilde c^2 + \tfrac12\tilde s\tilde h\big)M_{22}
      + \tilde s\,\tilde c\,M_{31} + \tilde h\,\tilde c\,M_{13}, \\
    G &= \tfrac12\tilde s\,B_a^2 + \tfrac12\tilde h\,B_b^2 + \tilde c\,B_aB_b, \\
    G' &= \tilde s\,B_aE_a + \tilde h\,B_bE_b + \tilde c\,\big(E_aB_b + B_aE_b\big), \\
    B_a &= \frac{1 - E_a}{a}, \quad E_a = e^{-aT}, \qquad B_b = \frac{1 - E_b}{b}, \quad E_b = e^{-bT}, \\
    M_{pq} &= \frac{1}{a^p\,b^q}\sum_{j=0}^{p}\sum_{k=0}^{q}(-1)^{j+k}\binom pj\binom qk\,\Phi(ja + kb), \qquad
      \Phi(0) = T, \quad \Phi(\kappa) = \frac{1 - e^{-\kappa T}}{\kappa} .
    \end{aligned}$$
    </div>
    <p>The upper sign is for a start in regime 1. With $\tilde h = \tilde c = 0$ the card is the
    <a href="./hull-white.html">Hull&ndash;White</a> one, since $M_{40} = I_4$.</p>
    <p>The memory term $G$ is the half-difference of the variance of $\int r$ between the regimes. With negative
    correlation its covariance part $\tilde c\,B_aB_b$ works against the two variance parts, and for some parameters
    $G$ passes through zero at a particular maturity. There the regimes price that bond alike at first order, and the
    second-order term $G'$ leads. This is the bond-price face of the effect on the
    <a href="./correlation.html">switching correlation</a> page.</p>
    <p>At $\lambda = 5$ and $T = 5$:</p>
''' + table(['quantity', 'value'], q_rows) + table(['bond ratio, regime 1', 'value', 'error'], o_rows) + r'''
    <h2>Results</h2>
    <p>The ratio of the regime-1 bond to the market bond at $T = 5$, and the error of each order of the card:</p>
''' + table(['switching rate', 'numerical', 'order 0', 'order 1', 'order 2'], l_rows) + r'''    <p>The error in $a_1(5)$ after each order of the <a href="./engine.html">engine</a>, against a numerical solution:</p>
''' + table(['switching rate'] + [f'order {o}' for o in range(7)], e_rows) + r'''    <p>Options on bonds use the same forcing with a terminal exponent in each factor, and the single variable
    $B_ax_T + B_bz_T$ in place of the short rate; see the <a href="./instruments.html">instruments</a> page.</p>
    <p>Certificate: <a href="''' + SRC + r'''">model_pages2.py</a>. In
    <a href="https://github.com/microprediction/regimelib">regimelib</a> this model is <code>SwitchingG2</code>.</p>
'''
    write('g2.html', 'G2++ with switching volatilities and correlation', body)


# ====================================================================================== equity with stochastic rates
def equity_rates_page():
    aR, bR, sR, r0 = 0.5, [0.05, 0.02], [0.015, 0.008], 0.03
    sS, rho, S0, q, T = [0.30, 0.15], -0.3, 100.0, 0.0, 1.0
    E, B, I1, I2, I3, I4 = Is(aR, T)
    (bb, bt), (srb, srt), (ssb, sst), (pb, pt) = half(bR), half([x * x for x in sR]), half([x * x for x in sS]), half([sS[i] * sR[i] for i in range(2)])

    def coef(w):
        c = 1 - 1j * w
        al = [-c * (aR * bR[i] + 1j * w * rho * sS[i] * sR[i]) for i in range(2)]
        be = [0.5 * c * c * sR[i] ** 2 for i in range(2)]
        ga = [-0.5 * sS[i] ** 2 * (w * w + 1j * w) for i in range(2)]
        return c, half(al), half(be), half(ga), al, be, ga

    def closed(w, lam, order, sign=1):
        eps = 1 / lam
        c, (alb, alt), (beb, bet), (gab, gat), *_ = coef(w)
        pre = S0 ** (1j * w) * cmath.exp(-1j * w * q * T - c * B * r0)
        avg = alb * I1 + beb * I2 + gab * T
        A = alt ** 2 * I2 + 2 * alt * bet * I3 + bet ** 2 * I4 + 2 * alt * gat * I1 + 2 * bet * gat * I2 + gat ** 2 * T
        G, Gp = alt * B + bet * B * B + gat, (alt + 2 * bet * B) * E
        if order == 0:
            return pre * cmath.exp(avg)
        if order == 1:
            return pre * cmath.exp(avg + eps / 2 * A) * (1 + sign * eps / 2 * G)
        return pre * cmath.exp(avg + eps / 2 * A - eps * eps / 8 * (G * G + gat * gat)) * (1 + sign * eps / 2 * G - sign * eps * eps / 4 * Gp)

    def numeric(w, lam, reg=0):
        c, _, _, _, al, be, ga = coef(w)
        Bf = lambda t: (1 - math.exp(-aR * t)) / aR
        gf = [lambda t, i=i: al[i] * Bf(t) + be[i] * Bf(t) ** 2 + ga[i] for i in range(2)]
        return S0 ** (1j * w) * cmath.exp(-1j * w * q * T - c * B * r0) * numerical_a_callable(T, sym(lam), gf, rtol=1e-12, is_complex=True)[reg]
    lam = 25.0; eps = 1 / lam
    c, (alb, alt), (beb, bet), (gab, gat), *_ = coef(U0)
    ref = numeric(U0, lam)
    assert abs(closed(U0, lam, 2) - ref) < 0.05 * abs(closed(U0, lam, 1) - ref) < 0.001 * abs(closed(U0, lam, 0) - ref)
    A = alt ** 2 * I2 + 2 * alt * bet * I3 + bet ** 2 * I4 + 2 * alt * gat * I1 + 2 * bet * gat * I2 + gat ** 2 * T
    G, Gp = alt * B + bet * B * B + gat, (alt + 2 * bet * B) * E
    q_rows = [['$c$', cz(c, 1)], ['$E$, $B$', f'{E:.6f}, &nbsp; {B:.6f}'],
              ['$I_1$, $I_2$, $I_3$, $I_4$', ', &nbsp; '.join(f'{x:.6f}' for x in (I1, I2, I3, I4))],
              [r'$\bar\alpha$, $\tilde\alpha$', cz(alb, 6) + ', &nbsp; ' + cz(alt, 6)],
              [r'$\bar\beta$, $\tilde\beta$', cg(beb) + ', &nbsp; ' + cg(bet)],
              [r'$\bar\gamma$, $\tilde\gamma$', cz(gab, 6) + ', &nbsp; ' + cz(gat, 6)],
              [r'averaged exponent $\bar\alpha I_1 + \bar\beta I_2 + \bar\gamma T$', cz(alb * I1 + beb * I2 + gab * T, 7)],
              ['$A$', cg(A)], ["$G$, $G'$", cz(G, 6) + ', &nbsp; ' + cz(Gp, 6)]]
    o_rows = [[lab, cz(closed(U0, lam, o), 6), e(abs(closed(U0, lam, o) - ref))] for o, lab in ((0, '0 (averaged model)'), (1, '1'), (2, '2'))]
    o_rows.append(['numerical solution', cz(ref, 6), ''])
    U = math.sqrt(80 / (ssb * T))

    def call(K, lam, order):
        psi = (lambda w: numeric(w, lam)) if order is None else (lambda w: closed(w, lam, order))
        return S0 * math.exp(-q * T) - math.sqrt(K) / math.pi * lewis(psi, -math.log(K), U)
    rows = []
    for lam_ in (25.0, 50.0, 100.0):
        for K in (90, 110):
            rf = call(K, lam_, None)
            rows.append([f'{lam_:g}', f'{K}', f'{rf:.6f}'] + [e(abs(call(K, lam_, o) - rf)) for o in (0, 1, 2)])
    print('hybrid', rows[0])
    assert abs(float(rows[0][2]) - 16.621952) < 2e-6          # regimelib's NumericalSwitchingEngine on SwitchingEquityRates
    bond = numeric(0.0, lam).real

    body = r'''    <h1>Equity options under stochastic rates</h1>
    <p class="subtitle">A stock and a short rate driven by one hidden regime, priced through a discounted characteristic function.</p>

    <h2>The model</h2>
    <p>A stock with <a href="./black-scholes.html">Black&ndash;Scholes</a> dynamics is discounted at a
    <a href="./regime-switching.html">Vasicek</a> short rate, and one hidden regime $y_t \in \{1, 2\}$ sets the
    parameters of both:</p>
    $$\frac{dS_t}{S_t} = (r_t - q)\,dt + \sigma_{y_t}\,dW^S_t, \qquad dr_t = a\,(b_{y_t} - r_t)\,dt + \eta_{y_t}\,dW^r_t,
      \qquad d\langle W^S, W^r\rangle = \rho\,dt .$$
    <p>The regime switches at rate $\lambda$ in each direction. Even with $\rho = 0$ the stock and the rate are
    dependent, because the same regime moves the equity volatility, the level the rate reverts to, and the rate
    volatility.</p>
    <p class="muted">Parameters: $\sigma = (0.30, 0.15)$, $a = 0.5$, $b = (0.05, 0.02)$, $\eta = (0.015, 0.008)$,
    $r_0 = 0.03$, $\rho = -0.3$, $q = 0$, $S_0 = 100$, $T = 1$, and a start in regime 1, where the stock is volatile
    and rates are heading up.</p>

    <h2>The quantity</h2>
    <p>With a stochastic rate the discount factor cannot be taken outside the expectation, so the object to compute is
    the discounted characteristic function</p>
    $$\Psi_i(w) = \mathbb{E}\big[e^{-\int_0^T r_s\,ds}\,S_T^{\,iw} \mid r_0,\ y_0 = i\big].$$
    <p>It contains the bond, $\Psi_i(0) = P_i(0,T)$, and the forward, $\Psi_i(-i) = S_0e^{-qT}$. A European call is
    <a href="./bibliography.html#Lewis2001">Lewis&apos;s formula</a> in discounted form,</p>
    $$C = S_0e^{-qT} - \frac{\sqrt K}{\pi}\int_0^\infty \operatorname{Re}\big[K^{-iu}\,\Psi(u - \tfrac i2)\big]\,\frac{du}{u^2 + 1/4}.$$
    <p>To see why, write the payoff as $S_T - \min(S_T, K)$. The first part is worth $S_0e^{-qT}$. For the second,
    $\min(S, K) = \frac{\sqrt K}{\pi}\int_0^\infty\operatorname{Re}\big[K^{-iu}S^{\,iu + 1/2}\big]\frac{du}{u^2 + 1/4}$,
    and taking the discounted expectation under the integral replaces $S^{\,i(u - i/2)}$ by $\Psi(u - \tfrac i2)$.</p>

    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is the Black&ndash;Scholes&ndash;Vasicek model with the
    averaged equity variance, rate level, rate variance and equity&ndash;rate covariance. Its discounted characteristic
    function is the exponential of a quadratic in $w$.</p>

    <h2>Reduction to a linear system</h2>
    <p>Solve the stock&apos;s equation:
    $S_T = S_0\exp\big(\int_0^T (r_s - q)\,ds + X_T\big)$ with
    $X_T = \int_0^T\sigma_{y_s}dW^S_s - \tfrac12\int_0^T\sigma_{y_s}^2ds$. Then</p>
    $$e^{-\int_0^T r}\,S_T^{\,iw} = S_0^{\,iw}\,e^{-iwqT}\,\exp\Big(-c\int_0^T r_s\,ds + iw\,X_T\Big), \qquad c = 1 - iw .$$
    <p>The rate is integrated with the complex weight $c$ in place of one. The function
    $f_i(t, r) = \mathbb{E}[\exp(-c\int_0^t r + iwX_t) \mid r_0 = r,\ y_0 = i]$ solves</p>
    $$\begin{aligned}\partial_t f_i = \;&a(b_i - r)\,\partial_r f_i + \tfrac12\eta_i^2\,\partial_{rr}f_i - c\,r\,f_i
      - \tfrac12\sigma_i^2(w^2 + iw)\,f_i \\ &+ iw\,\rho\,\sigma_i\eta_i\,\partial_r f_i + \sum_j Q_{ij}\,f_j, \qquad f_i(0, r) = 1 .\end{aligned}$$
    <p>The first three terms are Vasicek&apos;s with weight $c$. The fourth is the Black&ndash;Scholes exponent of
    $e^{iwX}$. The fifth is the cross term: a move in $W^S$ changes $e^{iwX}$ by the factor $iw\sigma_i\,dW^S$ and a
    correlated move in $W^r$ changes $r$, which gives $iw\rho\sigma_i\eta_i\,\partial_r$.</p>
    <p>Try $f_i = e^{-cB(t)\,r}\,a_i(t)$. Then $\partial_r f_i = -cB f_i$ and $\partial_{rr}f_i = c^2B^2f_i$, and the terms
    proportional to $r$ are</p>
    $$-cB'\,r = a\,cB\,r - c\,r,$$
    <p>which cancel when $B' = 1 - aB$, so $B(t) = (1 - e^{-at})/a$ as for a bond: the weight $c$ factors out. What
    remains involves $t$ only. So</p>
    $$\Psi_i(w) = S_0^{\,iw}\,e^{-iwqT}\,e^{-cB(T)\,r_0}\,a_i(T), \qquad a' = (Q + \operatorname{diag} g)a, \quad a(0) = \mathbf 1,$$
    $$g_i(t) = \underbrace{-a\,b_i\,c\,B + \tfrac12\eta_i^2\,c^2B^2}_{\text{rates, weight } c}\;
      \underbrace{-\,\tfrac12\sigma_i^2\,(w^2 + iw)}_{\text{equity}}\;\underbrace{-\,iw\,c\,\rho\,\sigma_i\eta_i\,B}_{\text{correlation}} .$$
    <p>The forcing is the sum of the rates forcing at weight $c = 1 - iw$ and the equity forcing at frequency $w$, plus
    the cross term. That is the whole construction: one linear system, with the discount and the return coupled
    through the regime.</p>
'''
    body += two_state_expansion(r'$\tilde g = \tilde\alpha\,B + \tilde\beta\,B^2 + \tilde\gamma$, with the coefficients given below',
                                complex_g='here $g$ is complex for every $w$',
                                extra=r''' The expansion needs $|\tilde g|/\lambda$ to be small, and $\tilde g$ grows like $w^2$. The Fourier integral is
    therefore stopped at the frequency where the averaged function falls below $e^{-40}$.''')
    body += r'''
    <h2>Explicit formulas</h2>
''' + SECOND_ORDER + r'''
    <h3>Step 1: the forcing</h3>
    <p>The forcing is a quadratic in $B$ with complex coefficients. Collect them by power of $B$:</p>
    $$g_i = \alpha_i\,B + \beta_i\,B^2 + \gamma_i, \qquad
      \alpha_i = -c\,\big(a\,b_i + iw\,\rho\,\sigma_i\eta_i\big), \quad \beta_i = \tfrac12c^2\eta_i^2, \quad
      \gamma_i = -\tfrac12\sigma_i^2\,(w^2 + iw),$$
    <p>and write $\bar\alpha, \tilde\alpha = \tfrac12(\alpha_1 \pm \alpha_2)$ and likewise for $\beta$ and $\gamma$. Then
    $\tilde g = \tilde\alpha B + \tilde\beta B^2 + \tilde\gamma$, with $\tilde g(0) = \tilde\gamma$, which is not zero,
    and $\tilde g' = (\tilde\alpha + 2\tilde\beta B)\,e^{-at}$.</p>

    <h3>Step 2: the integrals</h3>
    <p>With $I_n = \int_0^TB^n$,</p>
    $$\int_0^T\bar g = \bar\alpha\,I_1 + \bar\beta\,I_2 + \bar\gamma\,T,$$
    $$\int_0^T\tilde g^{\,2} = \tilde\alpha^2I_2 + 2\tilde\alpha\tilde\beta\,I_3 + \tilde\beta^2I_4
      + 2\tilde\alpha\tilde\gamma\,I_1 + 2\tilde\beta\tilde\gamma\,I_2 + \tilde\gamma^2\,T .$$
    <p>The first three terms are the Green&ndash;Kubo term of the rates alone, the last is that of the equity alone, and
    the two in the middle are the covariance between them: the regime that raises the equity volatility also moves the
    rate.</p>

    <h3>Step 3: the result</h3>
    <div class="equation-card">
    $$\begin{aligned}
    \Psi_{1,2}(w) = \;&S_0^{\,iw}\,e^{-iwqT}\,\exp\Big(-c\,B\,r_0 + \bar\alpha\,I_1 + \bar\beta\,I_2 + \bar\gamma\,T
      + \frac{\varepsilon}{2}\,A - \frac{\varepsilon^2}{8}\big(G^2 + \tilde\gamma^2\big)\Big) \\
    &\times\Big(1 \pm \frac{\varepsilon}{2}\,G \mp \frac{\varepsilon^2}{4}\,G'\Big) + O(\varepsilon^3),
    \end{aligned}$$
    $$\begin{aligned}
    A &= \tilde\alpha^2I_2 + 2\tilde\alpha\tilde\beta\,I_3 + \tilde\beta^2I_4
      + 2\tilde\alpha\tilde\gamma\,I_1 + 2\tilde\beta\tilde\gamma\,I_2 + \tilde\gamma^2\,T, \\
    G &= \tilde\alpha\,B + \tilde\beta\,B^2 + \tilde\gamma, \qquad G' = \big(\tilde\alpha + 2\tilde\beta\,B\big)\,E, \\
    \alpha_i &= -c\,\big(a\,b_i + iw\,\rho\,\sigma_i\eta_i\big), \qquad \beta_i = \tfrac12c^2\eta_i^2, \qquad
      \gamma_i = -\tfrac12\sigma_i^2\,(w^2 + iw), \qquad c = 1 - iw, \\
''' + I_LINES + r''' .
    \end{aligned}$$
    </div>
    <p>The upper sign is for a start in regime 1. Two checks are built in. At $w = 0$ the equity terms vanish,
    $c = 1$, and the card is the second-order <a href="./regime-switching.html">Vasicek bond</a>. At $w = -i$ the
    weight is $c = 0$, every $\alpha_i$, $\beta_i$ and $\gamma_i$ vanishes, and $\Psi = S_0e^{-qT}$ exactly.</p>

    <h3>A worked example</h3>
    <p>At $\lambda = 25$ and the Lewis frequency $w = 1 - i/2$:</p>
''' + table(['quantity', 'value'], q_rows) + table(['order', r'$\Psi_1(1 - i/2)$', 'error'], o_rows) + r'''
    <h2>Other pairs</h2>
    <ul>
      <li><strong>Hull&ndash;White rates.</strong> Write $r = x + \varphi(t)$ as on the
        <a href="./hull-white.html">Hull&ndash;White</a> page. The level term drops, $\alpha_i = -c\,iw\,\rho\,\sigma_i\eta_i$,
        and the prefactor $e^{-cBr_0}$ is replaced by $\exp\big(-c\int_0^T\varphi\big)$.</li>
      <li><strong>Heston equity.</strong> With $\rho = 0$ the equity part of the forcing is $\kappa\theta_iD(t; w)$ and the
        prefactor gains $e^{D(T;w)v_0}$, as on the <a href="./heston.html">Heston</a> page. An equity&ndash;rate
        correlation is not admitted: it adds a term $\rho\,\eta_i\sqrt v\,\partial_x\partial_r$ that no exponential-affine
        form absorbs, as noted in the <a href="./quantlib.html">catalogue</a>.</li>
    </ul>

    <h2>Results</h2>
    <p>Call prices from the numerical solution of the two-state system, and the error of the card after each order:</p>
''' + table(['switching rate', 'strike', 'numerical', 'order 0', 'order 1', 'order 2'], rows) + r'''    <p>The bond from the same regime at $\lambda = 25$ is $\Psi_1(0) = ''' + f'{bond:.6f}' + r'''$. The numerical prices agree with
    <a href="https://github.com/microprediction/regimelib">regimelib</a>&apos;s <code>SwitchingEquityRates</code> to the
    digits shown, and in the limit where the regimes coincide that class agrees with QuantLib&apos;s analytic
    Black&ndash;Scholes&ndash;Hull&ndash;White engine. Certificate: <a href="''' + SRC + r'''">model_pages2.py</a>.</p>
'''
    write('equity-rates.html', 'Equity options under stochastic rates', body)


# ====================================================================================== CEV
def cev_page():
    from scipy.stats import ncx2
    from scipy.integrate import solve_ivp
    from numpy.polynomial import chebyshev as ch
    S0, r, q, beta, K, T, sig = 100.0, 0.02, 0.0, 0.6, 100.0, 1.0, [2.5, 1.2]
    F = S0 * math.exp((r - q) * T)

    def C(v):
        den = (1 - beta) ** 2 * v
        x, y, d = K ** (2 * (1 - beta)) / den, F ** (2 * (1 - beta)) / den, 1 / (1 - beta)
        return F * (1 - ncx2.cdf(x, d + 2, y)) - K * ncx2.cdf(y, d, x)
    kh = 2 * (1 - beta) * (r - q)
    H1, H2, hT = (math.exp(kh * T) - 1) / kh, (math.exp(2 * kh * T) - 1) / (2 * kh), math.exp(kh * T)
    sb, st = half([s * s for s in sig])
    vbar, hw = sb * H1, st * H1                               # V lies in [vbar - hw, vbar + hw]
    deg = 28
    xs = np.cos(np.pi * (np.arange(60) + 0.5) / 60)
    cf = ch.chebfit(xs, [C(vbar + hw * x) for x in xs], deg)
    D = [ch.chebval(0.0, ch.chebder(cf, k)) / hw ** k if k else ch.chebval(0.0, cf) for k in range(5)]
    poly = ch.cheb2poly(cf)

    def exact(lam, reg):
        """E C(V) from the moments of x = (V - vbar) / hw, which solve m_k' = Q m_k + k rho_i(tau) m_{k-1}."""
        Q = np.array(sym(lam), float); sgn = np.array([1.0, -1.0])

        def rhs(tau, m):
            m = m.reshape(deg + 1, 2); out = np.zeros_like(m)
            rho_ = sgn * math.exp(kh * tau) / H1
            for k in range(deg + 1):
                out[k] = Q @ m[k] + (k * rho_ * m[k - 1] if k else 0.0)
            return out.ravel()
        m0 = np.zeros((deg + 1, 2)); m0[0] = 1.0
        m = solve_ivp(rhs, (0, T), m0.ravel(), method='DOP853', rtol=1e-13, atol=1e-15).y[:, -1].reshape(deg + 1, 2)
        return math.exp(-r * T) * float(poly @ m[:, reg])

    def order(lam, sign, o):
        eps = 1 / lam; disc = math.exp(-r * T)
        p = D[0]
        if o >= 1:
            p += eps / 2 * st * st * H2 * D[2] + sign * eps / 2 * st * hT * D[1]
        if o >= 2:
            p += eps * eps * (st ** 4 * H2 ** 2 / 8 * D[4] + sign * st ** 3 * H2 * hT / 4 * D[3]
                              - st * st * (hT * hT + 1) / 8 * D[2] - sign * st * kh * hT / 4 * D[1])
        return disc * p
    ex10 = exact(10.0, 0)
    print('CEV exact lam=10:', ex10, exact(10.0, 1), 'grid referee (regimelib, n=1601): 13.383591, 12.996193')
    assert abs(ex10 - 13.383591) < 1e-4 and abs(exact(10.0, 1) - 12.996193) < 1e-4
    rows = []
    for lam in (10.0, 20.0, 40.0):
        for reg, sign in ((0, 1), (1, -1)):
            ex = exact(lam, reg)
            rows.append([f'{lam:g}', f'{reg + 1}', f'{ex:.6f}'] + [e(abs(order(lam, sign, o) - ex)) for o in (0, 1, 2)])
    eps = 0.1
    q_rows = [['$F$', f'{F:.5f}'], [r'$\kappa_h$, $h_T$', f'{kh:.4f}, &nbsp; {hT:.6f}'], ['$H_1$, $H_2$', f'{H1:.6f}, &nbsp; {H2:.6f}'],
              [r'$\bar s$, $\tilde s$', f'{sb:.4f}, &nbsp; {st:.4f}'], [r'$\bar v = \bar sH_1$', f'{vbar:.6f}'],
              ['$C$', f'{D[0]:.6f}'], ['$C_v$, $C_{vv}$', f'{D[1]:.6f}, &nbsp; {D[2]:.6f}'.replace('-', '&minus;')],
              ['$C_{vvv}$, $C_{vvvv}$', f'{D[3]:.6f}, &nbsp; {D[4]:.6f}'.replace('-', '&minus;')],
              [r'Green&ndash;Kubo term $\frac\varepsilon2\tilde s^2H_2C_{vv}$, before discounting', f'{eps / 2 * st * st * H2 * D[2]:.6f}'.replace('-', '&minus;')],
              [r'memory term $\frac\varepsilon2\tilde sh_TC_v$, before discounting', f'{eps / 2 * st * hT * D[1]:.6f}']]
    o_rows = [[lab, f'{order(10.0, 1, o):.6f}', e(abs(order(10.0, 1, o) - ex10))] for o, lab in ((0, 'averaged CEV'), (1, 'first order'), (2, 'second order'))]
    o_rows.append(['exact mixture', f'{ex10:.6f}', ''])

    body = r'''    <h1>CEV with a switching volatility</h1>
    <p class="subtitle">A model that is not affine, in which the regime nevertheless factors out: the price is a mixture of CEV prices over one random variance.</p>

    <h2>The model</h2>
    <p>In the <a href="./bibliography.html#Schroder1989">constant elasticity of variance</a> model the volatility of
    the stock depends on its level through a power. Here the scale of that volatility is set by a hidden regime
    $y_t \in \{1, 2\}$:</p>
    $$dS_t = (r - q)\,S_t\,dt + \sigma_{y_t}\,S_t^{\beta}\,dW_t, \qquad 0 < \beta < 1 .$$
    <p>The regime switches at rate $\lambda$ in each direction. The option lives for $T$ and the regime changes every
    $1/\lambda$.</p>
    <p class="muted">Parameters: $\beta = 0.6$, $\sigma = (2.5, 1.2)$, which at $S = 100$ are lognormal volatilities of
    about 40% and 19%, $r = 0.02$, $q = 0$, $S_0 = K = 100$, $T = 1$.</p>

    <h2>The quantity</h2>
    <p>The European call $e^{-rT}\,\mathbb{E}[(S_T - K)^+ \mid y_0 = i]$.</p>

    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is the CEV model with the average squared scale
    $\bar\sigma^2 = \tfrac12(\sigma_1^2 + \sigma_2^2)$, priced by the noncentral chi-square formula below.</p>

    <h2>Why the affine reduction is not available, and what replaces it</h2>
    <p>The pricing equations are</p>
    $$\partial_t u_i + (r - q)\,S\,\partial_S u_i + \tfrac12\sigma_i^2\,S^{2\beta}\,\partial_{SS}u_i - r\,u_i + \sum_j Q_{ij}\,u_j = 0 .$$
    <p>There is no exponential-affine solution, because $S^{2\beta}$ is not linear in $S$. In these coordinates the
    switched operator $\mathcal A = \tfrac12S^{2\beta}\partial_{SS}$ also fails to commute with the drift, which is why
    the <a href="./quantlib.html">catalogue</a> first listed the model under the
    <a href="./any-equation.html">first-order rule</a>. The commutator, though, is a multiple of the operator itself:</p>
    $$\big[\,S\,\partial_S,\ S^{2\beta}\partial_{SS}\,\big] = 2(\beta - 1)\,S^{2\beta}\partial_{SS} .$$
    <p>To check it, apply both orders to a function $f$:
    $S\partial_S(S^{2\beta}f_{SS}) = 2\beta S^{2\beta}f_{SS} + S^{2\beta + 1}f_{SSS}$ and
    $S^{2\beta}\partial_{SS}(Sf_S) = 2S^{2\beta}f_{SS} + S^{2\beta + 1}f_{SSS}$. A commutator of this kind can be removed by a change
    of variable, and the variable is the forward.</p>

    <h3>Step 1: the forward</h3>
    <p>Let $F_t = S_t\,e^{(r - q)(T - t)}$, so that $F_T = S_T$. Its drift vanishes, and substituting
    $S = F e^{-(r - q)(T - t)}$ in the diffusion term gives</p>
    $$dF_t = \sigma_{y_t}\,\sqrt{h(T - t)}\;F_t^{\beta}\,dW_t, \qquad h(\tau) = e^{\kappa_h\tau}, \quad \kappa_h = 2(1 - \beta)(r - q) .$$
    <p>The regime and the calendar now enter only through the scalar $\sigma_{y_t}^2\,h(T - t)$ in front of one fixed
    operator, $\mathcal A_F = \tfrac12F^{2\beta}\partial_{FF}$. Operators of the form (scalar) $\times\,\mathcal A_F$ all
    commute.</p>

    <h3>Step 2: a random clock</h3>
    <p>A driftless diffusion whose variance rate is a scalar function of time is the same diffusion run on a
    different clock. So, given the regime path, $F_T$ is distributed as the unit-scale CEV diffusion
    $dG = G^\beta dW$ started at $F_0$ and run for the total variance</p>
    $$V = \int_0^T \sigma_{y_t}^2\,h(T - t)\,dt .$$
    <p>The regime path is independent of $W$, so the price is a mixture over $V$:</p>
    <div class="equation-card">
    $$u_i = e^{-rT}\;\mathbb{E}\big[\,C(F_0, V) \mid y_0 = i\,\big], \qquad F_0 = S_0\,e^{(r - q)T},$$
    </div>
    <p>where $C(F, v)$ is the undiscounted CEV call on a driftless forward with total variance $v$,</p>
    $$C(F, v) = F\,\big[1 - \chi^2\big(x;\ d + 2,\ y\big)\big] - K\,\chi^2\big(y;\ d,\ x\big), \qquad
      x = \frac{K^{2(1 - \beta)}}{(1 - \beta)^2\,v}, \quad y = \frac{F^{2(1 - \beta)}}{(1 - \beta)^2\,v}, \quad d = \frac{1}{1 - \beta},$$
    <p>with $\chi^2(\cdot\,;\ d,\ \nu)$ the noncentral chi-square distribution function with $d$ degrees of freedom and
    noncentrality $\nu$. This is exact for any chain and any number of regimes. When $r = q$ the weight $h$ is one and
    $V$ depends on the path only through the time spent in each regime.</p>

    <h2>Reduction to a linear system</h2>
    <p>All that is needed about the regime is the law of $V$, and its Laplace transform is a reduced system of the
    usual kind. In time to maturity $\tau$, the functions
    $a_i(\tau) = \mathbb{E}\big[\exp\big(-z\int\sigma_{y}^2h\big) \mid y = i\big]$ over the remaining life solve</p>
    $$a' = (Q + \operatorname{diag} g)\,a, \qquad a(0) = \mathbf 1, \qquad g_i(\tau) = -z\,\sigma_i^2\,h(\tau),$$
    <p>and $\mathbb{E}[e^{-zV} \mid y_0 = i] = a_i(T)$. The prefactor that an affine model would carry is absent; in its
    place the transform is turned back into a price. Since $C$ is smooth in $v$,
    $C(V) = e^{(V - \bar v)\partial_v}C(\bar v)$, and so</p>
    $$\mathbb{E}\big[C(V)\big] = \Big(e^{z\bar v}\,a_i(T)\Big)\Big|_{z = -\partial_v}\,C(\bar v), \qquad \bar v = \bar\sigma^2\int_0^Th :$$
    <p>every power of $z$ in the expansion of $a_i$ becomes a derivative of the CEV price in its variance argument.</p>
'''
    body += two_state_expansion(r'$\tilde g = -z\,\tilde s\,h(\tau), \qquad \tilde s = \tfrac12(\sigma_1^2 - \sigma_2^2)$')
    body += r'''
    <h2>The price in closed form</h2>
''' + SECOND_ORDER + r'''
    <h3>Step 1: the forcing and its integrals</h3>
    <p>With $\bar s, \tilde s = \tfrac12(\sigma_1^2 \pm \sigma_2^2)$, the forcing is $\bar g = -z\bar s\,h$ and
    $\tilde g = -z\tilde s\,h$, with $\tilde g(0) = -z\tilde s$ and $\tilde g' = -z\tilde s\,\kappa_h\,h$. The weight is
    an exponential, so</p>
    $$H_1 = \int_0^Th = \frac{e^{\kappa_hT} - 1}{\kappa_h}, \qquad H_2 = \int_0^Th^2 = \frac{e^{2\kappa_hT} - 1}{2\kappa_h}, \qquad h_T = e^{\kappa_hT},$$
    <p>and $\int\bar g = -z\,\bar s\,H_1 = -z\bar v$, $\int\tilde g^{\,2} = z^2\tilde s^2H_2$.</p>

    <h3>Step 2: the transform</h3>
    $$\mathbb{E}\big[e^{-zV}\big] = \exp\Big(-z\bar v + \frac{\varepsilon}{2}z^2\tilde s^2H_2 - \frac{\varepsilon^2}{8}z^2\tilde s^2\big(h_T^2 + 1\big)\Big)
      \Big(1 \mp \frac{\varepsilon}{2}\,z\,\tilde s\,h_T \pm \frac{\varepsilon^2}{4}\,z\,\tilde s\,\kappa_h\,h_T\Big) + O(\varepsilon^3).$$
    <p>Read as a statement about $V$: its mean is $\bar v \pm \tfrac12\varepsilon\tilde sh_T$, shifted towards the starting
    regime, and its variance is $\varepsilon\tilde s^2H_2$.</p>

    <h3>Step 3: the result</h3>
    <p>Expand in $\varepsilon$ and replace $z$ by $-\partial_v$:</p>
    <div class="equation-card">
    $$\begin{aligned}
    u_{1,2} = e^{-rT}\Big[\;&C + \frac{\varepsilon}{2}\,\tilde s^2H_2\,C_{vv} \pm \frac{\varepsilon}{2}\,\tilde s\,h_T\,C_v \\
      &+ \varepsilon^2\Big(\frac{\tilde s^4H_2^2}{8}\,C_{vvvv} \pm \frac{\tilde s^3H_2\,h_T}{4}\,C_{vvv}
      - \frac{\tilde s^2\,(h_T^2 + 1)}{8}\,C_{vv} \mp \frac{\tilde s\,\kappa_h\,h_T}{4}\,C_v\Big)\Big] + O(\varepsilon^3),
    \end{aligned}$$
    $$\begin{aligned}
    C &= C(F_0, \bar v) \text{ and its derivatives in } v, \qquad F_0 = S_0e^{(r - q)T}, \qquad \bar v = \bar s\,H_1, \\
    H_1 &= \frac{e^{\kappa_hT} - 1}{\kappa_h}, \qquad H_2 = \frac{e^{2\kappa_hT} - 1}{2\kappa_h}, \qquad h_T = e^{\kappa_hT},
      \qquad \kappa_h = 2(1 - \beta)(r - q), \\
    \bar s, \tilde s &= \frac{\sigma_1^2 \pm \sigma_2^2}{2}, \qquad \varepsilon = \frac1\lambda .
    \end{aligned}$$
    </div>
    <p>The upper sign is for a start in regime 1, the volatile one. The first-order terms have the meanings they have
    on every page: the Green&ndash;Kubo term is half the variance of $V$ times the convexity of the price in variance,
    and the memory term is the shift of the mean of $V$ times the sensitivity to variance. Because
    $\partial_vC = \mathcal A_FC$, the derivatives are also $C_v = \tfrac12F^{2\beta}C_{FF}$ and
    $C_{vv} = \mathcal A_F^2C$, which is how the <a href="./any-equation.html">first-order rule</a> writes the same two
    terms.</p>
    <p>At $\lambda = 10$ and a start in the volatile regime:</p>
''' + table(['quantity', 'value'], q_rows) + table(['call price', 'value', 'error'], o_rows) + r'''    <p>Here $C_{vv}$ is negative: at the money the price is concave in variance, so the randomness of $V$ lowers
    the price, while starting in the volatile regime raises it by three times as much.</p>

    <h2>Results</h2>
    <p>The exact mixture and the error of the card after each order. The exact value takes the moments of $V$ from the
    linear system $m_k' = Qm_k + k\,\sigma_i^2h\,m_{k-1}$ and applies them to a Chebyshev representation of $C$ on the
    range of $V$.</p>
''' + table(['switching rate', 'starting regime', 'exact', 'order 0', 'order 1', 'order 2'], rows) + r'''    <p>The mixture agrees with a finite-difference solution of the coupled equations in the original coordinates:
    <a href="https://github.com/microprediction/regimelib">regimelib</a>&apos;s <code>SwitchingFDReferee</code> on 1601
    points gives 13.38359 and 12.99619 at $\lambda = 10$ for the two starting regimes, against ''' + f'{ex10:.5f} and {exact(10.0, 1):.5f}' + r''' here, a
    difference of the size of the grid error. Its <code>FirstOrderFDEngine</code>, which applies the first-order rule on
    the grid without the change of variable, returns the same Green&ndash;Kubo and memory terms as the card, once discounted, to five
    digits. Certificate: <a href="''' + SRC + r'''">model_pages2.py</a>.</p>

    <h2>What does not transfer</h2>
    <p>The argument needs the switched parameter to scale one operator whose commutator with the rest is a multiple
    of itself. A switching elasticity $\beta$ changes the operator, and
    <a href="./cycle-smile.html">Heston with a switching volatility of variance</a> switches two operators,
    $v\,\partial_{vv}$ and $v\,\partial_{xv}$, that do not commute with the mean reversion in that way. Those stay with
    the first-order rule.</p>
'''
    write('cev.html', 'CEV with a switching volatility', body)


# ====================================================================================== instruments
def vasicek_survival(T, x, kap, th, sg, lam, sign, order):
    """The Vasicek bond or survival probability to second order (the card of the jumps page with no jumps)."""
    eps = 1 / lam
    E, B, I1, I2, I3, I4 = Is(kap, T)
    (thb, tht), (sb, st) = half(th), half([s * s for s in sg])
    avg = -B * x - kap * thb * I1 + 0.5 * sb * I2
    A = kap ** 2 * tht ** 2 * I2 - kap * tht * st * I3 + 0.25 * st * st * I4
    G = -kap * tht * B + 0.5 * st * B * B
    Gp = (-kap * tht + st * B) * E
    if order == 0:
        return math.exp(avg)
    if order == 1:
        return math.exp(avg + eps / 2 * A) * (1 + sign * eps / 2 * G)
    return math.exp(avg + eps / 2 * A - eps * eps / 8 * G * G) * (1 + sign * eps / 2 * G - sign * eps * eps / 4 * Gp)


def vasicek_numeric(T, x, kap, th, sg, lam, reg):
    Bf = lambda t: (1 - math.exp(-kap * t)) / kap
    gf = [lambda t, i=i: -kap * th[i] * Bf(t) + 0.5 * sg[i] ** 2 * Bf(t) ** 2 for i in range(2)]
    return math.exp(-Bf(T) * x) * numerical_a_callable(T, sym(lam), gf, rtol=1e-13)[reg].real


def instruments_page():
    # ---- credit default swap on a Vasicek intensity
    x0, kap, th, sg, R, rd = 0.02, 0.5, [0.06, 0.01], [0.01, 0.004], 0.4, 0.03
    times = [0.25 * k for k in range(1, 21)]

    def spread(Qf):
        prem = prot = 0.0; t0 = 0.0
        Dc = lambda t: math.exp(-rd * t)
        for t1 in times:
            tau, tm = t1 - t0, 0.5 * (t0 + t1)
            q0, q1 = (1.0 if t0 == 0 else Qf(t0)), Qf(t1)
            prem += tau * Dc(t1) * q1 + 0.5 * tau * Dc(tm) * (q0 - q1)
            prot += (1 - R) * Dc(tm) * (q0 - q1)
            t0 = t1
        return prot / prem
    cds_rows = []
    for lam in (2.0, 5.0, 10.0, 20.0):
        ref = spread(lambda t: vasicek_numeric(t, x0, kap, th, sg, lam, 0))
        cds_rows.append([f'{lam:g}', f'{ref * 1e4:.4f}'] + [e(abs(spread(lambda t: vasicek_survival(t, x0, kap, th, sg, lam, 1, o)) - ref) * 1e4) for o in (0, 1, 2)])
    # ---- geometric Asian under Black-Scholes
    sS, r, T, lam, S0 = [0.30, 0.15], 0.03, 1.0, 25.0, 100.0
    ssb, sst = half([s * s for s in sS])

    def asian_closed(u, lam, order, sign=1):
        eps = 1 / lam
        avg = 1j * u * (r - 0.5 * ssb) * T / 2 - u * u * ssb * T / 6
        A = 0.25 * sst ** 2 * T * (-u * u / 3 + 1j * u ** 3 / 2 + u ** 4 / 5)
        G, Gp = -0.5 * sst * (1j * u + u * u), -0.5 * sst * (1j * u + 2 * u * u) / T
        if order == 0:
            return cmath.exp(avg)
        if order == 1:
            return cmath.exp(avg + eps / 2 * A) * (1 + sign * eps / 2 * G)
        return cmath.exp(avg + eps / 2 * A - eps * eps / 8 * G * G) * (1 + sign * eps / 2 * G - sign * eps * eps / 4 * Gp)

    def asian_numeric(u, lam):
        gf = [lambda t, s=s: 1j * u * (r - 0.5 * s * s) * (t / T) - 0.5 * u * u * s * s * (t / T) ** 2 for s in sS]
        return numerical_a_callable(T, sym(lam), gf, rtol=1e-12, is_complex=True)[0]
    aref = asian_numeric(U0, lam)
    assert abs(asian_closed(U0, lam, 2) - aref) < 0.1 * abs(asian_closed(U0, lam, 1) - aref) < 0.01 * abs(asian_closed(U0, lam, 0) - aref)
    as_rows = [[lab, cz(asian_closed(U0, lam, o), 8), e(abs(asian_closed(U0, lam, o) - aref))] for o, lab in ((0, 'averaged model'), (1, 'first order'), (2, 'second order'))]
    as_rows.append(['numerical solution', cz(aref, 8), ''])
    U = math.sqrt(3 * 80 / (ssb * T))

    def asian_call(K, lam, order):
        phi = (lambda u: asian_numeric(u, lam)) if order is None else (lambda u: asian_closed(u, lam, order))
        FG = S0 * phi(-1j).real
        k, lf = math.log(FG / K), math.log(FG / S0)
        I0 = lewis(lambda z: phi(z) * cmath.exp(-1j * z * lf), k, U)
        return math.exp(-r * T) * (FG - math.sqrt(FG * K) / math.pi * I0)
    ap_rows = []
    for lam_ in (25.0, 50.0, 100.0):
        for K in (95, 105):
            rf = asian_call(K, lam_, None)
            ap_rows.append([f'{lam_:g}', f'{K}', f'{rf:.6f}'] + [e(abs(asian_call(K, lam_, o) - rf)) for o in (0, 1, 2)])
    # ---- parameter sensitivities of the first-order Vasicek bond
    Tg, lamg = 5.0, 5.0; eps = 1 / lamg
    E, B, I1, I2, I3, I4 = Is(kap, Tg)
    (thb, tht), (sb, st) = half(th), half([s * s for s in sg])
    A = kap ** 2 * tht ** 2 * I2 - kap * tht * st * I3 + 0.25 * st * st * I4
    G = -kap * tht * B + 0.5 * st * B * B
    cross = -kap * tht * I2 + 0.5 * st * I3                                  # int g~ B
    f_th1 = -kap / 2 * I1 - eps * kap / 2 * cross - eps * kap / 4 * B
    f_th2 = -kap / 2 * I1 + eps * kap / 2 * cross + eps * kap / 4 * B
    f_lam = -eps * eps * (0.5 * A + 0.5 * G)
    lp = lambda th_, lam_: math.log(vasicek_numeric(Tg, x0, kap, th_, sg, lam_, 0))
    d = 1e-5
    n_th1 = (lp([th[0] + d, th[1]], lamg) - lp([th[0] - d, th[1]], lamg)) / (2 * d)
    n_th2 = (lp([th[0], th[1] + d], lamg) - lp([th[0], th[1] - d], lamg)) / (2 * d)
    n_lam = (lp(th, lamg + 1e-3) - lp(th, lamg - 1e-3)) / 2e-3
    gr_rows = [[r'$\partial\log P_1/\partial\theta_1$', f'{f_th1:.6f}'.replace('-', '&minus;'), f'{n_th1:.6f}'.replace('-', '&minus;'), e(abs(f_th1 - n_th1))],
               [r'$\partial\log P_1/\partial\theta_2$', f'{f_th2:.6f}'.replace('-', '&minus;'), f'{n_th2:.6f}'.replace('-', '&minus;'), e(abs(f_th2 - n_th2))],
               [r'$\partial\log P_1/\partial\lambda$', f'{f_lam:.3e}'.replace('-', '&minus;'), f'{n_lam:.3e}'.replace('-', '&minus;'), e(abs(f_lam - n_lam))]]
    print('greeks', f_th1, n_th1, f_th2, n_th2, f_lam, n_lam)
    assert abs(f_th1 - n_th1) < 0.02 * abs(n_th1 - (-kap / 2 * I1)) + 1e-4 and abs(f_lam - n_lam) < 0.2 * abs(n_lam)

    body = r'''    <h1>Instruments</h1>
    <p class="subtitle">How each instrument is priced from the reduced system: what is solved, with which terminal vector, and what is then integrated.</p>

    <p class="lead">Every model page ends in the same object, the vector $a(t)$ solving
    $a' = (Q + \operatorname{diag} g)\,a$. This page lists what each instrument needs from it. The notation follows the
    model pages: a symmetric two-state chain switching at rate $\lambda = 1/\varepsilon$ where closed forms are written
    out, and any chain otherwise.</p>

    <h2>Bonds and claims on the regime</h2>
    <p>A zero-coupon bond, or a survival probability when the state is a default intensity, is</p>
    $$P_i(0,T) = e^{-B(T)\cdot x_0}\,a_i(T), \qquad a(0) = \mathbf 1 .$$
    <p>The terminal vector says what is paid in each regime at $T$. Starting from $a(0) = e_j$, the $j$-th unit vector,
    gives the value of one unit paid only if the regime at $T$ is $j$:</p>
    $$\mathbb{E}\big[e^{-\int_0^Tr}\,\mathbf 1\{y_T = j\} \mid y_0 = i\big] = e^{-B(T)\cdot x_0}\,a^{(j)}_i(T), \qquad a^{(j)}(0) = e_j .$$
    <p>The system is linear, so these add up to the bond. A payoff $\Phi_j$ that depends on the final regime is priced
    with $a(0) = \Phi$. For a two-state chain the closed forms for the bond are on the
    <a href="./regime-switching.html">regime-switching</a>, <a href="./jumps.html">jumps</a>, <a href="./cir.html">CIR</a>,
    <a href="./hull-white.html">Hull&ndash;White</a> and <a href="./g2.html">G2++</a> pages.</p>

    <h2>European options on equity</h2>
    <p>With $\phi_i(u)$ the characteristic function of the log return from regime $i$, a call is
    <a href="./bibliography.html#Lewis2001">Lewis&apos;s formula</a></p>
    $$C = S_0e^{-qT} - \frac{\sqrt{S_0K}\,e^{-rT}}{\pi}\int_0^\infty \operatorname{Re}\big[e^{iu\log(S_0/K)}\,\phi(u - \tfrac i2)\big]\,\frac{du}{u^2 + 1/4},$$
    <p>and a put follows by parity. The characteristic function is $\phi_i(u) = c(u)\,a_i(T; u)$, one solve of the
    reduced system per frequency, with the prefactor $c(u)$ and the forcing given on each model&apos;s page:
    <a href="./black-scholes.html">Black&ndash;Scholes</a>, <a href="./merton.html">Merton</a>,
    <a href="./variance-gamma.html">variance gamma</a>, <a href="./heston.html">Heston</a> and
    <a href="./bates.html">Bates</a>. Under stochastic rates the discounted form on the
    <a href="./equity-rates.html">equity with rates</a> page replaces it.</p>
    <p>A digital call paying one if $S_T > K$ is the Gil&ndash;Pelaez inversion of the same function,</p>
    $$e^{-rT}\,\mathbb{P}(S_T > K) = e^{-rT}\Big[\frac12 + \frac1\pi\int_0^\infty \operatorname{Re}\Big(\frac{e^{iu\log(S_0/K)}\,\phi(u)}{iu}\Big)\,du\Big].$$
    <p>The sensitivities to $S_0$ follow by differentiating under the integral. The derivative in $T$ needs
    $\partial_T\phi$, and the reduced system supplies it without a further solve:
    $\partial_Ta(T) = (Q + \operatorname{diag} g(T))\,a(T)$.</p>

    <h2>Options on bonds, swaptions and caps</h2>
    <p>Under a one-factor Gaussian short rate the bond at the option&apos;s expiry $T$, for maturity $S$, seen from regime $j$, is
    $A_j\,e^{-b\,x_T}$ with $A_j = a_j(S - T)$ and $b = B(S - T)$. The call pays $(A_je^{-bx_T} - K)^+$, which is positive
    when $x_T < x_j^* = \log(A_j/K)/b$: one exercise boundary per regime. Its value is a sum over the regime at
    expiry of two terms of the form</p>
    $$\mathbb{E}\big[e^{-\int_0^Tr}\,e^{-c\,x_T}\,\mathbf 1\{x_T < x_j^*\}\,\mathbf 1\{y_T = j\}\big]
      = \tfrac12\Psi_j(c) - \frac1\pi\int_0^\infty \operatorname{Im}\big(e^{-iux_j^*}\,\Psi_j(c - iu)\big)\,\frac{du}{u},$$
    <p>where $\Psi_j(c)$ is the reduced system with the terminal exponent $c$ and the terminal vector $e_j$. The
    terminal exponent changes the coefficient of the state from $B$ to
    $B_c(t) = c\,e^{-\kappa t} + (1 - e^{-\kappa t})/\kappa$ and leaves the form of the forcing alone. The derivation,
    and the closed forms, are on the <a href="./bond-options.html">bond options</a> page.</p>
    <p>In the same one-factor model, a coupon bond $\sum_k c_k\,A_{kj}\,e^{-b_kx}$ is decreasing in $x$ in every regime, so
    <a href="./bibliography.html#Jamshidian1989">Jamshidian&apos;s decomposition</a> holds regime by regime. In regime $j$
    there is one $x_j^*$ at which the coupon bond equals the strike, and the option on the coupon bond is the sum of
    options on its zero-coupon parts with strikes $K_{kj} = A_{kj}e^{-b_kx_j^*}$:</p>
    $$\Big(\sum_k c_kA_{kj}e^{-b_kx_T} - K\Big)^+ = \sum_k c_k\,\big(A_{kj}e^{-b_kx_T} - K_{kj}\big)^+ \qquad\text{on } \{y_T = j\} .$$
    <p>The strikes differ by regime, which is the only change from the model without regimes. From there:</p>
    <ul>
      <li>a receiver swaption is a call on the coupon bond with coupons $\tau_kK$ and the notional at the end, struck at
        par, and a payer swaption is the put;</li>
      <li>a caplet on the period $[t_{k-1}, t_k]$ with strike $K$ is $1 + \tau_kK$ puts on the bond maturing at $t_k$,
        struck at $1/(1 + \tau_kK)$ and expiring at $t_{k-1}$;</li>
      <li>under <a href="./hull-white.html">Hull&ndash;White</a> the deterministic shift scales each cash flow by
        $e^{-\int\varphi}$ and the rest is unchanged;</li>
      <li>under <a href="./g2.html">G2++</a> a single zero-coupon bond still depends on one Gaussian linear combination,
        so its option remains one-dimensional; this does not extend to a coupon bond.</li>
    </ul>
    <h3>Why G2++ swaptions are not Jamshidian decompositions</h3>
    <p>At expiry, a cash flow with residual maturity $\tau_k$ has loading vector</p>
    $$b_k = \big(B_a(\tau_k), B_b(\tau_k)\big), \qquad
      B_c(\tau) = \frac{1-e^{-c\tau}}{c}.$$
    <p>Thus the regime-$j$ coupon bond is
    $C_j(x,z)=\sum_k c_kA_{kj}\exp(-b_k\cdot(x,z))$. A linear scalar-Gaussian reduction of Jamshidian&apos;s kind exists
    exactly when all of the $b_k$ are collinear. For two distinct residual maturities this fails when $a\ne b$, because</p>
    $$R(\tau)=\frac{B_a(\tau)}{B_b(\tau)}, \qquad
      \frac{d}{d\tau}\log R(\tau)=\frac{a}{e^{a\tau}-1}-\frac{b}{e^{b\tau}-1},$$
    <p>and $c/(e^{c\tau}-1)$ is strictly decreasing in $c>0$. With the G2++ parameters on this site,
    $a=0.5$, $b=0.1$, and residual maturities one and four years,</p>
    $$\det(b_1,b_2)=0.94870454.$$
    <p>The exercise boundary is therefore curved rather than a Gaussian half-space. For two unit cash flows struck at
    two, the boundary through the origin has</p>
    $$z''(0)=\frac{2\det(b_1,b_2)^2}{(B_b(1)+B_b(4))^3}=0.02347513.$$
    <p>G2++ coupon-bond options and swaptions consequently require a two-dimensional Gaussian integral, or conditioning
    on one factor and solving the monotone boundary in the other. The one-factor decomposition above remains valid for
    Vasicek and Hull&ndash;White, while zero-coupon bond options and caplets remain one-dimensional under G2++.</p>
    <h3>An exact one-dimensional G2++ price</h3>
    <p>The curved boundary does not require two-dimensional numerical quadrature. In ordinary fixed-parameter G2++,
      work under the $T$-forward measure and write $(X,Z)$ for the bivariate Gaussian factors at option expiry. Put
      $w_k=c_kA_k&gt;0$, $p_k=B_a(\tau_k)$ and $q_k=B_b(\tau_k)$. For every fixed $x$,</p>
    $$C(x,z)=\sum_kw_k e^{-p_kx-q_kz}$$
    <p>decreases continuously from infinity to zero as $z$ increases, so there is a unique root
      $z=\zeta(x)$ of $C(x,z)=K$. If $Z\mid X=x\sim N(m(x),s^2)$, Gaussian exponential tilting gives the exact receiver
      forward value</p>
    $$\int \phi_X(x)\left[\sum_k w_k e^{-p_kx-q_km(x)+q_k^2s^2/2}
      \Phi\!\left({\zeta(x)-m(x)+q_ks^2\over s}\right)
      -K\Phi\!\left({\zeta(x)-m(x)\over s}\right)\right]dx.$$
    <p>The reduction is not special to two factors. Let $(Y,Z)$ be jointly Gaussian with
      $Y\in\mathbb R^{d-1}$ and write each loading as $(p_k,q_k)$, where $p_k\in\mathbb R^{d-1}$ and every
      $q_k&gt;0$. Conditional Gaussian regression gives $Z\mid Y=y\sim N(m(y),s^2)$ with affine $m$ and constant
      $s^2$. For each $y$, the equation
      $\sum_kw_k\exp(-p_k^\top y-q_kz)=K$ has one root $z=\zeta(y)$, and the same displayed tilted-normal formula,
      with $p_kx$ replaced by $p_k^\top y$, integrates out $Z$ exactly. A $d$-factor coupon option is therefore
      reduced to a $(d-1)$-dimensional Gaussian integral whenever one factor has strictly positive loadings for every
      cash flow.</p>
    <p>The boundary is a convex hypersurface. With
      $d_k=w_k\exp(-p_k^\top y-q_k\zeta(y))$ and
      $D=\sum_kd_kq_k$, implicit differentiation gives</p>
    $$\nabla\zeta=-{\sum_kd_kp_k\over D},\qquad
      \nabla^2\zeta={1\over D}\sum_kd_k
      (p_k+q_k\nabla\zeta)(p_k+q_k\nabla\zeta)^\top\succeq0.$$
    <p>If $r_k=p_k/q_k$, the Hessian rank is the affine-span dimension of the active ratio vectors $r_k$.
      Consequently it is positive definite exactly when those ratios affinely span
      $\mathbb R^{d-1}$. This is a local coupon-boundary rank statement under a fixed Gaussian expiry law; it is
      unrelated to fixed-parameter Green&ndash;Kubo rank or to rank after integrating maturity loadings.</p>
    <p>Multiplication by $P(0,T)$ gives the time-zero price; the payer follows by parity. For G2++ this is exact
      one-dimensional quadrature, not a Jamshidian sum. The remaining numerical truncation has a closed-form
      certificate. Write
      $X=\mu_X+\sigma_X\xi$, where $\xi$ is standard normal, and let
      $s^2=\sigma_Z^2(1-\rho^2)$ and $a_k=p_k\sigma_X+q_k\rho\sigma_Z$. If the receiver integral is restricted to
      $|\xi|\le L$, positivity and $(C-K)^+\le C$ give</p>
    $$0\le V-V_L\le\sum_k w_k
      e^{-p_k\mu_X-q_k\mu_Z+q_k^2s^2/2+a_k^2/2}
      \{\Phi(a_k-L)+\Phi(-a_k-L)\}.$$
    <p>This is the exact exponentially tilted Gaussian tail of each cash flow, not an asymptotic estimate. The payer
      tail is bounded more simply by $2K\Phi(-L)$. Both bounds are forward values; multiplication by $P(0,T)$ gives
      time-zero bounds. Implicit differentiation also quantifies the geometry. With
      $d_k=w_ke^{-p_kx-q_k\zeta(x)}$,</p>
    $$\zeta'(x)=-{\sum_kd_kp_k\over\sum_kd_kq_k},\qquad
      \zeta''(x)={\sum_kd_k(p_k+q_k\zeta'(x))^2\over\sum_kd_kq_k}\ge0.$$
    <p>The inequality is strict exactly when the positive-cash-flow loading vectors are not collinear. Thus unequal
      G2++ reversion speeds and at least two distinct payment maturities give a strictly convex boundary, while $a=b$
      recovers the one-factor Jamshidian limit.</p>
    <p>There is also an exact local variance identity. Set
      $\alpha_k=d_kq_k/\sum_jd_jq_j$, $r_k=p_k/q_k$ and
      $\bar r_\alpha=\sum_k\alpha_kr_k=-\zeta'(x)$. Then</p>
    $$\zeta''(x)=\sum_k\alpha_kq_k(r_k-\bar r_\alpha)^2.$$
    <p>Thus the boundary bends precisely where multiple loading directions remain relevant; its second derivative is
      a positive weighted dispersion, not merely a sign calculation.</p>
    <h3>The global boundary and its exact wings</h3>
    <p>Convexity is global and the two wings are determined by different cash flows. Put
      $r_k=p_k/q_k$, $r_{\min}=\min_kr_k$ and $r_{\max}=\max_kr_k$. Since</p>
    $$-\zeta'(x)={\sum_kd_kq_kr_k\over\sum_kd_kq_k},$$
    <p>the slope obeys
      $-r_{\max}\leq\zeta'(x)\leq-r_{\min}$, strictly between the endpoints whenever the loading directions differ.
      More precisely, if $I_R=\{k:r_k=r_{\min}\}$ and $I_L=\{k:r_k=r_{\max}\}$, let $c_R,c_L$ be the unique
      solutions of</p>
    $$\sum_{k\in I_R}w_ke^{-q_kc_R}=K,\qquad
      \sum_{k\in I_L}w_ke^{-q_kc_L}=K.$$
    <p>Then the exact asymptotes are</p>
    $$\zeta(x)+r_{\min}x\longrightarrow c_R\quad(x\to+\infty),\qquad
      \zeta(x)+r_{\max}x\longrightarrow c_L\quad(x\to-\infty).$$
    <p>Combining these limiting slopes with convexity gives the exact global slope-turning identities. If
      $\Gamma=\{(x,\zeta(x)):x\in\mathbb R\}$, then</p>
    $$\int_{-\infty}^{\infty}\zeta''(x)\,dx=r_{\max}-r_{\min},\qquad
      \int_{\Gamma}\kappa_{\rm geom}\,ds
      =\arctan r_{\max}-\arctan r_{\min}.$$
    <p>The second formula is the Euclidean turning angle of the graph. The distribution of bending depends on the
      cash-flow weights and strike, but its total second-derivative mass and total turning angle depend only on the
      two extreme loading directions.</p>
    <p>To prove this, substitute $z=-r_{\min}x+c$ in $C(x,z)=K$. Terms outside $I_R$ acquire the factor
      $\exp[-q_k(r_k-r_{\min})x]$ and vanish locally uniformly in $c$; the limiting left-hand side is continuous,
      strictly decreasing and crosses $K$ once. Monotonicity therefore carries its root to $c_R$. The left wing is
      identical after substituting $z=-r_{\max}x+c$ and sending $x$ to minus infinity.</p>
    <p>If each extreme is unique, attained at $i_R$ and $i_L$, this also supplies convergence rates:</p>
    $$\begin{aligned}
      \zeta(x)&=-r_{\min}x+{1\over q_{i_R}}\log{w_{i_R}\over K}
        +O(e^{-\delta_Rx}),
        &\delta_R&=\min_{k\ne i_R}q_k(r_k-r_{\min}),\\
      \zeta(x)&=-r_{\max}x+{1\over q_{i_L}}\log{w_{i_L}\over K}
        +O(e^{-\delta_L|x|}),
        &\delta_L&=\min_{k\ne i_L}q_k(r_{\max}-r_k).
    \end{aligned}$$
    <p>There is also an explicit finite-$x$ bound. Set
      $A_R=\sum_{k\ne i_R}w_ke^{-q_kc_R}$ and $u_R(x)=(A_R/K)e^{-\delta_Rx}$. Whenever $x\ge0$ and $u_R(x)&lt;1$,</p>
    $$0\leq \zeta(x)+r_{\min}x-c_R
      \leq-{1\over q_{i_R}}\log(1-u_R(x)).$$
    <p>The analogous left-wing bound has
      $A_L=\sum_{k\ne i_L}w_ke^{-q_kc_L}$ and $u_L(x)=(A_L/K)e^{-\delta_L|x|}$ for $x\le0$.
      Indeed the dominant term equals $K e^{-q_i\eta(x)}$ after subtracting the stated line. Positivity gives
      $\eta\ge0$, and bounding every competing exponential by $e^{-\delta|x|}$ gives the displayed inequality.
      For $a&gt;b$, the
      ratio $B_a(\tau)/B_b(\tau)$ decreases with residual maturity. Hence the longest payment determines the right
      wing and the shortest payment the left wing&mdash;a global obstruction to replacing the boundary by one
      Gaussian half-space.</p>
    <p>For the certificate above, $(r_{\min},c_R,\delta_R)=(0.524547948012,-0.119219437938,0.287765309607)$ and
      $(r_{\max},c_L,\delta_L)=(0.826941287566,-1.181063053835,0.996930222634)$. Boundary roots at increasing
      magnitudes recover exponential rates $0.287766603132$ and $0.996923880456$, respectively. At $x=20$ the
      right-wing error is $3.496798408893\times10^{-4}$ against the rigorous bound
      $3.497962883732\times10^{-4}$; at $x=-10$ the left-wing error is $1.622785325623\times10^{-3}$ against
      $1.631497250470\times10^{-3}$. Direct curvature quadrature on $[-50,100]$ gives
      $0.302393339554306$ against the exact extreme-ratio spread $0.302393339554316$.</p>
    <p><a href="./bibliography.html#ChoiShin2016">Choi and Shin (2016)</a> likewise formulate exact pricing in
      multi-factor Gaussian term-structure models by root-finding on the nonlinear boundary and integrating in the
      remaining dimensions, alongside a tangent-hyperplane approximation. Here the two-factor monotonicity makes
      the root unique for every conditioning value and the Gaussian tilting evaluates the other dimension exactly.
      This direct Gaussian formula is narrower than the general affine approximations of
      <a href="./bibliography.html#SchragerPelsser2006">Schrager and Pelsser (2006)</a>: it assumes a fixed-parameter
      Gaussian expiry law. It is exact for standard G2++ and its averaged fast-switching limit, not for the full
      finite-rate switching model.</p>
    <p>For the two-cash-flow certificate, conditional quadrature gives $0.669351292291$ and independent nested Gaussian
      quadrature gives the same value within $2.22\times10^{-16}$. In the $a=b$ limit it agrees with the scalar
      Jamshidian price within $1.67\times10^{-16}$. Truncating the standardized conditioning factor to $[-6,6]$
      changes the receiver value by less than $8.00\times10^{-9}$, inside the closed-form bound above. A separate
      three-factor certificate uses three maturities and a genuinely two-dimensional boundary: its Hessian eigenvalues
      are $5.14636\times10^{-5}$ and $3.65439\times10^{-2}$, the loading-ratio affine determinant is
      $8.73332\times10^{-3}$, and centered finite differences agree with the Hessian identity within
      $3.38\times10^{-9}$. The analytic conditional formula gives
      $1.252075854512$ at both 28- and 36-point tensor Gauss&ndash;Hermite orders and agrees with independent adaptive
      integration of the final Gaussian coordinate within $2.22\times10^{-16}$.</p>
    <p>For a 2&times;5 payer swaption under a two-regime Vasicek model, with mean levels 6% and 2% and volatilities 1.5%
    and 0.8%, the relative error of the expansion at each order against the numerical solution, by mean holding
    time, as computed by <a href="https://github.com/microprediction/regimelib/blob/main/papers/regimelib/verify_tables.py">regimelib</a>:</p>
''' + table([r'$\varepsilon$', 'order 0', 'order 1', 'order 2', 'order 3', 'order 4', 'order 6'], [
        ['0.050', '7.2e-3', '4.5e-5', '1.1e-7', '1.4e-7', '1.9e-9', '8.5e-11'],
        ['0.100', '1.5e-2', '1.7e-4', '9.9e-7', '2.5e-6', '1.9e-8', '1.9e-8'],
        ['0.200', '3.1e-2', '5.5e-4', '1.5e-5', '4.6e-5', '3.1e-6', 'diverged'],
        ['0.333', '5.3e-2', '5.9e-3', '1.6e-4', 'diverged', 'diverged', 'diverged']]) + r'''    <p>The series is asymptotic. At a holding time of four months it is best stopped at second order.</p>

    <h2>Credit default swaps</h2>
    <p>When $x_t$ is a default intensity, $Q(t) = e^{-B(t)x_0}a_i(t)$ is the survival probability. A swap paying the
    spread $s$ on the dates $t_1 < \dots < t_n$, with accrued premium paid at default and recovery $R$, has the legs</p>
    $$\text{premium} = s\sum_k\Big[\tau_k\,D(t_k)\,Q(t_k) + \tfrac12\tau_k\,D(t_k^m)\,\big(Q(t_{k-1}) - Q(t_k)\big)\Big],
      \qquad \text{protection} = (1 - R)\sum_k D(t_k^m)\,\big(Q(t_{k-1}) - Q(t_k)\big),$$
    <p>with $\tau_k = t_k - t_{k-1}$, $t_k^m$ the midpoint of the period and $D$ the discount curve. The fair spread is
    the ratio of the protection leg to the premium leg per unit spread. Only survival probabilities are needed, so the
    closed form is that of the bond. For a Vasicek intensity $dx = \kappa(\theta_y - x)dt + \sigma_y dW$,</p>
    <div class="equation-card">
    $$Q_{1,2}(t) = \exp\Big(-Bx_0 - \kappa\bar\theta\,I_1 + \tfrac12\bar s\,I_2 + \frac\varepsilon2A - \frac{\varepsilon^2}{8}G^2\Big)
      \Big(1 \pm \frac\varepsilon2G \mp \frac{\varepsilon^2}{4}G'\Big) + O(\varepsilon^3),$$
    $$A = \kappa^2\tilde\theta^2I_2 - \kappa\tilde\theta\,\tilde s\,I_3 + \tfrac14\tilde s^2I_4, \qquad
      G = -\kappa\tilde\theta\,B + \tfrac12\tilde s\,B^2, \qquad G' = \big(-\kappa\tilde\theta + \tilde s\,B\big)\,e^{-\kappa t},$$
    </div>
    <p>with $B$ and $I_n = \int_0^tB^n$ evaluated at each payment date, as on the <a href="./jumps.html">jumps</a> page.
    A five-year swap with quarterly premiums, $x_0 = 0.02$, $\kappa = 0.5$, $\theta = (0.06, 0.01)$,
    $\sigma = (0.01, 0.004)$, recovery 40% and a flat 3% discount curve, from the high-intensity regime; the fair
    spread in basis points and the error of each order, also in basis points:</p>
''' + table(['switching rate', 'fair spread, numerical', 'order 0', 'order 1', 'order 2'], cds_rows) + r'''    <p>Several names on one chain have joint survival equal to the bond of the summed forcing, so a first-to-default
    swap is a single-name swap on the sum, as on the <a href="./credit.html">two-name credit</a> page.</p>

    <h2>Geometric Asian options</h2>
    <p>Under <a href="./black-scholes.html">Black&ndash;Scholes</a> with a switching volatility, the continuous geometric
    average is $G = S_0e^{Y}$ with</p>
    $$Y = \frac1T\int_0^T\log\frac{S_t}{S_0}\,dt = \int_0^T\Big(1 - \frac tT\Big)\,d\log S_t .$$
    <p>The second form comes from exchanging the order of integration: the increment of $\log S$ at time $t$ is counted
    in the average for the remaining fraction $1 - t/T$ of the life. Each increment therefore enters $Y$ scaled by a
    deterministic weight, and in time to maturity $\tau = T - t$ that weight is $\tau/T$. The characteristic function
    $\mathbb{E}[e^{iuY} \mid y_0 = i] = a_i(T)$ solves the reduced system with</p>
    $$g_i(\tau) = iu\,\big(r - q - \tfrac12\sigma_i^2\big)\,\frac{\tau}{T} - \tfrac12u^2\sigma_i^2\,\frac{\tau^2}{T^2} .$$
    <p>The forcing is a polynomial in $\tau$, so every integral is elementary. With
    $\bar s, \tilde s = \tfrac12(\sigma_1^2 \pm \sigma_2^2)$ the half-difference is
    $\tilde g = -\tfrac12\tilde s\,\big(iu\,\tau/T + u^2\tau^2/T^2\big)$, which vanishes at $\tau = 0$, and</p>
    <div class="equation-card">
    $$\mathbb{E}\big[e^{iuY}\big]_{1,2} = \exp\Big(\int_0^T\bar g + \frac\varepsilon2A - \frac{\varepsilon^2}8G^2\Big)
      \Big(1 \pm \frac\varepsilon2G \mp \frac{\varepsilon^2}4G'\Big) + O(\varepsilon^3),$$
    $$\begin{aligned}
    \int_0^T\bar g &= \tfrac12\,iu\,\big(r - q - \tfrac12\bar s\big)\,T - \tfrac16u^2\,\bar s\,T, \\
    A &= \tfrac14\tilde s^2\,T\,\Big(-\frac{u^2}{3} + \frac{iu^3}{2} + \frac{u^4}{5}\Big), \\
    G &= -\tfrac12\tilde s\,(iu + u^2), \qquad G' = -\frac{\tilde s}{2T}\,(iu + 2u^2) .
    \end{aligned}$$
    </div>
    <p>The first line of the list is the Kemna&ndash;Vorst result: the average of a lognormal path has half the drift and a
    third of the variance. The option is then Lewis&apos;s formula on $G$, whose forward is
    $F_G = S_0\,\mathbb{E}[e^{Y}]$, the function above at $u = -i$:</p>
    $$C = e^{-rT}\Big[F_G - \frac{\sqrt{F_GK}}{\pi}\int_0^\infty\operatorname{Re}\Big(e^{iu\log(F_G/K)}\,\frac{\phi_Y(u - \tfrac i2)}{(F_G/S_0)^{\,i(u - i/2)}}\Big)\frac{du}{u^2 + 1/4}\Big].$$
    <p>With $\sigma = (0.30, 0.15)$, $r = 0.03$, $q = 0$, $T = 1$, $\lambda = 25$ and $u = 1 - i/2$, from the volatile
    regime:</p>
''' + table([r'$\mathbb{E}[e^{iuY}]$', 'value', 'error'], as_rows) + r'''    <p>Call prices on the geometric average with $S_0 = 100$, and the error of the card after each order:</p>
''' + table(['switching rate', 'strike', 'numerical', 'order 0', 'order 1', 'order 2'], ap_rows) + r'''    <p>The characteristic function of the average decays three times more slowly in $u^2$ than that of the terminal
    price, so the expansion is asked to work at higher frequencies than for a vanilla option, where $|\tilde g|/\lambda$
    is larger. At slower chains the series should be replaced by the numerical solution at the high frequencies.</p>

    <h2>Barrier, American and Bermudan instruments</h2>
    <p>These depend on the path of the state and not only on its terminal value, so the state does not factor out and
    there is no reduced system. They are priced on the coupled equations themselves, one unknown function per regime,</p>
    $$\partial_t u_i + L_iu_i + \sum_jQ_{ij}u_j = 0,$$
    <p>on a grid in the state with one block per regime:</p>
    <ul>
      <li>a knock-out barrier is the condition $u_i = \text{rebate}$ at the barrier in every regime, and a knock-in is the
        vanilla less the knock-out;</li>
      <li>an American option is the variational inequality
        $\max\big(\partial_t u_i + L_iu_i + \sum_jQ_{ij}u_j,\ \Phi - u_i\big) = 0$, with an exercise boundary that
        differs by regime;</li>
      <li>a Bermudan swaption projects onto the exercise value at each exercise date, and that value is the coupon bond
        in each regime from the reduced system.</li>
    </ul>
    <p>The <a href="./any-equation.html">first-order rule</a> still describes the operator, but the boundary adds layers
    that these pages do not treat, so no closed form is offered.</p>

    <h2>Sensitivities to the parameters</h2>
    <p>The closed forms are explicit in the parameters, so their derivatives are formulas. To first order the log price
    from a symmetric two-state chain is $\int\bar g + \tfrac12\varepsilon\int\tilde g^{\,2} + \log(1 \pm \tfrac12\varepsilon\tilde g(T))$.
    If a parameter $p_i$ of regime $i$ enters the forcing linearly, $g_i = p_i\,f(t) + \dots$, then
    $\partial\bar g/\partial p_1 = \partial\tilde g/\partial p_1 = \tfrac12f$ and</p>
    <div class="equation-card">
    $$\frac{\partial\log a_{1,2}}{\partial p_1} = \frac12\int_0^Tf + \frac\varepsilon2\int_0^T\tilde g\,f \pm \frac\varepsilon4\,f(T), \qquad
      \frac{\partial\log a_{1,2}}{\partial p_2} = \frac12\int_0^Tf - \frac\varepsilon2\int_0^T\tilde g\,f \mp \frac\varepsilon4\,f(T),$$
    $$\frac{\partial\log a_{1,2}}{\partial\lambda} = -\varepsilon^2\Big(\frac12\int_0^T\tilde g^{\,2} \pm \frac12\,\tilde g(T)\Big),$$
    </div>
    <p>each up to $O(\varepsilon^2)$, and $O(\varepsilon^3)$ for the last. The first term is the sensitivity of the
    averaged model, shared equally between the regimes. The second moves it towards the regime whose parameter matters
    more, and the third is the memory of the start. For the Vasicek mean level, $f = -\kappa B$ and
    $\int\tilde g\,f = -\kappa\,(-\kappa\tilde\theta I_2 + \tfrac12\tilde sI_3)$. With the intensity parameters of the
    swap above, $T = 5$ and $\lambda = 5$, against central differences of the numerical solution:</p>
''' + table(['sensitivity', 'formula', 'finite difference', 'difference'], gr_rows) + r'''    <p>For a general chain the first-order price depends on the chain through the stationary distribution $\pi$ and
    the group inverse $Q^\#$, and both have exact derivatives. For a perturbation $dQ$ with zero row sums,</p>
    $$d\pi = -\pi\,dQ\,Q^\#, \qquad dQ^\# = -Q^\#\,dQ\,Q^\# + \mathbf 1\,\pi\,dQ\,(Q^\#)^2 + (Q^\#)^2\,dQ\,\mathbf 1\,\pi .$$
    <p>To see the first, differentiate $\pi Q = 0$ to get $d\pi\,Q = -\pi\,dQ$ and multiply on the right by $Q^\#$,
    using $QQ^\# = I - \mathbf 1\pi$ and $d\pi\,\mathbf 1 = 0$. The Green&ndash;Kubo matrix
    $K_{jk} = -\pi\cdot(\tilde f_j\odot Q^\#\tilde f_k)$ and the memory vector $m = Q^\#\tilde g$ then vary by the
    product rule.</p>
    <p>Certificates: <a href="''' + SRC + r'''">model_pages2.py</a> and
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/verify_g2_coupon_geometry.py">verify_g2_coupon_geometry.py</a>.
    The instruments are implemented in
    <a href="https://github.com/microprediction/regimelib">regimelib</a>.</p>
'''
    write('instruments.html', 'Instruments', body)


def main():
    hull_white_page()
    g2_page()
    equity_rates_page()
    cev_page()
    instruments_page()


if __name__ == '__main__':
    main()
