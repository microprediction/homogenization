"""Pages for the regimelib models that had a catalogue row and no derivation: Merton, Bates, variance gamma, equity
with stochastic rates, and CEV. Each page works its model to second order in closed form, and every number on it is
computed here: the closed form against the numerical solution of the reduced system, and prices from both.

`python3 papers/fast-switching/model_pages.py && python3 tools/assemble.py && node docs/header-check.js`
"""
import cmath
import math
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fastswitch import FastSwitch, ExpSum, Cheb, numerical_a_callable
from models import gaussian_factors, heston_switching_theta
from quantlib_models import bates as bates_forcing
from pages_examples import table, e, write, sym, two_state_expansion

U0 = 1 - 0.5j                                     # the Lewis frequency used in the worked examples


def cz(z, p=7):
    z = complex(z)
    if abs(z.imag) < 5 * 10.0 ** (-p - 1):
        return f'{z.real:.{p}f}'.replace('-', '&minus;')
    return (f'{z.real:.{p}f}'.replace('-', '&minus;') + (' + ' if z.imag >= 0 else ' &minus; ') + f'{abs(z.imag):.{p}f} i')


def cg(z, p=4):
    z = complex(z)
    r = ('&minus;' if z.real < 0 else '') + f'{abs(z.real):.{p}g}'
    if abs(z.imag) < 1e-15 * max(1.0, abs(z.real)):
        return r
    return r + (' + ' if z.imag >= 0 else ' &minus; ') + f'{abs(z.imag):.{p}g} i'


def half(x):
    return (x[0] + x[1]) / 2, (x[0] - x[1]) / 2


def nodes(U, panels=1, n=96):
    x, w = np.polynomial.legendre.leggauss(n)
    us, ws = [], []
    for p in range(panels):
        lo, hi = U * p / panels, U * (p + 1) / panels
        us.extend((x + 1) * (hi - lo) / 2 + lo)
        ws.extend(w * (hi - lo) / 2)
    return us, ws


def lewis(phi, k, U, panels=1):
    """int_0^U Re[e^{iuk} phi(u - i/2)] / (u^2 + 1/4) du."""
    us, ws = nodes(U, panels)
    return sum(w * (cmath.exp(1j * u * k) * phi(u - 0.5j)).real / (u * u + 0.25) for u, w in zip(us, ws))


def exact_constant(gb, gt, lam, T, sign):
    """The symmetric two-state system with constant forcing, in closed form."""
    s = cmath.sqrt(lam * lam + gt * gt)
    return cmath.exp((gb - lam) * T) * (cmath.cosh(s * T) + (lam + sign * gt) / s * cmath.sinh(s * T))


def order_constant(gb, gt, lam, T, sign, order):
    eps, L = 1 / lam, 1 - cmath.exp(-2 * lam * T)
    if order == 0:
        return cmath.exp(gb * T)
    if order == 1:
        return cmath.exp(gb * T + eps / 2 * gt * gt * T) * (1 + sign * eps / 2 * gt * L)
    return cmath.exp(gb * T + eps / 2 * gt * gt * T - eps * eps / 4 * gt * gt * L) * (1 + sign * eps / 2 * gt * L)


SECOND_ORDER = r'''    <h3>Second order</h3>
    <p>Two more steps give the next order. The recursion gives $\omega_2 = -\omega_1'/2 = -\tilde g'/4$, which adds
    $-\tfrac14\varepsilon^2\int_0^T\tilde g\,\tilde g' = -\tfrac18\varepsilon^2\big(\tilde g(T)^2 - \tilde g(0)^2\big)$ to
    the exponent. The initial layer subtracts $\tfrac12\varepsilon\,\tilde g(0)\,e^{-2\lambda t}$ from $\omega$, and
    through $\int\tilde g\,\omega$ that adds $-\tfrac14\varepsilon^2\,\tilde g(0)^2$. Together, up to terms of size
    $e^{-2\lambda T}$,</p>
    <div class="equation-card">
    $$a_{1,2}(T) = \exp\Big(\int_0^T \bar g + \frac{\varepsilon}{2}\int_0^T \tilde g^{\,2}
      - \frac{\varepsilon^2}{8}\big(\tilde g(T)^2 + \tilde g(0)^2\big)\Big)
      \Big(1 \pm \frac{\varepsilon}{2}\,\tilde g(T) \mp \frac{\varepsilon^2}{4}\,\tilde g'(T)\Big) + O(\varepsilon^3).$$
    </div>
    <p>What remains is to evaluate $\int\bar g$, $\int\tilde g^{\,2}$, $\tilde g$ and $\tilde g'$ for this model.</p>
'''

CONSTANT_CARD = r'''    <div class="equation-card">
    $$\begin{aligned}
    \phi_{1,2} \;=\; &e^{(\bar g - \lambda)T}\Big(\cosh sT + \frac{\lambda \pm \tilde g}{s}\,\sinh sT\Big),
      \qquad s = \sqrt{\lambda^2 + \tilde g^2}, \\[4pt]
    \;=\; &\exp\Big(\bar g\,T + \frac{\varepsilon}{2}\,\tilde g^2\,T - \frac{\varepsilon^2}{4}\,\tilde g^2\big(1 - e^{-2\lambda T}\big)\Big)
      \Big(1 \pm \frac{\varepsilon}{2}\,\tilde g\,\big(1 - e^{-2\lambda T}\big)\Big) + O(\varepsilon^3).
    \end{aligned}$$
    </div>
'''

CONSTANT_PROOF = r'''    <p>The first line follows from the matrix exponential. The system is $a' = Ma$ with</p>
    $$M = \begin{pmatrix} g_1 - \lambda & \lambda \\ \lambda & g_2 - \lambda\end{pmatrix}
        = (\bar g - \lambda)\,I + N, \qquad N = \begin{pmatrix}\tilde g & \lambda \\ \lambda & -\tilde g\end{pmatrix},
        \qquad N^2 = (\lambda^2 + \tilde g^2)\,I = s^2 I,$$
    <p>so $e^{MT} = e^{(\bar g - \lambda)T}\big(\cosh sT\; I + s^{-1}\sinh sT\; N\big)$, and applying it to $\mathbf 1$
    gives the two rows $\lambda \pm \tilde g$. For the second line write $s = \lambda\sqrt{1 + \varepsilon^2\tilde g^2}
    = \lambda + \tfrac12\varepsilon\tilde g^2 + O(\varepsilon^3)$, so that $e^{(s - \lambda)T}$ supplies the
    Green&ndash;Kubo factor, and expand the remaining bracket in $\varepsilon$.</p>
'''

CONSTANT_SCOPE = r'''    <h3>Where the expansion applies</h3>
    <p>This expansion is pointwise in the Fourier frequency. Put $z = \varepsilon\tilde g$ and take the principal
    square root $d(z) = \sqrt{1 + z^2}$. The exact answer separates into its slow outer mode and its fast initial
    layer:</p>
    <div class="equation-card">
    $$\phi_\pm=e^{\bar gT}\left[A_\pm(z)e^{(d(z)-1)T/\varepsilon}
      +B_\pm(z)e^{-(d(z)+1)T/\varepsilon}\right],\qquad
      A_\pm=\frac12\left(1+\frac{1\pm z}{d}\right),\quad
      B_\pm=\frac12\left(1-\frac{1\pm z}{d}\right).$$
    </div>
    <p>The branch points $z=\pm i$ give the natural radius one. More quantitatively, fix $0&lt;\rho&lt;1$. If
    $|z|\leq\rho$, then the principal-root identity
    $(\operatorname{Re}d)^2=(|1+z^2|+\operatorname{Re}(1+z^2))/2$ gives
    $\operatorname{Re}d(z)\geq\sqrt{1-\rho^2}$, and hence</p>
    $$\left|e^{\bar gT}B_\pm(z)e^{-(d(z)+1)T/\varepsilon}\right|
      \leq e^{\operatorname{Re}\bar gT}
      \frac{\sqrt{1+\rho^2}+1+\rho}{2\sqrt{1-\rho^2}}
      e^{-(1+\sqrt{1-\rho^2})T/\varepsilon}.$$
    <p>For fixed $\tilde g$, the outer factor is analytic for $|z|&lt;1$. If $P_{2,\pm}$ is its quadratic Taylor
    polynomial,</p>
    $$P_{2,\pm}(z)=1+\frac{T\tilde g\pm1}{2}z+
      \left(\frac{(T\tilde g)^2}{8}\pm\frac{T\tilde g}{4}-\frac14\right)z^2,$$
    <p>then for $|z|\leq r&lt;\rho&lt;1$, Cauchy&apos;s estimate gives the explicit remainder</p>
    $$\left|A_\pm(z)e^{T\tilde g(d(z)-1)/z}-P_{2,\pm}(z)\right|
      \leq \frac{M_\rho}{\rho^3(1-r/\rho)}|z|^3,\qquad
      M_\rho=\max_{|w|=\rho}\left|A_\pm(w)e^{T\tilde g(d(w)-1)/w}\right|.$$
    <p>Combining the two estimates gives the absolute error bound</p>
    $$\left|\phi_\pm-e^{\bar gT}P_{2,\pm}(z)\right|\leq e^{\operatorname{Re}\bar gT}
      \left[\frac{M_\rho}{\rho^3(1-r/\rho)}|z|^3+
      \frac{\sqrt{1+\rho^2}+1+\rho}{2\sqrt{1-\rho^2}}
      e^{-(1+\sqrt{1-\rho^2})T/\varepsilon}\right].$$
    <p>Thus the fixed-frequency second-order error is $O(\lambda^{-3})$ plus an explicitly bounded exponential
    layer. A Fourier price needs the separate frequency-envelope condition
    $|\tilde g(u)|/\lambda\leq\rho$ on the part of the integral where the approximation is used; the statement is
    not uniform over all frequencies.</p>

    <h3>From a transform bound to a price bound</h3>
    <p>The passage through Lewis inversion can be made exact, but it introduces a separate tail term. Let
    $\phi_\lambda$ be the exact log-return transform, let $\widehat\phi_\lambda$ be any approximation used only on
    $0\leq u\leq R$, and suppose
    $|\phi_\lambda(u-i/2)-\widehat\phi_\lambda(u-i/2)|\leq E_\lambda(u)$ there. If
    $M_{1/2,\lambda}=\mathbb E[e^{X_\lambda/2}]&lt;\infty$, then the Lewis call with the approximate transform
    stopped at $R$ satisfies</p>
    <div class="equation-card">
    $$\begin{aligned}
    |C_\lambda-\widehat C_{\lambda,R}|
    \leq \frac{\sqrt{S_0K}\,e^{-rT}}{\pi}\left[
    \int_0^R\frac{E_\lambda(u)}{u^2+1/4}\,du
\mathrel{+}M_{1/2,\lambda}\{\pi-2\arctan(2R)\}\right].
    \end{aligned}$$
    </div>
    <p>Indeed, $|\phi_\lambda(u-i/2)|\leq M_{1/2,\lambda}$ and
    $\int_R^\infty(u^2+1/4)^{-1}du=\pi-2\arctan(2R)$. This proves convergence by first fixing $R$, taking the
    fast-switching limit on that compact interval, and then sending $R$ to infinity, provided the half-moments are
    uniformly bounded. It does <em>not</em> transfer the compact-frequency order automatically. If the forcing
    condition permits only $R=O(\lambda)$, as for variance gamma with switching martingale corrections, this
    model-independent tail certificate is only $O(\lambda^{-1})$; for a diffusive $u^2$ contrast and
    $R=O(\sqrt\lambda)$ it is only $O(\lambda^{-1/2})$. Faster price rates require a model-specific decay bound for
    the shifted characteristic function. These are upper-bound limitations, not lower bounds on the actual error.</p>

    <h3>Unequal transition rates</h3>
    <p>The radius-one conclusion is not an artifact of symmetric switching. Let the generator be
    $m\left(\begin{smallmatrix}-a&a\\b&-b\end{smallmatrix}\right)$ with $a,b&gt;0$, put
    $\kappa=a+b$, $\vartheta=(a-b)/\kappa$, $\varepsilon=(m\kappa)^{-1}$,
    $\delta=g_1-g_2$, $z=\varepsilon\delta$, and
    $\bar g_\pi=(bg_1+ag_2)/\kappa$. Direct diagonalization gives</p>
    <div class="equation-card">
    $$\begin{aligned}
    \phi_\pm={}&e^{\bar g_\pi T}\left[A_\pm(z)
      e^{\{d(z)-1+\vartheta z\}T/(2\varepsilon)}
      +B_\pm(z)e^{-\{d(z)+1-\vartheta z\}T/(2\varepsilon)}\right],\\
    d(z)={}&\sqrt{1-2\vartheta z+z^2},\qquad
    A_\pm=\frac12\left(1+\frac{1\pm z}{d}\right),\quad
    B_\pm=\frac12\left(1-\frac{1\pm z}{d}\right).
    \end{aligned}$$
    </div>
    <p>The branch points are
    $z=\vartheta\pm i\sqrt{1-\vartheta^2}$, again on the unit circle. Thus the slow mode is analytic for
    $|z|&lt;1$. On every closed disk $|z|\leq\rho&lt;1$,
    $\eta_{\vartheta,\rho}=\min\operatorname{Re}d(z)&gt;0$ and
    $|B_\pm(z)|\leq(1-\rho)^{-1}$, so the second term is bounded by</p>
    $$\frac{e^{\operatorname{Re}\bar g_\pi T}}{1-\rho}
      \exp\left[-\frac{\eta_{\vartheta,\rho}+1-|\vartheta|\rho}
      {2\varepsilon}T\right].$$
    <p>The quadratic slow-mode polynomial therefore has an $O((m\kappa)^{-3})$ fixed-frequency remainder plus this
    exponential layer by Cauchy&apos;s estimate on nested disks. Explicitly, if
    $c_1=T\delta(1-\vartheta^2)/4$, $c_2=T\delta\vartheta(1-\vartheta^2)/4$,
    $a_{1,\pm}=(\vartheta\pm1)/2$, and
    $a_{2,\pm}=(3\vartheta^2-1)/4\pm\vartheta/2$, the slow multiplier after
    $e^{\bar g_\pi T}$ is</p>
    $$P_{2,\pm}(z)=1+(a_{1,\pm}+c_1)z+
      \left(a_{2,\pm}+c_2+a_{1,\pm}c_1+\tfrac12c_1^2\right)z^2+O(z^3).$$
    <p>Symmetric switching is the special case $\vartheta=0$, $\delta=2\tilde g$. This transform is
    the finite-state Feynman&ndash;Kac counterpart of the two-state occupation transforms studied by
    <a href="https://doi.org/10.2307/3211908">Pedler (1971)</a>.</p>
'''


# ====================================================================================== Merton
def merton_mc(S0, K, T, r, sig, ell, mu, de, lam, N=400000, seed=7):
    """Exact regime paths from exponential holding times; given the time spent in regime 1 the model is Merton's with
    integrated variance V and integrated intensity L, priced by Merton's series of Black-Scholes prices."""
    rng = np.random.default_rng(seed)
    kbar = math.exp(mu + de * de / 2) - 1
    M = int(lam * T + 12 * math.sqrt(lam * T) + 20)
    vals = {k: [] for k in K}
    from scipy.stats import norm
    for _ in range(N // 50000):
        hold = rng.exponential(1 / lam, size=(50000, M))
        ends = np.cumsum(hold, axis=1)
        occ = np.clip(np.minimum(ends, T) - (ends - hold), 0, None)[:, 0::2].sum(axis=1)       # time in the starting regime
        V = sig[0] ** 2 * occ + sig[1] ** 2 * (T - occ)
        L = ell[0] * occ + ell[1] * (T - occ)
        for k in K:
            tot = np.zeros_like(V)
            w = np.exp(-L)
            for n in range(60):
                vn = V + n * de * de
                F = S0 * np.exp(r * T - L * kbar + n * (mu + de * de / 2))
                d1 = (np.log(F / k) + vn / 2) / np.sqrt(vn)
                tot += w * math.exp(-r * T) * (F * norm.cdf(d1) - k * norm.cdf(d1 - np.sqrt(vn)))
                w = w * L / (n + 1)
            vals[k].append(tot)
    out = {}
    for k in K:
        v = np.concatenate(vals[k])
        out[k] = (v.mean(), v.std() / math.sqrt(len(v)))
    return out


def merton_page():
    sig, ell, mu, de, r, S0, T = [0.25, 0.12], [2.0, 0.2], -0.08, 0.10, 0.03, 100.0, 1.0
    kbar = math.exp(mu + de * de / 2) - 1
    sb, st = half([s * s for s in sig]); lb, lt = half(ell)

    def gparts(u):
        phiJ = cmath.exp(1j * u * mu - 0.5 * u * u * de * de)
        g = [1j * u * (r - 0.5 * s * s - l * kbar) - 0.5 * u * u * s * s + l * (phiJ - 1) for s, l in zip(sig, ell)]
        return half(g) + (phiJ, g)
    gb, gt, phiJ, g = gparts(U0)
    assert abs(gb - (1j * U0 * (r - sb / 2 - lb * kbar) - U0 * U0 * sb / 2 + lb * (phiJ - 1))) < 1e-15
    assert abs(gt - (-0.5 * (1j * U0 + U0 * U0) * st + lt * (phiJ - 1 - 1j * U0 * kbar))) < 1e-15
    lam = 25.0
    ex = exact_constant(gb, gt, lam, T, +1)
    num = numerical_a_callable(T, sym(lam), [lambda t, c=c: c for c in g], rtol=1e-13)[0]
    assert abs(ex - num) < 1e-10, abs(ex - num)
    ex_rows = [[r'$\phi_J$, $\bar k$', cz(phiJ) + ', &nbsp; ' + cz(kbar), ''],
               [r'$\bar g$, $\tilde g$', cz(gb) + ', &nbsp; ' + cz(gt), ''],
               ['averaged Merton', cz(order_constant(gb, gt, lam, T, 1, 0), 8), e(abs(order_constant(gb, gt, lam, T, 1, 0) - ex))],
               ['first order', cz(order_constant(gb, gt, lam, T, 1, 1), 8), e(abs(order_constant(gb, gt, lam, T, 1, 1) - ex))],
               ['second order', cz(order_constant(gb, gt, lam, T, 1, 2), 8), e(abs(order_constant(gb, gt, lam, T, 1, 2) - ex))],
               ['exact', cz(ex, 8), '']]
    U = math.sqrt(80 / (sb * T))

    def call(K, lam, order):
        def phi(u):
            b, t = gparts(u)[:2]
            return exact_constant(b, t, lam, T, 1) if order is None else order_constant(b, t, lam, T, 1, order)
        return S0 - math.sqrt(S0 * K) * math.exp(-r * T) / math.pi * lewis(phi, math.log(S0 / K), U)
    rows, exact50 = [], {}
    for lam_ in (25.0, 50.0, 100.0):
        for K in (90, 110):
            ref = call(K, lam_, None)
            if lam_ == 50.0:
                exact50[K] = ref
            rows.append([f'{lam_:g}', f'{K}', f'{ref:.6f}'] + [e(abs(call(K, lam_, o) - ref)) for o in (0, 1, 2)])
    mc = merton_mc(S0, (90, 110), T, r, sig, ell, mu, de, 50.0)
    z = [abs(mc[K][0] - exact50[K]) / mc[K][1] for K in (90, 110)]
    z0 = [abs(mc[K][0] - call(K, 50.0, 0)) / mc[K][1] for K in (90, 110)]
    print('Merton MC', mc, 'exact', exact50, 'z', z, 'averaged z', z0)
    assert max(z) < 3

    body = r'''    <h1>Merton&apos;s jump diffusion with a switching jump intensity</h1>
    <p class="subtitle">A stock that jumps more often, and diffuses faster, in a stressed regime; European option prices in closed form.</p>

    <h2>The model</h2>
    <p>A stock price follows <a href="./bibliography.html#Merton1976">Merton&apos;s jump diffusion</a>, with the
    volatility and the jump intensity set by a hidden regime $y_t \in \{1, 2\}$:</p>
    $$\frac{dS_t}{S_{t-}} = \big(r - \ell_{y_t}\bar k\big)\,dt + \sigma_{y_t}\,dW_t + \big(e^{J} - 1\big)\,dN_t .$$
    <p>While $y_t = i$ the Poisson process $N$ jumps at rate $\ell_i$. Each jump multiplies the price by $e^J$ with
    $J$ normal, of mean $\mu_J$ and standard deviation $\delta$, and $\bar k = e^{\mu_J + \delta^2/2} - 1$ is the mean
    relative jump, so the drift term $-\ell_i\bar k$ keeps the discounted price a martingale in each regime.</p>
    <p>The regime switches at rate $\lambda$ in each direction. The option lives for $T$ and the regime changes every
    $1/\lambda$, so switching is fast when $\lambda T \gg 1$.</p>
    <p class="muted">Parameters: $\sigma = (0.25, 0.12)$, $\ell = (2, 0.2)$, $\mu_J = -0.08$, $\delta = 0.10$,
    $r = 0.03$, $S_0 = 100$, $T = 1$, and the stock starts in the stressed regime.</p>

    <h2>The quantity</h2>
    <p>European call prices, through the characteristic function of the log return</p>
    $$\phi_i(u) = \mathbb{E}\big[e^{iu\log(S_T/S_0)} \mid y_0 = i\big]$$
    <p>and
    <a href="./bibliography.html#Lewis2001">Lewis&apos;s formula</a></p>
    $$C = S_0 - \frac{\sqrt{S_0K}\,e^{-rT}}{\pi}\int_0^\infty \operatorname{Re}\big[e^{iu\log(S_0/K)}\,\phi(u - \tfrac i2)\big]\,\frac{du}{u^2 + 1/4}.$$
    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is Merton&apos;s model with the average variance
    $\bar\sigma^2 = \tfrac12(\sigma_1^2 + \sigma_2^2)$ and the average intensity $\bar\ell = \tfrac12(\ell_1 + \ell_2)$.
    The jump law is unchanged.</p>

    <h2>Reduction to a linear system</h2>
    <p>Condition on a short interval $dt$ at the start. While the regime is $i$ the log return moves in two independent
    ways. The diffusion adds a normal amount with mean $(r - \ell_i\bar k - \tfrac12\sigma_i^2)\,dt$ and variance
    $\sigma_i^2\,dt$. With probability $\ell_i\,dt$ there is a jump, which adds $J$ and multiplies $e^{iu\log S}$ by
    $e^{iuJ}$, whose mean is</p>
    $$\phi_J(u) = \mathbb{E}\big[e^{iuJ}\big] = \exp\big(iu\mu_J - \tfrac12u^2\delta^2\big).$$
    <p>So over $dt$ the quantity $e^{iu\log(S_T/S_0)}$ is multiplied on average by</p>
    $$\exp\Big(\big[iu(r - \ell_i\bar k - \tfrac12\sigma_i^2) - \tfrac12u^2\sigma_i^2 + \ell_i\,(\phi_J(u) - 1)\big]\,dt\Big).$$
    <p>The bracket is the L&eacute;vy exponent of regime $i$. Meanwhile the regime jumps at the rates in $Q$. So
    $\phi_i(u) = a_i(T)$, with no state to factor out, $a' = (Q + \operatorname{diag} g)a$, $a(0) = \mathbf 1$ and the
    constant, complex</p>
    $$g_i = iu\big(r - \tfrac12\sigma_i^2\big) - \tfrac12u^2\sigma_i^2 + \ell_i\,\big(\phi_J(u) - 1 - iu\bar k\big) .$$
    <p>The first two terms are the <a href="./black-scholes.html">Black&ndash;Scholes</a> forcing. The last is the
    compensated jump exponent, and it is proportional to the intensity.</p>
'''
    body += two_state_expansion(r'$\tilde g = -\tfrac12(iu + u^2)\,\tilde s + \tilde\ell\,\big(\phi_J(u) - 1 - iu\bar k\big), \qquad \tilde s = \tfrac12(\sigma_1^2 - \sigma_2^2), \quad \tilde\ell = \tfrac12(\ell_1 - \ell_2)$',
                                complex_g='here $g$ is complex for every $u$',
                                extra=r''' The expansion needs $|\tilde g|/\lambda$ to be small, and $\tilde g$ grows like $u^2$. The Fourier integral is
    therefore stopped at the frequency where the averaged characteristic function falls below $e^{-40}$; beyond it
    the integrand is negligible and the expansion would not apply.''')
    body += r'''
    <h2>The characteristic function in closed form</h2>
    <p>The forcing is constant, so the two-state system can be solved exactly. The characteristic function of the log
    return is</p>
''' + CONSTANT_CARD + r'''    <p>with</p>
    $$\begin{aligned}
    \bar g &= iu\big(r - \tfrac12\bar s\big) - \tfrac12u^2\,\bar s + \bar\ell\,\big(\phi_J(u) - 1 - iu\bar k\big), \\
    \tilde g &= -\tfrac12\big(iu + u^2\big)\,\tilde s + \tilde\ell\,\big(\phi_J(u) - 1 - iu\bar k\big), \\
    \phi_J(u) &= \exp\big(iu\mu_J - \tfrac12u^2\delta^2\big), \qquad \bar k = e^{\mu_J + \delta^2/2} - 1, \\
    \bar s, \tilde s &= \frac{\sigma_1^2 \pm \sigma_2^2}{2}, \qquad \bar\ell, \tilde\ell = \frac{\ell_1 \pm \ell_2}{2} .
    \end{aligned}$$
''' + CONSTANT_PROOF + CONSTANT_SCOPE + r'''    <p>The first line is exact. The second is its expansion in $\varepsilon = 1/\lambda$: Merton&apos;s characteristic
    function at the averaged variance and intensity, times the Green&ndash;Kubo factor $e^{\varepsilon\tilde g^2T/2}$
    and the memory of the starting regime. The upper sign is for a start in the stressed regime.</p>
    <p>The Green&ndash;Kubo exponent has three parts, from the square of $\tilde g$:</p>
    $$\frac{\varepsilon T}{2}\,\tilde g^2 = \frac{\varepsilon T}{2}\Big[\tfrac14(iu + u^2)^2\,\tilde s^2
      - (iu + u^2)\,\tilde s\,\tilde\ell\,c_J + \tilde\ell^{\,2}c_J^2\Big], \qquad c_J = \phi_J(u) - 1 - iu\bar k .$$
    <p>The first is the term of the Black&ndash;Scholes page. The last is the fluctuation of the jump count. The middle
    one is the covariance of the two: it is present because the same regime raises the volatility and the intensity.</p>
    <p>At $\lambda = 25$, $T = 1$ and the Lewis frequency $u = 1 - i/2$:</p>
''' + table(['quantity', 'value', 'error'], ex_rows) + r'''
    <h2>A jump law that switches</h2>
    <p>Nothing above used the fact that the two regimes share a jump law. If the jumps in regime $i$ have mean
    $\mu_{J,i}$ and standard deviation $\delta_i$, the only change is that the forcing carries its own exponent,</p>
    $$g_i = iu\big(r - \tfrac12\sigma_i^2\big) - \tfrac12u^2\sigma_i^2 + \ell_i\,\big(\phi_{J,i}(u) - 1 - iu\bar k_i\big),
      \qquad \phi_{J,i}(u) = e^{iu\mu_{J,i} - u^2\delta_i^2/2}, \quad \bar k_i = e^{\mu_{J,i} + \delta_i^2/2} - 1,$$
    <p>and the closed form holds with $\bar g$ and $\tilde g$ the half-sum and half-difference of these. The forcing is
    still constant, so the first line of the card is still exact.</p>

    <h2>Results</h2>
    <p>Call prices from the exact characteristic function, and the error of the expansion after each order:</p>
''' + table(['switching rate', 'strike', 'exact', 'order 0', 'order 1', 'order 2'], rows) + r'''    <p>A Monte Carlo check at $\lambda = 50$ samples 400,000 regime paths exactly, from exponential holding times.
    Given the time $\tau$ spent in the stressed regime the model is Merton&apos;s with integrated variance
    $\sigma_1^2\tau + \sigma_2^2(T - \tau)$ and integrated intensity $\ell_1\tau + \ell_2(T - \tau)$, so each path is
    priced by Merton&apos;s series of Black&ndash;Scholes prices with no time steps. It gives ''' + \
        f'{mc[90][0]:.4f} &plusmn; {mc[90][1]:.4f} and {mc[110][0]:.4f} &plusmn; {mc[110][1]:.4f}' + \
        r''' at strikes 90 and 110, within ''' + f'{max(z):.1f}' + r''' standard errors of the exact prices. The
    averaged model is off by more than ''' + f'{int(min(z0) // 10 * 10)}' + r''' standard errors. Certificate:
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/model_pages.py">model_pages.py</a>.</p>
    <p>In <a href="https://github.com/microprediction/regimelib">regimelib</a> this model is
    <code>SwitchingMerton76Process</code>.</p>
'''
    write('merton.html', 'Merton with a switching jump intensity', body)


# ====================================================================================== variance gamma
def variance_gamma_page():
    sg, nu, th, r, S0, T = [0.25, 0.12], [0.5, 0.2], [-0.25, -0.10], 0.03, 100.0, 1.0

    def psi(z, s, n, t):
        return -cmath.log(1 - 1j * t * n * z + 0.5 * s * s * n * z * z) / n
    om = [(-psi(-1j, s, n, t)).real for s, n, t in zip(sg, nu, th)]
    for s, n, t, o in zip(sg, nu, th, om):
        assert abs(o - math.log(1 - t * n - 0.5 * s * s * n) / n) < 1e-14

    def gparts(u):
        g = [1j * u * (r + o) + psi(u, s, n, t) for s, n, t, o in zip(sg, nu, th, om)]
        return half(g) + (g,)
    gb, gt, g = gparts(U0)
    lam = 25.0
    ex = exact_constant(gb, gt, lam, T, +1)
    num = numerical_a_callable(T, sym(lam), [lambda t, c=c: c for c in g], rtol=1e-13)[0]
    assert abs(ex - num) < 1e-10
    ex_rows = [[r'$\omega_1$, $\omega_2$', f'{om[0]:.7f}, &nbsp; {om[1]:.7f}'.replace('-', '&minus;'), ''],
               [r'$g_1$, $g_2$', cz(g[0]) + ', &nbsp; ' + cz(g[1]), ''],
               [r'$\bar g$, $\tilde g$', cz(gb) + ', &nbsp; ' + cz(gt), ''],
               ['averaged exponent', cz(order_constant(gb, gt, lam, T, 1, 0), 8), e(abs(order_constant(gb, gt, lam, T, 1, 0) - ex))],
               ['first order', cz(order_constant(gb, gt, lam, T, 1, 1), 8), e(abs(order_constant(gb, gt, lam, T, 1, 1) - ex))],
               ['second order', cz(order_constant(gb, gt, lam, T, 1, 2), 8), e(abs(order_constant(gb, gt, lam, T, 1, 2) - ex))],
               ['exact', cz(ex, 8), '']]
    U, panels = 600.0, 24

    def call(K, lam, order):
        def phi(u):
            b, t = gparts(u)[:2]
            return exact_constant(b, t, lam, T, 1) if order is None else order_constant(b, t, lam, T, 1, order)
        return S0 - math.sqrt(S0 * K) * math.exp(-r * T) / math.pi * lewis(phi, math.log(S0 / K), U, panels)
    assert abs(call(100, 25.0, None) - (S0 - math.sqrt(S0 * 100) * math.exp(-r * T) / math.pi * lewis(
        lambda u: exact_constant(*gparts(u)[:2], 25.0, T, 1), 0.0, 2 * U, 2 * panels))) < 1e-8
    rows = []
    for lam_ in (25.0, 50.0, 100.0):
        for K in (90, 110):
            ref = call(K, lam_, None)
            rows.append([f'{lam_:g}', f'{K}', f'{ref:.6f}'] + [e(abs(call(K, lam_, o) - ref)) for o in (0, 1, 2)])
    gbig = gparts(200 - 0.5j)
    omtilde = (om[0] - om[1]) / 2
    print('VG: |g~| at u = 200:', abs(gbig[1]), '|g~|/u:', abs(gbig[1]) / 200, 'omega~:', omtilde)

    body = r'''    <h1>Variance gamma with switching parameters</h1>
    <p class="subtitle">A pure-jump stock whose volatility, kurtosis and skew all change with a hidden regime; European option prices in closed form.</p>

    <h2>The model</h2>
    <p>In the <a href="./bibliography.html#MadanCarrChang1998">variance gamma model</a> the log price is a Brownian
    motion with drift $\theta$ and volatility $\sigma$, run on a gamma clock whose variance rate is $\nu$. Here all
    three parameters are set by a hidden regime $y_t \in \{1, 2\}$: while $y_t = i$,</p>
    $$d\log S_t = (r + \omega_i)\,dt + dX^{(i)}_t, \qquad X^{(i)}_t = \theta_i\,G^{(i)}_t + \sigma_i\,W_{G^{(i)}_t},$$
    <p>with $G^{(i)}$ a gamma process of unit mean rate and variance rate $\nu_i$. The constant</p>
    $$\omega_i = \frac{1}{\nu_i}\log\big(1 - \theta_i\nu_i - \tfrac12\sigma_i^2\nu_i\big)$$
    <p>keeps the discounted price a martingale in each regime. The regime switches at rate $\lambda$ in each direction.
    The option lives for $T$ and the regime changes every $1/\lambda$.</p>
    <p class="muted">Parameters: $\sigma = (0.25, 0.12)$, $\nu = (0.5, 0.2)$, $\theta = (-0.25, -0.10)$, $r = 0.03$,
    $S_0 = 100$, $T = 1$, and the stock starts in the first regime, the one with the fatter tails and the stronger
    skew.</p>

    <h2>The quantity</h2>
    <p>European call prices, through the characteristic function of the log return</p>
    $$\phi_i(u) = \mathbb{E}\big[e^{iu\log(S_T/S_0)} \mid y_0 = i\big]$$
    <p>and
    <a href="./bibliography.html#Lewis2001">Lewis&apos;s formula</a></p>
    $$C = S_0 - \frac{\sqrt{S_0K}\,e^{-rT}}{\pi}\int_0^\infty \operatorname{Re}\big[e^{iu\log(S_0/K)}\,\phi(u - \tfrac i2)\big]\,\frac{du}{u^2 + 1/4}.$$
    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is a L&eacute;vy process whose exponent is the average of
    the two regimes&apos; exponents. It is the sum of two independent variance gamma processes, each run at half
    speed, and it is not itself variance gamma unless the regimes agree.</p>

    <h2>Reduction to a linear system</h2>
    <p>Over an interval $dt$ spent in regime $i$ the log return has independent increments, so $e^{iu\log(S_T/S_0)}$ is
    multiplied on average by $e^{g_i\,dt}$ with $g_i$ the L&eacute;vy exponent of the regime. For variance gamma the
    exponent comes from conditioning on the gamma clock. Given the clock increment $dG$, the increment of $X$ is normal
    with mean $\theta_i\,dG$ and variance $\sigma_i^2\,dG$, so</p>
    $$\mathbb{E}\big[e^{iu\,dX} \mid dG\big] = e^{(iu\theta_i - \frac12\sigma_i^2u^2)\,dG},$$
    <p>and the gamma increment has $\mathbb{E}[e^{-z\,dG}] = (1 + \nu_i z)^{-dt/\nu_i}$. Putting
    $z = -iu\theta_i + \tfrac12\sigma_i^2u^2$ gives the exponent</p>
    $$\psi_i(u) = -\frac{1}{\nu_i}\log\big(1 - iu\,\theta_i\nu_i + \tfrac12\sigma_i^2\nu_i\,u^2\big).$$
    <p>Meanwhile the regime jumps at the rates in $Q$. So $\phi_i(u) = a_i(T)$, with no state to factor out,
    $a' = (Q + \operatorname{diag} g)a$, $a(0) = \mathbf 1$ and the constant, complex</p>
    $$g_i = iu\,(r + \omega_i) - \frac{1}{\nu_i}\log\big(1 - iu\,\theta_i\nu_i + \tfrac12\sigma_i^2\nu_i\,u^2\big) .$$
    <p>All three parameters sit inside $g_i$ and none of them touches a Riccati equation, because there is no state.
    That is why all three may switch.</p>
'''
    body += two_state_expansion(r'$\tilde g = \tfrac12\,iu\,(\omega_1 - \omega_2) + \tfrac12\big(\psi_1(u) - \psi_2(u)\big)$',
                                complex_g='here $g$ is complex for every $u$',
                                extra=r''' The L&eacute;vy-exponent difference grows logarithmically, but the martingale
    correction $iu\tilde\omega$ is linear unless $\omega_1=\omega_2$. Here $\tilde\omega = '''
                                + f'{omtilde:.8f}' + r'''$, and at $u=200-i/2$, $|\tilde g| = '''
                                + f'{abs(gbig[1]):.2f}' + r'''$. Thus the pointwise condition $|\tilde g|/\lambda&lt;1$
    gives a frequency window of order $\lambda$, wider than the order-$\sqrt{\lambda}$ window for the $u^2$ forcing
    in <a href="./black-scholes.html">Black&ndash;Scholes</a>, but not an exponentially wide window. The
    characteristic function falls like a power of $u$, not like a Gaussian, so the exact closed form below is used
    along the whole Fourier integral. The asymptotic price columns are finite-$\lambda$ diagnostics, not a uniform
    term-by-term expansion out to $u=600$.''')
    body += r'''
    <h2>The characteristic function in closed form</h2>
    <p>The forcing is constant, so the two-state system can be solved exactly. The characteristic function of the log
    return is</p>
''' + CONSTANT_CARD + r'''    <p>with</p>
    $$\begin{aligned}
    \bar g &= iu\,(r + \bar\omega) + \tfrac12\big(\psi_1(u) + \psi_2(u)\big), \\
    \tilde g &= iu\,\tilde\omega + \tfrac12\big(\psi_1(u) - \psi_2(u)\big), \\
    \psi_i(u) &= -\frac{1}{\nu_i}\log\big(1 - iu\,\theta_i\nu_i + \tfrac12\sigma_i^2\nu_i\,u^2\big), \\
    \omega_i &= \frac{1}{\nu_i}\log\big(1 - \theta_i\nu_i - \tfrac12\sigma_i^2\nu_i\big), \qquad
    \bar\omega, \tilde\omega = \frac{\omega_1 \pm \omega_2}{2} .
    \end{aligned}$$
''' + CONSTANT_PROOF + CONSTANT_SCOPE + r'''    <p>The first line is exact. The second is its expansion in $\varepsilon = 1/\lambda$: the averaged L&eacute;vy
    process, times the Green&ndash;Kubo factor $e^{\varepsilon\tilde g^2T/2}$ and the memory of the starting regime.
    The upper sign is for a start in the first regime.</p>
    <p>Written out, the characteristic function of the averaged process is</p>
    $$e^{\bar gT} = e^{iu(r + \bar\omega)T}\,
      \big(1 - iu\,\theta_1\nu_1 + \tfrac12\sigma_1^2\nu_1u^2\big)^{-T/(2\nu_1)}\,
      \big(1 - iu\,\theta_2\nu_2 + \tfrac12\sigma_2^2\nu_2u^2\big)^{-T/(2\nu_2)},$$
    <p>and the Green&ndash;Kubo factor is $\exp\big(\tfrac18\varepsilon T\,[\,2iu\tilde\omega + \psi_1(u) - \psi_2(u)\,]^2\big)$.</p>

    <h3>A variance-gamma tail certificate</h3>
    <p>For variance gamma the Fourier decay can be bounded directly, uniformly in the switching rate and, in fact,
    uniformly over the generator of any finite-state regime chain. Put</p>
    $$a_i=\tfrac12\sigma_i^2\nu_i,\qquad b_i=\theta_i\nu_i,\qquad
      c_i=1-\tfrac12b_i-\tfrac14a_i,$$
    <p>and assume $c_i&gt;0$ in every regime. On the Lewis line $z=u-i/2$,</p>
    $$\operatorname{Re}\big(1-ib_i z+a_i z^2\big)=c_i+a_i u^2.$$
    <p>Therefore, with</p>
    $$q=\min_i\frac{a_i}{c_i},\qquad \alpha=\min_i\frac1{\nu_i},\qquad
      H=\max_i\left\{\frac{r+\omega_i}{2}-\frac{\log c_i}{\nu_i}\right\},$$
    <p>the real part of every regime exponent obeys</p>
    $$\operatorname{Re}g_i(u-i/2)\leq H-\alpha\log(1+qu^2).$$
    <p>Conditional on the whole regime path, the transform is the exponential of the time integral of the active
    exponent. Taking absolute values before averaging gives</p>
    <div class="equation-card">
    $$|\phi_i(u-i/2)|\leq e^{HT}(1+qu^2)^{-\alpha T},$$
    $$\int_R^\infty\frac{|\phi_i(u-i/2)|}{u^2+1/4}\,du
      \leq\frac{e^{HT}q^{-\alpha T}}{2\alpha T+1}\,R^{-(2\alpha T+1)}.$$
    </div>
    <p>The last line uses $u^2+1/4\geq u^2$ and $1+qu^2\geq qu^2$. Thus an admissible
    $R=O(\lambda)$ window has a tail certificate of order $O(\lambda^{-(2\alpha T+1)})$, rather than the
    model-independent $O(\lambda^{-1})$. For the present parameters $q=0.001426250941$, $\alpha=2$ and
    $H=0.013005504422$, so the generator-uniform tail power is five. This improves the half-moment tail bound but,
    by itself, says nothing about the integrated approximation error on an expanding interior window.</p>

    <h3>A uniform cubic price theorem for this benchmark</h3>
    <p>The pointwise expansion does transfer to a third-order call-price approximation when it is used on a
    parabolic, rather than maximal, frequency window. Set $R_\lambda=c\sqrt\lambda$ for any fixed $c&gt;0$, use the
    quadratic slow multiplier $P_{2,+}$ only on $0\leq u\leq R_\lambda$, and discard the remaining approximate
    integral. Then, for fixed $S_0,K$ and the parameters above,</p>
    $$|C_\lambda-\widehat C_{\lambda,R_\lambda}|=O(\lambda^{-3}).$$
    <p>Here is a direct proof, including the two different tail mechanisms. On the Lewis line put
    $a_i=\sigma_i^2\nu_i/2$, $b_i=\theta_i\nu_i$ and
    $c_i=1-b_i/2-a_i/4&gt;0$. With $q=\min_i a_i/c_i$, the averaged factor obeys</p>
    $$|e^{\bar g(u-i/2)T}|\leq e^{\bar H T}(1+qu^2)^{-\beta},\qquad
      \beta=\frac{T}{2}\left(\frac1{\nu_1}+\frac1{\nu_2}\right)=\frac72.$$
    <p>Also $|\tilde g(u-i/2)|\leq L(1+u)$ for a finite model constant $L$. Write
    $z=\tilde g/\lambda$, $d=\sqrt{1+z^2}$ and note the exact identity</p>
    $$\tilde g\,\frac{d-1}{z}=\frac{\lambda^{-1}\tilde g^2}{d+1}.$$
    <p>On $u\leq c\sqrt\lambda$, $z\to0$ uniformly and the exponent on the right stays bounded. Differentiating
    $A_+(z)\exp\{T\lambda^{-1}\tilde g^2/(d+1)\}$ three times with respect to $\lambda^{-1}$ therefore gives,
    uniformly on this expanding window,</p>
    $$|\phi_\lambda-e^{\bar gT}P_{2,+}(z)|
      \leq C\lambda^{-3}|e^{\bar gT}|(1+|\tilde g|)^6+C e^{-c_0\lambda} |e^{\bar gT}|.$$
    <p>The first term is integrable against $(u^2+1/4)^{-1}du$ uniformly in the upper limit precisely because
    $\beta=7/2&gt;5/2$. Thus the integrated interior error is $O(\lambda^{-3})$.</p>
    <p>For the omitted exact tail, let $L_2$ be the time spent in regime 2. Conditional on a regime path, the
    shifted transform is bounded by</p>
    $$e^{HT}(1+qu^2)^{-\{2T+3L_2\}}.$$
    <p>The event $L_2\geq T/3$ therefore gives power $3T=3$. On its complement, the exact two-state occupation
    transform (the same Feynman&ndash;Kac calculation used by
    <a href="https://doi.org/10.2307/3211908">Pedler (1971)</a>) and a Chernoff bound at parameter $s=\lambda$ give</p>
    $$\Pr(L_2&lt;T/3)\leq \frac12\left(1+\frac3{\sqrt5}\right)
      \exp\left[-\left(\frac76-\frac{\sqrt5}{2}\right)\lambda T\right].$$
    <p>Consequently, for $T=1$ and $R&gt;0$,</p>
    $$\int_R^\infty\frac{|\phi_\lambda(u-i/2)|}{u^2+1/4}\,du
      \leq e^H\left[\frac{q^{-3}}7R^{-7}
      +\Pr(L_2&lt;1/3)\frac{q^{-2}}5R^{-5}\right].$$
    <p>At $R=R_\lambda$ this is $O(\lambda^{-7/2})$ plus an exponentially small term, so it is strictly smaller
    than the cubic interior remainder. The theorem is for this fixed parameter benchmark and this truncated
    approximation; it does not claim a uniform expansion of the characteristic function over all frequencies.</p>
    <p>At $\lambda = 25$, $T = 1$ and the Lewis frequency $u = 1 - i/2$:</p>
''' + table(['quantity', 'value', 'error'], ex_rows) + r'''
    <h2>Results</h2>
    <p>Call prices from the exact characteristic function, and the error of the expansion after each order:</p>
''' + table(['switching rate', 'strike', 'exact', 'order 0', 'order 1', 'order 2'], rows) + r'''    <p>The exact characteristic function agrees with the numerical solution of the two-state system to $10^{-10}$.
    The inversion certificate stops the quadratic slow mode at the largest window satisfying
    $|\tilde g|/\lambda\leq0.35$. For $\lambda=25,50,100,200$, those windows are
    $140.64,289.28,592.38,1199.97$. The generator-uniform VG tail ceilings are
    $5.59\times10^{-5},1.52\times10^{-6},4.22\times10^{-8},1.24\times10^{-9}$, with measured orders
    $5.203,5.170,5.092$. Adding the numerically integrated interior absolute errors gives triangle ceilings
    $6.94\times10^{-5},3.25\times10^{-6},2.62\times10^{-7},2.88\times10^{-8}$, which dominate the observed
    windowed-price errors $7.48\times10^{-6},9.38\times10^{-7},1.17\times10^{-7},1.47\times10^{-8}$.
    For the new analytic cubic theorem, the verifier instead takes $R_\lambda=10\sqrt\lambda$. At
    $\lambda=50,100,200,400$, the certified triangle bounds are
    $3.55\times10^{-4},1.84\times10^{-5},1.39\times10^{-6},1.24\times10^{-7}$ and dominate the observed price
    errors $2.04\times10^{-6},2.10\times10^{-7},1.55\times10^{-8},2.02\times10^{-9}$; the measured orders of the
    interior absolute error are $2.969,2.985,2.992$.</p>
    <p>Certificates:
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/model_pages.py">model_pages.py</a>
    for the prices,
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/verify_constant_forcing.py">verify_constant_forcing.py</a>
    for the analytic envelope, and
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/verify_quantlib_models.py">verify_quantlib_models.py</a>.</p>
    <p>In <a href="https://github.com/microprediction/regimelib">regimelib</a> this model is
    <code>SwitchingVarianceGammaProcess</code>.</p>
'''
    write('variance-gamma.html', 'Variance gamma with switching parameters', body)


# ====================================================================================== Bates
def bates_page():
    kap, ths, xi, rho, v0, S0, T, lam = 2.0, [0.09, 0.02], 0.4, -0.6, 0.04, 100.0, 1.0, 10.0
    ell, mu, de = [2.0, 0.2], -0.08, 0.10
    eps = 1 / lam
    kbar = math.exp(mu + de * de / 2) - 1
    thb, tht = half(ths); lb, lt = half(ell)

    def parts(u):
        b = kap - rho * xi * 1j * u
        d = cmath.sqrt(b * b + xi * xi * (1j * u + u * u))
        gam = (b - d) / (b + d); E = cmath.exp(-d * T)
        D = (b - d) / xi ** 2 * (1 - E) / (1 - gam * E)
        Dp = -0.5 * (u * u + 1j * u) - b * D + 0.5 * xi * xi * D * D
        L = cmath.log((1 - gam * E) / (1 - gam))
        I1 = ((b - d) * T - 2 * L) / xi ** 2
        al = (gam * gam - 1) / gam; be = 2 * gam - 2 - al
        I2 = ((b - d) / xi ** 2) ** 2 * (T + al * L / (gam * d) - be / (gam * d) * (1 / (1 - gam * E) - 1 / (1 - gam)))
        cJ = cmath.exp(1j * u * mu - 0.5 * u * u * de * de) - 1 - 1j * u * kbar
        return dict(d=d, gam=gam, D=D, Dp=Dp, I1=I1, I2=I2, cJ=cJ)

    def closed(u, order, sign=1):
        p = parts(u)
        D, Dp, I1, I2, cJ = p['D'], p['Dp'], p['I1'], p['I2'], p['cJ']
        avg = v0 * D + kap * thb * I1 + lb * cJ * T
        A = kap ** 2 * tht ** 2 * I2 + 2 * kap * tht * lt * cJ * I1 + lt ** 2 * cJ ** 2 * T
        G, G0 = kap * tht * D + lt * cJ, lt * cJ
        if order == 0:
            return cmath.exp(avg)
        if order == 1:
            return cmath.exp(avg + eps / 2 * A) * (1 + sign * eps / 2 * G)
        return cmath.exp(avg + eps / 2 * A - eps * eps / 8 * (G * G + G0 * G0)) * (1 + sign * eps / 2 * G - sign * eps * eps / 4 * kap * tht * Dp)

    def numeric(u, order=None):
        g, gf, D = bates_forcing(u, kap, ths, xi, rho, ell, mu, de, T)
        pre = cmath.exp(D(T) * v0)
        if order is None:
            return pre * numerical_a_callable(T, sym(lam), gf, rtol=1e-12)[0]
        return pre * FastSwitch(sym(lam), g, order=order).a(T, order)[0]
    p = parts(U0)
    ref = numeric(U0)
    A = kap ** 2 * tht ** 2 * p['I2'] + 2 * kap * tht * lt * p['cJ'] * p['I1'] + lt ** 2 * p['cJ'] ** 2 * T
    G, G0 = kap * tht * p['D'] + lt * p['cJ'], lt * p['cJ']
    for o in (0, 1, 2):
        print('Bates order', o, abs(closed(U0, o) - ref), 'engine', abs(numeric(U0, o) - ref))
    assert abs(closed(U0, 2) - ref) < 0.2 * abs(closed(U0, 1) - ref) < 0.04 * abs(closed(U0, 0) - ref)
    q_rows = [['$d$', cz(p['d'], 5)], [r'$\gamma$', cz(p['gam'], 6)],
              ["$D$, $D'$", cz(p['D'], 6) + ', &nbsp; ' + cz(p['Dp'], 6)],
              ['$I_1$, $I_2$', cz(p['I1'], 6) + ', &nbsp; ' + cz(p['I2'], 6)],
              ['$c_J$', cz(p['cJ'], 6)],
              [r'averaged exponent $v_0D + \kappa\bar\theta I_1 + \bar\ell c_JT$', cz(v0 * p['D'] + kap * thb * p['I1'] + lb * p['cJ'] * T, 7)],
              ['$A$', cg(A)], ['$G$, $G_0$', cz(G, 6) + ', &nbsp; ' + cz(G0, 6)],
              [r'first-order exponent $\frac\varepsilon2 A$', cg(eps / 2 * A)],
              [r'second-order exponent $-\frac{\varepsilon^2}8(G^2 + G_0^2)$', cg(-eps * eps / 8 * (G * G + G0 * G0))],
              [r'memory, first order $\frac\varepsilon2 G$', cg(eps / 2 * G)],
              [r"memory, second order $-\frac{\varepsilon^2}4\kappa\tilde\theta D'$", cg(-eps * eps / 4 * kap * tht * p['Dp'])]]
    o_rows = [[lab, cz(closed(U0, o)), e(abs(closed(U0, o) - ref))] for o, lab in ((0, '0 (averaged Bates)'), (1, '1'), (2, '2'))]
    U = 40.0
    us, ws = nodes(U)
    cfs = [([closed(u - 0.5j, o) for o in (0, 1, 2)], numeric(u - 0.5j), numeric(u - 0.5j, 4)) for u in us]

    def price(K, which):
        kk, tot = math.log(S0 / K), 0.0
        for (orders, ex, o4), u, w in zip(cfs, us, ws):
            phi = ex if which is None else (o4 if which == 4 else orders[which])
            tot += w * (cmath.exp(1j * u * kk) * phi).real / (u * u + 0.25)
        return S0 - math.sqrt(S0 * K) / math.pi * tot
    rows = [[f'{K}', f'{price(K, None):.5f}'] + [f'{price(K, o):.5f}' for o in (0, 1, 2, 4)] for K in (80, 90, 100, 110, 120)]

    body = r'''    <h1>Bates with a switching variance level and jump intensity</h1>
    <p class="subtitle">Stochastic volatility with jumps, where a hidden regime sets both the long-run variance and how often the price jumps.</p>

    <h2>The model</h2>
    <p><a href="./bibliography.html#Bates1996">Bates&apos;s model</a> adds Merton&apos;s lognormal jumps to
    Heston&apos;s stochastic variance. Here the long-run variance and the jump intensity are set by a hidden regime
    $y_t \in \{1, 2\}$, a turbulent market and a calm one. With $S_t = e^{X_t}$,</p>
    $$dX_t = \big(-\tfrac12 v_t - \ell_{y_t}\bar k\big)\,dt + \sqrt{v_t}\,dW_t + J\,dN_t, \qquad
      dv_t = \kappa\,(\theta_{y_t} - v_t)\,dt + \xi\sqrt{v_t}\,dZ_t, \qquad d\langle W, Z\rangle = \rho\,dt .$$
    <p>While $y_t = i$ the Poisson process $N$ jumps at rate $\ell_i$, by normal amounts $J$ with mean $\mu_J$ and
    standard deviation $\delta$, and $\bar k = e^{\mu_J + \delta^2/2} - 1$.</p>
    <p>The regime switches at rate $\lambda$ in each direction. The variance reverts over $1/\kappa$; the regime
    changes every $1/\lambda$.</p>
    <p class="muted">Parameters: $\kappa = 2$, $\theta = (0.09, 0.02)$, $\xi = 0.4$, $\rho = -0.6$, $v_0 = 0.04$,
    $\ell = (2, 0.2)$, $\mu_J = -0.08$, $\delta = 0.10$, $S_0 = 100$, zero interest, one year to expiry,
    $\lambda = 10$, and a start in the turbulent regime.</p>

    <h2>The quantity</h2>
    <p>European call prices, through the characteristic function</p>
    $$\phi_i(u) = \mathbb{E}\big[e^{iu(X_T - X_0)} \mid v_0,\ y_0 = i\big]$$
    <p>and
    <a href="./bibliography.html#Lewis2001">Lewis&apos;s formula</a></p>
    $$C = S_0 - \frac{\sqrt{S_0K}}{\pi}\int_0^\infty \operatorname{Re}\big[e^{iu\log(S_0/K)}\,\phi(u - \tfrac i2)\big]\,\frac{du}{u^2 + 1/4}.$$
    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is Bates&apos;s model with long-run variance
    $\bar\theta = \tfrac12(\theta_1 + \theta_2)$ and jump intensity $\bar\ell = \tfrac12(\ell_1 + \ell_2)$, and its
    closed-form characteristic function.</p>

    <h2>Reduction to a linear system</h2>
    <p>The characteristic function $\phi_i(t, v)$, over a remaining time $t$ and from variance $v$, solves the
    Feynman&ndash;Kac equations</p>
    $$\begin{aligned}\partial_t\phi_i = \;&\kappa(\theta_i - v)\,\partial_v\phi_i + \tfrac12\xi^2v\,\partial_{vv}\phi_i
      + iu\rho\xi v\,\partial_v\phi_i - \tfrac12(u^2 + iu)\,v\,\phi_i \\
      &+ \ell_i\big(\phi_J(u) - 1 - iu\bar k\big)\,\phi_i + \sum_j Q_{ij}\,\phi_j, \qquad \phi_i(0, v) = 1 .\end{aligned}$$
    <p>The first line is Heston&apos;s. The jump term is a multiple of $\phi_i$ because a jump in the log price
    multiplies $e^{iuX}$ by $e^{iuJ}$ and leaves the variance alone, with</p>
    $$\phi_J(u) = \exp\big(iu\mu_J - \tfrac12u^2\delta^2\big) .$$
    <p>Try the Heston form $\phi_i = e^{D(t)v}\,a_i(t)$. The terms in $v$ cancel when</p>
    $$D' = -\tfrac12(u^2 + iu) + (\rho\xi iu - \kappa)\,D + \tfrac12\xi^2D^2, \qquad D(0) = 0 .$$
    <p>This Riccati equation involves $\kappa$, $\xi$ and $\rho$ but neither $\theta$ nor $\ell$, so one $D$ serves both
    regimes. Its solution is</p>
    $$D(t) = \frac{\kappa - \rho\xi iu - d}{\xi^2}\;\frac{1 - e^{-dt}}{1 - \gamma e^{-dt}}, \qquad d = \sqrt{(\rho\xi iu - \kappa)^2 + \xi^2(iu + u^2)},$$
    <p>with</p>
    $$\gamma = (\kappa - \rho\xi iu - d)/(\kappa - \rho\xi iu + d).$$
    <p>What remains involves $t$ only. So $\phi_i = e^{D(T)v_0}a_i(T)$ with
    $a' = (Q + \operatorname{diag} g)a$, $a(0) = \mathbf 1$ and the complex</p>
    $$g_i(t) = \kappa\,\theta_i\,D(t) + \ell_i\,c_J, \qquad c_J = \phi_J(u) - 1 - iu\bar k .$$
    <p>The first term is the <a href="./heston.html">Heston</a> forcing and the second the compensated jump exponent of
    the <a href="./merton.html">Merton</a> page. The first vanishes at $t = 0$ and the second does not.</p>
'''
    body += two_state_expansion(r'$\tilde g = \kappa\tilde\theta\,D + \tilde\ell\,c_J, \qquad \tilde\theta = \tfrac12(\theta_1 - \theta_2), \quad \tilde\ell = \tfrac12(\ell_1 - \ell_2)$',
                                complex_g='here $g$ is complex for every $u$',
                                extra='The engine holds the complex $g_i$ as Chebyshev series.')
    body += r'''
    <h2>Explicit formulas</h2>
    <p>For Bates every term up to second order is in closed form. Write</p>
    $$b = \kappa - \rho\xi iu, \qquad d = \sqrt{b^2 + \xi^2(iu + u^2)}, \qquad \gamma = \frac{b - d}{b + d},$$
    <p>so that the regime-free coefficient and its derivative are</p>
    $$D(t) = \frac{b - d}{\xi^2}\;\frac{1 - e^{-dt}}{1 - \gamma e^{-dt}}, \qquad D'(t) = -\tfrac12(u^2 + iu) - b\,D(t) + \tfrac12\xi^2 D(t)^2 .$$
''' + SECOND_ORDER + r'''
    <h3>Step 1: the forcing</h3>
    <p>The forcing is a multiple of $D$ plus a constant, so its average and half-difference are</p>
    $$\bar g(t) = \kappa\,\bar\theta\,D(t) + \bar\ell\,c_J, \qquad \tilde g(t) = \kappa\,\tilde\theta\,D(t) + \tilde\ell\,c_J, \qquad
      \bar\theta, \tilde\theta = \frac{\theta_1 \pm \theta_2}{2}, \quad \bar\ell, \tilde\ell = \frac{\ell_1 \pm \ell_2}{2}.$$
    <p>At the ends, $\tilde g(0) = \tilde\ell\,c_J$ because $D(0) = 0$, and $\tilde g'(T) = \kappa\tilde\theta\,D'(T)$.</p>

    <h3>Step 2: the integrals</h3>
    <p>The averaged exponent needs $\int D$, the Heston integral. With $L(t) = \log\big((1 - \gamma e^{-dt})/(1 - \gamma)\big)$,</p>
    $$I_1 = \int_0^T D(s)\,ds = \frac{1}{\xi^2}\Big((b - d)\,T - 2L(T)\Big).$$
    <p>The first correction needs $\int\tilde g^{\,2}$, and the square has three parts:</p>
    $$\int_0^T\tilde g^{\,2} = \kappa^2\tilde\theta^2\,I_2 + 2\kappa\tilde\theta\,\tilde\ell\,c_J\,I_1 + \tilde\ell^{\,2}c_J^2\,T .$$
    <p>The first is Heston&apos;s, with</p>
    $$I_2 = \int_0^T D(s)^2\,ds = \Big(\frac{b-d}{\xi^2}\Big)^2\Big[T + \frac{\alpha}{\gamma d}\,L(T)
      - \frac{\beta}{\gamma d}\Big(\frac{1}{1 - \gamma e^{-dT}} - \frac{1}{1 - \gamma}\Big)\Big], \qquad
      \alpha = \frac{\gamma^2 - 1}{\gamma}, \quad \beta = 2\gamma - 2 - \alpha .$$
    <p>The last is the fluctuation of the jump count. The middle one is the covariance of the two, which is there
    because the turbulent regime has both the higher variance level and the higher jump intensity. It needs no new
    integral.</p>

    <h3>Step 3: the result</h3>
    <p>Every term, written out in the model&apos;s parameters:</p>
    <div class="equation-card">
    $$\begin{aligned}
    \phi_{1,2}(u) = \;&\exp\Big(v_0\,D + \kappa\bar\theta\,I_1 + \bar\ell\,c_J\,T + \frac{\varepsilon}{2}\,A
      - \frac{\varepsilon^2}{8}\big(G^2 + G_0^2\big)\Big) \\
    &\times\Big(1 \pm \frac{\varepsilon}{2}\,G \mp \frac{\varepsilon^2}{4}\,\kappa\tilde\theta\,D'\Big)
      + O(\varepsilon^3),
    \end{aligned}$$
    $$\begin{aligned}
    A &= \kappa^2\tilde\theta^2\,I_2 + 2\kappa\tilde\theta\,\tilde\ell\,c_J\,I_1 + \tilde\ell^{\,2}c_J^2\,T, \\
    G &= \kappa\tilde\theta\,D + \tilde\ell\,c_J, \qquad G_0 = \tilde\ell\,c_J, \\
    c_J &= e^{iu\mu_J - u^2\delta^2/2} - 1 - iu\,\big(e^{\mu_J + \delta^2/2} - 1\big), \\
    D &= \frac{b - d}{\xi^2}\,\frac{1 - E}{1 - \gamma E}, \\
    D' &= -\tfrac12(u^2 + iu) - b\,D + \tfrac12\xi^2 D^2, \\
    I_1 &= \frac{(b - d)\,T - 2L}{\xi^2}, \\
    I_2 &= \Big(\frac{b - d}{\xi^2}\Big)^2\Big[\,T + \frac{\alpha L}{\gamma d} - \frac{\beta}{\gamma d}\Big(\frac{1}{1 - \gamma E} - \frac{1}{1 - \gamma}\Big)\Big], \\
    L &= \log\frac{1 - \gamma E}{1 - \gamma}, \qquad E = e^{-dT}, \\
    b &= \kappa - \rho\xi iu, \qquad d = \sqrt{b^2 + \xi^2(iu + u^2)}, \qquad \gamma = \frac{b - d}{b + d}, \\
    \alpha &= \frac{\gamma^2 - 1}{\gamma}, \qquad \beta = 2\gamma - 2 - \alpha .
    \end{aligned}$$
    </div>
    <p>Here $D = D(T)$. The first three terms of the exponent are the log characteristic function of Bates&apos;s
    model at the averaged level and intensity. $A$ is the Green&ndash;Kubo correction and the bracket is the memory of
    the starting regime. The upper sign is for a start in regime 1. Setting $\tilde\ell = 0$ returns the
    <a href="./heston.html">Heston</a> card.</p>

    <h3>A worked example</h3>
    <p>With the parameters above, $\varepsilon = 0.1$, a start in the turbulent regime, and the Lewis frequency
    $u = 1 - i/2$:</p>
''' + table(['quantity', 'value'], q_rows) + r'''    <p>Assembling the terms gives the characteristic function at each order, against the numerical solution
    $''' + cz(ref).replace('&minus;', '-').replace(' i', r'\,i') + r'''$:</p>
''' + table(['order', r'$\phi_1(1 - i/2)$', 'error'], o_rows) + r'''    <p>Integrating these characteristic functions over $u$ with Lewis&apos;s formula gives the call prices below.</p>

    <h2>Results</h2>
    <p>Call prices from the numerical characteristic function, from the closed form above at orders 0, 1 and 2, and
    from the engine at order 4:</p>
''' + table(['strike', 'numerical', 'order 0', 'order 1', 'order 2', 'order 4'], rows) + r'''    <p>Certificates:
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/model_pages.py">model_pages.py</a>
    checks the closed form against the numerical solution, and
    <a href="https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/verify_quantlib_models.py">verify_quantlib_models.py</a>
    the convergence orders.</p>
    <p>In <a href="https://github.com/microprediction/regimelib">regimelib</a> this model is
    <code>SwitchingBatesModel</code>.</p>
'''
    write('bates.html', 'Bates with a switching variance level and jump intensity', body)


if __name__ == '__main__':
    merton_page()
    variance_gamma_page()
    bates_page()
    import model_pages2
    model_pages2.main()
