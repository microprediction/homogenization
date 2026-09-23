"""Compute the tables for the fast-switching example pages and write tools/pages/*.html.

`python3 papers/fast-switching/make_pages.py && python3 tools/assemble.py`
"""
import math
import cmath
import os
import sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from scipy.linalg import expm
from fastswitch import FastSwitch, ExpSum, numerical_a, numerical_a_callable
from models import gaussian_factors, cir_switching_mean, vasicek_jumps, mmpp, heston_switching_theta

PAGES = os.path.join(HERE, '..', '..', 'tools', 'pages')
SRC = 'https://github.com/microprediction/homogenization/blob/main/papers/fast-switching/'


def table(head, rows):
    h = ''.join(f'<th>{x}</th>' for x in head)
    b = ''.join('<tr>' + ''.join(f'<td>{x}</td>' for x in r) + '</tr>' for r in rows)
    return f'    <div class="table-wrap"><table class="impl"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table></div>\n'


def e(x):
    return f'{x:.1e}'.replace('e-0', 'e-').replace('e+0', 'e')


def write(name, title, body):
    with open(os.path.join(PAGES, name), 'w') as f:
        f.write(f'<!-- title: {title} · homogenization | math -->\n  <main>\n{body}  </main>\n')
    print('wrote', name)


def error_rows(make_g, t, lams, orders=range(0, 7), state=0, referee=None):
    rows = []
    for lam in lams:
        Q, g, gf = make_g(lam)
        ex = referee(t, Q, g, gf) if referee else numerical_a_callable(t, Q, gf, rtol=1e-13)
        fs = FastSwitch(Q, g, order=max(orders))
        rows.append([f'{lam:g}'] + [e(abs(fs.a(t, o)[state] - ex[state])) for o in orders])
    return rows


def sym(lam):
    return [[-lam, lam], [lam, -lam]]


# ------------------------------------------------------------------ engine
def engine_page():
    body = r'''    <h1>The fast-switching engine</h1>
    <p class="subtitle">One recursion for every model whose state factors out of an exponential-affine formula.</p>

    <p class="lead">Many expectations of the form $\mathbb{E}\big[e^{-\int_0^t c\cdot x_r\,dr}\,\phi(x_t)\big]$ are
    exponential-affine in the state $x$, with coefficients that solve ODEs. When a finite Markov chain $y_t$
    modulates the model but leaves the state-dependent coefficient unchanged, the state factors out and what
    remains is a linear system in time only:</p>
    $$u_i(t,x) = e^{-B(t)\cdot x}\,a_i(t), \qquad a' = \big(Q + \operatorname{diag} g(t)\big)\,a, \qquad a(0) = \mathbf 1 .$$
    <p>The model enters only through the functions $g_i(t)$, one per regime. The table lists them for the
    examples on this site.</p>
'''
    body += table(['Model', 'Switching parameters', 'Time function'], [
        ['<a href="./regime-switching.html">Vasicek</a>, <a href="./three-regimes.html">any number of regimes</a>', 'mean level, volatility', r'$-\kappa\theta_i B + \tfrac12\sigma_i^2 B^2$'],
        ['<a href="./credit.html">Gaussian factors, two-name credit</a>', 'means, volatilities, correlations', r'$-\sum_j \kappa_j\theta_{ji} B_j + \tfrac12\sum_{j,l}\rho_{jl,i}\sigma_{ji}\sigma_{li}B_jB_l$'],
        ['<a href="./cir.html">CIR</a>', 'mean level', r'$-\kappa\theta_i B_{\mathrm{CIR}}$'],
        ['<a href="./jumps.html">Vasicek with jumps</a>', 'mean, volatility, jump intensity', r'$-\kappa\theta_i B + \tfrac12\sigma_i^2B^2 + \ell_i\big(\tfrac{1}{1+mB} - 1\big)$'],
        ['<a href="./counts.html">Poisson counts</a>', 'arrival rate', r'$(z-1)\,\ell_i$'],
        ['<a href="./heston.html">Heston</a>', 'long-run variance', r'$\kappa\theta_i D(t)$, complex'],
    ])
    body += r'''
    <h2>The recursion</h2>
    <p>Write the generator as $Q = Q_0/\varepsilon$ with $\varepsilon$ the mean holding time, let $\pi$ be the
    stationary distribution, and set $s = \pi\cdot a$ and $w = a/s - \mathbf 1$, so that $\pi\cdot w = 0$. Then</p>
    <div class="equation-card">
    $$\begin{aligned}
      \frac{s'}{s} &= \bar g + \pi\cdot(g\,w), \qquad \bar g = \pi\cdot g, \\[4pt]
      \varepsilon\,w' &= Q_0\,w + \varepsilon\,F(w), \qquad F(w) = g\,(\mathbf 1 + w) - (\mathbf 1 + w)\,\big(\bar g + \pi\cdot(g\,w)\big).
    \end{aligned}$$
    </div>
    <p>The outer series $w = \sum_{n\ge1}\varepsilon^n w_n$ follows from $w_n = Q_0^{\#}\big(w_{n-1}' - F_{n-1}\big)$,
    where $Q_0^{\#}$ is the group inverse of $Q_0$ and $F_{n-1}$ collects the terms of $F$ of that order. Each
    step is solvable because $\pi$ annihilates the right-hand side. When every $g_i$ is a sum of exponentials
    $e^{-\alpha t}$ the class is closed under products and derivatives, so every $w_n$ is again such a sum and
    every term of $\log s$ integrates in closed form. Otherwise $g_i$ is held as a Chebyshev series.</p>
    <p>The outer series does not satisfy $w(0) = 0$. The initial layer $\eta = w - w_{\mathrm{outer}}$, in the fast
    time $\tau = t/\varepsilon$, solves $d\eta/d\tau = Q_0\eta + \varepsilon\,[F(w_{\mathrm{outer}}+\eta) - F(w_{\mathrm{outer}})]$
    with $\eta(0) = -w_{\mathrm{outer}}(0)$. It is solved order by order in the eigenbasis of $Q_0$, in the span of
    $\tau^k e^{\mu\tau}$, and it contributes to $\log s$ from order $\varepsilon^2$ or later. The result is
    $a_i = s\,(1 + w_i + \eta_i)$ with error $O(\varepsilon^{N+1})$ after $N$ orders.</p>

    <h2>Code and checks</h2>
    <ul>
      <li><a href="''' + SRC + '''fastswitch.py">fastswitch.py</a>: the engine, for any finite chain, real or complex coefficients.</li>
      <li><a href="''' + SRC + '''models.py">models.py</a>: the $g_i$ for each model, with the state-dependent factor.</li>
      <li><a href="''' + SRC + '''verify_engine.py">verify_engine.py</a>: agreement with the two-state code, and orders 1 to 6 on a three-state chain against a 30-digit numerical solution.</li>
      <li><a href="''' + SRC + '''verify_models.py">verify_models.py</a>: each model&apos;s reduction against Monte Carlo of the switching model, and each expansion&apos;s convergence.</li>
    </ul>
'''
    write('engine.html', 'The fast-switching engine', body)


# ------------------------------------------------------------------ three regimes
def three_regimes_page():
    k = 2.0
    th, s2 = [0.20, 0.08, 0.02], [0.04, 0.01, 0.0025]
    base = np.array([[-3, 2, 1], [1, -2, 1], [0.5, 1.5, -2]], float)

    def G(theta, v):
        return ExpSum({0: -theta + v / (2 * k * k), k: theta - 2 * v / (2 * k * k), 2 * k: v / (2 * k * k)})
    g = [G(th[i], s2[i]) for i in range(3)]
    rows = []
    for sc in (4, 8, 16):
        Q = sc * base
        ex = numerical_a(1.0, Q, g, dps=25)
        fs = FastSwitch(Q, g, order=6)
        rows.append([f'{sc}'] + [e(abs(fs.a(1.0, o)[0] - float(ex[0]))) for o in range(7)])
    pi = FastSwitch(4 * base, g, order=1).pi
    body = r'''    <h1>Three regimes</h1>
    <p class="subtitle">Vasicek with a mean level and volatility that move among three regimes.</p>
    <p>The short rate follows $dx = \kappa(\theta_{y} - x)\,dt + \sigma_y\,dW$ with $\kappa = 2$,
    $\theta = (0.20, 0.08, 0.02)$ and $\sigma^2 = (0.04, 0.01, 0.0025)$. The chain need not be reversible. Here
    its generator is $s\,Q_1$ with</p>
    $$Q_1 = \begin{pmatrix} -3 & 2 & 1 \\ 1 & -2 & 1 \\ 0.5 & 1.5 & -2 \end{pmatrix},$$
    <p>whose stationary distribution is $\pi = (''' + ', '.join(f'{p:.3f}' for p in pi) + r''')$. The averaged
    model is Vasicek&apos;s with the $\pi$-weighted mean level and variance. The table gives the error in the bond
    price $a_1(1)$, starting in the first regime, after each order, against a 30-digit numerical solution.</p>
'''
    body += table(['scale'] + [f'order {o}' for o in range(7)], rows)
    body += r'''    <p>Each doubling of the switching rates divides the order-$n$ error by about $2^{n+1}$. For $n$ regimes the
    recursion is unchanged: the two-state ratio becomes a vector $w$, and each order applies the group inverse
    of the generator once. See <a href="./engine.html">the engine</a>.</p>
'''
    write('three-regimes.html', 'Three regimes', body)


# ------------------------------------------------------------------ two factors and credit
def credit_page():
    kap, th, sg = [1.0, 1.5], [[0.06, 0.01], [0.05, 0.015]], [[0.02, 0.005], [0.02, 0.005]]
    rho = [[[1, 0], [0, 1]]] * 2
    x0, t = [0.03, 0.03], 3.0

    def surv(lam, w, order=None, numeric=False):
        Q = sym(lam)
        g, gf, pref = gaussian_factors(kap, th, sg, rho, w)
        a = numerical_a_callable(t, Q, gf, rtol=1e-13) if numeric else FastSwitch(Q, g, order=6).a(t, order)
        return a[0] * pref(t, x0)

    def dcorr(s1, s2, s12):
        p1, p2 = 1 - s1, 1 - s2
        return (1 - s1 - s2 + s12 - p1 * p2) / math.sqrt(p1 * (1 - p1) * p2 * (1 - p2))
    rows = []
    for lam in (2.0, 4.0, 8.0):
        ex = dcorr(surv(lam, [1, 0], numeric=True), surv(lam, [0, 1], numeric=True), surv(lam, [1, 1], numeric=True))
        vals = [dcorr(surv(lam, [1, 0], o), surv(lam, [0, 1], o), surv(lam, [1, 1], o)) for o in range(5)]
        rows.append([f'{lam:g}', f'{ex:.6f}'] + [f'{v + 0.0:.6f}'.replace('-0.000000', '0') for v in vals])
    body = r'''    <h1>Two factors and two-name credit</h1>
    <p class="subtitle">Default dependence created by a common regime, and where it enters the expansion.</p>
    <p>Two default intensities follow independent Ornstein&ndash;Uhlenbeck processes,
    $dx_j = \kappa_j(\theta_{j,y} - x_j)\,dt + \sigma_{j,y}\,dW_j$, whose mean levels and volatilities switch with a
    common two-state regime. The same reduction prices any positive combination $c_1x_1 + c_2x_2$, which covers
    two-factor Gaussian short rates as well. Survival of each name and of both names is</p>
    $$S_{c}(t) = \mathbb{E}\Big[e^{-\int_0^t (c_1 x_1 + c_2 x_2)}\Big] = e^{-c_1B_1x_1 - c_2B_2x_2}\,a_{c}(t),$$
    <p>with $c = (1,0)$, $(0,1)$ and $(1,1)$. Here $\kappa = (1, 1.5)$, $\theta_1 = (0.06, 0.01)$,
    $\theta_2 = (0.05, 0.015)$, $\sigma = (0.02, 0.005)$ for both names, $x_0 = (0.03, 0.03)$ and $t = 3$.</p>
    <p>The table gives the correlation of the two default indicators, numerically and after each order. The
    averaged model has independent names, so order 0 gives zero. The dependence appears at order $1/\lambda$.
    For a symmetric chain the first-order term is</p>
    $$\log\frac{S_{(1,1)}}{S_{(1,0)}\,S_{(0,1)}} = \frac{1}{\lambda}\int_0^t \tilde g_1(r)\,\tilde g_2(r)\,dr + O(\lambda^{-2}),$$
    <p>where $\tilde g_j = -\kappa_j\tilde\theta_j B_j + \tfrac12\tilde s_j B_j^2$ is each name&apos;s regime
    half-difference. It is positive when the regimes move both names the same way.</p>
'''
    body += table(['switching rate', 'numerical'] + [f'order {o}' for o in range(5)], rows)
    body += r'''    <p>The reduction was checked against Monte Carlo of the switching model in
    <a href="''' + SRC + '''verify_models.py">verify_models.py</a>.</p>
'''
    write('credit.html', 'Two factors and two-name credit', body)


# ------------------------------------------------------------------ Poisson counts
def counts_page():
    rates, t, M = [8.0, 1.0], 1.0, 64
    zs = np.exp(2j * np.pi * np.arange(M) / M)
    rows, moment_rows = [], []
    for lam in (5.0, 10.0, 20.0, 40.0):
        Q = np.array(sym(lam))
        exact = np.real(np.fft.fft([(expm((Q + np.diag([(z - 1) * r for r in rates])) * t) @ np.ones(2))[0] for z in zs])) / M
        errs = []
        for o in range(5):
            approx = np.real(np.fft.fft([FastSwitch(Q, mmpp(rates, z)[0], order=4).a(t, o)[0] for z in zs])) / M
            errs.append(np.abs(approx - exact).max())
        rows.append([f'{lam:g}'] + [e(x) for x in errs])
        k = np.arange(M)
        mean = (k * exact).sum()
        var = (k * k * exact).sum() - mean ** 2
        moment_rows.append([f'{lam:g}', f'{mean:.4f}', f'{var:.4f}', f'{var / mean:.4f}'])
    body = r'''    <h1>Poisson counts with a switching rate</h1>
    <p class="subtitle">A Markov-modulated Poisson process, and its distribution to any order.</p>
    <p>In a <a href="./bibliography.html#FischerMeierHellstern1993">Markov-modulated Poisson process</a> events arrive at rate $\ell_y$, with $\ell = (8, 1)$ per unit time and a symmetric chain switching at rate
    $\lambda$. The generating function $a_i(t) = \mathbb{E}[z^{N_t}\mid y_0 = i]$ solves
    $a' = \big(Q + (z-1)\operatorname{diag}\ell\big)a$, so $g_i = (z-1)\ell_i$ is constant and complex. Evaluating
    the expansion at $z$ on the unit circle and inverting by the discrete Fourier transform gives every
    probability $\Pr(N_t = k)$ at once.</p>
    <p>The averaged model is Poisson with rate $\bar\ell = 4.5$. The corrector carries the overdispersion. The
    first table gives the moments of $N_1$ starting in the high-rate regime, from the numerical solution.</p>
'''
    body += table(['switching rate', 'mean', 'variance', 'variance / mean'], moment_rows)
    body += '    <p>The second table gives the largest error over all probabilities after each order.</p>\n'
    body += table(['switching rate'] + [f'order {o}' for o in range(5)], rows)
    body += r'''    <p>The same calculation applies to claim counts, defaults in a portfolio, and arrivals of any kind whose rate
    follows a hidden regime.</p>
'''
    write('counts.html', 'Poisson counts with a switching rate', body)


# ------------------------------------------------------------------ CIR
def cir_page():
    kap, ths, sig, t = 1.5, [0.08, 0.02], 0.15, 3.0

    def mk(lam):
        g, gf, pref, B = cir_switching_mean(kap, ths, sig, 5.0)
        return sym(lam), g, gf
    rows = error_rows(mk, t, (5.0, 10.0, 20.0, 40.0))
    body = r'''    <h1>CIR with a switching mean level</h1>
    <p class="subtitle">A square-root intensity whose long-run level follows a regime.</p>
    <p>The intensity follows $dx = \kappa(\theta_y - x)\,dt + \sigma\sqrt{x}\,dW$. The coefficient of $x$ in the
    bond price solves $B' = 1 - \kappa B - \tfrac12\sigma^2 B^2$, which does not involve $\theta$, so the state
    factors out whenever only the mean level switches:</p>
    $$B(t) = \frac{2\,(e^{ht} - 1)}{(h+\kappa)(e^{ht}-1) + 2h}, \quad h = \sqrt{\kappa^2 + 2\sigma^2}, \qquad g_i(t) = -\kappa\,\theta_i\,B(t).$$
    <p>Here $\kappa = 1.5$, $\theta = (0.08, 0.02)$ and $\sigma = 0.15$. The table gives the error in $a_1(3)$
    after each order. $B$ is not a sum of exponentials, so the engine holds $g_i$ as a Chebyshev series.</p>
'''
    body += table(['switching rate'] + [f'order {o}' for o in range(7)], rows)
    body += r'''    <p>If $\sigma$ also switched, $B$ would differ between regimes and the state would no longer factor out.</p>
'''
    write('cir.html', 'CIR with a switching mean level', body)


# ------------------------------------------------------------------ jumps
def jumps_page():
    args = (2.0, [0.05, 0.02], [0.02, 0.01], [3.0, 0.2], 0.03, 5.0)

    def mk(lam):
        g, gf, pref = vasicek_jumps(*args)
        return sym(lam), g, gf
    rows = error_rows(mk, 3.0, (5.0, 10.0, 20.0, 40.0))
    body = r'''    <h1>Vasicek with jumps at a switching intensity</h1>
    <p class="subtitle">Upward jumps in the intensity that arrive more often in one regime.</p>
    <p>The intensity follows $dx = \kappa(\theta_y - x)\,dt + \sigma_y\,dW + dJ$, where $J$ jumps at rate $\ell_y$
    by exponential amounts of mean $m$. Jumps change only the time function:</p>
    $$g_i(t) = -\kappa\theta_i B + \tfrac12\sigma_i^2 B^2 + \ell_i\Big(\frac{1}{1 + m B} - 1\Big), \qquad B = \frac{1 - e^{-\kappa t}}{\kappa}.$$
    <p>Here $\kappa = 2$, $\theta = (0.05, 0.02)$, $\sigma = (0.02, 0.01)$, $\ell = (3, 0.2)$ and $m = 0.03$. The
    table gives the error in $a_1(3)$ after each order.</p>
'''
    body += table(['switching rate'] + [f'order {o}' for o in range(7)], rows)
    write('jumps.html', 'Vasicek with jumps at a switching intensity', body)


# ------------------------------------------------------------------ Heston
def heston_page():
    kap, ths, xi, rho, T, v0, S0, lam = 2.0, [0.09, 0.02], 0.4, -0.6, 1.0, 0.04, 100.0, 10.0
    Q = sym(lam)
    us, wts = np.polynomial.legendre.leggauss(96)
    U = 40.0
    us, wts = (us + 1) * U / 2, wts * U / 2
    cfs = []
    for u in us:
        uc = u - 0.5j
        g, gf, D = heston_switching_theta(uc, kap, ths, xi, rho, T)
        pre = cmath.exp(D(T) * v0)
        fs = FastSwitch(Q, g, order=4)
        cfs.append(([pre * fs.a(T, o)[0] for o in range(5)], pre * numerical_a_callable(T, Q, gf, rtol=1e-12)[0]))

    def price(K, which):
        kk, tot = math.log(S0 / K), 0.0
        for (orders, ex), u, w in zip(cfs, us, wts):
            phi = ex if which is None else orders[which]
            tot += w * (cmath.exp(1j * u * kk) * phi).real / (u * u + 0.25)
        return S0 - math.sqrt(S0 * K) / math.pi * tot
    rows = [[f'{K}', f'{price(K, None):.5f}'] + [f'{price(K, o):.5f}' for o in (0, 1, 2, 4)] for K in (80, 90, 100, 110, 120)]
    body = r'''    <h1>Heston with a switching long-run variance</h1>
    <p class="subtitle">Option prices by Fourier inversion, with the characteristic function expanded in $1/\lambda$.</p>
    <p>In <a href="./bibliography.html#Heston1993">Heston&apos;s model</a> the log-price follows $dX = -\tfrac12 v\,dt + \sqrt v\,dW$ and the variance
    $dv = \kappa(\theta_y - v)\,dt + \xi\sqrt v\,dZ$, with correlation $\rho$. The characteristic function is
    $\mathbb{E}[e^{iuX_T}] = e^{iuX_0 + D(T)v_0}\,a_i(T)$, where $D$ is Heston&apos;s Riccati solution, which does
    not involve $\theta$. The regime enters only through</p>
    $$g_i(t) = \kappa\,\theta_i\,D(t), \qquad D(t) = \frac{\kappa - \rho\xi iu - d}{\xi^2}\;\frac{1 - e^{-dt}}{1 - \gamma e^{-dt}},$$
    <p>with $d = \sqrt{(\rho\xi iu - \kappa)^2 + \xi^2(iu + u^2)}$ and $\gamma = (\kappa - \rho\xi iu - d)/(\kappa - \rho\xi iu + d)$.
    Call prices follow from <a href="./bibliography.html#Lewis2001">Lewis&apos;s formula</a>,
    $C = S_0 - \frac{\sqrt{S_0K}}{\pi}\int_0^\infty \operatorname{Re}\big[e^{iu\log(S_0/K)}\,\phi(u - \tfrac i2)\big]\,\frac{du}{u^2 + 1/4}$.</p>
    <p>Here $\kappa = 2$, $\theta = (0.09, 0.02)$, $\xi = 0.4$, $\rho = -0.6$, $v_0 = 0.04$, $S_0 = 100$, one year to
    expiry, a symmetric chain switching at $\lambda = 10$, and a start in the high-variance regime. The table gives
    call prices from the numerical characteristic function and after each order.</p>
'''
    body += table(['strike', 'numerical', 'order 0', 'order 1', 'order 2', 'order 4'], rows)
    body += r'''    <p>A Monte Carlo simulation of the switching model, with 200,000 paths, gives 22.13, 8.41 and 1.83 at strikes 80,
    100 and 120, each within its standard error of the numerical prices. The same construction applies to any
    affine stochastic volatility model in which only the long-run level switches.</p>
'''
    write('heston.html', 'Heston with a switching long-run variance', body)


if __name__ == '__main__':
    engine_page()
    three_regimes_page()
    credit_page()
    counts_page()
    cir_page()
    jumps_page()
    heston_page()
