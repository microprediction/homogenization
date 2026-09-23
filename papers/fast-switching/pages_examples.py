"""Self-contained example pages for the fast-switching engine (imported by make_pages.py).

Every page: the model with its two time scales, what is computed, averaging, the reduction to a linear system,
the expansion, and results with their checks.
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
from options import bs_call, zcb_call

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


def sym(lam):
    return [[-lam, lam], [lam, -lam]]


def error_rows(make, t, lams, orders=range(0, 7), state=0):
    rows = []
    for lam in lams:
        Q, g, gf = make(lam)
        ex = numerical_a_callable(t, Q, gf, rtol=1e-13)
        fs = FastSwitch(Q, g, order=max(orders))
        rows.append([f'{lam:g}'] + [e(abs(fs.a(t, o)[state] - ex[state])) for o in orders])
    return rows


def two_state_expansion(g_desc, extra=''):
    """The expansion paragraph for a symmetric two-state chain switching at rate lambda."""
    return r'''    <h2>The expansion</h2>
    <p>Let the regime switch at rate $\lambda$ in each direction and write $\varepsilon = 1/\lambda$,
    $\bar g = \tfrac12(g_1 + g_2)$ and $\tilde g = \tfrac12(g_1 - g_2)$, where ''' + g_desc + r'''. To first order,</p>
    $$a_i(t) = \exp\Big(\int_0^t \bar g + \frac{\varepsilon}{2}\int_0^t \tilde g^{\,2}\Big)\Big(1 \pm \frac{\varepsilon}{2}\,\tilde g(t)\Big) + O(\varepsilon^2),$$
    <p>with the upper sign for regime 1. The factor $e^{\int\bar g}$ is the averaged model.</p>
    <p>The term $\frac\varepsilon2\int\tilde g^2$ is shared by both regimes and grows with $t$. The bracket carries
    the starting regime and fades as $\lambda$ grows.</p>
    <p>Every higher order follows from one equation. With $m = \tfrac12(a_1 + a_2)$ and
    $\omega = (a_1 - a_2)/(a_1 + a_2)$,</p>
    $$\omega' = \tilde g\,(1 - \omega^2) - 2\lambda\,\omega, \quad \omega(0) = 0, \qquad \log m(t) = \int_0^t (\bar g + \tilde g\,\omega), \qquad a_{1,2} = m\,(1 \pm \omega).$$
    <p>Expanding $\omega = \sum_n \varepsilon^n\omega_n$ gives $\omega_1 = \tilde g/2$ and</p>
    $$\omega_{n+1} = -\tfrac12\big(\omega_n' + \tilde g\sum_{i+j=n}\omega_i\omega_j\big).$$
    <p>This series does not meet
    $\omega(0) = 0$, so an initial layer of width about $1/(2\lambda)$ is added and solved order by order in the
    same way. The <a href="./engine.html">engine</a> page gives the version for any number of regimes.</p>
''' + (('    <p>' + extra.strip() + '</p>\n') if extra else '')


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
        rows.append([f'{sc}'] + [e(abs(fs.a(1.0, o)[0] - ex[0])) for o in range(7)])
    pi = FastSwitch(4 * base, g, order=1).pi
    pis = ', '.join(f'{p:.3f}' for p in pi)
    body = r'''    <h1>Three regimes</h1>
    <p class="subtitle">A short rate whose mean level and volatility move among three regimes that switch quickly.</p>

    <h2>The model</h2>
    <p>The short rate $x_t$ is an Ornstein&ndash;Uhlenbeck process whose parameters are set by a hidden regime
    $y_t \in \{1, 2, 3\}$:</p>
    $$dx_t = \kappa\,(\theta_{y_t} - x_t)\,dt + \sigma_{y_t}\,dW_t .$$
    <p>The regime is a continuous-time Markov chain with generator $s\,Q_1$,</p>
    $$Q_1 = \begin{pmatrix} -3 & 2 & 1 \\ 1 & -2 & 1 \\ 0.5 & 1.5 & -2 \end{pmatrix},$$
    <p>so that from regime 1 it jumps to regime 2 at rate $2s$ and to regime 3 at rate $s$, and so on. The chain is
    not reversible.</p>
    <p>There are two time scales: the rate reverts towards its current level over $1/\kappa$, and the
    regime changes every $1/s$ or so. Switching is fast when $s \gg \kappa$.</p>
    <p class="muted">Parameters: $\kappa = 2$,
    $\theta = (0.20, 0.08, 0.02)$ and $\sigma^2 = (0.04, 0.01, 0.0025)$.</p>

    <h2>What is computed</h2>
    <p>The zero-coupon bond price</p>
    $$u_i(t,x) = \mathbb{E}\big[e^{-\int_0^t x_r\,dr} \mid x_0 = x,\ y_0 = i\big]$$
    <p>for
    maturity $t$, which is also the survival probability when $x$ is a default intensity.</p>

    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the rate sees only the stationary mixture. The chain&apos;s stationary
    distribution is $\pi = (''' + pis + r''')$, and the limit is Vasicek&apos;s model with mean level
    $\bar\theta = \pi\cdot\theta$ and variance $\bar s = \pi\cdot\sigma^2$.</p>

    <h2>Reduction to a linear system</h2>
    <p>Because $\kappa$ is the same in every regime, $u_i(t,x) = e^{-B(t)x}a_i(t)$ with
    $B(t) = (1 - e^{-\kappa t})/\kappa$, and</p>
    $$a' = \big(s\,Q_1 + \operatorname{diag} g(t)\big)\,a, \quad a(0) = \mathbf 1, \qquad g_i(t) = -\kappa\theta_i B(t) + \tfrac12\sigma_i^2 B(t)^2 .$$

    <h2>The expansion</h2>
    <p>Write $Q = s\,Q_1$ and $\bar g = \pi\cdot g$, and let $Q^{\#}$ be the group inverse of $Q$: the matrix that
    inverts $Q$ on vectors with $\pi\cdot v = 0$ and returns such vectors. To first order in $1/s$,</p>
    $$a_i(t) \approx \exp\Big(\int_0^t \big[\bar g - \pi\cdot\big(g\,Q^{\#}(g - \bar g)\big)\big]\Big)\,\Big(1 - \big[Q^{\#}(g - \bar g)\big]_i(t)\Big).$$
    <p>Higher orders follow from one recursion. With $s_\pi = \pi\cdot a$ and $w = a/s_\pi - \mathbf 1$, each order of
    $w$ applies $Q^{\#}$ once to terms already known, and an initial layer, solved in the eigenbasis of $Q$,
    restores $a(0) = \mathbf 1$. The <a href="./engine.html">engine</a> page gives the recursion in full.</p>

    <h2>Results</h2>
    <p>The error in $a_1(1)$, starting in regime 1, after each order, against a 30-digit numerical solution:</p>
'''
    body += table(['scale s'] + [f'order {o}' for o in range(7)], rows)
    body += r'''    <p>Each doubling of $s$ divides the order-$n$ error by about $2^{n+1}$.</p>
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
    <p class="subtitle">Default dependence created by a common hidden regime, and where it enters the expansion.</p>

    <h2>The model</h2>
    <p>Two companies have default intensities $x_1$ and $x_2$. Each follows its own Ornstein&ndash;Uhlenbeck process,
    driven by independent Brownian motions, but both take their parameters from one hidden regime
    $y_t \in \{1, 2\}$, a credit cycle:</p>
    $$dx_{j} = \kappa_j\,(\theta_{j,y_t} - x_j)\,dt + \sigma_{j,y_t}\,dW_j, \qquad j = 1, 2 .$$
    <p>The regime switches at rate $\lambda$ in each direction.</p>
    <p>There are two time scales: each intensity reverts
    over $1/\kappa_j$, and the cycle turns every $1/\lambda$.</p>
    <p class="muted">Parameters: $\kappa = (1, 1.5)$, the first company has
    $\theta_1 = (0.06, 0.01)$, the second $\theta_2 = (0.05, 0.015)$, both have $\sigma = (0.02, 0.005)$ in the two
    regimes, and $x_0 = (0.03, 0.03)$.</p>
    <p>The same algebra prices two-factor Gaussian short rates.</p>

    <h2>What is computed</h2>
    <p>Survival of each company and of both, to time $t = 3$,</p>
    $$S_c(t) = \mathbb{E}\Big[e^{-\int_0^t (c_1 x_1 + c_2 x_2)\,dr}\Big], \qquad c = (1,0),\ (0,1),\ (1,1),$$
    <p>and from them the correlation of the two default indicators,
    $(p_{12} - p_1p_2)/\sqrt{p_1(1-p_1)\,p_2(1-p_2)}$, with $p_1 = 1 - S_{(1,0)}$, $p_2 = 1 - S_{(0,1)}$ and
    $p_{12} = 1 - S_{(1,0)} - S_{(0,1)} + S_{(1,1)}$.</p>

    <h2>Averaging</h2>
    <p>When the cycle turns infinitely fast each intensity is a Vasicek process with averaged parameters, and the two
    are independent. The averaged model has no default correlation at all.</p>

    <h2>Reduction to a linear system</h2>
    <p>With $B_j(t) = c_j(1 - e^{-\kappa_j t})/\kappa_j$, $S_c = e^{-B_1x_1 - B_2x_2}\,a(t)$ and</p>
    $$a' = \big(Q + \operatorname{diag} g(t)\big)\,a, \quad a(0) = \mathbf 1, \qquad g_i = \sum_j\big(-\kappa_j\theta_{j,i}B_j + \tfrac12\sigma_{j,i}^2B_j^2\big).$$
'''
    body += two_state_expansion(r'$g$ is the function above for the chosen $c$',
                                r''' For two names the first-order term gives the dependence directly. The function $g$ for $c = (1,1)$ is the
    sum of the single-name functions, so $\tilde g_{(1,1)} = \tilde g_1 + \tilde g_2$ and</p>
    $$\log\frac{S_{(1,1)}}{S_{(1,0)}\,S_{(0,1)}} = \frac{1}{\lambda}\int_0^t \tilde g_1(r)\,\tilde g_2(r)\,dr + O(\lambda^{-2}),$$
    <p>where $\tilde g_j$ is company $j$&apos;s regime half-difference. It is positive when the cycle moves both
    companies the same way.''')
    body += r'''
    <h2>Results</h2>
    <p>The correlation of the default indicators, numerically and after each order:</p>
'''
    body += table(['switching rate', 'numerical'] + [f'order {o}' for o in range(5)], rows)
    body += r'''    <p>Order 0 is exactly zero, and order 1 already has most of the dependence. A Monte Carlo simulation of the
    switching model agrees with the numerical survival probabilities; see
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
    <p class="subtitle">Events whose arrival rate follows a hidden regime, and the distribution of their count.</p>

    <h2>The model</h2>
    <p>Events arrive one at a time, as insurance claims, defaults in a portfolio or requests to a server do.</p>
    <p>Their rate depends on a hidden regime $y_t \in \{1, 2\}$: while $y_t = i$ they arrive as a Poisson process with rate
    $\ell_i$.</p>
    <p>The regime switches at rate $\lambda$ in each direction. This is a
    <a href="./bibliography.html#FischerMeierHellstern1993">Markov-modulated Poisson process</a>.</p>
    <p>There are two time scales: events arrive every $1/\ell_i$ on average, and the regime changes every $1/\lambda$.</p>
    <p class="muted">Parameters: $\ell = (8, 1)$ per unit time, the process starts in the busy regime, and $N_t$ counts the events up to
    $t = 1$.</p>

    <h2>What is computed</h2>
    <p>The whole distribution $\Pr(N_t = k)$, $k = 0, 1, 2, \dots$, through the generating function</p>
    $$a_i(t) = \mathbb{E}\big[z^{N_t} \mid y_0 = i\big].$$
    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast, events arrive as a Poisson process with the average rate
    $\bar\ell = 4.5$, and the variance of the count equals its mean. Switching at a finite rate makes the count
    more variable than Poisson, and the corrector carries that overdispersion.</p>

    <h2>Reduction to a linear system</h2>
    <p>An event multiplies $z^{N}$ by $z$, so</p>
    $$a' = \big(Q + (z-1)\operatorname{diag}\ell\big)\,a, \qquad a(0) = \mathbf 1, \qquad g_i = (z-1)\,\ell_i ,$$
    <p>a constant, complex for complex $z$. Evaluating $a(t)$ at the 64 points $z = e^{2\pi i k/64}$ on the unit circle
    and applying the discrete Fourier transform returns all the probabilities at once.</p>
'''
    body += two_state_expansion(r'$\bar g = (z-1)\bar\ell$ and $\tilde g = \tfrac12(z-1)(\ell_1 - \ell_2)$ are constants',
                                r''' For the count, the shared first-order term multiplies the generating function by</p>
    $$\exp\big(\tfrac{t}{8\lambda}(z-1)^2(\ell_1 - \ell_2)^2\big),$$
    <p>which adds $t(\ell_1 - \ell_2)^2/(4\lambda)$ to the
    variance.''')
    body += r'''
    <h2>Results</h2>
    <p>Moments of $N_1$, starting in the busy regime, from the numerical solution:</p>
'''
    body += table(['switching rate', 'mean', 'variance', 'variance / mean'], moment_rows)
    body += '    <p>The largest error over all probabilities after each order:</p>\n'
    body += table(['switching rate'] + [f'order {o}' for o in range(5)], rows)
    write('counts.html', 'Poisson counts with a switching rate', body)


# ------------------------------------------------------------------ CIR
def cir_page():
    kap, ths, sig, t = 1.5, [0.08, 0.02], 0.15, 3.0

    def mk(lam):
        g, gf, pref, B = cir_switching_mean(kap, ths, sig, 5.0)
        return sym(lam), g, gf
    rows = error_rows(mk, t, (5.0, 10.0, 20.0, 40.0))
    body = r'''    <h1>CIR with a switching mean level</h1>
    <p class="subtitle">A square-root intensity whose long-run level follows a hidden regime.</p>

    <h2>The model</h2>
    <p>A default intensity, or short rate, $x_t$ follows a square-root (CIR) process whose long-run level is set by a
    hidden regime $y_t \in \{1, 2\}$:</p>
    $$dx_t = \kappa\,(\theta_{y_t} - x_t)\,dt + \sigma\sqrt{x_t}\,dW_t .$$
    <p>The volatility parameter $\sigma$ is the same in both regimes.</p>
    <p>The regime switches at rate $\lambda$ in each direction. The intensity reverts over $1/\kappa$; the regime changes every $1/\lambda$.</p>
    <p class="muted">Parameters: $\kappa = 1.5$,
    $\theta = (0.08, 0.02)$ and $\sigma = 0.15$.</p>

    <h2>What is computed</h2>
    <p>The survival probability, or bond price,</p>
    $$u_i(t,x) = \mathbb{E}\big[e^{-\int_0^t x_r\,dr} \mid x_0 = x,\ y_0 = i\big].$$
    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is the CIR model with mean level
    $\bar\theta = \tfrac12(\theta_1 + \theta_2)$ and its classical bond price.</p>

    <h2>Reduction to a linear system</h2>
    <p>The coefficient of $x$ in the CIR bond price solves $B' = 1 - \kappa B - \tfrac12\sigma^2 B^2$, $B(0) = 0$,
    which does not involve $\theta$:</p>
    $$B(t) = \frac{2\,(e^{ht} - 1)}{(h+\kappa)(e^{ht}-1) + 2h}, \qquad h = \sqrt{\kappa^2 + 2\sigma^2}.$$
    <p>So $u_i = e^{-B(t)x}a_i(t)$ with $a' = (Q + \operatorname{diag} g)a$, $a(0) = \mathbf 1$ and
    $g_i(t) = -\kappa\,\theta_i\,B(t)$. If $\sigma$ also switched, $B$ would differ between regimes and the state
    would not factor out.</p>
'''
    body += two_state_expansion(r'$g_i = -\kappa\theta_i B$, so that $\tilde g = -\kappa\tilde\theta B$ with $\tilde\theta = \tfrac12(\theta_1 - \theta_2)$',
                                r''' $B$ is not a sum of exponentials, so the engine holds $g_i$ as a Chebyshev series.''')
    body += r'''
    <h2>Results</h2>
    <p>The error in $a_1(3)$ after each order, against a numerical solution:</p>
'''
    body += table(['switching rate'] + [f'order {o}' for o in range(7)], rows)
    body += r'''    <p>A Monte Carlo simulation of the switching CIR model agrees with the numerical solution; see
    <a href="''' + SRC + '''verify_models.py">verify_models.py</a>.</p>
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
    <p class="subtitle">An intensity that jumps upward, more often in one regime than the other.</p>

    <h2>The model</h2>
    <p>A default intensity $x_t$ mean-reverts, diffuses, and jumps upward. The jumps arrive at a rate set by a hidden
    regime $y_t \in \{1, 2\}$, as do the mean level and volatility:</p>
    $$dx_t = \kappa\,(\theta_{y_t} - x_t)\,dt + \sigma_{y_t}\,dW_t + dJ_t .$$
    <p>While $y_t = i$, $J$ jumps at rate $\ell_i$ by exponentially distributed amounts with mean $m$.</p>
    <p>The regime switches at rate $\lambda$ in each direction. The intensity reverts over $1/\kappa$ and the regime changes every
    $1/\lambda$.</p>
    <p class="muted">Parameters: $\kappa = 2$, $\theta = (0.05, 0.02)$, $\sigma = (0.02, 0.01)$, $\ell = (3, 0.2)$ and
    $m = 0.03$: the first regime is a stressed one, with frequent jumps.</p>

    <h2>What is computed</h2>
    <p>The survival probability</p>
    $$u_i(t,x) = \mathbb{E}\big[e^{-\int_0^t x_r\,dr} \mid x_0 = x,\ y_0 = i\big].$$
    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is a Vasicek intensity with averaged mean and variance,
    plus jumps at the average rate $\bar\ell = \tfrac12(\ell_1 + \ell_2)$.</p>

    <h2>Reduction to a linear system</h2>
    <p>A jump of size $J$ multiplies $e^{-Bx}$ by $e^{-BJ}$, whose mean is $1/(1 + mB)$, so the jumps change only the
    time function. With $B(t) = (1 - e^{-\kappa t})/\kappa$, $u_i = e^{-B(t)x}a_i(t)$,
    $a' = (Q + \operatorname{diag} g)a$, $a(0) = \mathbf 1$ and</p>
    $$g_i(t) = -\kappa\theta_i B + \tfrac12\sigma_i^2 B^2 + \ell_i\Big(\frac{1}{1 + m B} - 1\Big).$$
'''
    body += two_state_expansion(r'$g_i$ is the function above',
                                r''' The jump term is not a sum of exponentials, so the engine holds $g_i$ as a Chebyshev series.''')
    body += r'''
    <h2>Results</h2>
    <p>The error in $a_1(3)$ after each order, against a numerical solution:</p>
'''
    body += table(['switching rate'] + [f'order {o}' for o in range(7)], rows)
    body += r'''    <p>A Monte Carlo simulation of the model with jumps agrees with the numerical solution; see
    <a href="''' + SRC + '''verify_models.py">verify_models.py</a>.</p>
'''
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
    <p class="subtitle">Stochastic volatility whose long-run level follows a hidden regime, and the option prices it implies.</p>

    <h2>The model</h2>
    <p>A stock price $S_t = e^{X_t}$ has stochastic variance $v_t$, as in
    <a href="./bibliography.html#Heston1993">Heston&apos;s model</a>, except that the level to which the variance
    reverts is set by a hidden regime $y_t \in \{1, 2\}$, a calm market and a turbulent one:</p>
    $$dX_t = -\tfrac12 v_t\,dt + \sqrt{v_t}\,dW_t, \qquad dv_t = \kappa\,(\theta_{y_t} - v_t)\,dt + \xi\sqrt{v_t}\,dZ_t, \qquad d\langle W, Z\rangle = \rho\,dt .$$
    <p>The regime switches at rate $\lambda$ in each direction. The variance reverts over $1/\kappa$; the regime
    changes every $1/\lambda$.</p>
    <p class="muted">Parameters: $\kappa = 2$, $\theta = (0.09, 0.02)$, $\xi = 0.4$, $\rho = -0.6$, $v_0 = 0.04$,
    $S_0 = 100$, zero interest, one year to expiry, $\lambda = 10$, and a start in the turbulent regime.</p>

    <h2>What is computed</h2>
    <p>European call prices, through the characteristic function</p>
    $$\phi_i(u) = \mathbb{E}\big[e^{iu(X_T - X_0)} \mid v_0,\ y_0 = i\big]$$
    <p>and
    <a href="./bibliography.html#Lewis2001">Lewis&apos;s formula</a></p>
    $$C = S_0 - \frac{\sqrt{S_0K}}{\pi}\int_0^\infty \operatorname{Re}\big[e^{iu\log(S_0/K)}\,\phi(u - \tfrac i2)\big]\,\frac{du}{u^2 + 1/4}.$$
    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is Heston&apos;s model with long-run variance
    $\bar\theta = \tfrac12(\theta_1 + \theta_2)$ and its closed-form characteristic function.</p>

    <h2>Reduction to a linear system</h2>
    <p>Heston&apos;s characteristic function is $e^{C(T) + D(T)v_0}$, where $D$ solves a Riccati equation that involves
    $\kappa$, $\xi$ and $\rho$ but not $\theta$:</p>
    $$D(t) = \frac{\kappa - \rho\xi iu - d}{\xi^2}\;\frac{1 - e^{-dt}}{1 - \gamma e^{-dt}}, \qquad d = \sqrt{(\rho\xi iu - \kappa)^2 + \xi^2(iu + u^2)},$$
    <p>with</p>
    $$\gamma = (\kappa - \rho\xi iu - d)/(\kappa - \rho\xi iu + d).$$
    <p>So $\phi_i = e^{D(T)v_0}a_i(T)$ with
    $a' = (Q + \operatorname{diag} g)a$, $a(0) = \mathbf 1$ and $g_i(t) = \kappa\,\theta_i\,D(t)$, which is complex.</p>
'''
    body += two_state_expansion(r'$g_i = \kappa\theta_i D$, so that $\tilde g = \kappa\tilde\theta D$',
                                r''' The engine holds the complex $g_i$ as Chebyshev series.''')
    body += r'''
    <h2>Results</h2>
    <p>Call prices from the numerical characteristic function and after each order:</p>
'''
    body += table(['strike', 'numerical', 'order 0', 'order 1', 'order 2', 'order 4'], rows)
    body += r'''    <p>A Monte Carlo simulation of the switching model, with 200,000 paths, gives 22.13, 8.41 and 1.83 at strikes 80,
    100 and 120, each within its standard error of the numerical prices.</p>
'''
    write('heston.html', 'Heston with a switching long-run variance', body)


# ------------------------------------------------------------------ Black-Scholes
def black_scholes_page():
    sig, r, T, S0 = [0.30, 0.15], 0.03, 1.0, 100.0
    rows = []
    for lam in (25.0, 50.0, 100.0):
        Q = sym(lam)
        for K in (90, 110):
            num = bs_call(S0, K, T, r, sig, Q, 0)
            rows.append([f'{lam:g}', f'{K}', f'{num:.6f}'] + [e(abs(bs_call(S0, K, T, r, sig, Q, 0, order=o) - num)) for o in range(5)])
    body = r'''    <h1>Black&ndash;Scholes with a switching volatility</h1>
    <p class="subtitle">A stock whose volatility jumps between two levels, and European option prices to any order.</p>

    <h2>The model</h2>
    <p>A stock price follows geometric Brownian motion whose volatility is set by a hidden regime
    $y_t \in \{1, 2\}$:</p>
    $$dS_t = r\,S_t\,dt + \sigma_{y_t}\,S_t\,dW_t .$$
    <p>The regime switches at rate $\lambda$ in each direction. The option lives for $T$ and the regime changes every
    $1/\lambda$, so switching is fast when $\lambda T \gg 1$.</p>
    <p class="muted">Parameters: $\sigma = (0.30, 0.15)$, $r = 0.03$, $S_0 = 100$,
    $T = 1$, and the stock starts in the volatile regime.</p>

    <h2>What is computed</h2>
    <p>European call prices, through the characteristic function of the log return</p>
    $$\phi_i(u) = \mathbb{E}\big[e^{iu\log(S_T/S_0)} \mid y_0 = i\big]$$
    <p>and
    <a href="./bibliography.html#Lewis2001">Lewis&apos;s formula</a></p>
    $$C = S_0 - \frac{\sqrt{S_0K}\,e^{-rT}}{\pi}\int_0^\infty \operatorname{Re}\big[e^{iu\log(S_0/K)}\,\phi(u - \tfrac i2)\big]\,\frac{du}{u^2 + 1/4}.$$
    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is Black&ndash;Scholes with the average variance
    $\bar\sigma^2 = \tfrac12(\sigma_1^2 + \sigma_2^2)$.</p>

    <h2>Reduction to a linear system</h2>
    <p>There is no state to factor out: $\phi_i(u) = a_i(T)$ directly, with $a' = (Q + \operatorname{diag} g)a$,
    $a(0) = \mathbf 1$ and the constant, complex</p>
    $$g_i = iu\big(r - \tfrac12\sigma_i^2\big) - \tfrac12u^2\sigma_i^2 .$$
'''
    body += two_state_expansion(r'$\tilde g = -\tfrac14(iu + u^2)(\sigma_1^2 - \sigma_2^2)$',
                                r''' The expansion needs $|\tilde g|/\lambda$ to be small, and $\tilde g$ grows like $u^2$. The Fourier integral is
    therefore stopped at the frequency where the averaged characteristic function falls below $e^{-40}$; beyond it
    the integrand is negligible and the expansion would not apply.''')
    body += r'''
    <h2>Results</h2>
    <p>Call prices from the numerical characteristic function, and the error after each order:</p>
'''
    body += table(['switching rate', 'strike', 'numerical'] + [f'order {o}' for o in range(5)], rows)
    body += r'''    <p>A Monte Carlo simulation of the switching model at $\lambda = 50$, with 400,000 paths, gives 16.581 and 6.758 at
    strikes 90 and 110, within one standard error of the numerical prices. <a href="./bibliography.html#Yin2009">Yin (2009)</a>
    develops asymptotic expansions for this model.</p>
'''
    write('black-scholes.html', 'Black-Scholes with a switching volatility', body)


# ------------------------------------------------------------------ bond options
def bond_options_page():
    kap, th, sg = 0.5, [0.05, 0.03], [0.015, 0.010]
    T, S, x0 = 1.0, 4.0, 0.04
    rows = []
    for lam in (25.0, 50.0, 100.0):
        Q = sym(lam)
        for K in (0.88, 0.90):
            num = zcb_call(T, S, K, x0, 0, kap, th, sg, Q)
            rows.append([f'{lam:g}', f'{K}', f'{num:.8f}'] + [e(abs(zcb_call(T, S, K, x0, 0, kap, th, sg, Q, order=o) - num)) for o in range(5)])
    body = r'''    <h1>Options on bonds under regime switching</h1>
    <p class="subtitle">A call on a zero-coupon bond when the short rate&apos;s mean level and volatility follow a hidden regime.</p>

    <h2>The model</h2>
    <p>The short rate $x_t$ is an Ornstein&ndash;Uhlenbeck process whose mean level and volatility are set by a hidden
    regime $y_t \in \{1, 2\}$ that switches at rate $\lambda$ in each direction:</p>
    $$dx_t = \kappa\,(\theta_{y_t} - x_t)\,dt + \sigma_{y_t}\,dW_t .$$
    <p>The rate reverts over $1/\kappa$ and the regime changes every $1/\lambda$.</p>
    <p class="muted">Parameters: $\kappa = 0.5$,
    $\theta = (0.05, 0.03)$, $\sigma = (0.015, 0.010)$, $x_0 = 0.04$, and the economy starts in regime 1.</p>

    <h2>What is computed</h2>
    <p>The price of a call, expiring at $T = 1$ with strike $K$, on the zero-coupon bond maturing at $S = 4$:</p>
    $$C = \mathbb{E}\Big[e^{-\int_0^T x_r\,dr}\,\big(P(T, S) - K\big)^+\Big].$$
    <p>The bond price at expiry depends on the regime then: $P(T,S) = A_{y_T}\,e^{-b\,x_T}$ with
    $b = (1 - e^{-\kappa(S-T)})/\kappa$ and $A_j$ the regime-$j$ factor for the remaining three years.</p>

    <h2>Averaging</h2>
    <p>When the regime switches infinitely fast the limit is Vasicek&apos;s model with averaged mean level and variance,
    and the option has Jamshidian&apos;s closed-form price.</p>

    <h2>Reduction to a linear system</h2>
    <p>In regime $j$ the call pays when $x_T < x_j^* = \log(A_j/K)/b$, so the price combines the quantities</p>
    $$\mathbb{E}\big[e^{-\int_0^T x_r\,dr - c\,x_T}\,\mathbf 1\{x_T < x_j^*,\ y_T = j\}\big]$$
    <p>for $c = b$ and $c = 0$. Each
    follows by Gil-Pelaez inversion from the same expectation without the indicator on $x_T$ and with $c$ replaced
    by $c - iu$. For a terminal exponent $c$ the coefficient of $x$ is</p>
    $$\tilde B(t) = c\,e^{-\kappa t} + (1 - e^{-\kappa t})/\kappa,$$
    <p>which does not depend on the regime, so</p>
    $$\mathbb{E}\big[e^{-\int_0^t x_r\,dr - c\,x_t}\,\mathbf 1\{y_t = j\} \mid y_0 = i\big] = e^{-\tilde B(t)x_0}\,a_i(t), \qquad a' = (Q + \operatorname{diag} g)\,a, \quad a(0) = e_j,$$
    <p>with</p>
    $$g_i = -\kappa\theta_i\tilde B + \tfrac12\sigma_i^2\tilde B^2.$$
    <p>The terminal vector is now the indicator
    $e_j$ of the regime at expiry rather than $\mathbf 1$.</p>

    <h2>The expansion</h2>
    <p>With $\varepsilon = 1/\lambda$, write the terminal vector $e_j$ as its stationary part $\tfrac12\mathbf 1$ plus a
    remainder. The stationary part is carried by the outer series exactly as when $a(0) = \mathbf 1$: to first order</p>
    $$\exp\big(\int\bar g + \tfrac\varepsilon2\int\tilde g^2\big)\big(1 \pm \tfrac\varepsilon2\tilde g\big),$$
    <p>with
    $\bar g$ and $\tilde g$ the regime average and half-difference of $g$. The remainder relaxes over the fast time
    $t/\varepsilon$ like $e^{-2\lambda t}$, so the initial layer now starts at order zero; it is solved order by order
    with the same recursion, described on the <a href="./engine.html">engine</a> page. The expansion needs
    $|\tilde g|/\lambda$ small at the frequencies $u$ that matter, and the Gil-Pelaez integrals stop at eight
    standard deviations of $x_T$ in frequency.</p>

    <h2>Results</h2>
    <p>Call prices from the numerical solution, and the error after each order:</p>
'''
    body += table(['switching rate', 'strike', 'numerical'] + [f'order {o}' for o in range(5)], rows)
    body += r'''    <p>A Monte Carlo simulation of the switching model at $\lambda = 50$, with 400,000 paths, gives 0.001361 at strike
    0.90, within one standard error of the numerical 0.001356.</p>
'''
    write('bond-options.html', 'Options on bonds under regime switching', body)
