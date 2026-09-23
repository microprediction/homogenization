"""Compute the tables for the fast-switching pages and write tools/pages/*.html.

`python3 papers/fast-switching/make_pages.py && python3 tools/assemble.py`
"""
import math
import os
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from pages_examples import (table, e, write, SRC, three_regimes_page, credit_page, counts_page, cir_page,
                            jumps_page, heston_page, black_scholes_page, bond_options_page)


def engine_page():
    body = r'''    <h1>The fast-switching engine</h1>
    <p class="subtitle">One recursion for every model whose state factors out of an exponential-affine formula.</p>

    <p class="lead">Each example on this site has a slow process $x_t$, the rate, intensity, variance or count of
    interest, and a fast process $y_t$ that it does not observe: a Markov chain with a finite number of regimes, or
    a fast mean-reverting diffusion. The quantity wanted is a Feynman&ndash;Kac expectation,</p>
    $$\mathbb{E}\big[e^{-\int_0^t c\cdot x_r\,dr}\,\phi(x_t, y_t)\big].$$
    <p>When the model is exponential-affine in $x$ and
    the coefficient of $x$ does not depend on the regime, the state factors out and what remains is a linear system
    in time only:</p>
    $$u_i(t,x) = e^{-B(t)\cdot x}\,a_i(t), \qquad a' = \big(Q + \operatorname{diag} g(t)\big)\,a, \qquad a(0) = \text{the payoff in each regime}.$$
    <p>The model enters only through the functions $g_i(t)$, one per regime, and the terminal vector.</p>
'''
    body += table(['Model', 'Switching parameters', 'Time function'], [
        ['<a href="./regime-switching.html">Vasicek</a>, <a href="./three-regimes.html">any number of regimes</a>', 'mean level, volatility', r'$-\kappa\theta_i B + \tfrac12\sigma_i^2 B^2$'],
        ['<a href="./credit.html">Gaussian factors, two-name credit</a>', 'means, volatilities, correlations', r'$-\sum_j \kappa_j\theta_{ji} B_j + \tfrac12\sum_{j,l}\rho_{jl,i}\sigma_{ji}\sigma_{li}B_jB_l$'],
        ['<a href="./cir.html">CIR</a>', 'mean level', r'$-\kappa\theta_i B_{\mathrm{CIR}}$'],
        ['<a href="./jumps.html">Vasicek with jumps</a>', 'mean, volatility, jump intensity', r'$-\kappa\theta_i B + \tfrac12\sigma_i^2B^2 + \ell_i\big(\tfrac{1}{1+mB} - 1\big)$'],
        ['<a href="./counts.html">Poisson counts</a>', 'arrival rate', r'$(z-1)\,\ell_i$'],
        ['<a href="./heston.html">Heston</a>', 'long-run variance', r'$\kappa\theta_i D(t)$, complex'],
        ['<a href="./black-scholes.html">Black&ndash;Scholes</a>', 'volatility', r'$iu(r - \tfrac12\sigma_i^2) - \tfrac12u^2\sigma_i^2$'],
        ['<a href="./bond-options.html">Bond options</a>', 'mean level, volatility', r'$-\kappa\theta_i \tilde B + \tfrac12\sigma_i^2\tilde B^2$, indicator terminal vector'],
        ['<a href="./fast-factor.html">Fast mean-reverting factor</a>', 'mean level and volatility, continuously', 'an operator in the Hermite basis, with correlation'],
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
    where $Q_0^{\#}$ is the group inverse of $Q_0$ and $F_{n-1}$ collects the terms of $F$ of that order. Each step
    is solvable because $\pi$ annihilates the right-hand side. The first order, with $Q^{\#}$ the group inverse of
    $Q$ itself, is</p>
    $$a_i(t) \approx \exp\Big(\int_0^t \big[\bar g - \pi\cdot\big(g\,Q^{\#}(g - \bar g)\big)\big]\Big)\,\Big(1 - \big[Q^{\#}(g - \bar g)\big]_i(t)\Big).$$
    <p>When every $g_i$ is a sum of exponentials $e^{-\alpha t}$ the class is closed under products and derivatives,
    so every $w_n$ is again such a sum and every term of $\log s$ integrates in closed form. Otherwise $g_i$ is held
    as a Chebyshev series.</p>
    <p>The outer series does not match the terminal vector. The initial layer $\eta = w - w_{\mathrm{outer}}$, in the fast
    time $\tau = t/\varepsilon$, solves</p>
    $$d\eta/d\tau = Q_0\eta + \varepsilon\,[F(w_{\mathrm{outer}}+\eta) - F(w_{\mathrm{outer}})]$$
    <p>with $\eta(0) = w(0) - w_{\mathrm{outer}}(0)$. It is solved order by order in the eigenbasis of $Q_0$, in the span of
    $\tau^k e^{\mu\tau}$. The result is $a_i = s\,(1 + w_i + \eta_i)$ with error $O(\varepsilon^{N+1})$ after $N$
    orders.</p>

    <h2>Other inputs</h2>
    <p>Time-dependent parameters, such as a Hull&ndash;White mean level $\theta(t)$ fitted to a yield curve, change only
    the functions $g_i$, which the engine then holds as Chebyshev series. A terminal exponential payoff
    $e^{-\beta x_T}$ changes only the starting value of $B$. A continuous fast factor in place of the chain is
    treated on its <a href="./fast-factor.html">own page</a>.</p>

    <h2>Code and checks</h2>
    <ul>
      <li><a href="''' + SRC + '''fastswitch.py">fastswitch.py</a>: the engine, for any finite chain, real or complex coefficients, any terminal vector.</li>
      <li><a href="''' + SRC + '''models.py">models.py</a> and <a href="''' + SRC + '''options.py">options.py</a>: the $g_i$ for each model, and option prices.</li>
      <li><a href="''' + SRC + '''verify_engine.py">verify_engine.py</a>: agreement with the two-state code, and orders 1 to 6 on a three-state chain against a 30-digit numerical solution.</li>
      <li><a href="''' + SRC + '''verify_models.py">verify_models.py</a>: each model&apos;s reduction against Monte Carlo of the switching model, and each expansion&apos;s convergence.</li>
    </ul>
'''
    write('engine.html', 'The fast-switching engine', body)


def fast_factor_page():
    from fastswitch_op import hermite_eval
    from fastswitch_gen import FastSwitchGen
    from models import fast_factor, fast_factor_exact
    par = dict(kappa=1.0, theta0=0.05, theta1=0.03, sig0=0.25, sig1=0.2)
    rho, t, y = -0.7, 1.0, 0.3
    L, pi, one, Gs = fast_factor(rho=rho, order=6, **par)
    rows = []
    for eps in (0.01, 0.0025, 0.000625):
        fg = FastSwitchGen(L, math.sqrt(eps), 2, Gs, pi, one, order=6)
        ex = fast_factor_exact(t, y, eps, rho=rho, **par)
        rows.append([f'{eps:g}'] + [e(abs(hermite_eval(fg.a(t, o), y) - ex)) for o in range(7)])
    body = r'''    <h1>A fast mean-reverting factor</h1>
    <p class="subtitle">The continuous version of regime switching, with correlation, to all orders in $\sqrt\varepsilon$.</p>

    <h2>The model</h2>
    <p>The short rate $x_t$ reverts to a level, and diffuses with a volatility, that both depend on a hidden factor
    $Y_t$ which moves much faster than the rate:</p>
    $$dx_t = \kappa\,\big(\theta(Y_t) - x_t\big)\,dt + \sigma(Y_t)\,dW_t, \qquad dY_t = -\frac{1}{\varepsilon}\,Y_t\,dt + \sqrt{\frac{2}{\varepsilon}}\,dZ_t, \qquad d\langle W, Z\rangle = \rho\,dt .$$
    <p>The factor is an Ornstein&ndash;Uhlenbeck process whose stationary law is standard normal. It forgets its
    starting point over a time of order $\varepsilon$, while the rate reverts over $1/\kappa$, so $\varepsilon \ll 1/\kappa$
    separates the two time scales.</p>
    <p>The correlation $\rho$ couples the shocks to the rate and to the factor. This is
    the setting of fast mean-reverting stochastic volatility for interest rates.</p>
    <p class="muted">Parameters: $\kappa = 1$,
    $\theta(y) = 0.05 + 0.03y$, $\sigma(y) = 0.25 + 0.2y$ and $\rho = -0.7$.</p>

    <h2>What is computed</h2>
    <p>The bond price</p>
    $$u(t, x, y) = \mathbb{E}\big[e^{-\int_0^t x_r\,dr} \mid x_0 = x,\ Y_0 = y\big].$$
    <h2>Averaging</h2>
    <p>When the factor is infinitely fast the rate sees only its stationary law, and the limit is Vasicek&apos;s model
    with mean level $\mathbb{E}[\theta(Y)]$ and variance $\mathbb{E}[\sigma(Y)^2]$, the expectations taken under the
    standard normal law.</p>

    <h2>Reduction to a linear system</h2>
    <p>Because $\kappa$ does not depend on $Y$, $u = e^{-B(t)x}a(t,y)$ with $B(t) = (1 - e^{-\kappa t})/\kappa$.
    With $\delta = \sqrt\varepsilon$,</p>
    $$a_t = \Big(\frac{1}{\delta^2}\mathcal L + \frac1\delta\,G_{-1}(t) + G_0(t)\Big)a, \qquad a(0,\cdot) = 1,$$
    $$\mathcal L = -y\,\partial_y + \partial_{yy}, \qquad G_{-1} = -\sqrt2\,\rho\,B(t)\,\sigma(y)\,\partial_y, \qquad G_0 = -\kappa\theta(y)B(t) + \tfrac12\sigma(y)^2B(t)^2 .$$
    <p>Write $a(t,\cdot)$ in Hermite coordinates, $a = \sum_n c_n(t)\,\mathrm{He}_n(y)$, with the probabilists&apos; Hermite
    polynomials $\mathrm{He}_0 = 1$, $\mathrm{He}_1 = y$, $\mathrm{He}_2 = y^2 - 1$, and so on. They are orthogonal under
    the standard normal law,</p>
    $$\mathcal L\,\mathrm{He}_n = -n\,\mathrm{He}_n, \qquad y\,\mathrm{He}_n = \mathrm{He}_{n+1} + n\,\mathrm{He}_{n-1}, \qquad \partial_y\mathrm{He}_n = n\,\mathrm{He}_{n-1}.$$
    <p>In these coordinates $\mathcal L$ is $\operatorname{diag}(0, -1, -2, \dots)$, the
    constant function is the unit vector $e_0$, and the stationary expectation reads off $c_0$. The system is a chain
    with states $0, 1, 2, \dots$ and decay rates $0, 1, 2, \dots$, coupled by the banded matrices $G_{-1}$ and $G_0$.</p>

    <h2>The expansion</h2>
    <p>Let $s = c_0$, the stationary expectation of $a$, and $v = a/s$. For each coupling $G_p$ let
    $F_p(v) = G_p v - v\,\mathbb{E}[G_p v]$. The outer terms of $v = 1 + \sum_{n\ge1}\delta^n w_n$ are</p>
    $$w_n = \mathcal L^{\#}\big(w_{n-2}' - [F_{-1}]_{n-1} - [F_0]_{n-2}\big),$$
    <p>where $\mathcal L^{\#}$ inverts $\mathcal L$ on functions of mean zero and $[F_p]_k$ is the part of $F_p$ of order
    $\delta^k$. Each order involves finitely many Hermite modes, because the couplings are polynomial in $y$. The
    first correction, of order $\sqrt\varepsilon$, needs the correlation:</p>
    $$\log\frac{s}{s\big|_{\rho = 0}} = -\sqrt{2\varepsilon}\,\rho\int_0^t B(r)\;\mathbb{E}\big[\sigma(Y)\,\partial_y\phi_r(Y)\big]\,dr + O(\varepsilon),
      \qquad \mathcal L\phi_r = \bar g(r) - g(r,\cdot),$$
    <p>with $g = G_0$ and $\bar g$ its stationary mean. An initial layer, solved in the eigenbasis of $\mathcal L$, restores
    $a(0,\cdot) = 1$. Without correlation only even orders appear and the expansion is in $\varepsilon$.</p>

    <h2>Results</h2>
    <p>With $\theta$ and $\sigma$ linear in $y$ the solution has the form $a = e^{A + C_1y + C_2y^2}$, with $A$, $C_1$
    and $C_2$ solving three ODEs; their numerical solution is the reference. The error at $t = 1$, $y = 0.3$, after
    each order in $\sqrt\varepsilon$:</p>
'''
    body += table(['epsilon'] + [f'order {o}' for o in range(7)], rows)
    body += r'''    <p>Each quartering of $\varepsilon$ halves $\sqrt\varepsilon$ and divides the order-$n$ error by about $2^{n+1}$,
    until round-off near $10^{-13}$. A Monte Carlo simulation of the two-factor model agrees with the reference solution;
    see <a href="''' + SRC + '''verify_fast_factor.py">verify_fast_factor.py</a>. The engines are
    <a href="''' + SRC + '''fastswitch_op.py">fastswitch_op.py</a> and
    <a href="''' + SRC + '''fastswitch_gen.py">fastswitch_gen.py</a>.</p>
'''
    write('fast-factor.html', 'A fast mean-reverting factor', body)


if __name__ == '__main__':
    engine_page()
    three_regimes_page()
    credit_page()
    counts_page()
    cir_page()
    jumps_page()
    heston_page()
    black_scholes_page()
    bond_options_page()
    fast_factor_page()
