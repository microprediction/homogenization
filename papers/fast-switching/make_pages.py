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

    <p class="lead">Each example has a slow process $x_t$, the rate, intensity, variance or count of
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
    <h2>Deriving the recursion</h2>
    <p>The derivation uses two ideas from the Background pages: the <a href="./solvability.html">solvability
    condition</a> and the <a href="./layers.html">initial layer</a>. It is written here for any number of regimes.</p>

    <h3>Step 1: separate the average from the shape</h3>
    <p>Let $\pi$ be the stationary distribution of the chain, so that $\pi Q = 0$. Split $a$ into its stationary
    average and its shape:</p>
    $$s = \pi\cdot a, \qquad w = \frac{a}{s} - \mathbf 1, \qquad \pi\cdot w = 0 .$$
    <p>Multiply the system $a' = (Q + \operatorname{diag} g)a$ on the left by $\pi$. The switching term drops out because
    $\pi Q = 0$, which leaves an equation for the average:</p>
    $$\frac{s'}{s} = \bar g + \pi\cdot(g\,w), \qquad \bar g = \pi\cdot g .$$
    <p>Subtracting this from the equation for $a/s$ gives the equation for the shape. Writing $Q = Q_0/\varepsilon$, with
    $\varepsilon$ a typical holding time,</p>
    <div class="equation-card">
    $$\varepsilon\,w' = Q_0\,w + \varepsilon\,F(w), \qquad F(w) = g\,(\mathbf 1 + w) - (\mathbf 1 + w)\,\big(\bar g + \pi\cdot(g\,w)\big).$$
    </div>
    <p>The right side of the shape equation always averages to zero under $\pi$. That is what makes the next step work.</p>

    <h3>Step 2: expand the shape</h3>
    <p>Write $w = \varepsilon w_1 + \varepsilon^2 w_2 + \cdots$ and match powers of $\varepsilon$. At order $\varepsilon^n$ the
    shape equation reads</p>
    $$Q_0\,w_n = w_{n-1}' - F_{n-1},$$
    <p>where $F_{n-1}$ collects the terms of $F$ of order $\varepsilon^{n-1}$. By the Fredholm alternative this has a
    solution because the right side averages to zero, and the solution with $\pi\cdot w_n = 0$ is</p>
    $$w_n = Q_0^{\#}\big(w_{n-1}' - F_{n-1}\big),$$
    <p>with $Q_0^{\#}$ the group inverse. Each order therefore costs one application of a fixed matrix to terms already
    known.</p>

    <h3>Step 3: the first order</h3>
    <p>At the first order $w_0 = 0$ and $F_0 = g - \bar g\,\mathbf 1$, so, with $Q^{\#} = \varepsilon\,Q_0^{\#}$ the group inverse
    of $Q$ itself,</p>
    $$\varepsilon\,w_1 = -Q^{\#}\big(g - \bar g\big).$$
    <p>Putting this into the equation for $s$ gives</p>
    $$a_i(t) \approx \exp\Big(\int_0^t \big[\bar g - \pi\cdot\big(g\,Q^{\#}(g - \bar g)\big)\big]\Big)\,\Big(1 - \big[Q^{\#}(g - \bar g)\big]_i(t)\Big).$$
    <p>The correction in the exponent has a direct meaning. Because $Q^{\#}f = -\int_0^\infty e^{Q\tau}f\,d\tau$ when
    $\pi\cdot f = 0$,</p>
    $$-\pi\cdot\big(g\,Q^{\#}(g - \bar g)\big) = \int_0^\infty \operatorname{Cov}_\pi\big(g(y_0),\,g(y_\tau)\big)\,d\tau .$$
    <p>This is a Green&ndash;Kubo integral of the autocovariance of $g$ along the chain. It is half the long-run variance
    rate of $\int g$, as on the <a href="./idea.html">idea</a> page, now for any chain. The factor in brackets
    is the memory of the starting regime.</p>

    <h3>Step 4: closed form</h3>
    <p>When every $g_i$ is a sum of exponentials $e^{-\alpha t}$, products and derivatives of such sums are again such
    sums. So every $w_n$ is a sum of exponentials, and every term of $\log s$ integrates in closed form. Otherwise
    $g_i$ is held as a Chebyshev series, which is also closed under these operations.</p>

    <h3>Step 5: the initial layer</h3>
    <p>The outer series does not match the terminal vector at $t = 0$. The mismatch is removed on the fast clock
    $\tau = t/\varepsilon$:</p>
    $$\eta = w - w_{\mathrm{outer}}, \qquad \frac{d\eta}{d\tau} = Q_0\,\eta + \varepsilon\,\big[F(w_{\mathrm{outer}}+\eta) - F(w_{\mathrm{outer}})\big], \qquad \eta(0) = w(0) - w_{\mathrm{outer}}(0).$$
    <p>It decays at the rates of the chain and is solved order by order in the eigenbasis of $Q_0$, in the span of
    $\tau^k e^{\mu\tau}$. The result is $a_i = s\,(1 + w_i + \eta_i)$, with error $O(\varepsilon^{N+1})$ after $N$
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
    <p class="subtitle">The continuous version of regime switching, with correlation, derived term by term.</p>

    <p class="lead">In most of the examples a Markov chain switches the parameters of a rate. Here a continuous
    factor does the same job: it moves the mean level and the volatility smoothly and quickly. The correction terms
    follow one at a time from the <a href="./idea.html">idea of averaging</a> and the
    <a href="./solvability.html">solvability condition</a>.</p>

    <h2>The model</h2>
    <p>The short rate $x_t$ reverts to a level, and diffuses with a volatility, that both depend on a hidden factor
    $Y_t$:</p>
    $$dx_t = \kappa\,\big(\theta(Y_t) - x_t\big)\,dt + \sigma(Y_t)\,dW_t .$$
    <p>The factor is a fast Ornstein&ndash;Uhlenbeck process,</p>
    $$dY_t = -\frac{1}{\varepsilon}\,Y_t\,dt + \sqrt{\frac{2}{\varepsilon}}\,dZ_t, \qquad d\langle W, Z\rangle = \rho\,dt .$$
    <p>Its stationary law is standard normal whatever the value of $\varepsilon$. It forgets its starting point over a
    time of order $\varepsilon$, while the rate reverts over $1/\kappa$. So $\varepsilon \ll 1/\kappa$ separates the two
    time scales.</p>
    <p>The correlation $\rho$ ties the shocks to the rate to the shocks to the factor. This is the setting of fast
    mean-reverting stochastic volatility for interest rates, with the mean level allowed to move as well.</p>
    <p>For a check with a closed-form answer, both functions are taken to be linear:</p>
    $$\theta(y) = \theta_0 + \theta_1\,y, \qquad \sigma(y) = \sigma_0 + \sigma_1\,y .$$
    <p class="muted">Parameters: $\kappa = 1$, $\theta_0 = 0.05$, $\theta_1 = 0.03$, $\sigma_0 = 0.25$, $\sigma_1 = 0.2$, $\rho = -0.7$.</p>

    <h2>The quantity</h2>
    <p>The bond price for maturity $t$, starting from rate $x$ and factor value $y$:</p>
    $$u(t, x, y) = \mathbb{E}\big[e^{-\int_0^t x_r\,dr} \mid x_0 = x,\ Y_0 = y\big].$$

    <h2>Averaging</h2>
    <p>When the factor is infinitely fast, the rate sees only its stationary law. The limit is Vasicek&apos;s model with
    the averaged mean level and variance,</p>
    $$\bar\theta = \mathbb{E}[\theta(Y)] = \theta_0, \qquad \bar\sigma^2 = \mathbb{E}[\sigma(Y)^2] = \sigma_0^2 + \sigma_1^2,$$
    <p>with the expectations taken under the standard normal law. The corrections below are what the average
    leaves out.</p>

    <h2>Step 1: the rate factors out</h2>
    <p>By the Feynman&ndash;Kac formula, $u$ solves a partial differential equation in $t$, $x$ and $y$. Try the Vasicek
    form in $x$:</p>
    $$u(t,x,y) = e^{-B(t)\,x}\,a(t,y), \qquad B(t) = \frac{1 - e^{-\kappa t}}{\kappa}.$$
    <p>Every term that involves $x$ cancels, because $B$ solves $B' = 1 - \kappa B$ and does not depend on $y$. What
    remains is an equation for $a$ alone. With $\delta = \sqrt\varepsilon$ it reads</p>
    $$\partial_t a = \frac{1}{\delta^2}\,\mathcal L a + \frac1\delta\,G_{-1}(t)\,a + g(t,y)\,a, \qquad a(0,y) = 1 .$$
    <p>Each operator has a source.</p>
    <ul>
      <li>$\mathcal L = -y\,\partial_y + \partial_{yy}$ is the generator of the factor on its own clock. It is multiplied by
        $1/\delta^2 = 1/\varepsilon$ because the factor is fast.</li>
      <li>The correlation term comes from the cross derivative $\rho\,\sigma(y)\sqrt{2/\varepsilon}\;\partial_x\partial_y u$.
        Acting on $e^{-Bx}$, the $\partial_x$ produces $-B$, which leaves</li>
    </ul>
    $$G_{-1} = -\sqrt2\,\rho\,B(t)\,\sigma(y)\,\partial_y .$$
    <ul>
      <li>The multiplication term collects the drift and the variance of the rate:</li>
    </ul>
    $$g(t,y) = -\kappa\,\theta(y)\,B(t) + \tfrac12\,\sigma(y)^2\,B(t)^2 .$$
    <p>For linear $\theta$ and $\sigma$ this is a quadratic in $y$. Writing $y^2 = \mathrm{He}_2(y) + 1$, with the Hermite
    polynomials $\mathrm{He}_1 = y$ and $\mathrm{He}_2 = y^2 - 1$,</p>
    $$g = (g_0 + g_2) + g_1\,\mathrm{He}_1 + g_2\,\mathrm{He}_2,$$
    <p>where the three coefficient functions are</p>
    $$g_0 = -\kappa\theta_0 B + \tfrac12\sigma_0^2B^2, \qquad g_1 = -\kappa\theta_1 B + \sigma_0\sigma_1B^2, \qquad g_2 = \tfrac12\sigma_1^2B^2 .$$
    <p>The stationary average of $g$ is $\bar g = g_0 + g_2$.</p>

    <h2>Step 2: split off the average</h2>
    <p>Let $s(t) = \mathbb{E}[a(t,Y)]$ be the stationary average of $a$, and $v = a/s$ its shape, so that
    $\mathbb{E}[v] = 1$.</p>
    <p>Take the stationary average of the equation for $a$. The term $\mathcal L a$ averages to zero, because the
    standard normal law is stationary. Dividing by $s$ gives the slow equation:</p>
    $$\frac{s'}{s} = \frac1\delta\,\mathbb{E}\big[G_{-1}v\big] + \mathbb{E}\big[g\,v\big].$$
    <p>Subtracting $v$ times this from the equation for $a/s$ gives the equation for the shape:</p>
    $$\partial_t v = \frac{1}{\delta^2}\,\mathcal L v + \frac1\delta\Big(G_{-1}v - v\,\mathbb{E}[G_{-1}v]\Big) + \Big(g\,v - v\,\mathbb{E}[g\,v]\Big).$$
    <p>Both brackets average to zero by construction. That is what will make every step solvable.</p>

    <h2>Step 3: expand the shape</h2>
    <p>Write the shape as a power series in $\delta$, with each correction averaging to zero:</p>
    $$v = 1 + \delta\,w_1 + \delta^2\,w_2 + \delta^3\,w_3 + \cdots, \qquad \mathbb{E}[w_n] = 0 .$$
    <p>Substitute and collect powers of $\delta$. Each power gives an equation $\mathcal L w_n = f_n$. By the
    <a href="./solvability.html">Fredholm alternative</a> it can be solved when $\mathbb{E}[f_n] = 0$, and the solution
    divides each Hermite coefficient of $f_n$ by minus its index:</p>
    $$\mathcal L\,\mathrm{He}_k = -k\,\mathrm{He}_k \quad\Longrightarrow\quad \mathcal L^{-1}\,\mathrm{He}_k = -\frac{\mathrm{He}_k}{k}.$$

    <h3>Order $\delta^{-1}$</h3>
    <p>The only terms are $\mathcal L w_1$ and the correlation term acting on the constant $1$:</p>
    $$\mathcal L\,w_1 = -\big(G_{-1}1 - \mathbb{E}[G_{-1}1]\big) = 0,$$
    <p>since $\partial_y 1 = 0$. So $w_1 = 0$. The factor has no effect at order $\delta$ on the shape.</p>

    <h3>Order $\delta^{0}$</h3>
    <p>Now $g$ enters:</p>
    $$\mathcal L\,w_2 = -\big(g - \bar g\big) = -g_1\,\mathrm{He}_1 - g_2\,\mathrm{He}_2 .$$
    <p>The right side averages to zero, so it can be solved by dividing by $-1$ and $-2$:</p>
    $$w_2 = g_1\,\mathrm{He}_1 + \tfrac12\,g_2\,\mathrm{He}_2 .$$
    <p>This is the corrector. It says how the bond price depends on the current factor value: a high factor raises the
    mean level and the volatility, and the corrector records by how much.</p>

    <h3>Order $\delta^{1}$</h3>
    <p>The correlation term now acts on $w_2$:</p>
    $$\mathcal L\,w_3 = -\big(G_{-1}w_2 - \mathbb{E}[G_{-1}w_2]\big).$$
    <p>Write $c(t) = \sqrt2\,\rho\,B(t)$, so that $G_{-1} = -c\,\sigma(y)\,\partial_y$. Since $\partial_y w_2 = g_1 + g_2\,y$,</p>
    $$G_{-1}w_2 = -c\,(\sigma_0 + \sigma_1 y)(g_1 + g_2 y) = -c\Big[(\sigma_0g_1 + \sigma_1g_2) + (\sigma_0g_2 + \sigma_1g_1)\,\mathrm{He}_1 + \sigma_1g_2\,\mathrm{He}_2\Big].$$
    <p>Removing the average and dividing by $-1$ and $-2$ gives</p>
    $$w_3 = -c\,(\sigma_0g_2 + \sigma_1g_1)\,\mathrm{He}_1 - \tfrac12\,c\,\sigma_1g_2\,\mathrm{He}_2 .$$

    <h2>Step 4: the slow equation, order by order</h2>
    <p>Substitute the expansion of $v$ into $s'/s = \delta^{-1}\mathbb{E}[G_{-1}v] + \mathbb{E}[g\,v]$ and collect powers
    of $\delta$. The Hermite polynomials are orthogonal with $\mathbb{E}[\mathrm{He}_k^2] = k!$, which makes each average
    a short sum:</p>
    $$\mathbb{E}[g] = \bar g, \qquad \mathbb{E}[G_{-1}w_2] = -c\,(\sigma_0g_1 + \sigma_1g_2),$$
    $$\mathbb{E}[g\,w_2] = g_1^2 + g_2^2, \qquad \mathbb{E}[G_{-1}w_3] = c^2\big(\sigma_0^2g_2 + \sigma_0\sigma_1g_1 + \sigma_1^2g_2\big).$$
    <p>Integrating in time, the stationary average of the bond factor is</p>
    <div class="equation-card">
    $$\log s(t) = \int_0^t \bar g \;-\; \delta\int_0^t c\,(\sigma_0g_1 + \sigma_1g_2) \;+\; \delta^2\int_0^t \Big[g_1^2 + g_2^2 + c^2\big(\sigma_0^2g_2 + \sigma_0\sigma_1g_1 + \sigma_1^2g_2\big)\Big] + O(\delta^3).$$
    </div>
    <p>Each term has a reading.</p>
    <ul>
      <li>The first is the averaged Vasicek model.</li>
      <li>The second, of order $\sqrt\varepsilon$, exists only with correlation. Shocks to the rate and to the factor
        move together, so the factor tends to be high when the rate has risen. This is the correlation
        correction of fast mean-reverting stochastic volatility.</li>
      <li>The third, of order $\varepsilon$, contains the fluctuation term $g_1^2 + g_2^2$. As on the
        <a href="./idea.html">idea</a> page, it is half the variance of the fluctuating integral of $g$, and it is
        present even without correlation.</li>
    </ul>

    <h2>Step 5: the bond price</h2>
    <p>Putting the pieces together, and writing the Hermite polynomials out,</p>
    $$u(t,x,y) = e^{-B(t)x}\,s(t)\,\Big(1 + \varepsilon\big[g_1\,y + \tfrac12 g_2\,(y^2 - 1)\big] + \varepsilon^{3/2}\big[w_3\big] + O(\varepsilon^2)\Big),$$
    <p>with $w_3$ from order $\delta^1$ above. The dependence on the current factor value $y$ first appears at order
    $\varepsilon$.</p>

    <h2>All orders</h2>
    <p>Every further order repeats steps 3 and 4. At order $\delta^{n-2}$ the equation is</p>
    $$\mathcal L\,w_n = w_{n-2}' - \big[F_{-1}\big]_{n-1} - \big[F_0\big]_{n-2},$$
    <p>where $F_p$ collects the terms of order $\delta^k$ in the brackets of the shape equation:</p>
    $$F_{-1}(v) = G_{-1}v - v\,\mathbb{E}[G_{-1}v], \qquad F_0(v) = g\,v - v\,\mathbb{E}[g\,v].$$
    <p>The right side always averages to zero, so every step is solvable. Because $g$ is a polynomial in $y$, each
    $w_n$ involves only finitely many Hermite polynomials, and each is a combination of powers of
    $e^{-\kappa t}$. The computation is exact at every order; only the truncation of the series introduces error.</p>
    <p>The shape starts at $v(0,y) = 1$, and the outer terms respect this for a while: $g$ vanishes at $t = 0$ because
    $B(0) = 0$, so $w_2(0) = w_3(0) = 0$. The first mismatch is at $w_4$, where the time derivative $w_2'$ enters.</p>
    <p>From there an <a href="./layers.html">initial layer</a> in the fast time $t/\varepsilon$ restores the starting
    condition. It is solved in the eigenbasis of $\mathcal L$, which here is the Hermite basis itself.</p>

    <h2>Results</h2>
    <p>For linear $\theta$ and $\sigma$ the solution also has the form $a = e^{A + C_1y + C_2y^2}$, with $A$, $C_1$ and
    $C_2$ solving three ordinary differential equations. Their numerical solution is the reference.</p>
    <p>The terms derived above agree with the code to $10^{-17}$. The table gives the error in the bond factor at
    $t = 1$, $y = 0.3$, after each order in $\sqrt\varepsilon$.</p>
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
