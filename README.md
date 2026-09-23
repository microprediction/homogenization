# homogenization (view as [web page](https://homogenization.microprediction.org))

Averaging a fast hidden variable, and computing the corrector that says what the average leaves out.

The first problem here comes from the second half of Peter Cotton's 2001 Stanford thesis: survival
probability, equivalently a zero-coupon bond price, when the hazard or short rate is an
Ornstein–Uhlenbeck process whose mean level and volatility switch between two regimes at rate λ.
The averaged model is Vasicek's with the regime averages. The corrector adds terms in ε = 1/λ.

## Result

With `B(r) = (1 - e^{-κr})/κ`, regime averages and half-differences `θ̄, θ̃` and `s̄, s̃` (of σ²),
`ḡ(r) = -κθ̄B + s̄B²/2` and `g̃(r) = -κθ̃B + s̃B²/2`,

    u_y(t,x) = exp(-B(t)x + ∫ḡ + (ε/2)∫g̃² - (ε²/8) g̃(t)²) · (1 ± (ε/2) g̃(t) ∓ (ε²/4) g̃'(t)) + O(ε³),

upper sign for regime 1. Against the exact solution (a two-state linear ODE, since κ does not
switch; the reduction is due to Elliott and Mamon 2002 and Elliott and Siu 2009) the error after
zero, one and two correction orders falls like ε, ε² and ε³.

The first-order terms in [homogenize](https://github.com/microprediction/homogenize) (2020) are
incorrect: with them the error stays of order ε. The formula above replaces them.

## Layout

```
papers/regime-switching-survival/
  exact.py               exact solution (two-state linear ODE, DOP853)
  expansion.py           the corrected expansion
  verify_expansion.py    certificate: Monte Carlo check of the exact solution, convergence orders, the package bug
docs/                    the site (GitHub Pages from main /docs)
  survival.js            the model in JavaScript, parity-checked against the Python
tools/assemble.py        builds docs/*.html from tools/pages/ with one canonical header
tools/biblio.json        the verified bibliography (every DOI checked against Crossref)
```

Run the certificate with `cd papers/regime-switching-survival && python3 verify_expansion.py`.
After editing a page, run `python3 tools/assemble.py && node docs/header-check.js`.

## Cite

Cotton, P. (2001). *An Analytic Approach to Ornstein–Uhlenbeck Processes with Fluctuating
Parameters and Applications in the Modeling of Fixed Income Securities.* PhD thesis, Stanford
University.
