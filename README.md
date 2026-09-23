# homogenization (view as [web page](https://homogenization.microprediction.org))

Averaging a fast hidden variable, and computing the corrector: the terms the average leaves out.

The site collects examples: regime-switching survival, fast mean-reverting volatility, periodic
media, choice among racing processes, conformal prediction and forecast calibration. The
regime-switching example is worked out in full here.

## Regime-switching survival

The hazard (or short rate) is an Ornstein–Uhlenbeck process whose mean level and volatility switch
between two regimes at rate λ. The problem comes from the second half of Peter Cotton's 2001 Stanford
thesis. The averaged model is Vasicek's with the regime averages, and the expansion in ε = 1/λ is
available to all orders:

- the ratio ρ = d/m of the regime difference to the regime mean solves one Riccati equation,
  and its outer series terms are polynomials in E = 1 − exp(−κt), integrated in closed form;
- the initial layer, which shifts log m from order ε⁴, is solved to all orders in the span of
  τᵏ exp(−2jτ), τ = λt.

The exact solution (a two-state linear ODE, since κ does not switch; Elliott and Mamon 2002, Elliott
and Siu 2009) is the reference. After n orders the error falls like ε^(n+1).

## Layout

```
papers/regime-switching-survival/
  exact.py                 exact solution (two-state linear ODE)
  expansion.py             the expansion to second order
  all_orders.py            the expansion to all orders, outer series plus initial layer
  verify_expansion.py      Monte Carlo check of the exact solution; orders 1 to 3
  verify_all_orders.py     orders 1 to 6 against a 30-digit exact solution
docs/                      the site (GitHub Pages from main /docs)
tools/assemble.py          builds docs/*.html from tools/pages/ with one canonical header
tools/biblio.json          the verified bibliography
```

After editing a page, run `python3 tools/assemble.py && node docs/header-check.js`.

## Cite

Cotton, P. (2001). *An Analytic Approach to Ornstein–Uhlenbeck Processes with Fluctuating
Parameters and Applications in the Modeling of Fixed Income Securities.* PhD thesis, Stanford
University.
