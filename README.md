# homogenization (view as [web page](https://homogenization.microprediction.org))

Averaging a fast hidden variable, and computing the corrector: the terms the average leaves out.

The site collects examples: regime-switching survival, fast switching in exponential-affine models,
fast mean-reverting volatility, periodic media, choice among racing processes, conformal prediction
and forecast calibration. The regime-switching and fast-switching examples are worked out in full here.

## Regime-switching survival

The hazard (or short rate) is an Ornstein–Uhlenbeck process whose mean level and volatility switch
between two regimes at rate λ. The problem comes from the second half of Peter Cotton's 2001 Stanford
thesis. The averaged model is Vasicek's with the regime averages, and the expansion in ε = 1/λ is
available to all orders:

- the ratio ρ = d/m of the regime difference to the regime mean solves one Riccati equation,
  and its outer series terms are polynomials in E = 1 − exp(−κt), integrated in closed form;
- the initial layer, which shifts log m from order ε⁴, is solved to all orders in the span of
  τᵏ exp(−2jτ), τ = λt.

The numerical solution of a two-state linear ODE (the reduction holds since κ does not switch; Elliott and Mamon 2002, Elliott
and Siu 2009) is the reference. After n orders the error falls like ε^(n+1).

## Layout

```
papers/regime-switching-survival/
  exact.py                 numerical solution of the two-state linear ODE
  expansion.py             the expansion to second order
  all_orders.py            the expansion to all orders, outer series plus initial layer
  verify_expansion.py      Monte Carlo check of the numerical solution; orders 1 to 3
  verify_all_orders.py     orders 1 to 6 against a 30-digit numerical solution
  general_orders.py        any switching rates (unequal occupancy), all orders
  verify_general_orders.py symmetric reduction; orders 1 to 6 at p = 0.3
papers/fast-switching/
  fastswitch.py            the engine: any finite chain, all orders, exponential-sum or Chebyshev coefficients
  fastswitch_op.py         a general coupling operator (a continuous fast factor in the Hermite basis)
  fastswitch_gen.py        several time scales: correlation in sqrt(eps)
  models.py                two-factor rates and credit, CIR, jumps, Poisson counts, Heston, a fast factor
  quantlib_models.py       Merton76, variance gamma and Bates with switched parameters
  explicit.py, bond_option_explicit.py   closed-form terms for the example pages
  verify_engine.py, verify_models.py, verify_fast_factor.py, verify_quantlib_models.py, verify_explicit.py   certificates
  make_pages.py            computes the tables on the example pages
papers/yield-curve/
  three_numbers.py         first-order Vasicek curve under a fast chain: three Green-Kubo numbers
  long_end.py              the long yield as a principal eigenvalue; its convergent series in the switching time
  verify_three_numbers.py  formula order, positive semidefiniteness, two-state equality, two matched chains, long end
  verify_uniform_long_end.py  the composite with the exact eigenvalue: uniform in maturity, two-state proof, chain A checked
  make_data.py             writes docs/three-numbers.js for the yield-curve page
papers/cumulants/
  polymoments.py           exact moments of integrated variance and the log return from polynomial moment systems
  verify_cumulants.py      Green-Kubo remainder bounds uniform in maturity, the log-return identity, the CIR case, leverage
papers/general/
  effective_generator.py   first-order rule for any pricing equation: L_bar + sum K_jk A_j A_k
  verify_general.py        random non-commuting operators, Vasicek speed and volatility, Black-Scholes smile, many-name credit
papers/smile/
  cycle_smile.py           Heston with switched level and vol-of-vol on a finite-difference grid; the first-order rule
  verify_cycle_smile.py    the direction of a regime cycle moves the smile, and the rule predicts it
docs/                      the site (GitHub Pages from main /docs)
tools/assemble.py          builds docs/*.html from tools/pages/ with one canonical header
tools/biblio.json          the verified bibliography
```

## Running the certificates

The certificates need Python 3 with NumPy, SciPy and mpmath:

```
python3 -m pip install -r requirements.txt
```

Each certificate runs from its own folder and ends with PASS or FAIL:

```
cd papers/fast-switching
python3 verify_engine.py
python3 verify_models.py
python3 verify_fast_factor.py
python3 verify_quantlib_models.py
python3 verify_explicit.py
cd ../regime-switching-survival
python3 verify_expansion.py
python3 verify_all_orders.py
python3 verify_general_orders.py
```

The other folders under `papers/` have their own `verify_*.py`, run the same way.

Optional: `verify_quantlib_models.py` also prices against QuantLib's engines when QuantLib is installed
(`python3 -m pip install QuantLib`); without it those comparisons are skipped.

## Editing the site

The site build needs Node for the header check. After editing a page, run
`python3 tools/assemble.py && node docs/header-check.js`.

## Cite

Cotton, P. (2001). *An Analytic Approach to Ornstein–Uhlenbeck Processes with Fluctuating
Parameters and Applications in the Modeling of Fixed Income Securities.* PhD thesis, Stanford
University.
