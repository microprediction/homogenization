# Yin and coauthors: the papers around the fast-switching expansion

Companion to yin_zhang_2013_notes.md. Read on 2026-09-25 after the book. The question is the same: who, after
Cotton (2001), expanded a *backward* equation with the regime coupled to the solution, and how far did they go.

## Status of each paper

| Paper | Venue | Access | Read |
|---|---|---|---|
| Khasminskii and Yin, Uniform asymptotic expansions for pricing European options | Appl. Math. Optim. 52 (2005) 279–296, DOI 10.1007/s00245-005-0833-2 | paywalled (Springer) | abstract only |
| Liu, Zhang and Yin, Option pricing in a regime-switching model using the fast Fourier transform | Int. J. Stoch. Anal. 2006, 18109, DOI 10.1155/JAMSA/2006/18109 | open access, CC BY (Wiley/Hindawi); the download was blocked from the shell, one click on the Wiley page fetches it | first page only |
| Yin, Asymptotic expansions of option price under regime-switching diffusions with a fast-varying switching process | Asymptotic Analysis 65 (2009) 203–222, DOI 10.3233/ASY-2009-0953 | paywalled (Sage, "Restricted access", purchase or DeepDyve) | abstract only |
| Yin and Nguyen, Asymptotic expansions of backward equations for two-time-scale Markov chains in continuous time | Acta Math. Appl. Sinica 25 (2009) 457–476 | paywalled (Springer) | as reproduced in the book, chapter 6 |
| Basu and Ghosh, Asymptotic analysis of option pricing in a Markov modulated market | Oper. Res. Lett. 37 (2009) 415–419, DOI 10.1016/j.orl.2009.06.005 | paywalled (Elsevier) | abstract only |
| Nguyen, Asymptotic expansions and stability of hybrid systems with two-time scales | Wayne State PhD dissertation, August 2011, digitalcommons.wayne.edu/oa_dissertations/324 | open access; PDF and text in this folder (gitignored) | chapters 1, 3, 5 in full |
| Nguyen and Yin, Asymptotic expansions for solutions of systems of Kolmogorov backward equations of two-time-scale switching jump diffusions | Comm. Statist. Theory Methods 40 (2011) 3425–3439, DOI 10.1080/03610926.2011.581166 | paywalled (Taylor and Francis) | via the dissertation |
| Nguyen and Yin, Asymptotic expansions of solutions of systems of Kolmogorov backward equations for two-time-scale switching diffusions | Quart. Appl. Math. 71 (2013) 601–628, DOI 10.1090/S0033-569X-2013-01277-X | free PDF on the AMS site (older than five years); the download was blocked from the shell | via the dissertation, whose chapter 3 is this paper |
| Zhou and Yin (2003), Yin and Zhou (2004), Zhang and Yin (2004), Zhang, Yin and Liu (2005) | mean–variance portfolio selection with regime switching; nearly optimal asset allocation; near-optimal selling rule for a two-time-scale market | paywalled | titles and abstracts only; control, not pricing |

## Nguyen (2011) dissertation, chapter 3, which is Nguyen and Yin (2013)

**Problem (3.2.4), p. 18.** Switching diffusion (X, α) on the circle, dX = b(X, α, t) dt + σ(X, α, t) dB,
with a state-dependent generator Q(x, t). Backward system −∂_t u(x, k, t) = L_k(x, t) u(x, k, t) + Q(x, t) u(x, ·, t)(k),
u(x, k, T) = g(x, k), where L_k = ½ a_k ∂²_x + b_k ∂_x. No zeroth-order term: the regime enters through the
diffusion coefficients only. Q^ε = Q̃(x, t)/ε + Q̂(x, t) (3.3.1) with Q̃ block diagonal over l recurrent groups
plus transient states with Q̃_* Hurwitz (3.3.3).

**Assumptions (A1)–(A5), p. 20.** Weak irreducibility of each block for every (x, t); Q̃, Q̂, a, b, g smooth
(2(n + 2) times in x, n + 2 times in t) and periodic in x.

**Construction, §3.3.1, pp. 21–24.** Stretched variable τ = (T − t)/ε. Ansatz u^ε ≈ Σ ε^j φ_j(x, t) + Σ ε^j ψ_j(x, τ)
(3.3.7). Outer recursion (3.3.8): Q̃ φ_0 = 0, Q̃ φ_{i+1} = −∂_t φ_i − (L + Q̂) φ_i. Layer recursion (3.3.11) from
Taylor-expanding Q̃ and L + Q̂ at T, giving ψ_0 = exp(Q̃(x, T) τ)(g − φ_0(x, T)) and ψ_i by variation of
constants (3.3.12). Lemma 3.3.1: the null space of Q̃ is spanned by 1̃(x, t) = block ones with transient rows
d_ı = −Q̃_*^{−1} Q̃_*^ı 1. Lemmas 3.3.2–3.3.3: exp(Q̃ τ) → P(x) = 1̃ ν exponentially fast, transient block included.

**Leading term, §3.3.2, p. 27.** φ_0 = 1̃ β_0 with β̇_0 = −ν (L + Q̂)(1̃ β_0) (3.3.17), the averaged operator
Σ_j ν_j^k L_{kj} acting on the group value, coupled through the aggregated generator ν Q̂ 1̃; terminal condition
β_0(x, T) = ν(x, T) g(x) (3.3.22) forced by ψ_0 → 0.

**Higher orders, §3.3.3, pp. 29–36.** Lemma 3.3.4: the stacked matrix Q_v = (Q̃; ν) has full column rank, so the
particular solution orthogonal to ν is φ̂_i = (Q_v' Q_v)^{−1} Q_v' (b̂_{i−1}; 0) (3.3.26), the least-squares form of
the group inverse. β_i solves (3.3.27) with terminal value β_i(x, T) = ∫_0^∞ ν(x, T) r_i(x, s) ds (3.3.32), the
layer's integrated residual. Proposition 3.3.5 smoothness bookkeeping; Lemma 3.3.6 and Proposition 3.3.7 exponential
decay of ψ_i and its x-derivatives.

**Error bound, §3.3.4, pp. 37–41.** Lemma 3.3.8 the probabilistic representation of the residual through Itô's
formula; Lemma 3.3.9 |L^ε ξ| ≤ C ε^κ ⇒ |ξ| ≤ C ε^κ; Theorem 3.3.10 sup |u^ε − Φ_n − Ψ_n| ≤ C ε^{n+1}, with the
usual "compute one more term" step (3.3.45).

**Fast diffusion, §3.4, pp. 42–62.** L^ε = L̃/ε + L̂ with the switching slow; expansion in the invariant density μ_k
of L̃_k (Lemma 3.4.1), Fredholm alternative through [ζ, μ] = 0 (Lemma 3.4.5), same layer and error structure.

**Illustrations and remarks, §3.5, pp. 63–67.** Weak-convergence limit dX = b̄ dt + σ̄ dB with ν-averaged
coefficients (3.5.2–3.5.3); finite-dimensional distributions of α^ε factor through ν(X(t)); a control cost
averaged by ν (3.5.4–3.5.5). Remark on absorbing plus transient states. Remark (3.5.7): the pure chain backward
equation du/dt = −Q(t) u with Q = Q̃/ε + Q̂, "a similar asymptotic expansion can also be constructed", which is
the Yin–Nguyen 2009 paper and the book's chapter 6. No option, bond, discount or potential anywhere in the
dissertation; the words do not occur.

## What this changes

- **Closest published relative of the thesis expansion is Nguyen and Yin (2011, 2013), not the book.** They
  expand −∂_t u = L u + Q^ε u, one diffusion operator per regime, to all orders with a terminal layer and a
  uniform ε^{n+1} bound. Set L_k = 0 and add a regime-dependent zeroth-order term g_k and the outer recursion is
  the reduced system a' = (Q + diag g) a of the thesis and the SIAM draft. Their proof does not use the absence
  of a zeroth-order term anywhere; the potential case is a special case of their argument in everything but the
  statement.
- **What they do not do.** No potential (so no bond, survival probability or discounted claim as stated); no
  identification of the first corrector as a Green–Kubo quadratic form in the group inverse; no closed-form
  two-state solution; no instruments. The first corrector is present implicitly in (3.3.26)–(3.3.27) as
  ν (L + Q̂) Q̃^# (L + Q̂) 1̃ β_0, which is the operator form of −π (f̃ ⊙ Q^# f̃).
- **Dates.** Thesis 2001; Yin option paper 2009; Yin–Nguyen chain backward equation 2009; Nguyen–Yin switching
  diffusion 2011 (jump diffusions, Comm. Statist.) and 2013 (QAM, state-dependent Q); book second edition 2013.
- **Paper.** The SIAM draft's introduction now cites Nguyen and Yin 2011 and 2013 with the "regime enters through
  the diffusion operator rather than a potential" wording, Basu and Ghosh 2009 for the fast and slow chain
  corrections to the risk-minimising price, and Khasminskii and Yin 2005 for the fast-diffusion volatility
  expansion; the Theorem 1 remark cites the 2013 paper beside the book. Commit on branch paper-siam.
- **Still to read if bought:** Yin 2009 (Sage) for whether the option expansion goes beyond leading order in
  practice and whether the averaged-volatility term is accompanied by a Green–Kubo variance correction; Basu and
  Ghosh 2009 for the form of their first-order correction.
