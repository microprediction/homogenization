"""Certificate for portfolio choice with fast regimes (page: portfolio.html).

1. Observed regime, CRRA: the certainty-equivalent rate psibar + (1-g) K(psi, psi) (+ memory term for a known start)
   against the numerical solution of the linear system (matrix exponential), with errors falling at second order;
   the engine's higher orders; zero first-order effect for log utility; a three-state chain with switching r;
   the optimal regime-dependent fractions equal the myopic Merton weights at every speed.
2. Mean-variance: Zhou-Yin's P system gives beta = E exp(-int lam^2); the frontier Var = beta/(1-beta)(E x_T - x0 e^{rT})^2
   agrees with their general formula when r is constant and with exact Monte Carlo of the optimal feedback;
   beta ~ exp(-T<lam^2> + T K(lam^2, lam^2)) with second-order errors.
3. Regime-blind constant fraction: CE(w) and the weight shift against a numerical solve and numerical maximization,
   and exact Monte Carlo over chain paths.  A nonreversible three-state example verifies that the drift/variance
   cross coefficient must be the symmetric Green-Kubo form; the one-sided coefficient fails at first order.
4. Gram identification: for a finite irreducible chain, the symmetric Green-Kubo Gram matrix has exactly the
   nullspace of the centered feature matrix.  A two-state example verifies that integrating maturity-dependent
   loading directions can nevertheless raise rank above every fixed-maturity rank.  A nonreversible example
   checks quantitative coercivity: the integrated Green-Kubo eigenvalues and condition number are controlled by
   the ordinary maturity-integrated loading Gram and the symmetric resolvent spectrum.
Writes results.json for the page. Ends with PASS or FAIL.
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from portfolio import *  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
R, MU, SIG = 0.03, [0.15, -0.05], [0.12, 0.30]   # bull, bear
Q1, Q2 = 3.0, 9.0                                # exit rates at speed 1: bull lasts 4 months, bear about 6 weeks
SPEEDS = [0.25, 0.5, 1, 2, 4]
GAMMA, T_CRRA, T_MV = 3.0, 5.0, 1.0


def main():
    ok, out = True, {}
    rr = [R, R]
    out['params'] = dict(r=R, mu=MU, sigma=SIG, q1=Q1, q2=Q2, gamma=GAMMA, T_crra=T_CRRA, T_mv=T_MV,
                         sharpe=sharpe(rr, MU, SIG).tolist(), pi=stationary(two_state(Q1, Q2)).tolist(),
                         psi=psi(rr, MU, SIG, GAMMA).tolist(), merton=merton_weights(rr, MU, SIG, GAMMA).tolist())

    print("1. observed regime, CRRA, gamma = 3, T = 5")
    rows, errs = [], []
    for m in SPEEDS:
        Q = two_state(m * Q1, m * Q2)
        p = psi(rr, MU, SIG, GAMMA)
        pb = float(stationary(Q) @ p)
        k2 = K2(m * Q1, m * Q2, p, p)
        assert abs(k2 - K(Q, p, p)) < 1e-15
        ex = ce_observed_exact(Q, rr, MU, SIG, GAMMA, T_CRRA)
        f1 = ce_observed_first(Q, rr, MU, SIG, GAMMA, T_CRRA)
        eng = ce_observed_engine(Q, rr, MU, SIG, GAMMA, T_CRRA, [1, 2, 4])
        exb = ce_observed_exact(Q, rr, MU, SIG, GAMMA, T_CRRA, start=0)
        f1b = ce_observed_first(Q, rr, MU, SIG, GAMMA, T_CRRA, start=0)
        exs = ce_observed_exact(Q, rr, MU, SIG, GAMMA, T_CRRA, start=1)
        f1s = ce_observed_first(Q, rr, MU, SIG, GAMMA, T_CRRA, start=1)
        rows.append(dict(m=m, psibar=pb, K=k2, exact=ex, first=f1, order2=eng[2], order4=eng[4],
                         bull_exact=exb, bull_first=f1b, bear_exact=exs, bear_first=f1s))
        errs.append(abs(ex - f1))
        print(f"   speed {m:4}: averaged {pb:.6f}  first order {f1:.6f}  order 2 {eng[2]:.6f}  order 4 {eng[4]:.7f}"
              f"  numerical {ex:.7f}   error {ex - f1:+.2e}   bull start {exb:.6f} vs {f1b:.6f}, bear start {exs:.6f} vs {f1s:.6f}")
        ok &= abs(eng[1] - f1) < 1e-12                          # the engine's first order is the closed form
        ok &= abs(eng[4] - ex) < abs(ex - f1)
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print("   error ratios per doubling of speed:", ", ".join(f"{x:.2f}" for x in ratios))
    ok &= all(3.3 < x < 4.7 for x in ratios[1:])
    out['crra'] = rows

    print("   risk aversion at speed 1 (effect in bp of CE rate)")
    rows = []
    Q = two_state(Q1, Q2)
    for gm in [1.0, 2.0, 3.0, 5.0, 10.0]:
        g_ = gm if gm != 1.0 else 1.0 + 1e-7
        p = psi(rr, MU, SIG, g_)
        pb = float(stationary(Q) @ p)
        ex = ce_observed_exact(Q, rr, MU, SIG, g_, T_CRRA)
        f1 = ce_observed_first(Q, rr, MU, SIG, g_, T_CRRA)
        rows.append(dict(gamma=gm, psibar=pb, exact_bp=1e4 * (ex - pb), first_bp=1e4 * (f1 - pb)))
        print(f"   gamma {gm:4}: psibar {pb:.5f}  effect numerical {1e4*(ex-pb):+.3f} bp, first order {1e4*(f1-pb):+.3f} bp")
        if gm == 1.0:
            ok &= abs(ex - pb) < 1e-8
    out['gamma'] = rows

    print("   optimal regime-dependent fractions (numerical maximization) against the myopic Merton weights")
    wm = merton_weights(rr, MU, SIG, GAMMA)
    rows = []
    for m in [0.25, 1, 4]:
        Q = two_state(m * Q1, m * Q2)
        res = minimize(lambda w: -ce_regime_fractions(Q, rr, MU, SIG, GAMMA, T_CRRA, w), x0=[1.0, 0.0],
                       method='Nelder-Mead', options=dict(xatol=1e-10, fatol=1e-14, maxiter=4000))
        rows.append(dict(m=m, w=res.x.tolist()))
        print(f"   speed {m:4}: numerical {res.x[0]:.6f}, {res.x[1]:.6f}   Merton {wm[0]:.6f}, {wm[1]:.6f}")
        ok &= np.abs(res.x - wm).max() < 1e-5
    out['fractions'] = rows

    print("   three-state chain Q_A with switching r, mu, sigma; gamma = 3, T = 5")
    QA = np.array([[-3, 2, 1], [1, -2, 1], [0.5, 1.5, -2]], float)
    r3, mu3, s3 = [0.05, 0.03, 0.01], [0.13, 0.08, -0.12], [0.12, 0.18, 0.35]
    rows, errs = [], []
    for m in [1, 2, 4, 8]:
        ex = ce_observed_exact(m * QA, r3, mu3, s3, GAMMA, T_CRRA)
        f1 = ce_observed_first(m * QA, r3, mu3, s3, GAMMA, T_CRRA)
        eng = ce_observed_engine(m * QA, r3, mu3, s3, GAMMA, T_CRRA, [4])[4]
        exi = ce_observed_exact(m * QA, r3, mu3, s3, GAMMA, T_CRRA, start=2)
        f1i = ce_observed_first(m * QA, r3, mu3, s3, GAMMA, T_CRRA, start=2)
        errs.append(abs(ex - f1))
        rows.append(dict(m=m, exact=ex, first=f1, order4=eng, crash_exact=exi, crash_first=f1i))
        print(f"   speed {m}: numerical {ex:.7f} first {f1:.7f} order 4 {eng:.8f}; from state 3 {exi:.6f} vs {f1i:.6f}")
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print("   error ratios:", ", ".join(f"{x:.2f}" for x in ratios))
    ok &= all(3.3 < x < 4.7 for x in ratios[1:])
    out['three_state'] = rows

    print("2. mean-variance frontier, T = 1")
    l2 = sharpe(rr, MU, SIG) ** 2
    rows, errs = [], []
    for m in SPEEDS:
        Q = two_state(m * Q1, m * Q2)
        row = dict(m=m, averaged=float(np.exp(-T_MV * stationary(Q) @ l2)))
        for name, st in [('pi', None), ('bull', 0), ('bear', 1)]:
            be, bf = beta_exact(Q, l2, T_MV, st), beta_first(Q, l2, T_MV, st)
            row[name] = dict(exact=be, first=bf, ratio_exact=frontier_ratio(be), ratio_first=frontier_ratio(bf))
            if st is not None:
                ok &= abs(beta_riccati(Q, R, l2, T_MV, st) - be) < 1e-14
        errs.append(abs(np.log(row['pi']['exact']) - np.log(row['pi']['first'])))
        rows.append(row)
        print(f"   speed {m:4}: beta averaged {row['averaged']:.5f} first {row['pi']['first']:.5f} numerical {row['pi']['exact']:.5f};"
              f" from bull {row['bull']['first']:.5f} vs {row['bull']['exact']:.5f}; from bear {row['bear']['first']:.5f} vs {row['bear']['exact']:.5f}")
    ratios = [errs[i] / errs[i + 1] for i in range(len(errs) - 1)]
    print("   error ratios (log beta):", ", ".join(f"{x:.2f}" for x in ratios))
    ok &= all(3.3 < x < 4.7 for x in ratios[1:])
    out['mv'] = rows

    print("   frontier: Zhou-Yin general formula with constant r, and exact Monte Carlo of the optimal feedback")
    rng = np.random.default_rng(20260923)
    x0, T, Q = 1.0, T_MV, two_state(Q1, Q2)
    mc = []
    for st, name in [(0, 'bull'), (1, 'bear')]:
        be = beta_exact(Q, l2, T, st)
        P0 = be * np.exp(2 * R * T)
        for z in [1.10, 1.20]:
            ours = be / (1 - be) * (z - x0 * np.exp(R * T)) ** 2
            zy = zhou_yin_frontier_var(P0, np.exp(-R * T), 0.0, x0, z)
            ok &= abs(ours - zy) < 1e-14
            d = (z - be * x0 * np.exp(R * T)) / (1 - be)          # Lagrange target for mean z
            E, V, se = mc_mean_variance(Q, R, l2, T, x0, d, st, 400000, rng)
            mc.append(dict(start=name, z=z, var_formula=ours, mc_mean=E, mc_mean_se=se, mc_var=V))
            print(f"   start {name}, E x_T = {z}: Var formula {ours:.6f}, Zhou-Yin {zy:.6f}; Monte Carlo mean {E:.5f} +- {se:.5f}, Var {V:.6f}")
            ok &= abs(E - z) < 4 * se and abs(V / ours - 1) < 0.02
    out['mv_mc'] = mc

    print("3. regime-blind constant fraction, gamma = 3, T = 5")
    rows, werr, cerr = [], [], []
    for m in SPEEDS:
        Q = two_state(m * Q1, m * Q2)
        w0, dw = blind_weights(Q, R, MU, SIG, GAMMA)
        wx, cex = blind_argmax(lambda w: ce_blind_exact(Q, w, R, MU, SIG, GAMMA, T_CRRA))
        w1, ce1 = blind_argmax(lambda w: ce_blind_first(Q, w, R, MU, SIG, GAMMA))
        w4, ce4 = blind_argmax(lambda w: ce_blind_engine(Q, w, R, MU, SIG, GAMMA, T_CRRA, 4), 0.0, 1.5)
        Kmm, Kms, Kss = blind_K(Q, MU, SIG)
        hb = float(stationary(Q) @ h_blind(w0, R, MU, SIG, GAMMA))
        ce_w0 = ce_blind_exact(Q, w0, R, MU, SIG, GAMMA, T_CRRA)
        ce_w0_first = ce_blind_first(Q, w0, R, MU, SIG, GAMMA)
        rows.append(dict(m=m, w0=w0, dw=dw, w_lin=w0 + dw, w_cubic=w1, w_order4=w4, w_exact=wx, Kmm=Kmm, Kms=Kms, Kss=Kss,
                         ce_avg=hb, ce_w0_exact=ce_w0, ce_w0_first=ce_w0_first, ce_opt_exact=cex, ce_opt_first=ce1,
                         ce_observed=ce_observed_exact(Q, [R, R], MU, SIG, GAMMA, T_CRRA)))
        werr.append(abs(w0 + dw - wx)); cerr.append(abs(ce_w0 - ce_w0_first))
        print(f"   speed {m:4}: w0 {w0:.4f}  w0+dw {w0+dw:.4f}  first-order argmax {w1:.4f}  order-4 argmax {w4:.4f}  numerical argmax {wx:.4f};"
              f"  CE(w0) averaged {hb:.5f} first {ce_w0_first:.6f} numerical {ce_w0:.6f}")
    rw = [werr[i] / werr[i + 1] for i in range(len(werr) - 1)]
    rc = [cerr[i] / cerr[i + 1] for i in range(len(cerr) - 1)]
    print("   weight error ratios:", ", ".join(f"{x:.2f}" for x in rw), "  CE error ratios:", ", ".join(f"{x:.2f}" for x in rc))
    ok &= all(3.3 < x < 4.7 for x in rw[1:]) and all(x > 3.3 for x in rc)
    out['blind'] = rows

    print("   exact Monte Carlo over chain paths of CE(w0) at speed 1")
    Q = two_state(Q1, Q2)
    w0, _ = blind_weights(Q, R, MU, SIG, GAMMA)
    hv = (1 - GAMMA) * h_blind(w0, R, MU, SIG, GAMMA)
    L = occupation_integral(Q, hv, T_CRRA, None, 400000, rng)
    e = np.exp(L)
    ce_mc = np.log(e.mean()) / ((1 - GAMMA) * T_CRRA)
    se = e.std() / np.sqrt(len(e)) / e.mean() / abs((1 - GAMMA) * T_CRRA)
    ce_ex = ce_blind_exact(Q, w0, R, MU, SIG, GAMMA, T_CRRA)
    print(f"   Monte Carlo {ce_mc:.6f} +- {se:.6f}, numerical {ce_ex:.6f}")
    ok &= abs(ce_mc - ce_ex) < 4 * se
    out['blind_mc'] = dict(mc=float(ce_mc), se=float(se), exact=ce_ex)

    print("   nonreversible three-state cross coefficient")
    Qc = np.array([[-4.2, 4.0, 0.2], [0.2, -4.2, 4.0], [4.0, 0.2, -4.2]])
    muc = np.array([1.0, 0.0, -1.0])
    sc = np.array([2.0, 3.0, 1.0])
    sigc = np.sqrt(sc)
    wc, rc, gc, tc = 0.5, 0.0, 3.0, 2.0
    k_mu_s = K(Qc, muc, sc)
    k_s_mu = K(Qc, sc, muc)
    k_sym = Ksym(Qc, muc, sc)
    k_mm, k_ss = K(Qc, muc, muc), K(Qc, sc, sc)
    hc = h_blind(wc, rc, muc, sigc, gc)
    k_direct = K(Qc, hc, hc)
    k_symmetric_expansion = (
        wc**2 * k_mm - gc * wc**3 * k_sym + 0.25 * gc**2 * wc**4 * k_ss
    )
    k_one_sided = (
        wc**2 * k_mm - gc * wc**3 * k_mu_s + 0.25 * gc**2 * wc**4 * k_ss
    )
    pi_c = stationary(Qc)
    hbar_c = float(pi_c @ hc)
    limit = (1.0 - gc) * k_direct
    speeds = np.array([16.0, 32.0, 64.0, 128.0, 256.0, 512.0])
    scaled = np.array(
        [m * (ce_blind_exact(m * Qc, wc, rc, muc, sigc, gc, tc) - hbar_c) for m in speeds]
    )
    errors = np.abs(scaled - limit)
    ratios = errors[:-1] / errors[1:]
    print(f"      K(mu,s) {k_mu_s:.12e}, K(s,mu) {k_s_mu:.12e}, symmetric {k_sym:.12e}")
    print(f"      K(h,h) direct {k_direct:.12e}, symmetric expansion {k_symmetric_expansion:.12e}, "
          f"one-sided expansion {k_one_sided:.12e}")
    print(f"      scaled exact coefficient at speed 512 {scaled[-1]:.12e}, limit {limit:.12e}, "
          f"one-sided prediction {(1.0-gc)*k_one_sided:.12e}")
    print("      first-order convergence ratios per speed doubling:", ", ".join(f"{x:.3f}" for x in ratios))
    ok &= abs(k_mu_s - k_s_mu) > 0.07
    ok &= abs(k_direct - k_symmetric_expansion) < 2e-16
    ok &= abs(k_one_sided / k_direct - 1.0) > 0.8
    ok &= abs(scaled[-1] / limit - 1.0) < 3e-4
    ok &= all(1.95 < x < 2.05 for x in ratios[1:])
    out['nonreversible_cross'] = dict(
        pi=pi_c.tolist(),
        K_mu_s=k_mu_s,
        K_s_mu=k_s_mu,
        K_sym=k_sym,
        K_h_h=k_direct,
        K_symmetric_expansion=k_symmetric_expansion,
        K_one_sided_expansion=k_one_sided,
        scaled_limit=limit,
        speeds=speeds.tolist(),
        scaled_exact=scaled.tolist(),
        convergence_ratios=ratios.tolist(),
    )

    print("4. exact Green-Kubo Gram rank and maturity aggregation")
    features = np.column_stack([muc, sc, muc + 2.0 * sc, np.ones(3)])
    centered = features - np.outer(np.ones(3), pi_c @ features)
    gram = green_kubo_gram(Qc, features)
    feature_rank = int(np.linalg.matrix_rank(centered, tol=1e-11))
    gram_rank = int(np.linalg.matrix_rank(gram, tol=1e-11))
    coeffs = [
        np.array([1.0, -0.4, 0.3, 2.0]),
        np.array([-0.2, 0.7, -0.5, 1.0]),
        np.array([0.6, 0.1, -0.8, -3.0]),
    ]
    dirichlet_errors = []
    for c in coeffs:
        h = centered @ c
        potential = -group_inverse(Qc) @ h
        energy = np.sum(
            pi_c[:, None] * Qc * (potential[None, :] - potential[:, None]) ** 2
        )
        dirichlet_errors.append(abs(2.0 * c @ gram @ c - energy))
    max_dirichlet_error = float(max(dirichlet_errors))
    print(f"      centered feature rank {feature_rank}, Gram rank {gram_rank}, "
          f"max Dirichlet identity error {max_dirichlet_error:.3e}")
    ok &= feature_rank == gram_rank == 2
    ok &= max_dirichlet_error < 2e-14

    Q_two = two_state(1.0, 1.0)
    contrast = np.array([1.0, -1.0])
    contrast_k = K(Q_two, contrast, contrast)
    nodes, weights = np.polynomial.legendre.leggauss(8)
    maturities, weights = 0.5 * (nodes + 1.0), 0.5 * weights
    integrated = sum(
        weight * green_kubo_gram(Q_two, np.outer(contrast, [1.0, maturity]))
        for maturity, weight in zip(maturities, weights)
    )
    integrated_exact = 0.5 * np.array([[1.0, 0.5], [0.5, 1.0 / 3.0]])
    quadrature_error = float(np.max(np.abs(integrated - integrated_exact)))
    integrated_det = float(np.linalg.det(integrated))
    pointwise_ranks = [
        int(np.linalg.matrix_rank(green_kubo_gram(Q_two, np.outer(contrast, [1.0, maturity])), tol=1e-12))
        for maturity in [0.0, 0.25, 0.5, 0.75, 1.0]
    ]
    integrated_rank = int(np.linalg.matrix_rank(integrated, tol=1e-12))
    print(f"      two-state K(z,z) {contrast_k:.12f}; pointwise ranks {pointwise_ranks}, "
          f"integrated rank {integrated_rank}, determinant {integrated_det:.12f}")
    print(f"      Gauss-Legendre versus exact integrated Gram error {quadrature_error:.3e}")
    ok &= abs(contrast_k - 0.5) < 1e-15
    ok &= pointwise_ranks == [1, 1, 1, 1, 1]
    ok &= integrated_rank == 2
    ok &= abs(integrated_det - 1.0 / 48.0) < 2e-15
    ok &= quadrature_error < 2e-15

    # Quantitative version of the rank theorem.  On centered L2(pi), let
    # R_sym be the self-adjoint part of (-Q)^(-1).  If its positive spectrum
    # lies in [a,b], then for M=int H^*H and Gbar=int H^*R_sym H,
    #
    #                 a M <= Gbar <= b M.
    #
    # Check the Loewner inequalities, their eigenvalue consequences, and the
    # resulting condition-number bound on a nonreversible, nonnormal chain.
    Q_quant = QA
    pi_quant = stationary(Q_quant)
    Pi_quant = np.diag(pi_quant)
    sqrt_pi = np.sqrt(pi_quant)
    inverse_centered = -group_inverse(Q_quant)
    inverse_adjoint = (
        np.diag(1.0 / pi_quant) @ inverse_centered.T @ Pi_quant
    )
    resolvent_sym = 0.5 * (inverse_centered + inverse_adjoint)
    resolvent_similarity = (
        np.diag(sqrt_pi)
        @ resolvent_sym
        @ np.diag(1.0 / sqrt_pi)
    )
    resolvent_eigenvalues = np.linalg.eigvalsh(resolvent_similarity)
    positive_resolvent_eigenvalues = resolvent_eigenvalues[
        resolvent_eigenvalues > 1e-12
    ]
    coercivity_lower = float(positive_resolvent_eigenvalues[0])
    coercivity_upper = float(positive_resolvent_eigenvalues[-1])

    mu_quant = muc - float(pi_quant @ muc)
    s_quant = sc - float(pi_quant @ sc)
    quantitative_gram = np.zeros((2, 2))
    loading_gram = np.zeros((2, 2))
    quantitative_pointwise_ranks = []
    for maturity, weight in zip(maturities, weights):
        state_direction = mu_quant + maturity * s_quant
        H_maturity = np.outer(state_direction, [1.0, maturity])
        pointwise_gram = green_kubo_gram(Q_quant, H_maturity)
        quantitative_gram += weight * pointwise_gram
        loading_gram += weight * H_maturity.T @ Pi_quant @ H_maturity
        quantitative_pointwise_ranks.append(
            int(np.linalg.matrix_rank(pointwise_gram, tol=1e-12))
        )

    lower_slack = np.linalg.eigvalsh(
        quantitative_gram - coercivity_lower * loading_gram
    )
    upper_slack = np.linalg.eigvalsh(
        coercivity_upper * loading_gram - quantitative_gram
    )
    loading_eigenvalues = np.linalg.eigvalsh(loading_gram)
    quantitative_eigenvalues = np.linalg.eigvalsh(quantitative_gram)
    quantitative_condition = float(np.linalg.cond(quantitative_gram))
    condition_bound = float(
        (coercivity_upper / coercivity_lower) * np.linalg.cond(loading_gram)
    )
    quantitative_rank = int(
        np.linalg.matrix_rank(quantitative_gram, tol=1e-12)
    )
    print(
        "      nonnormal resolvent interval "
        f"[{coercivity_lower:.12f}, {coercivity_upper:.12f}]"
    )
    print(
        "      loading eigenvalues "
        f"{loading_eigenvalues[0]:.12f}, {loading_eigenvalues[1]:.12f}; "
        "integrated Green-Kubo eigenvalues "
        f"{quantitative_eigenvalues[0]:.12f}, {quantitative_eigenvalues[1]:.12f}"
    )
    print(
        "      Loewner minimum slacks "
        f"{lower_slack[0]:.12e}, {upper_slack[0]:.12e}; "
        f"condition {quantitative_condition:.8f} <= {condition_bound:.8f}"
    )
    ok &= quantitative_pointwise_ranks == [1] * len(maturities)
    ok &= quantitative_rank == 2
    ok &= lower_slack[0] > -2e-14
    ok &= upper_slack[0] > -2e-14
    ok &= np.all(
        quantitative_eigenvalues
        >= coercivity_lower * loading_eigenvalues - 2e-14
    )
    ok &= np.all(
        quantitative_eigenvalues
        <= coercivity_upper * loading_eigenvalues + 2e-14
    )
    ok &= quantitative_condition <= condition_bound + 2e-12
    out['gram_rank'] = dict(
        nonreversible_feature_rank=feature_rank,
        nonreversible_gram_rank=gram_rank,
        max_dirichlet_identity_error=max_dirichlet_error,
        two_state_contrast_K=contrast_k,
        pointwise_ranks=pointwise_ranks,
        integrated_gram=integrated.tolist(),
        integrated_rank=integrated_rank,
        integrated_determinant=integrated_det,
        quadrature_error=quadrature_error,
        quantitative_pointwise_ranks=quantitative_pointwise_ranks,
        quantitative_integrated_rank=quantitative_rank,
        resolvent_interval=[coercivity_lower, coercivity_upper],
        loading_gram=loading_gram.tolist(),
        quantitative_integrated_gram=quantitative_gram.tolist(),
        loading_eigenvalues=loading_eigenvalues.tolist(),
        quantitative_eigenvalues=quantitative_eigenvalues.tolist(),
        lower_loewner_slack=lower_slack.tolist(),
        upper_loewner_slack=upper_slack.tolist(),
        quantitative_condition=quantitative_condition,
        condition_bound=condition_bound,
    )

    json.dump(out, open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == '__main__':
    main()
