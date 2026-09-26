"""Certificate for the maturity-uniform two-state initial-layer formulas.

The checks distinguish three statements which are easy to conflate:

1. For arbitrary q(0), adding the first layer makes the first-order error
   uniformly O(eps**2), while adding both layer orders makes the error
   uniformly O(eps**3).
2. If q(0) = 0, the second-order outer approximation has only O(eps**2)
   maximum error.  Adding the second-order layer makes it uniformly
   O(eps**3).
3. The omitted layer becomes as small as the nominal outer remainder at
   t = (eps / 2) log(1 / eps), not at a fixed multiple of eps.
"""
import json
import math
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp

from uniform_layer import first_order, second_order


HERE = Path(__file__).resolve().parent
T_MAX = 1.5
EPSILONS = np.array([1 / 20, 1 / 40, 1 / 80, 1 / 160], float)


def order(errors):
    return float(np.polyfit(np.log(EPSILONS), np.log(errors), 1)[0])


def exact_curve(eps, b, q, grid):
    def rhs(t, a):
        bp, qp = b(t), q(t)
        return [(bp + qp - 1 / eps) * a[0] + a[1] / eps,
                a[0] / eps + (bp - qp - 1 / eps) * a[1]]

    sol = solve_ivp(rhs, (0, float(grid[-1])), [1.0, 1.0], method='DOP853',
                    rtol=2e-12, atol=2e-14, dense_output=True,
                    max_step=min(0.002, eps / 8))
    assert sol.success
    return sol.sol(grid)[0]


def integrals(grid, eps, b_values, q_values):
    zeros = np.zeros(1)
    int_b = np.r_[zeros, cumulative_trapezoid(b_values, grid)]
    int_q2 = np.r_[zeros, cumulative_trapezoid(q_values ** 2, grid)]
    int_q_layer = np.r_[zeros, cumulative_trapezoid(q_values * np.exp(-2 * grid / eps), grid)]
    return int_b, int_q2, int_q_layer


def nonzero_start():
    b = lambda t: -0.04 - 0.015 * math.exp(-0.7 * t)
    q = lambda t: 0.65 + 0.35 * (1 - math.exp(-1.3 * t))
    qp = lambda t: 0.455 * math.exp(-1.3 * t)
    q0 = q(0.0)
    qp0 = qp(0.0)
    outer_errors, first_errors, second_errors, fixed_errors = [], [], [], []
    rows = []
    for eps in EPSILONS:
        # Resolve the fastest layer uniformly as eps decreases.
        grid = np.unique(np.r_[np.linspace(0, T_MAX, 3001), eps * np.linspace(0, 8, 1001)])
        grid = grid[grid <= T_MAX]
        bv = np.array([b(t) for t in grid])
        qv = np.array([q(t) for t in grid])
        qpv = np.array([qp(t) for t in grid])
        ib, iq2, iql = integrals(grid, eps, bv, qv)
        exact = exact_curve(eps, b, q, grid)
        outer = np.exp(ib + eps * iq2 / 2) * (1 + eps * qv / 2)
        comp1 = first_order(grid, eps, ib, iq2, qv, q0, iql)
        comp2 = second_order(
            grid, eps, ib, iq2, qv, qpv, q0, qp0, iql
        )
        oe = float(np.max(np.abs(outer - exact)))
        c1e = float(np.max(np.abs(comp1 - exact)))
        c2e = float(np.max(np.abs(comp2 - exact)))
        fe = float(abs(outer[-1] - exact[-1]))
        outer_errors.append(oe)
        first_errors.append(c1e)
        second_errors.append(c2e)
        fixed_errors.append(fe)
        rows.append(dict(
            epsilon=float(eps), outer_sup=oe, first_composite_sup=c1e,
            second_composite_sup=c2e, outer_fixed_T=fe,
        ))
    return (
        rows, order(outer_errors), order(first_errors), order(second_errors),
        order(fixed_errors),
    )


def zero_start():
    b = lambda t: -0.03 - 0.01 * math.exp(-0.9 * t)
    alpha, kappa = -0.8, 1.6
    q = lambda t: alpha * (1 - math.exp(-kappa * t))
    qp = lambda t: alpha * kappa * math.exp(-kappa * t)
    qp0 = qp(0.0)
    outer_errors, composite_errors, crossover_ratios = [], [], []
    rows = []
    for eps in EPSILONS:
        grid = np.unique(np.r_[np.linspace(0, T_MAX, 3001), eps * np.linspace(0, 8, 1001)])
        grid = grid[grid <= T_MAX]
        bv = np.array([b(t) for t in grid])
        qv = np.array([q(t) for t in grid])
        qpv = np.array([qp(t) for t in grid])
        ib, iq2, iql = integrals(grid, eps, bv, qv)
        exact = exact_curve(eps, b, q, grid)
        omega_outer = eps * qv / 2 - eps ** 2 * qpv / 4
        outer = np.exp(ib + eps * iq2 / 2 - eps ** 2 * qv ** 2 / 8) * (1 + omega_outer)
        comp = second_order(
            grid, eps, ib, iq2, qv, qpv, 0.0, qp0, iql
        )
        oe = float(np.max(np.abs(outer - exact)))
        ce = float(np.max(np.abs(comp - exact)))
        fe = float(abs(outer[-1] - exact[-1]))
        t_cross = eps * math.log(1 / eps) / 2
        layer_at_cross = eps ** 2 * abs(qp0) * math.exp(-2 * t_cross / eps) / 4
        crossover_ratios.append(layer_at_cross / eps ** 3)
        outer_errors.append(oe)
        composite_errors.append(ce)
        rows.append(dict(epsilon=float(eps), outer_sup=oe, composite_sup=ce,
                         outer_fixed_T=fe, crossover_ratio=float(crossover_ratios[-1])))
    return rows, order(outer_errors), order(composite_errors), crossover_ratios


def main():
    nz, nz_outer, nz_first, nz_second, nz_fixed = nonzero_start()
    z, z_outer, z_comp, cross = zero_start()
    print('1. q(0) != 0: outer, first composite, and second composite')
    for row in nz:
        print(f"   eps={row['epsilon']:.6f} outer sup={row['outer_sup']:.3e} "
              f"first comp={row['first_composite_sup']:.3e} "
              f"second comp={row['second_composite_sup']:.3e} "
              f"fixed-T outer={row['outer_fixed_T']:.3e}")
    print(f'   observed orders: outer {nz_outer:.6f}, first composite '
          f'{nz_first:.6f}, second composite {nz_second:.6f}, '
          f'fixed-T outer {nz_fixed:.6f}')
    print('2. q(0) = 0: second-order outer versus uniform composite')
    for row in z:
        print(f"   eps={row['epsilon']:.6f} outer sup={row['outer_sup']:.3e} "
              f"composite sup={row['composite_sup']:.3e} fixed-T outer={row['outer_fixed_T']:.3e}")
    print(f'   observed uniform orders: outer sup {z_outer:.6f}, composite sup {z_comp:.6f}')
    print(f'3. crossover layer / eps^3: min={min(cross):.12f}, max={max(cross):.12f}')
    assert 0.9 < nz_outer < 1.1
    assert 1.85 < nz_first < 2.15
    assert 2.8 < nz_second < 3.2
    assert 1.85 < nz_fixed < 2.15
    assert 1.85 < z_outer < 2.15
    assert 2.8 < z_comp < 3.2
    assert max(cross) - min(cross) < 1e-12
    out = dict(nonzero_start=dict(
                   rows=nz, outer_sup_order=nz_outer,
                   first_composite_sup_order=nz_first,
                   second_composite_sup_order=nz_second,
                   outer_fixed_order=nz_fixed),
               zero_start=dict(rows=z, outer_sup_order=z_outer,
                               composite_sup_order=z_comp,
                               crossover_ratio=cross[0]))
    (HERE / 'uniform_layer_results.json').write_text(json.dumps(out, indent=2) + '\n')
    print('PASS')


if __name__ == '__main__':
    main()
