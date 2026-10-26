"""Physical versus base-generator coefficients in the long-end expansion.

Checks the non-unit-epsilon identity and all-order clock normalization.
"""
from fractions import Fraction
import numpy as np
from three_numbers import coefficients, group_inverse
from long_end import eigen_series, long_yield


def scaled_error(actual, expected):
    """Sup-norm error relative to a unit-or-data scale."""
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    scale = max(1.0, float(np.max(np.abs(expected))))
    return float(np.max(np.abs(actual - expected))) / scale


def main():
    Q2 = np.array([[-1., 1.], [1., -1.]])
    theta, variance = np.array([1., -1.]), np.zeros(2)
    mu = eigen_series(Q2, -theta, 6)
    eps = Fraction(1, 10)
    assert eps*Fraction(1, 2) == Fraction(1, 20)
    c2, c3, c4 = coefficients(Q2/float(eps), theta, variance, 1.)
    physical = c2+c3+c4
    assert abs(mu[2]-.5) < 2e-12 and abs(physical-.05) < 2e-12
    assert abs(mu[2]-physical) > .4  # the former displayed identity fails
    print(f'Exact epsilon=1/10: base mu2={mu[2]:.6f}, physical GK={physical:.6f}, epsilon*mu2={float(eps)*mu[2]:.6f}')
    P = np.ones((3, 3))/3
    defective = P-np.eye(3)+.1*np.outer([1., -1., 0.], [1., 1., -2.])
    irreversible = np.array([[-3., 2., 1.], [1., -2., 1.], [.5, 1.5, -2.]])
    rng = np.random.default_rng(20261018)
    five = rng.uniform(.1, 1., (5, 5))
    np.fill_diagonal(five, 0)
    five -= np.diag(five.sum(axis=1))
    count, reparam, worst, homogeneity = 0, 0, 0., 0.
    for Q0 in (Q2, irreversible, defective, five):
        n = len(Q0)
        theta, variance = rng.uniform(.1, .8, n), rng.uniform(.05, .3, n)
        for kappa in (.7, 1.3):
            D = -theta+variance/(2*kappa*kappa)
            base = eigen_series(Q0, D, 6)
            cbase = np.array(coefficients(Q0, theta, variance, kappa))
            for eps in (.04, .2, .8):
                Q = Q0/eps
                c = np.array(coefficients(Q, theta, variance, kappa))
                physical = c @ np.array([kappa**-2, kappa**-3, kappa**-4])
                err = max(scaled_error(physical, eps*base[2]), scaled_error(c, eps*cbase),
                          scaled_error(group_inverse(Q), eps*group_inverse(Q0)))
                assert err < 2e-9
                worst = max(worst, err)
                count += 1
                for scale in (.3, 2.5):
                    changed = eigen_series(scale*Q0, D, 6)
                    terms = base[1:]*eps**np.arange(6)
                    same = changed[1:]*(scale*eps)**np.arange(6)
                    err = scaled_error(same, terms)
                    assert err < 2e-9
                    assert abs(long_yield(scale*Q0/(scale*eps), theta, variance, kappa)
                               -long_yield(Q, theta, variance, kappa)) < 2e-11
                    homogeneity = max(homogeneity, err)
                    reparam += 1
    print(f'{count} physical/base Green-Kubo checks: max scaled error={worst:.3e}')
    print(f'{reparam} normalization checks through order six: max scaled error={homogeneity:.3e}')
    print('PASS')


if __name__ == '__main__':
    main()
