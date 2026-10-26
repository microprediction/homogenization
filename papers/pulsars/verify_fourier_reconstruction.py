"""Exact finite-basis identities behind reconstruction-conditioned lag fits."""

import sympy as sp


def lag_weight(n, lag):
    """Symmetric matrix for the usual uncentered lag product."""
    W = sp.zeros(n)
    for j in range(n - lag):
        if lag == 0:
            W[j, j] += sp.Rational(1, n)
        else:
            weight = sp.Rational(1, 2 * (n - lag))
            W[j, j + lag] += weight
            W[j + lag, j] += weight
    return W


def vech_row(M):
    """Coefficients mapping vech(beta beta^T) to beta^T M beta."""
    d = M.rows
    return [M[i, j] if i == j else 2 * M[i, j]
            for i in range(d) for j in range(i, d)]


def main():
    n = 16
    times = [2 * sp.pi * j / n for j in range(n)]
    # Derivatives of the real Fourier modes at frequencies two and four.
    Phi = sp.Matrix([[-2 * sp.sin(2 * t), 2 * sp.cos(2 * t),
                      -4 * sp.sin(4 * t), 4 * sp.cos(4 * t)]
                     for t in times])

    trend = sp.Matrix([[1, j] for j in range(n)])
    R = sp.eye(n) - trend * (trend.T * trend).inv() * trend.T
    A = sp.simplify(R * Phi)
    beta = sp.Matrix([2, -1, 3, 1])
    x = A * beta

    rows = []
    for lag in range(11):
        W = lag_weight(n, lag)
        M = sp.simplify(A.T * W * A)
        assert sp.simplify((x.T * W * x)[0] - (beta.T * M * beta)[0]) == 0
        rows.append(vech_row(M))

    lag_map = sp.Matrix(rows)
    assert Phi.rank() == A.rank() == 4
    assert lag_map.rank() == 5
    assert lag_map[5:, :].rank() == 5

    Sigma = sp.diag(1, 2, 3, 4)
    reconstructed_covariance = sp.simplify(A * Sigma * A.T)
    assert reconstructed_covariance.rank() == 4

    print("11 exact lag-factorization identities")
    print("all-lag quadratic rank: 5")
    print("long-lag-only quadratic rank: 5")
    print("propagated reconstruction covariance rank: 4")
    print("PASS")


if __name__ == "__main__":
    main()
