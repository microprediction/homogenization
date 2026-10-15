"""Sharp Holder exponent and deterministic-noise lower bound, modulo reversal."""
from fractions import Fraction as F

import numpy as np
from scipy.linalg import null_space

from verify_covariance_identification import close, sym


def norm(a):
    return np.linalg.norm(a, 2)


def quotient(a, b):
    return min(norm(a-b), norm(a-b.T))


def skew(v):
    if len(v) == 1:
        return v[0]*np.array([[0, 1], [-1, 0]])
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def within(value, bound, scale=1):
    assert value <= bound+4096*np.finfo(float).eps*max(1, scale)


def main():
    rng = np.random.default_rng(20261015)
    for dimension in (1, 3):
        for _ in range(200):
            v = rng.normal(size=dimension)*10**rng.uniform(-3, 3)
            w = rng.normal(size=dimension)*10**rng.uniform(-3, 3)
            k, j = skew(v), skew(w)
            a = max(norm(k), norm(j))
            delta = norm(k@k-j@j)
            distance = min(norm(k-j), norm(k+j))
            within(distance, np.sqrt(2*delta), a)
            within(distance, 2*delta/a, a)
    # Orthogonal equal-length axial vectors attain the first skew-bound constant.
    k, j = skew([1, 0, 0]), skew([0, 1, 0])
    close(np.array([[min(norm(k-j), norm(k+j))]]), np.array([[np.sqrt(2*norm(k@k-j@j))]]))
    print("400 skew-pair bounds; sharp sqrt(2) constant for three-dimensional skew matrices")

    for n in (2, 3, 4):
        for _ in range(20):
            pi = rng.uniform(0.5, 1.5, n)
            pi /= pi.sum()
            u = null_space(np.sqrt(pi)[None, :])/np.sqrt(pi)[:, None]
            gs, ks, ls, qs = [], [], [], []
            for _ in range(2):
                x = rng.normal(size=(n-1, n-1))
                g = np.eye(n-1)+0.01*x.T@x/n
                x = rng.normal(size=(n-1, n-1))
                k = 0.01*(x-x.T)
                l = np.linalg.inv(g+k)
                q = -u@l@(u.T*pi)
                assert np.min(q[~np.eye(n, dtype=bool)]) > 0
                close(q@np.ones((n, 1)), np.zeros((n, 1)))
                close(pi[None, :]@q, np.zeros((1, n)))
                whitened = np.sqrt(pi)[:, None]*q/np.sqrt(pi)[None, :]
                gs.append(g); ks.append(k); ls.append(l); qs.append(whitened)
            g1, g2 = gs
            b1, b2 = [sym(np.linalg.matrix_power(g+k, 2)) for g, k in zip(gs, ks)]
            eg, eb = norm(g1-g2), norm(b1-b2)
            error = eb+(norm(g1)+norm(g2))*eg
            floor = min(np.linalg.eigvalsh(g)[0] for g in gs)
            a = max(norm(k) for k in ks)
            distance = quotient(qs[0], qs[1])
            within(distance, (eg+np.sqrt(2*error))/floor**2, max(map(norm, ls)))
            if a:
                within(distance, (eg+2*error/a)/floor**2, max(map(norm, ls)))
    print("60 positive-rate generator pairs with nonuniform pi: Holder and local Lipschitz bounds")

    u = np.sqrt(3)*null_space(np.ones((1, 3)))
    q0 = -u@u.T/3
    for k in (F(1, 2), F(1, 8), F(1, 32), F(1, 128)):
        eta = k*k/2
        boundary0, boundaryk, reported = F(1), 1-k*k, 1-eta
        assert abs(reported-boundary0) == eta == abs(reported-boundaryk)
        h = np.eye(2)+skew([float(k)])
        q = -u@np.linalg.solve(h, u.T)/3
        assert np.min(q[~np.eye(3, dtype=bool)]) > 0
        recovered_h = np.linalg.inv(-u.T@q@u/3)
        close(sym(recovered_h), np.eye(2))
        close(sym(recovered_h@recovered_h), float(boundaryk)*np.eye(2))
        distance = quotient(q0, q)
        predicted = float(k)/np.sqrt(1+float(k*k))
        close(np.array([[distance]]), np.array([[predicted]]))
        midpoint = (q0+q)/2
        close(np.array([[max(quotient(midpoint, q0), quotient(midpoint, q))]]), np.array([[predicted/2]]))
        print(f"eta={eta}: common boundary noise; minimax lower bound={predicted/2:.10g}")
    print("0<eta<1/6: lower bound sqrt(2 eta)/(2 sqrt(1+2 eta)); optimal exponent 1/2")
    print("PASS")


if __name__ == "__main__":
    main()
