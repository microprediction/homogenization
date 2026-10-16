"""Poisson block martingales and stitched all-start raw-window budgets.

Exact rational checks are distinct from floating continuous-time cross-checks.
The accompanying proof, not finite enumeration, covers an unbounded horizon.
"""
from fractions import Fraction as F
from itertools import combinations, product

import numpy as np
from scipy.linalg import expm, expm_frechet


def inverse(a):
    n = len(a)
    augmented = np.column_stack((a, [[F(int(i == j)) for j in range(n)] for i in range(n)]))
    for j in range(n):
        pivot = next(i for i in range(j, n) if augmented[i, j])
        augmented[[j, pivot]] = augmented[[pivot, j]]
        augmented[j] = augmented[j]/augmented[j, j]
        for i in range(n):
            if i != j:
                augmented[i] -= augmented[i, j]*augmented[j]
    return augmented[:, n:]


def coefficient(p):
    return max(sum(abs(a-b) for a, b in zip(x, y))/2 for x in p for y in p)


def radius(n, t, alpha, delta, length):
    epoch = length.bit_length()-1
    ell = np.log(4*n*n*(epoch+1)*(epoch+2)/delta)
    return n*t*t*(1+1/(1-alpha))*np.sqrt(ell/length)+np.sqrt(2)*t*t/((1-alpha)*length)


def kernels(q, t):
    n = len(q)
    derivatives = [expm_frechet(t*q, t*np.diag(f), compute_expm=False) for f in np.eye(n)]
    return [np.column_stack([da @ expm(i*t*q) @ db @ np.ones(n)
                             for da in derivatives for db in derivatives]) for i in (0, 1)]


def main():
    k = np.array([[F(1, 2), F(1, 3), F(1, 6)], [F(1, 6), F(1, 2), F(1, 3)],
                  [F(1, 3), F(1, 6), F(1, 2)]], dtype=object)
    p = k @ k @ k; projector = np.full((3, 3), F(1, 3))
    deviation = inverse(np.eye(3, dtype=int)-p+projector)-projector
    alpha = coefficient(p); c = 1/(1-alpha); width = 1+c
    ranges = diameters = laws = 0; max_mgf = max_tail = 0.
    for lag in (1, 2):
        paths = [[(end, start, a if lag == 1 else b, k[start, a]*k[a, b]*k[b, end])
                  for a, b, end in product(range(3), repeat=3)] for start in range(3)]
        reward = np.array([[sum(mass for _, a, b, mass in row if a == x and b == y)
                            for x, y in product(range(3), repeat=2)] for row in paths])
        target = np.array([sum(reward[:, j])/3 for j in range(9)])
        h = deviation @ reward
        assert np.all(h-p @ h == reward-np.outer(np.ones(3, dtype=int), target))
        for x, y in combinations(range(3), 2):
            assert sum((h[x]-h[y])**2) <= 2*c*c; diameters += 1
        for start, row in enumerate(paths):
            for j in range(9):
                x, y = divmod(j, 3)
                increments = [(int(a == x and b == y)+h[end, j]-h[start, j]-target[j], mass)
                              for end, a, b, mass in row]
                assert sum(d*mass for d, mass in increments) == 0
                possible = [d for d, _ in increments]
                assert max(possible)-min(possible) <= width; ranges += 1
                for tilt in (-2., -.5, .5, 2.):
                    mgf = sum(float(mass)*np.exp(tilt*float(d)-tilt*tilt*float(width)**2/8)
                              for d, mass in increments)
                    max_mgf = max(max_mgf, mgf)
                    assert mgf <= 1+64*np.finfo(float).eps
        # Exact first-passage probabilities, retaining reward/endpoint dependence.
        horizon = 16; threshold = F(3, 2)
        for j in (1, 6):
            x, y = divmod(j, 3)
            for start in range(3):
                distribution = {(start, 0): F(1)}; crossed = F(0)
                for length in range(1, horizon+1):
                    updated = {}
                    for (state, total), probability in distribution.items():
                        for end, a, b, mass in paths[state]:
                            total_new = total+int(a == x and b == y)
                            martingale = total_new-length*target[j]+h[end, j]-h[start, j]
                            if martingale >= threshold:
                                crossed += probability*mass
                            else:
                                key = end, total_new
                                updated[key] = updated.get(key, F(0))+probability*mass
                    distribution = updated
                    assert crossed+sum(distribution.values()) == 1
                # Exact weaker rational bound, plus exponential floating cross-check.
                exponent = 2*threshold**2/(horizon*width**2)
                assert crossed <= 1/(1+exponent)
                assert float(crossed) <= np.exp(-float(exponent))+64*np.finfo(float).eps
                max_tail = max(max_tail, float(crossed)); laws += 1
    allocation = sum(F(1, (j+1)*(j+2)) for j in range(64))
    assert allocation == 1-F(1, 65)
    print(f"exact block model: alpha={alpha}; entry range bound={width}")
    print(f"{ranges} exact zero-mean/range checks; {diameters} exact matrix-diameter checks")
    print(f"216 conditional MGF cross-checks; maximum normalized MGF={max_mgf:.6f}")
    print(f"{laws} exact first-passage laws through N={horizon}, threshold={threshold}; maximum probability={max_tail:.6f}")
    print(f"64 epoch allocations sum exactly to {allocation}, remaining mass=1/65")
    rng = np.random.default_rng(20261029); chains = []
    for n in range(2, 7):
        for _ in range(2):
            q = rng.uniform(.1, 1., (n, n)); np.fill_diagonal(q, 0.)
            chains.append(q-np.diag(q.sum(1)))
    for n in (3, 5):
        chains.append(np.roll(np.eye(n), 1, axis=1)-np.eye(n))
    chains.append(np.ones((3, 3))/3-np.eye(3)+.1*np.outer([1, -1, 0], [1, 1, -2]))
    count = 0; worst_residual = worst_diameter = 0.
    for q in chains:
        n = len(q); pi = np.linalg.solve(q.T+np.ones_like(q), np.ones(n))
        projector = np.outer(np.ones(n), pi)
        for exposure in (.2, 1.):
            t = exposure/max(-np.diag(q)); p = expm(3*t*q); alpha = coefficient(p)
            deviation = np.linalg.inv(np.eye(n)-p+projector)-projector
            for reward in kernels(q, t):
                h = deviation @ reward; target = pi @ reward; c = 1/(1-alpha)
                residual = np.max(abs(h-p @ h-reward+np.outer(np.ones(n), target)))
                scale = max(1., t*t, np.max(abs(h)))
                assert residual <= 32768*np.finfo(float).eps*scale
                worst_residual = max(worst_residual, residual/scale)
                for x, y in combinations(range(n), 2):
                    diameter = np.linalg.norm(h[x]-h[y]); bound = np.sqrt(2)*t*t*c
                    assert diameter <= bound+32768*np.finfo(float).eps*scale
                    worst_diameter = max(worst_diameter, diameter/bound); count += 1
    print(f"{count} continuous-time diameter checks on {len(chains)} chains; maximum bound ratio={worst_diameter:.6f}")
    print(f"maximum scaled Poisson residual={worst_residual:.3e}")
    eta = radius(2, np.log(2.), 1/8, .05, 1000000)
    print(f"anytime design n=2, T=log(2), alpha=1/8, delta=.05, N=1000000: radius={eta:.9f}")
    print("PASS")


if __name__ == '__main__':
    main()
