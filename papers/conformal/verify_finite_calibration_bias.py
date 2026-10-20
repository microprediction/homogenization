"""Finite iid calibration: latent path bias, beta attenuation and moment budgets."""
from math import pi, sqrt

import numpy as np
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import ndtr, roots_jacobi, roots_legendre
from scipy.stats import beta, norm


def stationary(q):
    matrix, rhs = q.T.copy(), np.zeros(len(q))
    matrix[-1], rhs[-1] = 1, 1
    return np.linalg.solve(matrix, rhs)


def attenuation(n, k):
    return quad(lambda z: norm.pdf(z)**2*beta.pdf(ndtr(z), k, n+1-k),
                -10, 10, epsabs=2e-11, epsrel=2e-11)[0]


def moment_budget(q, f, c, a, n):
    size, law = len(q), stationary(q)
    f = f-law@f
    inverse = np.linalg.inv(q+np.ones((size, 1))*law)-np.ones((size, 1))*law
    h = inverse@f
    matrix = np.kron(np.eye(5), c*q)
    for j in range(1, 5):
        matrix[j*size:(j+1)*size, (j-1)*size:j*size] = j*np.diag(f)
    initial = np.r_[np.ones(size), np.zeros(4*size)]
    moments = (expm(matrix)@initial).reshape(5, size)
    pooled = moments@law
    stationary_second = 2*(law*f)@(-c*h+(expm(c*q)-np.eye(size))@inverse@h)/c**2
    assert abs(pooled[2]-stationary_second) < 3e-9
    error = abs(a)*abs(expm(c*q)@h)/c+a*a*abs(moments[2]-pooled[2])/2
    error += abs(a)**3*abs(moments[3]-pooled[3])/6+a**4*(moments[4]+pooled[4])/24
    eta = a*a*pooled[2]/2
    transfer = min(norm.pdf(0)*min(1, n*eta), norm.pdf(norm.ppf(min(.5, eta))))
    error += abs(a*h)/c*transfer
    return h, error


def inversion(q, f, c, a, frequency_order=180, score_order=800):
    law = stationary(q)
    f = f-law@f
    t, wt = roots_legendre(frequency_order)
    t, wt = 6*(t+1), 6*wt  # Fourier cutoff 12; Gaussian tail < 2e-33.
    z, wz = roots_legendre(score_order)
    radius = abs(a)*max(abs(f))+9
    z, wz = radius*z, radius*wz
    transform = np.array([expm(c*q+1j*u*a*np.diag(f))@np.ones(len(q)) for u in t])
    phase = np.exp(-1j*z[:, None]*t)*np.exp(-t*t/2)
    cdfs = .5-np.imag((phase*(wt/t))@transform)/pi
    densities = np.real((phase*wt)@transform)/pi
    return law, z, wz, cdfs, densities


def coverage(data, n, k):
    law, z, weights, cdfs, densities = data
    pooled = np.clip(cdfs@law, 0, 1)
    order_density = beta.pdf(pooled, k, n+1-k)*(densities@law)
    result = (weights*order_density)@cdfs
    assert abs(law@result-k/(n+1)) < 5e-8
    return result


def binary_one_calibration(c, a):
    """Independent Poisson--Beta double occupation mixture, with no Fourier inversion."""
    nodes, plus, minus = [1., -1.], [np.exp(-c), 0.], [0., np.exp(-c)]
    probability = np.exp(-c)
    for jumps in range(1, int(c+12*sqrt(c)+30)):
        probability *= c/jumps
        if jumps % 2:
            alpha = beta_shape = (jumps+1)//2
        else:
            alpha, beta_shape = jumps//2+1, jumps//2
        x, w = roots_jacobi(24, beta_shape-1, alpha-1)
        x, w = x, probability*w/w.sum()
        nodes.extend(x); plus.extend(w); minus.extend(np.zeros(24))
        if jumps % 2:
            minus[-24:] = w
        else:
            nodes.extend(-x); plus.extend(np.zeros(24)); minus.extend(w)
    nodes, plus, minus = map(np.array, (nodes, plus, minus))
    assert max(abs(plus.sum()-1), abs(minus.sum()-1)) < 3e-9
    pooled = (plus+minus)/2
    value = 0.
    for j in range(0, len(nodes), 100):
        value += plus[j:j+100]@ndtr(a*(nodes[None, :]-nodes[j:j+100, None])/sqrt(2))@pooled
    return value


def main():
    cases = ((1, 1), (3, 2), (9, 9), (19, 18), (99, 90))
    js = {case: attenuation(*case) for case in cases}
    for eta in (.0001, .01, .1, .5, .9):
        u = np.linspace(0, 1-eta, 1001)
        j = lambda v: norm.pdf(norm.ppf(v))
        assert max(abs(j(u+eta)-j(u))) < j(min(eta, .5))+3e-9
    assert abs(js[1, 1]-1/(2*sqrt(pi))) < 3e-9
    for (n, k), j in js.items():
        assert 0 < j < norm.pdf(norm.ppf(k/(n+1)))
        assert abs(j-attenuation(n, n+1-k)) < 3e-9
        print(f'N={n}, k={k}, p={k/(n+1):.8f}: J={j:.10f}; population factor={norm.pdf(norm.ppf(k/(n+1))):.10f}')
    q = np.array([[-1.1, .8, .3], [.4, -1.4, 1], [.7, .5, -1.2]])
    f, a = np.array([-1., .35, 1.4]), 2.
    print(f'Q={q.tolist()}; f={f.tolist()}; mu=1; sigma=.5; a={a}; pi={stationary(q)}')
    print('Horizons 32,64,128,256,512,1024; fixed-N remainder O(c^-2), all-N bound O(c^-2 sqrt(log c))')
    largest_refinement, largest_ratio = 0., 0.
    residuals = {case: [] for case in cases}
    for c in (32, 64, 128, 256, 512, 1024):
        data = inversion(q, f, c, a)
        refined = inversion(q, f, c, a, 240, 1100)
        h, uniform_bound = moment_budget(q, f, c, a, 0)
        contrast = data[3]-(data[3]@data[0])[:, None]-a*norm.pdf(data[1])[:, None]*h/c
        assert np.max(abs(contrast)-uniform_bound) < 5e-8
        for n, k in cases:
            observed = coverage(data, n, k)-k/(n+1)
            discrepancy = max(abs(coverage(refined, n, k)-k/(n+1)-observed))
            largest_refinement = max(largest_refinement, discrepancy)
            h, bound = moment_budget(q, f, c, a, n)
            residual = abs(observed-a*h*js[n, k]/c)
            assert max(residual-bound) < 5e-8
            largest_ratio = max(largest_ratio, max(residual/bound))
            assert abs(stationary(q)@h) < 3e-9
            residuals[n, k].append(max(residual))
            if (n, k, c) == (9, 9, 256):
                print(f'N=9,k=9,c=256: coverages={observed+.9}; c*gaps={c*observed}')
                print(f'  predicted c*gaps={a*h*js[n,k]}; finite error budgets={bound}')
    assert largest_refinement < 5e-8
    for case, values in residuals.items():
        rate = np.log2(values[-2]/values[-1])
        assert 1.8 < rate < 2.2
        print(f'N,k={case}: final residual convergence order {rate:.5f}')
    qb, fb = np.array([[-1., 1.], [1., -1.]]), np.array([1., -1.])
    for c in (4, 16):
        value = coverage(inversion(qb, fb, c, a), 1, 1)[0]
        independent = binary_one_calibration(c, a)
        assert abs(value-independent) < 5e-8
        print(f'Binary c={c}, N=k=1: coverage={value:.10f}; mixture discrepancy={abs(value-independent):.3g}')
    print(f'30 three-state cases; max refinement {largest_refinement:.3g}; max residual/budget {largest_ratio:.5f}; tolerance 5e-8')
    print('PASS')


if __name__ == '__main__':
    main()
