"""Stationary Cox bin counts: uniform second-order TV decoupling."""

import numpy as np
from scipy.linalg import expm
from scipy.stats import nbinom, poisson

from polymoments import stationary


ROUND_TOL = 65536*np.finfo(float).eps
SIZE = 64


def envelope(q):
    p = expm(.5*q)
    rho = .5*max(np.sum(abs(a-b)) for a in p for b in p)
    assert 0 < rho < 1
    return 2/rho, -2*np.log(rho)


def kernels(q, intensity, speed, duration):
    z = np.exp(2j*np.pi*np.arange(SIZE)/SIZE)
    matrices = np.array([expm(duration*(speed*q+(s-1)*np.diag(intensity))) for s in z])
    return np.fft.fft(matrices, axis=0).real/SIZE


def conditional_start(blocks):
    u = np.ones(blocks[0].shape[-1])
    for k in blocks[::-1]:
        u = np.einsum('aij,...j->a...i', k, u)
    return u


def end_joint(blocks, pi):
    u = pi
    for k in blocks:
        u = np.einsum('...i,aij->...aj', u, k)
    return u


def product(vectors):
    result = np.array(1.)
    for vector in vectors:
        result = np.multiply.outer(result, vector)
    return result


def marginal_surrogate(q, intensity, speed, duration, pi):
    mean = (pi @ intensity)*duration
    if np.ptp(intensity) == 0:
        return poisson.pmf(np.arange(SIZE), mean), poisson.sf(SIZE-1, mean)
    f = intensity-pi @ intensity
    n = len(pi)
    moment = np.kron(np.eye(3), speed*q)
    for j in [1, 2]:
        moment[j*n:(j+1)*n, (j-1)*n:j*n] = j*np.diag(f)
    initial = np.zeros(3*n)
    initial[:n] = 1.
    variance = pi @ (expm(duration*moment) @ initial)[2*n:]
    shape, success = mean*mean/variance, mean/(mean+variance)
    assert abs(nbinom.mean(shape, success)-mean) <= ROUND_TOL*max(1., mean)
    assert abs(nbinom.var(shape, success)-mean-variance) <= ROUND_TOL*max(1., mean+variance)
    return nbinom.pmf(np.arange(SIZE), shape, success), nbinom.sf(SIZE-1, shape, success)


def check(q, intensity, speed, durations, known=None):
    pi = stationary(q)
    reverse = q.T*pi[None, :]/pi[:, None]
    c, gamma = known if known else envelope(q)
    cr, gr = known if known else envelope(reverse)
    fmax = max(abs(intensity-pi @ intensity))
    defect = lambda length, c, g: min(2., 2*c*fmax*(-np.expm1(-speed*g*length))/(speed*g))
    blocks = [kernels(q, intensity, speed, t) for t in durations]
    reversed_blocks = [kernels(reverse, intensity, speed, t) for t in durations[::-1]]
    u = conditional_start(blocks)
    joint = u @ pi
    axes = tuple(range(len(durations)))
    error = 0.
    allowance = ROUND_TOL*max(1., speed*np.linalg.norm(q, np.inf)*sum(durations))
    assert joint.min() >= -allowance and abs(joint.sum()-1) <= allowance
    memory = np.sum(abs(u-joint[..., None]), axis=axes).max()
    assert memory <= defect(sum(durations), c, gamma)+allowance
    reverse_u = conditional_start(reversed_blocks)
    reverse_u = reverse_u.transpose(list(range(len(durations)-1, -1, -1))+[len(durations)])
    reverse_memory = np.sum(abs(reverse_u-joint[..., None]), axis=axes).max()
    assert reverse_memory <= defect(sum(durations), cr, gr)+allowance
    ending = end_joint(blocks, pi)
    error = max(error, np.sum(abs(ending-reverse_u*pi)))
    # Check every cut by conditional past/future factorization through the boundary state.
    for cut in range(1, len(durations)):
        past = end_joint(blocks[:cut], pi)
        future = conditional_start(blocks[cut:])
        a, b = past.sum(axis=-1), future @ pi
        centered = np.tensordot(past-a[..., None]*pi, future-b[..., None], axes=([-1], [-1]))
        difference = joint-np.multiply.outer(a, b)
        error = max(error, np.sum(abs(difference-centered)))
        bound = .5*defect(sum(durations[:cut]), cr, gr)*defect(sum(durations[cut:]), c, gamma)
        assert .5*np.sum(abs(difference)) <= bound+allowance
    marginals = [np.einsum('i,aij->a', pi, k) for k in blocks]
    exact_product = product(marginals)
    measured = .5*np.sum(abs(joint-exact_product))
    alias = sum(poisson.sf(SIZE-1, max(intensity)*t) for t in durations)
    bound = sum(.5*defect(sum(durations[:j]), cr, gr)*defect(durations[j], c, gamma)
                for j in range(1, len(durations)))
    assert measured+2*alias <= bound+allowance
    replacements = [marginal_surrogate(q, intensity, speed, t, pi) for t in durations]
    independent_nb = product([p for p, tail in replacements])
    marginal_error = sum(.5*np.sum(abs(p-r))+poisson.sf(SIZE-1, max(intensity)*t)+tail/2
                         for p, (r, tail), t in zip(marginals, replacements, durations))
    nb_error = .5*np.sum(abs(joint-independent_nb))
    nb_tail = alias+sum(tail for p, tail in replacements)/2
    assert nb_error+nb_tail <= bound+marginal_error+allowance
    return measured, bound, nb_error, bound+marginal_error, error, alias, nb_tail


def main():
    designs = [
        (np.array([[-1., 1.], [1., -1.]]), np.array([3., 1.]), (1., 2.)),
        (np.array([[-1., 1., 0.], [0., -1., 1.], [4., 0., -4.]]), np.array([2., 0., 1.]), None),
        (np.roll(np.eye(4), 1, axis=1)-np.eye(4), np.array([3., 0., 2., 1.]), None),
    ]
    largest_error = largest_alias = largest_nb_tail = 0.
    cases = 0
    for index, (q, intensity, known) in enumerate(designs):
        for speed in [10., 20., 40., 80.]:
            for durations in [[.01, .1], [.1, 1.], [1., 3.], [.0001, 1.], [.2, .3, .5], [1/speed, 2/speed], [1., 1.]]:
                values = check(q, intensity, speed, durations, known)
                largest_error = max(largest_error, values[4])
                largest_alias = max(largest_alias, values[5])
                largest_nb_tail = max(largest_nb_tail, values[6])
                if index == 0 and speed == 80. and durations == [1., 1.]:
                    print("two unit bins, symmetric intensities (3,1), speed 80")
                    print(f"TV to independent exact marginals {values[0]:.10f}; bound {values[1]:.10f}")
                    print(f"TV to independent moment-fit NB {values[2]:.10f}; bound with marginal errors {values[3]:.10f}")
                cases += 1
    for constant in [0., 2.]:
        values = check(designs[0][0], np.full(2, constant), 10., [.1, 1.], (1., 2.))
        assert values[0] <= ROUND_TOL and values[2] <= ROUND_TOL
    assert largest_error <= ROUND_TOL*10
    print("stationary count-grid certificate; chain/grid/speed cases", cases)
    print(f"maximum reversal/cut identity L1 discrepancy {largest_error:.3e}; base tolerance {ROUND_TOL:.3e}")
    print(f"maximum exact-law modulo alias probability budget {largest_alias:.3e}")
    print(f"maximum NB comparison truncation/alias TV budget {largest_nb_tail:.3e}")
    print("constant and all-zero intensities have independent Poisson increments")
    print("PASS")


if __name__ == "__main__":
    main()
