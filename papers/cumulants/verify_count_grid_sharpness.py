"""Two-state count-grid TV: exact factorization and a sharp rate with remainder."""

import numpy as np
from scipy.linalg import expm
from scipy.stats import poisson

from verify_stationary_count_grid import ROUND_TOL, SIZE, kernels


def check(mean_rate, contrast, speed, first, second, gap):
    q = np.array([[-1., 1.], [1., -1.]])
    intensity = np.array([mean_rate+contrast, mean_rate-contrast])
    blocks = [kernels(q, intensity, speed, t) for t in [first, second]]
    conditional = [k.sum(axis=-1) for k in blocks]
    marginals = [u.mean(axis=-1) for u in conditional]
    defects = [(u[:, 0]-u[:, 1])/2 for u in conditional]
    joint = np.einsum('i,aij,jk,bkl->ab', [.5, .5], blocks[0],
                      expm(speed*gap*q), blocks[1])
    difference = joint-np.multiply.outer(*marginals)
    decay = np.exp(-2*speed*gap)
    factored = decay*np.multiply.outer(*defects)
    allowance = ROUND_TOL*max(1., 2*speed*(first+second+gap))
    identity_error = np.sum(abs(difference-factored))
    assert identity_error <= allowance
    measured = .5*np.sum(abs(difference))
    exact_factor = .5*decay*np.prod([np.sum(abs(d)) for d in defects])
    assert abs(measured-exact_factor) <= allowance
    # Folding modulo SIZE contracts l1: the same Duhamel bound applies to these laws.
    z = np.exp(2j*np.pi*np.arange(SIZE)/SIZE)
    for duration, defect in zip([first, second], defects):
        p = np.fft.fft(np.exp(mean_rate*duration*(z-1))).real/SIZE
        leading = contrast/(2*speed)*(np.roll(p, 1)-p)
        remainder = abs(contrast)*np.exp(-2*speed*duration)/speed
        remainder += 2*abs(contrast)**3*duration/speed**2
        assert np.sum(abs(defect-leading)) <= remainder+allowance
        assert np.sum(abs(defect)) <= abs(contrast)/speed+allowance
    modes = [poisson.pmf(np.floor(mean_rate*t), mean_rate*t) for t in [first, second]]
    coefficient = contrast**2/2*np.prod(modes)
    bound = decay*(contrast**2/(2*speed**2)
                   *(np.exp(-2*speed*first)+np.exp(-2*speed*second))
                   + contrast**4*(first+second)/speed**3)
    # Each bin is dominated by Pois(max(intensity)*T); joint and product each
    # differ from their modulo projections by at most the sum of bin tail masses.
    alias = sum(poisson.sf(SIZE-1, max(intensity)*t) for t in [first, second])
    assert abs(measured-decay*coefficient/speed**2) <= bound+2*alias+allowance
    return measured, coefficient, bound, identity_error, alias


def main():
    cases = 0
    largest_identity = largest_alias = 0.
    for mean_rate, contrast in [(2., 1.), (1., 1.), (3., -2.), (2., 0.)]:
        for speed in [.5, 2., 20., 80., 320.]:
            for first, second in [(.01, .1), (.2, .7), (1., 1.), (1/speed, 2/speed)]:
                for gap in [0., 1/speed, .1]:
                    values = check(mean_rate, contrast, speed, first, second, gap)
                    largest_identity = max(largest_identity, values[3])
                    largest_alias = max(largest_alias, values[4])
                    cases += 1
    print("symmetric two-state sharpness certificate; parameter/grid/speed/gap cases", cases)
    for speed in [20., 40., 80., 160., 320.]:
        measured, coefficient, bound, _, _ = check(2., 1., speed, 1., 1., 0.)
        print(f"speed {speed:.0f}: m^2 TV {speed**2*measured:.10f}; limit {coefficient:.10f}")
        if speed == 80.:
            print(f"speed 80 TV {measured:.10f}; leading term {coefficient/speed**2:.10f}")
            print(f"explicit finite-speed absolute error bound {bound:.10f}")
    print(f"maximum exact factorization L1 discrepancy {largest_identity:.3e}")
    print(f"maximum modulo alias probability budget {largest_alias:.3e}")
    print(f"base roundoff allowance {ROUND_TOL:.3e}, scaled by speed and horizon")
    print("PASS")


if __name__ == "__main__":
    main()
