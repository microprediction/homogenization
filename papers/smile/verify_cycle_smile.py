"""Certificate: the direction of a regime cycle moves the Heston smile, and the first-order rule predicts it.

Heston with rho = 0; a three-state chain cycling 1 -> 2 -> 3 -> 1 switches the variance level theta and the squared
vol-of-vol xi^2. Reversing the cycle keeps pi and the symmetric Green-Kubo matrix and flips its antisymmetric part.
Checks, on one finite-difference grid in the variance:
 1. the rule's implied-volatility error falls at second order as the chain speeds up;
 2. forward-minus-reverse implied volatilities are first order and the rule reproduces them;
 3. keeping only the symmetric part of K does not reach second order.
Writes results.json for the page.
"""
import json, math, os
import numpy as np
from cycle_smile import Grid, calls, implied_vol
from effective_generator import gk, stationary

QC = np.array([[-2.1, 2, 0.1], [0.1, -2.1, 2], [2, 0.1, -2.1]])
TH, XI2, KAPPA, V0, T, S0 = [0.02, 0.10, 0.05], [0.04, 0.16, 1.0], 2.0, 0.04, 1.0, 100.0
STRIKES = [70, 85, 100, 115, 130]
SCALES = [4, 8, 16, 32]


def main():
    g = Grid(vmax=1.0, n=121)
    out, ok = {'scales': SCALES, 'strikes': STRIKES, 'rows': []}, True
    for m in SCALES:
        res = {lab: calls(STRIKES, T, Q, TH, XI2, KAPPA, g, V0, S0=S0, n=40) for lab, Q in [('forward', m * QC), ('reverse', m * QC.T)]}
        iv = {lab: {k: [100 * implied_vol(c, S0, K, T) for c, K in zip(v, STRIKES)] for k, v in res[lab].items()} for lab in res}
        f = iv['forward']
        err = {k: max(abs(a - b) for a, b in zip(f[k], f['numerical'])) for k in ['averaged', 'symmetric part only', 'rule']}
        diff = {k: [a - b for a, b in zip(iv['forward'][k], iv['reverse'][k])] for k in ['numerical', 'rule']}
        K = gk(m * QC, [TH, XI2])
        out['rows'].append({'m': m, 'iv': iv, 'err': err, 'diff': diff, 'Kanti': 0.5 * (K[0, 1] - K[1, 0])})
        print(f"x{m:2d}  max IV error (vol points): averaged {err['averaged']:.3f}  symmetric only {err['symmetric part only']:.3f}"
              f"  rule {err['rule']:.4f}   forward-reverse ATM: numerical {diff['numerical'][2]:+.3f} rule {diff['rule'][2]:+.3f}")
    e = [r['err']['rule'] for r in out['rows']]
    s = [r['err']['symmetric part only'] for r in out['rows']]
    d = [abs(r['diff']['numerical'][2]) for r in out['rows']]
    rate = lambda z: math.log(z[-2] / z[-1]) / math.log(2)
    print(f"rates over the last doubling: rule {rate(e):.2f} (want 2), symmetric only {rate(s):.2f} (want 1), forward-reverse {rate(d):.2f} (want 1)")
    ok &= rate(e) > 1.6 and rate(s) < 1.4 and 0.7 < rate(d) < 1.3
    ok &= all(abs(a - b) < 0.2 * abs(a) + 0.002 for a, b in zip(out['rows'][-1]['diff']['numerical'], out['rows'][-1]['diff']['rule']))
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results.json'), 'w'), indent=1)
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
