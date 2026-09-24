"""Certificate: first-order Erlang-duration bond prices against the numerical solution of the stage chain.

Checks (1) the Green-Kubo number of the stage chain equals K1 = m1^2 m2^2 / (k (m1+m2)^3) by the group inverse,
(2) w from the closed form equals -Q# theta from the group inverse, stage by stage, and (3) the first-order yield
error falls like the square of the durations (the averaged model's error falls like the durations), for k = 1, 4, 16 and several ages."""
import numpy as np
from erlang_bonds import *

kap, th1, th2, sig, r0 = 0.5, 0.07, 0.01, 0.01, 0.03
ok = True
for k in (1, 2, 4, 16):
    m1, m2 = 0.3, 0.6
    Q = stage_chain(m1, m2, k); n = 2 * k
    pi = np.r_[np.full(k, m1 / k), np.full(k, m2 / k)] / (m1 + m2)
    P = np.outer(np.ones(n), pi); Qs = np.linalg.inv(P - Q) - P   # = -Q#
    f = np.r_[np.ones(k), np.zeros(k)] - pi[:k].sum()
    Kgi = pi @ (f * (Qs @ f))
    wgi = Qs @ (np.r_[np.full(k, th1), np.full(k, th2)] - (pi[:k].sum() * th1 + pi[k:].sum() * th2))
    wcf = np.r_[[w_first(1, (k - j) * m1 / k, m1, m2, k, th1 - th2) for j in range(k)],
                [w_first(2, (k - j) * m2 / k, m1, m2, k, th1 - th2) for j in range(k)]]
    e1, e2 = abs(Kgi / K1(m1, m2, k) - 1), np.max(np.abs(wgi - wcf))
    ok &= e1 < 1e-10 and e2 < 1e-12
    print(f"k={k:2d}  K1 group inverse / closed form - 1 = {e1:.1e}   max |w group inverse - closed form| = {e2:.1e}")

print("\nyield error (bp) of first order vs averaged Vasicek, T = 5, regime 1, against the numerical solution")
for k in (1, 4, 16):
    for age_frac in (0.0, 1.0, 2.0):
        errs, errs0 = [], []
        for s in (1.0, 0.5, 0.25):
            m1, m2 = 0.3 * s, 0.6 * s; T = 5.0
            ex = log_price_numerical(T, r0, 1, age_frac * m1, kap, th1, th2, sig, m1, m2, k)
            fo = log_price_first(T, r0, 1, age_frac * m1, kap, th1, th2, sig, m1, m2, k)
            av = log_vasicek(T, r0, kap, (m1 * th1 + m2 * th2) / (m1 + m2), sig)
            errs.append(-(fo - ex) / T * 1e4); errs0.append(-(av - ex) / T * 1e4)
        ratio = errs[0] / errs[2]
        ok &= (abs(ratio) > 8 or abs(errs[0]) < 0.01) and 3 < errs0[0] / errs0[2] < 5
        print(f"k={k:2d} age={age_frac:.0f} m1   first order: " + " ".join(f"{e:7.3f}" for e in errs)
              + f"   averaged: " + " ".join(f"{e:7.2f}" for e in errs0) + f"   error ratio s=1/s=1/4: {ratio:4.1f}")
print("\nPASS" if ok else "\nFAIL")
