"""Write three synthetic daily pain series into data/ (gitignored). No real patient data: every value is generated.

    dX = kappa (theta_y - X) dt + sqrt(s_y) dW,   hidden regime y: calm (0) -> bad (1) at rate a, bad -> calm at rate b.

Exact simulation (exponential holding times, exact OU steps), read once a day, 3,650 days after 30 discarded.
lam = a + b is the regime's relaxation rate; regime memory 1/lam. Seeds are fixed, so the files are reproducible."""
import csv, hashlib, math, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
KAPPA, THETA, S, P_BAD = 0.7, (2.0, 5.0), (1.0, 4.0), 1 / 3     # per day; levels; noise variance per day; share bad
SERIES = [("fast_1h", 1 / 24, 1117001), ("fast_2h", 1 / 12, 1117002), ("slow_10d", 10.0, 1117003)]


def simulate(days, a, b, seed, burn=30):
    rng = np.random.default_rng(seed)
    y = int(rng.random() < a / (a + b))
    x = (b * THETA[0] + a * THETA[1]) / (a + b)
    t, t_next, rows = 0.0, rng.exponential(1 / (a, b)[y]), []
    for k in range(1, days + burn + 1):
        T, bad = float(k), 0.0
        while True:
            end = min(t_next, T); h = end - t
            bad += h * y
            e = math.exp(-KAPPA * h)
            x = x * e + THETA[y] * (1 - e) + math.sqrt(S[y] * (1 - e * e) / (2 * KAPPA)) * rng.standard_normal()
            t = end
            if end == T:
                break
            y = 1 - y; t_next = t + rng.exponential(1 / (a, b)[y])
        if k > burn:
            rows.append((k - burn, f"{x:.6f}", int(min(10, max(0, round(x)))), y, f"{bad:.6f}"))
    return rows


def main():
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    for name, memory, seed in SERIES:
        lam = 1 / memory
        path = os.path.join(HERE, "data", f"{name}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["day", "level", "rating", "regime", "bad_share"])
            w.writerows(simulate(3650, lam * P_BAD, lam * (1 - P_BAD), seed))
        print(f"{name}: a {lam * P_BAD:.4g}/day, b {lam * (1 - P_BAD):.4g}/day, seed {seed}, "
              f"sha256 {hashlib.sha256(open(path, 'rb').read()).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
