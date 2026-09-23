"""Measured Green-Kubo number against its Markov equivalent.

For each detrended spin-down series x(t):
- K_meas: batch-means plateau of tau * Var(block mean) / 2 (as in 01_explore.py);
- tau_c: e-folding time of the autocorrelation (first lag where rho < 1/e);
- K_markov = Var(x) * tau_c, the Green-Kubo number of a Markov (exponentially correlated) process with the same
  variance and correlation time.
The ratio R = K_meas / K_markov is near 1 for Markov-like switching and much smaller than 1 when the switching is
regular, because successive swings cancel in the long-run integral. The batch-means curves are plotted to show
whether a plateau exists."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ex", os.path.join(HERE, "01_explore.py"))
ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)


def acf_efold(t, x):
    dt = np.median(np.diff(t))
    y = x - x.mean()
    n = len(y)
    f = np.fft.rfft(y, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n] / (y.var() * np.arange(n, 0, -1))
    k = np.argmax(ac < np.exp(-1))
    return k * dt, ac


def main():
    rows = json.load(open(os.path.join(HERE, "out", "01_summary.json")))
    fig, axes = plt.subplots(5, 4, figsize=(12, 12)); axes = axes.ravel()
    out = []
    for ax, r in zip(axes, rows):
        t, nd, x, _ = ex.load(r["psr"])
        taus, ks, K = ex.batch_K(t, x)
        tau_c, ac = acf_efold(t, x)
        Km = x.var() * tau_c
        out.append(dict(psr=r["psr"], K_meas=K, tau_c_days=tau_c, K_markov=Km, R=K / Km, period_days=r["period_days"]))
        ax.loglog(taus, ks, "o-", ms=3, color="#4a3aff"); ax.axhline(Km, color="#c2410c", ls="--", lw=1)
        ax.set_title(r["psr"], fontsize=9); ax.tick_params(labelsize=7)
    for ax in axes[len(rows):]:
        ax.axis("off")
    fig.suptitle("batch-means estimate of K against block length (days); dashed: Markov equivalent", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "out", "02_batch_means.png"), dpi=100)
    json.dump(out, open(os.path.join(HERE, "out", "02_regularity.json"), "w"), indent=1)
    print(f"{'pulsar':12s} {'K meas':>10s} {'tau_c d':>8s} {'K markov':>10s} {'R':>7s} {'period d':>9s}")
    for o in sorted(out, key=lambda o: o["R"]):
        print(f"{o['psr']:12s} {o['K_meas']:10.4g} {o['tau_c_days']:8.0f} {o['K_markov']:10.4g} {o['R']:7.3f} {o['period_days']:9.0f}")


if __name__ == "__main__":
    main()
