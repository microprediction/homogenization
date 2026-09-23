"""Explore the 17 spin-down series: detrend, two-level structure, dominant period, and the Green-Kubo number K.

K = int_0^inf Cov(nudot(0), nudot(t)) dt is measured by batch means: for blocks of length tau much longer than the
correlation time, tau * Var(block mean) / 2 -> K. The estimate is read at the plateau (median over tau from 1/20 to
1/6 of the span). Units: nudot in 1e-15 Hz/s, time in days, K in (1e-15 Hz/s)^2 * day."""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
OUT = os.path.join(HERE, "out")


def load(name):
    d = np.loadtxt(os.path.join(DATA, name + ".txt"))
    t, nd = d[:, 0], d[:, 1] * 1e15
    trend = np.polyfit(t - t.mean(), nd, 1)
    return t, nd, nd - np.polyval(trend, t - t.mean()), trend


def batch_K(t, x):
    dt = np.median(np.diff(t))
    span = t[-1] - t[0]
    taus = np.geomspace(span / 60, span / 4, 14)
    ks = []
    for tau in taus:
        m = max(1, int(round(tau / dt)))
        nb = len(x) // m
        means = x[:nb * m].reshape(nb, m).mean(1)
        ks.append(m * dt * means.var(ddof=1) / 2)
    ks = np.array(ks)
    plateau = ks[(taus >= span / 20) & (taus <= span / 6)]
    return taus, ks, float(np.median(plateau))


def main():
    names = sorted(os.path.basename(p)[:-4] for p in glob.glob(os.path.join(DATA, "*.txt")))
    rows = []
    fig, axes = plt.subplots(len(names), 1, figsize=(10, 1.4 * len(names)), sharex=False)
    for ax, name in zip(axes, names):
        t, nd, x, trend = load(name)
        dt = np.median(np.diff(t))
        # two-level structure: 1 vs 2 component Gaussian mixture by BIC
        X = x.reshape(-1, 1)
        g1, g2 = GaussianMixture(1).fit(X), GaussianMixture(2, n_init=5, random_state=0).fit(X)
        dbic = g1.bic(X) - g2.bic(X)
        # dominant period from the periodogram of the detrended series
        f = np.fft.rfftfreq(len(x), dt)
        P = np.abs(np.fft.rfft(x - x.mean())) ** 2
        k = 1 + np.argmax(P[1:])
        period = 1 / f[k]
        taus, ks, K = batch_K(t, x)
        rows.append(dict(psr=name, span_days=float(t[-1] - t[0]), nudot_mean=float(nd.mean()),
                         rel_std=float(x.std() / abs(nd.mean())), dBIC_two_levels=float(dbic),
                         period_days=float(period), K=K, K_rel=float(K / nd.mean() ** 2)))
        ax.plot(t, x, lw=0.7, color="#4a3aff")
        ax.set_ylabel(name, rotation=0, ha="right", fontsize=8)
        ax.tick_params(labelsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "01_series.png"), dpi=110)
    json.dump(rows, open(os.path.join(OUT, "01_summary.json"), "w"), indent=1)
    print(f"{'pulsar':12s} {'nudot':>9s} {'rel sd':>7s} {'dBIC 2 lev':>10s} {'period d':>9s} {'K':>10s}")
    for r in rows:
        print(f"{r['psr']:12s} {r['nudot_mean']:9.2f} {r['rel_std']:7.4f} {r['dBIC_two_levels']:10.1f} "
              f"{r['period_days']:9.0f} {r['K']:10.4g}")


if __name__ == "__main__":
    main()
