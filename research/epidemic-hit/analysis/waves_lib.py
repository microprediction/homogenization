"""Shared wave measurements for daily regional series anchored by serology.

growth(series): early growth rate per day from the contiguous rise of the 7-day mean between 10% and 60% of peak.
attack_at_peak(deaths, sero, sero_date): the serological attack scaled back to the death peak with cumulative deaths,
  antibodies at date t reflecting infections to t - 14 d, i.e. deaths to about t + 7 d.
R_from_r: gamma generation interval, mean 5.5 d, sd 2.1 d."""
import numpy as np, pandas as pd

MEAN_GI, SD_GI = 5.5, 2.1
K_GI, TH_GI = (MEAN_GI / SD_GI) ** 2, SD_GI ** 2 / MEAN_GI


def R_from_r(r):
    return (1 + r * TH_GI) ** K_GI


def smooth(s):
    return s.rolling(7, center=True, min_periods=4).mean()


def growth(s, start=None, end=None):
    """s: daily pd.Series indexed by date. Returns (r per day, rise start, rise end, peak date)."""
    x = smooth(s.loc[start:end].astype(float)).dropna()
    if len(x) < 20 or x.max() <= 0:
        return np.nan, None, None, None
    p = int(np.argmax(x.values)); v = x.values
    i1 = p
    while i1 > 0 and v[i1] > 0.6 * v[p]:
        i1 -= 1
    i0 = i1
    while i0 > 0 and v[i0 - 1] < v[i0] and v[i0 - 1] >= 0.1 * v[p]:
        i0 -= 1
    seg = np.arange(i0, i1 + 1)
    if len(seg) < 5 or v[i0] > 0.35 * v[p]:
        return np.nan, None, None, x.index[p]
    r = np.polyfit(seg, np.log(v[seg]), 1)[0]
    return r, x.index[i0], x.index[i1], x.index[p]


def attack_at_peak(deaths, sero, sero_date, peak_date):
    D = deaths.astype(float).cumsum()
    at_sero = D.loc[:pd.Timestamp(sero_date) + pd.Timedelta(days=7)].iloc[-1]
    at_peak = D.loc[:pd.Timestamp(peak_date)].iloc[-1]
    return sero * at_peak / at_sero if at_sero > 0 else np.nan


def summarize(name, deaths, admissions, sero, sero_date, end="2020-07-31"):
    rD, _, _, pD = growth(deaths, end=end)
    rA, _, _, pA = growth(admissions, end=end) if admissions is not None else (np.nan, None, None, None)
    a = attack_at_peak(deaths, sero, sero_date, pD) if pD is not None else np.nan
    out = dict(place=name, r_deaths=rD, r_adm=rA, R_deaths=R_from_r(rD), R_adm=R_from_r(rA),
               peak_deaths=None if pD is None else str(pD.date()), sero=sero, attack_at_peak=a)
    for key in ("deaths", "adm"):
        R = out[f"R_{key}"]
        out[f"textbook_{key}"] = 1 - 1 / R if R and R > 1 else np.nan
        out[f"lam_{key}"] = np.log(R) / -np.log(1 - a) if R and R > 1 and 0 < a < 1 else np.nan
    return out


def growth_weekly(s, start=None, end=None):
    """Weekly version: 3-week centred mean of weekly totals, contiguous rise between 10% and 60% of the peak.
    Returns (r per day, rise start, rise end, peak week end)."""
    w = s.loc[start:end].astype(float).resample("W-SUN").sum()
    x = w.rolling(3, center=True, min_periods=1).mean()
    v = x.values
    if len(v) < 6 or v.max() <= 0:
        return np.nan, None, None, None
    p = int(np.argmax(v))
    i1 = p
    while i1 > 0 and v[i1] > 0.6 * v[p]:
        i1 -= 1
    i0 = i1
    while i0 > 0 and v[i0 - 1] < v[i0] and v[i0 - 1] >= 0.1 * v[p]:
        i0 -= 1
    seg = np.arange(i0, i1 + 1)
    if len(seg) < 3 or v[i0] > 0.4 * v[p] or v[i0] <= 0:
        return np.nan, None, None, x.index[p]
    return np.polyfit(seg, np.log(v[seg]), 1)[0] / 7, x.index[i0], x.index[i1], x.index[p]
