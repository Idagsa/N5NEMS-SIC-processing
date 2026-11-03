"""Tie points and atmospheric noise reduction following Kolbe et al. (2025, RSE 328, 114815;
SCAMS), with one change: every window is the 14 days before the analysis day instead of
+-7 days around it.

  Tie-point candidates (Table 1):
    ice:   ERA5 SIC > 0.8,  100 K < Tb < 274 K (both channels)
    ocean: ERA5 SIC = 0,     90 K < Tb < 180 K (both channels)
  Ice types (Sect. 4.1.1): GR = (Tb31 - Tb22) / (Tb31 + Tb22); FYI (type A) GR >= -0.015,
    MYI (type B) GR < -0.015. Tie points are means, computed daily and smoothed over the window.
  One-channel ice tie point: mean of all ice candidates.
  RTM (Eq. 6): Tb = a V + b per channel and hemisphere, regression on the ocean candidates
    of the window.
  Correction (Eqs. 7-8): dTb = RTM(V_tp) - RTM(V) = a (V_tp - V), with V_tp the SIC-weighted
    mix of the ice and ocean tie-point water vapour; Tb_c = Tb + (1 - SIC) dTb, SIC the
    uncorrected hybrid SIC truncated to 0-1.
"""
import numpy as np
import pandas as pd

ICE_MIN_SIC = 0.8
ICE_TB = (100.0, 274.0)
OW_TB = (90.0, 180.0)
GR_THRESHOLD = -0.015
WINDOW_DAYS = 14
MIN_SAMPLES = 100


# Additions to Kolbe et al. (2025), on by default (set NEMS_SCAMS_PURE=1 to switch both off):
#   - open-water candidates need an ERA5 SST of 271-275 K (water near the ice edge);
#   - footprint box: ERA5 SIC is also checked at the two along-track neighbours (~100 km
#     either side, i.e. within one 200 x 300 km NEMS footprint): ice candidates need
#     ERA5 SIC > 0.8 at all three, open-water candidates ERA5 SIC < 0.01 at all three.
import os as _os
PURE = _os.environ.get("NEMS_SCAMS_PURE", "0") == "1"
OW_SST = (271.0, 275.0)
OW_BOX_MAX = 0.01
DECILES = _os.environ.get("NEMS_SCAMS_DECILES", "0") == "1"   # v4 FYI/MYI deciles instead of the GR split


def candidates(record, c22, c31):
    t22, t31, c = record[c22], record[c31], record["c_ice"]
    ice = (c > ICE_MIN_SIC) & t22.between(*ICE_TB, inclusive="neither") & t31.between(*ICE_TB, inclusive="neither")
    ow = (c <= 0.0) & t22.between(*OW_TB, inclusive="neither") & t31.between(*OW_TB, inclusive="neither")
    if not PURE:
        ice &= record["c_ice_box_min"] > ICE_MIN_SIC
        ow &= (record["c_ice_box_max"] < OW_BOX_MAX) & record["Ts"].between(*OW_SST)
    return ice, ow


def _window(daily, full):
    """Sum over the WINDOW_DAYS days before each day (the day itself excluded)."""
    d = daily.reindex(full, fill_value=0.0)
    return d.rolling(WINDOW_DAYS, min_periods=1).sum().shift(1)


def _daily_smoothed(values, mask, day, full):
    """Daily mean and std of each column over `mask`, averaged over the window days that
    have data; NaN where the window holds fewer than MIN_SAMPLES observations."""
    g = values[mask].groupby(day[mask])
    mean, std, n = g.mean(), g.std(ddof=0), g.size()
    has = (n > 0).astype(float)
    nwin = _window(n.astype(float), full)
    dwin = _window(has, full)
    out = {}
    for col in values.columns:
        out[f"{col}_mean"] = _window(mean[col].fillna(0.0), full) / dwin
        out[f"{col}_std"] = _window(std[col].fillna(0.0), full) / dwin
    out = pd.DataFrame(out, index=full)
    out[nwin < MIN_SAMPLES] = np.nan
    return out


def tiepoints(record, c22, c31):
    """Daily tie points (FYI/MYI/ICE/OW, 22 and 31 GHz, with std) and the tie-point water
    vapour, on the dates of `record`."""
    day = pd.to_datetime(record["datetime"]).dt.normalize()
    full = pd.date_range(day.min(), day.max(), freq="D")
    ice, ow = candidates(record, c22, c31)
    gr = (record[c31] - record[c22]) / (record[c31] + record[c22])
    vals = pd.DataFrame({"t22": record[c22], "t31": record[c31], "V": record["V"]})
    groups = {"ICE": ice, "FYI": ice & (gr >= GR_THRESHOLD), "MYI": ice & (gr < GR_THRESHOLD), "OW": ow}
    tp = pd.DataFrame(index=full)
    for name, m in groups.items():
        s = _daily_smoothed(vals, m, day, full)
        tp[f"{name}_tp22"], tp[f"{name}_tp31"] = s["t22_mean"], s["t31_mean"]
        tp[f"{name}_tp22_std"], tp[f"{name}_tp31_std"] = s["t22_std"], s["t31_std"]
        tp[f"{name}_V"] = s["V_mean"]
    if DECILES:
        # v4: FYI / MYI = mean of the warmest / coldest 10 % of the ice candidates of the
        # window, each channel ranked separately
        dd = day[ice]
        for ch, col in (("22", c22), ("31", c31)):
            t = record.loc[ice, col]
            by_day = {d: g.to_numpy() for d, g in t.groupby(dd)}
            hi, lo, shi, slo = [], [], [], []
            for d in full:
                w = [by_day[x] for x in pd.date_range(d - pd.Timedelta(days=WINDOW_DAYS), d - pd.Timedelta(days=1)) if x in by_day]
                v = np.sort(np.concatenate(w)) if w else np.array([])
                if v.size < MIN_SAMPLES:
                    hi.append(np.nan); lo.append(np.nan); shi.append(np.nan); slo.append(np.nan); continue
                k = max(1, int(round(0.1 * v.size)))
                hi.append(v[-k:].mean()); lo.append(v[:k].mean()); shi.append(v[-k:].std()); slo.append(v[:k].std())
            tp[f"FYI_tp{ch}"], tp[f"MYI_tp{ch}"] = hi, lo
            tp[f"FYI_tp{ch}_std"], tp[f"MYI_tp{ch}_std"] = shi, slo
    tp = tp.bfill().ffill()
    return tp, ice, ow


def regression(record, ow, c22, c31):
    """Slope a and offset b of Tb = a V + b per day, from the ocean candidates of the window."""
    day = pd.to_datetime(record["datetime"]).dt.normalize()
    full = pd.date_range(day.min(), day.max(), freq="D")
    V = record.loc[ow, "V"]
    d = day[ow]
    out = pd.DataFrame(index=full)
    n = _window(V.groupby(d).size().astype(float), full)
    sv = _window(V.groupby(d).sum(), full)
    svv = _window((V * V).groupby(d).sum(), full)
    for ch, col in (("22", c22), ("31", c31)):
        T = record.loc[ow, col]
        st = _window(T.groupby(d).sum(), full)
        svt = _window((V * T).groupby(d).sum(), full)
        a = (n * svt - sv * st) / (n * svv - sv ** 2)
        b = (st - a * sv) / n
        a[n < MIN_SAMPLES] = np.nan
        b[n < MIN_SAMPLES] = np.nan
        out[f"a{ch}"], out[f"b{ch}"] = a, b
    return out.bfill().ffill()


def correct(record, sic, tp, reg, c22="tb_22_measured", c31="tb_31_measured"):
    """Eqs. (7)-(8): returns the corrected 22 and 31 GHz Tb."""
    day = pd.to_datetime(record["datetime"]).dt.normalize()
    s = np.clip(np.asarray(sic, float), 0.0, 1.0)
    v_ice = tp["ICE_V"].reindex(day).to_numpy()
    v_ow = tp["OW_V"].reindex(day).to_numpy()
    v_tp = s * v_ice + (1.0 - s) * v_ow
    dV = v_tp - record["V"].to_numpy()
    out = []
    for ch, col in (("22", c22), ("31", c31)):
        a = reg[f"a{ch}"].reindex(day).to_numpy()
        out.append(record[col].to_numpy() + (1.0 - s) * a * dV)
    return out[0], out[1]
