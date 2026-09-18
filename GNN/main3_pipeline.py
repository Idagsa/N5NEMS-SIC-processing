"""
main3_pipeline.py — NEMS SIC physical retrieval pipeline

Plotting-free version of the pipeline half of the original main3.py. All
figure/analysis code that used to sit interleaved with this (as separate
"#%%" cells, several of them duplicated) has moved to main3_figures.py,
which uses the objects this script builds (arctic_record, antarctic_record,
ds, arctic_lmask, arctic_smask, ...) and expects to be run right after this
one, in the same session -- exactly how the original file was actually
used, just now split into two files instead of interleaved cells.

WHAT THIS DOES
--------------
  1. Load and merge NEMS-colocated observations into per-hemisphere
     DataFrames (arctic_record, antarctic_record).
  2. Compute uncorrected tie-points and an initial hybrid SIC from measured
     brightness temperatures.
  3. Apply RTM physics correction using a per-pixel, tie-point-weighted
     reference atmospheric forcing (Kolbe et al., 2024, ESSD).
  4. Recompute tie-points and hybrid SIC from the corrected brightness
     temperatures.
  5. Build the algorithm uncertainty budget (1-channel, 2-channel, hybrid,
     final selection).
  6. Resample each month to EASE2, apply the land-spillover correction, and
     form the total (algorithm + resampling) uncertainty
     (arctic_monthly_results, antarctic_monthly_results).
"""

import numpy as np
import pandas as pd
import xarray as xr
import glob
import os
from tiepoints_func import (analyze_sea_ice_tiepoints, analyze_ow_tiepoints,
                             extract_tiepoints1, extract_tiepoints2,
                             map_tiepoints_to_record)
from algos import hybrid, onechannel_22, bootstrapF
from RTM import RadiativeTransferModel
from uncertainty import (resample_to_ease2, calculate_resampling_uncertainty,
                          calculate_total_uncertainty, calculate_sigma_ow_ice,
                          calculate_weight, calculate_delta_2ch,
                          calculate_delta_hybrid, select_final_uncertainty)
from correction import apply_land_spillover_correction_reference

# Load and merge NEMS data
folder = r"C:\Users\user\OneDrive\Desktop\CCI\NEMS_colocated"
files = sorted(glob.glob(os.path.join(folder, "**.nc")))
ds = xr.open_mfdataset(files, combine="nested", concat_dim="t")

# Landsmasks, climatology and spillover for later use
arctic_lmask = r"C:\Users\user\OneDrive\Desktop\CCI\reference_files\LandOceanLakeMask_nh_ease2-250_v2.nc"
antarctic_lmask = r"C:\Users\user\OneDrive\Desktop\CCI\reference_files\LandOceanLakeMask_sh_ease2-250_v2.nc"

arctic_maskdata = xr.open_dataset(arctic_lmask)
antarctic_maskdata = xr.open_dataset(antarctic_lmask)

arctic_smask = arctic_maskdata['smask_sicci'].to_numpy().astype(int)
antarctic_smask = antarctic_maskdata['smask_sicci'].to_numpy().astype(int)

arctic_maskdata.close()
antarctic_maskdata.close()

# Pre-trained models
arctic_22_model = r"C:\Users\user\OneDrive\Desktop\CCI\trained_models\arctic_22_model.pkl"
arctic_31_model = r"C:\Users\user\OneDrive\Desktop\CCI\trained_models\arctic_31_model.pkl"
antarctic_22_model = r"C:\Users\user\OneDrive\Desktop\CCI\trained_models\antarctic_22_model.pkl"
antarctic_31_model = r"C:\Users\user\OneDrive\Desktop\CCI\trained_models\antarctic_31_model.pkl"

# Prepare list for all records
records = []

for frame in range(14):
    lat = ds["LAT"].isel(n16_frames=frame)
    lon = ds["LON"].isel(n16_frames=frame)
    V = ds["tcwv"].isel(n16_frames=frame)
    Ta = ds["t2m"].isel(n16_frames=frame)
    Ts = ds["sst"].isel(n16_frames=frame)
    c_ice = ds["siconc"].isel(n16_frames=frame)
    tcw = ds["tcw"].isel(n16_frames=frame)
    L = tcw - V
    W = np.sqrt(ds["v10"].isel(n16_frames=frame)**2 +
                ds["u10"].isel(n16_frames=frame)**2)
    
    tb_22_measured = ds["TBNEMS"].isel(n5_channels=0, n14_frames=frame)
    tb_31_measured = ds["TBNEMS"].isel(n5_channels=1, n14_frames=frame)
    time = pd.to_datetime(ds["Time"].isel(n16_frames=frame).values)
    
    frame_df = pd.DataFrame({
        "lat": lat.values.flatten(),
        "lon": lon.values.flatten(),
        "V": V.values.flatten(),
        "Ta": Ta.values.flatten(),
        "Ts": Ts.values.flatten(),
        "c_ice": c_ice.values.flatten(),
        "tcw": tcw.values.flatten(),
        "L": L.values.flatten(),
        "W": W.values.flatten(),
        "tb_22_measured": tb_22_measured.values.flatten(),
        "tb_31_measured": tb_31_measured.values.flatten(),
        "time": time})
    
    frame_df_filtered = frame_df[(frame_df["tb_22_measured"] >= 100) &
                                 (frame_df["tb_22_measured"] <= 290) &
                                 (frame_df["tb_31_measured"] >= 100) &
                                 (frame_df["tb_31_measured"] <= 290)].copy()
    
    frame_df_filtered['month'] = frame_df_filtered['time'].dt.month
    frame_df_filtered['year'] = frame_df_filtered['time'].dt.year
    frame_df_filtered['day'] = frame_df_filtered['time'].dt.day
    frame_df_filtered['datetime'] = frame_df_filtered['time']
    
    records.append(frame_df_filtered)

df = pd.concat(records, ignore_index=True).dropna()

# Hemisphere masks
mask_north = df['lat'] >= 42
mask_south = df['lat'] <= -42

# Southern hemisphere DataFrame
antarctic_record = pd.DataFrame({"tb_22_measured": df.loc[mask_south, "tb_22_measured"],
                                "tb_31_measured": df.loc[mask_south, "tb_31_measured"],
                                "c_ice": df.loc[mask_south, "c_ice"],
                                "W": df.loc[mask_south, "W"],
                                "lat": df.loc[mask_south, "lat"],
                                "lon": df.loc[mask_south, "lon"],
                                "Ts": df.loc[mask_south, "Ts"],
                                "Ta": df.loc[mask_south, "Ta"],
                                "V": df.loc[mask_south, "V"],
                                "L": df.loc[mask_south, "L"],
                                "datetime": df.loc[mask_south, "datetime"],
                                "year": df.loc[mask_south, "year"],
                                "month": df.loc[mask_south, "month"],
                                "day": df.loc[mask_south, "day"]})

# Northern hemisphere DataFrame
arctic_record = pd.DataFrame({"tb_22_measured": df.loc[mask_north, "tb_22_measured"],
                             "tb_31_measured": df.loc[mask_north, "tb_31_measured"],
                             "c_ice": df.loc[mask_north, "c_ice"],
                             "W": df.loc[mask_north, "W"],
                             "lat": df.loc[mask_north, "lat"],
                             "lon": df.loc[mask_north, "lon"],
                             "Ts": df.loc[mask_north, "Ts"],
                             "Ta": df.loc[mask_north, "Ta"],
                             "V": df.loc[mask_north, "V"],
                             "L": df.loc[mask_north, "L"],
                             "datetime": df.loc[mask_north, "datetime"],
                             "year": df.loc[mask_north, "year"],
                             "month": df.loc[mask_north, "month"],
                             "day": df.loc[mask_north, "day"]})


# Calculate UNCORRECTED tie-points from measured data
# For sea ice
antarctic_results = analyze_sea_ice_tiepoints(antarctic_record, 
                                              min_c_ice=0.95,
                                              rolling_mean_window=14,
                                              iqr_multiplier=1.0)

arctic_results = analyze_sea_ice_tiepoints(arctic_record, 
                                           min_c_ice=0.95,
                                           rolling_mean_window=14,
                                           iqr_multiplier=1.0)

# For Open Water
antarctic_OW = analyze_ow_tiepoints(antarctic_record,
                                    min_sst=271,
                                    max_sst=275,      
                                    max_c_ice=0.0,
                                    min_samples=100,
                                    rolling_mean_window=14)

arctic_OW = analyze_ow_tiepoints(arctic_record,
                                 min_sst=271,
                                 max_sst=275,
                                 max_c_ice=0.0,
                                 min_samples=100,
                                 rolling_mean_window=14)
# Extract corrected tiepoints
antarctic_tps = extract_tiepoints1(antarctic_results, antarctic_OW)
arctic_tps = extract_tiepoints1(arctic_results, arctic_OW)

# Map to records
map_tiepoints_to_record(arctic_record, arctic_tps)
map_tiepoints_to_record(antarctic_record, antarctic_tps)


# Fill any remaining NaNs
tiepoint_columns = ['FYI_tp22', 'FYI_tp31', 'MYI_tp22', 'MYI_tp31', 'OW_tp22', 'OW_tp31']
arctic_record[tiepoint_columns] = arctic_record[tiepoint_columns].bfill()
antarctic_record[tiepoint_columns] = antarctic_record[tiepoint_columns].bfill()

# Calculate initial SIC using HYBRID algorithm from measured TB

# Hybrid algorithm from measured TB
arctic_record['c_ice_hybrid'] = arctic_record.apply(
    lambda row: hybrid(
        row['tb_22_measured'],
        row['tb_31_measured'],
        row['FYI_tp22'],
        row['MYI_tp22'],
        row['OW_tp22'],
        row['FYI_tp31'],
        row['MYI_tp31'],
        row['OW_tp31']
    ) if all(pd.notna(row[col]) for col in ['FYI_tp22', 'MYI_tp22', 'OW_tp22',
                                              'FYI_tp31', 'MYI_tp31', 'OW_tp31',
                                              'tb_22_measured', 'tb_31_measured'])
    else np.nan,
    axis=1
)

antarctic_record['c_ice_hybrid'] = antarctic_record.apply(
    lambda row: hybrid(
        row['tb_22_measured'],
        row['tb_31_measured'],
        row['FYI_tp22'],
        row['MYI_tp22'],
        row['OW_tp22'],
        row['FYI_tp31'],
        row['MYI_tp31'],
        row['OW_tp31']
    ) if all(pd.notna(row[col]) for col in ['FYI_tp22', 'MYI_tp22', 'OW_tp22',
                                              'FYI_tp31', 'MYI_tp31', 'OW_tp31',
                                              'tb_22_measured', 'tb_31_measured'])
    else np.nan,
    axis=1
)


# Apply physics correction to brightness temperatures, following the
# dynamical-tie-point RTM correction approach of Kolbe et al. (2024, ESSD),
# with an added XGBoost bias-correction layer inside the RTM itself.
#
# Kolbe et al. do NOT use a single, static hemispheric-mean reference state.
# Instead, the "reference" atmospheric forcing for each pixel is a per-pixel
# mixture of the daily (15-day-smoothed) mean atmospheric state observed at
# the ice tie-point cluster and the open-water tie-point cluster, mixed in
# proportion to that pixel's own sea ice concentration:
#
#   X_ref(pixel) = c_ice(pixel) * X_ice_cluster(day)
#                + (1 - c_ice(pixel)) * X_water_cluster(day)
#
# for each atmospheric variable X in {V, W, L, Ta, Ts}. This makes the
# reference state track seasonal/regional atmospheric conditions rather than
# a single fixed climatology, while still isolating "weather noise" (the
# deviation of the pixel's *actual* conditions from this smoothly-varying
# reference) via the same before/after RTM differencing trick as before.
#
# Initialize RTM
rtm = RadiativeTransferModel(apply_correction=True)
rtm.set_bias_models(arctic_22_model, arctic_31_model, antarctic_22_model, antarctic_31_model)

ATMOS_VARS = ['V', 'W', 'L', 'Ta', 'Ts']
TIEPOINT_ROLLING_DAYS = 15  # matches Kolbe et al.'s 15-day (7 fwd/7 back) window


def compute_daily_cluster_means(record, ice_mask, water_mask, atmos_vars=ATMOS_VARS,
                                 window=TIEPOINT_ROLLING_DAYS):
    """
    Compute daily-mean atmospheric state at the ice and open-water tie-point
    clusters, smoothed with a centered rolling-mean window (Kolbe et al.,
    2024, Sect. 4.1). Returns a DataFrame indexed by date with columns
    '{var}_ice_daily' and '{var}_water_daily' for each atmospheric variable.
    """
    record = record.copy()
    record['date'] = pd.to_datetime(record['datetime']).dt.normalize()

    ice_daily = (record.loc[ice_mask, ['date'] + atmos_vars]
                 .groupby('date').mean())
    water_daily = (record.loc[water_mask, ['date'] + atmos_vars]
                   .groupby('date').mean())

    # Reindex onto the full daily range so the rolling window is continuous
    # even across days with no qualifying tie-point observations.
    full_range = pd.date_range(record['date'].min(), record['date'].max(), freq='D')
    ice_daily = ice_daily.reindex(full_range)
    water_daily = water_daily.reindex(full_range)

    ice_daily = ice_daily.rolling(window=window, center=True, min_periods=1).mean()
    water_daily = water_daily.rolling(window=window, center=True, min_periods=1).mean()

    ice_daily = ice_daily.add_suffix('_ice_daily')
    water_daily = water_daily.add_suffix('_water_daily')

    daily_cluster_means = ice_daily.join(water_daily)
    # Backward/forward fill any remaining gaps (e.g. missing files at the
    # very start/end of the record), same convention used for tie points
    # elsewhere in this script.
    daily_cluster_means = daily_cluster_means.bfill().ffill()
    return daily_cluster_means


def apply_weighted_reference_forcing(record, daily_cluster_means, atmos_vars=ATMOS_VARS):
    """
    Merge the daily ice/water cluster means onto each pixel by date, then
    mix them per-pixel using c_ice_hybrid as the mixing ratio to produce the
    reference forcing X_ref for each atmospheric variable.
    """
    record = record.copy()
    record['date'] = pd.to_datetime(record['datetime']).dt.normalize()
    record = record.join(daily_cluster_means, on='date')

    c = record['c_ice_hybrid']
    for var in atmos_vars:
        record[f'{var}_ref'] = (c * record[f'{var}_ice_daily']
                                 + (1.0 - c) * record[f'{var}_water_daily'])
    return record


# Ice / open-water cluster masks, matching the criteria already used for
# tie-point extraction (analyze_sea_ice_tiepoints / analyze_ow_tiepoints)
arctic_ice_mask = arctic_record['c_ice_hybrid'] >= 0.95
arctic_water_mask = ((arctic_record['c_ice_hybrid'] <= 0.0) &
                      (arctic_record['Ts'] >= 271) & (arctic_record['Ts'] <= 275) &
                      (arctic_record['W'] <= 15.0))

antarctic_ice_mask = antarctic_record['c_ice_hybrid'] >= 0.95
antarctic_water_mask = ((antarctic_record['c_ice_hybrid'] <= 0.0) &
                         (antarctic_record['Ts'] >= 271) & (antarctic_record['Ts'] <= 275) &
                         (antarctic_record['W'] <= 15.0))

arctic_daily_cluster_means = compute_daily_cluster_means(arctic_record, arctic_ice_mask, arctic_water_mask)
antarctic_daily_cluster_means = compute_daily_cluster_means(antarctic_record, antarctic_ice_mask, antarctic_water_mask)

arctic_record = apply_weighted_reference_forcing(arctic_record, arctic_daily_cluster_means)
antarctic_record = apply_weighted_reference_forcing(antarctic_record, antarctic_daily_cluster_means)

# Calculate reference RTM (per-pixel, tie-point-weighted forcing) - Arctic
rtm.set_region('arctic')
arctic_tb_ref = np.array([rtm.simulate(V=V_r, W=W_r, L=L_r, Ta=Ta_r, Ts=Ts_r, c_ice=c)
                          for V_r, W_r, L_r, Ta_r, Ts_r, c
                          in zip(arctic_record['V_ref'].values,
                                 arctic_record['W_ref'].values,
                                 arctic_record['L_ref'].values,
                                 arctic_record['Ta_ref'].values,
                                 arctic_record['Ts_ref'].values,
                                 arctic_record['c_ice_hybrid'].values)])

# Calculate reference RTM (per-pixel, tie-point-weighted forcing) - Antarctic
rtm.set_region('antarctic')
antarctic_tb_ref = np.array([rtm.simulate(V=V_r, W=W_r, L=L_r, Ta=Ta_r, Ts=Ts_r, c_ice=c)
                             for V_r, W_r, L_r, Ta_r, Ts_r, c
                             in zip(antarctic_record['V_ref'].values,
                                    antarctic_record['W_ref'].values,
                                    antarctic_record['L_ref'].values,
                                    antarctic_record['Ta_ref'].values,
                                    antarctic_record['Ts_ref'].values,
                                    antarctic_record['c_ice_hybrid'].values)])

arctic_record['tb_22_ref'] = arctic_tb_ref[:, 0]
arctic_record['tb_31_ref'] = arctic_tb_ref[:, 1]
antarctic_record['tb_22_ref'] = antarctic_tb_ref[:, 0]
antarctic_record['tb_31_ref'] = antarctic_tb_ref[:, 1]

# Simulate TB with RTM - Arctic
rtm.set_region('arctic')
arctic_tb_sim = np.array([rtm.simulate(V=V_i, W=W_i, L=L_i, Ta=Ta_i, Ts=Ts_i, c_ice=c_i)
                          for V_i, W_i, L_i, Ta_i, Ts_i, c_i 
                          in zip(arctic_record['V'].values,
                                 arctic_record['W'].values,
                                 arctic_record['L'].values,
                                 arctic_record['Ta'].values,
                                 arctic_record['Ts'].values,
                                 arctic_record['c_ice_hybrid'].values)])

arctic_record['tb_22_sim'] = arctic_tb_sim[:, 0]
arctic_record['tb_31_sim'] = arctic_tb_sim[:, 1]

# Simulate TB with RTM - Antarctic
rtm.set_region('antarctic')
antarctic_tb_sim = np.array([rtm.simulate(V=V_i, W=W_i, L=L_i, Ta=Ta_i, Ts=Ts_i, c_ice=c_i)
                             for V_i, W_i, L_i, Ta_i, Ts_i, c_i
                             in zip(antarctic_record['V'].values,
                                    antarctic_record['W'].values,
                                    antarctic_record['L'].values,
                                    antarctic_record['Ta'].values,
                                    antarctic_record['Ts'].values,
                                    antarctic_record['c_ice_hybrid'].values)])

antarctic_record['tb_22_sim'] = antarctic_tb_sim[:, 0]
antarctic_record['tb_31_sim'] = antarctic_tb_sim[:, 1]

# Apply physics correction
arctic_delta_22 = arctic_record['tb_22_ref'] - arctic_record['tb_22_sim']
arctic_delta_31 = arctic_record['tb_31_ref'] - arctic_record['tb_31_sim']
antarctic_delta_22 = antarctic_record['tb_22_ref'] - antarctic_record['tb_22_sim']
antarctic_delta_31 = antarctic_record['tb_31_ref'] - antarctic_record['tb_31_sim']

arctic_record['tb_22_phys_corrected'] = arctic_record['tb_22_measured'] + arctic_delta_22
arctic_record['tb_31_phys_corrected'] = arctic_record['tb_31_measured'] + arctic_delta_31
antarctic_record['tb_22_phys_corrected'] = antarctic_record['tb_22_measured'] + antarctic_delta_22
antarctic_record['tb_31_phys_corrected'] = antarctic_record['tb_31_measured'] + antarctic_delta_31

# Calculate CORRECTED tie-points from physics-corrected data
# For sea ice 
antarctic_results_corrected = analyze_sea_ice_tiepoints(
    antarctic_record, 
    min_c_ice=0.95,
    col_c_ice='c_ice_hybrid',
    col_22='tb_22_phys_corrected', 
    col_31='tb_31_phys_corrected',
    rolling_mean_window=14,
    iqr_multiplier=1.0)

arctic_results_corrected = analyze_sea_ice_tiepoints(
    arctic_record, 
    min_c_ice=0.95,
    col_c_ice='c_ice_hybrid',
    col_22='tb_22_phys_corrected', 
    col_31='tb_31_phys_corrected',
    rolling_mean_window=14,
    iqr_multiplier=1.0)

# For Open Water - WITH OUTLIER REMOVAL
antarctic_OW_corrected = analyze_ow_tiepoints(
    antarctic_record,
    col_22='tb_22_phys_corrected',
    col_31='tb_31_phys_corrected',
    min_sst=271,
    max_sst=275,
    max_c_ice=0.0,
    min_samples=100,
    rolling_mean_window=14)

arctic_OW_corrected = analyze_ow_tiepoints(
    arctic_record,
    col_22='tb_22_phys_corrected',
    col_31='tb_31_phys_corrected',
    min_sst=271,
    max_sst=275,
    max_c_ice=0.0,
    min_samples=100,
    rolling_mean_window=14
)

# Extract corrected tiepoints
antarctic_tps_corrected = extract_tiepoints2(antarctic_results_corrected, antarctic_OW_corrected)
arctic_tps_corrected = extract_tiepoints2(arctic_results_corrected, arctic_OW_corrected)

# Map to records
map_tiepoints_to_record(arctic_record, arctic_tps_corrected)
map_tiepoints_to_record(antarctic_record, antarctic_tps_corrected)

# Backward fill
corrected_cols = ['FYI_tp22_corrected', 'FYI_tp31_corrected', 'MYI_tp22_corrected', 
                  'MYI_tp31_corrected', 'OW_tp22_corrected', 'OW_tp31_corrected',
                  'FYI_tp22_corrected_std', 'FYI_tp31_corrected_std', 
                  'MYI_tp22_corrected_std', 'MYI_tp31_corrected_std', 
                  'OW_tp22_corrected_std', 'OW_tp31_corrected_std']
arctic_record[corrected_cols] = arctic_record[corrected_cols].bfill()
antarctic_record[corrected_cols] = antarctic_record[corrected_cols].bfill()


# Hybrid c_ice algorithm with corrected data
arctic_record['c_ice_hybrid_corrected'] = arctic_record.apply(
    lambda row: hybrid(
        row['tb_22_phys_corrected'],
        row['tb_31_phys_corrected'],
        row['FYI_tp22_corrected'],
        row['MYI_tp22_corrected'],
        row['OW_tp22_corrected'],
        row['FYI_tp31_corrected'],
        row['MYI_tp31_corrected'],
        row['OW_tp31_corrected']
    ) if all(pd.notna(row[col]) for col in ['FYI_tp22_corrected', 'MYI_tp22_corrected', 'OW_tp22_corrected',
                                              'FYI_tp31_corrected', 'MYI_tp31_corrected', 'OW_tp31_corrected',
                                              'tb_22_phys_corrected', 'tb_31_phys_corrected'])
    else np.nan,
    axis=1
)

antarctic_record['c_ice_hybrid_corrected'] = antarctic_record.apply(
    lambda row: hybrid(
        row['tb_22_phys_corrected'],
        row['tb_31_phys_corrected'],
        row['FYI_tp22_corrected'],
        row['MYI_tp22_corrected'],
        row['OW_tp22_corrected'],
        row['FYI_tp31_corrected'],
        row['MYI_tp31_corrected'],
        row['OW_tp31_corrected']
    ) if all(pd.notna(row[col]) for col in ['FYI_tp22_corrected', 'MYI_tp22_corrected', 'OW_tp22_corrected',
                                              'FYI_tp31_corrected', 'MYI_tp31_corrected', 'OW_tp31_corrected',
                                              'tb_22_phys_corrected', 'tb_31_phys_corrected'])
    else np.nan,
    axis=1
)


# ===========================================================================
# Algorithm uncertainties
# ===========================================================================
def calculate_delta_1ch(sic_1ch, tp_MYI, tp_FYI, tp_water,
                        std_MYI, std_FYI, std_water):
    """
    One-channel algorithm uncertainty (Equation 10).

    std_* are the tie-point standard deviations. tp_ice is the mean of the MYI
    and FYI tie points, so its uncertainty combines theirs in quadrature and
    halves.
    """
    if pd.isna(sic_1ch) or pd.isna(tp_MYI) or pd.isna(tp_FYI) or pd.isna(tp_water):
        return np.nan
    if pd.isna(std_MYI) or pd.isna(std_FYI) or pd.isna(std_water):
        return np.nan

    tp_ice = (tp_MYI + tp_FYI) / 2
    tp_diff = tp_ice - tp_water
    if tp_diff == 0:
        return np.nan

    delta_tp_ice = np.sqrt(std_MYI ** 2 + std_FYI ** 2) / 2
    delta_tp_water = std_water

    term1 = ((1 - sic_1ch) * delta_tp_water / tp_diff) ** 2
    term2 = (sic_1ch * delta_tp_ice / tp_diff) ** 2
    return np.sqrt(term1 + term2)


DELTA1_COLS = ['c_ice_hybrid_corrected',
               'MYI_tp22_corrected', 'FYI_tp22_corrected', 'OW_tp22_corrected',
               'MYI_tp22_corrected_std', 'FYI_tp22_corrected_std',
               'OW_tp22_corrected_std']

for _record in (arctic_record, antarctic_record):
    _record['delta_1ch'] = _record.apply(
        lambda row: calculate_delta_1ch(*[row[c] for c in DELTA1_COLS]), axis=1)


# Component retrievals, needed because calculate_sigma_ow_ice takes a
# 'c_ice_2ch_corrected' column. NOTE bootstrapF takes tb_31 BEFORE tb_22,
# the reverse of hybrid().
_TP_CORR = ['FYI_tp22_corrected', 'MYI_tp22_corrected', 'OW_tp22_corrected',
            'FYI_tp31_corrected', 'MYI_tp31_corrected', 'OW_tp31_corrected']
_TB_CORR = ['tb_22_phys_corrected', 'tb_31_phys_corrected']


def add_component_sic(record):
    """Add c_ice_1ch_corrected and c_ice_2ch_corrected."""
    ok = record[_TB_CORR + _TP_CORR].notna().all(axis=1)
    one = pd.Series(np.nan, index=record.index, dtype=float)
    two = pd.Series(np.nan, index=record.index, dtype=float)
    sub = record.loc[ok]
    one.loc[ok] = [onechannel_22(r['tb_22_phys_corrected'],
                                 r['FYI_tp22_corrected'],
                                 r['MYI_tp22_corrected'],
                                 r['OW_tp22_corrected'])
                   for _, r in sub.iterrows()]
    two.loc[ok] = [bootstrapF(r['tb_31_phys_corrected'],
                              r['tb_22_phys_corrected'],
                              r['FYI_tp22_corrected'],
                              r['MYI_tp22_corrected'],
                              r['OW_tp22_corrected'],
                              r['FYI_tp31_corrected'],
                              r['MYI_tp31_corrected'],
                              r['OW_tp31_corrected'])
                   for _, r in sub.iterrows()]
    record['c_ice_1ch_corrected'] = one
    record['c_ice_2ch_corrected'] = two
    return record


add_component_sic(arctic_record)
add_component_sic(antarctic_record)

# Calculate sigma_ow and sigma_ice using ERA5 c_ice as reference.
# Set SIGMA_SOURCE = 'c_ice_hybrid_corrected' to restore the previous behaviour.
SIGMA_SOURCE = 'c_ice_2ch_corrected'

arctic_sigma_ow, arctic_sigma_ice = calculate_sigma_ow_ice(
    arctic_record.dropna(subset=[SIGMA_SOURCE, 'c_ice']),
    c_ice_2ch_col=SIGMA_SOURCE, c_ice_ref_col='c_ice')
antarctic_sigma_ow, antarctic_sigma_ice = calculate_sigma_ow_ice(
    antarctic_record.dropna(subset=[SIGMA_SOURCE, 'c_ice']),
    c_ice_2ch_col=SIGMA_SOURCE, c_ice_ref_col='c_ice')

print(f'Arctic    sigma_ow {arctic_sigma_ow:.4f}  sigma_ice {arctic_sigma_ice:.4f}')
print(f'Antarctic sigma_ow {antarctic_sigma_ow:.4f}  sigma_ice {antarctic_sigma_ice:.4f}')


# Calculate 2-channel uncertainty 
arctic_record['delta_2ch'] = arctic_record['c_ice_hybrid_corrected'].apply(
    lambda sic: calculate_delta_2ch(sic, 
                                    arctic_sigma_ow,
                                    arctic_sigma_ice))

antarctic_record['delta_2ch'] = antarctic_record['c_ice_hybrid_corrected'].apply(
    lambda sic: calculate_delta_2ch(sic,
                                    antarctic_sigma_ow,
                                    antarctic_sigma_ice))

print("Calculating weight...")
# Calculate weight
arctic_record['weight'] = arctic_record['c_ice_hybrid_corrected'].apply(calculate_weight)
antarctic_record['weight'] = antarctic_record['c_ice_hybrid_corrected'].apply(calculate_weight)

print("Calculating hybrid uncertainties...")
# Calculate hybrid uncertainty
arctic_record['delta_hybrid'] = arctic_record.apply(
    lambda row: calculate_delta_hybrid(row['delta_1ch'],
                                       row['delta_2ch'],
                                       row['weight']), axis=1)

antarctic_record['delta_hybrid'] = antarctic_record.apply(
    lambda row: calculate_delta_hybrid(row['delta_1ch'],
                                       row['delta_2ch'],
                                       row['weight']),axis=1)

print("Hybrid algorithm uncertainty...")
# Select final algorithm uncertainty based on SIC value
arctic_record['delta_algorithm'] = arctic_record.apply(
    lambda row: select_final_uncertainty(row['c_ice_hybrid_corrected'],
                                         row['delta_1ch'],
                                         row['delta_2ch'],
                                         row['delta_hybrid']),axis=1)

antarctic_record['delta_algorithm'] = antarctic_record.apply(
    lambda row: select_final_uncertainty(row['c_ice_hybrid_corrected'],
                                         row['delta_1ch'],
                                         row['delta_2ch'],
                                         row['delta_hybrid']),axis=1)


# ===========================================================================
# Process all months
# ===========================================================================
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

RESAMPLE_FEATURES = ['tb_22_phys_corrected', 'tb_31_phys_corrected',
                     'c_ice_hybrid_corrected', 'uncertainty_algorithm']


def process_month(record, area, mask_path, smask, month,
                  features=RESAMPLE_FEATURES, filter_size=9):
    """Resample one month to EASE2, correct land spillover, form the total
    uncertainty. Returns None if the month holds no observations."""
    sub = record[record['month'] == month].copy()
    if len(sub) == 0:
        return None

    sub['uncertainty_algorithm'] = sub['delta_algorithm']

    grids = resample_to_ease2(sub, area, features,
                              mask_path=mask_path, month=month)
    sic_raw = grids[:, :, features.index('c_ice_hybrid_corrected')]
    unc_alg = grids[:, :, features.index('uncertainty_algorithm')]

    sic = apply_land_spillover_correction_reference(
        sic_raw, smask, filter_size=filter_size, validate=False)

    # Resampling term from the 3x3 spread of the delivered field. Pass sic_raw
    # instead if it should describe the resampling step alone.
    unc_resamp = calculate_resampling_uncertainty(sic, window_size=3)
    unc_total = calculate_total_uncertainty(unc_alg, unc_resamp)

    return {'sic': sic,
            'sic_before_spillover': sic_raw,
            'uncertainty': unc_total,
            'uncertainty_algorithm': unc_alg,
            'uncertainty_resampling': unc_resamp,
            'tb_22': grids[:, :, features.index('tb_22_phys_corrected')],
            'tb_31': grids[:, :, features.index('tb_31_phys_corrected')],
            'n_obs': len(sub)}


arctic_monthly_results = {}
antarctic_monthly_results = {}

for month in range(1, 13):
    arctic_monthly_results[month] = process_month(
        arctic_record, 'arctic', arctic_lmask, arctic_smask, month)
    antarctic_monthly_results[month] = process_month(
        antarctic_record, 'antarctic', antarctic_lmask, antarctic_smask, month)

    for _name, _res in (('Arctic', arctic_monthly_results[month]),
                        ('Antarctic', antarctic_monthly_results[month])):
        if _res is None:
            continue
        _shift = np.nanmean(np.abs(_res['sic'] - _res['sic_before_spillover']))
        print(f"{month_names[month - 1]} {_name:<10} n={_res['n_obs']:>7,}  "
              f"unc alg {100 * np.nanmean(_res['uncertainty_algorithm']):5.2f}%  "
              f"resamp {100 * np.nanmean(_res['uncertainty_resampling']):5.2f}%  "
              f"total {100 * np.nanmean(_res['uncertainty']):5.2f}%  "
              f"spillover moved SIC by {100 * _shift:5.2f}%")

# Sanity check: delta_algorithm should be non-zero over open water.
for _record, _name in ((arctic_record, 'Arctic'), (antarctic_record, 'Antarctic')):
    _v = _record.loc[_record['c_ice'] == 0, 'delta_algorithm'].dropna()
    if len(_v):
        print(f'{_name}: delta_algorithm over ERA5 open water  median '
              f'{100 * _v.median():.2f}%   exactly zero {int((_v == 0).sum()):,}'
              f' of {len(_v):,}')
        





# =============================================================================
# Sanity check: delta_1ch and tie-point spread
# =============================================================================
for rec, nm in ((arctic_record, 'Arctic'), (antarctic_record, 'Antarctic')):
    d1 = rec['delta_1ch'].dropna()
    print(f'{nm}: delta_1ch  n={len(d1):,}  max={d1.max():.6f}  '
          f'nonzero={int((d1 > 0).sum()):,}')
    print(f'{nm}: OW_tp22_corrected_std  '
          f'median={rec["OW_tp22_corrected_std"].median():.4f}  '
          f'zeros={int((rec["OW_tp22_corrected_std"] == 0).sum()):,}')


