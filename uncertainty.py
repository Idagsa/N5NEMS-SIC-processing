# -*- coding: utf-8 -*-
"""
Created on Fri Nov 7 14:26:15 2025
@author: user
"""
import numpy as np
import pandas as pd
import xarray as xr
from pyresample import kd_tree, geometry
from pyresample.utils import fwhm2sigma
import numpy as np
import pandas as pd
from scipy.ndimage import generic_filter

def calculate_weight(sic_1ch, low_threshold=0.40, high_threshold=0.75,
                     clamp=True):
    """Weight for the hybrid algorithm, w = (SIC - 0.4) / (0.75 - 0.4).
 
    clamp=True bounds w to [0, 1]. This cannot change delta_algorithm, since
    select_final_uncertainty only uses delta_hybrid inside the transition
    band; it only keeps delta_hybrid interpretable outside it. clamp=False
    reproduces the original behaviour exactly.
    """
    if pd.isna(sic_1ch):
        return np.nan
    weight = (sic_1ch - low_threshold) / (high_threshold - low_threshold)
    return float(np.clip(weight, 0.0, 1.0)) if clamp else weight


def calculate_delta_1ch(sic_1ch, tp_MYI, tp_FYI, tp_water,
                        std_MYI, std_FYI, std_water):
    """One-channel algorithm uncertainty (Equation 10).
 
    std_MYI, std_FYI, std_water are the tie-point standard deviations, i.e.
    the *_corrected_std columns. tp_ice is the mean of the MYI and FYI tie
    points, so its standard error combines theirs in quadrature and halves.
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
 


def calculate_delta_2ch(sic_2ch, sigma_ow, sigma_ice):
    """
    Calculate two-channel algorithm uncertainty (Equation 11)
    
    Parameters:
    -----------
    sic_2ch : float
        Sea ice concentration value from 2-channel algorithm
    sigma_ow : float
        Standard deviation of SIC for open water pixels (ERA5 c_ice == 0)
        Calculated once for entire dataset
    sigma_ice : float
        Standard deviation of SIC for ice pixels (ERA5 c_ice > 0.95)
        Calculated once for entire dataset
    
    Returns:
    --------
    float : Uncertainty value
    """
    # Handle NaN values
    if pd.isna(sic_2ch) or pd.isna(sigma_ow) or pd.isna(sigma_ice):
        return np.nan
    
    # Calculate uncertainty (Equation 11)
    term1 = (1 - sic_2ch) ** 2 * sigma_ow ** 2
    term2 = sic_2ch ** 2 * sigma_ice ** 2
    
    return np.sqrt(term1 + term2)

def calculate_delta_hybrid(delta_1ch, delta_2ch, weight):
    """
    Calculate hybrid algorithm uncertainty (Equation 12)
    """
    if any(pd.isna([delta_1ch, delta_2ch, weight])):
        return np.nan
    
    term1 = (1 - weight) ** 2 * delta_1ch ** 2
    term2 = weight ** 2 * delta_2ch ** 2
    
    return np.sqrt(term1 + term2)


def select_final_uncertainty(sic, delta_1ch, delta_2ch, delta_hybrid, 
                             low_threshold=0.40, high_threshold=0.75):
    """
    Select final uncertainty based on SIC value
    • SIC ≤ 40%: use delta_1ch
    • 40% < SIC < 75%: use delta_hybrid
    • SIC ≥ 75%: use delta_2ch
    """
    if pd.isna(sic):
        return np.nan
    
    if sic <= low_threshold:
        return delta_1ch
    elif sic >= high_threshold:
        return delta_2ch
    else:
        return delta_hybrid



def calculate_resampling_uncertainty(sic_grid, window_size=3, k_factor=0.29):
    """
    Calculate resampling uncertainty (Equation 13)
    
    δ_resampling = 0.29 * (max(SIC) - min(SIC)) in 3x3 window
    
    Parameters:
    -----------
    sic_grid : ndarray
        2D array of sea ice concentration values
    window_size : int
        Size of the window (default: 3 for 3x3 window)
    k_factor : float
        Calibration factor (default: 0.29, range: 0.288-0.292)
    
    Returns:
    --------
    ndarray
        Resampling uncertainty grid
    """
    def range_function(window):
        """Calculate max - min for valid values in window"""
        valid_vals = window[~np.isnan(window)]
        if len(valid_vals) == 0:
            return np.nan
        return np.max(valid_vals) - np.min(valid_vals)
    
    # Calculate range in 3x3 window
    sic_range = generic_filter(
        sic_grid, 
        range_function, 
        size=window_size,
        mode='constant',
        cval=np.nan
    )
    
    # Apply k factor
    return k_factor * sic_range


def calculate_total_uncertainty(uncertainty_algorithm, uncertainty_resampling):
    """
    Calculate total SIC uncertainty (Equation 9)
    
    s_total = √(δ²_algorithm + δ²_resampling)
    
    Parameters:
    -----------
    uncertainty_algorithm : float or array
        Algorithm uncertainty component
    uncertainty_resampling : float or array
        Resampling uncertainty component
    
    Returns:
    --------
    float or array
        Total uncertainty
    """
    return np.sqrt(uncertainty_algorithm**2 + uncertainty_resampling**2)


def calculate_sigma_ow_ice(df, c_ice_2ch_col='c_ice_2ch_corrected', c_ice_ref_col='c_ice'):
    """
    Calculate sigma_OW and sigma_ICE for two-channel uncertainty.
    
    σ_OW is computed as std of SIC_2ch for ocean surfaces (SIC_ERA5 = 0)
    σ_ICE is computed as std of SIC_2ch for ice surfaces (SIC_ERA5 > 0.95)
    
    Parameters:
    -----------
    df : DataFrame
        Data containing 2-channel SIC and reference SIC
    c_ice_2ch_col : str
        Column name for 2-channel SIC
    c_ice_ref_col : str
        Column name for reference SIC (e.g., ERA5)
    
    Returns:
    --------
    tuple
        (sigma_OW, sigma_ICE)
    """
    # Open water: reference SIC = 0
    open_water_mask = df[c_ice_ref_col] == 0
    sigma_OW = df.loc[open_water_mask, c_ice_2ch_col].std()
    
    # High ice: reference SIC > 0.95
    high_ice_mask = df[c_ice_ref_col] > 0.95
    sigma_ICE = df.loc[high_ice_mask, c_ice_2ch_col].std()
    
    # Handle NaN cases with default values
    if pd.isna(sigma_OW):
        sigma_OW = 0.05
    if pd.isna(sigma_ICE):
        sigma_ICE = 0.05
    
    return sigma_OW, sigma_ICE


# Configuration constants
EASE2_CONFIGS = {
    'arctic': {
        'grid_name': 'ease_nh',
        'description': 'Arctic EASE2 grid',
        'proj_id': 'ease2_nh-250',
        'lat_0': 90,
    },
    'antarctic': {
        'grid_name': 'ease_sh',
        'description': 'Antarctic EASE2 grid',
        'proj_id': 'ease2_sh-250',
        'lat_0': -90,
    }
}

EASE2_WIDTH = 432
EASE2_HEIGHT = 432
EASE2_AREA_EXTENT = (-5400000, -5400000, 5400000, 5400000)
RADIUS_OF_INFLUENCE = 87500

# Calculate sigma from FWHM (in km, converted to meters)
FWHM_KM = 192.0  # Full Width at Half Maximum in km https://ntrs.nasa.gov/api/citations/19750009839/downloads/19750009839.pdf
SIGMA_M = fwhm2sigma(FWHM_KM) * 1000  # Convert km to meters

def _get_area_definition(area):
    """Get EASE2 grid definition for specified polar region."""
    if area not in EASE2_CONFIGS:
        valid = ', '.join(EASE2_CONFIGS.keys())
        raise ValueError(f"area must be one of {valid}, got '{area}'")
    
    config = EASE2_CONFIGS[area]
    projection = {
        'proj': 'laea',
        'lat_0': config['lat_0'],
        'lon_0': 0,
        'ellps': 'WGS84',
        'datum': 'WGS84'
    }
    
    return geometry.AreaDefinition(
        config['grid_name'],
        config['description'],
        config['proj_id'],
        projection,
        width=EASE2_WIDTH,
        height=EASE2_HEIGHT,
        area_extent=EASE2_AREA_EXTENT
    )

def _normalize_coordinates(lons, lats):
    """Normalize longitude and latitude to standard ranges."""
    lons_norm = ((lons + 180) % 360) - 180
    lats_norm = ((lats + 90) % 180) - 90
    return lons_norm, lats_norm

def _prepare_features(df, features):
    """Extract and validate features from dataframe."""
    if isinstance(features, str):
        features = [features]
    
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise ValueError(f"Features not found in dataframe: {missing}")
    
    data = np.column_stack([df[col].values for col in features])
    return data, features

def _resample_features(swath_def, data, area_def, sigma=None):
    """Resample each feature using Gaussian weighting."""
    if sigma is None:
        sigma = SIGMA_M  # Use default sigma calculated from FWHM
    
    n_features = data.shape[1]
    resampled = np.full(
        (area_def.shape[0], area_def.shape[1], n_features),
        np.nan
    )
    
    for i in range(n_features):
        resampled[:, :, i] = kd_tree.resample_gauss(
            swath_def,
            data[:, i],
            area_def,
            radius_of_influence=RADIUS_OF_INFLUENCE,
            sigmas=sigma,  # Fixed: proper parameter name and value
            fill_value=np.nan
        )
    
    return resampled

import numpy as np
import xarray as xr


def _apply_masks(resampled, mask_path, features, month=None, climatology_var='climatology', 
                 landmask_var='lmask'):
    """
    Apply landmask and climatology mask to resampled data from a single file.
    
    Parameters
    ----------
    resampled : np.ndarray
        Resampled data with shape (432, 432, n_features)
    mask_path : str
        Path to NetCDF/HDF5 file containing both land/ocean mask and climatology data
    features : list of str
        Feature names corresponding to resampled data channels
    month : int, optional
        Month (1-12) for climatology mask. If None, climatology mask is not applied.
    climatology_var : str, optional
        Name or partial name of climatology variable (default: 'climatology').
        Will search for variable containing this string (case-insensitive).
    landmask_var : str, optional
        Name or partial name of landmask variable (default: 'lmask').
        Will search for variable containing this string (case-insensitive).
        Also searches for 'smask' as alternative.
    
    Returns
    -------
    np.ndarray
        Masked resampled data. Land areas set to NaN. 
        Where climatology==0:
        - SIC (c_ice*) features set to 0.0
        - Tb (tb_*) features set to NaN
        - Other features (uncertainty*) set to 0.0
        Where climatology==-1, all values set to NaN.
    
    Raises
    ------
    ValueError
        If mask file is invalid, variables not found, or month is invalid
    """
    
    try:
        ds = xr.open_dataset(mask_path)
        
        # Find landmask variable - check for both lmask and smask
        lmask = None
        search_terms = [landmask_var.lower(), 'smask', 'lmask']
        
        for var in ds.data_vars:
            for search_term in search_terms:
                if search_term in var.lower():
                    lmask = ds[var].values
                    print(f"  Found landmask variable: {var}")
                    break
            if lmask is not None:
                break
        
        if lmask is None:
            var_list = ', '.join(ds.data_vars)
            raise ValueError(
                f"No landmask variable found in {mask_path}. "
                f"Searched for: {search_terms}. "
                f"Available variables: {var_list}"
            )
        
        # Apply landmask (set land areas to NaN)
        # Assuming lmask > 0 means land, lmask == 0 means ocean
        is_land = (lmask > 0)
        masked_data = resampled.copy()
        
        if masked_data.ndim == 3:
            for i in range(masked_data.shape[2]):
                masked_data[:, :, i] = np.where(is_land, np.nan, masked_data[:, :, i])
        else:
            masked_data = np.where(is_land, np.nan, masked_data)
        
        # Apply climatology mask if month is provided
        if month is not None:
            if not (1 <= month <= 12):
                raise ValueError(f"month must be between 1 and 12, got {month}")
            
            # Find climatology variable
            climatology = None
            for var in ds.data_vars:
                if climatology_var.lower() in var.lower():
                    climatology = ds[var][int(month) - 1].to_numpy()
                    print(f"  Found climatology variable: {var}, using month {month}")
                    break
            
            if climatology is None:
                var_list = ', '.join(ds.data_vars)
                raise ValueError(
                    f"No climatology variable found in {mask_path}. "
                    f"Available variables: {var_list}"
                )
            
            # Handle NaN values in climatology
            climatology[np.isnan(climatology)] = -1
            climatology = climatology.astype(int)
            
            # Ensure climatology matches resampled shape
            if climatology.shape != (432, 432):
                raise ValueError(
                    f"Climatology shape {climatology.shape} does not match "
                    f"expected EASE2 grid shape (432, 432)"
                )
            
            # Apply climatology mask logic
            # Where climatology == 0: set SIC to 0.0, Tb to NaN, uncertainty to 0.0
            # Where climatology == -1: set all to NaN
            clim_zero_mask = climatology == 0
            clim_invalid_mask = climatology == -1
            
            # Apply to all features
            if masked_data.ndim == 3:
                for i in range(masked_data.shape[2]):
                    feature_name = features[i] if i < len(features) else ""
                    
                    # Check feature type
                    is_sic = 'c_ice' in feature_name.lower()
                    is_tb = 'tb_' in feature_name.lower()
                    is_uncertainty = 'uncertainty' in feature_name.lower()
                    
                    if clim_zero_mask.any():
                        if is_sic or is_uncertainty:
                            # For SIC and uncertainty: set to 0.0
                            masked_data[clim_zero_mask, i] = 0.0
                        elif is_tb:
                            # For Tb measurements: set to NaN
                            masked_data[clim_zero_mask, i] = np.nan
                        else:
                            # Default: set to 0.0 for other variables
                            masked_data[clim_zero_mask, i] = 0.0
                    
                    # Always set invalid climatology to NaN
                    if clim_invalid_mask.any():
                        masked_data[clim_invalid_mask, i] = np.nan
            else:
                # Handle 2D case
                masked_data[clim_zero_mask] = 0.0
                masked_data[clim_invalid_mask] = np.nan
        
        ds.close()
        return masked_data
        
    except FileNotFoundError:
        raise ValueError(f"Mask file not found: {mask_path}")
    except Exception as e:
        raise ValueError(f"Error applying masks: {str(e)}")


def resample_to_ease2(df, area, features, mask_path=None, month=None, sigma=None):
    """
    Resample data to EASE2 polar grid using Gaussian weighting.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe with columns: 'lon', 'lat', and feature columns
    area : str
        Either 'arctic' or 'antarctic'
    features : str or list of str
        Feature column name(s) to resample
    mask_path : str, optional
        Path to NetCDF/HDF5 file with land/ocean mask and climatology data.
        The file should contain variables with 'lmask' or 'smask' (for land mask) and 
        'climatology' (for climatology) in their names.
    month : int, optional
        Month (1-12) for climatology mask. If None, climatology mask is not applied.
    sigma : float, optional
        Gaussian sigma in meters for resampling. If None, uses SIGMA_M (from FWHM)
    
    Returns
    -------
    np.ndarray
        Resampled data with shape (432, 432, n_features). Land areas are NaN. 
        Where climatology==0:
        - SIC (c_ice*) features set to 0.0
        - Tb (tb_*) features set to NaN
        - Uncertainty features set to 0.0
        Where climatology==-1, all values are NaN.
    
    Raises
    ------
    ValueError
        If area is invalid, features not found, mask file invalid, or month is invalid
    """
    
    # Validate inputs
    if month is not None and not (1 <= month <= 12):
        raise ValueError(f"month must be between 1 and 12, got {month}")
    
    # Convert features to list if string
    if isinstance(features, str):
        features = [features]
    
    # Define target EASE2 grid
    area_def = _get_area_definition(area)
    
    # Prepare feature data
    data, features = _prepare_features(df, features)
    
    # Normalize coordinates
    lons, lats = _normalize_coordinates(df['lon'].values, df['lat'].values)
    
    # Create swath and resample
    swath_def = geometry.SwathDefinition(lons=lons, lats=lats)
    resampled = _resample_features(swath_def, data, area_def, sigma=sigma)
    
    # Apply landmask and climatology mask if provided
    if mask_path is not None:
        resampled = _apply_masks(resampled, mask_path, features, month=month)
    
    return resampled