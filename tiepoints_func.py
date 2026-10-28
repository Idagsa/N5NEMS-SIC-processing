import pandas as pd
import numpy as np

# Calculate percentiles in rolling 14-day backwards window with outlier removal
def rolling_percentile_stats(window_data, min_samples=100, outlier_method='iqr', iqr_multiplier=1.5):
    """
    Calculate top and bottom 10% statistics for a rolling window after removing outliers.
    
    Parameters:
    -----------
    window_data : DataFrame
        Data within the rolling window
    min_samples : int
        Minimum samples required after outlier removal
    outlier_method : str
        Method for outlier detection: 'iqr', 'zscore', or 'percentile'
    iqr_multiplier : float
        Multiplier for IQR method (default 1.5 for standard outliers)
    """
    if len(window_data) < min_samples:
        return None
    
    # Remove outliers based on BOTH tb_22 and tb_31
    clean_data = remove_outliers_multivariate(window_data.copy(), 
                                              columns=['tb_22', 'tb_31'], 
                                              method=outlier_method,
                                              iqr_multiplier=iqr_multiplier)
    
    # Check if we still have enough samples after outlier removal
    if len(clean_data) < min_samples:
        return None
    
    # Top 10%: highest brightness temperature (FYI - First Year Ice)
    # Do this separately for 22 GHz and 31 GHz
    top_10_count = max(1, int(len(clean_data) * 0.10))
    
    top_percent_22 = clean_data.nlargest(top_10_count, 'tb_22')
    top_percent_31 = clean_data.nlargest(top_10_count, 'tb_31')
    
    # Bottom 10%: lowest brightness temperature (MYI - Multi-Year Ice)
    # Do this separately for 22 GHz and 31 GHz
    bottom_10_count = max(1, int(len(clean_data) * 0.10))
    
    bottom_percent_22 = clean_data.nsmallest(bottom_10_count, 'tb_22')
    bottom_percent_31 = clean_data.nsmallest(bottom_10_count, 'tb_31')
    
    return {
        # 22 GHz statistics
        'top_10_tb_22_mean': top_percent_22['tb_22'].mean(),
        'top_10_tb_22_std': top_percent_22['tb_22'].std(),
        'top_10_tb_22_count': len(top_percent_22),
        'bottom_10_tb_22_mean': bottom_percent_22['tb_22'].mean(),
        'bottom_10_tb_22_std': bottom_percent_22['tb_22'].std(),
        'bottom_10_tb_22_count': len(bottom_percent_22),
        
        # 31 GHz statistics
        'top_10_tb_31_mean': top_percent_31['tb_31'].mean(),
        'top_10_tb_31_std': top_percent_31['tb_31'].std(),
        'top_10_tb_31_count': len(top_percent_31),
        'bottom_10_tb_31_mean': bottom_percent_31['tb_31'].mean(),
        'bottom_10_tb_31_std': bottom_percent_31['tb_31'].std(),
        'bottom_10_tb_31_count': len(bottom_percent_31),
        
        # General statistics
        'total_points': len(clean_data),
        'outliers_removed': len(window_data) - len(clean_data)
    }


def remove_outliers_multivariate(df, columns=['tb_22', 'tb_31'], method='iqr', 
                                 iqr_multiplier=1.5, zscore_threshold=3, 
                                 percentile_lower=1, percentile_upper=99):
    """
    Remove outliers from dataframe using various methods applied to multiple columns.
    A point is kept only if it's NOT an outlier in ANY of the specified columns.
    
    Parameters:
    -----------
    df : DataFrame
        Input data
    columns : list
        List of columns to use for outlier detection
    method : str
        Outlier detection method:
        - 'iqr': Interquartile Range (recommended)
        - 'zscore': Z-score method
        - 'percentile': Remove values outside percentile range
    iqr_multiplier : float
        Multiplier for IQR (1.5 = standard outliers, 3.0 = extreme outliers)
    zscore_threshold : float
        Threshold for z-score method (typically 3)
    percentile_lower : float
        Lower percentile cutoff (0-100)
    percentile_upper : float
        Upper percentile cutoff (0-100)
    """
    # Start with all points included
    mask = pd.Series([True] * len(df), index=df.index)
    
    # Apply outlier detection to each column
    for column in columns:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in dataframe, skipping")
            continue
            
        if method == 'iqr':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - iqr_multiplier * IQR
            upper_bound = Q3 + iqr_multiplier * IQR
            col_mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
            
        elif method == 'zscore':
            mean = df[column].mean()
            std = df[column].std()
            if std == 0:
                col_mask = pd.Series([True] * len(df), index=df.index)
            else:
                z_scores = np.abs((df[column] - mean) / std)
                col_mask = z_scores <= zscore_threshold
            
        elif method == 'percentile':
            lower_bound = df[column].quantile(percentile_lower / 100)
            upper_bound = df[column].quantile(percentile_upper / 100)
            col_mask = (df[column] >= lower_bound) & (df[column] <= upper_bound)
            
        else:
            raise ValueError(f"Unknown outlier method: {method}")
        
        # Combine masks (keep only if NOT outlier in all columns)
        mask = mask & col_mask
    
    return df[mask]


# Usage in main function:
def analyze_sea_ice_tiepoints(df, col_22='tb_22_measured', col_31='tb_31_measured',
                              min_c_ice=15, col_c_ice='c_ice', min_samples=100, 
                              rolling_mean_window=14, start_date='1972-12-17', 
                              outlier_method='iqr', iqr_multiplier=1.5,
                              zscore_threshold=3, percentile_lower=1, percentile_upper=99,
                              verbose=True):
    """
    Analyze sea ice brightness temperatures using top/bottom 10% tie-point selection.
    Removes outliers from BOTH 22 GHz and 31 GHz, then calculates separate top/bottom 10%
    for each frequency.
    
    Parameters:
    -----------
    col_22 : str
        Column name for 22 GHz brightness temperature
    col_31 : str
        Column name for 31 GHz brightness temperature
    min_c_ice : float
        Minimum sea ice concentration (%) to include data
    col_c_ice : str
        Column name for sea ice concentration (default: 'c_ice')
    min_samples : int
        Minimum samples required in rolling window (after outlier removal)
    rolling_mean_window : int
        Days for rolling mean smoothing of tie-points
    outlier_method : str
        Method for outlier detection: 'iqr', 'zscore', or 'percentile'
    iqr_multiplier : float
        Multiplier for IQR method (1.5 = standard, 3.0 = extreme outliers only)
    zscore_threshold : float
        Threshold for z-score method (typically 3)
    percentile_lower : float
        Lower percentile cutoff for percentile method (0-100)
    percentile_upper : float
        Upper percentile cutoff for percentile method (0-100)
    """
    import pandas as pd
    import numpy as np
    
    df = df.copy()
    
    # Rename selected columns
    df['tb_22'] = df[col_22]
    df['tb_31'] = df[col_31]
    
    # Remove any NaN values
    df = df.dropna(subset=['tb_22', 'tb_31'])
    
    # Apply sea ice concentration filter (only include ice)
    if col_c_ice in df.columns:
        df = df[df[col_c_ice] >= min_c_ice]
    
    # Create datetime column if not present
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(
            dict(year=df['year'], month=df['month'], day=df['day']))
    
    # Sort by datetime
    df = df.sort_values('datetime')
    df_indexed = df.set_index('datetime')
    
    # Explicit rolling approach
    results = []
    
    for date in df_indexed.index.unique():
        # Get data from past 14 days (not including current day)
        window_start = date - pd.Timedelta(days=14)
        window_data = df_indexed.loc[window_start:date].iloc[:-1]  # Exclude current day
        
        stats = rolling_percentile_stats(window_data, 
                                        min_samples=min_samples,
                                        outlier_method=outlier_method,
                                        iqr_multiplier=iqr_multiplier)
        if stats is not None:
            stats['datetime'] = date
            results.append(stats)
    
    stats_df = pd.DataFrame(results)
    
    if verbose and len(stats_df) == 0:
        print(f"WARNING: No tie-points generated!")
        print(f"Filtered records (c_ice >= {min_c_ice}): {len(df)}")
        print(f"This may indicate insufficient data or min_samples={min_samples} is too strict")
    
    # Apply rolling mean to smooth tie-points over time
    if rolling_mean_window > 1 and len(stats_df) > 0:
        # Set datetime as index for rolling operation
        stats_df = stats_df.sort_values('datetime')  # Ensure sorted
        stats_df_indexed = stats_df.set_index('datetime')
        
        # Apply rolling mean with backwards-looking window
        rolling_cols = ['top_10_tb_22_mean', 'top_10_tb_31_mean', 
                       'bottom_10_tb_22_mean', 'bottom_10_tb_31_mean',
                       'top_10_tb_22_std', 'top_10_tb_31_std',
                       'bottom_10_tb_22_std', 'bottom_10_tb_31_std']
        
        for col in rolling_cols:
            stats_df_indexed[col] = stats_df_indexed[col].rolling(
                window=f'{rolling_mean_window}D',
                closed='left',
                min_periods=1
            ).mean()
        
        # Reset index - this restores 'datetime' as a column
        stats_df = stats_df_indexed.reset_index()
    
    if verbose:
        print("="*80)
        print("Sea Ice 14-Day Rolling Window Analysis (With Outlier Removal)")
        print(f"Starting from: {pd.to_datetime(start_date).date()}")
        print(f"Using columns: {col_22}, {col_31}")
        print(f"Ice concentration column: {col_c_ice}")
        print(f"Minimum sea ice concentration: {min_c_ice}%")
        print(f"Outlier removal method: {outlier_method} (applied to BOTH 22 GHz and 31 GHz)")
        if outlier_method == 'iqr':
            print(f"IQR multiplier: {iqr_multiplier}")
        elif outlier_method == 'zscore':
            print(f"Z-score threshold: {zscore_threshold}")
        elif outlier_method == 'percentile':
            print(f"Percentile range: {percentile_lower}% - {percentile_upper}%")
        print(f"Minimum samples per window: {min_samples}")
        print(f"Rolling mean smoothing window: {rolling_mean_window} days")
        print(f"Selection method: Separate Top 10% and Bottom 10% for each frequency")
        print(f"Total tie-point dates: {len(stats_df)}")
        print("="*80)
        if len(stats_df) > 0:
            print(f"\nFirst date: {stats_df['datetime'].min().date()}")
            print(f"Last date: {stats_df['datetime'].max().date()}")
            print(f"Average points per window (after outlier removal): {stats_df['total_points'].mean():.0f}")
            print(f"Average outliers removed per window: {stats_df['outliers_removed'].mean():.1f}")
            outlier_pct = (stats_df['outliers_removed'].sum() / 
                          (stats_df['total_points'].sum() + stats_df['outliers_removed'].sum()) * 100)
            print(f"Total outlier removal rate: {outlier_pct:.1f}%")
            print(f"\nFYI 22GHz range: {stats_df['top_10_tb_22_mean'].min():.2f} to {stats_df['top_10_tb_22_mean'].max():.2f} K")
            print(f"MYI 22GHz range: {stats_df['bottom_10_tb_22_mean'].min():.2f} to {stats_df['bottom_10_tb_22_mean'].max():.2f} K")
            print(f"FYI 31GHz range: {stats_df['top_10_tb_31_mean'].min():.2f} to {stats_df['top_10_tb_31_mean'].max():.2f} K")
            print(f"MYI 31GHz range: {stats_df['bottom_10_tb_31_mean'].min():.2f} to {stats_df['bottom_10_tb_31_mean'].max():.2f} K")
    
    # Create tie-point dictionaries (return as Series for compatibility)
    # Handle empty case gracefully
    if len(stats_df) == 0:
        empty_series = pd.Series([], dtype='float64')
        MYI_tiepoints = {
            'datetime': pd.Series([], dtype='datetime64[ns]'),
            'tp_22': empty_series,
            'tp_31': empty_series,
            'tp_22_std': empty_series,
            'tp_31_std': empty_series
        }
        
        FYI_tiepoints = {
            'datetime': pd.Series([], dtype='datetime64[ns]'),
            'tp_22': empty_series,
            'tp_31': empty_series,
            'tp_22_std': empty_series,
            'tp_31_std': empty_series
        }
    else:
        MYI_tiepoints = {
            'datetime': stats_df['datetime'],
            'tp_22': stats_df['bottom_10_tb_22_mean'],
            'tp_31': stats_df['bottom_10_tb_31_mean'],
            'tp_22_std': stats_df['bottom_10_tb_22_std'],
            'tp_31_std': stats_df['bottom_10_tb_31_std']
        }
        
        FYI_tiepoints = {
            'datetime': stats_df['datetime'],
            'tp_22': stats_df['top_10_tb_22_mean'],
            'tp_31': stats_df['top_10_tb_31_mean'],
            'tp_22_std': stats_df['top_10_tb_22_std'],
            'tp_31_std': stats_df['top_10_tb_31_std']
        }
    
    return {
        'stats_df': stats_df,
        'MYI_tiepoints': MYI_tiepoints,
        'FYI_tiepoints': FYI_tiepoints
    }

def analyze_ow_tiepoints(df, min_sst, max_sst, max_c_ice,
                         outlier_sigma=1,
                         start_date='1972-12-17',
                         col_22='tb_22_measured', 
                         col_31='tb_31_measured',
                         min_samples=200,
                         rolling_mean_window=14):
    """
    Analyze open water brightness temperatures using backwards-looking rolling window.
    Then applies a rolling mean to smooth the tie-points over time.
    
    Parameters:
    -----------
    rolling_mean_window : int
        Number of days for rolling mean smoothing (default: 14)
    """
    import numpy as np
    
    df = df.copy()
    
    # Rename selected columns
    df['tb_22'] = df[col_22]
    df['tb_31'] = df[col_31]
    
    # Remove any NaN values
    df = df.dropna(subset=['tb_22', 'tb_31'])
    
    # Apply SST filter
    if 'Ts' in df.columns:
        df = df[(df['Ts'] >= min_sst) & (df['Ts'] <= max_sst)]
    
    # Apply sea ice concentration filter
    if 'c_ice' in df.columns:
        df = df[df['c_ice'] <= max_c_ice]
    
    # Remove outliers
    if len(df) > 0:
        z_22 = np.abs((df['tb_22'] - df['tb_22'].mean()) / df['tb_22'].std())
        z_31 = np.abs((df['tb_31'] - df['tb_31'].mean()) / df['tb_31'].std())
        df = df[(z_22 < outlier_sigma) & (z_31 < outlier_sigma)]
    
    # Create datetime column if not present
    if 'datetime' not in df.columns:
        df['datetime'] = pd.to_datetime(
            dict(year=df['year'], month=df['month'], day=df['day']))
    
    # Sort by datetime
    df = df.sort_values('datetime')
    df_indexed = df.set_index('datetime')
    
    # Use 14-day backwards-looking rolling window
    rolling_mean = df_indexed[['tb_22', 'tb_31']].rolling(
        window='14D', 
        closed='left'  # Excludes the current day, looks back 14 days
    ).mean()
    
    rolling_std = df_indexed[['tb_22', 'tb_31']].rolling(
        window='14D',
        closed='left'
    ).std()
    
    rolling_count = df_indexed[['tb_22']].rolling(
        window='14D',
        closed='left'
    ).count()
    
    # Combine statistics
    stats_combined = pd.DataFrame({
        'tb_22_mean': rolling_mean['tb_22'],
        'tb_31_mean': rolling_mean['tb_31'],
        'tb_22_std': rolling_std['tb_22'],
        'tb_31_std': rolling_std['tb_31'],
        'count': rolling_count['tb_22']
    }).dropna()
    
    # Filter intervals with insufficient samples
    stats_combined = stats_combined[stats_combined['count'] >= min_samples].reset_index()
    
    # Apply rolling mean to smooth tie-points over time
    if rolling_mean_window > 1 and len(stats_combined) > 0:
        # Set datetime as index for rolling operation
        stats_combined = stats_combined.sort_values('datetime')  # Ensure sorted
        stats_indexed = stats_combined.set_index('datetime')
        
        # Apply rolling mean with backwards-looking window
        rolling_cols = ['tb_22_mean', 'tb_31_mean', 'tb_22_std', 'tb_31_std']
        
        for col in rolling_cols:
            stats_indexed[col] = stats_indexed[col].rolling(
                window=f'{rolling_mean_window}D',
                closed='left',
                min_periods=1
            ).mean()
        
        # Reset index - this restores 'datetime' as a column
        stats_combined = stats_indexed.reset_index()
    
    # Create tie-point dictionary (return as Series for compatibility)
    # Handle empty case
    if len(stats_combined) == 0:
        empty_dt = pd.Series([], dtype='datetime64[ns]')
        empty_float = pd.Series([], dtype='float64')
        
        OW_tiepoints = {
            'datetime': empty_dt,
            'tp_22': empty_float,
            'tp_31': empty_float,
            'tp_22_std': empty_float,
            'tp_31_std': empty_float,
            'count': empty_float
        }
    else:
        OW_tiepoints = {
            'datetime': stats_combined['datetime'],
            'tp_22': stats_combined['tb_22_mean'],
            'tp_31': stats_combined['tb_31_mean'],
            'tp_22_std': stats_combined['tb_22_std'],
            'tp_31_std': stats_combined['tb_31_std'],
            'count': stats_combined['count']
        }
    
    return {
        'stats_df': stats_combined,
        'OW_tiepoints': OW_tiepoints,
        'filtered_data': df
    }


def extract_tiepoints1(results, ow_results, ice_types=['MYI', 'FYI'], channels=[22, 31]):
    """Extract tiepoints from results and convert to Series with datetime index"""
    tiepoints = {}
    
    # Ice tiepoints
    for ice_type in ice_types:
        key = f'{ice_type}_tiepoints'
        for ch in channels:
            col_name = f'tp_{ch}'
            series = pd.Series(
                data=results[key][col_name].values,
                index=results[key]['datetime'].values
            )
            tiepoints[f'{ice_type}_tp{ch}'] = series[~series.index.duplicated(keep='last')]
    
    # Open water tiepoints
    for ch in channels:
        col_name = f'tp_{ch}'
        series = pd.Series(
            data=ow_results['OW_tiepoints'][col_name].values,
            index=ow_results['OW_tiepoints']['datetime'].values
        )
        tiepoints[f'OW_tp{ch}'] = series[~series.index.duplicated(keep='last')]
    
    # Forward and backward fill
    for key in tiepoints:
        tiepoints[key] = tiepoints[key].ffill().bfill()
    
    return tiepoints

def extract_tiepoints2(results, ow_results, ice_types=['MYI', 'FYI'], channels=[22, 31]):
    """Extract tiepoints and std from results"""
    tiepoints = {}
    
    # Ice tiepoints (mean and std)
    for ice_type in ice_types:
        key = f'{ice_type}_tiepoints'
        for ch in channels:
            # Mean
            series = pd.Series(
                data=results[key][f'tp_{ch}'].values,
                index=results[key]['datetime'].values
            )
            tiepoints[f'{ice_type}_tp{ch}_corrected'] = series[~series.index.duplicated(keep='last')]
            
            # Std
            series_std = pd.Series(
                data=results[key][f'tp_{ch}_std'].values,
                index=results[key]['datetime'].values
            )
            tiepoints[f'{ice_type}_tp{ch}_corrected_std'] = series_std[~series_std.index.duplicated(keep='last')]
    
    # Open water tiepoints (mean and std)
    for ch in channels:
        # Mean
        series = pd.Series(
            data=ow_results['OW_tiepoints'][f'tp_{ch}'].values,
            index=ow_results['OW_tiepoints']['datetime'].values
        )
        tiepoints[f'OW_tp{ch}_corrected'] = series[~series.index.duplicated(keep='last')]
        
        # Std
        series_std = pd.Series(
            data=ow_results['OW_tiepoints'][f'tp_{ch}_std'].values,
            index=ow_results['OW_tiepoints']['datetime'].values
        )
        tiepoints[f'OW_tp{ch}_corrected_std'] = series_std[~series_std.index.duplicated(keep='last')]
    
    # Forward and backward fill
    for key in tiepoints:
        tiepoints[key] = tiepoints[key].ffill().bfill()
    
    return tiepoints

# Map to records
def map_tiepoints_to_record(record, tiepoints):
    """Map tiepoints to record using datetime index"""
    dt_norm = record['datetime'].dt.normalize()
    for name, series in tiepoints.items():
        record[name] = series.reindex(dt_norm, method='ffill').values
    
    # Backward fill any remaining NaNs
    tiepoint_cols = list(tiepoints.keys())
    record[tiepoint_cols] = record[tiepoint_cols].bfill()