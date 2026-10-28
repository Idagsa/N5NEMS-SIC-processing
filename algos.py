import numpy as np
from dictionaries import MYI_tiepoints, FYI_tiepoints, OW_tiepoints


def onechannel_22(tb_22,FYI_tiepoints_22,MYI_tiepoints_22,OW_tiepoints_22):
    """Compute onechannel_22 ice concentration for each row in a DataFrame."""
    fy_22 = FYI_tiepoints_22
    my_22 = MYI_tiepoints_22
    ow_22 = OW_tiepoints_22

    i_22 = (fy_22 + my_22) / 2
    c_ice_1c22 = (tb_22 - ow_22) / (i_22 - ow_22)
    c_ice_1c22 = np.clip(c_ice_1c22, 0, 1)
    return c_ice_1c22


def onechannel_31(tb_31,FYI_tiepoints_31,MYI_tiepoints_31,OW_tiepoints_31):
    """Compute onechannel_31 ice concentration for each row in a DataFrame."""
    fy_31 = FYI_tiepoints_31
    my_31 = MYI_tiepoints_31
    ow_31 = OW_tiepoints_31

    i_31 = (fy_31 + my_31) / 2
    c_ice_1c31 = (tb_31 - ow_31) / (i_31 - ow_31)
    c_ice_1c31 = np.clip(c_ice_1c31, 0, 1)
    return c_ice_1c31

def bootstrapF(tb_31,tb_22,FYI_tiepoints_22,MYI_tiepoints_22,OW_tiepoints_22,FYI_tiepoints_31,MYI_tiepoints_31,OW_tiepoints_31):
    """Compute bootstrapF (Comiso) ice concentration for each row in a DataFrame."""
    fy_22 = FYI_tiepoints_22
    my_22 = MYI_tiepoints_22
    ow_22 = OW_tiepoints_22

    fy_31 = FYI_tiepoints_31
    my_31 = MYI_tiepoints_31
    ow_31 = OW_tiepoints_31

    af = (fy_31 - my_31) / (fy_22 - my_22)
    bf = my_31 - af * my_22

    qf = (tb_31 - ow_31) / (tb_22 - ow_22)
    wf = ow_31 - qf * ow_22
    ti18vf = (bf - wf) / (qf - af)
    c_ice_2c = (tb_22 - ow_22) / (ti18vf - ow_22)
    c_ice_2c = np.clip(c_ice_2c, 0, 1)
    return c_ice_2c

import numpy as np
import pandas as pd

def hybrid(tb_22, tb_31, FYI_tiepoints_22, MYI_tiepoints_22, OW_tiepoints_22,
           FYI_tiepoints_31, MYI_tiepoints_31, OW_tiepoints_31):
    """
    Compute hybrid sea ice concentration combining single-channel (22 GHz) 
    and two-channel (bootstrap) algorithms.
    
    The hybrid algorithm uses:
    - Single-channel 22 GHz for SIC ≤ 40% (better in open water/low ice)
    - Linear blend between 40% < SIC < 75%
    - Two-channel bootstrap for SIC ≥ 75% (better for high ice/MYI)
    
    Parameters:
    -----------
    tb_22, tb_31 : scalar or array-like
        Brightness temperatures at 22 GHz and 31 GHz
    FYI_tiepoints_22, MYI_tiepoints_22, OW_tiepoints_22 : float or array-like
        Tie points for 22 GHz channel
    FYI_tiepoints_31, MYI_tiepoints_31, OW_tiepoints_31 : float or array-like
        Tie points for 31 GHz channel
        
    Returns:
    --------
    sic_hybrid : scalar or array
        Hybrid sea ice concentration (truncated to [0, 1])
    """
    
    # Compute single-channel 22 GHz SIC
    sic_1ch = onechannel_22(tb_22, FYI_tiepoints_22, MYI_tiepoints_22, OW_tiepoints_22)
    
    # Truncate to [0, 1] range
    sic_1ch = np.clip(sic_1ch, 0, 1)
    
    # Compute two-channel bootstrap SIC
    sic_2ch = bootstrapF(tb_31, tb_22, FYI_tiepoints_22, MYI_tiepoints_22, OW_tiepoints_22,
                         FYI_tiepoints_31, MYI_tiepoints_31, OW_tiepoints_31)
    
    # Truncate to [0, 1] range
    sic_2ch = np.clip(sic_2ch, 0, 1)
    
    # Check if inputs are scalar
    is_scalar = np.isscalar(tb_22) and np.isscalar(tb_31)
    
    if is_scalar:
        # Handle scalar case directly
        low_threshold = 0.4
        high_threshold = 0.75
        
        if sic_1ch <= low_threshold:
            return sic_1ch
        elif sic_1ch < high_threshold:
            weight = (sic_1ch - low_threshold) / (high_threshold - low_threshold)
            return (1 - weight) * sic_1ch + weight * sic_2ch
        else:
            return sic_2ch
    else:
        # Handle array case
        # Initialize hybrid SIC with single-channel values
        sic_hybrid = sic_1ch.copy()
        
        # Define thresholds
        low_threshold = 0.4   # 40%
        high_threshold = 0.75  # 75%
        
        # Apply hybrid algorithm logic
        # Case 1: SIC_1ch <= 40% -> use single-channel 22 GHz
        # (already initialized with sic_1ch)
        
        # Case 2: 40% < SIC_1ch < 75% -> linear blend
        transition_mask = (sic_1ch > low_threshold) & (sic_1ch < high_threshold)
        if np.any(transition_mask):
            # Calculate weight factor (Equation 4)
            weight = (sic_1ch[transition_mask] - low_threshold) / (high_threshold - low_threshold)
            
            # Calculate hybrid SIC (Equation 5)
            sic_hybrid[transition_mask] = ((1 - weight) * sic_1ch[transition_mask] + 
                                           weight * sic_2ch[transition_mask])
        
        # Case 3: SIC_1ch >= 75% -> use two-channel bootstrap
        high_mask = sic_1ch >= high_threshold
        sic_hybrid[high_mask] = sic_2ch[high_mask]
        
        # Final truncation to ensure [0, 1] range
        sic_hybrid = np.clip(sic_hybrid, 0, 1)
        
        return sic_hybrid