import numpy as np
from scipy import ndimage

def apply_land_spillover_correction_reference(sic, lmask, filter_size=9, validate=True):
    """
    Apply land spill-over correction matching the reference implementation.
    
    Parameters
    ----------
    sic : np.ndarray
        Sea ice concentration array (432, 432)
    lmask : np.ndarray
        Land mask array where values > 0 indicate land
    filter_size : int
        Size of uniform filter (default: 9 for 9x9 window)
    validate : bool
        If True, collect validation data
    
    Returns
    -------
    sic_corrected : np.ndarray
        Corrected SIC values
    spillover_mask : np.ndarray
        Boolean mask of corrected pixels
    land_spillover : np.ndarray
        The spillover values
    validation_data : dict or None
    """
    sic_corrected = sic.copy()
    
    # Create binary land mask
    land_binary = np.zeros_like(lmask, dtype=float)
    land_binary[lmask > 0.5] = 1.0
    
    # Create theoretical land concentration (land = 0.9)
    landnn = land_binary * 0.9
    
    # Apply uniform filter (Gaussian approximation with STD=50km)
    landnnb = ndimage.uniform_filter(landnn, size=filter_size)
    
    # Restore land pixels to 0.9 (don't filter land itself)
    landnnb[land_binary > 0.5] = 0.9
    
    # Set very small values to -1.0 (as in reference)
    landnnb[landnnb < 0.01] = -1.0
    
    # Calculate land spillover
    land_spillover = landnnb - landnn
    
    # Create spillover mask where spillover > SIC
    spillover_mask = land_spillover > sic
    
    # Apply correction: set to 0.0 where spillover exceeds SIC
    sic_corrected[spillover_mask] = 0.0
    
    # Validation data collection
    validation_data = None
    if validate:
        corrected_indices = np.argwhere(spillover_mask)
        if len(corrected_indices) > 0:
            validation_data = {'corrected_pixels': corrected_indices.tolist(),
                               'original_sic': sic[spillover_mask].tolist(),
                               'spillover_values': land_spillover[spillover_mask].tolist(),
                               'filtered_land': landnnb[spillover_mask].tolist(),
                               'n_corrected': np.sum(spillover_mask)}
    
    print(f"  Total pixels corrected: {np.sum(spillover_mask)}")
    print(f"  Mean spillover value: {np.mean(land_spillover[spillover_mask]):.4f}" if np.sum(spillover_mask) > 0 else "  No pixels corrected")
    
    return sic_corrected, spillover_mask, land_spillover, validation_data
