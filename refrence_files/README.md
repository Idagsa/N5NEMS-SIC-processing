# EASE2 Land-Ocean-Lake Mask Dataset

## Overview

This dataset contains land-ocean-lake mask data for polar regions on the EASE2 (Equal Area Scalable Earth 2) grid system. The masks distinguish between land, ocean, and lake pixels for both Arctic and Antarctic regions at 250 km spatial resolution.

## Files

### `LandOceanLakeMask_nh_ease2-250_v2.nc`
NetCDF file containing the land-ocean-lake mask for the **Northern Hemisphere (Arctic)** region at 250 km resolution on the EASE2 grid.

**Format:** NetCDF (Network Common Data Form)  
**Grid:** EASE2 Northern Hemisphere (ease2_nh-250)  
**Resolution:** 250 km  
**Dimensions:** 432 × 432 pixels

### `LandOceanLakeMask_sh_ease2-250_v2.nc`
NetCDF file containing the land-ocean-lake mask for the **Southern Hemisphere (Antarctic)** region at 250 km resolution on the EASE2 grid.

**Format:** NetCDF (Network Common Data Form)  
**Grid:** EASE2 Southern Hemisphere (ease2_sh-250)  
**Resolution:** 250 km  
**Dimensions:** 432 × 432 pixels

### `Ease_areas.txt`
Configuration file defining the EASE2 grid projection parameters and spatial extents for both hemispheres.

## EASE2 Grid System

The EASE2 grid is an equal-area projection system designed for global climate and remote sensing data. Key characteristics:

- **Projection:** Lambert Azimuthal Equal Area (LAEA)
- **Datum:** WGS84
- **Ellipsoid:** WGS84
- **Version:** EASE2 (improved over EASE 1)

### Northern Hemisphere Grid (ease2_nh-250)
- **Projection Center:** 90°N (North Pole)
- **Dimensions:** 432 × 432 pixels
- **Spatial Extent:** -5,400,000 to 5,400,000 meters (X and Y)
- **Coverage:** Arctic and Northern Hemisphere regions

### Southern Hemisphere Grid (ease2_sh-250)
- **Projection Center:** 90°S (South Pole)
- **Dimensions:** 432 × 432 pixels
- **Spatial Extent:** -5,400,000 to 5,400,000 meters (X and Y)
- **Coverage:** Antarctic and Southern Hemisphere regions

## Mask Classification

The land-ocean-lake masks typically classify pixels into the following categories:
- **Land:** Continental and terrestrial surfaces
- **Ocean:** Open water bodies (seas, oceans)
- **Lake:** Inland freshwater bodies
- **Undefined/Missing:** Areas without valid data (often at grid edges)

The exact encoding scheme depends on the specific implementation and should be verified in the NetCDF file metadata or documentation.

## Data Access

To work with these files, you will need:
- A NetCDF reader (e.g., Python libraries: `netCDF4`, `xarray`, `rasterio`)
- Knowledge of the EASE2 projection (may need to use `pyproj` or GDAL for coordinate transformations)

### Example (Python with xarray)
```python
import xarray as xr

# Load Northern Hemisphere mask
nh_mask = xr.open_dataset('LandOceanLakeMask_nh_ease2-250_v2.nc')
print(nh_mask)

# Load Southern Hemisphere mask
sh_mask = xr.open_dataset('LandOceanLakeMask_sh_ease2-250_v2.nc')
print(sh_mask)
```

## References

For more information on the EASE2 grid system, refer to:
- Brodzik, M. J., et al. (2012). EASE-Grid 2.0: Incremental but Significant Improvements for Earth-Gridded Data Sets. NSIDC Special Report, National Center for Atmospheric Research, Boulder, CO.
- NSIDC (National Snow and Ice Data Center) documentation on EASE2 grids

## Notes

- These are version 2 masks, indicating refinements over earlier versions
- The masks should be used in conjunction with the grid definition file (`Ease_areas.txt`) for proper coordinate transformation
- Ensure your analysis software properly interprets the EASE2 projection to avoid spatial misalignment
