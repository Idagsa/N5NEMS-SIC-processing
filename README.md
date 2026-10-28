# Nimbus-5 NEMS sea ice concentration, 1972–1973

Code for the two datasets described in

> Andersen, I. G.-S., Tonboe, R. T., Juul, J., Kolbe, W. M. and Tellefsen, E. H.:
> *Closing Gaps in the 1970s Satellite Sea Ice Concentration Record Using Nimbus-5 NEMS and Graph Neural Networks*.

`NEMS_SIC_GNN_pipeline.ipynb` runs the whole chain top to bottom and produces every figure, table and number in the paper. Its last cell prints each quoted number with its source.

| Part | What it does | Output |
|---|---|---|
| 1 | NEMS ingestion and screening, dynamic tie points, ERA5/RTM correction, hybrid SIC, uncertainty budget, daily EASE2 gridding with land-spillover correction | daily NEMS SIC, 150 days per hemisphere |
| 2 | Period compositing, unobserved-area masks, graph construction, model | – |
| 3 | Masked-reconstruction training, hold-out validation against baseline interpolators, robustness check (cell 3d) | checkpoints, validation tables |
| 4 | Inference, pole-hole fill, provenance flags, per-cell uncertainty, NetCDF export | 14-day gap-filled NEMS/ESMR SIC, 13 periods per hemisphere |
| 5 | Extent series, monthly table, NEMS–NIC statistics, diagnostics, regional zooms, tie points, screening | figures, tables, `manuscript_numbers.txt` |

## Requirements

Python 3.11 and the packages in `requirements.txt`. Install PyTorch Geometric against the PyTorch version you end up with.

```bash
pip install -r requirements.txt
```

## Input data

Set `NEMS_DATA_ROOT` to a folder with this layout:

```
NEMS_colocated/                   NEMS Tb co-located with ERA5 (one NetCDF per orbit file)
ESMR/<NH and SH ESA CCI ESMR v1.1 folders>
reference_files/LandOceanLakeMask_{nh,sh}_ease2-250_v2.nc
**/nic_weekly_*.bin               NIC weekly charts (NSIDC G02172), anywhere below the root
NIC/EASE2/*_NH_NIC-BIN.nc         optional: the same charts already on EASE2, used if present
G00917/arctic.dat, antarc.dat     NOAA/NMC/CAC monthly extent (NSIDC G00917)
```

Outputs go to `NEMS_OUTPUT_ROOT` (default `./output`).

## Project modules

The notebook imports `tiepoints_func`, `algos`, `RTM`, `uncertainty`, `correction` and `netcdf_writer` (and `algos` imports `dictionaries`). They are in `code/`; point the notebook at them with

```bash
export NEMS_MODULES_DIR=$PWD/code
```

or set `PROJECT_MODULES_DIR` in the first code cell.

## Reproducibility

Every random draw — the XGBoost split, the block hold-out, the mini-batch order and the network initialisation — is seeded from `SEED` in the setup cell. Training from that seed reproduces the published checkpoints exactly. The skill reported in the paper is computed on the three interleaved test periods only; the post-processing variant and all design choices were selected on the two validation periods.

Cell 3d reproduces the two robustness figures quoted in the paper (the spread over random initialisations, and the best simple method on the validation periods). It is off by default because it retrains the network three times per hemisphere; run it with

```bash
export NEMS_ROBUSTNESS=1
```

## Data

The two datasets are archived at DTU Data:

- daily NEMS SIC — https://doi.org/10.11583/DTU.31653223.v1
- 14-day gap-filled NEMS/ESMR SIC — https://doi.org/10.11583/DTU.33193608.v1
