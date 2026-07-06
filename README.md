# Polar Analysis Diagnostics Package

This repository contains the diagnostics and mode of variability analysis workflows for evaluating E3SM model simulations in polar regions (with a focus on the Southern Hemisphere). The workflows compare E3SM outputs against CMIP ensembles and observational datasets.

---

## Directory Structure

The repository is organized into a clean, modular structure consisting of the following four primary directories:

```
polar_analysis/
├── README.md               # This project documentation
├── .gitignore              # Files and directories ignored by Git
├── config/                 # Configurations for the diagnostic runs
├── jupyter/                # Jupyter Notebooks driving the diagnostic workflow
├── util/                   # Shared modular Python utility library (package)
└── scripts/                # Execution, validation, and alternative scripts
    ├── shell/              # Shell/Bash runner scripts
    └── ncl/                # Alternative and backup NCL scripts
```

---

## Component Details

### 1. Jupyter Notebooks (`jupyter/`)
These notebooks drive the diagnostic workflow and act as the main entry points:
- `Mean_Climate_Analysis.ipynb` — Portrait and parallel coordinate plots comparing mean climatologies.
- `Variability_Modes_Analysis.ipynb` — Consolidated metrics for modes of variability (ENSO, PDO, IPO, SAM, etc.).
- `ASL_Analysis.ipynb` — Amundsen Sea Low index generation and lead-lag analysis.
- `SAM_Analysis.ipynb` — Southern Annular Mode EOF and station-based index generation.
- `Nino_Analysis.ipynb` — Niño index generation and lead-lag analysis.
- `Regional_Analysis.ipynb` — Weighted atmospheric regional means and MPAS-Ocean/Sea-ice timeseries.
- `EOF_Analysis.ipynb` — Regional Empirical Orthogonal Function decomposition.
- `Trend_Analysis.ipynb` — Grid-point and regional trend calculations.
- `Weather_Regimes_Analysis.ipynb` — K-Means weather regime clustering.

### 2. Shared Utilities (`util/`)
A modular Python package (`util`) containing helper functions and classes imported by the Jupyter Notebooks:
- `common.py` — Common classes (e.g. `Case`), file I/O helpers, and plotting utilities.
- `asl.py` — Peak detection and low-pressure tracking algorithms for ASL.
- `sam.py` — Zonal difference and EOF calculation methods for SAM.
- `nino.py` — Niño SST index calculations and lead-lag regressions.
- `regional.py` — Weighted regional area averages and MPAS-Ocean regional timeseries.
- `eof.py` — Custom SVD-based EOF analysis fallback.
- `trend.py` — Linear regression trend calculations on grids.
- `weather_regimes.py` — K-Means clustering and standard scaling wrappers.
- `mean_climate.py` — Graphic plot parsers and portrait plot helpers.
- `mov_analysis.py` — Metric file loading and consolidated plotting helpers.

### 3. Scripts (`scripts/`)
Contains auxiliary scripts associated with the analysis:
- `check_imports.py` — Import validation script to check the integrity of the `util` package modules.
- `shell/run_notebooks.sh` — Bash script to run all Jupyter Notebooks headlessly using `jupyter nbconvert`.
- `ncl/` — Flattened backup and alternative NCL scripts for reference.

---

## Getting Started

### Environment Setup
These diagnostics are designed to run in an environment with standard climate analysis packages (e.g., `xarray`, `dask`, `cartopy`, `pcmdi_metrics`). 

On Chrysalis/LCRC, load the unified E3SM environment:
```bash
source /lcrc/soft/climate/e3sm-unified/load_latest_e3sm_unified_chrysalis.sh
```

### Verification
To verify that all shared library imports resolve correctly, run the verification script:
```bash
python scripts/check_imports.py
```

### Executing the Diagnostic Workflow
To run all Jupyter Notebooks headlessly in-place:
```bash
./scripts/shell/run_notebooks.sh
```
