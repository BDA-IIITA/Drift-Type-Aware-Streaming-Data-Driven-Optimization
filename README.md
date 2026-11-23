# GridDRIFT

## Overview

GridDRIFT is a Python application for studying how changes in electricity demand affect forecasting and energy planning. It replays historical hourly demand for the Germany–Luxembourg (DE-LU) bidding zone. At each simulated hour, a forecasting model predicts demand for the next 24 hours, and a Pyomo optimization model plans how to meet that demand using solar, wind, hydro, battery storage, conventional generation, and emergency supply.

As actual demand arrives, the application measures the error of the forecast made 24 hours earlier. ADWIN monitors those errors for changes; eligible drift events can lead to model retraining. A Streamlit dashboard displays the forecasts, drift events, dispatch plan, and modeled costs. GridDRIFT runs on historical data and does not control a live electricity grid.

The main code is in `src/griddrift/`, runnable entry points are in `scripts/`, and settings are in `configs/`.

## How to run

Run these commands from the repository root with Python 3.11.

### 1. Install dependencies

```powershell
conda env create -f environment.yml
conda activate griddrift
```

If you already have a Python 3.11 environment, use:

```powershell
python -m pip install -e ".[data,forecasting,drift,optimization,dashboard,dev]"
```

### 2. Prepare demand data

The replay needs a continuous hourly UTC CSV with `timestamp,demand_mw` columns. Set `ENTSOE_API_TOKEN` in your environment to download DE-LU demand:

```powershell
python scripts/download_entsoe_load.py --config configs/base.yaml --start 2021-01-01 --end 2024-03-31
```

The downloader prints the path of its cached CSV. For a replay starting in 2025, also obtain the later demand hours and combine nonoverlapping files with `scripts/merge_entsoe_load.py`. The dashboard's default demand path is `data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv`; you can choose another CSV in its sidebar.

### 3. Train or provide a forecast model

Train the MLP on the earlier demand data, replacing the example filename with the path printed by the downloader:

```powershell
python scripts/train_mlp.py --input data/raw/entsoe_cache/YOUR_TRAINING_LOAD.csv --output artifacts/models/de-lu-mlp
```

The dashboard also uses a saved XGBoost checkpoint and adaptation configuration files. These generated artifacts are excluded from Git. Their expected locations are defined in `src/griddrift/dashboard/model_selection.py`; provide them before starting the dashboard's default XGBoost session.

### 4. Prepare renewable and hydro inputs

The included root CSV files contain 2025 renewable data and PECD hydro reference data. Build the processed inputs used by the replay:

```powershell
python scripts/build_phase55_final_integration.py
```

This creates files under `artifacts/experiments/phase55/final_integration/`.

### 5. Start the dashboard

With the demand CSV, model checkpoints, adaptation configuration files, and processed resource inputs available:

```powershell
python -m streamlit run scripts/streamlit_app.py
```

Choose the input paths in the sidebar, click **Load / apply setup**, then use **Step** or **Step-N** to advance the replay.

For an offline MLP replay, use:

```powershell
python scripts/run_integrated_simulation.py --input data/raw/entsoe_cache/YOUR_FULL_LOAD.csv --model artifacts/models/de-lu-mlp --start 2025-01-08T00:00:00Z --hours 24 --output artifacts/integrated/example
```

To run a standalone dispatch example without demand data or trained models:

```powershell
python scripts/run_dispatch_demo.py --output artifacts/dispatch/example
```
