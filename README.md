# GridDRIFT

GridDRIFT is a Python research project for 24-hour electricity-demand forecasting, concept-drift monitoring, and cost-based energy dispatch. It replays historical German–Luxembourg (DE-LU) demand one hour at a time: issue 24 hourly forecasts, plan generation and storage with Pyomo and HiGHS, then reveal the next actual hour. It is a simulation, not a live grid controller.

The [project report](GridDRIFT_Report.pdf) describes the complete study. This repository contains the forecasting, ADWIN replay, dispatch, and Streamlit components. The source snapshot predates some later behavior discussed in the report; see [Implementation status](#implementation-status).

## System flow

1. A direct multi-output MLP or XGBoost model predicts demand for the next 24 UTC hours. A frozen model supplies a comparison stream.
2. When an actual hour arrives, ADWIN receives the absolute error of the forecast issued exactly 24 hours earlier.
3. An eligible error increase can trigger retraining after the configured cooldown and candidate validation. The artificial drift demo uses a separate policy.
4. A Pyomo model allocates solar, wind, hydro, battery, conventional generation, and emergency supply over 24 hours subject to capacity and state constraints.
5. Streamlit shows demand, forecasts, drift events, energy plans, and modeled costs.

The report evaluates a representative grid: demand and resources are multiplied by `0.001` for dispatch, while forecast errors remain in the original MW. Asset sizes and EUR/MWh rates are illustrative assumptions, not market prices.

## Contents

| Path | Purpose |
| --- | --- |
| `src/griddrift/` | Data, forecasting, drift, dispatch, simulation, and dashboard modules |
| `scripts/` | CLI entry points and experiment helpers |
| `configs/` | Data, asset, dispatch, and replay settings |
| `tests/` | Unit and integration tests |
| `docs/` | Data contract, assumptions, and methodology notes |
| `entsoe_de_lu_2025_hourly_features.csv` | Supplied 2025 demand and renewable source table |
| `PECD-hydro-*.csv` | Supplied hydro reference tables |
| `GridDRIFT_Report.pdf` | Thesis methodology, evaluation, and limitations |

Generated data, model checkpoints, and experiment outputs under `data/` and `artifacts/` are excluded from Git. The supplied root CSVs are inputs to resource preparation; they do not replace the long demand history and checkpoints needed for replay.

## Install

Use Python 3.11 from the project root:

```powershell
conda env create -f environment.yml
conda activate griddrift
```

Or install into an existing Python 3.11 environment:

```powershell
python -m pip install -e ".[data,forecasting,drift,optimization,dashboard,dev]"
```

Run tests with `python -m pytest -q -p no:cacheprovider`. The full suite includes tests for later features described below.

## Prepare data and models

The app expects continuous hourly demand in UTC with columns `timestamp,demand_mw`. Its default path is `data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv`. See [the data contract](docs/data_contract.md).

Set `ENTSOE_API_TOKEN` in your environment, then download the earlier training period. The command prints its cache filename:

```powershell
python scripts/download_entsoe_load.py --config configs/base.yaml --start 2021-01-01 --end 2024-03-31
python scripts/train_mlp.py --input data/raw/entsoe_cache/YOUR_TRAINING_LOAD.csv --output artifacts/models/de-lu-mlp
```

For replay, obtain subsequent demand through 2026-10-20. Use `scripts/merge_entsoe_load.py` for nonoverlapping cached ranges, or supply an existing CSV with the schema above. Keep the API token out of the repository.

The dashboard also expects a saved XGBoost model and selected adaptation configurations at paths defined in [model_selection.py](src/griddrift/dashboard/model_selection.py). These generated artifacts are not in Git. Obtain the original experiment artifacts or run the corresponding scripts under `scripts/`; training the MLP alone does not recreate the report's dashboard defaults.

Prepare the 2025 renewable and hydro inputs from the four root CSV files:

```powershell
python scripts/build_phase55_final_integration.py
```

Outputs go under `artifacts/experiments/phase55/final_integration/`. Planning uses ENTSO-E day-ahead solar and wind forecasts; actual generation is revealed for the executed hour. As the supplied table lacks forecast publication timestamps, the replay assumes availability at 00:00 UTC on the previous day. Hydro inflow is a modeled 2025 profile from historical PECD weekly medians, not observed 2025 inflow.

## Run

With the demand CSV, checkpoints, selection files, and processed resources in place:

```powershell
python -m streamlit run scripts/streamlit_app.py
```

Use **Load / apply setup**, then **Step** or **Step-N**. The four tabs cover Live Forecasting, Drift Monitoring, 24-Hour Energy Planning, and Cost & Analysis. Missing inputs are reported rather than silently replaced.

An offline MLP replay writes forecasts, plans, operations, errors, events, a JSON summary, and an HTML snapshot:

```powershell
python scripts/run_integrated_simulation.py --input data/raw/entsoe_cache/YOUR_FULL_LOAD.csv --model artifacts/models/de-lu-mlp --start 2025-01-08T00:00:00Z --hours 24 --output artifacts/integrated/example
```

The resource mode in `configs/integration.yaml` reads files produced by `build_phase55_final_integration.py`. A standalone dispatch example needs no trained forecast model:

```powershell
python scripts/run_dispatch_demo.py --output artifacts/dispatch/example
```

## Results in the report

These are results **reported in the supplied thesis**, under its 2025 replay and representative-grid assumptions. They have not been independently reproduced from this source snapshot.

| Evaluation | Reported result |
| --- | ---: |
| Adaptive XGBoost fixed 24-hour-ahead MAE on compared 2025–26 targets | 1,641 MW |
| 24-hour plan versus greedy dispatch, 8 January 2025 | EUR 1,092 lower (1.2%) |
| 24-hour plan versus greedy dispatch, 28 daily plans | EUR 109,964 lower (3.7%); cheaper on 28/28 days |
| 24-hour plan with battery versus greedy without battery, 28 days | EUR 124,784 lower (4.1%) |
| Realized versus committed planned cost, 1–8 January 2025 | EUR 4,770 higher (1.2%) |

The forecast comparison uses dates examined during development, so it is a comparative benchmark rather than an untouched test. Costs depend on assumed asset sizes and rates. There is no imbalance price; 2025 hydro inflow is modeled.

## Implementation status

The current source snapshot predates several behaviors described in the report and expected by some tests, including committed first-hour recourse, a parallel frozen-system cost comparison, hourly hydro rates, and parts of the three-shift demo. Thus the repository contains a working subset of the pipeline, but does not reproduce every report figure as published. The dashboard's default session also requires generated checkpoints, selection files, and long demand history.

For equations and modeling assumptions, see [dispatch design](docs/phase4_dispatch.md), [integration design](docs/phase5_integration.md), and the [report](GridDRIFT_Report.pdf).
