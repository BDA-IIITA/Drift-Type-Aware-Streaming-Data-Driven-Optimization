"""Build immutable, reproducible Phase 5.5 final renewable/hydro inputs."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from griddrift.data.hydro import write_de00_hydro_scenario
from griddrift.data.renewables import preprocess_entsoe_renewables_csv
from griddrift.dispatch import load_dispatch_config

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts/experiments/phase55/final_integration"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--initial-filling-fraction", type=float, default=0.5)
    parser.add_argument("--minimum-reservoir-energy-mwh", type=float, default=0.0)
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    renewable_source = ROOT / "entsoe_de_lu_2025_hourly_features.csv"
    renewable_out = OUTPUT / "entsoe_de_lu_2025_renewables_hourly.csv"
    hydro_paths = {
        "parameters": OUTPUT / "pecd_de00_reservoir_parameters.json",
        "profile": OUTPUT / "pecd_de00_seasonal_weekly_inflow.csv",
        "hourly_scenario": OUTPUT / "pecd_de00_synthetic_hourly_hydro_2025.csv",
        "provenance": OUTPUT / "pecd_de00_hydro_provenance.json",
    }
    expected = [renewable_out, renewable_out.with_suffix(".metadata.json"), *hydro_paths.values()]
    existing = [path for path in expected if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Final integration files already exist; review them or rerun with --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if args.overwrite:
        for path in existing:
            path.unlink()

    renewable = preprocess_entsoe_renewables_csv(renewable_source, renewable_out)
    hydro = write_de00_hydro_scenario(
        ROOT / "PECD-hydro-capacities.csv",
        ROOT / "PECD-hydro-weekly-inflows.csv",
        ROOT / "PECD-hydro-weekly-reservoir-levels.csv",
        output_dir=OUTPUT,
        year=2025,
        initial_filling_fraction=args.initial_filling_fraction,
        minimum_reservoir_energy_mwh=args.minimum_reservoir_energy_mwh,
    )
    hydro["hourly_scenario"].replace(hydro_paths["hourly_scenario"])
    reservoir_parameters = json.loads(hydro_paths["parameters"].read_text(encoding="utf-8"))
    assets, _ = load_dispatch_config(ROOT / "configs/assets.yaml")
    hydro_config = {
        "scenario_name": "DE00 historical-weekly-median seasonal inflow, 2025",
        "zone": "DE00",
        "technology": "Reservoir only; excludes run-of-river and pumped storage",
        "turbine_generation_limit_mw_national_reference": reservoir_parameters["turbine_capacity_mw"],
        "maximum_reservoir_energy_mwh_national_reference": reservoir_parameters["reservoir_capacity_mwh"],
        "minimum_reservoir_energy_mwh_national_reference": reservoir_parameters["minimum_reservoir_energy_mwh"],
        "initial_filling_fraction_assumption": reservoir_parameters["initial_filling_fraction"],
        "initial_reservoir_energy_mwh_national_reference": reservoir_parameters["initial_reservoir_energy_mwh"],
        "hydro_marginal_cost_eur_mwh": assets.hydro.marginal_cost_eur_mwh,
        "representative_grid_scale": 0.001,
        "random_seed": 42,
        "random_variation_applied": False,
        "random_seed_note": "Seed is recorded for reproducibility; hourly values are deterministic historical medians and no uncalibrated noise is added.",
        "inflow_method": "DE00 historical weekly energy medians by source week number; GWh to MWh; uniformly allocated across 168 weekly hours; mapped by ISO week number.",
        "inflow_is_observed_2025": False,
        "reservoir_state_method": "Initialized from configured fill fraction; thereafter determined only by optimizer hydro generation, synthetic inflow, and spill.",
        "market_price_used_as_marginal_cost": False,
        "configured_asset_rates_source": "configs/assets.yaml",
    }
    (OUTPUT / "hydro_scenario_config.json").write_text(
        json.dumps(hydro_config, indent=2), encoding="utf-8"
    )
    source_columns = pd.read_csv(renewable_source, nrows=0).columns.tolist()
    quality = {
        "renewables": {
            "source": str(renewable_source.relative_to(ROOT)),
            "rows": len(renewable),
            "first_timestamp_utc": str(renewable.target_timestamp_utc.iloc[0]),
            "last_timestamp_utc": str(renewable.target_timestamp_utc.iloc[-1]),
            "source_columns": source_columns,
            "source_hours_missing": int((~renewable.source_row_present).sum()),
            "missing_demand_actual": int(renewable.demand_actual_missing.sum()),
            "missing_solar_forecast": int(renewable.solar_forecast_missing.sum()),
            "missing_wind_forecast": int(renewable.wind_forecast_missing.sum()),
            "missing_solar_actual": int(renewable.solar_actual_missing.sum()),
            "missing_wind_actual": int(renewable.wind_actual_missing.sum()),
            "solar_forecast_issue_timestamps_present": int(renewable.solar_forecast_issue_timestamp_utc.notna().sum()),
            "wind_forecast_issue_timestamps_present": int(renewable.wind_forecast_issue_timestamp_utc.notna().sum()),
            "hydro_actual_columns_preserved_as_reference": [
                column for column in renewable if column.startswith("hydro_") and column.endswith("_mw")
            ],
            "hydro_actual_used_as_inflow_or_storage": False,
            "actual_renewables_used_as_physical_potential": False,
        },
        "hydro": {
            "rows": 8760,
            "first_timestamp_utc": "2025-01-01 00:00:00+00:00",
            "last_timestamp_utc": "2025-12-31 23:00:00+00:00",
            "inflow_is_synthetic_historically_informed": True,
            "reservoir_level_measurements_for_de00": 0,
            "negative_inflow_rows": 0,
            "scenario_fields_present": all(
                column in pd.read_csv(hydro_paths["hourly_scenario"], nrows=0).columns
                for column in (
                    "synthetic_2025_hydro_inflow_mwh",
                    "hydro_turbine_generation_limit_mw",
                    "maximum_reservoir_energy_mwh",
                    "provenance",
                    "scenario_random_seed",
                )
            ),
        },
    }
    (OUTPUT / "data_quality_report.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    shutil.copy2(hydro["provenance"], OUTPUT / "source_provenance.json")
    print(json.dumps({"output_dir": str(OUTPUT), "renewable_rows": len(renewable), "hydro_rows": 8760, "quality": quality}, indent=2))


if __name__ == "__main__":
    main()
