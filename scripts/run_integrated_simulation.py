"""Run a short offline integrated forecast-and-dispatch replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from train_mlp import _demand

from griddrift.dashboard.views import dashboard_snapshot_figure
from griddrift.data import (
    RenewableResourceProvider,
    load_assumed_hourly_inflow_series,
    load_reservoir_parameters_json,
)
from griddrift.dispatch import load_dispatch_config
from griddrift.forecasting.mlp import DirectMLPForecaster
from griddrift.simulation import IntegratedConfig, IntegratedSimulation, load_integrated_settings

ROOT = Path(__file__).resolve().parents[1]
FINAL_RESOURCE_DIR = ROOT / "artifacts/experiments/phase55/final_integration"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv")
    parser.add_argument("--model", default="artifacts/models/de-lu-mlp")
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--policy", choices=["static", "adaptive"])
    parser.add_argument("--scenario", choices=["normal", "low_renewables", "wind_scarcity"])
    parser.add_argument("--charging", choices=["renewable_only", "unrestricted"])
    parser.add_argument("--output", default="artifacts/integrated/offline-demo")
    parser.add_argument("--config", default="configs/integration.yaml")
    parser.add_argument("--renewables", help="Processed hourly renewable table for historical resource replay")
    parser.add_argument("--hydro-parameters", help="Processed PECD reservoir parameter JSON")
    parser.add_argument("--hydro-inflow", help="Processed PECD modeled hourly inflow CSV")
    parser.add_argument("--start", help="Optional replay start timestamp, interpreted as UTC")
    args = parser.parse_args()
    if args.hours < 0:
        parser.error("--hours must be non-negative")

    defaults = load_integrated_settings(args.config)
    resource_paths = (args.renewables, args.hydro_parameters, args.hydro_inflow)
    if not any(resource_paths) and defaults["renewable_data_mode"] == "entsoe_pecd":
        resource_paths = (
            str(FINAL_RESOURCE_DIR / "entsoe_de_lu_2025_renewables_hourly.csv"),
            str(FINAL_RESOURCE_DIR / "pecd_de00_reservoir_parameters.json"),
            str(FINAL_RESOURCE_DIR / "pecd_de00_synthetic_hourly_hydro_2025.csv"),
        )
    if any(resource_paths) and not all(resource_paths):
        parser.error("--renewables, --hydro-parameters, and --hydro-inflow must be supplied together")
    renewables_path, hydro_parameters_path, hydro_inflow_path = resource_paths
    input_frame = pd.read_csv(args.input, low_memory=False)
    if {"timestamp", "demand_mw"}.issubset(input_frame.columns):
        demand = _demand(args.input)
    elif {"target_timestamp_utc", "demand_actual_mw"}.issubset(input_frame.columns):
        index = pd.DatetimeIndex(pd.to_datetime(input_frame["target_timestamp_utc"], utc=True, errors="coerce"))
        values = pd.to_numeric(input_frame["demand_actual_mw"], errors="coerce")
        valid = index.notna() & values.notna()
        if not valid.all():
            parser.error("processed input contains invalid timestamps or missing actual demand")
        demand = pd.Series(values.to_numpy(dtype=float), index=index, name="demand_mw")
    else:
        parser.error("input must contain timestamp/demand_mw or target_timestamp_utc/demand_actual_mw")
    model = DirectMLPForecaster.load(args.model)
    assets, _ = load_dispatch_config("configs/assets.yaml")
    scale = defaults["demand_scale"] if args.scale is None else args.scale
    policy = defaults["policy"] if args.policy is None else args.policy
    scenario = defaults["resource_scenario"] if args.scenario is None else args.scenario
    charging = defaults["battery_charging_policy"] if args.charging is None else args.charging
    resource_provider = None
    hydro_parameters = None
    hydro_inflow = None
    if all(resource_paths):
        resource_provider = RenewableResourceProvider(
            renewables_path,
            issue_time_policy=defaults["renewable_forecast_issue_policy"],
        )
        hydro_parameters = load_reservoir_parameters_json(hydro_parameters_path)
        hydro_inflow = load_assumed_hourly_inflow_series(hydro_inflow_path)
    simulation = IntegratedSimulation(
        demand, model, assets,
        config=IntegratedConfig(demand_scale=scale, policy=policy,
                                resource_seed=defaults["resource_seed"],
                                resource_scenario=scenario,
                                battery_charging_policy=charging,
                                allow_hydro_battery_charging=defaults["allow_hydro_battery_charging"],
                                terminal_soc_policy=defaults["terminal_soc_policy"]),
        start_timestamp=args.start,
        renewable_provider=resource_provider,
        hydro_parameters=hydro_parameters,
        hydro_inflow_profile=hydro_inflow,
    )
    simulation.step_n(args.hours)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    simulation.forecast.rename("predicted_demand_mw").to_csv(output / "current_forecast_24h.csv", index_label="timestamp")
    simulation.plan.schedule.to_csv(output / "current_plan_24h.csv", index_label="timestamp")
    simulation.static_registry.to_frame().to_csv(output / "static_forecasts.csv", index=False)
    simulation.adaptive_registry.to_frame().to_csv(output / "adaptive_forecasts.csv", index=False)
    pd.DataFrame(simulation.realized).to_csv(output / "realized_operations.csv", index=False)
    pd.DataFrame(simulation.errors).to_csv(output / "forecast_errors.csv", index=False)
    pd.DataFrame(simulation.drift_events).to_csv(output / "drift_events.csv", index=False)
    pd.DataFrame([vars(event) for event in simulation.adaptation_events]).to_csv(output / "adaptation_events.csv", index=False)
    summary = {
        "timestamp": simulation.timestamp.isoformat(),
        "status": simulation.status,
        "forecast_issue_timestamp": simulation.timestamp.isoformat(),
        "forecast_horizon": len(simulation.forecast),
        "plan_horizon": len(simulation.plan.schedule),
        "model_version": simulation.active_model.model_version,
        "forecast_policy": policy,
        "resource_scenario": scenario,
        "allow_hydro_battery_charging": defaults["allow_hydro_battery_charging"],
        "renewable_data_status": (
            "ENTSO-E forecast columns with configured as-of assumption and revealed ENTSO-E actuals"
            if resource_provider is not None
            else "SIMULATED solar/wind forecasts and actual outcomes; seeded and separate"
        ),
        "renewable_forecast_issue_policy": defaults["renewable_forecast_issue_policy"] if resource_provider else None,
        "resource_source": str(Path(renewables_path)) if resource_provider else None,
        "hydro_inflow_source": str(Path(hydro_inflow_path)) if hydro_inflow is not None else None,
        "demand_source": str(Path(args.input)),
        "demand_scale_for_dispatch_only": scale,
        "plan_cost_eur": simulation.planned_cost_eur,
        "accumulated_realized_cost_eur": simulation.accumulated_realized_cost_eur,
        "state": {
            "battery_soc_mwh": simulation.state.battery_soc_mwh,
            "hydro_remaining_mwh": simulation.state.hydro_remaining_mwh,
            "conventional_output_mw": simulation.state.conventional_output_mw,
        },
        "metrics": simulation.metrics(),
        "realized_hours": len(simulation.realized),
        "drift_detections": len(simulation.drift_events),
        "adaptation_attempts": len(simulation.adaptation_events),
        "accepted_adaptations": sum(event.outcome == "adapted" for event in simulation.adaptation_events),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    dashboard_snapshot_figure(simulation).write_html(output / "dashboard_snapshot.html", include_plotlyjs=True)
    if policy == "adaptive":
        simulation.active_model.save(output / "final_adaptive_model", simulation.metrics())
    print(json.dumps(summary, indent=2, default=str))
    print(f"Saved offline replay outputs to {output}")


if __name__ == "__main__":
    main()
