"""Run a tiny real-checkpoint smoke replay for Phase 5.0 end-to-end wiring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from griddrift.dashboard.model_selection import (
    ADAPTIVE_MLP,
    ADAPTIVE_XGBOOST,
    load_selected_adaptation_config,
    load_selected_forecaster,
)
from griddrift.dashboard.views import validate_forecast_plan_alignment
from griddrift.dispatch import load_dispatch_config
from griddrift.simulation import IntegratedConfig, IntegratedSimulation, load_integrated_settings

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv"
FAMILY_NAMES = {ADAPTIVE_XGBOOST: "Adaptive XGBoost", ADAPTIVE_MLP: "Adaptive MLP"}


def _load_demand(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    if "demand_mw" not in frame:
        raise ValueError(f"{path} must contain a demand_mw column")
    return pd.Series(frame.demand_mw.to_numpy(dtype=float), index=index, name="demand_mw")


def _run_family(demand: pd.Series, family: str, start: pd.Timestamp, hours: int, assets, defaults) -> tuple[dict, IntegratedSimulation]:
    model = load_selected_forecaster(
        family,
        mlp_checkpoint=ROOT / "artifacts/models/de-lu-mlp",
        xgboost_checkpoint=ROOT / "artifacts/experiments/phase37/static/static-multioutput-20261023/static_xgboost_model",
    )
    selected_adaptation = load_selected_adaptation_config(family, ROOT)
    options = {
        "adaptation": selected_adaptation,
    } if family == ADAPTIVE_MLP else {
        "xgboost_adaptation": selected_adaptation,
    }
    simulation = IntegratedSimulation(
        demand,
        model,
        assets,
        config=IntegratedConfig(
            demand_scale=defaults["demand_scale"],
            resource_seed=defaults["resource_seed"],
            resource_scenario=defaults["resource_scenario"],
            battery_charging_policy=defaults["battery_charging_policy"],
            allow_hydro_battery_charging=defaults["allow_hydro_battery_charging"],
            terminal_soc_policy=defaults["terminal_soc_policy"],
            policy="adaptive",
            **options,
        ),
        start_timestamp=start,
        model_family=family,
    )
    if simulation.timestamp != start:
        raise AssertionError(f"replay started at {simulation.timestamp}, expected {start}")
    validate_forecast_plan_alignment(simulation)
    if len(simulation.forecast) != 24 or len(simulation.plan.schedule) != 24:
        raise AssertionError("initial forecast and dispatch plan must each contain 24 hours")
    if simulation.plan.termination_condition != "optimal":
        raise AssertionError(f"dispatch solver did not return optimal status: {simulation.plan.termination_condition}")
    expected_dispatch = simulation.forecast.to_numpy() * simulation.config.demand_scale
    if not (abs(simulation.plan.schedule.demand_mw.to_numpy() - expected_dispatch) < 1e-8).all():
        raise AssertionError("optimizer demand differs from the selected model's forecast")

    step_results = simulation.step_n(hours)
    if len(step_results) != hours:
        raise AssertionError(f"expected {hours} executed hours, got {len(step_results)}")
    validate_forecast_plan_alignment(simulation)
    errors = pd.DataFrame(simulation.errors)
    metrics = simulation.metrics()
    adaptation_attempts = sum(
        event.outcome in {"adapted", "candidate_rejected"}
        or event.outcome.startswith("candidate_rejected:")
        for event in simulation.adaptation_events
    )
    result = {
        "family": family,
        "label": FAMILY_NAMES[family],
        "initial_model_version": model.model_version,
        "final_adaptive_model_version": simulation.active_model.model_version,
        "replay_start_timestamp_utc": start.isoformat(),
        "hours_advanced": len(step_results),
        "current_timestamp_utc": simulation.timestamp.isoformat(),
        "active_forecast_issue_timestamp_utc": simulation.timestamp.isoformat(),
        "active_targets_first_utc": simulation.forecast.index[0].isoformat(),
        "active_targets_last_utc": simulation.forecast.index[-1].isoformat(),
        "active_target_count": len(simulation.forecast),
        "plan_target_count": len(simulation.plan.schedule),
        "solver_termination": simulation.plan.termination_condition,
        "fixed_24h_scored_rows": len(errors),
        "static_metrics_mw": {
            "mae": metrics["static_mae_mw"], "rmse": metrics["static_rmse_mw"],
        },
        "adaptive_metrics_mw": {
            "mae": metrics["adaptive_mae_mw"], "rmse": metrics["adaptive_rmse_mw"],
        },
        "drift_alarms": len(simulation.drift_events),
        "candidate_attempts": adaptation_attempts,
        "accepted_adaptations": sum(event.outcome == "adapted" for event in simulation.adaptation_events),
        "planned_24h_cost_eur": float(simulation.plan.total_cost_eur),
        "executed_cost_eur": float(simulation.accumulated_realized_cost_eur),
        "battery_soc_mwh": float(simulation.state.battery_soc_mwh),
        "hydro_remaining_mwh": float(simulation.state.hydro_remaining_mwh),
        "hydro_to_battery_enabled": simulation.config.allow_hydro_battery_charging,
        "hydro_to_battery_planned_mwh": float(
            simulation.plan.schedule.hydro_to_battery_mw.sum() * assets.interval_hours
        ),
    }
    return result, simulation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--hours", type=int, default=3)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/experiments/phase50/stage6/short_replay_2025-01-01_3h.json")
    args = parser.parse_args()
    if args.hours < 1 or args.hours > 24:
        parser.error("--hours must be between 1 and 24 for this short smoke replay")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Stage 6 result: {args.output}")

    demand = _load_demand(args.data)
    start = pd.Timestamp(args.start)
    start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
    defaults = load_integrated_settings(str(ROOT / "configs/integration.yaml"))
    assets, _ = load_dispatch_config(ROOT / "configs/assets.yaml")
    results, simulations = [], []
    for family in (ADAPTIVE_XGBOOST, ADAPTIVE_MLP):
        result, simulation = _run_family(demand, family, start, args.hours, assets, defaults)
        results.append(result)
        simulations.append(simulation)

    if simulations[0].detector is simulations[1].detector:
        raise AssertionError("model families unexpectedly share the same ADWIN detector")
    if simulations[0].adaptive_registry is simulations[1].adaptive_registry:
        raise AssertionError("model families unexpectedly share a forecast registry")
    output = {
        "stage": "phase50-stage6-short-replay",
        "dataset": str(args.data.relative_to(ROOT) if args.data.is_relative_to(ROOT) else args.data),
        "start_timestamp_utc": start.isoformat(),
        "hours_per_model": args.hours,
        "models": results,
        "state_isolation_verified": True,
        "training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"saved": str(args.output), "models": results}, indent=2, default=str))


if __name__ == "__main__":
    main()
