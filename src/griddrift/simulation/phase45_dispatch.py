"""Paired, forecast-driven planning and real-time recourse for Phase 4.5.

This experiment utility consumes saved predictions. It does not fit or update
either forecasting model, and it does not add balancing-market prices.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from griddrift.config.schema import AssetConfig
from griddrift.dispatch import DispatchState, optimize_dispatch
from griddrift.simulation.integrated import issue_resource_forecast, simulated_resource_actuals
from griddrift.simulation.phase45_inputs import ForecastVector, Phase45InputSettings


@dataclass(frozen=True)
class Phase45DispatchOutput:
    """Hourly plan rows and one executed/recourse row per issue and model."""

    plan_rows: pd.DataFrame
    execution_rows: pd.DataFrame
    settings: Phase45InputSettings

    def save(self, output_directory: str | Path, *, run_metadata: dict | None = None) -> tuple[Path, ...]:
        """Write one new run directory, refusing to overwrite any previous run."""
        output = Path(output_directory)
        output.mkdir(parents=True, exist_ok=False)
        plan_path = output / "hourly_plans.csv"
        execution_path = output / "hourly_execution.csv"
        metadata_path = output / "run_metadata.json"
        self.plan_rows.to_csv(plan_path, index=False)
        self.execution_rows.to_csv(execution_path, index=False)
        metadata = {
            "phase": "4.5",
            "runner": "paired_adaptive_mlp_xgboost_forecast_driven_dispatch",
            "settings": asdict(self.settings),
            "issue_count": int(self.execution_rows.issue_timestamp.nunique()),
            "execution_row_count": len(self.execution_rows),
            "plan_row_count": len(self.plan_rows),
            "realized_cost_eur_by_model": {
                str(model): float(group.realized_cost_eur.sum())
                for model, group in self.execution_rows.groupby("model")
            },
            "plan_objective_is_not_added_to_realized_cost": True,
            "metadata": run_metadata or {},
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
        return plan_path, execution_path, metadata_path


def _utc_series(actual_demand_mw: pd.Series) -> pd.Series:
    if not isinstance(actual_demand_mw.index, pd.DatetimeIndex):
        raise ValueError("actual demand must have a DatetimeIndex")
    index = actual_demand_mw.index
    if index.tz is None:
        raise ValueError("actual demand index must be timezone-aware")
    index = index.tz_convert("UTC")
    values = pd.to_numeric(actual_demand_mw, errors="raise").astype(float)
    values.index = index
    values = values.sort_index()
    if values.index.has_duplicates or not values.index.is_monotonic_increasing:
        raise ValueError("actual demand timestamps must be unique and chronological")
    if not np.isfinite(values.to_numpy()).all() or (values < 0).any():
        raise ValueError("actual demand must contain finite, non-negative MW values")
    if len(values) < 2 or not (values.index.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all():
        raise ValueError("actual demand must be a continuous hourly series")
    values.index.name = "timestamp"
    return values


def _commitment(schedule: pd.DataFrame) -> dict[str, float]:
    row = schedule.iloc[0]
    columns = (
        "solar_mw", "wind_mw", "hydro_mw", "conventional_mw",
        "battery_charge_mw", "battery_discharge_mw", "emergency_mw", "unserved_mw",
    )
    return {column: float(row[column]) for column in columns}


def run_phase45_dispatch_comparison(
    forecast_vectors: Sequence[ForecastVector],
    actual_demand_mw: pd.Series,
    assets: AssetConfig,
    *,
    settings: Phase45InputSettings | None = None,
    start_issue: pd.Timestamp | None = None,
    end_issue: pd.Timestamp | None = None,
    initial_state: DispatchState | None = None,
) -> Phase45DispatchOutput:
    """Run a bounded paired comparison on saved forecasts and revealed hours.

    At each issue, both 24-hour plans are completed using only each model's
    prediction vector and the common calendar resource forecast. Only then is
    target ``issue+1`` revealed to the execution path. Each model keeps its own
    realized storage/reservoir/generator state; both receive the same actual
    demand and seeded synthetic resource outcomes.
    """
    if not forecast_vectors:
        raise ValueError("at least one complete forecast vector is required")
    settings = settings or Phase45InputSettings()
    actuals = _utc_series(actual_demand_mw)
    vectors = sorted(forecast_vectors, key=lambda vector: vector.issue_timestamp)
    issues = pd.DatetimeIndex([vector.issue_timestamp for vector in vectors]).tz_convert("UTC")
    if issues.has_duplicates:
        raise ValueError("forecast issue timestamps must be unique")
    selected = np.ones(len(vectors), dtype=bool)
    if start_issue is not None:
        start = pd.Timestamp(start_issue)
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
        selected &= issues >= start
    if end_issue is not None:
        end = pd.Timestamp(end_issue)
        end = end.tz_localize("UTC") if end.tz is None else end.tz_convert("UTC")
        selected &= issues <= end
    vectors = [vector for vector, keep in zip(vectors, selected) if keep]
    if not vectors:
        raise ValueError("the requested issue window contains no forecast vectors")
    issues = pd.DatetimeIndex([vector.issue_timestamp for vector in vectors]).tz_convert("UTC")
    if len(issues) > 1 and not ((issues[1:] - issues[:-1]) == pd.Timedelta(hours=1)).all():
        raise ValueError("selected forecast issues must be contiguous hourly timestamps")

    for vector in vectors:
        execution_target = vector.target_timestamps[0]
        if execution_target not in actuals.index:
            raise ValueError(f"actual-demand source is missing executed target timestamp {execution_target}")

    initial = initial_state or DispatchState(
        assets.battery.initial_soc_mwh,
        assets.hydro.initial_energy_budget_mwh,
        0.0,
    )
    states = {
        "adaptive_mlp": DispatchState(
            initial.battery_soc_mwh, initial.hydro_remaining_mwh, initial.conventional_output_mw
        ),
        "adaptive_xgboost": DispatchState(
            initial.battery_soc_mwh, initial.hydro_remaining_mwh, initial.conventional_output_mw
        ),
    }
    # Generate outcomes over the full known timestamp index so a given seed
    # produces the same resource series regardless of which window is selected.
    actual_resource_profile = simulated_resource_actuals(
        actuals.index, assets, settings.resource_seed, settings.resource_scenario
    )

    plan_records: list[dict] = []
    execution_records: list[dict] = []
    for vector in vectors:
        target_index = pd.DatetimeIndex(vector.target_timestamps)
        renewable_forecast = issue_resource_forecast(target_index, assets, settings.resource_scenario)
        model_forecasts = {
            "adaptive_mlp": (vector.mlp_predicted_demand_mw, vector.mlp_model_version),
            "adaptive_xgboost": (vector.xgboost_predicted_demand_mw, vector.xgboost_model_version),
        }
        plans = {}
        for model_name, (forecast_mw, model_version) in model_forecasts.items():
            # This solve receives no actual series or target observations.
            plan = optimize_dispatch(
                np.asarray(forecast_mw, dtype=float) * settings.demand_scale,
                renewable_forecast,
                assets,
                states[model_name],
                timestamps=target_index,
                conventional_min_output_mw=settings.conventional_min_output_mw,
                terminal_soc_policy=settings.terminal_soc_policy,
                terminal_soc_target_mwh=settings.terminal_soc_target_mwh,
                battery_charging_policy=settings.battery_charging_policy,
                solver_name=settings.solver_name,
            )
            plans[model_name] = (plan, model_version)

        # Both model plans for this issue are complete before revealing the
        # common first target observation to either execution path.
        first_target = target_index[0]
        actual_value = float(actuals.loc[first_target])
        actual_resource = actual_resource_profile.loc[first_target]
        for model_name, (plan, model_version) in plans.items():
            recourse = optimize_dispatch(
                [actual_value * settings.demand_scale],
                {"solar": [float(actual_resource.solar_actual_mw)],
                 "wind": [float(actual_resource.wind_actual_mw)]},
                assets,
                states[model_name],
                timestamps=[first_target],
                conventional_min_output_mw=settings.conventional_min_output_mw,
                terminal_soc_policy="none",
                battery_charging_policy=settings.battery_charging_policy,
                solver_name=settings.solver_name,
                commitment_schedule=_commitment(plan.schedule),
            )
            planned_row = plan.schedule.iloc[0]
            realized_row = recourse.schedule.iloc[0]
            start_state = states[model_name]
            execution_record = {
                "issue_timestamp": vector.issue_timestamp,
                "target_timestamp": first_target,
                "model": model_name,
                "model_version": model_version,
                "actual_demand_mw": actual_value,
                "actual_solar_available_mw": float(actual_resource.solar_actual_mw),
                "actual_wind_available_mw": float(actual_resource.wind_actual_mw),
                "planned_target_demand_mw": float(planned_row.demand_mw / settings.demand_scale),
                "planned_first_hour_cost_eur": float(planned_row.cost_total_eur),
                "plan_24h_objective_eur": plan.total_cost_eur,
                "realized_cost_eur": recourse.total_cost_eur,
                **{
                    f"realized_cost_{source}_eur": amount
                    for source, amount in recourse.cost_breakdown_eur.items()
                },
                "commitment_deviation_mw": float(realized_row.commitment_deviation_mw),
                "initial_battery_soc_mwh": start_state.battery_soc_mwh,
                "initial_hydro_remaining_mwh": start_state.hydro_remaining_mwh,
                "initial_conventional_output_mw": start_state.conventional_output_mw,
                "ending_battery_soc_mwh": recourse.first_hour_state.battery_soc_mwh,
                "ending_hydro_remaining_mwh": recourse.first_hour_state.hydro_remaining_mwh,
                "ending_conventional_output_mw": recourse.first_hour_state.conventional_output_mw,
                "plan_solve_time_s": plan.solve_time_s,
                "recourse_solve_time_s": recourse.solve_time_s,
                "solver_status": recourse.solver_status,
                "termination_condition": recourse.termination_condition,
            }
            execution_record.update({
                f"planned_first_hour_{column}": value
                for column, value in planned_row.to_dict().items()
            })
            execution_record.update({
                f"recourse_{column}": value
                for column, value in realized_row.to_dict().items()
            })
            execution_records.append(execution_record)
            states[model_name] = recourse.first_hour_state

            for horizon, (target, row) in enumerate(plan.schedule.iterrows(), start=1):
                plan_records.append({
                    "issue_timestamp": vector.issue_timestamp,
                    "target_timestamp": target,
                    "horizon_hours": horizon,
                    "model": model_name,
                    "model_version": model_version,
                    "plan_24h_objective_eur": plan.total_cost_eur,
                    **row.to_dict(),
                })

    plan_frame = pd.DataFrame(plan_records)
    execution_frame = pd.DataFrame(execution_records)
    return Phase45DispatchOutput(plan_frame, execution_frame, settings)
