"""Audit physical, timeline, state, and cost accounting in a saved Phase 4.5 window."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from griddrift.dispatch import load_dispatch_config
from griddrift.simulation.integrated import load_integrated_settings

ROOT = Path(__file__).resolve().parents[1]
MODELS = {"adaptive_mlp", "adaptive_xgboost"}
TOL = 2e-5


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def check_close(left, right, message: str, *, atol: float = TOL) -> None:
    require(np.allclose(np.asarray(left, dtype=float), np.asarray(right, dtype=float), rtol=0, atol=atol), message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="saved Stage 3 pilot or Stage 4 window directory under phase45")
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    allowed = (ROOT / "artifacts/experiments/phase45").resolve()
    if not run_dir.resolve().is_relative_to(allowed):
        raise ValueError(f"pilot run must be within {allowed}")
    output_path = run_dir / "validation.json"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite validation report: {output_path}")
    plans = pd.read_csv(run_dir / "hourly_plans.csv", parse_dates=["issue_timestamp", "target_timestamp"])
    executed = pd.read_csv(run_dir / "hourly_execution.csv", parse_dates=["issue_timestamp", "target_timestamp"])
    assets, dispatch_settings = load_dispatch_config(ROOT / "configs/assets.yaml")
    integration = load_integrated_settings(str(ROOT / "configs/integration.yaml"))
    b, h, c = assets.battery, assets.hydro, assets.conventional

    require(set(plans.model) == MODELS and set(executed.model) == MODELS, "pilot must contain both expected model policies")
    require(len(executed) > 0 and len(plans) == len(executed) * 24, "unexpected plan/execution row counts")
    require(not executed.duplicated(["issue_timestamp", "target_timestamp", "model"]).any(), "duplicate executed interval")
    require((executed.termination_condition == "optimal").all(), "a recourse solve was not optimal")
    require((executed.solver_status == "ok").all(), "a recourse solver status was not ok")
    require((plans.horizon_hours.between(1, 24)).all(), "invalid plan horizon")
    for (issue, model), frame in plans.groupby(["issue_timestamp", "model"], sort=True):
        require(frame.horizon_hours.sort_values().tolist() == list(range(1, 25)), f"incomplete 24-hour plan: {issue} {model}")
        ordered = frame.sort_values("horizon_hours")
        init = executed.loc[(executed.issue_timestamp == issue) & (executed.model == model)].iloc[0]
        # Planned power balance and resource-use accounting.
        supply = ordered[["solar_to_demand_mw", "wind_to_demand_mw", "hydro_mw", "conventional_mw", "battery_discharge_mw", "emergency_mw", "unserved_mw"]].sum(axis=1)
        check_close(supply, ordered.demand_mw + ordered.battery_charge_mw, f"planned power balance failed: {issue} {model}")
        check_close(ordered.solar_to_demand_mw + ordered.solar_to_battery_mw + ordered.solar_curtailment_mw, ordered.solar_available_mw, f"planned solar allocation failed: {issue} {model}")
        check_close(ordered.wind_to_demand_mw + ordered.wind_to_battery_mw + ordered.wind_curtailment_mw, ordered.wind_available_mw, f"planned wind allocation failed: {issue} {model}")
        require((ordered.hydro_mw <= h.capacity_mw + TOL).all(), f"planned hydro power limit failed: {issue} {model}")
        require((ordered.hydro_remaining_mwh >= -TOL).all(), f"planned hydro budget became negative: {issue} {model}")
        require((ordered.soc_mwh.between(b.min_soc_mwh - TOL, b.max_soc_mwh + TOL)).all(), f"planned SOC bounds failed: {issue} {model}")
        require(close_scalar(ordered.soc_mwh.iloc[-1], init.initial_battery_soc_mwh, TOL), f"plan terminal SOC mismatch: {issue} {model}")
        prev_soc = float(init.initial_battery_soc_mwh)
        prev_hydro = float(init.initial_hydro_remaining_mwh)
        prev_conv = float(init.initial_conventional_output_mw)
        for _, row in ordered.iterrows():
            expected_soc = prev_soc + b.charge_efficiency * row.battery_charge_mw * assets.interval_hours - row.battery_discharge_mw * assets.interval_hours / b.discharge_efficiency
            require(abs(row.soc_mwh - expected_soc) <= TOL, f"planned SOC transition failed: {issue} {model}")
            expected_hydro = prev_hydro - row.hydro_mw * assets.interval_hours
            require(abs(row.hydro_remaining_mwh - expected_hydro) <= TOL, f"planned hydro transition failed: {issue} {model}")
            require(row.battery_charge_mw <= b.charge_limit_mw + TOL and row.battery_discharge_mw <= b.discharge_limit_mw + TOL, f"planned battery power limit failed: {issue} {model}")
            require(abs(row.conventional_mw - prev_conv) <= assets.conventional_ramp_up_mw_per_interval + TOL and abs(row.conventional_mw - prev_conv) <= assets.conventional_ramp_down_mw_per_interval + TOL, f"planned conventional ramp failed: {issue} {model}")
            prev_soc, prev_hydro, prev_conv = row.soc_mwh, row.hydro_remaining_mwh, row.conventional_mw

    executed = executed.sort_values(["model", "issue_timestamp"]).reset_index(drop=True)
    for model, frame in executed.groupby("model", sort=True):
        frame = frame.sort_values("issue_timestamp")
        require(frame.issue_timestamp.is_monotonic_increasing, f"issue order is not chronological for {model}")
        require((frame.target_timestamp == frame.issue_timestamp + pd.Timedelta(hours=1)).all(), f"wrong execution target for {model}")
        check_close(frame.actual_demand_mw, frame.recourse_demand_mw / integration["demand_scale"], f"actual demand mismatch for {model}")
        supply = frame[["recourse_solar_to_demand_mw", "recourse_wind_to_demand_mw", "recourse_hydro_mw", "recourse_conventional_mw", "recourse_battery_discharge_mw", "recourse_emergency_mw", "recourse_unserved_mw"]].sum(axis=1)
        check_close(supply, frame.recourse_demand_mw + frame.recourse_battery_charge_mw, f"recourse power balance failed for {model}")
        check_close(frame.recourse_solar_to_demand_mw + frame.recourse_solar_to_battery_mw + frame.recourse_solar_curtailment_mw, frame.recourse_solar_available_mw, f"recourse solar allocation failed for {model}")
        check_close(frame.recourse_wind_to_demand_mw + frame.recourse_wind_to_battery_mw + frame.recourse_wind_curtailment_mw, frame.recourse_wind_available_mw, f"recourse wind allocation failed for {model}")
        expected_soc = frame.initial_battery_soc_mwh + b.charge_efficiency * frame.recourse_battery_charge_mw * assets.interval_hours - frame.recourse_battery_discharge_mw * assets.interval_hours / b.discharge_efficiency
        check_close(frame.ending_battery_soc_mwh, expected_soc, f"recourse SOC transition failed for {model}")
        expected_hydro = frame.initial_hydro_remaining_mwh - frame.recourse_hydro_mw * assets.interval_hours
        check_close(frame.ending_hydro_remaining_mwh, expected_hydro, f"recourse hydro transition failed for {model}")
        require((frame.ending_battery_soc_mwh.between(b.min_soc_mwh - TOL, b.max_soc_mwh + TOL)).all(), f"recourse SOC bounds failed for {model}")
        require((frame.ending_hydro_remaining_mwh >= -TOL).all(), f"recourse hydro budget became negative for {model}")
        require((frame.recourse_hydro_mw <= h.capacity_mw + TOL).all(), f"recourse hydro power limit failed for {model}")
        require((frame.recourse_battery_charge_mw <= b.charge_limit_mw + TOL).all() and (frame.recourse_battery_discharge_mw <= b.discharge_limit_mw + TOL).all(), f"recourse battery power limit failed for {model}")
        require(((frame.recourse_conventional_mw - frame.initial_conventional_output_mw).abs() <= assets.conventional_ramp_up_mw_per_interval + TOL).all(), f"recourse conventional ramp-up failed for {model}")
        require(((frame.recourse_conventional_mw - frame.initial_conventional_output_mw).abs() <= assets.conventional_ramp_down_mw_per_interval + TOL).all(), f"recourse conventional ramp-down failed for {model}")
        check_close(frame.ending_battery_soc_mwh.iloc[:-1], frame.initial_battery_soc_mwh.iloc[1:], f"battery state continuity failed for {model}")
        check_close(frame.ending_hydro_remaining_mwh.iloc[:-1], frame.initial_hydro_remaining_mwh.iloc[1:], f"hydro state continuity failed for {model}")
        check_close(frame.ending_conventional_output_mw.iloc[:-1], frame.initial_conventional_output_mw.iloc[1:], f"generator state continuity failed for {model}")
        cost_columns = [column for column in frame.columns if column.startswith("realized_cost_") and column != "realized_cost_eur"]
        check_close(frame[cost_columns].sum(axis=1), frame.realized_cost_eur, f"recourse cost breakdown failed for {model}")

    # The same actual demand and seeded resource outcomes must be revealed to both paths.
    for field in ("actual_demand_mw", "actual_solar_available_mw", "actual_wind_available_mw"):
        require((executed.groupby("issue_timestamp")[field].nunique() == 1).all(), f"paired actual input mismatch: {field}")

    summary = {
        "status": "passed",
        "issue_count": int(executed.issue_timestamp.nunique()),
        "execution_rows": len(executed),
        "plan_rows": len(plans),
        "solver_statuses": executed.solver_status.value_counts().to_dict(),
        "termination_conditions": executed.termination_condition.value_counts().to_dict(),
        "models": {},
        "configuration": {
            "demand_scale": integration["demand_scale"],
            "resource_scenario": integration["resource_scenario"],
            "resource_seed": integration["resource_seed"],
            "battery_charging_policy": integration["battery_charging_policy"],
            "plan_terminal_soc_policy": integration["terminal_soc_policy"],
            "conventional_min_output_mw": dispatch_settings["conventional_min_output_mw"],
        },
        "checks": [
            "complete 24-hour plan timestamps and target alignment",
            "planned and recourse power balance",
            "solar/wind availability accounting",
            "battery efficiency, SOC and power bounds",
            "hydro energy and turbine limits",
            "conventional ramp constraints",
            "independent state continuity by model",
            "paired actual demand and resource outcomes",
            "recourse cost breakdown reconciles without adding rolling plan objectives",
            "preserve-initial-SOC terminal state for each plan",
        ],
    }
    for model, frame in executed.groupby("model", sort=True):
        summary["models"][model] = {
            "model_versions": sorted(frame.model_version.unique().tolist()),
            "planned_first_hour_cost_sum_eur": float(frame.planned_first_hour_cost_eur.sum()),
            "realized_recourse_cost_sum_eur": float(frame.realized_cost_eur.sum()),
            "mean_rolling_24h_plan_objective_eur_not_additive": float(frame.plan_24h_objective_eur.mean()),
            "final_battery_soc_mwh": float(frame.ending_battery_soc_mwh.iloc[-1]),
            "final_hydro_remaining_mwh": float(frame.ending_hydro_remaining_mwh.iloc[-1]),
            "total_recorded_solver_time_s": float((frame.plan_solve_time_s + frame.recourse_solve_time_s).sum()),
            "emergency_energy_mwh": float(frame.recourse_emergency_mw.sum() * assets.interval_hours),
            "unserved_energy_mwh": float(frame.recourse_unserved_mw.sum() * assets.interval_hours),
        }
    output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


def close_scalar(left: float, right: float, abs_tol: float) -> bool:
    return abs(float(left) - float(right)) <= abs_tol


if __name__ == "__main__":
    main()
