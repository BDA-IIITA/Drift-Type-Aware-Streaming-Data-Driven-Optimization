"""Validate saved Stage 5 replay outputs against processed resource inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", default="artifacts/experiments/phase55/stage5_short_replay")
    parser.add_argument(
        "--renewables", default="data/processed/entsoe_de_lu_2025_renewables_hourly.csv"
    )
    parser.add_argument(
        "--hydro-parameters", default="data/processed/pecd_de00_reservoir_parameters.json"
    )
    parser.add_argument(
        "--hydro-inflow", default="data/processed/pecd_de00_assumed_hourly_inflow_2025.csv"
    )
    parser.add_argument("--scale", type=float, default=0.001)
    parser.add_argument("--interval-hours", type=float, default=1.0)
    args = parser.parse_args()

    replay_dir = Path(args.replay)
    summary = json.loads((replay_dir / "summary.json").read_text(encoding="utf-8"))
    operations = pd.read_csv(replay_dir / "realized_operations.csv")
    plan = pd.read_csv(replay_dir / "current_plan_24h.csv")
    forecast = pd.read_csv(replay_dir / "current_forecast_24h.csv")
    source = pd.read_csv(args.renewables)
    source["target_timestamp_utc"] = pd.to_datetime(source["target_timestamp_utc"], utc=True)
    source = source.set_index("target_timestamp_utc")
    inflow = pd.read_csv(args.hydro_inflow)
    inflow["target_timestamp_utc"] = pd.to_datetime(inflow["target_timestamp_utc"], utc=True)
    inflow = inflow.set_index("target_timestamp_utc")
    params = json.loads(Path(args.hydro_parameters).read_text(encoding="utf-8"))

    operation_times = pd.DatetimeIndex(pd.to_datetime(operations["timestamp"], utc=True))
    plan_times = pd.DatetimeIndex(pd.to_datetime(plan["timestamp"], utc=True))
    forecast_times = pd.DatetimeIndex(pd.to_datetime(forecast["timestamp"], utc=True))
    assert len(operations) == summary["realized_hours"]
    assert len(plan) == len(forecast) == 24
    assert operation_times.is_monotonic_increasing and not operation_times.has_duplicates
    assert (operation_times.to_series().diff().iloc[1:] == pd.Timedelta(hours=1)).all()
    assert plan_times.equals(forecast_times)
    issue_time = pd.Timestamp(summary["forecast_issue_timestamp"])
    assert plan_times[0] == issue_time + pd.Timedelta(hours=1)
    assert plan_times[-1] == issue_time + pd.Timedelta(hours=24)

    actual_source = source.loc[operation_times]
    np.testing.assert_allclose(
        operations["actual_demand_unscaled_mw"].to_numpy(),
        actual_source["demand_actual_mw"].to_numpy(),
        rtol=0,
        atol=1e-7,
    )
    np.testing.assert_allclose(
        operations["solar_actual_source_scaled_mw"].to_numpy(),
        actual_source["solar_actual_mw"].to_numpy() * args.scale,
        rtol=0,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        operations["wind_actual_source_scaled_mw"].to_numpy(),
        actual_source["wind_actual_mw"].to_numpy() * args.scale,
        rtol=0,
        atol=1e-8,
    )
    expected_inflow_mwh = inflow.loc[operation_times]["assumed_inflow_mwh_per_hour"].to_numpy() * args.scale
    np.testing.assert_allclose(
        operations["hydro_inflow_source_mwh"].to_numpy(), expected_inflow_mwh, rtol=0, atol=1e-8
    )

    reservoir_capacity = float(params["reservoir_capacity_mwh"]) * args.scale
    reservoir_energy = float(params["initial_reservoir_energy_mwh"]) * args.scale
    max_balance_error = 0.0
    for row in operations.itertuples(index=False):
        expected_energy = (
            reservoir_energy
            + float(row.hydro_inflow_source_mwh)
            - (float(row.hydro_mw) + float(row.hydro_spill_mw)) * args.interval_hours
        )
        max_balance_error = max(max_balance_error, abs(expected_energy - float(row.ending_hydro_remaining_mwh)))
        assert -1e-7 <= float(row.ending_hydro_remaining_mwh) <= reservoir_capacity + 1e-7
        reservoir_energy = float(row.ending_hydro_remaining_mwh)
    assert max_balance_error <= 1e-6

    forecast_source = source.reindex(plan_times)
    for resource, source_column, issue_column, plan_column in (
        (
            "solar",
            "solar_day_ahead_forecast_mw",
            "solar_forecast_issue_timestamp_utc",
            "solar_available_mw",
        ),
        (
            "wind",
            "wind_day_ahead_forecast_mw",
            "wind_forecast_issue_timestamp_utc",
            "wind_available_mw",
        ),
    ):
        issue_values = pd.to_datetime(forecast_source[issue_column], utc=True, errors="coerce")
        assumed_issue = plan_times.normalize() - pd.Timedelta(days=1)
        effective_issue = issue_values.where(issue_values.notna(), assumed_issue)
        available_asof = effective_issue <= issue_time
        expected_values = forecast_source[source_column].where(available_asof, 0.0).fillna(0.0)
        np.testing.assert_allclose(
            plan[plan_column].to_numpy(),
            expected_values.to_numpy() * args.scale,
            rtol=0,
            atol=1e-8,
        )
    np.testing.assert_allclose(
        plan["cost_total_eur"].sum(), summary["plan_cost_eur"], rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        operations["cost_total_eur"].to_numpy(),
        operations["realized_cost_eur"].to_numpy(),
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        operations["realized_cost_eur"].sum(), summary["accumulated_realized_cost_eur"], rtol=0, atol=1e-5
    )
    np.testing.assert_allclose(
        operations["ending_hydro_remaining_mwh"].iloc[-1],
        summary["state"]["hydro_remaining_mwh"],
        rtol=0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        operations["ending_battery_soc_mwh"].iloc[-1],
        summary["state"]["battery_soc_mwh"],
        rtol=0,
        atol=1e-6,
    )

    report = {
        "validation": "passed",
        "replay": str(replay_dir),
        "requested_start_timestamp": "2025-01-02T00:00:00Z",
        "effective_first_executed_timestamp": operation_times[0].isoformat(),
        "effective_last_executed_timestamp": operation_times[-1].isoformat(),
        "executed_hour_count": len(operations),
        "forecast_issue_timestamp": issue_time.isoformat(),
        "forecast_and_plan_target_count": len(plan),
        "forecast_and_plan_targets_match": True,
        "source_actuals_match": True,
        "resource_forecasts_match_asof_day_ahead_values": True,
        "resource_forecast_issue_time_assumed_targets": int(
            pd.to_datetime(forecast_source["solar_forecast_issue_timestamp_utc"], utc=True, errors="coerce").isna().sum()
            + pd.to_datetime(forecast_source["wind_forecast_issue_timestamp_utc"], utc=True, errors="coerce").isna().sum()
        ),
        "maximum_hydro_balance_absolute_error_mwh": max_balance_error,
        "hydro_capacity_mwh_representative_grid": reservoir_capacity,
        "initial_hydro_mwh_representative_grid": float(params["initial_reservoir_energy_mwh"]) * args.scale,
        "final_hydro_mwh_representative_grid": float(operations["ending_hydro_remaining_mwh"].iloc[-1]),
        "final_battery_soc_mwh": float(operations["ending_battery_soc_mwh"].iloc[-1]),
        "realized_cost_sum_eur": float(operations["realized_cost_eur"].sum()),
        "final_24h_planned_cost_eur": float(plan["cost_total_eur"].sum()),
        "drift_detections": int(summary["drift_detections"]),
        "adaptation_attempts": int(summary["adaptation_attempts"]),
        "accepted_adaptations": int(summary["accepted_adaptations"]),
    }
    report_path = replay_dir / "stage5_verification.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))
    print(f"Saved verification report to {report_path}")


if __name__ == "__main__":
    main()
