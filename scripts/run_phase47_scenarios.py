"""Run the two reproducible Phase 4.7 hydro dispatch cases."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from griddrift.dispatch import DispatchState, load_dispatch_config, optimize_dispatch


def _cases() -> dict[str, dict[str, object]]:
    index = pd.date_range("2025-01-01 00:00", periods=24, freq="h", tz="UTC")
    conservation_demand = np.array([15.0] * 8 + [22.0] * 16)
    conservation_solar = np.array([8.0] * 8 + [0.0] * 16)
    conservation_wind = np.array([7.0] * 8 + [0.0] * 16)

    storage_demand = np.array([10.0] * 6 + [35.0] * 10 + [55.0] * 4 + [35.0] * 4)
    storage_solar = np.array([4.0] * 6 + [0.0] * 18)
    storage_wind = np.array([6.0] * 6 + [3.0] * 10 + [0.0] * 8)
    storage_turbine_limits = np.array([14.0] * 16 + [0.0] * 4 + [14.0] * 4)

    return {
        "water_conservation": {
            "index": index,
            "demand": conservation_demand,
            "solar": conservation_solar,
            "wind": conservation_wind,
            "hydro_budget_mwh": 40.0,
            "turbine_limits_mw": np.full(24, 14.0),
            "description": (
                "Renewables cover the first eight hours; later renewable availability is zero. "
                "The optimizer can preserve the limited hydro budget for later direct supply."
            ),
        },
        "hydro_to_battery_peak_derating": {
            "index": index,
            "demand": storage_demand,
            "solar": storage_solar,
            "wind": storage_wind,
            "hydro_budget_mwh": 50.0,
            "turbine_limits_mw": storage_turbine_limits,
            "description": (
                "Hydro turbine capacity is available earlier but derated to zero from 16:00-19:00 UTC. "
                "Demand is high during that four-hour interval, making stored energy potentially useful."
            ),
        },
    }


def _validate(schedule: pd.DataFrame, assets, initial: DispatchState, *, allow_hydro: bool) -> dict:
    dt = assets.interval_hours
    atol = 2e-5
    assert len(schedule) == 24
    assert np.allclose(schedule.hydro_mw, schedule.hydro_to_demand_mw + schedule.hydro_to_battery_mw, atol=atol)
    assert np.allclose(
        schedule.battery_charge_mw,
        schedule.solar_to_battery_mw + schedule.wind_to_battery_mw
        + (schedule.hydro_to_battery_mw if allow_hydro else 0.0),
        atol=atol,
    )
    assert np.allclose(
        schedule.soc_mwh.to_numpy(),
        initial.battery_soc_mwh + np.cumsum(
            (assets.battery.charge_efficiency * schedule.battery_charge_mw.to_numpy()
             - schedule.battery_discharge_mw.to_numpy() / assets.battery.discharge_efficiency) * dt
        ),
        atol=atol,
    )
    expected_hydro = initial.hydro_remaining_mwh + np.cumsum(
        - (schedule.hydro_mw.to_numpy() + schedule.hydro_spill_mw.to_numpy()) * dt
    )
    assert np.allclose(schedule.hydro_remaining_mwh.to_numpy(), expected_hydro, atol=atol)
    assert np.allclose(schedule.cost_total_eur, schedule.filter(regex=r"^cost_(?!total)").sum(axis=1), atol=atol)
    return {
        "hours": len(schedule),
        "source_split_valid": True,
        "battery_soc_transition_valid": True,
        "hydro_reservoir_transition_valid": True,
        "hourly_cost_reconciliation_valid": True,
        "terminal_soc_mwh": float(schedule.soc_mwh.iloc[-1]),
        "terminal_hydro_mwh": float(schedule.hydro_remaining_mwh.iloc[-1]),
        "hydro_to_battery_total_mwh": float(schedule.hydro_to_battery_mw.sum() * dt),
        "hydro_direct_total_mwh": float(schedule.hydro_to_demand_mw.sum() * dt),
        "battery_charge_total_mwh": float(schedule.battery_charge_mw.sum() * dt),
        "battery_discharge_total_mwh": float(schedule.battery_discharge_mw.sum() * dt),
        "hydro_spill_total_mwh": float(schedule.hydro_spill_mw.sum() * dt),
        "total_cost_eur": float(schedule.cost_total_eur.sum()),
        "cost_breakdown_eur": {
            col.removeprefix("cost_").removesuffix("_eur"): float(schedule[col].sum())
            for col in schedule if col.startswith("cost_") and col != "cost_total_eur"
        },
    }


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing Phase 4.7 results: {output}")
    output.mkdir(parents=True)
    assets, settings = load_dispatch_config("configs/assets.yaml")
    cases = _cases()
    artifact: dict[str, object] = {
        "phase": "4.7-stage4",
        "dispatch_config": settings,
        "assets": asdict(assets),
        "terminal_soc_policy": "preserve_initial",
        "charging_policy": "renewable_only",
        "initial_battery_soc_mwh": assets.battery.initial_soc_mwh,
        "cases": {},
    }

    for name, case in cases.items():
        index = case["index"]
        demand = np.asarray(case["demand"], dtype=float)
        solar = np.asarray(case["solar"], dtype=float)
        wind = np.asarray(case["wind"], dtype=float)
        limits = np.asarray(case["turbine_limits_mw"], dtype=float)
        state = DispatchState(
            assets.battery.initial_soc_mwh,
            float(case["hydro_budget_mwh"]),
            0.0,
        )
        raw_inputs = pd.DataFrame({
            "timestamp": index,
            "demand_mw": demand,
            "solar_available_mw": solar,
            "wind_available_mw": wind,
            "hydro_turbine_limit_mw": limits,
        })
        raw_inputs.to_csv(output / f"{name}_inputs.csv", index=False)

        pair: dict[str, object] = {"description": case["description"], "variants": {}}
        for enabled in (False, True):
            result = optimize_dispatch(
                demand,
                {"solar": solar, "wind": wind},
                assets,
                state,
                timestamps=index,
                hydro_generation_limits_mw=limits,
                terminal_soc_policy="preserve_initial",
                battery_charging_policy="renewable_only",
                allow_hydro_battery_charging=enabled,
            )
            schedule_path = output / f"{name}_hydro_charge_{'enabled' if enabled else 'disabled'}.csv"
            result.schedule.to_csv(schedule_path, index_label="timestamp")
            metrics = _validate(result.schedule, assets, state, allow_hydro=enabled)
            metrics["solver_status"] = result.solver_status
            metrics["hydro_used_mwh"] = result.hydro_used_mwh
            metrics["initial_hydro_mwh"] = state.hydro_remaining_mwh
            metrics["initial_soc_mwh"] = state.battery_soc_mwh
            metrics["schedule_file"] = schedule_path.name
            pair["variants"]["enabled" if enabled else "disabled"] = metrics

        disabled = pair["variants"]["disabled"]
        enabled = pair["variants"]["enabled"]
        pair["cost_difference_enabled_minus_disabled_eur"] = (
            enabled["total_cost_eur"] - disabled["total_cost_eur"]
        )
        pair["hydro_charge_decision"] = (
            "Hydro was routed to the battery in the enabled schedule."
            if enabled["hydro_to_battery_total_mwh"] > 1e-5
            else "The optimizer did not route hydro to the battery; the enabled option alone does not force charging."
        )
        artifact["cases"][name] = pair

    (output / "scenario_results.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    print(f"Saved Stage 4 schedules and metrics to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/experiments/phase47/stage4")
    run(Path(parser.parse_args().output))
