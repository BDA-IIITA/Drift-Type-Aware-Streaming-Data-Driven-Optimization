from dataclasses import replace
from pathlib import Path

import yaml

from griddrift.dispatch import DispatchState, load_dispatch_config, optimize_dispatch


def test_all_deterministic_scenarios_solve_and_demonstrate_required_behaviors():
    assets, settings = load_dispatch_config("configs/assets.yaml")
    outcomes = {}
    for path in sorted(Path("configs/scenarios").glob("*.yaml")):
        scenario = yaml.safe_load(path.read_text(encoding="utf-8"))
        configured_assets = replace(assets, **scenario.get("asset_overrides", {}))
        state_raw = scenario.get("initial_state", {})
        state = DispatchState(
            state_raw.get("battery_soc_mwh", configured_assets.battery.initial_soc_mwh),
            state_raw.get("hydro_remaining_mwh", configured_assets.hydro.initial_energy_budget_mwh),
            state_raw.get("conventional_output_mw", 0.0),
        )
        assert len(scenario["demand_mw"]) == settings["horizon_hours"] == 24
        result = optimize_dispatch(
            scenario["demand_mw"],
            {"solar": scenario["solar_available_mw"], "wind": scenario["wind_available_mw"]},
            configured_assets,
            state,
            hydro_inflow_mw=scenario.get("hydro_inflow_mw"),
            conventional_costs_eur_mwh=scenario.get("conventional_costs_eur_mwh"),
            emergency_costs_eur_mwh=scenario.get("emergency_costs_eur_mwh"),
            conventional_min_output_mw=settings["conventional_min_output_mw"],
            terminal_soc_policy=settings["terminal_soc_policy"],
        )
        assert result.termination_condition == "optimal"
        assert len(result.schedule) == 24
        assert (
            result.schedule.hydro_mw.sum()
            <= state.hydro_remaining_mwh + sum(scenario.get("hydro_inflow_mw", [0] * 24)) + 1e-6
        )
        outcomes[scenario["name"]] = result

    battery = outcomes["battery_storage"].schedule
    assert (battery.battery_charge_mw > 1e-4).any()
    assert (battery.battery_discharge_mw > 1e-4).any()
    hydro = outcomes["hydro_conservation"].schedule
    assert hydro.hydro_mw.iloc[:16].sum() < hydro.hydro_mw.iloc[16:].sum()
    shortage = outcomes["renewable_shortage"].schedule
    assert shortage.emergency_mw.sum() > 0
    assert shortage.unserved_mw.sum() > 0
    assert outcomes["normal_day"].schedule.solar_available_mw.sum() > 0
    assert (
        outcomes["evening_peak"].schedule.demand_mw.max()
        > outcomes["evening_peak"].schedule.demand_mw.iloc[:12].max()
    )
