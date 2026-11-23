from types import SimpleNamespace

import pandas as pd
import pytest

from griddrift.dispatch import DispatchState, load_dispatch_config, optimize_dispatch
from griddrift.dispatch.baselines import run_greedy_dispatch
from scripts.streamlit_app import (
    _active_plan_comparison_key,
    _compare_active_plan_with_greedy,
    _dispatch_table,
)


def test_dispatch_table_fills_rates_for_an_already_loaded_older_plan():
    assets, _ = load_dispatch_config("configs/assets.yaml")
    plan = optimize_dispatch(
        [10], {"solar": [2], "wind": [1]}, assets, terminal_soc_policy="none"
    )
    plan.schedule = plan.schedule.drop(columns=[
        "solar_cost_rate_eur_mwh", "wind_cost_rate_eur_mwh", "hydro_cost_rate_eur_mwh",
        "conventional_cost_rate_eur_mwh", "battery_degradation_rate_eur_mwh",
    ])
    sim = SimpleNamespace(
        plan=plan, assets=assets, forecast=pd.Series([10.0], index=plan.schedule.index)
    )

    table = _dispatch_table(sim)

    assert table.loc[0, "Solar €/MWh"] == assets.solar.marginal_cost_eur_mwh
    assert table.loc[0, "Wind €/MWh"] == assets.wind.marginal_cost_eur_mwh
    assert table.loc[0, "Conventional €/MWh"] == assets.conventional.marginal_cost_eur_mwh
    assert table.loc[0, "Battery wear €/MWh"] == assets.battery.degradation_cost_eur_mwh


def test_active_plan_greedy_comparison_uses_same_hourly_prices_and_invalidates_on_plan_change():
    assets, _ = load_dispatch_config("configs/assets.yaml")
    index = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    renewable = {"solar": [2.0, 1.0, 0.0], "wind": [1.0, 2.0, 1.0]}
    state = DispatchState(15.0, 20.0, 0.0)
    prices = [90.0, 110.0, 130.0]
    emergency_prices = [450.0, 460.0, 470.0]
    plan = optimize_dispatch(
        [10.0, 12.0, 11.0], renewable, assets, state,
        conventional_costs_eur_mwh=prices,
        emergency_costs_eur_mwh=emergency_prices,
        timestamps=index,
        terminal_soc_policy="preserve_initial",
    )
    sim = SimpleNamespace(
        plan=plan,
        assets=assets,
        timestamp=index[0],
        active_model=SimpleNamespace(model_version="test-model"),
        config=SimpleNamespace(
            policy="static", resource_scenario="normal",
            battery_charging_policy="renewable_only", terminal_soc_policy="preserve_initial",
        ),
        resource_forecast=renewable,
    )

    result = _compare_active_plan_with_greedy(sim)
    _, expected_raw, expected_settlement = run_greedy_dispatch(
        [10.0, 12.0, 11.0], renewable, assets, state,
        terminal_soc_target_mwh=state.battery_soc_mwh,
        timestamps=index,
        battery_charging_policy="renewable_only",
        conventional_costs_eur_mwh=prices,
        emergency_costs_eur_mwh=emergency_prices,
    )

    assert result["multi_period_total_eur"] == pytest.approx(plan.total_cost_eur)
    assert result["greedy_total_eur"] == pytest.approx(expected_raw + expected_settlement)
    assert result["savings_eur"] == pytest.approx(
        result["greedy_total_eur"] - result["multi_period_total_eur"]
    )
    original_key = _active_plan_comparison_key(sim)
    sim.plan.schedule.loc[index[0], "demand_mw"] += 1.0
    assert _active_plan_comparison_key(sim) != original_key
