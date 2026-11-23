from dataclasses import replace

import numpy as np
import pytest

from griddrift.config.schema import load_project_config
from griddrift.dispatch import DispatchState, load_dispatch_config, optimize_dispatch
from griddrift.simulation import load_integrated_settings


@pytest.fixture
def assets():
    return load_dispatch_config("configs/assets.yaml")[0]


def solve(assets, demand, *, solar=None, wind=None, state=None, **kwargs):
    n = len(demand)
    return optimize_dispatch(
        demand,
        {"solar": solar or [0.0] * n, "wind": wind or [0.0] * n},
        assets,
        state,
        terminal_soc_policy="none",
        **kwargs,
    )


def test_hydro_capacity_and_feature_default_load_from_both_configs():
    assets, settings = load_dispatch_config("configs/assets.yaml")
    project = load_project_config("configs/base.yaml")
    assert assets.hydro.reservoir_capacity_mwh == pytest.approx(120.0)
    assert project.assets.hydro.reservoir_capacity_mwh == pytest.approx(336.0)
    assert settings["allow_hydro_battery_charging"] is True
    assert load_integrated_settings("configs/integration.yaml")["allow_hydro_battery_charging"] is True


def test_hydro_generation_obeys_turbine_and_hourly_availability(assets):
    assets = replace(assets, hydro=replace(assets.hydro, capacity_mw=6.0, initial_energy_budget_mwh=12.0,
                                          reservoir_capacity_mwh=12.0))
    result = solve(
        assets, [12.0, 12.0], state=DispatchState(3.0, 12.0, 0.0),
        hydro_generation_limits_mw=[2.0, 0.0],
    )
    assert result.schedule.hydro_mw.iloc[0] <= 2.0 + 1e-7
    assert result.schedule.hydro_mw.iloc[1] == pytest.approx(0.0, abs=1e-7)
    assert (result.schedule.hydro_mw <= 6.0 + 1e-7).all()


def test_inflow_respects_configured_reservoir_capacity_with_spill(assets):
    hydro = replace(assets.hydro, capacity_mw=5.0, initial_energy_budget_mwh=4.0,
                    reservoir_capacity_mwh=5.0)
    assets = replace(assets, hydro=hydro)
    result = solve(assets, [0.0], state=DispatchState(3.0, 4.0, 0.0), hydro_inflow_mw=[4.0])
    row = result.schedule.iloc[0]
    assert row.hydro_spill_mw == pytest.approx(3.0, abs=1e-6)
    assert row.hydro_remaining_mwh == pytest.approx(5.0, abs=1e-6)
    assert row.hydro_remaining_mwh >= 0
    assert result.hydro_spilled_mwh == pytest.approx(3.0, abs=1e-6)


def test_initial_reservoir_state_cannot_exceed_capacity(assets):
    with pytest.raises(ValueError, match="exceeds reservoir capacity"):
        solve(assets, [1.0], state=DispatchState(3.0, 121.0, 0.0))


def test_hydro_allocation_is_not_double_counted_when_charging(assets):
    result = solve(
        assets, [10.0, 10.0], solar=[0.0, 0.0], wind=[0.0, 0.0],
        state=DispatchState(3.0, 8.0, 0.0),
        conventional_costs_eur_mwh=[10.0, 500.0],
        hydro_generation_limits_mw=[8.0, 0.0],
        allow_hydro_battery_charging=True,
    )
    f = result.schedule
    assert np.allclose(f.hydro_mw, f.hydro_to_demand_mw + f.hydro_to_battery_mw, atol=1e-6)
    assert np.allclose(
        f.battery_charge_mw,
        f.solar_to_battery_mw + f.wind_to_battery_mw + f.hydro_to_battery_mw,
        atol=1e-6,
    )
    assert f.hydro_to_battery_mw.iloc[0] > 1e-6
    assert f.hydro_to_battery_mw.iloc[0] <= f.hydro_mw.iloc[0] + 1e-7
    expected_soc = 3.0
    for t, row in f.iterrows():
        expected_soc += (
            assets.battery.charge_efficiency * row.battery_charge_mw
            - row.battery_discharge_mw / assets.battery.discharge_efficiency
        ) * assets.interval_hours
        assert row.soc_mwh == pytest.approx(expected_soc, abs=1e-6)
        assert assets.battery.min_soc_mwh - 1e-6 <= row.soc_mwh <= assets.battery.max_soc_mwh + 1e-6
    assert not ((f.battery_charge_mw > 1e-6) & (f.battery_discharge_mw > 1e-6)).any()


def test_hydro_battery_feature_reduces_cost_only_when_peak_turbine_is_unavailable(assets):
    state = DispatchState(3.0, 8.0, 0.0)
    common = {
        "conventional_costs_eur_mwh": [10.0, 500.0],
        "hydro_generation_limits_mw": [8.0, 0.0],
    }
    previous = solve(assets, [10.0, 10.0], state=state, **common)
    enabled = solve(assets, [10.0, 10.0], state=state,
                    allow_hydro_battery_charging=True, **common)
    assert enabled.schedule.hydro_to_battery_mw.iloc[0] > 1e-6
    assert enabled.schedule.battery_discharge_mw.iloc[1] > 1e-6
    assert enabled.total_cost_eur < previous.total_cost_eur


def test_conventional_and_emergency_cannot_charge_in_hydro_enabled_mode(assets):
    assets = replace(assets, hydro=replace(assets.hydro, initial_energy_budget_mwh=0.0,
                                          reservoir_capacity_mwh=0.0))
    result = solve(
        assets, [10.0, 10.0], state=DispatchState(3.0, 0.0, 0.0),
        allow_hydro_battery_charging=True,
    )
    f = result.schedule
    assert np.allclose(f.battery_charge_mw, 0.0, atol=1e-7)
    assert (f.conventional_mw > 0).any()
    assert np.allclose(f.emergency_mw, 0.0, atol=1e-7)

    no_conventional = replace(
        assets,
        conventional=replace(assets.conventional, capacity_mw=0.0),
        hydro=replace(assets.hydro, initial_energy_budget_mwh=0.0,
                      reservoir_capacity_mwh=0.0),
    )
    emergency_only = solve(
        no_conventional, [10.0, 10.0], state=DispatchState(3.0, 0.0, 0.0),
        allow_hydro_battery_charging=True,
    ).schedule
    assert (emergency_only.emergency_mw > 0).any()
    assert np.allclose(emergency_only.battery_charge_mw, 0.0, atol=1e-7)


def test_legacy_unrestricted_charging_remains_available_when_feature_is_off(assets):
    assets = replace(assets, conventional=replace(assets.conventional, marginal_cost_eur_mwh=10.0),
                     hydro=replace(assets.hydro, initial_energy_budget_mwh=0.0,
                                   reservoir_capacity_mwh=0.0))
    result = solve(
        assets, [0.0, 5.0], state=DispatchState(3.0, 0.0, 0.0),
        conventional_costs_eur_mwh=[10.0, 500.0],
        battery_charging_policy="unrestricted",
    )
    assert result.schedule.battery_charge_mw.iloc[0] > 0.0
    assert result.schedule.solar_to_battery_mw.iloc[0] == pytest.approx(0.0, abs=1e-7)
    assert result.schedule.wind_to_battery_mw.iloc[0] == pytest.approx(0.0, abs=1e-7)


def test_hydro_generation_cost_is_charged_once_and_costs_reconcile(assets):
    result = solve(
        assets, [10.0, 10.0], state=DispatchState(3.0, 8.0, 0.0),
        conventional_costs_eur_mwh=[10.0, 500.0],
        hydro_generation_limits_mw=[8.0, 0.0],
        allow_hydro_battery_charging=True,
    )
    f = result.schedule
    expected_hydro_cost = assets.hydro.marginal_cost_eur_mwh * f.hydro_mw.sum() * assets.interval_hours
    assert f.hydro_to_battery_mw.iloc[0] > 1e-6
    assert f.cost_hydro_eur.sum() == pytest.approx(expected_hydro_cost, abs=1e-6)
    assert result.cost_breakdown_eur["hydro"] == pytest.approx(expected_hydro_cost, abs=1e-6)
    assert sum(result.cost_breakdown_eur.values()) == pytest.approx(result.total_cost_eur, abs=1e-5)


def test_hydro_conservation_wins_when_turbine_is_available_later(assets):
    state = DispatchState(3.0, 5.0, 0.0)
    shared = {"hydro_generation_limits_mw": [5.0, 5.0, 5.0]}
    disabled = solve(assets, [0.0, 0.0, 20.0], state=state, **shared)
    enabled = solve(assets, [0.0, 0.0, 20.0], state=state,
                    allow_hydro_battery_charging=True, **shared)
    assert enabled.schedule.hydro_to_battery_mw.sum() == pytest.approx(0.0, abs=1e-6)
    assert enabled.schedule.hydro_mw.iloc[2] > 0
    assert enabled.total_cost_eur == pytest.approx(disabled.total_cost_eur, abs=1e-5)


def test_hydro_enabled_feature_requires_source_restricted_policy(assets):
    with pytest.raises(ValueError, match="requires battery_charging_policy"):
        solve(assets, [1.0], allow_hydro_battery_charging=True,
              battery_charging_policy="unrestricted")
