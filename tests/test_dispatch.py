from dataclasses import replace

import numpy as np
import pytest

from griddrift.dispatch import (
    DispatchSolveError,
    DispatchState,
    load_dispatch_config,
    optimize_dispatch,
)
from griddrift.dispatch.baselines import run_greedy_dispatch, terminal_soc_settlement


@pytest.fixture
def assets():
    return load_dispatch_config("configs/assets.yaml")[0]


def solve(assets, demand, solar=None, wind=None, state=None, **kwargs):
    n = len(demand)
    return optimize_dispatch(
        demand,
        {"solar": solar or [0.0] * n, "wind": wind or [0.0] * n},
        assets,
        state,
        terminal_soc_policy="none",
        **kwargs,
    )


def test_hourly_power_balance_and_renewable_accounting(assets):
    result = solve(assets, [8, 12], solar=[10, 3], wind=[4, 5])
    f = result.schedule
    supplied = f[
        [
            "solar_mw",
            "wind_mw",
            "hydro_mw",
            "conventional_mw",
            "battery_discharge_mw",
            "emergency_mw",
            "unserved_mw",
        ]
    ].sum(axis=1)
    required = f.demand_mw + f.battery_charge_mw
    assert np.allclose(supplied, required, atol=1e-6)
    assert np.allclose(f.solar_mw + f.solar_curtailment_mw, f.solar_available_mw)
    assert np.allclose(f.wind_mw + f.wind_curtailment_mw, f.wind_available_mw)
    assert np.allclose(f.solar_to_demand_mw + f.solar_to_battery_mw + f.solar_curtailment_mw, f.solar_available_mw)
    assert np.allclose(f.wind_to_demand_mw + f.wind_to_battery_mw + f.wind_curtailment_mw, f.wind_available_mw)
    assert np.allclose(f.battery_charge_mw, f.solar_to_battery_mw + f.wind_to_battery_mw)
    assert (f.solar_mw <= f.solar_available_mw + 1e-7).all()
    assert (f.wind_mw <= f.wind_available_mw + 1e-7).all()


def test_battery_soc_efficiency_bounds_and_no_simultaneous_operation(assets):
    assets = replace(assets, hydro=replace(assets.hydro, initial_energy_budget_mwh=0))
    result = solve(
        assets,
        [4, 4, 20, 20],
        solar=[12, 12, 0, 0],
        wind=[0, 0, 0, 0],
        state=DispatchState(8, 0, 0),
    )
    f, b, dt = result.schedule, assets.battery, assets.interval_hours
    previous = 8.0
    for _, row in f.iterrows():
        expected = (
            previous
            + b.charge_efficiency * row.battery_charge_mw * dt
            - row.battery_discharge_mw * dt / b.discharge_efficiency
        )
        assert row.soc_mwh == pytest.approx(expected, abs=1e-6)
        assert b.min_soc_mwh - 1e-6 <= row.soc_mwh <= b.max_soc_mwh + 1e-6
        assert not (row.battery_charge_mw > 1e-6 and row.battery_discharge_mw > 1e-6)
        previous = row.soc_mwh
    assert result.formulation == "LP"


def test_renewable_only_policy_prevents_grid_energy_from_charging_battery(assets):
    assets = replace(assets, hydro=replace(assets.hydro, initial_energy_budget_mwh=0))
    result = solve(
        assets,
        [5, 5],
        solar=[8, 0],
        wind=[4, 0],
        state=DispatchState(5, 0, 0),
    )
    frame = result.schedule
    assert np.allclose(frame.battery_charge_mw, frame.solar_to_battery_mw + frame.wind_to_battery_mw)
    assert frame.battery_charge_mw.iloc[0] > 1e-6
    assert frame.battery_charge_mw.iloc[1] == pytest.approx(0, abs=1e-7)


def test_milp_mode_binary_activates_when_linear_exclusivity_assumptions_do_not_hold(assets):
    battery = replace(assets.battery, degradation_cost_eur_mwh=0)
    assets = replace(
        assets, battery=battery, hydro=replace(assets.hydro, initial_energy_budget_mwh=0)
    )
    result = solve(assets, [10, 10], state=DispatchState(8, 0, 0))
    assert result.formulation == "MILP"
    assert not (
        (result.schedule.battery_charge_mw > 1e-6) & (result.schedule.battery_discharge_mw > 1e-6)
    ).any()


def test_hydro_energy_budget_persists_and_inflow_replenishes(assets):
    assets = replace(assets, hydro=replace(assets.hydro, capacity_mw=5, marginal_cost_eur_mwh=10))
    result = solve(
        assets,
        [10, 10, 10],
        solar=[0, 0, 0],
        wind=[0, 0, 0],
        state=DispatchState(15, 4, 0),
        hydro_inflow_mw=[0, 2, 0],
    )
    f = result.schedule
    assert f.hydro_mw.sum() * assets.interval_hours <= 6 + 1e-6
    assert f.hydro_remaining_mwh.iloc[-1] == pytest.approx(
        6 - f.hydro_mw.sum() * assets.interval_hours, abs=1e-6
    )
    assert result.first_hour_state.hydro_remaining_mwh == pytest.approx(4 - f.hydro_mw.iloc[0])
    next_call = optimize_dispatch(
        [10],
        {"solar": [0], "wind": [0]},
        assets,
        result.first_hour_state,
        terminal_soc_policy="none",
    )
    assert next_call.initial_state.hydro_remaining_mwh == pytest.approx(
        result.first_hour_state.hydro_remaining_mwh
    )


def test_hydro_is_conserved_for_later_tighter_hours(assets):
    assets = replace(
        assets,
        hydro=replace(
            assets.hydro, capacity_mw=5, marginal_cost_eur_mwh=10, initial_energy_budget_mwh=5
        ),
    )
    result = solve(assets, [10, 10, 20], solar=[10, 10, 0], wind=[0, 0, 0])
    assert result.schedule.hydro_mw.iloc[0] < 1e-6
    assert result.schedule.hydro_mw.iloc[1] < 1e-6
    assert result.schedule.hydro_mw.iloc[2] == pytest.approx(5, abs=1e-6)


def test_ramp_limits_and_first_hour_state(assets):
    assets = replace(
        assets, conventional_ramp_up_mw_per_interval=4, conventional_ramp_down_mw_per_interval=3
    )
    result = solve(assets, [20, 30], state=DispatchState(15, 0, 10))
    p = result.schedule.conventional_mw.to_numpy()
    assert p[0] <= 14 + 1e-6
    assert p[0] - 10 <= 4 + 1e-6
    assert 10 - p[0] <= 3 + 1e-6
    assert p[1] - p[0] <= 4 + 1e-6
    assert result.first_hour_state.conventional_output_mw == pytest.approx(p[0])


def test_terminal_soc_preserve_initial_and_optional_policy(assets):
    result = optimize_dispatch([10, 20], {"solar": [0, 0], "wind": [0, 0]}, assets)
    assert result.schedule.soc_mwh.iloc[-1] == pytest.approx(
        assets.battery.initial_soc_mwh, abs=1e-6
    )
    no_terminal = solve(assets, [10, 20])
    assert len(no_terminal.schedule) == 2
    reserve = optimize_dispatch(
        [10, 20],
        {"solar": [0, 0], "wind": [0, 0]},
        assets,
        terminal_soc_policy="target",
        terminal_soc_target_mwh=10,
    )
    assert reserve.schedule.soc_mwh.iloc[-1] >= 10 - 1e-6


def test_emergency_and_voll_unserved_are_explicit(assets):
    assets = replace(
        assets,
        conventional=replace(assets.conventional, capacity_mw=2),
        emergency_grid=replace(assets.emergency_grid, capacity_mw=3),
        hydro=replace(assets.hydro, initial_energy_budget_mwh=0),
    )
    result = solve(assets, [20], state=DispatchState(3, 0, 0))
    row = result.schedule.iloc[0]
    assert row.conventional_mw == pytest.approx(2)
    assert row.emergency_mw == pytest.approx(3)
    assert row.unserved_mw > 0
    assert result.cost_breakdown_eur["unserved_demand_voll"] == pytest.approx(
        row.unserved_mw * assets.value_of_lost_load_eur_mwh
    )


def test_cost_breakdown_and_storage_does_not_double_count_energy(assets):
    result = solve(assets, [20, 20, 20])
    assert sum(result.cost_breakdown_eur.values()) == pytest.approx(result.total_cost_eur)
    # Battery discharge has no second energy-purchase charge; only throughput wear is added.
    assert result.cost_breakdown_eur["battery_degradation"] == pytest.approx(
        assets.battery.degradation_cost_eur_mwh
        * (result.schedule.battery_charge_mw.sum() + result.schedule.battery_discharge_mw.sum())
        * assets.interval_hours
    )


def test_time_varying_conventional_prices_are_respected(assets):
    result = solve(assets, [20, 20], conventional_costs_eur_mwh=[30, 300])
    assert result.schedule.conventional_mw.iloc[0] >= result.schedule.conventional_mw.iloc[1]
    assert result.schedule.conventional_cost_rate_eur_mwh.tolist() == [30, 300]
    assert result.schedule.cost_conventional_eur.tolist() == pytest.approx(
        (result.schedule.conventional_mw * [30, 300] * assets.interval_hours).tolist()
    )


def test_unavailable_solver_and_infeasible_ramp_are_reported(assets):
    with pytest.raises(DispatchSolveError, match="unavailable"):
        solve(assets, [10], solver_name="not-a-real-solver")
    with pytest.raises(DispatchSolveError, match="did not solve optimally"):
        solve(
            replace(assets, conventional_ramp_down_mw_per_interval=0),
            [0],
            state=DispatchState(15, 0, 10),
        )


def test_myopic_baseline_carries_physical_state_and_multi_period_beats_it_on_ramp_case(assets):
    assets = replace(
        assets,
        conventional_ramp_up_mw_per_interval=5,
        hydro=replace(assets.hydro, initial_energy_budget_mwh=0),
    )
    demand = [20] * 10 + [24, 28, 34, 40, 46, 50, 52, 50, 46, 40, 34, 28, 24, 20]
    solar = [0, 0, 0, 0, 0, 1, 2, 4, 6, 8, 10, 12, 12, 10, 8, 5, 2, 0, 0, 0, 0, 0, 0, 0]
    wind = [8] * len(demand)
    state = DispatchState(8, 0, 10)
    multi = optimize_dispatch(demand, {"solar": solar, "wind": wind}, assets, state)
    fair_multi = optimize_dispatch(
        demand, {"solar": solar, "wind": wind}, assets, state, terminal_soc_policy="none"
    )
    greedy, raw_cost, settlement = run_greedy_dispatch(
        demand,
        {"solar": solar, "wind": wind},
        assets,
        state,
        terminal_soc_target_mwh=state.battery_soc_mwh,
    )
    assert (multi.schedule.battery_charge_mw > 0).any()
    multi_settlement = terminal_soc_settlement(
        fair_multi.schedule.soc_mwh.iloc[-1], state.battery_soc_mwh, assets,
        assets.conventional.marginal_cost_eur_mwh,
    )
    assert fair_multi.total_cost_eur + multi_settlement < raw_cost + settlement
    changes = greedy.conventional_mw.diff().dropna()
    assert (changes <= 5 + 1e-6).all()
    assert (changes >= -assets.conventional_ramp_down_mw_per_interval - 1e-6).all()


@pytest.mark.parametrize("policy, hydro", [("renewable_only", False), ("renewable_only", True), ("unrestricted", False)])
def test_energy_is_conserved_when_the_battery_charges(assets, policy, hydro):
    """Regression: renewable energy sent to storage must not be generated a second time."""
    assets = replace(assets, wind=replace(assets.wind, capacity_mw=30.0))
    result = solve(
        assets, [40.0, 40.0], wind=[20.0, 0.0], state=DispatchState(3.0, 0.0, 30.0),
        conventional_costs_eur_mwh=[50.0, 120.0],
        battery_charging_policy=policy, allow_hydro_battery_charging=hydro,
    )
    schedule = result.schedule
    generated = (
        schedule.solar_mw + schedule.wind_mw + schedule.hydro_mw + schedule.conventional_mw
        + schedule.battery_discharge_mw + schedule.emergency_mw + schedule.unserved_mw
    )
    np.testing.assert_allclose(generated, schedule.demand_mw + schedule.battery_charge_mw, atol=1e-6)
    # Charging 8 MW at night (EUR 50) to discharge at the EUR 120 peak is worth it.
    assert schedule.battery_charge_mw.iloc[0] == pytest.approx(8.0, abs=1e-6)
    assert schedule.battery_discharge_mw.iloc[1] > 0


def test_wind_charging_needs_no_extra_generation(assets):
    assets = replace(assets, wind=replace(assets.wind, capacity_mw=30.0))
    result = solve(
        assets, [40.0, 40.0], wind=[20.0, 0.0], state=DispatchState(3.0, 0.0, 30.0),
        conventional_costs_eur_mwh=[50.0, 120.0],
    )
    first = result.schedule.iloc[0]
    assert first.wind_to_battery_mw == pytest.approx(8.0, abs=1e-6)
    # Demand is met by the remaining 12 MW of wind plus 28 MW conventional, not 36 MW.
    assert first.conventional_mw == pytest.approx(28.0, abs=1e-6)


def test_hourly_hydro_prices_are_used_in_objective_and_reported(assets):
    result = solve(assets, [40.0, 40.0], hydro_costs_eur_mwh=[16.0, 28.0])
    schedule = result.schedule
    assert schedule.hydro_cost_rate_eur_mwh.tolist() == [16.0, 28.0]
    np.testing.assert_allclose(
        schedule.cost_hydro_eur, schedule.hydro_cost_rate_eur_mwh * schedule.hydro_mw, atol=1e-9
    )
    with pytest.raises(ValueError):
        solve(assets, [40.0, 40.0], hydro_costs_eur_mwh=[-1.0, 20.0])
