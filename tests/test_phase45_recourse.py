"""Focused tests for Phase 4.5's paired committed-plan recourse rule."""

import pytest

from griddrift.dispatch import DispatchState, load_dispatch_config, optimize_dispatch


@pytest.fixture
def assets():
    return load_dispatch_config("configs/assets.yaml")[0]


def first_hour_commitment(schedule):
    row = schedule.iloc[0]
    return {
        key: float(row[key])
        for key in (
            "solar_mw", "wind_mw", "hydro_mw", "conventional_mw",
            "battery_charge_mw", "battery_discharge_mw", "emergency_mw", "unserved_mw",
        )
    }


def test_committed_recourse_adjusts_schedule_and_preserves_physical_accounting(assets):
    state = DispatchState(15.0, 80.0, 10.0)
    plan = optimize_dispatch(
        [30.0, 32.0],
        {"solar": [8.0, 10.0], "wind": [7.0, 8.0]},
        assets,
        state,
        terminal_soc_policy="preserve_initial",
    )
    commitment = first_hour_commitment(plan.schedule)

    actual = optimize_dispatch(
        [38.0],
        {"solar": [2.0], "wind": [5.0]},
        assets,
        state,
        terminal_soc_policy="none",
        commitment_schedule=commitment,
    )

    row = actual.schedule.iloc[0]
    supplied = sum(row[key] for key in (
        "solar_mw", "wind_mw", "hydro_mw", "conventional_mw",
        "battery_discharge_mw", "emergency_mw", "unserved_mw",
    ))
    assert supplied == pytest.approx(row.demand_mw + row.battery_charge_mw, abs=1e-6)
    assert row.solar_mw <= row.solar_available_mw + 1e-7
    assert row.wind_mw <= row.wind_available_mw + 1e-7
    assert assets.battery.min_soc_mwh <= row.soc_mwh <= assets.battery.max_soc_mwh
    assert 0 <= row.hydro_remaining_mwh <= state.hydro_remaining_mwh
    assert abs(row.conventional_mw - state.conventional_output_mw) <= (
        assets.conventional_ramp_up_mw_per_interval + 1e-7
    )
    assert abs(row.conventional_mw - state.conventional_output_mw) <= (
        assets.conventional_ramp_down_mw_per_interval + 1e-7
    )

    expected_deviation = sum(abs(float(row[key]) - commitment[key]) for key in commitment)
    assert row.commitment_deviation_mw == pytest.approx(expected_deviation, abs=1e-6)
    assert row.commitment_policy == "minimum_l1_setpoint_adjustment_then_configured_cost"
    assert actual.total_cost_eur == pytest.approx(row.cost_total_eur, abs=1e-6)
    assert actual.total_cost_eur == pytest.approx(sum(actual.cost_breakdown_eur.values()), abs=1e-5)
    assert actual.initial_state == state


def test_recourse_is_symmetric_for_same_inputs_and_independent_states(assets):
    initial = DispatchState(15.0, 70.0, 12.0)
    left_state, right_state = DispatchState(**initial.__dict__), DispatchState(**initial.__dict__)
    plan = optimize_dispatch(
        [35.0], {"solar": [6.0], "wind": [9.0]}, assets, left_state,
        terminal_soc_policy="none",
    )
    commitment = first_hour_commitment(plan.schedule)

    left = optimize_dispatch(
        [37.0], {"solar": [5.0], "wind": [8.0]}, assets, left_state,
        terminal_soc_policy="none", commitment_schedule=commitment,
    )
    right = optimize_dispatch(
        [37.0], {"solar": [5.0], "wind": [8.0]}, assets, right_state,
        terminal_soc_policy="none", commitment_schedule=commitment,
    )

    assert left.schedule.equals(right.schedule)
    assert left.total_cost_eur == pytest.approx(right.total_cost_eur)
    assert left.initial_state == right.initial_state == initial
    assert left.first_hour_state == right.first_hour_state


def test_commitment_requires_exactly_one_hour_and_complete_setpoints(assets):
    with pytest.raises(ValueError, match="one-interval"):
        optimize_dispatch(
            [20.0, 21.0], {"solar": [0.0, 0.0], "wind": [0.0, 0.0]}, assets,
            commitment_schedule={key: 0.0 for key in first_hour_commitment.__annotations__},
        )

    with pytest.raises(ValueError, match="setpoints mismatch"):
        optimize_dispatch(
            [20.0], {"solar": [0.0], "wind": [0.0]}, assets,
            commitment_schedule={"conventional_mw": 20.0},
        )


def test_default_dispatch_has_no_commitment_deviation_column(assets):
    result = optimize_dispatch([20.0], {"solar": [3.0], "wind": [4.0]}, assets)
    assert "commitment_deviation_mw" not in result.schedule.columns
    assert "commitment_policy" not in result.schedule.columns


def test_solver_boundary_tolerance_is_normalized_for_next_terminal_target(assets):
    near_minimum = DispatchState(assets.battery.min_soc_mwh - 5e-9, 0.0, 0.0)
    result = optimize_dispatch(
        [10.0], {"solar": [0.0], "wind": [0.0]}, assets, near_minimum,
        terminal_soc_policy="preserve_initial",
    )
    assert result.initial_state.battery_soc_mwh == assets.battery.min_soc_mwh
    assert result.schedule.soc_mwh.iloc[-1] == pytest.approx(assets.battery.min_soc_mwh)
