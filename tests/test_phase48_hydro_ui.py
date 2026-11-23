from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from griddrift.dispatch import DispatchState, load_dispatch_config, optimize_dispatch
from griddrift.simulation import IntegratedConfig, IntegratedSimulation
from scripts.streamlit_app import (
    HYDRO_CHARGING_DEFAULT_SESSION_KEY,
    _active_plan_comparison_key,
    _dispatch_table,
    _sync_hydro_charging_default,
)


@pytest.fixture
def assets():
    return load_dispatch_config("configs/assets.yaml")[0]


class _FlatModel:
    model_version = "phase48-test"

    def __init__(self, index):
        self.feature_spec = SimpleNamespace(lookback_hours=168, horizon_hours=24)
        self.train_end_issue_time = index[-120]

    def predict_horizon(self, history, issue_time):
        target_index = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=24, freq="h")
        return pd.Series(float(history.iloc[-1]), index=target_index)


class _NoAlarm:
    def update(self, _value):
        return False


def test_hydro_toggle_reoptimizes_without_advancing_replay(assets, monkeypatch):
    from griddrift.simulation import integrated

    real_optimize = integrated.optimize_dispatch
    observed_flags = []

    def recording_optimize(*args, **kwargs):
        observed_flags.append(kwargs["allow_hydro_battery_charging"])
        return real_optimize(*args, **kwargs)

    monkeypatch.setattr(integrated, "optimize_dispatch", recording_optimize)
    index = pd.date_range("2024-01-01", periods=260, freq="h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    sim = IntegratedSimulation(
        demand,
        _FlatModel(index),
        assets,
        config=IntegratedConfig(demand_scale=0.001),
        detector=_NoAlarm(),
    )
    old_timestamp = sim.timestamp
    old_forecast = sim.forecast.copy()
    old_registry_count = len(sim.adaptive_registry.to_frame())
    old_plan = sim.plan

    assert sim.set_hydro_battery_charging(True) is True

    assert sim.config.allow_hydro_battery_charging is True
    assert sim.plan is not old_plan
    assert sim.timestamp == old_timestamp
    pd.testing.assert_series_equal(sim.forecast, old_forecast)
    assert len(sim.adaptive_registry.to_frame()) == old_registry_count

    assert sim.set_hydro_battery_charging(False) is True
    assert sim.config.allow_hydro_battery_charging is False
    assert observed_flags == [False, True, False]
    assert sim.timestamp == old_timestamp
    pd.testing.assert_series_equal(sim.forecast, old_forecast)
    assert len(sim.adaptive_registry.to_frame()) == old_registry_count


def test_cached_session_migrates_to_enabled_default_once_and_invalidates_old_plan(assets):
    index = pd.date_range("2024-01-01", periods=260, freq="h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    sim = IntegratedSimulation(
        demand,
        _FlatModel(index),
        assets,
        config=IntegratedConfig(demand_scale=0.001, allow_hydro_battery_charging=False),
        detector=_NoAlarm(),
    )
    old_plan = sim.plan
    old_comparison_key = _active_plan_comparison_key(sim)
    session_state = {
        HYDRO_CHARGING_DEFAULT_SESSION_KEY: None,
        f"hydro_battery_enabled_{id(sim)}": False,
    }
    old_time = sim.timestamp

    assert _sync_hydro_charging_default(session_state, sim, True) is True
    assert sim.config.allow_hydro_battery_charging is True
    assert sim.plan is not old_plan
    assert _active_plan_comparison_key(sim) != old_comparison_key
    assert session_state[f"hydro_battery_enabled_{id(sim)}"] is True
    assert sim.timestamp == old_time

    assert sim.set_hydro_battery_charging(False) is True
    assert _sync_hydro_charging_default(session_state, sim, True) is False
    assert sim.config.allow_hydro_battery_charging is False


def test_hydro_allocation_fields_remain_in_planning_table(assets):
    index = pd.date_range("2025-01-01", periods=2, freq="h", tz="UTC")
    state = DispatchState(battery_soc_mwh=3.0, hydro_remaining_mwh=8.0)
    renewable = {"solar": [0.0, 0.0], "wind": [0.0, 0.0]}
    hydro_limits = [8.0, 0.0]
    prices = [10.0, 500.0]
    emergency_prices = [450.0, 450.0]
    plan = optimize_dispatch(
        [10.0, 10.0], renewable, assets, state,
        timestamps=index,
        hydro_generation_limits_mw=hydro_limits,
        conventional_costs_eur_mwh=prices,
        emergency_costs_eur_mwh=emergency_prices,
        terminal_soc_policy="none",
    )
    table_sim = SimpleNamespace(
        plan=plan,
        assets=assets,
        forecast=pd.Series([10.0, 10.0], index=index),
    )

    table = _dispatch_table(table_sim)
    assert "Hydro to demand (MW)" in table
    assert "Hydro to battery (MW)" in table
    assert "Hydro inflow (MW)" in table
