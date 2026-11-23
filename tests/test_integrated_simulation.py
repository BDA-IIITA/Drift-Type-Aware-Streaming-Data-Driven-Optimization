from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from griddrift.dashboard.views import (
    dashboard_snapshot_figure,
    daily_planned_cost_eur,
    demand_chart_frame,
    dispatch_download_frame,
    streaming_forecast_frame,
)
from griddrift.dispatch import load_dispatch_config
from griddrift.simulation import IntegratedConfig, IntegratedSimulation, load_integrated_settings
from scripts.streamlit_app import _figure, _replay_config_for_data


@dataclass
class ForecastSpec:
    lookback_hours: int = 168
    horizon_hours: int = 24


class SimpleModel:
    """Fast deterministic model double for integration sequencing tests."""

    def __init__(self, index):
        self.feature_spec = ForecastSpec()
        self.model_version = "test-static"
        self.train_end_issue_time = index[-120]

    def predict_horizon(self, history, issue_time):
        targets = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=24, freq="1h")
        level = float(history.loc[:issue_time].iloc[-1])
        return pd.Series(np.full(24, level), index=targets, name="predicted_demand_mw")


class NoAlarmDetector:
    def update(self, value):
        return False


class AlarmDetector:
    last_change = None

    def __init__(self):
        self.called = False

    def update(self, value):
        if not self.called:
            self.called = True
            return True
        return False


def _simulation(demand):
    assets = load_dispatch_config("configs/assets.yaml")[0]
    model = SimpleModel(demand.index)
    return IntegratedSimulation(demand, model, assets, config=IntegratedConfig(demand_scale=0.001), detector=NoAlarmDetector())


def test_complete_forecast_and_dispatch_share_exact_future_timestamps():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    simulation = _simulation(demand)
    forecast_times = simulation.forecast.index
    plan_times = pd.DatetimeIndex(simulation.plan.schedule.index)
    assert len(forecast_times) == len(plan_times) == 24
    assert forecast_times.equals(plan_times)
    assert forecast_times[0] == simulation.timestamp + pd.Timedelta(hours=1)
    schedule = simulation.plan.schedule
    assert np.allclose(schedule.battery_charge_mw, schedule.solar_to_battery_mw + schedule.wind_to_battery_mw)


def test_integrated_yaml_exposes_scaling_and_renewable_charging_policy():
    settings = load_integrated_settings("configs/integration.yaml")
    assert settings["demand_scale"] == 0.001
    assert settings["battery_charging_policy"] == "renewable_only"
    assert settings["allow_hydro_battery_charging"] is True


def test_integrated_simulation_records_planned_and_realized_hydro_flows_and_state():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    assets = load_dispatch_config("configs/assets.yaml")[0]
    simulation = IntegratedSimulation(
        demand,
        SimpleModel(index),
        assets,
        config=IntegratedConfig(demand_scale=0.001, allow_hydro_battery_charging=True),
        detector=NoAlarmDetector(),
    )
    assert "hydro_to_battery_mw" in simulation.plan.schedule
    assert "hydro_to_demand_mw" in simulation.plan.schedule
    planned_hydro_charge = simulation.plan.schedule.hydro_to_battery_mw.iloc[0]
    initial_state = simulation.state

    simulation.step()
    row = simulation.realized[-1]
    assert "hydro_to_battery_mw" in row
    assert "hydro_to_demand_mw" in row
    assert "planned_first_hour_hydro_to_battery_mw" in row
    assert row["planned_first_hour_hydro_to_battery_mw"] == pytest.approx(planned_hydro_charge)
    assert row["hydro_mw"] == pytest.approx(row["hydro_to_demand_mw"] + row["hydro_to_battery_mw"])
    expected_soc = initial_state.battery_soc_mwh + (
        assets.battery.charge_efficiency * row["battery_charge_mw"]
        - row["battery_discharge_mw"] / assets.battery.discharge_efficiency
    ) * assets.interval_hours
    expected_hydro = initial_state.hydro_remaining_mwh - (
        row["hydro_mw"] + row["hydro_spill_mw"]
    ) * assets.interval_hours
    assert row["ending_battery_soc_mwh"] == pytest.approx(expected_soc)
    assert row["ending_hydro_remaining_mwh"] == pytest.approx(expected_hydro)
    assert row["realized_cost_eur"] == pytest.approx(row["cost_total_eur"])
    assert row["ending_battery_soc_mwh"] == pytest.approx(simulation.state.battery_soc_mwh)
    assert row["ending_hydro_remaining_mwh"] == pytest.approx(simulation.state.hydro_remaining_mwh)


def test_integrated_config_rejects_ambiguous_hydro_charging_policy():
    with pytest.raises(ValueError, match="requires renewable_only"):
        IntegratedConfig(battery_charging_policy="unrestricted", allow_hydro_battery_charging=True)


def test_daily_planned_cost_uses_only_current_utc_day_plan():
    simulation = SimpleNamespace(
        timestamp=pd.Timestamp("2025-01-01 12:00", tz="UTC"),
        realized=[
            {"timestamp": pd.Timestamp("2024-12-31 23:00", tz="UTC"), "realized_cost_eur": 90.0},
            {"timestamp": pd.Timestamp("2025-01-01 01:00", tz="UTC"), "realized_cost_eur": 10.0},
            {"timestamp": pd.Timestamp("2025-01-01 11:00", tz="UTC"), "realized_cost_eur": 20.0},
        ],
        plan=SimpleNamespace(schedule=pd.DataFrame(
            {"cost_total_eur": [30.0, 40.0, 50.0]},
            index=pd.date_range("2025-01-01 13:00", periods=3, freq="10h", tz="UTC"),
        )),
    )

    assert daily_planned_cost_eur(simulation) == 70.0


def test_no_future_actual_is_exposed_and_each_step_reveals_one_observation():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    demand = pd.Series(np.arange(len(index), dtype=float) + 40_000, index=index)
    simulation = _simulation(demand)
    issue = simulation.timestamp
    first_target = issue + pd.Timedelta(hours=1)
    assert simulation.history_frame().index.max() == issue
    assert first_target not in simulation.history_frame().index
    simulation.step()
    assert simulation.history_frame().index.max() == first_target
    assert simulation.timestamp == first_target
    assert simulation.forecast.index[0] == first_target + pd.Timedelta(hours=1)


def test_fixed_24_hour_error_matching_and_state_carryover():
    index = pd.date_range("2024-01-01", periods=270, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    simulation = _simulation(demand)
    issue = simulation.timestamp
    simulation.step_n(24)
    matches = [row for row in simulation.errors if row["horizon_hours"] == 24]
    assert matches
    assert any(pd.Timestamp(row["issue_timestamp"]) == issue for row in matches)
    assert any(pd.Timestamp(row["target_timestamp"]) == issue + pd.Timedelta(hours=24) for row in matches)
    assert len(simulation.realized) == 24
    last = simulation.realized[-1]
    assert simulation.state.battery_soc_mwh == last["soc_mwh"]
    assert simulation.state.hydro_remaining_mwh == last["hydro_remaining_mwh"]
    assert simulation.accumulated_realized_cost_eur == sum(r["realized_cost_eur"] for r in simulation.realized)


def test_selected_replay_start_scores_each_new_hour_from_an_eligible_24h_issue():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    demand = pd.Series(np.arange(len(index), dtype=float) + 40_000, index=index)
    assets = load_dispatch_config("configs/assets.yaml")[0]
    model = SimpleModel(index)
    model.train_end_issue_time = index[200]
    simulation = IntegratedSimulation(
        demand, model, assets, config=IntegratedConfig(demand_scale=0.001),
        detector=NoAlarmDetector(), start_timestamp=index[230],
    )

    assert simulation.timestamp == index[230]
    assert simulation.history_frame().index.max() == index[230]
    simulation.step()

    scored = pd.DataFrame(simulation.errors)
    scored = scored.loc[scored.target_timestamp == index[231]]
    assert set(scored.model_policy) == {"static", "adaptive"}
    assert set(pd.to_datetime(scored.issue_timestamp, utc=True)) == {index[207]}
    assert index[207] > model.train_end_issue_time
    assert np.isfinite(simulation.metrics()["static_mae_mw"])


def test_replay_start_cannot_expose_actuals_past_dataset_end():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    simulation = _simulation(demand)
    try:
        simulation.reset(start_timestamp=index[-1] + pd.Timedelta(hours=1))
    except ValueError as exc:
        assert "after the last available actual observation" in str(exc)
    else:
        raise AssertionError("replay must not start beyond available actual demand")


def test_reset_reproduces_initial_plan_and_actual_stream():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    simulation = _simulation(demand)
    initial_plan = simulation.plan.schedule.copy()
    timestamp = simulation.timestamp
    simulation.step_n(3)
    simulation.reset()
    assert simulation.timestamp == timestamp
    pd.testing.assert_frame_equal(simulation.plan.schedule, initial_plan)
    assert len(simulation.realized) == 0


def test_future_actual_resources_do_not_change_already_issued_plan():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    simulation = _simulation(demand)
    plan = simulation.plan.schedule.copy()
    future = simulation._resource_actuals.index > simulation.timestamp
    simulation._resource_actuals.loc[future, ["solar_actual_mw", "wind_actual_mw"]] = 0.0
    pd.testing.assert_frame_equal(simulation.plan.schedule, plan)


def test_dashboard_views_show_only_observed_actuals_and_24_future_values():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    simulation = _simulation(demand)
    view = demand_chart_frame(simulation)
    assert view.actual_demand_mw.notna().sum() == len(simulation.actual_history)
    assert view.predicted_demand_mw.notna().sum() == 24
    assert view.loc[view.index > simulation.timestamp, "actual_demand_mw"].isna().all()
    downloaded = dispatch_download_frame(simulation)
    assert len(downloaded) == 24
    assert pd.DatetimeIndex(downloaded.timestamp).equals(simulation.forecast.index)
    assert len(dashboard_snapshot_figure(simulation).data) >= 9


def test_streaming_chart_frame_keeps_recent_actuals_and_all_future_predictions():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    simulation = _simulation(demand)
    simulation.step_n(3)
    frame = streaming_forecast_frame(simulation, window_hours=48)
    assert frame.index.min() == simulation.timestamp - pd.Timedelta(hours=47)
    assert frame.actual_demand_mw.notna().sum() == 48
    assert frame.predicted_demand_mw.notna().sum() == 24
    assert frame.loc[frame.index > simulation.timestamp, "actual_demand_mw"].isna().all()


def test_streaming_chart_frame_rejects_empty_window():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    simulation = _simulation(pd.Series(np.full(len(index), 45_000.0), index=index))
    try:
        streaming_forecast_frame(simulation, window_hours=0)
    except ValueError as exc:
        assert "window_hours must be positive" in str(exc)
    else:
        raise AssertionError("a non-positive chart history window must be rejected")


def test_live_chart_shows_adaptive_after_static_and_marks_accepted_update():
    index = pd.date_range("2024-01-01", periods=260, freq="1h", tz="UTC")
    simulation = _simulation(pd.Series(np.full(len(index), 45_000.0), index=index))
    target = simulation.timestamp
    simulation.errors = [
        {
            "target_timestamp": target,
            "predicted_demand_mw": 44_000.0,
            "model_policy": "static",
            "model_version": "test-static",
        },
        {
            "target_timestamp": target,
            "predicted_demand_mw": 47_000.0,
            "model_policy": "adaptive",
            "model_version": "test-static.adapt1",
        },
    ]
    simulation.adaptation_events = [SimpleNamespace(
        outcome="adapted",
        drift_timestamp=target,
        previous_model_version="test-static",
        new_model_version="test-static.adapt1",
    )]

    figure = _figure(simulation)
    names = [trace.name for trace in figure.data]
    assert names.index("Adaptive 24h-ahead") < names.index("Static 24h-ahead (frozen)")
    assert "Accepted adaptive update" in names
    adaptive = next(trace for trace in figure.data if trace.name == "Adaptive 24h-ahead")
    assert "test-static.adapt1" in adaptive.customdata


def test_three_shift_demo_uses_unvalidated_mixed_history_fine_tuning():
    demo_config = _replay_config_for_data(
        "data/processed/entsoe_actual_load_DE_LU_demo_drift_3events.csv"
    )
    real_config = _replay_config_for_data(
        "data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv"
    )
    assert demo_config.adaptation_policy == "increase_only"
    assert demo_config.adaptation_strategy == "mixed_history_fine_tune"
    # Stronger demo fine-tune: full learning rate over every recent sample.
    assert demo_config.retraining_window_days == 14
    assert demo_config.fine_tune_learning_rate_factor == 1.0
    assert demo_config.fine_tune_max_updates >= 1_000
    assert demo_config.cooldown_days == 7
    assert demo_config.validation_days == 7
    assert demo_config.min_validation_improvement_fraction == 0.05
    assert demo_config.representative_samples_per_month == 24
    assert real_config.adaptation_policy == "validated"
    # Normal replay keeps the frozen Phase 3.9 fine-tune settings.
    assert real_config.retraining_window_days == 60
    assert real_config.fine_tune_learning_rate_factor == 0.1
    assert real_config.fine_tune_max_updates == 8
    assert real_config.cooldown_days == 7


def test_demo_xgboost_retrains_on_recent_window_without_gate_but_real_data_does_not():
    from scripts.streamlit_app import _xgboost_config_for_data

    demo = _xgboost_config_for_data("data/processed/entsoe_actual_load_DE_LU_demo_drift_3events_feb.csv")
    real = _xgboost_config_for_data(
        "data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv"
    )
    assert (demo.strategy, demo.recent_window_days, demo.require_validation) == ("recent", 60, False)
    assert demo.cooldown_days == 7
    assert (real.strategy, real.require_validation, real.cooldown_days) == ("full_history", True, 7)


def test_adaptive_model_is_not_changed_without_an_adwin_alarm():
    index = pd.date_range("2024-01-01", periods=270, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    assets = load_dispatch_config("configs/assets.yaml")[0]
    simulation = IntegratedSimulation(
        demand,
        SimpleModel(index),
        assets,
        config=IntegratedConfig(policy="adaptive"),
        detector=NoAlarmDetector(),
    )
    start_version = simulation.active_model.model_version
    simulation.step_n(30)
    assert simulation.active_model.model_version == start_version
    assert not simulation.adaptation_events


def test_alarm_is_logged_but_an_unqualified_change_does_not_adapt():
    index = pd.date_range("2024-01-01", periods=270, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    assets = load_dispatch_config("configs/assets.yaml")[0]
    simulation = IntegratedSimulation(
        demand,
        SimpleModel(index),
        assets,
        config=IntegratedConfig(policy="adaptive"),
        detector=AlarmDetector(),
    )
    start_version = simulation.active_model.model_version
    simulation.step_n(24)
    assert len(simulation.drift_events) == 1
    assert simulation.active_model.model_version == start_version
    assert simulation.adaptation_events[0].outcome == "error_increase_not_meaningful"


def test_adaptive_comparison_updates_in_parallel_when_static_drives_dispatch():
    index = pd.date_range("2024-01-01", periods=270, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    assets = load_dispatch_config("configs/assets.yaml")[0]
    simulation = IntegratedSimulation(
        demand,
        SimpleModel(index),
        assets,
        config=IntegratedConfig(policy="static"),
        detector=AlarmDetector(),
    )

    simulation.step_n(24)

    assert simulation.config.policy == "static"
    assert len(simulation.drift_events) == 1
    assert simulation.adaptation_events[0].outcome == "error_increase_not_meaningful"
    assert simulation.active_model is simulation.static_model


class IncreaseAlarmDetector:
    """Raise one alarm whose ADWIN cut kept the last 10 hourly errors."""

    def __init__(self):
        self.called = False
        self.last_change = None

    def update(self, value):
        from griddrift.drift.adwin import ErrorDistributionChange

        if self.called:
            self.last_change = None
            return False
        self.called = True
        self.last_change = ErrorDistributionChange(
            error_mean_before_mw=1_000.0, error_mean_after_mw=1_100.0,
            window_width_before=50, window_width_after=10,
        )
        return True


def test_drift_timeline_dates_the_drift_range_and_explains_the_decision():
    from griddrift.dashboard.views import drift_timeline_frame

    index = pd.date_range("2024-01-01", periods=270, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    assets = load_dispatch_config("configs/assets.yaml")[0]
    simulation = IntegratedSimulation(
        demand, SimpleModel(index), assets,
        config=IntegratedConfig(policy="adaptive"), detector=IncreaseAlarmDetector(),
    )
    simulation.step_n(24)

    timeline = drift_timeline_frame(simulation)
    assert len(timeline) == 1
    row = timeline.iloc[0]
    assert row.onset_timestamp == row.alarm_timestamp - pd.Timedelta(hours=9)
    assert row.error_increased
    assert row.error_change_fraction == pytest.approx(0.10)
    # +100 MW is below the 250 MW trigger, so no candidate is trained.
    assert row.decision_kind == "too_small"
    assert not row.retraining_attempted
    assert "250 MW" in row.decision
    _figure(simulation)  # overlays must render without error


def test_delayed_retrain_collects_post_drift_data_then_adapts_once():
    from griddrift.dashboard.views import drift_timeline_frame

    class BigIncreaseAlarm(IncreaseAlarmDetector):
        def update(self, value):
            fired = super().update(value)
            if fired:  # +1,000 MW and +100%: passes the 250 MW / 10% trigger
                from griddrift.drift.adwin import ErrorDistributionChange

                self.last_change = ErrorDistributionChange(1_000.0, 2_000.0, 50, 10)
            return fired

    index = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    demand = pd.Series(np.full(len(index), 45_000.0), index=index)
    assets = load_dispatch_config("configs/assets.yaml")[0]
    simulation = IntegratedSimulation(
        demand, SimpleModel(index), assets,
        config=IntegratedConfig(policy="adaptive", retrain_delay_hours=6), detector=BigIncreaseAlarm(),
    )
    calls = []
    simulation.replay._adapt = lambda timestamp, change: calls.append(timestamp)

    while not simulation.drift_events:
        simulation.step()
    alarm = simulation.drift_events[0]["drift_timestamp"]
    assert simulation.drift_events[0]["retraining_scheduled_for"] == alarm + pd.Timedelta(hours=6)
    assert simulation.pending_retrain is not None and calls == []
    assert drift_timeline_frame(simulation).iloc[0].decision_kind == "retrain_scheduled"
    _figure(simulation)

    simulation.step_n(5)
    assert calls == []  # still collecting post-drift data
    simulation.step()
    assert calls == [alarm + pd.Timedelta(hours=6)]
    assert simulation.pending_retrain is None
    simulation.step_n(10)
    assert len(calls) == 1


def _recourse_simulation(**config):
    index = pd.date_range("2024-01-01", periods=300, freq="1h", tz="UTC")
    demand = pd.Series(45_000.0 + 3_000.0 * np.sin(np.arange(len(index)) / 4), index=index)
    assets = load_dispatch_config("configs/assets.yaml")[0]
    return IntegratedSimulation(
        demand, SimpleModel(index), assets,
        config=IntegratedConfig(policy="adaptive", **config), detector=NoAlarmDetector(),
    )


def test_realized_hour_is_committed_to_the_plan_first_hour_by_default():
    simulation = _recourse_simulation()
    assert simulation.config.realization == "committed_recourse"
    planned = simulation.plan.schedule.iloc[0]
    simulation.step()
    realized = simulation.realized[-1]
    # Least-change recourse records how far real time moved from the committed plan.
    assert realized["commitment_deviation_mw"] >= 0.0
    supplied = sum(realized[key] for key in (
        "solar_mw", "wind_mw", "hydro_mw", "conventional_mw",
        "battery_discharge_mw", "emergency_mw", "unserved_mw",
    ))
    assert supplied == pytest.approx(realized["demand_mw"] + realized["battery_charge_mw"], abs=1e-6)
    assert realized["planned_first_hour_cost_eur"] == pytest.approx(float(planned.cost_total_eur))


def test_counterfactual_static_system_is_operated_alongside_the_live_one():
    simulation = _recourse_simulation(track_counterfactual=True)
    assert simulation.counterfactual_policy == "static"
    simulation.step_n(6)
    assert len(simulation.counterfactual_realized) == 6
    assert {row["policy"] for row in simulation.counterfactual_realized} == {"static"}
    assert simulation.counterfactual_cost_eur == pytest.approx(
        sum(row["realized_cost_eur"] for row in simulation.counterfactual_realized)
    )
    # Identical forecasts (no retraining) must give identical realized costs.
    assert simulation.counterfactual_cost_eur == pytest.approx(simulation.accumulated_realized_cost_eur)


def test_perfect_hindsight_mode_ignores_the_plan():
    simulation = _recourse_simulation(realization="perfect_hindsight")
    simulation.step()
    assert "commitment_deviation_mw" not in simulation.realized[-1] or (
        simulation.realized[-1]["commitment_deviation_mw"] == 0.0
    )


def test_peak_off_peak_conventional_rates_reach_plan_and_realized_hour():
    blocks = (70.0, 90.0, 115.0, 100.0, 120.0, 75.0)
    simulation = _recourse_simulation(conventional_cost_blocks_eur_mwh=blocks)
    plan = simulation.plan.schedule
    expected = np.asarray(blocks)[pd.DatetimeIndex(plan.index).hour // 4]
    np.testing.assert_allclose(plan.conventional_cost_rate_eur_mwh.to_numpy(), expected)
    simulation.step()
    realized = simulation.realized[-1]
    assert realized["conventional_cost_rate_eur_mwh"] == blocks[realized["timestamp"].hour // 4]


def test_conventional_cost_blocks_default_to_flat_asset_rate_and_validate():
    simulation = _recourse_simulation()
    flat = simulation.assets.conventional.marginal_cost_eur_mwh
    assert (simulation.plan.schedule.conventional_cost_rate_eur_mwh == flat).all()
    with pytest.raises(ValueError):
        IntegratedConfig(conventional_cost_blocks_eur_mwh=(70.0, 90.0))


def test_hydro_price_blocks_reach_the_plan():
    blocks = (16.0, 20.0, 26.0, 22.0, 28.0, 20.0)
    simulation = _recourse_simulation(hydro_cost_blocks_eur_mwh=blocks)
    plan = simulation.plan.schedule
    expected = np.asarray(blocks)[pd.DatetimeIndex(plan.index).hour // 4]
    np.testing.assert_allclose(plan.hydro_cost_rate_eur_mwh.to_numpy(), expected)
    assert load_integrated_settings("configs/integration.yaml")["hydro_cost_blocks_eur_mwh"] == blocks
