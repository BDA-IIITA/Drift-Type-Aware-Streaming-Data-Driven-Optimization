import numpy as np
import pandas as pd

from griddrift.dashboard.model_selection import (
    ADAPTIVE_MLP,
    ADAPTIVE_XGBOOST,
    load_selected_adaptation_config,
)
from griddrift.dispatch import DispatchState, load_dispatch_config
from griddrift.drift import ErrorDistributionChange
from griddrift.forecasting.features import DirectFeatureSpec
from griddrift.forecasting.xgboost import DirectXGBoostForecaster
from griddrift.simulation import integrated, xgboost_replay
from griddrift.simulation.integrated import IntegratedConfig, IntegratedSimulation
from griddrift.simulation.xgboost_replay import XGBoostReplayConfig, XGBoostStreamingReplay


class _BaseModel(DirectXGBoostForecaster):
    def predict_feature_frame(self, features):
        return np.full((len(features), 24), 100.0)


class _Candidate:
    model_version = "xgb-test-v1.adapt1"

    def predict_feature_frame(self, features):
        return np.zeros((len(features), 24))


class _WorseCandidate:
    model_version = "xgb-test-v1.adapt1"

    def predict_feature_frame(self, features):
        return np.full((len(features), 24), 5_000.0)


class _QuietDetector:
    last_change = None

    def update(self, error):
        return False


class _AlarmDetector:
    last_change = ErrorDistributionChange(100.0, 1_000.0, 400, 100)

    def update(self, error):
        return True


class _CountingQuietDetector:
    last_change = None

    def __init__(self):
        self.count = 0

    def update(self, error):
        self.count += 1
        return False


def _model(train_end):
    return _BaseModel(
        models=[], feature_spec=DirectFeatureSpec(), model_version="xgb-test-v1",
        feature_columns=[], train_end_issue_time=train_end, parameters={}, multi_output=True,
    )


def test_runtime_adaptation_configs_match_frozen_phase_selections():
    mlp = load_selected_adaptation_config(ADAPTIVE_MLP, ".")
    xgb = load_selected_adaptation_config(ADAPTIVE_XGBOOST, ".")
    assert mlp.adaptation_strategy == "mixed_history_fine_tune"
    assert mlp.retraining_window_days == 60
    assert mlp.min_validation_improvement_fraction == 0.05
    assert mlp.cooldown_days == 7
    assert xgb.strategy == "full_history"
    assert xgb.min_validation_improvement_fraction == 0.005
    assert xgb.cooldown_days == 7
    assert (xgb.adwin.delta, xgb.adwin.clock, xgb.adwin.min_window_length) == (0.002, 32, 5)


def test_xgboost_never_adapts_without_an_adwin_alarm():
    index = pd.date_range("2024-01-01", periods=800, freq="h", tz="UTC")
    demand = pd.Series(0.0, index=index)
    model = _model(index[100])
    replay = XGBoostStreamingReplay(model, config=XGBoostReplayConfig(strategy="full_history"), detector=_QuietDetector())
    assert replay.observe_error(index[700], 500.0, demand.loc[:index[700]]) is False
    assert replay.adaptive_model.model_version == model.model_version
    assert replay.events == []


def test_xgboost_candidate_uses_only_observed_targets_and_cooldown(monkeypatch):
    index = pd.date_range("2024-01-01", periods=900, freq="h", tz="UTC")
    demand = pd.Series(0.0, index=index)
    timestamp = index[800]
    model = _model(index[100])
    captured = {}

    def fit_candidate(X, y, **kwargs):
        captured["train_index"] = X.index.copy()
        captured["targets"] = y.copy()
        captured["version"] = kwargs["model_version"]
        return _Candidate()

    monkeypatch.setattr(xgboost_replay, "fit_direct_xgboost_samples", fit_candidate)
    replay = XGBoostStreamingReplay(
        model,
        config=XGBoostReplayConfig(
            strategy="full_history", min_validation_improvement_fraction=0.005,
        ),
        detector=_AlarmDetector(),
    )

    assert replay.observe_error(timestamp, 1_000.0, demand.loc[:timestamp]) is True
    event = replay.events[0]
    assert event.outcome == "adapted"
    assert replay.adaptive_model.model_version == "xgb-test-v1.adapt1"
    assert captured["version"] == "xgb-test-v1.adapt1"
    assert captured["train_index"][-1] + pd.Timedelta(hours=24) <= timestamp
    assert event.latest_training_target_timestamp <= timestamp
    assert event.validation_last_target_timestamp <= timestamp

    cooldown_timestamp = timestamp + pd.Timedelta(hours=1)
    assert replay.observe_error(cooldown_timestamp, 1_000.0, demand.loc[:cooldown_timestamp]) is True
    assert replay.events[-1].outcome == "cooldown_skipped"
    assert replay.adaptive_model.model_version == "xgb-test-v1.adapt1"


def test_rejected_xgboost_candidate_does_not_replace_deployed_model(monkeypatch):
    index = pd.date_range("2024-01-01", periods=900, freq="h", tz="UTC")
    demand = pd.Series(0.0, index=index)
    monkeypatch.setattr(xgboost_replay, "fit_direct_xgboost_samples", lambda *args, **kwargs: _WorseCandidate())
    replay = XGBoostStreamingReplay(
        _model(index[100]),
        config=XGBoostReplayConfig(strategy="full_history", min_validation_improvement_fraction=0.005),
        detector=_AlarmDetector(),
    )
    assert replay.observe_error(index[800], 1_000.0, demand.loc[:index[800]]) is True
    assert replay.events[-1].outcome == "candidate_rejected"
    assert replay.adaptive_model.model_version == "xgb-test-v1"


def test_integrated_replay_scores_only_matching_24_hour_error_then_updates_adwin(monkeypatch):
    index = pd.date_range("2024-01-01", periods=900, freq="h", tz="UTC")
    demand = pd.Series(10_000.0 + np.arange(len(index)), index=index, name="demand_mw")
    assets, _ = load_dispatch_config("configs/assets.yaml")
    detector = _CountingQuietDetector()

    class FakeDispatch:
        total_cost_eur = 0.0
        schedule = pd.DataFrame([{
            "cost_total_eur": 0.0, "hydro_to_demand_mw": 0.0,
            "hydro_to_battery_mw": 0.0, "hydro_spill_mw": 0.0,
        }])
        first_hour_state = DispatchState(15.0, 120.0, 0.0)

    class TimelineModel(_BaseModel):
        def predict_horizon(self, history, issue_time):
            targets = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=24, freq="h")
            base = float(history.loc[:issue_time].iloc[-1])
            return pd.Series(base + np.arange(24), index=targets)

    monkeypatch.setattr(integrated, "optimize_dispatch", lambda *args, **kwargs: FakeDispatch())
    simulation = IntegratedSimulation(
        demand, TimelineModel(
            models=[], feature_spec=DirectFeatureSpec(), model_version="xgb-test-v1",
            feature_columns=[], train_end_issue_time=index[100], parameters={}, multi_output=True,
        ), assets,
        # The fake dispatch has no setpoint columns to commit; this test is about scoring.
        config=IntegratedConfig(policy="adaptive", realization="perfect_hindsight"),
        detector=detector, start_timestamp=index[300], model_family="xgboost",
    )
    result = simulation.step()
    assert result["timestamp"] == index[301]
    assert result["drift"] is False
    assert simulation.detector.count == 1
    matched = [row for row in simulation.errors if row["model_policy"] == "adaptive"]
    assert len(matched) == 1
    assert matched[0]["target_timestamp"] == index[301]
    assert matched[0]["issue_timestamp"] == index[277]
    assert matched[0]["target_timestamp"] - matched[0]["issue_timestamp"] == pd.Timedelta(hours=24)
    assert simulation.adaptation_events == []
    assert simulation.active_model.model_version == "xgb-test-v1"


def test_demo_xgboost_retrains_on_recent_window_including_newest_days(monkeypatch):
    index = pd.date_range("2024-01-01", periods=2_400, freq="h", tz="UTC")
    demand = pd.Series(0.0, index=index)
    timestamp = index[-1]
    captured = {}

    def fit_candidate(X, y, **kwargs):
        captured["train_index"] = X.index.copy()
        return _Candidate()

    monkeypatch.setattr(xgboost_replay, "fit_direct_xgboost_samples", fit_candidate)
    replay = XGBoostStreamingReplay(
        _model(index[100]),
        config=XGBoostReplayConfig(strategy="recent", recent_window_days=30, require_validation=False),
        detector=_AlarmDetector(),
    )
    assert replay.observe_error(timestamp, 1_000.0, demand) is True

    event = replay.events[0]
    train = captured["train_index"]
    assert event.outcome == "adapted"
    assert event.candidate_validation_mae_mw is None  # no hold-out gate in demo mode
    assert event.training_strategy == "recent_unvalidated"
    assert replay.adaptive_model.model_version == "xgb-test-v1.adapt1"
    # Newest fully observed issue is included; nothing older than the window.
    assert train[-1] + pd.Timedelta(hours=24) == timestamp
    assert train[0] + pd.Timedelta(hours=24) > timestamp - pd.Timedelta(days=30)


def test_observe_error_can_defer_adaptation_to_the_caller(monkeypatch):
    index = pd.date_range("2024-01-01", periods=900, freq="h", tz="UTC")
    monkeypatch.setattr(xgboost_replay, "fit_direct_xgboost_samples", lambda *a, **k: _Candidate())
    replay = XGBoostStreamingReplay(_model(index[100]), detector=_AlarmDetector())
    assert replay.observe_error(index[800], 1_000.0, pd.Series(0.0, index=index), adapt=False) is True
    assert replay.events == []
    assert replay.is_meaningful_increase(_AlarmDetector.last_change)
