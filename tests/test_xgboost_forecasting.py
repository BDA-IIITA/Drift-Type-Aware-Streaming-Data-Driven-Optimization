import numpy as np
import pandas as pd
import pytest

from griddrift.drift import ErrorDistributionChange
from griddrift.forecasting.features import DirectFeatureSpec, build_feature_row
from griddrift.forecasting.xgboost import DirectXGBoostForecaster, train_direct_xgboost
from griddrift.simulation.xgboost_replay import run_xgboost_replay


@pytest.fixture
def demand():
    index = pd.date_range("2023-01-01", periods=600, freq="h", tz="UTC")
    values = 50_000 + 4_000 * np.sin(np.arange(len(index)) * 2 * np.pi / 24)
    return pd.Series(values, index=index, name="demand_mw")


def test_xgboost_prediction_has_aligned_24_hour_horizon(demand):
    spec = DirectFeatureSpec()
    history = demand.iloc[:300]
    issue = history.index[-1]
    result = train_direct_xgboost(
        demand.iloc[:500], feature_spec=spec, train_end_issue_time=demand.index[350],
        parameters={"n_estimators": 3, "max_depth": 2, "n_jobs": 1},
    )
    prediction = result.model.predict_horizon(history, issue)
    assert len(prediction) == 24
    assert prediction.index.equals(pd.date_range(issue + pd.Timedelta(hours=1), periods=24, freq="h"))
    assert (prediction >= 0).all()


def test_features_ignore_values_after_issue_time(demand):
    spec = DirectFeatureSpec()
    issue = demand.index[300]
    history = demand.loc[:issue]
    extended = demand.copy()
    extended.loc[extended.index > issue] = 999_999
    assert build_feature_row(history, issue, spec) == build_feature_row(extended, issue, spec)


def test_checkpoint_persists_model_and_version(demand, tmp_path):
    result = train_direct_xgboost(
        demand.iloc[:500], train_end_issue_time=demand.index[350],
        parameters={"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
    )
    result.model.save(tmp_path, result.validation_metrics)
    restored = DirectXGBoostForecaster.load(tmp_path)
    issue = demand.index[400]
    pd.testing.assert_series_equal(
        restored.predict_horizon(demand.iloc[:401], issue),
        result.model.predict_horizon(demand.iloc[:401], issue),
    )
    assert restored.model_version == result.model.model_version
    repeat = train_direct_xgboost(
        demand.iloc[:500], train_end_issue_time=demand.index[350],
        parameters={"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
    )
    pd.testing.assert_series_equal(
        repeat.model.predict_horizon(demand.iloc[:401], issue),
        result.model.predict_horizon(demand.iloc[:401], issue),
    )


def test_training_respects_issue_cutoff_and_validation_embargo(demand):
    result = train_direct_xgboost(
        demand.iloc[:550], train_end_issue_time=demand.index[400], validation_fraction=0.2,
        parameters={"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
    )
    assert result.model.train_end_issue_time <= demand.index[400]
    assert result.train_samples > result.validation_samples
    assert result.validation_train_end_issue_time + pd.Timedelta(hours=24) < result.validation_first_issue_time
    assert result.validation_metrics["mae_mw"] >= 0


def test_replay_uses_matching_24_hour_issue_and_does_not_adapt_without_alarm(demand):
    class QuietDetector:
        last_change = None

        def update(self, absolute_error_mw):
            return False

    model = train_direct_xgboost(
        demand.iloc[:500], train_end_issue_time=demand.index[350],
        parameters={"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
    ).model
    replay = run_xgboost_replay(
        demand.iloc[:431], model, start=demand.index[400], end=demand.index[430], detector=QuietDetector(),
    )
    replay_repeat = run_xgboost_replay(
        demand.iloc[:431], model, start=demand.index[400], end=demand.index[430], detector=QuietDetector(),
    )
    assert replay.adaptation_events.empty
    assert replay.final_model.model_version == model.model_version
    assert not replay.adaptive_errors.empty
    difference = (
        pd.to_datetime(replay.adaptive_errors["target_timestamp"], utc=True)
        - pd.to_datetime(replay.adaptive_errors["issue_timestamp"], utc=True)
    )
    assert (difference == pd.Timedelta(hours=24)).all()
    assert np.allclose(replay.static_errors["actual_demand_mw"], replay.adaptive_errors["actual_demand_mw"])
    assert len(replay.adaptive_registry.to_frame()) == 31 * 24
    pd.testing.assert_frame_equal(replay.adaptive_errors, replay_repeat.adaptive_errors)


def test_alarm_candidate_is_validated_and_cooldown_is_enforced(demand, monkeypatch):
    from griddrift.simulation import xgboost_replay
    from griddrift.simulation.xgboost_replay import XGBoostReplayConfig

    class AlarmEveryTime:
        last_change = ErrorDistributionChange(1_000.0, 3_000.0, 300, 150)

        def update(self, absolute_error_mw):
            return True

    class RejectedCandidate:
        model_version = "candidate"

        def predict_feature_frame(self, features):
            return np.zeros((len(features), 24))

    monkeypatch.setattr(xgboost_replay, "fit_direct_xgboost_samples", lambda *a, **k: RejectedCandidate())
    model = train_direct_xgboost(
        demand.iloc[:500], train_end_issue_time=demand.index[350],
        parameters={"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
    ).model
    result = run_xgboost_replay(
        demand, model, start=demand.index[500], end=demand.index[526],
        config=XGBoostReplayConfig(strategy="recent", recent_window_days=60,
                                   min_validation_improvement_fraction=1.1),
        detector=AlarmEveryTime(),
    )
    decisions = result.adaptation_events["decision"].tolist()
    assert decisions[0] == "rejected_validation"
    assert "skipped_cooldown" in decisions
    assert result.final_model.model_version == model.model_version
    event = result.adaptation_events.iloc[0]
    assert pd.Timestamp(event.training_latest_target_timestamp) <= pd.Timestamp(event.drift_timestamp)
    assert pd.Timestamp(event.validation_last_target_timestamp) <= pd.Timestamp(event.drift_timestamp)


def test_validated_candidate_can_be_accepted_and_version_changes(demand, monkeypatch):
    from griddrift.simulation import xgboost_replay
    from griddrift.simulation.xgboost_replay import XGBoostReplayConfig

    class AlarmOnce:
        last_change = ErrorDistributionChange(1_000.0, 3_000.0, 300, 150)

        def __init__(self):
            self.used = False

        def update(self, absolute_error_mw):
            if self.used:
                return False
            self.used = True
            return True

    class TestCandidate:
        model_version = "test-candidate"

        def predict_feature_frame(self, features):
            # Test double only: oracle outputs force the promotion branch to be covered.
            result = np.zeros((len(features), 24))
            for i, issue in enumerate(features.index):
                result[i] = demand.loc[issue + pd.Timedelta(hours=1):issue + pd.Timedelta(hours=24)].to_numpy()
            return result

        def predict_horizon(self, history, issue_time):
            targets = pd.date_range(issue_time + pd.Timedelta(hours=1), periods=24, freq="h")
            return pd.Series(demand.loc[targets].to_numpy(), index=targets, name="predicted_demand_mw")

    monkeypatch.setattr(xgboost_replay, "fit_direct_xgboost_samples", lambda *a, **k: TestCandidate())
    model = train_direct_xgboost(
        demand.iloc[:500], train_end_issue_time=demand.index[350],
        parameters={"n_estimators": 2, "max_depth": 2, "n_jobs": 1},
    ).model
    result = run_xgboost_replay(
        demand, model, start=demand.index[500], end=demand.index[526],
        config=XGBoostReplayConfig(strategy="recent", recent_window_days=60), detector=AlarmOnce(),
    )
    assert result.adaptation_events.iloc[0]["decision"] == "accepted"
    assert result.adaptation_events.iloc[0]["new_model_version"] == "test-candidate"
    assert result.final_model.model_version == "test-candidate"
