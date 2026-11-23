import numpy as np
import pandas as pd
import pytest

from griddrift.data.synthetic import make_synthetic_scenario
from griddrift.drift import ErrorDistributionChange
from griddrift.forecasting.features import build_supervised_dataset
from griddrift.forecasting.mlp import train_direct_mlp
from griddrift.simulation import ReplayConfig, StreamingReplay


class SequenceDetector:
    """Deterministic stand-in used to verify replay sequencing, not ADWIN itself."""

    def __init__(self, alarms=(), change=None):
        self.alarms = list(alarms)
        self.values = []
        self.last_change = change

    def update(self, value):
        self.values.append(value)
        return self.alarms.pop(0) if self.alarms else False


@pytest.fixture(scope="module")
def replay_inputs():
    demand, _ = make_synthetic_scenario(periods=24 * 94, seed=17)
    history = demand["demand_mw"]
    trained = train_direct_mlp(
        history.iloc[: 24 * 72], hidden_layer_sizes=(6,), max_iter=35, random_state=4
    )
    return history, trained.model


def _end(model, hours=40):
    return model.train_end_issue_time + pd.Timedelta(hours=24 + hours)


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_replay_matches_only_the_fixed_24_hour_forecast(replay_inputs):
    demand, model = replay_inputs
    replay = StreamingReplay(demand, model, detector=SequenceDetector())
    result = replay.run(end_timestamp=_end(model))
    assert not result.adaptive_errors.empty
    assert (result.adaptive_errors["horizon_hours"] == 24).all()
    assert (
        pd.to_datetime(result.adaptive_errors["target_timestamp"], utc=True)
        == pd.to_datetime(result.adaptive_errors["issue_timestamp"], utc=True) + pd.Timedelta(hours=24)
    ).all()
    assert (pd.to_datetime(result.adaptive_errors["target_timestamp"], utc=True) <= _end(model)).all()


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_no_retraining_without_detector_alarm(replay_inputs):
    demand, model = replay_inputs
    result = StreamingReplay(demand, model, detector=SequenceDetector()).run(end_timestamp=_end(model))
    assert result.adaptation_count == 0
    assert result.final_adaptive_model.model_version == model.model_version


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_adaptation_uses_only_fully_observed_targets(replay_inputs):
    demand, model = replay_inputs
    # First evaluated 24-hour error triggers retraining; prior data are all that is visible then.
    result = StreamingReplay(
        demand,
        model,
        config=ReplayConfig(adaptation_max_iter=25),
        detector=SequenceDetector([True]),
    ).run(end_timestamp=_end(model))
    event = next(event for event in result.adaptation_events if event.outcome == "adapted")
    assert event.training_window_end <= event.drift_timestamp
    assert event.latest_training_target_timestamp <= event.drift_timestamp
    assert event.new_model_version != event.previous_model_version


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_cooldown_blocks_repeated_detected_drifts(replay_inputs):
    demand, model = replay_inputs
    result = StreamingReplay(
        demand,
        model,
        config=ReplayConfig(cooldown_days=7, adaptation_max_iter=25),
        detector=SequenceDetector([True] * 60),
    ).run(end_timestamp=_end(model, hours=38))
    assert result.adaptation_count == 1
    assert any(event.outcome == "cooldown_skipped" for event in result.adaptation_events)
    assert result.drift_count == len(result.adaptation_events)


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_replay_reset_is_reproducible(replay_inputs):
    demand, model = replay_inputs
    replay = StreamingReplay(demand, model, detector=SequenceDetector())
    first = replay.run(end_timestamp=_end(model, hours=30))
    replay.reset()
    second = replay.run(end_timestamp=_end(model, hours=30))
    pd.testing.assert_frame_equal(first.adaptive_errors, second.adaptive_errors)
    assert first.final_adaptive_model.model_version == second.final_adaptive_model.model_version


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_validated_candidate_uses_observed_targets_and_can_be_rejected(replay_inputs):
    demand, model = replay_inputs
    increase = ErrorDistributionChange(1000.0, 1400.0, 200, 40)
    replay = StreamingReplay(
        demand,
        model,
        config=ReplayConfig(
            adaptation_policy="validated",
            adaptation_strategy="fine_tune",
            retraining_window_days=30,
            fine_tune_epochs=2,
            min_validation_improvement_fraction=0.99,
        ),
        detector=SequenceDetector([True], change=increase),
    )
    result = replay.run(end_timestamp=_end(model, hours=25))
    event = next(event for event in result.adaptation_events if event.outcome == "candidate_rejected")
    assert event.latest_training_target_timestamp <= event.drift_timestamp
    assert event.training_window_end < event.drift_timestamp
    assert event.latest_training_target_timestamp < event.validation_first_target_timestamp
    assert event.validation_last_target_timestamp <= event.drift_timestamp
    assert event.candidate_validation_mae_mw is not None
    assert event.incumbent_validation_mae_mw is not None
    assert event.new_model_version is None
    assert result.final_adaptive_model.model_version == model.model_version
    for before, after in zip(model.model.coefs_, result.final_adaptive_model.model.coefs_):
        np.testing.assert_array_equal(before, after)


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_gentle_finetune_continues_weights_optimizer_and_scalers(replay_inputs):
    demand, model = replay_inputs
    timestamp = demand.index[-1]
    replay = StreamingReplay(
        demand,
        model,
        config=ReplayConfig(
            adaptation_strategy="gentle_fine_tune",
            retraining_window_days=14,
            validation_days=7,
            fine_tune_learning_rate_factor=0.1,
            fine_tune_max_updates=3,
            fine_tune_batch_size=64,
        ),
        detector=SequenceDetector(),
    )
    original_weights = [weight.copy() for weight in model.model.coefs_]
    original_steps = model.model.t_
    original_lr = float(model.model._optimizer.learning_rate)
    candidate, X_validation, _, _, _, _, _, training_seconds = replay._fit_validated_candidate(
        timestamp, replay.adaptive_model, f"{model.model_version}.adapt1"
    )

    assert candidate.model is not model.model
    assert candidate.feature_scaler is replay.adaptive_model.feature_scaler
    assert candidate.target_scaler is replay.adaptive_model.target_scaler
    assert candidate.model.t_ > original_steps
    assert candidate.model._optimizer.learning_rate < original_lr
    assert candidate.model.n_iter_ - model.model.n_iter_ <= 3
    assert candidate.eligible_window_end + pd.Timedelta(hours=24) <= timestamp
    assert X_validation.index[-1] + pd.Timedelta(hours=24) <= timestamp
    assert training_seconds >= 0
    assert any(not np.array_equal(before, after) for before, after in zip(original_weights, candidate.model.coefs_))
    for before, after in zip(original_weights, model.model.coefs_):
        np.testing.assert_array_equal(before, after)


@pytest.mark.filterwarnings("ignore:Stochastic Optimizer:sklearn.exceptions.ConvergenceWarning")
def test_mixed_history_finetuning_includes_older_samples_and_two_validation_periods(replay_inputs):
    demand, model = replay_inputs
    timestamp = demand.index[-1]
    replay = StreamingReplay(
        demand,
        model,
        config=ReplayConfig(
            adaptation_strategy="mixed_history_fine_tune",
            retraining_window_days=14,
            validation_days=3,
            validation_periods=2,
            fine_tune_max_updates=2,
            fine_tune_batch_size=64,
        ),
        detector=SequenceDetector(),
    )
    candidate, X_validation, y_validation, *_ = replay._fit_validated_candidate(
        timestamp, replay.adaptive_model, f"{model.model_version}.adapt1"
    )

    assert len(X_validation) == 24 * 6
    first_validation_issue = X_validation.index.min()
    assert candidate.eligible_window_end + pd.Timedelta(hours=24) <= first_validation_issue
    recent_start = candidate.eligible_window_end - pd.Timedelta(days=14)
    assert candidate.eligible_window_start < recent_start
    assert y_validation.index.equals(X_validation.index)


def test_selected_adaptation_samples_match_canonical_feature_pipeline(replay_inputs):
    demand, model = replay_inputs
    full_X, full_y = build_supervised_dataset(demand, model.feature_spec)
    selected = full_X.index[-12:]
    selected_X, selected_y = StreamingReplay._build_selected_samples(demand, selected, model)

    pd.testing.assert_frame_equal(selected_X, full_X.loc[selected])
    pd.testing.assert_frame_equal(selected_y, full_y.loc[selected])
