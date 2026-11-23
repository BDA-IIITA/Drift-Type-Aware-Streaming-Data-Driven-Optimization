import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from griddrift.config.schema import DriftConfig
from griddrift.drift import ErrorDistributionChange
from griddrift.simulation import ReplayConfig, StreamingReplay
from griddrift.simulation.threshold_experiments import load_threshold_replay_experiments


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_CONFIG = ROOT / "artifacts/experiments/phase39/threshold_experiment_config.json"
EXPECTED_THRESHOLDS = (0.0, 0.005, 0.01, 0.02, 0.05)


class StatefulDetector:
    def __init__(self):
        self.calls = 0

    def update(self, value):
        self.calls += 1
        return False


def _threshold_config(threshold: float) -> ReplayConfig:
    experiments = load_threshold_replay_experiments(EXPERIMENT_CONFIG)
    return next(
        experiment.replay_config
        for experiment in experiments
        if experiment.min_validation_improvement_fraction == threshold
    )


def _make_replay(threshold: float, candidate_mae: float, incumbent_mae: float = 100.0):
    index = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    demand = pd.Series(np.linspace(1000.0, 1100.0, len(index)), index=index)
    static = SimpleNamespace(model_version="seed-model")
    detector = StatefulDetector()
    replay = StreamingReplay(
        demand,
        static,
        config=_threshold_config(threshold),
        detector=detector,
    )
    issue = index[-1]
    candidate = copy.deepcopy(replay.adaptive_model)
    candidate.model_version = "seed-model.adapt1"
    x_validation = pd.DataFrame({"feature": [1.0]}, index=[issue - pd.Timedelta(hours=2)])
    y_validation = pd.DataFrame({"last_target": [0.0]}, index=x_validation.index)
    train_end = issue - pd.Timedelta(hours=48)
    replay._fit_validated_candidate = lambda ts, old, version: (
        candidate,
        x_validation,
        y_validation,
        train_end - pd.Timedelta(days=1),
        train_end,
        issue - pd.Timedelta(hours=3),
        issue - pd.Timedelta(hours=1),
        0.01,
    )
    replay._validation_predictions = lambda model, X: np.array(
        [candidate_mae if model.model_version.endswith("adapt1") else incumbent_mae]
    )
    return replay, issue, detector


@pytest.mark.parametrize(
    ("threshold", "candidate_mae", "expected_outcome"),
    [
        (0.005, 99.5, "adapted"),
        (0.01, 99.5, "candidate_rejected"),
    ],
)
def test_validation_threshold_accepts_or_rejects_at_configured_boundary(
    threshold, candidate_mae, expected_outcome
):
    replay, issue, _ = _make_replay(threshold, candidate_mae)
    replay._adapt(issue, ErrorDistributionChange(1000.0, 1400.0, 200, 40))

    assert replay.events[-1].outcome == expected_outcome
    assert replay.config.min_validation_improvement_fraction == threshold


def test_zero_threshold_accepts_exact_zero_improvement():
    replay, issue, _ = _make_replay(0.0, candidate_mae=100.0)
    replay._adapt(issue, ErrorDistributionChange(1000.0, 1400.0, 200, 40))

    assert replay.events[-1].validation_improvement_fraction == 0.0
    assert replay.events[-1].outcome == "adapted"
    assert replay.adaptive_model.model_version == "seed-model.adapt1"


def test_rejected_candidate_does_not_replace_deployed_model():
    replay, issue, _ = _make_replay(0.01, candidate_mae=99.5)
    incumbent = replay.adaptive_model
    replay._adapt(issue, ErrorDistributionChange(1000.0, 1400.0, 200, 40))

    assert replay.events[-1].outcome == "candidate_rejected"
    assert replay.adaptive_model is incumbent
    assert replay.adaptive_model.model_version == "seed-model"
    assert replay.events[-1].new_model_version is None


def test_reset_isolates_model_detector_cooldown_and_event_state():
    replay, issue, original_detector = _make_replay(0.0, candidate_mae=90.0)
    replay.detector.update(123.0)
    replay._adapt(issue, ErrorDistributionChange(1000.0, 1400.0, 200, 40))
    assert replay.adaptive_model.model_version == "seed-model.adapt1"
    assert replay.last_adaptation_timestamp == issue
    assert replay.events

    replay.reset()

    assert replay.adaptive_model.model_version == "seed-model"
    assert replay.adaptive_model is not replay.static_model
    assert replay.last_adaptation_timestamp is None
    assert replay.events == []
    assert replay.detector is not original_detector
    assert replay.detector.calls == 0
    assert replay.config.fine_tune_random_state == 42


def test_independent_threshold_replays_start_from_the_same_frozen_checkpoint():
    index = pd.date_range("2024-01-01", periods=72, freq="h", tz="UTC")
    demand = pd.Series(np.linspace(1000.0, 1100.0, len(index)), index=index)
    frozen = SimpleNamespace(model_version="seed-model")
    first = StreamingReplay(
        demand,
        frozen,
        config=_threshold_config(0.0),
        detector=StatefulDetector(),
    )
    second = StreamingReplay(
        demand,
        frozen,
        config=_threshold_config(0.005),
        detector=StatefulDetector(),
    )

    assert first.adaptive_model is not frozen
    assert second.adaptive_model is not frozen
    assert first.adaptive_model is not second.adaptive_model
    assert first.adaptive_model.model_version == second.adaptive_model.model_version == "seed-model"
    assert first.detector is not second.detector
    first.adaptive_model.model_version = "seed-model.adapt1"
    assert second.adaptive_model.model_version == frozen.model_version == "seed-model"


def test_experiment_config_changes_only_validation_threshold():
    data = json.loads(EXPERIMENT_CONFIG.read_text(encoding="utf-8"))
    thresholds = tuple(
        item["min_validation_improvement_fraction"] for item in data["thresholds"]
    )
    assert thresholds == EXPECTED_THRESHOLDS

    loaded = load_threshold_replay_experiments(EXPERIMENT_CONFIG)
    assert tuple(item.min_validation_improvement_fraction for item in loaded) == EXPECTED_THRESHOLDS
    configs = [item.replay_config for item in loaded]
    normalized = []
    for config in configs:
        record = asdict(config)
        record.pop("min_validation_improvement_fraction")
        normalized.append(record)
    assert all(record == normalized[0] for record in normalized[1:])
    assert data["controls"]["production_config_modified"] is False
    assert data["controls"]["each_run_starts_from_source_checkpoint"] is True


@pytest.mark.parametrize("threshold", [-0.001, 1.001])
def test_replay_config_rejects_out_of_range_threshold(threshold):
    with pytest.raises(ValueError, match="between 0 and 1"):
        ReplayConfig(min_validation_improvement_fraction=threshold)


def test_project_drift_config_rejects_out_of_range_threshold():
    with pytest.raises(ValueError, match="between 0 and 1"):
        DriftConfig(min_validation_improvement_fraction=1.001)
