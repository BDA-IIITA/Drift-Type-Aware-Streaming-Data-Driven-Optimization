"""Load isolated Phase 3.9 validation-threshold replay configurations."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from griddrift.drift import ADWINConfig
from griddrift.simulation.replay import ReplayConfig


@dataclass(frozen=True)
class ThresholdReplayExperiment:
    run_id: str
    min_validation_improvement_fraction: float
    replay_config: ReplayConfig


def load_threshold_replay_experiments(
    path: str | Path,
) -> tuple[ThresholdReplayExperiment, ...]:
    """Build replay configs from the Phase 3.9 matrix, without changing defaults."""
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    fixed_settings = dict(document["fixed_settings"])
    fixed_settings["adwin"] = ADWINConfig(**fixed_settings["adwin"])

    experiments = []
    run_ids: set[str] = set()
    thresholds: set[float] = set()
    for item in document["thresholds"]:
        run_id = str(item["run_id"])
        threshold = float(item["min_validation_improvement_fraction"])
        if not run_id or run_id in run_ids:
            raise ValueError("threshold experiment run IDs must be non-empty and unique")
        if threshold in thresholds:
            raise ValueError("threshold experiment values must be unique")
        replay_config = ReplayConfig(
            **fixed_settings,
            min_validation_improvement_fraction=threshold,
        )
        experiments.append(ThresholdReplayExperiment(run_id, threshold, replay_config))
        run_ids.add(run_id)
        thresholds.add(threshold)

    if not experiments:
        raise ValueError("threshold experiment configuration must contain at least one run")
    return tuple(experiments)
