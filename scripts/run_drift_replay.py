"""Replay a demand history and adapt the MLP only on ADWIN alarms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from griddrift.config import load_project_config
from griddrift.drift import ADWINConfig
from griddrift.forecasting.mlp import DirectMLPForecaster
from griddrift.simulation import ReplayConfig, StreamingReplay


def _demand(path: str) -> pd.Series:
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return pd.Series(frame["demand_mw"].to_numpy(dtype=float), index=index, name="demand_mw")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Canonical hourly demand CSV")
    parser.add_argument("--model", required=True, help="Frozen Phase 2 model directory")
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="artifacts/replays/de-lu")
    parser.add_argument("--end", help="Optional inclusive UTC replay end timestamp")
    parser.add_argument("--start", help="Optional UTC issue-time start for a separate evaluation period")
    parser.add_argument("--policy", choices=("original", "increase_only", "validated"), default="original")
    parser.add_argument("--strategy", choices=("scratch", "fine_tune", "recent_plus_history"), default="scratch")
    parser.add_argument("--window-days", type=int, default=60)
    parser.add_argument("--window-months", type=int)
    args = parser.parse_args()

    project = load_project_config(args.config)
    drift = project.drift
    replay_config = ReplayConfig(
        cooldown_days=drift.cooldown_days,
        adaptation_policy=args.policy,
        adaptation_strategy=args.strategy,
        retraining_window_days=args.window_days,
        retraining_window_months=args.window_months,
        min_error_increase_mw=drift.min_error_increase_mw,
        min_error_increase_fraction=drift.min_error_increase_fraction,
        validation_days=drift.validation_days,
        min_validation_improvement_fraction=drift.min_validation_improvement_fraction,
        fine_tune_epochs=drift.fine_tune_epochs,
        adwin=ADWINConfig(delta=drift.delta, clock=drift.clock, min_window_length=drift.min_window_length),
    )
    result = StreamingReplay(
        _demand(args.input), DirectMLPForecaster.load(args.model), config=replay_config
    ).run(start_timestamp=args.start, end_timestamp=args.end)

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result.static_errors.to_csv(output / "static_24h_errors.csv", index=False)
    result.adaptive_errors.to_csv(output / "adaptive_24h_errors.csv", index=False)
    result.static_registry.save_csv(output / "static_forecast_registry.csv")
    result.adaptive_registry.save_csv(output / "adaptive_forecast_registry.csv")
    result.adaptation_frame().to_csv(output / "adaptation_events.csv", index=False)
    result.final_adaptive_model.save(output / "adaptive_model", result.adaptive_metrics)
    summary = {
        "drift_detections": result.drift_count,
        "adaptations": result.adaptation_count,
        "adaptation_attempts": result.adaptation_attempt_count,
        "rejected_adaptations": result.rejected_adaptation_count,
        "runtime_seconds": result.runtime_seconds,
        "static_metrics": result.static_metrics,
        "adaptive_metrics": result.adaptive_metrics,
        "final_model_version": result.final_adaptive_model.model_version,
        "policy": args.policy,
        "strategy": args.strategy,
        "window_days": args.window_days,
        "window_months": args.window_months,
        "start_timestamp": args.start,
        "end_timestamp": args.end,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved replay outputs to {output}")


if __name__ == "__main__":
    main()
