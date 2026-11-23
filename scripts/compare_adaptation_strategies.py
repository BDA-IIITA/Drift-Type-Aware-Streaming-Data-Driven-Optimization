"""Short, reproducible Phase 3.5 adaptation comparison on a shared replay span."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from griddrift.config import load_project_config
from griddrift.drift import ADWINConfig
from griddrift.forecasting.mlp import DirectMLPForecaster
from griddrift.simulation import ReplayConfig, StreamingReplay
from run_drift_replay import _demand


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="artifacts/experiments/phase35-short")
    parser.add_argument("--days", type=int, default=24)
    parser.add_argument("--max-iter", type=int, default=100)
    args = parser.parse_args()
    if args.days < 1:
        parser.error("--days must be positive")

    demand = _demand(args.input)
    static_model = DirectMLPForecaster.load(args.model)
    drift = load_project_config(args.config).drift
    first_position = int(
        demand.index.searchsorted(
            static_model.train_end_issue_time + pd.Timedelta(hours=24), side="left"
        )
    )
    stop_position = min(len(demand) - 1, first_position + args.days * 24)
    end_timestamp = demand.index[stop_position]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    cases = [
        ("original_scratch_60d", "original", "scratch", 60, None),
        ("increase_only_scratch_60d", "increase_only", "scratch", 60, None),
        ("validated_scratch_30d", "validated", "scratch", 30, None),
        ("validated_scratch_60d", "validated", "scratch", 60, None),
        ("validated_scratch_120d", "validated", "scratch", 120, None),
        ("validated_scratch_3mo", "validated", "scratch", 60, 3),
        ("validated_finetune_60d", "validated", "fine_tune", 60, None),
        ("validated_recent_plus_history_60d", "validated", "recent_plus_history", 60, None),
    ]
    summaries = []
    for name, policy, strategy, window_days, window_months in cases:
        config = ReplayConfig(
            cooldown_days=drift.cooldown_days,
            adwin=ADWINConfig(
                delta=drift.delta,
                clock=drift.clock,
                min_window_length=drift.min_window_length,
            ),
            adaptation_policy=policy,
            adaptation_strategy=strategy,
            retraining_window_days=window_days,
            retraining_window_months=window_months,
            min_error_increase_mw=drift.min_error_increase_mw,
            min_error_increase_fraction=drift.min_error_increase_fraction,
            validation_days=drift.validation_days,
            min_validation_improvement_fraction=drift.min_validation_improvement_fraction,
            fine_tune_epochs=drift.fine_tune_epochs,
            adaptation_max_iter=args.max_iter,
        )
        started = time.perf_counter()
        result = StreamingReplay(demand, static_model, config=config).run(end_timestamp=end_timestamp)
        elapsed = time.perf_counter() - started
        event_path = output / f"{name}_events.csv"
        result.adaptation_frame().to_csv(event_path, index=False)
        summaries.append(
            {
                "strategy": name,
                "policy": policy,
                "training_strategy": strategy,
                "window_days": window_days if window_months is None else None,
                "window_calendar_months": window_months,
                "drift_alarms": result.drift_count,
                "adaptation_attempts": result.adaptation_attempt_count,
                "accepted_adaptations": result.adaptation_count,
                "rejected_adaptations": result.rejected_adaptation_count,
                "mae_mw": result.adaptive_metrics["mae_mw"],
                "rmse_mw": result.adaptive_metrics["rmse_mw"],
                "static_mae_mw": result.static_metrics["mae_mw"],
                "static_rmse_mw": result.static_metrics["rmse_mw"],
                "count": result.adaptive_metrics["count"],
                "runtime_seconds": elapsed,
                "end_timestamp": end_timestamp.isoformat(),
                "event_log": event_path.name,
            }
        )
        print(json.dumps(summaries[-1]), flush=True)

    summary = {
        "static_model_version": static_model.model_version,
        "replay_start": demand.index[first_position].isoformat(),
        "replay_end": end_timestamp.isoformat(),
        "replay_days_requested": args.days,
        "max_iter": args.max_iter,
        "seed": 42,
        "strategies": summaries,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pd.DataFrame(summaries).to_csv(output / "summary.csv", index=False)


if __name__ == "__main__":
    main()
