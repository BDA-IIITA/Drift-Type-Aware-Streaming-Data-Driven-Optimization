"""Run the Phase 3.9 five-threshold MLP replay on the 2024 development split.

Previously completed 0%, 1%, 2%, and 5% runs are reused from Phase 3.6 only
after their saved configuration is verified against the frozen Phase 3.9
matrix. The missing 0.5% run is replayed from the same frozen checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_drift_replay import _demand  # noqa: E402
from griddrift.forecasting.mlp import DirectMLPForecaster  # noqa: E402
from griddrift.simulation import StreamingReplay  # noqa: E402
from griddrift.simulation.threshold_experiments import (  # noqa: E402
    load_threshold_replay_experiments,
)


CONFIG_PATH = ROOT / "artifacts/experiments/phase39/threshold_experiment_config.json"
SOURCE_ROOT = ROOT / "artifacts/experiments/phase36/threshold-sweep/threshold-dev-20261023"
OUTPUT_ROOT = ROOT / "artifacts/experiments/phase39/development"
SOURCE_NAMES = {
    0.0: "gentle_mixed_60d_threshold_00pct",
    0.01: "gentle_mixed_60d_threshold_01pct",
    0.02: "gentle_mixed_60d_threshold_02pct",
    0.05: "gentle_mixed_60d_threshold_05pct",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=True), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric_row(frame: pd.DataFrame, *, actual_col: str, predicted_col: str) -> dict[str, float]:
    if frame.empty:
        return {"mae_mw": float("nan"), "rmse_mw": float("nan"), "mape_pct": float("nan"), "smape_pct": float("nan"), "count": 0.0}
    actual = frame[actual_col].to_numpy(dtype=float)
    predicted = frame[predicted_col].to_numpy(dtype=float)
    error = np.abs(actual - predicted)
    nonzero = np.abs(actual) > 1e-12
    smape_denominator = np.abs(actual) + np.abs(predicted)
    smape_valid = smape_denominator > 1e-12
    return {
        "mae_mw": float(error.mean()),
        "rmse_mw": float(np.sqrt(np.mean(np.square(error)))),
        "mape_pct": float(np.mean(error[nonzero] / np.abs(actual[nonzero])) * 100.0) if nonzero.any() else float("nan"),
        "smape_pct": float(np.mean(2.0 * error[smape_valid] / smape_denominator[smape_valid]) * 100.0) if smape_valid.any() else float("nan"),
        "count": float(len(frame)),
    }


def _target_period(frame: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame.empty:
        return frame
    targets = pd.to_datetime(frame["target_timestamp"], utc=True)
    return frame.loc[(targets >= start) & (targets <= end)].copy()


def _registry_metrics(path: Path, demand: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    records = pd.read_csv(path, parse_dates=["issue_timestamp", "target_timestamp"])
    records["target_timestamp"] = pd.to_datetime(records["target_timestamp"], utc=True)
    records = records.loc[(records["target_timestamp"] >= start) & (records["target_timestamp"] <= end)].copy()
    records["actual_demand_mw"] = records["target_timestamp"].map(demand)
    records = records.dropna(subset=["actual_demand_mw"])
    return _metric_row(records, actual_col="actual_demand_mw", predicted_col="predicted_demand_mw")


def _event_counts(events: pd.DataFrame) -> dict[str, int]:
    outcomes = events["outcome"].astype(str) if not events.empty else pd.Series(dtype=str)
    accepted = int(outcomes.eq("adapted").sum())
    rejected = int(outcomes.eq("candidate_rejected").sum() + outcomes.str.startswith("candidate_rejected:").sum())
    return {
        "drift_alarms": int(len(events)),
        "adaptation_attempts": accepted + rejected,
        "accepted_adaptations": accepted,
        "rejected_adaptations": rejected,
        "cooldown_skips": int(outcomes.eq("cooldown_skipped").sum()),
        "non_increase_or_unusable_alarm_skips": int(outcomes.eq("error_increase_not_meaningful").sum()),
    }


def _before_after(events: pd.DataFrame, errors: pd.DataFrame) -> pd.DataFrame:
    columns = ["adaptation_timestamp", "new_model_version", "before_mae_mw", "before_count", "after_mae_mw", "after_count", "after_minus_before_mae_mw", "later_window_result"]
    rows: list[dict[str, object]] = []
    if events.empty or errors.empty:
        return pd.DataFrame(rows, columns=columns)
    target = pd.to_datetime(errors["target_timestamp"], utc=True)
    for event in events.loc[events["outcome"] == "adapted"].itertuples(index=False):
        time = pd.Timestamp(event.drift_timestamp)
        before = errors.loc[(target >= time - pd.Timedelta(days=7)) & (target < time) & (errors["model_version"] == event.previous_model_version)]
        after = errors.loc[(target >= time + pd.Timedelta(hours=24)) & (target <= time + pd.Timedelta(days=7)) & (errors["model_version"] == event.new_model_version)]
        before_mae = float(before["absolute_error_mw"].mean()) if not before.empty else float("nan")
        after_mae = float(after["absolute_error_mw"].mean()) if not after.empty else float("nan")
        label = "not_measurable"
        if np.isfinite(before_mae) and np.isfinite(after_mae):
            label = "improved" if after_mae < before_mae else "worsened_or_equal"
        rows.append({
            "adaptation_timestamp": time,
            "new_model_version": event.new_model_version,
            "before_mae_mw": before_mae,
            "before_count": len(before),
            "after_mae_mw": after_mae,
            "after_count": len(after),
            "after_minus_before_mae_mw": after_mae - before_mae,
            "later_window_result": label,
        })
    return pd.DataFrame(rows, columns=columns)


def _fixed_metrics(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    errors = pd.read_csv(path, parse_dates=["target_timestamp"])
    errors["target_timestamp"] = pd.to_datetime(errors["target_timestamp"], utc=True)
    errors = _target_period(errors, start, end)
    return _metric_row(errors, actual_col="actual_demand_mw", predicted_col="predicted_demand_mw")


def _verify_reusable_source(item: dict[str, object], source_name: str, fixed: dict[str, object], train_end: str) -> dict[str, object]:
    metadata = json.loads((SOURCE_ROOT / "run_metadata.json").read_text(encoding="utf-8"))
    metric_rows = pd.read_csv(SOURCE_ROOT / "metrics.csv")
    row = metric_rows.loc[metric_rows["strategy"] == source_name]
    if len(row) != 1:
        raise ValueError(f"expected one saved metrics row for {source_name}")
    recorded = json.loads(row.iloc[0]["configuration"])
    expected = {
        "strategy": "mixed_history_fine_tune",
        "window_days": 60,
        "validation_days": int(fixed["validation_days"]),
        "validation_periods": int(fixed["validation_periods"]),
        "validation_threshold_fraction": float(item["min_validation_improvement_fraction"]),
        "horizon_hours": 24,
        "retraining_window_days": 60,
        "cooldown_days": int(fixed["cooldown_days"]),
        "adaptation_policy": "validated",
        "adaptation_strategy": "mixed_history_fine_tune",
        "min_error_increase_mw": float(fixed["min_error_increase_mw"]),
        "min_error_increase_fraction": float(fixed["min_error_increase_fraction"]),
        "min_validation_improvement_fraction": float(item["min_validation_improvement_fraction"]),
        "fine_tune_epochs": int(fixed["fine_tune_epochs"]),
        "fine_tune_learning_rate_factor": float(fixed["fine_tune_learning_rate_factor"]),
        "fine_tune_max_updates": int(fixed["fine_tune_max_updates"]),
        "fine_tune_batch_size": int(fixed["fine_tune_batch_size"]),
        "fine_tune_random_state": int(fixed["fine_tune_random_state"]),
        "representative_samples_per_month": int(fixed["representative_samples_per_month"]),
        "adwin": fixed["adwin"],
    }
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise ValueError(f"saved Phase 3.6 {source_name} does not match Phase 3.9 on {key}: {recorded.get(key)!r} != {value!r}")
    source_train_end = pd.Timestamp(metadata["model_train_end"]).tz_convert("UTC").isoformat().replace("+00:00", "Z")
    if source_train_end != train_end:
        raise ValueError(f"saved Phase 3.6 training boundary mismatch: {source_train_end!r} != {train_end!r}")
    checks = {
        "initial_model_version": "mlp-1209a11e1cf2",
        "replay_start": "2024-04-09T00:00:00Z",
        "replay_end": "2024-12-31T23:00:00Z",
        "cooldown_days": int(fixed["cooldown_days"]),
        "fine_tune_random_state": int(fixed["fine_tune_random_state"]),
        "representative_samples_per_month": int(fixed["representative_samples_per_month"]),
    }
    for key, value in checks.items():
        if metadata.get(key) != value:
            raise ValueError(f"saved Phase 3.6 run metadata mismatch on {key}: {metadata.get(key)!r} != {value!r}")
    source_strategy = SOURCE_ROOT / "strategies" / source_name
    required = ["adaptation_events.csv", "adaptive_24h_errors.csv", "adaptive_forecast_registry.csv", "final_adaptive_model/metadata.json", "final_adaptive_model/model.joblib"]
    if any(not (source_strategy / name).is_file() for name in required):
        raise FileNotFoundError(f"saved Phase 3.6 output is incomplete: {source_strategy}")
    source_checkpoint = SOURCE_ROOT / "initial_model.joblib"
    current_checkpoint = ROOT / "artifacts/models/de-lu-mlp/model.joblib"
    source_hash = _sha256(source_checkpoint)
    current_hash = _sha256(current_checkpoint)
    if source_hash != current_hash:
        raise ValueError(f"saved Phase 3.6 {source_name} used a different frozen checkpoint")
    return {
        "source_metrics": row.iloc[0].to_dict(),
        "source_name": source_name,
        "source_strategy_path": str(source_strategy),
        "checkpoint_sha256": current_hash,
    }


def _copy_static_reference() -> None:
    static_root = OUTPUT_ROOT / "static_reference"
    if static_root.exists():
        return
    static_root.mkdir(parents=True, exist_ok=False)
    for filename in ("static_24h_errors.csv", "static_forecast_registry.csv", "initial_model.joblib", "initial_model_metadata.json"):
        shutil.copy2(SOURCE_ROOT / filename, static_root / filename)


def _run_new_experiment(experiment: object, demand: pd.Series, source_model: DirectMLPForecaster, config_doc: dict[str, object], train_end: str, target_start: pd.Timestamp, target_end: pd.Timestamp) -> dict[str, object]:
    run_id = experiment.run_id
    runs_root = OUTPUT_ROOT / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)
    final_dir = runs_root / run_id
    if final_dir.exists():
        summary_path = final_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "complete":
                print(f"Reusing completed Phase 3.9 run {run_id}", flush=True)
                return summary
        raise FileExistsError(f"incomplete or non-complete output exists at {final_dir}; preserving it")
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    temp_dir = runs_root / f".{run_id}.{stamp}.partial"
    temp_dir.mkdir(parents=True, exist_ok=False)
    fixed = config_doc["fixed_settings"]
    run_config = experiment.replay_config
    run_meta = {
        "status": "running",
        "run_id": run_id,
        "source_model_version": source_model.model_version,
        "source_model_train_end_issue_time_utc": train_end,
        "development_replay_start_issue_time_utc": config_doc["development_replay_period_utc"]["start_inclusive"],
        "development_replay_end_issue_time_utc": config_doc["development_replay_period_utc"]["end_inclusive"],
        "development_fixed_24h_target_start_utc": target_start.isoformat(),
        "development_fixed_24h_target_end_utc": target_end.isoformat(),
        "dataset_path": config_doc["source_dataset"],
        "threshold_fraction": experiment.min_validation_improvement_fraction,
        "fixed_settings": fixed,
        "source_checkpoint_path": config_doc["source_checkpoint"],
        "created_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(temp_dir / "run_metadata.json", run_meta)
    started = time.perf_counter()
    result = StreamingReplay(demand, source_model, config=run_config).run(
        start_timestamp=config_doc["development_replay_period_utc"]["start_inclusive"],
        end_timestamp=config_doc["development_replay_period_utc"]["end_inclusive"],
    )
    runtime = time.perf_counter() - started
    for name, frame in (("adaptive_24h_errors.csv", result.adaptive_errors), ("adaptation_events.csv", result.adaptation_frame())):
        frame.to_csv(temp_dir / name, index=False)
    result.adaptive_registry.save_csv(temp_dir / "adaptive_forecast_registry.csv")
    result.final_adaptive_model.save(temp_dir / "final_adaptive_model", result.adaptive_metrics)
    after_before = _before_after(result.adaptation_frame(), _target_period(result.adaptive_errors, target_start, target_end))
    after_before.to_csv(temp_dir / "before_after_adaptations.csv", index=False)
    counts = _event_counts(result.adaptation_frame())
    fixed_static = _fixed_metrics(OUTPUT_ROOT / "static_reference/static_24h_errors.csv", target_start, target_end)
    fixed_adaptive = _metric_row(_target_period(result.adaptive_errors, target_start, target_end), actual_col="actual_demand_mw", predicted_col="predicted_demand_mw")
    all_static = _registry_metrics(OUTPUT_ROOT / "static_reference/static_forecast_registry.csv", demand, target_start, target_end)
    all_adaptive = _registry_metrics(temp_dir / "adaptive_forecast_registry.csv", demand, target_start, target_end)
    later = after_before["later_window_result"].value_counts().to_dict() if not after_before.empty else {}
    summary = {
        "status": "complete",
        "run_id": run_id,
        "threshold_fraction": experiment.min_validation_improvement_fraction,
        "fixed_24h_static": fixed_static,
        "fixed_24h_adaptive": fixed_adaptive,
        "all_horizon_static": all_static,
        "all_horizon_adaptive": all_adaptive,
        **counts,
        "accepted_updates_later_window_improved": int(later.get("improved", 0)),
        "accepted_updates_later_window_worsened_or_equal": int(later.get("worsened_or_equal", 0)),
        "accepted_updates_later_window_not_measurable": int(later.get("not_measurable", 0)),
        "candidate_training_runtime_seconds": result.candidate_training_runtime_seconds,
        "inference_runtime_seconds": result.inference_runtime_seconds,
        "runtime_seconds": runtime,
        "final_model_version": result.final_adaptive_model.model_version,
        "static_reference_sha256": _sha256(OUTPUT_ROOT / "static_reference/static_24h_errors.csv"),
    }
    _write_json(temp_dir / "summary.json", summary)
    run_meta["status"] = "complete"
    run_meta["completed_utc"] = datetime.now(UTC).isoformat()
    _write_json(temp_dir / "run_metadata.json", run_meta)
    temp_dir.rename(final_dir)
    print(f"Completed {run_id}: fixed24 MAE {fixed_adaptive['mae_mw']:.2f} MW; {counts['accepted_adaptations']} accepted; {runtime:.1f}s", flush=True)
    return summary


def _reuse_experiment(experiment: object, source: dict[str, object], demand: pd.Series, config_doc: dict[str, object], train_end: str, target_start: pd.Timestamp, target_end: pd.Timestamp) -> dict[str, object]:
    run_id = experiment.run_id
    run_dir = OUTPUT_ROOT / "runs" / run_id
    if run_dir.exists():
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("status") == "complete":
                print(f"Reusing completed Phase 3.9 copy {run_id}", flush=True)
                return summary
        raise FileExistsError(f"incomplete Phase 3.9 output exists at {run_dir}; preserving it")
    src = Path(str(source["source_strategy_path"]))
    run_dir.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, run_dir)
    metric = source["source_metrics"]
    fixed_static = _fixed_metrics(OUTPUT_ROOT / "static_reference/static_24h_errors.csv", target_start, target_end)
    fixed_adaptive = _fixed_metrics(run_dir / "adaptive_24h_errors.csv", target_start, target_end)
    all_static = _registry_metrics(OUTPUT_ROOT / "static_reference/static_forecast_registry.csv", demand, target_start, target_end)
    all_adaptive = _registry_metrics(run_dir / "adaptive_forecast_registry.csv", demand, target_start, target_end)
    events = pd.read_csv(run_dir / "adaptation_events.csv")
    counts = _event_counts(events)
    adaptive_errors = pd.read_csv(run_dir / "adaptive_24h_errors.csv", parse_dates=["target_timestamp"])
    adaptive_errors["target_timestamp"] = pd.to_datetime(adaptive_errors["target_timestamp"], utc=True)
    after_before = _before_after(events, _target_period(adaptive_errors, target_start, target_end))
    after_before.to_csv(run_dir / "before_after_adaptations.csv", index=False)
    later = after_before["later_window_result"].value_counts().to_dict() if not after_before.empty else {}
    summary = {
        "status": "complete",
        "run_id": run_id,
        "threshold_fraction": experiment.min_validation_improvement_fraction,
        "fixed_24h_static": fixed_static,
        "fixed_24h_adaptive": fixed_adaptive,
        "all_horizon_static": all_static,
        "all_horizon_adaptive": all_adaptive,
        **counts,
        "accepted_updates_later_window_improved": int(later.get("improved", 0)),
        "accepted_updates_later_window_worsened_or_equal": int(later.get("worsened_or_equal", 0)),
        "accepted_updates_later_window_not_measurable": int(later.get("not_measurable", 0)),
        "candidate_training_runtime_seconds": float(metric["candidate_training_runtime_seconds"]),
        "inference_runtime_seconds": float(metric["inference_runtime_seconds"]),
        "runtime_seconds": float(metric["runtime_seconds"]),
        "final_model_version": json.loads((run_dir / "final_adaptive_model/metadata.json").read_text(encoding="utf-8"))["model_version"],
        "reused_from": str(source["source_strategy_path"]),
        "source_static_reference_sha256": _sha256(OUTPUT_ROOT / "static_reference/static_24h_errors.csv"),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "run_metadata.json", {
        "status": "complete",
        "run_id": run_id,
        "threshold_fraction": experiment.min_validation_improvement_fraction,
        "reused_from": str(source["source_strategy_path"]),
        "source_root_run_metadata": str(SOURCE_ROOT / "run_metadata.json"),
        "source_model_version": "mlp-1209a11e1cf2",
        "source_model_train_end_issue_time_utc": train_end,
        "development_replay_start_issue_time_utc": config_doc["development_replay_period_utc"]["start_inclusive"],
        "development_replay_end_issue_time_utc": config_doc["development_replay_period_utc"]["end_inclusive"],
        "development_fixed_24h_target_start_utc": target_start.isoformat(),
        "development_fixed_24h_target_end_utc": target_end.isoformat(),
        "reuse_verified_configuration": source["verification"],
    })
    print(f"Reused verified Phase 3.6 replay {run_id}: fixed24 MAE {fixed_adaptive['mae_mw']:.2f} MW", flush=True)
    return summary


def main() -> None:
    config_doc = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    experiments = load_threshold_replay_experiments(CONFIG_PATH)
    model_path = ROOT / config_doc["source_checkpoint"]
    dataset_path = ROOT / config_doc["source_dataset"]
    source_model = DirectMLPForecaster.load(model_path)
    train_end = source_model.train_end_issue_time.isoformat().replace("+00:00", "Z")
    period = config_doc["development_replay_period_utc"]
    replay_start = pd.Timestamp(period["start_inclusive"])
    replay_end = pd.Timestamp(period["end_inclusive"])
    target_start = replay_start + pd.Timedelta(hours=24)
    target_end = replay_end
    last_training_target = source_model.train_end_issue_time + pd.Timedelta(hours=source_model.feature_spec.horizon_hours)
    if last_training_target >= replay_start:
        raise ValueError(f"development issue period overlaps training targets: {last_training_target} >= {replay_start}")
    if SOURCE_ROOT.exists():
        for experiment in experiments:
            threshold = experiment.min_validation_improvement_fraction
            if threshold not in SOURCE_NAMES:
                continue
            verified = _verify_reusable_source(
                {"min_validation_improvement_fraction": threshold},
                SOURCE_NAMES[threshold], config_doc["fixed_settings"], train_end,
            )
            verified["verification"] = "matching replay settings, source checkpoint, train boundary, and development issue range"
            verified_sources[threshold] = verified
    demand = _demand(str(dataset_path))
    if replay_start not in demand.index or replay_end not in demand.index:
        raise ValueError("development replay boundaries are not exact hourly observations in the dataset")
    if last_training_target not in demand.index:
        raise ValueError("dataset does not contain the frozen model's final training-target observation")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(OUTPUT_ROOT / "chronology_audit.json", {
        "status": "passed",
        "model_version": source_model.model_version,
        "checkpoint_sha256": _sha256(model_path / "model.joblib"),
        "train_end_issue_time_utc": source_model.train_end_issue_time.isoformat(),
        "last_training_target_time_utc": last_training_target.isoformat(),
        "development_first_forecast_issue_time_utc": replay_start.isoformat(),
        "development_first_fixed_24h_target_time_utc": target_start.isoformat(),
        "development_last_target_time_utc": target_end.isoformat(),
        "gap_from_last_training_target_to_first_development_issue_hours": (replay_start - last_training_target).total_seconds() / 3600.0,
        "fixed_24h_target_count_expected": len(pd.date_range(target_start, target_end, freq="h")),
        "sequential_replay_enforces_observation_availability": True,
        "adaptation_training_labels_checked_as_observed_at_event_time": True,
        "note": "The frozen model is trained through its final 24-hour target. Development predictions begin after that target has been observed. Replay scores a prediction only when the matching target timestamp is revealed.",
    })
    _copy_static_reference()
    # Store an auditable copy of the frozen checkpoint once; prior runs may load the same shared checkpoint.
    frozen_copy = OUTPUT_ROOT / "source_model"
    if not frozen_copy.exists():
        shutil.copytree(model_path, frozen_copy)

    _write_json(OUTPUT_ROOT / "reused_source_verification.json", {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "phase36_source_root": str(SOURCE_ROOT),
        "current_source_checkpoint_sha256": _sha256(model_path / "model.joblib"),
        "reused_threshold_runs": {
            str(threshold): {
                "run_id": next(item.run_id for item in experiments if item.min_validation_improvement_fraction == threshold),
                "source_name": value["source_name"],
                "source_strategy_path": value["source_strategy_path"],
                "checkpoint_sha256": value["checkpoint_sha256"],
                "configuration_check": value["verification"],
            }
            for threshold, value in verified_sources.items()
        },
    })

    summaries = []
    for experiment in experiments:
        threshold = experiment.min_validation_improvement_fraction
        if threshold in verified_sources:
            source = verified_sources[threshold]
            summary = _reuse_experiment(experiment, source, demand, config_doc, train_end, target_start, target_end)
        else:
            summary = _run_new_experiment(experiment, demand, source_model, config_doc, train_end, target_start, target_end)
        summaries.append(summary)
        # Checkpoint each completed threshold before moving to the next.
        _write_json(OUTPUT_ROOT / "completed_runs.json", {"completed_run_ids": [row["run_id"] for row in summaries]})

    rows = []
    for summary in summaries:
        row = {
            "run_id": summary["run_id"],
            "threshold_fraction": summary["threshold_fraction"],
            "static_mae_mw": summary["fixed_24h_static"]["mae_mw"],
            "static_rmse_mw": summary["fixed_24h_static"]["rmse_mw"],
            "adaptive_mae_mw": summary["fixed_24h_adaptive"]["mae_mw"],
            "adaptive_rmse_mw": summary["fixed_24h_adaptive"]["rmse_mw"],
            "adaptive_mape_pct": summary["fixed_24h_adaptive"]["mape_pct"],
            "adaptive_smape_pct": summary["fixed_24h_adaptive"]["smape_pct"],
            "matched_fixed_24h_predictions": summary["fixed_24h_adaptive"]["count"],
            "all_horizon_static_mae_mw": summary["all_horizon_static"]["mae_mw"],
            "all_horizon_static_rmse_mw": summary["all_horizon_static"]["rmse_mw"],
            "all_horizon_adaptive_mae_mw": summary["all_horizon_adaptive"]["mae_mw"],
            "all_horizon_adaptive_rmse_mw": summary["all_horizon_adaptive"]["rmse_mw"],
            "all_horizon_adaptive_mape_pct": summary["all_horizon_adaptive"]["mape_pct"],
            "all_horizon_adaptive_smape_pct": summary["all_horizon_adaptive"]["smape_pct"],
            "matched_all_horizon_predictions": summary["all_horizon_adaptive"]["count"],
            "drift_alarms": summary["drift_alarms"],
            "adaptation_attempts": summary["adaptation_attempts"],
            "accepted_adaptations": summary["accepted_adaptations"],
            "rejected_adaptations": summary["rejected_adaptations"],
            "candidate_training_runtime_seconds": summary["candidate_training_runtime_seconds"],
            "inference_runtime_seconds": summary["inference_runtime_seconds"],
            "runtime_seconds": summary["runtime_seconds"],
            "accepted_update_later_window_improved": summary["accepted_updates_later_window_improved"],
            "accepted_update_later_window_worsened_or_equal": summary["accepted_updates_later_window_worsened_or_equal"],
            "accepted_update_later_window_not_measurable": summary["accepted_updates_later_window_not_measurable"],
            "result_origin": "reused_verified_phase36" if "reused_from" in summary else "phase39_replay",
        }
        rows.append(row)
    results = pd.DataFrame(rows).sort_values("threshold_fraction")
    results.to_csv(OUTPUT_ROOT / "threshold_results.csv", index=False)
    _write_json(OUTPUT_ROOT / "threshold_results.json", rows)
    print(f"Saved Phase 3.9 development threshold results to {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    verified_sources: dict[float, dict[str, object]] = {}
    main()
