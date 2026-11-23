"""Resumable development selection and frozen benchmark for Phase 3.8 Random Forest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from griddrift.forecasting.features import (
    DirectFeatureSpec,
    _build_feature_row_validated,
    _validate_demand,
    build_supervised_dataset,
)
from griddrift.forecasting.random_forest import (
    DirectRandomForestForecaster,
    fit_direct_random_forest_samples,
)

TRAIN_CUTOFF = pd.Timestamp("2024-03-15T09:00:00Z")
DEV_START = pd.Timestamp("2024-04-09T00:00:00Z")
DEV_END = pd.Timestamp("2024-12-31T23:00:00Z")
FINAL_START = pd.Timestamp("2025-01-01T00:00:00Z")
FINAL_END = pd.Timestamp("2026-10-20T23:00:00Z")
SEED = 42
CONFIGURATIONS: dict[str, dict[str, object]] = {
    "rf60_f05_leaf2": {"n_estimators": 60, "max_features": 0.5, "min_samples_leaf": 2, "n_jobs": 4},
    "rf100_f05_leaf2": {"n_estimators": 100, "max_features": 0.5, "min_samples_leaf": 2, "n_jobs": 4},
    "rf60_f1_leaf4": {"n_estimators": 60, "max_features": 1.0, "min_samples_leaf": 4, "n_jobs": 4},
}


def _json_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_demand(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return _validate_demand(pd.Series(frame["demand_mw"].to_numpy(float), index=index))


def _features_for_issues(
    demand: pd.Series, issues: pd.DatetimeIndex, spec: DirectFeatureSpec
) -> pd.DataFrame:
    rows = [_build_feature_row_validated(demand, issue, spec) for issue in issues]
    return pd.DataFrame(rows, index=issues)


def _metric_dict(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    actual = np.asarray(actual, dtype=float).reshape(-1)
    predicted = np.asarray(predicted, dtype=float).reshape(-1)
    return {
        "mae_mw": float(mean_absolute_error(actual, predicted)),
        "rmse_mw": float(np.sqrt(mean_squared_error(actual, predicted))),
        "count": int(actual.size),
    }


def _development_case(
    model: DirectRandomForestForecaster,
    demand: pd.Series,
    run_root: Path,
    config_name: str,
) -> dict[str, object]:
    output = run_root / "development" / config_name
    prediction_path = output / "fixed_24h_predictions.csv"
    metrics_path = output / "metrics.json"
    if prediction_path.exists() and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    target_index = pd.date_range(DEV_START, DEV_END, freq="h")
    issue_index = target_index - pd.Timedelta(hours=24)
    features = _features_for_issues(demand, issue_index, model.feature_spec)
    predictions = model.predict_feature_frame(features)[:, -1]
    frame = pd.DataFrame({
        "issue_timestamp": issue_index,
        "target_timestamp": target_index,
        "actual_demand_mw": demand.loc[target_index].to_numpy(),
        "predicted_demand_mw": predictions,
    })
    frame["absolute_error_mw"] = (frame.actual_demand_mw - frame.predicted_demand_mw).abs()
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(prediction_path, index=False)
    metrics: dict[str, object] = {
        "config_name": config_name,
        "period_start": DEV_START,
        "period_end": DEV_END,
        "selection_metric": "fixed_24h_mae_mw",
        "fixed_24h": _metric_dict(frame.actual_demand_mw, frame.predicted_demand_mw),
        "predictions_path": str(prediction_path),
    }
    _json_write(metrics_path, metrics)
    return metrics


def run_development(demand: pd.Series, run_root: Path) -> dict[str, object]:
    selection_path = run_root / "selection.json"
    comparison_path = run_root / "development" / "configuration_comparison.csv"
    if selection_path.exists() and comparison_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    spec = DirectFeatureSpec()
    training_data = demand.loc[: TRAIN_CUTOFF + pd.Timedelta(hours=spec.horizon_hours)]
    X, y = build_supervised_dataset(training_data, spec)
    eligible = X.index <= TRAIN_CUTOFF
    X, y = X.loc[eligible], y.loc[eligible]
    if X.empty or X.index[-1] > TRAIN_CUTOFF:
        raise ValueError("training labels crossed the fixed training cutoff")

    results: list[dict[str, object]] = []
    for name, parameters in CONFIGURATIONS.items():
        case_root = run_root / "development" / name
        model_path = case_root / "model" / "model.joblib"
        if model_path.exists():
            model = DirectRandomForestForecaster.load(model_path.parent)
        else:
            started = time.perf_counter()
            model = fit_direct_random_forest_samples(
                X, y, feature_spec=spec, parameters=parameters, random_state=SEED,
                model_version=f"rf38-{name}-{TRAIN_CUTOFF:%Y%m%dT%H%MZ}",
            )
            model.save(case_root / "model")
            _json_write(case_root / "fit_metadata.json", {
                "config_name": name, "parameters": parameters, "random_state": SEED,
                "training_cutoff_issue_timestamp": TRAIN_CUTOFF,
                "training_latest_target_timestamp": X.index[-1] + pd.Timedelta(hours=spec.horizon_hours),
                "training_first_issue_timestamp": X.index[0],
                "training_last_issue_timestamp": X.index[-1],
                "training_sample_count": len(X),
                "feature_count": X.shape[1],
                "fit_runtime_seconds": time.perf_counter() - started,
                "selection_data": "2024 development period only",
            })
        metrics = _development_case(model, demand, run_root, name)
        results.append(metrics)

    comparison = pd.DataFrame([
        {"config_name": row["config_name"], **row["fixed_24h"]} for row in results
    ]).sort_values("mae_mw", kind="stable")
    comparison.to_csv(comparison_path, index=False)
    selected_name = str(comparison.iloc[0]["config_name"])
    selected = {
        "selected_config": selected_name,
        "selected_on": "2024 development only",
        "selection_metric": "fixed_24h_mae_mw",
        "training_cutoff_issue_timestamp": TRAIN_CUTOFF,
        "development_target_start": DEV_START,
        "development_target_end": DEV_END,
        "random_state": SEED,
        "configurations": CONFIGURATIONS,
        "comparison_path": str(comparison_path),
        "results": results,
        "final_period_used_for_selection": False,
    }
    _json_write(selection_path, selected)
    return selected


def run_final(demand: pd.Series, run_root: Path) -> dict[str, object]:
    selection_path = run_root / "selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("run development selection first; selection.json is missing")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    name = selection["selected_config"]
    model = DirectRandomForestForecaster.load(run_root / "development" / name / "model")
    final_root = run_root / "final"
    prediction_path = final_root / "all_horizon_predictions.csv"
    metrics_path = final_root / "metrics.json"
    if prediction_path.exists() and metrics_path.exists():
        return json.loads(metrics_path.read_text(encoding="utf-8"))

    if prediction_path.exists():
        frame = pd.read_csv(
            prediction_path,
            parse_dates=["issue_timestamp", "target_timestamp"],
        )
        for column in ("issue_timestamp", "target_timestamp"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
    else:
        issue_index = pd.date_range(
            FINAL_START - pd.Timedelta(hours=model.feature_spec.horizon_hours),
            FINAL_END - pd.Timedelta(hours=1), freq="h",
        )
        features = _features_for_issues(demand, issue_index, model.feature_spec)
        prediction_matrix = model.predict_feature_frame(features)
        target_matrix = np.column_stack([
            demand.reindex(issue_index + pd.Timedelta(hours=step)).to_numpy()
            for step in range(1, model.feature_spec.horizon_hours + 1)
        ])
        issue_times_ns = issue_index.as_unit("ns").asi8
        target_times_ns = issue_times_ns[:, None] + (
            np.arange(1, model.feature_spec.horizon_hours + 1, dtype=np.int64)[None, :]
            * pd.Timedelta(hours=1).value
        )
        valid = (target_times_ns >= FINAL_START.value) & (target_times_ns <= FINAL_END.value)
        issue_values_ns = np.repeat(issue_times_ns[:, None], model.feature_spec.horizon_hours, axis=1)
        horizon_values = np.broadcast_to(
            np.arange(1, model.feature_spec.horizon_hours + 1), target_times_ns.shape
        )
        frame = pd.DataFrame({
            "issue_timestamp": pd.to_datetime(issue_values_ns[valid], utc=True),
            "target_timestamp": pd.to_datetime(target_times_ns[valid], utc=True),
            "horizon_hours": horizon_values[valid],
            "actual_demand_mw": target_matrix[valid],
            "predicted_demand_mw": prediction_matrix[valid],
        })
        frame["absolute_error_mw"] = (frame.actual_demand_mw - frame.predicted_demand_mw).abs()
        final_root.mkdir(parents=True, exist_ok=True)
        frame.to_csv(prediction_path, index=False)
    fixed = frame.loc[frame.horizon_hours == model.feature_spec.horizon_hours]
    metrics = {
        "model": "Random Forest",
        "model_version": model.model_version,
        "selected_config": name,
        "evaluation_period_start": FINAL_START,
        "evaluation_period_end": FINAL_END,
        "evaluation_basis": "frozen model, all 24 horizons; configuration fixed using 2024 development targets",
        "all_horizons": _metric_dict(frame.actual_demand_mw, frame.predicted_demand_mw),
        "fixed_24h": _metric_dict(fixed.actual_demand_mw, fixed.predicted_demand_mw),
        "prediction_rows": len(frame),
        "predictions_path": str(prediction_path),
        "prior_inspection_limitation": "2025-2026 was previously inspected in Phases 3.6/3.7; not a fresh untouched test set.",
    }
    final_root.mkdir(parents=True, exist_ok=True)
    model.save(final_root / "frozen_model", metrics)
    _json_write(metrics_path, metrics)
    _json_write(final_root / "configuration.json", {
        "selected_config": name,
        "parameters": CONFIGURATIONS[name],
        "random_state": SEED,
        "training_cutoff_issue_timestamp": TRAIN_CUTOFF,
        "training_latest_target_timestamp": model.train_end_issue_time + pd.Timedelta(hours=model.feature_spec.horizon_hours),
        "feature_spec": asdict(model.feature_spec),
    })
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "final", "all"), default="all")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input", required=True)
    args = parser.parse_args()
    run_root = ROOT / "artifacts" / "experiments" / "phase38" / "random_forest" / args.run_id
    demand = _load_demand(Path(args.input))
    if demand.index[-1] < FINAL_END:
        raise ValueError(f"input does not cover the frozen final interval through {FINAL_END}")
    source_digest = hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    metadata_path = run_root / "run_metadata.json"
    metadata = {
        "run_id": args.run_id,
        "dataset": args.input,
        "dataset_sha256": source_digest,
        "dataset_start": demand.index[0],
        "dataset_end": demand.index[-1],
        "training_cutoff_issue_timestamp": TRAIN_CUTOFF,
        "development_target_start": DEV_START,
        "development_target_end": DEV_END,
        "final_target_start": FINAL_START,
        "final_target_end": FINAL_END,
        "random_state": SEED,
        "configurations": CONFIGURATIONS,
        "stage": 4,
    }
    if metadata_path.exists():
        prior = json.loads(metadata_path.read_text(encoding="utf-8"))
        if prior.get("dataset_sha256") != source_digest or prior.get("run_id") != args.run_id:
            raise ValueError("run ID already belongs to different inputs; choose a new run ID")
    else:
        _json_write(metadata_path, metadata)
    if args.phase in ("development", "all"):
        result = run_development(demand, run_root)
        print(json.dumps({"phase": "development", "selected_config": result["selected_config"]}, indent=2))
    if args.phase in ("final", "all"):
        result = run_final(demand, run_root)
        print(json.dumps(result, indent=2, default=str))
    print(f"Saved Stage 4 outputs to {run_root}")


if __name__ == "__main__":
    main()
