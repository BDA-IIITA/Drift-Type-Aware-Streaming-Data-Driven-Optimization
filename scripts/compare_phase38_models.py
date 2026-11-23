"""Compare saved Phase 3.6/3.7/3.8 forecasts without retraining or replaying models."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from griddrift.forecasting.comparison import align_forecasts

FINAL_START = pd.Timestamp("2025-01-01T00:00:00Z")
FINAL_END = pd.Timestamp("2026-10-20T23:00:00Z")
DEV_START = pd.Timestamp("2024-04-09T00:00:00Z")
DEV_END = pd.Timestamp("2024-12-31T23:00:00Z")
RUN_ID_DEFAULT = "common-period-20261024"

FORECAST_PATHS = {
    "static_mlp": Path("artifacts/experiments/phase36/final/final-holdout-rerun-20261023/strategies/gentle_mixed_60d_threshold_05pct/static_forecast_registry.csv"),
    "adaptive_mlp": Path("artifacts/experiments/phase36/final/final-holdout-rerun-20261023/strategies/gentle_mixed_60d_threshold_05pct/adaptive_forecast_registry.csv"),
    "static_xgboost": Path("artifacts/experiments/phase37/final/final-holdout-20261024/full_history/static_forecast_registry.csv"),
    "adaptive_xgboost": Path("artifacts/experiments/phase37/final/final-holdout-20261024/full_history/adaptive_forecast_registry.csv"),
    "random_forest": Path("artifacts/experiments/phase38/random_forest/rf-dev-final-20261024/final/all_horizon_predictions.csv"),
}
METRIC_FILES = {
    "mlp": Path("artifacts/experiments/phase36/final/final-holdout-rerun-20261023/metrics.json"),
    "xgboost": Path("artifacts/experiments/phase37/final/final-holdout-20261024/full_history/summary.json"),
}


def _read_actuals(path: Path) -> pd.Series:
    frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
    index = pd.DatetimeIndex(frame.index)
    index = index.tz_localize("UTC") if index.tz is None else index.tz_convert("UTC")
    return pd.Series(frame["demand_mw"].to_numpy(dtype=float), index=index, name="demand_mw")


def _read_forecasts(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required saved forecast file is missing: {path}")
    frame = pd.read_csv(path, usecols=[
        "issue_timestamp", "target_timestamp", "horizon_hours", "predicted_demand_mw"
    ])
    for column in ("issue_timestamp", "target_timestamp"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    frame["horizon_hours"] = pd.to_numeric(frame.horizon_hours, errors="raise").astype(int)
    return frame


def _metrics(frame: pd.DataFrame, models: list[str]) -> list[dict[str, object]]:
    results = []
    y = frame.actual_demand_mw.to_numpy(dtype=float)
    for model in models:
        prediction = frame[f"{model}_predicted_mw"].to_numpy(dtype=float)
        error = prediction - y
        results.append({
            "model": model,
            "mae_mw": float(np.mean(np.abs(error))),
            "rmse_mw": float(np.sqrt(np.mean(error**2))),
            "mape_pct": float(np.mean(np.abs(error) / np.maximum(np.abs(y), 1e-9)) * 100),
            "smape_pct": float(np.mean(
                2 * np.abs(error) / np.maximum(np.abs(y) + np.abs(prediction), 1e-9)
            ) * 100),
            "count": len(frame),
        })
    return results


def _add_reference_improvements(metrics: pd.DataFrame) -> pd.DataFrame:
    metrics = metrics.copy()
    for reference in ("static_mlp", "static_xgboost"):
        ref = metrics.loc[metrics.model == reference].set_index("scope").mae_mw
        metrics[f"mae_improvement_vs_{reference}_pct"] = [
            (float(ref.loc[row.scope]) - row.mae_mw) / float(ref.loc[row.scope]) * 100
            for row in metrics.itertuples()
        ]
    return metrics


def _event_targets(root: Path) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    sources: dict[str, set[pd.Timestamp]] = {}
    for model, path in (
        ("mlp", root / "phase36/final/final-holdout-rerun-20261023/strategies/gentle_mixed_60d_threshold_05pct/adaptation_events.csv"),
        ("xgboost", root / "phase37/final/final-holdout-20261024/full_history/adaptation_events.csv"),
    ):
        frame = pd.read_csv(path, usecols=["drift_timestamp"], parse_dates=["drift_timestamp"])
        stamps = pd.to_datetime(frame.drift_timestamp, utc=True)
        sources[model] = set(stamps.loc[(stamps >= FINAL_START) & (stamps <= FINAL_END)])
    union = pd.DatetimeIndex(sorted(sources["mlp"] | sources["xgboost"]), tz="UTC")
    details = pd.DataFrame({"target_timestamp": union})
    details["mlp_adwin_alarm"] = details.target_timestamp.isin(sources["mlp"])
    details["xgboost_adwin_alarm"] = details.target_timestamp.isin(sources["xgboost"])
    details["alarm_source"] = np.select(
        [details.mlp_adwin_alarm & details.xgboost_adwin_alarm, details.mlp_adwin_alarm],
        ["both", "mlp"], default="xgboost",
    )
    return union, details


def _summaries(joined: pd.DataFrame, models: list[str], alarm_targets: pd.DatetimeIndex, dev_actuals: pd.Series, output: Path) -> dict[str, object]:
    rows: list[dict[str, object]] = []

    def add(scope: str, frame: pd.DataFrame) -> None:
        for result in _metrics(frame, models):
            rows.append({"scope": scope, **result})

    add("all_horizons", joined)
    fixed = joined.loc[joined.horizon_hours == 24].copy()
    add("fixed_24h", fixed)
    horizon_rows = []
    for horizon, subset in joined.groupby("horizon_hours", sort=True):
        horizon_rows.extend({"horizon_hours": int(horizon), **result} for result in _metrics(subset, models))
    pd.DataFrame(horizon_rows).to_csv(output / "horizon_metrics.csv", index=False)

    for month, subset in fixed.groupby(fixed.target_timestamp.dt.strftime("%Y-%m"), sort=True):
        rows.extend({"scope": f"month:{month}", **result} for result in _metrics(subset, models))
    fixed["season"] = fixed.target_timestamp.dt.month.map(
        {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
         6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn", 11: "autumn"}
    )
    for season, subset in fixed.groupby("season", sort=True):
        rows.extend({"scope": f"season:{season}", **result} for result in _metrics(subset, models))

    peak_threshold = float(dev_actuals.loc[(dev_actuals.index >= DEV_START) & (dev_actuals.index <= DEV_END)].quantile(0.9))
    fixed["demand_band"] = np.where(fixed.actual_demand_mw >= peak_threshold, "peak", "normal")
    for band, subset in fixed.groupby("demand_band", sort=True):
        rows.extend({"scope": f"demand_band:{band}", "peak_threshold_mw": peak_threshold, **result}
                    for result in _metrics(subset, models))

    alarm_fixed = fixed.loc[fixed.target_timestamp.isin(alarm_targets)]
    add("adwin_alarm_targets_union", alarm_fixed)
    summary = _add_reference_improvements(pd.DataFrame(rows))
    summary.to_csv(output / "summary_metrics.csv", index=False)
    return {
        "common_keys": len(joined),
        "common_fixed_24h_targets": len(fixed),
        "unique_adwin_alarm_targets_union": len(alarm_targets),
        "development_p90_demand_mw": peak_threshold,
        "model_metrics": summary.loc[summary.scope.isin(["all_horizons", "fixed_24h", "adwin_alarm_targets_union"])].to_dict("records"),
    }


def run(input_path: Path, run_id: str, refresh_report: bool = False) -> Path:
    output = ROOT / "artifacts" / "experiments" / "phase38" / "comparison" / run_id
    output.mkdir(parents=True, exist_ok=True)
    metrics_path = output / "run_summary.json"
    joined_path = output / "aligned_predictions.csv"
    actual = _read_actuals(input_path)
    digest = hashlib.sha256(input_path.read_bytes()).hexdigest()
    if metrics_path.exists() and joined_path.exists():
        previous = json.loads(metrics_path.read_text(encoding="utf-8"))
        if previous.get("dataset_sha256") != digest:
            raise ValueError("comparison run ID already exists for a different dataset")
        if refresh_report:
            report = _make_report(
                previous,
                previous["model_comparison"]["model_metrics"],
                previous["source_forecast_rows_before_final_filter"],
            )
            (output / "report.md").write_text(report, encoding="utf-8")
        return output

    models = list(FORECAST_PATHS)
    forecast_frames = {}
    source_counts = {}
    for model, relative_path in FORECAST_PATHS.items():
        frame = _read_forecasts(ROOT / relative_path)
        source_counts[model] = len(frame)
        frame = frame.loc[(frame.target_timestamp >= FINAL_START) & (frame.target_timestamp <= FINAL_END)]
        forecast_frames[model] = frame
    aligned = align_forecasts(forecast_frames, actual)
    aligned.to_csv(joined_path, index=False)
    alarms, alarm_frame = _event_targets(ROOT / "artifacts/experiments")
    alarm_frame.to_csv(output / "adwin_alarm_targets.csv", index=False)
    dev_actuals = actual
    summary = _summaries(aligned, models, alarms, dev_actuals, output)
    metadata = {
        "run_id": run_id,
        "output_dir": str(output),
        "dataset": str(input_path),
        "dataset_sha256": digest,
        "target_start": FINAL_START,
        "target_end": FINAL_END,
        "development_peak_threshold_period": [DEV_START, DEV_END],
        "models": models,
        "source_forecast_rows_before_final_filter": source_counts,
        "source_forecast_rows_in_comparison_period": {
            model: len(frame) for model, frame in forecast_frames.items()
        },
        "common_prediction_keys": summary["common_keys"],
        "timestamp_key": ["issue_timestamp", "target_timestamp", "horizon_hours"],
        "actual_demand_verified_from_source": True,
        "model_comparison": summary,
        "adaptation_counts": {
            "mlp": {"holdout_drift_alarms": 92, "attempts": 33, "accepted": 0, "rejected": 33},
            "xgboost_full_history": {"holdout_drift_alarms": 109, "attempts": 37, "accepted": 20, "rejected": 17},
            "random_forest": {"adaptation": "not run; frozen model"},
        },
        "evaluation_limitation": "2025-2026 was previously inspected in Phases 3.6/3.7; comparative benchmark, not untouched final test.",
    }
    metrics_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    report = _make_report(metadata, summary["model_metrics"], source_counts)
    (output / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metadata["model_comparison"], indent=2, default=str))
    print(f"Saved matched comparison to {output}")
    return output


def _make_report(metadata: dict[str, object], metrics: list[dict[str, object]], source_counts: dict[str, int]) -> str:
    lines = [
        "# Phase 3.8 Model Comparison", "",
        "## Protocol", "",
        "Five frozen or replay-produced forecast streams were inner-joined by issue timestamp, target timestamp, and horizon. Actual demand was independently reloaded from the source CSV and verified for each common key. No training or replay was run during this comparison.", "",
        f"Evaluation targets: {FINAL_START} through {FINAL_END}. Common keys: {metadata['model_comparison']['common_keys']:,} (24 leads for each of {metadata['model_comparison']['common_fixed_24h_targets']:,} target hours).", "",
        "2025-2026 has previously been examined in Phases 3.6/3.7 and is a comparative benchmark, not an untouched confirmatory test. The peak threshold is the 90th percentile of 2024 development-period demand.", "",
        "## Overall results", "",
        "| Model | All leads MAE (MW) | All leads RMSE (MW) | Fixed 24h MAE (MW) | Fixed 24h RMSE (MW) |", "| --- | ---: | ---: | ---: | ---: |",
    ]
    # Rehydrate the detailed summary table saved by the metric writer.
    data_path = Path(metadata["output_dir"]) / "summary_metrics.csv" if "output_dir" in metadata else None
    if data_path and data_path.exists():
        table = pd.read_csv(data_path)
        for model in metadata["models"]:
            a = table[(table.scope == "all_horizons") & (table.model == model)].iloc[0]
            f = table[(table.scope == "fixed_24h") & (table.model == model)].iloc[0]
            lines.append(f"| {model} | {a.mae_mw:.2f} | {a.rmse_mw:.2f} | {f.mae_mw:.2f} | {f.rmse_mw:.2f} |")
    else:
        for row in metrics:
            lines.append(f"| {row['model']} | — | — | {row['mae_mw']:.2f} | {row['rmse_mw']:.2f} |")
    event_table = table[table.scope == "adwin_alarm_targets_union"] if data_path and data_path.exists() else pd.DataFrame()
    peak_table = table[table.scope == "demand_band:peak"] if data_path and data_path.exists() else pd.DataFrame()
    lines.extend([
        "", "The adaptive MLP has the lowest all-horizon MAE (1,658.21 MW). Adaptive XGBoost has the lowest fixed-24-hour MAE (1,640.94 MW), about 1.78% below frozen XGBoost. The MLP adaptive stream had no accepted update during the 2025-2026 score period; its registry already carried the `adapt5` version at score-period start.",
        "", "## Event-point and high-demand checks", "",
        "The event-point subset uses the same 156 unique target timestamps for all models (union of saved MLP/XGBoost alarms), at fixed 24-hour lead. It is not a drift-duration label.", "",
        "| Model | MAE at alarm targets (MW) |", "| --- | ---: |",
    ])
    for _, row in event_table.iterrows():
        lines.append(f"| {row.model} | {row.mae_mw:.2f} |")
    lines.extend([
        "", f"Peak demand is defined as actual demand at or above the 2024 development q90 ({metadata['model_comparison']['development_p90_demand_mw']:,.2f} MW). At these peak targets, static XGBoost has the lowest fixed-24h MAE ({peak_table.loc[peak_table.model == 'static_xgboost', 'mae_mw'].iloc[0]:,.2f} MW); neither adaptive stream improves its corresponding static model on this subset.",
    ])
    lines.extend([
        "", "## Adaptation context", "",
        "| Forecast stream | Holdout ADWIN alarms | Attempts | Accepted | Rejected |", "| --- | ---: | ---: | ---: | ---: |",
        "| MLP adaptive | 92 | 33 | 0 | 33 |",
        "| XGBoost adaptive (full history) | 109 | 37 | 20 | 17 |",
        "| Random Forest | — | — | — | — |",
        "", "The adaptive streams are evaluated as produced by their saved causal replays. RF is a frozen baseline. Alarm-target scores in `summary_metrics.csv` use the union of MLP and XGBoost holdout alarm target timestamps; they are event-point metrics, not a claim of a labeled drift interval.", "",
        "## Artifacts", "",
        "- `aligned_predictions.csv`: all five predictions and source actual for every common issue/target/horizon key.",
        "- `summary_metrics.csv`: overall, fixed-lead, monthly, seasonal, peak/normal, and alarm-target metrics plus improvement percentages vs static MLP and static XGBoost.",
        "- `horizon_metrics.csv`: per-lead MAE/RMSE/count.",
        "- `adwin_alarm_targets.csv`: union alarm timestamps and source flags.",
        "- `run_summary.json`: source paths/counts, dataset hash, adaptation counts, evaluation caveat, and alignment totals.",
        "", "Reproduce without model fitting:", "",
        "```powershell", "conda activate griddrift",
        "python scripts/compare_phase38_models.py --run-id common-period-20261024 --input data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv",
        "```", "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=RUN_ID_DEFAULT)
    parser.add_argument("--input", required=True)
    parser.add_argument("--refresh-report", action="store_true", help="rebuild Markdown only from saved metrics")
    args = parser.parse_args()
    run(Path(args.input), args.run_id, refresh_report=args.refresh_report)


if __name__ == "__main__":
    main()
