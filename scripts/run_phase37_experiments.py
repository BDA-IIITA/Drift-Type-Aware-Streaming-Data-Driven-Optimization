"""Reproducible direct-XGBoost and ADWIN experiments; writes only under phase37."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_drift_replay import _demand

from griddrift.drift import ADWINConfig
from griddrift.forecasting.mlp import DirectMLPForecaster
from griddrift.forecasting.registry import ForecastRecord, ForecastRegistry
from griddrift.forecasting.xgboost import DirectXGBoostForecaster, train_direct_xgboost
from griddrift.simulation.xgboost_replay import (
    XGBoostReplayConfig,
    XGBoostReplayResult,
    run_xgboost_replay,
)

DEV_START = pd.Timestamp("2024-04-09T00:00:00Z")
DEV_END = pd.Timestamp("2024-12-31T23:00:00Z")
FINAL_START = pd.Timestamp("2025-01-01T00:00:00Z")
FINAL_END_DEFAULT = pd.Timestamp("2026-10-20T23:00:00Z")
PARAMETERS = {
    "n_estimators": 80, "max_depth": 4, "learning_rate": 0.05,
    "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 2.0,
    "n_jobs": 4, "tree_method": "hist", "random_state": 42,
    "multi_strategy": "one_output_per_tree",
}


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"mae_mw": float("nan"), "rmse_mw": float("nan"), "count": 0.0}
    errors = frame["absolute_error_mw"].to_numpy(dtype=float)
    return {"mae_mw": float(errors.mean()), "rmse_mw": float(np.sqrt(np.mean(errors**2))), "count": float(len(errors))}


def _run_case(demand: pd.Series, model, args, strategy: str, window: int | None, start: pd.Timestamp, end: pd.Timestamp, evaluation_start: pd.Timestamp):
    config = XGBoostReplayConfig(
        strategy=strategy,
        recent_window_days=window or 60,
        cooldown_days=7,
        validation_days=7,
        min_error_increase_mw=250,
        min_error_increase_fraction=0.10,
        min_validation_improvement_fraction=args.validation_threshold,
        random_state=42,
        adwin=ADWINConfig(delta=0.002, clock=32, min_window_length=5),
    )
    return run_xgboost_replay(
        demand, model, start=start, end=end, evaluation_start=evaluation_start,
        config=config, parameters=PARAMETERS,
    )


def _save_replay(output: Path, name: str, result, metadata: dict[str, object]) -> dict[str, object]:
    folder = output / name
    folder.mkdir(parents=True, exist_ok=True)
    result.static_errors.to_csv(folder / "static_24h_errors.csv", index=False)
    result.adaptive_errors.to_csv(folder / "adaptive_24h_errors.csv", index=False)
    result.adaptation_events.to_csv(folder / "adaptation_events.csv", index=False)
    result.static_registry.save_csv(folder / "static_forecast_registry.csv")
    result.adaptive_registry.save_csv(folder / "adaptive_forecast_registry.csv")
    for version, candidate in result.candidate_models.items():
        candidate.save(folder / "candidate_models" / version)
    adaptive_metrics = _metrics(result.adaptive_errors)
    static_metrics = _metrics(result.static_errors)
    result.final_model.save(folder / "final_adaptive_model", adaptive_metrics)
    events = result.adaptation_events
    decisions = events.get("decision", pd.Series(dtype=str))
    attempts = ~decisions.str.startswith("skipped") if not events.empty else pd.Series(dtype=bool)
    event_times = pd.to_datetime(events["drift_timestamp"], utc=True) if not events.empty else pd.Series(dtype="datetime64[ns, UTC]")
    eval_start = pd.Timestamp(metadata.get("evaluation_start", metadata.get("replay_start")))
    eval_start = eval_start.tz_localize("UTC") if eval_start.tz is None else eval_start.tz_convert("UTC")
    warmup_mask = event_times < eval_start if not events.empty else pd.Series(dtype=bool)
    holdout_mask = event_times >= eval_start if not events.empty else pd.Series(dtype=bool)

    def event_counts(mask: pd.Series) -> dict[str, int]:
        if events.empty:
            return {"alarms": 0, "attempts": 0, "accepted": 0, "rejected": 0}
        subset = events.loc[mask]
        subset_decisions = subset["decision"].astype(str)
        return {
            "alarms": len(subset),
            "attempts": int((~subset_decisions.str.startswith("skipped")).sum()),
            "accepted": int((subset_decisions == "accepted").sum()),
            "rejected": int(subset_decisions.str.startswith("rejected").sum()),
        }
    warmup_counts = event_counts(warmup_mask)
    holdout_counts = event_counts(holdout_mask)
    summary: dict[str, object] = {
        **metadata,
        "static_metrics": static_metrics,
        "adaptive_metrics": adaptive_metrics,
        "drift_alarms": len(result.adaptation_events),
        "adaptation_attempts": int(attempts.sum()) if len(attempts) else 0,
        "accepted_adaptations": int((decisions == "accepted").sum()) if len(decisions) else 0,
        "rejected_adaptations": int(decisions.str.startswith("rejected").sum()) if len(decisions) else 0,
        "warmup_events_before_evaluation_start": warmup_counts,
        "holdout_events_from_evaluation_start": holdout_counts,
        "runtime_seconds": result.runtime_seconds,
        "inference_runtime_seconds": result.inference_runtime_seconds,
        "candidate_training_runtime_seconds": float(result.adaptation_events.get("training_seconds", pd.Series(dtype=float)).sum()) if not result.adaptation_events.empty else 0.0,
        "final_model_version": result.final_model.model_version,
    }
    _write_json(folder / "summary.json", summary)
    return summary


def _load_replay_case(folder: Path, summary: dict[str, object]) -> XGBoostReplayResult:
    def load_errors(path: Path) -> pd.DataFrame:
        frame = pd.read_csv(path)
        for column in ("issue_timestamp", "target_timestamp"):
            frame[column] = pd.to_datetime(frame[column], utc=True)
        return frame

    def load_registry(path: Path) -> ForecastRegistry:
        frame = pd.read_csv(path)
        records = []
        for row in frame.itertuples(index=False):
            records.append(ForecastRecord(
                pd.Timestamp(row.issue_timestamp), pd.Timestamp(row.target_timestamp),
                int(row.horizon_hours), float(row.predicted_demand_mw), str(row.model_version),
            ))
        return ForecastRegistry(records)

    events_path = folder / "adaptation_events.csv"
    events = pd.read_csv(events_path) if events_path.stat().st_size else pd.DataFrame()
    if not events.empty:
        events["drift_timestamp"] = pd.to_datetime(events["drift_timestamp"], utc=True)
    candidate_models = {
        child.name: DirectXGBoostForecaster.load(child)
        for child in (folder / "candidate_models").iterdir()
        if child.is_dir()
    } if (folder / "candidate_models").exists() else {}
    return XGBoostReplayResult(
        static_errors=load_errors(folder / "static_24h_errors.csv"),
        adaptive_errors=load_errors(folder / "adaptive_24h_errors.csv"),
        adaptation_events=events,
        static_registry=load_registry(folder / "static_forecast_registry.csv"),
        adaptive_registry=load_registry(folder / "adaptive_forecast_registry.csv"),
        final_model=DirectXGBoostForecaster.load(folder / "final_adaptive_model"),
        candidate_models=candidate_models,
        runtime_seconds=float(summary["runtime_seconds"]),
        inference_runtime_seconds=float(summary["inference_runtime_seconds"]),
    )


def _mlp_final_comparison(output: Path, mlp_results: Path, xgb_static: pd.DataFrame,
                          xgb_adaptive: pd.DataFrame, actual_demand: pd.Series) -> pd.DataFrame:
    static = pd.read_csv(mlp_results / "static_24h_errors.csv", parse_dates=["issue_timestamp", "target_timestamp"])
    adaptive = pd.read_csv(mlp_results / "adaptive_24h_errors.csv", parse_dates=["issue_timestamp", "target_timestamp"])
    for frame in (static, adaptive):
        frame["issue_timestamp"] = pd.to_datetime(frame["issue_timestamp"], utc=True)
        frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], utc=True)
        frame.drop(frame.index[frame["target_timestamp"] < FINAL_START], inplace=True)
    pairs = {"static_mlp": static, "adaptive_mlp": adaptive,
             "static_xgboost": xgb_static, "adaptive_xgboost": xgb_adaptive}
    reference_keys = pd.MultiIndex.from_frame(static[["issue_timestamp", "target_timestamp"]])
    for name, frame in pairs.items():
        keys = pd.MultiIndex.from_frame(frame[["issue_timestamp", "target_timestamp"]])
        if not keys.equals(reference_keys):
            raise ValueError(f"{name} does not cover the exact same issue/target timestamps as frozen MLP")
    # Join all strategies on the actual fixed-lead target timestamps, and verify source values.
    merged = None
    for name, frame in pairs.items():
        if frame.empty:
            raise ValueError(f"no final holdout rows available for {name}")
        columns = ["issue_timestamp", "target_timestamp", "actual_demand_mw", "predicted_demand_mw", "absolute_error_mw", "model_version"]
        renamed = frame.loc[:, columns].rename(columns={
            "actual_demand_mw": f"{name}_actual_mw",
            "predicted_demand_mw": f"{name}_predicted_mw",
            "absolute_error_mw": f"{name}_absolute_error_mw",
            "model_version": f"{name}_model_version",
        })
        merged = renamed if merged is None else merged.merge(renamed, on=["issue_timestamp", "target_timestamp"], how="inner", validate="one_to_one")
    actual_cols = [c for c in merged.columns if c.endswith("_actual_mw")]
    if not np.allclose(merged[actual_cols].to_numpy(), merged[actual_cols[0]].to_numpy()[:, None], atol=1e-6):
        raise ValueError("MLP/XGBoost actual demand values differ after target timestamp alignment")
    expected = actual_demand.reindex(pd.DatetimeIndex(merged["target_timestamp"])).to_numpy(dtype=float)
    if np.isnan(expected).any() or not np.allclose(merged[actual_cols[0]].to_numpy(dtype=float), expected, atol=1e-6):
        raise ValueError("forecast actuals do not match the source ENTSO-E demand CSV")
    if not (merged["target_timestamp"] - merged["issue_timestamp"] == pd.Timedelta(hours=24)).all():
        raise ValueError("final comparison contains a forecast that is not exactly 24 hours ahead")
    output.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output / "mlp_xgboost_final_aligned_predictions.csv", index=False)
    return merged


def _save_plot(fig, plots: Path, name: str) -> None:
    plots.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots / f"{name}.png", dpi=240, bbox_inches="tight")
    fig.savefig(plots / f"{name}.svg", bbox_inches="tight")


def _final_report(output: Path, merged: pd.DataFrame, summaries: list[dict[str, object]], events: pd.DataFrame,
                  metadata: dict[str, object]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots = output / "plots"
    models = ["static_mlp", "adaptive_mlp", "static_xgboost", "adaptive_xgboost"]
    rows = []
    for name in models:
        error = merged[f"{name}_absolute_error_mw"].to_numpy(dtype=float)
        rows.append({"model": name, "mae_mw": float(error.mean()), "rmse_mw": float(np.sqrt(np.mean(error**2)))})
    metric_frame = pd.DataFrame(rows)
    metric_frame.to_csv(output / "final_model_metrics.csv", index=False)
    metric_frame.to_json(output / "final_model_metrics.json", orient="records", indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, title in zip(axes, ("mae_mw", "rmse_mw"), ("MAE comparison", "RMSE comparison")):
        ax.bar(metric_frame["model"], metric_frame[metric], color=["#8094a8", "#38a5e6", "#c58b3b", "#4caf76"])
        ax.set_title(title); ax.set_ylabel("Forecast error (MW; lower is better)")
        ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); _save_plot(fig, plots, "mlp_vs_xgboost_forecasting_performance"); plt.close(fig)

    view = merged.sort_values("target_timestamp").iloc[: 30 * 24]
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(view["target_timestamp"], view["static_mlp_actual_mw"], color="black", lw=1.2, label="Actual")
    for name, label, color in (("static_mlp", "Frozen MLP", "#8b929b"),
                               ("adaptive_mlp", "Adaptive MLP", "#2472b8"),
                               ("static_xgboost", "Frozen XGBoost", "#d89026"),
                               ("adaptive_xgboost", "Adaptive XGBoost", "#2e9d62")):
        ax.plot(view["target_timestamp"], view[f"{name}_predicted_mw"], lw=1, label=label, color=color)
    ax.set_title("Actual vs predicted DE-LU demand — first 30 final-holdout days")
    ax.set_ylabel("Demand (MW)"); ax.set_xlabel("Target timestamp (UTC)"); ax.legend(ncol=3); ax.grid(alpha=.2)
    fig.tight_layout(); _save_plot(fig, plots, "actual_vs_predicted_demand"); plt.close(fig)

    # Event stream for the selected adaptive XGB model. Errors are fixed 24-hour lead only.
    adaptive = merged[["target_timestamp", "adaptive_xgboost_absolute_error_mw"]].copy()
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.plot(adaptive["target_timestamp"], adaptive["adaptive_xgboost_absolute_error_mw"], lw=.8, label="Adaptive XGBoost absolute error")
    if not events.empty:
        times = pd.to_datetime(events["drift_timestamp"], utc=True)
        shown = events.loc[times >= FINAL_START]
        for event in shown.itertuples(index=False):
            at = pd.Timestamp(event.drift_timestamp)
            ax.axvline(at, color="#e5484d", alpha=.35, ls="--")
            if getattr(event, "decision", "") == "accepted":
                ax.scatter([at], [adaptive.loc[adaptive["target_timestamp"] >= at, "adaptive_xgboost_absolute_error_mw"].iloc[0]], marker="D", color="#9b59b6", label="Accepted update")
    ax.set_title("ADWIN changes and accepted XGBoost updates — final holdout")
    ax.set_ylabel("Absolute error (MW)"); ax.set_xlabel("Actual target timestamp (UTC)"); ax.grid(alpha=.2); ax.legend()
    fig.tight_layout(); _save_plot(fig, plots, "adwin_drift_detection"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 5))
    window = 24 * 7
    for name, label, color in (("static_xgboost_absolute_error_mw", "Frozen XGBoost", "#d89026"),
                               ("adaptive_xgboost_absolute_error_mw", "ADWIN-adaptive XGBoost", "#2e9d62")):
        ax.plot(merged["target_timestamp"], merged[name].rolling(window, min_periods=24).mean(), label=label, color=color)
    ax.set_title("Static vs adaptive XGBoost — rolling 7-day fixed-lead MAE")
    ax.set_ylabel("Rolling mean absolute error (MW)"); ax.set_xlabel("Target timestamp (UTC)"); ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); _save_plot(fig, plots, "static_vs_adaptive_xgboost"); plt.close(fig)

    if not events.empty:
        target = pd.to_datetime(merged["target_timestamp"], utc=True)
        post_masks = []
        for event in events.itertuples(index=False):
            alarm = pd.Timestamp(event.drift_timestamp)
            post_masks.append((target >= alarm + pd.Timedelta(hours=24)) & (target <= alarm + pd.Timedelta(days=7)))
        mask = np.logical_or.reduce(post_masks) if post_masks else np.zeros(len(merged), dtype=bool)
    else:
        mask = np.zeros(len(merged), dtype=bool)
    selected = merged.loc[mask]
    drift_metrics = []
    for name in models:
        errs = selected[f"{name}_absolute_error_mw"].to_numpy(dtype=float)
        drift_metrics.append({"model": name, "post_drift_mae_mw": float(errs.mean()) if len(errs) else float("nan"), "count": len(errs)})
    pd.DataFrame(drift_metrics).to_csv(output / "post_drift_metrics.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([r["model"] for r in drift_metrics], [r["post_drift_mae_mw"] for r in drift_metrics], color=["#8094a8", "#38a5e6", "#c58b3b", "#4caf76"])
    ax.set_title("Forecast error in the 7 days after detected changes")
    ax.set_ylabel("MAE (MW; lower is better)"); ax.tick_params(axis="x", rotation=25); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); _save_plot(fig, plots, "drift_period_performance"); plt.close(fig)

    strategy_rows = []
    for summary in summaries:
        strategy_rows.append({"strategy": summary.get("strategy_name"), "mae_mw": summary["adaptive_metrics"]["mae_mw"],
                              "runtime_seconds": summary["runtime_seconds"], "accepted": summary["accepted_adaptations"],
                              "attempts": summary["adaptation_attempts"]})
    strategy_frame = pd.DataFrame(strategy_rows)
    strategy_frame.to_csv(output / "adaptation_strategy_comparison.csv", index=False)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(strategy_frame["strategy"], strategy_frame["mae_mw"], color="#4a90c2", label="MAE (MW)")
    ax1.set_ylabel("MAE (MW)"); ax1.tick_params(axis="x", rotation=20); ax1.grid(axis="y", alpha=.25)
    ax2 = ax1.twinx(); ax2.plot(strategy_frame["strategy"], strategy_frame["runtime_seconds"], color="#d9534f", marker="o", label="Runtime (s)")
    ax2.set_ylabel("Runtime (seconds)"); ax1.set_title("Full-history vs recent-window adaptation")
    fig.tight_layout(); _save_plot(fig, plots, "adaptation_strategy_comparison"); plt.close(fig)

    baseline = metric_frame.set_index("model")
    md = ["# GridDRIFT Phase 3.7 — XGBoost and drift adaptation", "",
          "## Protocol", "",
          "XGBoost 3.2 uses native multi-output regression (`one_output_per_tree`) for all 24 direct future intervals and GridDRIFT's 168-hour past-only lag/rolling/calendar feature pipeline. Its initial training cutoff matches the frozen MLP issue-time cutoff. ADWIN monitors only the fixed 24-hour-ahead adaptive-XGBoost absolute error. Candidate training uses labels whose complete 24-hour target window was observed at alarm time, with a 24-hour train/validation embargo and a seven-day cooldown.", "",
          "Final evaluation is the untouched 2025-01-01 through dataset-end period. Development configuration selection is completed separately. MLP and XGBoost predictions are compared only after inner-joining issue and target timestamps and verifying identical actual demand. The Phase 3.6 adaptive MLP's score includes its one accepted 2024 warm-up update; it accepted no further update in the final holdout.", "",
          "## Final holdout results", "", "| Model | MAE (MW) | RMSE (MW) | vs frozen MLP MAE |", "|---|---:|---:|---:|"]
    baseline_mae = float(baseline.loc["static_mlp", "mae_mw"])
    for row in rows:
        delta = (row["mae_mw"] - baseline_mae) / baseline_mae * 100
        md.append(f"| {row['model']} | {row['mae_mw']:.2f} | {row['rmse_mw']:.2f} | {delta:+.2f}% |")
    md.extend(["", f"Development selected: `{metadata['development_selected_strategy']}` with a {metadata['selected_validation_threshold_fraction']:.0%} validation-improvement threshold; the final holdout was not used to choose it."])
    md.extend(["", "## Adaptation activity", "", "| Strategy | MAE (MW) | RMSE (MW) | ADWIN alarms | attempts | accepted | rejected | warm-up accepted | final accepted | runtime (s) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for summary in summaries:
        warmup = summary.get("warmup_events_before_evaluation_start", {}).get("accepted", 0)
        holdout = summary.get("holdout_events_from_evaluation_start", {}).get("accepted", summary["accepted_adaptations"])
        md.append(f"| {summary.get('strategy_name')} | {summary['adaptive_metrics']['mae_mw']:.2f} | {summary['adaptive_metrics']['rmse_mw']:.2f} | {summary['drift_alarms']} | {summary['adaptation_attempts']} | {summary['accepted_adaptations']} | {summary['rejected_adaptations']} | {warmup} | {holdout} | {summary['runtime_seconds']:.1f} |")
    md.extend(["", "## Interpretation and limitations", "",
               "A lower MAE/RMSE is better; adaptation is useful only if the measured holdout supports it. The best strategy is selected on the 2024 development period, not on the final holdout. XGBoost candidates are retrained from scratch; warm-start tree growth was not enabled because it can grow the ensemble without a controlled validation budget. All six figures have PNG and SVG variants and can be regenerated from saved CSVs by rerunning the report helper in this script (without fitting models).", "",
               "## Artifacts", "", "Predictions and exact timestamp joins: `mlp_xgboost_final_aligned_predictions.csv`; model metrics: `final_model_metrics.csv/json`; adaptation events: `adaptation_events.csv`; plot files: `plots/`; strategy metrics: `adaptation_strategy_comparison.csv`.",
               "", "## Reproduction commands", "", "Run these from the repository root in PowerShell after activating the `griddrift` environment:", "",
               "```powershell",
               f"python scripts/run_phase37_experiments.py --profile static --run-id {metadata['static_run_id']} --input {metadata['dataset']} --mlp-model artifacts/models/de-lu-mlp --output artifacts/experiments/phase37",
               f"python scripts/run_phase37_experiments.py --profile development --run-id {metadata['development_run_id']} --input {metadata['dataset']} --mlp-model artifacts/models/de-lu-mlp --xgb-model {metadata['static_model_directory']}",
               f"python scripts/run_phase37_experiments.py --profile final --run-id {metadata['run_id']} --input {metadata['dataset']} --mlp-model artifacts/models/de-lu-mlp --xgb-model {metadata['static_model_directory']} --selection {metadata['development_selection_path']} --phase36-results {metadata['mlp_phase36_results_path']}",
               f"python scripts/run_phase37_experiments.py --profile report --run-id {metadata['run_id']} --input {metadata['dataset']} --output artifacts/experiments/phase37",
               "```"])
    (output / "report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("static", "short", "development", "threshold", "final", "report"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--mlp-model", default="artifacts/models/de-lu-mlp")
    parser.add_argument("--xgb-model")
    parser.add_argument("--output", default="artifacts/experiments/phase37")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--end", help="Inclusive end for short/development/final replay")
    parser.add_argument("--validation-threshold", type=float, default=0.02)
    parser.add_argument("--selection", help="Development selection JSON required for final")
    parser.add_argument("--strategy", choices=("full_history", "recent"), help="Single strategy for threshold development runs")
    parser.add_argument("--window-days", type=int, help="Recent window for a threshold development run")
    parser.add_argument("--phase36-results", default="artifacts/experiments/phase36/final/final-holdout-rerun-20261023/strategies/gentle_mixed_60d_threshold_05pct")
    args = parser.parse_args()
    demand = _demand(args.input)
    output = Path(args.output)
    run_root = output / args.profile / args.run_id
    if args.profile == "report":
        final_root = output / "final" / args.run_id
        merged = pd.read_csv(final_root / "mlp_xgboost_final_aligned_predictions.csv", parse_dates=["issue_timestamp", "target_timestamp"])
        merged["issue_timestamp"] = pd.to_datetime(merged["issue_timestamp"], utc=True)
        merged["target_timestamp"] = pd.to_datetime(merged["target_timestamp"], utc=True)
        summaries = json.loads((final_root / "metrics.json").read_text(encoding="utf-8"))
        events = pd.read_csv(final_root / "adaptation_events.csv")
        events["drift_timestamp"] = pd.to_datetime(events["drift_timestamp"], utc=True)
        metadata = json.loads((final_root / "run_metadata.json").read_text(encoding="utf-8"))
        _final_report(final_root, merged, summaries, events, metadata)
        print(f"Regenerated report and plots from saved CSVs under {final_root}")
        return
    if run_root.exists() and any(run_root.iterdir()) and args.profile not in {"development", "threshold", "final"}:
        parser.error(f"run directory is non-empty and will not be overwritten: {run_root}; use a new --run-id")
    mlp = DirectMLPForecaster.load(args.mlp_model)
    params = dict(PARAMETERS)
    run_root.mkdir(parents=True, exist_ok=True)

    if args.profile == "static":
        start = time.perf_counter()
        training = train_direct_xgboost(
            demand, feature_spec=mlp.feature_spec,
            train_end_issue_time=mlp.train_end_issue_time, random_state=42,
            model_version="xgb-de-lu-static-42", parameters=params,
        )
        model_path = Path(args.xgb_model) if args.xgb_model else run_root / "static_xgboost_model"
        training.model.save(model_path, training.validation_metrics)
        metadata = {
            "profile": "static", "dataset": str(args.input), "model_type": "native XGBoost multi-output, one_output_per_tree; 24 outputs",
            "version": "3.2.0", "training_cutoff_issue_timestamp": mlp.train_end_issue_time.isoformat(),
            "training_samples": training.train_samples, "validation_samples": training.validation_samples,
            "final_model_fit_samples": training.final_fit_samples,
            "validation_train_end_issue_timestamp": training.validation_train_end_issue_time.isoformat(),
            "validation_first_issue_timestamp": training.validation_first_issue_time.isoformat(),
            "validation_metrics": training.validation_metrics, "parameters": params,
            "seed": 42, "training_seconds": training.training_seconds,
            "feature_spec": {"lookback_hours": mlp.feature_spec.lookback_hours, "horizon_hours": mlp.feature_spec.horizon_hours},
        }
        _write_json(run_root / "static_training.json", metadata)
        print(json.dumps(metadata, indent=2))
        return

    if not args.xgb_model:
        parser.error("--xgb-model must point to a saved Phase 3.7 static XGBoost model")
    from griddrift.forecasting.xgboost import DirectXGBoostForecaster
    model = DirectXGBoostForecaster.load(args.xgb_model)
    start = DEV_START
    end = pd.Timestamp(args.end, tz="UTC") if args.end else (DEV_START + pd.Timedelta(days=30) - pd.Timedelta(hours=1) if args.profile == "short" else (DEV_END if args.profile in {"development", "threshold"} else FINAL_END_DEFAULT))
    evaluation_start = DEV_START if args.profile in {"short", "development", "threshold"} else FINAL_START
    if args.profile == "short":
        cases = [("recent_60d_short", "recent", 60)]
    elif args.profile == "development":
        cases = [("full_history", "full_history", None), ("recent_60d", "recent", 60),
                 ("recent_120d", "recent", 120), ("recent_180d", "recent", 180)]
    elif args.profile == "threshold":
        if not args.strategy:
            parser.error("--strategy is required for a threshold development run")
        if args.strategy == "recent" and not args.window_days:
            parser.error("--window-days is required for a recent strategy")
        name = f"recent_{args.window_days}d" if args.strategy == "recent" else "full_history"
        threshold_label = f"{args.validation_threshold * 100:g}".replace(".", "p")
        cases = [(f"threshold_{name}_{threshold_label}pct", args.strategy, args.window_days)]
    else:
        if not args.selection:
            parser.error("--selection is required for final profile")
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        if "selected_validation_threshold_fraction" in selection:
            args.validation_threshold = float(selection["selected_validation_threshold_fraction"])
        if "best_recent_window_days" in selection:
            selected_days = int(selection["best_recent_window_days"])
        else:
            development_root = Path(args.selection).parent
            development_summaries = json.loads((development_root / "metrics.json").read_text(encoding="utf-8"))
            recent_summaries = [r for r in development_summaries if r["strategy"] == "recent"]
            selected_days = int(min(recent_summaries, key=lambda r: r["adaptive_metrics"]["mae_mw"])["window_days"])
        cases = [("full_history", "full_history", None), (f"recent_{selected_days}d", "recent", selected_days)]

    summaries = []
    replay_results = {}
    for name, strategy, window in cases:
        case_folder = run_root / name
        summary_path = case_folder / "summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            result = _load_replay_case(case_folder, summary)
            replay_results[name] = result
            summaries.append({"strategy_name": name, **summary})
            print(f"Reusing completed saved case {name} from {case_folder}", flush=True)
            continue
        print(f"Starting {name}: {start.isoformat()} through {end.isoformat()} (model fitting may take several minutes)", flush=True)
        result = _run_case(demand, model, args, strategy, window, start, end, evaluation_start)
        replay_results[name] = result
        summary = _save_replay(run_root, name, result, {
            "strategy": strategy, "window_days": window, "replay_start": start.isoformat(),
            "replay_end": end.isoformat(), "evaluation_start": evaluation_start.isoformat(),
            "adwin": {"delta": 0.002, "clock": 32, "min_window_length": 5},
            "cooldown_days": 7, "validation_threshold_fraction": args.validation_threshold,
            "parameters": params, "seed": 42,
        })
        summaries.append({"strategy_name": name, **summary})
        print(json.dumps({"strategy": name, "static": summary["static_metrics"], "adaptive": summary["adaptive_metrics"],
                          "alarms": summary["drift_alarms"], "attempts": summary["adaptation_attempts"],
                          "accepted": summary["accepted_adaptations"], "rejected": summary["rejected_adaptations"],
                          "runtime_seconds": summary["runtime_seconds"]}, indent=2), flush=True)

    pd.DataFrame(summaries).to_csv(run_root / "metrics.csv", index=False)
    _write_json(run_root / "metrics.json", summaries)
    if args.profile == "development":
        eligible = [r for r in summaries if r["adaptive_metrics"]["count"] > 0]
        chosen = min(eligible, key=lambda r: r["adaptive_metrics"]["mae_mw"])
        recent = [r for r in eligible if r["strategy"] == "recent"]
        best_recent = min(recent, key=lambda r: r["adaptive_metrics"]["mae_mw"])
        selection = {
            "best_recent_window_days": best_recent["window_days"],
            "selected_strategy": chosen["strategy_name"], "selected_on": "development only",
            "selection_metric": "adaptive fixed-24-hour MAE", "selected_development_mae_mw": chosen["adaptive_metrics"]["mae_mw"],
            "static_development_mae_mw": chosen["static_metrics"]["mae_mw"],
            "validation_threshold_fraction": args.validation_threshold,
        }
        _write_json(run_root / "selection.json", selection)
        print(f"Development selection saved to {run_root / 'selection.json'}")
    if args.profile == "final":
        final_selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        selected_strategy = final_selection["selected_strategy"]
        selected_name = selected_strategy if selected_strategy == "full_history" else f"recent_{int(cases[-1][2])}d"
        selected_result = replay_results[selected_name]
        aligned = _mlp_final_comparison(
            run_root, Path(args.phase36_results), replay_results["full_history"].static_errors,
            selected_result.adaptive_errors, demand,
        )
        selected_result.adaptation_events.to_csv(run_root / "adaptation_events.csv", index=False)
        # Per-lead error is retained separately from the fixed 24-hour ADWIN stream.
        demand_lookup = demand.rename("actual_demand_mw").rename_axis("target_timestamp").reset_index()
        horizon_rows = []
        for name, result in (("static_xgboost", replay_results["full_history"]), ("adaptive_xgboost", selected_result)):
            registry = result.static_registry if name == "static_xgboost" else result.adaptive_registry
            frame = registry.to_frame()
            frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], utc=True)
            joined = frame.merge(demand_lookup, on="target_timestamp", how="inner")
            joined = joined.loc[joined["target_timestamp"] >= FINAL_START]
            joined["absolute_error_mw"] = (joined["actual_demand_mw"] - joined["predicted_demand_mw"]).abs()
            for horizon, group in joined.groupby("horizon_hours"):
                err = group["absolute_error_mw"].to_numpy(dtype=float)
                horizon_rows.append({"model": name, "horizon_hours": int(horizon), "mae_mw": float(err.mean()),
                                     "rmse_mw": float(np.sqrt(np.mean(err**2))), "count": len(err)})
        pd.DataFrame(horizon_rows).to_csv(run_root / "horizon_wise_metrics.csv", index=False)
        metadata = {
            "profile": "final", "run_id": args.run_id, "dataset": args.input,
            "dataset_start": demand.index[0].isoformat(), "dataset_end": demand.index[-1].isoformat(),
            "training_cutoff_issue_timestamp": mlp.train_end_issue_time.isoformat(),
            "replay_start": start.isoformat(), "evaluation_start": FINAL_START.isoformat(), "evaluation_end": end.isoformat(),
            "xgb_version": "3.2.0", "xgb_parameters": params, "adwin": {"delta": .002, "clock": 32, "min_window_length": 5},
            "cooldown_days": 7, "comparison_aligned_rows": len(aligned),
            "mlp_phase36_results_path": args.phase36_results,
            "best_recent_window_days": cases[-1][2],
            "development_selected_strategy": selected_strategy,
            "selected_validation_threshold_fraction": args.validation_threshold,
            "static_run_id": Path(args.xgb_model).parents[0].name,
            "static_model_directory": args.xgb_model,
            "development_run_id": Path(args.selection).parent.name,
            "development_selection_path": args.selection,
            "limitation": "Adaptation decisions occurred during 2024 development and final replay; final scores remain untouched for model/threshold selection.",
        }
        _write_json(run_root / "run_metadata.json", metadata)
        _final_report(run_root, aligned, summaries, selected_result.adaptation_events, metadata)
    print(f"Saved Phase 3.7 results under {run_root}")


if __name__ == "__main__":
    main()
