"""Run reproducible Phase 3.6 MLP-only development/final replay experiments."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from run_drift_replay import _demand

from griddrift.config import load_project_config
from griddrift.drift import ADWINConfig
from griddrift.forecasting.mlp import DirectMLPForecaster
from griddrift.simulation import ReplayConfig, StreamingReplay

DEV_START = "2024-04-09T00:00:00Z"
DEV_END = "2024-12-31T23:00:00Z"
FINAL_START = "2025-01-01T00:00:00Z"


def _strategy_cases(profile: str) -> list[dict[str, object]]:
    if profile == "threshold-sweep":
        return [
            {
                "name": f"gentle_mixed_60d_threshold_{round(threshold * 100):02d}pct",
                "strategy": "mixed_history_fine_tune",
                "window_days": 60,
                "validation_days": 7,
                "validation_periods": 1,
                "validation_threshold_fraction": threshold,
            }
            for threshold in (0.0, 0.01, 0.02, 0.05)
        ]
    cases = [
        {
            "name": "previous_validated_finetune_60d",
            "strategy": "fine_tune",
            "window_days": 60,
            "validation_days": 7,
            "validation_periods": 1,
        }
    ]
    if profile == "short":
        return cases + [
            {
                "name": "gentle_recent_30d",
                "strategy": "gentle_fine_tune",
                "window_days": 30,
                "validation_days": 7,
                "validation_periods": 1,
            },
            {
                "name": "gentle_mixed_30d",
                "strategy": "mixed_history_fine_tune",
                "window_days": 30,
                "validation_days": 7,
                "validation_periods": 1,
            },
        ]
    for days in (7, 14, 30, 60):
        cases.append(
            {
                "name": f"gentle_recent_{days}d",
                "strategy": "gentle_fine_tune",
                "window_days": days,
                "validation_days": 7,
                "validation_periods": 1,
            }
        )
    for days in (30, 60):
        cases.append(
            {
                "name": f"gentle_mixed_{days}d",
                "strategy": "mixed_history_fine_tune",
                "window_days": days,
                "validation_days": 7,
                "validation_periods": 1,
            }
        )
    cases.append(
        {
            "name": "gentle_recent_60d_validation_14d_2periods",
            "strategy": "gentle_fine_tune",
            "window_days": 60,
            "validation_days": 7,
            "validation_periods": 2,
        }
    )
    return cases


def _metric_row(errors: pd.DataFrame) -> dict[str, float]:
    if errors.empty:
        return {"mae_mw": float("nan"), "rmse_mw": float("nan"), "count": 0.0}
    absolute = errors["absolute_error_mw"].to_numpy(dtype=float)
    return {
        "mae_mw": float(absolute.mean()),
        "rmse_mw": float(np.sqrt(np.mean(np.square(absolute)))),
        "count": float(len(absolute)),
    }


def _filter_period(errors: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if errors.empty:
        return errors
    target = pd.to_datetime(errors["target_timestamp"], utc=True)
    mask = pd.Series(True, index=errors.index)
    if start:
        mask &= target >= pd.Timestamp(start)
    if end:
        mask &= target <= pd.Timestamp(end)
    return errors.loc[mask].copy()


def _accepted_adaptations(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return events
    return events.loc[events["outcome"] == "adapted"].copy()


def _before_after(events: pd.DataFrame, errors: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if events.empty or errors.empty:
        return pd.DataFrame(records)
    target = pd.to_datetime(errors["target_timestamp"], utc=True)
    for event in _accepted_adaptations(events).itertuples(index=False):
        timestamp = pd.Timestamp(event.drift_timestamp)
        before = errors.loc[
            (target >= timestamp - pd.Timedelta(days=days))
            & (target < timestamp)
            & (errors["model_version"] == event.previous_model_version)
        ]
        # Fixed 24-hour-ahead errors attributable to the newly deployed
        # version cannot arrive until one full day after the drift timestamp.
        after = errors.loc[
            (target >= timestamp + pd.Timedelta(hours=24))
            & (target <= timestamp + pd.Timedelta(days=days))
            & (errors["model_version"] == event.new_model_version)
        ]
        records.append(
            {
                "adaptation_timestamp": timestamp,
                "model_version": event.new_model_version,
                "before_days": days,
                "before_mae_mw": _metric_row(before)["mae_mw"],
                "before_count": len(before),
                "after_days": days,
                "after_mae_mw": _metric_row(after)["mae_mw"],
                "after_count": len(after),
                "after_minus_before_mae_mw": _metric_row(after)["mae_mw"] - _metric_row(before)["mae_mw"],
            }
        )
    return pd.DataFrame(records)


def _drift_period_mae(errors: pd.DataFrame, events: pd.DataFrame, days: int = 7) -> float:
    if errors.empty or events.empty:
        return float("nan")
    target = pd.to_datetime(errors["target_timestamp"], utc=True)
    mask = pd.Series(False, index=errors.index)
    for timestamp in pd.to_datetime(events["drift_timestamp"], utc=True):
        # The first newly-issued 24-hour forecast is scoreable at t+24h.
        mask |= (target >= timestamp + pd.Timedelta(hours=24)) & (
            target <= timestamp + pd.Timedelta(days=days)
        )
    return _metric_row(errors.loc[mask])["mae_mw"]


def _post_accepted_mae(errors: pd.DataFrame, events: pd.DataFrame, days: int = 7) -> float:
    frame = _before_after(events, errors, days)
    return float(frame["after_mae_mw"].mean()) if not frame.empty else float("nan")


def _save_plots(root: Path, strategy_summaries: list[dict[str, object]], results: dict[str, dict[str, object]], *, period_label: str) -> None:
    plots = root / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(strategy_summaries)
    metrics = summary.loc[summary["kind"].isin(["static", "adaptive"])].copy()
    metrics["label"] = metrics["strategy"]
    labels = metrics["label"].tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.4), 5.5))
    ax.bar(x - width / 2, metrics["mae_mw"], width, label="MAE")
    ax.bar(x + width / 2, metrics["rmse_mw"], width, label="RMSE")
    ax.set_title(f"GridDRIFT MLP adaptation comparison — {period_label}")
    ax.set_ylabel("Forecast error (MW; lower is better)")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "forecasting_model_comparison.png", dpi=220)
    fig.savefig(plots / "forecasting_model_comparison.svg")
    plt.close(fig)

    # Use fixed 24-hour-ahead scored values; these are aligned by target time.
    selected_key = next((key for key in results if key != "static"), None)
    if selected_key is not None:
        data = results[selected_key]
        static_errors = data["static_errors"].copy()
        adaptive_errors = data["adaptive_errors"].copy()
        if not static_errors.empty and not adaptive_errors.empty:
            target_col = "target_timestamp"
            start = pd.to_datetime(static_errors[target_col], utc=True).min()
            end = min(
                start + pd.Timedelta(days=30),
                pd.to_datetime(static_errors[target_col], utc=True).max(),
            )
            s = static_errors.loc[pd.to_datetime(static_errors[target_col], utc=True).between(start, end)]
            a = adaptive_errors.loc[pd.to_datetime(adaptive_errors[target_col], utc=True).between(start, end)]
            fig, ax = plt.subplots(figsize=(13, 5.5))
            ax.plot(pd.to_datetime(s[target_col], utc=True), s["actual_demand_mw"], color="#222222", lw=1.6, label="Actual demand")
            ax.plot(pd.to_datetime(s[target_col], utc=True), s["predicted_demand_mw"], color="#1f77b4", lw=1.1, label="Frozen MLP prediction")
            ax.plot(pd.to_datetime(a[target_col], utc=True), a["predicted_demand_mw"], color="#ff7f0e", lw=1.1, label="Adaptive MLP prediction")
            ax.set_title(f"Actual vs frozen and adaptive MLP — first 30 scored days ({period_label})")
            ax.set_ylabel("Electricity demand (MW)")
            ax.set_xlabel("Target timestamp (UTC)")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots / "actual_vs_predicted.png", dpi=220)
            fig.savefig(plots / "actual_vs_predicted.svg")
            plt.close(fig)

        if not adaptive_errors.empty:
            events = data["events"]
            fig, ax = plt.subplots(figsize=(13, 5.5))
            ts = pd.to_datetime(adaptive_errors["target_timestamp"], utc=True)
            ax.plot(ts, adaptive_errors["absolute_error_mw"], lw=0.9, color="#2864a5", label="Adaptive 24-hour absolute error")
            if not events.empty:
                for event in events.itertuples(index=False):
                    drift_ts = pd.Timestamp(event.drift_timestamp)
                    ax.axvline(drift_ts, color="#d62728", alpha=0.45, lw=1)
                    if event.outcome == "adapted":
                        ax.scatter([drift_ts], [0], marker="^", color="#2ca02c", s=45, label="Accepted adaptation" if "Accepted adaptation" not in ax.get_legend_handles_labels()[1] else None)
            ax.set_title(f"ADWIN drift and accepted adaptation events — {period_label}")
            ax.set_ylabel("Absolute forecast error (MW)")
            ax.set_xlabel("Actual target timestamp (UTC)")
            ax.legend()
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots / "adwin_drift_detection.png", dpi=220)
            fig.savefig(plots / "adwin_drift_detection.svg")
            plt.close(fig)

            change = _before_after(events, adaptive_errors)
            change.to_csv(root / "before_after_adaptations.csv", index=False)
            if not change.empty:
                event_x = np.arange(len(change))
                fig, ax = plt.subplots(figsize=(max(8, len(change) * 1.0), 5.5))
                ax.plot(event_x, change["before_mae_mw"], marker="o", label="7-day MAE before adaptation")
                ax.plot(event_x, change["after_mae_mw"], marker="o", label="7-day MAE after adaptation")
                ax.set_title(f"Forecast MAE around accepted adaptations — {period_label}")
                ax.set_ylabel("Fixed 24-hour-ahead MAE (MW)")
                ax.set_xlabel("Accepted adaptation event")
                ax.set_xticks(event_x, pd.to_datetime(change["adaptation_timestamp"], utc=True).dt.strftime("%Y-%m-%d"), rotation=30, ha="right")
                ax.legend()
                ax.grid(alpha=0.25)
                fig.tight_layout()
                fig.savefig(plots / "before_vs_after_adaptation.png", dpi=220)
                fig.savefig(plots / "before_vs_after_adaptation.svg")
                plt.close(fig)

    fine = summary.loc[summary["kind"] == "adaptive"].copy()
    if not fine.empty:
        fine = fine.loc[
            fine["strategy"].str.contains("previous_validated|gentle_recent|gentle_mixed", regex=True)
        ]
        if not fine.empty:
            labels = fine["strategy"].tolist()
            x = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(max(9, len(labels) * 1.3), 5.5))
            ax.bar(x, fine["mae_mw"], color=["#1f77b4" if "recent" in s else "#2ca02c" for s in labels])
            ax.set_title(f"Previous, gentle recent-only, and mixed-history fine-tuning — {period_label}")
            ax.set_ylabel("Fixed 24-hour-ahead MAE (MW; lower is better)")
            ax.set_xticks(x, labels, rotation=25, ha="right")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(plots / "fine_tuning_strategy_comparison.png", dpi=220)
            fig.savefig(plots / "fine_tuning_strategy_comparison.svg")
            plt.close(fig)


def _report(summary: pd.DataFrame, metadata: dict[str, object], path: Path) -> None:
    columns = ["strategy", "mae_mw", "rmse_mw", "drift_period_mae_mw", "post_accepted_adaptation_mae_mw", "drift_alarms", "adaptation_attempts", "accepted_adaptations", "rejected_adaptations", "candidate_training_runtime_seconds", "inference_runtime_seconds", "runtime_seconds"]
    table_frame = summary[columns].copy()
    static_row = summary.loc[summary["kind"] == "static"].iloc[0]
    table_frame["mae_vs_static_pct"] = (summary["mae_mw"] / static_row["mae_mw"] - 1.0) * 100.0
    table_frame["rmse_vs_static_pct"] = (summary["rmse_mw"] / static_row["rmse_mw"] - 1.0) * 100.0
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for record in table_frame.itertuples(index=False, name=None):
        cells = [f"{value:.2f}" if isinstance(value, (int, float, np.number)) and np.isfinite(value) else ("" if pd.isna(value) else str(value)) for value in record]
        body.append("| " + " | ".join(cells) + " |")
    table = "\n".join([header, separator, *body])
    if metadata["profile"] == "final":
        warmup_count = int(metadata.get("pre_evaluation_accepted_adaptations", 0))
        warmup_versions = ", ".join(metadata.get("pre_evaluation_adapted_model_versions", [])) or "none"
        eval_counts = metadata.get("evaluation_adaptation_counts", {})
        selection_text = (
            f"The strategy and threshold were fixed from development selection file "
            f"`{metadata['selection_file']}` before this evaluation. Jan 2025–Oct 2026 is the "
            "untouched score period; its metrics were not used to select the configuration. "
            "Causal replay begins in Apr 2024 so the detector and adaptive model can evolve "
            f"using only observations that precede each simulated time. The warm-up accepted "
            f"{warmup_count} adaptation(s) before the score period (resulting model version(s): "
            f"{warmup_versions}); during the score period there were "
            f"{eval_counts.get('alarms', 0)} alarms, {eval_counts.get('attempts', 0)} attempts, "
            f"{eval_counts.get('accepted', 0)} accepted and {eval_counts.get('rejected', 0)} rejected."
        )
    else:
        selection_text = (
            "The best adaptive strategy is the lowest-MAE adaptive row within this reported "
            "development period only; it is not final-test evidence. If no adaptive row beats "
            "frozen MAE, the frozen MLP remains the best performer. Candidate selection at "
            "each drift event uses only the past validation window available at that event."
        )
    selection_file_arg = (
        f" --selection {metadata['selection_file']}" if metadata.get("selection_file") else ""
    )
    report = f"""# GridDRIFT Phase 3.6 MLP adaptation experiment

## Objective and data

Compare the frozen DE-LU MLP with the previous validated fine-tuning method and conservative ADWIN-triggered continuation strategies. Dataset: hourly ENTSO-E DE-LU actual demand, UTC. Model checkpoint: `{metadata['initial_model_version']}`; frozen model training boundary: `{metadata['model_train_end']}`. This report labels its score period as **{metadata['period_label']}**.

## Method and controls

All adaptive strategies use the same replay observations, persistent ADWIN settings, fixed 24-hour-ahead errors, and {metadata['cooldown_days']}-day adaptation cooldown. Candidate validation requires at least {100 * metadata['minimum_validation_improvement_fraction']:.1f}% improvement in MAE. Candidate training labels are fully observed and separated from validation by the configured 24-hour embargo. Fine-tuning copies the deployed model and reuses its existing preprocessing scalers; the gentle strategies lower the copied Adam learning rate to {metadata['learning_rate_factor']} of its current rate and cap updates at {metadata['max_updates']} mini-batches. Mixed-history runs add up to {metadata['representative_samples_per_month']} evenly spaced historical samples per older month. Seeds: initial model {metadata['model_random_state']}; fine-tune batch sampling {metadata['fine_tune_random_state']}.

## Results

{table}

The `static` row is the frozen baseline. {selection_text}

In the comparison columns, negative percentages mean lower error than the frozen baseline; positive percentages mean worse error.

## Limitations

ADWIN alarms identify distribution changes in the absolute-error stream; they do not guarantee recoverable model drift. Validation is based on historical observations from one demand series and does not prove generalization to future seasons. The plots are regenerated from saved fixed-horizon error tables and adaptation logs; no model retraining is needed to regenerate them.

## Reproduction

```powershell
conda activate griddrift
python scripts/run_phase36_experiments.py --profile {metadata['profile']} --input {metadata['input_path']} --model artifacts/models/de-lu-mlp --config configs/base.yaml --run-id {metadata['run_id']}{selection_file_arg}
```
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("short", "development", "threshold-sweep", "final"), required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--output", default="artifacts/experiments/phase36")
    parser.add_argument("--run-id", default=datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"))
    parser.add_argument("--dev-start", default=DEV_START)
    parser.add_argument("--dev-end", default=DEV_END)
    parser.add_argument("--final-start", default=FINAL_START)
    parser.add_argument("--end", help="Inclusive final replay end; defaults to the last dataset observation")
    parser.add_argument("--selection", help="Development selection.json; required for final profile")
    args = parser.parse_args()

    output_root = Path(args.output) / args.profile / args.run_id
    if output_root.exists() and any(output_root.iterdir()):
        parser.error(f"output already exists and is non-empty: {output_root}; choose a new --run-id")
    output_root.mkdir(parents=True, exist_ok=True)

    demand = _demand(args.input)
    initial_model = DirectMLPForecaster.load(args.model)
    project = load_project_config(args.config)
    drift = project.drift
    cases = _strategy_cases(args.profile)
    if args.profile == "final":
        if not args.selection:
            parser.error("--selection must point to the frozen development shortlist JSON")
        selection = json.loads(Path(args.selection).read_text(encoding="utf-8"))
        if "selected_configuration" in selection:
            cases = [dict(selection["selected_configuration"])]
        else:
            best = selection["selected_strategy"]
            cases = [case for case in cases if case["name"] == best]
        if not cases:
            parser.error(f"selected strategy {best!r} is not available in this script")

    replay_start = args.dev_start
    evaluation_start = None if args.profile != "final" else args.final_start
    short_end = (pd.Timestamp(args.dev_start) + pd.Timedelta(days=30) - pd.Timedelta(hours=1)).isoformat()
    replay_end = args.end or (
        short_end if args.profile == "short" else
        args.dev_end if args.profile in {"development", "threshold-sweep"} else demand.index[-1].isoformat()
    )
    period_label = {
        "short": "short development",
        "development": "development",
        "threshold-sweep": "development threshold sweep",
        "final": "untouched final evaluation",
    }[args.profile]

    metadata: dict[str, object] = {
        "profile": args.profile,
        "run_id": args.run_id,
        "input_path": args.input,
        "model_path": args.model,
        "config_path": args.config,
        "selection_file": args.selection if args.profile == "final" else None,
        "initial_model_version": initial_model.model_version,
        "model_train_end": initial_model.train_end_issue_time.isoformat(),
        "model_random_state": initial_model.model.random_state,
        "fine_tune_random_state": 42,
        "replay_start": replay_start,
        "evaluation_start": evaluation_start,
        "replay_end": replay_end,
        "period_label": period_label,
        "dataset_first_timestamp": demand.index[0].isoformat(),
        "dataset_last_timestamp": demand.index[-1].isoformat(),
        "adwin": {"delta": drift.delta, "clock": drift.clock, "min_window_length": drift.min_window_length},
        "cooldown_days": drift.cooldown_days,
        "minimum_error_increase_mw": drift.min_error_increase_mw,
        "minimum_error_increase_fraction": drift.min_error_increase_fraction,
        "minimum_validation_improvement_fraction": (
            float(cases[0].get("validation_threshold_fraction", drift.min_validation_improvement_fraction))
            if len(cases) == 1 else drift.min_validation_improvement_fraction
        ),
        "learning_rate_factor": 0.1,
        "max_updates": 8,
        "fine_tune_batch_size": 200,
        "representative_samples_per_month": 24,
        "command": " ".join(sys.argv),
        "strategies": cases,
    }
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    shutil.copy2(Path(args.model) / "metadata.json", output_root / "initial_model_metadata.json")
    shutil.copy2(Path(args.model) / "model.joblib", output_root / "initial_model.joblib")

    rows: list[dict[str, object]] = []
    results: dict[str, dict[str, object]] = {}
    static_summary: dict[str, float] | None = None
    best_case: dict[str, object] | None = None
    for case in cases:
        name = str(case["name"])
        strategy_output = output_root / "strategies" / name
        strategy_output.mkdir(parents=True, exist_ok=True)
        replay_config = ReplayConfig(
            cooldown_days=drift.cooldown_days,
            adaptation_policy="validated",
            adaptation_strategy=str(case["strategy"]),
            retraining_window_days=int(case["window_days"]),
            min_error_increase_mw=drift.min_error_increase_mw,
            min_error_increase_fraction=drift.min_error_increase_fraction,
            validation_days=int(case["validation_days"]),
            validation_periods=int(case["validation_periods"]),
            min_validation_improvement_fraction=float(
                case.get("validation_threshold_fraction", drift.min_validation_improvement_fraction)
            ),
            fine_tune_epochs=drift.fine_tune_epochs,
            fine_tune_learning_rate_factor=0.1,
            fine_tune_max_updates=8,
            fine_tune_batch_size=200,
            fine_tune_random_state=42,
            representative_samples_per_month=24,
            adwin=ADWINConfig(delta=drift.delta, clock=drift.clock, min_window_length=drift.min_window_length),
        )
        started = time.perf_counter()
        result = StreamingReplay(demand, initial_model, config=replay_config).run(
            start_timestamp=replay_start, end_timestamp=replay_end
        )
        runtime = time.perf_counter() - started
        static_errors = _filter_period(result.static_errors, evaluation_start, replay_end if args.profile == "final" else None)
        adaptive_errors = _filter_period(result.adaptive_errors, evaluation_start, replay_end if args.profile == "final" else None)
        static_metrics = _metric_row(static_errors)
        adaptive_metrics = _metric_row(adaptive_errors)
        all_events = result.adaptation_frame()
        event_frame = all_events
        if evaluation_start is not None and not all_events.empty:
            drift_timestamps = pd.to_datetime(all_events["drift_timestamp"], utc=True)
            event_frame = all_events.loc[drift_timestamps >= pd.Timestamp(evaluation_start)].copy()
        drift_events = event_frame
        drift_static_mae = _drift_period_mae(static_errors, drift_events)
        drift_adaptive_mae = _drift_period_mae(adaptive_errors, drift_events)
        if static_summary is None:
            static_summary = static_metrics
            static_errors.to_csv(output_root / "static_24h_errors.csv", index=False)
            result.static_registry.save_csv(output_root / "static_forecast_registry.csv")
            rows.append({
                "kind": "static", "strategy": "frozen_static_mlp", **static_metrics,
                "drift_period_mae_mw": drift_static_mae,
                "post_accepted_adaptation_mae_mw": float("nan"),
                "drift_alarms": 0, "adaptation_attempts": 0, "accepted_adaptations": 0,
                "rejected_adaptations": 0, "candidate_training_runtime_seconds": 0.0,
                "inference_runtime_seconds": result.inference_runtime_seconds,
                "runtime_seconds": runtime,
            })
        all_events.to_csv(strategy_output / "adaptation_events.csv", index=False)
        event_frame.to_csv(strategy_output / "evaluation_adaptation_events.csv", index=False)
        result.static_errors.to_csv(strategy_output / "static_24h_errors.csv", index=False)
        result.adaptive_errors.to_csv(strategy_output / "adaptive_24h_errors.csv", index=False)
        result.static_registry.save_csv(strategy_output / "static_forecast_registry.csv")
        result.adaptive_registry.save_csv(strategy_output / "adaptive_forecast_registry.csv")
        result.final_adaptive_model.save(strategy_output / "final_adaptive_model", adaptive_metrics)
        evaluation_training_seconds = float(
            event_frame.get("candidate_training_seconds", pd.Series(dtype=float)).fillna(0).sum()
        )
        evaluation_attempts = int(
            event_frame["outcome"].isin(["adapted", "candidate_rejected"])
            .sum()
        ) if not event_frame.empty else 0
        evaluation_accepted = int((event_frame["outcome"] == "adapted").sum()) if not event_frame.empty else 0
        evaluation_rejected = int(
            event_frame["outcome"].astype(str).str.startswith("candidate_rejected").sum()
        ) if not event_frame.empty else 0
        summary_record = {
            "kind": "adaptive", "strategy": name, **adaptive_metrics,
            "static_mae_mw": static_metrics["mae_mw"],
            "static_rmse_mw": static_metrics["rmse_mw"],
            "drift_period_mae_mw": drift_adaptive_mae,
            "static_drift_period_mae_mw": drift_static_mae,
            "post_accepted_adaptation_mae_mw": _post_accepted_mae(adaptive_errors, event_frame),
            "drift_alarms": len(event_frame),
            "adaptation_attempts": evaluation_attempts,
            "accepted_adaptations": evaluation_accepted,
            "rejected_adaptations": evaluation_rejected,
            "candidate_training_runtime_seconds": evaluation_training_seconds,
            "full_replay_candidate_training_runtime_seconds": result.candidate_training_runtime_seconds,
            "inference_runtime_seconds": result.inference_runtime_seconds,
            "runtime_seconds": runtime,
            "configuration": json.dumps({**case, **replay_config.__dict__, "adwin": replay_config.adwin.__dict__}, default=str),
        }
        rows.append(summary_record)
        results[name] = {
            "static_errors": static_errors,
            "adaptive_errors": adaptive_errors,
            "events": event_frame,
        }
        if best_case is None or adaptive_metrics["mae_mw"] < float(best_case["mae_mw"]):
            best_case = {
                "name": name,
                "mae_mw": adaptive_metrics["mae_mw"],
                "model": result.final_adaptive_model,
                "configuration": dict(case),
            }
        print(json.dumps({key: value for key, value in summary_record.items() if key != "configuration"}), flush=True)

    summary_frame = pd.DataFrame(rows)
    summary_frame.to_csv(output_root / "metrics.csv", index=False)
    (output_root / "metrics.json").write_text(summary_frame.to_json(orient="records", indent=2), encoding="utf-8")
    if evaluation_start is not None:
        drift_times = pd.to_datetime(all_events["drift_timestamp"], utc=True) if not all_events.empty else pd.Series(dtype="datetime64[ns, UTC]")
        pre_evaluation_events = all_events.loc[drift_times < pd.Timestamp(evaluation_start)] if not all_events.empty else all_events
        pre_evaluation_adaptations = pre_evaluation_events.loc[
            pre_evaluation_events["outcome"] == "adapted"
        ] if not pre_evaluation_events.empty else pre_evaluation_events
        metadata["pre_evaluation_accepted_adaptations"] = len(pre_evaluation_adaptations)
        metadata["pre_evaluation_adapted_model_versions"] = (
            pre_evaluation_adaptations["new_model_version"].dropna().astype(str).tolist()
            if not pre_evaluation_adaptations.empty else []
        )
        metadata["evaluation_adaptation_counts"] = {
            "alarms": int(summary_frame.iloc[-1]["drift_alarms"]),
            "attempts": int(summary_frame.iloc[-1]["adaptation_attempts"]),
            "accepted": int(summary_frame.iloc[-1]["accepted_adaptations"]),
            "rejected": int(summary_frame.iloc[-1]["rejected_adaptations"]),
        }
    _save_plots(output_root, rows, results, period_label=period_label)
    metadata["best_development_candidate"] = None if best_case is None else best_case["name"]
    if args.profile != "final" and best_case is not None:
        best_model_path = output_root / "best_development_candidate"
        best_case["model"].save(best_model_path, {"development_mae_mw": best_case["mae_mw"]})
        selected = {
            "selected_strategy": best_case["name"],
            "selected_on": period_label,
            "selection_metric": "fixed_24h_mae_mw",
            "selected_mae_mw": best_case["mae_mw"],
            "static_mae_mw": static_summary["mae_mw"] if static_summary else None,
            "candidate_is_better_than_static": bool(static_summary and best_case["mae_mw"] < static_summary["mae_mw"]),
            "development_run_id": args.run_id,
            "selected_configuration": best_case["configuration"],
        }
        (output_root / "selection.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
        metadata["selection"] = selected
    _report(summary_frame, metadata, output_root / "report.md")
    (output_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved Phase 3.6 artifacts to {output_root}", flush=True)


if __name__ == "__main__":
    main()
