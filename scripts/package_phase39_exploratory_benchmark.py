"""Package and audit existing causal benchmark outputs for Phase 3.9 Stage 5.

The selected 5% MLP configuration exactly matches the saved Phase 3.6 holdout
replay, so this stage reuses that replay and the Phase 3.8 matched comparison
instead of running a duplicate multi-year model replay.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = ROOT / "artifacts/experiments/phase39"
OUTPUT = PHASE_ROOT / "benchmark/exploratory-2025-2026"
SELECTED = PHASE_ROOT / "selected_configuration.json"
THRESHOLD_CONFIG = PHASE_ROOT / "threshold_experiment_config.json"
PH36 = ROOT / "artifacts/experiments/phase36/final/final-holdout-rerun-20261023"
PH36_STRATEGY = PH36 / "strategies/gentle_mixed_60d_threshold_05pct"
PH37 = ROOT / "artifacts/experiments/phase37/final/final-holdout-20261024"
PH38 = ROOT / "artifacts/experiments/phase38/comparison/common-period-20261024"
PH38_EFFECT = ROOT / "artifacts/experiments/phase38/adaptation_validation/holdout-paired-snapshots-20261024"
DATASET = ROOT / "data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_dict(value: object) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    import ast

    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("expected a dictionary in saved experiment metadata")
    return parsed


def _check_selected_configuration() -> tuple[dict[str, object], dict[str, object], pd.Series, dict[str, object]]:
    selected = json.loads(SELECTED.read_text(encoding="utf-8"))
    threshold_config = json.loads(THRESHOLD_CONFIG.read_text(encoding="utf-8"))
    if selected.get("status") != "frozen_for_stage5" or float(selected["selected_threshold_fraction"]) != 0.05:
        raise ValueError("Stage 4 selected configuration is not the frozen 5% setting")
    phase36_meta = json.loads((PH36 / "run_metadata.json").read_text(encoding="utf-8"))
    phase36_metrics = pd.read_csv(PH36 / "metrics.csv")
    mlp_row = phase36_metrics.loc[
        (phase36_metrics["kind"] == "adaptive")
        & (phase36_metrics["strategy"] == "gentle_mixed_60d_threshold_05pct")
    ]
    if len(mlp_row) != 1:
        raise ValueError("expected exactly one saved Phase 3.6 adaptive MLP result")
    recorded = json.loads(mlp_row.iloc[0]["configuration"])
    fixed = selected["frozen_experiment_configuration"]
    expected = {
        "horizon_hours": fixed["horizon_hours"],
        "retraining_window_days": fixed["retraining_window_days"],
        "cooldown_days": fixed["cooldown_days"],
        "adaptation_policy": fixed["adaptation_policy"],
        "adaptation_strategy": fixed["adaptation_strategy"],
        "validation_days": fixed["validation_days"],
        "validation_periods": fixed["validation_periods"],
        "min_error_increase_mw": fixed["min_error_increase_mw"],
        "min_error_increase_fraction": fixed["min_error_increase_fraction"],
        "min_validation_improvement_fraction": selected["selected_threshold_fraction"],
        "fine_tune_epochs": fixed["fine_tune_epochs"],
        "fine_tune_learning_rate_factor": fixed["fine_tune_learning_rate_factor"],
        "fine_tune_max_updates": fixed["fine_tune_max_updates"],
        "fine_tune_batch_size": fixed["fine_tune_batch_size"],
        "fine_tune_random_state": fixed["fine_tune_random_state"],
        "representative_samples_per_month": fixed["representative_samples_per_month"],
        "adwin": fixed["adwin"],
    }
    for key, value in expected.items():
        if recorded.get(key) != value:
            raise ValueError(f"selected configuration does not match saved Phase 3.6 replay on {key}")
    if phase36_meta["initial_model_version"] != threshold_config["source_model_version"]:
        raise ValueError("selected and saved Phase 3.6 runs use different initial model versions")
    if pd.Timestamp(phase36_meta["model_train_end"]) != pd.Timestamp(threshold_config["source_model_train_end_issue_time_utc"]):
        raise ValueError("selected and saved Phase 3.6 runs have different training boundaries")
    if phase36_meta["evaluation_start"] != "2025-01-01T00:00:00Z" or phase36_meta["replay_end"] != "2026-10-20T23:00:00+00:00":
        raise ValueError("saved Phase 3.6 replay does not cover the selected benchmark period")
    current_checkpoint = ROOT / threshold_config["source_checkpoint"] / "model.joblib"
    source_checkpoint = PH36 / "initial_model.joblib"
    checkpoint_hash = _sha256(current_checkpoint)
    if checkpoint_hash != _sha256(source_checkpoint):
        raise ValueError("selected MLP checkpoint differs from the saved Phase 3.6 replay checkpoint")
    audit = json.loads((PHASE_ROOT / "development/chronology_audit.json").read_text(encoding="utf-8"))
    if checkpoint_hash != audit["checkpoint_sha256"]:
        raise ValueError("Phase 3.9 selected checkpoint hash differs from the audited development checkpoint")

    # Confirm the warm-up adaptation path that produced the model used on 2025-01-01.
    dev_events = pd.read_csv(PHASE_ROOT / "development/runs/mlp-threshold-05pct-dev/adaptation_events.csv")
    full_events = pd.read_csv(PH36_STRATEGY / "adaptation_events.csv")
    full_events = full_events.loc[pd.to_datetime(full_events["drift_timestamp"], utc=True) < pd.Timestamp("2025-01-01T00:00:00Z")]
    fields = ["drift_timestamp", "previous_model_version", "new_model_version", "outcome", "candidate_validation_mae_mw", "incumbent_validation_mae_mw", "validation_improvement_fraction"]
    if not dev_events[fields].fillna("").equals(full_events[fields].fillna("").reset_index(drop=True)):
        raise ValueError("Phase 3.9 development and Phase 3.6 benchmark warm-up event paths differ")
    return selected, phase36_meta, mlp_row.iloc[0], {"checkpoint_sha256": checkpoint_hash, "warmup_events": len(dev_events)}


def _metric_copy_with_alias(metrics: pd.DataFrame, scopes: set[str]) -> pd.DataFrame:
    selected = metrics.loc[metrics["scope"].isin(scopes)].copy()
    adaptive = selected.loc[selected["model"] == "adaptive_mlp"].copy()
    adaptive["model"] = "selected_adaptive_mlp_5pct_equivalent"
    adaptive["comparison_note"] = "Exact same 5% replay as previous_adaptive_mlp_5pct; not an independent run"
    selected["comparison_note"] = "Source Phase 3.8 matched-period metric"
    result = pd.concat([selected, adaptive], ignore_index=True)
    return result


def _format_metric_rows(metrics: pd.DataFrame, scope: str, labels: dict[str, str]) -> list[str]:
    frame = metrics.loc[metrics["scope"] == scope]
    rows = []
    for row in frame.itertuples(index=False):
        rows.append(
            f"| {labels.get(row.model, row.model)} | {row.mae_mw:,.2f} | {row.rmse_mw:,.2f} | {row.mape_pct:.3f}% | {row.smape_pct:.3f}% | {int(row.count):,} |"
        )
    return rows


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite Phase 3.9 Stage 5 output: {OUTPUT}")
    selected, phase36_meta, phase36_metric, checkpoint = _check_selected_configuration()
    comparison_summary = json.loads((PH38 / "run_summary.json").read_text(encoding="utf-8"))
    dataset_hash = _sha256(DATASET)
    if dataset_hash != comparison_summary["dataset_sha256"]:
        raise ValueError("current demand dataset differs from Phase 3.8 common comparison data")
    if comparison_summary["target_start"] != "2025-01-01 00:00:00+00:00" or comparison_summary["target_end"] != "2026-10-20 23:00:00+00:00":
        raise ValueError("Phase 3.8 common comparison has an unexpected benchmark period")
    if comparison_summary["common_prediction_keys"] != 379008 or comparison_summary["model_comparison"]["common_fixed_24h_targets"] != 15792:
        raise ValueError("Phase 3.8 predictions are not aligned on the expected common target set")

    source_metrics = pd.read_csv(PH38 / "summary_metrics.csv")
    phase36_mae = float(phase36_metric["mae_mw"])
    phase36_rmse = float(phase36_metric["rmse_mw"])
    common_adaptive = source_metrics.loc[(source_metrics["scope"] == "fixed_24h") & (source_metrics["model"] == "adaptive_mlp")].iloc[0]
    if not (np.isclose(phase36_mae, float(common_adaptive["mae_mw"])) and np.isclose(phase36_rmse, float(common_adaptive["rmse_mw"]))):
        raise ValueError("Phase 3.6 selected MLP and Phase 3.8 matched comparison metrics disagree")

    partial_root = OUTPUT.parent
    partial_root.mkdir(parents=True, exist_ok=True)
    partials = sorted(partial_root.glob(".exploratory-2025-2026-*.partial"))
    if partials:
        temp = partials[-1]
    else:
        temp = partial_root / f".exploratory-2025-2026-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}.partial"
        temp.mkdir(parents=True, exist_ok=False)

    aligned = pd.read_csv(PH38 / "aligned_predictions.csv")
    if len(aligned) != comparison_summary["common_prediction_keys"]:
        raise ValueError("aligned prediction row count differs from Phase 3.8 metadata")
    # The chosen 5% run is exactly the same stream as the previous 5% adaptive MLP.
    aligned["selected_adaptive_mlp_predicted_mw"] = aligned["adaptive_mlp_predicted_mw"]
    aligned.to_csv(temp / "matched_predictions.csv", index=False)

    metrics = _metric_copy_with_alias(source_metrics, {"all_horizons", "fixed_24h"})
    metrics.to_csv(temp / "model_comparison.csv", index=False)
    monthly = _metric_copy_with_alias(source_metrics, {scope for scope in source_metrics["scope"].unique() if str(scope).startswith("month:")})
    monthly.to_csv(temp / "monthly_model_comparison.csv", index=False)
    horizon = pd.read_csv(PH38 / "horizon_metrics.csv")
    selected_horizon = horizon.loc[horizon["model"] == "adaptive_mlp"].copy()
    selected_horizon["model"] = "selected_adaptive_mlp_5pct_equivalent"
    selected_horizon["comparison_note"] = "Exact same 5% replay as previous_adaptive_mlp_5pct"
    horizon["comparison_note"] = "Source Phase 3.8 matched-period metric"
    pd.concat([horizon, selected_horizon], ignore_index=True).to_csv(temp / "horizon_metrics.csv", index=False)

    copy_jobs = {
        "adwin_alarm_targets.csv": PH38 / "adwin_alarm_targets.csv",
        "phase36_mlp_metrics.csv": PH36 / "metrics.csv",
        "phase36_mlp_run_metadata.json": PH36 / "run_metadata.json",
        "mlp_all_replay_adaptation_events.csv": PH36_STRATEGY / "adaptation_events.csv",
        "mlp_holdout_adaptation_events.csv": PH36_STRATEGY / "evaluation_adaptation_events.csv",
        "phase37_xgboost_metrics.csv": PH37 / "metrics.csv",
        "phase37_xgboost_run_metadata.json": PH37 / "run_metadata.json",
        "xgboost_all_replay_adaptation_events.csv": PH37 / "adaptation_events.csv",
        "xgboost_adaptation_effectiveness.csv": PH38_EFFECT / "adaptation_effectiveness.csv",
        "xgboost_monthly_seasonal_consistency.csv": PH38_EFFECT / "xgboost_monthly_seasonal_consistency.csv",
        "phase38_common_run_summary.json": PH38 / "run_summary.json",
        "phase38_comparison_report.md": PH38 / "report.md",
        "phase38_summary_metrics.csv": PH38 / "summary_metrics.csv",
    }
    for name, source in copy_jobs.items():
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, temp / name)

    xgb_events = pd.read_csv(PH37 / "adaptation_events.csv", parse_dates=["drift_timestamp"])
    holdout_events = xgb_events.loc[pd.to_datetime(xgb_events["drift_timestamp"], utc=True) >= pd.Timestamp("2025-01-01T00:00:00Z")]
    holdout_events.to_csv(temp / "xgboost_holdout_adaptation_events.csv", index=False)
    xgb_metric = pd.read_csv(PH37 / "metrics.csv").loc[lambda f: f["strategy_name"] == "full_history"].iloc[0]
    mlp_counts = {
        "holdout_drift_alarms": int(phase36_metric["drift_alarms"]),
        "holdout_attempts": int(phase36_metric["adaptation_attempts"]),
        "holdout_accepted": int(phase36_metric["accepted_adaptations"]),
        "holdout_rejected": int(phase36_metric["rejected_adaptations"]),
        "accepted_warmup_updates_before_benchmark": int(phase36_meta["pre_evaluation_accepted_adaptations"]),
    }
    xgb_counts = {
        "holdout_drift_alarms": int(xgb_metric["holdout_events_from_evaluation_start"].split("'alarms': ")[1].split(",")[0]),
        "holdout_attempts": int(xgb_metric["holdout_events_from_evaluation_start"].split("'attempts': ")[1].split(",")[0]),
        "holdout_accepted": int(xgb_metric["holdout_events_from_evaluation_start"].split("'accepted': ")[1].split(",")[0]),
        "holdout_rejected": int(xgb_metric["holdout_events_from_evaluation_start"].split("'rejected': ")[1].split("}")[0]),
    }
    if len(holdout_events) != xgb_counts["holdout_drift_alarms"]:
        raise ValueError("filtered XGBoost holdout event count differs from saved metrics")

    model_labels = {
        "static_mlp": "Frozen MLP",
        "adaptive_mlp": "Previous adaptive MLP (5%)",
        "selected_adaptive_mlp_5pct_equivalent": "Phase 3.9 selected adaptive MLP (5%; same stream)",
        "static_xgboost": "Frozen XGBoost",
        "adaptive_xgboost": "Adaptive XGBoost",
        "random_forest": "Frozen Random Forest",
    }
    base_fixed = source_metrics.loc[source_metrics["scope"] == "fixed_24h"]
    baseline = base_fixed.loc[base_fixed["model"] == "static_mlp"].iloc[0]
    selected_mlp = base_fixed.loc[base_fixed["model"] == "adaptive_mlp"].iloc[0]
    xgb_effect = pd.read_csv(PH38_EFFECT / "adaptation_effectiveness.csv")
    deployed = xgb_effect.loc[xgb_effect["window"] == "deployed_interval_up_to_7d"]
    xgb_effect_counts = deployed["fixed24_effect"].value_counts().to_dict()
    xgb_old_mae = float(deployed["fixed24_previous_mae_mw"].mean())
    xgb_new_mae = float(deployed["fixed24_updated_mae_mw"].mean())
    xgb_pooled_change_pct = (xgb_new_mae - xgb_old_mae) / xgb_old_mae * 100.0
    summary = {
        "status": "complete",
        "benchmark_label": "Exploratory historical benchmark; this period was examined in Phases 3.6-3.8",
        "target_period_utc": {"start": comparison_summary["target_start"], "end": comparison_summary["target_end"]},
        "common_fixed_24h_target_count": comparison_summary["model_comparison"]["common_fixed_24h_targets"],
        "common_all_horizon_prediction_count": comparison_summary["common_prediction_keys"],
        "actual_demand_verified_from_dataset": comparison_summary["actual_demand_verified_from_source"],
        "dataset_sha256": dataset_hash,
        "selected_threshold_fraction": selected["selected_threshold_fraction"],
        "selected_mlp_reused_previous_adaptive_stream": True,
        "selected_mlp_reuse_reason": "same selected 5% threshold, fixed training/adaptation settings, frozen checkpoint SHA-256, warm-up event path, holdout dates, and matched metrics",
        "selected_mlp_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "mlp_adaptation_counts": mlp_counts,
        "xgboost_adaptation_counts": xgb_counts,
        "xgboost_accepted_update_7day_outcomes": {
            "improved": int(xgb_effect_counts.get("improved", 0)),
            "worsened": int(xgb_effect_counts.get("worsened", 0)),
            "inconclusive": int(xgb_effect_counts.get("inconclusive", 0)),
            "accepted_updates_evaluated": int(len(deployed)),
            "pooled_previous_mae_mw": xgb_old_mae,
            "pooled_updated_mae_mw": xgb_new_mae,
            "pooled_mae_change_pct": xgb_pooled_change_pct,
        },
        "runtime_seconds_from_saved_replays": {
            "mlp_full_replay": float(phase36_metric["runtime_seconds"]),
            "mlp_holdout_candidate_training": float(phase36_metric["candidate_training_runtime_seconds"]),
            "xgboost_full_replay": float(xgb_metric["runtime_seconds"]),
            "xgboost_holdout_candidate_training": float(xgb_metric["candidate_training_runtime_seconds"]),
            "new_replay_run_in_stage5": False,
        },
        "outputs": [
            "matched_predictions.csv",
            "model_comparison.csv",
            "monthly_model_comparison.csv",
            "horizon_metrics.csv",
            "mlp_holdout_adaptation_events.csv",
            "xgboost_holdout_adaptation_events.csv",
            "xgboost_adaptation_effectiveness.csv",
        ],
    }
    (temp / "benchmark_summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8")

    all_table = ["| Model | MAE (MW) | RMSE (MW) | MAPE | sMAPE | Matched forecasts |", "|---|---:|---:|---:|---:|---:|"]
    fixed_table = ["| Model | MAE (MW) | RMSE (MW) | MAPE | sMAPE | Matched targets |", "|---|---:|---:|---:|---:|---:|"]
    for scope, table in (("all_horizons", all_table), ("fixed_24h", fixed_table)):
        table.extend(_format_metric_rows(metrics, scope, model_labels))
    report = f"""# GridDRIFT Phase 3.9 Stage 5 — Exploratory 2025–2026 benchmark

## Benchmark scope and reuse

Evaluation targets: **{comparison_summary['target_start']} through {comparison_summary['target_end']}**. The common comparison contains **{comparison_summary['model_comparison']['common_fixed_24h_targets']:,} fixed-24-hour target timestamps** and **{comparison_summary['common_prediction_keys']:,} issue/target/horizon predictions** across matching models. Actual demand was verified against the current source dataset (SHA-256 `{dataset_hash}`).

This is an **exploratory historical benchmark**, not a fresh holdout: the period was already examined in Phases 3.6–3.8. The selected 5% configuration is identical to the prior Phase 3.6 adaptive MLP configuration. The selected checkpoint hash, complete replay settings, development warm-up event path, replay dates, and fixed-24-hour metrics were checked and matched, so the existing causal replay was reused instead of rerunning the same multi-year replay. The new selected MLP row is therefore an alias of the previous 5% adaptive MLP stream, not an independent experiment.

## Overall matched results

All-lead metrics summarize 24 forecast horizons per target timestamp. Fixed-24-hour metrics are the primary comparison for ADWIN.

### All horizons

{chr(10).join(all_table)}

### Fixed 24-hour ahead

{chr(10).join(fixed_table)}

The frozen MLP recorded MAE/RMSE **{float(baseline['mae_mw']):,.2f}/{float(baseline['rmse_mw']):,.2f} MW**. The selected/previous adaptive MLP recorded **{float(selected_mlp['mae_mw']):,.2f}/{float(selected_mlp['rmse_mw']):,.2f} MW** on fixed 24-hour targets. It had **{mlp_counts['holdout_accepted']} accepted adaptations during the evaluation period**; one 2024 warm-up adaptation was already active at the start of 2025. As no holdout update was accepted, within-holdout post-accepted-update effectiveness is not applicable.

Adaptive XGBoost had {xgb_counts['holdout_drift_alarms']} holdout alarms, {xgb_counts['holdout_attempts']} attempts, {xgb_counts['holdout_accepted']} accepted updates, and {xgb_counts['holdout_rejected']} rejected updates. Paired follow-up analysis of its 20 accepted snapshots classified **{int(xgb_effect_counts.get('improved', 0))} improved, {int(xgb_effect_counts.get('worsened', 0))} worsened, and {int(xgb_effect_counts.get('inconclusive', 0))} inconclusive** over the primary up-to-seven-day windows. Pooled old/new MAE was {xgb_old_mae:,.2f}/{xgb_new_mae:,.2f} MW ({xgb_pooled_change_pct:+.2f}% change). This pooled decrease does not mean every update helped.

## Drift and runtime

| Replay stream | Holdout alarms | Attempts | Accepted | Rejected | Full replay runtime (s) | Candidate training time during holdout (s) |
|---|---:|---:|---:|---:|---:|---:|
| Selected / previous adaptive MLP (5%) | {mlp_counts['holdout_drift_alarms']} | {mlp_counts['holdout_attempts']} | {mlp_counts['holdout_accepted']} | {mlp_counts['holdout_rejected']} | {float(phase36_metric['runtime_seconds']):,.1f} | {float(phase36_metric['candidate_training_runtime_seconds']):.2f} |
| Adaptive XGBoost (0.5%) | {xgb_counts['holdout_drift_alarms']} | {xgb_counts['holdout_attempts']} | {xgb_counts['holdout_accepted']} | {xgb_counts['holdout_rejected']} | {float(xgb_metric['runtime_seconds']):,.1f} | {float(xgb_metric['candidate_training_runtime_seconds']):,.1f} |

The runtimes are the values saved by the original Phase 3.6/3.7 replay jobs, not time spent packaging this Stage 5 report. Monthly and horizon-wise metrics are included in the CSV outputs. Full MLP/XGBoost event logs and the XGBoost paired update analysis are saved alongside matched forecasts.

## Limitations

- The 5% threshold was chosen on 2024 development data, but the 2025–2026 period had already been examined in prior phases. Treat every result here as exploratory.
- The selected adaptive MLP and previous adaptive MLP are the same model stream under the exact same threshold and protocol; there is no separate model to compare between those labels.
- The adaptive MLP improved aggregate holdout fixed-24h MAE relative to the frozen MLP, but it accepted no adaptation during the holdout period. That aggregate comparison reflects the model state carried into 2025 after warm-up, not a successful 2025–2026 update.
- XGBoost accepted many more updates, but per-update paired windows were mostly inconclusive. Do not infer that all accepted updates were beneficial.

## Reproduction and files

Package the verified saved benchmark outputs without model fitting or replay:

```powershell
conda run -n griddrift python scripts/package_phase39_exploratory_benchmark.py
```

The main predictions are in `matched_predictions.csv`; aggregate, monthly, and horizon comparisons are in `model_comparison.csv`, `monthly_model_comparison.csv`, and `horizon_metrics.csv`. Event and update-effectiveness logs are saved in the same directory. The source Phase 3.6–3.8 artifacts remain unchanged.
"""
    (temp / "stage5_exploratory_benchmark.md").write_text(report, encoding="utf-8")

    file_hashes = {path.name: _sha256(path) for path in temp.iterdir() if path.is_file()}
    manifest = {
        "status": "complete",
        "selected_config_path": str(SELECTED.relative_to(ROOT)),
        "selected_config_threshold": selected["selected_threshold_fraction"],
        "phase36_run_id": phase36_meta["run_id"],
        "phase37_run_id": "final-holdout-20261024",
        "phase38_run_id": comparison_summary["run_id"],
        "mlp_replay_reused_after_exact_config_checkpoint_warmup_path_and_metrics_match": True,
        "selected_mlp_source_checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "dataset_sha256": dataset_hash,
        "copied_files_sha256": file_hashes,
        "created_utc": datetime.now(UTC).isoformat(),
        "no_models_retrained_or_replayed_in_stage5": True,
    }
    (temp / "stage5_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temp.rename(OUTPUT)
    print(f"Packaged verified exploratory benchmark outputs at {OUTPUT}")


if __name__ == "__main__":
    main()
