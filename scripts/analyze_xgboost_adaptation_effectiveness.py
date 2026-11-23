"""Validate saved XGBoost updates on paired, subsequent unseen demand forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from griddrift.forecasting.adaptation_analysis import (
    build_feature_matrix_for_issues,
    compare_absolute_errors,
    season_name,
)
from griddrift.forecasting.xgboost import DirectXGBoostForecaster


def _utc(value: object) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


def _load_demand(path: Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    index = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"], utc=True))
    return pd.Series(frame["demand_mw"].to_numpy(dtype=float), index=index, name="demand_mw")


def _model_path(version: str, replay_root: Path, static_root: Path) -> Path:
    if version == "xgb-de-lu-static-42":
        return static_root
    return replay_root / "candidate_models" / version


def _load_model(version: str, replay_root: Path, static_root: Path) -> DirectXGBoostForecaster:
    path = _model_path(version, replay_root, static_root)
    model = DirectXGBoostForecaster.load(path)
    if model.model_version != version:
        raise ValueError(f"checkpoint version mismatch for {version}: {model.model_version}")
    return model


def _effect_row(
    frame: pd.DataFrame,
    *,
    window_name: str,
    window_end: pd.Timestamp,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, object]:
    fixed = frame.loc[frame["horizon_hours"] == 24]
    metrics = compare_absolute_errors(
        fixed["actual_demand_mw"].to_numpy(),
        fixed["previous_prediction_mw"].to_numpy(),
        fixed["updated_prediction_mw"].to_numpy(),
        bootstrap_replicates=bootstrap_replicates,
        seed=seed,
    )
    all_horizons = compare_absolute_errors(
        frame["actual_demand_mw"].to_numpy(),
        frame["previous_prediction_mw"].to_numpy(),
        frame["updated_prediction_mw"].to_numpy(),
        # Descriptive full-horizon metrics only; block-bootstrap inference is
        # reserved for the single fixed lead monitored by ADWIN.
        bootstrap_replicates=0,
        seed=seed + 1,
    )
    return {
        "window": window_name,
        "window_issue_end_exclusive": window_end.isoformat(),
        **{f"fixed24_{key}": value for key, value in metrics.items()},
        **{f"all_horizons_{key}": value for key, value in all_horizons.items()},
    }


def _evaluate_event(
    event: dict[str, str],
    event_number: int,
    accepted_events: list[dict[str, str]],
    demand: pd.Series,
    replay_root: Path,
    static_root: Path,
    feature_cache: pd.DataFrame,
    fixed_registry: pd.DataFrame,
    *,
    followup_days: int,
    deployment_window_days: int,
    bootstrap_replicates: int,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    event_time = _utc(event["drift_timestamp"])
    old_version, new_version = event["previous_model_version"], event["new_model_version"]
    for field in ("training_latest_target_timestamp", "validation_last_target_timestamp"):
        timestamp_value = event.get(field, "")
        if timestamp_value and _utc(timestamp_value) > event_time:
            raise ValueError(f"{field} for {new_version} is after its adaptation timestamp")

    later_accepts = [
        _utc(other["drift_timestamp"])
        for other in accepted_events
        if _utc(other["drift_timestamp"]) > event_time
    ]
    next_update = min(later_accepts) if later_accepts else pd.NaT
    data_end_issue = demand.index[-1] - pd.Timedelta(hours=24)
    fixed_end = event_time + pd.Timedelta(days=deployment_window_days)
    if not pd.isna(next_update):
        fixed_end = min(fixed_end, next_update)
    long_end = min(event_time + pd.Timedelta(days=followup_days), data_end_issue + pd.Timedelta(hours=1))
    primary_issues = demand.index[(demand.index >= event_time) & (demand.index < fixed_end)]
    long_issues = demand.index[(demand.index >= event_time) & (demand.index < long_end)]
    primary_issues = primary_issues[primary_issues <= data_end_issue]
    long_issues = long_issues[long_issues <= data_end_issue]
    if not len(long_issues):
        raise ValueError(f"no fully observable follow-up targets for {new_version}")

    old_model = _load_model(old_version, replay_root, static_root)
    new_model = _load_model(new_version, replay_root, static_root)
    if old_model.feature_spec != new_model.feature_spec or old_model.feature_columns != new_model.feature_columns:
        raise ValueError(f"feature contract changed between {old_version} and {new_version}")
    features = feature_cache.reindex(long_issues).loc[:, new_model.feature_columns]
    old_predictions = old_model.predict_feature_frame(features)
    new_predictions = new_model.predict_feature_frame(features)
    if old_predictions.shape != new_predictions.shape or old_predictions.shape != (len(long_issues), 24):
        raise ValueError(f"unexpected prediction shape for {new_version}")

    issue_rows = np.repeat(long_issues.to_numpy(), 24)
    horizons = np.tile(np.arange(1, 25), len(long_issues))
    targets = pd.DatetimeIndex(issue_rows) + pd.to_timedelta(horizons, unit="h")
    actual = demand.reindex(targets)
    if actual.isna().any():
        raise ValueError(f"missing actual targets in the post-update window for {new_version}")
    primary_cutoff = primary_issues[-1] + pd.Timedelta(hours=1) if len(primary_issues) else event_time
    primary_mask = pd.DatetimeIndex(issue_rows) < primary_cutoff
    predictions = pd.DataFrame({
        "issue_timestamp": issue_rows,
        "target_timestamp": targets,
        "horizon_hours": horizons,
        "actual_demand_mw": actual.to_numpy(dtype=float),
        "previous_model_version": old_version,
        "previous_prediction_mw": old_predictions.reshape(-1),
        "updated_model_version": new_version,
        "updated_prediction_mw": new_predictions.reshape(-1),
        "primary_deployment_window": primary_mask,
        "event_timestamp": event_time,
    })

    # Confirm the reconstructed new snapshot matches the prediction actually stored
    # for the first post-update issue, when the registry contains that issue.
    actual_issued = fixed_registry.loc[[event_time]] if event_time in fixed_registry.index else pd.DataFrame()
    if len(actual_issued) != 1 or actual_issued.iloc[0]["model_version"] != new_version:
        raise ValueError(f"saved replay registry does not show {new_version} issuing at {event_time}")
    reconstructed = float(new_predictions[0, 23])
    if not np.isclose(reconstructed, float(actual_issued.iloc[0]["predicted_demand_mw"]), atol=1e-5, rtol=1e-7):
        raise ValueError(f"reconstructed forecast does not match replay for {new_version}")

    metric_rows: list[dict[str, object]] = []
    base = {
        "event_number": event_number,
        "drift_timestamp": event_time.isoformat(),
        "previous_model_version": old_version,
        "new_model_version": new_version,
        "candidate_training_first_issue_timestamp": event.get("training_first_issue_timestamp", ""),
        "candidate_training_last_issue_timestamp": event.get("training_last_issue_timestamp", ""),
        "candidate_training_latest_target_timestamp": event.get("training_latest_target_timestamp", ""),
        "candidate_validation_first_target_timestamp": event.get("validation_first_target_timestamp", ""),
        "candidate_validation_last_target_timestamp": event.get("validation_last_target_timestamp", ""),
        "candidate_validation_mae_mw": event.get("candidate_validation_mae_mw", ""),
        "incumbent_validation_mae_mw": event.get("incumbent_validation_mae_mw", ""),
        "next_accepted_update_timestamp": "" if pd.isna(next_update) else next_update.isoformat(),
        "primary_window_issue_start": event_time.isoformat(),
        "primary_window_issue_end_exclusive": fixed_end.isoformat(),
        "followup_window_issue_end_exclusive": long_end.isoformat(),
    }
    primary = predictions.loc[predictions["primary_deployment_window"]]
    if len(primary):
        metric_rows.append({**base, **_effect_row(
            primary, window_name="deployed_interval_up_to_7d", window_end=fixed_end,
            bootstrap_replicates=bootstrap_replicates, seed=42 + event_number,
        )})
    else:
        metric_rows.append({**base, "window": "deployed_interval_up_to_7d", "fixed24_effect": "inconclusive"})
    metric_rows.append({**base, **_effect_row(
        predictions, window_name=f"fixed_snapshot_{followup_days}d", window_end=long_end,
        bootstrap_replicates=bootstrap_replicates, seed=1_000 + event_number,
    )})
    return predictions, metric_rows


def _monthly_consistency(aligned_path: Path) -> pd.DataFrame:
    frame = pd.read_csv(aligned_path, parse_dates=["issue_timestamp", "target_timestamp"])
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], utc=True)
    frame["month"] = frame["target_timestamp"].dt.strftime("%Y-%m")
    frame["season"] = frame["target_timestamp"].map(season_name)
    rows: list[dict[str, object]] = []
    for period_type, column in (("month", "month"), ("season", "season")):
        for period, group in frame.groupby(column, sort=True):
            static_mae = float(group["static_xgboost_absolute_error_mw"].mean())
            adaptive_mae = float(group["adaptive_xgboost_absolute_error_mw"].mean())
            rows.append({
                "period_type": period_type,
                "period": period,
                "count": len(group),
                "static_xgboost_mae_mw": static_mae,
                "adaptive_xgboost_mae_mw": adaptive_mae,
                "mae_change_mw": adaptive_mae - static_mae,
                "mae_change_percent": (adaptive_mae - static_mae) / static_mae * 100 if static_mae else np.nan,
                "adaptive_lower_mae": adaptive_mae < static_mae,
            })
    return pd.DataFrame(rows)


def _write_report(run_root: Path, metrics: pd.DataFrame, monthly: pd.DataFrame, manifest: dict[str, object]) -> None:
    primary = metrics.loc[metrics["window"] == "deployed_interval_up_to_7d"]
    effects = primary["fixed24_effect"].value_counts().to_dict()
    long_window = metrics.loc[metrics["window"] == f"fixed_snapshot_{manifest['followup_days']}d"]

    def aggregate_fixed24(frame: pd.DataFrame) -> dict[str, object]:
        weights = frame["fixed24_sample_count"].to_numpy(dtype=float)
        total = float(weights.sum())
        old_mae = float(np.average(frame["fixed24_previous_mae_mw"], weights=weights))
        new_mae = float(np.average(frame["fixed24_updated_mae_mw"], weights=weights))
        old_rmse = float(np.sqrt(np.average(frame["fixed24_previous_rmse_mw"] ** 2, weights=weights)))
        new_rmse = float(np.sqrt(np.average(frame["fixed24_updated_rmse_mw"] ** 2, weights=weights)))
        changes = frame["fixed24_mae_change_mw"]
        return {
            "count": int(total), "old_mae": old_mae, "new_mae": new_mae,
            "old_rmse": old_rmse, "new_rmse": new_rmse,
            "change_percent": (new_mae - old_mae) / old_mae * 100 if old_mae else float("nan"),
            "point_lower": int((changes < 0).sum()), "point_higher": int((changes > 0).sum()),
            "effects": frame["fixed24_effect"].value_counts().to_dict(),
        }

    primary_summary = aggregate_fixed24(primary)
    long_summary = aggregate_fixed24(long_window)
    month = monthly.loc[monthly["period_type"] == "month"]
    decisive = primary.loc[primary["fixed24_effect"].isin(["improved", "worsened"])]
    season = monthly.loc[monthly["period_type"] == "season"].set_index("period")
    report = [
        "# Phase 3.8 Stage 2 — XGBoost adaptation effectiveness",
        "",
        "The saved accepted model snapshots were evaluated counterfactually on the same future forecast issues and targets. No model was retrained. Candidate and preceding models were held fixed in each paired window; this isolates the effect of a model version and is not a rerun of the continuously adapting policy.",
        "",
        f"- Accepted holdout updates evaluated: {manifest['accepted_holdout_updates']}",
        f"- Primary window: up to {manifest['deployment_window_days']} days after activation, ending earlier at the next accepted update.",
        f"- Sustained snapshot window: fixed {manifest['followup_days']} days where the actual series covers the whole window.",
        "- Every saved training/validation latest target was checked to be available by the adaptation timestamp; post-update targets begin after that timestamp. The reconstructed new snapshot's first fixed-24-hour prediction was also checked against the saved replay registry.",
        "- Classification uses a deterministic 95% moving-block bootstrap interval (24-hour blocks) of paired fixed-24-hour absolute-error change; negative means the update lowered error.",
        "- All-horizon metrics are descriptive; ADWIN's directly monitored outcome is the 24-hour-ahead error.",
        "",
        "## Primary paired 24-hour-ahead outcomes",
        "",
        f"Improved: {effects.get('improved', 0)}; worsened: {effects.get('worsened', 0)}; inconclusive: {effects.get('inconclusive', 0)} of {len(primary)} update windows.",
        "",
        "Statistically clearer 7-day outcomes: " + ("; ".join(
            f"{row.drift_timestamp} ({row.new_model_version}: {row.fixed24_effect}, {row.fixed24_mae_change_percent:+.2f}% MAE)"
            for row in decisive.itertuples(index=False)
        ) or "none"),
        "",
        "| Paired window | Paired target rows | Old MAE / RMSE (MW) | Updated MAE / RMSE (MW) | MAE change | Per-update point MAE lower / higher | 95% block-CI improved / worsened / inconclusive |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        f"| Up to 7-day deployed interval | {primary_summary['count']:,} | {primary_summary['old_mae']:.2f} / {primary_summary['old_rmse']:.2f} | {primary_summary['new_mae']:.2f} / {primary_summary['new_rmse']:.2f} | {primary_summary['change_percent']:+.2f}% | {primary_summary['point_lower']} / {primary_summary['point_higher']} | {primary_summary['effects'].get('improved', 0)} / {primary_summary['effects'].get('worsened', 0)} / {primary_summary['effects'].get('inconclusive', 0)} |",
        f"| Fixed {manifest['followup_days']}-day snapshots | {long_summary['count']:,} | {long_summary['old_mae']:.2f} / {long_summary['old_rmse']:.2f} | {long_summary['new_mae']:.2f} / {long_summary['new_rmse']:.2f} | {long_summary['change_percent']:+.2f}% | {long_summary['point_lower']} / {long_summary['point_higher']} | {long_summary['effects'].get('improved', 0)} / {long_summary['effects'].get('worsened', 0)} / {long_summary['effects'].get('inconclusive', 0)} |",
        "",
        "The 28-day windows for different updates can overlap in calendar time. Its paired-row total and pooled metrics therefore summarize event-specific counterfactual comparisons, not unique target timestamps; use the per-update rows for interpretation.",
        "",
        f"The updated snapshot had a lower point MAE in {int(month['adaptive_lower_mae'].sum())} of {len(month)} months; monthly percent changes ranged from {month['mae_change_percent'].min():+.2f}% to {month['mae_change_percent'].max():+.2f}%.",
        "",
        "## Seasonal context",
        "",
        "| Season (UTC target month) | Frozen XGBoost MAE (MW) | Adaptive XGBoost MAE (MW) | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for season_name_value in ("winter", "spring", "summer", "autumn"):
        if season_name_value in season.index:
            row = season.loc[season_name_value]
            report.append(f"| {season_name_value.title()} | {row.static_xgboost_mae_mw:.2f} | {row.adaptive_xgboost_mae_mw:.2f} | {row.mae_change_percent:+.2f}% |")
    report.extend([
        "",
        "## Interpretation caution",
        "",
        "The aggregate Phase 3.7 MAE reduction is not proof that each accepted update helped. Per-event short-term and 28-day paired snapshot metrics are in `adaptation_effectiveness.csv`; monthly and seasonal context is in `xgboost_monthly_seasonal_consistency.csv`. The comparative period had already been examined in Phase 3.7 and is not a fresh independent test.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "conda activate griddrift",
        "python scripts/analyze_xgboost_adaptation_effectiveness.py --input data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv --run-dir " + str(run_root).replace("/", "\\") + " --followup-days 28 --deployment-window-days 7 --bootstrap-replicates 2000",
        "# Rebuild only the Markdown summary from the saved metrics, without loading models or rerunning predictions:",
        "python scripts/analyze_xgboost_adaptation_effectiveness.py --input data/raw/entsoe_cache/entsoe_actual_load_DE_LU_20210101T0000Z_20261021T0000Z_1h.csv --run-dir " + str(run_root).replace("/", "\\") + " --followup-days 28 --deployment-window-days 7 --bootstrap-replicates 2000 --refresh-report",
        "```",
        "",
    ])
    (run_root / "report.md").write_text("\n".join(report), encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    run_root: Path = args.run_dir
    replay_root = Path("artifacts/experiments/phase37/final/final-holdout-20261024/full_history")
    static_root = Path("artifacts/experiments/phase37/static/static-multioutput-20261023/static_xgboost_model")
    source_events = pd.read_csv(replay_root / "adaptation_events.csv").fillna("")
    source_events = source_events.loc[source_events["decision"] == "accepted"].to_dict("records")
    accepted = [event for event in source_events if _utc(event["drift_timestamp"]) >= _utc("2025-01-01T00:00:00Z")]
    if len(accepted) != 20:
        raise ValueError(f"expected 20 accepted holdout updates, found {len(accepted)}")
    accepted.sort(key=lambda event: _utc(event["drift_timestamp"]))
    manifest_path = run_root / "run_metadata.json"
    manifest = {
        "run_id": run_root.name,
        "source_replay": str(replay_root),
        "source_static_checkpoint": str(static_root),
        "accepted_holdout_updates": len(accepted),
        "evaluation_start": "2025-01-01T00:00:00Z",
        "evaluation_end": "2026-10-20T23:00:00Z",
        "deployment_window_days": args.deployment_window_days,
        "followup_days": args.followup_days,
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_block_hours": 24,
        "seed": 42,
        "models_retrained": False,
        "status": "running",
    }
    updates_dir = run_root / "updates"
    updates_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        for key in ("accepted_holdout_updates", "deployment_window_days", "followup_days", "bootstrap_replicates"):
            if previous.get(key) != manifest[key]:
                raise ValueError(f"cannot resume with changed parameter {key}; use a new run directory")
        manifest.update({k: v for k, v in previous.items() if k not in {"status"}})
    else:
        (run_root / "README.md").write_text(
            "# XGBoost adaptation paired-snapshot run\n\n"
            "This run is resumable per accepted update. Files in `updates/` are completed paired forecasts; existing event files are reused unchanged.\n",
            encoding="utf-8",
        )
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    demand = _load_demand(args.input)
    data_end_issue = demand.index[-1] - pd.Timedelta(hours=24)
    all_long_issues = pd.DatetimeIndex(sorted({
        issue
        for event in accepted
        for issue in demand.index[
            (demand.index >= _utc(event["drift_timestamp"]))
            & (demand.index < _utc(event["drift_timestamp"]) + pd.Timedelta(days=args.followup_days))
            & (demand.index <= data_end_issue)
        ]
    }))
    # Materialize the exact past-only feature rows once for all update windows.
    reference_model = _load_model(accepted[0]["new_model_version"], replay_root, static_root)
    feature_cache = build_feature_matrix_for_issues(demand, all_long_issues, reference_model.feature_spec)
    registry = pd.read_csv(
        replay_root / "adaptive_forecast_registry.csv",
        usecols=["issue_timestamp", "horizon_hours", "predicted_demand_mw", "model_version"],
    )
    registry["issue_timestamp"] = pd.to_datetime(registry["issue_timestamp"], utc=True)
    fixed_registry = registry.loc[registry["horizon_hours"] == 24].set_index("issue_timestamp")
    if fixed_registry.index.has_duplicates:
        raise ValueError("adaptive registry has duplicate fixed-lead forecasts")
    metric_rows: list[dict[str, object]] = []
    for position, event in enumerate(accepted, start=1):
        event_time = _utc(event["drift_timestamp"])
        event_id = f"update-{position:02d}_{event_time.strftime('%Y%m%dT%H%MZ')}"
        prediction_path = updates_dir / f"{event_id}_paired_predictions.csv"
        metric_path = updates_dir / f"{event_id}_metrics.json"
        if prediction_path.exists() and metric_path.exists():
            predictions = pd.read_csv(prediction_path, parse_dates=["issue_timestamp", "target_timestamp", "event_timestamp"])
            rows = json.loads(metric_path.read_text(encoding="utf-8"))
        else:
            predictions, rows = _evaluate_event(
                event, position, accepted, demand, replay_root, static_root,
                feature_cache, fixed_registry,
                followup_days=args.followup_days,
                deployment_window_days=args.deployment_window_days,
                bootstrap_replicates=args.bootstrap_replicates,
            )
            predictions.to_csv(prediction_path, index=False)
            metric_path.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")
        metric_rows.extend(rows)
        print(f"[{position}/{len(accepted)}] {event_id}: {len(predictions):,} paired horizon rows saved/reused")

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(run_root / "adaptation_effectiveness.csv", index=False)
    monthly = _monthly_consistency(
        Path("artifacts/experiments/phase37/final/final-holdout-20261024/mlp_xgboost_final_aligned_predictions.csv")
    )
    monthly.to_csv(run_root / "xgboost_monthly_seasonal_consistency.csv", index=False)
    manifest["status"] = "complete"
    manifest["prediction_rows"] = int(sum(
        pd.read_csv(path, usecols=["horizon_hours"]).shape[0]
        for path in updates_dir.glob("*_paired_predictions.csv")
    ))
    manifest["completed_update_files"] = len(list(updates_dir.glob("*_paired_predictions.csv")))
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_report(run_root, metrics, monthly, manifest)
    print(f"Saved Stage 2 paired-snapshot analysis to {run_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--followup-days", type=int, default=28)
    parser.add_argument("--deployment-window-days", type=int, default=7)
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--refresh-report", action="store_true", help="rebuild report.md from completed CSV/JSON outputs only")
    args = parser.parse_args()
    if args.followup_days < args.deployment_window_days or args.deployment_window_days < 1:
        parser.error("followup-days must be >= deployment-window-days >= 1")
    if args.refresh_report:
        manifest_path = args.run_dir / "run_metadata.json"
        metrics_path = args.run_dir / "adaptation_effectiveness.csv"
        monthly_path = args.run_dir / "xgboost_monthly_seasonal_consistency.csv"
        if not all(path.exists() for path in (manifest_path, metrics_path, monthly_path)):
            parser.error("completed run metadata and summary CSVs are required to refresh the report")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        _write_report(args.run_dir, pd.read_csv(metrics_path), pd.read_csv(monthly_path), manifest)
        print(f"Regenerated report from saved data at {args.run_dir}")
        return
    if args.run_dir.exists() and any(args.run_dir.iterdir()):
        if not (args.run_dir / "run_metadata.json").exists():
            parser.error("run directory is non-empty and has no resumable run metadata; choose a new path")
        existing_manifest = json.loads((args.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
        if existing_manifest.get("status") == "complete":
            print(f"Run already complete; reusing saved outputs at {args.run_dir}")
            return
    args.run_dir.mkdir(parents=True, exist_ok=True)
    run_analysis(args)


if __name__ == "__main__":
    main()
