"""Analyze saved Phase 3.9 development runs and freeze the selected threshold."""

from __future__ import annotations

import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PHASE_ROOT = ROOT / "artifacts/experiments/phase39"
DEV_ROOT = PHASE_ROOT / "development"
RESULTS_PATH = DEV_ROOT / "threshold_results.csv"
OUTPUT_CSV = PHASE_ROOT / "development_threshold_comparison.csv"
MONTHLY_CSV = PHASE_ROOT / "monthly_threshold_comparison.csv"
SELECTION_JSON = PHASE_ROOT / "selected_configuration.json"
REPORT_MD = PHASE_ROOT / "threshold_selection.md"


def _metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"mae_mw": float("nan"), "rmse_mw": float("nan"), "count": 0.0}
    errors = frame["absolute_error_mw"].to_numpy(dtype=float)
    return {
        "mae_mw": float(errors.mean()),
        "rmse_mw": float(np.sqrt(np.mean(np.square(errors)))),
        "count": float(len(errors)),
    }


def _read_errors(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["target_timestamp"])
    frame["target_timestamp"] = pd.to_datetime(frame["target_timestamp"], utc=True)
    return frame


def _summary_by_month(threshold: float, run_id: str, static: pd.DataFrame, adaptive: pd.DataFrame) -> list[dict[str, object]]:
    static = static.copy()
    adaptive = adaptive.copy()
    static["month"] = static["target_timestamp"].dt.strftime("%Y-%m")
    adaptive["month"] = adaptive["target_timestamp"].dt.strftime("%Y-%m")
    rows = []
    for month, adaptive_month in adaptive.groupby("month", sort=True):
        static_month = static.loc[static["month"] == month]
        s = _metrics(static_month)
        a = _metrics(adaptive_month)
        rows.append({
            "run_id": run_id,
            "threshold_fraction": threshold,
            "month": month,
            "static_mae_mw": s["mae_mw"],
            "static_rmse_mw": s["rmse_mw"],
            "adaptive_mae_mw": a["mae_mw"],
            "adaptive_rmse_mw": a["rmse_mw"],
            "mae_change_pct_vs_static": (a["mae_mw"] - s["mae_mw"]) / s["mae_mw"] * 100.0,
            "count": a["count"],
        })
    return rows


def _drift_window_metrics(events: pd.DataFrame, static: pd.DataFrame, adaptive: pd.DataFrame) -> tuple[dict[str, float], dict[str, float]]:
    if events.empty:
        return _metrics(static.iloc[0:0]), _metrics(adaptive.iloc[0:0])
    target = pd.to_datetime(adaptive["target_timestamp"], utc=True)
    mask = pd.Series(False, index=adaptive.index)
    for timestamp in pd.to_datetime(events["drift_timestamp"], utc=True):
        # A fixed 24-hour forecast is scoreable after its target is revealed.
        mask |= (target >= timestamp + pd.Timedelta(hours=24)) & (target <= timestamp + pd.Timedelta(days=7))
    targets = set(adaptive.loc[mask, "target_timestamp"])
    static_window = static.loc[static["target_timestamp"].isin(targets)]
    adaptive_window = adaptive.loc[mask]
    return _metrics(static_window), _metrics(adaptive_window)


def main() -> None:
    outputs = (OUTPUT_CSV, MONTHLY_CSV, SELECTION_JSON, REPORT_MD)
    existing = [output for output in outputs if output.exists()]
    if len(existing) == len(outputs):
        try:
            frozen = json.loads(SELECTION_JSON.read_text(encoding="utf-8")).get("status") == "frozen_for_stage5"
        except (json.JSONDecodeError, OSError):
            frozen = False
        if frozen:
            raise FileExistsError("complete Stage 4 outputs already exist; refusing to overwrite them")
    elif existing:
        # Keep interrupted output from a failed attempt while allowing the analysis
        # to resume from the immutable Stage 3 inputs.
        partial_dir = PHASE_ROOT / f"stage4-incomplete-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        partial_dir.mkdir(parents=True, exist_ok=False)
        for output in existing:
            shutil.move(str(output), str(partial_dir / output.name))

    source_results = pd.read_csv(RESULTS_PATH)
    if len(source_results) != 5 or source_results["threshold_fraction"].nunique() != 5:
        raise ValueError("Stage 3 results must contain exactly five distinct thresholds")
    audit = json.loads((DEV_ROOT / "chronology_audit.json").read_text(encoding="utf-8"))
    if audit.get("status") != "passed":
        raise ValueError("Stage 3 chronology audit did not pass")
    static_path = DEV_ROOT / "static_reference/static_24h_errors.csv"
    static = _read_errors(static_path)

    overall_rows: list[dict[str, object]] = []
    monthly_rows: list[dict[str, object]] = []
    detailed: dict[str, dict[str, object]] = {}
    for record in source_results.to_dict(orient="records"):
        threshold = float(record["threshold_fraction"])
        run_id = str(record["run_id"])
        run_dir = DEV_ROOT / "runs" / run_id
        summary_path = run_dir / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") != "complete":
            raise ValueError(f"run {run_id} is not complete")
        adaptive = _read_errors(run_dir / "adaptive_24h_errors.csv")
        events = pd.read_csv(run_dir / "adaptation_events.csv", parse_dates=["drift_timestamp"])
        events["drift_timestamp"] = pd.to_datetime(events["drift_timestamp"], utc=True)
        before_after = pd.read_csv(run_dir / "before_after_adaptations.csv")
        if len(adaptive) != int(record["matched_fixed_24h_predictions"]):
            raise ValueError(f"fixed-24-hour count mismatch for {run_id}")

        drift_static, drift_adaptive = _drift_window_metrics(events, static, adaptive)
        directions = events["error_change_direction"].fillna("").value_counts().to_dict()
        error_before = pd.to_numeric(events.get("error_mean_before_mw", pd.Series(dtype=float)), errors="coerce")
        error_after = pd.to_numeric(events.get("error_mean_after_mw", pd.Series(dtype=float)), errors="coerce")
        later = before_after["later_window_result"].fillna("not_measurable").value_counts().to_dict() if not before_after.empty else {}
        row = {
            **record,
            "mae_change_pct_vs_static": (float(record["adaptive_mae_mw"]) - float(record["static_mae_mw"])) / float(record["static_mae_mw"]) * 100.0,
            "drift_window_static_mae_mw": drift_static["mae_mw"],
            "drift_window_adaptive_mae_mw": drift_adaptive["mae_mw"],
            "drift_window_static_rmse_mw": drift_static["rmse_mw"],
            "drift_window_adaptive_rmse_mw": drift_adaptive["rmse_mw"],
            "drift_window_matched_targets": drift_adaptive["count"],
            "error_increased_alarms": int(directions.get("error_increased", 0)),
            "error_decreased_alarms": int(directions.get("error_decreased", 0)),
            "other_direction_alarms": int(len(events) - directions.get("error_increased", 0) - directions.get("error_decreased", 0)),
            "adwin_mean_error_before_mw": float(error_before.mean()),
            "adwin_mean_error_after_mw": float(error_after.mean()),
            "adwin_mean_shift_mw": float(error_after.mean() - error_before.mean()),
            "accepted_updates_later_window_improved": int(later.get("improved", 0)),
            "accepted_updates_later_window_worsened_or_equal": int(later.get("worsened_or_equal", 0)),
            "accepted_updates_later_window_not_measurable": int(later.get("not_measurable", 0)),
        }
        overall_rows.append(row)
        monthly_rows.extend(_summary_by_month(threshold, run_id, static, adaptive))
        detailed[run_id] = {
            "summary": summary,
            "drift_window_static": drift_static,
            "drift_window_adaptive": drift_adaptive,
            "later_window_counts": {"improved": int(later.get("improved", 0)), "worsened_or_equal": int(later.get("worsened_or_equal", 0)), "not_measurable": int(later.get("not_measurable", 0))},
        }

    comparison = pd.DataFrame(overall_rows).sort_values("threshold_fraction")
    monthly = pd.DataFrame(monthly_rows).sort_values(["threshold_fraction", "month"])
    # Fixed-24-hour MAE is the primary metric used in earlier phase selection.
    # RMSE breaks a tie, followed by fewer accepted model changes.
    selected_row = comparison.sort_values(
        ["adaptive_mae_mw", "adaptive_rmse_mw", "accepted_adaptations", "threshold_fraction"],
        ascending=[True, True, True, False],
    ).iloc[0]
    selected_threshold = float(selected_row["threshold_fraction"])
    selected_run_id = str(selected_row["run_id"])
    static_mae = float(selected_row["static_mae_mw"])
    adaptive_mae = float(selected_row["adaptive_mae_mw"])
    adaptive_rmse = float(selected_row["adaptive_rmse_mw"])
    mae_improvement_pct = (static_mae - adaptive_mae) / static_mae * 100.0
    selected = {
        "phase": "3.9",
        "stage": 4,
        "status": "frozen_for_stage5",
        "selected_threshold_fraction": selected_threshold,
        "selected_threshold_percent": selected_threshold * 100.0,
        "selected_run_id": selected_run_id,
        "selected_on": "2024 development replay only",
        "selection_metric": "fixed_24h_mae_mw",
        "tie_breakers": ["fixed_24h_rmse_mw", "fewer_accepted_adaptations", "higher_threshold_if_still_tied"],
        "selected_fixed_24h_mae_mw": adaptive_mae,
        "selected_fixed_24h_rmse_mw": adaptive_rmse,
        "static_fixed_24h_mae_mw": static_mae,
        "mae_improvement_vs_static_pct": mae_improvement_pct,
        "accepted_adaptations": int(selected_row["accepted_adaptations"]),
        "development_period_utc": {
            "first_fixed_24h_target": "2024-04-10T00:00:00Z",
            "last_target": "2024-12-31T23:00:00Z",
            "matched_fixed_24h_targets": int(selected_row["matched_fixed_24h_predictions"]),
        },
        "frozen_experiment_configuration": json.loads((PHASE_ROOT / "threshold_experiment_config.json").read_text(encoding="utf-8"))["fixed_settings"],
        "production_configuration_changed": False,
        "stage5_period_is_exploratory": True,
        "stage5_may_start_only_after_explicit_user_request": True,
        "notes": [
            "Selection prioritizes fixed 24-hour-ahead forecasting accuracy, not adaptation count.",
            "The chosen threshold had the lowest development fixed-24h MAE and RMSE and one accepted adaptation; its observed subsequent seven-day window improved.",
            "The MAE margin is small and development history has prior exposure. This is a frozen development choice, not an untouched-test conclusion.",
        ],
    }

    comparison.to_csv(OUTPUT_CSV, index=False)
    monthly.to_csv(MONTHLY_CSV, index=False)
    SELECTION_JSON.write_text(json.dumps(selected, indent=2, allow_nan=True), encoding="utf-8")

    table = ["| Threshold | Fixed-24h MAE / RMSE (MW) | Change vs static MAE | Accepted | Post-update 7-day windows improved / worse | Drift-window MAE static / adaptive (MW) |", "|---:|---:|---:|---:|---:|---:|"]
    for row in comparison.to_dict(orient="records"):
        table.append(
            f"| {float(row['threshold_fraction']):.1%} | {float(row['adaptive_mae_mw']):,.2f} / {float(row['adaptive_rmse_mw']):,.2f} | {float(row['mae_change_pct_vs_static']):+.3f}% | {int(row['accepted_adaptations'])} | {int(row['accepted_updates_later_window_improved'])} / {int(row['accepted_updates_later_window_worsened_or_equal'])} | {float(row['drift_window_static_mae_mw']):,.2f} / {float(row['drift_window_adaptive_mae_mw']):,.2f} |"
        )
    adwin_table = ["| Threshold | Increased / decreased alarms | Mean ADWIN error before → after (MW) | Post-alarm 7-day MAE static / adaptive (MW) |", "|---:|---:|---:|---:|"]
    for row in comparison.to_dict(orient="records"):
        adwin_table.append(
            f"| {float(row['threshold_fraction']):.1%} | {int(row['error_increased_alarms'])} / {int(row['error_decreased_alarms'])} | {float(row['adwin_mean_error_before_mw']):,.1f} → {float(row['adwin_mean_error_after_mw']):,.1f} (shift {float(row['adwin_mean_shift_mw']):+,.1f}) | {float(row['drift_window_static_mae_mw']):,.2f} / {float(row['drift_window_adaptive_mae_mw']):,.2f} |"
        )
    month_wins_vs_static = monthly.assign(wins=monthly["adaptive_mae_mw"] < monthly["static_mae_mw"]).groupby("threshold_fraction")["wins"].sum().to_dict()
    selected_month_best_count = int(monthly.loc[(monthly["threshold_fraction"] == selected_threshold) & monthly.apply(lambda row: np.isclose(row["adaptive_mae_mw"], monthly.loc[monthly["month"] == row["month"], "adaptive_mae_mw"].min()), axis=1)].shape[0])
    report = f"""# GridDRIFT Phase 3.9 Stage 4 — Threshold selection

## Decision

Freeze **{selected_threshold:.1%}** as the Phase 3.9 selected validation-improvement threshold for Stage 5. Selection used only the saved 2024 development replay; no 2025–2026 data was read for this decision. The production YAML and deployed dashboard settings were not changed.

## Comparison

The primary metric is fixed-24-hour MAE, consistent with the drift monitor and prior MLP selection. RMSE is a secondary check. The static MLP scored MAE {static_mae:,.2f} MW and RMSE {float(selected_row['static_rmse_mw']):,.2f} MW on the same {int(selected_row['matched_fixed_24h_predictions']):,} targets.

{chr(10).join(table)}

The 5% threshold gave the lowest fixed-24-hour MAE and RMSE and accepted one candidate. Its MAE was {mae_improvement_pct:.3f}% lower than frozen static. The next-lowest fixed-24-hour MAE was {float(comparison['adaptive_mae_mw'].nsmallest(2).iloc[-1]):,.2f} MW; the gap to the selected result is only {float(comparison['adaptive_mae_mw'].nsmallest(2).iloc[-1] - adaptive_mae):.2f} MW, so the accuracy advantage is modest.

The 0% threshold accepted 14 updates, of which {int(comparison.loc[comparison['threshold_fraction'] == 0.0, 'accepted_updates_later_window_worsened_or_equal'].iloc[0])} had equal or worse MAE in the following seven-day window; its aggregate MAE was higher than at 5%. This is evidence that accepting more candidates did not improve this replay. At 0.5% and 1%, two updates were accepted and both had lower MAE in their following seven-day windows; those runs tied on aggregate fixed-24-hour accuracy. At 5%, the single accepted update also had lower subsequent seven-day MAE. These per-event windows are descriptive comparisons, not causal counterfactuals.

ADWIN reported increases and decreases based on its stream comparison. The table below summarizes the recorded mean error on each side of alarms and the union of scoreable fixed-24-hour targets in the seven days after each alarm. The latter includes overlapping alarm windows only once and is descriptive, not a separate causal estimate.

{chr(10).join(adwin_table)}

In the month-by-month file, adaptive MAE beat static in {int(month_wins_vs_static.get(selected_threshold, 0))} of 9 scored months for the selected threshold; the selected threshold had the lowest adaptive MAE in {selected_month_best_count} months (ties included). Monthly values are retained for review because one aggregate development score can hide seasonal variation.

## Interpretation and limitations

- The 5% choice is conservative and best on both primary development metrics, with the fewest accepted updates among thresholds that beat static MAE. It offers a modest gain, not proof of a reliable general improvement.
- The 0.5% and 1% settings have slightly better all-lead MAE than 5%, while fixed-24-hour MAE is the specified primary selection metric. All-lead predictions overlap in target hours and are secondary.
- One accepted update at 5% is a small sample. The 2024 development period has prior exposure from Phase 3.6, so this selection must be validated cautiously and Stage 5 results must be called exploratory.
- Earlier Stage 1 holdout candidates had no recorded candidate reaching 5%; the selected threshold may accept few or no updates in Stage 5. The 5% selection is not changed based on that prior period.

## Frozen configuration and files

The selected threshold and fixed replay settings are saved in `selected_configuration.json`. Stage 5 must use that file unchanged. Full aggregate and monthly metrics are in `development_threshold_comparison.csv` and `monthly_threshold_comparison.csv`; the detailed source remains in `development/runs/`.

Reproduce the read-only analysis:

```powershell
conda run -n griddrift python scripts/analyze_phase39_threshold_selection.py
```

The analysis script refuses to overwrite completed Stage 4 files. No models were trained, no replay was run, and no presentation plots were generated in this stage.
"""
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"Selected {selected_threshold:.1%} from development only; saved Stage 4 outputs under {PHASE_ROOT}")


if __name__ == "__main__":
    main()
